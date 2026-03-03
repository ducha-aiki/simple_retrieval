"""MASt3R-based global (ASMK) and local feature extraction.

Keeps all MASt3R/dust3r dependencies in one place so that global_feature.py
and local_feature.py remain agnostic to this specific backbone.
"""
import os
import tempfile
import cv2
import h5py
import numpy as np
import torch
import kornia.feature as KF
from collections import Counter
from itertools import groupby
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------

def _predict_dust3r_shape(W1, H1, size=512, patch_size=16):
    """Predict dust3r preprocessing output (H_out, W_out) from input (W1, H1).

    Replicates load_images() resize+crop logic without loading pixels, so we
    can pre-sort images by output shape for same-shape GPU batching.
    """
    S = max(W1, H1)
    W = int(round(W1 * size / S))
    H = int(round(H1 * size / S))
    cx, cy = W // 2, H // 2
    halfw = ((2 * cx) // patch_size) * patch_size / 2
    halfh = ((2 * cy) // patch_size) * patch_size / 2
    if W == H:  # square_ok=False is the load_images default
        halfh = 3 * halfw / 4
    return int(2 * halfh), int(2 * halfw)


def _mast3r_patch_centers(topk_indices, true_shape, patch_size=16):
    """Convert patch indices → (x, y) pixel centres in processed-image space.

    Args:
        topk_indices: LongTensor (B, nfeat)
        true_shape:   LongTensor (B, 2) = [H, W]
    Returns:
        FloatTensor (B, nfeat, 2) — (x, y)
    """
    W_patches = true_shape[:, 1:2] // patch_size   # (B, 1)
    rows = (topk_indices // W_patches).float()
    cols = (topk_indices  % W_patches).float()
    return torch.stack([(cols + 0.5) * patch_size,
                        (rows + 0.5) * patch_size], dim=-1)  # (B, nfeat, 2)


def _read_image_hw(path):
    """Read (H, W) from image header + EXIF orientation, without loading pixels."""
    import PIL.Image
    _ROTATE_TAGS = {5, 6, 7, 8}
    with PIL.Image.open(path) as img:
        W, H = img.size
        try:
            exif = img._getexif()
            if exif:
                orient = exif.get(274)   # 274 = Orientation tag
                if orient in _ROTATE_TAGS:
                    W, H = H, W
        except Exception:
            pass
    return H, W


def _sorted_by_dust3r_shape(paths, imsize, num_workers=8):
    """Parallel header-only reads → sort indices for same-shape GPU batching.

    Returns:
        order           – list[int], sorted_i -> orig_i
        sorted_paths    – paths in sorted order
        sorted_orig_hws – (H,W) tuples in sorted order
        orig_hws        – (H,W) tuples in original order
    """
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        orig_hws = list(ex.map(_read_image_hw, paths))

    out_shapes = [_predict_dust3r_shape(hw[1], hw[0], imsize) for hw in orig_hws]
    order = sorted(range(len(paths)), key=lambda i: out_shapes[i])
    return (order,
            [paths[i]    for i in order],
            [orig_hws[i] for i in order],
            orig_hws)


# ---------------------------------------------------------------------------
# MASt3R ASMK global retrieval
# ---------------------------------------------------------------------------

class MASt3RASMKRetrieval:
    """MASt3R local features + ASMK inverted-file index for image retrieval.

    Args:
        mast3r_dir: path to the cloned naver/mast3r repository.
        retrieval_model_path: path to the *_trainingfree.pth checkpoint.
            The matching *_codebook.pkl must be in the same directory.
        device: 'cuda' or 'cpu'.
    """

    def __init__(self, mast3r_dir, retrieval_model_path, device='cuda'):
        import sys
        mast3r_dir = os.path.abspath(mast3r_dir)
        if mast3r_dir not in sys.path:
            sys.path.insert(0, mast3r_dir)
        import mast3r.utils.path_to_dust3r  # noqa: F401
        from mast3r.retrieval.processor import Retriever
        from mast3r.model import AsymmetricMASt3R
        backbone = None
        dname = os.path.dirname(os.path.abspath(retrieval_model_path))
        bname = os.path.basename(retrieval_model_path)
        backbone_name = bname.replace('_retrieval_trainingfree', '')
        backbone_path = os.path.join(dname, backbone_name)
        if os.path.isfile(backbone_path):
            backbone = AsymmetricMASt3R.from_pretrained(backbone_path)
        self.retriever = Retriever(retrieval_model_path, backbone=backbone, device=device)
        self.device = device
        self.asmk_dataset = None

    def _extract_chunk_batched(self, images, batch_size=8, num_workers=4):
        """Extract MASt3R local features, batching same-shape images on the GPU.

        Returns (feat_list, kpts_list, orig_hws) — all in original image order.
            feat_list  – list of np.float32 (nfeat_i, D)
            kpts_list  – list of np.float32 (nfeat_i, 2)  (x,y) in original px
            orig_hws   – list of (H_orig, W_orig) tuples
        """
        from mast3r.retrieval.model import Dust3rInputFromImageList

        imsize = self.retriever.imsize
        order, sorted_images, sorted_orig_hws, orig_hws = \
            _sorted_by_dust3r_shape(images, imsize)

        shape_counts = Counter(_predict_dust3r_shape(h[1], h[0], imsize)
                               for h in orig_hws)
        most_common_shape, most_common_n = shape_counts.most_common(1)[0]
        print(f"[MASt3R-batch] {len(images)} images, {len(shape_counts)} unique output shapes; "
              f"most common {most_common_shape} × {most_common_n} imgs", flush=True)

        dataset = Dust3rInputFromImageList(sorted_images, imsize=imsize)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=lambda items: items,
        )

        feat_list   = [None] * len(images)
        kpts_list   = [None] * len(images)
        flush_sizes = []
        sorted_offset = 0

        for dl_batch in tqdm(loader, desc="MASt3R extract", leave=False):
            keyed = [
                (tuple(item['true_shape'][0].tolist()), sorted_offset + k, item)
                for k, item in enumerate(dl_batch)
            ]
            sorted_offset += len(dl_batch)

            for shape, grp in groupby(keyed, key=lambda x: x[0]):
                grp = list(grp)
                flush_sizes.append(len(grp))
                imgs_t = torch.cat([item['img'] for _, _, item in grp], dim=0
                                   ).to(self.device, non_blocking=True)
                sh_t   = torch.tensor([list(shape)] * len(grp),
                                      dtype=torch.int64, device=self.device)
                with torch.no_grad():
                    feat, _, topk_idx = self.retriever.model.forward_local(
                        {'img': imgs_t, 'true_shape': sh_t})
                feat_cpu  = feat.cpu()
                kpts_proc = _mast3r_patch_centers(topk_idx.cpu(), sh_t.cpu())
                H_proc, W_proc = shape
                for j, (_, sorted_i, _) in enumerate(grp):
                    orig_i         = order[sorted_i]
                    H_orig, W_orig = sorted_orig_hws[sorted_i]
                    kpts           = kpts_proc[j].clone()
                    kpts[:, 0]    *= W_orig / W_proc
                    kpts[:, 1]    *= H_orig / H_proc
                    feat_list[orig_i] = feat_cpu[j].numpy()
                    kpts_list[orig_i] = kpts.numpy()

        if flush_sizes:
            avg_bs = sum(flush_sizes) / len(flush_sizes)
            print(f"[MASt3R-batch] {len(flush_sizes)} GPU forward passes, "
                  f"avg batch={avg_bs:.1f} (min={min(flush_sizes)} max={max(flush_sizes)})",
                  flush=True)

        return feat_list, kpts_list, orig_hws

    def index_images(self, fnames, h5_path, chunk_size=2000, batch_size=8,
                     local_feature_dir=None):
        """Extract features in chunks, stream to HDF5, build ASMK IVF.

        Chunks limit peak RAM (each chunk's feat_list lives in memory until
        the writer thread picks it up).  The writer thread keeps all H5 files
        open for the full duration — no repeated open/close per chunk.

        If *local_feature_dir* is given, writes descriptors/keypoints/lafs/hw
        H5 files there in the same forward pass — no second model run needed.
        """
        import queue, threading

        images = list(fnames)
        N      = len(images)

        # Sanity-check: run first image, print descriptor stats, get dim
        feat_list_1, _, _ = self._extract_chunk_batched(images[:1], batch_size=1)
        f   = feat_list_1[0]
        dim = f.shape[1]
        print(f"[MASt3R-ASMK] first image: {f.shape[0]} kpts, dim={dim}, "
              f"min={f.min():.4f} max={f.max():.4f} mean={f.mean():.4f} std={f.std():.4f}",
              flush=True)
        del feat_list_1

        write_q = queue.Queue(maxsize=4)   # bound queue for backpressure

        def writer_worker():
            lf = {}
            with h5py.File(h5_path, 'w') as hf:
                ds_feat = hf.create_dataset('feat', shape=(0, dim),
                                            maxshape=(None, dim),
                                            dtype='float32', chunks=(1024, dim))
                ds_ids  = hf.create_dataset('ids',  shape=(0,),
                                            maxshape=(None,),
                                            dtype='int64', chunks=(1024,))
                if local_feature_dir:
                    os.makedirs(local_feature_dir, exist_ok=True)
                    lf['desc'] = h5py.File(f'{local_feature_dir}/descriptors.h5', 'w')
                    lf['kp']   = h5py.File(f'{local_feature_dir}/keypoints.h5',   'w')
                    lf['laf']  = h5py.File(f'{local_feature_dir}/lafs.h5',        'w')
                    lf['hw']   = h5py.File(f'{local_feature_dir}/hw.h5',           'w')
                try:
                    write_pos = 0
                    while True:
                        item = write_q.get()
                        if item is None:    # sentinel — no more chunks
                            break
                        img_start, feat_list, kpts_list, orig_hws, chunk_paths = item
                        feat_np = np.concatenate(feat_list, axis=0)
                        ids_np  = np.concatenate([
                            np.full(feat_list[i].shape[0], img_start + i, dtype=np.int64)
                            for i in range(len(chunk_paths))
                        ], axis=0)
                        n = len(feat_np)
                        ds_feat.resize(write_pos + n, axis=0)
                        ds_feat[write_pos:write_pos + n] = feat_np
                        ds_ids.resize(write_pos + n, axis=0)
                        ds_ids[write_pos:write_pos + n]  = ids_np
                        write_pos += n
                        if lf:
                            for i, path in enumerate(chunk_paths):
                                lafs = KF.laf_from_center_scale_ori(
                                    torch.from_numpy(kpts_list[i]).unsqueeze(0)).numpy()
                                lf['desc'][path] = feat_list[i]
                                lf['kp'][path]   = kpts_list[i]
                                lf['laf'][path]  = lafs
                                lf['hw'][path]   = np.array(list(orig_hws[i]))
                        write_q.task_done()
                finally:
                    for f in lf.values():
                        f.close()

        writer = threading.Thread(target=writer_worker, daemon=True)
        writer.start()

        for start in range(0, N, chunk_size):
            chunk = images[start:start + chunk_size]
            feat_list, kpts_list, orig_hws = self._extract_chunk_batched(
                chunk, batch_size=batch_size)
            write_q.put((start, feat_list, kpts_list, orig_hws, chunk))
            print(f"[MASt3R-ASMK] {min(start + chunk_size, N)}/{N} images extracted",
                  flush=True)
            del feat_list, kpts_list, orig_hws

        write_q.put(None)   # sentinel
        writer.join()

        self._build_ivf_from_h5(h5_path)

    def _build_ivf_from_h5(self, h5_path, kpt_chunk=500_000):
        """Build ASMK IVF aligned to image boundaries (no split kpt runs)."""
        print("[MASt3R-ASMK] building IVF from HDF5...", flush=True)
        builder = self.retriever.asmk.create_ivf_builder()
        with h5py.File(h5_path, 'r') as hf:
            total = hf['feat'].shape[0]
            start = 0
            while start < total:
                end = min(start + kpt_chunk, total)
                if end < total:
                    lookahead_end = min(end + 10_000, total)
                    ids_look = hf['ids'][end - 1:lookahead_end]
                    end = (end - 1) + int(np.searchsorted(ids_look, int(ids_look[0]) + 1))
                builder.add(hf['feat'][start:end], hf['ids'][start:end])
                print(f"[MASt3R-ASMK] IVF: {end}/{total} kpts", flush=True)
                start = end
        self.asmk_dataset = self.retriever.asmk.add_ivf_builder(builder)
        print("[MASt3R-ASMK] IVF built.", flush=True)

    def rebuild_ivf(self, h5_path):
        self._build_ivf_from_h5(h5_path)

    def query(self, img_np):
        """Query with a single RGB numpy image → (ranks, scores)."""
        from mast3r.retrieval.model import extract_local_features
        assert self.asmk_dataset is not None, "Call index_images or rebuild_ivf first"
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmppath = f.name
        try:
            cv2.imwrite(tmppath, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            feat, ids = extract_local_features(
                self.retriever.model, [tmppath],
                self.retriever.imsize, tocpu=True, device=self.device,
            )
            feat, ids = feat.numpy(), ids.numpy()
        finally:
            os.unlink(tmppath)
        _meta, _qids, ranks, ranked_scores = self.asmk_dataset.query_ivf(feat, ids)
        return ranks[0], ranked_scores[0]


# ---------------------------------------------------------------------------
# MASt3R local features (same backbone, no extra GPU pass)
# ---------------------------------------------------------------------------

def detect_mast3r_dir(img_fnames, model, imsize, feature_dir,
                      batch_size=8, num_workers=4,
                      device=torch.device('cuda')):
    """Extract MASt3R local features for all DB images, save to H5 files.

    Output layout (same as detect_xfeat_dir / detect_sift_dir):
        {feature_dir}/descriptors.h5, keypoints.h5, lafs.h5, hw.h5

    Keypoints are in original-image pixel coordinates.

    Args:
        model:  RetrievalModel (retriever.model from MASt3RASMKRetrieval)
        imsize: dust3r processing size (typically 512)
    """
    try:
        from mast3r.retrieval.model import Dust3rInputFromImageList
    except ImportError:
        from dust3r.retrieval.model import Dust3rInputFromImageList

    if os.path.exists(os.path.join(feature_dir, 'descriptors.h5')):
        print(f"[MASt3R local] cache exists in {feature_dir}, skipping")
        return

    os.makedirs(feature_dir, exist_ok=True)

    order, sorted_fnames, sorted_orig_hws, _ = \
        _sorted_by_dust3r_shape(img_fnames, imsize)

    dataset = Dust3rInputFromImageList(sorted_fnames, imsize=imsize)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=lambda items: items,
    )

    with h5py.File(f'{feature_dir}/descriptors.h5', 'w') as f_desc, \
         h5py.File(f'{feature_dir}/keypoints.h5',   'w') as f_kp,   \
         h5py.File(f'{feature_dir}/lafs.h5',        'w') as f_laf,  \
         h5py.File(f'{feature_dir}/hw.h5',          'w') as f_hw:

        sorted_offset = 0
        for dl_batch in tqdm(loader, desc="MASt3R local extract"):
            keyed = [
                (tuple(item['true_shape'][0].tolist()), sorted_offset + k, item)
                for k, item in enumerate(dl_batch)
            ]
            sorted_offset += len(dl_batch)

            for shape, grp in groupby(keyed, key=lambda x: x[0]):
                grp = list(grp)
                imgs_t = torch.cat([item['img'] for _, _, item in grp], dim=0
                                   ).to(device, non_blocking=True)
                sh_t   = torch.tensor([list(shape)] * len(grp),
                                      dtype=torch.int64, device=device)
                with torch.inference_mode():
                    feat, _, topk_idx = model.forward_local(
                        {'img': imgs_t, 'true_shape': sh_t})
                feat_cpu  = feat.cpu()
                kpts_proc = _mast3r_patch_centers(topk_idx.cpu(), sh_t.cpu())
                H_proc, W_proc = shape
                for j, (_, sorted_i, _) in enumerate(grp):
                    path           = sorted_fnames[sorted_i]
                    H_orig, W_orig = sorted_orig_hws[sorted_i]
                    kpts           = kpts_proc[j].clone()
                    kpts[:, 0]    *= W_orig / W_proc
                    kpts[:, 1]    *= H_orig / H_proc
                    kpts_np        = kpts.numpy()
                    lafs = KF.laf_from_center_scale_ori(
                        torch.from_numpy(kpts_np).unsqueeze(0)).numpy()
                    f_desc[path] = feat_cpu[j].numpy()
                    f_kp[path]   = kpts_np
                    f_laf[path]  = lafs
                    f_hw[path]   = np.array([H_orig, W_orig])


def detect_mast3r_single(img, model, imsize, device=torch.device('cuda')):
    """Extract MASt3R features from one RGB numpy image.

    Returns (kpts, descs, lafs) matching detect_sift_single / detect_xfeat_single.
    """
    from dust3r.utils.image import load_images
    H_orig, W_orig = img.shape[:2]

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        tmppath = f.name
    try:
        cv2.imwrite(tmppath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        sample = load_images([tmppath], size=imsize, verbose=False)[0]
    finally:
        os.unlink(tmppath)

    true_shape = torch.from_numpy(sample['true_shape']).to(device)
    img_t      = sample['img'].to(device)
    H_proc, W_proc = true_shape[0, 0].item(), true_shape[0, 1].item()

    with torch.inference_mode():
        feat, _, topk_idx = model.forward_local({'img': img_t, 'true_shape': true_shape})

    kpts = _mast3r_patch_centers(topk_idx[0].cpu().unsqueeze(0), true_shape.cpu())[0]
    kpts[:, 0] *= W_orig / W_proc
    kpts[:, 1] *= H_orig / H_proc
    kpts_np = kpts.numpy()

    lafs = KF.laf_from_center_scale_ori(torch.from_numpy(kpts_np).unsqueeze(0))
    return kpts_np, feat[0].cpu(), lafs
