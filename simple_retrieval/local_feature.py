
import torch
import cv2
import h5py
import kornia as K
import os
from tqdm import tqdm
import numpy as np
import kornia.feature as KF
from multiprocessing import Pool, cpu_count
from kornia_moons.feature import laf_from_opencv_SIFT_kpts
from simple_retrieval.pile_of_garbage import CustomImageFolderFromFileList, collate_with_string, no_collate, H5LocalFeatureDataset, H5MASt3RLocalFeatureDataset
from simple_retrieval.xfeat import XFeat, LighterGlue
from simple_retrieval.clidd import CLIDD
from simple_retrieval.mast3r_feature import detect_mast3r_dir, detect_mast3r_single  # noqa: F401
import torchvision.transforms as T
from torch.utils.data import DataLoader
from kornia_moons.feature import kornia_matches_from_cv2

def _rootsift_numpy(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Pure-numpy RootSIFT: L1-normalise → sqrt → L2-normalise."""
    x = x.astype(np.float32)
    x /= (np.linalg.norm(x, ord=1, axis=-1, keepdims=True) + eps)
    np.clip(x, eps, None, out=x)
    np.sqrt(x, out=x)
    x /= (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + eps)
    return x


def _laf_from_opencv_kpts_numpy(kpts, mr_size: float = 6.0) -> np.ndarray:
    """
    Build a (1, N, 2, 3) LAF array from a list of cv2.KeyPoint objects.
    Pure numpy — safe to call inside forked worker processes.
    """
    n = len(kpts)
    xy     = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts], dtype=np.float32)
    sizes  = np.array([kp.size for kp in kpts], dtype=np.float32)
    angles = np.array([-kp.angle * np.pi / 180.0 for kp in kpts], dtype=np.float32)
    scale  = sizes / (2.0 * mr_size)
    cos_a  = np.cos(angles) * scale
    sin_a  = np.sin(angles) * scale
    lafs = np.zeros((1, n, 2, 3), dtype=np.float32)
    lafs[0, :, 0, 0] =  cos_a
    lafs[0, :, 0, 1] = -sin_a
    lafs[0, :, 0, 2] =  xy[:, 0]
    lafs[0, :, 1, 0] =  sin_a
    lafs[0, :, 1, 1] =  cos_a
    lafs[0, :, 1, 2] =  xy[:, 1]
    return lafs


def _sift_worker(args):
    """
    Module-level worker for parallel SIFT extraction.
    Pure numpy/OpenCV — no torch/kornia — safe under multiprocessing fork.
    """
    img_path, num_feats, resize_to = args
    try:
        sift = cv2.SIFT_create(num_feats, edgeThreshold=-1000, contrastThreshold=-1000)
        img1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if img1 is None:
            return img_path, None
        hw1 = np.array(img1.shape[:2])
        if resize_to:
            img1 = cv2.resize(img1, resize_to)
            hw1_new = np.array(img1.shape[:2])
            scale_w = float(hw1[1]) / float(hw1_new[1])
            scale_h = float(hw1[0]) / float(hw1_new[0])
        else:
            scale_w = scale_h = 1.0
        kpts1, descs1 = sift.detectAndCompute(img1, None)
        if kpts1 is None or len(kpts1) == 0:
            return img_path, None
        lafs_np = _laf_from_opencv_kpts_numpy(kpts1)
        if resize_to:
            lafs_np[0, :, 0, :] *= scale_w
            lafs_np[0, :, 1, :] *= scale_h
        avg_sf   = (scale_w + scale_h) / 2.0
        scales1  = np.array([kp.size / 2.0 * avg_sf for kp in kpts1], dtype=np.float32)
        descs_np = _rootsift_numpy(descs1)
        kpts_np  = lafs_np[0, :, :, 2]          # centre = last column of each 2×3 block
        return img_path, (kpts_np, lafs_np, descs_np, hw1, scales1)
    except Exception as e:
        print(f"[SIFT worker] Error on {img_path}: {e}")
        return img_path, None


def load_torch_image(fname, device=torch.device('cpu')):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.
    img = K.color.bgr_to_rgb(img.to(device))
    return img


def sift_to_rootsift(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, p=1, dim=-1, eps=eps)
    x.clip_(min=eps).sqrt_()
    return torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)


def quantize_descriptors(descs: np.ndarray) -> np.ndarray:
    """L2-normalised float descriptors → uint8 (scale=512, clipped to [0,255])."""
    return (descs * 512).clip(0, 255).astype(np.uint8)


def get_input_xfeat_transform(image_size=None):
    if image_size:
        return T.Compose([
            T.Resize(image_size,  interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
        ])

def detect_sift_dir(img_fnames,
                num_feats=2048,
                device=torch.device('cpu'),
                feature_dir='.featureout', resize_to=(800, 600),
                num_workers=None):
    """Extract SIFT features for a list of images in parallel.

    Uses ``num_workers`` processes for extraction (default: cpu_count()-1) and
    writes to HDF5 files in the main process (single writer).
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    os.makedirs(feature_dir, exist_ok=True)

    worker_args = [(p, num_feats, resize_to) for p in img_fnames]

    with Pool(num_workers) as pool, \
         h5py.File(f'{feature_dir}/lafs.h5', mode='w') as f_laf, \
         h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
         h5py.File(f'{feature_dir}/hw.h5', mode='w') as f_hw, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc, \
         h5py.File(f'{feature_dir}/scales.h5', mode='w') as f_scales:
        for img_path, result in tqdm(
                pool.imap_unordered(_sift_worker, worker_args),
                total=len(worker_args), desc="SIFT extraction"):
            if result is None:
                print(f"[detect_sift_dir] Skipping {img_path} (no keypoints or read error)")
                continue
            kpts_np, lafs_np, descs_np, hw1, scales1 = result
            key = img_path
            f_laf[key]    = lafs_np
            f_kp[key]     = kpts_np
            f_desc[key]   = descs_np
            f_hw[key]     = hw1
            f_scales[key] = scales1
    return

def detect_sift_single(img, num_feats=2048, resize_to=(800, 600)):
    device=torch.device('cpu')
    sift = cv2.SIFT_create(num_feats, edgeThreshold=-1000, contrastThreshold=-1000)
    hw1 = torch.tensor(img.shape[:2])
    if resize_to:
        img = cv2.resize(img, resize_to)
        hw1_new = torch.tensor(img.shape[:2], device=device)
    kpts, descs = sift.detectAndCompute(img, None)
    descs1 = sift_to_rootsift(torch.from_numpy(descs)).to(device)
    lafs1 = laf_from_opencv_SIFT_kpts(kpts)
    if resize_to:
        lafs1[:, :, 0, :] *= hw1[1] / hw1_new[1]
        lafs1[:, :, 1, :] *= hw1[0] / hw1_new[0]
    kpts = KF.get_laf_center(lafs1).reshape(-1, 2).detach().cpu().numpy()
    return kpts, descs1, lafs1


def detect_xfeat_single(img, num_feats=2048, resize_to=(800, 600)):
    device=torch.device('cpu')
    model = XFeat(top_k=num_feats, detection_threshold=0.0)
    hw1 = torch.tensor(img.shape[:2])
    if resize_to:
        img = cv2.resize(img, resize_to)
        hw1_new = torch.tensor(img.shape[:2], device=device)
    res = model.detectAndCompute(K.image_to_tensor(img,None).float(), top_k=num_feats)
    keypoints, descriptors = res[0]['keypoints'], res[0]['descriptors']
    lafs1 = K.feature.laf_from_center_scale_ori(keypoints.reshape(1, -1, 2))
    kpts = KF.get_laf_center(lafs1).reshape(-1, 2).detach().cpu().numpy()
    if resize_to:
        lafs1[:, :, 0, :] *= hw1[1] / hw1_new[1]
        lafs1[:, :, 1, :] *= hw1[0] / hw1_new[0]
    kpts = KF.get_laf_center(lafs1).reshape(-1, 2).detach().cpu().numpy()
    return kpts, descriptors, lafs1


def detect_xfeat_dir(img_fnames,
                num_feats=2048,
                device=torch.device('cpu'),
                feature_dir='.featureout', resize_to=(600, 800),
                num_workers=1,
                batch_size=1,
                pin_memory=False,
                quantize=False):

    model = XFeat(top_k=num_feats, detection_threshold=0.0)
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    ds = CustomImageFolderFromFileList(img_fnames,
                                       transform=get_input_xfeat_transform(resize_to))
    dev = device
    dtype = torch.float16 if 'cuda' in str(device) else torch.float32
    bs = batch_size
    dl = DataLoader(ds,
                    batch_size=bs,
                    num_workers=num_workers,
                    collate_fn=collate_with_string, 
                    persistent_workers=True, pin_memory = pin_memory)
    model = model.to(device, dtype)
    with torch.inference_mode():
        with h5py.File(f'{feature_dir}/lafs.h5', mode='w') as f_laf, \
                h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
                h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc, \
                h5py.File(f'{feature_dir}/hw.h5', mode='w') as f_hw:
            for img_batch, hs, ws, fnames in tqdm(dl):
                res = model.detectAndCompute(img_batch.to(dev, dtype), top_k=num_feats)
                for i, img_path in enumerate(fnames):
                    key = img_path
                    keypoints, descriptors = res[i]['keypoints'], res[i]['descriptors']
                    lafs1 = K.feature.laf_from_center_scale_ori(keypoints.reshape(1, -1, 2))
                    if resize_to:
                        lafs1[:, :, 0, :]  *= ws[i] / resize_to[1]
                        lafs1[:, :, 1, :]  *= hs[i] / resize_to[0]
                    desc_dim = descriptors.shape[-1]
                    kpts = KF.get_laf_center(lafs1).reshape(-1, 2).detach().cpu().numpy()
                    descriptors = descriptors.reshape(-1, desc_dim).detach().cpu().numpy()
                    lafs_np = lafs1.detach().cpu().numpy()
                    if quantize:
                        descriptors = quantize_descriptors(descriptors)
                        lafs_np = lafs_np.astype(np.float16)
                        kpts = kpts.astype(np.float16)
                    f_laf[key] = lafs_np
                    f_kp[key] = kpts
                    f_desc[key] = descriptors
                    f_hw[key] = np.array([hs[i], ws[i]])
            print ("Done")
    return

def detect_clidd_single(img, num_feats=2048, resize_to=(800, 600), weights_path=None, device=torch.device('cuda')):
    model = CLIDD(cfg='E128', top_k=num_feats, score=-1e9, weights_path=weights_path).eval().to(device)
    hw1 = torch.tensor(img.shape[:2])
    if resize_to:
        img = cv2.resize(img, resize_to)
        hw1_new = torch.tensor(img.shape[:2], device=device)
    img_t = torch.from_numpy(img).float().permute(2, 0, 1)[None].to(device) / 255.0
    res = model(img_t)
    keypoints = res[0]['keypoints']   # (N, 2)
    descriptors = res[0]['descriptors']  # (N, 128)
    lafs1 = K.feature.laf_from_center_scale_ori(keypoints.reshape(1, -1, 2))
    if resize_to:
        lafs1[:, :, 0, :] *= hw1[1] / hw1_new[1]
        lafs1[:, :, 1, :] *= hw1[0] / hw1_new[0]
    kpts = KF.get_laf_center(lafs1).reshape(-1, 2).detach().cpu().numpy()
    return kpts, descriptors, lafs1


def detect_clidd_dir(img_fnames,
                     num_feats=2048,
                     device=torch.device('cpu'),
                     feature_dir='.featureout', resize_to=(600, 800),
                     num_workers=1,
                     batch_size=1,
                     pin_memory=False,
                     weights_path=None,
                     quantize=False):
    model = CLIDD(cfg='E128', top_k=num_feats, score=-1e9, weights_path=weights_path).eval()
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    ds = CustomImageFolderFromFileList(img_fnames,
                                       transform=get_input_xfeat_transform(resize_to))
    dev = device
    dtype = torch.float16 if 'cuda' in str(device) else torch.float32
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=collate_with_string,
                    persistent_workers=True, pin_memory=pin_memory)
    model = model.to(device, dtype)
    with h5py.File(f'{feature_dir}/lafs.h5', mode='w') as f_laf, \
            h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
            h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc, \
            h5py.File(f'{feature_dir}/hw.h5', mode='w') as f_hw:
        for img_batch, hs, ws, fnames in tqdm(dl):
            res = model(img_batch.to(dev, dtype))
            for i, img_path in enumerate(fnames):
                key = img_path
                keypoints = res[i]['keypoints']
                descriptors = res[i]['descriptors']
                lafs1 = K.feature.laf_from_center_scale_ori(keypoints.reshape(1, -1, 2))
                if resize_to:
                    lafs1[:, :, 0, :] *= ws[i] / resize_to[1]
                    lafs1[:, :, 1, :] *= hs[i] / resize_to[0]
                desc_dim = descriptors.shape[-1]
                kpts = KF.get_laf_center(lafs1).reshape(-1, 2).detach().cpu().numpy()
                descriptors = descriptors.reshape(-1, desc_dim).detach().cpu().numpy()
                lafs_np = lafs1.detach().cpu().numpy()
                if quantize:
                    descriptors = quantize_descriptors(descriptors)
                    lafs_np = lafs_np.astype(np.float16)
                    kpts = kpts.astype(np.float16)
                f_laf[key] = lafs_np
                f_kp[key] = kpts
                f_desc[key] = descriptors
                f_hw[key] = np.array([hs[i], ws[i]])
        print("Done")


def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = kp1[idxs[:, 0]]
    mkpts2 = kp2[idxs[:, 1]]
    return mkpts1, mkpts2

def match_query_to_db(query_desc, query_laf, query_hw, db_dir, fnames, matching_method='smnn', feature_name='xfeat', device=torch.device('cpu'), **kwargs):
    dtype = torch.float16 if 'cuda' in str(device) else torch.float32
    matching_keypoints=[]
    out_hw2 = []
    if matching_method == 'flann':
        FLANN_INDEX_KDTREE = 1  # FLANN_INDEX_KDTREE
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=2)
        search_params = dict(checks=32)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        flann.add([query_desc.detach().float().cpu().numpy()])
    elif matching_method == 'faiss_gpu':
        import faiss
        res = faiss.StandardGpuResources()
        temp_index = faiss.IndexFlatL2(query_desc.shape[-1])
        index = faiss.index_cpu_to_gpu(res, 0, temp_index)
        index.add(query_desc.detach().float().cpu().numpy())
        print ("faiss index created")
    elif matching_method == 'faiss_cpu':
        import faiss
        index = faiss.IndexFlatL2(query_desc.shape[-1])
        index.add(query_desc.detach().float().cpu().numpy())
        print ("faiss CPU index created")
    elif matching_method =='lightglue':
        if feature_name =='xfeat':
            matcher = LighterGlue(device=device).eval().to(dtype)
    if feature_name == 'mast3r':
        shortlist_local_feature_dataset = H5MASt3RLocalFeatureDataset(
            db_dir, kwargs['mast3r_asmk_h5'], fnames, kwargs['fname_to_idx'])
    else:
        shortlist_local_feature_dataset = H5LocalFeatureDataset(db_dir, fnames)
    kp1 = K.feature.get_laf_center(query_laf).reshape(-1, 2)
    batch_size = 2
    if len(kp1) <= 4096:
        batch_size = 16
    if len(kp1) <= 2048:
        batch_size = 32
    if len(kp1) <= 1024:
        batch_size = 64
    if str(device) != 'cuda':
        batch_size = 1
    lf_data_loader = DataLoader(shortlist_local_feature_dataset, batch_size=batch_size, num_workers=2, collate_fn=no_collate)
    for descs2_batch, lafs2_batch, hw2_batch, fnames_batch in tqdm(lf_data_loader):
        if matching_method == 'lightglue':
            #rint ("batched_method")
            if feature_name ==  'xfeat':
                kp2 = K.feature.get_laf_center(torch.cat(lafs2_batch, dim=0).to(device, dtype))
                data = {'descriptors0': query_desc.to(device, dtype)[None].expand(len(fnames_batch), -1, -1),
                        'keypoints0': kp1.to(device, dtype).expand(len(fnames_batch), -1, -1),
                        'image_size0': query_hw.reshape(1,2).flip(1).to(device, dtype).repeat(len(fnames_batch), 1),
                        'descriptors1': torch.stack(descs2_batch, axis=0).to(device, dtype),
                        'keypoints1': kp2,
                        'image_size1': torch.stack(hw2_batch).to(device, dtype).reshape(-1,2).flip(1)}
                out = matcher(data)
                for i in range(len(fnames_batch)):
                    idxs = out["matches"][i].detach().cpu()
                    mkpts1, mkpts2  = get_matching_keypoints(kp1.cpu(), kp2[i].reshape(-1, 2).cpu(), idxs)
                    matching_keypoints.append((mkpts1, mkpts2))
                    out_hw2.append(hw2_batch[i])
        else:
            for i in range(len(fnames_batch)):
                descs2 = descs2_batch[i]
                lafs2 = lafs2_batch[i]
                hw2 = hw2_batch[i]
                fname = fnames_batch[i]
                if matching_method == 'adalam':
                    dists, idxs = K.feature.match_adalam(query_desc, descs2.to(device, dtype),
                                    query_laf,
                                    lafs2.to(device, dtype),  # Adalam takes into account also geometric information
                                    hw1=query_hw, hw2=hw2.to(device, dtype))  # Adalam also benefits from knowing image size
                elif matching_method == 'smnn':
                    matcher = K.feature.match_smnn
                    dists, idxs = matcher(query_desc, descs2.to(device, dtype), 0.99)
                elif matching_method == 'snn':
                    matcher = K.feature.match_snn
                    dists, idxs = matcher(query_desc, descs2.to(device, dtype), 0.95)
                elif matching_method in ['faiss_gpu', 'faiss_cpu']:	
                    D, I = index.search(descs2.cpu().numpy(), 2)
                    idxs = torch.from_numpy(I[:, 0])
                    snn_ratio = D[:, 0] / (1e-8 + D[:, 1])
                    idxs = torch.cat([idxs.reshape(-1, 1), torch.arange(len(idxs)).reshape(-1, 1)], dim=1)
                    mask = snn_ratio <=  0.95
                    idxs = idxs[mask]
                elif matching_method == 'lightglue':
                    if feature_name ==  'xfeat':
                        data = {'descriptors0': query_desc.to(device, dtype)[None],
                                'keypoints0': K.feature.get_laf_center(query_laf).to(device, dtype),
                                'image_size0': query_hw.reshape(1,2).flip(1).to(device, dtype),
                                'descriptors1': descs2.to(device, dtype)[None],
                                'keypoints1': K.feature.get_laf_center(lafs2.to(device, dtype)),
                                'image_size1': hw2.to(device, dtype).reshape(1,2).flip(1)}
                        #import pdb; pdb.set_trace()
                        out = matcher(data)	
                        idxs = out["matches"][0].detach().cpu()
                elif matching_method == 'flann':
                    matches = flann.knnMatch(descs2.float().numpy(), k=2)
                    valid_matches = []
                    for cur_match in matches:
                        tmp_valid_matches = [
                            nn_1 for nn_1, nn_2 in zip(cur_match[:-1], cur_match[1:])
                            if nn_1.distance <= 0.95 * nn_2.distance
                        ]
                        valid_matches.extend(tmp_valid_matches)
                    dists, idxs = kornia_matches_from_cv2(valid_matches)
                    idxs = idxs.flip(1)
                else:
                    raise NotImplementedError
                kp2 = K.feature.get_laf_center(lafs2).reshape(-1, 2)
                mkpts1, mkpts2  = get_matching_keypoints(kp1, kp2, idxs.cpu())
                matching_keypoints.append((mkpts1, mkpts2))
                out_hw2.append(hw2)
    return matching_keypoints, out_hw2


def match_feature_pair(fname1: str, fname2: str, feature_dir: str,
                       matching_method: str = 'smnn',
                       device: torch.device = torch.device('cpu')):
    """
    Match local features between two DB images stored in feature_dir h5 files.

    Returns:
        mkpts1 (Tensor, N×2), mkpts2 (Tensor, N×2), hw1 (Tensor), hw2 (Tensor)
    """
    dtype = torch.float16 if 'cuda' in str(device) else torch.float32
    with h5py.File(os.path.join(feature_dir, 'descriptors.h5'), 'r') as f_desc, \
         h5py.File(os.path.join(feature_dir, 'lafs.h5'), 'r') as f_laf, \
         h5py.File(os.path.join(feature_dir, 'hw.h5'), 'r') as f_hw:
        descs1 = torch.from_numpy(f_desc[fname1][...].astype(np.float32)).to(device, dtype)
        descs2 = torch.from_numpy(f_desc[fname2][...].astype(np.float32)).to(device, dtype)
        lafs1 = torch.from_numpy(f_laf[fname1][...]).to(device, dtype)
        lafs2 = torch.from_numpy(f_laf[fname2][...]).to(device, dtype)
        hw1 = torch.from_numpy(f_hw[fname1][...])
        hw2 = torch.from_numpy(f_hw[fname2][...])

    kp1 = KF.get_laf_center(lafs1).reshape(-1, 2)
    kp2 = KF.get_laf_center(lafs2).reshape(-1, 2)

    if matching_method == 'smnn':
        dists, idxs = K.feature.match_smnn(descs1, descs2, 0.99)
    elif matching_method == 'snn':
        dists, idxs = K.feature.match_snn(descs1, descs2, 0.95)
    elif matching_method in ('faiss_gpu', 'faiss_cpu'):
        import faiss
        if matching_method == 'faiss_gpu':
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(descs1.shape[-1]))
        else:
            index = faiss.IndexFlatL2(descs1.shape[-1])
        index.add(descs1.float().cpu().numpy())
        D, I = index.search(descs2.float().cpu().numpy(), 2)
        snn_ratio = D[:, 0] / (1e-8 + D[:, 1])
        valid = snn_ratio <= 0.95
        idxs = torch.from_numpy(
            np.stack([I[:, 0], np.arange(len(I))], axis=1)[valid])
    else:
        raise NotImplementedError(f"matching_method={matching_method!r}")

    mkpts1 = kp1[idxs[:, 0]].cpu()
    mkpts2 = kp2[idxs[:, 1]].cpu()
    return mkpts1, mkpts2, hw1, hw2


def get_scale_factor(H):
    # Normalize the Homography matrix
    H = H / H[2, 2]
    # Extract the 2x2 linear transformation part
    A = H[:2, :2]
    # Compute SVD of H_2x2
    U, S, Vt = np.linalg.svd(A)
    # Singular values are in S
    sigma1, sigma2 = S
    # Compute the scale factor as the geometric mean
    scale_factor = np.sqrt(sigma1 * sigma2)
    return scale_factor


def get_convex_hull_area(points):
    # Ensure points are in the correct shape for cv2.convexHull (Nx1x2)
    points = np.array(points, dtype=np.float32).reshape(-1, 1,  2)
    hull = cv2.convexHull(points)
    area = cv2.contourArea(hull)
    return area

def get_scale_factor_via_convex_hull(kpts1, kpts2):
    area1 = get_convex_hull_area(kpts1)
    area2 = get_convex_hull_area(kpts2)
    area_ratio = area2 / area1
    scale_factor = np.sqrt(area_ratio)
    return scale_factor


def spatial_scoring(matching_keypoints, criterion='num_inliers', config={"inl_th": 3.0, "num_iter": 1000}):
    new_shortlist_scores = []
    Hs = []
    for i, (mkpts1, mkpts2) in enumerate(tqdm(matching_keypoints)):
        if len(mkpts1) < 50:
            new_shortlist_scores.append(0)
            Hs.append(np.zeros((3, 3)))
            continue
        if criterion == 'scale_factor_min':
            H, inliers = cv2.findHomography(
                mkpts2.detach().cpu().numpy(),
                mkpts1.detach().cpu().numpy(),
                cv2.USAC_MAGSAC,
                config['inl_th'],
                0.999,
                config['num_iter']
            )
            if H is not None:
                H = np.linalg.inv(H)
            else:
                H = np.zeros((3, 3))
        else:
            H, inliers = cv2.findHomography(
                mkpts1.detach().cpu().numpy(),
                mkpts2.detach().cpu().numpy(),
                cv2.USAC_MAGSAC,
                config['inl_th'],
                0.999,
                config['num_iter']
            )            
        inliers = (inliers > 0).reshape(-1)
        num_inl = inliers.sum()
        fail = False
        if H is None:
            H = np.zeros((3, 3))
            fail = True
        if num_inl>50:
            if criterion == 'num_inliers':
                new_shortlist_scores.append(num_inl)
            elif criterion == 'scale_factor_min':
                print (mkpts2.detach().cpu().numpy().shape, inliers.shape)
                scale_factor = get_scale_factor_via_convex_hull(mkpts2.detach().cpu().numpy()[inliers],
                                                mkpts1.detach().cpu().numpy()[inliers])
                if (scale_factor <= 0.1) or scale_factor > 10:
                    scale_factor = 0.0
                print (f"Scale factor: {scale_factor}, num_inliers: {num_inl}")
                new_shortlist_scores.append(scale_factor)
            elif criterion == 'scale_factor_max':
                scale_factor = get_scale_factor_via_convex_hull(mkpts1.detach().cpu().numpy()[inliers],
                                                mkpts2.detach().cpu().numpy()[inliers])
                if (scale_factor <= 0.12) or scale_factor > 8:
                    scale_factor = 0.0
                print (f"Scale factor: {scale_factor}, num_inliers: {num_inl}")
                new_shortlist_scores.append(scale_factor)
        else:
            print ("Too little inliers")
            if criterion == 'num_inliers':
                new_shortlist_scores.append(num_inl)
            else:
                new_shortlist_scores.append(0)
        Hs.append(H)
    new_shortlist_scores = np.array(new_shortlist_scores)
    return new_shortlist_scores, Hs