import multiprocessing as mp
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch
from time import time
from tqdm import tqdm
import numpy as np
import os
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset
import os
from typing import List, Tuple, Callable, Optional
from PIL import Image
from torch.utils.data import Dataset
import h5py
from scipy.sparse import csr_matrix
from simple_retrieval.global_feature import get_dinov2salad, get_input_transform_siglip, get_input_transform_dinosalad, get_input_transform_dinov3large, dataset_inference, siglip2, DINOv3Large, DINOv3LargeGeM
from simple_retrieval.local_feature import detect_sift_single, detect_sift_dir, detect_xfeat_single, detect_xfeat_dir, detect_clidd_single, detect_clidd_dir, match_query_to_db, spatial_scoring
from simple_retrieval.mast3r_feature import MASt3RASMKRetrieval, detect_mast3r_single, detect_mast3r_dir
from simple_retrieval.sift_asmk import SIFTASMKRetrieval
from simple_retrieval.pile_of_garbage import CustomImageFolder
from simple_retrieval.manifold_diffusion import sim_kernel, sim_kernel_torch, topK_to_csr, get_W_sparse, normalize_connection_graph, topK_W, cg_diffusion
import cv2
# VLM resorting — imported lazily to keep startup cost low
_vlm_resort = None


def get_default_config():
    conf = {"local_features": "xfeat",
            "quantize_local_desc": False,
            "global_features": "dinosalad",
            "inl_th": 2.0,
            "num_iter": 2000,
            "num_local_features": 4096,
            "local_desc_image_size": (800, 600),
            "local_desc_batch_size": 2,
            "num_workers": 1,
            "use_diffusion": False,
            "global_desc_image_size": 448,
            "global_desc_batch_size": 2,
            "device": "cpu",
            "force_recache": False,
            "ransac_type": "homography",
            "matching_method": "smnn",
            "num_nn": 100,
            "resort_criterion": "scale_factor_max",
            # VLM resorting options (used when resort_criterion starts with 'vlm_')
            "vlm_model": "qwen2vl-2b",   # model shortcut or full HF path
            "vlm_bbox": None,             # [x1,y1,x2,y2] query ROI, or None
            # SIFT+ASMK options (used when global_features='sift_asmk')
            "sift_asmk_vocab_size": 65536,
            "sift_asmk_sample_n": 500_000,
            "sift_asmk_topk": 1000,
            "sift_asmk_binary": True,
            "sift_asmk_gpu_id": None,
            "sift_asmk_scale_sigma": 1.0,
            "sift_asmk_multiple_assignment": 1,
            # HQE options
            "hqe_topk_verify": 50,
            "hqe_max_iters": 2,
            "hqe_overlap_thresh": 0.5,
            "hqe_consistency_thresh": 0.1,
            }
    return conf
    

class SimpleRetrieval:
    def __init__(self, img_dir=None, index_dir=None, config = get_default_config()):
        self.img_dir = img_dir
        self.index_dir = index_dir
        self.config = config
        dev = torch.device(self.config["device"])
        dtype = torch.float16 if 'cuda' in self.config["device"] else torch.float32
        img_size = self.config["global_desc_image_size"]
        if config.get("global_features") == "dinosalad":
            self.global_model = get_dinov2salad(device=dev, dtype=dtype).eval()
            self.global_transform = get_input_transform_dinosalad((img_size, img_size))
        elif config.get("global_features") == "dinov3large":
            self.global_model = DINOv3Large(device=dev).to(device=dev, dtype=dtype).eval()
            self.global_transform = get_input_transform_dinov3large((img_size, img_size))
        elif config.get("global_features") == "dinov3large_gem":
            self.global_model = DINOv3LargeGeM(device=dev).to(device=dev, dtype=dtype).eval()
            self.global_transform = get_input_transform_dinov3large((img_size, img_size))
        elif config.get("global_features") == "mast3r_asmk":
            self.mast3r_retrieval = MASt3RASMKRetrieval(
                mast3r_dir=config.get("mast3r_dir", "../mast3r"),
                retrieval_model_path=config.get("mast3r_retrieval_model",
                    "../mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"),
                device=config.get("device", "cuda"),
            )
            self.global_model = None
            self.global_transform = None
        elif config.get("global_features") == "sift_asmk":
            self.sift_asmk = SIFTASMKRetrieval(
                vocab_size=config.get("sift_asmk_vocab_size", 65536),
                topk=config.get("sift_asmk_topk", 1000),
                binary=config.get("sift_asmk_binary", True),
                gpu_id=config.get("sift_asmk_gpu_id", None),
                scale_sigma=config.get("sift_asmk_scale_sigma", 1.0),
                multiple_assignment=config.get("sift_asmk_multiple_assignment", 1),
            )
            self.global_model = None
            self.global_transform = None
        else:
            self.global_model  = siglip2(device=dev).eval()
            self.global_transform = get_input_transform_siglip((384, 384))
        return
    
    def __repr__(self):
        return f"""SimpleRetrieval(img_dir={self.img_dir},
        index_dir={self.index_dir}, config={self.config})"""
    
    def get_cache_dir_name(self, img_dir):
        if self.index_dir is not None:
            return self.index_dir
        return f"./tmp/{img_dir.replace('/', '_')}"
    
    def get_global_index_fname(self, img_dir):
        return os.path.join(self.get_cache_dir_name(img_dir), f"{self.config['global_features']}_global_index.pth")
    
    def get_global_index_Wn_name(self, img_dir):
        return os.path.join(self.get_cache_dir_name(img_dir),  f"{self.config['global_features']}_Wn.pth")

    def get_random_shortlist(self, num_nn = 100):
        """Returns a random shortlist of images.
        Args:
            num_nn (int): The number of nearest neighbors to return.
        Returns:
            idxs (np.ndarray): The indices of the nearest neighbors.
            sims (np.ndarray): The similarity score of the nearest neighbors.
        """
        idxs = np.random.choice(len(self.ds), num_nn, replace=False)
        sims = np.random.rand(num_nn)
        return idxs, sims

    def get_local_feature_dir(self, img_dir):
        local_desc_name = f"{self.config['local_features']}_{self.config['num_local_features']}"
        if self.config.get("quantize_local_desc", False):
            local_desc_name += "_uint8"
        return os.path.join(self.config['local_desc_dir'], local_desc_name)

    def create_global_descriptor_index(self, img_dir, index_dir):
        """Creates a global descriptor index from a directory of images."""
        index_dir = self.get_cache_dir_name(img_dir)
        os.makedirs(index_dir, exist_ok=True)
        self.ds = CustomImageFolder(img_dir, transform=self.global_transform)

        if self.config.get("global_features") == "mast3r_asmk":
            h5_path = os.path.join(index_dir, "mast3r_asmk.h5")
            self.mast3r_asmk_h5_path = h5_path
            if os.path.exists(h5_path) and not self.config["force_recache"]:
                print(f"Loading MASt3R ASMK features from {h5_path}")
                self.mast3r_retrieval.rebuild_ivf(h5_path)
            else:
                print(f"Extracting MASt3R ASMK features for {len(self.ds.samples)} images")
                lf_dir = None
                if self.config.get("local_features") == "mast3r":
                    lf_dir = self.get_local_feature_dir(img_dir)
                self.mast3r_retrieval.index_images(
                    self.ds.samples, h5_path=h5_path, local_feature_dir=lf_dir, batch_size=self.config["local_desc_batch_size"], num_workers=self.config["num_workers"])
                print(f"Features saved to {h5_path}")
            self.Wn = None
            return

        if self.config.get("global_features") == "sift_asmk":
            # SIFT+ASMK: local features serve as both global retrieval and local reranking.
            # Step 1: ensure SIFT features are extracted.
            local_feat_dir = self.get_local_feature_dir(img_dir)
            os.makedirs(local_feat_dir, exist_ok=True)
            sentinel = os.path.join(local_feat_dir, 'descriptors.h5')
            if not os.path.exists(sentinel) or self.config["force_recache"]:
                print(f"Extracting SIFT features for {len(self.ds.samples)} images...")
                detect_sift_dir(self.ds.samples,
                                feature_dir=local_feat_dir,
                                num_feats=self.config["num_local_features"],
                                resize_to=self.config["local_desc_image_size"])
            else:
                print(f"SIFT features already exist in {local_feat_dir}")
            self.local_feature_dir = local_feat_dir

            # Step 2: train vocabulary + build IVF (both cache-backed).
            vocab_cache = os.path.join(index_dir, "sift_asmk_vocab.pkl")
            ivf_cache   = os.path.join(index_dir, "sift_asmk_ivf.pkl")
            aux_path    = os.path.join(index_dir, "sift_asmk_aux.pkl")

            if os.path.exists(vocab_cache) and os.path.exists(ivf_cache) \
                    and os.path.exists(aux_path) and not self.config["force_recache"]:
                print("Loading SIFT+ASMK index from cache...")
                self.sift_asmk.load_aux(aux_path)
                self.sift_asmk.rebuild_ivf(vocab_cache, ivf_cache)
            else:
                self.sift_asmk.train_vocabulary(
                    local_feat_dir, self.ds.samples,
                    sample_n=self.config.get("sift_asmk_sample_n", 500_000),
                    cache_path=vocab_cache)
                self.sift_asmk.build_ivf(
                    local_feat_dir, self.ds.samples,
                    cache_path=ivf_cache)
                self.sift_asmk.save_aux(aux_path)
            self.Wn = None
            return

        global_index_fname = self.get_global_index_fname(img_dir)
        Wn_fname = self.get_global_index_Wn_name(img_dir)
        
        if os.path.exists(global_index_fname) and not self.config["force_recache"]:
            print (f"Loading global index from {global_index_fname}")
            self.global_descs = torch.load(global_index_fname, weights_only=False)
           
        else:
            print (f"Creating global index from images in: {img_dir}")
            self.global_descs = dataset_inference(self.global_model,
                                                        self.ds,
                                                        batch_size=self.config["global_desc_batch_size"],
                                                        device=self.config["device"],
                                                        num_workers=self.config["num_workers"])
            torch.save(self.global_descs, global_index_fname)
            print (f"Global index saved to:  {global_index_fname}")
            self.global_descs = torch.load(global_index_fname, weights_only=False)
        print (self.global_descs.shape, self.global_descs.dtype)
        print (self.global_descs[0])
        if os.path.exists(Wn_fname) and not self.config["force_recache"]:
            self.Wn = torch.load(Wn_fname, weights_only=False)
        else:
            self.Wn = self.get_Wn(self.global_descs.T, K = 100)
            torch.save(self.Wn, Wn_fname)
        return 

    def get_Wn(self, X, K = 100, max_size = 1000):
        print (f"Computing Wn for {X.shape} samples")
        num_samples = X.shape[1]
        if num_samples > max_size:
            W = get_W_sparse(X, K)
            W = W.minimum(W.T)
        else:
            A = np.dot(X.T, X)
            W = sim_kernel(A)
            W = topK_W(W, K).astype(np.float32)
        Wn = normalize_connection_graph(W)
        return Wn

    def create_local_descriptor_index(self, img_dir):
        # Placeholder function for creating a local descriptor index
        self.local_feature_dir = self.get_local_feature_dir(img_dir)
        t=time()
        # sift_asmk already extracts SIFT in create_global_descriptor_index
        if self.config.get("global_features") == "sift_asmk":
            print(f"SIFT features managed by sift_asmk pipeline, already in {self.local_feature_dir}")
            return
        sentinel = 'keypoints.h5' if self.config["local_features"] == "mast3r" else 'descriptors.h5'
        if (not os.path.exists(os.path.join(self.local_feature_dir, sentinel))) or (self.config["force_recache"]):
            fnames_list = self.ds.samples
            if self.config["local_features"] == "sift":
                detect_sift_dir(fnames_list, feature_dir=self.local_feature_dir, num_feats=self.config["num_local_features"], device=self.config["device"],
                                resize_to=self.config["local_desc_image_size"])
            elif self.config["local_features"] == "xfeat":
                detect_xfeat_dir(fnames_list,
                                 feature_dir=self.local_feature_dir,
                                 num_feats=self.config["num_local_features"],
                                 resize_to=self.config["local_desc_image_size"],
                                 device=self.config["device"],
                                 batch_size=self.config["local_desc_batch_size"],
                                 num_workers=self.config["num_workers"],
                                 pin_memory=False,
                                 quantize=self.config["quantize_local_desc"])
            elif self.config["local_features"] == "clidd":
                detect_clidd_dir(fnames_list,
                                 feature_dir=self.local_feature_dir,
                                 num_feats=self.config["num_local_features"],
                                 resize_to=self.config["local_desc_image_size"],
                                 device=self.config["device"],
                                 batch_size=self.config["local_desc_batch_size"],
                                 num_workers=self.config["num_workers"],
                                 pin_memory=False,
                                 quantize=self.config["quantize_local_desc"])
            elif self.config["local_features"] == "mast3r":
                assert hasattr(self, 'mast3r_retrieval'), \
                    "local_features='mast3r' requires global_features='mast3r_asmk'"
                detect_mast3r_dir(fnames_list,
                                  model=self.mast3r_retrieval.retriever.model,
                                  imsize=self.mast3r_retrieval.retriever.imsize,
                                  feature_dir=self.local_feature_dir,
                                  batch_size=self.config["local_desc_batch_size"],
                                  num_workers=self.config["num_workers"],
                                  device=torch.device(self.config["device"]))
            print (f"{self.config['local_features']} index of created from images in: {img_dir} in {time()-t:.2f} sec, saved to {self.local_feature_dir}")
        else:
            print (f"Local index already exists in {self.local_feature_dir}")
        return 

    def describe_query(self, img):
        dev = torch.device(self.config["device"])
        dtype = torch.float16 if 'cuda' in str(self.config["device"]) else torch.float32
        t = time()
        model = self.global_model.to(dev, dtype)
        model.eval()
        with torch.inference_mode():
            global_desc = model(self.global_transform(img)[None].to(dev, dtype)).cpu()
        print (f"Describe query in: {time()-t:.2f} sec")
        return global_desc.reshape(1, -1).detach().cpu().numpy()
    
    def get_shortlist(self, img, num_nn = 1000, manifold_diffusion=False, Wn=None):
        """Returns a shortlist of images based on the global similarity query image.
        Args:
            query_fname (str): The filename of the query image.
            num_nn (int): The number of nearest neighbors to return.
        Returns:
            idxs (np.ndarray): The indices of the nearest neighbors.
            sims (np.ndarray): The similarity score of the nearest neighbors.
        """
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        if isinstance(img, np.ndarray):
            img_np = img
            img = Image.fromarray(img)
        else:
            img_np = np.array(img)
        t=time()
        if self.config.get("global_features") == "mast3r_asmk":
            ranks, scores = self.mast3r_retrieval.query(img_np)
            idxs = np.array(ranks[:num_nn], dtype=np.int64)
            out_vals = np.array(scores[:num_nn], dtype=np.float32)
            print(f"MASt3R ASMK shortlist in: {time()-t:.2f} sec")
            return idxs, out_vals

        if self.config.get("global_features") == "sift_asmk":
            _, q_descs, q_lafs = detect_sift_single(
                img_np,
                num_feats=self.config["num_local_features"],
                resize_to=self.config["local_desc_image_size"])
            q_descs_np = q_descs.float().cpu().numpy()
            ranks, scores = self.sift_asmk.query(q_descs_np, topk=num_nn)
            scores = self.sift_asmk.scale_biased_rescore(q_lafs, ranks, scores)
            print(f"SIFT+ASMK shortlist in: {time()-t:.2f} sec")
            return ranks[:num_nn], scores[:num_nn]

        query = self.describe_query(img)
        if manifold_diffusion:
            print("Diffusion")
            Q = query.reshape(-1, 1).astype(np.float32)
            X = self.global_descs.T
            K = 100 # approx 50 mutual nns
            QUERYKNN = 50
            R = 2000
            alpha = 0.9
            sim  = np.dot(X.T, Q)
            qsim = sim_kernel(sim).T
            sortidxs = np.argsort(-qsim, axis = 1)
            for i in range(len(qsim)):
                qsim[i,sortidxs[i,QUERYKNN:]] = 0
            if Wn is None:
                W = sim_kernel(np.dot(X.T, X))
                W = topK_W(W, K)
                Wn = normalize_connection_graph(W)
            cg_sims = cg_diffusion(qsim, Wn, alpha)
            dists = -cg_sims.reshape(-1)
            idxs = np.argsort(dists)[:num_nn]
            out_vals = (2-dists[idxs])/2.0
        else:
            if torch.cuda.is_available() and len(self.global_descs) > 10000:
                dists = torch.norm(torch.from_numpy(self.global_descs).to(torch.device('cuda'), torch.float16) - torch.from_numpy(query).to(torch.device('cuda'), torch.float16), dim=1)
                vals, idxs = torch.topk(dists, num_nn, dim=0, largest=False)
                vals = vals.detach().cpu().numpy()
                idxs = idxs.detach().cpu().numpy()
                out_vals = (2-vals)/2.0
            else:
                dists = np.linalg.norm(self.global_descs - query, axis=1)
                idxs = np.argsort(dists)[:num_nn]
                out_vals = (2-dists[idxs])/2.0

        #print (f"Distances: {dists[idxs]}")
        print (f"Shortlist in: {time()-t:.2f} sec")
        return idxs, out_vals
    
    def resort_shortlist(self, query, shortlist, criterion='num_inliers', device='cpu',
                         matching_method='smnn', bbox=None, vlm_model=None,
                         vlm_collective=True):
        """Re-rank a shortlist of retrieved images.

        For geometry-based criteria ('num_inliers', 'scale_factor_max',
        'scale_factor_min') local features + RANSAC are used.

        For VLM-based criteria ('vlm_zoom_in', 'vlm_zoom_out', 'vlm_relevant')
        a vision-language model is used instead; in this case `corners_norm` is
        returned as None.

        Args:
            query:          Query image as np.ndarray (H,W,3) RGB.
            shortlist:      1-D int array of DB indices from get_shortlist.
            criterion:      Scoring criterion (see above).
            device:         'cuda' or 'cpu'.
            matching_method: Local feature matching method (geometry path only).
            bbox:           Optional [x1,y1,x2,y2] query ROI for VLM path.
            vlm_model:      VLM model name override; falls back to config value.

        Returns:
            sorted_shortlist (np.ndarray), scores (np.ndarray), corners_norm
        """
        if criterion.startswith("vlm_"):
            return self._resort_shortlist_vlm(
                query, shortlist, criterion=criterion, device=device,
                bbox=bbox, vlm_model=vlm_model, collective=vlm_collective)

        # --- Geometry path ---
        new_shortlist_scores = []
        matching_keypoints = []
        dtype = torch.float16 if 'cuda' in str(device) else torch.float32
        hw1 = torch.tensor(query.shape[:2]).to(device, dtype)
        hq, wq = query.shape[:2]
        if self.config["local_features"] == "sift":
            kpt, descs, lafs1 = detect_sift_single(query,
                                                   num_feats=self.config["num_local_features"],
                                                   resize_to=self.config["local_desc_image_size"])
        elif self.config["local_features"] == "xfeat":
            kpt, descs, lafs1 = detect_xfeat_single(query,
                                                    num_feats=self.config["num_local_features"],
                                                    resize_to=self.config["local_desc_image_size"])
        elif self.config["local_features"] == "clidd":
            kpt, descs, lafs1 = detect_clidd_single(query,
                                                    num_feats=self.config["num_local_features"],
                                                    resize_to=self.config["local_desc_image_size"],
                                                    device=torch.device(device))
        elif self.config["local_features"] == "mast3r":
            assert hasattr(self, 'mast3r_retrieval'), \
                "local_features='mast3r' requires global_features='mast3r_asmk'"
            kpt, descs, lafs1 = detect_mast3r_single(
                query,
                model=self.mast3r_retrieval.retriever.model,
                imsize=self.mast3r_retrieval.retriever.imsize,
                device=torch.device(device))
        descs = descs.to(device, dtype)
        lafs1 = lafs1.to(device, dtype)
        fnames = [self.ds.samples[i] for i in shortlist]
        tt = time()
        extra_kwargs = {}
        if self.config['local_features'] == 'mast3r':
            if not hasattr(self, '_fname_to_idx'):
                self._fname_to_idx = {p: i for i, p in enumerate(self.ds.samples)}
            extra_kwargs = {'mast3r_asmk_h5': self.mast3r_asmk_h5_path,
                            'fname_to_idx':   self._fname_to_idx}
        with torch.inference_mode():
            matching_keypoints, hw2_s = match_query_to_db(descs, lafs1, hw1,
                                               self.local_feature_dir,
                                               fnames,
                                               feature_name=self.config['local_features'],
                                               matching_method=matching_method,
                                               device=torch.device(device),
                                               **extra_kwargs)
        print(f"Matching {matching_method} in {time()-tt:.4f} sec")
        tt = time()
        new_shortlist_scores, Hs = spatial_scoring(
            matching_keypoints, criterion=criterion, config=self.config)

        # Build normalised corner projections for visualisation
        topleft     = np.array([0,  0,  1])
        bottomleft  = np.array([0,  hq, 1])
        topright    = np.array([wq, 0,  1])
        bottomright = np.array([wq, hq, 1])
        corners = np.stack([topleft, bottomleft, bottomright, topright, topleft], axis=0)
        Hs_arr = np.array(Hs)   # (N,3,3)
        proj = np.einsum("nij,kj->nki", Hs_arr, corners)  # (N,5,3)
        z = proj[:, :, 2:3]
        # Avoid divide-by-zero for failed homographies
        safe_z = np.where(np.abs(z) < 1e-8, 1.0, z)
        proj_norm = proj[:, :, :2] / safe_z
        wh2 = np.array(hw2_s)[:, ::-1][:, np.newaxis, :]   # (N,1,2)
        corners_norm = proj_norm / np.where(wh2 == 0, 1.0, wh2)

        print(f"RANSAC in {time()-tt:.4f} sec")
        sorted_idxs = np.argsort(new_shortlist_scores)[::-1]
        return (shortlist[sorted_idxs],
                np.array(new_shortlist_scores)[sorted_idxs],
                corners_norm[sorted_idxs])

    def hqe_query(self, img, topk_asmk: int = None, topk_verify: int = None,
                  max_hqe_iters: int = None, device: str = None,
                  matching_method: str = None) -> tuple:
        """
        Full HQE pipeline (Phases 1–4) for SIFT+ASMK global_features.

        Runs ASMK retrieval, scale-biased re-scoring, RANSAC verification,
        result grouping, geometric consistency check, and iterative query
        expansion.

        Args:
            img: Query image — path (str), PIL Image, or np.ndarray (H,W,3 RGB).
            topk_asmk:        Number of ASMK candidates per iteration (default from config).
            topk_verify:      Number of candidates for RANSAC verification (default from config).
            max_hqe_iters:    Maximum HQE expansion iterations (default from config).
            device:           'cuda' or 'cpu' (default from config).
            matching_method:  Local matching method (default from config).
        Returns:
            sorted_idxs  (np.ndarray): DB image indices, best-first.
            final_scores (np.ndarray): Corresponding scores.
        """
        assert self.config.get("global_features") == "sift_asmk", \
            "hqe_query() requires global_features='sift_asmk'"

        if isinstance(img, str):
            img = np.array(__import__('PIL').Image.open(img).convert("RGB"))
        elif not isinstance(img, np.ndarray):
            img = np.array(img)

        dev = torch.device(device or self.config["device"])
        mm  = matching_method or self.config.get("matching_method", "smnn")

        _, q_descs, q_lafs = detect_sift_single(
            img,
            num_feats=self.config["num_local_features"],
            resize_to=self.config["local_desc_image_size"])

        q_hw = np.array(img.shape[:2])

        idxs, scores = self.sift_asmk.hqe_query(
            q_descs=q_descs.float().cpu().numpy(),
            q_lafs=q_lafs.cpu().numpy(),
            q_hw=q_hw,
            feature_dir=self.local_feature_dir,
            topk_asmk=topk_asmk or self.config.get("sift_asmk_topk", 1000),
            topk_verify=topk_verify or self.config.get("hqe_topk_verify", 50),
            max_hqe_iters=max_hqe_iters if max_hqe_iters is not None
                          else self.config.get("hqe_max_iters", 2),
            overlap_thresh=self.config.get("hqe_overlap_thresh", 0.5),
            consistency_thresh=self.config.get("hqe_consistency_thresh", 0.1),
            device=dev,
            matching_method=mm,
            inl_th=self.config.get("inl_th", 3.0),
            num_ransac_iter=self.config.get("num_iter", 1000),
        )
        return idxs, scores

    def _resort_shortlist_vlm(self, query, shortlist, criterion='vlm_zoom_in',
                               device='cuda', bbox=None, vlm_model=None,
                               collective=True):
        """VLM-based resorting. Returns (sorted_shortlist, scores, None).

        Args:
            collective: If True (default), send all candidates in one/few VLM
                        calls with a disk cache (faster on repeated queries).
                        If False, run one VLM call per candidate (no caching).
        """
        model_name = vlm_model or self.config.get("vlm_model", "qwen2vl-2b")
        query_bbox = bbox if bbox is not None else self.config.get("vlm_bbox")
        if collective:
            from simple_retrieval.llms import resort_shortlist_vlm_collective as _fn
            cache_dir = self.config.get("vlm_cache_dir", "./tmp/vlm_cache")
            max_per = self.config.get("vlm_max_images_per_call", 8)
            sorted_sl, scores = _fn(
                query_img=query,
                shortlist=shortlist,
                db_fnames=self.ds.samples,
                model_name=model_name,
                criterion=criterion,
                bbox=query_bbox,
                device=device,
                max_images_per_call=max_per,
                cache_dir=cache_dir,
            )
        else:
            from simple_retrieval.llms import resort_shortlist_vlm as _fn
            sorted_sl, scores = _fn(
                query_img=query,
                shortlist=shortlist,
                db_fnames=self.ds.samples,
                model_name=model_name,
                criterion=criterion,
                bbox=query_bbox,
                device=device,
            )
        return sorted_sl, scores, None

def main():
    # Example usage of the retrieve_data function
    r = SimpleRetrieval()
    print (r)
    r.create_global_descriptor_index('/Users/oldufo/datasets/goose',
                                     './tmp/global_desc')
    r.create_local_descriptor_index('/Users/oldufo/datasets/goose')
    query_fname = '/Users/oldufo/datasets/goose/goose1.png'
    
    #r.create_global_descriptor_index('/Users/oldufo/datasets/oxford5k',
    #                                 './tmp/global_desc_ox5k')
    #r.create_local_descriptor_index('/Users/oldufo/datasets/oxford5k')
    #query_fname = '/Users/oldufo/datasets/oxford5k/all_souls_000006.jpg'

    #query_fname = '/Users/oldufo/datasets/EVD/1/graf.png'
    
    shortlist_idxs, shortlist_scores = r.get_shortlist(query_fname, num_nn=r.config["num_nn"], manifold_diffusion=r.config["use_diffusion"])
    fnames = r.ds.samples  
    q_img = cv2.cvtColor(cv2.imread(query_fname), cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        idxs, scores, corners_norm = r.resort_shortlist(
            q_img, shortlist_idxs,
            matching_method=r.config["matching_method"],
            criterion=r.config["resort_criterion"])
    print([fnames[i] for i in idxs], scores)

if __name__ == "__main__":
    main()