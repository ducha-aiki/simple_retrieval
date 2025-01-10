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

from simple_retrieval.global_feature import get_dinov2salad, get_input_transform, dataset_inference
from simple_retrieval.local_feature import detect_sift_single, detect_sift_dir, detect_xfeat_single, detect_xfeat_dir, get_matching_keypoints
from simple_retrieval.pile_of_garbage import CustomImageFolder
import cv2
from kornia_moons.feature import kornia_matches_from_cv2


def get_default_config():
    conf = {"local_features": "xfeat",
            "global_features": "dinosalad",
            "inl_th": 2.0,
            "num_iter": 2000,
            "num_local_features": 4096,
            "local_desc_image_size": (1024,768),
            
            "global_desc_image_size": 448,
            "global_desc_batch_size": 2,
            "device": "mps",
            "force_recache": False,
            "ransac_type": "homography",
            "matching_method": "smnn",
            "num_nn": 100,
            "use_diffusion": False,
            "resort_criterion": "scale_factor_max"}
    return conf
    
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

class SimpleRetrieval:
    def __init__(self, img_dir=None, index_dir=None, config = get_default_config()):
        self.img_dir = img_dir
        self.index_dir = index_dir
        self.config = config
        dev = torch.device(self.config["device"])
        dtype = torch.float16 if 'cuda' in self.config["device"] else torch.float32
        self.global_model = get_dinov2salad(device=dev, dtype=dtype).eval()
        img_size = self.config["global_desc_image_size"]
        self.global_transform = get_input_transform((img_size, img_size))
        return
    
    def __repr__(self):
        return f"""SimpleRetrieval(img_dir={self.img_dir},
        index_dir={self.index_dir}, config={self.config})"""
    
    def get_cache_dir_name(self, img_dir):
        return f"./tmp/{img_dir.replace('/', '_')}"
    
    def get_global_index_fname(self, img_dir):
        return os.path.join(self.get_cache_dir_name(img_dir), "global_index.pth")

    def get_local_feature_dir(self, img_dir):
        local_desc_name = f"{self.config['local_features']}_{self.config['num_local_features']}"
        return os.path.join(self.get_cache_dir_name(img_dir), local_desc_name)

    def create_global_descriptor_index(self, img_dir, index_dir):
        """Creates a global descriptor index from a directory of images."""
        index_dir = self.get_cache_dir_name(img_dir)
        os.makedirs(index_dir, exist_ok=True)
        self.ds = CustomImageFolder(img_dir, transform=self.global_transform)
        global_index_fname = self.get_global_index_fname(img_dir)
        if os.path.exists(global_index_fname) and not self.config["force_recache"]:
            print (f"Loading global index from {global_index_fname}")
            self.global_descs = torch.load(global_index_fname)
        else:
            print (f"Creating global index from images in: {img_dir}")
            self.global_descs = dataset_inference(self.global_model,
                                                        self.ds,
                                                        batch_size=self.config["global_desc_batch_size"],
                                                        device=self.config["device"],
                                                        num_workers=1)
            print (self.global_descs[0])
            torch.save(self.global_descs, global_index_fname)
            print (f"Global index saved to:  {global_index_fname}")
            self.global_descs = torch.load(global_index_fname)
        print (self.global_descs[0])
        print (self.global_descs.shape, self.global_descs.dtype)
        return 

    def create_local_descriptor_index(self, img_dir):
        # Placeholder function for creating a local descriptor index
        self.local_feature_dir = self.get_local_feature_dir(img_dir)
        t=time()
        if (not os.path.exists(os.path.join(self.local_feature_dir, 'descriptors.h5'))) or (self.config["force_recache"]):
            fnames_list = self.ds.samples
            if self.config["local_features"] == "sift":
                detect_sift_dir(fnames_list, feature_dir=self.local_feature_dir, num_feats=self.config["num_local_features"], device=self.config["device"],
                                resize_to=self.config["local_desc_image_size"])
            if self.config["local_features"] == "xfeat":
                detect_xfeat_dir(fnames_list,
                                 feature_dir=self.local_feature_dir,
                                 num_feats=self.config["num_local_features"],
                                 resize_to=self.config["local_desc_image_size"],
                                 device=self.config["device"])
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
    
    def get_shortlist(self, query_fname, num_nn = 1000):
        """Returns a shortlist of images based on the global similarity query image.
        Args:
            query_fname (str): The filename of the query image.
            num_nn (int): The number of nearest neighbors to return.
        Returns:
            idxs (np.ndarray): The indices of the nearest neighbors.
            sims (np.ndarray): The similarity score of the nearest neighbors.
        """
        img = Image.open(query_fname).convert("RGB")
        query = self.describe_query(img)
        t=time()
        dists = np.linalg.norm(self.global_descs - query, axis=1)
        idxs = np.argsort(dists)[:num_nn]
        print (f"Distances: {dists[idxs]}")
        print (f"Shortlist in: {time()-t:.2f} sec")
        return idxs, (2-dists[idxs])/2.0
    
    def resort_shortlist(self, query, shortlist, criterion = 'num_inl', device='cpu', matching_method='smnn'):
        # Placeholder function for retrieving data
        ### First, we need to get the local descriptors of the query image
        assert matching_method in ['adalam', 'smnn', 'flann']
        import kornia as K
        hw1 = torch.tensor(query.shape[:2])
        new_shortlist_scores = []
        matching_keypoints = []

        if self.config["local_features"] == "sift":
            kpt, descs, lafs1 = detect_sift_single(query,
                                                   num_feats=self.config["num_local_features"],
                                                   resize_to=self.config["local_desc_image_size"])
        if self.config["local_features"] == "xfeat":
            kpt, descs, lafs1 = detect_xfeat_single(query,
                                                    num_feats=self.config["num_local_features"],
                                                    resize_to=self.config["local_desc_image_size"])
        if matching_method == 'flann':
            FLANN_INDEX_KDTREE = 1  # FLANN_INDEX_KDTREE
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=2)
            search_params = dict(checks=32)  # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            flann.add([descs.detach().cpu().numpy()])
        tt=time()
        with h5py.File(f'{self.local_feature_dir}/descriptors.h5', mode='r') as f_desc, \
            h5py.File(f'{self.local_feature_dir}/lafs.h5', mode='r') as f_laf, \
            h5py.File(f'{self.local_feature_dir}/hw.h5', mode='r') as f_hw :
                for i, idx in tqdm(enumerate(shortlist)):
                    fname = self.ds.samples[idx]
                    hw2 = torch.from_numpy(f_hw[fname][...])
                    descs2 = torch.from_numpy(f_desc[fname][...])
                    lafs2 = torch.from_numpy(f_laf[fname][...])
                    if matching_method == 'adalam':
                        dists, idxs = K.feature.match_adalam(descs, descs2,
                                        lafs1,
                                        lafs2,  # Adalam takes into account also geometric information
                                        hw1=hw1, hw2=hw2)  # Adalam also benefits from knowing image size
                    elif matching_method == 'smnn':
                        matcher = K.feature.match_smnn
                        dists, idxs = matcher(descs, descs2, 0.99)
                    elif matching_method == 'flann':
                        matches = flann.knnMatch(descs2.numpy(), k=2)
                        valid_matches = []
                        for cur_match in matches:
                            tmp_valid_matches = [
                                nn_1 for nn_1, nn_2 in zip(cur_match[:-1], cur_match[1:])
                                if nn_1.distance <= 0.9 * nn_2.distance
                            ]
                            valid_matches.extend(tmp_valid_matches)
                        dists, idxs = kornia_matches_from_cv2(valid_matches)
                        idxs = idxs.flip(1)
                    else:
                        raise NotImplementedError
                    kp1 = K.feature.get_laf_center(lafs1).reshape(-1, 2)
                    kp2 = K.feature.get_laf_center(lafs2).reshape(-1, 2)
                    mkpts1, mkpts2  = get_matching_keypoints(kp1, kp2, idxs)
                    matching_keypoints.append((mkpts1, mkpts2))
        print (f"Matching {matching_method} in {time()-tt:.4f} sec")
        tt=time()
        for i, idx in tqdm(enumerate(shortlist)):
            mkpts1, mkpts2 = matching_keypoints[i]
            if len(mkpts1) < 5:
                new_shortlist_scores.append(0)
                continue
            H, inliers = cv2.findHomography(
                mkpts1.detach().cpu().numpy(),
                mkpts2.detach().cpu().numpy(),
                cv2.USAC_MAGSAC,
                self.config['inl_th'],
                0.999,
                self.config['num_iter']
            )
            inliers = inliers > 0
            num_inl = inliers.sum()
            if num_inl>20:
                if criterion == 'num_inl':
                    new_shortlist_scores.append(num_inl)
                elif criterion == 'scale_factor_min':
                    scale_factor = get_scale_factor(H)
                    new_shortlist_scores.append(1.0/scale_factor)
                elif criterion == 'scale_factor_max':
                    scale_factor = get_scale_factor(H)
                    new_shortlist_scores.append(scale_factor)
            else:
                if criterion == 'num_inl':
                    new_shortlist_scores.append(num_inl)
                else:
                    new_shortlist_scores.append(0)
                #print (f"Found {num_inl} inliers in {fname}")
                #print (H)
        print (f"RANSAC in {time()-tt:.4f} sec")
        new_shortlist_scores = np.array(new_shortlist_scores)
        sorted_idxs = np.argsort(new_shortlist_scores)[::-1]
        return shortlist[sorted_idxs], new_shortlist_scores[sorted_idxs]

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
    
    shortlist_idxs, shortlist_scores = r.get_shortlist(query_fname, num_nn=r.config["num_nn"])
    fnames = r.ds.samples  
    q_img = cv2.cvtColor(cv2.imread(query_fname), cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        idxs, scores = r.resort_shortlist(q_img, shortlist_idxs, matching_method=r.config["matching_method"],
                                 criterion=r.config["resort_criterion"])
    print ([fnames[i] for i in idxs], scores)  

if __name__ == "__main__":
    main()