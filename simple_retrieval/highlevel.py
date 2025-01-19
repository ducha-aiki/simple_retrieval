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

from simple_retrieval.global_feature import get_dinov2salad, get_input_transform_siglip, get_input_transform_dinosalad, dataset_inference, siglip2
from simple_retrieval.local_feature import detect_sift_single, detect_sift_dir, detect_xfeat_single, detect_xfeat_dir, match_query_to_db, spatial_scoring
from simple_retrieval.pile_of_garbage import CustomImageFolder
from simple_retrieval.manifold_diffusion import sim_kernel, normalize_connection_graph, topK_W, cg_diffusion
import cv2


def get_default_config():
    conf = {"local_features": "xfeat",
            "global_features": "dinosalad",
            "inl_th": 2.0,
            "num_iter": 2000,
            "num_local_features": 4096,
            "local_desc_image_size": (1024,768),
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
            "use_diffusion": False,
            "resort_criterion": "scale_factor_max"}
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
        else:
            self.global_model  = siglip2(device=dev).eval()
            self.global_transform = get_input_transform_siglip((384, 384))
        return
    
    def __repr__(self):
        return f"""SimpleRetrieval(img_dir={self.img_dir},
        index_dir={self.index_dir}, config={self.config})"""
    
    def get_cache_dir_name(self, img_dir):
        return f"./tmp/{img_dir.replace('/', '_')}"
    
    def get_global_index_fname(self, img_dir):
        return os.path.join(self.get_cache_dir_name(img_dir), f"{self.config['global_features']}_global_index.pth")
    
    def get_global_index_Wn_name(self, img_dir):
        return os.path.join(self.get_cache_dir_name(img_dir),  f"{self.config['global_features']}_'Wn'.pth")

    def get_local_feature_dir(self, img_dir):
        local_desc_name = f"{self.config['local_features']}_{self.config['num_local_features']}"
        return os.path.join(self.get_cache_dir_name(img_dir), local_desc_name)

    def create_global_descriptor_index(self, img_dir, index_dir):
        """Creates a global descriptor index from a directory of images."""
        index_dir = self.get_cache_dir_name(img_dir)
        os.makedirs(index_dir, exist_ok=True)
        self.ds = CustomImageFolder(img_dir, transform=self.global_transform)
        global_index_fname = self.get_global_index_fname(img_dir)
        Wn_fname = self.get_global_index_Wn_name(img_dir)
        
        if os.path.exists(global_index_fname) and not self.config["force_recache"]:
            print (f"Loading global index from {global_index_fname}")
            self.global_descs = torch.load(global_index_fname)
           
        else:
            print (f"Creating global index from images in: {img_dir}")
            self.global_descs = dataset_inference(self.global_model,
                                                        self.ds,
                                                        batch_size=self.config["global_desc_batch_size"],
                                                        device=self.config["device"],
                                                        num_workers=self.config["num_workers"])
            torch.save(self.global_descs, global_index_fname)
            print (f"Global index saved to:  {global_index_fname}")
            self.global_descs = torch.load(global_index_fname)
        if os.path.exists(Wn_fname) and not self.config["force_recache"]:
            self.Wn = torch.load(Wn_fname)
        else:
            print(self.global_descs.shape)
            self.Wn = None#self.get_Wn(self.global_descs.T, K = 100)
            #torch.save(self.Wn, Wn_fname)
        print (self.global_descs[0])
        print (self.global_descs.shape, self.global_descs.dtype)
        return 

    def get_Wn(self, X, K = 100):
        W = sim_kernel(np.dot(X.T, X))
        W = topK_W(W, K)
        Wn = normalize_connection_graph(W)
        return Wn

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
                                 device=self.config["device"],
                                 batch_size=self.config["local_desc_batch_size"],
                                 num_workers=self.config["num_workers"],
                                 pin_memory=False)
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
    
    def get_shortlist(self, query_fname, num_nn = 1000, manifold_diffusion=False, Wn=None):
        """Returns a shortlist of images based on the global similarity query image.
        Args:
            query_fname (str): The filename of the query image.
            num_nn (int): The number of nearest neighbors to return.
        Returns:
            idxs (np.ndarray): The indices of the nearest neighbors.
            sims (np.ndarray): The similarity score of the nearest neighbors.
        """
        img = Image.open(query_fname).convert("RGB")
        t=time()
        query = self.describe_query(img)
        if manifold_diffusion:
            Q = query.reshape(-1, 1)
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
        else:
            dists = np.linalg.norm(self.global_descs - query, axis=1)
        idxs = np.argsort(dists)[:num_nn]
        #print (f"Distances: {dists[idxs]}")
        print (f"Shortlist in: {time()-t:.2f} sec")
        return idxs, (2-dists[idxs])/2.0
    
    def resort_shortlist(self, query, shortlist, criterion = 'num_inl', device='cpu', matching_method='smnn'):
        # Placeholder function for retrieving data
        ### First, we need to get the local descriptors of the query image
        new_shortlist_scores = []
        matching_keypoints = []
        dtype = torch.float16 if 'cuda' in str(device) else torch.float32
        hw1 = torch.tensor(query.shape[:2]).to(device, dtype)

        if self.config["local_features"] == "sift":
            kpt, descs, lafs1 = detect_sift_single(query,
                                                   num_feats=self.config["num_local_features"],
                                                   resize_to=self.config["local_desc_image_size"])
        if self.config["local_features"] == "xfeat":
            kpt, descs, lafs1 = detect_xfeat_single(query,
                                                    num_feats=self.config["num_local_features"],
                                                    resize_to=self.config["local_desc_image_size"])
        descs = descs.to(device, dtype)
        lafs1 = lafs1.to(device, dtype)
        fnames = [self.ds.samples[i] for i in shortlist]
        tt=time()
        matching_keypoints = match_query_to_db(descs, lafs1, hw1,
                                               self.local_feature_dir,
                                               fnames,
                                               matching_method=matching_method,
                                               device=torch.device(device))
        print (f"Matching {matching_method} in {time()-tt:.4f} sec")
        tt=time()
        new_shortlist_scores = spatial_scoring(matching_keypoints, criterion=criterion, config=self.config)
        print (f"RANSAC in {time()-tt:.4f} sec")
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
    
    shortlist_idxs, shortlist_scores = r.get_shortlist(query_fname, num_nn=r.config["num_nn"], manifold_diffusion=r.config["use_diffusion"])
    fnames = r.ds.samples  
    q_img = cv2.cvtColor(cv2.imread(query_fname), cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        idxs, scores = r.resort_shortlist(q_img, shortlist_idxs, matching_method=r.config["matching_method"],
                                 criterion=r.config["resort_criterion"])
    print ([fnames[i] for i in idxs], scores)  

if __name__ == "__main__":
    main()