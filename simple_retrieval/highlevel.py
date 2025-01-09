
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

from simple_retrieval.global_feature import get_dinov2salad, get_input_transform
from simple_retrieval.local_feature import detect_sift_single, detect_sift_dir
import cv2


def get_default_config():
    conf = {"local_features": "sift",
            "global_features": "dinosalad",
            "inl_th": 5.0,
            "num_iter": 1000,
            "global_desc_image_size": 392,
            "global_desc_batch_size": 2,
            "device": "mps",
            "ransac_type": "homography",
            "num_nn": 1000,
            "use_diffusion": False,
            "resort_criterion": "num_inl"}
    return conf

def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = kp1[idxs[:, 0]]
    mkpts2 = kp2[idxs[:, 1]]
    return mkpts1, mkpts2

class CustomImageFolder(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None, extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".gif")):
        """
        Args:
            root (str): Root directory path.
            transform (Callable, optional): A function/transform to apply to the images.
            extensions (tuple): Tuple of allowed file extensions (default: common image formats).
        """
        self.root = root
        self.transform = transform
        self.extensions = extensions
        self.samples = self._make_dataset()

    def _is_valid_file(self, filename: str) -> bool:
        """Checks if a file is a valid image file based on its extension."""
        return filename.lower().endswith(self.extensions)

    def _make_dataset(self) -> List[str]:
        """Indexes all valid image files in the directory and its subdirectories."""
        images = []
        for root, _, files in os.walk(self.root):
            for file in sorted(files):
                if self._is_valid_file(file):
                    images.append(os.path.join(root, file))
        return images

    def __len__(self) -> int:
        """Returns the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int):
        """Returns the image and its file path at the given index."""
        filepath = self.samples[index]
        with Image.open(filepath) as img:
            img = img.convert("RGB")  # Ensure all images are in RGB format
            if self.transform:
                img = self.transform(img)
        return img, filepath
    

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
    

    def create_global_descriptor_index(self, img_dir, index_dir, ):
        global_desc = []
        t = time()
        dev = torch.device(self.config["device"])
        dtype = torch.float16 if 'cuda' in self.config["device"] else torch.float32
        if self.config["global_features"] == "dinosalad":
            model = self.global_model
            bs = self.config["global_desc_batch_size"]
            transform = self.global_transform
            self.ds = CustomImageFolder(img_dir, transform=transform)
            dl = DataLoader(self.ds, batch_size=bs, num_workers=1)
            with torch.inference_mode():
                for img, _ in tqdm(dl):
                    global_desc.append(model(img.to(dev)).cpu())
        # Placeholder function for creating a global descriptor index
        self.global_descs = np.concatenate(global_desc, axis=0)
        print (f"{self.config['global_features']} index of {len(self.global_descs)} created from images in: {img_dir} in {time()-t:.2f} sec, saved to {index_dir}")
        return 

    def create_local_descriptor_index(self, img_dir, index_dir):
        # Placeholder function for creating a local descriptor index
        t=time()
        fnames_list = self.ds.samples
        if self.config["local_features"] == "sift":
            detect_sift_dir(fnames_list, feature_dir=index_dir)
        self.local_feature_dir=index_dir
        print (f"{self.config['local_features']} index of created from images in: {img_dir} in {time()-t:.2f} sec, saved to {index_dir}")
        return 

    def describe_query(self, img):
        dev = torch.device(self.config["device"])
        dtype = torch.float16 if 'cuda' in self.config["device"] else torch.float32
        t = time()
        if self.config["global_features"] == "dinosalad":
            model = self.global_model
            with torch.inference_mode():
                global_desc = model(self.global_transform(img)[None].to(dev)).cpu()
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
        print (f"Shortlist in: {time()-t:.2f} sec")
        return idxs, 2-dists[idxs]
    
    def resort_shortlist(self, query, shortlist, criterion = 'num_inl'):
        # Placeholder function for retrieving data
        ### First, we need to get the local descriptors of the query image
        import kornia as K
        matcher = K.feature.match_adalam
        hw1 = torch.tensor(query.shape[:2])
        new_shortlist_scores = []

        if self.config["local_features"] == "sift":
            kpt, descs, lafs1 = detect_sift_single(query)
            with h5py.File(f'{self.local_feature_dir}/descriptors.h5', mode='r') as f_desc, \
                h5py.File(f'{self.local_feature_dir}/lafs.h5', mode='r') as f_laf:
                for i, idx in tqdm(enumerate(shortlist)):
                    fname = self.ds.samples[idx]
                    img = Image.open(fname).convert("RGB")
                    wh2 = img.size
                    hw2 = torch.tensor([wh2[1], wh2[0]])
                    descs2 = torch.from_numpy(f_desc[fname][...])
                    lafs2 = torch.from_numpy(f_laf[fname][...])
                    dists, idxs = matcher(descs, descs2,
                                      lafs1,
                                      lafs2,  # Adalam takes into account also geometric information
                                      hw1=hw1, hw2=hw2)  # Adalam also benefits from knowing image size
                    if criterion == 'num_inl':
                        kp1 = K.feature.get_laf_center(lafs1).reshape(-1, 2)
                        kp2 = K.feature.get_laf_center(lafs2).reshape(-1, 2)
                        mkpts1, mkpts2  = get_matching_keypoints(kp1, kp2, idxs)
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
                        new_shortlist_scores.append(inliers.sum())
                    else:
                        raise NotImplementedError
        new_shortlist_scores = np.array(new_shortlist_scores)
        sorted_idxs = np.argsort(new_shortlist_scores)[::-1]
        print (sorted_idxs)
        return shortlist[sorted_idxs], new_shortlist_scores[sorted_idxs]




def main():
    # Example usage of the retrieve_data function
    r = SimpleRetrieval()
    print (r)
    r.create_global_descriptor_index('/Users/oldufo/datasets/goose',
                                     './tmp/global_desc')
    r.create_local_descriptor_index('/Users/oldufo/datasets/goose',
                                     './tmp/local_desc')
    
    #query_fname = '/Users/oldufo/datasets/EVD/1/graf.png'
    query_fname = '/Users/oldufo/datasets/goose/IMG_3926.jpg'
    shortlist_idxs, shortlist_scores = r.get_shortlist(query_fname)
    print (shortlist_idxs, shortlist_scores)
    q_img = cv2.cvtColor(cv2.imread(query_fname), cv2.COLOR_BGR2RGB)
    out = r.resort_shortlist(q_img, shortlist_idxs)
    print(out)

if __name__ == "__main__":
    main()