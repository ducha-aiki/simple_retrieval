
import torch
import cv2
import h5py
import kornia as K
import os
from tqdm import tqdm

import kornia.feature as KF
from kornia_moons.feature import laf_from_opencv_SIFT_kpts


def load_torch_image(fname, device=torch.device('cpu')):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.
    img = K.color.bgr_to_rgb(img.to(device))
    return img


def sift_to_rootsift(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, p=1, dim=-1, eps=eps)
    x.clip_(min=eps).sqrt_()
    return torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)


def detect_sift_dir(img_fnames,
                segmentations=None,
                num_feats=2048,
                device=torch.device('cpu'),
                feature_dir='.featureout', resize_to=(800, 600)):
    sift = cv2.SIFT_create(num_feats, edgeThreshold=-1000, contrastThreshold=-1000)
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    with h5py.File(f'{feature_dir}/lafs.h5', mode='w') as f_laf, \
            h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
            h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
        for i, img_path in tqdm(enumerate(img_fnames)):
            if segmentations is not None:
                seg = cv2.imread(segmentations[i], cv2.IMREAD_GRAYSCALE)
            else:
                seg = None
            img1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            hw1 = torch.tensor(img1.shape[:2], device=device)
            if resize_to:
                img1 = cv2.resize(img1, resize_to)
                hw1_new = torch.tensor(img1.shape[:2], device=device)
            #img_fname = img_path.split('/')[-1]
            key = img_path
            kpts1, descs1 = sift.detectAndCompute(img1, seg)
            lafs1 = laf_from_opencv_SIFT_kpts(kpts1)
            if resize_to:
                lafs1[..., 0] *= hw1_new[1] / hw1[1]
                lafs1[..., 1] *= hw1_new[0] / hw1[0]
            descs1 = sift_to_rootsift(torch.from_numpy(descs1)).to(device)
            desc_dim = descs1.shape[-1]
            kpts = KF.get_laf_center(lafs1).reshape(-1, 2).detach().cpu().numpy()
            descs1 = descs1.reshape(-1, desc_dim).detach().cpu().numpy()
            f_laf[key] = lafs1.detach().cpu().numpy()
            f_kp[key] = kpts
            f_desc[key] = descs1
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
        lafs1[..., 0] *= hw1_new[1] / hw1[1]
        lafs1[..., 1] *= hw1_new[0] / hw1[0]
    kpts = KF.get_laf_center(lafs1).reshape(-1, 2).detach().cpu().numpy()
    return kpts, descs1, lafs1