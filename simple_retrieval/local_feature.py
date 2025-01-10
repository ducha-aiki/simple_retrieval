
import torch
import cv2
import h5py
import kornia as K
import os
from tqdm import tqdm
import numpy as np
import kornia.feature as KF
from kornia_moons.feature import laf_from_opencv_SIFT_kpts
from simple_retrieval.pile_of_garbage import CustomImageFolderFromFileList, collate_with_string
import torchvision.transforms as T
from torch.utils.data import DataLoader


def load_torch_image(fname, device=torch.device('cpu')):
    img = K.image_to_tensor(cv2.imread(fname), False).float() / 255.
    img = K.color.bgr_to_rgb(img.to(device))
    return img


def sift_to_rootsift(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, p=1, dim=-1, eps=eps)
    x.clip_(min=eps).sqrt_()
    return torch.nn.functional.normalize(x, p=2, dim=-1, eps=eps)


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
                feature_dir='.featureout', resize_to=(800, 600)):
    sift = cv2.SIFT_create(num_feats, edgeThreshold=-1000, contrastThreshold=-1000)
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    with h5py.File(f'{feature_dir}/lafs.h5', mode='w') as f_laf, \
            h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
            h5py.File(f'{feature_dir}/hw.h5', mode='w') as f_hw, \
            h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
        for i, img_path in tqdm(enumerate(img_fnames)):
            img1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            hw1 = torch.tensor(img1.shape[:2], device=device)
            if resize_to:
                img1 = cv2.resize(img1, resize_to)
                hw1_new = torch.tensor(img1.shape[:2], device=device)
            #img_fname = img_path.split('/')[-1]
            key = img_path
            kpts1, descs1 = sift.detectAndCompute(img1, None)
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
            f_hw[key] = hw1.detach().cpu().numpy()
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


def detect_xfeat_single(img, num_feats=2048, resize_to=(800, 600)):
    device=torch.device('cpu')
    from xfeat import XFeat
    model = XFeat()
    hw1 = torch.tensor(img.shape[:2])
    if resize_to:
        img = cv2.resize(img, resize_to)
        hw1_new = torch.tensor(img.shape[:2], device=device)
    res = model.detectAndCompute(K.image_to_tensor(img,None).float(), top_k=num_feats)
    keypoints, descriptors = res[0]['keypoints'], res[0]['descriptors']
    lafs1 = K.feature.laf_from_center_scale_ori(keypoints.reshape(1, -1, 2))
    if resize_to:
        lafs1[..., 0] *= hw1_new[1] / hw1[1]
        lafs1[..., 1] *= hw1_new[0] / hw1[0]
    kpts = KF.get_laf_center(lafs1).reshape(-1, 2).detach().cpu().numpy()
    return kpts, descriptors, lafs1


def detect_xfeat_dir(img_fnames,
                num_feats=2048,
                device=torch.device('cpu'),
                feature_dir='.featureout', resize_to=(600, 800)):
    from xfeat import XFeat
    model = XFeat()
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)

    ds = CustomImageFolderFromFileList(img_fnames,
                                       transform=get_input_xfeat_transform(resize_to))
    dev = device
    dtype = torch.float16 if 'cuda' in str(device) else torch.float32
    bs = 1
    dl = DataLoader(ds,
                    batch_size=bs,
                    num_workers=1,
                    collate_fn=collate_with_string,
                    persistent_workers=False, pin_memory = False)
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
                        lafs1[..., 0] *= ws[i] / resize_to[1]
                        lafs1[..., 1] *= hs[i] / resize_to[0]
                    desc_dim = descriptors.shape[-1]
                    kpts = KF.get_laf_center(lafs1).reshape(-1, 2).detach().cpu().numpy()
                    descriptors = descriptors.reshape(-1, desc_dim).detach().cpu().numpy()
                    f_laf[key] = lafs1.detach().cpu().numpy()
                    f_kp[key] = kpts
                    f_desc[key] = descriptors
                    f_hw[key] = np.array([hs[i], ws[i]])
            print ("Done")
    return

def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = kp1[idxs[:, 0]]
    mkpts2 = kp2[idxs[:, 1]]
    return mkpts1, mkpts2
