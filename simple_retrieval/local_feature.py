
import torch
import cv2
import h5py
import kornia as K
import os
from tqdm import tqdm
import numpy as np
import kornia.feature as KF
from kornia_moons.feature import laf_from_opencv_SIFT_kpts
from simple_retrieval.pile_of_garbage import CustomImageFolderFromFileList, collate_with_string, no_collate, H5LocalFeatureDataset
from simple_retrieval.xfeat import XFeat, LighterGlue
import torchvision.transforms as T
from torch.utils.data import DataLoader
from kornia_moons.feature import kornia_matches_from_cv2

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
            hw1 = torch.tensor(img1.shape[:2])
            if resize_to:
                img1 = cv2.resize(img1, resize_to)
                hw1_new = torch.tensor(img1.shape[:2])
            #img_fname = img_path.split('/')[-1]
            key = img_path
            kpts1, descs1 = sift.detectAndCompute(img1, None)
            lafs1 = laf_from_opencv_SIFT_kpts(kpts1)
            if resize_to:
                lafs1[:, :, 0, :] *= hw1[1] / hw1_new[1]
                lafs1[:, :, 1, :] *= hw1[0] / hw1_new[0]
            descs1 = sift_to_rootsift(torch.from_numpy(descs1))
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
                pin_memory=False):

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

def match_query_to_db(query_desc, query_laf, query_hw, db_dir, fnames, matching_method='smnn' , feature_name='xfeat', device=torch.device('cpu')):
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
    for i, idx in tqdm(enumerate(matching_keypoints)):
        mkpts1, mkpts2 = matching_keypoints[i]
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