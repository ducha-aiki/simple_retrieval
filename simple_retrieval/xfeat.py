# Adapted from https://github.com/verlab/accelerated_features
# Original authors: Guilherme Potje · Felipe Cadar · Andre Araujo · Renato Martins · Erickson R. Nascimento
# License: Apache License 2.0

"""
    "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""
import time
import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kornia.feature.lightglue import LightGlue


class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions. """ 
    def __init__(self, mode = 'bicubic', align_corners = False): 
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """ Normalize coords to [-1,1]. """
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode = self.mode , align_corners = False)
        return x.permute(0,2,3,1).squeeze(-2)


class BasicLayer(nn.Module):
    """
      Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
                                      nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
                                      nn.BatchNorm2d(out_channels, affine=False),
                                      nn.ReLU(inplace = True),
                                    )

    def forward(self, x):
      return self.layer(x)

class XFeatModel(nn.Module):
    """
       Implementation of architecture described in 
       "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    """

    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm2d(1)


        ########### ⬇️ CNN Backbone & Heads ⬇️ ###########

        self.skip1 = nn.Sequential(	 nn.AvgPool2d(4, stride = 4),
                                       nn.Conv2d (1, 24, 1, stride = 1, padding=0) )

        self.block1 = nn.Sequential(
                                        BasicLayer( 1,  4, stride=1),
                                        BasicLayer( 4,  8, stride=2),
                                        BasicLayer( 8,  8, stride=1),
                                        BasicLayer( 8, 24, stride=2),
                                    )

        self.block2 = nn.Sequential(
                                        BasicLayer(24, 24, stride=1),
                                        BasicLayer(24, 24, stride=1),
                                     )

        self.block3 = nn.Sequential(
                                        BasicLayer(24, 64, stride=2),
                                        BasicLayer(64, 64, stride=1),
                                        BasicLayer(64, 64, 1, padding=0),
                                     )
        self.block4 = nn.Sequential(
                                        BasicLayer(64, 64, stride=2),
                                        BasicLayer(64, 64, stride=1),
                                        BasicLayer(64, 64, stride=1),
                                     )

        self.block5 = nn.Sequential(
                                        BasicLayer( 64, 128, stride=2),
                                        BasicLayer(128, 128, stride=1),
                                        BasicLayer(128, 128, stride=1),
                                        BasicLayer(128,  64, 1, padding=0),
                                     )

        self.block_fusion =  nn.Sequential(
                                        BasicLayer(64, 64, stride=1),
                                        BasicLayer(64, 64, stride=1),
                                        nn.Conv2d (64, 64, 1, padding=0)
                                     )

        self.heatmap_head = nn.Sequential(
                                        BasicLayer(64, 64, 1, padding=0),
                                        BasicLayer(64, 64, 1, padding=0),
                                        nn.Conv2d (64, 1, 1),
                                        nn.Sigmoid()
                                    )


        self.keypoint_head = nn.Sequential(
                                        BasicLayer(64, 64, 1, padding=0),
                                        BasicLayer(64, 64, 1, padding=0),
                                        BasicLayer(64, 64, 1, padding=0),
                                        nn.Conv2d (64, 65, 1),
                                    )


          ########### ⬇️ Fine Matcher MLP ⬇️ ###########

        self.fine_matcher =  nn.Sequential(
                                            nn.Linear(128, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                              nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                              nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                              nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                              nn.ReLU(inplace = True),
                                            nn.Linear(512, 64),
                                        )

    def _unfold2d(self, x, ws = 2):
        """
            Unfolds tensor in 2D with desired ws (window size) and concat the channels
        """
        B, C, H, W = x.shape
        x = x.unfold(2,  ws , ws).unfold(3, ws,ws)                             \
            .reshape(B, C, H//ws, W//ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)


    def forward(self, x):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:
                feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
                keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
                heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

        """
        #dont backprop through normalization
        with torch.no_grad():
            x = x.mean(dim=1, keepdim = True)
            x = self.norm(x)

        #main backbone
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        #pyramid fusion
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        feats = self.block_fusion( x3 + x4 + x5 )

        #heads
        heatmap = self.heatmap_head(feats) # Reliability map
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits

        return feats, keypoints, heatmap

class XFeat(nn.Module):
    """ 
        Implements the inference module for XFeat. 
        It supports inference for both sparse and semi-dense feature extraction & matching.
    """

    def __init__(self,
              weights = os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat.pt',
              top_k = 4096,
              detection_threshold=0.05,
              device = torch.device('cpu')):
        super().__init__()
        self.dev = device
        self.net = XFeatModel().to(self.dev).eval()
        self.top_k = top_k
        self.detection_threshold = detection_threshold

        if weights is not None:
            if isinstance(weights, str):
                if os.path.exists(weights):
                    state_dict = torch.load(weights, map_location=self.dev)
                else:
                    state_dict = torch.hub.load_state_dict_from_url("https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt")
                self.net.load_state_dict(state_dict)

        self.interpolator = InterpolateSparse2d('bicubic')

        #Try to import LightGlue from Kornia
        self.kornia_available = False
        self.lighterglue = None
        try:
            import kornia
            self.kornia_available=True
        except:
            pass


    @torch.inference_mode()
    def detectAndCompute(self, x, top_k = None, detection_threshold = None):
        """
            Compute sparse keypoints & descriptors. Supports batched mode.

            input:
                x -> torch.Tensor(B, C, H, W): grayscale or rgb image
                top_k -> int: keep best k features
            return:
                List[Dict]: 
                    'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
                    'scores'       ->   torch.Tensor(N,): keypoint scores
                    'descriptors'  ->   torch.Tensor(N, 64): local features
        """
        if top_k is None: top_k = self.top_k
        if detection_threshold is None: detection_threshold = self.detection_threshold
        x, rh1, rw1 = self.preprocess_tensor(x)

        B, _, _H1, _W1 = x.shape
        
        M1, K1, H1 = self.net(x)
        M1 = F.normalize(M1, dim=1)

        #Convert logits to heatmap and extract kpts
        K1h = self.get_kpts_heatmap(K1)
        mkpts = self.NMS(K1h, threshold=detection_threshold, kernel_size=5)

        #Compute reliability scores
        _nearest = InterpolateSparse2d('nearest')
        _bilinear = InterpolateSparse2d('bilinear')
        scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)
        scores[torch.all(mkpts == 0, dim=-1)] = -1

        #Select top-k features
        idxs = torch.argsort(-scores)
        mkpts_x  = torch.gather(mkpts[...,0], -1, idxs)[:, :top_k]
        mkpts_y  = torch.gather(mkpts[...,1], -1, idxs)[:, :top_k]
        mkpts = torch.cat([mkpts_x[...,None], mkpts_y[...,None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)[:, :top_k]

        #Interpolate descriptors at kpts positions
        feats = self.interpolator(M1, mkpts, H = _H1, W = _W1)

        #L2-Normalize
        feats = F.normalize(feats, dim=-1)

        #Correct kpt scale
        mkpts = mkpts * torch.tensor([rw1,rh1], device=mkpts.device).view(1, 1, -1)

        valid = scores > 0
        return [  
                   {'keypoints': mkpts[b][valid[b]],
                    'scores': scores[b][valid[b]],
                    'descriptors': feats[b][valid[b]]} for b in range(B) 
               ]

    @torch.inference_mode()
    def detectAndComputeDense(self, x, top_k = None, multiscale = True):
        """
            Compute dense *and coarse* descriptors. Supports batched mode.

            input:
                x -> torch.Tensor(B, C, H, W): grayscale or rgb image
                top_k -> int: keep best k features
            return: features sorted by their reliability score -- from most to least
                List[Dict]: 
                    'keypoints'    ->   torch.Tensor(top_k, 2): coarse keypoints
                    'scales'       ->   torch.Tensor(top_k,): extraction scale
                    'descriptors'  ->   torch.Tensor(top_k, 64): coarse local features
        """
        if top_k is None: top_k = self.top_k
        if multiscale:
            mkpts, sc, feats = self.extract_dualscale(x, top_k)
        else:
            mkpts, feats = self.extractDense(x, top_k)
            sc = torch.ones(mkpts.shape[:2], device=mkpts.device)

        return {'keypoints': mkpts,
                'descriptors': feats,
                'scales': sc }


    @torch.inference_mode()
    def match_lighterglue(self, d0, d1, min_conf = 0.1):
        """
            Match XFeat sparse features with LightGlue (smaller version) -- currently does NOT support batched inference because of padding, but its possible to implement easily.
            input:
                d0, d1: Dict('keypoints', 'scores, 'descriptors', 'image_size (Width, Height)')
            output:
                mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
                                idx              -> np.ndarray (N,2) the indices of the matching features
                
        """
        if not self.kornia_available:
            raise RuntimeError('We rely on kornia for LightGlue. Install with: pip install kornia')
        elif self.lighterglue is None:
            self.lighterglue = LighterGlue(device=self.dev)

        data = {
                'keypoints0': d0['keypoints'][None, ...],
                'keypoints1': d1['keypoints'][None, ...],
                'descriptors0': d0['descriptors'][None, ...],
                'descriptors1': d1['descriptors'][None, ...],
                'image_size0': torch.tensor(d0['image_size']).to(self.dev)[None, ...],
                'image_size1': torch.tensor(d1['image_size']).to(self.dev)[None, ...]
        }

        #Dict -> log_assignment: [B x M+1 x N+1] matches0: [B x M] matching_scores0: [B x M] matches1: [B x N] matching_scores1: [B x N] matches: List[[Si x 2]], scores: List[[Si]]
        out = self.lighterglue(data, min_conf = min_conf)

        idxs = out['matches'][0]

        return d0['keypoints'][idxs[:, 0]].cpu().numpy(), d1['keypoints'][idxs[:, 1]].cpu().numpy(), out['matches'][0].cpu().numpy()


    @torch.inference_mode()
    def match_xfeat(self, img1, img2, top_k = None, min_cossim = -1):
        """
            Simple extractor and MNN matcher.
            For simplicity it does not support batched mode due to possibly different number of kpts.
            input:
                img1 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
                img2 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
                top_k -> int: keep best k features
            returns:
                mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
        """
        if top_k is None: top_k = self.top_k
        img1 = self.parse_input(img1)
        img2 = self.parse_input(img2)

        out1 = self.detectAndCompute(img1, top_k=top_k)[0]
        out2 = self.detectAndCompute(img2, top_k=top_k)[0]

        idxs0, idxs1 = self.match(out1['descriptors'], out2['descriptors'], min_cossim=min_cossim )

        return out1['keypoints'][idxs0].cpu().numpy(), out2['keypoints'][idxs1].cpu().numpy()

    @torch.inference_mode()
    def match_xfeat_star(self, im_set1, im_set2, top_k = None):
        """
            Extracts coarse feats, then match pairs and finally refine matches, currently supports batched mode.
            input:
                im_set1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
                im_set2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
                top_k -> int: keep best k features
            returns:
                matches -> List[torch.Tensor(N, 4)]: List of size B containing tensor of pairwise matches (x1,y1,x2,y2)
        """
        if top_k is None: top_k = self.top_k
        im_set1 = self.parse_input(im_set1)
        im_set2 = self.parse_input(im_set2)

        #Compute coarse feats
        out1 = self.detectAndComputeDense(im_set1, top_k=top_k)
        out2 = self.detectAndComputeDense(im_set2, top_k=top_k)

        #Match batches of pairs
        idxs_list = self.batch_match(out1['descriptors'], out2['descriptors'] )
        B = len(im_set1)

        #Refine coarse matches
        #this part is harder to batch, currently iterate
        matches = []
        for b in range(B):
            matches.append(self.refine_matches(out1, out2, matches = idxs_list, batch_idx=b))

        return matches if B > 1 else (matches[0][:, :2].cpu().numpy(), matches[0][:, 2:].cpu().numpy())

    def preprocess_tensor(self, x):
        """ Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
        if isinstance(x, np.ndarray):
            if len(x.shape) == 3:
                x = torch.tensor(x).permute(2,0,1)[None]
            elif len(x.shape) == 2:
                x = torch.tensor(x[..., None]).permute(2,0,1)[None]
            else:
                raise RuntimeError('For numpy arrays, only (H,W) or (H,W,C) format is supported.')
        
        if len(x.shape) != 4:
            raise RuntimeError('Input tensor needs to be in (B,C,H,W) format')
        #x = x.to(self.dev).float()

        H, W = x.shape[-2:]
        _H, _W = (H//32) * 32, (W//32) * 32
        rh, rw = H/_H, W/_W

        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        return x, rh, rw

    def get_kpts_heatmap(self, kpts, softmax_temp = 1.0):
        scores = F.softmax(kpts*softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
        return heatmap

    def NMS(self, x, threshold = 0.05, kernel_size = 5):
        B, _, H, W = x.shape
        pad=kernel_size//2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

        pad_val = max([len(x) for x in pos_batched])
        pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

        #Pad kpts and build (B, N, 2) tensor
        for b in range(len(pos_batched)):
            pos[b, :len(pos_batched[b]), :] = pos_batched[b]

        return pos

    @torch.inference_mode()
    def batch_match(self, feats1, feats2, min_cossim = -1):
        B = len(feats1)
        cossim = torch.bmm(feats1, feats2.permute(0,2,1))
        match12 = torch.argmax(cossim, dim=-1)
        match21 = torch.argmax(cossim.permute(0,2,1), dim=-1)

        idx0 = torch.arange(len(match12[0]), device=match12.device)

        batched_matches = []

        for b in range(B):
            mutual = match21[b][match12[b]] == idx0

            if min_cossim > 0:
                cossim_max, _ = cossim[b].max(dim=1)
                good = cossim_max > min_cossim
                idx0_b = idx0[mutual & good]
                idx1_b = match12[b][mutual & good]
            else:
                idx0_b = idx0[mutual]
                idx1_b = match12[b][mutual]

            batched_matches.append((idx0_b, idx1_b))

        return batched_matches

    def subpix_softmax2d(self, heatmaps, temp = 3):
        N, H, W = heatmaps.shape
        heatmaps = torch.softmax(temp * heatmaps.view(-1, H*W), -1).view(-1, H, W)
        x, y = torch.meshgrid(torch.arange(W, device =  heatmaps.device ), torch.arange(H, device =  heatmaps.device ), indexing = 'xy')
        x = x - (W//2)
        y = y - (H//2)

        coords_x = (x[None, ...] * heatmaps)
        coords_y = (y[None, ...] * heatmaps)
        coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H*W, 2)
        coords = coords.sum(1)

        return coords

    def refine_matches(self, d0, d1, matches, batch_idx, fine_conf = 0.25):
        idx0, idx1 = matches[batch_idx]
        feats1 = d0['descriptors'][batch_idx][idx0]
        feats2 = d1['descriptors'][batch_idx][idx1]
        mkpts_0 = d0['keypoints'][batch_idx][idx0]
        mkpts_1 = d1['keypoints'][batch_idx][idx1]
        sc0 = d0['scales'][batch_idx][idx0]

        #Compute fine offsets
        offsets = self.net.fine_matcher(torch.cat([feats1, feats2],dim=-1))
        conf = F.softmax(offsets*3, dim=-1).max(dim=-1)[0]
        offsets = self.subpix_softmax2d(offsets.view(-1,8,8))

        mkpts_0 += offsets* (sc0[:,None]) #*0.9 #* (sc0[:,None])

        mask_good = conf > fine_conf
        mkpts_0 = mkpts_0[mask_good]
        mkpts_1 = mkpts_1[mask_good]

        return torch.cat([mkpts_0, mkpts_1], dim=-1)

    @torch.inference_mode()
    def match(self, feats1, feats2, min_cossim = 0.82):

        cossim = feats1 @ feats2.t()
        cossim_t = feats2 @ feats1.t()
        
        _, match12 = cossim.max(dim=1)
        _, match21 = cossim_t.max(dim=1)

        idx0 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0

        if min_cossim > 0:
            cossim, _ = cossim.max(dim=1)
            good = cossim > min_cossim
            idx0 = idx0[mutual & good]
            idx1 = match12[mutual & good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]

        return idx0, idx1

    def create_xy(self, h, w, dev):
        y, x = torch.meshgrid(torch.arange(h, device = dev), 
                                torch.arange(w, device = dev), indexing='ij')
        xy = torch.cat([x[..., None],y[..., None]], -1).reshape(-1,2)
        return xy

    def extractDense(self, x, top_k = 8_000):
        if top_k < 1:
            top_k = 100_000_000

        x, rh1, rw1 = self.preprocess_tensor(x)

        M1, K1, H1 = self.net(x)
        
        B, C, _H1, _W1 = M1.shape
        
        xy1 = (self.create_xy(_H1, _W1, M1.device) * 8).expand(B,-1,-1)

        M1 = M1.permute(0,2,3,1).reshape(B, -1, C)
        H1 = H1.permute(0,2,3,1).reshape(B, -1)

        _, top_k = torch.topk(H1, k = min(len(H1[0]), top_k), dim=-1)

        feats = torch.gather( M1, 1, top_k[...,None].expand(-1, -1, 64))
        mkpts = torch.gather(xy1, 1, top_k[...,None].expand(-1, -1, 2))
        mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1,-1)

        return mkpts, feats

    def extract_dualscale(self, x, top_k, s1 = 0.6, s2 = 1.3):
        x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode='bilinear')
        x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode='bilinear')

        B, _, _, _ = x.shape

        mkpts_1, feats_1 = self.extractDense(x1, int(top_k*0.20))
        mkpts_2, feats_2 = self.extractDense(x2, int(top_k*0.80))

        mkpts = torch.cat([mkpts_1/s1, mkpts_2/s2], dim=1)
        sc1 = torch.ones(mkpts_1.shape[:2], device=mkpts_1.device) * (1/s1)
        sc2 = torch.ones(mkpts_2.shape[:2], device=mkpts_2.device) * (1/s2)
        sc = torch.cat([sc1, sc2],dim=1)
        feats = torch.cat([feats_1, feats_2], dim=1)

        return mkpts, sc, feats

    def parse_input(self, x):
        if len(x.shape) == 3:
            x = x[None, ...]

        if isinstance(x, np.ndarray):
            x = torch.tensor(x).permute(0,3,1,2)/255

        return x

class LighterGlue(nn.Module):
    """
        Lighter version of LightGlue :)
    """

    default_conf_xfeat = {
    "name": "xfeat",  # just for interfacing
    "input_dim": 64,  # input descriptor dimension (autoselected from weights)
    "descriptor_dim": 96,
    "add_scale_ori": False,
    "add_laf": False,  # for KeyNetAffNetHardNet
    "scale_coef": 1.0,  # to compensate for the SIFT scale bigger than KeyNet
    "n_layers": 6,
    "num_heads": 1,
    "flash": True,  # enable FlashAttention if available.
    "mp": False,  # enable mixed precision
    "depth_confidence": -1,  # early stopping, disable with -1
    "width_confidence": 0.95,  # point pruning, disable with -1
    "filter_threshold": 0.1,  # match threshold
    "weights": None,
    }

    def __init__(self, weights = os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat-lighterglue.pt', device=torch.device('cpu')):
        super().__init__()
        LightGlue.default_conf = self.default_conf_xfeat
        self.net = LightGlue(None)
        self.dev = device

        if os.path.exists(weights):
            state_dict = torch.load(weights, map_location=self.dev)
        else:
            state_dict = torch.hub.load_state_dict_from_url("https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt")

        # rename old state dict entries
        for i in range(self.net.conf.n_layers):
            pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            state_dict = {k.replace('matcher.', ''): v for k, v in state_dict.items()}

        self.net.load_state_dict(state_dict, strict=False)
        self.net.to(self.dev)

    @torch.inference_mode()
    def forward(self, data, min_conf = 0.1):
        self.net.conf.filter_threshold = min_conf
        result = self.net( {   'image0': {'keypoints': data['keypoints0'], 'descriptors': data['descriptors0'], 'image_size': data['image_size0']},
                               'image1': {'keypoints': data['keypoints1'], 'descriptors': data['descriptors1'], 'image_size': data['image_size1']}  
                           } )
        return result