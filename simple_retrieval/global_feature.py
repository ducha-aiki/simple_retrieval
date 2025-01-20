# Taken from https://github.com/serizba/salad/
import math
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from time import time
from tqdm import tqdm
from simple_retrieval.pile_of_garbage import CustomImageFolder

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=2,
            norm_layer=False,
            return_token=False
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).eval()
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token


    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """
        self.model.to(x.device)
        self.model.patch_embed.proj.to(x.device)
        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)
        
        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score = 1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n-m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(
        log_a,
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg
    )
    return log_P - norm


class SALAD(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """
    def __init__(self,
            num_channels=1536,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
            dropout=0.3,
        ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters= num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        
        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.))


    def forward(self, x):
        """
        x (tuple): A tuple containing two elements, f and t. 
            (torch.Tensor): The feature tensors (t_i) [B, C, H // 14, W // 14].
            (torch.Tensor): The token tensor (t_{n+1}) [B, C].

        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + g]
        """
        x, t = x # Extract features and token

        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        # Sinkhorn algorithm
        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)
        # Normalize to maintain mass
        p = p[:, :-1, :]


        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        f = torch.cat([
            nn.functional.normalize(t, p=2, dim=-1),
            nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        ], dim=-1)

        return nn.functional.normalize(f, p=2, dim=-1)

def get_input_transform_dinosalad(image_size=None):
    MEAN=[0.485, 0.456, 0.406]; STD=[0.229, 0.224, 0.225]
    if image_size:
        return T.Compose([
            T.Resize(image_size,  interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

def get_input_transform_siglip(image_size=None):
    MEAN=[0.5, 0.5, 0.5]; STD=[0.5, 0.5, 0.5]
    if image_size:
        return T.Compose([
            T.Resize(image_size,  interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

class SimpleDINOSALAD(nn.Module):
    def __init__(self, model, aggregator):
        super().__init__()
        self.backbone = model
        self.aggregator = aggregator
        self.backbone.eval()
        self.aggregator.eval()
    
    def forward(self, x):
        x = self.backbone(x)
        return self.aggregator(x)


class siglip2(nn.Module):
    def __init__(self,device='cpu'):
        super().__init__()
        self.device=device
        dtype = torch.float16 if 'cuda' in str(device) else torch.float32
        self.dtype=dtype
        from transformers import SiglipProcessor, SiglipVisionModel
        self.model = SiglipVisionModel.from_pretrained(
            "google/siglip-large-patch16-384",
           # attn_implementation="flash_attention_2",
            torch_dtype=dtype,
            device_map=device,
        ).eval()
        self.processor = SiglipProcessor.from_pretrained("google/siglip-large-patch16-384")
    def forward(self, x):
        from transformers.feature_extraction_utils  import BatchFeature
        with torch.inference_mode():
            inp = BatchFeature(data={"pixel_values": x}, tensor_type='pt').to(self.device, self.dtype)
            outputs = self.model(**inp)
            features = outputs.pooler_output 
        return nn.functional.normalize(features, p=2, dim=-1)

def get_dinov2salad(device='cpu', dtype=torch.float32):
    """
    This function returns a DINOv2 + SALAD model.
    """
    backbone_config={
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
    }
    dino = DINOv2(model_name='dinov2_vitb14', **backbone_config)
    agg_config={
            'num_channels': 768,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        }
    salad = SALAD(**agg_config).eval()
    model = SimpleDINOSALAD(dino, salad)
    model.load_state_dict(torch.load('dino_salad.ckpt', map_location=torch.device('cpu')))
    model.eval()
    model=model.to(device=device, dtype=dtype)
    return model


def dataset_inference(model, ds, batch_size=4, device=torch.device('cpu'), num_workers=1):
    global_desc = []
    t = time()
    dev = device
    dtype = torch.float16 if 'cuda' in str(device) else torch.float32
    bs = batch_size
    model.eval()
    dl = DataLoader(ds, batch_size=bs, num_workers=num_workers, pin_memory=True)
    with torch.inference_mode():
        for img, _ in tqdm(dl):
            global_desc.append(model(img.to(dev, dtype)).cpu())
    # Placeholder function for creating a global descriptor index
    global_descs = np.concatenate(global_desc, axis=0)
    print (f"{len(global_descs)} global descs in {time()-t:.2f} sec")
    return global_descs