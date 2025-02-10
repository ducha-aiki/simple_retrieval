# Author: Dmytro Mishkin, 2018.
# This is simple partial re-implementation of papers 
# Fast Spectral Ranking for Similarity Search, CVPR 2018. https://arxiv.org/abs/1703.06935
# and  Efficient Diffusion on Region Manifolds: Recovering Small Objects with Compact CNN Representations, CVPR 2017 http://people.rennes.inria.fr/Ahmet.Iscen/diffusion.html


import os
import numpy as np
import torch
from scipy.io import loadmat
from scipy.sparse import csr_matrix, eye, diags
from scipy.sparse import linalg as s_linalg
from time import time
from tqdm import tqdm

def sim_kernel(dot_product):
    return np.maximum(np.power(dot_product,3),0)

def sim_kernel_torch(dot_product):
    return torch.clamp(torch.pow(dot_product, 3), 0)

 
def normalize_connection_graph(G):
    W = csr_matrix(G)
    W = W - diags(W.diagonal())
    D = np.array(1./ np.sqrt(W.sum(axis = 1)))
    D[np.isnan(D)] = 0
    D[np.isinf(D)] = 0
    D_mh = diags(D.reshape(-1))
    Wn = D_mh * W * D_mh
    return Wn

def topK_W(G, K = 100, symmetric = True):
    sortidxs = np.argsort(-G, axis = 1)
    for i in range(G.shape[0]):
        G[i,sortidxs[i,K:]] = 0
    if symmetric:
        G = np.minimum(G, G.T)
    return G

def topK_W_torch(G, K = 100, symmetric = True, return_sparse = False):
    top_val, top_idx = torch.topk(G, K, dim = 1)
    G_out = torch.zeros_like(G)
    G_out.scatter_(1, top_idx, top_val)
    if symmetric:
        G_out = torch.min(G_out, G_out.t())
    return G_out

def topK_to_csr(G, K=100):
    # Compute top-K values and indices on GPU
    top_val, top_idx = torch.topk(G, K, dim=1, sorted=False)
    
    # Create row indices for each top-K element.
    n = G.size(0)
    row_idx = torch.arange(n, device=G.device).unsqueeze(1).expand(n, K)
    # Flatten and move the indices and data to CPU
    row_idx_np = row_idx.reshape(-1).cpu().numpy()
    col_idx_np = top_idx.reshape(-1).cpu().numpy()
    data_np    = top_val.reshape(-1).cpu().numpy().astype(np.float32)
    
    # Construct the CSR matrix from the sparse data.
    csr = csr_matrix((data_np, (row_idx_np, col_idx_np)), shape=G.shape)
    return csr

def get_W_sparse(X, K=100, bs=32):
    """
    Compute a huge sparse similarity matrix W as a CSR matrix using GPU batch processing.
    
    Parameters:
      X : np.ndarray
          Input data of shape (D, num_samples). (D is the feature dimension.)
      K : int, default 100
          The number of top neighbors (nonzeros) per row.
      bs : int, default 32
          Batch size (number of columns to process at a time).
          
    Returns:
      W_sparse : csr_matrix
          A (num_samples x num_samples) sparse CSR matrix with at most K nonzeros per row.
          Symmetry is enforced via elementwise minimum with its transpose.
          
    Note:
      This function assumes CUDA is available. If not, it will raise an error.
      The similarity kernel function 'sim_kernel_torch' must be defined externally.
    """
    # Ensure CUDA is available (we only support the GPU path here)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for get_W_sparse")
    
    device = torch.device('cuda')
    dtype = torch.float16  # using half precision for speed on GPU
    
    # Convert the input array to a torch tensor on GPU.
    X_torch = torch.from_numpy(X).to(device, dtype)
    num_samples = X.shape[1]
    
    # Lists to collect sparse indices and data from each batch.
    all_rows = []
    all_cols = []
    all_data = []
    
    # Process the samples in batches.
    with torch.inference_mode():
        for i in tqdm(range(0, num_samples, bs), desc="Processing batches"):
            # current_X is a (D, current_bs) chunk.
            current_X = X_torch[:, i:i+bs]
            # Compute the similarity kernel between current_X and the whole X.
            # current_X.T @ X_torch produces a (current_bs x num_samples) matrix.
            current_W = sim_kernel_torch(current_X.T @ X_torch)
            
            # For each row in the current batch, extract the top-K entries.
            # top_val and top_idx have shape (current_bs, K)
            top_val, top_idx = torch.topk(current_W, K, dim=1, sorted=False)
            current_bs = current_W.size(0)
            
            # The global row indices for this batch (offset by i).
            batch_rows = torch.arange(i, i + current_bs, device=device).unsqueeze(1).expand(current_bs, K)
            
            # Flatten the batch tensors and move them to CPU.
            all_rows.append(batch_rows.reshape(-1).cpu().numpy())
            all_cols.append(top_idx.reshape(-1).cpu().numpy())
            all_data.append(top_val.reshape(-1).cpu().numpy().astype(np.float32))
    
    # Concatenate all the collected indices and data.
    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    data = np.concatenate(all_data)
    
    # Build the huge sparse matrix.
    W_sparse = csr_matrix((data, (rows, cols)), shape=(num_samples, num_samples))
    
    # Enforce symmetry (each edge weight becomes the minimum of its two directions).
    W_sparse = W_sparse.minimum(W_sparse.transpose())
    
    return W_sparse



def find_trunc_graph(qs, W, levels = 3):
    needed_idxs = []
    needed_idxs = list(np.nonzero(qs > 0)[0])
    for l in range(levels):
        idid = W.nonzero()[1]
        needed_idxs.extend(list(idid))
        needed_idxs =list(set(needed_idxs))
    return np.array(needed_idxs), W[needed_idxs,:][:,needed_idxs]

def dfs_trunk(sim, A,alpha = 0.99, QUERYKNN = 10, maxiter = 8, K = 100, tol = 1e-3):
    qsim = sim_kernel(sim).T
    sortidxs = np.argsort(-qsim, axis = 1)
    for i in range(len(qsim)):
        qsim[i,sortidxs[i,QUERYKNN:]] = 0
    qsims = sim_kernel(qsim)
    W = sim_kernel(A)
    W = csr_matrix(topK_W(W, K))
    out_ranks = []
    t =time()
    for i in range(qsims.shape[0]):
        qs =  qsims[i,:]
        tt = time() 
        w_idxs, W_trunk = find_trunc_graph(qs, W, 2)
        Wn = normalize_connection_graph(W_trunk)
        Wnn = eye(Wn.shape[0]) - alpha * Wn
        f,inf = s_linalg.minres(Wnn, qs[w_idxs], tol=tol, maxiter=maxiter)
        ranks = w_idxs[np.argsort(-f.reshape(-1))]
        missing = np.setdiff1d(np.arange(A.shape[1]), ranks)
        out_ranks.append(np.concatenate([ranks.reshape(-1,1), missing.reshape(-1,1)], axis = 0))
    #print time() -t, 'qtime'
    out_ranks = np.concatenate(out_ranks, axis = 1)
    return out_ranks

def cg_diffusion(qsims, Wn, alpha = 0.99, maxiter = 10, tol = 1e-3):
    Wnn = eye(Wn.shape[0]) - alpha * Wn
    out_sims = []
    for i in range(qsims.shape[0]):
        #f,inf = s_linalg.cg(Wnn, qsims[i,:], tol=tol, maxiter=maxiter)
        f,inf = s_linalg.minres(Wnn, qsims[i,:], tol=tol, maxiter=maxiter)
        out_sims.append(f.reshape(-1,1))
    out_sims = np.concatenate(out_sims, axis = 1)
    #ranks = np.argsort(-out_sims, axis = 0)
    return out_sims

def fsr_rankR(qsims, Wn, alpha = 0.99, R = 2000):
    vals, vecs = s_linalg.eigsh(Wn, k = R)
    p2 = diags((1.0 - alpha) / (1.0 - alpha*vals))
    vc = csr_matrix(vecs)
    p3 =  vc.dot(p2)
    vc_norm =  (vc.multiply(vc)).sum(axis = 0)
    out_sims = []
    for i in range(qsims.shape[0]):
        qsims_sparse = csr_matrix(qsims[i:i+1,:])
        p1 =(vc.T).dot(qsims_sparse.T)
        diff_sim = csr_matrix(p3)*csr_matrix(p1)
        out_sims.append(diff_sim.todense().reshape(-1,1))
    out_sims = np.concatenate(out_sims, axis = 1)
    #ranks = np.argsort(-out_sims, axis = 0)
    return out_sims