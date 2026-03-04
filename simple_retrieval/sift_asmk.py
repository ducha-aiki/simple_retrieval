"""SIFT + ASMK retrieval implementing the pipeline from:
  "Efficient Image Detail Mining" (Mikulik, Radenovic, Chum, Matas; ACCV 2014)

Four phases:
  1. Visual vocabulary (ASMK codebook) + inverted file indexing and query
  2. Scale-biased re-scoring (log-scale bins, 4-bit / 16 values)
  3. Result grouping (H-backprojection overlap >= 50%) + HQE loop
  4. Geometric consistency test (triple-compose homography check)
"""

import os
import pickle
import numpy as np
import h5py
import cv2
import torch
import kornia.feature as KF
from tqdm import tqdm

import asmk
import asmk.codebook as cdb_pkg
import asmk.inverted_file as ivf_pkg
import asmk.kernel as kern_pkg
import asmk.index as idx_pkg

# ---------------------------------------------------------------------------
# Scale-bin helpers
# ---------------------------------------------------------------------------

N_SCALE_BINS = 16
_LOG2_MIN = -8.0   # log2(scale) range: 2^-8 = 1/256 px to 2^8 = 256 px
_LOG2_MAX =  8.0


def _scales_from_lafs(lafs: np.ndarray) -> np.ndarray:
    """Compute per-keypoint scale (in pixels) from LAF array of shape (..., 2, 3)."""
    lafs = lafs.reshape(-1, 2, 3)
    a, b = lafs[:, 0, 0], lafs[:, 0, 1]
    c, d = lafs[:, 1, 0], lafs[:, 1, 1]
    scale = np.sqrt(np.abs(a * d - b * c))   # sqrt(|det|)
    return scale.astype(np.float32)


def _log2_scale_to_bin(log2_scale: float, n_bins: int = N_SCALE_BINS) -> int:
    """Map a log2-scale value to an integer bin index in [0, n_bins-1]."""
    t = (log2_scale - _LOG2_MIN) / (_LOG2_MAX - _LOG2_MIN) * n_bins
    return int(np.clip(t, 0, n_bins - 1))


def _scales_to_bins(scales: np.ndarray, n_bins: int = N_SCALE_BINS) -> np.ndarray:
    """Map (N,) pixel-scale array to (N,) uint8 bin indices."""
    log2_scales = np.log2(np.clip(scales, 1e-4, None))
    t = (log2_scales - _LOG2_MIN) / (_LOG2_MAX - _LOG2_MIN) * n_bins
    return np.clip(t, 0, n_bins - 1).astype(np.uint8)


def _scale_compat_matrix(n_bins: int = N_SCALE_BINS, sigma: float = 1.0) -> np.ndarray:
    """
    Build NxN matrix S where S[i,j] = exp(-((i-j)*bin_width)^2 / (2*sigma^2)).
    sigma is measured in log2-scale units.
    """
    bin_width = (_LOG2_MAX - _LOG2_MIN) / n_bins
    idx = np.arange(n_bins, dtype=np.float32)
    diff = (idx[:, None] - idx[None, :]) * bin_width      # (N, N) log2-scale diff
    return np.exp(-(diff ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


# ---------------------------------------------------------------------------
# Polygon helpers for result grouping
# ---------------------------------------------------------------------------

def _poly_intersection_area(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Area of intersection of two convex polygons (Nx2 float32 arrays)."""
    p1 = poly1.astype(np.float32).reshape(-1, 1, 2)
    p2 = poly2.astype(np.float32).reshape(-1, 1, 2)
    retval, _ = cv2.intersectConvexConvex(p1, p2)
    return float(retval)


# ---------------------------------------------------------------------------
# Default ASMK params
# ---------------------------------------------------------------------------

def _default_params(vocab_size: int = 65536, topk: int = 1000,
                    binary: bool = True, gpu_id=None,
                    multiple_assignment: int = 1) -> dict:
    """Return a complete ASMK parameter dict."""
    return {
        "index": {"gpu_id": gpu_id},
        "train_codebook": {
            "codebook": {"size": vocab_size},
        },
        "build_ivf": {
            "kernel": {"binary": binary},
            "quantize": {"multiple_assignment": multiple_assignment},
            "aggregate": {},
            "ivf": {"use_idf": True},
        },
        "query_ivf": {
            "quantize": {"multiple_assignment": multiple_assignment},
            "aggregate": {},
            "search": {"topk": topk},
            "similarity": {"alpha": 3.0, "similarity_threshold": 0.0},
        },
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SIFTASMKRetrieval:
    """
    SIFT + ASMK retrieval with scale-biased re-scoring, result grouping, HQE
    loop, and geometric consistency test.

    Typical usage::

        ret = SIFTASMKRetrieval(vocab_size=65536)
        ret.train_vocabulary(feature_dir, fnames, cache_path='vocab.pkl')
        ret.build_ivf(feature_dir, fnames, cache_path='ivf.pkl')
        ret.save_aux(aux_path)           # saves fnames + scale histograms

        # query
        ranks, scores = ret.query(q_descs, topk=1000)
        ranks, scores = ret.scale_biased_rescore(q_lafs, ranks, scores)
        idxs, scores = ret.hqe_query(q_descs, q_lafs, q_hw, feature_dir)
    """

    def __init__(self, vocab_size: int = 65536, topk: int = 1000,
                 binary: bool = True, gpu_id=None,
                 scale_sigma: float = 1.0, multiple_assignment: int = 1):
        self.params = _default_params(vocab_size=vocab_size, topk=topk,
                                      binary=binary, gpu_id=gpu_id,
                                      multiple_assignment=multiple_assignment)
        self.asmk_method = None          # trained + indexed ASMKMethod
        self.fnames: list = []           # ordered DB image paths
        self.db_scale_hists = None       # (n_images, N_SCALE_BINS) float32 normalised histograms
        self.scale_compat = _scale_compat_matrix(sigma=scale_sigma)

    # ------------------------------------------------------------------
    # Phase 1: vocabulary + IVF
    # ------------------------------------------------------------------

    def train_vocabulary(self, feature_dir: str, fnames: list,
                         sample_n: int = 500_000, cache_path: str = None):
        """Sample descriptors and train ASMK codebook."""
        print(f"Loading SIFT descriptors for vocabulary training (sample_n={sample_n})...")
        vecs = self._load_sample_descs(feature_dir, fnames, sample_n)
        print(f"Training codebook on {len(vecs)} descriptors, "
              f"vocab_size={self.params['train_codebook']['codebook']['size']}...")
        m = asmk.ASMKMethod.initialize_untrained(self.params)
        self.asmk_method = m.train_codebook(vecs, cache_path=cache_path)
        print("Codebook trained.")

    def build_ivf(self, feature_dir: str, fnames: list, cache_path: str = None):
        """
        Quantize all DB SIFT descriptors and build inverted file.
        Also collects per-image scale histograms for Phase 2.
        """
        assert self.asmk_method is not None, "Call train_vocabulary() first."
        self.fnames = list(fnames)
        n_images = len(fnames)
        scale_hists = np.zeros((n_images, N_SCALE_BINS), dtype=np.float32)

        print(f"Building IVF for {n_images} images...")
        all_vecs = []
        all_imids = []

        with h5py.File(os.path.join(feature_dir, 'descriptors.h5'), 'r') as f_desc, \
             h5py.File(os.path.join(feature_dir, 'lafs.h5'), 'r') as f_laf:
            for img_idx, fname in enumerate(tqdm(fnames, desc="Loading SIFT")):
                descs = f_desc[fname][...].astype(np.float32)
                if descs.dtype == np.uint8:
                    descs = descs.astype(np.float32) / 512.0
                lafs = f_laf[fname][...]              # (1, N, 2, 3) or (N, 2, 3)
                scales = _scales_from_lafs(lafs)
                bins = _scales_to_bins(scales)        # (N,) uint8

                # Normalised histogram for Phase 2 scale bias
                hist = np.bincount(bins.astype(np.int64), minlength=N_SCALE_BINS).astype(np.float32)
                if hist.sum() > 0:
                    hist /= hist.sum()
                scale_hists[img_idx] = hist

                all_vecs.append(descs)
                all_imids.append(np.full(len(descs), img_idx, dtype=np.int32))

        self.db_scale_hists = scale_hists

        all_vecs_np = np.concatenate(all_vecs, axis=0)
        all_imids_np = np.concatenate(all_imids, axis=0)

        self.asmk_method = self.asmk_method.build_ivf(
            all_vecs_np, all_imids_np, cache_path=cache_path)
        print(f"IVF built. Stats: {self.asmk_method.inverted_file.stats}")

    def save_aux(self, path: str):
        """Save fnames + scale histograms (the parts ASMK cache_path does not cover)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        aux = {
            "fnames": self.fnames,
            "db_scale_hists": self.db_scale_hists,
            "params": self.params,
        }
        with open(path, 'wb') as f:
            pickle.dump(aux, f)
        print(f"Auxiliary state saved to {path}")

    def load_aux(self, path: str):
        """Load fnames + scale histograms saved by save_aux()."""
        with open(path, 'rb') as f:
            aux = pickle.load(f)
        self.fnames = aux["fnames"]
        self.db_scale_hists = aux["db_scale_hists"]
        self.params = aux["params"]
        print(f"Auxiliary state loaded from {path} ({len(self.fnames)} images)")

    def rebuild_ivf(self, vocab_cache: str, ivf_cache: str):
        """
        Reconstruct asmk_method from on-disk caches (fast path, no re-training).
        Must have called load_aux() (or train+build) beforehand to set self.params.
        """
        m = asmk.ASMKMethod.initialize_untrained(self.params)
        # train_codebook with cache → loads without re-clustering
        m2 = m.train_codebook(np.zeros((0, 128), dtype=np.float32), cache_path=vocab_cache)
        # build_ivf with cache → loads without re-indexing
        self.asmk_method = m2.build_ivf(
            np.zeros((0, 128), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            cache_path=ivf_cache)
        print("ASMK method rebuilt from cache.")

    # ------------------------------------------------------------------
    # Phase 1: query
    # ------------------------------------------------------------------

    def query(self, q_descs, topk: int = None) -> tuple:
        """
        Run ASMK query.

        Args:
            q_descs: (N, D) numpy array or torch.Tensor of query descriptors.
            topk:    Number of results to return (None → use params default).
        Returns:
            ranks  (np.ndarray int64):  DB image indices, best first.
            scores (np.ndarray float32): Corresponding similarity scores.
        """
        assert self.asmk_method is not None, "Build IVF first."
        if isinstance(q_descs, torch.Tensor):
            q_descs = q_descs.float().cpu().numpy()
        q_descs = np.ascontiguousarray(q_descs, dtype=np.float32)
        q_imids = np.zeros(len(q_descs), dtype=np.int32)

        step_params = None
        if topk is not None:
            step_params = {**self.params["query_ivf"],
                           "search": {**self.params["query_ivf"]["search"], "topk": topk}}

        meta, images, ranks_2d, scores_2d = self.asmk_method.query_ivf(
            q_descs, q_imids, step_params=step_params)

        return ranks_2d[0].astype(np.int64), scores_2d[0].astype(np.float32)

    # ------------------------------------------------------------------
    # Phase 2: scale-biased re-scoring
    # ------------------------------------------------------------------

    def scale_biased_rescore(self, q_lafs, shortlist_idxs: np.ndarray,
                              shortlist_scores: np.ndarray) -> np.ndarray:
        """
        Re-weight ASMK scores by query↔DB scale histogram compatibility.

        For each shortlisted image, the weight is the dot product of the
        query scale histogram with the scale-compatibility-weighted DB
        histogram: w_i = q_hist @ scale_compat @ db_hist[i].

        Args:
            q_lafs:           Query LAFs as numpy array (1, N, 2, 3) or torch.Tensor.
            shortlist_idxs:   DB image indices from query().
            shortlist_scores: ASMK scores from query().
        Returns:
            re-weighted scores (same shape).
        """
        if self.db_scale_hists is None:
            return shortlist_scores

        if isinstance(q_lafs, torch.Tensor):
            q_lafs = q_lafs.float().cpu().numpy()
        q_scales = _scales_from_lafs(q_lafs)
        q_bins = _scales_to_bins(q_scales)
        q_hist = np.bincount(q_bins.astype(np.int64), minlength=N_SCALE_BINS).astype(np.float32)
        if q_hist.sum() > 0:
            q_hist /= q_hist.sum()

        # weighted_q[j] = sum_i q_hist[i] * scale_compat[i, j]
        weighted_q = q_hist @ self.scale_compat                # (N_SCALE_BINS,)

        new_scores = shortlist_scores.copy()
        for rank_pos, img_idx in enumerate(shortlist_idxs):
            db_hist = self.db_scale_hists[img_idx]             # (N_SCALE_BINS,)
            compat = float(np.dot(weighted_q, db_hist))        # scalar in [0, 1]
            new_scores[rank_pos] *= compat
        return new_scores

    # ------------------------------------------------------------------
    # Phase 3: result grouping
    # ------------------------------------------------------------------

    @staticmethod
    def group_results_by_overlap(shortlist_idxs: np.ndarray, Hs: list,
                                  hw_query: tuple, overlap_thresh: float = 0.5) -> list:
        """
        Greedy grouping of retrieved images by query-box backprojection IoU.

        Images are assumed already sorted best-first. Each group is a list of
        positions into shortlist_idxs (i.e. indices into Hs).

        Returns:
            groups: list of lists of ints (positions in shortlist_idxs).
        """
        hq, wq = hw_query[:2]
        query_corners = np.float32([[0, 0], [wq, 0], [wq, hq], [0, hq]])

        # Project query box into each DB image
        projected = []
        for H in Hs:
            if H is None or np.linalg.norm(H) < 1e-8:
                projected.append(None)
                continue
            try:
                pts = cv2.perspectiveTransform(query_corners.reshape(1, -1, 2), H)
                projected.append(pts.reshape(-1, 2))
            except Exception:
                projected.append(None)

        n = len(shortlist_idxs)
        assigned = [False] * n
        groups = []

        for i in range(n):
            if assigned[i] or projected[i] is None:
                continue
            group = [i]
            assigned[i] = True
            hull_i = cv2.convexHull(projected[i].astype(np.float32))
            area_i = cv2.contourArea(hull_i)

            for j in range(i + 1, n):
                if assigned[j] or projected[j] is None:
                    continue
                hull_j = cv2.convexHull(projected[j].astype(np.float32))
                area_j = cv2.contourArea(hull_j)
                inter = _poly_intersection_area(hull_i.reshape(-1, 2), hull_j.reshape(-1, 2))
                union = area_i + area_j - inter
                iou = inter / (union + 1e-6)
                if iou >= overlap_thresh:
                    group.append(j)
                    assigned[j] = True
            groups.append(group)

        return groups

    # ------------------------------------------------------------------
    # Phase 4: geometric consistency
    # ------------------------------------------------------------------

    @staticmethod
    def geometric_consistency(H_qi: np.ndarray, H_qj: np.ndarray,
                               H_ij: np.ndarray, thresh: float = 0.1) -> bool:
        """
        Check triple-compose consistency: H_qi ≈ H_qj @ H_ij.

        H_qi: homography query → image_i   (3×3)
        H_qj: homography query → image_j   (3×3)
        H_ij: homography image_i → image_j (3×3)

        Returns True if the Frobenius norm of the difference (on the
        normalised 3×3 matrices, excluding the last row) is below thresh.
        """
        for H in (H_qi, H_qj, H_ij):
            if H is None or np.linalg.norm(H) < 1e-8:
                return False
        H_qi_n = H_qi / (H_qi[2, 2] + 1e-12)
        H_qj_n = H_qj / (H_qj[2, 2] + 1e-12)
        H_ij_n = H_ij / (H_ij[2, 2] + 1e-12)
        H_composed = H_qj_n @ H_ij_n
        H_composed /= (H_composed[2, 2] + 1e-12)
        diff = np.linalg.norm(H_composed[:2, :] - H_qi_n[:2, :], 'fro')
        return diff < thresh

    # ------------------------------------------------------------------
    # Full HQE pipeline
    # ------------------------------------------------------------------

    def hqe_query(self, q_descs, q_lafs, q_hw, feature_dir: str,
                  topk_asmk: int = 1000, topk_verify: int = 50,
                  max_hqe_iters: int = 2, overlap_thresh: float = 0.5,
                  consistency_thresh: float = 0.1,
                  device: torch.device = torch.device('cpu'),
                  matching_method: str = 'smnn',
                  inl_th: float = 3.0, num_ransac_iter: int = 1000) -> tuple:
        """
        Full HQE pipeline.

        Each iteration:
          1. ASMK query (with scale-bias re-scoring)
          2. RANSAC verification on top ``topk_verify`` results
          3. Result grouping by H-backprojection overlap
          4. Geometric consistency check on each group (triple-compose)
          5. Expand query descriptors with the best consistent group
        Repeat for ``max_hqe_iters`` iterations.

        Returns:
            sorted_idxs  (np.ndarray int64):  DB indices, best-first.
            final_scores (np.ndarray float32): Corresponding merged scores.
        """
        from simple_retrieval.local_feature import match_query_to_db, spatial_scoring

        if isinstance(q_descs, torch.Tensor):
            q_descs_np = q_descs.float().cpu().numpy()
        else:
            q_descs_np = np.ascontiguousarray(q_descs, dtype=np.float32)

        if isinstance(q_lafs, torch.Tensor):
            q_lafs_np = q_lafs.float().cpu().numpy()
        else:
            q_lafs_np = np.asarray(q_lafs, dtype=np.float32)

        hw = tuple(int(x) for x in q_hw[:2])

        # accumulated best score per DB image
        acc_scores: dict = {}

        cur_descs = q_descs_np
        cur_lafs  = q_lafs_np

        for iteration in range(max_hqe_iters + 1):
            print(f"\n[HQE iter {iteration}] Running ASMK query with {len(cur_descs)} descriptors...")
            ranks, scores = self.query(cur_descs, topk=topk_asmk)

            # Phase 2: scale-biased re-scoring
            scores = self.scale_biased_rescore(cur_lafs, ranks, scores)

            # Merge into accumulated scores
            for idx, sc in zip(ranks, scores):
                key = int(idx)
                if sc > acc_scores.get(key, -1e9):
                    acc_scores[key] = float(sc)

            if iteration == max_hqe_iters:
                break

            # Phase 3+4: geo verification on top-K
            verify_idxs = ranks[:topk_verify]
            verify_fnames = [self.fnames[i] for i in verify_idxs]
            dtype = torch.float16 if 'cuda' in str(device) else torch.float32

            cur_descs_t = torch.from_numpy(cur_descs).to(device, dtype)
            cur_lafs_t  = torch.from_numpy(cur_lafs).to(device, dtype)
            hw_t = torch.tensor(hw, dtype=dtype).to(device)

            print(f"[HQE iter {iteration}] Verifying {len(verify_fnames)} candidates...")
            with torch.inference_mode():
                mkps, hw2_list = match_query_to_db(
                    cur_descs_t, cur_lafs_t, hw_t,
                    feature_dir, verify_fnames,
                    feature_name='sift', matching_method=matching_method,
                    device=device)

            geo_scores, Hs = spatial_scoring(
                mkps, criterion='num_inliers',
                config={'inl_th': inl_th, 'num_iter': num_ransac_iter})

            # Boost accumulated scores by geo evidence
            for pos, img_idx in enumerate(verify_idxs):
                if geo_scores[pos] > 0:
                    # Normalise geo score to [0,1] range and combine
                    norm_geo = float(geo_scores[pos]) / (float(geo_scores[pos]) + 100.0)
                    key = int(img_idx)
                    acc_scores[key] = max(acc_scores.get(key, 0.0),
                                          acc_scores.get(key, 0.0) * 0.5 + norm_geo * 0.5)

            # Phase 3: group by H-backprojection overlap
            groups = self.group_results_by_overlap(
                verify_idxs, Hs, hw, overlap_thresh=overlap_thresh)

            if not groups:
                print("[HQE] No groups found, stopping expansion.")
                break

            # Phase 4: find the largest geometrically consistent group
            best_group_positions = self._find_consistent_group(
                groups, Hs, verify_fnames, feature_dir, device, dtype,
                matching_method, inl_th, num_ransac_iter, consistency_thresh)

            if not best_group_positions:
                print("[HQE] No consistent group found, stopping expansion.")
                break

            print(f"[HQE iter {iteration}] Expanding query with "
                  f"{len(best_group_positions)} images.")

            # Phase 5: expand query with features from the consistent group
            with h5py.File(os.path.join(feature_dir, 'descriptors.h5'), 'r') as f_desc, \
                 h5py.File(os.path.join(feature_dir, 'lafs.h5'), 'r') as f_laf:
                new_descs = [cur_descs]
                new_lafs  = [cur_lafs]
                for pos in best_group_positions:
                    fname = verify_fnames[pos]
                    d = f_desc[fname][...].astype(np.float32)
                    if d.dtype == np.uint8:
                        d = d.astype(np.float32) / 512.0
                    l = f_laf[fname][...]           # (1, N, 2, 3)
                    new_descs.append(d)
                    new_lafs.append(l.reshape(1, -1, 2, 3))
                cur_descs = np.concatenate(new_descs, axis=0)
                cur_lafs  = np.concatenate(new_lafs, axis=1)  # concat along keypoint axis

        # Build final sorted result
        final_idxs   = np.array(
            sorted(acc_scores.keys(), key=lambda x: -acc_scores[x]), dtype=np.int64)
        final_scores = np.array([acc_scores[i] for i in final_idxs], dtype=np.float32)
        return final_idxs, final_scores

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_consistent_group(self, groups: list, Hs: list, verify_fnames: list,
                                feature_dir: str, device, dtype,
                                matching_method: str, inl_th: float,
                                num_ransac_iter: int, thresh: float) -> list:
        """
        For each group, check pairwise geometric consistency (triple-compose).
        Return positions (into verify_fnames / Hs) of the best consistent sub-group.
        """
        from simple_retrieval.local_feature import match_feature_pair, spatial_scoring

        best: list = []
        for group in groups:
            if len(group) == 0:
                continue
            if len(group) == 1:
                if len(group) > len(best):
                    best = group
                continue

            consistent = [group[0]]           # seed: highest-scoring image in group
            seed_pos   = group[0]
            H_qi       = Hs[seed_pos]

            for pos in group[1:]:
                H_qj = Hs[pos]
                fname_i = verify_fnames[seed_pos]
                fname_j = verify_fnames[pos]
                try:
                    mkpts1, mkpts2, hw1, hw2 = match_feature_pair(
                        fname_i, fname_j, feature_dir,
                        matching_method=matching_method, device=device)
                    if len(mkpts1) < 10:
                        continue
                    H_ij, inliers = cv2.findHomography(
                        mkpts1.cpu().numpy(), mkpts2.cpu().numpy(),
                        cv2.USAC_MAGSAC, inl_th, 0.999, num_ransac_iter)
                    if H_ij is None:
                        continue
                    if self.geometric_consistency(H_qi, H_qj, H_ij, thresh=thresh):
                        consistent.append(pos)
                except Exception as e:
                    print(f"[HQE consistency] Error: {e}")
                    continue

            if len(consistent) > len(best):
                best = consistent

        return best

    def _load_sample_descs(self, feature_dir: str, fnames: list,
                            sample_n: int) -> np.ndarray:
        """Load a random sample of descriptors from h5 for codebook training."""
        all_descs = []
        with h5py.File(os.path.join(feature_dir, 'descriptors.h5'), 'r') as f:
            n_per_img = max(1, sample_n // len(fnames))
            for fname in tqdm(fnames, desc="Sampling descriptors"):
                d = f[fname][...]
                if d.dtype == np.uint8:
                    d = d.astype(np.float32) / 512.0
                else:
                    d = d.astype(np.float32)
                if len(d) > n_per_img:
                    idx = np.random.choice(len(d), n_per_img, replace=False)
                    d = d[idx]
                all_descs.append(d)
        return np.concatenate(all_descs, axis=0)
