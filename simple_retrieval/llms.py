"""
VLM-based shortlist resorting for image retrieval.

Supports Qwen2-VL, Gemma3, and PaliGemma for zoom-in / zoom-out / relevance scoring.
Model is specified by a short name or a full HuggingFace model ID.
"""

import hashlib
import json
import os
import re
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

# Disable all HuggingFace download timeouts so slow connections don't fail
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "0")   # 0 = no timeout
os.environ.setdefault("HUGGINGFACE_HUB_VERBOSITY", "warning")

# ---------------------------------------------------------------------------
# Model name shortcuts
# ---------------------------------------------------------------------------
MODEL_SHORTCUTS = {
    # Qwen2-VL (transformers ≥ 4.48)
    "qwen2vl-2b":       "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2vl-7b":       "Qwen/Qwen2-VL-7B-Instruct",
    # Qwen2.5-VL (transformers ≥ 4.52)
    "qwen2.5vl-3b":     "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5vl-7b":     "Qwen/Qwen2.5-VL-7B-Instruct",
    # Gemma 3 with vision (transformers ≥ 4.51; gated — requires HF login)
    "gemma3-4b":        "google/gemma-3-4b-it",
    "gemma3-12b":       "google/gemma-3-12b-it",
    "gemma3-27b":       "google/gemma-3-27b-it",
    # DeepSeek-VL (transformers ≥ 4.57; deepseek-community namespace)
    "deepseek-vl-1.3b": "deepseek-community/deepseek-vl-1.3b-chat",
    "deepseek-vl-7b":   "deepseek-community/deepseek-vl-7b-chat",
    # PaliGemma (single-image VQA, falls back to composite image)
    "paligemma-3b":     "google/paligemma-3b-mix-448",
}

# Categories the VLM can return
ZOOM_IN   = "zoom_in"
ZOOM_OUT  = "zoom_out"
SAME      = "same"
IRRELEVANT = "irrelevant"

# Score weights used for each criterion
# criterion → {category: weight}
_CRITERION_WEIGHTS = {
    "vlm_zoom_in":  {ZOOM_IN: 1.0, SAME: 0.3, ZOOM_OUT: 0.1, IRRELEVANT: 0.0},
    "vlm_zoom_out": {ZOOM_OUT: 1.0, SAME: 0.3, ZOOM_IN: 0.1, IRRELEVANT: 0.0},
    "vlm_relevant": {ZOOM_IN: 1.0, ZOOM_OUT: 1.0, SAME: 1.0, IRRELEVANT: 0.0},
}


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------

def resolve_model_name(name: str) -> str:
    """Expand a shortcut to a full HuggingFace model ID."""
    return MODEL_SHORTCUTS.get(name.lower(), name)


def get_model_family(model_id: str) -> str:
    mid = model_id.lower()
    if "qwen2.5-vl" in mid or "qwen2.5_vl" in mid:
        return "qwen2vl"   # same chat API as Qwen2-VL
    if "qwen2-vl" in mid or "qwen2_vl" in mid:
        return "qwen2vl"
    if "paligemma" in mid:
        return "paligemma"
    if "gemma-3" in mid or "gemma3" in mid:
        return "gemma3"
    if "deepseek-vl" in mid or "deepseek_vl" in mid:
        return "deepseek"
    return "auto"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    raise TypeError(f"Expected PIL Image or np.ndarray, got {type(img)}")


def draw_bbox(img: Image.Image, bbox: List[int],
              color: str = "red", width: int = 5) -> Image.Image:
    """Draw bbox [x1,y1,x2,y2] in pixel coords on image."""
    img = img.copy()
    ImageDraw.Draw(img).rectangle(bbox, outline=color, width=width)
    return img


def crop_bbox(img: Image.Image, bbox: Optional[List[int]]) -> Image.Image:
    """Crop image to [x1,y1,x2,y2]; return full image if bbox is None."""
    if bbox is None:
        return img
    return img.crop(bbox)


def resize_for_vlm(img: Image.Image, max_size: int = 448) -> Image.Image:
    """Cap the longest edge of an image to `max_size` pixels.

    Oxford5k images can exceed 3000px; feeding them raw to Qwen2-VL blows
    up VRAM because the visual encoder tokenises every patch.  Resizing to
    ≤448px keeps each image at ~256 visual tokens (16×16 ViT patches).
    """
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                      Image.LANCZOS)


def make_side_by_side(left: Image.Image, right: Image.Image,
                      target_height: int = 448) -> Image.Image:
    """Resize both images to the same height and paste side-by-side."""
    def _resize_h(im, h):
        w = max(1, int(im.width * h / im.height))
        return im.resize((w, h), Image.LANCZOS)

    left  = _resize_h(left,  target_height)
    right = _resize_h(right, target_height)
    canvas = Image.new("RGB", (left.width + right.width, target_height))
    canvas.paste(left,  (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def load_vlm(model_name: str, device: str = "cuda",
             dtype: torch.dtype = torch.float16):
    """Load (and cache) a VLM model + processor.

    Returns:
        model, processor, family (str)
    """
    model_id = resolve_model_name(model_name)
    cache_key = (model_id, device)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    family = get_model_family(model_id)
    print(f"Loading VLM '{model_id}' (family={family}) on {device} …")

    if family == "qwen2vl":
        try:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            processor = Qwen2VLProcessor.from_pretrained(model_id)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, device_map=device)
        except Exception:
            # Qwen2.5-VL or future variant — fall back to Auto classes
            from transformers import AutoModelForVision2Seq, AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForVision2Seq.from_pretrained(
                model_id, torch_dtype=dtype, device_map=device)
    elif family == "gemma3":
        from transformers import Gemma3ForConditionalGeneration, Gemma3Processor
        try:
            processor = Gemma3Processor.from_pretrained(model_id)
            # Gemma3 is trained in bfloat16; using float16 causes degenerate
            # (all-zero) generation outputs regardless of the caller's dtype.
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map=device)
        except OSError as e:
            if "gated repo" in str(e) or "401" in str(e):
                raise RuntimeError(
                    f"Gemma 3 is a gated model. You must:\n"
                    f"  1. Accept the license at https://huggingface.co/{model_id}\n"
                    f"  2. Run:  huggingface-cli login  (or set HF_TOKEN env var)\n"
                    f"Original error: {e}"
                ) from e
            raise
    elif family == "deepseek":
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id)
        if cfg.model_type == "deepseek_vl_hybrid":
            from transformers import (DeepseekVLHybridForConditionalGeneration,
                                      DeepseekVLHybridProcessor)
            processor = DeepseekVLHybridProcessor.from_pretrained(model_id)
            model = DeepseekVLHybridForConditionalGeneration.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map=device)
        else:  # deepseek_vl (1.3B / 7B non-hybrid)
            from transformers import (DeepseekVLForConditionalGeneration,
                                      DeepseekVLProcessor)
            processor = DeepseekVLProcessor.from_pretrained(model_id)
            model = DeepseekVLForConditionalGeneration.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map=device)
    elif family == "paligemma":
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, device_map=device)
    else:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, dtype=dtype, device_map=device)

    model.eval()
    _model_cache[cache_key] = (model, processor, family)
    print(f"VLM loaded.")
    return model, processor, family


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_QWEN_SYSTEM = (
    "You are a precise visual analysis assistant. "
    "Answer only with the exact format requested."
)

_SCALE_INSTRUCTION = (
    "You are given two images.\n"
    "• Image 1 (query): shows the scene / region of interest.\n"
    "• Image 2 (candidate): a retrieved database image.\n\n"
    "Task: decide whether Image 2 shows the SAME content as Image 1, "
    "and at what relative scale.\n\n"
    "Reply with EXACTLY one line in this format:\n"
    "  ZOOM_IN <conf>   — content visible in Image 2 at LARGER scale (zoomed in)\n"
    "  ZOOM_OUT <conf>  — content visible in Image 2 at SMALLER scale (zoomed out)\n"
    "  SAME <conf>      — content visible in Image 2 at roughly SAME scale\n"
    "  IRRELEVANT       — content NOT visible in Image 2\n"
    "where <conf> is a confidence 0.1–1.0.\n"
    "Output nothing else."
)

_PALIGEMMA_PREFIX = (
    "The left half is a query image. The right half is a candidate. "
    "Does the candidate show the same content as the query at: "
    "zoom_in (larger scale), zoom_out (smaller scale), same scale, or irrelevant? "
    "Answer:"
)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _infer_qwen2vl(model, processor, images: List[Image.Image],
                   prompt: str, device: str, max_new_tokens: int = 32,
                   max_image_size: int = 448) -> str:
    """Run Qwen2-VL chat inference with a list of images.

    Images are resized to `max_image_size` on their longest edge before
    tokenisation to avoid VRAM explosions on high-resolution inputs.
    """
    images = [resize_for_vlm(img, max_image_size) for img in images]

    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)
    # Trim prompt tokens
    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_ids, skip_special_tokens=True).strip()


def _infer_paligemma(model, processor, composite: Image.Image,
                     prefix: str, device: str, max_new_tokens: int = 16) -> str:
    """Run PaliGemma VQA inference on a composite (side-by-side) image.

    PaliGemma requires an explicit <image> token at the start of the text;
    without it the processor warns and may not bind the image features correctly.
    """
    text = "<image>" + prefix
    inputs = processor(text=text, images=composite, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)
    return processor.decode(out_ids[0][input_len:], skip_special_tokens=True).strip()


def _infer_gemma3(model, processor, images: List[Image.Image],
                  prompt: str, device: str, max_new_tokens: int = 32,
                  max_image_size: int = 448) -> str:
    """Run Gemma3 chat inference with multiple images.

    Gemma3 uses the same chat-template + separate images API as Qwen2-VL.
    Each image is referenced by an {"type": "image"} entry in the content list.
    """
    images = [resize_for_vlm(img, max_image_size) for img in images]

    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)
    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_ids, skip_special_tokens=True).strip()


def _infer_deepseek(model, processor, images: List[Image.Image],
                    prompt: str, device: str, max_new_tokens: int = 32,
                    max_image_size: int = 448) -> str:
    """Run DeepSeek-VL chat inference with multiple images.

    DeepSeek-VL uses the standard chat-template API with <image_placeholder>
    tokens.  Images are resized before tokenisation to avoid VRAM spikes.
    """
    images = [resize_for_vlm(img, max_image_size) for img in images]

    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt")
    # Cast float tensors to the model's dtype (bfloat16) — the hybrid model's
    # high-res vision encoder (SAM-based) errors if pixel_values stay float32.
    model_dtype = next(model.parameters()).dtype
    inputs = {
        k: v.to(device=device, dtype=model_dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)
    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_vlm_response(response: str) -> Tuple[str, float]:
    """Parse VLM text output → (category, confidence).

    Handles:
      "ZOOM_IN 0.9"    → (zoom_in, 0.9)
      "ZOOM_IN <0.9>"  → (zoom_in, 0.9)   # Gemma3 wraps conf in angle brackets
      "ZOOM_OUT 0.7"   → (zoom_out, 0.7)
      "SAME 0.85"      → (same, 0.85)
      "IRRELEVANT"     → (irrelevant, 0.0)
      plus lowercase / partial matches
    """
    r = response.strip().upper()

    # Match "<CATEGORY> <float>"  or  "CATEGORY <0.9>"  (angle-bracket variant)
    m = re.search(r"(ZOOM_IN|ZOOM_OUT|SAME|IRRELEVANT)\s*<?([0-9.]+)>?", r)
    if m:
        cat_raw = m.group(1)
        conf_str = m.group(2)
        cat = cat_raw.lower()  # zoom_in / zoom_out / same / irrelevant
        try:
            conf = float(conf_str)
            conf = max(0.0, min(1.0, conf))
        except ValueError:
            conf = 0.5 if cat != IRRELEVANT else 0.0
        return cat, conf

    # Fallback keyword scan
    if "IRRELEVANT" in r or "NOT VISIBLE" in r or "NOT PRESENT" in r:
        return IRRELEVANT, 0.0
    if "ZOOM_IN" in r or "LARGER" in r or "CLOSER" in r or "BIGGER" in r:
        return ZOOM_IN, 0.5
    if "ZOOM_OUT" in r or "SMALLER" in r or "FARTHER" in r or "WIDER" in r:
        return ZOOM_OUT, 0.5
    if "SAME" in r or "SIMILAR" in r:
        return SAME, 0.5

    return IRRELEVANT, 0.0


def score_from_parse(category: str, confidence: float, criterion: str) -> float:
    """Convert parsed (category, confidence) to a scalar resort score."""
    weights = _CRITERION_WEIGHTS.get(criterion, _CRITERION_WEIGHTS["vlm_relevant"])
    return weights.get(category, 0.0) * confidence


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def resort_shortlist_vlm(
    query_img,
    shortlist: np.ndarray,
    db_fnames: List[str],
    model_name: str = "qwen2vl-2b",
    criterion: str = "vlm_zoom_in",
    bbox: Optional[List[int]] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Re-rank a retrieval shortlist using a VLM zoom/relevance score.

    Args:
        query_img:   Query image (PIL Image or np.ndarray, RGB).
        shortlist:   1-D array of DB indices (as returned by get_shortlist).
        db_fnames:   Full list of DB filenames (indexed by shortlist values).
        model_name:  Short name (see MODEL_SHORTCUTS) or full HF model ID.
        criterion:   One of 'vlm_zoom_in', 'vlm_zoom_out', 'vlm_relevant'.
        bbox:        Optional [x1,y1,x2,y2] in pixel coords of the query ROI.
                     If given, the query is cropped to this region for the VLM.
                     If None, the full query image is used (with no annotation).
        device:      'cuda' or 'cpu'.
        verbose:     Print per-image VLM response if True.

    Returns:
        sorted_idxs:   shortlist re-ranked by descending VLM score.
        sorted_scores: corresponding scores (higher = better match).
    """
    query_pil = _to_pil(query_img)

    # --- Prepare query representation for the VLM ---
    if bbox is not None:
        # Show annotated full image + cropped region as a two-image pair
        query_annotated = draw_bbox(query_pil, bbox)
        query_crop      = crop_bbox(query_pil, bbox)
        query_for_vlm   = [query_annotated, query_crop]   # list of images
        prompt_extra    = (
            "Image 1 (query) has a red bounding box marking the region of interest. "
            "Image 2 (query crop) shows only that region. "
            "Image 3 (candidate): the database image to evaluate.\n\n"
        )
        # Insert crop as extra image for qwen (3-image), or combine for paligemma
    else:
        query_for_vlm = [query_pil]
        prompt_extra  = ""

    model, processor, family = load_vlm(model_name, device=device)

    scores = np.zeros(len(shortlist), dtype=np.float32)
    fnames = [db_fnames[i] for i in shortlist]

    it = tqdm(enumerate(fnames), total=len(fnames),
              desc=f"VLM resort ({criterion})")

    for rank, fname in it:
        try:
            candidate_pil = Image.open(fname).convert("RGB")
        except Exception as e:
            if verbose:
                print(f"  [skip {fname}]: {e}")
            continue

        # --- Compose and run inference ---
        all_images = query_for_vlm + [candidate_pil]
        if len(all_images) == 3:
            header = (
                "You are given three images: "
                "Image 1 = query with red bbox, "
                "Image 2 = cropped region of interest, "
                "Image 3 = candidate database image.\n\n"
            )
        else:
            header = (
                "You are given two images: "
                "Image 1 = query, "
                "Image 2 = candidate database image.\n\n"
            )
        full_prompt = header + _SCALE_INSTRUCTION

        if family == "qwen2vl":
            response = _infer_qwen2vl(model, processor, all_images, full_prompt, device)
        elif family == "gemma3":
            response = _infer_gemma3(model, processor, all_images, full_prompt, device)
        elif family == "deepseek":
            response = _infer_deepseek(model, processor, all_images, full_prompt, device)
        elif family == "paligemma":
            left_img = query_crop if bbox is not None else query_pil
            composite = make_side_by_side(left_img, candidate_pil)
            response = _infer_paligemma(model, processor, composite, _PALIGEMMA_PREFIX, device)
        else:
            # Fallback: AutoModelForVision2Seq with chat template
            response = _infer_qwen2vl(model, processor, all_images, full_prompt, device)

        category, confidence = parse_vlm_response(response)
        score = score_from_parse(category, confidence, criterion)
        scores[rank] = score

        if verbose:
            it.write(f"  [{rank:3d}] {category:10s} {confidence:.2f} → {score:.3f}  |  {response!r}")

    sorted_idxs = np.argsort(-scores)
    return shortlist[sorted_idxs], scores[sorted_idxs]


def unload_vlm(model_name: str, device: str = "cuda") -> None:
    """Remove a cached VLM from memory."""
    model_id = resolve_model_name(model_name)
    key = (model_id, device)
    if key in _model_cache:
        model, _, _ = _model_cache.pop(key)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Unloaded VLM '{model_id}'.")


# ---------------------------------------------------------------------------
# Disk response cache
# ---------------------------------------------------------------------------

class ResponseCache:
    """Persistent JSON cache mapping (model, query_hash, candidate, bbox) → (category, conf).

    This avoids re-running VLM inference when the same query/candidate pair
    has already been evaluated in a previous run.
    """

    def __init__(self, cache_dir: str):
        self.path = os.path.join(cache_dir, "vlm_response_cache.json")
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.exists(self.path):
            with open(self.path) as f:
                self._data: Dict[str, list] = json.load(f)
        else:
            self._data = {}
        self._dirty = False

    @staticmethod
    def _key(model_id: str, query_hash: str, candidate_fname: str,
             bbox: Optional[List[int]]) -> str:
        raw = f"{model_id}|{query_hash}|{candidate_fname}|{bbox}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, model_id: str, query_hash: str, candidate_fname: str,
            bbox: Optional[List[int]]) -> Optional[Tuple[str, float]]:
        val = self._data.get(self._key(model_id, query_hash, candidate_fname, bbox))
        if val is None:
            return None
        return val[0], float(val[1])

    def put(self, model_id: str, query_hash: str, candidate_fname: str,
            bbox: Optional[List[int]], category: str, confidence: float) -> None:
        self._data[self._key(model_id, query_hash, candidate_fname, bbox)] = [category, confidence]
        self._dirty = True

    def flush(self) -> None:
        if self._dirty:
            with open(self.path, "w") as f:
                json.dump(self._data, f)
            self._dirty = False

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass


def _image_hash(img: Image.Image) -> str:
    """Cheap perceptual hash for cache keying (not for security)."""
    thumb = img.resize((16, 16), Image.BILINEAR).tobytes()
    return hashlib.md5(thumb).hexdigest()


# ---------------------------------------------------------------------------
# Collective (all-shortlist-at-once) prompts
# ---------------------------------------------------------------------------

def _build_collective_prompt_qwen(n_candidates: int, has_bbox: bool) -> str:
    """Build a single prompt that asks the VLM to score ALL candidates at once."""
    if has_bbox:
        header = (
            f"You are given {n_candidates + 2} images:\n"
            "• Image 1: query image with a red bounding box marking the region of interest.\n"
            "• Image 2: cropped region of interest from the query.\n"
            f"• Images 3–{n_candidates + 2}: candidate database images.\n\n"
        )
        ref = "the region shown in Images 1–2"
    else:
        header = (
            f"You are given {n_candidates + 1} images:\n"
            "• Image 1: the query image.\n"
            f"• Images 2–{n_candidates + 1}: candidate database images.\n\n"
        )
        ref = "the content in Image 1"

    offset = 3 if has_bbox else 2
    body = (
        f"For each candidate image, determine if it shows {ref} "
        "and at what relative scale.\n\n"
        "Output EXACTLY one line per candidate in this format:\n"
        "  Image_<k>: ZOOM_IN <conf>   (same content, LARGER scale / zoomed in)\n"
        "  Image_<k>: ZOOM_OUT <conf>  (same content, SMALLER scale / zoomed out)\n"
        "  Image_<k>: SAME <conf>      (same content, similar scale)\n"
        "  Image_<k>: IRRELEVANT       (content not present)\n"
        "where <conf> is 0.1–1.0 and k is the image number.\n\n"
        f"Output lines for Image_{offset} through Image_{offset + n_candidates - 1}. "
        "Output nothing else."
    )
    return header + body


def _parse_collective_response(response: str, n_candidates: int,
                                offset: int) -> List[Tuple[str, float]]:
    """Parse multi-line collective VLM output into a list of (category, conf)."""
    results: List[Tuple[str, float]] = [(IRRELEVANT, 0.0)] * n_candidates

    pattern = re.compile(
        r"Image_?(\d+)\s*[:\-]\s*(ZOOM_IN|ZOOM_OUT|SAME|IRRELEVANT)\s*<?([0-9.]*)>?",
        re.IGNORECASE,
    )
    for m in pattern.finditer(response.upper()):
        img_num = int(m.group(1))
        idx = img_num - offset            # 0-based candidate index
        if 0 <= idx < n_candidates:
            cat = m.group(2).lower()
            cat = cat.replace("zoom_in", ZOOM_IN).replace("zoom_out", ZOOM_OUT)
            conf_str = m.group(3)
            try:
                conf = max(0.0, min(1.0, float(conf_str)))
            except ValueError:
                conf = 0.5 if cat != IRRELEVANT else 0.0
            results[idx] = (cat, conf)

    return results


# ---------------------------------------------------------------------------
# Collective resorting (whole shortlist in one / few VLM calls)
# ---------------------------------------------------------------------------

def resort_shortlist_vlm_collective(
    query_img,
    shortlist: np.ndarray,
    db_fnames: List[str],
    model_name: str = "qwen2vl-2b",
    criterion: str = "vlm_zoom_in",
    bbox: Optional[List[int]] = None,
    device: str = "cuda",
    max_images_per_call: int = 8,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Re-rank a shortlist by sending all candidates in a single VLM call.

    Instead of one VLM call per candidate, this submits all candidates
    together so the model can compare them against each other — which
    produces more consistent relative rankings.  When the shortlist is
    larger than `max_images_per_call`, it is split into chunks (each
    chunk is one VLM call) and scores are merged.

    A disk cache (JSON) is maintained under `cache_dir` so individual
    (query, candidate) pairs are not re-evaluated on re-runs.  Pass
    ``cache_dir=None`` to disable caching.

    Args:
        query_img:           Query image (PIL Image or np.ndarray, RGB).
        shortlist:           1-D int array of DB indices.
        db_fnames:           Full DB filename list (indexed by shortlist values).
        model_name:          Short name or full HF model ID.
        criterion:           'vlm_zoom_in', 'vlm_zoom_out', or 'vlm_relevant'.
        bbox:                Optional [x1,y1,x2,y2] query ROI in pixel coords.
        device:              'cuda' or 'cpu'.
        max_images_per_call: Max number of candidate images per VLM call.
                             Qwen2-VL-2B handles ~16–24 comfortably.
        cache_dir:           Directory for persistent response cache.
                             Defaults to './tmp/vlm_cache'. Use None to disable.
        verbose:             Print chunk-level progress.

    Returns:
        sorted_idxs:   shortlist re-ranked by descending VLM score.
        sorted_scores: corresponding scores.
    """
    if cache_dir is None:
        cache_dir = "./tmp/vlm_cache"
    cache = ResponseCache(cache_dir)

    query_pil = _to_pil(query_img)
    query_hash = _image_hash(query_pil)
    model_id = resolve_model_name(model_name)

    has_bbox = bbox is not None
    if has_bbox:
        query_annotated = draw_bbox(query_pil, bbox)
        query_crop      = crop_bbox(query_pil, bbox)
        query_images    = [query_annotated, query_crop]
        offset          = 3    # candidates start at Image_3
    else:
        query_images = [query_pil]
        offset       = 2       # candidates start at Image_2

    model, processor, family = load_vlm(model_name, device=device)

    fnames = [db_fnames[i] for i in shortlist]
    scores = np.zeros(len(shortlist), dtype=np.float32)

    # ----------------------------------------------------------------
    # Load all candidate images (skipping broken files)
    # ----------------------------------------------------------------
    candidate_pils: List[Optional[Image.Image]] = []
    for fname in fnames:
        try:
            candidate_pils.append(Image.open(fname).convert("RGB"))
        except Exception as e:
            if verbose:
                print(f"  [skip {fname}]: {e}")
            candidate_pils.append(None)

    # ----------------------------------------------------------------
    # Check cache for already-evaluated pairs
    # ----------------------------------------------------------------
    cached_mask = np.zeros(len(shortlist), dtype=bool)
    for rank, fname in enumerate(fnames):
        hit = cache.get(model_id, query_hash, fname, bbox)
        if hit is not None:
            cat, conf = hit
            scores[rank] = score_from_parse(cat, conf, criterion)
            cached_mask[rank] = True

    remaining = [i for i in range(len(shortlist)) if not cached_mask[i]]
    n_cached  = cached_mask.sum()
    if verbose and n_cached:
        print(f"  Cache hits: {n_cached}/{len(shortlist)}")

    # ----------------------------------------------------------------
    # Process remaining images in chunks
    # ----------------------------------------------------------------
    chunks = [remaining[i:i + max_images_per_call]
              for i in range(0, len(remaining), max_images_per_call)]

    for chunk_idx, chunk in enumerate(chunks):
        chunk_pils = [candidate_pils[i] for i in chunk if candidate_pils[i] is not None]
        valid_chunk = [i for i in chunk if candidate_pils[i] is not None]
        n = len(chunk_pils)
        if n == 0:
            continue

        if verbose:
            print(f"  Chunk {chunk_idx + 1}/{len(chunks)}: {n} candidates "
                  f"(indices {valid_chunk[0]}–{valid_chunk[-1]})")

        if family in ("qwen2vl", "auto"):
            all_images = query_images + chunk_pils
            prompt = _build_collective_prompt_qwen(n, has_bbox)
            response = _infer_qwen2vl(
                model, processor, all_images, prompt, device,
                max_new_tokens=20 * n)
        elif family == "gemma3":
            all_images = query_images + chunk_pils
            prompt = _build_collective_prompt_qwen(n, has_bbox)
            response = _infer_gemma3(
                model, processor, all_images, prompt, device,
                max_new_tokens=20 * n)
        elif family == "deepseek":
            all_images = query_images + chunk_pils
            prompt = _build_collective_prompt_qwen(n, has_bbox)
            response = _infer_deepseek(
                model, processor, all_images, prompt, device,
                max_new_tokens=20 * n)
        elif family == "paligemma":
            # PaliGemma is single-image only — process candidates one by one
            for local_i, global_i in enumerate(valid_chunk):
                left = query_crop if has_bbox else query_pil
                composite = make_side_by_side(left, chunk_pils[local_i])
                r = _infer_paligemma(model, processor, composite,
                                     _PALIGEMMA_PREFIX, device)
                if verbose:
                    print(f"    [{global_i:3d}] Response: {r!r}")
                cat, conf = parse_vlm_response(r)
                score = score_from_parse(cat, conf, criterion)
                scores[global_i] = score
                cache.put(model_id, query_hash, fnames[global_i], bbox, cat, conf)
            continue

        if verbose:
            print(f"    Response: {response!r}")

        parsed = _parse_collective_response(response, n, offset)
        for local_i, global_i in enumerate(valid_chunk):
            cat, conf = parsed[local_i]
            score = score_from_parse(cat, conf, criterion)
            scores[global_i] = score
            cache.put(model_id, query_hash, fnames[global_i], bbox, cat, conf)

    cache.flush()

    sorted_idxs = np.argsort(-scores)
    return shortlist[sorted_idxs], scores[sorted_idxs]


# ---------------------------------------------------------------------------
# Self-test  (python -m simple_retrieval.llms  or  python simple_retrieval/llms.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import glob
    import sys

    parser = argparse.ArgumentParser(
        description="Quick VLM resorting smoke-test on Oxford5k images."
    )
    parser.add_argument("--oxford_dir", default=os.path.expanduser("~/datasets/Oxford5k"),
                        help="Directory containing Oxford5k .jpg images")
    parser.add_argument("--query",      default=None,
                        help="Query image path (default: first all_souls image found)")
    parser.add_argument("--model",      default="qwen2vl-2b",
                        help="VLM model shortcut or full HF path. "
                             f"Shortcuts: {list(MODEL_SHORTCUTS.keys())}")
    parser.add_argument("--criterion",  default="vlm_zoom_in",
                        choices=["vlm_zoom_in", "vlm_zoom_out", "vlm_relevant"])
    parser.add_argument("--n",          type=int, default=50,
                        help="Number of shortlist images to score (default 50)")
    parser.add_argument("--bbox",       nargs=4, type=int, default=None,
                        metavar=("X1","Y1","X2","Y2"),
                        help="Optional query ROI bbox in pixel coords")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--per_image",  action="store_true",
                        help="Use one VLM call per candidate (default: collective/chunked)")
    parser.add_argument("--chunk",      type=int, default=8,
                        help="Max candidates per collective VLM call (default 8)")
    parser.add_argument("--cache_dir",  default="./tmp/vlm_cache",
                        help="Directory for the response cache")
    args = parser.parse_args()

    # ---- Collect candidate images from Oxford5k ----
    oxford_dir = os.path.expanduser(args.oxford_dir)
    all_imgs = sorted(glob.glob(os.path.join(oxford_dir, "*.jpg")))
    if not all_imgs:
        sys.exit(f"No .jpg images found in {oxford_dir}")

    # Pick a query
    if args.query:
        query_path = os.path.expanduser(args.query)
    else:
        # Default: first all_souls image
        candidates_q = [p for p in all_imgs if "all_souls" in os.path.basename(p)]
        query_path = candidates_q[0] if candidates_q else all_imgs[0]

    print(f"Query : {query_path}")
    query_img = Image.open(query_path).convert("RGB")

    # Build shortlist: pick args.n images, excluding the query itself
    shortlist_fnames = [p for p in all_imgs if p != query_path][: args.n]
    # shortlist indices are just 0..n-1 into our local list
    shortlist = np.arange(len(shortlist_fnames))
    db_fnames = shortlist_fnames          # db_fnames[shortlist[i]] == shortlist_fnames[i]

    print(f"Shortlist size : {len(shortlist)}")
    print(f"Model          : {args.model}  ({resolve_model_name(args.model)})")
    print(f"Criterion      : {args.criterion}")
    print(f"Mode           : {'per-image' if args.per_image else f'collective (chunk={args.chunk})'}")
    print(f"BBox           : {args.bbox}")
    print(f"Device         : {args.device}")
    print()

    if args.per_image:
        sorted_sl, scores = resort_shortlist_vlm(
            query_img   = query_img,
            shortlist   = shortlist,
            db_fnames   = db_fnames,
            model_name  = args.model,
            criterion   = args.criterion,
            bbox        = args.bbox,
            device      = args.device,
        )
    else:
        sorted_sl, scores = resort_shortlist_vlm_collective(
            query_img           = query_img,
            shortlist           = shortlist,
            db_fnames           = db_fnames,
            model_name          = args.model,
            criterion           = args.criterion,
            bbox                = args.bbox,
            device              = args.device,
            max_images_per_call = args.chunk,
            cache_dir           = args.cache_dir,
        )

    print("\n=== Results (descending score) ===")
    for rank, (idx, score) in enumerate(zip(sorted_sl, scores)):
        tag = " ← top" if rank == 0 else ""
        print(f"  {rank:3d} | score={score:.3f} | {os.path.basename(db_fnames[idx])}{tag}")
