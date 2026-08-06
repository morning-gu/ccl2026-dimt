"""PowerPaint erasure API server.

Wraps open-mmlab/PowerPaint (prompt-augmented diffusion inpainting) as a
standalone FastAPI service.  PowerPaint is fine-tuned Stable Diffusion with
task-specific prompts; for text erasure we use the "Remove the masked area"
prompt which tells the model to generate seamless background instead of
hallucinating new content.

Prerequisites:
    - pip install fastapi uvicorn pydantic numpy opencv-python Pillow torch diffusers transformers accelerate
    - PowerPaint weights auto-download from HuggingFace (set POWERPAINT_MODEL_ID to override).

Environment variables:
    POWERPAINT_MODEL_ID         HuggingFace model ID for PowerPaint weights.
                                Default: "SanityZero/PowerPaint-v2"
    POWERPAINT_BASE_MODEL       Base SD model (only used if BrushNet pipeline
                                is needed). Default: "runwayml/stable-diffusion-inpainting"
    POWERPAINT_BRUSHNET         BrushNet model ID. Default: "SanityZero/PowerPaint-BrushNet"
    POWERPAINT_PROMPT           Task prompt for erasure.
                                Default: "Remove the masked area"
    POWERPAINT_NEGATIVE_PROMPT  Negative prompt. Default: ""
    POWERPAINT_STEPS            Number of inference steps. Default: 25
    POWERPAINT_GUIDANCE         Guidance scale. Default: 7.5
    DEVICE                      "cuda" or "cpu". Default: "cuda"

Usage:
  python powerpaint_server.py --host 0.0.0.0 --port 8014
"""
import argparse
import logging
import os
from typing import List

import numpy as np
from fastapi import FastAPI

from _shared import (
    EraseRequest,
    RenderResponse,
    RegionData,
    decode_erase_request,
    encode_image,
    build_mask,
)

logger = logging.getLogger("powerpaint_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

app = FastAPI(title="PowerPaint Eraser API")

_pipeline = None
_device = None
_config = {}


def _init_model():
    global _pipeline, _device, _config

    import torch

    _device = "cuda" if os.environ.get("DEVICE", "cuda") == "cuda" else "cpu"
    dtype = torch.float16 if _device == "cuda" else torch.float32

    model_id = os.environ.get("POWERPAINT_MODEL_ID", "SanityZero/PowerPaint-v2")
    base_model = os.environ.get(
        "POWERPAINT_BASE_MODEL", "runwayml/stable-diffusion-inpainting"
    )

    _config["prompt"] = os.environ.get(
        "POWERPAINT_PROMPT", "Remove the masked area"
    )
    _config["negative_prompt"] = os.environ.get(
        "POWERPAINT_NEGATIVE_PROMPT", ""
    )
    _config["steps"] = int(os.environ.get("POWERPAINT_STEPS", "25"))
    _config["guidance"] = float(os.environ.get("POWERPAINT_GUIDANCE", "7.5"))

    # Try PowerPaint-specific pipeline first, fall back to BrushNet, then SDInpaint.
    try:
        from diffusers import PowerPaintPipeline  # type: ignore

        logger.info("Loading PowerPaintPipeline from %s", model_id)
        _pipeline = PowerPaintPipeline.from_pretrained(
            model_id, torch_dtype=dtype
        )
    except (ImportError, Exception) as e:
        logger.info("PowerPaintPipeline unavailable (%s), trying BrushNet pipeline", e)
        try:
            from diffusers import BrushNetPipeline, BrushNet  # type: ignore

            brushnet_id = os.environ.get(
                "POWERPAINT_BRUSHNET", "SanityZero/PowerPaint-BrushNet"
            )
            logger.info(
                "Loading BrushNetPipeline (base=%s, brushnet=%s)",
                base_model, brushnet_id,
            )
            brushnet = BrushNet.from_pretrained(brushnet_id, torch_dtype=dtype)
            _pipeline = BrushNetPipeline.from_pretrained(
                base_model, brushnet=brushnet, torch_dtype=dtype
            )
        except (ImportError, Exception) as e2:
            logger.info(
                "BrushNet unavailable (%s), falling back to SDInpaintPipeline", e2
            )
            from diffusers import StableDiffusionInpaintPipeline

            logger.info("Loading SDInpaintPipeline from %s", model_id)
            _pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_id, torch_dtype=dtype
            )

    if _device == "cuda":
        _pipeline = _pipeline.to("cuda")

    logger.info(
        "PowerPaint initialised (model=%s, prompt=%r, steps=%d, guidance=%.1f)",
        model_id, _config["prompt"], _config["steps"], _config["guidance"],
    )


def _band_pixels(
    image: np.ndarray, x1: int, y1: int, x2: int, y2: int,
    width: int = 4, inside: bool = True,
) -> np.ndarray:
    """Return pixels in a width-px band along the bbox edge (for color correction)."""
    h, w = image.shape[:2]
    if inside:
        ox1, oy1, ox2, oy2 = x1, y1, x2, y2
        ix1, iy1 = x1 + width, y1 + width
        ix2, iy2 = max(ix1, x2 - width), max(iy1, y2 - width)
    else:
        ox1, oy1, ox2, oy2 = (
            max(0, x1 - width), max(0, y1 - width),
            min(w, x2 + width), min(h, y2 + width),
        )
        ix1, iy1, ix2, iy2 = x1, y1, x2, y2
    band = np.zeros((h, w), dtype=bool)
    band[oy1:oy2, ox1:ox2] = True
    band[iy1:iy2, ix1:ix2] = False
    return image[band]


def _color_correct(
    orig: np.ndarray, fill: np.ndarray,
    regions: List[RegionData], dilate: int,
) -> np.ndarray:
    """Shift fill per region so its boundary matches the surrounding background."""
    h, w = orig.shape[:2]
    out = fill.copy()
    for region in regions:
        x1 = max(0, int(region.bbox[0]) - dilate)
        y1 = max(0, int(region.bbox[1]) - dilate)
        x2 = min(w, int(region.bbox[2]) + dilate)
        y2 = min(h, int(region.bbox[3]) + dilate)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        outside = _band_pixels(orig, x1, y1, x2, y2, width=4, inside=False)
        inside_band = _band_pixels(fill, x1, y1, x2, y2, width=4, inside=True)
        if len(outside) < 4 or len(inside_band) < 4:
            continue
        delta = np.median(outside, axis=0) - np.median(inside_band, axis=0)
        delta = np.clip(delta, -30, 30)
        out[y1:y2, x1:x2] += delta.astype(np.int16)
    return out


def _erase_powerpaint(
    image: np.ndarray, regions: List[RegionData], dilate_pixels: int,
) -> np.ndarray:
    import torch
    from PIL import Image
    import cv2

    dilate = dilate_pixels or 5

    # 1. Full bbox mask for stable generation (same strategy as LaMa plugin).
    full_mask = build_mask(image.shape[:2], regions, dilate, image=None)

    h, w = image.shape[:2]
    img_pil = Image.fromarray(image).convert("RGB").resize((512, 512))
    mask_pil = Image.fromarray(full_mask).convert("L").resize((512, 512))

    # 2. Run PowerPaint with the "Remove" prompt.
    with torch.no_grad():
        output = _pipeline(
            prompt=_config["prompt"],
            negative_prompt=_config["negative_prompt"],
            image=img_pil,
            mask_image=mask_pil,
            num_inference_steps=_config["steps"],
            guidance_scale=_config["guidance"],
        ).images[0]

    fill = np.array(output.resize((w, h)))
    if fill.shape[:2] != (h, w):
        fill = cv2.resize(fill, (w, h))
    fill = fill.astype(np.int16)
    orig = image.astype(np.int16)

    # 3. Per-region color correction (same as LaMa plugin).
    fill = _color_correct(orig, fill, regions, dilate)

    # 4. Pixel-level text mask for compositing (preserve original background).
    text_mask = build_mask(
        image.shape[:2], regions, dilate, image=image, keep_thin_lines=True,
    )

    # 5. Feathered blending for seamless transition.
    k = 2 * dilate + 1
    feather = cv2.GaussianBlur(text_mask, (k, k), 0)
    weight = (feather.astype(np.float32) / 255.0)[..., None]
    result = (
        weight * fill.astype(np.float32)
        + (1.0 - weight) * orig.astype(np.float32)
    )
    return np.clip(result, 0, 255).astype(np.uint8)


@app.post("/erase", response_model=RenderResponse)
def erase(req: EraseRequest):
    if _pipeline is None:
        _init_model()

    image, regions, dilate_pixels = decode_erase_request(req)
    if not regions:
        return RenderResponse(image=encode_image(image))

    result = _erase_powerpaint(image, regions, dilate_pixels)
    return RenderResponse(image=encode_image(result))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="PowerPaint Eraser API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "powerpaint_server:app",
        host=args.host, port=args.port, reload=args.reload,
    )
