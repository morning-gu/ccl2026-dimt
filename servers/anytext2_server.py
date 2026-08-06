"""AnyText2 rendering API server.

Wraps tyxsspa/AnyText2 as a standalone FastAPI service.

Prerequisites:
    - AnyText2 repo cloned: https://github.com/tyxsspa/AnyText2
    - pip install -r AnyText2/requirements.txt  (controlnet_aux, einops, timm, torch, ...)
    - pip install fastapi uvicorn pydantic numpy opencv-python Pillow
    - Set ANYTEXT2_MODEL_PATH env var to the repo root.

Usage:
  python anytext2_server.py --host 0.0.0.0 --port 8003
"""
import argparse
import logging
import os
import sys
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException

from _shared import (
    RenderRequest,
    RenderResponse,
    RegionData,
    decode_request,
    encode_image,
)

logger = logging.getLogger("anytext2_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

app = FastAPI(title="AnyText2 Renderer API")

# ------------------------------------------------------------------
# Global model state (lazy-loaded on first request)
# ------------------------------------------------------------------

_anytext2 = None
_anytext2_repo = None
_device = None


def _init_model():
    """Load AnyText2 model."""
    global _anytext2, _anytext2_repo, _device

    repo = os.environ.get("ANYTEXT2_MODEL_PATH", "AnyText2")
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from ms_wrapper import AnyText2Model  # type: ignore

    device = os.environ.get("ANYTEXT2_DEVICE", "cuda")
    infer_params = {
        "use_fp16": device == "cuda",
        "use_translator": False,
        "font_path": os.path.join(repo, "font", "Arial_Unicode.ttf"),
    }
    ckpt = os.environ.get("ANYTEXT2_CKPT", "")
    if ckpt:
        infer_params["model_path"] = ckpt
    _anytext2 = AnyText2Model(model_dir=os.path.join(repo, "models"), **infer_params)
    if device == "cuda":
        _anytext2 = _anytext2.cuda(0)
    _anytext2_repo = repo
    _device = device
    logger.info("AnyText2 model initialized from %s", repo)


# ------------------------------------------------------------------
# Core rendering
# ------------------------------------------------------------------

def _render_anytext2(image, regions, style_reference):
    import cv2

    h, w = image.shape[:2]
    texts = [r.translated_text for r in regions]
    text_prompt = "#".join(texts)
    pos = np.zeros((h, w), dtype=np.uint8)
    for r in regions:
        x1, y1, x2, y2 = [int(v) for v in r.bbox[:4]]
        pos[y1:y2, x1:x2] = 255
    color_parts = []
    for r in regions:
        c = r.style_info.get("color")
        if isinstance(c, (tuple, list)) and len(c) >= 3:
            color_parts.append(",".join(str(int(v)) for v in c[:3]))
        else:
            color_parts.append("500,500,500")
    text_colors = " ".join(color_parts)
    params = {
        "mode": "edit",
        "sort_priority": "\u2193\u2193\u2192",
        "show_debug": False,
        "revise_pos": False,
        "image_count": 1,
        "ddim_steps": 20,
        "image_width": w,
        "image_height": h,
        "strength": 1.0,
        "attnx_scale": 1.0,
        "font_hollow": None,
        "cfg_scale": 9.0,
        "eta": 0.0,
        "a_prompt": "best quality, extremely detailed,4k, HD, supper legible text, clear text edges, clear strokes, neat writing, no watermarks",
        "n_prompt": "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture",
        "base_model_path": "",
        "lora_path_ratio": "",
        "glyline_font_path": ["None"] * len(regions),
        "font_hint_image": [None] * len(regions),
        "font_hint_mask": [None] * len(regions),
        "text_colors": text_colors,
    }
    input_data = {
        "img_prompt": image[..., ::-1].copy(),
        "text_prompt": text_prompt,
        "seed": 1,
        "draw_pos": pos,
        "ori_image": image[..., ::-1].copy(),
    }
    results, rtn_code, rtn_warning, debug_info = _anytext2(input_data, **params)
    if rtn_code < 0 or not results:
        raise RuntimeError(f"AnyText2 rendering failed (rtn_code={rtn_code}): {rtn_warning}")
    return np.array(results[0])[..., ::-1]


# ------------------------------------------------------------------
# FastAPI endpoint
# ------------------------------------------------------------------

@app.post("/render", response_model=RenderResponse)
def render(req: RenderRequest):
    if _anytext2 is None:
        _init_model()

    image, regions, style_ref = decode_request(req)
    if not regions:
        return RenderResponse(image=encode_image(image))

    result = _render_anytext2(image, regions, style_ref)
    return RenderResponse(image=encode_image(result))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _anytext2 is not None}


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="AnyText2 Renderer API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "anytext2_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
