"""SSER (Self-supervised Text Erasing) API server.

Wraps alimama-creative/Self-supervised-Text-Erasing (STRnet2) as a standalone
FastAPI service.

Prerequisites:
    - Repo cloned: https://github.com/alimama-creative/Self-supervised-Text-Erasing
    - Checkpoints: https://huggingface.co/alimama-creative/Self-Supervised-Text-Erasing
    - pip install fastapi uvicorn pydantic numpy opencv-python torch
    - Set SSER_REPO and SSER_CKPT env vars.

Usage:
  python sser_server.py --host 0.0.0.0 --port 8013
"""
import argparse
import logging
import os
import sys
from typing import List

from contextlib import asynccontextmanager

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

logger = logging.getLogger("sser_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

_model = None
_device = None


def _init_model():
    global _model, _device

    repo = os.environ.get("SSER_REPO", "")
    if not repo:
        raise RuntimeError(
            "SSER_REPO not set. Clone "
            "https://github.com/alimama-creative/Self-supervised-Text-Erasing "
            "and set SSER_REPO to the repo root."
        )
    if repo not in sys.path:
        sys.path.insert(0, repo)

    import torch
    from models.src.erase.sa_gan import STRnet2  # type: ignore

    _model = STRnet2(3)
    ckpt = os.environ.get("SSER_CKPT", "")
    if not ckpt:
        raise RuntimeError(
            "SSER_CKPT not set. Download best_net_G.pth from "
            "https://huggingface.co/alimama-creative/Self-Supervised-Text-Erasing "
            "and set SSER_CKPT to the file path."
        )

    state_dict = torch.load(ckpt, map_location="cpu")
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    missing, unexpected = _model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        logger.warning(
            "SSER state_dict mismatch -- missing %d keys, unexpected %d keys",
            len(missing), len(unexpected),
        )
    _model.eval()
    want_cuda = os.environ.get("DEVICE", "cuda") == "cuda"
    _device = "cuda" if (want_cuda and torch.cuda.is_available()) else "cpu"
    _model = _model.to(_device)
    logger.info("SSER (STRnet2) initialized from %s", ckpt)


def _erase_sser(image, regions: List[RegionData], dilate_pixels: int) -> np.ndarray:
    import torch
    import cv2

    dilate = dilate_pixels or 5
    mask = build_mask(image.shape[:2], regions, dilate, image=image)

    h, w = image.shape[:2]
    h_pad = h + (32 - h % 32) % 32
    w_pad = w + (32 - w % 32) % 32
    img_rgb = image[..., ::-1].copy()
    if h_pad != h or w_pad != w:
        img_rgb = cv2.resize(img_rgb, (w_pad, h_pad))

    img_t = torch.from_numpy(img_rgb.astype("float32") / 255.0)
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)
    dev = next(_model.parameters()).device
    img_t = img_t.to(dev)

    with torch.no_grad():
        outputs = _model(img_t)
        fake_b = outputs[4]

    out = fake_b[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = (out * 255).astype(np.uint8)
    if h_pad != h or w_pad != w:
        out = cv2.resize(out, (w, h))
    out = out[..., ::-1]

    keep = mask == 0
    if image.ndim == out.ndim and image.shape[:2] == out.shape[:2]:
        out[keep] = image[keep]
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_model()
    yield


app = FastAPI(title="SSER Eraser API", lifespan=lifespan)


@app.post("/erase", response_model=RenderResponse)
def erase(req: EraseRequest):
    image, regions, dilate_pixels = decode_erase_request(req)
    if not regions:
        return RenderResponse(image=encode_image(image))

    result = _erase_sser(image, regions, dilate_pixels)
    return RenderResponse(image=encode_image(result))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="SSER Eraser API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("sser_server:app", host=args.host, port=args.port, reload=args.reload)
