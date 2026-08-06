"""PERT erasure API server.

Wraps wangyuxin87/PERT as a standalone FastAPI service.

Prerequisites:
    - PERT repo cloned: https://github.com/wangyuxin87/PERT
    - pip install fastapi uvicorn pydantic numpy opencv-python torch
    - Set PERT_REPO and PERT_CKPT env vars.

Usage:
  python pert_server.py --host 0.0.0.0 --port 8012
"""
import argparse
import logging
import os
import sys
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

logger = logging.getLogger("pert_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

app = FastAPI(title="PERT Eraser API")

_model = None
_device = None


def _init_model():
    global _model, _device

    repo = os.environ.get("PERT_REPO", "")
    if not repo:
        raise RuntimeError("PERT_REPO not set. Clone https://github.com/wangyuxin87/PERT and set PERT_REPO.")
    if repo not in sys.path:
        sys.path.insert(0, repo)

    import torch
    from networks_sfnet import Pert  # type: ignore

    _device = "cuda" if os.environ.get("DEVICE", "cuda") == "cuda" else "cpu"
    _model = Pert(use_GPU=(_device == "cuda"))
    ckpt = os.environ.get("PERT_CKPT", "")
    if not ckpt:
        raise RuntimeError("PERT_CKPT not set. Download pretrained weights and set PERT_CKPT.")
    state_dict = torch.load(ckpt, map_location="cpu")
    _model.load_state_dict(state_dict, strict=True)
    _model.eval()
    if _device == "cuda":
        _model = _model.cuda()
    logger.info("PERT initialized from %s", ckpt)


def _erase_pert(image, regions: List[RegionData], dilate_pixels: int) -> np.ndarray:
    import torch

    dilate = dilate_pixels or 5
    mask = build_mask(image.shape[:2], regions, dilate, image=image)

    h, w = image.shape[:2]
    dev = next(_model.parameters()).device
    h_pad = h + (32 - h % 32) % 32
    w_pad = w + (32 - w % 32) % 32
    img_rgb = image[..., ::-1].copy()
    if h_pad != h or w_pad != w:
        import cv2
        img_rgb = cv2.resize(img_rgb, (w_pad, h_pad))
    img_t = torch.from_numpy(img_rgb.astype("float32") / 255.0)
    img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(dev)
    with torch.no_grad():
        out, _, _, _, _ = _model(img_t)
    out = out[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    out = (out * 255).astype("uint8")
    if h_pad != h or w_pad != w:
        import cv2
        out = cv2.resize(out, (w, h))
    out = out[..., ::-1]
    keep = mask == 0
    if image.ndim == out.ndim and image.shape[:2] == out.shape[:2]:
        out[keep] = image[keep]
    return out


@app.post("/erase", response_model=RenderResponse)
def erase(req: EraseRequest):
    if _model is None:
        _init_model()

    image, regions, dilate_pixels = decode_erase_request(req)
    if not regions:
        return RenderResponse(image=encode_image(image))

    result = _erase_pert(image, regions, dilate_pixels)
    return RenderResponse(image=encode_image(result))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="PERT Eraser API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("pert_server:app", host=args.host, port=args.port, reload=args.reload)
