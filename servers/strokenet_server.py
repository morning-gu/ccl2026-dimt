"""STRNet erasure API server.

Wraps SceneTextRemover-pytorch (STRNet) as a standalone FastAPI service.

Prerequisites:
    - SceneTextRemover-pytorch repo cloned
    - pip install fastapi uvicorn pydantic numpy opencv-python torch
    - Set STROKENET_REPO and STROKENET_CKPT env vars.

Usage:
  python strokenet_server.py --host 0.0.0.0 --port 8011
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

logger = logging.getLogger("strokenet_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

app = FastAPI(title="STRNet Eraser API")

_model = None
_device = None


def _init_model():
    global _model, _device

    repo = os.environ.get("STROKENET_REPO", "SceneTextRemover-pytorch")
    if repo not in sys.path:
        sys.path.insert(0, repo)

    import torch
    from network import STRNet  # type: ignore

    _model = STRNet()
    ckpt = os.environ.get("STROKENET_CKPT", "")
    if not ckpt:
        raise RuntimeError("STROKENET_CKPT not set.")
    _model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    _model.eval()
    _device = "cuda" if os.environ.get("DEVICE", "cuda") == "cuda" else "cpu"
    if _device == "cuda":
        _model = _model.cuda()
    logger.info("STRNet initialized from %s", ckpt)


def _erase_strokenet(image, regions: List[RegionData], dilate_pixels: int) -> np.ndarray:
    import torch

    dilate = dilate_pixels or 5
    mask = build_mask(image.shape[:2], regions, dilate, image=image)

    h, w = image.shape[:2]
    dev = next(_model.parameters()).device
    img_t = torch.from_numpy(image[..., ::-1].copy()).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
    m_t = torch.from_numpy((mask > 0).astype("float32")).float().unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        _, _, _, ite_ = _model(img_t, m_t)
    out = ite_[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    out = (out * 255).astype("uint8")[..., ::-1]
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

    result = _erase_strokenet(image, regions, dilate_pixels)
    return RenderResponse(image=encode_image(result))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="STRNet Eraser API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("strokenet_server:app", host=args.host, port=args.port, reload=args.reload)
