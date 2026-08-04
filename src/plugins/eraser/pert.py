"""PERT scene text removal erasure plugin (Solution A optional)."""
import logging
import numpy as np
from typing import List
from interfaces.base import StageType
from interfaces.eraser import IEraserPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion
from ._common import build_mask

logger = logging.getLogger(__name__)


@register_plugin(StageType.ERASER, "pert")
class PERTEraserPlugin(IEraserPlugin):
    """PERT scene text removal, migrated from TextEraser._erase_pert."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None

    def _init_pert(self):
        import os, sys, torch
        repo = os.environ.get("PERT_REPO", "")
        if not repo:
            raise RuntimeError("PERT_REPO not set. Clone https://github.com/wangyuxin87/PERT and set PERT_REPO.")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from networks_sfnet import Pert  # type: ignore
        self._model = Pert(use_GPU=(self.config.device == "cuda"))
        ckpt = os.environ.get("PERT_CKPT", "")
        if not ckpt:
            raise RuntimeError("PERT_CKPT not set. Download pretrained weights and set PERT_CKPT.")
        state_dict = torch.load(ckpt, map_location="cpu")
        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval()
        if self.config.device == "cuda":
            self._model = self._model.cuda()
        logger.info("PERT initialized from %s", ckpt)

    def erase(self, image: np.ndarray, regions: List[TextRegion], dilate_pixels: int = 0) -> np.ndarray:
        if not regions:
            return image
        if self._model is None:
            self._init_pert()
        mask = build_mask(
            image.shape[:2], regions,
            dilate_pixels or self.config.erasure_dilate_pixels,
            image=image,
        )
        import torch
        h, w = image.shape[:2]
        dev = next(self._model.parameters()).device
        h_pad = h + (32 - h % 32) % 32
        w_pad = w + (32 - w % 32) % 32
        img_rgb = image[..., ::-1].copy()
        if h_pad != h or w_pad != w:
            import cv2
            img_rgb = cv2.resize(img_rgb, (w_pad, h_pad))
        img_t = torch.from_numpy(img_rgb.astype("float32") / 255.0)
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            out, _, _, _, _ = self._model(img_t)
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
