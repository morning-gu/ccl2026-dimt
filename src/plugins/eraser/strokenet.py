"""STRNet stroke-level erasure plugin."""
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


@register_plugin(StageType.ERASER, "strokenet")
class StrokenetEraserPlugin(IEraserPlugin):
    """STRNet stroke-level erasure, migrated from TextEraser._erase_strokenet."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None

    def _init_strokenet(self):
        import os, sys, torch
        repo = os.environ.get("STROKENET_REPO", "SceneTextRemover-pytorch")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from network import STRNet  # type: ignore
        self._model = STRNet()
        ckpt = os.environ.get("STROKENET_CKPT", "")
        if not ckpt:
            raise RuntimeError("STROKENET_CKPT not set.")
        self._model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self._model.eval()
        if self.config.device == "cuda":
            self._model = self._model.cuda()
        logger.info("STRNet initialized from %s", ckpt)

    def erase(self, image: np.ndarray, regions: List[TextRegion], dilate_pixels: int = 0) -> np.ndarray:
        if not regions:
            return image
        if self._model is None:
            self._init_strokenet()
        mask = build_mask(
            image.shape[:2], regions,
            dilate_pixels or self.config.erasure_dilate_pixels,
            image=image,
        )
        import torch
        h, w = image.shape[:2]
        dev = next(self._model.parameters()).device
        img_t = torch.from_numpy(image[..., ::-1].copy()).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
        m_t = torch.from_numpy((mask > 0).astype("float32")).float().unsqueeze(0).unsqueeze(0).to(dev)
        with torch.no_grad():
            _, _, _, ite_ = self._model(img_t, m_t)
        out = ite_[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        out = (out * 255).astype("uint8")[..., ::-1]
        keep = mask == 0
        if image.ndim == out.ndim and image.shape[:2] == out.shape[:2]:
            out[keep] = image[keep]
        return out
