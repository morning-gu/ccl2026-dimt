"""LaMA inpainting erasure plugin (Solution B)."""
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


@register_plugin(StageType.ERASER, "lama")
class LaMaEraserPlugin(IEraserPlugin):
    """LaMA inpainting, migrated from TextEraser._erase_lama."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None

    def _init_lama(self):
        import torch
        from simple_lama_inpainting import SimpleLama

        # TorchScript models saved on CUDA need map_location='cpu'
        # when loading on a machine without CUDA (e.g. macOS MPS).
        _orig_load = torch.jit.load
        def _patched_load(*args, **kwargs):
            if "map_location" not in kwargs:
                kwargs["map_location"] = "cpu"
            return _orig_load(*args, **kwargs)
        torch.jit.load = _patched_load
        try:
            self._model = SimpleLama()
        finally:
            torch.jit.load = _orig_load
        logger.info("LaMA erasure model initialized")

    def erase(self, image: np.ndarray, regions: List[TextRegion], dilate_pixels: int = 0) -> np.ndarray:
        if not regions:
            return image
        if self._model is None:
            self._init_lama()
        mask = build_mask(
            image.shape[:2], regions,
            dilate_pixels or self.config.erasure_dilate_pixels,
            image=image,
        )
        from PIL import Image
        result = self._model(Image.fromarray(image), Image.fromarray(mask))
        return np.array(result)
