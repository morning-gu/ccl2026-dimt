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


def _band_pixels(image: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                 width: int = 4, inside: bool = True) -> np.ndarray:
    """Return pixels in a ``width``-px band along the bbox edge.

    ``inside=True``  -> the outer ``width`` pixels of the rect (LaMA fill).
    ``inside=False`` -> a band just outside the rect (original background).
    """
    h, w = image.shape[:2]
    if inside:
        ox1, oy1, ox2, oy2 = x1, y1, x2, y2
        ix1, iy1 = x1 + width, y1 + width
        ix2, iy2 = max(ix1, x2 - width), max(iy1, y2 - width)
    else:
        ox1, oy1, ox2, oy2 = max(0, x1 - width), max(0, y1 - width), min(w, x2 + width), min(h, y2 + width)
        ix1, iy1, ix2, iy2 = x1, y1, x2, y2
    band = np.zeros((h, w), dtype=bool)
    band[oy1:oy2, ox1:ox2] = True
    band[iy1:iy2, ix1:ix2] = False
    return image[band]


@register_plugin(StageType.ERASER, "lama")
class LaMaEraserPlugin(IEraserPlugin):
    """LaMA inpainting, migrated from TextEraser._erase_lama.

    LaMA is fed a full bounding-box mask for stable generation (fragmented
    pixel masks make it hallucinate noise). Its result is then composited back
    only onto the actual text pixels (pixel-level mask), preserving the
    original background inside the bbox so no rectangular color patch remains.
    A per-region color correction aligns LaMA's fill with the surrounding
    background and the composite mask is feathered for a seamless transition.
    """

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

    @staticmethod
    def _color_correct(orig: np.ndarray, fill: np.ndarray,
                       regions: List[TextRegion], dilate: int) -> np.ndarray:
        """Shift LaMA fill per region so its boundary matches the background.

        Uses the median (robust to neighbouring text/graphics) of a band just
        outside the bbox (real background) vs. just inside (LaMA fill), and
        removes that offset from the whole bbox region.
        """
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
            inside = _band_pixels(fill, x1, y1, x2, y2, width=4, inside=True)
            if len(outside) < 4 or len(inside) < 4:
                continue
            delta = np.median(outside, axis=0) - np.median(inside, axis=0)
            delta = np.clip(delta, -30, 30)
            out[y1:y2, x1:x2] += delta.astype(np.int16)
        return out

    def erase(self, image: np.ndarray, regions: List[TextRegion], dilate_pixels: int = 0) -> np.ndarray:
        if not regions:
            return image
        if self._model is None:
            self._init_lama()
        import cv2
        from PIL import Image

        dilate = dilate_pixels or self.config.erasure_dilate_pixels

        # 1. Full bbox mask -> LaMA (stable generation, no hallucinated noise).
        full_mask = build_mask(image.shape[:2], regions, dilate, image=None)
        lama_fill = np.array(self._model(Image.fromarray(image), Image.fromarray(full_mask)))
        if lama_fill.shape[:2] != image.shape[:2]:
            lama_fill = cv2.resize(lama_fill, (image.shape[1], image.shape[0]))
        lama_fill = lama_fill.astype(np.int16)
        orig = image.astype(np.int16)

        # 2. Per-region color correction: align LaMA boundary with real background.
        lama_fill = self._color_correct(orig, lama_fill, regions, dilate)

        # 3. Pixel-level text mask -> composite only where text was, preserving
        #    the original background elsewhere (kills the rectangular patch).
        # keep_thin_lines=True: thin/small strokes (e.g. a horizontal stroke that
        # Otsu splits off as its own component) are real text and must be replaced
        # too; LaMA's fill is already clean so there is no smudging risk.
        text_mask = build_mask(image.shape[:2], regions, dilate, image=image,
                               keep_thin_lines=True)

        # 4. Feather the composite mask and blend.
        k = 2 * dilate + 1
        feather = cv2.GaussianBlur(text_mask, (k, k), 0)
        weight = (feather.astype(np.float32) / 255.0)[..., None]
        result = weight * lama_fill.astype(np.float32) + (1.0 - weight) * orig.astype(np.float32)
        return np.clip(result, 0, 255).astype(np.uint8)
