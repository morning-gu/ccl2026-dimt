"""OpenCV inpaint erasure plugin (Solution C)."""
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


@register_plugin(StageType.ERASER, "opencv")
class OpenCVEraserPlugin(IEraserPlugin):
    """OpenCV Telea inpainting, migrated from TextEraser._erase_opencv."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def erase(self, image: np.ndarray, regions: List[TextRegion], dilate_pixels: int = 0) -> np.ndarray:
        if not regions:
            return image
        import cv2
        mask = build_mask(
            image.shape[:2], regions,
            dilate_pixels or self.config.erasure_dilate_pixels,
            image=image,
        )
        result = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        smoothed = cv2.medianBlur(result, 3)
        mask_bool = mask > 0
        result[mask_bool] = smoothed[mask_bool]
        return result
