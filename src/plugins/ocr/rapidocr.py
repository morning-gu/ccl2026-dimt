"""RapidOCR (PP-OCRv4 ONNX) OCR detection plugin."""
import logging
import math
from typing import List, Tuple
import numpy as np

from interfaces.base import StageType
from interfaces.ocr import IOCRPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion

logger = logging.getLogger(__name__)


@register_plugin(StageType.OCR, "rapidocr")
class RapidOCRPlugin(IOCRPlugin):
    """RapidOCR detection plugin, migrated from common/ocr_detector.py."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._ocr = None
        self._symbol_chars = {"十", "✚", "✛", "✜", "✝", "†", "‡", "+", "＋", "✠", "✡"}

    def _init_rapidocr(self):
        from rapidocr_onnxruntime import RapidOCR
        self._ocr = RapidOCR()
        logger.info("RapidOCR initialized")

    def _ensure_initialized(self):
        if self._ocr is None:
            self._init_rapidocr()

    def detect(self, image: np.ndarray) -> List[TextRegion]:
        self._ensure_initialized()
        return self._detect_rapidocr(image)

    def detect_from_path(self, image_path: str) -> Tuple[List[TextRegion], np.ndarray]:
        try:
            import cv2
            image = cv2.imread(image_path)
        except ImportError:
            from PIL import Image
            img_pil = Image.open(image_path)
            image = np.array(img_pil)[:, :, ::-1]
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        regions = self.detect(image)
        return regions, image

    def _detect_rapidocr(self, image: np.ndarray) -> List[TextRegion]:
        result, elapse = self._ocr(image)
        regions = []
        if result:
            for bbox_poly, text, conf in result:
                xs = [p[0] for p in bbox_poly]
                ys = [p[1] for p in bbox_poly]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                dx = bbox_poly[1][0] - bbox_poly[0][0]
                dy = bbox_poly[1][1] - bbox_poly[0][1]
                angle = math.degrees(math.atan2(dy, dx))
                regions.append(TextRegion(
                    text=text,
                    bbox=bbox,
                    confidence=float(conf),
                    bbox_poly=[[float(p[0]), float(p[1])] for p in bbox_poly],
                    angle=angle,
                ))
        regions = [r for r in regions if not self._is_likely_symbol(r)]
        logger.debug("RapidOCR detected %d regions", len(regions))
        return regions

    def _is_likely_symbol(self, region: TextRegion) -> bool:
        text = region.text.strip()
        if len(text) != 1:
            return False
        if text in self._symbol_chars:
            logger.debug("Filtering symbol region: %r", text)
            return True
        return False
