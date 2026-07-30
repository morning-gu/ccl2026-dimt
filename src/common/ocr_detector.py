"""OCR detection module using RapidOCR as primary backend.

RapidOCR is the ONNX runtime port of PP-OCRv4 (the model family used in the
AnyTrans / AnyText2 papers). No fallback backends: if RapidOCR is not
installed, import-time or first-use raises so the missing dependency is
visible instead of silently producing empty results.
"""
import logging
from typing import List, Optional, Tuple
import numpy as np

from .config import PipelineConfig
from .selective_translator import TextRegion

logger = logging.getLogger(__name__)


class OCRDetector:
    """Detect text regions using RapidOCR (PP-OCRv4 ONNX).

    Single backend, no degradation. Requires `rapidocr-onnxruntime`.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._ocr = None
        self._backend = "rapidocr"

    def _init_rapidocr(self):
        """Initialize RapidOCR. Raises if the package is not installed."""
        from rapidocr_onnxruntime import RapidOCR
        self._ocr = RapidOCR()
        logger.info("RapidOCR initialized")

    def _ensure_initialized(self):
        if self._ocr is None and self._backend == "rapidocr":
            self._init_rapidocr()

    def detect(self, image: np.ndarray) -> List[TextRegion]:
        """Detect text regions. No fallback backends."""
        self._ensure_initialized()
        return self._detect_rapidocr(image)

    def detect_from_path(self, image_path: str) -> Tuple[List[TextRegion], np.ndarray]:
        """Detect text regions from an image file path.

        Returns:
            Tuple of (regions, image_array).
        """
        try:
            import cv2
            image = cv2.imread(image_path)
        except ImportError:
            from PIL import Image
            img_pil = Image.open(image_path)
            image = np.array(img_pil)[:, :, ::-1]  # RGB to BGR
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        regions = self.detect(image)
        return regions, image
    def _detect_rapidocr(self, image: np.ndarray) -> List[TextRegion]:
        """Run RapidOCR detection."""
        result, elapse = self._ocr(image)
        regions = []
        if result:
            for bbox_poly, text, conf in result:
                xs = [p[0] for p in bbox_poly]
                ys = [p[1] for p in bbox_poly]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                regions.append(TextRegion(
                    text=text,
                    bbox=bbox,
                    confidence=float(conf),
                ))
        logger.debug("RapidOCR detected %d regions", len(regions))
        return regions
