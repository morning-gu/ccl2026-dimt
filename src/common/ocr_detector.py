"""OCR detection module using RapidOCR as primary backend.

Supports RapidOCR (ONNX-based PaddleOCR) with fallback to PaddleOCR,
EasyOCR, and a stub for testing.
"""
import logging
from typing import List, Optional, Tuple
import numpy as np

from .config import PipelineConfig
from .selective_translator import TextRegion

logger = logging.getLogger(__name__)


class OCRDetector:
    """Detect text regions in images using PaddleOCR."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._ocr = None
        self._backend = "rapidocr"

    def _init_rapidocr(self):
        """Lazy-initialize RapidOCR (ONNX-based, no PaddlePaddle needed)."""
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
            self._backend = "rapidocr"
            logger.info("RapidOCR initialized successfully")
        except ImportError:
            logger.warning("RapidOCR not available, trying PaddleOCR fallback")
            self._init_paddleocr()

    def _init_paddleocr(self):
        """Fallback to PaddleOCR."""
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.config.ocr_lang,
                use_gpu=(self.config.device == "cuda"),
                show_log=False,
            )
            self._backend = "paddleocr"
            logger.info("PaddleOCR initialized successfully")
        except ImportError:
            logger.warning("PaddleOCR not available, trying EasyOCR fallback")
            self._init_easyocr()

    def _init_easyocr(self):
        """Fallback to EasyOCR."""
        try:
            import easyocr
            lang_map = {"ch": ["ch_sim", "en"], "en": ["en"]}
            langs = lang_map.get(self.config.ocr_lang, ["ch_sim", "en"])
            self._ocr = easyocr.Reader(langs, gpu=(self.config.device == "cuda"))
            self._backend = "easyocr"
            logger.info("EasyOCR initialized successfully")
        except ImportError:
            logger.warning("EasyOCR not available either, using stub OCR")
            self._ocr = None
            self._backend = "stub"

    def _ensure_initialized(self):
        if self._ocr is None and self._backend == "rapidocr":
            self._init_rapidocr()

    def detect(self, image: np.ndarray) -> List[TextRegion]:
        """Detect text regions in an image.

        Args:
            image: numpy array (H, W, 3) in BGR or RGB format.

        Returns:
            List of TextRegion objects.
        """
        self._ensure_initialized()

        if self._backend == "rapidocr":
            return self._detect_rapidocr(image)
        elif self._backend == "paddleocr":
            return self._detect_paddleocr(image)
        elif self._backend == "easyocr":
            return self._detect_easyocr(image)
        else:
            return self._detect_stub(image)

    def _detect_paddleocr(self, image: np.ndarray) -> List[TextRegion]:
        """Run PaddleOCR detection."""
        results = self._ocr.ocr(image, cls=True)
        regions = []
        if results and results[0]:
            for line in results[0]:
                # PaddleOCR returns: [polygon_points, (text, confidence)]
                poly = line[0]
                text, conf = line[1]
                # Convert polygon to [x1, y1, x2, y2]
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                regions.append(TextRegion(
                    text=text,
                    bbox=bbox,
                    confidence=float(conf),
                ))
        logger.debug("PaddleOCR detected %d regions", len(regions))
        return regions

    def _detect_easyocr(self, image: np.ndarray) -> List[TextRegion]:
        """Run EasyOCR detection."""
        results = self._ocr.readtext(image)
        regions = []
        for (bbox_poly, text, conf) in results:
            xs = [p[0] for p in bbox_poly]
            ys = [p[1] for p in bbox_poly]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            regions.append(TextRegion(
                text=text,
                bbox=bbox,
                confidence=float(conf),
            ))
        logger.debug("EasyOCR detected %d regions", len(regions))
        return regions

    def _detect_stub(self, image: np.ndarray) -> List[TextRegion]:
        """Stub OCR for testing without any OCR library."""
        h, w = image.shape[:2]
        logger.warning("Using stub OCR - no text will be detected")
        return []

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
