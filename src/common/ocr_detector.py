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
        # Single-char detections that are almost always graphical symbols
        # (crosses, plus signs, daggers) rather than real text in product images.
        self._symbol_chars = {"十", "✚", "✛", "✜", "✝", "†", "‡", "+", "＋", "✠", "✡"}

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
                # Rotation angle from the top edge (poly[0] -> poly[1]).
                import math
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
        # Filter single-char graphical symbols (cross/十字架 etc.)
        # OCR often misidentifies cross shapes as the character "十".
        regions = [r for r in regions if not self._is_likely_symbol(r)]
        # Re-recognize long text regions that may have been truncated
        regions = self._rerecognize_long_regions(image, regions)
        logger.debug("RapidOCR detected %d regions", len(regions))
        return regions

    def _rerecognize_long_regions(self, image: np.ndarray,
                                  regions: List[TextRegion],
                                  min_width: int = 400) -> List[TextRegion]:
        """Re-recognize wide text regions by splitting and re-reading.

        PP-OCRv4's recognizer has a max output length (~25 chars).  Wide
        regions with long text get truncated.  This method splits wide
        crops into overlapping halves, recognizes each separately, and
        concatenates the results when the combined text is longer.
        """
        try:
            import cv2
        except ImportError:
            return regions

        for region in regions:
            x1, y1, x2, y2 = [int(v) for v in region.bbox[:4]]
            w = x2 - x1
            if w < min_width:
                continue
            # Crop with small padding
            pad = 3
            h_img, w_img = image.shape[:2]
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w_img, x2 + pad)
            cy2 = min(h_img, y2 + pad)
            crop = image[cy1:cy2, cx1:cx2]
            if crop.shape[0] < 5 or crop.shape[1] < 5:
                continue
            # Split into overlapping halves
            cw = crop.shape[1]
            overlap = max(10, int(cw * 0.08))
            mid = cw // 2
            left = crop[:, :mid + overlap]
            right = crop[:, mid - overlap:]
            # Recognize each half
            try:
                rec_results, _ = self._ocr.text_rec([left, right])
            except Exception:
                continue
            if len(rec_results) < 2:
                continue
            left_text, left_conf = rec_results[0]
            right_text, right_conf = rec_results[1]
            # Concatenate; the overlap may cause duplicate chars at the
            # boundary, but more text is better than truncated text.
            combined = left_text + right_text
            # Only replace if the combined text is meaningfully longer
            if len(combined) > len(region.text) + 2:
                region.text = combined
                region.confidence = min(left_conf, right_conf)
                logger.debug("Re-recognized long region: %d -> %d chars",
                             len(region.text), len(combined))
        return regions

    def _is_likely_symbol(self, region: TextRegion) -> bool:
        """Return True if the region is a single-char graphical symbol, not text."""
        text = region.text.strip()
        if len(text) != 1:
            return False
        if text in self._symbol_chars:
            logger.debug("Filtering symbol region: %r (conf=%.3f)", text, region.confidence)
            return True
        return False
