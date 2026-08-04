"""Style extraction plugin (Solution B)."""
import logging
from typing import List, Tuple
import numpy as np
from interfaces.base import StageType
from interfaces.style_extractor import IStyleExtractorPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion

logger = logging.getLogger(__name__)


@register_plugin(StageType.STYLE_EXTRACTOR, "style")
class StyleExtractorPlugin(IStyleExtractorPlugin):
    """Extract per-region style info, migrated from solution_b StyleExtractor."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def extract_style(self, image: np.ndarray, regions: List[TextRegion]) -> List[TextRegion]:
        for region in regions:
            region.style_info = self._extract_single(image, region)
        return regions

    def _extract_single(self, image: np.ndarray, region: TextRegion) -> dict:
        x1, y1, x2, y2 = [int(v) for v in region.bbox[:4]]
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return {}
        style = {}
        box_h = y2 - y1
        style["font_size"] = max(10, int(box_h * 0.75))
        style["color"] = self._detect_text_color(roi)
        style["bg_color"] = self._detect_bg_color(roi)
        style["font_weight"] = self._detect_weight(roi)
        style["alignment"] = "center"
        box_w = x2 - x1
        style["is_vertical"] = box_h > box_w * 2
        return style

    def _detect_text_color(self, roi):
        try:
            import cv2
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean = np.mean(gray)
            if mean > 128:
                mask = gray < mean - 30
            else:
                mask = gray > mean + 30
            if mask.any():
                text_pixels = roi[mask]
                color = tuple(int(c) for c in np.median(text_pixels, axis=0))
            else:
                color = (0, 0, 0)
            return color
        except Exception:
            return (0, 0, 0)

    def _detect_bg_color(self, roi):
        try:
            import cv2
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean = np.mean(gray)
            if mean > 128:
                mask = gray >= mean - 30
            else:
                mask = gray <= mean + 30
            if mask.any():
                bg_pixels = roi[mask]
                color = tuple(int(c) for c in np.median(bg_pixels, axis=0))
            else:
                color = (255, 255, 255)
            return color
        except Exception:
            return (255, 255, 255)

    def _detect_weight(self, roi):
        try:
            import cv2
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
            median_width = np.median(dist[dist > 0]) if (dist > 0).any() else 1
            return "bold" if median_width > 3.5 else "normal"
        except Exception:
            return "normal"
