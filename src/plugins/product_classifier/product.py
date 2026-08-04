"""Product image classifier plugin (Solution C)."""
from typing import List, Tuple
import numpy as np
from interfaces.base import StageType
from interfaces.product_classifier import IProductClassifierPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion


@register_plugin(StageType.PRODUCT_CLASSIFIER, "product")
class ProductClassifierPlugin(IProductClassifierPlugin):
    """Product type + layout classification, migrated from solution_c."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._type_keywords = {
            "electronics": ["??", "??", "??", "??", "??", "??", "??", "??",
                           "phone", "laptop", "earphone", "charger", "battery", "screen"],
            "clothing": ["??", "??", "??", "T?",
                        "dress", "pants", "shoes", "shirt", "jacket"],
            "food": ["??", "??", "??", "??", "??",
                    "snack", "drink", "fruit", "tea", "coffee"],
            "cosmetics": ["??", "??", "??", "??", "??", "??",
                         "lipstick", "foundation", "mask", "skincare", "perfume"],
            "home": ["??", "??", "??", "??",
                    "furniture", "bed", "so" + "fa", "kitchen", "storage"],
        }

    def classify(self, image: np.ndarray, regions: List[TextRegion]) -> Tuple[str, str]:
        all_text = " ".join(r.text for r in regions)
        product_type = self._classify_product_type(all_text)
        layout = self._classify_layout(image, regions)
        return (product_type, layout)

    def _classify_product_type(self, all_text: str) -> str:
        text_lower = all_text.lower()
        best_type = "other"
        best_score = 0
        for ptype, keywords in self._type_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_type = ptype
        return best_type

    def _classify_layout(self, image: np.ndarray, regions: List[TextRegion]) -> str:
        h, w = image.shape[:2]
        aspect = w / h if h > 0 else 1
        if not regions:
            return "unknown"
        if aspect > 2.0:
            return "banner"
        total_text_area = sum(r.area for r in regions)
        image_area = h * w
        text_ratio = total_text_area / image_area if image_area > 0 else 0
        if text_ratio > 0.5:
            return "text_heavy"
        if len(regions) > 10:
            return "comparison"
        if aspect < 0.6 and len(regions) > 5:
            return "detail_page"
        center_x, center_y = w / 2, h / 2
        for r in regions:
            rx, ry = r.center
            if abs(rx - center_x) < w * 0.2 and abs(ry - center_y) < h * 0.2:
                return "watermarked"
        return "product_card"
