"""E-commerce selective classifier plugin."""
import re
from typing import List
from interfaces.base import StageType
from interfaces.classifier import IClassifierPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion, SelectiveTranslator


# Promotional patterns copied from solution_c/pipeline.py
_PROMO_PATTERNS = [
    (re.compile(r"限时[优特]惠"), "time_limited_offer"),
    (re.compile(r"新品上市"), "new_arrival"),
    (re.compile(r"爆款"), "bestseller"),
    (re.compile(r"热卖"), "hot_sale"),
    (re.compile(r"满\d+减\d+"), "discount_threshold"),
    (re.compile(r"买\d+送\d+"), "buy_get_free"),
    (re.compile(r"包邮"), "free_shipping"),
    (re.compile(r"7天无理由退换"), "7_day_return"),
]

# Customer service keywords
_CS_KEYWORDS = ["客服", "售后", "退换", "保修", "发票", "配送", "物流", "安装"]


@register_plugin(StageType.CLASSIFIER, "ecommerce")
class EcommerceClassifierPlugin(IClassifierPlugin):
    """E-commerce extended classifier, migrated from solution_c.

    Uses composition instead of inheritance to avoid LSP issues.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._base = SelectiveTranslator(
            preserve_brand=config.preserve_brand,
            preserve_logo=config.preserve_logo,
            logo_threshold=config.logo_detection_threshold,
        )

    def classify_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        return [self._classify_region(r) for r in regions]

    def _classify_region(self, region: TextRegion) -> TextRegion:
        text = region.text.strip()

        # Check promotional patterns first
        for pattern, promo_type in _PROMO_PATTERNS:
            if pattern.search(text):
                region.is_translatable = True
                region.region_type = "promo"
                region.style_info["promo_type"] = promo_type
                return region

        # Customer service / policy text should be translated
        if any(kw in text for kw in _CS_KEYWORDS):
            region.is_translatable = True
            region.region_type = "service"
            return region

        # Feature list items (bullet points, numbered lists)
        if re.match(r'^[•·▪▸➤◆★☆]\s*', text) or re.match(r'^\d+[.、)\]]\s*', text):
            region.is_translatable = True
            region.region_type = "feature"
            return region

        # Fall back to base classification
        return self._base.classify_region(region)
