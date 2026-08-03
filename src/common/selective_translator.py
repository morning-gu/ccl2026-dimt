"""Selective translation: detect which text regions to translate vs preserve.

Core innovation for the CCL2026 competition: brand names, logos, specs,
and other non-translatable content must be preserved in the source language
while translatable content (descriptions, slogans, features) gets translated.
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

from .config import BRAND_KEYWORDS_DEFAULT, LOGO_DETECTION_CLASSES

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """A detected text region in the image."""
    text: str
    bbox: List[float]  # [x1, y1, x2, y2] or polygon points
    confidence: float = 1.0
    is_translatable: bool = True
    preserve_reason: str = ""
    style_info: dict = field(default_factory=dict)  # font, size, color, etc.
    translated_text: str = ""
    region_type: str = "text"  # text, brand, logo, spec, price, url, code

    @property
    def area(self) -> float:
        if len(self.bbox) >= 4:
            w = self.bbox[2] - self.bbox[0]
            h = self.bbox[3] - self.bbox[1]
            return abs(w * h)
        return 0.0

    @property
    def center(self) -> tuple:
        if len(self.bbox) >= 4:
            return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)
        return (0, 0)


class SelectiveTranslator:
    """Decide which text regions to translate and which to preserve.

    This is the key differentiator for the competition - no existing paper
    addresses selective in-image translation.
    """

    def __init__(
        self,
        brand_keywords: Optional[List[str]] = None,
        preserve_brand: bool = True,
        preserve_logo: bool = True,
        logo_threshold: float = 0.7,
    ):
        self.brand_keywords: Set[str] = set(kw.lower() for kw in (brand_keywords or BRAND_KEYWORDS_DEFAULT))
        self.preserve_brand = preserve_brand
        self.preserve_logo = preserve_logo
        self.logo_threshold = logo_threshold

        # Patterns for non-translatable content
        self._price_pattern = re.compile(
            r'[\¥\$\€\£\₹]\s*\d[\d,.]*'  # currency symbols
            r'|\d[\d,.]*\s*(元|美元|欧元|英镑|円|円|₩)'  # trailing currency
            r'|\d+\.\d{2}'  # decimal prices like 99.99
        )
        self._url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self._code_pattern = re.compile(r'^[A-Z0-9][-A-Z0-9]{4,}$')  # product codes
        self._spec_pattern = re.compile(
            r'\d+\s*(GB|MB|TB|KB|GHz|MHz|Hz|MP|W|V|A|mAh|mm|cm|m|kg|g|mg|L|ml|寸|英寸|像素|核|线程)'
            r'|\d+[\.\d]*[\"″\']'  # screen sizes like 6.7"
        )
        self._trademark_pattern = re.compile(r'[®™©]')
        self._pure_number_pattern = re.compile(r'^[\d,.\s%]+$')
        self._pure_english_pattern = re.compile(r'^[A-Za-z0-9\s\-_\.&/]+$')

    def classify_region(self, region: TextRegion) -> TextRegion:
        """Classify a text region as translatable or preservable."""
        text = region.text.strip()
        if not text:
            region.is_translatable = False
            region.preserve_reason = "empty"
            region.region_type = "text"
            return region

        # 1. Price / currency
        if self._price_pattern.search(text):
            region.is_translatable = False
            region.preserve_reason = "price"
            region.region_type = "price"
            return region

        # 2. URL
        if self._url_pattern.search(text):
            region.is_translatable = False
            region.preserve_reason = "url"
            region.region_type = "url"
            return region

        # 3. Product code / model number
        if self._code_pattern.match(text):
            region.is_translatable = False
            region.preserve_reason = "product_code"
            region.region_type = "code"
            return region

        # 4. Technical specifications
        if self._spec_pattern.search(text):
            # Only preserve as spec when the text is essentially just numbers + units.
            # If translatable words remain after removing spec matches (e.g. "周长"
            # in "周长50cm"), let the translator handle it — it can preserve units
            # while translating the surrounding text.
            stripped = self._spec_pattern.sub("", text)
            stripped = re.sub(r'[\d\s,./\-:%~≈至到\[\]（）()]+', '', stripped)
            if not stripped:
                region.is_translatable = False
                region.preserve_reason = "specification"
                region.region_type = "spec"
                return region
            # Mixed spec + translatable text → fall through to normal translation
            # (e.g. "周长50cm", "重量2kg", "长35cm宽20cm")

        # 5. Trademark symbols
        if self._trademark_pattern.search(text):
            region.is_translatable = False
            region.preserve_reason = "trademark"
            region.region_type = "brand"
            return region

        # 6. Brand name detection
        if self.preserve_brand:
            text_lower = text.lower()
            for kw in self.brand_keywords:
                if self._brand_keyword_match(kw, text_lower):
                    region.is_translatable = False
                    region.preserve_reason = f"brand:{kw}"
                    region.region_type = "brand"
                    return region

        # 7. Pure numbers / percentages
        if self._pure_number_pattern.match(text):
            region.is_translatable = False
            region.preserve_reason = "pure_number"
            region.region_type = "spec"
            return region

        # 8. Logo detection (by region_type hint from OCR)
        if self.preserve_logo and region.region_type in LOGO_DETECTION_CLASSES:
            region.is_translatable = False
            region.preserve_reason = "logo"
            region.region_type = "logo"
            return region

        # 9. Very short pure English in Chinese context (likely brand)
        if len(text) <= 5 and self._pure_english_pattern.match(text) and not text.isnumeric():
            # Short English words in Chinese e-commerce images are often brand names
            common_english_words = {"new", "hot", "top", "best", "sale", "off", "set", "kit", "pro", "max", "plus", "mini", "lite", "go", "ai", "hd", "4k", "5g", "6g", "usb", "led", "lcd", "oled", "cpu", "gpu", "ram", "rom", "ssd", "hdd"}
            if text.lower() not in common_english_words:
                region.is_translatable = False
                region.preserve_reason = "short_english_likely_brand"
                region.region_type = "brand"
                return region

        # Default: translatable
        region.is_translatable = True
        region.region_type = "text"
        return region

    def _brand_keyword_match(self, keyword: str, text_lower: str) -> bool:
        """Match brand keyword with word-boundary awareness.

        Short keywords (<=2 chars) require a word boundary so that single
        letters like 'C' or 'R' don't match substrings of arbitrary text
        (e.g. 'c' in 'cm', 'r' in 'artwork').
        """
        if len(keyword) <= 2:
            # Require a non-alphanumeric boundary on both sides
            pattern = r'(?<![a-z0-9])' + re.escape(keyword) + r'(?![a-z0-9])'
            return re.search(pattern, text_lower) is not None
        return keyword in text_lower

    def classify_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Classify all regions."""
        return [self.classify_region(r) for r in regions]

    def get_translatable_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Return only regions that should be translated."""
        classified = self.classify_regions(regions)
        return [r for r in classified if r.is_translatable]

    def get_preserved_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Return only regions that should be preserved."""
        classified = self.classify_regions(regions)
        return [r for r in classified if not r.is_translatable]

    def merge_nearby_regions(self, regions: List[TextRegion], max_gap: float = 20.0) -> List[TextRegion]:
        """Merge nearby text regions that belong to the same logical text block.

        This helps when OCR splits a single text block into multiple fragments.
        """
        if len(regions) <= 1:
            return regions

        # Sort by vertical position, then horizontal
        sorted_regions = sorted(regions, key=lambda r: (r.center[1], r.center[0]))
        merged = [sorted_regions[0]]

        for region in sorted_regions[1:]:
            prev = merged[-1]
            # Check if regions are close enough to merge
            dx = region.center[0] - prev.center[0]
            dy = region.center[1] - prev.center[1]
            dist = (dx ** 2 + dy ** 2) ** 0.5

            if dist < max_gap and prev.is_translatable == region.is_translatable:
                # Merge: extend bbox and concatenate text
                new_bbox = [
                    min(prev.bbox[0], region.bbox[0]),
                    min(prev.bbox[1], region.bbox[1]),
                    max(prev.bbox[2], region.bbox[2]),
                    max(prev.bbox[3], region.bbox[3]),
                ]
                prev.bbox = new_bbox
                prev.text = prev.text + region.text
                prev.confidence = min(prev.confidence, region.confidence)
            else:
                merged.append(region)

        return merged
