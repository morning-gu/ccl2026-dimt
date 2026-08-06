"""NoOp plugin implementations (Null Object Pattern)."""
from typing import List, Tuple, Dict
import numpy as np
from common.selective_translator import TextRegion
from .style_extractor import IStyleExtractorPlugin
from .context_analyzer import IContextAnalyzerPlugin
from .product_classifier import IProductClassifierPlugin
from .box_resizer import IBoxResizerPlugin
from .quality_checker import IQualityCheckerPlugin
from .translator import ITranslatorPlugin
from .classifier import IClassifierPlugin


class NoOpStyleExtractor(IStyleExtractorPlugin):
    def __init__(self, config):
        pass
    def extract_style(self, image, regions):
        return regions


class NoOpContextAnalyzer(IContextAnalyzerPlugin):
    def __init__(self, config):
        pass
    def analyze(self, image):
        return ""


class NoOpProductClassifier(IProductClassifierPlugin):
    def __init__(self, config):
        pass
    def classify(self, image, regions):
        return ("unknown", "unknown")


class NoOpBoxResizer(IBoxResizerPlugin):
    def __init__(self, config):
        pass
    def resize_regions(self, regions, source_lang, target_lang):
        return regions


class NoOpQualityChecker(IQualityCheckerPlugin):
    def __init__(self, config):
        pass
    def check(self, original, result, regions):
        return {}

class NoOpTranslator(ITranslatorPlugin):
    """NoOp translator: copies text as-is (for renderer roundtrip eval)."""

    def __init__(self, config):
        pass

    def translate_regions(self, regions, target_lang, image_context=""):
        for r in regions:
            r.translated_text = r.text
        return regions


class NoOpClassifier(IClassifierPlugin):
    """NoOp classifier: marks all regions as translatable."""

    def __init__(self, config):
        pass

    def classify_regions(self, regions):
        for r in regions:
            r.is_translatable = True
            r.region_type = "text"
        return regions