"""Selective translation classifier plugin."""
from typing import List
from interfaces.base import StageType
from interfaces.classifier import IClassifierPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion, SelectiveTranslator


@register_plugin(StageType.CLASSIFIER, "selective")
class SelectiveClassifierPlugin(IClassifierPlugin):
    """Base selective classifier, migrated from common/selective_translator.py."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._inner = SelectiveTranslator(
            preserve_brand=config.preserve_brand,
            preserve_logo=config.preserve_logo,
            logo_threshold=config.logo_detection_threshold,
        )

    def classify_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        return self._inner.classify_regions(regions)
