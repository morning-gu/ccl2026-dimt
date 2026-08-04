"""CoT + VLM context-aware translator plugin (Solution B/C)."""
from typing import List
from interfaces.base import StageType
from interfaces.translator import ITranslatorPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion
from common.translator import ContextAwareTranslator


@register_plugin(StageType.TRANSLATOR, "context_aware")
class ContextAwareTranslatorPlugin(ITranslatorPlugin):
    """CoT + VLM context-aware translation, migrated from common/translator.py."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._inner = ContextAwareTranslator(config)

    def translate_regions(
        self,
        regions: List[TextRegion],
        target_lang: str,
        image_context: str = "",
    ) -> List[TextRegion]:
        return self._inner.translate_regions(
            regions, target_lang, image_context=image_context
        )
