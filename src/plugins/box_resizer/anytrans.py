"""AnyTrans box resize plugin (Solution A, AnyTrans Section 3.3)."""
from typing import List
from interfaces.base import StageType
from interfaces.box_resizer import IBoxResizerPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion
from common.box_resize import resize_regions


@register_plugin(StageType.BOX_RESIZER, "anytrans")
class AnyTransBoxResizerPlugin(IBoxResizerPlugin):
    """Anticipated box resize, migrated from common/box_resize.py."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def resize_regions(
        self,
        regions: List[TextRegion],
        source_lang: str,
        target_lang: str,
    ) -> List[TextRegion]:
        return resize_regions(regions, source_lang, target_lang)
