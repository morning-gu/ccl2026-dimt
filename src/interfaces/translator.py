"""Translation plugin interface."""
from abc import ABC, abstractmethod
from typing import List
from common.selective_translator import TextRegion


class ITranslatorPlugin(ABC):
    @abstractmethod
    def translate_regions(
        self,
        regions: List[TextRegion],
        target_lang: str,
        image_context: str = "",
    ) -> List[TextRegion]: ...
