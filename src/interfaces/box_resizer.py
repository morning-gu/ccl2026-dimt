"""Box resize plugin interface."""
from abc import ABC, abstractmethod
from typing import List
from common.selective_translator import TextRegion


class IBoxResizerPlugin(ABC):
    @abstractmethod
    def resize_regions(
        self,
        regions: List[TextRegion],
        source_lang: str,
        target_lang: str,
    ) -> List[TextRegion]: ...
