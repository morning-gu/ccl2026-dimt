"""Region classification plugin interface."""
from abc import ABC, abstractmethod
from typing import List
from common.selective_translator import TextRegion


class IClassifierPlugin(ABC):
    @abstractmethod
    def classify_regions(self, regions: List[TextRegion]) -> List[TextRegion]: ...
