"""Style extraction plugin interface."""
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from common.selective_translator import TextRegion


class IStyleExtractorPlugin(ABC):
    @abstractmethod
    def extract_style(self, image: np.ndarray, regions: List[TextRegion]) -> List[TextRegion]: ...
