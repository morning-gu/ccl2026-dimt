"""Product classification plugin interface."""
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from common.selective_translator import TextRegion


class IProductClassifierPlugin(ABC):
    @abstractmethod
    def classify(self, image: np.ndarray, regions: List[TextRegion]) -> Tuple[str, str]: ...
