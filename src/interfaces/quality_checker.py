"""Quality check plugin interface."""
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np
from common.selective_translator import TextRegion


class IQualityCheckerPlugin(ABC):
    @abstractmethod
    def check(
        self,
        original: np.ndarray,
        result: np.ndarray,
        regions: List[TextRegion],
    ) -> Dict[str, float]: ...
