"""Text erasure plugin interface."""
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from common.selective_translator import TextRegion


class IEraserPlugin(ABC):
    @abstractmethod
    def erase(
        self,
        image: np.ndarray,
        regions: List[TextRegion],
        dilate_pixels: int = 0,
    ) -> np.ndarray: ...
