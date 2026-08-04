"""Text rendering plugin interface."""
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from common.selective_translator import TextRegion


class IRendererPlugin(ABC):
    @abstractmethod
    def render(
        self,
        image: np.ndarray,
        regions: List[TextRegion],
        style_reference: Optional[np.ndarray] = None,
    ) -> np.ndarray: ...
