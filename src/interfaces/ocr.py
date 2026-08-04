"""OCR detection plugin interface."""
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from common.selective_translator import TextRegion


class IOCRPlugin(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[TextRegion]: ...

    @abstractmethod
    def detect_from_path(self, image_path: str) -> Tuple[List[TextRegion], np.ndarray]: ...
