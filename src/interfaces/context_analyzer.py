"""Image context analysis plugin interface."""
from abc import ABC, abstractmethod
import numpy as np


class IContextAnalyzerPlugin(ABC):
    @abstractmethod
    def analyze(self, image: np.ndarray) -> str: ...
