"""Pipeline context: shared state passed between stages."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from .selective_translator import TextRegion


@dataclass
class PipelineContext:
    """Pipeline context passed between stages."""
    image: np.ndarray
    image_path: str
    regions: List[TextRegion] = field(default_factory=list)
    target_lang: str = ""
    image_context: str = ""
    product_type: str = ""
    layout: str = ""
    erased_image: Optional[np.ndarray] = None
    quality_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
