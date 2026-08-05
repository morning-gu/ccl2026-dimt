"""Plugin package: import all plugin modules to trigger registration."""
from .registry import registry, register_plugin, PluginRegistry
from interfaces.base import StageType
from interfaces.noop import (
    NoOpStyleExtractor, NoOpContextAnalyzer,
    NoOpProductClassifier, NoOpBoxResizer, NoOpQualityChecker,
)

# Register NoOp plugins
registry.register(StageType.STYLE_EXTRACTOR, "noop", NoOpStyleExtractor)
registry.register(StageType.CONTEXT_ANALYZER, "noop", NoOpContextAnalyzer)
registry.register(StageType.PRODUCT_CLASSIFIER, "noop", NoOpProductClassifier)
registry.register(StageType.BOX_RESIZER, "noop", NoOpBoxResizer)
registry.register(StageType.QUALITY_CHECKER, "noop", NoOpQualityChecker)

# Concrete plugins
from .ocr import rapidocr  # noqa: E402,F401
from .classifier import selective, ecommerce  # noqa: E402,F401
from .style_extractor import style  # noqa: E402,F401
from .context_analyzer import vlm  # noqa: E402,F401
from .product_classifier import product  # noqa: E402,F401
from .translator import anytrans, context_aware  # noqa: E402,F401
from .eraser import lama, opencv, sd_inpaint, pert, strokenet  # noqa: E402,F401
from .box_resizer import anytrans as box_anytrans  # noqa: E402,F401
from .renderer import anytext2, pil  # noqa: E402,F401
from .quality_checker import basic  # noqa: E402,F401
from .quality_checker import competition  # noqa: E402,F401
