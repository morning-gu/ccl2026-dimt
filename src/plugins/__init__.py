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

# Concrete plugins (uncomment as each is created in Tasks 3-6):
# from .ocr import rapidocr  # noqa
# from .classifier import selective, ecommerce  # noqa
# from .style_extractor import style  # noqa
# from .context_analyzer import vlm  # noqa
# from .product_classifier import product  # noqa
# from .translator import anytrans, context_aware  # noqa
# from .eraser import lama, opencv, sd_inpaint, pert, strokenet  # noqa
# from .box_resizer import anytrans as box_anytrans  # noqa
# from .renderer import anytext2, pil  # noqa
# from .quality_checker import basic  # noqa
