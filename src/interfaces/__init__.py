"""Unified pipeline stage interfaces."""
from .base import StageType, _STAGE_INTERFACES
from .ocr import IOCRPlugin
from .classifier import IClassifierPlugin
from .style_extractor import IStyleExtractorPlugin
from .context_analyzer import IContextAnalyzerPlugin
from .product_classifier import IProductClassifierPlugin
from .translator import ITranslatorPlugin
from .eraser import IEraserPlugin
from .box_resizer import IBoxResizerPlugin
from .renderer import IRendererPlugin
from .quality_checker import IQualityCheckerPlugin

_STAGE_INTERFACES.update({
    StageType.OCR: IOCRPlugin,
    StageType.CLASSIFIER: IClassifierPlugin,
    StageType.STYLE_EXTRACTOR: IStyleExtractorPlugin,
    StageType.CONTEXT_ANALYZER: IContextAnalyzerPlugin,
    StageType.PRODUCT_CLASSIFIER: IProductClassifierPlugin,
    StageType.TRANSLATOR: ITranslatorPlugin,
    StageType.ERASER: IEraserPlugin,
    StageType.BOX_RESIZER: IBoxResizerPlugin,
    StageType.RENDERER: IRendererPlugin,
    StageType.QUALITY_CHECKER: IQualityCheckerPlugin,
})

__all__ = [
    "StageType", "_STAGE_INTERFACES",
    "IOCRPlugin", "IClassifierPlugin", "IStyleExtractorPlugin",
    "IContextAnalyzerPlugin", "IProductClassifierPlugin", "ITranslatorPlugin",
    "IEraserPlugin", "IBoxResizerPlugin", "IRendererPlugin", "IQualityCheckerPlugin",
]
