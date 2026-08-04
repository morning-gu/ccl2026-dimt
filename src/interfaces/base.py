"""Pipeline stage type enum and interface mapping."""
from enum import Enum


class StageType(Enum):
    OCR = "ocr"
    CLASSIFIER = "classifier"
    STYLE_EXTRACTOR = "style_extractor"
    CONTEXT_ANALYZER = "context_analyzer"
    PRODUCT_CLASSIFIER = "product_classifier"
    TRANSLATOR = "translator"
    ERASER = "eraser"
    BOX_RESIZER = "box_resizer"
    RENDERER = "renderer"
    QUALITY_CHECKER = "quality_checker"


# Populated by interfaces/__init__.py - maps each StageType to its ABC.
_STAGE_INTERFACES: dict = {}
