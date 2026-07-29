"""Common modules shared across all solutions."""
from .config import PipelineConfig, LangConfig, TARGET_LANGUAGES, BRAND_KEYWORDS_DEFAULT, LOGO_DETECTION_CLASSES, load_config_from_env
from .selective_translator import TextRegion, SelectiveTranslator
from .ocr_detector import OCRDetector
from .translator import ContextAwareTranslator
from .renderer import TextEraser, TextRenderer
from .submission import SubmissionPackager
