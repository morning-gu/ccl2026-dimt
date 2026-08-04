"""Common modules shared across all solutions."""
from .config import PipelineConfig, LangConfig, TARGET_LANGUAGES, BRAND_KEYWORDS_DEFAULT, LOGO_DETECTION_CLASSES, load_config_from_env
from .selective_translator import TextRegion, SelectiveTranslator
from .translator import ContextAwareTranslator
from .submission import SubmissionPackager
