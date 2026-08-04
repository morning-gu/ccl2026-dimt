"""Shared configuration for all solutions."""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional


def _load_dotenv_file():
    """Load .env file from project root into os.environ (without overwriting)."""
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
    except ImportError:
        from pathlib import Path
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if env_path.exists():
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if key and key not in os.environ:
                        os.environ[key] = value


_load_dotenv_file()


@dataclass
class LangConfig:
    """Target language configuration."""
    code: str
    name_cn: str
    name_en: str
    qwen_lang: str


TARGET_LANGUAGES = [
    LangConfig("en", "英语", "English", "English"),
    LangConfig("es", "西班牙语", "Spanish", "Spanish"),
    LangConfig("pt", "葡萄牙语", "Portuguese", "Portuguese"),
    LangConfig("ja", "日语", "Japanese", "Japanese"),
    LangConfig("fr", "法语", "French", "French"),
]


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    input_dir: str = ""
    output_dir: str = ""
    solution_name: str = "default"
    source_lang: str = "zh"
    target_langs: List[str] = field(default_factory=lambda: ["en", "es", "pt", "ja", "fr"])
    # Plugin composition: maps stage name -> plugin name
    plugins: Dict[str, str] = field(default_factory=lambda: {
        "ocr": "rapidocr",
        "classifier": "selective",
        "style_extractor": "noop",
        "context_analyzer": "noop",
        "product_classifier": "noop",
        "translator": "context_aware",
        "eraser": "opencv",
        "box_resizer": "noop",
        "renderer": "pil",
        "quality_checker": "noop",
    })
    # OCR settings
    ocr_det_model: str = "PP-OCRv4"
    ocr_rec_model: str = "PP-OCRv4"
    ocr_lang: str = "ch"
    ocr_det_limit_side: int = 960
    # Translation settings
    translation_model: str = "GLM-5.1"
    translation_use_vlm: bool = True
    translation_use_cot: bool = True
    translation_max_tokens: int = 2048
    translation_temperature: float = 0.3
    # Gateway compatibility: some OpenAI-compatible gateways (e.g. enterprise
    # MAAS) use a self-signed CA not in certifi, or force SSE streaming.
    translation_verify_ssl: bool = True
    translation_api_base: str = "http://127.0.0.1:8082/v1"
    translation_api_key: str = ""
    # VLM (Vision-Language Model) settings for image context analysis
    vlm_api_base: str = ""
    vlm_api_key: str = ""
    vlm_model: str = "qwen3.7-plus"
    # Selective translation settings
    selective_enabled: bool = True
    brand_keywords_file: str = ""
    logo_detection_threshold: float = 0.7
    preserve_brand: bool = True
    preserve_logo: bool = True
    # Rendering settings
    render_model: str = "anytext2"
    render_preserve_style: bool = True
    render_preserve_background: bool = True
    # Erasure settings
    erasure_model: str = "lama"
    erasure_dilate_pixels: int = 5
    # Submission settings
    submission_zip_name: str = "submission.zip"
    supported_image_formats: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"]
    )
    # Hardware settings
    device: str = "cuda"
    batch_size: int = 4
    # Logging
    log_level: str = "INFO"
    log_dir: str = ""
    # Debug intermediate file settings
    debug_enabled: bool = True          # Master switch for debug output
    debug_dir: str = ""                 # Debug output directory (auto-derived if empty)
    debug_ocr: bool = True              # Save OCR detection visualization (bbox + text overlay)
    debug_mask: bool = True             # Save erasure mask image
    debug_erased: bool = True           # Save image after text erasure (before rendering)
    debug_translation: bool = True      # Save per-region translation mapping as JSON
    debug_style: bool = True            # Save style extraction results (Solution B)
    debug_classification: bool = True   # Save selective classification results
    debug_quality: bool = True          # Save quality check results (Solution C)
    debug_render: bool = True           # Save rendered result image (for comparison)
    debug_original: bool = True         # Save a copy of the original source image
    debug_context: bool = True          # Save VLM image context analysis results (Solution B)
    debug_product: bool = True          # Save product type & layout classification (Solution C)


BRAND_KEYWORDS_DEFAULT = [
    # Chinese brands
    "华为", "小米", "OPPO", "vivo", "荣耀", "魅族", "一加",
    "海尔", "美的", "格力", "TCL", "海信", "创维",
    "李宁", "安踏", "特步", "鸿星尔克",
    "阿里巴巴", "淘宝", "天猫", "京东", "拼多多",
    "微信", "支付宝", "抖音", "快手",
    # International brands
    "Apple", "Samsung", "Nike", "Adidas", "PUMA", "SONY", "Panasonic",
    "Canon", "Nikon", "LEGO", "IKEA", "ZARA", "H&M", "UNIQLO",
    "Louis Vuitton", "Gucci", "Chanel", "Prada", "Dior", "Hermes",
    "Coca-Cola", "Pepsi", "McDonald", "KFC", "Starbucks",
    # Common marks
    "R", "TM", "C",
]

LOGO_DETECTION_CLASSES = [
    "logo", "brand", "trademark", "emblem", "badge", "seal",
]


def load_config_from_env(cfg: PipelineConfig) -> PipelineConfig:
    """Override config fields from environment variables."""
    env_map = {
        "INPUT_DIR": "input_dir",
        "OUTPUT_DIR": "output_dir",
        "SOURCE_LANG": "source_lang",
        "TRANSLATION_MODEL": "translation_model",
        "TRANSLATION_API_BASE": "translation_api_base",
        "TRANSLATION_API_KEY": "translation_api_key",
        "VLM_API_BASE": "vlm_api_base",
        "VLM_API_KEY": "vlm_api_key",
        "VLM_MODEL": "vlm_model",
        "DEVICE": "device",
        "BATCH_SIZE": "batch_size",
        "LOG_LEVEL": "log_level",
        "DEBUG_ENABLED": "debug_enabled",
        "DEBUG_DIR": "debug_dir",
        "DEBUG_OCR": "debug_ocr",
        "DEBUG_MASK": "debug_mask",
       "DEBUG_ERASED": "debug_erased",
       "DEBUG_TRANSLATION": "debug_translation",
        "TARGET_LANGS": "target_langs",
    }

    bool_fields = {
        "RENDER_PRESERVE_STYLE": "render_preserve_style",
        "RENDER_PRESERVE_BACKGROUND": "render_preserve_background",
        "SELECTIVE_ENABLED": "selective_enabled",
        "PRESERVE_BRAND": "preserve_brand",
        "PRESERVE_LOGO": "preserve_logo",
        "TRANSLATION_VERIFY_SSL": "translation_verify_ssl",
        "DEBUG_ENABLED": "debug_enabled",
        "DEBUG_OCR": "debug_ocr",
        "DEBUG_MASK": "debug_mask",
        "DEBUG_ERASED": "debug_erased",
        "DEBUG_TRANSLATION": "debug_translation",
        "DEBUG_STYLE": "debug_style",
        "DEBUG_CLASSIFICATION": "debug_classification",
        "DEBUG_QUALITY": "debug_quality",
        "DEBUG_RENDER": "debug_render",
        "DEBUG_ORIGINAL": "debug_original",
        "DEBUG_CONTEXT": "debug_context",
        "DEBUG_PRODUCT": "debug_product",
    }

    int_fields = {
        "BATCH_SIZE": "batch_size",
        "TRANSLATION_MAX_TOKENS": "translation_max_tokens",
        "OCR_DET_LIMIT_SIDE": "ocr_det_limit_side",
        "ERASURE_DILATE_PIXELS": "erasure_dilate_pixels",
    }

    float_fields = {
        "TRANSLATION_TEMPERATURE": "translation_temperature",
        "LOGO_DETECTION_THRESHOLD": "logo_detection_threshold",
    }

    list_fields = {
        "TARGET_LANGS": "target_langs",
    }

    env_map.update(bool_fields)
    env_map.update(int_fields)
    env_map.update(float_fields)
    env_map.update(list_fields)

    for env_key, field_name in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            if field_name in int_fields.values():
                val = int(val)
            elif field_name in float_fields.values():
                val = float(val)
            elif field_name in bool_fields.values():
                val = val.lower() in ("1", "true", "yes")
            elif field_name in list_fields.values():
                # Comma- or space-separated list, e.g. "en,es,pt" or "en es pt"
                val = [x.strip() for x in val.replace(",", " ").split() if x.strip()]
            setattr(cfg, field_name, val)
    return cfg
