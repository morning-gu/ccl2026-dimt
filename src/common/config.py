"""Shared configuration for all solutions."""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional


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
    source_lang: str = "zh"
    target_langs: List[str] = field(default_factory=lambda: ["en", "es", "pt", "ja", "fr"])
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
    translation_api_base: str = "http://127.0.0.1:8082/v1"
    translation_api_key: str = "sk-12345679"
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
        "RENDER_MODEL": "render_model",
        "ERASURE_MODEL": "erasure_model",
        "DEVICE": "device",
        "BATCH_SIZE": "batch_size",
        "LOG_LEVEL": "log_level",
        "DEBUG_ENABLED": "debug_enabled",
        "DEBUG_DIR": "debug_dir",
        "DEBUG_OCR": "debug_ocr",
        "DEBUG_MASK": "debug_mask",
        "DEBUG_ERASED": "debug_erased",
        "DEBUG_TRANSLATION": "debug_translation",
    }
    for env_key, field_name in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            if field_name in ("batch_size",):
                val = int(val)
            elif field_name in ("debug_enabled", "debug_ocr", "debug_mask",
                                "debug_erased", "debug_translation",
                                "debug_style", "debug_classification",
                                "debug_quality", "debug_render",
                                "debug_original", "debug_context",
                                "debug_product"):
                val = val.lower() in ("1", "true", "yes")
            setattr(cfg, field_name, val)
    return cfg
