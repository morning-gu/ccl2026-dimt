"""Per-image configuration overrides for isolated tuning.

This module provides a mechanism to apply per-image parameter overrides
without affecting other images. It reads a JSON config file that maps
image stems (e.g. "010") to parameter overrides for erasure, rendering,
OCR, and pipeline stages.

Usage:
    config = ImageConfig.load()           # load from default path
    overrides = config.get("010")         # get overrides for image 010
    if overrides.erasure_threshold_method:
        ...  # use the override instead of the default

The config file (image_overrides.json) lives at the project root.
If it does not exist, all queries return empty overrides (no-op).
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ImageOverrides:
    """Parameter overrides for a single image."""
    # Erasure parameters
    erasure_threshold_method: Optional[str] = None  # "border", "otsu", "auto"
    erasure_dilate_pixels: Optional[int] = None
    erasure_cc_min_area: Optional[int] = None
    erasure_cc_min_fill_ratio: Optional[float] = None
    erasure_cc_max_thin_length: Optional[int] = None

    # Rendering parameters
    render_color_method: Optional[str] = None  # "otsu_contrast", "percentile"
    render_color_percentile: Optional[float] = None
    render_font_size_factor: Optional[float] = None

    # OCR parameters
    ocr_merge_adjacent: Optional[bool] = None
    ocr_merge_max_gap: Optional[float] = None

    # Pipeline parameters
    pipeline_skip_translation_patterns: Optional[list] = None

    def has_erasure_overrides(self) -> bool:
        return any(v is not None for v in [
            self.erasure_threshold_method, self.erasure_dilate_pixels,
            self.erasure_cc_min_area, self.erasure_cc_min_fill_ratio,
            self.erasure_cc_max_thin_length,
        ])

    def has_render_overrides(self) -> bool:
        return any(v is not None for v in [
            self.render_color_method, self.render_color_percentile,
            self.render_font_size_factor,
        ])

    def has_ocr_overrides(self) -> bool:
        return any(v is not None for v in [
            self.ocr_merge_adjacent, self.ocr_merge_max_gap,
        ])

    def has_pipeline_overrides(self) -> bool:
        return self.pipeline_skip_translation_patterns is not None


class ImageConfig:
    """Load and query per-image override configurations."""

    def __init__(self, overrides: dict = None):
        self._overrides: dict[str, ImageOverrides] = overrides or {}

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "ImageConfig":
        """Load image overrides from a JSON file."""
        if config_path is None:
            root = Path(__file__).resolve().parent.parent.parent
            config_path = root / "image_overrides.json"
        path = Path(config_path)
        if not path.exists():
            logger.debug("Image overrides file not found: %s (using defaults)", path)
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load image overrides from %s: %s", path, e)
            return cls()
        overrides = {}
        for stem, params in raw.items():
            if not isinstance(params, dict):
                continue
            overrides[stem] = ImageOverrides(
                erasure_threshold_method=params.get("erasure_threshold_method"),
                erasure_dilate_pixels=params.get("erasure_dilate_pixels"),
                erasure_cc_min_area=params.get("erasure_cc_min_area"),
                erasure_cc_min_fill_ratio=params.get("erasure_cc_min_fill_ratio"),
                erasure_cc_max_thin_length=params.get("erasure_cc_max_thin_length"),
                render_color_method=params.get("render_color_method"),
                render_color_percentile=params.get("render_color_percentile"),
                render_font_size_factor=params.get("render_font_size_factor"),
                ocr_merge_adjacent=params.get("ocr_merge_adjacent"),
                ocr_merge_max_gap=params.get("ocr_merge_max_gap"),
                pipeline_skip_translation_patterns=params.get("pipeline_skip_translation_patterns"),
            )
        logger.info("Loaded image overrides for %d images from %s", len(overrides), path)
        return cls(overrides)

    def get(self, image_stem: str) -> ImageOverrides:
        """Get overrides for an image stem (e.g. "010")."""
        return self._overrides.get(image_stem, ImageOverrides())

    def has_overrides(self, image_stem: str) -> bool:
        """Check if any overrides exist for this image stem."""
        return image_stem in self._overrides
