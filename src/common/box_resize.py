"""Anticipated Box Resize (AnyTrans Section 3.3).

When the translated text length differs significantly from the source text
length, the original OCR box may be too small or too large for clean text
rendering. This module resizes the anticipated target box based on the
word/character count ratio, using language-pair-specific char-length ratios
as described in the AnyTrans paper.

Rules (from AnyTrans Section 3.3):
  - If word_count_ratio > 1.2: enlarge the box (translated text is longer)
  - If word_count_ratio < 0.8: shrink the box (translated text is shorter)
  - Otherwise: keep the original box size
  - For zh->en, one Chinese character ~= 2.5 English letters
  - Resize the dimension that matches the text orientation (width for
    horizontal text, height for vertical text)
"""
import logging
from typing import List

from .selective_translator import TextRegion

logger = logging.getLogger(__name__)

# Language-pair-specific character length ratios.
# Key format: "src->tgt". Value = how many target chars one source char
# roughly maps to. E.g., zh->en: one Chinese char ~= 2.5 English letters.
CHAR_LENGTH_RATIOS = {
    "zh->en": 2.5,
    "zh->es": 2.5,
    "zh->pt": 2.5,
    "zh->fr": 2.5,
    "zh->ja": 1.0,
    "en->zh": 0.4,
    "ja->zh": 1.0,
    "es->zh": 0.4,
    "pt->zh": 0.4,
    "fr->zh": 0.4,
}

# Thresholds from the AnyTrans paper
RATIO_UPPER = 1.2  # enlarge if ratio > 1.2
RATIO_LOWER = 0.8  # shrink if ratio < 0.8


def _count_effective_chars(text: str, lang: str) -> int:
    """Count effective characters for length comparison.

    For CJK languages (zh, ja), each character is roughly square.
    For Latin languages (en, es, pt, fr), each letter is narrower.
    We count non-space characters in both cases for simplicity;
    the language-pair ratio handles the width difference.
    """
    if not text:
        return 0
    return len(text.replace(" ", "").replace("\n", ""))


def _compute_length_ratio(
    src_text: str,
    tgt_text: str,
    src_lang: str,
    tgt_lang: str,
) -> float:
    """Compute the effective length ratio between source and translated text.

    Returns the ratio of target effective length to source effective length,
    adjusted by the language-pair character length ratio.

    A ratio > 1.0 means the translated text needs more space than the source.
    """
    src_chars = _count_effective_chars(src_text, src_lang)
    tgt_chars = _count_effective_chars(tgt_text, tgt_lang)
    if src_chars == 0:
        return 1.0

    pair_key = f"{src_lang}->{tgt_lang}"
    char_ratio = CHAR_LENGTH_RATIOS.get(pair_key, 1.0)

    # Effective space needed relative to source char width:
    # For zh->en (char_ratio=2.5): one Chinese char = 2.5 English letters
    #   src has src_chars Chinese chars (each = 1 width unit)
    #   tgt has tgt_chars English letters (each = 1/2.5 width unit)
    #   tgt_effective = tgt_chars / 2.5
    #   ratio = tgt_effective / src_chars
    if char_ratio != 1.0:
        tgt_effective = tgt_chars / char_ratio
    else:
        tgt_effective = tgt_chars

    return tgt_effective / src_chars


def resize_anticipated_box(
    region: TextRegion,
    src_lang: str,
    tgt_lang: str,
    max_enlarge: float = 2.0,
    max_shrink: float = 0.5,
) -> TextRegion:
    """Resize the OCR box for a translated text region.

    Implements the Anticipated Box Resize strategy from AnyTrans Section 3.3.
    Modifies the region's bbox in-place and returns the region.

    Args:
        region: A TextRegion with translated_text set.
        src_lang: Source language code (e.g. "zh").
        tgt_lang: Target language code (e.g. "en").
        max_enlarge: Maximum enlargement factor (safety clamp).
        max_shrink: Maximum shrink factor (safety clamp).

    Returns:
        The same region with potentially resized bbox.
    """
    src_text = region.text
    tgt_text = region.translated_text
    if not src_text or not tgt_text:
        return region

    ratio = _compute_length_ratio(src_text, tgt_text, src_lang, tgt_lang)

    # Only resize if ratio deviates significantly (AnyTrans thresholds)
    if RATIO_LOWER <= ratio <= RATIO_UPPER:
        return region

    # Clamp the resize factor for safety
    resize_factor = max(min(ratio, max_enlarge), max_shrink)

    x1, y1, x2, y2 = [float(v) for v in region.bbox[:4]]
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return region

    # Determine text orientation: if box is taller than wide, it is likely
    # vertical text (resize height); otherwise horizontal (resize width).
    is_vertical = box_h > box_w * 1.5

    if is_vertical:
        new_h = box_h * resize_factor
        cy = (y1 + y2) / 2
        new_y1 = max(0, cy - new_h / 2)
        region.bbox = [x1, new_y1, x2, new_y1 + new_h]
    else:
        new_w = box_w * resize_factor
        cx = (x1 + x2) / 2
        new_x1 = max(0, cx - new_w / 2)
        region.bbox = [new_x1, y1, new_x1 + new_w, y2]

    logger.debug(
        "Box resized: %s->%s ratio=%.2f factor=%.2f %s bbox=%s",
        src_lang, tgt_lang, ratio, resize_factor,
        "vertical" if is_vertical else "horizontal",
        [round(v, 1) for v in region.bbox],
    )
    return region


def resize_regions(
    regions: List[TextRegion],
    src_lang: str,
    tgt_lang: str,
) -> List[TextRegion]:
    """Resize all translatable regions with translated text.

    Preserved (non-translatable) regions are left unchanged.
    """
    for r in regions:
        if r.is_translatable and r.translated_text:
            resize_anticipated_box(r, src_lang, tgt_lang)
    return regions
