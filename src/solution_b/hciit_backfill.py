"""HCIIT Stage 2: Image Backfilling with Style Consistency.

Implements the image generation stage from the HCIIT paper
(Fu et al., "Ensuring Consistency for In-Image Translation"):

  Stage 2 ensures Image Generation Consistency via:
    a) Style Latent Module: encodes style image S + background image B
       -> Zs = g(D(S) + D(B))
       This conditions the diffusion model to generate text matching the
       original style (font, color, thickness).
    b) Glyph Latent Module: encodes glyph + position + masked image
       -> Za = f(G(lg) + P(lp) + D(lm))
       (Consistent with AnyText, provides text content and position info.)
    c) Text Erase Model: removes original text from style image to get
       background image B (clean background for style conditioning).

  OPEN-SOURCE STATUS:
    The HCIIT paper's custom style-consistent diffusion model (trained on
    400K pseudo pairs with Style Latent Module + TextControlNet) has NO
    open-source implementation or pretrained weights available.

    Therefore, we use AnyText2 as the rendering backend, which provides:
    - Glyph Latent Module equivalent (glyph encoder + position encoder)
    - Font Encoder (AnyText2's font encoder, not in original AnyText)
    - Color Encoder (AnyText2's color encoder)
    - WriteNet+AttnX architecture for text-image fusion

    We enhance AnyText2 with HCIIT's Style Latent Module concept by:
    1. Extracting the style image (original text region) as font_hint
    2. Extracting the background image (erased region) for conditioning
    3. Passing style info (color, font hints) to AnyText2's rendering API

  This module wraps the common renderer with HCIIT-specific preprocessing.
"""
import logging
from typing import List, Optional, Tuple
import numpy as np

from common.config import PipelineConfig
from common.selective_translator import TextRegion
from common.renderer import TextEraser, TextRenderer

logger = logging.getLogger(__name__)


class StyleLatentExtractor:
    """Extract style and background information for HCIIT Stage 2.

    In the paper, Style Latent Module computes:
        Zs = g(D(S) + D(B))
    where S is the style image (original text region) and B is the
    background image (text erased). Since we use AnyText2 instead of
    the paper's custom diffusion model, we extract discrete style
    attributes and font hint images that AnyText2 can consume.

    This provides a practical approximation of the Style Latent Module:
    - Font hint images: cropped text regions from the original image
      (fed to AnyText2's font_hint_image parameter)
    - Text colors: per-region dominant text color
      (fed to AnyText2's text_colors parameter)
    - Background image: the erased image
      (used as the base image for rendering)
    """

    def __init__(self):
        pass

    def extract_style_hints(
        self,
        original_image: np.ndarray,
        erased_image: np.ndarray,
        regions: List[TextRegion],
    ) -> List[dict]:
        """Extract per-region style hints for HCIIT-style backfilling.

        Args:
            original_image: Original source image (BGR).
            erased_image: Image after text erasure (BGR).
            regions: Translatable text regions with style_info populated.

        Returns:
            List of dicts with keys:
              - font_hint_image: cropped original text region (RGB numpy)
              - font_hint_mask: binary mask of the text region
              - color: detected text color (R,G,B)
              - bg_color: detected background color (R,G,B)
        """
        h, w = original_image.shape[:2]
        hints = []
        for region in regions:
            hint = {}
            x1, y1, x2, y2 = [int(v) for v in region.bbox[:4]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Font hint image: crop from original (contains text style info)
            # This is the HCIIT Style Image S for this region
            if x2 > x1 and y2 > y1:
                style_crop = original_image[y1:y2, x1:x2]
                # Convert BGR -> RGB for AnyText2
                hint["font_hint_image"] = style_crop[:, :, ::-1].copy()

                # Font hint mask: white rectangle on black canvas
                mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                mask[:, :] = 255
                hint["font_hint_mask"] = mask
            else:
                hint["font_hint_image"] = None
                hint["font_hint_mask"] = None

            # Colors from style_info (extracted by StyleExtractor earlier)
            style = region.style_info or {}
            hint["color"] = style.get("color", (0, 0, 0))
            hint["bg_color"] = style.get("bg_color", (255, 255, 255))

            hints.append(hint)
        return hints


class HCIITBackfiller:
    """HCIIT Stage 2: Image backfilling with style consistency.

    Two-stage process (matching the paper):
      1. Text Erase: Remove original text to get background image B
         (HCIIT's "Text Erase Model" step)
      2. Style-Consistent Render: Render translated text with style
         conditioning from original text regions
         (HCIIT's "Style Latent Module + Glyph Latent Module" step)

    Since the paper's custom diffusion model is not open-sourced,
    we use AnyText2 as the rendering backend with HCIIT-style
    preprocessing (font hints + color conditioning).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.eraser = TextEraser(config)
        self.renderer = TextRenderer(config)
        self.style_extractor = StyleLatentExtractor()

    def erase_text(
        self,
        image: np.ndarray,
        regions: List[TextRegion],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Erase text from image to get background image B.

        This is HCIIT's "Text Erase Model" step (Figure 3).
        The erased image serves as the background input for the
        Style Latent Module: Zs = g(D(S) + D(B)).

        Args:
            image: Original source image (BGR).
            regions: Translatable text regions to erase.

        Returns:
            Tuple of (mask, erased_image).
        """
        if not regions:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            return mask, image

        mask = self.eraser._build_mask(
            image.shape[:2], regions,
            dilate=self.config.erasure_dilate_pixels,
        )
        erased = self.eraser.erase(image, regions)
        return mask, erased

    def backfill(
        self,
        original_image: np.ndarray,
        erased_image: np.ndarray,
        regions: List[TextRegion],
    ) -> np.ndarray:
        """Render translated text with style consistency (HCIIT Stage 2).

        Implements the paper's image backfilling with style conditioning.
        Since the paper's custom diffusion model is not available, we use
        AnyText2 enhanced with HCIIT's style conditioning approach:

        1. Extract style hints from original text regions (Style Image S)
        2. Use erased image as background (Background Image B)
        3. Pass style hints to AnyText2 for font/color conditioning

        Args:
            original_image: Original source image (for style reference).
            erased_image: Image after text erasure (background B).
            regions: Translatable regions with translated_text set.

        Returns:
            Final image with translated text rendered.
        """
        render_regions = [r for r in regions if r.translated_text]
        if not render_regions:
            return erased_image

        # Extract HCIIT-style hints (approximation of Style Latent Module)
        style_hints = self.style_extractor.extract_style_hints(
            original_image, erased_image, render_regions
        )

        # Enrich region style_info with HCIIT style hints for AnyText2
        for region, hint in zip(render_regions, style_hints):
            if not region.style_info:
                region.style_info = {}
            # Ensure color is set from style extraction
            if "color" not in region.style_info or not region.style_info["color"]:
                region.style_info["color"] = hint["color"]
            # Store font hint references for AnyText2 rendering
            region.style_info["font_hint_image"] = hint["font_hint_image"]
            region.style_info["font_hint_mask"] = hint["font_hint_mask"]

        # Render using AnyText2 (with HCIIT style conditioning)
        result = self.renderer.render(
            erased_image, render_regions, style_reference=original_image
        )
        return result
