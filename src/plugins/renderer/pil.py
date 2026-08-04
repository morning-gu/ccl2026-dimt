"""PIL rendering plugin (Solution C)."""
import logging
import numpy as np
from typing import List, Optional
from interfaces.base import StageType
from interfaces.renderer import IRendererPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion

logger = logging.getLogger(__name__)

_font_manager = None


def _get_font_manager():
    global _font_manager
    if _font_manager is None:
        from common.font_manager import FontManager
        _font_manager = FontManager()
    return _font_manager


@register_plugin(StageType.RENDERER, "pil")
class PILRendererPlugin(IRendererPlugin):
    """PIL-based rendering, migrated from common/renderer.py TextRenderer."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def render(self, image, regions, style_reference=None):
        if not regions:
            return image
        render_regions = [r for r in regions if r.translated_text]
        if not render_regions:
            return image
        return self._render_pil(image, render_regions, style_reference)

    def _render_pil(self, image, regions, style_reference=None):
        from PIL import Image, ImageDraw
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        for region in regions:
            if (not region.style_info.get("color")) and style_reference is not None:
                region.style_info = _PilStyleHelper.enrich(region.style_info, style_reference, region)
            self._draw_single(draw, img_pil, region)
        return np.array(img_pil)
    def _draw_single(self, draw, img_pil, region):
        from PIL import ImageFont
        x1, y1, x2, y2 = [int(v) for v in region.bbox[:4]]
        box_w = x2 - x1
        box_h = y2 - y1
        text = region.translated_text
        if not text or box_w <= 0 or box_h <= 0:
            return
        # If the OCR polygon indicates significant rotation, render at that angle.
        if self._draw_rotated(img_pil, region):
            return
        weight = region.style_info.get("font_weight")
        font_size = int(region.style_info.get("font_size") or max(10, int(box_h * 0.8)))
        font_size = self._fit_font_size(draw, text, font_size, box_w, box_h)
        font = self._load_font(font_size, text, weight)
        fill_color = region.style_info.get("color", (0, 0, 0))
        if not isinstance(fill_color, (tuple, list)) or len(fill_color) < 3:
            fill_color = (0, 0, 0)
        fill_color = tuple(int(c) for c in fill_color[:3])
        lines = self._wrap_text(text, font, box_w)
        line_h = draw.textbbox((0, 0), "Ag", font=font)[3]
        total_h = line_h * len(lines)
        alignment = region.style_info.get("alignment", "center")
        ty = y1 + max(0, (box_h - total_h) // 2)
        for ln in lines:
            ln_w = draw.textbbox((0, 0), ln, font=font)[2]
            if alignment == "left":
                tx = x1 + 2
            elif alignment == "right":
                tx = x1 + max(0, box_w - ln_w - 2)
            else:
                tx = x1 + max(0, (box_w - ln_w) // 2)
            draw.text((tx, ty), ln, font=font, fill=fill_color)
            ty += line_h
    def _draw_rotated(self, img_pil, region):
        """Render translated text at the angle of the original OCR polygon.

        Returns True if the rotated path was used, False to fall back to the
        regular horizontal renderer (no polygon or near-zero angle).
        """
        from PIL import Image, ImageDraw
        poly = getattr(region, "bbox_poly", None)
        if not poly or len(poly) < 4:
            return False
        # Use pre-computed angle from OCR (fallback to 0 if absent).
        angle_deg = float(getattr(region, "angle", 0.0))
        if abs(angle_deg) < 5:
            return False
        text = region.translated_text
        if not text:
            return True
        import math
        dx = poly[1][0] - poly[0][0]
        dy = poly[1][1] - poly[0][1]
        # Polygon dimensions: top-edge = text width, left-edge = text height.
        top_len = int(math.hypot(dx, dy))
        left_len = int(math.hypot(poly[3][0] - poly[0][0], poly[3][1] - poly[0][1]))
        if top_len <= 0 or left_len <= 0:
            return False
        weight = region.style_info.get("font_weight")
        font_size = int(region.style_info.get("font_size") or max(10, int(left_len * 0.8)))
        # Render into a transparent canvas, then rotate and paste.
        pad = max(top_len, left_len)
        canvas = Image.new("RGBA", (top_len + pad, left_len + pad), (0, 0, 0, 0))
        cdraw = ImageDraw.Draw(canvas)
        font_size = self._fit_font_size(cdraw, text, font_size, top_len, left_len)
        font = self._load_font(font_size, text, weight)
        fill_color = region.style_info.get("color", (0, 0, 0))
        if not isinstance(fill_color, (tuple, list)) or len(fill_color) < 3:
            fill_color = (0, 0, 0)
        fill_color = tuple(int(c) for c in fill_color[:3])
        lines = self._wrap_text(text, font, top_len)
        line_h = cdraw.textbbox((0, 0), "Ag", font=font)[3]
        total_h = line_h * len(lines)
        alignment = region.style_info.get("alignment", "center")
        cw, ch = canvas.size
        ty = (ch - total_h) // 2
        for ln in lines:
            ln_w = cdraw.textbbox((0, 0), ln, font=font)[2]
            if alignment == "left":
                tx = (cw - top_len) // 2 + 2
            elif alignment == "right":
                tx = (cw - top_len) // 2 + max(0, top_len - ln_w - 2)
            else:
                tx = (cw - ln_w) // 2
            cdraw.text((tx, ty), ln, font=font, fill=fill_color)
            ty += line_h
        rotated = canvas.rotate(-angle_deg, expand=True, resample=Image.BICUBIC)
        cx = int((region.bbox[0] + region.bbox[2]) / 2)
        cy = int((region.bbox[1] + region.bbox[3]) / 2)
        img_pil.paste(rotated, (cx - rotated.width // 2, cy - rotated.height // 2), rotated)
        return True
    def _fit_font_size(self, draw, text, size, box_w, box_h):
        size = max(10, size)
        for _ in range(12):
            font = self._load_font(size, text)
            lines = self._wrap_text(text, font, box_w)
            line_h = draw.textbbox((0, 0), "Ag", font=font)[3]
            total_h = line_h * len(lines)
            max_w = max((draw.textbbox((0, 0), ln, font=font)[2] for ln in lines), default=0)
            if max_w <= box_w and total_h <= box_h:
                return size
            if size <= 10:
                return 10
            size = max(10, int(size * 0.9))
        return size
    def _wrap_text(self, text, font, max_width):
        from PIL import Image, ImageDraw
        tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        if max_width <= 0:
            return [text]
        words = text.split(" ")
        lines, cur = [], ""
        for w in words:
            trial = w if not cur else cur + " " + w
            if tmp.textbbox((0, 0), trial, font=font)[2] <= max_width or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        final = []
        for ln in lines:
            if tmp.textbbox((0, 0), ln, font=font)[2] <= max_width:
                final.append(ln)
                continue
            cur = ""
            for ch in ln:
                trial = cur + ch
                if tmp.textbbox((0, 0), trial, font=font)[2] <= max_width:
                    cur = trial
                else:
                    if cur:
                        final.append(cur)
                    cur = ch
            if cur:
                final.append(cur)
        return final or [text]
    def _load_font(self, size, text, weight=None):
        from PIL import ImageFont
        bold = (weight == "bold")
        return _get_font_manager().load_font(size, text, bold)
class _PilStyleHelper:
    """On-the-fly style extraction for the PIL renderer (Solution C)."""
    @staticmethod
    def enrich(style_info, image, region):
        style = dict(style_info or {})
        try:
            x1, y1, x2, y2 = [max(0, int(v)) for v in region.bbox[:4]]
            h, w = image.shape[:2]
            x2 = min(w, x2); y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return style
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                return style
            if not style.get("color"):
                poly = getattr(region, "bbox_poly", None)
                style["color"] = _PilStyleHelper._text_color(roi, poly, x1, y1)
            if not style.get("font_size"):
                style["font_size"] = max(10, int((y2 - y1) * 0.75))
        except Exception:
            pass
        return style
    @staticmethod
    def _text_color(roi, polygon=None, ox=0, oy=0):
        try:
            import cv2
            if roi.ndim != 3 or roi.shape[0] < 3 or roi.shape[1] < 3:
                return (0, 0, 0)
            h, w = roi.shape[:2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # -- Estimate the local background colour --
            # When the OCR polygon is available, sample the area *outside*
            # the polygon (but inside the bbox) — this is the text's
            # immediate background, which may differ from the bbox border
            # (e.g. a light strip behind text in a dark product image).
            bg = None
            if polygon and len(polygon) >= 4:
                pts = np.array([[(int(p[0]) - ox, int(p[1]) - oy) for p in polygon]], dtype=np.int32)
                pmask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(pmask, pts, 255)
                outside = pmask == 0
                if outside.sum() > h * w * 0.1:
                    bg = np.median(roi[outside].reshape(-1, 3).astype(np.float32), axis=0)
            if bg is None:
                border = np.concatenate([roi[0, :, :], roi[-1, :, :], roi[:, 0, :], roi[:, -1, :]]).astype(np.float32)
                bg = np.median(border, axis=0)
            # -- Otsu threshold to separate the two dominant colour classes --
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dark_m = binary == 0
            light_m = binary == 255
            dark_mean = roi[dark_m].reshape(-1, 3).mean(axis=0) if dark_m.any() else bg
            light_mean = roi[light_m].reshape(-1, 3).mean(axis=0) if light_m.any() else bg
            # Text is the class that contrasts *more* with the local background.
            dark_contrast = float(np.linalg.norm(dark_mean - bg))
            light_contrast = float(np.linalg.norm(light_mean - bg))
            text_color = dark_mean if dark_contrast > light_contrast else light_mean
            return tuple(int(c) for c in np.clip(text_color, 0, 255))
        except Exception:
            return (0, 0, 0)
