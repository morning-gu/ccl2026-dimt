"""Competition-aligned quality checker for CCL2026-Eval (14 dimensions).

Computes local proxies for all 14 scoring dimensions used by the official
evaluator, so the pipeline can self-assess without submitting:

  Translation quality (t_):  t_pixel t_pos t_layout t_font t_color
                              t_hallu t_omiss t_size
  Source preservation  (s_):  s_pixel s_pos s_color s_size s_hallu s_omiss

All s_ metrics and the geometry-based t_ metrics are computed from
(original, result, regions) alone. t_pixel / t_hallu use optional OCR
re-read of the rendered image (set QUALITY_OCR_VERIFY=1 to enable); when
disabled, t_hallu falls back to a structure-based heuristic and t_pixel is
reported as 0.0 (it genuinely requires text verification).
"""
import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np

from interfaces.base import StageType
from interfaces.quality_checker import IQualityCheckerPlugin
from plugins.registry import register_plugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion

logger = logging.getLogger(__name__)

_WEIGHTS = {
    "t_pixel": 3.0, "t_pos": 3.0, "t_layout": 3.0, "t_font": 3.0,
    "t_color": 3.0, "t_hallu": 3.0, "t_omiss": 3.0, "t_size": 3.0,
    "s_pixel": 3.0, "s_pos": 3.0, "s_color": 3.0, "s_size": 3.0,
    "s_hallu": 3.0, "s_omiss": 3.0,
}
_TOTAL = sum(_WEIGHTS.values())  # 42.0


def _clip_box(bbox, w, h):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _safe_roi(img, bbox, w, h):
    x1, y1, x2, y2 = _clip_box(bbox, w, h)
    roi = img[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


def _to_gray(img):
    if img.ndim == 2:
        return img
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@register_plugin(StageType.QUALITY_CHECKER, "competition")
class CompetitionQualityCheckerPlugin(IQualityCheckerPlugin):
    """14-dimension self-assessment aligned with CCL2026-Eval."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._ocr_verify = os.environ.get("QUALITY_OCR_VERIFY", "0") == "1"
        self._ocr = None  # lazy RapidOCR for re-read verification

    # -- public entry -----------------------------------------------------

    def check(self, original: np.ndarray, result: np.ndarray,
              regions: List[TextRegion]) -> Dict[str, float]:
        c: Dict[str, float] = {}
        h, w = original.shape[:2]
        mask = self._bg_mask(h, w, regions)

        # Source-image preservation (s_)
        c["s_pixel"] = self._s_pixel(original, result, mask)
        c["s_color"] = self._s_color(original, result, mask)
        c["s_size"] = self._s_size(original, result)
        c["s_pos"] = self._s_pos(original, result, mask)
        c["s_hallu"] = self._s_hallu(original, result, mask)
        c["s_omiss"] = self._s_omiss(original, result, mask)

        # Translation quality (t_)
        translated = [r for r in regions
                      if r.is_translatable and r.translated_text]
        c["t_omiss"] = self._t_omiss(result, translated, w, h)
        c["t_pos"] = self._t_pos(result, translated, w, h)
        c["t_layout"] = self._t_layout(result, translated, w, h)
        c["t_size"] = self._t_size(result, translated, w, h)
        c["t_color"] = self._t_color(result, translated, w, h)
        c["t_font"] = self._t_font(result, translated, w, h, original)

        if self._ocr_verify:
            c["t_pixel"], c["t_hallu"] = self._t_ocr(result, translated, w, h)
        else:
            c["t_hallu"] = self._t_hallu_no_ocr(result, translated, w, h)
            c["t_pixel"] = 0.0

        c["score"] = self._aggregate(c)
        return c

    # -- background mask --------------------------------------------------

    def _bg_mask(self, h, w, regions):
        mask = np.ones((h, w), dtype=bool)
        for r in regions:
            x1, y1, x2, y2 = _clip_box(r.bbox, w, h)
            mask[y1:y2, x1:x2] = False
        return mask

    # -- source preservation (s_) ----------------------------------------

    def _s_pixel(self, original, result, mask):
        try:
            if not mask.any():
                return 1.0
            diff = np.abs(original.astype(np.float32) - result.astype(np.float32))
            if diff.ndim == 3:
                diff = diff.mean(axis=2)
            return float(max(0.0, 1.0 - diff[mask].mean() / 255.0))
        except Exception:
            return 0.5

    def _s_color(self, original, result, mask):
        try:
            if not mask.any():
                return 1.0
            if original.ndim == 3:
                o = original[mask].astype(np.float32).mean(axis=0)
                r = result[mask].astype(np.float32).mean(axis=0)
            else:
                o = np.array([original[mask].mean()], dtype=np.float32)
                r = np.array([result[mask].mean()], dtype=np.float32)
            dist = float(np.linalg.norm(o - r))
            max_dist = 255.0 * np.sqrt(len(o))
            return float(max(0.0, 1.0 - dist / max_dist))
        except Exception:
            return 0.5

    def _s_size(self, original, result):
        return 1.0 if original.shape[:2] == result.shape[:2] else 0.0

    def _s_pos(self, original, result, mask):
        # Structural preservation via gradient similarity on background.
        try:
            if not mask.any():
                return 1.0
            import cv2
            go = cv2.Laplacian(_to_gray(original), cv2.CV_32F)
            gr = cv2.Laplacian(_to_gray(result), cv2.CV_32F)
            d = np.abs(go - gr)[mask]
            return float(max(0.0, 1.0 - d.mean() / 255.0))
        except Exception:
            return 0.5

    def _s_hallu(self, original, result, mask):
        # Artifact fraction: bg pixels that changed a lot look like inpaint
        # hallucination / bleeding outside text boxes.
        try:
            if not mask.any():
                return 1.0
            diff = np.abs(original.astype(np.float32) - result.astype(np.float32))
            if diff.ndim == 3:
                diff = diff.mean(axis=2)
            frac = float((diff[mask] > 30).mean())
            return float(max(0.0, 1.0 - frac))
        except Exception:
            return 0.5

    def _s_omiss(self, original, result, mask):
        # Content loss: bg pixels that became near-blank while original was not.
        try:
            if not mask.any():
                return 1.0
            r = _to_gray(result)[mask].astype(np.float32)
            o = _to_gray(original)[mask].astype(np.float32)
            became_blank = ((r > 240) | (r < 15)) & ((o <= 240) & (o >= 15))
            frac = float(became_blank.mean())
            return float(max(0.0, 1.0 - frac))
        except Exception:
            return 0.5

    # -- text geometry (t_) ----------------------------------------------

    def _text_pixels(self, roi_gray):
        """Return a boolean mask of foreground (text) pixels via Otsu."""
        import cv2
        if roi_gray.size == 0 or roi_gray.std() < 5:
            return np.zeros_like(roi_gray, dtype=bool)
        _, bw = cv2.threshold(roi_gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Text is the minority class (typical for text-on-background).
        fg = bw == 0
        if fg.mean() > 0.5:
            fg = ~fg
        return fg

    def _t_omiss(self, result, translated, w, h):
        if not translated:
            return 1.0
        g = _to_gray(result)
        present = 0
        for r in translated:
            roi, _ = _safe_roi(g, r.bbox, w, h)
            if roi.size > 0 and roi.std() > 10:
                present += 1
        return float(present / len(translated))

    def _t_pos(self, result, translated, w, h):
        # Centroid of rendered text should sit near box center.
        if not translated:
            return 1.0
        g = _to_gray(result)
        scores = []
        for r in translated:
            roi, (x1, y1, x2, y2) = _safe_roi(g, r.bbox, w, h)
            if roi.size == 0:
                continue
            fg = self._text_pixels(roi)
            if not fg.any():
                scores.append(0.0)
                continue
            ys, xs = np.nonzero(fg)
            cx, cy = xs.mean() + x1, ys.mean() + y1
            bx, by = (x1 + x2) / 2, (y1 + y2) / 2
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            off = np.sqrt(((cx - bx) / bw) ** 2 + ((cy - by) / bh) ** 2)
            scores.append(float(max(0.0, 1.0 - off)))
        return float(np.mean(scores)) if scores else 0.0

    def _t_layout(self, result, translated, w, h):
        # Layout fit: rendered text bbox should cover the box well (no overflow,
        # not tiny/cramped). Score by min(width/height coverage) capped at 1.0.
        if not translated:
            return 1.0
        g = _to_gray(result)
        scores = []
        for r in translated:
            roi, _ = _safe_roi(g, r.bbox, w, h)
            if roi.size == 0:
                continue
            fg = self._text_pixels(roi)
            if not fg.any():
                scores.append(0.0)
                continue
            ys, xs = np.nonzero(fg)
            rh, rw = roi.shape[:2]
            tw = (xs.max() - xs.min() + 1) / max(1, rw)
            th = (ys.max() - ys.min() + 1) / max(1, rh)
            scores.append(float(min(tw, th, 1.0)))
        return float(np.mean(scores)) if scores else 0.0

    def _t_size(self, result, translated, w, h):
        # Fill ratio: rendered text area / box area. Too sparse = unreadable
        # (the weakest dimension for top teams, 2.364/3.0); too dense = overflow.
        if not translated:
            return 1.0
        g = _to_gray(result)
        scores = []
        for r in translated:
            roi, _ = _safe_roi(g, r.bbox, w, h)
            if roi.size == 0:
                continue
            fg = self._text_pixels(roi)
            fill = fg.sum() / fg.size
            # Ideal ~0.18-0.45; gaussian-style scoring around 0.28.
            ideal = 0.28
            sigma = 0.18
            s = np.exp(-0.5 * ((fill - ideal) / sigma) ** 2)
            scores.append(float(s))
        return float(np.mean(scores)) if scores else 0.0

    def _t_color(self, result, translated, w, h):
        # Rendered text color should match style_info['color'] if available.
        if not translated:
            return 1.0
        scores = []
        for r in translated:
            roi, _ = _safe_roi(result, r.bbox, w, h)
            if roi.size == 0 or "color" not in (r.style_info or {}):
                continue
            fg = self._text_pixels(_to_gray(roi))
            if not fg.any():
                scores.append(0.0)
                continue
            tex_color = roi[fg].astype(np.float32).mean(axis=0)
            ref = np.array(r.style_info["color"], dtype=np.float32)
            dist = float(np.linalg.norm(tex_color - ref))
            scores.append(float(max(0.0, 1.0 - dist / 441.7)))
        return float(np.mean(scores)) if scores else 0.5

    def _t_font(self, result, translated, w, h, original):
        # Stroke width via distance transform; compare rendered vs original.
        if not translated:
            return 1.0
        import cv2
        scores = []
        for r in translated:
            r_roi, _ = _safe_roi(_to_gray(result), r.bbox, w, h)
            o_roi, _ = _safe_roi(_to_gray(original), r.bbox, w, h)
            if r_roi.size == 0 or o_roi.size == 0:
                continue
            r_w = self._stroke_width(r_roi)
            o_w = self._stroke_width(o_roi)
            if o_w <= 0:
                scores.append(0.5)
                continue
            ratio = min(r_w, o_w) / max(r_w, o_w)
            scores.append(float(ratio))
        return float(np.mean(scores)) if scores else 0.5

    def _stroke_width(self, gray_roi):
        try:
            import cv2
            fg = self._text_pixels(gray_roi)
            if not fg.any():
                return 0.0
            dist = cv2.distanceTransform(fg.astype(np.uint8) * 255,
                                         cv2.DIST_L2, 3)
            return float(np.median(dist[dist > 0])) if (dist > 0).any() else 0.0
        except Exception:
            return 0.0

    # -- OCR re-read verification (optional) ------------------------------

    def _ensure_ocr(self):
        if self._ocr is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._ocr = RapidOCR()
                logger.info("CompetitionQualityChecker: RapidOCR ready for verify")
            except Exception as e:
                logger.warning("OCR verify unavailable: %s", e)
                self._ocr = False
        return self._ocr

    def _t_ocr(self, result, translated, w, h):
        ocr = self._ensure_ocr()
        if not ocr or not translated:
            return 0.0, self._t_hallu_no_ocr(result, translated, w, h)
        sims = []
        for r in translated:
            x1, y1, x2, y2 = _clip_box(r.bbox, w, h)
            crop = result[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                res, _ = ocr(crop)
            except Exception:
                res = None
            read = "".join(t for _, t, _ in res) if res else ""
            if not read and not r.translated_text:
                sims.append(1.0)
                continue
            if not read or not r.translated_text:
                sims.append(0.0)
                continue
            sims.append(self._edit_sim(read, r.translated_text))
        pixel = float(np.mean(sims)) if sims else 0.0
        # t_hallu: readable & matches expected text => low hallucination.
        hallu = pixel
        return pixel, hallu

    def _t_hallu_no_ocr(self, result, translated, w, h):
        # Heuristic: rendered region should have text-like structure (bimodal
        # intensity), not noise (uniform) or solid block.
        if not translated:
            return 1.0
        g = _to_gray(result)
        scores = []
        for r in translated:
            roi, _ = _safe_roi(g, r.bbox, w, h)
            if roi.size == 0:
                continue
            std = roi.std()
            # text-like: moderate std (10-90); noise/solid: extreme or ~0
            if std < 10:
                scores.append(0.0)
            elif std > 110:
                scores.append(0.3)
            else:
                scores.append(float(min(1.0, std / 40.0)))
        return float(np.mean(scores)) if scores else 0.5

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _edit_sim(a, b):
        """Normalized edit similarity in [0,1]."""
        a = (a or "").strip().lower()
        b = (b or "").strip().lower()
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            nd = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                nd[j] = min(dp[j] + 1, nd[j - 1] + 1, dp[j - 1] + cost)
            dp = nd
        dist = dp[n]
        return float(1.0 - dist / max(m, n))

    @staticmethod
    def _aggregate(c):
        total = 0.0
        for k, w in _WEIGHTS.items():
            total += c.get(k, 0.0) * w
        return float(total / _TOTAL)
