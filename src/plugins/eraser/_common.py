"""Shared utilities for eraser plugins."""
import numpy as np
from typing import List
from common.selective_translator import TextRegion


def build_mask(shape, regions: List[TextRegion], dilate: int, image=None) -> np.ndarray:
    """Build erasure mask covering text regions. Migrated from TextEraser._build_mask."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in regions:
        x1 = max(0, int(region.bbox[0]) - dilate)
        y1 = max(0, int(region.bbox[1]) - dilate)
        x2 = min(w, int(region.bbox[2]) + dilate)
        y2 = min(h, int(region.bbox[3]) + dilate)
        if image is not None and (x2 - x1) > 2 and (y2 - y1) > 2:
            roi = image[y1:y2, x1:x2]
            text_mask = text_pixel_mask(roi, region, x1, y1)
            if text_mask is not None:
                mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], text_mask)
                continue
        mask[y1:y2, x1:x2] = 255
    return mask


def text_pixel_mask(roi, region, ox, oy):
    """Build pixel-level text mask. Migrated from TextEraser._text_pixel_mask."""
    try:
        import cv2
        rh, rw = roi.shape[:2]
        if rh < 5 or rw < 5:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        poly = getattr(region, "bbox_poly", None)

        if poly and len(poly) >= 4:
            pts = np.array([[(int(p[0]) - ox, int(p[1]) - oy) for p in poly]], dtype=np.int32)
            pmask = np.zeros((rh, rw), dtype=np.uint8)
            cv2.fillPoly(pmask, pts, 255)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            outside = pmask == 0
            if outside.sum() > 4:
                bg_gray = float(np.median(gray[outside]))
            else:
                border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
                bg_gray = float(np.median(border))
            dark_sel = (binary == 0) & (pmask > 0)
            light_sel = (binary == 255) & (pmask > 0)
            dark_mean = float(gray[dark_sel].mean()) if dark_sel.any() else bg_gray
            light_mean = float(gray[light_sel].mean()) if light_sel.any() else bg_gray
            if abs(dark_mean - bg_gray) > abs(light_mean - bg_gray):
                tmask = dark_sel.astype(np.uint8) * 255
            else:
                tmask = light_sel.astype(np.uint8) * 255
        else:
            border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
            bg = float(np.median(border))
            diff = np.abs(gray.astype(np.int16) - bg)
            thresh = max(40, int(np.std(border) * 2))
            tmask = (diff > thresh).astype(np.uint8) * 255

        num, labels, stats, _ = cv2.connectedComponentsWithStats(tmask, connectivity=8)
        filtered = np.zeros_like(tmask)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            fill_ratio = area / max(1, bw * bh)
            min_dim = min(bw, bh)
            max_dim = max(bw, bh)
            is_thin_line = min_dim <= 2 and max_dim > 15
            if area < 10 or fill_ratio < 0.1 or is_thin_line:
                continue
            filtered[labels == i] = 255
        tmask = filtered

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        tmask = cv2.dilate(tmask, k, iterations=2)
        return tmask
    except Exception:
        return None
