"""Shared utilities for renderer API servers.

Self-contained module with no imports from the pipeline project.  Provides
image base64 <-> numpy codec, region serialization models, mask building,
and common FastAPI request/response schemas shared by all servers.
"""
import base64
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Image codec: base64 PNG <-> numpy BGR array
# ------------------------------------------------------------------

def encode_image(image: np.ndarray) -> str:
    """Encode a BGR numpy array to a base64 PNG string."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode image as PNG")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def decode_image(b64: str) -> np.ndarray:
    """Decode a base64 PNG string to a BGR numpy array."""
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from base64")
    return img


# ------------------------------------------------------------------
# Region data model (mirrors pipeline TextRegion, standalone)
# ------------------------------------------------------------------

@dataclass
class RegionData:
    """Mirror of common.selective_translator.TextRegion for server-side use."""
    text: str
    bbox: List[float]
    confidence: float = 1.0
    is_translatable: bool = True
    preserve_reason: str = ""
    style_info: dict = field(default_factory=dict)
    translated_text: str = ""
    region_type: str = "text"
    bbox_poly: Optional[List[List[float]]] = None
    angle: float = 0.0


# ------------------------------------------------------------------
# Pydantic request/response schemas
# ------------------------------------------------------------------

class RegionModel(BaseModel):
    text: str
    bbox: List[float]
    confidence: float = 1.0
    is_translatable: bool = True
    preserve_reason: str = ""
    style_info: dict = Field(default_factory=dict)
    translated_text: str = ""
    region_type: str = "text"
    bbox_poly: Optional[List[List[float]]] = None
    angle: float = 0.0

    def to_data(self) -> RegionData:
        return RegionData(
            text=self.text,
            bbox=self.bbox,
            confidence=self.confidence,
            is_translatable=self.is_translatable,
            preserve_reason=self.preserve_reason,
            style_info=self.style_info,
            translated_text=self.translated_text,
            region_type=self.region_type,
            bbox_poly=self.bbox_poly,
            angle=self.angle,
        )


class RenderRequest(BaseModel):
    image: str  # base64 PNG
    regions: List[RegionModel]
    style_reference: Optional[str] = None  # base64 PNG or null


class RenderResponse(BaseModel):
    image: str  # base64 PNG


# ------------------------------------------------------------------
# Convenience: decode a full request into (image, regions, style_ref)
# ------------------------------------------------------------------

def decode_request(req: RenderRequest):
    """Decode a RenderRequest into (image_bgr, regions, style_ref_bgr_or_None)."""
    image = decode_image(req.image)
    regions = [r.to_data() for r in req.regions]
    style_ref = decode_image(req.style_reference) if req.style_reference else None
    return image, regions, style_ref


# ------------------------------------------------------------------
# Eraser request schema
# ------------------------------------------------------------------

class EraseRequest(BaseModel):
    image: str  # base64 PNG
    regions: List[RegionModel]
    dilate_pixels: int = 0


def decode_erase_request(req: EraseRequest):
    """Decode an EraseRequest into (image_bgr, regions, dilate_pixels)."""
    image = decode_image(req.image)
    regions = [r.to_data() for r in req.regions]
    return image, regions, req.dilate_pixels


# ------------------------------------------------------------------
# Mask building (standalone version of eraser _common.build_mask)
# ------------------------------------------------------------------

def text_pixel_mask(roi, region: RegionData, ox, oy, keep_thin_lines: bool = False):
    """Build pixel-level text mask (standalone, mirrors eraser _common)."""
    try:
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
            if fill_ratio < 0.1:
                continue
            if not keep_thin_lines and (area < 10 or is_thin_line):
                continue
            filtered[labels == i] = 255
        tmask = filtered

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        tmask = cv2.dilate(tmask, k, iterations=2)
        return tmask
    except Exception:
        return None


def build_mask(shape, regions: List[RegionData], dilate: int, image=None,
               keep_thin_lines: bool = False) -> np.ndarray:
    """Build erasure mask covering text regions (standalone, mirrors eraser _common)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in regions:
        x1 = max(0, int(region.bbox[0]) - dilate)
        y1 = max(0, int(region.bbox[1]) - dilate)
        x2 = min(w, int(region.bbox[2]) + dilate)
        y2 = min(h, int(region.bbox[3]) + dilate)
        if image is not None and (x2 - x1) > 2 and (y2 - y1) > 2:
            roi = image[y1:y2, x1:x2]
            text_mask = text_pixel_mask(roi, region, x1, y1, keep_thin_lines=keep_thin_lines)
            if text_mask is not None:
                mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], text_mask)
                continue
        mask[y1:y2, x1:x2] = 255
    return mask
