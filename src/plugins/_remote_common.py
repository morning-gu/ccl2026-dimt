"""Shared utilities for remote plugin clients (renderers and erasers).

Provides image base64 <-> numpy codec, TextRegion serialisation, and a
mixin class that manages the httpx client lifecycle.  Both RemoteRendererBase
and RemoteEraserBase use this to avoid duplication.
"""
import base64
import logging
import os
from typing import List, Optional

import httpx
import numpy as np

from common.config import PipelineConfig
from common.selective_translator import TextRegion

logger = logging.getLogger(__name__)


def encode_image(image: np.ndarray) -> str:
    """Encode a BGR numpy array to a base64 PNG string."""
    import cv2
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode image as PNG")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def decode_image(b64: str) -> np.ndarray:
    """Decode a base64 PNG string to a BGR numpy array."""
    import cv2
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from base64")
    return img


def serialize_region(r: TextRegion) -> dict:
    """Serialise a TextRegion to a JSON-friendly dict."""
    return {
        "text": r.text,
        "bbox": list(r.bbox),
        "confidence": r.confidence,
        "is_translatable": r.is_translatable,
        "preserve_reason": r.preserve_reason,
        "style_info": r.style_info,
        "translated_text": r.translated_text,
        "region_type": r.region_type,
        "bbox_poly": r.bbox_poly,
        "angle": r.angle,
    }


class RemotePluginMixin:
    """Mixin managing httpx client init and POST transport.

    Subclasses set _api_url_env, _plugin_name, _default_timeout, then call
    _init_client() and _post() from their interface method.
    """

    _api_url_env: str = ""
    _plugin_name: str = ""
    _default_timeout: float = 120.0

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._client: Optional[httpx.Client] = None
        self._api_url: str = ""

    def _init_client(self):
        url = os.environ.get(self._api_url_env, "")
        if not url:
            raise RuntimeError(
                f"{self._api_url_env} is not set. Start the {self._plugin_name} "
                f"server and set this env var to its endpoint URL."
            )
        self._api_url = url
        timeout_env = f"{self._plugin_name.upper()}_API_TIMEOUT"
        timeout = float(os.environ.get(timeout_env, self._default_timeout))
        self._client = httpx.Client(timeout=timeout)
        logger.info("%s remote plugin initialised (API: %s, timeout: %.0fs)",
                     self._plugin_name, url, timeout)

    def _post(self, payload: dict) -> np.ndarray:
        """POST a JSON payload to the API and return the decoded result image."""
        try:
            resp = self._client.post(self._api_url, json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = e.response.text[:500] if e.response.text else ""
            raise RuntimeError(
                f"{self._plugin_name} server returned HTTP {e.response.status_code}: {detail}"
            ) from e
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"Cannot connect to {self._plugin_name} server at {self._api_url}: {e}"
            ) from e
        except httpx.TimeoutException:
            raise RuntimeError(
                f"{self._plugin_name} server timed out. "
                f"Increase {self._plugin_name.upper()}_API_TIMEOUT if needed."
            )
        return decode_image(resp.json()["image"])
