"""Base class for remote renderer plugins that call an HTTP API."""
import logging
from typing import List, Optional

import numpy as np

from interfaces.renderer import IRendererPlugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion
from plugins._remote_common import RemotePluginMixin, encode_image, serialize_region

logger = logging.getLogger(__name__)


class RemoteRendererBase(IRendererPlugin, RemotePluginMixin):
    """Base class for remote renderer plugins.

    Subclasses set _api_url_env and _plugin_name.
    """

    def render(
        self,
        image: np.ndarray,
        regions: List[TextRegion],
        style_reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not regions:
            return image
        render_regions = [r for r in regions if r.translated_text]
        if not render_regions:
            return image
        if self._client is None:
            self._init_client()

        payload = {
            "image": encode_image(image),
            "regions": [serialize_region(r) for r in render_regions],
            "style_reference": encode_image(style_reference) if style_reference is not None else None,
        }
        return self._post(payload)
