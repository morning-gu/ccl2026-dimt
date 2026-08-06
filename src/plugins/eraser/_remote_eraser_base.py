"""Base class for remote eraser plugins that call an HTTP API."""
import logging
from typing import List

import numpy as np

from interfaces.eraser import IEraserPlugin
from common.config import PipelineConfig
from common.selective_translator import TextRegion
from plugins._remote_common import RemotePluginMixin, encode_image, serialize_region

logger = logging.getLogger(__name__)


class RemoteEraserBase(IEraserPlugin, RemotePluginMixin):
    """Base class for remote eraser plugins.

    Subclasses set _api_url_env and _plugin_name.
    """

    def erase(
        self,
        image: np.ndarray,
        regions: List[TextRegion],
        dilate_pixels: int = 0,
    ) -> np.ndarray:
        if not regions:
            return image
        if self._client is None:
            self._init_client()

        payload = {
            "image": encode_image(image),
            "regions": [serialize_region(r) for r in regions],
            "dilate_pixels": dilate_pixels,
        }
        return self._post(payload)
