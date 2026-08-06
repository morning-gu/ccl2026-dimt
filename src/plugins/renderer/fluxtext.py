"""FluxText rendering plugin (remote client).

Calls the FluxText API server (servers/fluxtext_server.py) via HTTP.
All model loading and inference happen server-side; this plugin only
handles serialisation and transport.

Configuration:
  FLUXTEXT_API_URL      URL of the FluxText server render endpoint
                        (e.g. http://localhost:8002/render)
  FLUXTEXT_API_TIMEOUT  Request timeout in seconds (default: 120)
"""
import logging

from interfaces.base import StageType
from plugins.registry import register_plugin
from ._remote_base import RemoteRendererBase

logger = logging.getLogger(__name__)


@register_plugin(StageType.RENDERER, "fluxtext")
class FluxTextRendererPlugin(RemoteRendererBase):
    """FLUX-Text scene text editing via remote API.

    Unlike PIL/AnyText2 which render onto a pre-erased image, FluxText takes
    the *original* image and performs inpainting-based text replacement in one
    pass.  When this plugin is active, set the pipeline eraser to ``noop``
    so the original image is passed through unchanged.
    """

    _api_url_env = "FLUXTEXT_API_URL"
    _plugin_name = "fluxtext"
    _default_timeout = 120.0
