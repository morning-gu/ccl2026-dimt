"""EasyText rendering plugin (remote client).

Calls the EasyText API server (servers/easytext_server.py) via HTTP.
All model loading and inference happen server-side; this plugin only
handles serialisation and transport.

Configuration:
  EASYTEXT_API_URL      URL of the EasyText server render endpoint
                        (e.g. http://localhost:8001/render)
  EASYTEXT_API_TIMEOUT  Request timeout in seconds (default: 120)
"""
import logging

from interfaces.base import StageType
from plugins.registry import register_plugin
from ._remote_base import RemoteRendererBase

logger = logging.getLogger(__name__)


@register_plugin(StageType.RENDERER, "easytext")
class EasyTextRendererPlugin(RemoteRendererBase):
    """EasyText multilingual text rendering via remote FLUX DiT + LoRA API."""

    _api_url_env = "EASYTEXT_API_URL"
    _plugin_name = "easytext"
    _default_timeout = 120.0
