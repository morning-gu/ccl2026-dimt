"""AnyText2 rendering plugin (remote client).

Calls the AnyText2 API server (servers/anytext2_server.py) via HTTP.
All model loading and inference happen server-side; this plugin only
handles serialisation and transport.

Configuration:
  ANYTEXT2_API_URL      URL of the AnyText2 server render endpoint
                        (e.g. http://localhost:8003/render)
  ANYTEXT2_API_TIMEOUT  Request timeout in seconds (default: 120)
"""
import logging

from interfaces.base import StageType
from plugins.registry import register_plugin
from ._remote_base import RemoteRendererBase

logger = logging.getLogger(__name__)


@register_plugin(StageType.RENDERER, "anytext2")
class AnyText2RendererPlugin(RemoteRendererBase):
    """AnyText2 rendering via remote API."""

    _api_url_env = "ANYTEXT2_API_URL"
    _plugin_name = "anytext2"
    _default_timeout = 120.0
