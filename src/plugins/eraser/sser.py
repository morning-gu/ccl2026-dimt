"""Self-supervised Text Erasing plugin (remote client).

Calls the SSER API server (servers/sser_server.py) via HTTP.

Configuration:
  SSER_API_URL      URL of the SSER server erase endpoint
                    (e.g. http://localhost:8013/erase)
  SSER_API_TIMEOUT  Request timeout in seconds (default: 120)
"""
import logging

from interfaces.base import StageType
from plugins.registry import register_plugin
from ._remote_eraser_base import RemoteEraserBase

logger = logging.getLogger(__name__)


@register_plugin(StageType.ERASER, "sser")
class SSEREraserPlugin(RemoteEraserBase):
    """Self-supervised Text Erasing via STRnet2 remote API."""

    _api_url_env = "SSER_API_URL"
    _plugin_name = "sser"
    _default_timeout = 120.0
