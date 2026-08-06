"""STRNet stroke-level erasure plugin (remote client).

Calls the STRNet API server (servers/strokenet_server.py) via HTTP.

Configuration:
  STROKENET_API_URL      URL of the STRNet server erase endpoint
                         (e.g. http://localhost:8011/erase)
  STROKENET_API_TIMEOUT  Request timeout in seconds (default: 120)
"""
import logging

from interfaces.base import StageType
from plugins.registry import register_plugin
from ._remote_eraser_base import RemoteEraserBase

logger = logging.getLogger(__name__)


@register_plugin(StageType.ERASER, "strokenet")
class StrokenetEraserPlugin(RemoteEraserBase):
    """STRNet stroke-level erasure via remote API."""

    _api_url_env = "STROKENET_API_URL"
    _plugin_name = "strokenet"
    _default_timeout = 120.0
