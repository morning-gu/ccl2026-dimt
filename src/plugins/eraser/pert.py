"""PERT scene text removal erasure plugin (remote client).

Calls the PERT API server (servers/pert_server.py) via HTTP.

Configuration:
  PERT_API_URL      URL of the PERT server erase endpoint
                    (e.g. http://localhost:8012/erase)
  PERT_API_TIMEOUT  Request timeout in seconds (default: 120)
"""
import logging

from interfaces.base import StageType
from plugins.registry import register_plugin
from ._remote_eraser_base import RemoteEraserBase

logger = logging.getLogger(__name__)


@register_plugin(StageType.ERASER, "pert")
class PERTEraserPlugin(RemoteEraserBase):
    """PERT scene text removal via remote API."""

    _api_url_env = "PERT_API_URL"
    _plugin_name = "pert"
    _default_timeout = 120.0
