"""PowerPaint erasure plugin (remote client).

Calls the PowerPaint API server (servers/powerpaint_server.py) via HTTP.
PowerPaint uses prompt-augmented diffusion training for SOTA inpainting
quality.  The server handles the "Remove the masked area" prompt, color
correction, and feathered blending.

Configuration:
  POWERPAINT_API_URL      URL of the PowerPaint server erase endpoint
                          (e.g. http://localhost:8014/erase)
  POWERPAINT_API_TIMEOUT  Request timeout in seconds (default: 120)
"""
import logging

from interfaces.base import StageType
from plugins.registry import register_plugin
from ._remote_eraser_base import RemoteEraserBase

logger = logging.getLogger(__name__)


@register_plugin(StageType.ERASER, "powerpaint")
class PowerPaintEraserPlugin(RemoteEraserBase):
    """PowerPaint prompt-augmented diffusion inpainting via remote API.

    Server-side, PowerPaint uses the "Remove the masked area" prompt to
    generate seamless background restoration.  Post-processing (color
    correction + feathered compositing) matches the LaMa plugin for
    clean boundary transitions.
    """

    _api_url_env = "POWERPAINT_API_URL"
    _plugin_name = "powerpaint"
    _default_timeout = 120.0
