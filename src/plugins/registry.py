"""Plugin registry: register, lookup, and instantiate stage plugins."""
from typing import Type, Dict, Any
from interfaces.base import StageType, _STAGE_INTERFACES
from common.config import PipelineConfig


class PluginRegistry:
    """Central registry for pipeline stage plugins."""

    def __init__(self):
        self._plugins: Dict[StageType, Dict[str, Type]] = {}

    def register(self, stage: StageType, name: str, plugin_cls: Type):
        """Register a plugin implementation class for a stage."""
        interface = _STAGE_INTERFACES.get(stage)
        if interface is None:
            raise ValueError(f"Unknown stage type: {stage}")
        if not issubclass(plugin_cls, interface):
            raise TypeError(
                f"{plugin_cls.__name__} must implement {interface.__name__}"
            )
        self._plugins.setdefault(stage, {})[name] = plugin_cls

    def create(self, stage: StageType, name: str, config: PipelineConfig) -> Any:
        """Factory: create a plugin instance by name."""
        plugin_cls = self._plugins.get(stage, {}).get(name)
        if plugin_cls is None:
            available = self.available(stage)
            raise ValueError(
                f"No plugin registered for stage={stage.value!r}, name={name!r}. "
                f"Available: {available}"
            )
        return plugin_cls(config)

    def available(self, stage: StageType) -> list:
        """List all registered plugin names for a stage."""
        return list(self._plugins.get(stage, {}).keys())

    def is_registered(self, stage: StageType, name: str) -> bool:
        return name in self._plugins.get(stage, {})


# Global registry instance
registry = PluginRegistry()


def register_plugin(stage: StageType, name: str):
    """Decorator: register a plugin class to the global registry."""
    def decorator(cls):
        registry.register(stage, name, cls)
        return cls
    return decorator
