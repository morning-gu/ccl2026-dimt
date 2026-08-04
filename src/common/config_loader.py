"""YAML config loader with extends inheritance.

Loads PipelineConfig from YAML files. Supports `extends` field for
configuration inheritance with deep merge of `plugins` dicts.

Priority: CLI args > env vars > YAML > code defaults.
"""
from pathlib import Path
from typing import Any, Dict

import yaml

from .config import PipelineConfig, load_config_from_env


def load_config_from_yaml(yaml_path: str) -> PipelineConfig:
    """Load config from a YAML file, processing extends inheritance.

    Then applies env var overrides (non-destructive to CLI-set values).
    """
    path = Path(yaml_path).resolve()
    data = _load_with_inheritance(path)
    cfg = PipelineConfig()
    _apply_dict_to_config(cfg, data)
    cfg = load_config_from_env(cfg)
    return cfg


def _load_with_inheritance(path: Path) -> dict:
    """Load YAML, recursively processing the extends field.

    Child values override parent; plugins dicts are deep-merged.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    if parent:
        parent_data = _load_with_inheritance(path.parent / parent)
        # Deep merge plugins: parent first, child overrides
        child_plugins = data.pop("plugins", None)
        parent_data.update(data)
        if child_plugins is not None:
            parent_data.setdefault("plugins", {}).update(child_plugins)
        return parent_data
    return data


def _apply_dict_to_config(cfg: PipelineConfig, data: dict):
    """Apply a dict of config values to a PipelineConfig dataclass."""
    for key, value in data.items():
        if key == "plugins":
            cfg.plugins = {str(k): str(v) for k, v in value.items()}
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
