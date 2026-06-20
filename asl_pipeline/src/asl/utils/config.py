from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    def __getattr__(self, item: str) -> Any:
        if item in self:
            value = self[item]
            return Config(value) if isinstance(value, dict) else value
        raise AttributeError(item)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get(self, key, default=None):
        value = super().get(key, default)
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw)


def load_class_names(path: str | Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]
