"""Utilities for reading/writing YAML while preserving comments."""

from pathlib import Path
from ruamel.yaml import YAML

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.width = 4096  # prevent line wrapping


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return _yaml.load(f)


def save_yaml(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        _yaml.dump(data, f)


def update_yaml(path: Path, updates: dict):
    """Load yaml, apply nested updates, save preserving comments.

    updates is a dict of {section: {key: value}}, e.g.:
        {"extrinsics": {"dk": -5}}
    """
    data = load_yaml(path)
    for section, kvs in updates.items():
        if section not in data:
            data[section] = {}
        for key, val in kvs.items():
            data[section][key] = val
    save_yaml(path, data)
