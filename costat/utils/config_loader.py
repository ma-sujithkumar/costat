"""Loads the single project configuration file and the JSON registries.

Everything tunable lives in config/config.ini and the JSON maps it points at.
Modules ask this loader for values rather than hardcoding them, which keeps the
sources free of magic numbers and paths.
"""

import configparser
import json
import os
from typing import Any, Dict, List


class ConfigLoader:
    """Reads config.ini (with cross-section interpolation) and JSON registries.

    A single instance is shared across the pipeline. The class exposes typed
    getters so callers never have to remember which value is an int or a list.
    """

    def __init__(self, config_path: str) -> None:
        if not os.path.isfile(config_path):
            raise FileNotFoundError("config.ini not found at: " + config_path)
        self.config_path: str = config_path
        self.parser: configparser.ConfigParser = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
        self.parser.read(config_path)

    def get_str(self, section: str, key: str) -> str:
        return self.parser.get(section, key)

    def get_int(self, section: str, key: str) -> int:
        return self.parser.getint(section, key)

    def get_float(self, section: str, key: str) -> float:
        return self.parser.getfloat(section, key)

    def get_int_list(self, section: str, key: str) -> List[int]:
        raw_value: str = self.parser.get(section, key)
        return [int(item.strip()) for item in raw_value.split(",") if item.strip()]

    def get_str_list(self, section: str, key: str) -> List[str]:
        raw_value: str = self.parser.get(section, key)
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    def get_path(self, key: str) -> str:
        """Return a path from the [paths] section, creating parent dirs lazily."""
        return self.parser.get("paths", key)

    @staticmethod
    def load_json(json_path: str) -> Dict[str, Any]:
        if not os.path.isfile(json_path):
            raise FileNotFoundError("JSON registry not found at: " + json_path)
        with open(json_path, "r", encoding="ascii") as handle:
            return json.load(handle)
