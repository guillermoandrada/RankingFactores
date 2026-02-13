"""
Application settings loaded from config.json.
"""

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.json"


def _load_config() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


_config = _load_config()

# Expose as module-level constants (snake_case in JSON → UPPER for Python)
DB_URL = _config["db_url"]
FIXED_COLUMNS = _config["fixed_columns"]
DEFAULT_INPUT_FILE = _config["default_input_file"]
