"""Shared environment loading utilities."""

import importlib
from pathlib import Path
from typing import Optional


def load_env(env_file: Optional[str] = None) -> None:
    """Load environment variables from .env file if python-dotenv is available."""
    try:
        dotenv_mod = importlib.import_module("dotenv")
        loader = getattr(dotenv_mod, "load_dotenv", None)
        if callable(loader):
            if env_file is not None:
                loader(Path(env_file))
            else:
                loader()
    except Exception:
        pass
