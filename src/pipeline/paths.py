"""Conventions for MemDB / experiment paths (candidate extract vs ingest vs preds)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def safe_model_tag(model_name: str) -> str:
    """Model name safe for directory segments (no path separators or odd chars)."""
    tag = re.sub(r"[/:\\s]+", "_", str(model_name).strip())
    tag = re.sub(r"[^a-zA-Z0-9_.-]+", "", tag)
    return tag or "model"


def default_candidates_dir(
    *,
    benchmark: str,
    extract_model: str,
    suffix: str = "default",
    root: Optional[Path] = None,
) -> Path:
    """
    Shared atomic candidates for all ingest methods:
    ``MemDB/candidates/{benchmark}/{extract_model_tag}/{suffix}/``
    """
    base = root if root is not None else Path("MemDB") / "candidates"
    return base / benchmark.strip() / safe_model_tag(extract_model) / suffix.strip()


def default_ingest_dir(
    *,
    benchmark: str,
    manager_model: str,
    method_name: str,
    root: Optional[Path] = None,
) -> Path:
    """Per-method vector DB: ``MemDB/ingest/{benchmark}/{manager_tag}/{method}/``."""
    base = root if root is not None else Path("MemDB") / "ingest"
    return base / benchmark.strip() / safe_model_tag(manager_model) / method_name.strip()


def default_memory_trace_dir(
    *,
    benchmark: str,
    manager_model: str,
    method_name: str,
    root: Optional[Path] = None,
) -> Path:
    """Trace logs mirroring ingest layout."""
    base = root if root is not None else Path("logs") / "memory_trace" / "ingest"
    return base / benchmark.strip() / safe_model_tag(manager_model) / method_name.strip()
