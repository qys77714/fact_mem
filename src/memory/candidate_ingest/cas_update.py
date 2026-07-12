"""Metadata helpers for relation_decision ingest."""

from __future__ import annotations

from typing import Any


def metadata_for_new_primary(
    metadata_base: dict[str, Any],
    m_new: str,
    *,
    lme_update_method: str = "relation_decision",
) -> dict[str, Any]:
    """Build metadata for a new primary row."""
    meta = dict(metadata_base)
    meta["memory_role"] = "primary"
    meta["lme_update_method"] = lme_update_method
    return meta
