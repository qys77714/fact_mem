"""Post-ingest memory fusion (relational bundles → single fused rows).

Heavy imports (faiss / LocalFaiss) are lazy so ``memory.fusion.bundle_prompt_render`` works without faiss.
"""

from __future__ import annotations

__all__ = [
    "fuse_local_faiss_database",
    "is_local_faiss_database_fused",
    "list_depth_one_leaf_star_packages",
    "list_disjoint_depth_one_partition_packages",
    "list_fusion_packages",
    "list_multimember_depth_one_partition_wave",
    "list_whole_tree_fusion_packages",
    "render_fusion_user_prompt",
]


def __getattr__(name: str):
    if name == "render_fusion_user_prompt":
        from .bundle_prompt_render import render_fusion_user_prompt

        return render_fusion_user_prompt
    if name in (
        "fuse_local_faiss_database",
        "is_local_faiss_database_fused",
        "list_depth_one_leaf_star_packages",
        "list_disjoint_depth_one_partition_packages",
        "list_fusion_packages",
        "list_multimember_depth_one_partition_wave",
        "list_whole_tree_fusion_packages",
    ):
        from . import lme_bundle_fusion as m

        return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
