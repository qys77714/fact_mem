from .base import BaseMemorySystem, RetrievedMemory


def get_memory_system(
    method_name: str,
    embed_model_name: str,
    embed_client=None,
    database_root: str = None,
    *,
    use_hybrid_retrieval: bool = False,
    hybrid_dense_weight: float = 0.5,
    hybrid_bm25_weight: float = 0.5,
    hybrid_pool_mult: int = 4,
    hybrid_full_corpus_pool: bool = False,
    unfused_rank_database_root: str | None = None,
    language: str = "en",
    **kwargs,
) -> BaseMemorySystem:
    """Return a memory system for the answer stage (prebuilt dense retrieval only)."""
    from .prebuilt import PrebuiltMemorySystem

    if method_name == "prebuilt":
        return PrebuiltMemorySystem(
            embed_model_name=embed_model_name,
            embed_client=embed_client,
            database_root=database_root,
            use_hybrid_retrieval=use_hybrid_retrieval,
            hybrid_dense_weight=hybrid_dense_weight,
            hybrid_bm25_weight=hybrid_bm25_weight,
            hybrid_pool_mult=hybrid_pool_mult,
            hybrid_full_corpus_pool=hybrid_full_corpus_pool,
            unfused_rank_database_root=unfused_rank_database_root,
            language=language,
            **kwargs,
        )
    raise ValueError(f"Unknown memory method: {method_name}")


def __getattr__(name: str):
    if name == "PrebuiltMemorySystem":
        from .prebuilt import PrebuiltMemorySystem

        return PrebuiltMemorySystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseMemorySystem",
    "RetrievedMemory",
    "get_memory_system",
    "PrebuiltMemorySystem",
]
