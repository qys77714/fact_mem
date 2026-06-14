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
    rerank_qwen3_vllm: bool = False,
    rerank_qwen3_vllm_base_url: str | None = None,
    rerank_qwen3_vllm_api_key: str | None = None,
    rerank_qwen3_vllm_model: str = "Qwen3-Reranker-0.6B",
    rerank_qwen3_vllm_timeout_s: float = 120.0,
    rerank_top_k: int | None = None,
    **kwargs,
) -> BaseMemorySystem:
    """Return a memory system for ``pipeline_lme_generate`` (prebuilt dense retrieval only)."""
    from .baselines.lme_prebuilt import LmePrebuiltMemorySystem

    if method_name == "lme_prebuilt":
        return LmePrebuiltMemorySystem(
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
            rerank_qwen3_vllm=rerank_qwen3_vllm,
            rerank_qwen3_vllm_base_url=rerank_qwen3_vllm_base_url,
            rerank_qwen3_vllm_api_key=rerank_qwen3_vllm_api_key,
            rerank_qwen3_vllm_model=rerank_qwen3_vllm_model,
            rerank_qwen3_vllm_timeout_s=rerank_qwen3_vllm_timeout_s,
            rerank_top_k=rerank_top_k,
            **kwargs,
        )
    raise ValueError(f"Unknown memory method: {method_name}")


def __getattr__(name: str):
    if name == "LmePrebuiltMemorySystem":
        from .baselines.lme_prebuilt import LmePrebuiltMemorySystem

        return LmePrebuiltMemorySystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseMemorySystem",
    "RetrievedMemory",
    "get_memory_system",
    "LmePrebuiltMemorySystem",
]
