from .base import BaseMemorySystem, RetrievedMemory
from .baselines.full_context import FullContextMemorySystem
from .baselines.rag import RagMemorySystem
from .baselines.only_query import OnlyQueryMemorySystem
from .amem import AMemMemorySystem
from .mem0 import Mem0MemorySystem

def get_memory_system(
    method_name: str,
    embed_model_name: str,
    embed_client=None,
    database_root: str = None,
    **kwargs
) -> BaseMemorySystem:
    """
    根据 method_name 获取不同的记忆系统实例
    """
    if method_name == "full_context":
        return FullContextMemorySystem(
            embed_model_name=embed_model_name,
            embed_client=embed_client,
            database_root=database_root
        )
    elif method_name == "rag":
        granularity = kwargs.get("granularity", 1)
        return RagMemorySystem(
            granularity=granularity,
            embed_model_name=embed_model_name,
            embed_client=embed_client,
            database_root=database_root
        )
    elif method_name == "only_query":
        return OnlyQueryMemorySystem()
    elif method_name == "amem":
        llm_client = kwargs.get("llm_client")
        if llm_client is None:
            raise ValueError("amem requires llm_client (via kwargs)")
        return AMemMemorySystem(
            embed_model_name=embed_model_name,
            llm_client=llm_client,
            embed_client=embed_client,
            database_root=database_root,
            related_memory_top_k=kwargs.get("related_memory_top_k", kwargs.get("retrieve_topk", 5)),
            language=kwargs.get("language"),
            granularity=kwargs.get("granularity", "all"),
            trace_log_dir=kwargs.get("trace_log_dir"),
        )
    elif method_name == "mem0":
        llm_client = kwargs.get("llm_client")
        if llm_client is None:
            raise ValueError("mem0 requires llm_client (via kwargs)")
        return Mem0MemorySystem(
            embed_model_name=embed_model_name,
            llm_client=llm_client,
            embed_client=embed_client,
            database_root=database_root,
            related_memory_top_k=kwargs.get("related_memory_top_k", kwargs.get("retrieve_topk", 5)),
            language=kwargs.get("language"),
            granularity=kwargs.get("granularity", "all"),
            trace_log_dir=kwargs.get("trace_log_dir"),
            dialogue_format=kwargs.get("dialogue_format", "user_assistant"),
        )
    else:
        raise ValueError(f"Unknown memory method: {method_name}")

__all__ = [
    "BaseMemorySystem",
    "RetrievedMemory",
    "get_memory_system",
    "FullContextMemorySystem",
    "RagMemorySystem",
    "OnlyQueryMemorySystem",
    "AMemMemorySystem",
    "Mem0MemorySystem",
]
