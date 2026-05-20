from .base import BaseBenchmark, MemoryEpisode, ChatSession, ChatTurn, QuestionItem
from .datasets import DEFAULT_BENCHMARK_DATASETS, resolve_benchmark_data_path
from .locomo import LocomoBenchmark
from .lme import LMEBenchmark
from .event_bench import EventBenchmark
from .meme import MEMEBenchmark

def get_benchmark(task_name: str, file_path: str, lang: str = "en") -> BaseBenchmark:
    """
    根据 task_name 返回对应的 Benchmark 实例
    """
    task_name = task_name.lower()
    
    if task_name.startswith("meme"):
        return MEMEBenchmark(file_path, lang=lang)
    elif task_name.startswith("lme"):
        return LMEBenchmark(file_path, lang=lang)
    elif "locomo" in task_name:
        return LocomoBenchmark(file_path, lang=lang)
    elif "lmb" in task_name or "emb" in task_name or "event" in task_name:
        return EventBenchmark(file_path, lang=lang)
    else:
        # 默认回退到通用格式 (因为格式基本一致，可用 LMEBenchmark 作为 fallback)
        return LMEBenchmark(file_path, lang=lang)

__all__ = [
    "BaseBenchmark",
    "MemoryEpisode",
    "ChatSession",
    "ChatTurn",
    "QuestionItem",
    "get_benchmark",
    "LocomoBenchmark",
    "LMEBenchmark",
    "EventBenchmark",
    "MEMEBenchmark",
    "DEFAULT_BENCHMARK_DATASETS",
    "resolve_benchmark_data_path",
]
