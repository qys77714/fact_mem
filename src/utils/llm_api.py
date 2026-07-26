import os

from utils.env import load_env
from utils.openai_client import AsyncOpenAIClient, OpenAIClient


def _get_required_env(key: str) -> str:
	value = os.getenv(key)
	if not value:
		raise ValueError(f"Missing required environment variable: {key}")
	return value


def load_api_chat_completion(model_name, async_=False, *args, **kargs):
	load_env()

	model_name_qwen = {
		# Qwen
		"qwen3-max": "qwen3-max",
		"qwen-plus": "qwen-plus",
		"qwen-turbo": "qwen-turbo",
		"QWQ-32B": "qwq-32b-preview",
		"qwen2.5-72b-instruct": "qwen2.5-72b-instruct",
		"qwen2.5-32b-instruct": "qwen2.5-32b-instruct",
		"qwen1.5-72b-chat": "qwen1.5-72b-chat",
		"glm-5": "glm-5",
		# Qwen3 via DashScope (Aliyun API)
		"Qwen3-32B": "qwen3-32b",
		# # deepseek
		# "deepseek-v3": "deepseek-v3",
		# "deepseek-r1": "deepseek-r1",
	}
	model_name_deepseek = {
		# deepseek
		"deepseek-v3": "deepseek-chat",
		"deepseek-r1": "deepseek-reasoner",
		"deepseek-v4-flash": "deepseek-v4-flash",
		"deepseek-v4-pro": "deepseek-v4-pro",
	}
	model_name_openai = {
		# openai
		"gpt-4o-mini": "gpt-4o-mini",
		"gpt-4o": "gpt-4o",
		"gpt-3.5-turbo": "gpt-3.5-turbo-0125",
		"gpt-4.1": "gpt-4.1",
		"gpt-4.1-nano": "gpt-4.1-nano",
		"gpt-5.4": "gpt-5.4",
	}
	model_name_01 = {
		"yi-large": "yi-large",
		"yi-34b-chat": "yi-medium"
	}
	model_name_intern = {
		"internlm2_5-20b-chat": "internlm2.5-latest"
	}


	model_name_vllm = {
		"Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
		"Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instruct",
		"Ministral-8B-Instruct": "Ministral-8B-Instruct",
		"Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
		"qwen3-moe": "qwen3-moe",
		"Qwen3-8B": "Qwen3-8B",
		# Qwen3-32B moved to DashScope API (see model_name_qwen)
		"Qwen3-4B": "Qwen3-4B",
		"Qwen3-0.6B": "Qwen3-0.6B",
		"qwen3-30b-moe": "qwen3-30b-moe",
		"Qwen3.5-27B-FP8": "Qwen3.5-27B-FP8",
		"Qwen3.5-27B": "Qwen3.5-27B",
		"Qwen3.5-4B": "Qwen3.5-4B",
		"Qwen3.5-9B": "Qwen3.5-9B",
		"Qwen3-30B": "Qwen3-30B-A3B-Thinking-2507",
		"gemma4-12b": "gemma-4-12B-it",
		"gemma4-26B": "gemma-4-26B-A4B-it",
		"gemma4-31B": "gemma-4-31B-it",
		"gemma4-e2b": "gemma-4-E2B-it",
		"gemma4-e4b": "gemma-4-E4B-it",
	}

	if model_name in list(model_name_vllm.keys()):
		model_orig_name = model_name
		model_name = model_name_vllm[model_name]
		api_key = os.getenv("VLLM_API_KEY", "EMPTY")
		
		# Allow reading VLLM_{MODEL}_PORT or PORT_{MODEL} from .env for dynamic port assignment
		env_model_name = model_orig_name.upper().replace("-", "_").replace(".", "_")
		model_specific_port = os.getenv(f"PORT_{env_model_name}") or os.getenv(f"VLLM_{env_model_name}_PORT")
		if model_specific_port:
			base_url = f"http://localhost:{model_specific_port}/v1/"
		else:
			base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1/")
	# elif model_name == "qwen3-32b":
	#     api_key = "mem"
	#     base_url = "http://10.0.2.219:1146/v1"
	elif model_name in list(model_name_qwen.keys()):
		api_key = _get_required_env("DASHSCOPE_API_KEY")
		base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
		model_name = model_name_qwen[model_name]
	elif model_name in list(model_name_deepseek.keys()):
		api_key = _get_required_env("DEEPSEEK_API_KEY")
		base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
		model_name = model_name_deepseek[model_name]
	elif model_name in list(model_name_openai.keys()):
		api_key = _get_required_env("OPENAI_API_KEY")
		base_url = os.getenv("OPENAI_BASE_URL", "https://foundationmodeldepartment.openai.azure.com/openai/v1")
		model_name = model_name_openai[model_name]
	elif model_name in list(model_name_01.keys()):
		api_key = _get_required_env("YI_API_KEY")
		base_url = os.getenv("YI_BASE_URL", "https://api.lingyiwanwu.com/v1")
		model_name = model_name_01[model_name]
	elif model_name in list(model_name_intern.keys()):
		api_key = _get_required_env("INTERN_API_KEY")
		base_url = os.getenv("INTERN_BASE_URL", "https://chat.intern-ai.org.cn/api/v1/")
		model_name = model_name_intern[model_name]
	else:
		raise ValueError(f"Unknown model: {model_name}")

	if not async_:
		client = OpenAIClient(api_key=api_key, base_url=base_url, model=model_name,
		                      dashscope_mode=("dashscope" in base_url))
	else:
		client = AsyncOpenAIClient(api_key=api_key, base_url=base_url, model=model_name,
		                           dashscope_mode=("dashscope" in base_url))

	return client


__all__ = ["load_api_chat_completion"]
