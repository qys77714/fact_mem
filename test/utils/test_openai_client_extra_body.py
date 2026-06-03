from utils.openai_client import _prepare_extra_body_for_model


def test_qwen_models_disable_thinking_by_default():
    body = _prepare_extra_body_for_model("Qwen3-4B", None, qwen_enable_thinking=False)
    assert body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_qwen_models_enable_thinking_when_requested():
    body = _prepare_extra_body_for_model("Qwen3-4B", None, qwen_enable_thinking=True)
    assert body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_non_qwen_models_drop_chat_template_kwargs():
    body = _prepare_extra_body_for_model(
        "gemma-4-26B-A4B-it",
        {"chat_template_kwargs": {"enable_thinking": True}},
        qwen_enable_thinking=False,
    )
    assert body is None
