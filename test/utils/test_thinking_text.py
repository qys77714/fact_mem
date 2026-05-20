from utils.thinking_text import split_embedded_thinking


def test_split_embedded_thinking_redacted_block():
    body = "Okay, let's see. The answer should be no."
    raw = f"<think>{body}</think>\n\nno"
    think, tail = split_embedded_thinking(raw)
    assert body in think
    assert tail == "no"


def test_split_embedded_thinking_plain_content():
    assert split_embedded_thinking("  yes  ") == ("", "yes")
