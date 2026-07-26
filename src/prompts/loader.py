from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


def _strip_time(value: str) -> str:
    """去除时间字符串末尾的 HH:MM 部分。
    '2023/05/28 (Sun) 00:37' → '2023/05/28 (Sun)'
    '2023/05/22' → '2023/05/22'
    """
    return re.sub(r'\s+\d{2}:\d{2}$', '', (value or ''))


@lru_cache(maxsize=1)
def _get_env() -> Environment:
    templates_dir = Path(__file__).resolve().parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        variable_start_string="[[",
        variable_end_string="]]",
    )
    env.filters['strip_time'] = _strip_time
    return env


def render_prompt(template_name: str, **context: Any) -> str:
    env = _get_env()
    template = env.get_template(template_name)
    return template.render(**context).strip()
