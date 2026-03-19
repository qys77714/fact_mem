from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


@lru_cache(maxsize=1)
def _get_env() -> Environment:
    templates_dir = Path(__file__).resolve().parent / "templates"
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        variable_start_string="[[",
        variable_end_string="]]",
    )


def render_prompt(template_name: str, **context: Any) -> str:
    env = _get_env()
    template = env.get_template(template_name)
    return template.render(**context).strip()
