#!/usr/bin/env python3
"""Load run_exp parallel / token_limits / debug / prompts / paths from YAML and emit bash assignments."""

from __future__ import annotations

import os
import re
import shlex
import sys
from pathlib import Path

import yaml

_VAR = re.compile(r"\$\{([^}]+)\}")

# 与 run_exp.sh 管线的 Jinja 模板名对齐；在 YAML 的 prompts 中可只写需要覆盖的键。
# 见 run_exp.config.yaml 中分节说明；空字符串表示「由对应步骤脚本内置默认处理」
# （如 mem_extract_template / mem_extract_aspect_template_* / mem_extract_aspects_only / fusion_bundle_template_zh 置空则不向 CLI 传该参数）。
_DEFAULT_PROMPTS: dict[str, str] = {
    # --- extract_candidates：--mem-extract-template；空 = 由 benchmark+language 解析默认 ---
    "mem_extract_template": "0_mem_extract.jinja",
    # --- extract_candidates：0 = 仅主模板；1 = 仅三方面（须配合 aspect 模板非空）---
    "mem_extract_aspects_only": "0",
    # --- extract_candidates：--mem-extract-extra-template（顺序 1→2→3；空 = 不传；仅 aspects_only=1 时使用）---
    "mem_extract_aspect_template_1": "",
    "mem_extract_aspect_template_2": "",
    "mem_extract_aspect_template_3": "",
    # --- ingest relation_decision：成对五类关系 ---
    "relation_classification_system_en": "lme_relation_classification_system_en_v2.jinja",
    "relation_classification_system_zh": "lme_relation_classification_system_zh_v2.jinja",
    "relation_classification_user": "lme_relation_classification_user.jinja",
    # --- fuse_lme_memory_bundles：关系包 user + 行前缀边标签 ---
    "fusion_bundle_template_en": "lme_fuse_memory_bundle_en_v3.jinja",
    "fusion_bundle_template_zh": "",
    "fusion_edge_labels_template_en": "lme_fuse_memory_bundle_edge_labels_en_v2.jinja",
    "fusion_edge_labels_template_zh": "lme_fuse_memory_bundle_edge_labels_zh_v2.jinja",
    # --- pipeline_evaluate：LLM Judge ---
    "judge_oqa_template": "pipeline_eval_oqa.jinja",
    "judge_mcq_template": "pipeline_eval_mcq.jinja",
    "judge_system_template": "pipeline_eval_system.jinja",
    # --- pipeline_generate / agent + LME 检索上下文（load_config 只导出，run_exp 不传 CLI；与代码一致）---
    "generate_agent_prompt_mcq_en": "agent_prompt_en_mcq.jinja",
    "generate_agent_prompt_mcq_zh": "agent_prompt_zh_mcq.jinja",
    "generate_agent_prompt_open_en": "agent_prompt_en_open.jinja",
    "generate_agent_prompt_open_zh": "agent_prompt_zh_open.jinja",
    "generate_context_empty_en": "agent_context_empty_en.jinja",
    "generate_context_empty_zh": "agent_context_empty_zh.jinja",
    "generate_context_unit_en": "lme_memory_context_unit_en.jinja",
    "generate_context_unit_zh": "lme_memory_context_unit_zh.jinja",
    # --- memory.base：非 LME 子类 format_retrieved_for_context 用的 unit 行 ---
    "base_memory_context_unit_en": "agent_context_unit_en.jinja",
    "base_memory_context_unit_zh": "agent_context_unit_zh.jinja",
    # --- mem0 管线（ingest mem0 等；run_exp 默认未开该分支，仅作单一配置源）---
    "mem0_fact_retrieval_en": "mem0_fact_retrieval_en.jinja",
    "mem0_fact_retrieval_zh": "mem0_fact_retrieval_zh.jinja",
    "mem0_fact_retrieval_multi_en": "mem0_fact_retrieval_multi_en.jinja",
    "mem0_fact_retrieval_multi_zh": "mem0_fact_retrieval_multi_zh.jinja",
    "mem0_current_memory_part_en": "mem0_current_memory_part_en.jinja",
    "mem0_current_memory_part_zh": "mem0_current_memory_part_zh.jinja",
    "mem0_current_memory_empty_en": "mem0_current_memory_empty_en.jinja",
    "mem0_current_memory_empty_zh": "mem0_current_memory_empty_zh.jinja",
    "mem0_update_memory_default_en": "mem0_update_memory_default_en.jinja",
    "mem0_update_memory_default_zh": "mem0_update_memory_default_zh.jinja",
    "mem0_update_memory_no_delete_en": "mem0_update_memory_no_delete_en.jinja",
    "mem0_update_memory_no_delete_zh": "mem0_update_memory_no_delete_zh.jinja",
    "mem0_context_unit_en": "mem0_context_unit_en.jinja",
    "mem0_context_unit_zh": "mem0_context_unit_zh.jinja",
}


def _expand_once(template: str, env: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in env:
            return m.group(0)
        return env[key]

    return _VAR.sub(repl, template)


def expand_nested(templates: dict[str, object], base: dict[str, str]) -> dict[str, str]:
    """Resolve ${var} with iterative substitution so paths can reference each other."""
    str_templates = {str(k): str(v) for k, v in templates.items()}
    out = {k: str_templates[k] for k in str_templates}
    for _ in range(256):
        e = {**base, **out}
        new_out = {k: _expand_once(t, e) for k, t in str_templates.items()}
        if new_out == out:
            for k, v in new_out.items():
                if "${" in v:
                    raise SystemExit(
                        f"run_exp_load_config: unresolved placeholder in paths.{k}={v!r}"
                    )
            return new_out
        out = new_out
    raise SystemExit("run_exp_load_config: path expansion did not converge")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: run_exp_load_config.py CONFIG.yaml", file=sys.stderr)
        raise SystemExit(2)
    cfg_path = Path(sys.argv[1])
    if not cfg_path.is_file():
        print(f"run_exp_load_config: not a file: {cfg_path}", file=sys.stderr)
        raise SystemExit(1)

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    parallel = raw.get("parallel") or {}
    token_limits = raw.get("token_limits") or {}
    debug = raw.get("debug") or {}
    prompts_raw = raw.get("prompts")
    if prompts_raw is None:
        prompts: dict[str, object] = dict(_DEFAULT_PROMPTS)
    elif not isinstance(prompts_raw, dict):
        raise SystemExit("run_exp_load_config: prompts must be a mapping")
    else:
        prompts = {
            **_DEFAULT_PROMPTS,
            **{str(k): ("" if v is None else str(v)) for k, v in prompts_raw.items()},
        }
    paths = raw.get("paths") or {}

    base: dict[str, str] = {k: v for k, v in os.environ.items() if isinstance(v, str)}
    for name, section in (
        ("parallel", parallel),
        ("token_limits", token_limits),
        ("debug", debug),
        ("prompts", prompts),
    ):
        if not isinstance(section, dict):
            raise SystemExit(f"run_exp_load_config: {name} must be a mapping")
        for k, v in section.items():
            base[str(k)] = "" if v is None else str(v)

    if not isinstance(paths, dict):
        raise SystemExit("run_exp_load_config: paths must be a mapping")
    expanded_paths = expand_nested(paths, base)

    for section in (parallel, token_limits, debug, prompts):
        for k, v in section.items():
            print(f"{k}={shlex.quote('' if v is None else str(v))}")
    for k, v in expanded_paths.items():
        print(f"{k}={shlex.quote(v)}")


if __name__ == "__main__":
    main()
