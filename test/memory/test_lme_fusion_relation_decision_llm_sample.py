"""
关系包融合 prompt（lme_fuse_memory_bundle_zh/en.jinja）+ relation_decision 灌库形态的记忆元数据，
用 Qwen3-32B 采样调用，便于在终端查看完整 user prompt 与模型输出。

无 LLM 的用例默认随 pytest 运行；真实 API 用例需同时设置：

  export RUN_FUSION_LLM_SAMPLE=1
  export VLLM_API_KEY=...   # 与 llm_api 中 Qwen3-32B 一致，可选 VLLM_BASE_URL

可选环境变量：

  LME_RELATION_DECISION_ROOT   未融合 relation_decision 根目录（其下为各 history 子目录）。
                               默认尝试项目内常见路径；不存在则仅用内置合成包。
  LME_FUSION_SAMPLE_COUNT      采样包数量上限（默认 3）
  LME_FUSION_SAMPLE_LANGUAGE   zh / en / both（默认 zh）

建议查看输入输出时加 ``-s`` 不关 stdout::

  RUN_FUSION_LLM_SAMPLE=1 uv run pytest test/memory/test_lme_fusion_relation_decision_llm_sample.py -s -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory.base import RetrievedMemory  # noqa: E402
from memory.fusion.bundle_prompt_render import render_fusion_user_prompt  # noqa: E402
from memory.fusion.lme_bundle_fusion import (  # noqa: E402
    _fuse_bundle_with_llm,
    list_whole_tree_fusion_packages,
)
from memory.storage.local_faiss import LocalFaissDatabase  # noqa: E402


def _default_relation_decision_roots() -> List[Path]:
    return [
        PROJECT_ROOT / "MemDB" / "ingest" / "lme_s_cand0406_Qwen3-32B_allset_0406" / "relation_decision",
        PROJECT_ROOT / "MemDB" / "ingest" / "lme_s_cand0406_Qwen3-32B_allset_0406_limit1024" / "relation_decision",
    ]


def _resolve_rd_root() -> Path | None:
    env = (os.getenv("LME_RELATION_DECISION_ROOT") or "").strip()
    if env:
        p = Path(env).expanduser()
        return p if p.is_dir() else None
    for p in _default_relation_decision_roots():
        if p.is_dir():
            return p
    return None


def _iter_multi_member_packages(rd_root: Path, *, max_episodes: int = 80) -> Iterator[Tuple[str, List[RetrievedMemory]]]:
    """遍历子目录，返回 (history_name, members)。"""
    subdirs = sorted(p for p in rd_root.iterdir() if p.is_dir())
    for ep in subdirs[:max_episodes]:
        try:
            db = LocalFaissDatabase(namespace=ep.name, database_root=str(rd_root))
            for pkg in list_whole_tree_fusion_packages(db):
                if len(pkg) >= 2:
                    yield ep.name, pkg
        except (OSError, ValueError, RuntimeError):
            continue


def _synthetic_relation_decision_bundles() -> List[List[RetrievedMemory]]:
    """与 relation_decision 写库一致的 metadata 形状（primary + evidence 树）。"""
    root = RetrievedMemory(
        memory_id="pid-001",
        text="Alice 于 2024 年 3 月任项目经理。",
        source_index="c0",
        time="2024/03/01",
        score=0.0,
        metadata={"memory_role": "primary"},
    )
    equiv = RetrievedMemory(
        memory_id="eid-002",
        text="2024 年 3 月起 Alice 担任项目经理一职。",
        source_index="c1",
        time="2024/03/05",
        score=0.0,
        metadata={
            "memory_role": "evidence",
            "parent_primary": "pid-001",
            "lme_edge": "EQUIV",
            "evidence_depth": 1,
        },
    )
    attach = RetrievedMemory(
        memory_id="eid-003",
        text="她同时负责跨团队排期与风险评审。",
        source_index="c2",
        time="2024/04/10",
        score=0.0,
        metadata={
            "memory_role": "evidence",
            "parent_primary": "pid-001",
            "lme_edge": "ATTACH",
            "evidence_depth": 1,
        },
    )
    nested = RetrievedMemory(
        memory_id="eid-004",
        text="风险评审每双周一次。",
        source_index="c3",
        time="2024/04/12",
        score=0.0,
        metadata={
            "memory_role": "evidence",
            "parent_primary": "eid-003",
            "lme_edge": "ATTACH",
            "evidence_depth": 2,
        },
    )
    upd = RetrievedMemory(
        memory_id="eid-005",
        text="先前 2023 年 12 月 Alice 为高级工程师。",
        source_index="c4",
        time="2023/12/01",
        score=0.0,
        metadata={
            "memory_role": "evidence",
            "parent_primary": "pid-001",
            "lme_edge": "UPDATE",
            "evidence_depth": 1,
        },
    )
    return [
        [root, equiv],
        [root, equiv, attach, nested],
        [root, upd, equiv],
    ]


def _collect_sample_bundles(limit: int) -> List[Tuple[str, List[RetrievedMemory]]]:
    out: List[Tuple[str, List[RetrievedMemory]]] = []
    rd = _resolve_rd_root()
    label = "disk"
    if rd is not None:
        for hist, pkg in _iter_multi_member_packages(rd):
            out.append((f"{label}:{hist}", pkg))
            if len(out) >= limit:
                return out
    for i, pkg in enumerate(_synthetic_relation_decision_bundles()):
        out.append((f"synthetic:{i}", pkg))
        if len(out) >= limit:
            break
    return out


def _languages_from_env() -> List[str]:
    raw = (os.getenv("LME_FUSION_SAMPLE_LANGUAGE") or "zh").strip().lower()
    if raw == "both":
        return ["zh", "en"]
    if raw in ("zh", "en"):
        return [raw]
    return ["zh"]


def _sample_limit() -> int:
    try:
        return max(1, int(os.getenv("LME_FUSION_SAMPLE_COUNT", "3")))
    except ValueError:
        return 3


def test_render_fusion_prompt_relation_decision_style_zh_en() -> None:
    """不调用 LLM：确认 relation_decision 形态下中英文模板均能渲染且含树/边信息。"""
    bundles = _synthetic_relation_decision_bundles()
    pkg = bundles[1]
    zh = render_fusion_user_prompt(pkg, language="zh")
    en = render_fusion_user_prompt(pkg, language="en")
    assert "主事实" in zh or "包内树" in zh
    assert "[primary]" in en or "[bundle tree]" in en
    # 子边为「子→父」：evidence 挂在 attach 行下则为 4→3
    assert "4→3" in zh and "4→3" in en
    assert "父节点（第" in zh
    assert "【附属】对父节点（第 3 条）的补充、细化或从属信息" in zh
    assert "parent node (row 3)" in en
    assert "[attached] supplements or refines parent node (row 3)" in en


@pytest.mark.skipif(
    not (os.getenv("RUN_FUSION_LLM_SAMPLE") or "").strip(),
    reason="设置 RUN_FUSION_LLM_SAMPLE=1 才调用 Qwen3-32B（避免 CI 消耗与依赖外网）",
)
def test_qwen3_32b_fuse_sample_bundles_stdout() -> None:
    """调用 VLLM 上的 Qwen3-32B，打印各采样包的融合 user prompt 与模型输出。"""
    from utils.env import load_env
    from utils.llm_api import load_api_chat_completion

    load_env(str(PROJECT_ROOT / ".env"))
    if not (os.getenv("VLLM_API_KEY") or "").strip():
        pytest.skip("需要 VLLM_API_KEY（Qwen3-32B）")

    limit = _sample_limit()
    bundles = _collect_sample_bundles(limit)
    if not bundles:
        pytest.skip("无可用关系包（磁盘目录为空且合成列表异常）")

    llm = load_api_chat_completion("Qwen3-32B", async_=False)
    langs = _languages_from_env()

    for tag, members in bundles:
        for lang in langs:
            user_prompt = render_fusion_user_prompt(members, language=lang)
            print("\n" + "=" * 80)
            print(f"[sample] bundle={tag} language={lang} members={len(members)}")
            print("-" * 80)
            print("--- LLM user message (full) ---")
            print(user_prompt)
            print("-" * 80)
            fused = _fuse_bundle_with_llm(
                members,
                llm_client=llm,
                language=lang,
                max_new_tokens=512,
            )
            print("--- LLM merged paragraph ---")
            print(fused)
            print("=" * 80)

        assert len(members) >= 1

    # 至少验证最后一次调用有非空返回（单成员包不走 LLM，此处均为多成员）
    last_members = bundles[-1][1]
    if len(last_members) >= 2:
        out = _fuse_bundle_with_llm(
            last_members,
            llm_client=llm,
            language=langs[-1],
            max_new_tokens=512,
        )
        assert isinstance(out, str) and out.strip()
