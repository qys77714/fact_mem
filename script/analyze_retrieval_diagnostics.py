#!/usr/bin/env python3
"""
Retrieval Mechanism Analysis / Retrieval Process Diagnostics
============================================================
分析 N=8 实验的检索诊断指标。

对每个 manager model × method 组合，计算：
  - Any gold retrieved (%)
  - All gold retrieved (%)
  - Gold recall (%)
  - Confounder token share (%)
  - Retrieved entry count (in final 256-token context)

用法:
  PYTHONPATH=src uv run --no-sync python script/analyze_retrieval_diagnostics.py
  PYTHONPATH=src uv run --no-sync python script/analyze_retrieval_diagnostics.py --output-dir results/retrieval_diag
"""

import json
import os
import re
import sys
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

# ---- Configuration ----

HYBRID_GOLDEN_PATH = os.path.join(REPO, "data/preprocessed/longmemeval_s_hybrid_golden.json")
ARTIFACTS_DIR = os.path.join(REPO, "artifacts")
DEFAULT_OUTPUT_DIR = os.path.join(REPO, "results/retrieval_diagnostics")

# Experiment runs: (run_id, method_key, ingest_id, answer_id)
# Key: (manager_model, method_name)
RUN_CONFIGS = {
    ("gemma4-e4b", "add_all"): {
        "run_id": "lme_s_hybrid_filler_n8_gemma4-e4b_rd_addall_add_all+relation_decision_tl256--3547654a",
        "method_key": "add_all",
        "ingest_id": "455348c2",
        "answer_id": "8d2c1673",
    },
    ("gemma4-e4b", "relation_decision"): {
        "run_id": "lme_s_hybrid_filler_n8_gemma4-e4b_rd_addall_add_all+relation_decision_tl256--3547654a",
        "method_key": "relation_decision",
        "ingest_id": "dd9d919c",
        "answer_id": "2b666548",
    },
    ("gemma4-e4b", "evermemos"): {
        "run_id": "lme_s_hybrid_filler_n8_gemma4-e4b_evm_evermemos_tl256--d445ee5d",
        "method_key": "evermemos",
        "ingest_id": "9be1a72f",
        "answer_id": "d46fe4e6",
    },
    ("gemma4-e4b", "mem0"): {
        "run_id": "lme_s_hybrid_filler_n8_gemma4-e4b_mem0_mem0_tl256--390d7a29",
        "method_key": "mem0",
        "ingest_id": "d522d1c6",
        "answer_id": "529c95b2",
    },
    ("Qwen3.5-4B", "add_all"): {
        "run_id": "lme_s_hybrid_filler_n8_qwen3-5-4b_rd_addall_add_all+relation_decision_tl256--f1153694",
        "method_key": "add_all",
        "ingest_id": "684ac1e5",
        "answer_id": "d32aae13",
    },
    ("Qwen3.5-4B", "relation_decision"): {
        "run_id": "lme_s_hybrid_filler_n8_qwen3-5-4b_rd_addall_add_all+relation_decision_tl256--f1153694",
        "method_key": "relation_decision",
        "ingest_id": "507ca0d2",
        "answer_id": "3aa75bae",
    },
    ("Qwen3.5-4B", "evermemos"): {
        "run_id": "lme_s_hybrid_filler_n8_qwen3-5-4b_evm_evermemos_tl256--cefcf552",
        "method_key": "evermemos",
        "ingest_id": "fc8e285b",
        "answer_id": "a8cd3129",
    },
    ("Qwen3.5-4B", "mem0"): {
        "run_id": "lme_s_hybrid_filler_n8_qwen3-5-4b_mem0_mem0_tl256--bab40ec3",
        "method_key": "mem0",
        "ingest_id": "75ffdf44",
        "answer_id": "2c0e71fd",
    },
}

# Method display names
METHOD_DISPLAY = {
    "add_all": "Append-all",
    "relation_decision": "RD",
    "evermemos": "EverMemOS-style",
    "mem0": "Mem0-style",
}


# ---- Data Structures ----

@dataclass
class QuestionResult:
    """Per-question diagnostic result."""
    manager_model: str
    method: str
    run_id: int  # always 0 for single run
    query_id: str  # UUID question_id
    history_name: str
    gold_memory_ids: List[str] = field(default_factory=list)
    retrieved_entry_ids: List[str] = field(default_factory=list)  # in final context
    retrieved_source_memory_ids: List[str] = field(default_factory=list)  # all source IDs (incl. fused)
    retrieved_confounder_ids: List[str] = field(default_factory=list)
    num_required_gold: int = 0
    num_retrieved_gold: int = 0
    any_gold_retrieved: bool = False
    all_gold_retrieved: bool = False
    gold_recall: float = 0.0
    memory_context_tokens: int = 0
    confounder_tokens: int = 0
    mixed_or_unattributed_tokens: int = 0
    confounder_token_share: float = 0.0
    retrieved_entry_count: int = 0
    final_memory_context: str = ""
    confusion_type: str = ""
    # Extra diagnostics
    retrieved_bg_ids: List[str] = field(default_factory=list)
    bg_tokens: int = 0
    gold_tokens: int = 0
    evidence_has_gold_primary_missed: bool = False
    is_mixed_fused: bool = False  # RD fused entry contains both gold and confounder
    # New fields for provenance audit
    required_gold_ids: List[str] = field(default_factory=list)  # = gold_memory_ids (alias for clarity)
    retrieved_gold_ids: List[str] = field(default_factory=list)  # = retrieved_gold_ids (actual intersection)
    unmatched_required_gold_ids: List[str] = field(default_factory=list)
    duplicate_gold_match_count: int = 0  # how many times the same gold ID matched
    empty_context_reason: str = ""


# ---- Helper Functions ----

def load_hybrid_golden_data(path: str = HYBRID_GOLDEN_PATH) -> Dict:
    """Load hybrid golden data, indexed by question_id (which equals history_name)."""
    with open(path) as f:
        data = json.load(f)
    return {item["question_id"]: item for item in data}


def load_tokenizer():
    """Load the Qwen3-8B tokenizer used in experiments."""
    from transformers import AutoTokenizer
    tokenizer_path = os.environ.get(
        "ANSWER_TOKENIZER_PATH", "/data/zjj/models/Qwen/Qwen3-8B"
    )
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def token_count(text: str, tokenizer) -> int:
    """Count tokens in text using the given tokenizer."""
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def extract_memory_units_from_prompt(prompt: str) -> List[str]:
    """
    Extract individual memory unit blocks from the formatted prompt.
    Returns list of full memory unit text blocks (including formatting).

    Uses a two-step approach:
    1. First try to match complete units with </MemoryContent> closing tag
    2. For any unmatched trailing unit (truncated by token limit), match without closing tag
    """
    # Match complete memory units
    pattern_complete = r'(### Memory Unit \d+.*?</MemoryContent>)'
    complete = re.findall(pattern_complete, prompt, re.DOTALL)

    # Check for a trailing incomplete unit (truncated before </MemoryContent>)
    # Find the position after the last complete unit
    last_end = 0
    for m in re.finditer(pattern_complete, prompt, re.DOTALL):
        last_end = m.end()

    remaining = prompt[last_end:]

    # Check if remaining starts with a new memory unit header but lacks closing tag
    pattern_partial = r'(### Memory Unit \d+.*?)(?=### Memory Unit \d+|### Question Details|\Z)'
    partial_matches = re.findall(pattern_partial, remaining, re.DOTALL)
    partial = [m.strip() for m in partial_matches
               if m.strip().startswith('### Memory Unit') and '</MemoryContent>' not in m]

    return [m.strip() for m in complete] + partial


def extract_memory_content_from_unit(unit_text: str) -> str:
    """Extract just the content text from a memory unit block.
    Handles both complete units (with </MemoryContent>) and truncated ones (without)."""
    # Try complete unit first
    m = re.search(r'<MemoryContent>\s*(.*?)\s*</MemoryContent>', unit_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try truncated unit (no closing tag)
    m = re.search(r'<MemoryContent>\s*(.*)', unit_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def build_text_to_id_mapping(ingest_dir: str, history_names: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Build text→memory_id mapping for each history from ingest data.
    Returns {history_name: {text: memory_id}}
    """
    mapping = {}
    import glob as _glob
    for hn in history_names:
        hist_dir = os.path.join(ingest_dir, hn)
        if not os.path.isdir(hist_dir):
            # Check for hash-named directories
            for d in os.listdir(ingest_dir):
                dp = os.path.join(ingest_dir, d)
                if os.path.isdir(dp):
                    # Check .memory_ready.json for history_name
                    mrf = os.path.join(dp, ".memory_ready.json")
                    if os.path.exists(mrf):
                        with open(mrf) as f:
                            mr = json.load(f)
                        if mr.get("history_name") == hn:
                            hist_dir = dp
                            break
        if not os.path.isdir(hist_dir):
            continue
        texts_path = os.path.join(hist_dir, "texts.json")
        ids_path = os.path.join(hist_dir, "ids.json")
        if not os.path.exists(texts_path) or not os.path.exists(ids_path):
            continue
        with open(texts_path) as f:
            texts = json.load(f)
        with open(ids_path) as f:
            ids = json.load(f)
        mapping[hn] = {t: i for t, i in zip(texts, ids)}
    return mapping


def find_history_dir(ingest_base: str, history_name: str) -> Optional[str]:
    """Find the ingest subdirectory for a given history_name.
    All ingest directories use history_name directly as directory names."""
    direct = os.path.join(ingest_base, history_name)
    if os.path.isdir(direct):
        return direct
    return None


def _fuzzy_match_text(
    target: str, candidates: List[str], threshold: float = 0.7
) -> Optional[Tuple[int, float]]:
    """Find best fuzzy match for target in candidates, returning (index, ratio) or None."""
    import difflib
    best_idx, best_ratio = -1, 0.0
    for i, c in enumerate(candidates):
        ratio = difflib.SequenceMatcher(None, target, c).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
    if best_ratio >= threshold:
        return (best_idx, best_ratio)
    return None


def build_gold_confounder_id_map(
    ingest_dir: str,
    hybrid_data: Dict,
    history_names: List[str],
    method: str = "",
) -> Dict[str, Dict]:
    """
    For each history_name, identify gold memory managed IDs and confounder managed IDs
    by matching texts from hybrid_golden data to ingest texts.

    Uses STABLE original gold keys (gold_0, gold_1, ...) independent of whether
    a particular method's ingest preserves the gold text. This ensures num_required_gold
    is always correct even when mem0 rewrites/loses gold facts.

    For mem0 (which rewrites memory texts), uses fuzzy matching as fallback.

    Returns: {history_name: {
        "num_required_gold": int,            # always correct count
        "original_gold_keys": ["gold_0", ...],  # stable keys
        "gold_key_to_text": {gold_key: original_text},
        "gold_key_to_managed_ids": {gold_key: set(ingest_memory_ids)},
        "conf_managed_ids": set(ingest_memory_ids),  # confounders in this ingest
        "gold_match_type": {gold_key: "exact"|"fuzzy"|"lost"},
        "confusion_type": str,
        "new_value_golden_idx": int|None,
    }}
    """
    result = {}
    for hn in history_names:
        hr = hybrid_data.get(hn)
        if not hr:
            continue

        hist_dir = find_history_dir(ingest_dir, hn)
        if not hist_dir:
            continue

        texts_path = os.path.join(hist_dir, "texts.json")
        ids_path = os.path.join(hist_dir, "ids.json")
        if not os.path.exists(texts_path) or not os.path.exists(ids_path):
            continue

        with open(texts_path) as f:
            texts = json.load(f)
        with open(ids_path) as f:
            ids = json.load(f)

        text_to_id = dict(zip(texts, ids))

        # Build stable gold keys
        golden_memory_list = hr.get("golden_memory", [])
        num_required = len(golden_memory_list)
        original_gold_keys = [f"gold_{i}" for i in range(num_required)]
        gold_key_to_text = {}
        gold_key_to_managed_ids = {}
        gold_match_type = {}

        for i, gm in enumerate(golden_memory_list):
            gk = f"gold_{i}"
            gt = gm["text"]
            gold_key_to_text[gk] = gt

            managed_ids = set()
            match_type = "lost"

            # 1. Try exact match
            if gt in text_to_id:
                managed_ids.add(text_to_id[gt])
                match_type = "exact"
            # 2. For mem0, try fuzzy matching
            elif method == "mem0":
                fm = _fuzzy_match_text(gt, texts, threshold=0.65)
                if fm is not None:
                    idx, ratio = fm
                    managed_ids.add(ids[idx])
                    match_type = f"fuzzy_{ratio:.2f}"

            gold_key_to_managed_ids[gk] = managed_ids
            gold_match_type[gk] = match_type

        # Confounder managed IDs
        conf_managed_ids = set()
        for d in hr.get("distractors", []):
            dt = d["text"]
            if dt in text_to_id:
                conf_managed_ids.add(text_to_id[dt])
            elif method == "mem0":
                fm = _fuzzy_match_text(dt, texts, threshold=0.65)
                if fm is not None:
                    idx, ratio = fm
                    conf_managed_ids.add(ids[idx])

        result[hn] = {
            "num_required_gold": num_required,
            "original_gold_keys": original_gold_keys,
            "gold_key_to_text": gold_key_to_text,
            "gold_key_to_managed_ids": gold_key_to_managed_ids,
            "conf_managed_ids": conf_managed_ids,
            "gold_match_type": gold_match_type,
            "confusion_type": hr.get("confusion_type", ""),
            "new_value_golden_idx": hr.get("new_value_golden_idx"),
        }

    return result


def match_memory_units_to_retrieved(
    memory_units: List[str],
    retrieved: List[Dict],
) -> List[Tuple[str, Optional[Dict]]]:
    """
    Match each memory unit in the prompt to its corresponding retrieved entry.
    Returns list of (unit_text, retrieved_entry_or_None).

    For truncated units (missing </MemoryContent>), uses prefix matching since
    the full text was cut off by token truncation.
    """
    results = []
    # Build lookup: text -> retrieved entry
    text_to_entry = {}
    for r in retrieved:
        t = r["text"].strip()
        text_to_entry[t] = r

    for unit in memory_units:
        content = extract_memory_content_from_unit(unit)
        is_truncated = '</MemoryContent>' not in unit

        # Try exact match first
        entry = text_to_entry.get(content)
        if entry is not None:
            results.append((unit, entry))
            continue

        if is_truncated and content:
            # For truncated units: find the retrieved entry whose text STARTS WITH
            # the truncated content (token truncation cuts from the end)
            best_entry = None
            best_len = 0
            for r in retrieved:
                rt = r["text"].strip()
                if rt.startswith(content[:50]):  # at least first 50 chars match
                    if len(rt) > best_len:
                        best_entry = r
                        best_len = len(rt)
            if best_entry is not None:
                results.append((unit, best_entry))
                continue

        # Try without trailing punctuation and whitespace normalization
        content_norm = content.rstrip(".,;:!? ").strip()
        for r in retrieved:
            rt = r["text"].strip().rstrip(".,;:!? ").strip()
            if rt == content_norm or (not is_truncated and content_norm in rt):
                entry = r
                break
        results.append((unit, entry))
    return results


def classify_entry(
    memory_id: str,
    entry: Dict,
    id_map: Dict,
) -> Tuple[str, List[str]]:
    """
    Classify a retrieved entry as 'gold', 'confounder', 'bg', or 'mixed'.
    For RD fused entries, check fused_member_ids.

    Returns (category, list_of_matched_source_ids)
    """
    gold_ids = id_map.get("gold_ids", set())
    conf_ids = id_map.get("conf_ids", set())

    meta = entry.get("metadata", {})
    fused_ids = meta.get("fused_member_ids", []) if meta.get("answer_fused") else []

    if fused_ids:
        # Check all fused member IDs
        fused_gold = [fid for fid in fused_ids if fid in gold_ids]
        fused_conf = [fid for fid in fused_ids if fid in conf_ids]

        if fused_gold and fused_conf:
            return "mixed", fused_gold + fused_conf
        elif fused_gold:
            return "gold", fused_gold
        elif fused_conf:
            return "confounder", fused_conf
        else:
            return "bg", []
    else:
        # Single entry
        if memory_id in gold_ids:
            return "gold", [memory_id]
        elif memory_id in conf_ids:
            return "confounder", [memory_id]
        else:
            return "bg", []


def analyze_one_trace(
    trace_path: str,
    id_map: Dict,
    tokenizer,
    history_name: str,
    hybrid_item: Dict,
    num_required_gold_actual: int = 0,  # from hybrid data, independent of id_map matching
) -> List[QuestionResult]:
    """
    Analyze one agent trace file. Returns list of QuestionResult (one per QA event).

    Gold coverage: set intersection of required_gold_ids ∩ all_source_ids_in_final_context.
    This handles RD fusion correctly by expanding fused_member_ids, and uses set
    deduplication to prevent double-counting.
    """
    results = []
    if not os.path.exists(trace_path):
        return results

    with open(trace_path) as f:
        lines = f.readlines()

    required_gold_keys = id_map.get("original_gold_keys", [])
    gold_key_to_managed = id_map.get("gold_key_to_managed_ids", {})
    conf_managed_ids = id_map.get("conf_managed_ids", set())
    gold_key_to_text = id_map.get("gold_key_to_text", {})

    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event_type") != "question_answer":
            continue

        qid = event["question_id"]
        prompt = event.get("prompt", "")
        retrieved = event.get("retrieved", [])

        # Extract memory units from prompt (these are in the final context)
        memory_units = extract_memory_units_from_prompt(prompt)

        # Match units to retrieved entries
        matched = match_memory_units_to_retrieved(memory_units, retrieved)

        # ---- Collect ALL source memory IDs in final context ----
        all_context_source_ids = set()
        final_entry_ids = []
        for _, entry in matched:
            if entry is None:
                continue
            final_entry_ids.append(entry["memory_id"])
            all_context_source_ids.add(entry["memory_id"])
            meta = entry.get("metadata", {})
            if meta.get("answer_fused"):
                all_context_source_ids.update(meta.get("fused_member_ids", []))

        # ---- Gold coverage: stable gold keys × managed ID intersection ----
        retrieved_gold_keys = []
        unmatched_gold_keys = []
        for gk in required_gold_keys:
            managed_ids = gold_key_to_managed.get(gk, set())
            if managed_ids and (managed_ids & all_context_source_ids):
                retrieved_gold_keys.append(gk)
            else:
                unmatched_gold_keys.append(gk)

        num_required = num_required_gold_actual
        num_retrieved_gold = len(retrieved_gold_keys)
        any_gold = num_retrieved_gold > 0
        all_gold = (num_retrieved_gold == num_required) if num_required > 0 else True
        gold_recall = num_retrieved_gold / num_required if num_required > 0 else 1.0

        # ---- Confounder coverage ----
        retrieved_conf_managed = conf_managed_ids & all_context_source_ids

        # ---- Token-level classification ----
        conf_tokens = 0
        gold_tokens = 0
        bg_tokens = 0
        mixed_tokens = 0
        is_mixed = False

        for unit_text, entry in matched:
            unit_tokens = token_count(unit_text, tokenizer)
            if entry is None:
                mixed_tokens += unit_tokens
                continue

            entry_source_ids = {entry["memory_id"]}
            meta = entry.get("metadata", {})
            if meta.get("answer_fused"):
                entry_source_ids.update(meta.get("fused_member_ids", []))

            # Check if this entry contains gold (any gold key's managed IDs overlap)
            has_gold = any(
                bool(gold_key_to_managed.get(gk, set()) & entry_source_ids)
                for gk in required_gold_keys
            )
            has_conf = bool(conf_managed_ids & entry_source_ids)

            if has_gold and has_conf:
                mixed_tokens += unit_tokens
                is_mixed = True
            elif has_gold:
                gold_tokens += unit_tokens
            elif has_conf:
                conf_tokens += unit_tokens
            else:
                bg_tokens += unit_tokens

        # Total memory context tokens
        total_mem_tokens = gold_tokens + conf_tokens + bg_tokens + mixed_tokens

        conf_share = conf_tokens / total_mem_tokens if total_mem_tokens > 0 else float('nan')

        # Count entries in final context
        entry_count = len(final_entry_ids)

        # Check if evidence has gold but primary missed
        evidence_has_gold_primary_missed = False
        if not any_gold:
            for r in retrieved:
                if r.get("metadata", {}).get("memory_role") == "evidence":
                    ev_source_ids = {r["memory_id"]}
                    ev_meta = r.get("metadata", {})
                    if ev_meta.get("answer_fused"):
                        ev_source_ids.update(ev_meta.get("fused_member_ids", []))
                    ev_has_gold = any(
                        bool(gold_key_to_managed.get(gk, set()) & ev_source_ids)
                        for gk in required_gold_keys
                    )
                    if ev_has_gold:
                        evidence_has_gold_primary_missed = True
                        break

        # Empty context diagnostics
        empty_context_reason = ""
        if total_mem_tokens == 0:
            if len(memory_units) == 0:
                empty_context_reason = "no_memory_units_in_prompt"
            elif len(retrieved) == 0:
                empty_context_reason = "no_retrieved_entries"
            elif all(e is None for _, e in matched):
                empty_context_reason = "no_matched_entries"
            else:
                empty_context_reason = "all_units_zero_tokens"

        # Collect all managed gold IDs in final context (for display/debug)
        all_managed_gold_in_context = set()
        for gk in retrieved_gold_keys:
            all_managed_gold_in_context.update(gold_key_to_managed.get(gk, set()))

        # Duplicate gold match: count managed IDs appearing in context that belong to gold
        duplicate_gold_count = sum(
            1 for sid in all_context_source_ids
            if any(sid in gold_key_to_managed.get(gk, set()) for gk in required_gold_keys)
        ) - len(retrieved_gold_keys)

        qr = QuestionResult(
            manager_model="",
            method="",
            run_id=0,
            query_id=qid,
            history_name=history_name,
            gold_memory_ids=required_gold_keys,  # now stores stable gold keys
            retrieved_entry_ids=final_entry_ids,
            retrieved_source_memory_ids=sorted(all_context_source_ids),
            retrieved_confounder_ids=sorted(retrieved_conf_managed),
            num_required_gold=num_required,
            num_retrieved_gold=num_retrieved_gold,
            any_gold_retrieved=any_gold,
            all_gold_retrieved=all_gold and num_required > 0,
            gold_recall=gold_recall,
            memory_context_tokens=total_mem_tokens,
            confounder_tokens=conf_tokens,
            mixed_or_unattributed_tokens=mixed_tokens,
            confounder_token_share=conf_share,
            retrieved_entry_count=entry_count,
            final_memory_context="\n".join(memory_units),
            confusion_type=hybrid_item.get("confusion_type", ""),
            retrieved_bg_ids=[],
            bg_tokens=bg_tokens,
            gold_tokens=gold_tokens,
            evidence_has_gold_primary_missed=evidence_has_gold_primary_missed,
            is_mixed_fused=is_mixed,
            required_gold_ids=required_gold_keys,  # stable gold keys
            retrieved_gold_ids=retrieved_gold_keys,  # stable gold keys actually retrieved
            unmatched_required_gold_ids=unmatched_gold_keys,
            duplicate_gold_match_count=duplicate_gold_count,
            empty_context_reason=empty_context_reason,
        )
        results.append(qr)

    return results


def compute_summary_metrics(results: List[QuestionResult]) -> Dict:
    """Compute aggregate metrics from per-question results."""
    n = len(results)
    if n == 0:
        return {}

    any_gold_pct = sum(1 for r in results if r.any_gold_retrieved) / n * 100
    all_gold_pct = sum(1 for r in results if r.all_gold_retrieved) / n * 100
    gold_recall = np.mean([r.gold_recall for r in results]) * 100
    conf_share_values = [r.confounder_token_share for r in results
                         if not (isinstance(r.confounder_token_share, float) and np.isnan(r.confounder_token_share))]
    conf_share = np.mean(conf_share_values) * 100 if conf_share_values else 0.0
    entry_count_mean = np.mean([r.retrieved_entry_count for r in results])
    entry_count_std = np.std([r.retrieved_entry_count for r in results])
    mem_tokens_mean = np.mean([r.memory_context_tokens for r in results])
    mem_tokens_std = np.std([r.memory_context_tokens for r in results])

    # Percentiles for entry count
    entry_counts = sorted([r.retrieved_entry_count for r in results])
    p25 = entry_counts[int(n * 0.25)] if n > 0 else 0
    p50 = entry_counts[int(n * 0.50)] if n > 0 else 0
    p75 = entry_counts[int(n * 0.75)] if n > 0 else 0

    # Gold recall by num_required_gold
    recall_by_num = defaultdict(list)
    for r in results:
        recall_by_num[r.num_required_gold].append(r.gold_recall)
    recall_by_num_avg = {k: np.mean(v) * 100 for k, v in sorted(recall_by_num.items())}

    # By confusion type
    type_i = [r for r in results if r.confusion_type == "type_i"]
    type_ii = [r for r in results if r.confusion_type == "type_ii"]
    type_i_gold_recall = np.mean([r.gold_recall for r in type_i]) * 100 if type_i else 0
    type_ii_gold_recall = np.mean([r.gold_recall for r in type_ii]) * 100 if type_ii else 0

    # Over-256-token check
    over_256 = sum(1 for r in results if r.memory_context_tokens > 256)

    # Mixed fused count (RD only)
    mixed_count = sum(1 for r in results if r.is_mixed_fused)

    # Evidence has gold but primary missed
    ev_gold_missed = sum(1 for r in results if r.evidence_has_gold_primary_missed)

    return {
        "n_questions": n,
        "any_gold_retrieved_pct": any_gold_pct,
        "all_gold_retrieved_pct": all_gold_pct,
        "gold_recall_pct": gold_recall,
        "confounder_token_share_pct": conf_share,
        "retrieved_entry_count_mean": entry_count_mean,
        "retrieved_entry_count_std": entry_count_std,
        "retrieved_entry_count_p25": p25,
        "retrieved_entry_count_p50": p50,
        "retrieved_entry_count_p75": p75,
        "memory_context_tokens_mean": mem_tokens_mean,
        "memory_context_tokens_std": mem_tokens_std,
        "over_256_tokens_count": over_256,
        "gold_recall_by_num_required": recall_by_num_avg,
        "type_i_gold_recall_pct": type_i_gold_recall,
        "type_ii_gold_recall_pct": type_ii_gold_recall,
        "mixed_fused_count": mixed_count,
        "evidence_has_gold_primary_missed": ev_gold_missed,
    }


def generate_latex_table(all_summaries: Dict) -> str:
    """Generate LaTeX table for the main results."""
    model_order = ["gemma4-e4b", "Qwen3.5-4B"]
    method_order = ["add_all", "mem0", "evermemos", "relation_decision"]
    model_display = {"gemma4-e4b": "Gemma 4 E4B", "Qwen3.5-4B": "Qwen3.5-4B"}

    metrics = [
        ("any_gold_retrieved_pct", "Any gold retrieved (\\%)", "↑"),
        ("all_gold_retrieved_pct", "All gold retrieved (\\%)", "↑"),
        ("gold_recall_pct", "Gold recall (\\%)", "↑"),
        ("confounder_token_share_pct", "Confounder tokens (\\%)", "↓"),
        ("retrieved_entry_count_mean", "Retrieved entries", ""),
    ]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Retrieval diagnostics for $N=8$ confounder condition.}")
    lines.append(r"\label{tab:retrieval_diagnostics}")

    col_spec = "l" + "c" * (len(method_order) + 1)
    lines.append(r"\begin{tabular}{" + col_spec + "}")

    header = " & \\multicolumn{" + str(len(method_order) + 1) + "}{c}{Method} \\\\"
    # lines.append(header)
    # Simpler approach:
    header2 = "Manager model & Metric & " + " & ".join(METHOD_DISPLAY[m] for m in method_order) + r" \\"
    lines.append(r"\toprule")
    lines.append(header2)
    lines.append(r"\midrule")

    for mi, model in enumerate(model_order):
        for metric_key, metric_name, arrow in metrics:
            label = model_display[model] if metric_key == "any_gold_retrieved_pct" else ""
            row = f"{label} & {metric_name} {arrow}"
            for method in method_order:
                summary = all_summaries.get((model, method), {})
                val = summary.get(metric_key, 0)
                if metric_key in ("any_gold_retrieved_pct", "all_gold_retrieved_pct",
                                  "gold_recall_pct", "confounder_token_share_pct"):
                    row += f" & {val:.1f}"
                else:
                    row += f" & {val:.1f}"
            row += r" \\"
            lines.append(row)
        if mi < len(model_order) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Retrieval diagnostics analysis")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for results")
    parser.add_argument("--method", default=None,
                        help="Only analyze specific method (add_all, relation_decision, evermemos, mem0)")
    parser.add_argument("--model", default=None,
                        help="Only analyze specific model (gemma4-e4b, Qwen3.5-4B)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading hybrid golden data...")
    hybrid_data = load_hybrid_golden_data()
    print(f"  {len(hybrid_data)} questions with gold/confounder annotations")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    print(f"  Tokenizer: {type(tokenizer).__name__}")

    # Collect all history names from hybrid data
    all_history_names = sorted(hybrid_data.keys())

    all_results: Dict[Tuple[str, str], List[QuestionResult]] = {}
    all_summaries: Dict[Tuple[str, str], Dict] = {}

    for (model, method), cfg in RUN_CONFIGS.items():
        if args.model and args.model != model:
            continue
        if args.method and args.method != method:
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing: {model} / {method}")
        print(f"  Run: {cfg['run_id']}")
        print(f"  Ingest: {cfg['ingest_id']}")
        print(f"  Answer: {cfg['answer_id']}")

        # Build ingest path
        ingest_dir = os.path.join(ARTIFACTS_DIR, "stages", "ingest", method, cfg["ingest_id"])

        # Build gold/confounder ID mapping from ingest data
        print(f"  Building gold/confounder ID mapping from ingest (method={method})...")
        id_map = build_gold_confounder_id_map(ingest_dir, hybrid_data, all_history_names, method=method)
        # Count gold matching stats using stable gold keys
        n_exact = 0
        n_fuzzy = 0
        n_lost = 0
        for hn in all_history_names:
            im = id_map.get(hn, {})
            for mt in im.get("gold_match_type", {}).values():
                if mt == "exact":
                    n_exact += 1
                elif isinstance(mt, str) and mt.startswith("fuzzy"):
                    n_fuzzy += 1
                elif mt == "lost":
                    n_lost += 1
        n_total_gold = n_exact + n_fuzzy + n_lost
        print(f"  Total gold items across all histories: {n_total_gold}")
        print(f"    Exact matches: {n_exact}, Fuzzy: {n_fuzzy}, Lost: {n_lost}")

        # Analyze traces
        trace_dir = os.path.join(
            ARTIFACTS_DIR, "runs", cfg["run_id"],
            "answer", cfg["method_key"], cfg["answer_id"], "agent_trace"
        )

        results: List[QuestionResult] = []
        missing_traces = 0
        no_gold_ids = 0

        for hn in all_history_names:
            trace_path = os.path.join(trace_dir, f"{hn}.jsonl")
            if not os.path.exists(trace_path):
                missing_traces += 1
                continue

            hr = hybrid_data.get(hn, {})
            im = id_map.get(hn, {"original_gold_keys": [], "gold_key_to_managed_ids": {},
                                  "conf_managed_ids": set(), "confusion_type": ""})

            # Count gold items that couldn't be matched to any managed ID
            lost_keys = [gk for gk in im.get("original_gold_keys", [])
                        if not im.get("gold_key_to_managed_ids", {}).get(gk, set())]
            no_gold_ids += len(lost_keys)

            num_gold_actual = len(hr.get("golden_memory", []))
            q_results = analyze_one_trace(trace_path, im, tokenizer, hn, hr,
                                          num_required_gold_actual=num_gold_actual)
            for qr in q_results:
                qr.manager_model = model
                qr.method = method
            results.extend(q_results)

        print(f"  Total QA events: {len(results)}")
        print(f"  Missing traces: {missing_traces}")
        print(f"  Gold items not matched to any managed ID: {no_gold_ids}")

        # Include ALL analyzable questions.
        # For questions where gold IDs couldn't be found (e.g., mem0 rewrote text),
        # the metrics are 0 (gold not retrievable).
        results_with_gold = [r for r in results if r.num_required_gold > 0]
        print(f"  Questions with gold annotations: {len(results_with_gold)}")

        # For mem0: also report questions where gold wasn't found vs total
        if method == "mem0":
            n_gold_found = sum(1 for r in results_with_gold if r.num_retrieved_gold > 0)
            n_gold_total = len(results_with_gold)
            print(f"    Questions where gold was FOUND in memory store: {n_gold_found}/{n_gold_total}")

        all_results[(model, method)] = results_with_gold

        # Compute summary metrics
        summary = compute_summary_metrics(results_with_gold)
        all_summaries[(model, method)] = summary

        # Print summary
        print(f"\n  --- Summary for {model} / {method} ---")
        print(f"  Any gold retrieved: {summary['any_gold_retrieved_pct']:.1f}%")
        print(f"  All gold retrieved: {summary['all_gold_retrieved_pct']:.1f}%")
        print(f"  Gold recall: {summary['gold_recall_pct']:.1f}%")
        print(f"  Confounder token share: {summary['confounder_token_share_pct']:.1f}%")
        print(f"  Retrieved entries (mean): {summary['retrieved_entry_count_mean']:.1f}")
        print(f"  Memory context tokens (mean): {summary['memory_context_tokens_mean']:.1f}")
        print(f"  Over 256 tokens: {summary['over_256_tokens_count']}")
        print(f"  Mixed fused (RD): {summary['mixed_fused_count']}")
        print(f"  Evidence gold missed: {summary['evidence_has_gold_primary_missed']}")

        # Save per-question results
        per_q_path = os.path.join(
            args.output_dir, f"per_question_{model}_{method}.jsonl"
        )
        with open(per_q_path, "w") as f:
            for qr in results_with_gold:
                f.write(json.dumps(qr.__dict__, ensure_ascii=False) + "\n")
        print(f"  Per-question results saved to: {per_q_path}")

    # ---- Cross-checks and diagnostics ----

    print(f"\n{'='*60}")
    print("CROSS-CHECKS AND DIAGNOSTICS")

    # Check 1: Verify all methods cover the same 470 questions
    print("\n1. Question coverage:")
    for (model, method), results in sorted(all_results.items()):
        qids = set(r.query_id for r in results)
        print(f"  {model}/{method}: {len(qids)} unique questions")

    # Check 2: add_all results comparison between models (use history_name, not query_id)
    if ("gemma4-e4b", "add_all") in all_results and ("Qwen3.5-4B", "add_all") in all_results:
        g_add = all_results[("gemma4-e4b", "add_all")]
        q_add = all_results[("Qwen3.5-4B", "add_all")]
        g_sum = all_summaries[("gemma4-e4b", "add_all")]
        q_sum = all_summaries[("Qwen3.5-4B", "add_all")]

        # Compare per-question by history_name (stable across runs)
        g_by_hn = {r.history_name: r for r in g_add}
        q_by_hn = {r.history_name: r for r in q_add}
        common = set(g_by_hn) & set(q_by_hn)
        identical = sum(1 for hn in common
                        if g_by_hn[hn].any_gold_retrieved == q_by_hn[hn].any_gold_retrieved
                        and g_by_hn[hn].retrieved_entry_count == q_by_hn[hn].retrieved_entry_count)
        print(f"\n2. add_all consistency between models (by history_name):")
        print(f"  Common histories: {len(common)}")
        print(f"  Identical (any_gold+entry_count): {identical}/{len(common)}")
        print(f"  gemma4-e4b AnyGold={g_sum['any_gold_retrieved_pct']:.1f}%, Qwen3.5-4B={q_sum['any_gold_retrieved_pct']:.1f}%")
        if identical == len(common) and len(common) == 470:
            print("  >> add_all results are IDENTICAL between models — can be POOLED.")
        else:
            diff_hns = [hn for hn in common
                         if g_by_hn[hn].any_gold_retrieved != q_by_hn[hn].any_gold_retrieved]
            print(f"  >> Differences in {len(diff_hns)} questions" + (f": {diff_hns[:5]}" if diff_hns else ""))

    # Check 3: Over-256 tokens
    print("\n3. Context token limit check (should be 0 over 256):")
    for (model, method), summary in sorted(all_summaries.items()):
        over = summary.get("over_256_tokens_count", "N/A")
        mean = summary.get("memory_context_tokens_mean", 0)
        print(f"  {model}/{method}: over_256={over}, mean_tokens={mean:.1f}")

    # Check 4: Entry count distribution
    print("\n4. Retrieved entry count distribution:")
    for (model, method), summary in sorted(all_summaries.items()):
        print(f"  {model}/{method}: mean={summary['retrieved_entry_count_mean']:.1f}, "
              f"std={summary['retrieved_entry_count_std']:.1f}, "
              f"p25={summary['retrieved_entry_count_p25']}, "
              f"p50={summary['retrieved_entry_count_p50']}, "
              f"p75={summary['retrieved_entry_count_p75']}")

    # Check 5: Gold recall by num_required
    print("\n5. Gold recall by num_required_gold:")
    for (model, method), summary in sorted(all_summaries.items()):
        rec = summary.get("gold_recall_by_num_required", {})
        print(f"  {model}/{method}: {rec}")

    # Check 6: Type I vs Type II
    print("\n6. Confusion type breakdown:")
    for (model, method), summary in sorted(all_summaries.items()):
        print(f"  {model}/{method}: Type I={summary['type_i_gold_recall_pct']:.1f}%, "
              f"Type II={summary['type_ii_gold_recall_pct']:.1f}%")

    # Check 7: Mixed fused (RD)
    print("\n7. RD mixed gold/confounder fused entries:")
    for (model, method), summary in sorted(all_summaries.items()):
        if method == "relation_decision":
            print(f"  {model}: {summary['mixed_fused_count']}")

    # Check 8: Evidence gold missed
    print("\n8. Evidence has gold but primary missed:")
    for (model, method), summary in sorted(all_summaries.items()):
        print(f"  {model}/{method}: {summary['evidence_has_gold_primary_missed']}")

    # ---- Generate LaTeX table ----
    latex = generate_latex_table(all_summaries)
    latex_path = os.path.join(args.output_dir, "main_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nLaTeX table saved to: {latex_path}")

    # ---- Generate summary JSON ----
    summary_path = os.path.join(args.output_dir, "summary.json")
    # Convert tuple keys to strings for JSON
    serializable_summaries = {f"{m}__{mt}": s for (m, mt), s in all_summaries.items()}
    with open(summary_path, "w") as f:
        json.dump(serializable_summaries, f, indent=2, ensure_ascii=False)
    print(f"Summary JSON saved to: {summary_path}")

    # ---- Print final aggregated table ----
    print(f"\n{'='*60}")
    print("FINAL RESULTS TABLE")
    print(f"{'='*60}")

    model_order = ["gemma4-e4b", "Qwen3.5-4B"]
    method_order = ["add_all", "mem0", "evermemos", "relation_decision"]

    header = f"{'Manager model':<16} {'Metric':<30} " + "".join(f"{METHOD_DISPLAY[m]:>16}" for m in method_order)
    print(f"\n{header}")
    print("-" * len(header))

    for model in model_order:
        for metric_key, metric_name, _ in [
            ("any_gold_retrieved_pct", "Any gold retrieved (%) ↑", ""),
            ("all_gold_retrieved_pct", "All gold retrieved (%) ↑", ""),
            ("gold_recall_pct", "Gold recall (%) ↑", ""),
            ("confounder_token_share_pct", "Confounder tokens (%) ↓", ""),
            ("retrieved_entry_count_mean", "Retrieved entries", ""),
        ]:
            label = model if metric_key == "any_gold_retrieved_pct" else ""
            row = f"{label:<16} {metric_name:<30}"
            for method in method_order:
                s = all_summaries.get((model, method), {})
                val = s.get(metric_key, 0)
                row += f" {val:>15.1f}"
            print(row)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
