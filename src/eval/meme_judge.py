"""
MEME task-specific LLM judge (aligned with MEME-public code/eval/judge.py).

Each (question, agent_answer, gold) is scored with a per-task prompt.
ER uses deterministic substring match; Cas/Abs/Del support trivial-pass
classification when before-phase results are available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.thinking_text import split_embedded_thinking


BEFORE_COMMON_PROMPT = """You are a judge evaluating whether an AI agent's answer is semantically correct.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

Does the agent's answer contain the correct information matching the gold answer?

Rules:
- Focus on semantic equivalence, not exact wording
- If the gold value is present in the agent's answer, it is correct — regardless of any additional information, future possibilities, or extra details the agent mentions
- "Dentist every 6 months; dermatologist monthly (if you change residence)" → gold is "dentist (every 6 months)" → YES (core answer correct, extra info is irrelevant)
- "40 minutes" = "40 min" → YES
- "dentist appointment every 6 months" = "dentist (every 6 months)" → YES
- If agent says "I don't know" and gold is a specific value → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

CASCADE_D_PROMPT = """You are a judge evaluating whether an AI agent correctly reports a CASCADE-DEPENDENT value after an upstream change.

QUESTION: {question}
GOLD (new cascaded value): {gold}
AGENT: {agent_answer}

The upstream entity changed, triggering a cascade. The agent should now report ONLY the new value.

Rules:
- Agent states the new value as the sole, definitive answer → YES
- Agent lists BOTH old and new values together → NO
- Agent states only the old value → NO
- Agent lists multiple options or hedges ("might", "considering") → NO
- Agent uses future tense ("will change to X", "will be X") instead of confirming as current → NO
- "I don't know" → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

CASCADE_U_PROMPT = """You are a judge evaluating whether an AI agent appropriately handles UNCERTAINTY after an upstream entity changed.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

The upstream entity changed, making the previous answer potentially invalid. The agent should express uncertainty.

Rules:
- Agent says "I don't know" / "not sure" / "Unknown" / "None" / expresses uncertainty → YES
- Agent acknowledges the upstream change and questions validity → YES
- Agent confidently states the OLD value as definitive current fact → NO
- Agent provides the old value with no hedging or uncertainty → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

DELETION_PROMPT = """You are a judge evaluating whether an AI agent correctly handles a DELETED piece of information.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

The information was explicitly deleted/removed. The agent should not return it.

Rules:
- Agent says "I don't have that information" / "no longer available" / "Unknown" / "None" / similar WITHOUT revealing the deleted value → YES
- Agent indicates the item no longer exists or applies WITHOUT revealing the deleted value (e.g., "No food allergies" when the deleted value was a specific allergy) → YES
- Agent reveals the deleted value in any way (even while saying it was deleted/removed) → NO
- Agent returns the deleted value as if it still exists → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

MULTIHOP_PROMPT = """You are a judge evaluating whether an AI agent's answer contains specific values.

QUESTION: {question}
TARGET VALUES: {gold_values}
AGENT: {agent_answer}

For EACH target value, check whether it appears (exact or paraphrased) in the agent's answer.

Rules:
- The value appears in the answer (exact or paraphrased) → true
- The value is absent → false
- "I don't know" or "I don't have that information" → all false

Output ONLY JSON: {{"results": [{{"value": "...", "present": true/false}}], "reason": "one sentence"}}"""

MULTIPLE_UPDATE_PROMPT = """You are a judge evaluating whether an AI agent correctly recalls a sequence of historical values for an entity that changed multiple times.

QUESTION: {question}
GOLD (ordered history values, comma-separated): {gold}
AGENT: {agent_answer}

The user's information changed multiple times. The question asks for the FULL HISTORY in chronological order (earliest to latest).

Rules:
- Agent's answer must contain ALL gold values in the EXACT order specified
- Order matters: the values must appear earliest-to-latest as in the gold
- Extra surrounding text is fine, but the gold sequence must be preserved
- Missing any value → NO
- Wrong order → NO
- Only some values → NO
- "I don't know" → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

EXACT_PROMPT = """You are a judge evaluating whether an AI agent correctly recalls an EXACT value verbatim.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

Does the agent's answer contain the exact value?

Rules:
- Check if the gold value appears VERBATIM as a substring in the agent's answer
- If the gold value is fully contained in the answer, even with extra words before/after it → YES
- Example: gold="OOM killed by container runtime", agent="OOM killed by container runtime errors" → YES (gold is fully preserved)
- Minor formatting differences are acceptable (e.g., extra spaces, capitalization)
- Missing or substituted words WITHIN the gold value → NO
- "I don't know" or "I don't have that information" → NO
- A completely different value → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

JUDGE_PROMPTS = {
    "Tr": MULTIPLE_UPDATE_PROMPT,
    "Cas": CASCADE_D_PROMPT,
    "Abs": CASCADE_U_PROMPT,
    "Del": DELETION_PROMPT,
    "Agg": MULTIHOP_PROMPT,
    "ER": EXACT_PROMPT,
}

TRIVIAL_PASS_TASKS = frozenset({"Cas", "Abs", "Del"})


def task_base(task_type: str) -> str:
    return str(task_type or "").split(" (")[0]


def _parse_json_response(text: str) -> Dict[str, Any]:
    content = text.strip()
    _, tail = split_embedded_thinking(content)
    core = tail.strip()
    if core.startswith("```"):
        core = core.split("\n", 1)[1]
        if core.endswith("```"):
            core = core[:-3]
        core = core.strip()
    return json.loads(core)


def _er_substring_match(gold_value: str, agent_answer: str) -> Dict[str, Any]:
    gold_norm = " ".join(str(gold_value).lower().split())
    agent_norm = " ".join(str(agent_answer).lower().split())
    matched = gold_norm in agent_norm
    return {
        "u_pass": matched,
        "u_reason": (
            "Gold value found in answer (substring match)"
            if matched
            else "Gold value not found in answer (substring match)"
        ),
    }


@dataclass
class MemeJudgeResult:
    u_pass: bool
    u_reason: str = ""
    pass_type: Optional[str] = None
    u_pass_count: Optional[int] = None
    u_pass_total: Optional[int] = None
    u_pass_per_entity: Optional[Dict[str, bool]] = None


class MemeLLMJudge:
    """Async MEME judge using fact_memory's chat completion client."""

    def __init__(self, client, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    async def _call(self, prompt: str, max_new_tokens: int = 512) -> Dict[str, Any]:
        responses = await self.client.get_response_chat(
            [[{"role": "user", "content": prompt}]],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            max_concurrency=1,
            use_tqdm=False,
            verbose=False,
        )
        text = responses[0] if responses else ""
        if isinstance(text, dict):
            text = text.get("content") or text.get("response") or ""
        return _parse_json_response(str(text))

    async def u_check(
        self,
        *,
        question: str,
        gold_value: Any,
        agent_answer: str,
        task_type: str,
        phase: str = "after",
        max_new_tokens: int = 512,
    ) -> MemeJudgeResult:
        task_key = task_base(task_type)

        if task_key == "ER":
            raw = _er_substring_match(gold_value, agent_answer)
            return MemeJudgeResult(u_pass=raw["u_pass"], u_reason=raw["u_reason"])

        if task_key == "Tr":
            history = gold_value if isinstance(gold_value, list) else [gold_value]
            gold_str = ", ".join(str(v) for v in history)
            prompt = JUDGE_PROMPTS["Tr"].format(
                question=question, gold=gold_str, agent_answer=agent_answer
            )
            result = await self._call(prompt, max_new_tokens=max_new_tokens)
            return MemeJudgeResult(
                u_pass=bool(result.get("correct", False)),
                u_reason=str(result.get("reason", "")),
            )

        if phase == "before" and task_key != "Agg":
            template = BEFORE_COMMON_PROMPT
        else:
            template = JUDGE_PROMPTS[task_key]

        prompt = template.format(
            question=question,
            gold=gold_value,
            agent_answer=agent_answer,
        )
        result = await self._call(prompt, max_new_tokens=max_new_tokens)
        return MemeJudgeResult(
            u_pass=bool(result.get("correct", False)),
            u_reason=str(result.get("reason", "")),
        )

    async def u_check_multi(
        self,
        *,
        question: str,
        entity_values: Dict[str, Any],
        agent_answer: str,
        max_new_tokens: int = 512,
    ) -> MemeJudgeResult:
        entity_order = list(entity_values.keys())
        values_list = [entity_values[e] for e in entity_order]
        gold_values_str = ", ".join(f'"{v}"' for v in values_list)
        prompt = MULTIHOP_PROMPT.format(
            question=question,
            gold_values=gold_values_str,
            agent_answer=agent_answer,
        )
        result = await self._call(prompt, max_new_tokens=max_new_tokens)
        results_list = result.get("results", [])
        per_entity = {}
        for i, entity in enumerate(entity_order):
            per_entity[entity] = (
                results_list[i].get("present", False) if i < len(results_list) else False
            )
        pass_count = sum(1 for v in per_entity.values() if v)
        total = len(per_entity)
        return MemeJudgeResult(
            u_pass=pass_count == total,
            u_reason=str(result.get("reason", "")),
            u_pass_count=pass_count,
            u_pass_total=total,
            u_pass_per_entity=per_entity,
        )


def classify_trivial_pass(
    *,
    task_type: str,
    entity_key: str,
    before_pass_by_entity: Dict[str, bool],
    after_u_pass: bool,
) -> Optional[str]:
    """Return pass_type for Cas/Abs/Del after-phase answers; None for other tasks."""
    if task_base(task_type) not in TRIVIAL_PASS_TASKS:
        return None
    b_pass = before_pass_by_entity.get(entity_key, False)
    if after_u_pass and b_pass:
        return "real"
    if after_u_pass and not b_pass:
        return "trivial"
    if not after_u_pass and b_pass:
        return "knew_but_failed"
    return "never_knew"


def counts_toward_meme_score(row: Dict[str, Any]) -> bool:
    """Whether an *after* question counts as correct in the headline MEME score.

    Denominator is after questions only. Cas/Abs/Del after counts only when
    ``pass_type == "real"`` (after u_pass AND matching before u_pass on same entity).
    Other after types use raw ``u_pass``. Before rows never enter the score.
    """
    if row.get("judge_api_failed"):
        return False
    if str(row.get("phase", "after")) != "after":
        return False
    tb = task_base(str(row.get("question_type", "")))
    if tb in TRIVIAL_PASS_TASKS:
        return row.get("pass_type") == "real"
    return bool(row.get("u_pass", False))


def row_in_meme_phase_totals(row: Dict[str, Any]) -> Optional[str]:
    """Return ``before`` / ``after`` if the row belongs in phase denominators, else None."""
    if row.get("judge_api_failed"):
        return None
    phase = str(row.get("phase", "after"))
    if phase == "before" and task_base(str(row.get("question_type", ""))) == "ER":
        return None
    return phase if phase in ("before", "after") else None


def aggregate_meme_metrics(judged_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute MEME metrics from judged pred rows.

    ``meme_score`` = after-only accuracy (denominator = after questions). Cas/Abs/Del
    after counts only ``pass_type == "real"``. ``meme_score_raw`` = all after u_pass.
    ``meme_score_judge_totals`` = (before_pass + after_pass) / (before + after) as in
    MEME-public ``judge.py`` GRAND TOTAL (diagnostic; includes trivial after passes).
    """
    per_type: Dict[str, Dict[str, int]] = {}
    trivial_analysis: Dict[str, Dict[str, int]] = {
        t: {"total": 0, "real_pass": 0, "trivial_pass": 0, "knew_but_failed": 0, "never_knew": 0}
        for t in TRIVIAL_PASS_TASKS
    }

    before_pass = 0
    before_total = 0
    after_pass = 0
    after_total = 0
    before_pass_raw = 0
    after_pass_raw = 0

    for row in judged_rows:
        q_type = str(row.get("question_type", "unknown"))
        tb = task_base(q_type)
        ok_raw = bool(row.get("u_pass", False))

        bucket = per_type.setdefault(
            q_type, {"correct": 0, "judged": 0, "api_failed": 0}
        )
        if row.get("judge_api_failed"):
            bucket["api_failed"] += 1
            continue
        # per_type accuracy: after questions only (same denominator rule as meme_score)
        if str(row.get("phase", "after")) == "after":
            bucket["judged"] += 1
            if counts_toward_meme_score(row):
                bucket["correct"] += 1

        phase_slot = row_in_meme_phase_totals(row)
        if phase_slot == "before":
            before_total += 1
            if ok_raw:
                before_pass += 1
                before_pass_raw += 1
        elif phase_slot == "after":
            after_total += 1
            if counts_toward_meme_score(row):
                after_pass += 1
            if ok_raw:
                after_pass_raw += 1
            pt = row.get("pass_type")
            if tb in TRIVIAL_PASS_TASKS and pt:
                ta = trivial_analysis[tb]
                ta["total"] += 1
                if pt == "real":
                    ta["real_pass"] += 1
                elif pt == "trivial":
                    ta["trivial_pass"] += 1
                elif pt == "knew_but_failed":
                    ta["knew_but_failed"] += 1
                else:
                    ta["never_knew"] += 1

    total_after = after_total
    total_pass_after = after_pass
    total_pass_after_raw = after_pass_raw
    total_judge_style = after_total + before_total
    total_pass_judge_style = after_pass_raw + before_pass_raw
    per_type_out = {}
    for q_type, v in per_type.items():
        judged = v["judged"]
        per_type_out[q_type] = {
            "accuracy": (v["correct"] / judged) if judged else 0.0,
            "correct": v["correct"],
            "judged": judged,
            "api_failed": v["api_failed"],
        }

    return {
        "n_samples": len(judged_rows),
        "judged_count": total_after,
        "api_failure_count": sum(v["api_failed"] for v in per_type.values()),
        "overall_accuracy": (total_pass_after / total_after) if total_after else 0.0,
        "meme_score": (total_pass_after / total_after) if total_after else 0.0,
        "meme_score_raw": (total_pass_after_raw / total_after) if total_after else 0.0,
        "meme_score_judge_totals": (
            (total_pass_judge_style / total_judge_style) if total_judge_style else 0.0
        ),
        "before_pass": before_pass,
        "before_total": before_total,
        "after_pass": after_pass,
        "after_total": after_total,
        "before_pass_raw": before_pass_raw,
        "after_pass_raw": after_pass_raw,
        "per_type": per_type_out,
        "trivial_analysis": trivial_analysis,
        "scoring_note": (
            "meme_score: after-only; Cas/Abs/Del after needs pass_type=real; "
            "meme_score_judge_totals: (before+after) raw u_pass like judge.py"
        ),
    }
