#!/usr/bin/env python3
"""
Build LME golden memory v2: per-evidence-session extraction with has_answer turn anchoring.

Key change from v1: one golden memory per evidence session (not flat list).
Each memory is {session_id, content}, golden_memory count == evidence_session_ids count.

Usage:
    PYTHONPATH=src uv run --no-sync python script/build_lme_golden_memory_v2.py \
        [--limit N] [--max-workers 20] [--out data/preprocessed/longmemeval_s_golden.json]
"""

import argparse
import json
import sys
import time
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prompts import render_prompt
from utils.llm_api import load_api_chat_completion


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_original_data(path: str = "data/raw_data/longmemeval_s_cleaned.json") -> list:
    with open(path) as f:
        return json.load(f)


def find_evidence_session_indices(item: dict) -> list[tuple[str, int]]:
    """Return [(session_id, index_in_haystack), ...] for each evidence session."""
    result = []
    hs_ids = item["haystack_session_ids"]
    for aid in item["answer_session_ids"]:
        if aid in hs_ids:
            result.append((aid, hs_ids.index(aid)))
    return result


def count_has_answer_turns(session: list) -> int:
    """Count turns with has_answer=True in a session."""
    return sum(1 for t in session if isinstance(t, dict) and t.get("has_answer"))


def is_abstention(item: dict) -> bool:
    """An item is abstention if its question_id ends with '_abs'.

    These questions have answers saying "not enough information" —
    the evidence is either empty or incomplete. Some _abs questions
    do have has_answer turns (partial evidence) but still can't be answered.
    """
    return item["question_id"].endswith("_abs")


# ---------------------------------------------------------------------------
# Session text formatting
# ---------------------------------------------------------------------------

def format_session_text(session: list) -> str:
    """Format one session's turns, highlighting has_answer=True turns with ⚑ markers."""
    lines = []
    for turn in session:
        if not isinstance(turn, dict):
            continue
        speaker = turn.get("speaker", turn.get("role", "unknown"))
        content = turn.get("content", "").strip()
        has_answer = turn.get("has_answer", False)

        if has_answer:
            lines.append(f"⚑ [{speaker}]: {content}")
        else:
            lines.append(f"[{speaker}]: {content}")

    return "\n".join(lines)


def format_has_answer_turns(session: list) -> str:
    """Extract only has_answer=True turns from a session, with speaker labels."""
    lines = []
    for turn in session:
        if not isinstance(turn, dict):
            continue
        if turn.get("has_answer"):
            speaker = turn.get("speaker", turn.get("role", "unknown"))
            content = turn.get("content", "").strip()
            lines.append(f"[{speaker}]: {content}")
    return "\n".join(lines) if lines else "(no marked turns)"


# ---------------------------------------------------------------------------
# LLM consolidation step
# ---------------------------------------------------------------------------

def consolidate_golden_memories(
    client,
    question: str,
    answer: str,
    question_type: str,
    question_date: str,
    session_gms: list[dict],  # [{content, session_date, has_answer_turns_text}]
    max_new_tokens: int = 1024,
) -> list[str] | None:
    """Consolidate per-session golden memories into a flat list of facts.

    Returns list of fact strings, or None on failure.
    LLM may merge, split, or drop facts freely.
    """
    # Build sessions summary
    summaries = []
    for i, sg in enumerate(session_gms):
        sdate = sg.get("session_date", "unknown")
        gm = sg.get("content") or "(extraction failed — no memory)"
        ha_text = sg.get("has_answer_turns_text", "(no marked turns)")
        summaries.append(
            f"Session {i+1} (date: {sdate}):\n"
            f"Extracted GM: {gm}\n"
            f"Key turns:\n{ha_text}\n"
        )
    sessions_summary = "\n---\n".join(summaries)

    prompt = render_prompt(
        "lme_golden_memory_consolidate_en.jinja",
        question=question,
        answer=answer,
        question_type=question_type,
        question_date=question_date,
        sessions_summary=sessions_summary,
    )

    messages = [{"role": "user", "content": prompt}]
    response = client.get_response_chat(
        messages, max_new_tokens=max_new_tokens, temperature=0.0
    )

    if response is None:
        return None

    # Parse flat JSON array of strings
    try:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return None
        # Filter out non-strings, strip whitespace
        result = [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
        return result
    except (json.JSONDecodeError) as e:
        print(f"  Consolidation JSON parse error: {e}")
        return None


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

def extract_one_golden_memory(
    client,
    question: str,
    answer: str,
    question_type: str,
    session_text: str,
    session_date: str,
    has_marked_turns: bool = True,
    max_new_tokens: int = 256,
) -> str | None:
    """Call LLM to distill one golden memory from one evidence session.

    Returns the content string, or None on failure.
    """
    prompt = render_prompt(
        "lme_golden_memory_distill_v2_en.jinja",
        question=question,
        answer=answer,
        question_type=question_type,
        session_text=session_text,
        session_date=session_date,
        has_marked_turns=has_marked_turns,
    )

    messages = [{"role": "user", "content": prompt}]
    response = client.get_response_chat(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    if response is None:
        return None

    # Parse JSON from response
    try:
        text = response.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Find the first complete JSON object (handle trailing text/gibberish)
        # Try to parse progressively shorter suffixes
        best_parse = None
        for end in range(len(text), 0, -1):
            candidate = text[:end].strip()
            try:
                parsed = json.loads(candidate)
                best_parse = parsed
                break
            except json.JSONDecodeError:
                continue

        if best_parse is None:
            raise json.JSONDecodeError("No valid JSON found", text, 0)

        gm = best_parse.get("golden_memory")
        if gm is None:
            return None
        if isinstance(gm, str) and gm.strip():
            return gm.strip()
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  JSON parse error: {e}, response: {response[:200]}")
        return None


def build_extraction_work_items(original: list) -> list[dict]:
    """Build flat list of per-session extraction tasks.

    Each work item: {question_id, question, answer, question_type,
                      session_id, session_date, session_text}
    """
    work_items = []
    for item in original:
        if is_abstention(item):
            continue

        qid = item["question_id"]
        question = item["question"]
        answer = item["answer"]
        qtype = item["question_type"]

        for sid, idx in find_evidence_session_indices(item):
            sess = item["haystack_sessions"][idx]
            date = item["haystack_dates"][idx] if idx < len(item.get("haystack_dates", [])) else "unknown"

            session_text = format_session_text(sess)
            has_marked = count_has_answer_turns(sess) > 0
            ha_turns_text = format_has_answer_turns(sess)

            work_items.append({
                "question_id": qid,
                "question": question,
                "answer": answer,
                "question_type": qtype,
                "session_id": sid,
                "session_date": date,
                "session_text": session_text,
                "has_marked_turns": has_marked,
                "has_answer_turns_text": ha_turns_text,
            })

    return work_items


# ---------------------------------------------------------------------------
# Verification (answer → judge)
# ---------------------------------------------------------------------------

def answer_with_golden_memory(
    client,
    golden_memories: list,
    question: str,
    question_time: str,
    max_new_tokens: int = 512,
) -> str | None:
    """Answer using the same template as run_exp_lme.py: pipeline_answer.jinja"""
    context_units = []
    for i, gm in enumerate(golden_memories):
        if isinstance(gm, dict):
            content = gm.get("content", "")
            date = gm.get("date", "")
        else:
            content = str(gm)
            date = ""
        if not content:
            continue
        unit = render_prompt(
            "pipeline_answer_memory_unit.jinja",
            index=i + 1,
            text=content,
            time=date,
            metadata={},
            show_time=bool(date),
        )
        context_units.append(unit)
    context_block = "\n".join(context_units)

    prompt = render_prompt(
        "pipeline_answer.jinja",
        context_block=context_block,
        question=question,
        question_time=question_time,
    )
    messages = [{"role": "user", "content": prompt}]
    response = client.get_response_chat(
        messages, max_new_tokens=max_new_tokens, temperature=0.0
    )
    if response is None:
        return None
    return response.strip()


def judge_answer(
    client,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    question_type: str = "",
    max_new_tokens: int = 128,
) -> bool:
    """Judge using same templates as run_exp_lme.py: pipeline_eval_system + pipeline_eval_oqa"""
    if candidate_answer is None:
        return False

    system_prompt = render_prompt("pipeline_eval_system.jinja")
    user_prompt = render_prompt(
        "pipeline_eval_oqa.jinja",
        question=question,
        reference=reference_answer,
        candidate=candidate_answer,
        use_cot=False,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.get_response_chat(
        messages, max_new_tokens=max_new_tokens, temperature=0.0
    )
    if response is None:
        return False
    text_lower = response.strip().lower()
    return "yes" in text_lower and "no" not in text_lower


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_extraction_phase(
    work_items: list[dict],
    gemma_client,
    max_workers: int = 20,
) -> dict[str, list[dict]]:
    """Phase 1: Extract one golden memory per evidence session.

    Returns: {question_id: [{session_id, content}, ...]}
    """
    # Group by question_id for progress tracking
    qid_to_items = defaultdict(list)
    for wi in work_items:
        qid_to_items[wi["question_id"]].append(wi)

    total_sessions = len(work_items)
    total_questions = len(qid_to_items)

    print(f"Phase 1: Extracting golden memories from {total_sessions} sessions "
          f"across {total_questions} questions...")

    results: dict[str, list[dict]] = defaultdict(list)
    completed = 0
    failed = 0

    def process_item(wi):
        content = extract_one_golden_memory(
            gemma_client,
            wi["question"],
            wi["answer"],
            wi["question_type"],
            wi["session_text"],
            wi["session_date"],
            wi["has_marked_turns"],
        )
        return {
            "question_id": wi["question_id"],
            "session_id": wi["session_id"],
            "content": content,
            "session_date": wi.get("session_date", ""),
            "has_answer_turns_text": wi.get("has_answer_turns_text", ""),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, wi): wi for wi in work_items}
        for future in as_completed(futures):
            result = future.result()
            qid = result["question_id"]
            results[qid].append(result)
            completed += 1

            if result["content"] is None:
                failed += 1

            if completed % 100 == 0:
                print(f"  Progress: {completed}/{total_sessions} sessions "
                      f"({failed} failed so far)")

    print(f"  Extraction done: {completed} sessions, {failed} failed")
    return dict(results)


def run_consolidation_phase(
    extraction_results: dict[str, list[dict]],
    original_data: list,
    client,
) -> dict[str, list[str]]:
    """Phase 1.5: Consolidate per-session GM into flat fact lists per question."""
    orig_by_qid = {item["question_id"]: item for item in original_data}
    total = len(extraction_results)
    print(f"\nPhase 1.5: Consolidating golden memories for {total} questions...")

    def consolidate_one(qid):
        orig = orig_by_qid.get(qid)
        gms = extraction_results[qid]
        if orig is None:
            # Fallback: keep extracted content as flat list
            facts = [gm["content"] for gm in gms if gm.get("content")]
            return qid, facts
        session_gms = []
        for gm in gms:
            session_gms.append({
                "content": gm.get("content"),
                "session_date": gm.get("session_date", ""),
                "has_answer_turns_text": gm.get("has_answer_turns_text", "(no marked turns)"),
            })
        qdate = orig.get("question_date", "unknown")
        qtype = orig.get("question_type", "")
        refined = consolidate_golden_memories(
            client, orig["question"], orig["answer"], qtype, qdate, session_gms
        )
        if refined is None or len(refined) == 0:
            # Consolidation failed or returned empty — keep original extractions
            refined = [gm["content"] for gm in gms if gm.get("content")]
        return qid, refined

    consolidated = {}
    completed = 0
    qids = list(extraction_results.keys())
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(consolidate_one, qid): qid for qid in qids}
        for future in as_completed(futures):
            qid, facts = future.result()
            consolidated[qid] = facts
            completed += 1
            if completed % 50 == 0:
                total_facts = sum(len(v) for v in consolidated.values())
                print(f"  Progress: {completed}/{total}, {total_facts} facts so far")

    total_facts = sum(len(v) for v in consolidated.values())
    print(f"  Consolidation done: {total_facts} total facts")

    # Phase 1.6: Match dates to each fact
    consolidated = _match_dates(consolidated, extraction_results, client)
    return consolidated


def _match_dates(
    consolidated: dict[str, list[str]],
    extraction_results: dict[str, list[dict]],
    client,
) -> dict[str, list[dict]]:
    """Match each consolidated fact to a date from its source evidence session."""
    total = len(consolidated)
    print(f"\nPhase 1.6: Matching dates for {total} questions...")

    def match_one(qid):
        facts = consolidated[qid]
        gms = extraction_results.get(qid, [])
        if not facts or not gms:
            return qid, [{"content": f, "date": ""} for f in facts]

        # Build context: per-session date + key turns
        session_info = []
        for gm in gms:
            sdate = gm.get("session_date", "")
            ha = gm.get("has_answer_turns_text", "")
            session_info.append(f"Date: {sdate}\nTurns: {ha[:500]}")

        prompt = f"""Match each golden memory fact to the date of the evidence session it came from.

Rules:
- Each fact must be matched to EXACTLY ONE date from the available sessions below.
- If a fact merges info from multiple sessions, pick the MOST RELEVANT date.
- If multiple facts come from the same session, they all get the same date.
- Order the output chronologically (earliest first) for knowledge-update questions.
- Only use dates from the sessions listed below.

FACTS:
{json.dumps(facts, indent=2)}

AVAILABLE SESSION DATES & CONTENT:
{chr(10).join(session_info)}

Return ONLY a JSON array:
[{{"content": "The user ...", "date": "2023/05/20"}}, ...]"""

        messages = [{"role": "user", "content": prompt}]
        response = client.get_response_chat(messages, max_new_tokens=1024, temperature=0.0)
        if response is None:
            return qid, [{"content": f, "date": ""} for f in facts]

        try:
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return qid, [{"content": e.get("content", ""), "date": e.get("date", "")} for e in parsed]
        except json.JSONDecodeError:
            pass
        return qid, [{"content": f, "date": ""} for f in facts]

    result = {}
    completed = 0
    qids = list(consolidated.keys())
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(match_one, qid): qid for qid in qids}
        for future in as_completed(futures):
            qid, matched = future.result()
            result[qid] = matched
            completed += 1
            if completed % 100 == 0:
                print(f"  Progress: {completed}/{total}")

    print(f"  Date matching done")
    return result


def _fill_missing_dates(
    final_results: list[dict],
    extraction_cache: dict[str, list[dict]],
    client,
) -> list[dict]:
    """Post-evaluation: use LLM to fill dates for facts missing one."""
    need_fix_qids = []
    for r in final_results:
        if r.get("abstention"):
            continue
        gms = r.get("golden_memory", [])
        if any(isinstance(gm, dict) and not gm.get("date") for gm in gms):
            need_fix_qids.append(r["question_id"])

    if not need_fix_qids:
        return final_results

    print(f"  Filling dates for {len(need_fix_qids)} questions...")
    fixed = 0
    for r in final_results:
        if r["question_id"] not in need_fix_qids:
            continue
        gms = extraction_cache.get(r["question_id"], [])
        if not gms:
            continue
        facts = r["golden_memory"]
        # Build session info
        session_info = []
        for gm in gms:
            sdate = gm.get("session_date", "")
            ha = gm.get("has_answer_turns_text", "")
            session_info.append(f"Date: {sdate}\nTurns: {ha[:500]}")

        prompt = f"""Match each fact to the date of the evidence session it came from.

FACTS:
{json.dumps([f['content'] if isinstance(f, dict) else f for f in facts], indent=2)}

AVAILABLE SESSION DATES & CONTENT:
{chr(10).join(session_info)}

Return ONLY a JSON array:
[{{"content": "...", "date": "YYYY/MM/DD"}}, ...]
Every fact MUST have a date."""

        messages = [{"role": "user", "content": prompt}]
        response = client.get_response_chat(messages, max_new_tokens=1024, temperature=0.0)
        if response is None:
            continue
        try:
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].startswith("```"): lines = lines[:-1]
                text = "\n".join(lines).strip()
            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) == len(facts):
                for i, entry in enumerate(parsed):
                    if isinstance(facts[i], dict):
                        facts[i]["date"] = entry.get("date", "")
                fixed += 1
        except json.JSONDecodeError:
            pass

    # Regex fallback for any remaining undated facts
    import re
    date_patterns = [
        (r'\b(20\d{2}/\d{2}/\d{2})\b', lambda m: m.group(1)),       # 2023/05/20
        (r'\b(\d{4}/\d{2}/\d{2})\b', lambda m: m.group(1)),          # 2023/05/20 (full)
        (r'\bon ([A-Z][a-z]+ \d{1,2}(?:st|nd|rd|th)?,? \d{4})\b', lambda m: m.group(1)),
        (r'\bin ([A-Z][a-z]+ \d{4})\b', lambda m: m.group(1)),
        (r'\bin (\d{4})\b', lambda m: m.group(1)),
    ]
    still_no_date = 0
    for r in final_results:
        for gm in r.get("golden_memory", []):
            if isinstance(gm, dict) and not gm.get("date") and gm.get("content"):
                for pat, fn in date_patterns:
                    m = re.search(pat, gm["content"])
                    if m:
                        gm["date"] = fn(m)
                        break
                if not gm["date"]:
                    still_no_date += 1
    if still_no_date:
        print(f"  Regex fallback: {still_no_date} facts still undated")

    print(f"  Fixed {fixed}/{len(need_fix_qids)} questions")
    return final_results


def _holistic_extract(
    client,
    orig_item: dict,
    cached_gms: list[dict],
    failed_answer: str = "",
    max_new_tokens: int = 512,
) -> list[str] | None:
    """Fallback: give ALL has_answer turns to LLM for direct holistic extraction.
    If failed_answer is provided, include it as feedback to avoid repeating the mistake."""
    all_turns = []
    for gm in cached_gms:
        sdate = gm.get("session_date", "")
        ha_text = gm.get("has_answer_turns_text", "")
        if ha_text and ha_text != "(no marked turns)":
            all_turns.append(f"Session date {sdate}:\n{ha_text}")

    if not all_turns:
        return None

    evidence = "\n\n---\n\n".join(all_turns)

    feedback = ""
    if failed_answer:
        feedback = f"""\nPREVIOUS ATTEMPT FAILED. The golden memories produced this WRONG answer: "{failed_answer}"
The CORRECT answer should be: "{orig_item['answer']}"
Fix the golden memories so they produce the correct answer.\n"""

    prompt = f"""Extract the minimal set of atomic facts about the user needed to answer this question correctly.{feedback}
Rules:
- Each fact = one atomic declarative sentence starting with "The user".
- Include dates where time is relevant.
- Split compound facts into separate entries.
- Only facts grounded in the evidence below.

QUESTION: {orig_item['question']}

GOLD ANSWER: {orig_item['answer']}

EVIDENCE TURNS:
{evidence}

Return ONLY a JSON array of strings:
["The user ...", "The user ..."]"""

    messages = [{"role": "user", "content": prompt}]
    response = client.get_response_chat(
        messages, max_new_tokens=max_new_tokens, temperature=0.0
    )
    if response is None:
        return None

    try:
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
    except json.JSONDecodeError:
        pass
    return None


def run_evaluation_phase(
    consolidated_results: dict[str, list[dict]],
    original_data: list,
    gemma_client,
    judge_client,
    extraction_cache: dict[str, list[dict]] | None = None,
) -> dict:
    """Phase 2: Answer using golden memories, judge vs gold answer.

    If a question fails, fall back to holistic extraction from all has_answer turns.
    """
    orig_by_qid = {item["question_id"]: item for item in original_data}
    total = len(consolidated_results)
    print(f"\nPhase 2: Evaluating {total} questions (answer → judge)...")

    final_results = []
    fallback_count = 0

    def evaluate_one(qid):
        nonlocal fallback_count
        orig = orig_by_qid.get(qid)
        if orig is None:
            return None
        facts = consolidated_results[qid]
        qt = orig["question_type"]
        qtime = orig.get("question_date", "unknown")
        max_retries = 5

        for attempt in range(max_retries + 1):
            # Retry answer step on transient failures
            candidate = None
            for _ in range(3):
                candidate = answer_with_golden_memory(
                    gemma_client, facts, orig["question"], qtime
                )
                if candidate is not None:
                    break

            if candidate is None:
                # Answer step completely failed — try holistic extraction
                if extraction_cache is not None and attempt == 0:
                    fallback_facts = _holistic_extract(
                        gemma_client, orig, extraction_cache.get(qid, []),
                        failed_answer="",
                    )
                    if fallback_facts:
                        facts = fallback_facts
                        fallback_count += 1
                        continue
                break

            judged_correct = judge_answer(
                judge_client, orig["question"], orig["answer"], candidate,
                question_type=qt,
            )
            # Knowledge-update: must have ≥2 GMs
            if judged_correct and qt == 'knowledge-update' and len(facts) < 2:
                judged_correct = False  # force holistic retry

            if judged_correct:
                break

            # Safety: never allow empty GM for non-abstention
            if len(facts) == 0 and extraction_cache is not None:
                cache_gms = extraction_cache.get(qid, [])
                facts = [{"content": gm["content"], "date": gm.get("session_date", "")}
                         for gm in cache_gms if gm.get("content")]

            # Failed — try holistic extraction with feedback
            if extraction_cache is not None:
                fallback_facts = _holistic_extract(
                    gemma_client, orig, extraction_cache.get(qid, []),
                    failed_answer=candidate or "",
                )
                if fallback_facts:
                    # Assign empty dates for now (will be matched later)
                    facts = [{"content": f, "date": ""} for f in fallback_facts]
                    fallback_count += 1
                    continue
            break  # no more strategies

        return {
            "question_id": qid,
            "question": orig["question"],
            "answer": orig["answer"],
            "question_type": qt,
            "evidence_session_ids": orig["answer_session_ids"],
            "abstention": False,
            "golden_memory": facts,
            "judged_correct": judged_correct,
        }

    qids = list(consolidated_results.keys())
    completed = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(evaluate_one, qid): qid for qid in qids}
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            final_results.append(result)
            completed += 1
            if completed % 50 == 0:
                c = sum(1 for r in final_results if r["judged_correct"])
                print(f"  Progress: {completed}/{total}, {c} correct so far")

    final_results.sort(key=lambda r: qids.index(r["question_id"]) if r["question_id"] in qids else 999)
    print(f"  Fallback rescued: {fallback_count} questions")

    type_stats = {}
    for r in final_results:
        qt = r["question_type"]
        if qt not in type_stats:
            type_stats[qt] = {"correct": 0, "total": 0}
        type_stats[qt]["total"] += 1
        if r["judged_correct"]:
            type_stats[qt]["correct"] += 1

    # Add abstention items
    for item in original_data:
        if is_abstention(item):
            final_results.append({
                "question_id": item["question_id"],
                "question": item["question"],
                "answer": item["answer"],
                "question_type": item["question_type"],
                "evidence_session_ids": item["answer_session_ids"],
                "abstention": True,
                "golden_memory": [],
                "judged_correct": True,
            })

    # Print accuracy summary
    active = [r for r in final_results if not r["abstention"]]
    correct = sum(1 for r in active if r["judged_correct"])
    total_facts = sum(len(r["golden_memory"]) for r in final_results)
    print(f"\n  Overall accuracy: {correct}/{len(active)} = {100*correct/len(active):.1f}%")
    print(f"  Total facts: {total_facts}")
    print(f"\n  Per-type accuracy:")
    for qt in sorted(type_stats.keys()):
        ts = type_stats[qt]
        acc = 100 * ts["correct"] / ts["total"] if ts["total"] > 0 else 0
        print(f"    {qt:<30s} {ts['correct']:>3d}/{ts['total']:<3d} = {acc:.1f}%")

    return final_results, type_stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build LME golden memory v2")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions to process (for testing)")
    parser.add_argument("--max-workers", type=int, default=16,
                        help="Max concurrent LLM calls for extraction")
    parser.add_argument("--out", type=str,
                        default="data/preprocessed/longmemeval_s_golden.json",
                        help="Output JSON path")
    parser.add_argument("--cache", type=str,
                        default="data/preprocessed/lme_extraction_cache.json",
                        help="Cache path for extraction results (skip extraction if exists)")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip extraction, use cached results")
    args = parser.parse_args()

    # Load data
    print("Loading original data...")
    original = load_original_data()

    if args.limit:
        original = original[:args.limit]
        print(f"  Limited to {args.limit} questions for testing")

    abstention_count = sum(1 for item in original if is_abstention(item))
    active_count = len(original) - abstention_count
    print(f"  Total: {len(original)} questions ({active_count} active, "
          f"{abstention_count} abstention)")

    # Initialize clients
    print("\nInitializing LLM clients...")
    gemma_client = load_api_chat_completion("gemma4-26B", async_=False)
    judge_client = load_api_chat_completion("gemma4-26B", async_=False)

    # Phase 1: Per-session extraction (with cache)
    t0 = time.time()
    cache_path = Path(args.cache)
    if args.skip_extraction and cache_path.exists():
        print(f"\nLoading cached extraction from {cache_path}...")
        with open(cache_path) as f:
            extraction_results = json.load(f)
        # Convert back to list-of-dicts per qid
        extraction_results = {
            qid: [dict(gm) for gm in gms]
            for qid, gms in extraction_results.items()
        }
        print(f"  Loaded {sum(len(v) for v in extraction_results.values())} sessions "
              f"across {len(extraction_results)} questions")
    else:
        work_items = build_extraction_work_items(original)
        print(f"  Extraction work items: {len(work_items)} sessions")
        extraction_results = run_extraction_phase(
            work_items, gemma_client, max_workers=args.max_workers
        )
        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(extraction_results, f, ensure_ascii=False)
        print(f"  Extraction cache saved to {cache_path}")
    t1 = time.time()
    print(f"  Phase 1 time: {t1 - t0:.1f}s")

    # Phase 1.5: Consolidate (dedup + denoise across sessions)
    consolidated_results = run_consolidation_phase(
        extraction_results,
        original,
        gemma_client,
    )
    t1c = time.time()
    print(f"  Phase 1.5 time: {t1c - t1:.1f}s")

    # Phase 2: Evaluate accuracy (answer with GM → judge vs gold)
    final_results, accuracy_stats = run_evaluation_phase(
        consolidated_results,
        original,
        gemma_client,
        judge_client,
        extraction_cache=extraction_results,
    )
    t2 = time.time()
    print(f"  Phase 2 time: {t2 - t1c:.1f}s")

    # Phase 2.5: Fill missing dates for fallback-generated facts
    final_results = _fill_missing_dates(final_results, extraction_results, gemma_client)
    t2c = time.time()
    print(f"  Phase 2.5 time: {t2c - t2:.1f}s")

    # Save
    print(f"\nSaving to {args.out}...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Summary
    total = len(final_results)
    active = [r for r in final_results if not r["abstention"]]
    correct = sum(1 for r in active if r["judged_correct"])
    gm_count = sum(len(r["golden_memory"]) for r in final_results)
    dated_count = sum(1 for r in final_results for gm in r.get("golden_memory", [])
                      if isinstance(gm, dict) and gm.get("date"))
    print(f"\nDone. Accuracy: {correct}/{len(active)} = {100*correct/len(active):.1f}% "
          f"({total} total, {total - len(active)} abstention). "
          f"Total facts: {gm_count} ({dated_count} with dates). "
          f"Total time: {t2 - t0:.1f}s")
    print(f"Output: {out_path.resolve()}")


if __name__ == "__main__":
    main()
