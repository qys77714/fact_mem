#!/usr/bin/env python3
"""relation_decision 实验实时聚合看板。

扫描 <experiment_run>/memory_trace/relation_decision/*.jsonl，聚合内部运行状态，
重点是三类 LLM/分类调用：classify(五分类) / verify(LLM 复核) / answer_fuse(融合)。

用法：
  # 自动定位最新的 relation_decision trace 目录
  uv run --no-sync python script/monitor_relation_decision.py

  # 指定 trace 目录或实验 run 根目录
  uv run --no-sync python script/monitor_relation_decision.py --trace-dir experiment/<run>/memory_trace/relation_decision

  # 单次输出（不循环刷新），适合管道/重定向
  uv run --no-sync python script/monitor_relation_decision.py --once

  # 刷新间隔（秒）
  uv run --no-sync python script/monitor_relation_decision.py --interval 3

延迟说明：
  per-call latency_ms 来自代码埋点（classify/verify/fuse 三处），仅对加埋点之后启动的
  运行可得；旧运行无此字段时该列显示 n/a。整体吞吐 calls/s 用记录时间戳跨度估算，任何运行都可用。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

EXPERIMENT_ROOT = "experiment"
METHOD_SUBDIR = os.path.join("memory_trace", "relation_decision")

PURPOSE_CLASSIFY = "lme_candidate_relation_decision_classify_relation"
PURPOSE_VERIFY = "lme_candidate_relation_decision_verify_relation"
PURPOSE_FUSE = "lme_candidate_relation_decision_answer_fuse"

LABELS = ["IND", "EQV", "NSO", "OSN", "CON"]


def find_latest_trace_dir() -> Optional[Path]:
    """在 experiment/ 下找最新修改的 relation_decision trace 目录。"""
    candidates = glob.glob(os.path.join(EXPERIMENT_ROOT, "*", METHOD_SUBDIR))
    dirs = [Path(c) for c in candidates if os.path.isdir(c)]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def _percentile(sorted_vals, q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


class Stats:
    def __init__(self) -> None:
        self.episodes: dict[str, dict] = {}  # filename -> {records, last_ts}
        self.calls = Counter()               # purpose -> count
        self.errors = Counter()              # purpose -> error count
        self.classify_labels = Counter()     # label -> count (classifier 原始预测)
        self.verify_by_relation = defaultdict(lambda: [0, 0])  # relation -> [correct, total]
        self.latency = defaultdict(list)     # purpose -> [ms,...]
        self.memops = Counter()              # operation -> count
        self.ts_min: Optional[datetime] = None
        self.ts_max: Optional[datetime] = None
        self.backend_seen = set()
        # 非 IND classify 明细：list of {episode, old, new, label, ts}
        self.non_ind: list[dict] = []
        # verify 结果按 (old,new,relation) → correct，用于给 non_ind 标注最终是否保留
        self.verify_outcome: dict[tuple, bool] = {}

    def _track_ts(self, rec: dict) -> None:
        ts = rec.get("timestamp")
        if not ts:
            return
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            return
        if self.ts_min is None or dt < self.ts_min:
            self.ts_min = dt
        if self.ts_max is None or dt > self.ts_max:
            self.ts_max = dt

    def add_record(self, fname: str, rec: dict) -> None:
        self._track_ts(rec)
        ep = self.episodes.setdefault(fname, {"records": 0, "last_ts": ""})
        ep["records"] += 1
        ep["last_ts"] = rec.get("timestamp", ep["last_ts"])

        etype = rec.get("event_type")
        if etype == "memory_operation":
            self.memops[rec.get("operation", "?")] += 1
            return
        if etype != "llm_interaction":
            return

        purpose = rec.get("purpose", "?")
        self.calls[purpose] += 1
        if rec.get("status") == "error" or rec.get("error"):
            self.errors[purpose] += 1

        meta = rec.get("metadata") or {}
        lat = meta.get("latency_ms")
        if isinstance(lat, (int, float)):
            self.latency[purpose].append(float(lat))

        resp = rec.get("response")
        if purpose == PURPOSE_CLASSIFY:
            if meta.get("backend"):
                self.backend_seen.add(meta["backend"])
            label = None
            if isinstance(resp, dict):
                label = resp.get("relation")
            if label in LABELS:
                self.classify_labels[label] += 1
            if label in ("EQV", "NSO", "OSN", "CON"):
                old, new = _extract_old_new(rec)
                self.non_ind.append({
                    "episode": fname.replace(".jsonl", ""),
                    "old": old,
                    "new": new,
                    "label": label,
                    "ts": rec.get("timestamp", ""),
                })
        elif purpose == PURPOSE_VERIFY:
            rel = meta.get("relation", "?")
            correct = _extract_verify_correct(resp)
            bucket = self.verify_by_relation[rel]
            bucket[1] += 1
            if correct:
                bucket[0] += 1
            old, new = _extract_old_new(rec)
            if old is not None:
                self.verify_outcome[(old, new, rel)] = correct


def _extract_old_new(rec: dict) -> tuple:
    """从 trace 记录的 request.messages 还原 (old, new) 文本。

    classifier backend: 单条 user 消息 "old: ...\\nnew: ..."。
    verify: system + user，user 内含 OLD FACT / NEW FACT 段。
    """
    import re
    msgs = ((rec.get("request") or {}).get("messages")) or []
    contents = [m.get("content", "") for m in msgs if isinstance(m, dict)]
    text = "\n".join(c for c in contents if isinstance(c, str))
    # classifier 紧凑格式
    m = re.search(r"old:\s*(.*?)\nnew:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # verify/LLM 模板格式：OLD FACT: ... NEW FACT: ...
    old = new = None
    mo = re.search(r"OLD FACT:\s*(.*?)(?:\n\n|\nNEW FACT:)", text, re.IGNORECASE | re.DOTALL)
    mn = re.search(r"NEW FACT:\s*(.*?)(?:\n\n|\nPREDICTED|$)", text, re.IGNORECASE | re.DOTALL)
    if mo:
        old = mo.group(1).strip()
    if mn:
        new = mn.group(1).strip()
    return old, new


def _extract_verify_correct(resp) -> bool:
    payload = resp
    if isinstance(payload, (list, tuple)):
        payload = payload[0] if payload else None
    if isinstance(payload, dict):
        return bool(payload.get("correct"))
    if isinstance(payload, str):
        import re
        m = re.search(r"\{.*\}", payload, re.DOTALL)
        if m:
            try:
                return bool(json.loads(m.group(0)).get("correct"))
            except (ValueError, TypeError):
                return False
    return False


def scan(trace_dir: Path) -> Stats:
    st = Stats()
    for fp in sorted(trace_dir.glob("*.jsonl")):
        fname = fp.name
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 末行可能写到一半
                    st.add_record(fname, rec)
        except OSError:
            continue
    return st


def _fmt_lat(vals) -> str:
    if not vals:
        return "n/a"
    s = sorted(vals)
    avg = sum(s) / len(s)
    return f"avg {avg:7.1f}  p50 {_percentile(s,0.5):7.1f}  p95 {_percentile(s,0.95):7.1f} ms"


def render(st: Stats, trace_dir: Path) -> str:
    lines = []
    W = 78
    lines.append("=" * W)
    lines.append(f" relation_decision 监控  |  {datetime.now().strftime('%H:%M:%S')}")
    lines.append(f" trace: {trace_dir}")
    lines.append("=" * W)

    # episode 进度（每个 episode 一个 <name>_<phase>.jsonl 文件）
    n_files = len(st.episodes)
    total_records = sum(e["records"] for e in st.episodes.values())
    span_s = 0.0
    if st.ts_min and st.ts_max:
        span_s = (st.ts_max - st.ts_min).total_seconds()
    total_calls = sum(st.calls.values())
    cps = (total_calls / span_s) if span_s > 0 else 0.0
    lines.append(f" episode 文件: {n_files:4d}    trace 记录: {total_records:7d}    "
                 f"耗时跨度: {span_s:6.1f}s")
    lines.append(f" LLM/分类调用总数: {total_calls:7d}    吞吐: {cps:6.1f} calls/s")
    if st.backend_seen:
        lines.append(f" relation backend: {', '.join(sorted(st.backend_seen))}")
    lines.append("-" * W)

    # 三类调用：次数 + 延迟
    lines.append(" 调用类型           次数    错误   延迟")
    label_map = [
        ("classify", PURPOSE_CLASSIFY),
        ("verify  ", PURPOSE_VERIFY),
        ("fuse    ", PURPOSE_FUSE),
    ]
    for short, purpose in label_map:
        n = st.calls.get(purpose, 0)
        err = st.errors.get(purpose, 0)
        lat = _fmt_lat(st.latency.get(purpose, []))
        lines.append(f"  {short}        {n:7d} {err:6d}   {lat}")
    lines.append("-" * W)

    # classify 五分类标签分布
    total_lbl = sum(st.classify_labels.values())
    lines.append(f" classify 五分类标签分布 (共 {total_lbl}):")
    if total_lbl:
        parts = []
        for lab in LABELS:
            c = st.classify_labels.get(lab, 0)
            pct = (c / total_lbl * 100) if total_lbl else 0
            parts.append(f"{lab} {c}({pct:4.1f}%)")
        # 两个一行排版
        lines.append("   " + "   ".join(parts))
        non_ind = total_lbl - st.classify_labels.get("IND", 0)
        nr = (non_ind / total_lbl * 100) if total_lbl else 0
        lines.append(f"   非 IND（进入 verify 复核）: {non_ind} ({nr:.1f}%)")
    else:
        lines.append("   （暂无）")
    lines.append("-" * W)

    # verify 复核：各关系通过率
    if st.verify_by_relation:
        lines.append(" verify 复核通过率（correct=保留该关系，否则退回 IND）:")
        tot_ok = tot_all = 0
        for rel in ["CON", "OSN", "NSO", "EQV"]:
            ok, all_ = st.verify_by_relation.get(rel, [0, 0])
            tot_ok += ok
            tot_all += all_
            if all_:
                rate = ok / all_ * 100
                lines.append(f"   {rel}:  {ok:4d}/{all_:<4d}  通过 {rate:5.1f}%")
        if tot_all:
            lines.append(f"   合计: {tot_ok}/{tot_all}  通过 {tot_ok/tot_all*100:.1f}%  "
                         f"否决 {(tot_all-tot_ok)/tot_all*100:.1f}%")
        lines.append("-" * W)

    # 写库操作
    if st.memops:
        lines.append(" 写库操作:")
        items = sorted(st.memops.items(), key=lambda x: -x[1])
        row = "   " + "   ".join(f"{op}:{c}" for op, c in items[:8])
        lines.append(row)
        lines.append("-" * W)

    lines.append(f" (latency n/a 表示该运行启动时尚无埋点；吞吐始终可用)")
    lines.append(f" 非 IND 判定明细共 {len(st.non_ind)} 条，用 --detail 查看 / --dump <file> 导出")
    lines.append("=" * W)
    return "\n".join(lines)


def _verify_tag(st: Stats, item: dict) -> str:
    """给一条 non_ind 明细标注 verify 最终结果。"""
    key = (item["old"], item["new"], item["label"])
    outcome = st.verify_outcome.get(key)
    if outcome is None:
        return "?待复核 "
    return "✓保留   " if outcome else "✗退回IND"


def render_detail(st: Stats, label_filter: Optional[str], limit: int) -> str:
    """逐条列出被判为非 IND 的 fact 对（含 verify 最终结果）。"""
    items = st.non_ind
    if label_filter:
        items = [it for it in items if it["label"] == label_filter.upper()]
    total = len(items)
    shown = items[-limit:] if limit and limit > 0 else items
    lines = []
    W = 100
    lines.append("=" * W)
    title = f" 非 IND 判定明细  共 {total} 条"
    if label_filter:
        title += f"（仅 {label_filter.upper()}）"
    if limit and total > len(shown):
        title += f"，显示最近 {len(shown)} 条"
    lines.append(title)
    lines.append(" 标注: ✓保留=verify 通过建边 | ✗退回IND=verify 否决 | ?待复核=verify 尚未跑到")
    lines.append("=" * W)
    for it in shown:
        tag = _verify_tag(st, it)
        old = (it["old"] or "")[:80]
        new = (it["new"] or "")[:80]
        ep = it["episode"]
        lines.append(f"[{it['label']}] {tag}  {ep}")
        lines.append(f"    old: {old}")
        lines.append(f"    new: {new}")
    lines.append("=" * W)
    # 按标签的保留/否决小计
    by_label = defaultdict(lambda: [0, 0, 0])  # label -> [kept, rejected, pending]
    for it in items:
        o = st.verify_outcome.get((it["old"], it["new"], it["label"]))
        b = by_label[it["label"]]
        if o is True:
            b[0] += 1
        elif o is False:
            b[1] += 1
        else:
            b[2] += 1
    for lab in ["CON", "OSN", "NSO", "EQV"]:
        if lab in by_label:
            k, r, p = by_label[lab]
            lines.append(f" {lab}: 保留 {k}  退回 {r}  待复核 {p}")
    lines.append("=" * W)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="relation_decision 实时聚合监控看板")
    ap.add_argument("--trace-dir", default=None,
                    help="trace 目录或实验 run 根目录；缺省自动定位最新")
    ap.add_argument("--interval", type=float, default=3.0, help="刷新间隔秒")
    ap.add_argument("--once", action="store_true", help="只输出一次，不循环刷新")
    ap.add_argument("--detail", action="store_true",
                    help="显示被判为非 IND 的 fact 明细（old/new/label + verify 结果）")
    ap.add_argument("--label", default=None,
                    help="明细只看某个标签：CON/OSN/NSO/EQV")
    ap.add_argument("--limit", type=int, default=40,
                    help="明细最多显示最近 N 条（0=全部），默认 40")
    ap.add_argument("--dump", default=None,
                    help="把全部非 IND 明细导出为 JSONL 到指定文件后退出")
    args = ap.parse_args()

    if args.trace_dir:
        td = Path(args.trace_dir)
        # 允许传实验 run 根目录，自动补 memory_trace/relation_decision
        if td.is_dir() and not list(td.glob("*.jsonl")):
            cand = td / METHOD_SUBDIR
            if cand.is_dir():
                td = cand
    else:
        td = find_latest_trace_dir()
        if td is None:
            print("未找到 relation_decision trace 目录（experiment/*/memory_trace/relation_decision）。"
                  "\n实验是否已进入 run 阶段？或用 --trace-dir 指定。", file=sys.stderr)
            return 1

    if not td.is_dir():
        print(f"trace 目录不存在: {td}", file=sys.stderr)
        return 1

    if args.dump:
        st = scan(td)
        with open(args.dump, "w", encoding="utf-8") as f:
            for it in st.non_ind:
                rec = dict(it)
                rec["verify"] = st.verify_outcome.get((it["old"], it["new"], it["label"]))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"已导出 {len(st.non_ind)} 条非 IND 明细 -> {args.dump}")
        return 0

    if args.once:
        st = scan(td)
        if args.detail:
            print(render_detail(st, args.label, args.limit))
        else:
            print(render(st, td))
        return 0

    try:
        while True:
            st = scan(td)
            os.system("clear")
            if args.detail:
                print(render_detail(st, args.label, args.limit))
            else:
                print(render(st, td))
            print(f"\n每 {args.interval:.0f}s 刷新，Ctrl-C 退出。")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已退出监控。")
        return 0


if __name__ == "__main__":
    sys.exit(main())
