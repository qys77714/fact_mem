"""实验身份与阶段指纹（Experiment Identity & Stage Fingerprints）。

面向 LME 实验流水线（extract → ingest → generate → evaluate）提供：

1. 确定性的 canonical JSON / sha256 hash：任意 JSON/YAML 风格的嵌套配置
   （dict / list / tuple / set / Path / 标量）都能被规范化为稳定字符串，
   字典键顺序不影响结果，集合按排序后的规范表示写出。
2. ``ExperimentIdentity``：由 resolved_config 推导出可读且路径安全的
   slug、``run_id``（``{slug}--{8位hash}``）与 ``run_root``，并可
   ``materialize()`` 写出 ``manifest.json`` / ``run.yaml`` 两个可复现文件。
   写入幂等：同一 run_root 已存在且 resolved_config_hash 不同则拒绝覆盖。
3. 阶段指纹：``candidate_fingerprint`` / ``ingest_fingerprint`` /
   ``answer_fingerprint`` / ``judge_fingerprint``，采用最小依赖原则——
   每个阶段的指纹只吸收真正影响该阶段产出的配置字段，不相关字段的变化
   不会污染指纹（例如 answer 阶段的参数变化不应影响 candidate/ingest）。

仅依赖标准库与项目已有依赖（PyYAML）；不发起任何网络请求，不加载任何模型。
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
import re
import secrets
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows/non-POSIX explicitly unsupported
    fcntl = None

__all__ = [
    "canonicalize",
    "canonical_json",
    "sha256_hash",
    "short_hash",
    "candidate_fingerprint",
    "ingest_fingerprint",
    "answer_fingerprint",
    "judge_fingerprint",
    "ExperimentIdentity",
    "MaterializedRun",
    "ArtifactLayout",
]

SCHEMA_VERSION = 1
DEFAULT_ARTIFACTS_ROOT = Path("artifacts/runs")
DEFAULT_STAGES_ROOT = Path("artifacts/stages")
_RUN_ID_HASH_LEN = 8
_MATERIALIZE_LOCK = threading.RLock()
_SAFE_METHOD_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_STAGE_LOCKS_GUARD = threading.Lock()
_STAGE_THREAD_LOCKS: dict[Path, threading.Lock] = {}


# ---------------------------------------------------------------------------
# canonical JSON / hash
# ---------------------------------------------------------------------------


def canonicalize(value: Any) -> Any:
    """把任意嵌套值规范化为可确定序列化的 JSON 兼容结构。

    - Mapping：键转为 str，递归规范化（顺序不重要，由 ``json.dumps(sort_keys=True)`` 保证）。
    - list/tuple：保持顺序，逐元素递归规范化。
    - set/frozenset：规范化后按其 JSON 表示排序，消除迭代顺序的不确定性。
    - Path：转为 POSIX 风格字符串。
    - str/int/float/bool/None：原样返回。
    - 其他未知类型：回退为 ``str()``，保证仍然可序列化且确定。
    """
    if isinstance(value, Mapping):
        return {str(k): canonicalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonicalize(v) for v in value]
    if isinstance(value, (set, frozenset)):
        items = [canonicalize(v) for v in value]
        items.sort(key=lambda v: json.dumps(v, sort_keys=True, default=str))
        return items
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    """返回 ``value`` 的确定性 canonical JSON 字符串（排序键、紧凑分隔符）。"""
    return json.dumps(
        canonicalize(value),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_hash(value: Any) -> str:
    """canonical JSON 的完整 sha256 十六进制摘要。"""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def short_hash(value: Any, length: int = _RUN_ID_HASH_LEN) -> str:
    """sha256 摘要的前 ``length`` 位十六进制字符（默认 8 位）。"""
    return sha256_hash(value)[:length]


# ---------------------------------------------------------------------------
# 配置字段访问辅助
# ---------------------------------------------------------------------------


def _get_path(config: Any, dotted: str, default: Any = None) -> Any:
    """按 ``a.b.c`` 形式的点分路径从嵌套对象中取值，缺失则返回 default。

    支持 ``Mapping``（dict）和带有属性访问的对象（如 Pydantic BaseModel）。
    """
    node = config
    for part in dotted.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif hasattr(node, part):
            node = getattr(node, part)
        else:
            return default
    return node


def _extract_fields(config: Mapping, dotted_paths: Sequence[str]) -> dict:
    return {path: _get_path(config, path) for path in dotted_paths}


# ---------------------------------------------------------------------------
# 阶段指纹：最小依赖字段表
# ---------------------------------------------------------------------------

_CANDIDATE_FIELDS: Sequence[str] = (
    "experiment.benchmark",
    "models.extract",
    "extract.candidate_suffix",
    "extract.granularity",
    "extract.turn_overlap",
    "extract.language",
    "extract.aspect_templates",
    "token_limits.extract_max_new_tokens",
)

_INGEST_COMMON_FIELDS: Sequence[str] = (
    "models.manager",
    "models.embedding",
    "token_limits.ingest_relation_max_new_tokens",
    "token_limits.ingest_manager_max_new_tokens",
    "token_limits.fusion_max_new_tokens",
    "prompts.relation_user_en",
    "prompts.relation_user_zh",
)

_ANSWER_FIELDS: Sequence[str] = (
    "experiment.benchmark",
    "models.answer",
    "models.embedding",
    "generate.retrieve_topk",
    "generate.memory_token_limit",
    "generate.answer_stratified_sample",
    "generate.answer_sample_seed",
    "generate.show_memory_time",
    "generate.hybrid",
)

_JUDGE_FIELDS: Sequence[str] = (
    "experiment.benchmark",
    "models.judge",
    "evaluate.use_cot",
    "evaluate.judge_stratified_sample",
    "evaluate.judge_sample_seed",
    "prompts.judge_template",
    "token_limits.evaluate_max_new_tokens",
)


def _template_content_identities(
    references: Sequence[Any], template_root: Optional[Path]
) -> list[dict[str, str]]:
    """返回模板引用及其内容 hash，找不到时使用明确且稳定的 missing 标记。"""
    root = Path(template_root).resolve() if template_root is not None else None
    identities = []
    for reference in references:
        reference_text = "" if reference is None else str(reference)
        if not reference_text:
            identities.append({"reference": reference_text, "content_sha256": "missing:empty"})
            continue
        if root is None:
            identities.append(
                {"reference": reference_text, "content_sha256": "missing:template-root-unset"}
            )
            continue

        candidate = (root / reference_text).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            identities.append(
                {"reference": reference_text, "content_sha256": "missing:outside-template-root"}
            )
            continue

        try:
            content = candidate.read_bytes()
        except FileNotFoundError:
            identities.append(
                {"reference": reference_text, "content_sha256": "missing:not-found"}
            )
            continue
        except OSError:
            identities.append(
                {"reference": reference_text, "content_sha256": "missing:unreadable"}
            )
            continue
        identities.append(
            {
                "reference": reference_text,
                "content_sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return identities


def _template_references(config: Mapping, dotted_paths: Sequence[str]) -> list[Any]:
    references = []
    for dotted_path in dotted_paths:
        value = _get_path(config, dotted_path)
        if isinstance(value, (list, tuple)):
            references.extend(value)
        else:
            references.append(value)
    return references


def candidate_fingerprint(
    resolved_config: Mapping, *, template_root: Optional[Path] = None
) -> str:
    """候选记忆抽取阶段指纹：只依赖 benchmark + extract 相关配置。"""
    payload = {
        "stage": "candidate",
        "fields": _extract_fields(resolved_config, _CANDIDATE_FIELDS),
        "template_contents": _template_content_identities(
            _template_references(resolved_config, ("extract.aspect_templates",)),
            template_root,
        ),
    }
    return short_hash(payload)


def ingest_fingerprint(
    resolved_config: Mapping, method: str, *, template_root: Optional[Path] = None
) -> str:
    """灌库阶段指纹：候选指纹 + manager 模型/相关 prompt/token 上限 + 该 method 的全部专属配置。

    只纳入 ``methods.<method>`` 子树，因此某一方法阈值的变化不会影响其他方法的
    ingest 指纹；也不吸收 generate/evaluate 相关字段。
    """
    payload = {
        "stage": "ingest",
        "method": method,
        "candidate": candidate_fingerprint(resolved_config, template_root=template_root),
        "common_fields": _extract_fields(resolved_config, _INGEST_COMMON_FIELDS),
        "method_config": _get_path(resolved_config, f"methods.{method}", {}),
        "template_contents": _template_content_identities(
            _template_references(
                resolved_config,
                (
                    "prompts.relation_user_en",
                    "prompts.relation_user_zh",
                ),
            ),
            template_root,
        ),
    }
    return short_hash(payload)


def answer_fingerprint(
    resolved_config: Mapping,
    method: str,
    ingest_id: str,
    *,
    template_root: Optional[Path] = None,
) -> str:
    """答题阶段指纹：上游 ingest_id + answer 模型 + retrieve/生成相关配置。"""
    payload = {
        "stage": "answer",
        "method": method,
        "ingest_id": ingest_id,
        "fields": _extract_fields(resolved_config, _ANSWER_FIELDS),
    }
    return short_hash(payload)


def judge_fingerprint(
    resolved_config: Mapping,
    method: str,
    answer_id: str,
    *,
    template_root: Optional[Path] = None,
) -> str:
    """评测阶段指纹：上游 answer_id + judge 模型 + judge 相关配置/模板。"""
    payload = {
        "stage": "judge",
        "method": method,
        "answer_id": answer_id,
        "fields": _extract_fields(resolved_config, _JUDGE_FIELDS),
        "template_contents": _template_content_identities(
            _template_references(
                resolved_config,
                ("prompts.judge_template",),
            ),
            template_root,
        ),
    }
    return short_hash(payload)


# ---------------------------------------------------------------------------
# slug / run_id
# ---------------------------------------------------------------------------


def _slugify(text: Any) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    text = text.strip("-_")
    return text or "x"


def _enabled_method_names(resolved_config: Mapping) -> list:
    methods = _get_path(resolved_config, "methods", {})
    if not isinstance(methods, Mapping):
        return []
    names = [
        str(name)
        for name, cfg in methods.items()
        if isinstance(cfg, Mapping) and cfg.get("enabled")
    ]
    return sorted(names)


def build_slug(resolved_config: Mapping) -> str:
    """从 resolved_config 中挑取可读片段拼出路径安全的 slug（不含 hash）。"""
    parts = []

    benchmark = _get_path(resolved_config, "experiment.benchmark")
    if benchmark:
        parts.append(_slugify(benchmark))

    suffix = _get_path(resolved_config, "experiment.suffix")
    if suffix:
        parts.append(_slugify(suffix))

    methods = _enabled_method_names(resolved_config)
    if methods:
        parts.append("+".join(_slugify(m) for m in methods))

    token_limit = _get_path(resolved_config, "generate.memory_token_limit")
    if token_limit is not None:
        parts.append(f"tl{token_limit}")

    if not parts:
        parts.append("run")

    slug = "_".join(parts)
    slug = slug[:96].strip("_-")
    return slug or "run"


# ---------------------------------------------------------------------------
# git commit（best-effort，容忍 git 缺失/非仓库/超时）
# ---------------------------------------------------------------------------


def _resolve_git_commit(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# ExperimentIdentity / materialize
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterializedRun:
    """``ExperimentIdentity.materialize()`` 的返回结果。"""

    run_root: Path
    manifest_path: Path
    run_yaml_path: Path
    manifest: dict
    reused: bool


@dataclass(frozen=True)
class ExperimentIdentity:
    """由一份 resolved_config 推导出的实验身份：slug / run_id / run_root，并支持落盘。"""

    resolved_config: Mapping[str, Any]
    source_config_path: Optional[Path] = None
    repo_root: Path = field(default_factory=Path.cwd)
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolved_config", deepcopy(self.resolved_config))
        object.__setattr__(self, "repo_root", Path(self.repo_root).resolve())
        object.__setattr__(self, "artifacts_root", Path(self.artifacts_root))
        if self.source_config_path is not None:
            object.__setattr__(
                self, "source_config_path", Path(self.source_config_path).resolve()
            )

    # -- 派生属性 ----------------------------------------------------------

    @property
    def resolved_config_hash(self) -> str:
        return sha256_hash(self.resolved_config)

    @property
    def short_hash(self) -> str:
        return self.resolved_config_hash[:_RUN_ID_HASH_LEN]

    @property
    def slug(self) -> str:
        return build_slug(self.resolved_config)

    @property
    def run_id(self) -> str:
        return f"{self.slug}--{self.short_hash}"

    @property
    def run_root(self) -> Path:
        root = self.artifacts_root
        if not root.is_absolute():
            root = self.repo_root / root
        return root / self.run_id

    # -- manifest ------------------------------------------------------

    def build_manifest(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "slug": self.slug,
            "resolved_config": canonicalize(self.resolved_config),
            "resolved_config_hash": self.resolved_config_hash,
            "source_config_path": (
                str(self.source_config_path) if self.source_config_path is not None else None
            ),
            "git_commit": _resolve_git_commit(self.repo_root),
            "created_at": _utc_now_iso(),
        }

    def materialize(self) -> MaterializedRun:
        """写出 ``run.yaml`` 与 ``manifest.json``。

        幂等：若 run_root 下已有 manifest.json，比较其 resolved_config_hash：
        - 相同 → 直接复用现有文件，不重写（避免 created_at 被覆盖）。
        - 不同 → 抛出 ValueError，不修改已存在的文件。
        """
        run_root = self.run_root
        manifest_path = run_root / "manifest.json"
        run_yaml_path = run_root / "run.yaml"

        new_hash = self.resolved_config_hash

        with _MATERIALIZE_LOCK:
            run_root.mkdir(parents=True, exist_ok=True)
            if manifest_path.exists():
                try:
                    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid manifest.json at {manifest_path}: {exc}"
                    ) from exc
                if not isinstance(existing_manifest, dict):
                    raise ValueError(
                        f"invalid manifest.json at {manifest_path}: expected a JSON object"
                    )

                existing_hash = existing_manifest.get("resolved_config_hash")
                if existing_hash != new_hash:
                    raise ValueError(
                        f"run '{self.run_id}' already exists at {run_root} with a "
                        f"different resolved_config_hash "
                        f"(existing={existing_hash!r}, new={new_hash!r}); refusing to "
                        "overwrite. Remove the existing run directory or use a "
                        "different artifacts_root if this is intentional."
                    )
                if not run_yaml_path.exists():
                    _atomic_write_text(
                        run_yaml_path,
                        yaml.safe_dump(existing_manifest, sort_keys=True, allow_unicode=True),
                    )
                return MaterializedRun(
                    run_root=run_root,
                    manifest_path=manifest_path,
                    run_yaml_path=run_yaml_path,
                    manifest=existing_manifest,
                    reused=True,
                )

            manifest = self.build_manifest()
            _atomic_write_text(
                manifest_path,
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
            )
            _atomic_write_text(
                run_yaml_path,
                yaml.safe_dump(manifest, sort_keys=True, allow_unicode=True),
            )
        return MaterializedRun(
            run_root=run_root,
            manifest_path=manifest_path,
            run_yaml_path=run_yaml_path,
            manifest=manifest,
            reused=False,
        )


def _safe_method(method: Any) -> str:
    """校验并返回路径安全的 method 名，拒绝空值/越权字符（``/``、``\\``、空格、``.``）。"""
    text = "" if method is None else str(method)
    if not _SAFE_METHOD_RE.fullmatch(text):
        raise ValueError(
            f"invalid method name {method!r}: must match {_SAFE_METHOD_RE.pattern!r} "
            "(non-empty, no path separators, dots, or spaces)"
        )
    return text


def _generate_attempt_id() -> str:
    """UTC 时间戳（微秒精度）+ 随机短标识，保证同一进程内连续调用也不重复。"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    token = secrets.token_hex(4)
    return f"{timestamp}-{token}"


# ---------------------------------------------------------------------------
# ArtifactLayout：内容寻址的 stage 目录布局
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactLayout:
    """由 ``ExperimentIdentity`` + ``template_root`` 推导出的产物目录布局。

    设计要点：
    - candidate / ingest 落在与 ``run_root`` 独立的全局内容寻址阶段根
      （默认 ``<repo_root>/artifacts/stages``），只要 candidate/ingest 指纹
      不变（例如仅改变 ``generate.memory_token_limit``），不同的 run 就会
      复用同一份 candidate/ingest 目录，避免重复抽取/灌库。
    - answer / judge / attempts 落在该 run 专属的 ``run_root`` 下，因为它们
      依赖 ``generate``/``evaluate`` 相关配置，天然随 run 变化。
    - 所有 method 相关路径都会先经过 ``_safe_method`` 校验，拒绝空值或包含
      路径分隔符/``..``/空格的非法 method，防止越过预期目录。
    - ``stage_nonce``：可选的强制去重标记（如某次 ``full_pipeline`` 统计重复的
      ``variant_id``）。非空时会被混入 candidate_id / ingest_id 的计算，
      使得同一份 resolved_config 在不同 nonce 下产生互不相同、互不复用的
      candidate/ingest 目录；answer/judge 因为始终基于
      ``self.ingest_id(method)``（而不是原始 ``ingest_fingerprint``）计算，
      会自动依赖 nonce 之后的 ingest。为空（``None``/``""``）时行为与不设置
      完全一致。
    """

    identity: ExperimentIdentity
    template_root: Optional[Path] = None
    stages_root: Optional[Path] = None
    stage_nonce: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "template_root",
            Path(self.template_root).resolve() if self.template_root is not None else None,
        )
        root = Path(self.stages_root) if self.stages_root is not None else DEFAULT_STAGES_ROOT
        if not root.is_absolute():
            root = self.identity.repo_root / root
        object.__setattr__(self, "stages_root", root)
        object.__setattr__(self, "stage_nonce", self.stage_nonce or None)

    # -- run_root 便捷代理 --------------------------------------------------

    @property
    def run_root(self) -> Path:
        return self.identity.run_root

    # -- candidate -----------------------------------------------------

    @property
    def candidate_id(self) -> str:
        base = candidate_fingerprint(
            self.identity.resolved_config, template_root=self.template_root
        )
        if self.stage_nonce:
            return short_hash({"base_candidate_id": base, "stage_nonce": self.stage_nonce})
        return base

    @property
    def candidate_dir(self) -> Path:
        return self.stages_root / "candidates" / self.candidate_id

    # -- shared stage locks ------------------------------------------------

    @property
    def locks_root(self) -> Path:
        """跨进程锁文件专用目录，刻意与实际阶段数据目录分离。"""
        return self.stages_root / "locks"

    def _lock_path(self, stage: str, method: Optional[str] = None) -> Path:
        if stage == "candidate":
            return self.locks_root / f"candidate--{self.candidate_id}.lock"
        if stage == "ingest" and method is not None:
            safe_method = _safe_method(method)
            return self.locks_root / f"ingest--{safe_method}--{self.ingest_id(safe_method)}.lock"
        raise ValueError(f"unsupported shared lock stage {stage!r}")

    @contextmanager
    def _shared_stage_lock(self, lock_path: Path):
        """取得线程内互斥 + Linux advisory flock 的共享阶段锁。"""
        if fcntl is None:
            raise RuntimeError(
                "shared artifact stage locks require fcntl.flock; this platform is unsupported"
            )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with _STAGE_LOCKS_GUARD:
            thread_lock = _STAGE_THREAD_LOCKS.setdefault(lock_path, threading.Lock())
        with thread_lock:
            with lock_path.open("a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def candidate_lock(self):
        """返回 candidate 内容寻址阶段的跨进程锁 contextmanager。"""
        return self._shared_stage_lock(self._lock_path("candidate"))

    # -- ingest ----------------------------------------------------------

    def ingest_id(self, method: str) -> str:
        safe_method = _safe_method(method)
        base = ingest_fingerprint(
            self.identity.resolved_config, safe_method, template_root=self.template_root
        )
        if self.stage_nonce:
            return short_hash(
                {
                    "base_ingest_id": base,
                    "method": safe_method,
                    "stage_nonce": self.stage_nonce,
                }
            )
        return base

    def ingest_dir(self, method: str) -> Path:
        safe_method = _safe_method(method)
        return self.stages_root / "ingest" / safe_method / self.ingest_id(safe_method)

    def ingest_lock(self, method: str):
        """返回指定 method 的 ingest 内容寻址阶段的跨进程锁 contextmanager。"""
        return self._shared_stage_lock(self._lock_path("ingest", method))

    # -- answer ------------------------------------------------------------

    def answer_id(self, method: str) -> str:
        safe_method = _safe_method(method)
        return answer_fingerprint(
            self.identity.resolved_config,
            safe_method,
            self.ingest_id(safe_method),
            template_root=self.template_root,
        )

    def answer_dir(self, method: str) -> Path:
        safe_method = _safe_method(method)
        return self.run_root / "answer" / safe_method / self.answer_id(safe_method)

    # -- judge -------------------------------------------------------------

    def judge_id(self, method: str) -> str:
        safe_method = _safe_method(method)
        return judge_fingerprint(
            self.identity.resolved_config,
            safe_method,
            self.answer_id(safe_method),
            template_root=self.template_root,
        )

    def judge_dir(self, method: str) -> Path:
        safe_method = _safe_method(method)
        return self.run_root / "judge" / safe_method / self.judge_id(safe_method)

    # -- attempts ------------------------------------------------------

    def attempt_dir(self, attempt_id: str) -> Path:
        return self.run_root / "attempts" / attempt_id

    def new_attempt_dir(self) -> Path:
        """生成新的 attempt_id 并只创建该目录；不做任何模型调用/其他副作用。"""
        attempt_id = _generate_attempt_id()
        path = self.attempt_dir(attempt_id)
        path.mkdir(parents=True, exist_ok=False)
        return path

    # -- shared stage manifests -------------------------------------------

    def _shared_stage_manifest(
        self, stage: str, *, method: Optional[str], upstream_stage_ids: list[str]
    ) -> dict:
        stage_id = self.candidate_id if stage == "candidate" else self.ingest_id(method or "")
        return {
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "stage_id": stage_id,
            "method": method,
            "upstream_stage_ids": upstream_stage_ids,
            "producer_run_id": self.identity.run_id,
            "producer_resolved_config_hash": self.identity.resolved_config_hash,
            "producer_run_root": str(self.run_root),
            "created_at": _utc_now_iso(),
        }

    def _write_shared_stage_manifest(
        self,
        stage_dir: Path,
        stage: str,
        *,
        method: Optional[str],
        upstream_stage_ids: list[str],
    ) -> Path:
        """在持有相应 stage lock 时创建或验证共享 stage manifest。"""
        manifest = self._shared_stage_manifest(
            stage, method=method, upstream_stage_ids=upstream_stage_ids
        )
        path = stage_dir / "stage_manifest.json"
        stage_dir.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid shared stage manifest at {path}: {exc}") from exc
            required = ("stage", "stage_id", "method", "upstream_stage_ids")
            if not isinstance(existing, dict) or any(
                existing.get(field) != manifest[field] for field in required
            ):
                raise ValueError(
                    f"incompatible shared stage manifest at {path}; refusing to overwrite"
                )
            return path
        _atomic_write_text(
            path, json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
        )
        return path

    def write_candidate_stage_manifest(self) -> Path:
        return self._write_shared_stage_manifest(
            self.candidate_dir, "candidate", method=None, upstream_stage_ids=[]
        )

    def write_ingest_stage_manifest(self, method: str) -> Path:
        safe_method = _safe_method(method)
        return self._write_shared_stage_manifest(
            self.ingest_dir(safe_method),
            "ingest",
            method=safe_method,
            upstream_stage_ids=[self.candidate_id],
        )

    # -- stage manifests（内存 mapping，是否落盘由编排层决定）--------------

    def candidate_manifest(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "stage": "candidate",
            "stage_id": self.candidate_id,
            "run_id": self.identity.run_id,
            "method": None,
            "upstream_stage_ids": [],
            "resolved_config_hash": self.identity.resolved_config_hash,
        }

    def ingest_manifest(self, method: str) -> dict:
        safe_method = _safe_method(method)
        return {
            "schema_version": SCHEMA_VERSION,
            "stage": "ingest",
            "stage_id": self.ingest_id(safe_method),
            "run_id": self.identity.run_id,
            "method": safe_method,
            "upstream_stage_ids": [self.candidate_id],
            "resolved_config_hash": self.identity.resolved_config_hash,
        }

    def answer_manifest(self, method: str) -> dict:
        safe_method = _safe_method(method)
        return {
            "schema_version": SCHEMA_VERSION,
            "stage": "answer",
            "stage_id": self.answer_id(safe_method),
            "run_id": self.identity.run_id,
            "method": safe_method,
            "upstream_stage_ids": [self.ingest_id(safe_method)],
            "resolved_config_hash": self.identity.resolved_config_hash,
        }

    def judge_manifest(self, method: str) -> dict:
        safe_method = _safe_method(method)
        return {
            "schema_version": SCHEMA_VERSION,
            "stage": "judge",
            "stage_id": self.judge_id(safe_method),
            "run_id": self.identity.run_id,
            "method": safe_method,
            "upstream_stage_ids": [self.answer_id(safe_method)],
            "resolved_config_hash": self.identity.resolved_config_hash,
        }


def _atomic_write_text(path: Path, text: str) -> None:
    """以临时文件 + os.replace 原子替换单个 artifact 文件。"""
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
