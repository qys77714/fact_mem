"""relation_decision 就地融合答题记忆 C + LLM 复核 的行为测试。

不依赖真实 backbone / embedding / LLM：用假对象注入，通过 __new__ 绕过 __init__。
"""
import numpy as np
import pytest


class _Trace:
    def log_llm_interaction(self, **k):
        pass

    def log_memory_operation(self, **k):
        pass


def _make_system(*, classify_label="CON", fuse_text="FUSED C"):
    """构造系统：classifier 永远给 classify_label，融合返回 fuse_text。"""
    from memory.candidate_ingest import memory_system as ms

    sysobj = ms.LmeCandidateRelationDecisionMemorySystem.__new__(
        ms.LmeCandidateRelationDecisionMemorySystem
    )
    sysobj.language = "en"
    sysobj._relation_backend = "llm"
    sysobj._relation_system_en_template = None
    sysobj._relation_system_zh_template = None
    sysobj._relation_max_new_tokens = 64
    sysobj._answer_fuse_max_new_tokens = 64
    sysobj._rc_backend = None
    sysobj._fusion_enabled = True
    sysobj.related_memory_top_k = 5
    sysobj._pairwise_sim_threshold = 0.0
    sysobj._condition_sim_threshold = 0.0
    sysobj._relation_concurrency = 1
    sysobj._active_relations = None

    # 8 维稳定 embedding：按文本 hash 落在单位向量上（保证可被检索到）
    def _embed(texts):
        out = []
        for t in texts:
            v = np.zeros(8, dtype=np.float32)
            v[0] = 1.0
            v[1] = (abs(hash(t)) % 100) / 1000.0
            out.append(v)
        return np.array(out, dtype=np.float32)

    sysobj._embed_texts = _embed  # type: ignore
    sysobj.build_text_for_embedding = lambda text, metadata=None: text  # type: ignore

    class FakeLLM:
        def __init__(self):
            self.purposes = []

        def get_response_chat(self, messages, **k):
            rf = k.get("response_format")
            # classification 调用带 response_format（schema）
            if rf is not None:
                self.purposes.append("classify")
                return {"relation": classify_label}
            # 否则是融合调用（无 schema）
            self.purposes.append("fuse")
            return fuse_text

    sysobj.llm_client = FakeLLM()
    return sysobj, ms


def _new_db(tmp_path):
    from memory.storage.local_faiss import LocalFaissDatabase

    return LocalFaissDatabase(namespace="ep", database_root=str(tmp_path))


def _meta_base():
    return {"date": "2024-01-01", "lme_update_method": "relation_decision"}


def test_con_creates_answer_memory_and_hides_new_primary(tmp_path, monkeypatch):
    """CON 成立：m_new 成 primary、old 降 evidence，产出 C(role=answer)，隐藏 m_new。"""
    s, ms = _make_system(classify_label="CON", fuse_text="C-TEXT")
    db = _new_db(tmp_path)
    # 预置一条 old primary
    old_id = db.add("Alice likes blue", "session_0", "2023-12-01", {}, s._embed_texts(["Alice likes blue"])[0])

    # classifier 给 CON
    monkeypatch.setattr(s, "_classify_relation", lambda old, new, scope, trace: "CON")

    ops = s._run_pairwise_relation_decision(db, "Alice likes red", _meta_base(), 1, "scope", _Trace())
    assert ops == 1
    assert "fuse" in s.llm_client.purposes

    rows = {r.memory_id: r for r in db.list_all_memories()}
    roles = sorted((r.metadata.get("memory_role", "primary")) for r in rows.values())
    # old 变 evidence；新增 m_new(primary, hidden) + C(answer)
    assert "answer" in roles, roles
    assert "evidence" in roles, roles
    # C 行内容是融合文本
    c_rows = [r for r in rows.values() if r.metadata.get("memory_role") == "answer"]
    assert len(c_rows) == 1 and c_rows[0].text == "C-TEXT"
    # m_new primary 被 answer_hidden
    new_primary = [
        r for r in rows.values()
        if r.metadata.get("memory_role", "primary") == "primary" and r.text == "Alice likes red"
    ]
    assert len(new_primary) == 1
    assert new_primary[0].metadata.get("answer_hidden") is True
    assert new_primary[0].metadata.get("answer_id") == c_rows[0].memory_id



def test_eqv_hides_old_primary_keeps_m_new_as_evidence(tmp_path, monkeypatch):
    """EQV 成立：m_new 挂 evidence，old 仍是 primary 但被 C 覆盖 -> old 被 hidden。"""
    s, ms = _make_system(classify_label="EQV", fuse_text="C-EQV")
    db = _new_db(tmp_path)
    old_id = db.add("Alice works at Google", "session_0", "2023-12-01", {}, s._embed_texts(["Alice works at Google"])[0])

    monkeypatch.setattr(s, "_classify_relation", lambda old, new, scope, trace: "EQV")

    ops = s._run_pairwise_relation_decision(db, "Alice is employed by Google", _meta_base(), 1, "scope", _Trace())
    assert ops == 1

    rows = {r.memory_id: r for r in db.list_all_memories()}
    # old 仍是 primary 但 answer_hidden
    old_row = rows[old_id]
    assert old_row.metadata.get("memory_role", "primary") == "primary"
    assert old_row.metadata.get("answer_hidden") is True
    # 有一条 C
    c_rows = [r for r in rows.values() if r.metadata.get("memory_role") == "answer"]
    assert len(c_rows) == 1 and c_rows[0].text == "C-EQV"
    assert old_row.metadata.get("answer_id") == c_rows[0].memory_id
