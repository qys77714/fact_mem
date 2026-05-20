"""Sliding-window turn chunking for candidate extract (LoCoMo-style sessions)."""

from pipeline.extract_candidates import _iter_turn_chunks, _normalize_turn_overlap


def test_iter_turn_chunks_no_overlap_matches_fixed_stride():
    turns = list(range(10))
    out = _iter_turn_chunks(turns, 4, overlap_turns=0)
    indices = [(a, b) for a, b, _ in out]
    assert indices == [(0, 3), (4, 7), (8, 9)]


def test_iter_turn_chunks_overlap_one_slides_by_three():
    turns = list(range(10))
    out = _iter_turn_chunks(turns, 4, overlap_turns=1)
    indices = [(a, b) for a, b, chunks in out]
    assert indices == [(0, 3), (3, 6), (6, 9), (9, 9)]
    assert [len(c) for _, _, c in out] == [4, 4, 4, 1]


def test_normalize_turn_overlap_rejects_when_not_smaller_than_granularity():
    assert _normalize_turn_overlap("0", 4) == 0
    assert _normalize_turn_overlap("3", 4) == 3
    try:
        _normalize_turn_overlap("4", 4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_normalize_turn_overlap_all_granularity_forces_zero():
    assert _normalize_turn_overlap("5", "all") == 0
