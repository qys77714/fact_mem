from utils import qa_metrics as qm


def test_compute_f1_whitespace_overlap():
    # Same tokens after normalize
    f1 = qm.compute_f1("The cat", "cat", "whitespace")
    assert f1 == 1.0
    assert qm.compute_exact("The cat", "cat", "whitespace") is True


def test_compute_f1_partial_overlap():
    f1 = qm.compute_f1("cat dog", "cat fish", "whitespace")
    assert 0.0 < f1 < 1.0


def test_compute_f1_empty_both():
    assert qm.compute_f1("", "", "whitespace") == 1.0
    assert qm.compute_exact("", "", "whitespace") is True


def test_compute_f1_empty_gold():
    assert qm.compute_f1("x", "", "whitespace") == 0.0


def test_char_mode():
    assert qm.compute_exact("北京", "北京", "char") is True
    f1 = qm.compute_f1("北 京", "北京", "char")
    assert f1 == 1.0
