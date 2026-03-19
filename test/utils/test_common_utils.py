import time

import numpy as np

from utils.common_utils import Timer, configure_logging, set_seed


def test_set_seed_reproducible():
    set_seed(42)
    a = np.random.rand(3)
    set_seed(42)
    b = np.random.rand(3)
    assert np.allclose(a, b)


def test_timer_end_requires_start():
    t = Timer()
    try:
        t.end()
        assert False, "should raise"
    except ValueError:
        pass


def test_timer_basic():
    t = Timer()
    t.start()
    time.sleep(0.01)
    h, m, s = t.end()
    assert h >= 0 and m >= 0 and s >= 0


def test_configure_logging_creates_file(tmp_path):
    log_file = tmp_path / "x" / "a.log"
    logger = configure_logging(str(log_file), override=True)
    logger.info("hello")
    assert log_file.exists()
