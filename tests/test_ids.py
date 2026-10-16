import time
from sitegraph.ids import gid, ulid, now_ms


def test_gid_is_deterministic_and_prefixed():
    assert gid("st", "a", 1) == gid("st", "a", 1)
    g = gid("st", "a")
    assert g.startswith("st:") and len(g) == 3 + 16


def test_gid_differs_by_parts_but_not_by_prefix_hash():
    assert gid("st", "a") != gid("st", "b")
    assert gid("st", "a")[3:] == gid("el", "a")[3:]
    assert gid("st", None) == gid("st", "")


def test_ulid_is_sortable_and_unique():
    a = ulid("run")
    time.sleep(0.003)
    b = ulid("run")
    assert a.startswith("run:") and a < b and a != ulid("run")


def test_now_ms_is_int():
    assert isinstance(now_ms(), int)
