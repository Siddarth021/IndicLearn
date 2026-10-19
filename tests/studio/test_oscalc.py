"""OS numerical solvers checked against textbook examples (Silberschatz-Galvin) and a real IIIT Sri City mid-term."""

from __future__ import annotations

import random

import pytest

from indiclearn.studio.courses import oscalc as C


def P(*rows):
    return [{"pid": f"P{i + 1}", "at": r[0], "bt": r[1], "pr": r[2] if len(r) > 2 else 0} for i, r in enumerate(rows)]


def test_fcfs_sjf_srtf_rr_priority_textbook():
    assert C.schedule(P((0, 24), (0, 3), (0, 3)), "fcfs")["avg_wt"] == "17"
    assert C.schedule(P((0, 6), (0, 8), (0, 7), (0, 3)), "sjf")["avg_wt"] == "7"
    assert C.schedule(P((0, 8), (1, 4), (2, 9), (3, 5)), "srtf")["avg_wt"] == "26/4 = 6.50"
    assert C.schedule(P((0, 24), (0, 3), (0, 3)), "rr", quantum=4)["avg_wt"] == "17/3 = 5.67"
    assert C.schedule(P((0, 10, 3), (0, 1, 1), (0, 2, 4), (0, 1, 5), (0, 5, 2)), "prio_np")["avg_wt"] == "41/5 = 8.20"


def test_iiits_midterm_preemptive_priority_question():
    r = C.schedule(P((0, 5, 4), (1, 10, 3), (2, 3, 1), (3, 2, 6), (4, 7, 6), (5, 5, 2)), "prio_p", larger_is_higher=True)
    assert [x["ct"] for x in r["rows"]] == [14, 24, 32, 5, 12, 29]
    assert r["avg_tat"] == "101/6 = 16.83" and r["avg_wt"] == "69/6 = 11.50"


def test_page_replacement_textbook():
    refs = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
    assert [C.replace_pages(refs, 3, a)["faults"] for a in ("fifo", "lru", "opt")] == [15, 12, 9]
    belady = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    assert C.replace_pages(belady, 3, "fifo")["faults"] == 9 and C.replace_pages(belady, 4, "fifo")["faults"] == 10


def test_bankers_textbook():
    r = C.bankers([[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]], [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]], [3, 3, 2])
    assert r["safe"] and r["sequence"] == [1, 3, 0, 2, 4]  # lowest-numbered runnable process first
    assert r["need"][0] == [7, 4, 3]


def test_disk_textbook():
    q = [98, 183, 37, 122, 14, 124, 65, 67]
    assert C.disk(53, q, "fcfs", 200)["total"] == 640
    assert C.disk(53, q, "sstf", 200)["total"] == 236
    assert C.disk(53, q, "scan", 200, toward_high=False)["total"] == 236
    assert C.disk(53, q, "look", 200)["total"] == 299
    assert C.disk(53, q, "clook", 200)["total"] == 322


def test_fit_textbook():
    holes, procs = [100, 500, 200, 300, 600], [212, 417, 112, 426]
    assert C.fit(holes, procs, "first")["placed"] == [1, 4, 1, None]
    assert C.fit(holes, procs, "best")["placed"] == [3, 1, 2, 4]
    assert C.fit(holes, procs, "worst")["placed"] == [4, 1, 4, None]


@pytest.mark.parametrize("kind", list(C.GENERATORS))
@pytest.mark.parametrize("diff", ["easy", "medium", "hard"])
def test_every_generator_makes_a_complete_question(kind, diff):
    for seed in range(15):
        q = C.generate(kind, random.Random(seed), 4, diff, "x")
        assert q["blocks"] and q["answer"] and all(("p" in b) ^ ("table" in b) for b in q["blocks"] + q["answer"])
