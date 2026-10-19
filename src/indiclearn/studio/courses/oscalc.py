"""Exact solvers + question generators for Operating Systems numericals.

Every numerical in a generated paper is produced here, so its answer is computed, never guessed by a
language model: CPU scheduling, page replacement, Banker's algorithm, disk scheduling, contiguous
allocation (first/best/worst fit), paging address translation and effective access time.

A question is a dict of printable "blocks" (paragraphs and tables) plus an answer made of blocks too:
    {"kind": "numerical", "topic": "fcfs-sjf", "marks": 4, "blocks": [...], "answer": [...]}
    block = {"p": "text"} | {"table": [[header...], [row...], ...]}
Conventions (ties, queue order) are stated in the question itself so the student and the key agree.
"""

from __future__ import annotations

import random
from collections import deque
from fractions import Fraction
from typing import Any

# --------------------------------------------------------------------------- helpers


def _avg(values: list[int]) -> str:
    f = Fraction(sum(values), len(values))
    return str(f.numerator) if f.denominator == 1 else f"{sum(values)}/{len(values)} = {float(f):.2f}"


def _gantt_rows(gantt: list[tuple[str, int, int]]) -> list[list[str]]:
    return [["Slot", *[g[0] for g in gantt]], ["Time", *[f"{g[1]}-{g[2]}" for g in gantt]]]


def _merge(gantt: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for g in gantt:
        if out and out[-1][0] == g[0] and out[-1][2] == g[1]:
            out[-1] = (g[0], out[-1][1], g[2])
        else:
            out.append(g)
    return out


# --------------------------------------------------------------------------- CPU scheduling

CPU_ALGOS = {
    "fcfs": "First Come First Serve (FCFS)",
    "sjf": "non-preemptive Shortest Job First (SJF)",
    "srtf": "Shortest Remaining Time First (SRTF, preemptive SJF)",
    "prio_np": "non-preemptive priority scheduling",
    "prio_p": "preemptive priority scheduling",
    "rr": "Round Robin",
}


def schedule(procs: list[dict[str, int]], algo: str, quantum: int = 2, larger_is_higher: bool = False) -> dict[str, Any]:
    """procs: [{"pid": "P1", "at": 0, "bt": 5, "pr": 2}, ...] -> gantt, CT/TAT/WT/RT per process, averages.
    Ties are broken by arrival time, then by process number (P1 before P2)."""
    ps = [dict(p) for p in procs]
    order = {p["pid"]: i for i, p in enumerate(ps)}
    rem = {p["pid"]: p["bt"] for p in ps}
    first: dict[str, int] = {}
    done: dict[str, int] = {}
    gantt: list[tuple[str, int, int]] = []
    t = min(p["at"] for p in ps)

    def prio_key(p: dict[str, int]) -> int:
        return -p["pr"] if larger_is_higher else p["pr"]

    if algo == "rr":
        q: deque[str] = deque()
        arrived: set[str] = set()

        def admit(upto: int) -> None:
            for p in sorted(ps, key=lambda p: (p["at"], order[p["pid"]])):
                if p["pid"] not in arrived and p["at"] <= upto:
                    arrived.add(p["pid"])
                    q.append(p["pid"])

        admit(t)
        while len(done) < len(ps):
            if not q:
                nxt = min(p["at"] for p in ps if p["pid"] not in arrived)
                gantt.append(("idle", t, nxt))
                t = nxt
                admit(t)
                continue
            pid = q.popleft()
            first.setdefault(pid, t)
            run = min(quantum, rem[pid])
            gantt.append((pid, t, t + run))
            t += run
            rem[pid] -= run
            admit(t)  # processes arriving during (or at the end of) the slice join the queue before the preempted one
            if rem[pid]:
                q.append(pid)
            else:
                done[pid] = t
    else:
        preemptive = algo in ("srtf", "prio_p")
        while len(done) < len(ps):
            ready = [p for p in ps if p["at"] <= t and p["pid"] not in done]
            if not ready:
                nxt = min(p["at"] for p in ps if p["pid"] not in done)
                gantt.append(("idle", t, nxt))
                t = nxt
                continue
            if algo == "fcfs":
                key = lambda p: (p["at"], order[p["pid"]])  # noqa: E731
            elif algo in ("sjf", "srtf"):
                key = lambda p: (rem[p["pid"]], p["at"], order[p["pid"]])  # noqa: E731
            else:
                key = lambda p: (prio_key(p), p["at"], order[p["pid"]])  # noqa: E731
            p = min(ready, key=key)
            pid = p["pid"]
            first.setdefault(pid, t)
            if preemptive:
                future = [x["at"] for x in ps if x["at"] > t and x["pid"] not in done]
                run = min([rem[pid]] + [a - t for a in future])
            else:
                run = rem[pid]
            gantt.append((pid, t, t + run))
            t += run
            rem[pid] -= run
            if not rem[pid]:
                done[pid] = t
    rows = []
    for p in ps:
        ct = done[p["pid"]]
        tat = ct - p["at"]
        rows.append({"pid": p["pid"], "at": p["at"], "bt": p["bt"], "pr": p.get("pr"), "ct": ct, "tat": tat, "wt": tat - p["bt"], "rt": first[p["pid"]] - p["at"]})
    return {"gantt": _merge(gantt), "rows": rows, "avg_tat": _avg([r["tat"] for r in rows]), "avg_wt": _avg([r["wt"] for r in rows]),
            "avg_rt": _avg([r["rt"] for r in rows])}


def gen_cpu(rng: random.Random, marks: int, difficulty: str, topic: str = "fcfs-sjf") -> dict[str, Any]:
    n = {"easy": 4, "medium": 5, "hard": 6}.get(difficulty, 5)
    pool = {"fcfs-sjf": ["fcfs", "sjf", "srtf"], "priority-sched": ["prio_np", "prio_p"], "round-robin": ["rr"]}.get(topic, list(CPU_ALGOS))
    algo = rng.choice(pool)
    if difficulty == "easy" and algo == "srtf":
        algo = "sjf"
    procs = []
    at = 0
    for i in range(n):
        procs.append({"pid": f"P{i + 1}", "at": at, "bt": rng.randint(1, 10), "pr": rng.randint(1, n + 1)})
        at += rng.choice([0, 1, 1, 2, 3]) if i else rng.choice([0, 0, 1])
    procs[0]["at"] = 0
    quantum = rng.choice([2, 3, 4]) if algo == "rr" else 0
    larger = rng.random() < 0.5
    two = marks >= 6 and algo in ("fcfs", "sjf", "srtf")
    algos = [algo] + ([rng.choice([a for a in ("fcfs", "sjf", "srtf") if a != algo])] if two else [])
    has_prio = any(a.startswith("prio") for a in algos)
    head = ["Process", "Arrival time", "Burst time"] + (["Priority"] if has_prio else [])
    table = [head] + [[p["pid"], str(p["at"]), str(p["bt"])] + ([str(p["pr"])] if has_prio else []) for p in procs]
    names = " and ".join(CPU_ALGOS[a] + (f" with time quantum {quantum}" if a == "rr" else "") for a in algos)
    rule = ""
    if has_prio:
        rule = " A larger priority number means higher priority." if larger else " A smaller priority number means higher priority."
    if "rr" in algos:
        rule += " A process arriving at the same instant a time slice ends joins the ready queue before the preempted process."
    blocks = [{"p": f"Consider the processes below. Draw the Gantt chart and find the completion time, turnaround time and waiting "
                    f"time of each process, and the average turnaround and waiting time, using {names}.{rule} Ties are broken by "
                    "arrival time, then by process number."}, {"table": table}]
    answer = []
    for a in algos:
        r = schedule(procs, a, quantum, larger)
        answer.append({"p": f"{CPU_ALGOS[a]}{f' (q = {quantum})' if a == 'rr' else ''} - Gantt chart:"})
        answer.append({"table": _gantt_rows(r["gantt"])})
        answer.append({"table": [["Process", "AT", "BT", "CT", "TAT = CT - AT", "WT = TAT - BT"]]
                       + [[x["pid"], str(x["at"]), str(x["bt"]), str(x["ct"]), str(x["tat"]), str(x["wt"])] for x in r["rows"]]})
        answer.append({"p": f"Average turnaround time = {r['avg_tat']}; average waiting time = {r['avg_wt']}."})
    return {"kind": "numerical", "topic": topic, "marks": marks, "blocks": blocks, "answer": answer,
            "data": {"procs": procs, "algos": algos, "quantum": quantum, "larger_is_higher": larger}}


# --------------------------------------------------------------------------- page replacement


def replace_pages(refs: list[int], frames: int, algo: str) -> dict[str, Any]:
    mem: list[int | None] = [None] * frames
    loaded: dict[int, int] = {}
    last: dict[int, int] = {}
    steps, faults = [], 0
    for i, r in enumerate(refs):
        hit = r in mem
        if not hit:
            faults += 1
            if None in mem:
                slot = mem.index(None)
            elif algo == "fifo":
                slot = min(range(frames), key=lambda k: loaded[mem[k]])  # type: ignore[index]
            elif algo == "lru":
                slot = min(range(frames), key=lambda k: last[mem[k]])  # type: ignore[index]
            else:  # optimal: evict the page used farthest in the future (never again = farthest; ties -> lowest frame)
                def nxt(k: int) -> int:
                    page = mem[k]
                    return next((j for j in range(i + 1, len(refs)) if refs[j] == page), len(refs) + k)
                slot = max(range(frames), key=lambda k: (nxt(k), -k))
            if mem[slot] is not None:
                loaded.pop(mem[slot], None)  # type: ignore[arg-type]
            mem[slot] = r
            loaded[r] = i
        last[r] = i
        steps.append({"ref": r, "frames": list(mem), "fault": not hit})
    return {"steps": steps, "faults": faults, "hits": len(refs) - faults}


PAGE_ALGOS = {"fifo": "FIFO", "lru": "LRU", "opt": "Optimal"}


def gen_pages(rng: random.Random, marks: int, difficulty: str, topic: str = "page-replacement") -> dict[str, Any]:
    length = {"easy": 12, "medium": 15, "hard": 20}.get(difficulty, 15)
    refs = [rng.randint(0, 7) for _ in range(length)]
    frames = rng.choice([3, 3, 4]) if difficulty != "easy" else 3
    algos = list(PAGE_ALGOS) if marks >= 5 else rng.sample(list(PAGE_ALGOS), 2 if marks >= 3 else 1)
    blocks = [{"p": f"Consider the page reference string {', '.join(map(str, refs))} with {frames} page frames, initially empty. "
                    f"How many page faults occur with {', '.join(PAGE_ALGOS[a] for a in algos)} page replacement? Show the frame contents after every reference."}]
    answer = []
    for a in algos:
        r = replace_pages(refs, frames, a)
        table = [["Ref", *[str(s["ref"]) for s in r["steps"]]]]
        for f in range(frames):
            table.append([f"F{f + 1}", *[("-" if s["frames"][f] is None else str(s["frames"][f])) for s in r["steps"]]])
        table.append(["Fault", *["F" if s["fault"] else "" for s in r["steps"]]])
        answer += [{"p": f"{PAGE_ALGOS[a]}:"}, {"table": table}, {"p": f"{PAGE_ALGOS[a]} page faults = {r['faults']} (hits = {r['hits']})."}]
    return {"kind": "numerical", "topic": topic, "marks": marks, "blocks": blocks, "answer": answer, "data": {"refs": refs, "frames": frames, "algos": algos}}


# --------------------------------------------------------------------------- Banker's algorithm


def bankers(alloc: list[list[int]], mx: list[list[int]], avail: list[int]) -> dict[str, Any]:
    n, m = len(alloc), len(avail)
    need = [[mx[i][j] - alloc[i][j] for j in range(m)] for i in range(n)]
    work, finish, seq, trace = list(avail), [False] * n, [], []
    while True:  # always run the lowest-numbered process that can finish, then rescan from P0
        i = next((i for i in range(n) if not finish[i] and all(need[i][j] <= work[j] for j in range(m))), None)
        if i is None:
            break
        work = [work[j] + alloc[i][j] for j in range(m)]
        finish[i] = True
        seq.append(i)
        trace.append((i, list(work)))
    return {"need": need, "safe": all(finish), "sequence": seq, "trace": trace}


def gen_bankers(rng: random.Random, marks: int, difficulty: str, topic: str = "bankers") -> dict[str, Any]:
    n = {"easy": 4, "medium": 5, "hard": 5}.get(difficulty, 5)
    m = 3
    for _ in range(200):
        mx = [[rng.randint(1, 9) for _ in range(m)] for _ in range(n)]
        alloc = [[rng.randint(0, mx[i][j]) for j in range(m)] for i in range(n)]
        avail = [rng.randint(1, 5) for _ in range(m)]
        r = bankers(alloc, mx, avail)
        if r["safe"] or (difficulty == "hard" and rng.random() < 0.3):
            break
    res = "ABC"
    head = ["Process", *[f"Alloc {c}" for c in res], *[f"Max {c}" for c in res]]
    table = [head] + [[f"P{i}", *map(str, alloc[i]), *map(str, mx[i])] for i in range(n)]
    blocks = [{"p": f"A system has {n} processes P0-P{n - 1} and three resource types A, B, C. Available = ({', '.join(map(str, avail))}). "
                    "Using the Banker's algorithm: (a) compute the Need matrix; (b) is the system in a safe state? If yes, give a safe sequence. "
                    "When several processes can run, pick the lowest-numbered one."}, {"table": table}]
    answer = [{"p": "Need = Max - Allocation:"}, {"table": [["Process", *[f"Need {c}" for c in res]]] + [[f"P{i}", *map(str, r["need"][i])] for i in range(n)]}]
    answer.append({"table": [["Step", "Process run", "Work after release"]] + [[str(k + 1), f"P{i}", f"({', '.join(map(str, w))})"] for k, (i, w) in enumerate(r["trace"])]})
    if r["safe"]:
        answer.append({"p": f"Safe state. Safe sequence: <{', '.join(f'P{i}' for i in r['sequence'])}>."})
    else:
        stuck = [f"P{i}" for i in range(n) if i not in r["sequence"]]
        answer.append({"p": f"Unsafe state: {', '.join(stuck)} can never get their Need with the available resources, so no safe sequence exists."})
    return {"kind": "numerical", "topic": topic, "marks": marks, "blocks": blocks, "answer": answer, "data": {"alloc": alloc, "max": mx, "avail": avail}}


# --------------------------------------------------------------------------- disk scheduling

DISK_ALGOS = {"fcfs": "FCFS", "sstf": "SSTF", "scan": "SCAN", "cscan": "C-SCAN", "look": "LOOK", "clook": "C-LOOK"}


def disk(head: int, reqs: list[int], algo: str, cylinders: int, toward_high: bool = True) -> dict[str, Any]:
    """Head path and total head movement. SCAN/C-SCAN go to the last cylinder; the C-SCAN/C-LOOK return jump IS counted."""
    lo, hi = sorted(r for r in reqs if r < head), sorted(r for r in reqs if r >= head)
    if algo == "fcfs":
        path = list(reqs)
    elif algo == "sstf":
        left, path, cur = list(reqs), [], head
        while left:
            nxt = min(left, key=lambda r: (abs(r - cur), r))
            path.append(nxt)
            left.remove(nxt)
            cur = nxt
    elif toward_high:
        if algo == "scan":
            path = hi + ([cylinders - 1] if lo and (not hi or hi[-1] != cylinders - 1) else []) + lo[::-1]
        elif algo == "cscan":
            path = hi + ([cylinders - 1, 0] if lo else []) + lo
        elif algo == "look":
            path = hi + lo[::-1]
        else:
            path = hi + lo
    else:
        lo_d, hi_a = lo[::-1], hi
        if algo == "scan":
            path = lo_d + ([0] if hi_a and (not lo_d or lo_d[-1] != 0) else []) + hi_a
        elif algo == "cscan":
            path = lo_d + ([0, cylinders - 1] if hi_a else []) + hi_a[::-1]
        elif algo == "look":
            path = lo_d + hi_a
        else:
            path = lo_d + hi_a[::-1]
    total, cur = 0, head
    for p in path:
        total += abs(p - cur)
        cur = p
    return {"path": [head, *path], "total": total}


def gen_disk(rng: random.Random, marks: int, difficulty: str, topic: str = "disk-sched") -> dict[str, Any]:
    cyl = rng.choice([200, 200, 500])
    k = {"easy": 6, "medium": 8, "hard": 9}.get(difficulty, 8)
    reqs = rng.sample(range(1, cyl - 1), k)
    head = rng.randint(cyl // 5, cyl * 3 // 5)
    algos = rng.sample(list(DISK_ALGOS), 4 if marks >= 6 else 2 if marks >= 3 else 1)
    up = rng.random() < 0.6
    blocks = [{"p": f"A disk has {cyl} cylinders (0-{cyl - 1}). The head is at cylinder {head} and moving towards "
                    f"{'higher' if up else 'lower'} cylinder numbers. The request queue is: {', '.join(map(str, reqs))}. Find the order of "
                    f"service and the total head movement for {', '.join(DISK_ALGOS[a] for a in algos)}. "
                    "For SCAN and C-SCAN the head travels to the end of the disk; the return jump of C-SCAN and C-LOOK is counted."}]
    answer = []
    for a in algos:
        r = disk(head, reqs, a, cyl, up)
        answer.append({"p": f"{DISK_ALGOS[a]}: {' -> '.join(map(str, r['path']))}. Total head movement = {r['total']} cylinders."})
    return {"kind": "numerical", "topic": topic, "marks": marks, "blocks": blocks, "answer": answer, "data": {"head": head, "reqs": reqs, "cyl": cyl, "up": up, "algos": algos}}


# --------------------------------------------------------------------------- contiguous allocation


def fit(holes: list[int], procs: list[int], algo: str) -> dict[str, Any]:
    free = list(holes)
    placed: list[int | None] = []
    for size in procs:
        cands = [i for i, h in enumerate(free) if h >= size]
        if not cands:
            placed.append(None)
            continue
        i = cands[0] if algo == "first" else min(cands, key=lambda k: (free[k], k)) if algo == "best" else max(cands, key=lambda k: (free[k], -k))
        free[i] -= size
        placed.append(i)
    return {"placed": placed, "left": free}


def gen_fit(rng: random.Random, marks: int, difficulty: str, topic: str = "contiguous") -> dict[str, Any]:
    holes = [rng.choice(range(100, 700, 50)) for _ in range(5)]
    procs = [rng.randint(80, 500) for _ in range(4)]
    blocks = [{"p": f"Given free memory holes of {', '.join(f'{h} KB' for h in holes)} (in order), and processes of "
                    f"{', '.join(f'{p} KB' for p in procs)} (in order), how would First fit, Best fit and Worst fit place them? "
                    "A hole keeps its leftover space after a placement. Which algorithm uses memory best here?"}]
    answer, used = [], {}
    for a, name in (("first", "First fit"), ("best", "Best fit"), ("worst", "Worst fit")):
        r = fit(holes, procs, a)
        cells = [f"{p} KB -> hole {i + 1} ({holes[i]} KB)" if i is not None else f"{p} KB must wait" for p, i in zip(procs, r["placed"])]
        used[name] = sum(1 for i in r["placed"] if i is not None)
        answer.append({"p": f"{name}: " + "; ".join(cells) + f". Holes left: {', '.join(map(str, r['left']))} KB."})
    best = max(used.values())
    answer.append({"p": f"Most processes placed: {', '.join(k for k, v in used.items() if v == best)} ({best} of {len(procs)})."})
    return {"kind": "numerical", "topic": topic, "marks": marks, "blocks": blocks, "answer": answer, "data": {"holes": holes, "procs": procs}}


# --------------------------------------------------------------------------- paging and EAT


def gen_paging(rng: random.Random, marks: int, difficulty: str, topic: str = "paging") -> dict[str, Any]:
    page_kb = rng.choice([1, 2, 4])
    page = page_kb * 1024
    pages = rng.choice([8, 16])
    frames = list(range(pages * 2))
    table = rng.sample(frames, pages)
    addrs = [rng.randrange(0, pages * page) for _ in range(3)]
    blocks = [{"p": f"A system uses paging with a page size of {page_kb} KB ({page} bytes) and a logical address space of {pages} pages. "
                    f"The page table maps page i to the frame below. (a) How many bits are in a logical address, and how many are page "
                    f"number and offset? (b) Translate the logical addresses {', '.join(map(str, addrs))} to physical addresses."},
              {"table": [["Page", *map(str, range(pages))], ["Frame", *map(str, table)]]}]
    off_bits = page.bit_length() - 1
    pg_bits = pages.bit_length() - 1
    answer = [{"p": f"(a) Offset bits = log2({page}) = {off_bits}; page-number bits = log2({pages}) = {pg_bits}; logical address = {off_bits + pg_bits} bits."}]
    rows = [["Logical", "Page = L div size", "Offset = L mod size", "Frame", "Physical = frame x size + offset"]]
    for a in addrs:
        p, o = divmod(a, page)
        rows.append([str(a), str(p), str(o), str(table[p]), str(table[p] * page + o)])
    answer += [{"p": "(b)"}, {"table": rows}]
    return {"kind": "numerical", "topic": topic, "marks": marks, "blocks": blocks, "answer": answer, "data": {"page": page, "table": table, "addrs": addrs}}


def gen_eat(rng: random.Random, marks: int, difficulty: str, topic: str = "tlb") -> dict[str, Any]:
    t = rng.choice([10, 20, 20, 30])
    m = rng.choice([50, 80, 100, 120])
    h = rng.choice([80, 85, 90, 95, 98])
    blocks = [{"p": f"A paging system has a TLB lookup time of {t} ns and a memory access time of {m} ns. The TLB hit ratio is {h}%. "
                    "Find the effective memory access time (EAT). The TLB is searched before the page table; on a miss the page table is "
                    "read from memory, then the data."}]
    hit, miss = t + m, t + 2 * m
    eat = Fraction(h, 100) * hit + Fraction(100 - h, 100) * miss
    answer = [{"p": f"Hit: {t} + {m} = {hit} ns. Miss: {t} + 2 x {m} = {miss} ns."},
              {"p": f"EAT = {h / 100:.2f} x {hit} + {(100 - h) / 100:.2f} x {miss} = {float(eat):g} ns."}]
    return {"kind": "numerical", "topic": topic, "marks": marks, "blocks": blocks, "answer": answer, "data": {"t": t, "m": m, "h": h}}


GENERATORS = {"cpu": gen_cpu, "pages": gen_pages, "bankers": gen_bankers, "disk": gen_disk, "fit": gen_fit, "paging": gen_paging, "eat": gen_eat}


def generate(kind: str, rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    return GENERATORS[kind](rng, marks, difficulty, topic)
