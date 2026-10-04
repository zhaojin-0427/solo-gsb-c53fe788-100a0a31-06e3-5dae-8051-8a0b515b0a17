"""Randomised fuzz tests: every produced plan is checked by an independent
verifier that re-derives all constraints straight from the specification."""
from __future__ import annotations

import random

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


# --------------------------------------------------------------------------
# independent plan verifier
# --------------------------------------------------------------------------

def verify(request: dict, plan: dict) -> None:
    assert plan["status"] in ("optimal", "feasible")
    deadline = request["deadline"]
    assert plan["address_bound"] <= request["memory_limit"]

    # global tensor definitions (consistent across paths by validation)
    defs = {}
    for p in request["paths"]:
        for t in p["tensors"]:
            defs.setdefault(t["id"], t)

    def root_of(tid):
        off = 0
        cur = tid
        while defs[cur]["base"] is not None:
            off += defs[cur]["offset"]
            cur = defs[cur]["base"]
        return cur, off

    peaks = []
    for p_req, p_plan in zip(request["paths"], plan["paths"]):
        assert p_req["id"] == p_plan["id"]
        ops = {o["id"]: o for o in p_req["operators"]}
        tensors = {t["id"]: t for t in p_req["tensors"]}
        starts = {o["id"]: o["start"] for o in p_plan["operators"]}
        ends = {o["id"]: o["end"] for o in p_plan["operators"]}

        for oid, o in ops.items():
            assert ends[oid] - starts[oid] == o["duration"], "duration"
            assert 0 <= starts[oid] and ends[oid] <= deadline, "deadline"
            for d in o["dependencies"]:
                assert starts[oid] >= ends[d], "dependency order"

        if ops:
            assert p_plan["start"] == min(starts.values())
            assert p_plan["end"] == max(ends.values())
        else:
            assert p_plan["start"] == 0 and p_plan["end"] == 0
        assert p_plan["end"] <= deadline

        # producer -> consumer data flow
        for t in tensors.values():
            if t["producer"] is not None:
                for c in t["consumers"]:
                    assert starts[c] >= ends[t["producer"]], "data flow"

        addrs = {t["id"]: t["address"] for t in p_plan["tensors"]}
        # address derivation + alignment
        for tid, t in tensors.items():
            root, off = root_of(tid)
            assert addrs[tid] == addrs[root] + off, "view address"
            assert addrs[tid] % t["alignment"] == 0, "alignment"
            assert addrs[tid] >= 0

        # lifetimes of base tensors (views extend their root)
        children = {}
        for tid, t in tensors.items():
            if t["base"] is not None:
                children.setdefault(t["base"], []).append(tid)

        def views_of(rid):
            out, stack = [], list(children.get(rid, ()))
            while stack:
                u = stack.pop()
                out.append(u)
                stack.extend(children.get(u, ()))
            return out

        def birth(t):
            if t["producer"] is not None:
                return ends[t["producer"]]
            return p_plan["start"]

        def deaths(t):
            out = [ends[c] for c in t["consumers"]]
            if t["is_output"]:
                out.append(p_plan["end"])
            return out

        windows = {}
        for tid, t in tensors.items():
            if t["base"] is not None:
                continue
            related = [tid] + views_of(tid)
            b = min(birth(tensors[u]) for u in related)
            d = max([b] + [x for u in related for x in deaths(tensors[u])])
            windows[tid] = (b, d)

        # simultaneously alive base tensors must not overlap
        ids = sorted(windows)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b_ = ids[i], ids[j]
                b0, b1 = windows[a]
                d0, d1 = windows[b_]
                # half-open intersection test (empty windows never overlap)
                if max(b0, d0) < min(b1, d1):
                    lo = max(addrs[a], addrs[b_])
                    hi = min(addrs[a] + tensors[a]["bytes"],
                             addrs[b_] + tensors[b_]["bytes"])
                    assert lo >= hi, f"address overlap {a} {b_}"

        peak = 0
        for tid, (b, d) in windows.items():
            if d > b:
                peak = max(peak, addrs[tid] + tensors[tid]["bytes"])
        assert p_plan["peak"] == peak, "per-path peak"
        peaks.append(peak)

    assert plan["address_bound"] == max(peaks), "address bound"


# --------------------------------------------------------------------------
# random instance generator
# --------------------------------------------------------------------------

def random_request(rng: random.Random) -> dict:
    n_paths = rng.randint(1, 3)
    shared = []
    if rng.random() < 0.7:
        shared = [{
            "id": "w_shared",
            "bytes": rng.choice([16, 32, 64]),
            "alignment": rng.choice([1, 2, 4, 8, 16]),
            "base": None,
            "offset": 0,
            "producer": None,
            "consumers": [],
            "is_output": False,
        }]
    paths = []
    for pi in range(n_paths):
        n_ops = rng.randint(1, 8)
        ops = []
        for oi in range(n_ops):
            deps = [f"o{j}" for j in range(oi) if rng.random() < 0.35]
            ops.append({
                "id": f"o{oi}",
                "duration": rng.randint(1, 6),
                "dependencies": deps,
            })
        tensors = []
        n_tensors = rng.randint(0, 6)
        made = []
        for ti in range(n_tensors):
            tid = f"t{pi}_{ti}"  # tensor ids are path-local except shared ones
            is_view = made and rng.random() < 0.3
            if is_view:
                base = rng.choice(made)
                base_def = next(t for t in tensors if t["id"] == base)
                size = rng.randint(1, base_def["bytes"])
                tensors.append({
                    "id": tid,
                    "bytes": size,
                    "alignment": 1,  # keep views satisfiable
                    "base": base,
                    "offset": rng.randint(0, base_def["bytes"] - size),
                    "producer": rng.choice(ops)["id"] if rng.random() < 0.8 else None,
                    "consumers": [],
                    "is_output": False,
                })
            else:
                tensors.append({
                    "id": tid,
                    "bytes": rng.randint(1, 64),
                    "alignment": rng.choice([1, 2, 4, 8]),
                    "base": None,
                    "offset": 0,
                    "producer": rng.choice(ops)["id"] if rng.random() < 0.8 else None,
                    "consumers": [],
                    "is_output": False,
                })
            made.append(tid)
        # wire consumers: each tensor is consumed by random later ops so the
        # data-flow graph stays acyclic
        op_index = {o["id"]: i for i, o in enumerate(ops)}
        for t in tensors:
            if t["producer"] is not None:
                later = [o["id"] for o in ops
                         if op_index[o["id"]] > op_index[t["producer"]]]
                t["consumers"] = rng.sample(
                    later, k=min(rng.randint(0, 2), len(later)))
            if not t["consumers"] and rng.random() < 0.4:
                t["is_output"] = True
        for s in shared:
            t = dict(s)
            t["consumers"] = rng.sample([o["id"] for o in ops],
                                        k=rng.randint(1, min(2, n_ops)))
            tensors.append(t)
        paths.append({"id": f"path{pi}", "operators": ops, "tensors": tensors})

    longest = 0
    for p in paths:  # crude longest-chain bound so most instances are feasible
        longest = max(longest, sum(o["duration"] for o in p["operators"]))
    return {
        "deadline": rng.randint(longest, longest + 10),
        "memory_limit": rng.randint(256, 8192),
        "paths": paths,
    }


def test_fuzz_random_instances_are_valid():
    rng = random.Random(20261004)
    checked = 0
    for _ in range(60):
        payload = random_request(rng)
        resp = client.post("/plans", json=payload)
        assert resp.status_code == 200, resp.text
        plan = resp.json()
        if plan["status"] == "infeasible":
            continue
        verify(payload, plan)
        checked += 1
    assert checked >= 40  # most random instances should be feasible
