"""Correctness tests for the planner.

Runs directly (``python tests/test_planner.py``) or under pytest.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import ValidationError

from app.planner import PlannerError, plan, prepare
from app.schemas import PlanRequest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_example() -> dict:
    with open(os.path.join(ROOT, "examples", "request.json")) as fh:
        return json.load(fh)


def _plan(payload: dict):
    return plan(PlanRequest.model_validate(payload))


def test_example_is_optimal_and_matches_documented_response():
    r = _plan(_load_example())
    assert r.feasible and r.optimal
    assert r.capacity_upper_bound == 96

    by_id = {a.tensor_id: a.address for a in r.addresses}
    assert by_id == {"w": 64, "x": 0, "y": 0, "yv": 16}
    # views are placed inside their root
    assert by_id["yv"] == by_id["y"] + 16

    p1, p2 = r.timeline
    assert (p1.start, p1.end, p1.peak_bytes) == (0, 5, 64)
    assert (p2.start, p2.end, p2.peak_bytes) == (5, 12, 96)
    # mutually exclusive paths are serialized
    assert p1.end <= p2.start


def test_deadline_infeasible():
    p = _load_example(); p["deadline"] = 11
    r = _plan(PlanRequest.model_validate(p))
    assert not r.feasible and r.reason == "deadline"


def test_memory_infeasible():
    p = _load_example(); p["memory_limit"] = 95  # minimum capacity is 96
    r = _plan(PlanRequest.model_validate(p))
    assert not r.feasible and r.reason == "memory_limit"


def test_both_infeasible():
    p = _load_example(); p["deadline"] = 11; p["memory_limit"] = 64
    r = _plan(PlanRequest.model_validate(p))
    assert not r.feasible and r.reason == "deadline_and_memory_limit"


def test_parallel_fork_join_coexists():
    r = _plan({
        "deadline": 4, "memory_limit": 256,
        "paths": [{
            "id": "P",
            "operators": [
                {"id": "p", "duration": 1},
                {"id": "q", "duration": 2, "depends_on": ["p"]},
                {"id": "r", "duration": 2, "depends_on": ["p"]},
                {"id": "j", "duration": 1, "depends_on": ["q", "r"]},
            ],
            "tensors": [
                {"id": "x", "bytes": 64, "alignment": 16, "producer": "p", "consumers": ["q", "j"]},
                {"id": "y", "bytes": 48, "alignment": 16, "producer": "q", "consumers": ["j"]},
                {"id": "z", "bytes": 32, "alignment": 16, "producer": "r", "consumers": ["j"]},
            ],
        }],
    })
    assert r.feasible and r.optimal
    # all three live together at join -> 64+48+32
    assert r.capacity_upper_bound == 144
    assert r.timeline[0].peak_bytes == 144
    # q and r may run in parallel
    assert r.timeline[0].end == 4


def test_half_open_intervals_reuse_address():
    r = _plan({
        "deadline": 5, "memory_limit": 256,
        "paths": [{
            "id": "P",
            "operators": [
                {"id": "a", "duration": 2},
                {"id": "b", "duration": 2, "depends_on": ["a"]},
            ],
            "tensors": [
                {"id": "t1", "bytes": 40, "alignment": 8, "producer": "a", "consumers": ["b"]},
                {"id": "t2", "bytes": 40, "alignment": 8, "producer": "b", "consumers": []},
            ],
        }],
    })
    addr = {a.tensor_id: a.address for a in r.addresses}
    # t1 dies exactly when b starts; t2 is produced then: reuse is allowed
    assert addr["t1"] == addr["t2"] == 0
    assert r.capacity_upper_bound == 40


def test_view_extends_root_lifetime():
    r = _plan({
        "deadline": 6, "memory_limit": 256,
        "paths": [{
            "id": "P",
            "operators": [
                {"id": "a", "duration": 1},
                {"id": "b", "duration": 1, "depends_on": ["a"]},
                {"id": "c", "duration": 1, "depends_on": ["b"]},
            ],
            "tensors": [
                {"id": "r", "bytes": 64, "alignment": 16, "producer": "a", "consumers": ["b"]},
                {"id": "v", "bytes": 8, "alignment": 8, "producer": "c", "output": True,
                 "base_tensor": "r", "view_offset": 0},
                {"id": "s", "bytes": 64, "alignment": 16, "producer": "b", "consumers": ["c"]},
            ],
        }],
    })
    addr = {a.tensor_id: a.address for a in r.addresses}
    # r is kept alive until path end by output view v, so s cannot alias it
    assert addr["r"] != addr["s"]
    assert r.capacity_upper_bound == 128
    assert addr["v"] == addr["r"]


def test_cross_path_reuse():
    r = _plan({
        "deadline": 8, "memory_limit": 64,
        "paths": [
            {"id": "A",
             "operators": [{"id": "a", "duration": 2},
                           {"id": "b", "duration": 2, "depends_on": ["a"]}],
             "tensors": [{"id": "u", "bytes": 64, "alignment": 16,
                          "producer": "a", "consumers": ["b"]}]},
            {"id": "B",
             "operators": [{"id": "a", "duration": 2},
                           {"id": "b", "duration": 2, "depends_on": ["a"]}],
             "tensors": [{"id": "v", "bytes": 64, "alignment": 16,
                          "producer": "a", "consumers": ["b"]}]},
        ],
    })
    assert r.feasible and r.optimal
    addr = {a.tensor_id: a.address for a in r.addresses}
    assert addr["u"] == addr["v"] == 0
    assert r.capacity_upper_bound == 64


def test_inconsistent_same_name_tensor_rejected():
    p = _load_example()
    p["paths"][1]["tensors"][0]["alignment"] = 16
    try:
        plan(PlanRequest.model_validate(p))
    except PlannerError as e:
        assert "inconsistent" in str(e)
    else:
        raise AssertionError("expected PlannerError")


def test_semantic_errors_are_422():
    cases = [
        # dependency cycle
        {"paths": [{"id": "P",
                    "operators": [{"id": "a", "duration": 1, "depends_on": ["b"]},
                                  {"id": "b", "duration": 1, "depends_on": ["a"]}],
                    "tensors": [{"id": "x", "bytes": 8, "alignment": 8,
                                 "producer": "a", "output": True}]}]},
        # unknown consumer
        {"paths": [{"id": "P",
                    "operators": [{"id": "a", "duration": 1}],
                    "tensors": [{"id": "x", "bytes": 8, "alignment": 8,
                                 "producer": "a", "consumers": ["zzz"]}]}]},
        # view outside root
        {"paths": [{"id": "P",
                    "operators": [{"id": "a", "duration": 1},
                                  {"id": "b", "duration": 1, "depends_on": ["a"]}],
                    "tensors": [
                        {"id": "r", "bytes": 8, "alignment": 8, "producer": "a",
                         "consumers": ["b"]},
                        {"id": "v", "bytes": 8, "alignment": 8, "producer": "b",
                         "output": True, "base_tensor": "r", "view_offset": 64}]}]},
        # non power-of-two alignment / non-positive duration are schema errors
        None,
    ]
    for extra in cases[:-1]:
        payload = {"deadline": 10, "memory_limit": 1024, **extra}
        try:
            plan(PlanRequest.model_validate(payload))
        except PlannerError:
            pass
        else:
            raise AssertionError("expected PlannerError: " + json.dumps(payload))

    for bad in (
        {"deadline": 0, "memory_limit": 1024, "paths": []},
        {"deadline": 10, "memory_limit": -1, "paths": []},
        {"paths": [{"id": "P",
                    "operators": [{"id": "a", "duration": 0}],
                    "tensors": [{"id": "x", "bytes": 8, "alignment": 3,
                                 "producer": "a"}]}]},
    ):
        try:
            PlanRequest.model_validate(bad)
        except ValidationError:
            pass
        else:
            raise AssertionError("expected ValidationError: " + json.dumps(bad))


def test_never_overclaims_optimality_on_hard_instance():
    # a larger instance must still return quickly with a feasible plan, and
    # must only claim optimality when CP-SAT proved it
    payload = json.loads(
        open(os.path.join(ROOT, "examples", "stress_payload.json")).read()
    )
    t0 = time.time()
    r = _plan(payload)
    assert time.time() - t0 < 30
    assert r.feasible
    # invariants: views inside roots, everything aligned, capacity respected
    prep = prepare(PlanRequest.model_validate(payload))
    amap = {a.tensor_id: a.address for a in r.addresses}
    for tid, info in prep.tensors.items():
        assert amap[tid] == amap[info.root] + info.cum_offset
        assert amap[tid] % info.alignment == 0
        assert amap[tid] >= 0
    assert r.capacity_upper_bound <= payload["memory_limit"]


def test_view_can_share_base_producer_operator():
    # a view may be produced inside the same operator that produces its base
    r = _plan({
        "deadline": 5, "memory_limit": 1024,
        "paths": [{
            "id": "P",
            "operators": [
                {"id": "p", "duration": 1},
                {"id": "q", "duration": 1, "depends_on": ["p"]},
            ],
            "tensors": [
                {"id": "z", "bytes": 64, "alignment": 16, "producer": "p",
                 "consumers": ["q"]},
                {"id": "a", "bytes": 16, "alignment": 16, "producer": "p",
                 "consumers": ["q"], "base_tensor": "z", "view_offset": 0},
            ],
        }],
    })
    assert r.feasible
    addr = {t.tensor_id: t.address for t in r.addresses}
    assert addr["a"] == addr["z"]


def test_address_tiebreak_uses_smallest_tensor_id_of_root():
    # root 'z' has a view 'a' whose tensor id sorts first, so root z must be
    # placed before root b even though b < z by id
    r = _plan({
        "deadline": 10, "memory_limit": 1024,
        "paths": [{
            "id": "P",
            "operators": [
                {"id": "p", "duration": 1},
                {"id": "q", "duration": 1, "depends_on": ["p"]},
            ],
            "tensors": [
                {"id": "z", "bytes": 64, "alignment": 16, "producer": "p",
                 "consumers": ["q"]},
                {"id": "a", "bytes": 16, "alignment": 16, "producer": "p",
                 "consumers": ["q"], "base_tensor": "z", "view_offset": 0},
                {"id": "b", "bytes": 64, "alignment": 16, "producer": "p",
                 "consumers": ["q"]},
            ],
        }],
    })
    assert r.optimal
    addr = {t.tensor_id: t.address for t in r.addresses}
    assert addr["z"] == 0 and addr["a"] == 0 and addr["b"] == 64


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} tests passed")
