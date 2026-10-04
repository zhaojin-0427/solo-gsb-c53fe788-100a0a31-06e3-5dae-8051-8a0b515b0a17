"""End-to-end tests for the buffer planner (solver + HTTP layer)."""
from __future__ import annotations

import copy

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def op(oid, duration, deps=()):
    return {"id": oid, "duration": duration, "dependencies": list(deps)}


def tensor(tid, size, align=1, base=None, offset=0, producer=None,
           consumers=(), is_output=False):
    return {
        "id": tid,
        "bytes": size,
        "alignment": align,
        "base": base,
        "offset": offset,
        "producer": producer,
        "consumers": list(consumers),
        "is_output": is_output,
    }


def path(pid, operators, tensors):
    return {"id": pid, "operators": operators, "tensors": tensors}


def post(payload):
    return client.post("/plans", json=payload)


def solve(payload):
    resp = post(payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


def addr_of(plan, path_id, tensor_id):
    p = next(p for p in plan["paths"] if p["id"] == path_id)
    return next(t["address"] for t in p["tensors"] if t["id"] == tensor_id)


def op_window(plan, path_id, op_id):
    p = next(p for p in plan["paths"] if p["id"] == path_id)
    o = next(o for o in p["operators"] if o["id"] == op_id)
    return o["start"], o["end"]


# --------------------------------------------------------------------------
# scheduling + packing semantics
# --------------------------------------------------------------------------

def test_disjoint_lifetimes_share_address():
    payload = {
        "deadline": 10,
        "memory_limit": 1000,
        "paths": [path("p",
                       [op("a", 2), op("b", 3, ["a"]), op("c", 1, ["b"])],
                       [tensor("x", 8, 4, producer="a", consumers=["b"]),
                        tensor("y", 8, 4, producer="b", consumers=["c"])])],
    }
    plan = solve(payload)
    assert plan["status"] == "optimal"
    # x lives [2,5), y lives [5,6): disjoint -> same address.
    assert plan["address_bound"] == 8
    assert addr_of(plan, "p", "x") == 0
    assert addr_of(plan, "p", "y") == 0
    assert plan["paths"][0]["start"] == 0
    assert plan["paths"][0]["end"] == 6
    assert plan["paths"][0]["peak"] == 8
    assert op_window(plan, "p", "a") == (0, 2)
    assert op_window(plan, "p", "b") == (2, 5)
    assert op_window(plan, "p", "c") == (5, 6)


def test_overlapping_tensors_get_distinct_addresses():
    payload = {
        "deadline": 10,
        "memory_limit": 1000,
        "paths": [path("p",
                       [op("a", 2), op("b", 2), op("c", 1, ["a", "b"])],
                       [tensor("x", 8, 4, producer="a", consumers=["c"]),
                        tensor("y", 4, 4, producer="b", consumers=["c"])])],
    }
    plan = solve(payload)
    assert plan["status"] == "optimal"
    # x and y are both alive during [2,3).
    assert plan["address_bound"] == 12
    # tensor-id lexicographic order: x is minimised first.
    assert addr_of(plan, "p", "x") == 0
    assert addr_of(plan, "p", "y") == 8


def test_lexicographic_address_order_follows_tensor_id():
    # t_a sorts before t_b; t_a must get the lower address even though
    # it is the smaller tensor.
    payload = {
        "deadline": 10,
        "memory_limit": 1000,
        "paths": [path("p",
                       [op("a", 1), op("b", 1), op("c", 1, ["a", "b"])],
                       [tensor("t_a", 4, 4, producer="a", consumers=["c"]),
                        tensor("t_b", 8, 4, producer="b", consumers=["c"])])],
    }
    plan = solve(payload)
    assert addr_of(plan, "p", "t_a") == 0
    assert addr_of(plan, "p", "t_b") == 4
    assert plan["address_bound"] == 12


def test_alignment_is_respected():
    payload = {
        "deadline": 10,
        "memory_limit": 1000,
        "paths": [path("p",
                       [op("a", 1), op("b", 1), op("c", 1, ["a", "b"])],
                       [tensor("x", 4, 8, producer="a", consumers=["c"]),
                        tensor("y", 4, 8, producer="b", consumers=["c"])])],
    }
    plan = solve(payload)
    assert addr_of(plan, "p", "x") == 0
    assert addr_of(plan, "p", "y") == 8  # 4 would violate 8-byte alignment
    assert plan["address_bound"] == 12


def test_view_borrows_base_and_extends_its_lifetime():
    payload = {
        "deadline": 10,
        "memory_limit": 1000,
        "paths": [path("p",
                       [op("a", 1), op("b", 1, ["a"]), op("c", 1, ["b"])],
                       [tensor("x", 16, 8, producer="a", consumers=["b"]),
                        tensor("v", 4, 4, base="x", offset=4,
                               producer="b", consumers=["c"]),
                        tensor("y", 8, 8, producer="a", consumers=["c"])])],
    }
    plan = solve(payload)
    assert plan["status"] == "optimal"
    # v keeps x alive until t=3, so x overlaps y ([1,3)) and they may
    # not share bytes.
    assert addr_of(plan, "p", "x") == 0
    assert addr_of(plan, "p", "v") == 4  # base address + offset
    assert addr_of(plan, "p", "y") == 16
    assert plan["address_bound"] == 24


def test_final_output_lives_until_path_end():
    payload = {
        "deadline": 10,
        "memory_limit": 1000,
        "paths": [path("p",
                       [op("a", 1), op("b", 1, ["a"])],
                       [tensor("x", 8, 8, producer="a", is_output=True),
                        tensor("y", 8, 8, consumers=["b"])])],
    }
    plan = solve(payload)
    # x (output) is alive [1,2); y (input) is alive [0,2) -> overlap.
    assert addr_of(plan, "p", "x") == 0
    assert addr_of(plan, "p", "y") == 8
    assert plan["address_bound"] == 16


def test_mutually_exclusive_paths_reuse_addresses():
    payload = {
        "deadline": 10,
        "memory_limit": 1000,
        "paths": [
            path("p1", [op("a", 1)],
                 [tensor("t1", 16, 8, consumers=["a"])]),
            path("p2", [op("b", 1)],
                 [tensor("t2", 8, 8, consumers=["b"])]),
        ],
    }
    plan = solve(payload)
    assert plan["status"] == "optimal"
    assert addr_of(plan, "p1", "t1") == 0
    assert addr_of(plan, "p2", "t2") == 0
    # bound is the max over paths, not the sum.
    assert plan["address_bound"] == 16
    assert plan["paths"][0]["peak"] == 16
    assert plan["paths"][1]["peak"] == 8


def test_shared_tensor_has_one_address_in_every_path():
    payload = {
        "deadline": 10,
        "memory_limit": 1000,
        "paths": [
            path("p1",
                 [op("a", 1), op("b", 1, ["a"])],
                 [tensor("w", 8, 4, consumers=["a", "b"]),
                  tensor("x", 8, 4, producer="a", consumers=["b"])]),
            path("p2",
                 [op("c", 2)],
                 [tensor("w", 8, 4, consumers=["c"]),
                  tensor("y", 4, 4, producer="c", is_output=True)]),
        ],
    }
    plan = solve(payload)
    assert plan["status"] == "optimal"
    assert addr_of(plan, "p1", "w") == 0
    assert addr_of(plan, "p2", "w") == 0  # same shared buffer
    assert addr_of(plan, "p1", "x") == 8  # overlaps w in p1
    assert plan["address_bound"] == 16
    assert plan["paths"][1]["peak"] == 8


def test_deadline_exact_boundary_is_feasible():
    payload = {
        "deadline": 6,
        "memory_limit": 1000,
        "paths": [path("p", [op("a", 2), op("b", 4, ["a"])], [])],
    }
    plan = solve(payload)
    assert plan["status"] == "optimal"
    assert plan["paths"][0]["end"] == 6


def test_infeasible_deadline():
    payload = {
        "deadline": 9,
        "memory_limit": 1000,
        "paths": [path("p", [op("a", 5), op("b", 5, ["a"])], [])],
    }
    plan = solve(payload)
    assert plan["status"] == "infeasible"
    assert plan["paths"] is None
    assert plan["address_bound"] is None


def test_infeasible_memory_limit():
    payload = {
        "deadline": 10,
        "memory_limit": 15,
        "paths": [path("p",
                       [op("a", 1), op("b", 1), op("c", 1, ["a", "b"])],
                       [tensor("x", 8, 8, producer="a", consumers=["c"]),
                        tensor("y", 8, 8, producer="b", consumers=["c"])])],
    }
    plan = solve(payload)
    assert plan["status"] == "infeasible"


def test_view_alignment_can_make_instance_infeasible():
    # base is 4-aligned, so base+2 can never be 4-aligned.
    payload = {
        "deadline": 10,
        "memory_limit": 1000,
        "paths": [path("p",
                       [op("a", 1), op("b", 1, ["a"])],
                       [tensor("x", 8, 4, producer="a", consumers=["b"]),
                        tensor("v", 2, 4, base="x", offset=2,
                               producer="a", consumers=["b"])])],
    }
    plan = solve(payload)
    assert plan["status"] == "infeasible"


def test_zero_lifetime_tensor_does_not_inflate_peak():
    # z is produced by the last operator: [prod_end, path_end) is empty.
    payload = {
        "deadline": 10,
        "memory_limit": 10000,
        "paths": [path("p",
                       [op("a", 1), op("b", 1, ["a"])],
                       [tensor("x", 8, 8, producer="a", consumers=["b"]),
                        tensor("z", 1000, 8, producer="b", is_output=True)])],
    }
    plan = solve(payload)
    assert plan["status"] == "optimal"
    assert plan["address_bound"] == 8
    assert addr_of(plan, "p", "z") == 0


def test_two_path_readme_example():
    payload = {
        "deadline": 12,
        "memory_limit": 4096,
        "paths": [
            path("chat",
                 [op("embed", 2), op("attn", 3, ["embed"]), op("head", 2, ["attn"])],
                 [tensor("weights", 1024, 256, consumers=["attn"]),
                  tensor("hidden", 512, 256, producer="embed", consumers=["attn"]),
                  tensor("kv", 768, 256, producer="attn", consumers=["head"]),
                  tensor("kv_tail", 256, 256, base="kv", offset=512,
                         producer="attn", consumers=["head"]),
                  tensor("logits", 512, 256, producer="head", is_output=True)]),
            path("embed_only",
                 [op("embed", 2), op("proj", 2, ["embed"])],
                 [tensor("weights", 1024, 256, consumers=["proj"]),
                  tensor("hidden", 512, 256, producer="embed", consumers=["proj"]),
                  tensor("vec", 256, 256, producer="proj", is_output=True)]),
        ],
    }
    plan = solve(payload)
    assert plan["status"] == "optimal"
    assert plan["address_bound"] == 1536
    chat, emb = plan["paths"]
    assert (chat["start"], chat["end"], chat["peak"]) == (0, 7, 1536)
    assert (emb["start"], emb["end"], emb["peak"]) == (0, 4, 1536)
    # tensor-id lexicographic order: hidden < weights, so hidden packs first.
    assert addr_of(plan, "chat", "hidden") == 0
    assert addr_of(plan, "chat", "weights") == 512
    assert addr_of(plan, "chat", "kv") == 0
    assert addr_of(plan, "chat", "kv_tail") == 512  # kv + offset
    # shared tensors keep the same address in both paths.
    assert addr_of(plan, "embed_only", "hidden") == 0
    assert addr_of(plan, "embed_only", "weights") == 512


def test_readme_example_files_are_in_sync():
    """examples/plan_request.json posted to /plans must yield exactly
    examples/plan_response.json (the README shows these files)."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    payload = json.loads((root / "examples" / "plan_request.json").read_text())
    expected = json.loads((root / "examples" / "plan_response.json").read_text())
    resp = post(payload)
    assert resp.status_code == 200
    assert resp.json() == expected


def test_data_flow_cycle_is_infeasible_not_an_error():
    # dependencies are acyclic, but producer/consumer edges form a cycle:
    # o1 -> t1 -> o2 -> t2 -> o1 can never be scheduled.
    payload = {
        "deadline": 100,
        "memory_limit": 1000,
        "paths": [path("p",
                       [op("o1", 1), op("o2", 1)],
                       [tensor("t1", 8, 1, producer="o1", consumers=["o2"]),
                        tensor("t2", 8, 1, producer="o2", consumers=["o1"])])],
    }
    plan = solve(payload)
    assert plan["status"] == "infeasible"


def test_long_dependency_chain():
    n = 50
    ops = [op(f"o{i}", 1, [f"o{i-1}"] if i else []) for i in range(n)]
    tensors = [tensor(f"t{i}", 8, 8, producer=f"o{i}", consumers=[f"o{i+1}"])
               for i in range(n - 1)]
    payload = {"deadline": n, "memory_limit": 1000,
               "paths": [path("p", ops, tensors)]}
    plan = solve(payload)
    assert plan["status"] == "optimal"
    # consecutive tensors never overlap in time: everything sits at 0.
    assert plan["address_bound"] == 8
    assert plan["paths"][0]["end"] == n


# --------------------------------------------------------------------------
# validation -> HTTP 422
# --------------------------------------------------------------------------

def _valid_payload():
    return {
        "deadline": 10,
        "memory_limit": 1000,
        "paths": [path("p", [op("a", 1)],
                       [tensor("x", 8, 4, producer="a", is_output=True)])],
    }


def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200


def test_422_on_negative_deadline():
    payload = _valid_payload()
    payload["deadline"] = -1
    assert post(payload).status_code == 422


def test_422_on_empty_paths():
    payload = _valid_payload()
    payload["paths"] = []
    assert post(payload).status_code == 422


def test_422_on_zero_duration():
    payload = _valid_payload()
    payload["paths"][0]["operators"][0]["duration"] = 0
    assert post(payload).status_code == 422


def test_422_on_zero_bytes():
    payload = _valid_payload()
    payload["paths"][0]["tensors"][0]["bytes"] = 0
    assert post(payload).status_code == 422


@pytest.mark.parametrize("align", [0, 3, 6, 12, 100])
def test_422_on_non_power_of_two_alignment(align):
    payload = _valid_payload()
    payload["paths"][0]["tensors"][0]["alignment"] = align
    assert post(payload).status_code == 422


def test_422_on_dependency_cycle():
    payload = _valid_payload()
    payload["paths"][0]["operators"] = [op("a", 1, ["b"]), op("b", 1, ["a"])]
    assert post(payload).status_code == 422


def test_422_on_self_dependency():
    payload = _valid_payload()
    payload["paths"][0]["operators"] = [op("a", 1, ["a"])]
    assert post(payload).status_code == 422


def test_422_on_unknown_dependency():
    payload = _valid_payload()
    payload["paths"][0]["operators"] = [op("a", 1, ["nope"])]
    assert post(payload).status_code == 422


def test_422_on_unknown_producer():
    payload = _valid_payload()
    payload["paths"][0]["tensors"][0]["producer"] = "nope"
    assert post(payload).status_code == 422


def test_422_on_unknown_consumer():
    payload = _valid_payload()
    payload["paths"][0]["tensors"][0]["consumers"] = ["nope"]
    assert post(payload).status_code == 422


def test_422_on_unknown_base():
    payload = _valid_payload()
    payload["paths"][0]["tensors"].append(
        tensor("v", 4, 4, base="nope", offset=0, producer="a"))
    assert post(payload).status_code == 422


def test_422_on_view_exceeding_base():
    payload = _valid_payload()
    payload["paths"][0]["tensors"].append(
        tensor("v", 8, 4, base="x", offset=4, producer="a"))  # 4+8 > 8
    assert post(payload).status_code == 422


def test_422_on_offset_without_base():
    payload = _valid_payload()
    payload["paths"][0]["tensors"][0]["offset"] = 4
    assert post(payload).status_code == 422


def test_422_on_cyclic_base_chain():
    payload = _valid_payload()
    payload["paths"][0]["tensors"] = [
        tensor("x", 8, 4, base="y", offset=0, producer="a"),
        tensor("y", 8, 4, base="x", offset=0, producer="a"),
    ]
    assert post(payload).status_code == 422


def test_422_on_inconsistent_shared_tensor():
    payload = _valid_payload()
    payload["paths"].append(
        path("q", [op("b", 1)],
             [tensor("x", 16, 4, producer="b", is_output=True)]))  # bytes differ
    assert post(payload).status_code == 422


def test_422_on_duplicate_path_id():
    payload = _valid_payload()
    payload["paths"].append(copy.deepcopy(payload["paths"][0]))
    assert post(payload).status_code == 422


def test_422_on_duplicate_operator_id():
    payload = _valid_payload()
    payload["paths"][0]["operators"].append(op("a", 2))
    assert post(payload).status_code == 422


def test_422_on_duplicate_tensor_id():
    payload = _valid_payload()
    payload["paths"][0]["tensors"].append(
        tensor("x", 8, 4, producer="a", is_output=True))
    assert post(payload).status_code == 422


def test_422_on_producer_also_consumer():
    payload = _valid_payload()
    payload["paths"][0]["tensors"][0]["consumers"] = ["a"]
    assert post(payload).status_code == 422


def test_422_on_duplicate_consumers():
    payload = _valid_payload()
    payload["paths"][0]["tensors"][0]["consumers"] = ["a", "a"]
    payload["paths"][0]["tensors"][0]["producer"] = None
    assert post(payload).status_code == 422


def test_422_on_malformed_json_body():
    resp = client.post("/plans", json={"deadline": "soon"})
    assert resp.status_code == 422
