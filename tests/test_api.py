"""HTTP-level tests for POST /plans using FastAPI's in-process client."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(ROOT, "examples", "request.json")) as fh:
    EXAMPLE = json.load(fh)
with open(os.path.join(ROOT, "examples", "response.json")) as fh:
    DOC_RESPONSE = json.load(fh)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_plan_matches_documented_example():
    r = client.post("/plans", json=EXAMPLE)
    assert r.status_code == 200
    assert r.json() == DOC_RESPONSE


def test_infeasible_deadline():
    p = json.loads(json.dumps(EXAMPLE)); p["deadline"] = 11
    r = client.post("/plans", json=p)
    assert r.status_code == 200
    assert r.json()["feasible"] is False and r.json()["reason"] == "deadline"


def test_semantic_422():
    p = json.loads(json.dumps(EXAMPLE))
    p["paths"][1]["tensors"][0]["alignment"] = 16  # inconsistent with P1
    r = client.post("/plans", json=p)
    assert r.status_code == 422
    assert "inconsistent" in r.json()["detail"][0]["msg"]


def test_schema_422():
    p = json.loads(json.dumps(EXAMPLE)); p["deadline"] = -1
    r = client.post("/plans", json=p)
    assert r.status_code == 422


def test_malformed_json_422():
    r = client.post("/plans", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_unknown_field_422():
    p = json.loads(json.dumps(EXAMPLE)); p["bogus"] = 1
    r = client.post("/plans", json=p)
    assert r.status_code == 422


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} tests passed")
