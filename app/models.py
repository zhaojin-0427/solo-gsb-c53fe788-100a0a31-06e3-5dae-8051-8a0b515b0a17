"""Request/response schemas for the inference-graph buffer planner.

All semantic validation lives here: any violation raises ``ValueError``
inside a Pydantic validator, which FastAPI surfaces as HTTP 422.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class TensorDef(BaseModel):
    """A tensor definition, local to one path.

    The same tensor id may appear in several paths (a shared tensor); the
    physical attributes ``bytes``/``alignment``/``base``/``offset`` must then
    be identical in every definition (checked on ``PlanRequest``).
    """

    id: str = Field(min_length=1)
    bytes: int = Field(ge=1)
    alignment: int = Field(ge=1)
    base: Optional[str] = None          # set => this tensor is a view of `base`
    offset: int = Field(default=0, ge=0)  # byte offset of the view inside `base`
    producer: Optional[str] = None      # operator id; None => external input
    consumers: list[str] = Field(default_factory=list)
    is_output: bool = False             # final output: kept alive until path end

    @field_validator("alignment")
    @classmethod
    def _alignment_is_power_of_two(cls, v: int) -> int:
        if v & (v - 1) != 0:
            raise ValueError(f"alignment must be a power of two, got {v}")
        return v

    @model_validator(mode="after")
    def _check_view_fields(self) -> "TensorDef":
        if self.base is None and self.offset != 0:
            raise ValueError(f"tensor {self.id!r}: offset requires a base tensor")
        if self.base is not None and self.base == self.id:
            raise ValueError(f"tensor {self.id!r} cannot be its own base")
        if len(set(self.consumers)) != len(self.consumers):
            raise ValueError(f"tensor {self.id!r}: duplicate consumer ids")
        if self.producer is not None and self.producer in self.consumers:
            raise ValueError(
                f"tensor {self.id!r}: producer {self.producer!r} is also a consumer"
            )
        return self


class OperatorDef(BaseModel):
    id: str = Field(min_length=1)
    duration: int = Field(ge=1)
    dependencies: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_deps(self) -> "OperatorDef":
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError(f"operator {self.id!r}: duplicate dependency ids")
        if self.id in self.dependencies:
            raise ValueError(f"operator {self.id!r} cannot depend on itself")
        return self


class PathDef(BaseModel):
    """One mutually exclusive execution path of the inference graph."""

    id: str = Field(min_length=1)
    operators: list[OperatorDef] = Field(default_factory=list)
    tensors: list[TensorDef] = Field(default_factory=list)


class PlanRequest(BaseModel):
    deadline: int = Field(ge=0)
    memory_limit: int = Field(ge=0)
    paths: list[PathDef] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_semantics(self) -> "PlanRequest":
        path_ids = [p.id for p in self.paths]
        if len(set(path_ids)) != len(path_ids):
            raise ValueError("duplicate path ids")

        # Physical definition of every tensor id, must be consistent wherever
        # the same id is re-defined across paths.
        defs: dict[str, TensorDef] = {}

        for path in self.paths:
            op_ids = [o.id for o in path.operators]
            if len(set(op_ids)) != len(op_ids):
                raise ValueError(f"path {path.id!r}: duplicate operator ids")
            ops = {o.id: o for o in path.operators}

            for o in path.operators:
                for d in o.dependencies:
                    if d not in ops:
                        raise ValueError(
                            f"path {path.id!r}: operator {o.id!r} depends on "
                            f"unknown operator {d!r}"
                        )
            self._check_acyclic(path)

            tensor_ids = [t.id for t in path.tensors]
            if len(set(tensor_ids)) != len(tensor_ids):
                raise ValueError(f"path {path.id!r}: duplicate tensor ids")
            tensors = {t.id: t for t in path.tensors}

            for t in path.tensors:
                if t.producer is not None and t.producer not in ops:
                    raise ValueError(
                        f"path {path.id!r}: tensor {t.id!r} has unknown "
                        f"producer {t.producer!r}"
                    )
                for c in t.consumers:
                    if c not in ops:
                        raise ValueError(
                            f"path {path.id!r}: tensor {t.id!r} has unknown "
                            f"consumer {c!r}"
                        )
                if t.base is not None:
                    if t.base not in tensors:
                        raise ValueError(
                            f"path {path.id!r}: view {t.id!r} refers to base "
                            f"{t.base!r} which is not defined in this path"
                        )
                    base = tensors[t.base]
                    if t.offset + t.bytes > base.bytes:
                        raise ValueError(
                            f"path {path.id!r}: view {t.id!r} "
                            f"(offset {t.offset} + {t.bytes} bytes) exceeds "
                            f"base {t.base!r} ({base.bytes} bytes)"
                        )

                prev = defs.get(t.id)
                if prev is None:
                    defs[t.id] = t
                else:
                    same = (
                        prev.bytes == t.bytes
                        and prev.alignment == t.alignment
                        and prev.base == t.base
                        and prev.offset == t.offset
                    )
                    if not same:
                        raise ValueError(
                            f"inconsistent definitions of tensor {t.id!r} "
                            f"across paths"
                        )

        # View chains must be acyclic (base references form a DAG).
        for tid in defs:
            seen = {tid}
            cur = tid
            while defs[cur].base is not None:
                cur = defs[cur].base
                if cur in seen:
                    raise ValueError(f"cyclic base chain involving tensor {cur!r}")
                seen.add(cur)

        return self

    @staticmethod
    def _check_acyclic(path: PathDef) -> None:
        indegree = {o.id: 0 for o in path.operators}
        outgoing: dict[str, list[str]] = defaultdict(list)
        for o in path.operators:
            for d in set(o.dependencies):
                outgoing[d].append(o.id)
                indegree[o.id] += 1
        queue = deque(oid for oid, deg in indegree.items() if deg == 0)
        visited = 0
        while queue:
            u = queue.popleft()
            visited += 1
            for w in outgoing[u]:
                indegree[w] -= 1
                if indegree[w] == 0:
                    queue.append(w)
        if visited != len(path.operators):
            raise ValueError(f"path {path.id!r}: dependency graph has a cycle")


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class OperatorPlan(BaseModel):
    id: str
    start: int
    end: int


class TensorPlan(BaseModel):
    id: str
    address: int


class PathPlan(BaseModel):
    id: str
    start: int
    end: int
    peak: int
    operators: list[OperatorPlan]
    tensors: list[TensorPlan]


class PlanResponse(BaseModel):
    # "optimal"   - every lexicographic objective was proven optimal
    # "feasible"  - a valid plan, but optimality was not proven (time limit)
    # "infeasible"- no plan satisfies the deadline / memory limit
    # "unknown"   - the solver could not find any plan within the time limit
    status: str
    address_bound: Optional[int] = None
    paths: Optional[list[PathPlan]] = None
