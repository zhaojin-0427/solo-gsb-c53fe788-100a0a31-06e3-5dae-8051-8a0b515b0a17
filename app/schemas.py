"""Request/response schemas for the buffer planning API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _as_int(name: str, v: object) -> int:
    # Reject booleans: in Python ``True`` is also an ``int``.
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{name} must be an integer")
    return v


class OperatorIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    duration: int
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("operator id must be a non-empty string")
        return v

    @field_validator("duration")
    @classmethod
    def _duration_positive(cls, v: int) -> int:
        v = _as_int("duration", v)
        if v <= 0:
            raise ValueError("duration must be a positive integer")
        return v

    @field_validator("depends_on", mode="before")
    @classmethod
    def _deps_list(cls, v: object) -> object:
        if not isinstance(v, list):
            raise ValueError("depends_on must be a list of operator ids")
        return v

    @field_validator("depends_on")
    @classmethod
    def _deps_dedup(cls, v: list[str]) -> list[str]:
        if any(not isinstance(d, str) or not d.strip() for d in v):
            raise ValueError("depends_on entries must be non-empty strings")
        if len(set(v)) != len(v):
            raise ValueError("depends_on contains duplicate operator ids")
        return v


class TensorIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    bytes: int
    alignment: int
    producer: str
    consumers: list[str] = Field(default_factory=list)
    output: bool = False
    base_tensor: Optional[str] = None
    view_offset: int = 0

    @field_validator("id", "producer")
    @classmethod
    def _nonempty(cls, v: str, info):
        if not v.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return v

    @field_validator("bytes")
    @classmethod
    def _bytes_positive(cls, v: int) -> int:
        v = _as_int("bytes", v)
        if v <= 0:
            raise ValueError("bytes must be a positive integer")
        return v

    @field_validator("alignment")
    @classmethod
    def _alignment_pow2(cls, v: int) -> int:
        v = _as_int("alignment", v)
        if v <= 0 or (v & (v - 1)) != 0:
            raise ValueError("alignment must be a positive power of two")
        return v

    @field_validator("view_offset")
    @classmethod
    def _offset_nonneg(cls, v: int) -> int:
        v = _as_int("view_offset", v)
        if v < 0:
            raise ValueError("view_offset must be a non-negative integer")
        return v

    @field_validator("base_tensor")
    @classmethod
    def _base_nonempty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("base_tensor must reference a tensor id")
        return v

    @field_validator("consumers", mode="before")
    @classmethod
    def _consumers_list(cls, v: object) -> object:
        if not isinstance(v, list):
            raise ValueError("consumers must be a list of operator ids")
        return v

    @field_validator("consumers")
    @classmethod
    def _consumers_dedup(cls, v: list[str]) -> list[str]:
        if any(not isinstance(c, str) or not c.strip() for c in v):
            raise ValueError("consumers entries must be non-empty strings")
        if len(set(v)) != len(v):
            raise ValueError("consumers contains duplicate operator ids")
        return v

    @field_validator("output", mode="before")
    @classmethod
    def _output_bool(cls, v: object) -> bool:
        if not isinstance(v, bool):
            raise ValueError("output must be a boolean")
        return v


class PathIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    operators: list[OperatorIn]
    tensors: list[TensorIn]

    @field_validator("id")
    @classmethod
    def _id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("path id must be a non-empty string")
        return v

    @field_validator("operators", "tensors")
    @classmethod
    def _nonempty_list(cls, v: list, info):
        if not v:
            raise ValueError(f"{info.field_name} must not be empty")
        return v


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deadline: int
    memory_limit: int
    paths: list[PathIn]

    @field_validator("deadline", "memory_limit")
    @classmethod
    def _positive(cls, v: int, info):
        v = _as_int(info.field_name, v)
        if v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v

    @field_validator("paths")
    @classmethod
    def _paths_nonempty(cls, v: list) -> list:
        if not v:
            raise ValueError("paths must not be empty")
        return v


class OpSchedule(BaseModel):
    operator_id: str
    start: int
    end: int


class PathSchedule(BaseModel):
    path_id: str
    start: int
    end: int
    peak_bytes: int
    operators: list[OpSchedule]


class TensorAddress(BaseModel):
    tensor_id: str
    address: int
    bytes: int


class PlanResponse(BaseModel):
    feasible: bool
    optimal: bool
    reason: Optional[str] = None
    capacity_upper_bound: int
    timeline: list[PathSchedule]
    addresses: list[TensorAddress]
