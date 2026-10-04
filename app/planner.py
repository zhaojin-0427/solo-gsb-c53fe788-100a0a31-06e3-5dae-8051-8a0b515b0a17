"""Buffer / schedule planning with OR-Tools CP-SAT.

The planner jointly solves:

* a non-preemptive schedule of each path's operator DAG (ready operators may
  run in parallel), with mutually-exclusive paths serialized on one global
  timeline and every path finishing before the deadline;
* placement of every root tensor in one shared byte-addressable buffer under
  power-of-two alignment, where tensors that are never simultaneously live in
  a path may share bytes and mutually-exclusive paths may reuse them.

Optimization is strictly lexicographic:

1. minimize the address-space upper bound (capacity);
2. minimize tensor addresses in tensor-id order;
3. minimize path starts in path-id order;
4. minimize operator starts in operator-id order (operators tied across paths
   are ordered by path id).

Only when CP-SAT proves every lexicographic stage optimal is the result marked
``optimal``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ortools.sat.python import cp_model

from .schemas import (
    OpSchedule,
    PathSchedule,
    PlanRequest,
    PlanResponse,
    TensorAddress,
)


class PlannerError(ValueError):
    """Semantic validation error; mapped to HTTP 422."""


@dataclass
class Occurrence:
    producer: str
    consumers: list[str]
    output: bool


@dataclass
class TInfo:
    id: str
    size: int
    alignment: int
    base: Optional[str]
    offset: int
    root: str = ""
    cum_offset: int = 0
    occ: dict[str, Occurrence] = field(default_factory=dict)


@dataclass
class Prepared:
    path_ids: list[str]
    path_ops: dict[str, dict]       # path -> {op_id: (duration, deps)}
    path_op_order: dict[str, list[str]]
    tensors: dict[str, TInfo]
    roots: list[str]
    root_size: dict[str, int]
    root_align: dict[str, int]
    live_roots: set[str]            # roots with a non-empty lifetime somewhere
    H: int
    S: int


@dataclass
class ModelArtifacts:
    model: cp_model.CpModel
    s: dict          # (pid, oid) -> start var (global timeline)
    f: dict          # (pid, oid) -> finish var
    sp: dict         # pid -> path start var
    ep: dict         # pid -> path end var
    addr: dict       # root id -> address var
    cap: object
    lstart: dict     # (pid, root) -> liveness start expr
    lend: dict       # (pid, root) -> liveness end expr
    presence: dict   # (pid, root) -> bool var, True iff non-empty lifetime


# --------------------------------------------------------------------------- #
# Validation / preparation
# --------------------------------------------------------------------------- #
def prepare(req: PlanRequest) -> Prepared:
    path_ids: list[str] = []
    path_ops: dict[str, dict] = {}
    path_op_order: dict[str, list[str]] = {}
    canon: dict[str, tuple] = {}
    tensors: dict[str, TInfo] = {}

    if len({p.id for p in req.paths}) != len(req.paths):
        raise PlannerError("duplicate path ids are not allowed")

    for path in req.paths:
        pid = path.id
        path_ids.append(pid)

        if len({o.id for o in path.operators}) != len(path.operators):
            raise PlannerError(f"path {pid}: duplicate operator ids")
        ops = {o.id: (o.duration, list(o.depends_on)) for o in path.operators}
        path_ops[pid] = ops
        path_op_order[pid] = [o.id for o in path.operators]

        local = {t.id: t for t in path.tensors}
        if len(local) != len(path.tensors):
            raise PlannerError(f"path {pid}: duplicate tensor ids")

        for o in path.operators:
            for d in o.depends_on:
                if d == o.id:
                    raise PlannerError(
                        f"path {pid}: operator {o.id!r} cannot depend on itself"
                    )
                if d not in ops:
                    raise PlannerError(
                        f"path {pid}: operator {o.id!r} depends on unknown operator {d!r}"
                    )

        for t in path.tensors:
            if t.producer not in ops:
                raise PlannerError(
                    f"path {pid}: tensor {t.id!r} producer {t.producer!r} is not an operator"
                )
            for c in t.consumers:
                if c not in ops:
                    raise PlannerError(
                        f"path {pid}: tensor {t.id!r} consumer {c!r} is not an operator"
                    )
            if t.base_tensor is not None:
                if t.base_tensor == t.id:
                    raise PlannerError(f"path {pid}: tensor {t.id!r} cannot view itself")
                if t.base_tensor not in local:
                    raise PlannerError(
                        f"path {pid}: tensor {t.id!r} base_tensor {t.base_tensor!r} not found"
                    )

            key = (t.bytes, t.alignment, t.base_tensor, t.view_offset)
            if t.id in canon and canon[t.id] != key:
                raise PlannerError(
                    f"tensor {t.id!r} has inconsistent definitions across paths"
                )
            canon[t.id] = key

            info = tensors.get(t.id)
            if info is None:
                info = TInfo(
                    id=t.id, size=t.bytes, alignment=t.alignment,
                    base=t.base_tensor, offset=t.view_offset,
                )
                tensors[t.id] = info
            info.occ[pid] = Occurrence(t.producer, list(t.consumers), t.output)

    # resolve view chains
    for tid, info in tensors.items():
        chain_ids: list[str] = []
        cur = info
        cum = 0
        while cur.base is not None:
            if cur.id in chain_ids:
                raise PlannerError(f"tensor {tid!r}: cyclic view chain via {cur.id!r}")
            chain_ids.append(cur.id)
            nxt = tensors.get(cur.base)
            if nxt is None:  # pragma: no cover - guarded per path above
                raise PlannerError(f"tensor {tid!r}: unknown base tensor {cur.base!r}")
            cum += cur.offset
            cur = nxt
        info.root = cur.id
        info.cum_offset = cum
        if cum % info.alignment != 0:
            raise PlannerError(
                f"tensor {tid!r}: cumulative view offset {cum} is not aligned to "
                f"its alignment {info.alignment}"
            )
        if info.root != info.id and cum + info.size > cur.size:
            raise PlannerError(
                f"tensor {tid!r}: view [offset={cum}, bytes={info.size}) exceeds "
                f"root tensor {cur.id!r} of {cur.size} bytes"
            )

    # effective root alignment: strictest alignment in every view chain
    root_align: dict[str, int] = {}
    for info in tensors.values():
        chain_max = info.alignment
        cur = info
        while cur.base is not None:
            cur = tensors[cur.base]
            chain_max = max(chain_max, cur.alignment)
        root_align[info.root] = max(root_align.get(info.root, 1), chain_max)

    # DAG check over explicit deps + producer/consumer + view-producer edges
    for pid in path_ids:
        ops = path_ops[pid]
        local_ids = {tid for tid, info in tensors.items() if pid in info.occ}
        succ: dict[str, list[str]] = {o: [] for o in ops}
        for oid, (_, deps) in ops.items():
            for d in deps:
                succ[d].append(oid)
        for tid in local_ids:
            info = tensors[tid]
            oc = info.occ[pid]
            for c in oc.consumers:
                if c != oc.producer:  # same-operator access imposes no order
                    succ[oc.producer].append(c)
            if info.base is not None and info.base in local_ids:
                bp = tensors[info.base].occ[pid].producer
                if bp != oc.producer:  # view produced within the base producer
                    succ[bp].append(oc.producer)
        _assert_acyclic(succ, pid)

    roots = sorted({info.root for info in tensors.values()})
    root_size = {r: tensors[r].size for r in roots}
    live_roots = {
        info.root
        for info in tensors.values()
        for oc in info.occ.values()
        if oc.output or oc.consumers
    }
    H = sum(d for ops in path_ops.values() for d, _ in ops.values())
    # Sufficient-memory bound for the diagnostic probes: placing live roots
    # sequentially at aligned addresses uses at most sum(size + align - 1)
    # bytes (the slack accounts for every alignment gap).
    S = sum(
        root_size[r] + root_align.get(r, 1) - 1
        for r in roots
        if r in live_roots
    )
    return Prepared(
        path_ids=path_ids,
        path_ops=path_ops,
        path_op_order=path_op_order,
        tensors=tensors,
        roots=roots,
        root_size=root_size,
        root_align=root_align,
        live_roots=live_roots,
        H=max(H, 1),
        S=max(S, 1),
    )


def _assert_acyclic(succ: dict[str, list[str]], pid: str) -> None:
    color = {o: 0 for o in succ}

    def dfs(u: str) -> None:
        color[u] = 1
        for v in succ[u]:
            if color[v] == 1:
                raise PlannerError(
                    f"path {pid}: operator dependency cycle involving {v!r}"
                )
            if color[v] == 0:
                dfs(v)
        color[u] = 2

    for o in succ:
        if color[o] == 0:
            dfs(o)


# --------------------------------------------------------------------------- #
# CP-SAT model
# --------------------------------------------------------------------------- #
def build_model(prep: Prepared, deadline: int, memory_limit: int) -> ModelArtifacts:
    m = cp_model.CpModel()
    H = prep.H
    s: dict = {}
    f: dict = {}

    for pid in prep.path_ids:
        for oid, (dur, deps) in prep.path_ops[pid].items():
            sv = m.NewIntVar(0, H, f"s_{pid}_{oid}")
            fv = m.NewIntVar(0, H, f"f_{pid}_{oid}")
            m.Add(fv == sv + dur)
            s[(pid, oid)] = sv
            f[(pid, oid)] = fv
            for d in deps:
                m.Add(sv >= f[(pid, d)])

    # data precedences: producer -> consumers, base producer -> view producer.
    # Same-operator reads/views impose no ordering (tensor available within the
    # producing operator) and would be degenerate self constraints.
    for info in prep.tensors.values():
        for pid, oc in info.occ.items():
            for c in oc.consumers:
                if c != oc.producer:
                    m.Add(s[(pid, c)] >= f[(pid, oc.producer)])
            if info.base is not None and pid in prep.tensors[info.base].occ:
                bp = prep.tensors[info.base].occ[pid].producer
                if bp != oc.producer:
                    m.Add(s[(pid, oc.producer)] >= f[(pid, bp)])

    # path windows; mutually-exclusive paths are serialized on the timeline
    sp, ep, path_ivs = {}, {}, []
    for pid in prep.path_ids:
        op_ids = list(prep.path_ops[pid])
        sv = m.NewIntVar(0, H, f"S_{pid}")
        ev = m.NewIntVar(0, H, f"E_{pid}")
        dur = m.NewIntVar(0, H, f"D_{pid}")
        m.AddMaxEquality(ev, [f[(pid, o)] for o in op_ids])
        m.AddMinEquality(sv, [s[(pid, o)] for o in op_ids])
        m.Add(dur == ev - sv)
        iv = m.NewIntervalVar(sv, dur, ev, f"iv_{pid}")
        path_ivs.append(iv)
        m.Add(ev <= deadline)
        sp[pid], ep[pid] = sv, ev
    m.AddNoOverlap(path_ivs)

    # root tensor addresses in the shared buffer
    addr: dict = {}
    cap = m.NewIntVar(0, memory_limit, "capacity")
    zero = m.NewConstant(0)
    y_iv: dict = {}
    for r in prep.roots:
        if r not in prep.live_roots:
            # empty lifetime everywhere: reserves nothing, pinned at address 0
            av = m.NewConstant(0)
            y_iv[r] = None
        else:
            av = m.NewIntVar(0, memory_limit, f"a_{r}")
            m.Add(av + prep.root_size[r] <= cap)
            m.AddModuloEquality(zero, av, prep.root_align[r])
            y_iv[r] = m.NewFixedSizeIntervalVar(
                av, prep.root_size[r], f"yiv_{r}"
            )
        addr[r] = av
    m.Add(cap <= memory_limit)

    # liveness intervals of roots within each path; each non-empty lifetime
    # becomes a (time x bytes) rectangle. AddNoOverlap2D enforces, in one
    # constraint, that simultaneously-live roots (in any paths) are disjoint
    # in bytes and that every rectangle stays inside the global capacity.
    lstart: dict = {}
    lend: dict = {}
    presence: dict = {}
    x_rects: list = []
    y_rects: list = []

    for pid in prep.path_ids:
        members: dict[str, list[TInfo]] = {}
        for info in prep.tensors.values():
            if pid in info.occ:
                members.setdefault(info.root, []).append(info)
        for r, infos in members.items():
            starts, ends = [], []
            for info in infos:
                oc = info.occ[pid]
                pf = f[(pid, oc.producer)]
                starts.append(pf)
                if oc.output:
                    ends.append(ep[pid])
                for c in oc.consumers:
                    ends.append(f[(pid, c)])
            if not ends:
                # half-open empty interval [t, t): no rectangle, never occupies
                base = starts[0]
                presence[(pid, r)] = m.NewConstant(0)
                lstart[(pid, r)] = base
                lend[(pid, r)] = base
                continue
            ls = m.NewIntVar(0, H, f"ls_{pid}_{r}")
            le = m.NewIntVar(0, H, f"le_{pid}_{r}")
            # occupied as soon as any view of the root is produced ...
            if len(starts) == 1:
                m.Add(ls == starts[0])
            else:
                m.AddMinEquality(ls, starts)
            # ... until the last consumer (of any view) or path end for outputs
            m.AddMaxEquality(le, ends)
            wvar = m.NewIntVar(0, H, f"lw_{pid}_{r}")
            m.Add(wvar == le - ls)
            iv = m.NewIntervalVar(ls, wvar, le, f"xiv_{pid}_{r}")
            presence[(pid, r)] = m.NewConstant(1)
            lstart[(pid, r)] = ls
            lend[(pid, r)] = le
            if y_iv[r] is not None:
                x_rects.append(iv)
                y_rects.append(y_iv[r])

    m.AddNoOverlap2D(x_rects, y_rects)

    return ModelArtifacts(m, s, f, sp, ep, addr, cap, lstart, lend, presence)


# --------------------------------------------------------------------------- #
# Solving (lexicographic)
# --------------------------------------------------------------------------- #
_TIME_STAGE1 = 10.0
_TIME_STAGE = 0.3
_TIME_OP_STAGE = 0.08
_TIME_PROBE = 5.0


def _new_solver(time_limit: float) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 42
    return solver


def plan(req: PlanRequest) -> PlanResponse:
    prep = prepare(req)
    art = build_model(prep, req.deadline, req.memory_limit)
    m = art.model

    solver = _new_solver(_TIME_STAGE1)
    m.Minimize(art.cap)
    st = solver.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _infeasible_response(prep, req)

    all_optimal = st == cp_model.OPTIMAL

    def fix(var) -> None:
        m.Add(var == solver.Value(var))

    def hint_current() -> None:
        # Warm-start the next solve with the incumbent solution so that the
        # lexicographic stages only need to prove a tiny improvement.
        m.ClearHints()

        def hint(var) -> None:
            # only genuine, as-yet-unfixed decision variables can be hinted
            if isinstance(var, cp_model.IntVar):
                proto = var.Proto()
                if proto.domain and proto.domain[0] < proto.domain[1]:
                    m.AddHint(var, solver.Value(var))

        for var in art.addr.values():
            hint(var)
        hint(art.cap)
        for pid in prep.path_ids:
            hint(art.sp[pid])
            hint(art.ep[pid])
            for oid in prep.path_ops[pid]:
                hint(art.s[(pid, oid)])
                hint(art.f[(pid, oid)])
        for (pid, r), pvar in art.presence.items():
            if isinstance(pvar, cp_model.IntVar) and solver.Value(pvar) == 1:
                hint(art.lstart[(pid, r)])
                hint(art.lend[(pid, r)])

    def stage(var, time_limit: float) -> None:
        nonlocal all_optimal
        hint_current()
        solver.parameters.max_time_in_seconds = time_limit
        m.Minimize(var)
        status = solver.Solve(m)
        if status == cp_model.OPTIMAL:
            fix(var)
            return True
        if status == cp_model.FEASIBLE:
            fix(var)
        # not proved optimal: lexicographic optimality cannot be recovered,
        # but the incumbent is still a valid feasible solution
        all_optimal = False
        return False

    # 1. capacity proved/held -> fix it
    fix(art.cap)

    def run_all() -> bool:
        # 2. tensor addresses in tensor-id order. One variable decides the
        # address of a whole root (views move with it), so roots are ordered
        # by the smallest tensor id among the root and its views.
        root_priority: dict[str, str] = {}
        for tid, info in prep.tensors.items():
            r = info.root
            if r in prep.live_roots:
                root_priority[r] = min(root_priority.get(r, tid), tid)
        for r in sorted(root_priority, key=lambda x: root_priority[x]):
            if not stage(art.addr[r], _TIME_STAGE):
                return False

        # 3. path starts in path-id order
        for pid in prep.path_ids:
            if not stage(art.sp[pid], _TIME_STAGE):
                return False

        # 4. operator starts by (operator id, path id)
        op_keys = sorted(art.s.keys(), key=lambda k: (k[1], k[0]))
        for key in op_keys:
            if not stage(art.s[key], _TIME_OP_STAGE):
                return False
        return True

    run_all()
    return _build_response(prep, art, solver, optimal=all_optimal)


def _build_response(
    prep: Prepared, art: ModelArtifacts, solver: cp_model.CpSolver, optimal: bool
) -> PlanResponse:
    root_addr = {r: solver.Value(art.addr[r]) for r in prep.roots}

    timeline: list[PathSchedule] = []
    for pid in prep.path_ids:
        ops = [
            OpSchedule(
                operator_id=oid,
                start=solver.Value(art.s[(pid, oid)]),
                end=solver.Value(art.f[(pid, oid)]),
            )
            for oid in prep.path_op_order[pid]
        ]
        peak = _path_peak(prep, art, solver, pid)
        timeline.append(
            PathSchedule(
                path_id=pid,
                start=solver.Value(art.sp[pid]),
                end=solver.Value(art.ep[pid]),
                peak_bytes=peak,
                operators=ops,
            )
        )

    addresses = [
        TensorAddress(
            tensor_id=tid,
            address=root_addr[info.root] + info.cum_offset,
            bytes=info.size,
        )
        for tid, info in sorted(prep.tensors.items())
    ]
    return PlanResponse(
        feasible=True,
        optimal=optimal,
        capacity_upper_bound=solver.Value(art.cap),
        timeline=timeline,
        addresses=addresses,
    )


def _path_peak(
    prep: Prepared, art: ModelArtifacts, solver: cp_model.CpSolver, pid: str
) -> int:
    events: list[tuple[int, int, int]] = []  # (time, kind, delta); end before start
    for r in prep.roots:
        if (pid, r) not in art.presence or solver.Value(art.presence[(pid, r)]) == 0:
            continue
        ls = solver.Value(art.lstart[(pid, r)])
        le = solver.Value(art.lend[(pid, r)])
        if le <= ls:
            continue
        events.append((ls, 1, prep.root_size[r]))   # acquire
        events.append((le, 0, -prep.root_size[r]))  # release first
    events.sort()
    cur = peak = 0
    for _, _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def _infeasible_response(prep: Prepared, req: PlanRequest) -> PlanResponse:
    # Relax one resource at a time using its guaranteed-sufficient bound:
    # S = sum of all live root sizes always fits memory; H = total work
    # always fits the serialized timeline.
    deadline_ok = _feasible(prep, req.deadline, prep.S)
    memory_ok = _feasible(prep, prep.H, req.memory_limit)
    if not deadline_ok and not memory_ok:
        reason = "deadline_and_memory_limit"
    elif not deadline_ok:
        reason = "deadline"
    else:
        reason = "memory_limit"
    return PlanResponse(
        feasible=False,
        optimal=False,
        reason=reason,
        capacity_upper_bound=0,
        timeline=[],
        addresses=[],
    )


def _feasible(prep: Prepared, deadline: int, memory_limit: int) -> bool:
    art = build_model(prep, deadline, memory_limit)
    solver = _new_solver(_TIME_PROBE)
    status = solver.Solve(art.model)
    return status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
