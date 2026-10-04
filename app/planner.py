"""CP-SAT based buffer planner.

Model
-----
* Every operator gets a start time; dependencies and producer->consumer
  data flow impose precedence constraints.  Operators without a precedence
  relation may run in parallel (no resource constraints).
* A path spans [min operator start, max operator end]; it must finish before
  the deadline.  Paths are mutually exclusive, so their schedules are
  independent and their address spaces may be reused.
* A tensor is alive from the end of its producer (or the path start when it
  is an external input) until the end of its last consumer; final outputs
  stay alive until the path end.  A view borrows the bytes of its base
  tensor at ``base_address + offset`` and extends the base tensor's
  occupancy to cover the view's own lifetime.
* Inside one path, simultaneously alive base tensors must not overlap in
  memory: this is a 2D no-overlap constraint (time x address).
* The address-space upper bound is the maximum, over all paths, of the
  per-path peak address.  It is minimised first; then tensor addresses
  (tensor id order), path starts (path id order) and operator starts
  (operator id order) are minimised lexicographically.  A stage only
  contributes to an "optimal" status when CP-SAT proves its optimum.
"""
from __future__ import annotations

import os
import time

from ortools.sat.python import cp_model

from .models import (
    OperatorPlan,
    PathPlan,
    PlanRequest,
    PlanResponse,
    TensorDef,
    TensorPlan,
)

# Per-solve and total wall-clock budgets (seconds), tunable via environment.
SOLVE_SECONDS = float(os.environ.get("PLANNER_SOLVE_SECONDS", "5"))
TOTAL_SECONDS = float(os.environ.get("PLANNER_TOTAL_SECONDS", "60"))


def plan(request: PlanRequest) -> PlanResponse:
    return _Planner(request).run()


class _Planner:
    def __init__(self, request: PlanRequest) -> None:
        self.req = request
        self.model = cp_model.CpModel()
        self.deadline = request.deadline

        # Global tensor definitions (consistency already validated).
        self.defs: dict[str, TensorDef] = {}
        for p in request.paths:
            for t in p.tensors:
                self.defs.setdefault(t.id, t)

        # Root base tensor and accumulated view offset for every tensor id.
        self.root: dict[str, str] = {}
        self.root_offset: dict[str, int] = {}
        for tid in self.defs:
            offset = 0
            cur = tid
            while self.defs[cur].base is not None:
                offset += self.defs[cur].offset
                cur = self.defs[cur].base
            self.root[tid] = cur
            self.root_offset[tid] = offset

        # Per-path lookup tables.
        self.path_tensors = [{t.id: t for t in p.tensors} for p in request.paths]
        self.path_ops = [{o.id: o for o in p.operators} for p in request.paths]

    # ------------------------------------------------------------------ build

    def _build(self) -> None:
        m = self.model
        req = self.req
        deadline = self.deadline
        mem = req.memory_limit

        # One address variable per root (non-view) tensor; views derive their
        # address as root_address + accumulated offset.
        self.addr: dict[str, cp_model.IntVar] = {}
        for tid, t in self.defs.items():
            if t.base is None:
                v = m.new_int_var(0, mem, f"addr[{tid}]")
                k = m.new_int_var(0, mem // t.alignment, f"align[{tid}]")
                m.add(v == t.alignment * k)
                self.addr[tid] = v
        # A view's own alignment constrains the root tensor's address.
        for tid, t in self.defs.items():
            if t.base is not None:
                m.add_modulo_equality(
                    0, self.addr[self.root[tid]] + self.root_offset[tid], t.alignment
                )

        self.op_start: list[dict[str, cp_model.IntVar]] = []
        self.path_start: list[cp_model.IntVar] = []
        self.path_end: list[cp_model.IntVar] = []
        self.birth: list[dict[str, cp_model.IntVar]] = []
        self.death: list[dict[str, cp_model.IntVar]] = []
        self.bound = m.new_int_var(0, mem, "address_bound")

        for pi, path in enumerate(req.paths):
            ops = self.path_ops[pi]
            tensors = self.path_tensors[pi]

            starts: dict[str, cp_model.IntVar] = {}
            self.birth.append({})
            self.death.append({})
            for oid, o in ops.items():
                s = m.new_int_var(0, deadline, f"start[{path.id},{oid}]")
                m.add(s + o.duration <= deadline)
                starts[oid] = s
            for oid, o in ops.items():
                for d in set(o.dependencies):
                    m.add(starts[oid] >= starts[d] + ops[d].duration)

            ps = m.new_int_var(0, deadline, f"path_start[{path.id}]")
            pe = m.new_int_var(0, deadline, f"path_end[{path.id}]")
            if ops:
                m.add_min_equality(ps, list(starts.values()))
                m.add_max_equality(
                    pe, [starts[oid] + o.duration for oid, o in ops.items()]
                )
            else:
                m.add(ps == 0)
                m.add(pe == 0)

            # Producer -> consumer data flow: a consumer starts only after
            # the producer of every tensor it reads has finished.
            for t in tensors.values():
                if t.producer is not None:
                    for c in set(t.consumers):
                        m.add(starts[c] >= starts[t.producer] + ops[t.producer].duration)

            def birth(t: TensorDef):
                if t.producer is not None:
                    return starts[t.producer] + ops[t.producer].duration
                return ps  # external input: alive from the path start

            def deaths(t: TensorDef):
                out = [starts[c] + ops[c].duration for c in t.consumers]
                if t.is_output:
                    out.append(pe)  # final outputs live until the path end
                return out

            children: dict[str, list[str]] = {}
            for tid, t in tensors.items():
                if t.base is not None:
                    children.setdefault(t.base, []).append(tid)

            def transitive_views(rid: str) -> list[str]:
                found: list[str] = []
                stack = list(children.get(rid, ()))
                while stack:
                    u = stack.pop()
                    found.append(u)
                    stack.extend(children.get(u, ()))
                return found

            x_intervals, y_intervals, demands = [], [], []
            for tid, t in tensors.items():
                if t.base is not None:
                    continue  # views borrow their root's bytes
                views = transitive_views(tid)
                birth_exprs = [birth(t)] + [birth(tensors[v]) for v in views]
                death_exprs = list(deaths(t))
                for v in views:
                    death_exprs.extend(deaths(tensors[v]))

                bv = m.new_int_var(0, deadline, f"birth[{path.id},{tid}]")
                dv = m.new_int_var(0, deadline, f"death[{path.id},{tid}]")
                m.add_min_equality(bv, birth_exprs)
                m.add_max_equality(dv, death_exprs + [bv])
                self.birth[pi][tid] = bv
                self.death[pi][tid] = dv

                span = m.new_int_var(0, deadline, f"span[{path.id},{tid}]")
                m.add(span == dv - bv)
                av = self.addr[tid]
                # A tensor claims its byte range only while its lifetime
                # window is non-empty (half-open semantics: [birth, death)).
                non_empty = m.new_bool_var(f"live[{path.id},{tid}]")
                m.add(dv >= bv + 1).only_enforce_if(non_empty)
                m.add(dv == bv).only_enforce_if(non_empty.Not())
                m.add(av + t.bytes <= self.bound).only_enforce_if(non_empty)
                # Optional intervals: a tensor with an empty window is absent
                # from the 2D packing (and zero-size intervals must never be
                # passed to no_overlap_2d - they crash the solver).
                x_intervals.append(
                    m.new_optional_interval_var(
                        bv, span, dv, non_empty, f"life[{path.id},{tid}]")
                )
                y_intervals.append(
                    m.new_optional_interval_var(
                        av, t.bytes, av + t.bytes, non_empty,
                        f"bytes[{path.id},{tid}]")
                )
                demands.append(t.bytes)

            # Simultaneously alive base tensors of one path must not overlap.
            if x_intervals:
                m.add_no_overlap_2d(x_intervals, y_intervals)
                # Redundant but strong relaxation: at any instant the alive
                # tensors' disjoint byte ranges fit inside [0, bound).
                m.add_cumulative(x_intervals, demands, self.bound)

            self.op_start.append(starts)
            self.path_start.append(ps)
            self.path_end.append(pe)

        # bound is already constrained by every per-tensor contribution and
        # bounded above by the memory limit via its domain.

    # ------------------------------------------------------------- objectives

    def _stages(self) -> list:
        """Lexicographic objectives, in order."""
        stages = [self.bound]
        for tid in sorted(self.defs):
            stages.append(self.addr[self.root[tid]] + self.root_offset[tid])
        order = sorted(range(len(self.req.paths)),
                       key=lambda i: self.req.paths[i].id)
        for pi in order:
            stages.append(self.path_start[pi])
        for pi in order:
            for oid in sorted(self.path_ops[pi]):
                stages.append(self.op_start[pi][oid])
        return stages

    # ------------------------------------------------------------------ solve

    def _greedy_hints(self) -> None:
        """Feed the solver a greedy ASAP schedule + first-fit packing as a
        starting hint.  Hints need not be feasible; they only guide search."""
        op_hint: dict[tuple[int, str], int] = {}
        addr_hint: dict[str, int] = {}
        bound_hint = 0
        for pi, path in enumerate(self.req.paths):
            ops = self.path_ops[pi]
            tensors = self.path_tensors[pi]
            # ASAP schedule respecting dependencies and producer->consumer.
            # Kahn-style longest path; if the data-flow graph has a cycle the
            # instance is infeasible anyway, so cyclic nodes just get 0.
            preds: dict[str, set[str]] = {oid: set() for oid in ops}
            for oid, o in ops.items():
                preds[oid].update(d for d in o.dependencies if d in ops)
            for t in tensors.values():
                if t.producer in ops:
                    for c in t.consumers:
                        if c in ops:
                            preds[c].add(t.producer)
            succ: dict[str, list[str]] = {oid: [] for oid in ops}
            indeg = {oid: len(ps_) for oid, ps_ in preds.items()}
            for oid, ps_ in preds.items():
                for p in ps_:
                    succ[p].append(oid)
            queue = [oid for oid in ops if indeg[oid] == 0]
            starts: dict[str, int] = {oid: 0 for oid in queue}
            head = 0
            while head < len(queue):
                cur = queue[head]
                head += 1
                for nxt in succ[cur]:
                    indeg[nxt] -= 1
                    starts[nxt] = max(starts.get(nxt, 0),
                                      starts[cur] + ops[cur].duration)
                    if indeg[nxt] == 0:
                        queue.append(nxt)
            for oid in ops:
                starts.setdefault(oid, 0)  # cyclic leftovers: arbitrary hint
            ends = {oid: starts[oid] + o.duration for oid, o in ops.items()}
            p_start = min(starts.values(), default=0)
            p_end = max(ends.values(), default=0)
            for oid, s in starts.items():
                op_hint[(pi, oid)] = s

            # Lifetime windows of base tensors under this schedule.
            def birth(t: TensorDef) -> int:
                return ends[t.producer] if t.producer in ops else p_start

            def death(t: TensorDef) -> int:
                d = max([ends[c] for c in t.consumers if c in ops]
                        + ([p_end] if t.is_output else []), default=0)
                return d

            children: dict[str, list[str]] = {}
            for tid, t in tensors.items():
                if t.base is not None:
                    children.setdefault(t.base, []).append(tid)

            def views(rid: str) -> list[str]:
                out, stack = [], list(children.get(rid, ()))
                while stack:
                    u = stack.pop()
                    out.append(u)
                    stack.extend(children.get(u, ()))
                return out

            windows: dict[str, tuple[int, int]] = {}
            for tid, t in tensors.items():
                if t.base is not None:
                    continue
                related = [tid] + views(tid)
                b = min(birth(tensors[u]) for u in related)
                d = max([b] + [death(tensors[u]) for u in related])
                if d > b:
                    windows[tid] = (b, d)

            # First-fit packing in birth order, lowest aligned address.
            placed: dict[str, int] = {}
            peak = 0
            for tid in sorted(windows, key=lambda t: (windows[t][0], t)):
                if tid in addr_hint:
                    placed[tid] = addr_hint[tid]  # shared tensor: keep hint
                    continue
                b0, b1 = windows[tid]
                align = self.defs[tid].alignment
                size = self.defs[tid].bytes
                address = 0
                while True:
                    conflict = any(
                        max(b0, windows[o][0]) < min(b1, windows[o][1])
                        and address < placed[o] + self.defs[o].bytes
                        and placed[o] < address + size
                        for o in placed
                    )
                    if not conflict:
                        break
                    address += align
                placed[tid] = address
                addr_hint.setdefault(tid, address)
                peak = max(peak, address + size)
            bound_hint = max(bound_hint, peak)

        for (pi, oid), val in op_hint.items():
            self.model.add_hint(self.op_start[pi][oid], val)
        for tid, val in addr_hint.items():
            self.model.add_hint(self.addr[tid], val)
        self.model.add_hint(self.bound, bound_hint)

    def run(self) -> PlanResponse:
        self._build()
        self._greedy_hints()
        solver = cp_model.CpSolver()
        wall_deadline = time.monotonic() + TOTAL_SECONDS

        snapshot = None
        all_optimal = True
        for expr in self._stages():
            remaining = wall_deadline - time.monotonic()
            if remaining <= 0:
                all_optimal = False
                break
            self.model.clear_objective()
            self.model.minimize(expr)
            solver.parameters.max_time_in_seconds = min(SOLVE_SECONDS, remaining)
            status = solver.solve(self.model)
            if status == cp_model.INFEASIBLE:
                return PlanResponse(status="infeasible")
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                all_optimal = False
                break
            snapshot = self._snapshot(solver)
            if status != cp_model.OPTIMAL:
                all_optimal = False
                break
            # Fix this objective at its proven optimum before the next stage.
            self.model.add(expr == int(solver.objective_value))

        if snapshot is None:
            return PlanResponse(status="unknown")
        return self._response(snapshot, "optimal" if all_optimal else "feasible")

    def _snapshot(self, solver: cp_model.CpSolver) -> dict:
        return {
            "bound": solver.value(self.bound),
            "addr": {tid: solver.value(v) for tid, v in self.addr.items()},
            "path_start": [solver.value(v) for v in self.path_start],
            "path_end": [solver.value(v) for v in self.path_end],
            "birth": [
                {tid: solver.value(v) for tid, v in per_path.items()}
                for per_path in self.birth
            ],
            "death": [
                {tid: solver.value(v) for tid, v in per_path.items()}
                for per_path in self.death
            ],
            "op_start": [
                {oid: solver.value(s) for oid, s in starts.items()}
                for starts in self.op_start
            ],
        }

    def _response(self, snap: dict, status: str) -> PlanResponse:
        paths = []
        for pi, path in enumerate(self.req.paths):
            operators = [
                OperatorPlan(
                    id=oid,
                    start=snap["op_start"][pi][oid],
                    end=snap["op_start"][pi][oid] + self.path_ops[pi][oid].duration,
                )
                for oid in sorted(self.path_ops[pi])
            ]
            tensors = []
            peak = 0
            for tid in sorted(self.path_tensors[pi]):
                address = snap["addr"][self.root[tid]] + self.root_offset[tid]
                tensors.append(TensorPlan(id=tid, address=address))
                # Peak: highest byte claimed by a base tensor whose lifetime
                # window is non-empty in this path.
                if self.defs[tid].base is None:
                    born = snap["birth"][pi][tid]
                    dead = snap["death"][pi][tid]
                    if dead > born:
                        peak = max(peak, address + self.defs[tid].bytes)
            paths.append(
                PathPlan(
                    id=path.id,
                    start=snap["path_start"][pi],
                    end=snap["path_end"][pi],
                    peak=peak,
                    operators=operators,
                    tensors=tensors,
                )
            )
        return PlanResponse(
            status=status, address_bound=snap["bound"], paths=paths
        )
