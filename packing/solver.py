"""
Solver layer: PackingProblem in, SolveResult out.

This is the only file that imports gurobipy. It returns plain data
(dataclasses), never text meant for users; the UI decides how to say things.
To switch solvers later, only this file changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from .problem import Issue, PackingProblem, validate


# ---------------------------------------------------------------- results
@dataclass
class SackLoad:
    sack: str
    items: list[str]
    weight_used: float
    weight_cap: float
    volume_used: float
    volume_cap: float


@dataclass
class Solution:
    total_value: int
    max_value: int            # value if every item were packed
    loads: list[SackLoad]
    left_behind: list[str]


@dataclass
class Conflict:
    """One requirement that is part of a contradiction (from the IIS)."""
    kind: str      # category_min | category_max | sack_weight | sack_volume | sack_rules
    subject: str   # category or sack name
    limit: float   # the current value of that requirement


@dataclass
class Fix:
    """A concrete change that would make the problem feasible (from feasRelax)."""
    kind: str      # category_min | category_max | sack_weight | sack_volume
    subject: str
    current: float
    suggested: float


@dataclass
class SolveResult:
    status: str    # optimal | time_limit | infeasible | invalid | error
    issues: list[Issue] = field(default_factory=list)
    solution: Solution | None = None
    conflicts: list[Conflict] = field(default_factory=list)
    fixes: list[Fix] = field(default_factory=list)
    detail: str = ""


# How reluctant we are to suggest changing each kind of requirement, per unit
# of change. Category limits are preferences (cheap); bag capacities are
# physical (expensive). Packing rules and "pack once" are never relaxed.
RELAX_PENALTY = {
    "category_min": 1.0,
    "category_max": 1.0,
    "sack_weight": 10.0,
    "sack_volume": 10.0,
}


# ---------------------------------------------------------------- model
def _build(p: PackingProblem, env: gp.Env):
    """Build the model. Returns (model, x, tags) where tags maps each
    constraint name to (kind, subject, limit) so results can be explained."""
    n, m = len(p.items), len(p.sacks)
    I, J = range(n), range(m)
    tags: dict[str, tuple[str, str, float]] = {}

    model = gp.Model("college_packing", env=env)
    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")

    model.setObjective(
        gp.quicksum(p.items[i].value * x[i, j] for i in I for j in J), GRB.MAXIMIZE
    )

    # (1) each item at most once
    model.addConstrs((x.sum(i, "*") <= 1 for i in I), name="pack_once")

    # (2), (3) capacities
    for j, s in enumerate(p.sacks):
        model.addConstr(
            gp.quicksum(p.items[i].weight * x[i, j] for i in I) <= s.weight_cap,
            name=f"weight_cap[{j}]",
        )
        tags[f"weight_cap[{j}]"] = ("sack_weight", s.name, s.weight_cap)

        model.addConstr(
            gp.quicksum(p.items[i].volume * x[i, j] for i in I) <= s.volume_cap,
            name=f"volume_cap[{j}]",
        )
        tags[f"volume_cap[{j}]"] = ("sack_volume", s.name, s.volume_cap)

    # (4), (5) category limits
    for g, c in enumerate(p.categories):
        members = p.members(c.name)
        if not members:
            continue                    # validation already caught min > 0 here
        count = gp.quicksum(x[i, j] for i in members for j in J)
        if c.min_items > 0:
            model.addConstr(count >= c.min_items, name=f"cat_min[{g}]")
            tags[f"cat_min[{g}]"] = ("category_min", c.name, c.min_items)
        if c.max_items is not None:
            model.addConstr(count <= c.max_items, name=f"cat_max[{g}]")
            tags[f"cat_max[{g}]"] = ("category_max", c.name, c.max_items)

    # (6) prohibited items
    for j, s in enumerate(p.sacks):
        banned = [i for i, it in enumerate(p.items) if s.name in it.prohibited_in]
        if banned:
            model.addConstr(gp.quicksum(x[i, j] for i in banned) == 0,
                            name=f"prohibited[{j}]")
            tags[f"prohibited[{j}]"] = ("sack_rules", s.name, 0)

    model.update()
    return model, x, tags


# ---------------------------------------------------------------- reading results
def _assignment(p: PackingProblem, value_of) -> dict[int, int]:
    """item index -> sack index, for items that were packed."""
    placed = {}
    for i in range(len(p.items)):
        for j in range(len(p.sacks)):
            if value_of(i, j) > 0.5:
                placed[i] = j
    return placed


def _solution(p: PackingProblem, placed: dict[int, int]) -> Solution:
    loads = []
    for j, s in enumerate(p.sacks):
        inside = [i for i, jj in placed.items() if jj == j]
        loads.append(SackLoad(
            sack=s.name,
            items=[p.items[i].name for i in inside],
            weight_used=sum(p.items[i].weight for i in inside),
            weight_cap=s.weight_cap,
            volume_used=sum(p.items[i].volume for i in inside),
            volume_cap=s.volume_cap,
        ))
    return Solution(
        total_value=sum(p.items[i].value for i in placed),
        max_value=sum(it.value for it in p.items),
        loads=loads,
        left_behind=[it.name for i, it in enumerate(p.items) if i not in placed],
    )


# ---------------------------------------------------------------- diagnosis
def _conflicts(model: gp.Model, tags) -> list[Conflict]:
    """Which requirements contradict each other (an IIS)."""
    model.computeIIS()
    return [Conflict(*tags[c.ConstrName])
            for c in model.getConstrs()
            if c.IISConstr and c.ConstrName in tags]


def _fixes(p: PackingProblem, model: gp.Model, tags) -> list[Fix]:
    """Smallest weighted change to the requirements that makes packing possible."""
    relax = model.copy()
    try:
        names = [name for name, t in tags.items() if t[0] in RELAX_PENALTY]
        if not names:
            return []
        constrs = [relax.getConstrByName(name) for name in names]
        penalties = [RELAX_PENALTY[tags[name][0]] for name in names]
        relax.feasRelax(0, False, None, None, None, constrs, penalties)
        relax.optimize()
        if relax.SolCount == 0:
            return []           # only packing rules could fix it; no suggestion

        vals = {v.VarName: v.X for v in relax.getVars()}
        placed = _assignment(p, lambda i, j: vals[f"x[{i},{j}]"])
        load = _solution(p, placed)

        fixes = []
        for g, c in enumerate(p.categories):
            count = sum(1 for i in p.members(c.name) if i in placed)
            if f"cat_min[{g}]" in tags and count < c.min_items:
                fixes.append(Fix("category_min", c.name, c.min_items, count))
            if f"cat_max[{g}]" in tags and count > c.max_items:
                fixes.append(Fix("category_max", c.name, c.max_items, count))
        for s, ld in zip(p.sacks, load.loads):
            if ld.weight_used > s.weight_cap + 1e-6:
                fixes.append(Fix("sack_weight", s.name, s.weight_cap,
                                 math.ceil(ld.weight_used * 10) / 10))
            if ld.volume_used > s.volume_cap + 1e-6:
                fixes.append(Fix("sack_volume", s.name, s.volume_cap,
                                 math.ceil(ld.volume_used * 10) / 10))
        return fixes
    finally:
        relax.dispose()


# ---------------------------------------------------------------- entry point
def solve(p: PackingProblem, time_limit: float = 30.0) -> SolveResult:
    issues = validate(p)
    if any(i.level == "error" for i in issues):
        return SolveResult("invalid", issues=issues)

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        model, x, tags = _build(p, env)
        try:
            model.Params.TimeLimit = time_limit
            model.optimize()
            if model.Status == GRB.INF_OR_UNBD:
                model.Params.DualReductions = 0
                model.optimize()

            if model.SolCount > 0:
                vals = model.getAttr("X", x)
                placed = _assignment(p, lambda i, j: vals[i, j])
                status = "optimal" if model.Status == GRB.OPTIMAL else "time_limit"
                return SolveResult(status, issues=issues, solution=_solution(p, placed))

            if model.Status == GRB.INFEASIBLE:
                return SolveResult(
                    "infeasible", issues=issues,
                    conflicts=_conflicts(model, tags),
                    fixes=_fixes(p, model, tags),
                )

            return SolveResult("error", issues=issues,
                               detail=f"Gurobi status code {model.Status}")
        finally:
            model.dispose()
