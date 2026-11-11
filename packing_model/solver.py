"""Solver layer for the college packing model.

Takes a PackingProblem and returns a SolveResult. This is the only
module that imports gurobipy, so switching solvers only changes this
file. Results are plain dataclasses, never text meant for users; the
UI decides how to word them.

The model is a multi-knapsack that maximizes total item value minus
penalties for missed soft category minimums. Constraint numbers in
the comments, (1)-(11), match the write-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from .problem import Issue, PackingProblem, validate


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class SackLoad:
    """What ended up in one sack, and how full it is."""

    sack: str
    items: list[str]
    weight_used: float
    weight_cap: float
    volume_used: float
    volume_cap: float


@dataclass
class SoftResult:
    """How one soft category minimum (g in S) turned out.

    Attributes:
        category: Category name.
        packed: Number of its items packed.
        minimum: The soft minimum l_g.
        met: True if ``packed >= minimum``.
        penalty: beta_g * shortfall, or 0 when the minimum is met.
    """

    category: str
    packed: int
    minimum: int
    met: bool
    penalty: float


@dataclass
class Solution:
    """A packing plan.

    Attributes:
        total_value: Sum of v_i over the packed items.
        max_value: Value if every item were packed.
        loads: One SackLoad per sack, in input order.
        left_behind: Names of the items not packed.
        soft: One SoftResult per soft category.
    """

    total_value: int
    max_value: int
    loads: list[SackLoad]
    left_behind: list[str]
    soft: list[SoftResult] = field(default_factory=list)

    @property
    def objective(self) -> float:
        """Item value plus soft-minimum penalties (the objective)."""
        return self.total_value + sum(r.penalty for r in self.soft)


@dataclass
class Conflict:
    """One requirement that is part of a contradiction (from the IIS).

    Attributes:
        kind: One of category_min, category_max, sack_weight,
            sack_volume, sack_rules or required.
        subject: Name of the category, sack or item concerned.
        limit: The current value of that requirement.
    """

    kind: str
    subject: str
    limit: float


@dataclass
class Fix:
    """A change that would make the problem feasible (from feasRelax).

    Attributes:
        kind: One of category_min, category_max, sack_weight or
            sack_volume.
        subject: Name of the category or sack to change.
        current: Its current limit.
        suggested: The limit that would make packing possible.
    """

    kind: str
    subject: str
    current: float
    suggested: float


@dataclass
class SolveResult:
    """Everything solve() found out.

    Attributes:
        status: optimal, time_limit, infeasible, invalid or error.
        issues: Input issues from validate(); errors mean "invalid".
        solution: The packing plan, if one was found.
        conflicts: Requirements that contradict each other, if
            infeasible.
        fixes: Smallest changes that would make it feasible, if
            infeasible.
        detail: Extra information when status is "error".
    """

    status: str
    issues: list[Issue] = field(default_factory=list)
    solution: Solution | None = None
    conflicts: list[Conflict] = field(default_factory=list)
    fixes: list[Fix] = field(default_factory=list)
    detail: str = ""


# Cost per unit of change when suggesting fixes. Category limits are
# preferences (cheap); bag capacities are physical (expensive). Packing
# rules, required items and "pack once" are never relaxed, and soft
# minimums never cause infeasibility, so they are not listed.
RELAX_PENALTY = {
    "category_min": 1.0,
    "category_max": 1.0,
    "sack_weight": 10.0,
    "sack_volume": 10.0,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def big_m(minimum: int, n_members: int) -> int:
    """Return the smallest big-M that makes (6) and (7) exact.

    For a soft category, l_g - count_g lies in [l_g - |C_g|, l_g].
    Constraint (7) with ind = 0 needs M >= l_g, and (6) with ind = 1
    needs M >= |C_g| - l_g + 1.

    Args:
        minimum: The soft minimum l_g.
        n_members: Number of items in the category, |C_g|.

    Returns:
        A positive integer M.
    """
    return max(minimum, n_members - minimum + 1, 1)


def _build(p: PackingProblem, env: gp.Env):
    """Build the Gurobi model for a validated problem.

    Args:
        p: The problem. It should already pass validate().
        env: The Gurobi environment to build the model in.

    Returns:
        A tuple ``(model, x, ind, tags)``:
            model: The gp.Model, updated and ready to optimize.
            x: x[i, j] packing variables.
            ind: ind[g] soft-minimum indicators, keyed by category
                index.
            tags: Maps each diagnosable constraint name to
                ``(kind, subject, limit)`` so the IIS and feasRelax
                results can be explained (see Conflict).
    """
    n, m = len(p.items), len(p.sacks)
    I, J = range(n), range(m)
    tags: dict[str, tuple[str, str, float]] = {}

    model = gp.Model("college_packing", env=env)

    # (10) x[i, j] = 1 if item i is packed in sack j.
    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")

    # (11) ind[g] = 1 if soft category g meets its minimum.
    soft = [g for g, c in enumerate(p.categories) if c.soft]
    ind = model.addVars(soft, vtype=GRB.BINARY, name="ind")

    members = {g: p.members(c.name) for g, c in enumerate(p.categories)}
    count = {g: gp.quicksum(x[i, j] for i in members[g] for j in J)
             for g in members}

    # Objective: total value + sum over g in S of
    #     beta_g * (l_g - count_g) * (1 - ind_g).
    total_value = gp.quicksum(p.items[i].value * x[i, j]
                              for i in I for j in J)
    penalty = gp.quicksum(
        p.categories[g].beta
        * (p.categories[g].min_items - count[g])
        * (1 - ind[g])
        for g in soft
    )
    model.setObjective(total_value + penalty, GRB.MAXIMIZE)
    if soft:
        model.Params.NonConvex = 2  # The penalty term is quadratic.

    # (1) Pack each item at most once.
    model.addConstrs((x.sum(i, "*") <= 1 for i in I), name="pack_once")

    # (2) Weight and (3) usable volume capacity of each sack.
    for j, s in enumerate(p.sacks):
        model.addConstr(
            gp.quicksum(p.items[i].weight * x[i, j] for i in I)
            <= s.weight_cap,
            name=f"weight_cap[{j}]",
        )
        tags[f"weight_cap[{j}]"] = ("sack_weight", s.name, s.weight_cap)

        model.addConstr(
            gp.quicksum(p.items[i].volume * x[i, j] for i in I)
            <= s.volume_cap,
            name=f"volume_cap[{j}]",
        )
        tags[f"volume_cap[{j}]"] = ("sack_volume", s.name, s.volume_cap)

    # (4) Hard category minimums and (5) category maximums. Empty
    # categories are skipped; validate() already rejected a hard
    # minimum above zero for them.
    for g, c in enumerate(p.categories):
        if not members[g]:
            continue
        if not c.soft and c.min_items > 0:
            name = f"cat_min[{g}]"
            model.addConstr(count[g] >= c.min_items, name=name)
            tags[name] = ("category_min", c.name, c.min_items)
        if c.max_items is not None:
            name = f"cat_max[{g}]"
            model.addConstr(count[g] <= c.max_items, name=name)
            tags[name] = ("category_max", c.name, c.max_items)

    # (6), (7) Soft minimums: together they force ind[g] = 1 exactly
    # when count_g >= l_g.
    for g in soft:
        c = p.categories[g]
        M = big_m(c.min_items, len(members[g]))
        model.addConstr(c.min_items - count[g] >= 1 - M * ind[g],
                        name=f"soft_min_met[{g}]")
        model.addConstr(c.min_items - count[g] <= M * (1 - ind[g]),
                        name=f"soft_min_not_met[{g}]")

    # (8) Prohibited items stay out of the sacks that ban them.
    for j, s in enumerate(p.sacks):
        banned = [i for i, it in enumerate(p.items)
                  if s.name in it.prohibited_in]
        if banned:
            name = f"prohibited[{j}]"
            model.addConstr(gp.quicksum(x[i, j] for i in banned) == 0,
                            name=name)
            tags[name] = ("sack_rules", s.name, 0)

    # (9) Required items must be packed.
    for i in p.required():
        name = f"required[{i}]"
        model.addConstr(x.sum(i, "*") == 1, name=name)
        tags[name] = ("required", p.items[i].name, 1)

    model.update()
    return model, x, ind, tags


# ---------------------------------------------------------------------------
# Reading results
# ---------------------------------------------------------------------------

def _assignment(p: PackingProblem, value_of) -> dict[int, int]:
    """Map each packed item index to its sack index.

    Args:
        p: The problem.
        value_of: Function ``(i, j) -> float`` giving the value of
            x[i, j] in some solution.

    Returns:
        ``{item index: sack index}`` for every packed item.
    """
    placed = {}
    for i in range(len(p.items)):
        for j in range(len(p.sacks)):
            if value_of(i, j) > 0.5:
                placed[i] = j
    return placed


def _solution(p: PackingProblem, placed: dict[int, int]) -> Solution:
    """Build a Solution from an _assignment() result."""
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

    soft = []
    for c in p.categories:
        if not c.soft:
            continue
        packed = sum(1 for i in p.members(c.name) if i in placed)
        met = packed >= c.min_items
        shortfall_cost = 0.0 if met else c.beta * (c.min_items - packed)
        soft.append(SoftResult(c.name, packed, c.min_items, met,
                               shortfall_cost))

    return Solution(
        total_value=sum(p.items[i].value for i in placed),
        max_value=sum(it.value for it in p.items),
        loads=loads,
        left_behind=[it.name for i, it in enumerate(p.items)
                     if i not in placed],
        soft=soft,
    )


# ---------------------------------------------------------------------------
# Diagnosing infeasibility
# ---------------------------------------------------------------------------

def _conflicts(model: gp.Model, tags) -> list[Conflict]:
    """Return the tagged constraints in an IIS of the model."""
    model.computeIIS()
    return [Conflict(*tags[c.ConstrName])
            for c in model.getConstrs()
            if c.IISConstr and c.ConstrName in tags]


def _fixes(p: PackingProblem, model: gp.Model, tags) -> list[Fix]:
    """Find the cheapest change to the limits that makes it feasible.

    Relaxes a copy of the model with feasRelax, weighting each
    constraint kind by RELAX_PENALTY, then compares the relaxed
    packing with the current limits.

    Args:
        p: The problem.
        model: The infeasible model from _build(). It is not changed.
        tags: The constraint tags from _build().

    Returns:
        One Fix per limit that the relaxed packing breaks. Empty if
        only packing rules or required items could fix the problem.
    """
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
            return []

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def solve(p: PackingProblem, time_limit: float = 30.0) -> SolveResult:
    """Validate, build and solve a packing problem.

    Args:
        p: The problem to solve.
        time_limit: Gurobi time limit in seconds.

    Returns:
        A SolveResult. Its status is:
            invalid: validate() found errors; only ``issues`` is set.
            optimal / time_limit: ``solution`` is set.
            infeasible: ``conflicts`` and ``fixes`` explain why.
            error: Gurobi stopped for another reason; see ``detail``.
    """
    issues = validate(p)
    if any(i.level == "error" for i in issues):
        return SolveResult("invalid", issues=issues)

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        model, x, _ind, tags = _build(p, env)
        try:
            model.Params.TimeLimit = time_limit
            model.optimize()
            # Re-solve without dual reductions to tell infeasible
            # apart from unbounded.
            if model.Status == GRB.INF_OR_UNBD:
                model.Params.DualReductions = 0
                model.optimize()

            if model.SolCount > 0:
                vals = model.getAttr("X", x)
                placed = _assignment(p, lambda i, j: vals[i, j])
                status = ("optimal" if model.Status == GRB.OPTIMAL
                          else "time_limit")
                return SolveResult(status, issues=issues,
                                   solution=_solution(p, placed))

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
