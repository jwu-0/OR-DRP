"""
Data layer: what a packing problem looks like, how it is saved,
and checks that catch bad input before the solver ever runs.

Nothing here knows about Gurobi or about the UI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


# ---------------------------------------------------------------- data model
@dataclass
class Sack:
    name: str
    weight_limit: float          # airline / bag limit, including the bag itself
    volume: float                # nominal volume
    empty_weight: float = 0.0    # weight of the empty bag
    alpha: float = 1.0           # packing efficiency in (0, 1]

    @property
    def weight_cap(self) -> float:
        """Weight available for contents (kw_j)."""
        return self.weight_limit - self.empty_weight

    @property
    def volume_cap(self) -> float:
        """Usable volume (alpha_j * kv_j)."""
        return self.alpha * self.volume


@dataclass
class Category:
    name: str
    min_items: int = 0
    max_items: int | None = None  # None = no maximum


@dataclass
class Item:
    name: str
    value: int
    weight: float
    volume: float
    categories: list[str] = field(default_factory=list)
    prohibited_in: list[str] = field(default_factory=list)   # sack names


@dataclass
class PackingProblem:
    sacks: list[Sack]
    categories: list[Category]
    items: list[Item]
    units: dict = field(default_factory=lambda: {"weight": "lb", "volume": "L"})

    # ---- save / load -------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "PackingProblem":
        return cls(
            sacks=[Sack(**s) for s in d.get("sacks", [])],
            categories=[Category(**c) for c in d.get("categories", [])],
            items=[Item(**it) for it in d.get("items", [])],
            units=d.get("units", {"weight": "lb", "volume": "L"}),
        )

    @classmethod
    def from_json(cls, text: str) -> "PackingProblem":
        return cls.from_dict(json.loads(text))

    # ---- helpers -----------------------------------------------------
    def members(self, category: str) -> list[int]:
        """Indices of items that belong to a category."""
        return [i for i, it in enumerate(self.items) if category in it.categories]

    def fits_somewhere(self, item: Item) -> bool:
        """Can this item go in at least one sack on its own?"""
        return any(
            s.name not in item.prohibited_in
            and item.weight <= s.weight_cap
            and item.volume <= s.volume_cap
            for s in self.sacks
        )


# ---------------------------------------------------------------- validation
@dataclass
class Issue:
    level: str                 # "error" blocks solving, "warning" does not
    code: str                  # machine-readable; the UI turns it into a sentence
    subject: str               # the item / sack / category concerned
    detail: dict = field(default_factory=dict)


def validate(p: PackingProblem) -> list[Issue]:
    issues: list[Issue] = []

    def err(code, subject, **detail):
        issues.append(Issue("error", code, subject, detail))

    def warn(code, subject, **detail):
        issues.append(Issue("warning", code, subject, detail))

    if not p.sacks:
        err("no_sacks", "")
    if not p.items:
        err("no_items", "")

    for table, rows in (("bags", p.sacks), ("items", p.items), ("categories", p.categories)):
        seen = set()
        for r in rows:
            if r.name in seen:
                err("duplicate_name", r.name, table=table)
            seen.add(r.name)

    for s in p.sacks:
        if s.weight_cap <= 0:
            err("sack_no_weight", s.name, cap=s.weight_cap)
        if s.volume <= 0:
            err("sack_no_volume", s.name)
        if not 0 < s.alpha <= 1:
            err("bad_alpha", s.name, alpha=s.alpha)

    sack_names = {s.name for s in p.sacks}
    cat_names = {c.name for c in p.categories}
    for it in p.items:
        if it.weight < 0 or it.volume < 0 or it.value < 0:
            err("negative_number", it.name)
        for c in it.categories:
            if c not in cat_names:
                err("unknown_category", it.name, category=c)
        for s in it.prohibited_in:
            if s not in sack_names:
                err("unknown_sack", it.name, sack=s)

    if any(i.level == "error" for i in issues):
        return issues          # later checks assume the basics are sound

    for it in p.items:
        if not p.fits_somewhere(it):
            warn("item_fits_nowhere", it.name)

    for c in p.categories:
        if c.min_items < 0 or (c.max_items is not None and c.max_items < 0):
            err("negative_limit", c.name)
            continue
        if c.max_items is not None and c.min_items > c.max_items:
            err("min_above_max", c.name, min=c.min_items, max=c.max_items)
            continue
        members = p.members(c.name)
        packable = [i for i in members if p.fits_somewhere(p.items[i])]
        if c.min_items > len(members):
            err("min_above_members", c.name, min=c.min_items, members=len(members))
        elif c.min_items > len(packable):
            err("min_above_packable", c.name, min=c.min_items, packable=len(packable))

    return issues
