"""Data layer for the college packing model.

Defines what a packing problem looks like, how it is saved to and
loaded from a trip JSON file, and the checks that catch bad input
before the solver runs. Nothing here knows about Gurobi or the UI.

A trip JSON file holds all the input in one place::

    {
      "units": {"weight": "lb", "volume": "L"},
      "sacks": [
        {"name": "Carry-on", "weight_limit": 22, "volume": 38,
         "empty_weight": 7, "alpha": 0.85},
        ...
      ],
      "categories": [
        {"name": "clothing", "min_items": 6, "max_items": 11},
        {"name": "snacks", "min_items": 3, "max_items": 8,
         "soft": true, "penalty": -10},
        ...
      ],
      "items": [
        {"name": "Laptop", "value": 10, "weight": 3.5, "volume": 1.2,
         "categories": ["electronics", "school"],
         "prohibited_in": [], "required": true},
        ...
      ]
    }

Fields left out of a sack, category or item take the defaults set in
the dataclasses below.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

DEFAULT_UNITS = {"weight": "lb", "volume": "L"}

# beta_g used for a soft category whose penalty is left at 0. Item
# values run 1-10, so each missing item costs as much as the most
# valuable item could be worth. A placeholder until beta is tuned.
DEFAULT_PENALTY = -10.0


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Sack:
    """A bag j that items can be packed into.

    Attributes:
        name: Unique bag name.
        weight_limit: Airline or bag limit, including the bag itself.
        volume: Nominal volume kv_j.
        empty_weight: Weight of the empty bag.
        alpha: Packing efficiency alpha_j, in (0, 1].
    """

    name: str
    weight_limit: float
    volume: float
    empty_weight: float = 0.0
    alpha: float = 1.0

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
    """A category g with bounds on how many of its items are packed.

    Attributes:
        name: Unique category name.
        min_items: Minimum l_g.
        max_items: Maximum u_g, or None for no maximum.
        soft: True if g is in S, so missing the minimum costs a
            penalty instead of making the model infeasible.
        penalty: beta_g, the (negative) points lost per item short
            of the minimum. 0 means DEFAULT_PENALTY. Only used when
            ``soft`` is True; see ``beta``.
    """

    name: str
    min_items: int = 0
    max_items: int | None = None
    soft: bool = False
    penalty: float = 0.0

    @property
    def beta(self) -> float:
        """beta_g: ``penalty``, or DEFAULT_PENALTY if it is 0."""
        return self.penalty or DEFAULT_PENALTY


@dataclass
class Item:
    """An item i that may be packed.

    Attributes:
        name: Unique item name.
        value: v_i, how much the item is wanted.
        weight: w_i.
        volume: f_i.
        categories: Names of the categories the item belongs to.
        prohibited_in: Names of the sacks the item may not go in.
        required: True if the item is in C_R and must be packed.
    """

    name: str
    value: int
    weight: float
    volume: float
    categories: list[str] = field(default_factory=list)
    prohibited_in: list[str] = field(default_factory=list)
    required: bool = False


@dataclass
class PackingProblem:
    """A full packing problem: sacks, categories, items and units."""

    sacks: list[Sack]
    categories: list[Category]
    items: list[Item]
    units: dict = field(default_factory=lambda: dict(DEFAULT_UNITS))

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> PackingProblem:
        """Build a problem from a dict in the trip JSON layout.

        Args:
            d: Parsed trip JSON (see the module docstring).

        Returns:
            The PackingProblem, with defaults for any missing
            optional fields.

        Raises:
            TypeError: If a sack, category or item has an unknown
                field or is missing a required one.
        """
        return cls(
            sacks=[Sack(**s) for s in d.get("sacks", [])],
            categories=[Category(**c) for c in d.get("categories", [])],
            items=[Item(**it) for it in d.get("items", [])],
            units=d.get("units", dict(DEFAULT_UNITS)),
        )

    @classmethod
    def from_json(cls, text: str) -> PackingProblem:
        return cls.from_dict(json.loads(text))

    def required(self) -> list[int]:
        """Return the indices of the required items (C_R)."""
        return [i for i, it in enumerate(self.items) if it.required]

    def members(self, category: str) -> list[int]:
        """Return the indices of the items in a category (C_g)."""
        return [i for i, it in enumerate(self.items)
                if category in it.categories]

    def fits_somewhere(self, item: Item) -> bool:
        """Return True if the item alone fits in some allowed sack."""
        return any(
            s.name not in item.prohibited_in
            and item.weight <= s.weight_cap
            and item.volume <= s.volume_cap
            for s in self.sacks
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    """One problem found in the input.

    Attributes:
        level: "error" blocks solving; "warning" does not.
        code: Machine-readable code; the UI turns it into a sentence.
        subject: Name of the item, sack or category concerned.
        detail: Extra values used in the message, such as the minimum.
    """

    level: str
    code: str
    subject: str
    detail: dict = field(default_factory=dict)


def validate(p: PackingProblem) -> list[Issue]:
    """Check a problem for input mistakes before solving.

    Args:
        p: The problem to check.

    Returns:
        Every issue found, in the order checked; empty if the input
        is fine. If a basic check (names, signs, unknown references)
        fails, the later checks are skipped because they rely on it.
    """
    issues: list[Issue] = []

    def err(code, subject, **detail):
        issues.append(Issue("error", code, subject, detail))

    def warn(code, subject, **detail):
        issues.append(Issue("warning", code, subject, detail))

    if not p.sacks:
        err("no_sacks", "")
    if not p.items:
        err("no_items", "")

    tables = (("bags", p.sacks), ("items", p.items),
              ("categories", p.categories))
    for table, rows in tables:
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
        return issues

    for it in p.items:
        if not p.fits_somewhere(it):
            if it.required:
                err("required_fits_nowhere", it.name)
            else:
                warn("item_fits_nowhere", it.name)

    for c in p.categories:
        if c.min_items < 0 or (c.max_items is not None and c.max_items < 0):
            err("negative_limit", c.name)
            continue
        if c.max_items is not None and c.min_items > c.max_items:
            err("min_above_max", c.name, min=c.min_items, max=c.max_items)
            continue
        if c.soft and c.penalty > 0:
            err("soft_needs_penalty", c.name, penalty=c.penalty)
            continue

        members = p.members(c.name)
        if c.max_items is not None:
            n_required = sum(1 for i in p.required() if i in members)
            if n_required > c.max_items:
                err("required_above_max", c.name,
                    required=n_required, max=c.max_items)
                continue

        # A missed soft minimum only costs a penalty, so just warn.
        report = warn if c.soft else err
        packable = [i for i in members if p.fits_somewhere(p.items[i])]
        if c.min_items > len(members):
            report("min_above_members", c.name,
                   min=c.min_items, members=len(members))
        elif c.min_items > len(packable):
            report("min_above_packable", c.name,
                   min=c.min_items, packable=len(packable))

    return issues
