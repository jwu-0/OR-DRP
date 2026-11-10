"""
UI helpers with no Streamlit dependency:
  * convert between editable tables (DataFrames) and a PackingProblem
  * turn the solver's structured output into plain-language sentences
"""

from __future__ import annotations

import pandas as pd

from packing.problem import Category, Issue, Item, PackingProblem, Sack

SACK_COLS = ["Bag", "Weight limit", "Empty bag weight", "Volume", "Packing efficiency"]
CAT_COLS = ["Category", "Minimum", "Maximum"]
ITEM_COLS = ["Item", "Value", "Weight", "Volume", "Categories", "Not allowed in"]


# ---------------------------------------------------------------- tables
def problem_to_frames(p: PackingProblem):
    sacks = pd.DataFrame(
        [[s.name, s.weight_limit, s.empty_weight, s.volume, s.alpha] for s in p.sacks],
        columns=SACK_COLS,
    )
    cats = pd.DataFrame(
        [[c.name, c.min_items, c.max_items] for c in p.categories], columns=CAT_COLS
    )
    items = pd.DataFrame(
        [[it.name, it.value, it.weight, it.volume,
          ", ".join(it.categories), ", ".join(it.prohibited_in)] for it in p.items],
        columns=ITEM_COLS,
    )
    return sacks, cats, items


def _blank(v) -> bool:
    return v is None or (not isinstance(v, str) and pd.isna(v)) or str(v).strip() == ""


def _num(v, default=0.0) -> float:
    return default if _blank(v) else float(v)


def _names(v) -> list[str]:
    return [] if _blank(v) else [s.strip() for s in str(v).split(",") if s.strip()]


def frames_to_problem(sacks: pd.DataFrame, cats: pd.DataFrame,
                      items: pd.DataFrame, units: dict) -> PackingProblem:
    """Rows without a name are ignored (e.g. a half-typed new row)."""
    return PackingProblem(
        sacks=[Sack(name=str(r["Bag"]).strip(),
                    weight_limit=_num(r["Weight limit"]),
                    empty_weight=_num(r["Empty bag weight"]),
                    volume=_num(r["Volume"]),
                    alpha=_num(r["Packing efficiency"], 1.0))
               for _, r in sacks.iterrows() if not _blank(r["Bag"])],
        categories=[Category(name=str(r["Category"]).strip(),
                             min_items=int(_num(r["Minimum"])),
                             max_items=None if _blank(r["Maximum"]) else int(r["Maximum"]))
                    for _, r in cats.iterrows() if not _blank(r["Category"])],
        items=[Item(name=str(r["Item"]).strip(),
                    value=int(round(_num(r["Value"]))),
                    weight=_num(r["Weight"]),
                    volume=_num(r["Volume"]),
                    categories=_names(r["Categories"]),
                    prohibited_in=_names(r["Not allowed in"]))
               for _, r in items.iterrows() if not _blank(r["Item"])],
        units=units,
    )


# ---------------------------------------------------------------- wording
def describe_issue(issue: Issue, units: dict) -> str:
    s, d, wu = issue.subject, issue.detail, units["weight"]
    return {
        "no_sacks": "Add at least one bag.",
        "no_items": "Add at least one item.",
        "duplicate_name": f"“{s}” appears more than once in {d.get('table')}. Give each one a different name.",
        "sack_no_weight": f"{s}: the empty bag weight is at least the weight limit, so nothing fits. Check both numbers.",
        "sack_no_volume": f"{s}: volume must be greater than 0.",
        "bad_alpha": f"{s}: packing efficiency must be between 0 and 1 (it’s {d.get('alpha')}).",
        "negative_number": f"{s}: value, weight, and volume can’t be negative.",
        "unknown_category": f"{s} is in category “{d.get('category')}”, which isn’t on the category list. Add it there or fix the spelling.",
        "unknown_sack": f"{s} is not allowed in “{d.get('sack')}”, but there’s no bag with that name. Fix the spelling.",
        "negative_limit": f"{s}: minimum and maximum can’t be negative.",
        "min_above_max": f"{s}: the minimum ({d.get('min')}) is higher than the maximum ({d.get('max')}).",
        "min_above_members": f"{s}: you need at least {d.get('min')}, but only {d.get('members')} items are in this category.",
        "min_above_packable": f"{s}: you need at least {d.get('min')}, but only {d.get('packable')} items in this category fit in any bag.",
        "item_fits_nowhere": f"{s} is too heavy or too big for every bag it’s allowed in, so it will be left behind.",
    }.get(issue.code, f"{s}: {issue.code}")


def describe_conflict(c, units: dict) -> str:
    wu, vu = units["weight"], units["volume"]
    return {
        "category_min": f"You need at least {c.limit:g} {c.subject} items.",
        "category_max": f"You allow at most {c.limit:g} {c.subject} items.",
        "sack_weight": f"{c.subject} can hold {c.limit:g} {wu} of contents.",
        "sack_volume": f"{c.subject} has about {c.limit:g} {vu} of usable space.",
        "sack_rules": f"Some items aren’t allowed in {c.subject}.",
    }[c.kind]


def describe_fix(f, units: dict) -> str:
    wu, vu = units["weight"], units["volume"]
    if f.kind == "category_min":
        return f"Lower the {f.subject} minimum from {f.current:g} to {f.suggested:g}."
    if f.kind == "category_max":
        return f"Raise the {f.subject} maximum from {f.current:g} to {f.suggested:g}."
    if f.kind == "sack_weight":
        return (f"{f.subject} would need {f.suggested - f.current:.1f} {wu} more capacity. "
                f"Swap in lighter items or use a bigger bag.")
    return (f"{f.subject} would need {f.suggested - f.current:.1f} {vu} more space. "
            f"Swap in smaller items or use a bigger bag.")
