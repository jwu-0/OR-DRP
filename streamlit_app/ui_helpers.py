"""Helpers for the Streamlit app that don't depend on Streamlit.

Converts between the editable tables (DataFrames) and a
PackingProblem, and turns the solver's structured output into
plain-language sentences.
"""

from __future__ import annotations

import pandas as pd

from packing_model.problem import Category, Issue, Item, PackingProblem, Sack

SACK_COLS = ["Bag", "Weight limit", "Empty bag weight", "Volume",
             "Packing efficiency"]
CAT_COLS = ["Category", "Minimum", "Maximum", "Soft minimum", "Penalty"]
ITEM_COLS = ["Item", "Value", "Weight", "Volume", "Categories",
             "Not allowed in", "Required"]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def problem_to_frames(p: PackingProblem):
    """Split a problem into the three tables the app edits.

    Args:
        p: The problem to show.

    Returns:
        A tuple ``(sacks, cats, items)`` of DataFrames with columns
        SACK_COLS, CAT_COLS and ITEM_COLS. Lists of names (an item's
        categories and banned bags) become comma-separated strings.
    """
    sacks = pd.DataFrame(
        [[s.name, s.weight_limit, s.empty_weight, s.volume, s.alpha]
         for s in p.sacks],
        columns=SACK_COLS,
    )
    cats = pd.DataFrame(
        [[c.name, c.min_items, c.max_items, c.soft, c.penalty]
         for c in p.categories],
        columns=CAT_COLS,
    )
    items = pd.DataFrame(
        [[it.name, it.value, it.weight, it.volume,
          ", ".join(it.categories), ", ".join(it.prohibited_in),
          it.required]
         for it in p.items],
        columns=ITEM_COLS,
    )
    return sacks, cats, items


def _blank(v) -> bool:
    return (v is None or (not isinstance(v, str) and pd.isna(v))
            or str(v).strip() == "")


def _num(v, default=0.0) -> float:
    return default if _blank(v) else float(v)


def _bool(v) -> bool:
    return False if _blank(v) else bool(v)


def _names(v) -> list[str]:
    return [] if _blank(v) else [s.strip() for s in str(v).split(",")
                                 if s.strip()]


def frames_to_problem(sacks: pd.DataFrame, cats: pd.DataFrame,
                      items: pd.DataFrame, units: dict) -> PackingProblem:
    """Build a problem from the three edited tables.

    The inverse of problem_to_frames(). Rows with a blank name, such
    as a half-typed new row, are ignored. Blank numbers become 0
    (packing efficiency becomes 1), and a blank Maximum means no
    maximum.

    Args:
        sacks: Table with columns SACK_COLS.
        cats: Table with columns CAT_COLS.
        items: Table with columns ITEM_COLS.
        units: ``{"weight": ..., "volume": ...}`` unit labels.

    Returns:
        The PackingProblem. It is not validated here.
    """
    return PackingProblem(
        sacks=[
            Sack(name=str(r["Bag"]).strip(),
                 weight_limit=_num(r["Weight limit"]),
                 empty_weight=_num(r["Empty bag weight"]),
                 volume=_num(r["Volume"]),
                 alpha=_num(r["Packing efficiency"], 1.0))
            for _, r in sacks.iterrows() if not _blank(r["Bag"])
        ],
        categories=[
            Category(name=str(r["Category"]).strip(),
                     min_items=int(_num(r["Minimum"])),
                     max_items=(None if _blank(r["Maximum"])
                                else int(r["Maximum"])),
                     soft=_bool(r["Soft minimum"]),
                     penalty=_num(r["Penalty"]))
            for _, r in cats.iterrows() if not _blank(r["Category"])
        ],
        items=[
            Item(name=str(r["Item"]).strip(),
                 value=int(round(_num(r["Value"]))),
                 weight=_num(r["Weight"]),
                 volume=_num(r["Volume"]),
                 categories=_names(r["Categories"]),
                 prohibited_in=_names(r["Not allowed in"]),
                 required=_bool(r["Required"]))
            for _, r in items.iterrows() if not _blank(r["Item"])
        ],
        units=units,
    )


# ---------------------------------------------------------------------------
# Wording
# ---------------------------------------------------------------------------

def describe_issue(issue: Issue, units: dict) -> str:
    """Return one sentence describing a validation Issue."""
    s, d = issue.subject, issue.detail
    messages = {
        "no_sacks": "Add at least one bag.",
        "no_items": "Add at least one item.",
        "duplicate_name": (f"“{s}” appears more than once in "
                           f"{d.get('table')}. Give each one a different "
                           f"name."),
        "sack_no_weight": (f"{s}: the empty bag weight is at least the "
                           f"weight limit, so nothing fits. Check both "
                           f"numbers."),
        "sack_no_volume": f"{s}: volume must be greater than 0.",
        "bad_alpha": (f"{s}: packing efficiency must be between 0 and 1 "
                      f"(it’s {d.get('alpha')})."),
        "negative_number": (f"{s}: value, weight, and volume can’t be "
                            f"negative."),
        "unknown_category": (f"{s} is in category “{d.get('category')}”, "
                             f"which isn’t on the category list. Add it "
                             f"there or fix the spelling."),
        "unknown_sack": (f"{s} is not allowed in “{d.get('sack')}”, but "
                         f"there’s no bag with that name. Fix the "
                         f"spelling."),
        "negative_limit": f"{s}: minimum and maximum can’t be negative.",
        "min_above_max": (f"{s}: the minimum ({d.get('min')}) is higher "
                          f"than the maximum ({d.get('max')})."),
        "min_above_members": (f"{s}: you need at least {d.get('min')}, but "
                              f"only {d.get('members')} items are in this "
                              f"category."),
        "min_above_packable": (f"{s}: you need at least {d.get('min')}, "
                               f"but only {d.get('packable')} items in "
                               f"this category fit in any bag."),
        "item_fits_nowhere": (f"{s} is too heavy or too big for every bag "
                              f"it’s allowed in, so it will be left "
                              f"behind."),
        "required_fits_nowhere": (f"{s} is required, but it’s too heavy or "
                                  f"too big for every bag it’s allowed "
                                  f"in."),
        "required_above_max": (f"{s}: {d.get('required')} required items "
                               f"are in this category, but the maximum is "
                               f"{d.get('max')}."),
        "soft_needs_penalty": (f"{s}: a soft minimum’s penalty must be "
                               f"negative (it’s {d.get('penalty')}), for "
                               f"example -5, or 0 for the default."),
    }
    return messages.get(issue.code, f"{s}: {issue.code}")


def describe_conflict(c, units: dict) -> str:
    """Return one sentence describing a solver Conflict."""
    wu, vu = units["weight"], units["volume"]
    return {
        "category_min": f"You need at least {c.limit:g} {c.subject} items.",
        "category_max": f"You allow at most {c.limit:g} {c.subject} items.",
        "sack_weight": f"{c.subject} can hold {c.limit:g} {wu} of contents.",
        "sack_volume": (f"{c.subject} has about {c.limit:g} {vu} of usable "
                        f"space."),
        "sack_rules": f"Some items aren’t allowed in {c.subject}.",
        "required": f"{c.subject} is required.",
    }[c.kind]


def describe_fix(f, units: dict) -> str:
    """Return one sentence describing a suggested solver Fix."""
    wu, vu = units["weight"], units["volume"]
    if f.kind == "category_min":
        return (f"Lower the {f.subject} minimum from {f.current:g} to "
                f"{f.suggested:g}.")
    if f.kind == "category_max":
        return (f"Raise the {f.subject} maximum from {f.current:g} to "
                f"{f.suggested:g}.")
    if f.kind == "sack_weight":
        return (f"{f.subject} would need {f.suggested - f.current:.1f} {wu} "
                f"more capacity. Swap in lighter items or use a bigger bag.")
    return (f"{f.subject} would need {f.suggested - f.current:.1f} {vu} "
            f"more space. Swap in smaller items or use a bigger bag.")
