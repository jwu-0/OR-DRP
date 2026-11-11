"""Solve a trip from the command line and print the packing plan.

Usage::

    python -m packing_model streamlit_app/data/example_trip.json

Input is one trip JSON file (layout in ``packing_model.problem``).
Output, on stdout, is any input issues, the solve status, then either
each sack's contents and fill level, soft-minimum results and the
items left behind, or the conflicts and suggested fixes.
"""

import argparse

from .problem import PackingProblem
from .solver import solve


def main():
    parser = argparse.ArgumentParser(description="College packing model")
    parser.add_argument("trip_json", help="path to a trip JSON file")
    args = parser.parse_args()

    with open(args.trip_json, encoding="utf-8") as f:
        problem = PackingProblem.from_json(f.read())

    result = solve(problem)
    for issue in result.issues:
        print(f"[{issue.level}] {issue.code}: {issue.subject} "
              f"{issue.detail or ''}")
    print(f"Status: {result.status}")

    sol = result.solution
    if sol is None:
        for c in result.conflicts:
            print(f"  conflict: {c.kind} {c.subject} (limit {c.limit:g})")
        for f in result.fixes:
            print(f"  fix: {f.kind} {f.subject} "
                  f"{f.current:g} -> {f.suggested:g}")
        return

    print(f"Objective: {sol.objective:g} "
          f"(item value {sol.total_value} of {sol.max_value})")
    for ld in sol.loads:
        print(f"\n{ld.sack}: {ld.weight_used:.2f}/{ld.weight_cap:g} weight, "
              f"{ld.volume_used:.2f}/{ld.volume_cap:.2f} volume")
        for name in ld.items:
            print(f"    {name}")
    if sol.soft:
        print("\nSoft category minimums:")
        for r in sol.soft:
            status = "met" if r.met else f"NOT met (penalty {r.penalty:g})"
            print(f"    {r.category}: {r.packed}/{r.minimum} {status}")
    if sol.left_behind:
        print(f"\nLeft behind: {', '.join(sol.left_behind)}")


if __name__ == "__main__":
    main()
