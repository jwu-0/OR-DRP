"""
Streamlit UI. Run with:  streamlit run app.py

The UI never touches Gurobi. It builds a PackingProblem from the tables,
calls packing.solver.solve(), and turns the SolveResult into sentences.
"""

from pathlib import Path

import streamlit as st

from packing.problem import PackingProblem
from packing.solver import solve
from ui_helpers import (describe_conflict, describe_fix, describe_issue,
                        frames_to_problem, problem_to_frames)

EXAMPLE = Path(__file__).parent / "example_trip.json"

st.set_page_config(page_title="Pack my bags", page_icon="🧳", layout="wide")


# ---------------------------------------------------------------- state
def load_into_state(problem: PackingProblem):
    s, c, i = problem_to_frames(problem)
    st.session_state.sacks_df, st.session_state.cats_df, st.session_state.items_df = s, c, i
    st.session_state.units = problem.units
    st.session_state.ver = st.session_state.get("ver", 0) + 1   # fresh editors
    st.session_state.pop("last", None)


if "ver" not in st.session_state:
    load_into_state(PackingProblem.from_json(EXAMPLE.read_text()))
units = st.session_state.units
wu, vu = units["weight"], units["volume"]
ver = st.session_state.ver


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Your trip")
    upload = st.file_uploader("Open a saved trip", type="json")
    if upload is not None:
        upload_id = getattr(upload, "file_id", f"{upload.name}-{upload.size}")
        if st.session_state.get("upload_id") != upload_id:
            st.session_state.upload_id = upload_id
            try:
                load_into_state(PackingProblem.from_json(upload.getvalue().decode()))
                st.rerun()
            except Exception as e:  # malformed file
                st.error(f"Couldn’t open that file: {e}")
    if st.button("Reset to example trip"):
        load_into_state(PackingProblem.from_json(EXAMPLE.read_text()))
        st.rerun()


# ---------------------------------------------------------------- inputs
st.title("Pack my bags")
st.write("List your bags and what you’d like to bring. Rate each item’s value "
         "from 1 to 10, and the solver packs the combination worth the most.")

tab_bags, tab_items, tab_cats = st.tabs(["Bags", "Items", "Category rules"])
num = st.column_config.NumberColumn

with tab_bags:
    sacks_df = st.data_editor(
        st.session_state.sacks_df, key=f"sacks_{ver}", num_rows="dynamic", hide_index=True,
        column_config={
            "Weight limit": num(f"Weight limit ({wu})", min_value=0.0, help="Airline limit for this bag, including the bag itself"),
            "Empty bag weight": num(f"Empty bag weight ({wu})", min_value=0.0),
            "Volume": num(f"Volume ({vu})", min_value=0.0),
            "Packing efficiency": num(min_value=0.0, max_value=1.0, step=0.05,
                                      help="Share of the volume you can really fill. Soft bags ≈ 0.85, hard shells ≈ 0.80"),
        },
    )

with tab_items:
    st.caption("Separate several categories or bags with commas. Names must match the other tabs exactly.")
    items_df = st.data_editor(
        st.session_state.items_df, key=f"items_{ver}", num_rows="dynamic", hide_index=True,
        column_config={
            "Value": num(min_value=0, max_value=10, step=1, help="How much you want it, 1–10"),
            "Weight": num(f"Weight ({wu})", min_value=0.0, step=0.1),
            "Volume": num(f"Volume ({vu})", min_value=0.0, step=0.1),
        },
    )

with tab_cats:
    st.caption("Leave Maximum blank for no limit.")
    cats_df = st.data_editor(
        st.session_state.cats_df, key=f"cats_{ver}", num_rows="dynamic", hide_index=True,
        column_config={"Minimum": num(min_value=0, step=1), "Maximum": num(min_value=0, step=1)},
    )

problem = frames_to_problem(sacks_df, cats_df, items_df, units)

with st.sidebar:
    st.download_button("Save this trip", problem.to_json(),
                       file_name="my_trip.json", mime="application/json")


# ---------------------------------------------------------------- solve
if st.button("Find the best packing", type="primary"):
    with st.spinner("Packing…"):
        st.session_state.last = (problem.to_dict(), solve(problem))

last = st.session_state.get("last")
if last is None:
    st.stop()
if last[0] != problem.to_dict():
    st.info("You’ve changed your trip since the last run. Select **Find the best packing** to update.")
    st.stop()
result = last[1]


# ---------------------------------------------------------------- results
st.divider()
errors = [i for i in result.issues if i.level == "error"]
warnings = [i for i in result.issues if i.level == "warning"]

if result.status == "invalid":
    st.error("Fix these entries, then try again:\n\n"
             + "\n".join(f"- {describe_issue(i, units)}" for i in errors))
    st.stop()

for w in warnings:
    st.warning(describe_issue(w, units))

if result.status == "infeasible":
    st.error("Your rules can’t all be met at once. These ones conflict:\n\n"
             + "\n".join(f"- {describe_conflict(c, units)}" for c in result.conflicts))
    if result.fixes:
        st.info("Smallest change that would make it work:\n\n"
                + "\n".join(f"- {describe_fix(f, units)}" for f in result.fixes)
                + "\n\nMake the change in the tabs above and run it again.")
    else:
        st.info("Changing category limits or bag sizes alone won’t fix this. "
                "Check which items are not allowed in which bags, or add a bag.")
    st.stop()

if result.status == "error":
    st.error(f"The solver stopped unexpectedly ({result.detail}). Try again, "
             "or save your trip and check the numbers.")
    st.stop()

sol = result.solution
if result.status == "time_limit":
    st.warning("This is the best packing found within the time limit. It may not be the very best.")

packed = sum(len(ld.items) for ld in sol.loads)
st.success(f"Packed {packed} of {packed + len(sol.left_behind)} items, "
           f"worth {sol.total_value} of {sol.max_value} possible points.")

for col, ld in zip(st.columns(len(sol.loads)), sol.loads):
    with col:
        st.subheader(ld.sack)
        st.progress(min(ld.weight_used / ld.weight_cap, 1.0),
                    text=f"Weight: {ld.weight_used:.1f} of {ld.weight_cap:g} {wu}")
        st.progress(min(ld.volume_used / ld.volume_cap, 1.0),
                    text=f"Space: {ld.volume_used:.1f} of {ld.volume_cap:g} {vu}")
        st.markdown("\n".join(f"- {name}" for name in ld.items) or "_Empty_")

if sol.left_behind:
    with st.expander(f"Left behind ({len(sol.left_behind)})"):
        st.markdown("\n".join(f"- {name}" for name in sol.left_behind))
