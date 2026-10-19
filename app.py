"""
Interactive demo of the NFL Fourth-Down Decision Assistant.

  pip install -r requirements-demo.txt
  streamlit run app.py

Uses the trained models in results/models.joblib (run `python fourth_down.py build` first if
that file is missing). Nothing here is fitted: it only calls the same option_values() engine
that produced the paper's results.
"""
import os

import joblib
import pandas as pd
import streamlit as st

import fourth_down as fd

st.set_page_config(page_title="Fourth-Down Decision Assistant", layout="centered")


@st.cache_resource
def load_bundle():
    path = os.path.join(fd.OUT_DIR, "models.joblib")
    if not os.path.exists(path):
        st.error("results/models.joblib not found. Run `python fourth_down.py build` first.")
        st.stop()
    return joblib.load(path)


bundle = load_bundle()
st.title("Fourth-Down Decision Assistant")
st.caption("Expected win probability of each choice, from models trained on 2021-2024 NFL plays "
           "(nflverse) and evaluated on the 2025 season.")

c1, c2 = st.columns(2)
with c1:
    ydstogo = st.number_input("Yards to go", 1, 30, 4)
    side = st.radio("Ball on", ["Opponent's side", "Own side"], horizontal=True)
    yard = st.slider("Yard line", 1, 50, 35)
    yardline_100 = yard if side == "Opponent's side" else 100 - yard
    st.caption(f"{yardline_100} yards from the end zone")
with c2:
    score_diff = st.number_input("Score difference (your team minus opponent)", -40, 40, -7)
    qtr = st.selectbox("Quarter", [1, 2, 3, 4], index=3)
    clock = st.text_input("Game clock in this quarter (mm:ss)", "15:00")
    t1, t2 = st.columns(2)
    timeouts = t1.selectbox("Your timeouts", [3, 2, 1, 0])
    opp_timeouts = t2.selectbox("Opponent timeouts", [3, 2, 1, 0])
    home = st.checkbox("Home team", value=True)

try:
    mm, ss = clock.split(":")
    q_left = int(mm) * 60 + int(ss)
    assert 0 <= q_left <= 900
except Exception:
    st.error("Enter the clock as mm:ss, between 00:00 and 15:00.")
    st.stop()
if ydstogo > yardline_100:
    st.error("Yards to go cannot be more than the distance to the end zone.")
    st.stop()

game_left = (4 - qtr) * 900 + q_left
half_left = q_left + (900 if qtr in (1, 3) else 0)
s = pd.DataFrame([{"ydstogo": ydstogo, "yardline_100": yardline_100, "score_differential": score_diff,
                   "qtr": qtr, "game_seconds_remaining": game_left, "half_seconds_remaining": half_left,
                   "posteam_timeouts_remaining": timeouts, "defteam_timeouts_remaining": opp_timeouts,
                   "is_home": int(home)}])
o = fd.option_values(s, bundle, "final").iloc[0]
rule = fd.rule_based_decision(ydstogo, yardline_100)
names = {"go": "GO FOR IT", "field_goal": "FIELD GOAL", "punt": "PUNT"}
strength = "toss-up" if o.margin < fd.LEAN else ("lean" if o.margin < fd.CLEAR else "clear")

st.divider()
st.subheader(f"Recommendation: {names[o.recommendation]}")
st.write(f"Margin over the next-best choice: **{o.margin * 100:.1f} win-probability points** ({strength}).")

rows = []
for k in fd.OPTIONS:
    v = o["wp_" + k]
    rows.append({"Choice": names[k], "Expected win probability": None if pd.isna(v) else round(v * 100, 1)})
tbl = pd.DataFrame(rows).set_index("Choice")
st.bar_chart(tbl.dropna(), horizontal=True, height=180)
st.dataframe(tbl.fillna("out of range").astype(str).rename(
    columns={"Expected win probability": "Expected win probability (%)"}), width="stretch")

st.write(f"Conversion probability: **{o.p_convert:.1%}**  ·  "
         f"Field-goal make probability ({yardline_100 + 18}-yard try): **{o.p_fg_make:.1%}**")
st.info(f"Rule baseline (yards to go + field position only) says: **{names[rule]}**"
        + ("" if rule == o.recommendation else " — the engine disagrees because it also uses score, time and timeouts."))
with st.expander("Known weak spots (see paper, Discussion)"):
    st.markdown("- Deep in its own territory (own 15 or deeper), the engine can favor GO on 4th & 3-6; "
                "very few teams have tried this, so the conversion estimate there is uncertain. A tested "
                "correction did not change this.\n"
                "- Treat toss-ups (margin under 1 WP point) as close calls, not firm advice.")
