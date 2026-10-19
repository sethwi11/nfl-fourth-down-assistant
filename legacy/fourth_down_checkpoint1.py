"""
NFL Fourth-Down Decision Assistant - Practical Checkpoint 1
Author: Seth Williams (COMP 365)

What this script does
  build   : downloads nflverse play-by-play data (2021-2025), filters to fourth-down
            plays, cleans them, saves fourth_down_clean.csv, runs the rule-based
            baseline and a distance-bucket conversion baseline, and saves results.
  predict : takes ONE fourth-down situation and prints the baseline recommendation
            plus an estimated conversion probability.

Usage
  python fourth_down_checkpoint1.py build
  python fourth_down_checkpoint1.py predict --ydstogo 2 --yardline_100 45 \
         --score_diff -3 --qtr 4 --seconds_left 300

Data: nflverse / nflfastR play-by-play (https://github.com/nflverse/nflverse-data/releases/tag/pbp)
"""
import argparse
import json
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEASONS = [2021, 2022, 2023, 2024, 2025]
TRAIN_SEASONS = [2021, 2022, 2023, 2024]   # temporal split: train on past ...
TEST_SEASON = 2025                          # ... test on the most recent season
DATA_DIR = "data"
CLEAN_CSV = "fourth_down_clean.csv"
URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.csv.gz"

KEEP_COLS = [
    "season", "game_id", "down", "play_id", "posteam", "defteam", "qtr",
    "game_seconds_remaining", "half_seconds_remaining", "yardline_100", "ydstogo",
    "score_differential", "posteam_timeouts_remaining", "wp", "play_type",
    "fourth_down_converted", "fourth_down_failed",
]
FEATURES = ["ydstogo", "yardline_100", "score_differential", "qtr", "game_seconds_remaining"]


# ----------------------------------------------------------------------------------
# Data loading and cleaning
# ----------------------------------------------------------------------------------
def download(season):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"pbp_{season}.csv.gz")
    if not os.path.exists(path):
        print(f"Downloading {season} ...")
        urllib.request.urlretrieve(URL.format(season), path)
    return path


def load_clean():
    frames = []
    for s in SEASONS:
        frames.append(pd.read_csv(download(s), usecols=KEEP_COLS, low_memory=False))
    raw = pd.concat(frames, ignore_index=True)
    stats = {"total_plays": int(len(raw))}

    d = raw[raw["down"] == 4].copy()
    stats["fourth_down_rows"] = int(len(d))

    # Keep only real decisions: go for it (run/pass), punt, field goal
    d = d[d["play_type"].isin(["run", "pass", "punt", "field_goal"])].copy()
    stats["after_play_type_filter"] = int(len(d))

    d["decision"] = d["play_type"].map(
        {"run": "go", "pass": "go", "punt": "punt", "field_goal": "field_goal"}
    )

    # Conversion label only exists for go-for-it plays
    d["converted"] = np.where(d["fourth_down_converted"] == 1, 1,
                      np.where(d["fourth_down_failed"] == 1, 0, np.nan))

    # Drop rows missing any model feature
    before = len(d)
    d = d.dropna(subset=FEATURES + ["wp"])
    stats["dropped_missing_features"] = int(before - len(d))

    # Go-for-it plays with no clear conversion result (e.g., penalty) are unusable for labels
    go_unlabeled = int(((d["decision"] == "go") & d["converted"].isna()).sum())
    stats["go_plays_without_label"] = go_unlabeled
    d = d[~((d["decision"] == "go") & d["converted"].isna())]

    d = d.drop(columns=["down", "fourth_down_converted", "fourth_down_failed"]).reset_index(drop=True)
    stats["clean_rows"] = int(len(d))
    return d, stats


# ----------------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------------
def rule_based_decision(ydstogo, yardline_100):
    """Non-AI baseline from the proposal: go only when distance is short, otherwise
    kick a field goal if in range (<= 37 yards from goal line = ~54-yd kick), else punt."""
    if ydstogo <= 2 and yardline_100 <= 50:
        return "go"
    if yardline_100 <= 37:
        return "field_goal"
    return "punt"


def bucket(y):
    """Distance bucket used by the conversion baseline."""
    return np.minimum(np.asarray(y), 10).astype(int)   # 1..9 individually, 10+ together


def fit_bucket_baseline(train_go):
    """Baseline conversion probability = historical conversion rate for that yards-to-go bucket."""
    rates = train_go.groupby(bucket(train_go["ydstogo"]))["converted"].mean().to_dict()
    overall = float(train_go["converted"].mean())
    return rates, overall


def predict_bucket(rates, overall, ydstogo):
    return np.array([rates.get(int(b), overall) for b in bucket(ydstogo)])


def metrics(y, p):
    return {
        "accuracy": round(float(accuracy_score(y, (p >= 0.5).astype(int))), 3),
        "roc_auc": round(float(roc_auc_score(y, p)), 3) if len(set(p)) > 1 else 0.5,
        "brier": round(float(brier_score_loss(y, p)), 3),
    }


# ----------------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------------
def build():
    d, stats = load_clean()
    d.to_csv(CLEAN_CSV, index=False)
    print(f"Saved {CLEAN_CSV}: {len(d)} rows")

    go = d[d["decision"] == "go"]
    train = go[go["season"].isin(TRAIN_SEASONS)]
    test = go[go["season"] == TEST_SEASON]
    stats.update({
        "decision_counts": d["decision"].value_counts().to_dict(),
        "go_train_n": int(len(train)), "go_test_n": int(len(test)),
        "go_conversion_rate_train": round(float(train["converted"].mean()), 3),
        "go_conversion_rate_test": round(float(test["converted"].mean()), 3),
    })

    y = test["converted"].astype(int).values

    # Baseline A: constant league-average conversion rate
    const = np.full(len(test), train["converted"].mean())
    res = {"constant_rate": metrics(y, const)}

    # Baseline B: distance-bucket historical rate
    rates, overall = fit_bucket_baseline(train)
    res["distance_bucket_rate"] = metrics(y, predict_bucket(rates, overall, test["ydstogo"]))

    # Preliminary ML (first look only; full model comparison is Checkpoint 2)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(train[FEATURES], train["converted"].astype(int))
    res["logistic_regression_prelim"] = metrics(y, lr.predict_proba(test[FEATURES])[:, 1])

    # Rule-based decision baseline vs. what NFL teams actually did (test season)
    t = d[d["season"] == TEST_SEASON].copy()
    t["rule"] = [rule_based_decision(a, b) for a, b in zip(t["ydstogo"], t["yardline_100"])]
    agree = float((t["rule"] == t["decision"]).mean())
    res["rule_vs_actual"] = {
        "n_plays": int(len(t)),
        "agreement": round(agree, 3),
        "rule_recommends": t["rule"].value_counts().to_dict(),
        "actual": t["decision"].value_counts().to_dict(),
        "crosstab": pd.crosstab(t["decision"], t["rule"]).to_dict(),
    }
    out = {"cleaning": stats, "conversion_prediction_test_2025": res,
           "bucket_rates_train": {str(k): round(v, 3) for k, v in rates.items()}}
    with open("baseline_results.json", "w") as f:
        json.dump(out, f, indent=2, default=int)
    print(json.dumps(out, indent=2, default=int))
    make_figure(train, test, rates, overall, t)


def make_figure(train, test, rates, overall, t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})

    # Left: conversion rate by yards to go
    xs = np.arange(1, 11)
    tr = train.groupby(bucket(train["ydstogo"]))["converted"].agg(["mean", "size"])
    te = test.groupby(bucket(test["ydstogo"]))["converted"].agg(["mean", "size"])
    ax[0].plot(xs, [tr.loc[x, "mean"] * 100 for x in xs], "o-", color="#1F4E79",
               label="Baseline (2021-24 rate)")
    ax[0].plot(xs, [te.loc[x, "mean"] * 100 if x in te.index else np.nan for x in xs], "s--",
               color="#C55A11", label="Actual 2025")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([str(x) if x < 10 else "10+" for x in xs])
    ax[0].set_xlabel("Yards to go")
    ax[0].set_ylabel("4th-down conversion rate (%)")
    ax[0].set_title("Go-for-it conversion rate by distance", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # Right: rule-based recommendation vs actual coach decision (2025)
    labels = ["go", "field_goal", "punt"]
    ct = pd.crosstab(t["decision"], t["rule"]).reindex(index=labels, columns=labels, fill_value=0)
    im = ax[1].imshow(ct.values, cmap="Blues")
    ax[1].set_xticks(range(3)); ax[1].set_yticks(range(3))
    ax[1].set_xticklabels(["Go", "FG", "Punt"]); ax[1].set_yticklabels(["Go", "FG", "Punt"])
    ax[1].set_xlabel("Rule-based recommendation")
    ax[1].set_ylabel("What teams actually did")
    for i in range(3):
        for j in range(3):
            v = ct.values[i, j]
            ax[1].text(j, i, str(v), ha="center", va="center",
                       color="white" if v > ct.values.max() * 0.55 else "black", fontsize=10)
    agree = (t["rule"] == t["decision"]).mean() * 100
    ax[1].set_title(f"Rule baseline vs actual, 2025 ({agree:.0f}% agree)", fontsize=10)

    plt.tight_layout()
    plt.savefig("checkpoint1_result.png", dpi=200)
    print("Saved checkpoint1_result.png")


# ----------------------------------------------------------------------------------
# predict (sample input -> observable output)
# ----------------------------------------------------------------------------------
def predict(args):
    if not os.path.exists(CLEAN_CSV):
        raise SystemExit("Run `python fourth_down_checkpoint1.py build` first.")
    d = pd.read_csv(CLEAN_CSV)
    go = d[(d["decision"] == "go") & (d["season"].isin(TRAIN_SEASONS))]
    rates, overall = fit_bucket_baseline(go)
    p = float(predict_bucket(rates, overall, [args.ydstogo])[0])
    rec = rule_based_decision(args.ydstogo, args.yardline_100)
    print("\n=== Fourth-Down Situation ===")
    print(f"Yards to go: {args.ydstogo} | Yard line (to goal): {args.yardline_100} | "
          f"Score diff: {args.score_diff:+d} | Qtr: {args.qtr} | Seconds left: {args.seconds_left}")
    print("\n=== Baseline Output ===")
    print(f"Rule-based recommendation : {rec.replace('_', ' ').upper()}")
    print(f"Est. conversion probability (distance-bucket baseline): {p:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    pr = sub.add_parser("predict")
    pr.add_argument("--ydstogo", type=int, required=True)
    pr.add_argument("--yardline_100", type=int, required=True, help="yards from opponent goal line")
    pr.add_argument("--score_diff", type=int, default=0)
    pr.add_argument("--qtr", type=int, default=2)
    pr.add_argument("--seconds_left", type=int, default=1800)
    a = ap.parse_args()
    build() if a.cmd == "build" else predict(a)
