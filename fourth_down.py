"""
NFL Fourth-Down Decision Assistant
Author: Seth Williams

Recommends GO, FIELD GOAL or PUNT on fourth down by estimating the expected win probability
of each choice, and evaluates those recommendations on the held-out 2025 NFL season.

Pipeline (all trained on 2021-2024, evaluated on 2025):
  1. Conversion models: two baselines (league rate, distance-bucket rate) and four learned
     models (logistic regression, random forest, gradient boosting, situation-neutral spline
     logistic regression), compared on the same 931 held-out 2025 go-for-it plays.
     Model choice uses leave-one-season-out CV on 2021-24 only.
  2. Win-probability (WP) model: monotonic, regularized gradient boosting trained on
     165,285 regulation plays from 2021-24.
  3. Decision engine: E[WP] of GO, FIELD GOAL and PUNT; recommends the largest.
  4. Recommendation evaluation on 3,973 regulation fourth downs in 2025 (T3-T5 in README).

Commands
  python fourth_down.py build                     # train + evaluate everything -> results/
  python fourth_down.py build --source nflverse   # same, but re-download fresh nflverse files
  python fourth_down.py predict --ydstogo 4 --yardline_100 35 --score_diff -7 --qtr 4 --seconds_left 900
  python fourth_down.py snapshot                  # rebuild data/snapshot from data/pbp_*.csv.gz

Data: nflverse play-by-play, CC BY 4.0 (https://github.com/nflverse/nflverse-data/releases/tag/pbp)
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, StandardScaler

SEASONS = [2021, 2022, 2023, 2024, 2025]
TRAIN_SEASONS = [2021, 2022, 2023, 2024]
TEST_SEASON = 2025
DATA_DIR = "data"
OUT_DIR = "results"
SNAPSHOT = os.path.join("data", "snapshot", "nflverse_pbp_2021_2025_subset.csv.gz")
URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.csv.gz"
RNG = 365

COLS = [
    "season", "season_type", "game_id", "play_id", "posteam", "defteam", "posteam_type",
    "down", "qtr", "game_seconds_remaining", "half_seconds_remaining", "yardline_100",
    "ydstogo", "score_differential", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "wp", "wpa", "play_type", "result",
    "fourth_down_converted", "fourth_down_failed", "yards_gained", "touchdown",
    "field_goal_result", "kick_distance",
]
CONV_FEATURES = ["ydstogo", "yardline_100", "score_differential", "qtr", "game_seconds_remaining"]
WP_FEATURES = ["score_differential", "diff_time_ratio", "game_seconds_remaining",
               "half_seconds_remaining", "yardline_100", "down", "ydstogo",
               "posteam_timeouts_remaining", "defteam_timeouts_remaining", "is_home"]
WP_FEATURES_V2 = WP_FEATURES + ["red_zone", "goal_to_go", "log_yardline", "late_sd",
                                 "need_td_late", "need_fg_late", "lead_late"]
OPTIONS = ["go", "field_goal", "punt"]
# Margin labels used everywhere (CLI, app, paper): toss-up < 1, lean 1-3, clear >= 3 WP points
LEAN, CLEAR = 0.01, 0.03
DEEP_YL, SHRINK_K = 85, 50   # sparse region for conversion (own 15 or deeper) and shrinkage strength
# Engine variants. "final" is the default used by predict/app and the headline results.
ENGINES = {
    "final":       {"conv": "neutral", "wp": "v2", "shrink": True},   # both weak-case fixes
    "wp_fix_only": {"conv": "neutral", "wp": "v2", "shrink": False},  # ablation: WP v2 only
    "checkpoint":  {"conv": "neutral", "wp": "v1", "shrink": False},  # engine reported at Checkpoint 2
    "full_conv":   {"conv": "full",    "wp": "v2", "shrink": False},  # ablation: 5-feature conversion model
}


# ==================================================================================
# Data
# ==================================================================================
def download(season):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"pbp_{season}.csv.gz")
    if not os.path.exists(path):
        print(f"Downloading {season} ...")
        urllib.request.urlretrieve(URL.format(season), path)
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(source="snapshot"):
    """source='snapshot': the frozen column subset shipped in this repo (exact reproduction).
    source='nflverse': download the full season files from nflverse (may differ if nflverse
    has re-released the files since the snapshot was taken)."""
    if source == "snapshot":
        if not os.path.exists(SNAPSHOT):
            raise SystemExit(f"{SNAPSHOT} not found. Run with --source nflverse, or "
                             "`python fourth_down.py snapshot` after downloading.")
        return pd.read_csv(SNAPSHOT, low_memory=False)
    return pd.concat([pd.read_csv(download(s), usecols=COLS, low_memory=False) for s in SEASONS],
                     ignore_index=True)


def make_snapshot():
    """Write the 26 columns this project uses, for all 2021-25 plays, to one compressed file."""
    raw = load_raw("nflverse")[COLS]
    os.makedirs(os.path.dirname(SNAPSHOT), exist_ok=True)
    raw.to_csv(SNAPSHOT, index=False, compression={"method": "gzip", "mtime": 0})
    hashes = {f"pbp_{s}.csv.gz": sha256(os.path.join(DATA_DIR, f"pbp_{s}.csv.gz")) for s in SEASONS}
    hashes[os.path.basename(SNAPSHOT)] = sha256(SNAPSHOT)
    with open(os.path.join(DATA_DIR, "snapshot", "SHA256SUMS.json"), "w") as f:
        json.dump({"downloaded_utc": time.strftime("%Y-%m-%d", time.gmtime()), "sha256": hashes}, f, indent=2)
    print(f"Wrote {SNAPSHOT}: {len(raw):,} plays, {len(COLS)} columns")


def add_wp_features(d):
    d = d.copy()
    elapsed = (3600 - d["game_seconds_remaining"]) / 3600
    d["diff_time_ratio"] = d["score_differential"] * np.exp(4 * elapsed)  # as in nflfastR
    if "is_home" not in d:
        d["is_home"] = (d["posteam_type"] == "home").astype(int)
    # v2 features (weak-case fixes): goal-line and late-game structure
    late = d["game_seconds_remaining"] <= 300
    sd = d["score_differential"]
    d["red_zone"] = (d["yardline_100"] <= 20).astype(int)
    d["goal_to_go"] = (d["ydstogo"] >= d["yardline_100"]).astype(int)
    d["log_yardline"] = np.log(d["yardline_100"].clip(1, 99))
    d["late_sd"] = sd * late
    d["need_td_late"] = (sd.between(-8, -4) & late).astype(int)
    d["need_fg_late"] = (sd.between(-3, -1) & late).astype(int)
    d["lead_late"] = (sd.between(1, 8) & late).astype(int)
    return d


def clean_fourth_downs(raw):
    """Fourth-down decisions only (go / punt / field goal) -> 20,267 rows."""
    stats = {"total_plays": int(len(raw))}
    d = raw[raw["down"] == 4].copy()
    stats["fourth_down_rows"] = int(len(d))
    d = d[d["play_type"].isin(["run", "pass", "punt", "field_goal"])].copy()
    d["decision"] = d["play_type"].map({"run": "go", "pass": "go", "punt": "punt",
                                        "field_goal": "field_goal"})
    d["converted"] = np.where(d["fourth_down_converted"] == 1, 1,
                              np.where(d["fourth_down_failed"] == 1, 0, np.nan))
    d = d.dropna(subset=CONV_FEATURES + ["wp"])
    d = d[~((d["decision"] == "go") & d["converted"].isna())].reset_index(drop=True)
    stats["clean_rows"] = int(len(d))
    stats["decision_counts"] = d["decision"].value_counts().to_dict()
    return add_wp_features(d), stats


# ==================================================================================
# Models
# ==================================================================================
def conv_models():
    return {
        "logistic_regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=25,
                                                random_state=RNG, n_jobs=-1),
        # monotonic: more yards to go can only lower conversion probability
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=40,
            monotonic_cst=[-1, 0, 0, 0, 0], random_state=RNG),
    }


def neutral_conv_model():
    """Situation-neutral conversion model: ONLY yards to go and field position (smooth splines).
    Score/time are left out on purpose: in the data they mostly encode WHEN teams choose to go
    (selection), not how hard the play is. Score/time enter the decision via the WP model."""
    return make_pipeline(
        ColumnTransformer([("s", SplineTransformer(n_knots=4, degree=2),
                            ["ydstogo", "yardline_100"])]),
        LogisticRegression(max_iter=3000, C=0.5))


def bucket(y):
    return np.minimum(np.asarray(y), 10).astype(int)


def bucket_rates(train_go):
    return (train_go.groupby(bucket(train_go["ydstogo"]))["converted"].mean().to_dict(),
            float(train_go["converted"].mean()))


def predict_bucket(rates, overall, ydstogo):
    return np.array([rates.get(int(b), overall) for b in bucket(ydstogo)])


def metrics(y, p):
    return {"accuracy": round(float(accuracy_score(y, (p >= 0.5).astype(int))), 3),
            "roc_auc": round(float(roc_auc_score(y, p)), 3) if len(set(p)) > 1 else 0.5,
            "brier": round(float(brier_score_loss(y, p)), 3)}


class GameBoot:
    """Bootstrap that resamples whole GAMES with replacement (plays in a game are related)."""
    def __init__(self, games, seed=RNG):
        _, inv = np.unique(np.asarray(games), return_inverse=True)
        order = np.argsort(inv, kind="stable")
        self.groups = np.split(order, np.cumsum(np.bincount(inv))[:-1])
        self.n_rows = len(order)
        self.rng = np.random.default_rng(seed)

    def sample(self):
        pick = self.rng.integers(0, len(self.groups), len(self.groups))
        return np.concatenate([self.groups[k] for k in pick])

    def sample_plays(self):
        return self.rng.integers(0, self.n_rows, self.n_rows)


def ci95(v, nd=3):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return [round(float(np.percentile(v, 2.5)), nd), round(float(np.percentile(v, 97.5)), nd)]


def bootstrap_vs(y, p_model, p_base, games, n=2000):
    """Paired bootstrap of (model - baseline) AUC and Brier on the same plays.
    Main CI resamples games; the play-level CI is kept only to show the difference."""
    out = {}
    for label, use_games in [("game", True), ("play", False)]:
        gb = GameBoot(games)
        da, db = [], []
        for _ in range(n):
            i = gb.sample() if use_games else gb.sample_plays()
            if len(set(y[i])) < 2:
                continue
            da.append(roc_auc_score(y[i], p_model[i]) - roc_auc_score(y[i], p_base[i]))
            db.append(brier_score_loss(y[i], p_model[i]) - brier_score_loss(y[i], p_base[i]))
        key = "" if use_games else "_play_level"
        out["auc_diff_ci95" + key], out["brier_diff_ci95" + key] = ci95(da), ci95(db)
    out["resampling_unit"] = "game"
    out["n_games"] = int(len(np.unique(games)))
    return out


def season_cv_brier(model, train):
    """Leave-one-season-out CV on 2021-24 only (used to pick the model, not the test set)."""
    scores = []
    for s in TRAIN_SEASONS:
        tr, va = train[train["season"] != s], train[train["season"] == s]
        m = model.fit(tr[CONV_FEATURES], tr["converted"].astype(int))
        scores.append(brier_score_loss(va["converted"].astype(int),
                                       m.predict_proba(va[CONV_FEATURES])[:, 1]))
    return round(float(np.mean(scores)), 4)


def wp_training_frame(raw, seasons):
    p = raw[raw["season"].isin(seasons) & raw["down"].isin([1, 2, 3, 4]) &
            (raw["qtr"] <= 4) & raw["posteam"].notna() & raw["result"].notna() &
            (raw["result"] != 0)].dropna(subset=["yardline_100", "score_differential",
                                                  "posteam_timeouts_remaining"])
    p = add_wp_features(p)
    home_won = p["result"] > 0
    return p, np.where(p["is_home"] == 1, home_won, ~home_won).astype(int)


WP_V1_CST = [1, 1, 0, 0, -1, -1, -1, 1, -1, 1]
WP_V2_CST = WP_V1_CST + [0, 0, -1, 1, 0, 0, 0]
WP_V1_HP = dict(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=1000, l2_regularization=1.0)
WP_V2_HP = dict(WP_V1_HP, max_iter=300)   # chosen by grouped CV in tune_wp.py (config "v2b")


def wp_spec(version):
    return (WP_FEATURES, WP_V1_CST, WP_V1_HP) if version == "v1" else (WP_FEATURES_V2, WP_V2_CST, WP_V2_HP)


def train_wp_model_version(raw, version):
    p, y = wp_training_frame(raw, TRAIN_SEASONS)
    feats, cst, hp = wp_spec(version)
    m = HistGradientBoostingClassifier(monotonic_cst=cst, random_state=RNG, **hp).fit(p[feats], y)
    m.feature_list_ = feats
    return m, int(len(p))


def train_wp_model(raw):
    """Win probability for the team with the ball, trained on 2021-24 regulation plays."""
    p = raw[raw["season"].isin(TRAIN_SEASONS) & raw["down"].isin([1, 2, 3, 4]) &
            (raw["qtr"] <= 4) & raw["posteam"].notna() & raw["result"].notna() &
            (raw["result"] != 0)].dropna(subset=["yardline_100", "score_differential",
                                                  "posteam_timeouts_remaining"])
    p = add_wp_features(p)
    home_won = p["result"] > 0
    y = np.where(p["is_home"] == 1, home_won, ~home_won).astype(int)
    # Monotonic constraints (order = WP_FEATURES): lead +, farther from goal -, later down -,
    # more yards to go -, own timeouts +, opponent timeouts -, home +.
    # Strong regularization: 165k plays but only ~1,100 game outcomes, so a flexible model
    # memorizes game noise (CP2 first attempt: WP jumped 4 points over 10 seconds of clock).
    # Settings chosen by 5-fold CV grouped by game on 2021-24 (see tune_wp.py).
    cst = [1, 1, 0, 0, -1, -1, -1, 1, -1, 1]
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                       min_samples_leaf=1000, l2_regularization=1.0,
                                       monotonic_cst=cst, random_state=RNG)
    m.fit(p[WP_FEATURES], y)
    return m, int(len(p))


def eval_wp_model(wp_model, raw):
    t = raw[(raw["season"] == TEST_SEASON) & raw["down"].isin([1, 2, 3, 4]) & (raw["qtr"] <= 4) &
            raw["posteam"].notna() & raw["result"].notna() & (raw["result"] != 0)].dropna(
        subset=["yardline_100", "score_differential", "posteam_timeouts_remaining", "wp"])
    t = add_wp_features(t)
    home_won = t["result"] > 0
    y = np.where(t["is_home"] == 1, home_won, ~home_won).astype(int)
    ours = wp_model.predict_proba(t[getattr(wp_model, "feature_list_", WP_FEATURES)])[:, 1]
    late = (t["game_seconds_remaining"] <= 300).values
    rz = (t["yardline_100"] <= 20).values
    return {"n_plays_2025": int(len(t)),
            "brier_ours": round(float(brier_score_loss(y, ours)), 4),
            "brier_nflverse_wp": round(float(brier_score_loss(y, t["wp"])), 4),
            "brier_coinflip": 0.25,
            "brier_ours_last5min": round(float(brier_score_loss(y[late], ours[late])), 4),
            "brier_nflverse_last5min": round(float(brier_score_loss(y[late], t["wp"].values[late])), 4),
            "brier_ours_redzone": round(float(brier_score_loss(y[rz], ours[rz])), 4),
            "brier_nflverse_redzone": round(float(brier_score_loss(y[rz], t["wp"].values[rz])), 4),
            "n_last5min": int(late.sum()), "n_redzone": int(rz.sum())}


def fit_transition_tables(raw):
    """Simple empirical models for what happens after each choice (2021-24 only)."""
    r = raw[raw["season"].isin(TRAIN_SEASONS)].sort_values(["game_id", "play_id"]).copy()
    nxt = r.groupby("game_id")[["posteam", "yardline_100"]].shift(-1)
    r["next_posteam"], r["next_yl"] = nxt["posteam"], nxt["yardline_100"]

    # FG make probability by kick distance (logistic regression)
    fg = r[(r["play_type"] == "field_goal") & r["kick_distance"].notna()]
    fg_model = LogisticRegression().fit(fg[["kick_distance"]],
                                        (fg["field_goal_result"] == "made").astype(int))

    # Punt: where does the opponent start? (5-yard bins of the punting team's yardline_100)
    pu = r[(r["play_type"] == "punt") & (r["next_posteam"] == r["defteam"]) &
           r["next_yl"].notna()]
    punt_tbl = pu.groupby((pu["yardline_100"] // 5) * 5)["next_yl"].mean().to_dict()

    # Successful conversions: average yards gained by yards to go (beyond the line to gain)
    cv = r[(r["down"] == 4) & (r["fourth_down_converted"] == 1) &
           r["play_type"].isin(["run", "pass"])]
    gain_tbl = cv.groupby(bucket(cv["ydstogo"]))["yards_gained"].median().to_dict()

    # Opponent start after a kickoff that follows a score (most recent rules: 2024 season)
    # (nflverse lists the RECEIVING team as posteam on kickoff plays)
    ko = r[(r["play_type"] == "kickoff") & (r["season"] == 2024) & r["next_yl"].notna() &
           (r["next_posteam"] == r["posteam"])]
    ko_start = float(ko["next_yl"].median()) if len(ko) else 70.0
    return {"fg_model": fg_model,
            "punt_tbl": {int(k): float(v) for k, v in punt_tbl.items()},
            "gain_tbl": {int(k): float(v) for k, v in gain_tbl.items()},
            "ko_start": ko_start}


def describe_transitions(tr):
    return {"ko_start_yardline_100": tr["ko_start"],
            "punt_opp_start_by_bin": tr["punt_tbl"], "conv_median_gain": tr["gain_tbl"]}


# ==================================================================================
# Decision engine
# ==================================================================================
def _wp(wp_model, rows):
    df = add_wp_features(pd.DataFrame(rows))
    return wp_model.predict_proba(df[getattr(wp_model, "feature_list_", WP_FEATURES)])[:, 1]


def fit_deep_offset(train_go, conv_model):
    """Sparse-region correction for conversion (weak case 1). Teams rarely go for it from their own
    15 or deeper, so the model extrapolates there. Shift its logit in that region toward the observed
    2021-24 rate, shrunk by n/(n+K): with few attempts the shift stays small."""
    deep = train_go[train_go["yardline_100"] >= DEEP_YL]
    n = len(deep)
    obs = float(deep["converted"].mean())
    pred = float(conv_model.predict_proba(deep[CONV_FEATURES])[:, 1].mean())
    lg = lambda q: np.log(q / (1 - q))
    off = (lg(obs) - lg(pred)) * n / (n + SHRINK_K)
    return {"n": n, "observed_rate": round(obs, 3), "model_mean_p": round(pred, 3),
            "logit_offset": float(off), "k": SHRINK_K}


def conversion_prob(s, bundle, spec):
    p = bundle["conv_models"][spec["conv"]].predict_proba(s[CONV_FEATURES])[:, 1]
    if spec["shrink"]:
        off = bundle["deep_offset"]["logit_offset"]
        deep = s["yardline_100"].values >= DEEP_YL
        lg = np.log(p / (1 - p))
        p = np.where(deep, 1 / (1 + np.exp(-(lg + off))), p)
    return p


def _state(sd, gsec, hsec, yl, down, ytg, to_pos, to_def, home):
    return {"score_differential": sd, "game_seconds_remaining": max(gsec, 0),
            "half_seconds_remaining": max(hsec, 0), "yardline_100": float(np.clip(yl, 1, 99)),
            "down": down, "ydstogo": ytg, "posteam_timeouts_remaining": to_pos,
            "defteam_timeouts_remaining": to_def, "is_home": home}


def option_values(situations, bundle, engine="final"):
    """For each situation (DataFrame with CONV_FEATURES + timeouts/home/half seconds),
    return expected win probability of GO, FIELD GOAL and PUNT plus their parts."""
    spec = ENGINES[engine]
    wp_m, tr = bundle["wp_models"][spec["wp"]], bundle["transitions"]
    s = situations.reset_index(drop=True)
    p_conv = conversion_prob(s, bundle, spec)
    p_fg = tr["fg_model"].predict_proba(pd.DataFrame({"kick_distance": s["yardline_100"] + 18}))[:, 1]

    us_rows, opp_rows = [], []   # states to score with the WP model
    for _, r in s.iterrows():
        sd, g, h, yl, ytg = r.score_differential, r.game_seconds_remaining, \
            r.half_seconds_remaining, r.yardline_100, r.ydstogo
        tp, td, home = r.posteam_timeouts_remaining, r.defteam_timeouts_remaining, r.is_home
        gain = max(ytg, tr["gain_tbl"].get(int(min(ytg, 10)), ytg))
        # GO success: TD if the gain reaches the goal line, else 1st down further on
        if gain >= yl:
            opp_rows.append(_state(-(sd + 7), g - 10, h - 10, tr["ko_start"], 1, 10, td, tp, 1 - home))
        else:
            ny = yl - gain
            us_rows.append(_state(sd, g - 6, h - 6, ny, 1, min(10, ny), tp, td, home))
        # GO failure: opponent takes over at the spot
        opp_rows.append(_state(-sd, g - 6, h - 6, 100 - yl, 1, 10, td, tp, 1 - home))
        # FG made: opponent receives kickoff down 3 more
        opp_rows.append(_state(-(sd + 3), g - 10, h - 10, tr["ko_start"], 1, 10, td, tp, 1 - home))
        # FG missed: opponent ball at spot of kick (min own 20)
        opp_rows.append(_state(-sd, g - 6, h - 6, min(93 - yl, 80), 1, 10, td, tp, 1 - home))
        # PUNT: opponent starts where punts from this area usually leave them
        b = int(min((yl // 5) * 5, 95))
        keys = sorted(tr["punt_tbl"])
        p_yl = tr["punt_tbl"].get(b, tr["punt_tbl"][keys[0]] if b < keys[0] else 80.0)
        opp_rows.append(_state(-sd, g - 8, h - 8, p_yl, 1, 10, td, tp, 1 - home))

    us = list(_wp(wp_m, us_rows)) if us_rows else []
    opp = list(1 - _wp(wp_m, opp_rows))  # convert to OUR win probability
    out = []
    for i, r in s.iterrows():
        gain = max(r.ydstogo, tr["gain_tbl"].get(int(min(r.ydstogo, 10)), r.ydstogo))
        wp_go_succ = opp.pop(0) if gain >= r.yardline_100 else us.pop(0)
        wp_go_fail, wp_fg_make, wp_fg_miss, wp_punt = opp.pop(0), opp.pop(0), opp.pop(0), opp.pop(0)
        go = p_conv[i] * wp_go_succ + (1 - p_conv[i]) * wp_go_fail
        fg = p_fg[i] * wp_fg_make + (1 - p_fg[i]) * wp_fg_miss
        # Field goals beyond ~63 yards are not realistic options
        if r.yardline_100 + 18 > 63:
            fg = np.nan
        out.append({"p_convert": p_conv[i], "p_fg_make": p_fg[i],
                    "wp_go": go, "wp_field_goal": fg, "wp_punt": wp_punt})
    o = pd.DataFrame(out)
    vals = o[["wp_go", "wp_field_goal", "wp_punt"]].to_numpy()
    best = np.nanargmax(vals, axis=1)
    o["recommendation"] = [OPTIONS[b] for b in best]
    srt = -np.sort(-np.nan_to_num(vals, nan=-1), axis=1)
    o["margin"] = srt[:, 0] - srt[:, 1]
    return o


def rule_based_decision(ydstogo, yardline_100):
    """Rule baseline (from the project proposal). Uses ONLY yards to go and field position."""
    if ydstogo <= 2 and yardline_100 <= 50:
        return "go"
    if yardline_100 <= 37:
        return "field_goal"
    return "punt"


REC_CODE = {"go": 0, "field_goal": 1, "punt": 2}
N_SIT = 6 * 4 * 5 * 2


def situation_code(t):
    """Situation strata for the matched comparison: yards to go (1, 2, 3, 4-5, 6-9, 10+) x
    field position (opp 1-20, opp 21-49, own 50-21, own 20-1) x score (down 9+, down 1-8, tied,
    up 1-8, up 9+) x period (Q1-3, Q4) = 240 cells."""
    ytg = np.digitize(t["ydstogo"].values, [2, 3, 4, 6, 10])
    fld = np.digitize(t["yardline_100"].values, [21, 50, 80])
    sco = np.digitize(t["score_differential"].values, [-8, 0, 1, 9])
    per = (t["qtr"].values == 4).astype(int)
    return ((ytg * 4 + fld) * 5 + sco) * 2 + per


def _simple_gap(rec, fol, wpa, idx, min_n=20):
    """Within each recommendation group: mean WPA(followed) - mean WPA(not followed); the overall
    value weights groups by size (groups with >= min_n plays on both sides). WP points."""
    key = rec[idx] * 2 + fol[idx]
    n = np.bincount(key, minlength=6)
    sm = np.bincount(key, weights=wpa[idx], minlength=6)
    diffs, num, den = np.full(3, np.nan), 0.0, 0.0
    for k in range(3):
        nf, nn = n[2 * k + 1], n[2 * k]
        if nf > 0 and nn > 0:
            diffs[k] = sm[2 * k + 1] / nf - sm[2 * k] / nn
        if nf >= min_n and nn >= min_n:
            num += diffs[k] * (nf + nn)
            den += nf + nn
    return 100 * num / den, 100 * diffs


def _matched_gap(strata, fol, wpa, idx, n_strata):
    """Same comparison, but only between plays in the SAME recommendation group AND the same
    situation stratum; differences averaged with stratum-size weights. Returns (gap, share used)."""
    key = strata[idx] * 2 + fol[idx]
    n = np.bincount(key, minlength=2 * n_strata).reshape(n_strata, 2)
    sm = np.bincount(key, weights=wpa[idx], minlength=2 * n_strata).reshape(n_strata, 2)
    ok = (n[:, 0] > 0) & (n[:, 1] > 0)
    d = sm[ok, 1] / n[ok, 1] - sm[ok, 0] / n[ok, 0]
    w = n[ok].sum(1)
    return 100 * float((d * w).sum() / w.sum()), float(w.sum() / len(idx))


def association_tests(t, rec_cols, n_boot=1000):
    """T3 for several recommenders on the same plays, with game-level bootstrap CIs.
    rec_cols: {label: column with 'go'/'field_goal'/'punt'}. These are ASSOCIATIONS, not effects."""
    wpa = t["wpa"].values
    dec = t["decision"].map(REC_CODE).values
    sit = situation_code(t)
    arrs = {}
    for lab, col in rec_cols.items():
        rec = t[col].map(REC_CODE).values
        arrs[lab] = (rec, (dec == rec).astype(int), rec * N_SIT + sit)
    def stats(idx):
        out = {}
        for lab, (rec, fol, strata) in arrs.items():
            g, per = _simple_gap(rec, fol, wpa, idx)
            m, share = _matched_gap(strata, fol, wpa, idx, 3 * N_SIT)
            out[lab] = (g, per, m, share)
        return out
    point = stats(np.arange(len(t)))
    gb = GameBoot(t["game_id"].values)
    boots = [stats(gb.sample()) for _ in range(n_boot)]
    play = [stats(gb.sample_plays()) for _ in range(n_boot)]
    res = {}
    for lab in rec_cols:
        g, per, m, share = point[lab]
        res[lab] = {
            "weighted_gap_pp": round(g, 2),
            "weighted_gap_ci95_game": ci95([b[lab][0] for b in boots], 2),
            "weighted_gap_ci95_play_level": ci95([b[lab][0] for b in play], 2),
            "per_group_gap_pp": {k: round(float(per[REC_CODE[k]]), 2) for k in OPTIONS},
            "per_group_gap_ci95_game": {k: ci95([b[lab][1][REC_CODE[k]] for b in boots], 2) for k in OPTIONS},
            "matched_gap_pp": round(m, 2),
            "matched_gap_ci95_game": ci95([b[lab][2] for b in boots], 2),
            "matched_share_of_plays_used": round(share, 3),
        }
    labs = list(rec_cols)
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            a, b = labs[i], labs[j]
            res[f"{a}_minus_{b}"] = {
                "weighted_gap_diff_ci95_game": ci95([x[a][0] - x[b][0] for x in boots], 2),
                "matched_gap_diff_ci95_game": ci95([x[a][2] - x[b][2] for x in boots], 2)}
    res["n_games"] = int(t["game_id"].nunique())
    res["n_boot"] = n_boot
    return res


def margin_label(m):
    return np.where(m < LEAN, "toss_up", np.where(m < CLEAR, "lean", "clear"))


def evaluate_engine(t, bundle, engine):
    """All recommendation tests for one engine on the 2025 fourth downs."""
    t = t.reset_index(drop=True).copy()
    ov = option_values(t, bundle, engine)
    t = pd.concat([t, ov], axis=1)
    t["rule"] = [rule_based_decision(a, b) for a, b in zip(t["ydstogo"], t["yardline_100"])]
    t["model_follow"] = t["decision"] == t["recommendation"]
    t["rule_follow"] = t["decision"] == t["rule"]
    t["wp_after"] = t["wp"] + t["wpa"]   # nflverse WP after the play (independent of our model)
    t["pred_taken"] = np.choose(t["decision"].map(REC_CODE).values,
                                [t["wp_go"].values, t["wp_field_goal"].fillna(0).values,
                                 t["wp_punt"].values])
    t["margin_label"] = margin_label(t["margin"].values)

    ev = {"n_plays": int(len(t)), "n_games": int(t["game_id"].nunique()),
          "model_recommends": t["recommendation"].value_counts().to_dict(),
          "rule_recommends": t["rule"].value_counts().to_dict(),
          "actual": t["decision"].value_counts().to_dict(),
          "agreement_model_vs_teams": round(float(t["model_follow"].mean()), 3),
          "agreement_rule_vs_teams": round(float(t["rule_follow"].mean()), 3),
          "model_vs_rule_differ": int((t["recommendation"] != t["rule"]).sum()),
          "margin_labels": t["margin_label"].value_counts().to_dict(),
          "go_margin_labels": t.loc[t["recommendation"] == "go", "margin_label"].value_counts().to_dict()}

    # T3 raw group means (for the table)
    def follow_table(col_rec, col_follow):
        rows = {}
        for rec in OPTIONS:
            g = t[t[col_rec] == rec]
            f, nf = g[g[col_follow]], g[~g[col_follow]]
            rows[rec] = {"n_followed": int(len(f)), "n_not": int(len(nf)),
                         "wpa_followed": round(float(f["wpa"].mean()) * 100, 2) if len(f) else None,
                         "wpa_not": round(float(nf["wpa"].mean()) * 100, 2) if len(nf) else None}
        return rows
    ev["wpa_follow_model"] = follow_table("recommendation", "model_follow")
    ev["wpa_follow_rule"] = follow_table("rule", "rule_follow")
    ev["association"] = association_tests(t, {"engine": "recommendation", "rule": "rule"})

    # GO calls by strength, same thresholds as the CLI/app labels (lean >= 1, clear >= 3 WP points)
    for lab, thr in [("clear_go_calls", CLEAR), ("lean_or_clear_go_calls", LEAN)]:
        g = t[(t["recommendation"] == "go") & (t["margin"] >= thr)]
        went = g["decision"] == "go"
        ev[lab] = {"threshold_pts": thr * 100, "n": int(len(g)), "teams_went": int(went.sum()),
                   "wpa_went": round(float(g.loc[went, "wpa"].mean()) * 100, 2),
                   "wpa_kicked": round(float(g.loc[~went, "wpa"].mean()) * 100, 2)}

    # T4 consistency check: our predicted WP for the action taken vs nflverse's post-play WP estimate
    bias = {}
    for a in OPTIONS:
        g = t[t["decision"] == a]
        bias[a] = {"n": int(len(g)),
                   "mean_predicted": round(float(g["pred_taken"].mean()), 3),
                   "mean_nflverse_after": round(float(g["wp_after"].mean()), 3),
                   "mae": round(float((g["pred_taken"] - g["wp_after"]).abs().mean()), 3)}
    ev["value_consistency_check"] = bias

    # T5 sensitivity: do score and time actually change the recommendation?
    neutral = t.copy()
    neutral["score_differential"], neutral["qtr"] = 0, 2
    neutral["game_seconds_remaining"], neutral["half_seconds_remaining"] = 2700, 900
    neutral["posteam_timeouts_remaining"] = neutral["defteam_timeouts_remaining"] = 3
    nv = option_values(neutral, bundle, engine)
    ev["sensitivity"] = {"changed_when_score_time_neutralized": int((nv["recommendation"] != t["recommendation"]).sum()),
                         "share": round(float((nv["recommendation"] != t["recommendation"]).mean()), 3)}
    return ev, t


def decision_grid(bundle, engine):
    """Recommendations for a tied game, Q2 15:00, averaged over home/away (Figure: decision chart)."""
    yls, ytgs = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90], [1, 2, 3, 4, 5, 7, 10]
    s = pd.DataFrame([{"ydstogo": yy, "yardline_100": yl, "score_differential": 0, "qtr": 2,
                       "game_seconds_remaining": 2700, "half_seconds_remaining": 900,
                       "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3, "is_home": h}
                      for yl in yls for yy in ytgs for h in (0, 1)])
    s = pd.concat([s, option_values(s, bundle, engine)], axis=1)
    s.loc[s["ydstogo"] > s["yardline_100"], ["wp_go", "wp_field_goal", "wp_punt"]] = np.nan
    g = s.groupby(["yardline_100", "ydstogo"])[["wp_go", "wp_field_goal", "wp_punt"]].mean()
    best = g.fillna(-1).idxmax(axis=1).unstack().reindex(index=yls, columns=ytgs)
    gap = ((g["wp_go"] - g[["wp_field_goal", "wp_punt"]].max(axis=1)) * 100).unstack().reindex(index=yls, columns=ytgs)
    return yls, ytgs, best, gap


def weak_case_checks(d, bundle, tabs):
    """Numbers behind the weak cases, before (checkpoint engine) and after the fixes (final engine)."""
    out = {}
    go = d[(d["decision"] == "go") & d["season"].isin(TRAIN_SEASONS)]
    deep = go[go["yardline_100"] >= DEEP_YL]
    out["train_go_attempts_own15_or_deeper"] = {"n": int(len(deep)),
                                                "conversion_rate": round(float(deep["converted"].mean()), 3)}
    out["deep_offset"] = bundle["deep_offset"]
    case1 = pd.DataFrame([{"ydstogo": 5, "yardline_100": 90, "score_differential": 0, "qtr": 2,
                           "game_seconds_remaining": 2700, "half_seconds_remaining": 900,
                           "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3, "is_home": 1}])
    case2 = pd.DataFrame([{"ydstogo": 8, "yardline_100": 30, "score_differential": -4, "qtr": 4,
                           "game_seconds_remaining": 120, "half_seconds_remaining": 120,
                           "posteam_timeouts_remaining": 1, "defteam_timeouts_remaining": 3, "is_home": 1}])
    for eng in ["checkpoint", "wp_fix_only", "final"]:
        c1 = option_values(case1, bundle, eng).iloc[0]
        c2 = option_values(case2, bundle, eng).iloc[0]
        tt = tabs[eng]
        dz = tt[(tt["yardline_100"] >= DEEP_YL) & tt["ydstogo"].between(3, 6) & (tt["recommendation"] == "go")]
        yls, ytgs, best, gap = decision_grid(bundle, eng)
        own = {f"own {100 - yl}": [k.replace("wp_", "") for k in best.loc[yl].tolist()] for yl in [90, 80, 70]}
        out[eng] = {
            "weak_case_1_4th5_own10_tied_Q2": {"p_convert": round(float(c1.p_convert), 3),
                                               "wp_go": round(float(c1.wp_go), 3), "wp_punt": round(float(c1.wp_punt), 3),
                                               "margin": round(float(c1.margin), 3), "recommendation": c1.recommendation},
            "weak_case_2_4th8_opp30_down4_2min": {"wp_go": round(float(c2.wp_go), 3),
                                                  "wp_field_goal": round(float(c2.wp_field_goal), 3),
                                                  "wp_punt": round(float(c2.wp_punt), 3),
                                                  "margin": round(float(c2.margin), 3), "recommendation": c2.recommendation},
            "2025_go_calls_own15_or_deeper_ytg3to6": {"n": int(len(dz)),
                                                      "teams_punted": int((dz["decision"] == "punt").sum())},
            "decision_grid_own_territory_tied_Q2_ytg_1_2_3_4_5_7_10": own,
        }
    for ver in ["v1", "v2"]:
        yls_ = [2, 5, 10, 20, 52, 80]
        wps = _wp(bundle["wp_models"][ver], [_state(0, 2694, 894, v, 1, min(10, v), 3, 3, 0) for v in yls_])
        late = _wp(bundle["wp_models"][ver], [_state(sd, 110, 110, 70, 1, 10, 3, 1, 0) for sd in (1, 4)])
        out[f"wp_{ver}_curves"] = {
            "opp_wp_1st_down_by_yards_from_goal": {str(v): round(float(w), 3) for v, w in zip(yls_, wps)},
            "gap_5_vs_20_yards_pts": round(float(wps[1] - wps[3]) * 100, 1),
            "opp_leading_by_1_vs_4_with_1m50_left_own30": [round(float(x), 3) for x in late]}
    out["punt_from_own_10_opp_start_yards_from_goal"] = round(float(bundle["transitions"]["punt_tbl"][90]), 1)
    q = pd.DataFrame([{"ydstogo": 6, "yardline_100": 72, "score_differential": 0, "qtr": 1,
                       "game_seconds_remaining": 3500}])
    preds = {}
    for name, m in conv_models().items():
        m.fit(go[CONV_FEATURES], go["converted"].astype(int))
        preds[name] = round(float(m.predict_proba(q)[0, 1]), 3)
    preds["situation_neutral_lr"] = round(float(bundle["conv_models"]["neutral"].predict_proba(q)[0, 1]), 3)
    out["selection_bias_4th6_own28_tied_Q1"] = {"model_predictions": preds,
                                                "league_rate_at_6_yards_2021_24":
                                                    round(float(go.loc[go["ydstogo"] == 6, "converted"].mean()), 3)}
    return out


NFL4TH_CSV = os.path.join("data", "nfl4th_2025_recommendations.csv")


def nfl4th_benchmark(t):
    """External benchmark: nfl4th (Baldwin) recommendations for the same 2025 plays, produced by
    benchmark_nfl4th.R. Agreement + the same association tests on the plays both systems cover."""
    if not os.path.exists(NFL4TH_CSV):
        return {"available": False}
    n4 = pd.read_csv(NFL4TH_CSV)
    m = t.merge(n4[["game_id", "play_id", "nfl4th_rec", "nfl4th_go_wp", "nfl4th_fg_wp", "nfl4th_punt_wp"]],
                on=["game_id", "play_id"], how="inner")
    m = m[m["nfl4th_rec"].isin(OPTIONS)].reset_index(drop=True)
    out = {"available": True, "n_matched": int(len(m)), "n_engine_plays": int(len(t)),
           "nfl4th_recommends": m["nfl4th_rec"].value_counts().to_dict(),
           "agreement": {"engine_vs_nfl4th": round(float((m["recommendation"] == m["nfl4th_rec"]).mean()), 3),
                         "rule_vs_nfl4th": round(float((m["rule"] == m["nfl4th_rec"]).mean()), 3),
                         "teams_vs_nfl4th": round(float((m["decision"] == m["nfl4th_rec"]).mean()), 3),
                         "engine_vs_teams_on_matched": round(float((m["recommendation"] == m["decision"]).mean()), 3)},
           "crosstab_engine_rows_nfl4th_cols": pd.crosstab(m["recommendation"], m["nfl4th_rec"]).to_dict(orient="index"),
           "association": association_tests(m, {"engine": "recommendation", "nfl4th": "nfl4th_rec", "rule": "rule"})}
    # when they disagree, which choice did go-for-it vs kick fall on?
    dis = m[m["recommendation"] != m["nfl4th_rec"]]
    out["disagreements"] = {"n": int(len(dis)),
                            "engine_go_nfl4th_kick": int(((dis["recommendation"] == "go") & (dis["nfl4th_rec"] != "go")).sum()),
                            "engine_kick_nfl4th_go": int(((dis["recommendation"] != "go") & (dis["nfl4th_rec"] == "go")).sum())}
    return out

# ==================================================================================
# Build: train + evaluate everything
# ==================================================================================
def build(source="snapshot"):
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = load_raw(source)
    d, stats = clean_fourth_downs(raw)
    d.to_csv(os.path.join(OUT_DIR, "fourth_down_clean.csv"), index=False)
    go = d[d["decision"] == "go"]
    train, test = go[go["season"].isin(TRAIN_SEASONS)], go[go["season"] == TEST_SEASON]
    y = test["converted"].astype(int).values
    res = {"cleaning": stats, "n_train_go": int(len(train)), "n_test_go": int(len(test)),
           "n_test_go_games": int(test["game_id"].nunique())}

    # ---- T1. Conversion: baselines vs ML on the same 931 plays -----------------------
    rates, overall = bucket_rates(train)
    preds = {"constant_rate": np.full(len(test), overall),
             "distance_bucket": predict_bucket(rates, overall, test["ydstogo"])}
    cv = {}
    for name, m in conv_models().items():
        cv[name] = season_cv_brier(m, train)
        m.fit(train[CONV_FEATURES], train["converted"].astype(int))
        preds[name] = m.predict_proba(test[CONV_FEATURES])[:, 1]
    chosen = min(cv, key=cv.get)
    conv_model = conv_models()[chosen].fit(train[CONV_FEATURES], train["converted"].astype(int))
    neutral = neutral_conv_model()
    cv["situation_neutral_lr"] = season_cv_brier(neutral_conv_model(), train)
    neutral.fit(train[CONV_FEATURES], train["converted"].astype(int))
    preds["situation_neutral_lr"] = neutral.predict_proba(test[CONV_FEATURES])[:, 1]
    games = test["game_id"].values
    res["conversion"] = {
        "cv_brier_2021_24_leave_one_season_out": cv, "chosen_by_cv": chosen,
        "test_2025": {k: metrics(y, p) for k, p in preds.items()},
        "chosen_vs_distance_bucket_bootstrap": bootstrap_vs(y, preds[chosen], preds["distance_bucket"], games),
        "neutral_vs_distance_bucket_bootstrap": bootstrap_vs(y, preds["situation_neutral_lr"],
                                                             preds["distance_bucket"], games),
    }
    sl = {}
    for lab, mask in {"1-2 yds": test["ydstogo"] <= 2, "3-5 yds": test["ydstogo"].between(3, 5),
                      "6-9 yds": test["ydstogo"].between(6, 9), "10+ yds": test["ydstogo"] >= 10,
                      "own half (yl>50)": test["yardline_100"] > 50,
                      "opp half (yl<=50)": test["yardline_100"] <= 50}.items():
        mk = mask.values
        sl[lab] = {"n": int(mk.sum()), "actual_rate": round(float(y[mk].mean()), 3),
                   "model_mean_p": round(float(preds[chosen][mk].mean()), 3),
                   "bucket_mean_p": round(float(preds["distance_bucket"][mk].mean()), 3),
                   "brier_model": round(float(brier_score_loss(y[mk], preds[chosen][mk])), 3),
                   "brier_bucket": round(float(brier_score_loss(y[mk], preds["distance_bucket"][mk])), 3)}
    res["conversion"]["slices_chosen_model"] = sl

    # ---- T2. Win-probability models (v1 = checkpoint, v2 = weak-case fix) + transitions --
    wp_v1, n_wp = train_wp_model_version(raw, "v1")
    wp_v2, _ = train_wp_model_version(raw, "v2")
    res["wp_model"] = {"n_train_plays": n_wp, "v1": eval_wp_model(wp_v1, raw), "v2": eval_wp_model(wp_v2, raw)}
    tr = fit_transition_tables(raw)
    res["transitions"] = {**describe_transitions(tr),
                          "fg_make_p_at_40yd": round(float(tr["fg_model"].predict_proba(
                              pd.DataFrame({"kick_distance": [40]}))[0, 1]), 3),
                          "fg_make_p_at_55yd": round(float(tr["fg_model"].predict_proba(
                              pd.DataFrame({"kick_distance": [55]}))[0, 1]), 3)}
    bundle = {"wp_models": {"v1": wp_v1, "v2": wp_v2}, "transitions": tr, "conv_name": chosen,
              "conv_models": {"full": conv_model, "neutral": neutral}}
    bundle["deep_offset"] = fit_deep_offset(train, neutral)
    deep_test = test["yardline_100"].values >= DEEP_YL
    p_fix = conversion_prob(test, bundle, ENGINES["final"])
    bundle["deep_offset"]["test_2025_deep_attempts"] = {
        "n": int(deep_test.sum()), "actual_rate": round(float(y[deep_test].mean()), 3) if deep_test.any() else None,
        "mean_p_before": round(float(preds["situation_neutral_lr"][deep_test].mean()), 3) if deep_test.any() else None,
        "mean_p_after": round(float(p_fix[deep_test].mean()), 3) if deep_test.any() else None}
    res["conversion"]["test_2025"]["situation_neutral_lr_with_deep_fix"] = metrics(y, p_fix)
    joblib.dump(bundle, os.path.join(OUT_DIR, "models.joblib"))

    # ---- T3-T5. Recommendations for every 2025 regulation fourth down, every engine ------
    t = d[(d["season"] == TEST_SEASON) & (d["qtr"] <= 4)].reset_index(drop=True)
    t = t.dropna(subset=["posteam_timeouts_remaining", "defteam_timeouts_remaining", "wpa"]).reset_index(drop=True)
    evals, tabs = {}, {}
    for e in ENGINES:
        evals[e], tabs[e] = evaluate_engine(t, bundle, e)
    res["recommendation_eval_2025"] = evals
    tf = tabs["final"].copy()
    for e in ENGINES:
        if e != "final":
            tf[f"recommendation_{e}"] = tabs[e]["recommendation"].values
    tf.to_csv(os.path.join(OUT_DIR, "recommendations_2025.csv"), index=False)
    res["nfl4th_benchmark"] = nfl4th_benchmark(tabs["final"])
    res["weak_cases"] = weak_case_checks(d, bundle, tabs)

    conv_ = lambda o: o.item() if hasattr(o, "item") else str(o)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(res, f, indent=2, default=conv_)
    make_figures(test, y, preds, chosen, tabs["final"], bundle)
    make_paper_figures(y, preds, chosen, tabs, bundle, res)
    import matplotlib, sklearn
    info = {"source": source, "runtime_seconds": round(time.time() - t0, 1),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
            "packages": {"pandas": pd.__version__, "numpy": np.__version__,
                         "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                         "joblib": joblib.__version__},
            "random_seed": RNG,
            "input_sha256": sha256(SNAPSHOT) if source == "snapshot" else
            {f"pbp_{s}.csv.gz": sha256(os.path.join(DATA_DIR, f"pbp_{s}.csv.gz")) for s in SEASONS}}
    with open(os.path.join(OUT_DIR, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"Done in {info['runtime_seconds']} s. Results in {OUT_DIR}/")


def make_figures(test, y, preds, chosen, t, bundle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, grey = "#1F4E79", "#C55A11", "#7F7F7F"

    # Figure 2: calibration (left) + score/time sensitivity example (right)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.7))
    for name, col, mk in [(chosen, blue, "o-"), ("situation_neutral_lr", "#2E7D32", "^-"),
                          ("distance_bucket", orange, "s--")]:
        p = preds[name]
        bins = np.quantile(p, np.linspace(0, 1, 9))
        idx = np.clip(np.digitize(p, bins[1:-1]), 0, 7)
        xs = [p[idx == k].mean() for k in range(8) if (idx == k).any()]
        ys = [y[idx == k].mean() for k in range(8) if (idx == k).any()]
        lbl = {chosen: chosen.replace("_", " ") + " (5 features)",
               "situation_neutral_lr": "situation-neutral LR (used by engine)",
               "distance_bucket": "distance-bucket baseline"}[name]
        ax[0].plot(np.array(xs) * 100, np.array(ys) * 100, mk, color=col, label=lbl)
    ax[0].plot([0, 100], [0, 100], ":", color=grey, label="perfect calibration")
    ax[0].set_xlim(10, 90); ax[0].set_ylim(10, 90)
    ax[0].set_xlabel("Predicted conversion probability (%)")
    ax[0].set_ylabel("Observed conversion rate (%)")
    ax[0].set_title("Calibration on held-out 2025 (n = 931, 8 equal-size bins)", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # Right: 4th & 4 at opponent 35 -- identical inputs to the rule, different game states
    secs = np.arange(3600, -1, -60)
    rows = []
    for sd in [-7, 0, 7]:
        for g in secs:
            rows.append({"ydstogo": 4, "yardline_100": 35, "score_differential": sd,
                         "qtr": min(4, int((3600 - g) // 900) + 1), "game_seconds_remaining": g,
                         "half_seconds_remaining": g % 1800 if g % 1800 else (1800 if g == 3600 else 0),
                         "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3, "is_home": 1})
    ov = option_values(pd.DataFrame(rows), bundle)
    ov["gap"] = (ov["wp_go"] - ov[["wp_field_goal", "wp_punt"]].max(axis=1)) * 100
    for k, (sd, col) in enumerate([(-7, orange), (0, grey), (7, blue)]):
        seg = ov.iloc[k * len(secs):(k + 1) * len(secs)]
        ax[1].plot((3600 - secs) / 60, seg["gap"].values, color=col,
                   label=f"score diff {sd:+d}")
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_xlabel("Game minute"); ax[1].set_ylabel("WP(go) - WP(best kick), points")
    ax[1].set_title("4th & 4 at opp. 35: above 0 = GO, below 0 = kick\n"
                    "(rule baseline says FIELD GOAL in every case)", fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig2_calibration_sensitivity.png"), dpi=200); plt.savefig(os.path.join(OUT_DIR, "fig2_calibration_sensitivity.pdf"))
    plt.close()

    # Figure 3: model and rule vs what teams did (2025)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    labs = ["go", "field_goal", "punt"]
    for a, col, title in [(ax[0], "rule", "Rule baseline"), (ax[1], "recommendation", "WP engine (final)")]:
        ct = pd.crosstab(t["decision"], t[col]).reindex(index=labs, columns=labs, fill_value=0)
        a.imshow(ct.values, cmap="Blues")
        a.set_xticks(range(3)); a.set_yticks(range(3))
        a.set_xticklabels(["Go", "FG", "Punt"]); a.set_yticklabels(["Go", "FG", "Punt"])
        a.set_xlabel(f"{title} recommends"); a.set_ylabel("What teams did")
        for i in range(3):
            for j in range(3):
                v = ct.values[i, j]
                a.text(j, i, str(v), ha="center", va="center",
                       color="white" if v > ct.values.max() * 0.55 else "black")
        a.set_title(f"{title} (n = {len(t):,})", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig3_recommend_vs_actual.png"), dpi=200); plt.savefig(os.path.join(OUT_DIR, "fig3_recommend_vs_actual.pdf"))
    plt.close()
    # Figure 4: decision chart for a neutral situation (tied, Q2, 15:00 left in half)
    yls, ytgs = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90], [1, 2, 3, 4, 5, 7, 10]
    rows = [{"ydstogo": y, "yardline_100": yl, "score_differential": 0, "qtr": 2,
             "game_seconds_remaining": 2700, "half_seconds_remaining": 900,
             "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3, "is_home": h}
            for yl in yls for y in ytgs for h in (0, 1)]
    s = pd.DataFrame(rows)
    s = pd.concat([s, option_values(s, bundle)], axis=1)
    s.loc[s["ydstogo"] > s["yardline_100"], ["wp_go", "wp_field_goal", "wp_punt"]] = np.nan
    g = s.groupby(["yardline_100", "ydstogo"])[["wp_go", "wp_field_goal", "wp_punt"]].mean()
    best = g.fillna(-1).idxmax(axis=1)
    alt = g[["wp_field_goal", "wp_punt"]].max(axis=1)
    gapv = ((g["wp_go"] - alt) * 100).unstack().reindex(index=yls, columns=ytgs)
    lab = best.unstack().reindex(index=yls, columns=ytgs)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    im = ax.imshow(gapv.values, cmap="RdBu", vmin=-4, vmax=4, aspect="auto")
    for i, yl in enumerate(yls):
        for j, y in enumerate(ytgs):
            v = gapv.values[i, j]
            if np.isnan(v):
                continue
            k = {"wp_go": "GO", "wp_field_goal": "FG", "wp_punt": "PUNT"}[lab.values[i, j]]
            ax.text(j, i, f"{k}\n{v:+.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 2.5 else "black")
    ax.set_xticks(range(len(ytgs))); ax.set_xticklabels([str(y) for y in ytgs])
    ax.set_yticks(range(len(yls)))
    ax.set_yticklabels([f"opp {y}" if y < 50 else ("50" if y == 50 else f"own {100 - y}") for y in yls])
    ax.set_xlabel("Yards to go"); ax.set_ylabel("Line of scrimmage")
    ax.set_title("Engine recommendation, tied game, Q2 15:00 (cell: choice and\n"
                 "WP(go) minus best kick, in WP points)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.04, label="go advantage (WP pts)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig4_decision_chart.png"), dpi=200); plt.savefig(os.path.join(OUT_DIR, "fig4_decision_chart.pdf"))
    plt.close()
    print("Saved figures to", OUT_DIR)


# Paper figures: vector PDF sized for the two-column ACM sigconf template (column = 3.33 in).
# Palette = validated colorblind-safe categorical hues; series also differ by marker shape.
PAPER_BLUE, PAPER_ORANGE, PAPER_AQUA, PAPER_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b6b"


def _paper_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # ACMSE allows only Linux Libertine/Biolinum: use Biolinum in figures when it is installed
    # (Debian/Ubuntu: apt-get install fonts-linuxlibertine); otherwise fall back to the default font.
    # TrueType copies in docs/fonts (converted from the OFL-licensed Linux Biolinum OpenType files)
    # embed cleanly in PDF; the system OpenType files are a fallback.
    here = os.path.dirname(os.path.abspath(__file__))
    for fp in [os.path.join(here, "docs", "fonts", f"LinBiolinum_{v}.ttf") for v in ("R", "RB", "RI")] or []:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
    if any("Biolinum" in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "Linux Biolinum O"
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.rm"] = "Linux Biolinum O"
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                         "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#444444", "grid.color": "#dddddd", "grid.linewidth": 0.5,
                         "lines.linewidth": 1.4, "pdf.fonttype": 42, "savefig.bbox": "tight",
                         "savefig.pad_inches": 0.02})
    return plt


def make_paper_figures(y, preds, chosen, tabs, bundle, res):
    plt = _paper_style()
    out = lambda n: os.path.join(OUT_DIR, n)

    # P1: calibration of conversion models on the 931 held-out plays
    fig, ax = plt.subplots(figsize=(3.33, 2.55))
    for name, col, mk, lbl in [
            ("distance_bucket", PAPER_ORANGE, "s", "Distance-bucket baseline"),
            (chosen, PAPER_BLUE, "o", "Logistic regression (5 features)"),
            ("situation_neutral_lr", PAPER_AQUA, "^", "Situation-neutral LR (engine)")]:
        p = preds[name]
        bins = np.quantile(p, np.linspace(0, 1, 9))
        idx = np.clip(np.digitize(p, bins[1:-1]), 0, 7)
        xs = [p[idx == k].mean() * 100 for k in range(8) if (idx == k).any()]
        ys = [y[idx == k].mean() * 100 for k in range(8) if (idx == k).any()]
        ax.plot(xs, ys, marker=mk, ms=4, color=col, label=lbl)
    ax.plot([15, 80], [15, 80], ":", color=PAPER_GRAY, lw=1, label="Perfect calibration")
    ax.set_xlim(15, 80); ax.set_ylim(15, 80)
    ax.set_xlabel("Predicted conversion probability (%)")
    ax.set_ylabel("Observed conversion rate (%)")
    ax.grid(True); ax.legend(loc="upper left", frameon=False)
    fig.savefig(out("paper_fig_calibration.pdf")); fig.savefig(out("paper_fig_calibration.png"), dpi=300)
    plt.close(fig)

    # P2: association gaps with game-level 95% CIs, all recommenders on the SAME plays
    bench = res.get("nfl4th_benchmark", {})
    if bench.get("available"):
        assoc = bench["association"]
        who = [("rule", "Rule baseline", PAPER_ORANGE, "s"), ("nfl4th", "nfl4th", PAPER_GRAY, "D"),
               ("engine", "Our engine", PAPER_BLUE, "o")]
        n_note = f"{bench['n_matched']:,} plays"
    else:
        assoc = res["recommendation_eval_2025"]["final"]["association"]
        who = [("rule", "Rule baseline", PAPER_ORANGE, "s"), ("engine", "Our engine", PAPER_BLUE, "o")]
        n_note = f"{res['recommendation_eval_2025']['final']['n_plays']:,} plays"
    rows = []
    for key, lab, col, mk in who:
        a_ = assoc[key]
        rows.append((f"{lab}: by group", a_["weighted_gap_pp"], *a_["weighted_gap_ci95_game"], col, mk, False))
        rows.append((f"{lab}: situation-matched", a_["matched_gap_pp"], *a_["matched_gap_ci95_game"], col, mk, True))
    fig, ax = plt.subplots(figsize=(3.33, 2.1))
    for i, (lab, m, lo, hi, col, mk, filled) in enumerate(rows[::-1]):
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt=mk, color=col, ms=4,
                    mfc=col if filled else "white", elinewidth=1.1, capsize=2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=6.5)
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_xlabel(f"WPA followed minus not followed (WP pts), {n_note}")
    ax.grid(True, axis="x")
    fig.savefig(out("paper_fig_follow_gap.pdf")); fig.savefig(out("paper_fig_follow_gap.png"), dpi=300)
    plt.close(fig)

    # P3: score/time sensitivity for one fixed situation (4th & 4 at the opponent 35), final engine
    secs = np.arange(3600, -1, -60)
    srows = []
    for sd in [-7, 0, 7]:
        for g in secs:
            srows.append({"ydstogo": 4, "yardline_100": 35, "score_differential": sd,
                          "qtr": min(4, int((3600 - g) // 900) + 1), "game_seconds_remaining": g,
                          "half_seconds_remaining": g % 1800 if g % 1800 else (1800 if g == 3600 else 0),
                          "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3, "is_home": 1})
    ov = option_values(pd.DataFrame(srows), bundle, "final")
    ov["gap"] = (ov["wp_go"] - ov[["wp_field_goal", "wp_punt"]].max(axis=1)) * 100
    fig, ax = plt.subplots(figsize=(3.33, 2.2))
    for k, (sd, col, ls, lbl) in enumerate([(-7, PAPER_ORANGE, "-", "Trailing by 7"),
                                            (0, PAPER_GRAY, "--", "Tied"),
                                            (7, PAPER_BLUE, "-.", "Leading by 7")]):
        seg = ov.iloc[k * len(secs):(k + 1) * len(secs)]
        ax.plot((3600 - secs) / 60, seg["gap"].values, color=col, ls=ls, label=lbl)
    ax.axhline(0, color="#222222", lw=0.8)
    ax.text(61, 0.3, "GO\nbetter", fontsize=6, color="#444444", va="bottom", clip_on=False)
    ax.text(61, -0.3, "kick\nbetter", fontsize=6, color="#444444", va="top", clip_on=False)
    ax.set_xlim(0, 60); ax.set_xticks([0, 15, 30, 45, 60])
    ax.set_xlabel("Game minute")
    ax.set_ylabel("WP(go) - WP(best kick), pts")
    ax.grid(True); ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=6)
    ax.set_ylim(min(-4, ov["gap"].min() - 0.5), ov["gap"].max() + 2.2)
    fig.savefig(out("paper_fig_sensitivity.pdf")); fig.savefig(out("paper_fig_sensitivity.png"), dpi=300)
    plt.close(fig)

    # P4: decision chart, tied game, Q2 15:00, checkpoint engine (top) vs final engine (bottom)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.3), sharex=True)
    for ax, eng, title in [(axes[0], "checkpoint", "(a) Checkpoint engine"),
                           (axes[1], "final", "(b) Final engine (WP v2 + sparse-region conversion fix)")]:
        yls, ytgs, best, gapv = decision_grid(bundle, eng)
        G, B = gapv.values.T[:, ::-1], best.values.T[:, ::-1]
        xl = [f"own {100 - v}" if v > 50 else ("50" if v == 50 else f"opp {v}") for v in yls[::-1]]
        im = ax.imshow(G, cmap="RdBu", vmin=-4, vmax=4, aspect="auto")
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                v = G[i, j]
                if np.isnan(v):
                    continue
                k = {"wp_go": "GO", "wp_field_goal": "FG", "wp_punt": "PUNT"}[B[i, j]]
                ax.text(j, i, f"{k} {v:+.1f}", ha="center", va="center", fontsize=5.8,
                        color="white" if abs(v) > 2.6 else "#111111")
        ax.set_yticks(range(len(ytgs))); ax.set_yticklabels([str(v) for v in ytgs])
        ax.set_ylabel("Yards to go"); ax.set_title(title, fontsize=7.5, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[1].set_xticks(range(len(xl))); axes[1].set_xticklabels(xl)
    axes[1].set_xlabel("Line of scrimmage")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("GO advantage over best kick (WP pts)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    fig.savefig(out("paper_fig_decision_chart.pdf")); fig.savefig(out("paper_fig_decision_chart.png"), dpi=300)
    plt.close(fig)
    print("Saved paper figures to", OUT_DIR)


# ==================================================================================
# Predict: one situation -> recommendation
# ==================================================================================
def predict(a):
    path = os.path.join(OUT_DIR, "models.joblib")
    if not os.path.exists(path):
        raise SystemExit("Run `python fourth_down.py build` first.")
    bundle = joblib.load(path)
    half = a.seconds_left - 1800 if a.qtr <= 2 else a.seconds_left
    s = pd.DataFrame([{"ydstogo": a.ydstogo, "yardline_100": a.yardline_100,
                       "score_differential": a.score_diff, "qtr": a.qtr,
                       "game_seconds_remaining": a.seconds_left, "half_seconds_remaining": max(half, 0),
                       "posteam_timeouts_remaining": a.timeouts, "defteam_timeouts_remaining": a.opp_timeouts,
                       "is_home": int(a.home)}])
    o = option_values(s, bundle, "final").iloc[0]
    rule = rule_based_decision(a.ydstogo, a.yardline_100)
    fmt = lambda v: "n/a (out of range)" if pd.isna(v) else f"{v:.1%}"
    print("\n=== Fourth-Down Situation ===")
    print(f"4th & {a.ydstogo} | {a.yardline_100} yds from goal | score diff {a.score_diff:+d} | "
          f"Q{a.qtr} | {a.seconds_left}s left in game | timeouts {a.timeouts}-{a.opp_timeouts}")
    full_p = bundle["conv_models"]["full"].predict_proba(s[CONV_FEATURES])[0, 1]
    print(f"\nConversion probability (situation-neutral model used by engine): {o.p_convert:.1%}")
    print(f"  (5-feature {bundle['conv_name'].replace('_', ' ')} for comparison: {full_p:.1%})")
    print(f"FG make probability ({a.yardline_100 + 18}-yd attempt): {o.p_fg_make:.1%}")
    print("\nExpected win probability by choice:")
    for k in OPTIONS:
        star = "  <-- recommended" if k == o.recommendation else ""
        print(f"  {k.replace('_', ' ').upper():<11} {fmt(o['wp_' + k])}{star}")
    strength = "toss-up" if o.margin < LEAN else ("lean" if o.margin < CLEAR else "clear")
    print(f"\nRecommendation: {o.recommendation.replace('_', ' ').upper()} "
          f"(margin {o.margin * 100:.1f} WP points, {strength})")
    print(f"Rule baseline (yards to go + field position only): {rule.replace('_', ' ').upper()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", choices=["snapshot", "nflverse"], default="snapshot")
    sub.add_parser("snapshot")
    pr = sub.add_parser("predict")
    pr.add_argument("--ydstogo", type=int, required=True)
    pr.add_argument("--yardline_100", type=int, required=True, help="yards from opponent goal line")
    pr.add_argument("--score_diff", type=int, default=0, help="offense score minus defense score")
    pr.add_argument("--qtr", type=int, default=2)
    pr.add_argument("--seconds_left", type=int, default=1800, help="seconds left in the GAME")
    pr.add_argument("--timeouts", type=int, default=3)
    pr.add_argument("--opp_timeouts", type=int, default=3)
    pr.add_argument("--home", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.source)
    elif a.cmd == "snapshot":
        make_snapshot()
    else:
        predict(a)
