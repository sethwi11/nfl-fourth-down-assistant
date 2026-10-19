"""
Choose the win-probability (WP) model settings using 2021-24 data only (2025 is never touched).

  python tune_wp.py        # about 5-8 minutes on 2 CPUs

Each configuration is scored by 5-fold cross-validation GROUPED BY GAME (a game's plays never sit
in both the training and the validation fold). Printed per configuration:
  all        Brier score on every play (lower = better)
  last5min   Brier on plays with <= 5:00 left in the game (weak case 2: late deficits)
  redzone    Brier on plays within 20 yards of the goal line (weak case 1: goal-line flatness)
  10s jump   change in WP between 15:00 and 14:50 of Q2 at own 30, tied (memorized-noise check)

Configurations
  cp2_first  first Checkpoint 2 attempt (too flexible; WP jumps ~4 points in 10 seconds)
  v1         Checkpoint 2 model (WP_FEATURES, strong regularization)
  v2a        v1 + goal-line and late-game features (fourth_down.WP_FEATURES_V2)
  v2b        v2a with 300 boosting iterations          <- selected as the final WP model
  v2c        v2a with min_samples_leaf 500
Selection rule (set before looking at 2025): lowest overall grouped-CV Brier; ties broken by the
last-5-minutes Brier. v1 -> v2b: all 0.1662 -> 0.1661, last5min 0.0837 -> 0.0828.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold

import fourth_down as f

raw = f.load_raw("snapshot")
p, y = f.wp_training_frame(raw, f.TRAIN_SEASONS)
cfgs = {
    "cp2_first": (f.WP_FEATURES, [1, 1, 0, 0, -1, 0, 0, 0, 0, 0],
                  dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=200)),
    "v1": (f.WP_FEATURES, f.WP_V1_CST, f.WP_V1_HP),
    "v2a": (f.WP_FEATURES_V2, f.WP_V2_CST, f.WP_V1_HP),
    "v2b": (f.WP_FEATURES_V2, f.WP_V2_CST, f.WP_V2_HP),
    "v2c": (f.WP_FEATURES_V2, f.WP_V2_CST, dict(f.WP_V1_HP, min_samples_leaf=500)),
}
late = (p["game_seconds_remaining"] <= 300).values
rz = (p["yardline_100"] <= 20).values
print(f"{len(p):,} plays, {p['game_id'].nunique():,} games, 5-fold grouped CV")
for name, (feats, cst, hp) in cfgs.items():
    pred = np.zeros(len(p))
    for a, b in GroupKFold(5).split(p, y, p["game_id"]):
        m = HistGradientBoostingClassifier(monotonic_cst=cst, random_state=f.RNG, **hp)
        m.fit(p[feats].iloc[a], y[a])
        pred[b] = m.predict_proba(p[feats].iloc[b])[:, 1]
    full = HistGradientBoostingClassifier(monotonic_cst=cst, random_state=f.RNG, **hp).fit(p[feats], y)
    full.feature_list_ = feats
    j = f._wp(full, [f._state(0, 2700, 900, 70, 1, 10, 3, 3, 0), f._state(0, 2690, 890, 70, 1, 10, 3, 3, 0)])
    print(f"{name:10s} all {brier_score_loss(y, pred):.4f}  last5min {brier_score_loss(y[late], pred[late]):.4f}  "
          f"redzone {brier_score_loss(y[rz], pred[rz]):.4f}  10s jump {abs(j[1] - j[0]) * 100:.1f} pts", flush=True)
