"""
Check that a rebuild reproduces the reported results.

  python verify_results.py

Compares results/results.json (from `python fourth_down.py build`) against
expected/expected_results.json, the reference run used in the paper. Every number is compared
(including nested tables). Also checks the SHA-256 of the data snapshot.
Exit code 0 = everything matches within tolerance.
"""
import hashlib
import json
import math
import os
import sys

TOL = 1e-6          # results are rounded to 2-4 decimals, so a correct rebuild matches exactly
SNAPSHOT = os.path.join("data", "snapshot", "nflverse_pbp_2021_2025_subset.csv.gz")
SUMS = os.path.join("data", "snapshot", "SHA256SUMS.json")


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


def main():
    ok = True
    if os.path.exists(SNAPSHOT) and os.path.exists(SUMS):
        h = hashlib.sha256(open(SNAPSHOT, "rb").read()).hexdigest()
        want = json.load(open(SUMS))["sha256"][os.path.basename(SNAPSHOT)]
        print(f"[{'OK' if h == want else 'MISMATCH'}] data snapshot SHA-256")
        ok &= h == want
    exp = flatten(json.load(open(os.path.join("expected", "expected_results.json"))))
    got = flatten(json.load(open(os.path.join("results", "results.json"))))
    bad = []
    for k, v in exp.items():
        g = got.get(k, "<missing>")
        if isinstance(v, (int, float)) and isinstance(g, (int, float)):
            if not math.isclose(v, g, rel_tol=0, abs_tol=TOL):
                bad.append((k, v, g))
        elif v != g:
            bad.append((k, v, g))
    print(f"[{'OK' if not bad else 'MISMATCH'}] {len(exp) - len(bad)}/{len(exp)} reported values reproduced")
    for k, v, g in bad[:40]:
        print(f"    {k}: expected {v}, got {g}")
    ok &= not bad
    print("ALL RESULTS REPRODUCED" if ok else "SOME RESULTS DIFFER (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
