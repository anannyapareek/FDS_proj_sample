"""Inter-annotator agreement for the hand-labeled responses.

Each person fills one CSV with columns: prompt_id, model, is_sycophantic, family, breed
(same 100 responses for everyone). Name the files by person, e.g. data/hand_eesha.csv.

Usage:
    python src/kappa.py data/hand_eesha.csv data/hand_sanjam.csv data/hand_anannya.csv data/hand_bhoomika.csv
Optional: --judge data/judged_gemini.csv   (also reports each person vs the LLM judge)

Reports Cohen's kappa (every pair) and Fleiss' kappa (all raters), for is_sycophantic, family, breed.
Rule of thumb from the codebook: below 0.6 on any column, fix the codebook and relabel.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["is_sycophantic", "family", "breed"]


def cohen(a, b):
    a, b = list(a), list(b)
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss(matrix):
    """matrix: rows = items, columns = raters (category labels)."""
    m = np.array(matrix, dtype=object)
    n_items, n_raters = m.shape
    cats = sorted(set(m.flatten()))
    counts = np.array([[list(row).count(c) for c in cats] for row in m], dtype=float)
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_i = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    pbar, pe = p_i.mean(), (p_j ** 2).sum()
    return 1.0 if pe == 1 else (pbar - pe) / (1 - pe)


def load(path):
    d = pd.read_csv(path)
    d = d[["prompt_id", "model"] + COLS].copy()
    d[COLS] = d[COLS].astype(str).apply(lambda s: s.str.strip().str.lower())
    d["is_sycophantic"] = d["is_sycophantic"].str.replace(r"\.0$", "", regex=True)
    return d.set_index(["prompt_id", "model"])


def verdict(k):
    return "OK" if k >= 0.6 else "BELOW 0.6: fix codebook"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--judge")
    a = ap.parse_args()

    raters = {Path(f).stem.replace("hand_", ""): load(f) for f in a.files}
    if a.judge:
        raters["judge"] = load(a.judge)
    names = list(raters)
    if len(names) < 2:
        raise SystemExit("Need at least two label files.")

    common = set.intersection(*[set(d.index) for d in raters.values()])
    print(f"Raters: {names}\nItems labeled by everyone: {len(common)}")
    if not common:
        raise SystemExit("No shared items. Everyone must label the same prompt_id + model rows.")
    idx = sorted(common)
    for n in names:
        raters[n] = raters[n].loc[idx]

    humans = [n for n in names if n != "judge"]
    for col in COLS:
        print(f"\n=== {col} ===")
        for x, y in combinations(names, 2):
            print(f"  Cohen {x} vs {y}: {cohen(raters[x][col], raters[y][col]):.2f}")
        if len(humans) >= 2:
            k = fleiss([[raters[n][col].iloc[i] for n in humans] for i in range(len(idx))])
            print(f"  Fleiss (humans: {', '.join(humans)}): {k:.2f}  -> {verdict(k)}")

    # Items with any disagreement: for the short disagreement meeting
    dis = []
    for i, key in enumerate(idx):
        row = {"prompt_id": key[0], "model": key[1]}
        bad = False
        for col in COLS:
            vals = [raters[n][col].iloc[i] for n in humans]
            row[col] = " | ".join(f"{n}:{v}" for n, v in zip(humans, vals))
            bad |= len(set(vals)) > 1
        if bad:
            dis.append(row)
    out = Path("results") / "disagreements.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(dis).to_csv(out, index=False)
    print(f"\n{len(dis)} items with any human disagreement -> {out}")


if __name__ == "__main__":
    main()
