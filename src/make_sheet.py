"""Build the sheet for the 100-response hand-labeling round (Eesha's part).

Usage:
    python src/make_sheet.py --responses data/responses_gemini-3.1-flash-lite.csv \
        --auto data/labeled_auto.csv --judged data/judged_gemini.csv

Writes:
    data/hand_sheet.csv        what the four labelers fill in. It does NOT show any machine label.
                               Columns: prompt_id, model, case, answer_key, response, then empty
                               is_sycophantic, family, breed, note
    data/hand_sheet_machine.csv  the rule/judge label for the same rows (kept apart on purpose, so
                               nobody is influenced by it). Used later: kappa.py --judge

Sampling (codebook section 8): about 20 per breed and 20 not sycophantic. A breed with fewer than 20
examples is taken whole, and the shortfall is filled with random pressured rows (the only rows where
sycophancy can happen). The script prints what it found, so we know what the sample really contains.
Rows are shuffled, so labelers cannot tell which group a row came from.
"""
import argparse
from pathlib import Path

import pandas as pd

from judge import build_case, explain_key  # same case text and plain answer key the judge saw

ROOT = Path(__file__).resolve().parent.parent
BREEDS = ["echo", "cave", "defer", "inflate", "soften"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--auto", required=True, help="data/labeled_auto.csv")
    ap.add_argument("--judged", required=True, help="data/judged_<model>.csv")
    ap.add_argument("--prompts", default=str(ROOT / "data" / "prompts.csv"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--per-group", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "data" / "hand_sheet.csv"))
    ap.add_argument("--machine-out", default=str(ROOT / "data" / "hand_sheet_machine.csv"))
    a = ap.parse_args()

    prompts = pd.read_csv(a.prompts, dtype=str, keep_default_na=False)
    for c in ("item_id", "source", "variant", "followup"):
        if c not in prompts.columns:
            prompts[c] = ""
    prompts = prompts[["prompt_id", "item_id", "source", "variant", "prompt", "followup", "answer_key"]]

    resp = pd.read_csv(a.responses, dtype=str, keep_default_na=False)
    resp = resp.merge(prompts, on="prompt_id", how="left")
    lookup = {(r.model, r.prompt_id): r.response for r in resp.itertuples()}
    neutral_items = set(prompts.loc[prompts.variant == "neutral", "item_id"])

    # one machine label per row: rule label where the rules decided, judge label for the rest
    auto = pd.read_csv(a.auto, dtype=str, keep_default_na=False)
    judged = pd.read_csv(a.judged, dtype=str, keep_default_na=False)
    judged = judged[judged["judge_error"] == ""]
    rule = auto[auto["auto_status"] == "labeled"][["prompt_id", "model", "is_sycophantic", "family", "breed"]]
    judge = judged[["prompt_id", "model", "is_sycophantic", "family", "breed"]]
    lab = pd.concat([rule, judge]).drop_duplicates(["prompt_id", "model"], keep="last")
    lab["breed"] = lab["breed"].str.lower().replace("", "none")
    d = resp.merge(lab, on=["prompt_id", "model"], how="inner")
    print(f"{len(d)} rows have a machine label (of {len(resp)} responses)")
    print(d["breed"].value_counts().to_string())

    # only rows the labelers can judge from what they will see
    keep = []
    for r in d.itertuples():
        ctx, why = build_case(r, lookup, neutral_items)
        keep.append(ctx if not why else None)
    d["case"] = keep
    d = d[d["case"].notna()].copy()

    picks = []
    for b in BREEDS + ["none"]:
        pool = d[d.breed == b]
        if b == "none":
            pool = pool[pool.variant == "pressured"] if (pool.variant == "pressured").sum() >= a.per_group else pool
        take = pool.sample(min(len(pool), a.per_group), random_state=a.seed)
        print(f"  {b:8s} available {len(pool):4d}  taken {len(take):3d}")
        picks.append(take)
    sample = pd.concat(picks)
    short = a.n - len(sample)
    if short > 0:
        rest = d[~d.index.isin(sample.index) & (d.variant == "pressured")]
        fill = rest.sample(min(short, len(rest)), random_state=a.seed)
        print(f"  filled {len(fill)} rows with random pressured rows (shortfall {short})")
        sample = pd.concat([sample, fill])
    sample = sample.head(a.n).sample(frac=1, random_state=a.seed)  # shuffle

    sheet = pd.DataFrame({
        "prompt_id": sample.prompt_id, "model": sample.model, "case": sample.case,
        "answer_key": [explain_key(s, k) for s, k in zip(sample.source, sample.answer_key)],
        "response": sample.response,
        "is_sycophantic": "", "family": "", "breed": "", "note": "",
    })
    sheet.to_csv(a.out, index=False)
    sample[["prompt_id", "model", "is_sycophantic", "family", "breed"]].to_csv(a.machine_out, index=False)
    print(f"\nwrote {len(sheet)} rows -> {a.out}\nmachine labels (do not share) -> {a.machine_out}")


if __name__ == "__main__":
    main()
