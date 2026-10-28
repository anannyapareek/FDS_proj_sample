"""Merge contributor custom-prompt files into data/custom_prompts.csv.

Usage (from the repo root, before src/prepare_prompts.py):
    python src/merge_custom.py

Reads data/prompts_eesha_all50.csv and data/prompts_bhoomika_all50.csv
(prompt_id, source, variant, prompt, answer_key) and writes
data/custom_prompts.csv in the format prepare_prompts.py's build_custom()
expects (item_id, variant, prompt, source, followup, answer_key).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "custom_prompts.csv"

SOURCES = {
    DATA / "prompts_eesha_all50.csv": "custom_eesha",
    DATA / "prompts_bhoomika_all50.csv": "custom_bhoomika",
}

COLUMNS = ["item_id", "variant", "prompt", "source", "followup", "answer_key"]


def main():
    frames = []
    for path, source in SOURCES.items():
        if not path.exists():
            raise SystemExit(f"missing input file: {path}")
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = {"prompt_id", "variant", "prompt"} - set(df.columns)
        if missing:
            raise SystemExit(f"{path.name} is missing columns: {sorted(missing)}")
        df = df.rename(columns={"prompt_id": "item_id"})
        df["source"] = source
        df["followup"] = ""
        if "answer_key" not in df.columns:
            df["answer_key"] = ""
        frames.append(df[COLUMNS])

    merged = pd.concat(frames, ignore_index=True)
    if merged["item_id"].duplicated().any():
        dups = merged[merged["item_id"].duplicated()]["item_id"].tolist()
        raise SystemExit(f"duplicate item_id across custom files: {dups}")

    merged.to_csv(OUT, index=False)
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(merged)} rows")
    print(merged.groupby(["source", "variant"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
