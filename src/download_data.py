"""Download the raw datasets into data/raw/.

Usage (from the repo root):
    python src/download_data.py

Anthropic evals and meg-tong sycophancy-eval come straight from GitHub.
EleutherAI/sycophancy comes from HuggingFace through the `datasets` library.
Beacon is NOT downloaded here: it is held out for the final test.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

GITHUB_FILES = {
    "anthropic": {
        "base": "https://raw.githubusercontent.com/anthropics/evals/main/sycophancy/",
        "files": [
            "sycophancy_on_nlp_survey.jsonl",
            "sycophancy_on_philpapers2020.jsonl",
            "sycophancy_on_political_typology_quiz.jsonl",
        ],
    },
    "meg_tong": {
        "base": "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/",
        "files": ["answer.jsonl", "are_you_sure.jsonl", "feedback.jsonl"],
    },
}


def fetch(url: str, dest: Path, tries: int = 3) -> None:
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            return
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt}/{tries} failed: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"could not download {url}")


def download_github() -> None:
    for name, spec in GITHUB_FILES.items():
        out = RAW / name
        out.mkdir(parents=True, exist_ok=True)
        for fname in spec["files"]:
            dest = out / fname
            if dest.exists() and dest.stat().st_size > 0:
                print(f"skip   {dest.relative_to(RAW.parent.parent)} (already there)")
                continue
            print(f"fetch  {spec['base'] + fname}")
            fetch(spec["base"] + fname, dest)
            n = sum(1 for _ in open(dest, encoding="utf-8"))
            print(f"  saved {dest.name}: {n} rows")


def download_eleuther() -> None:
    out = RAW / "eleuther"
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.jsonl")):
        print("skip   data/raw/eleuther (already there)")
        return
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError:
        print("EleutherAI skipped: run `pip install datasets` first.")
        return
    try:
        configs = get_dataset_config_names("EleutherAI/sycophancy")
        print(f"EleutherAI/sycophancy configs: {configs}")
        for cfg in configs:
            ds = load_dataset("EleutherAI/sycophancy", cfg)
            for split, d in ds.items():
                dest = out / f"{cfg}__{split}.jsonl"
                with open(dest, "w", encoding="utf-8") as f:
                    for row in d:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  saved {dest.name}: {len(d)} rows, columns={d.column_names}")
    except Exception as e:  # noqa: BLE001
        print(f"EleutherAI download failed ({e}).")
        print("The pipeline still works without it; prepare_prompts.py skips it.")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    download_github()
    download_eleuther()
    print("done.")
    sys.exit(0)
