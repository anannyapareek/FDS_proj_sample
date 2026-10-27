"""Send prompts.csv to one model and save its responses. Works for Gemini and Groq.

Examples (run from the repo root):
    python src/collect.py --provider gemini --model <gemini-model-name> --rpm 10
    python src/collect.py --provider groq   --model <groq-model-name>   --rpm 25
    python src/collect.py --provider mock   --model mock-a --limit 20      # no key needed

Output: data/responses_<model>.csv with columns prompt_id, model, response.
Safe to stop and restart: rows already saved are skipped, and every call is saved at once.
Failed calls are printed and NOT saved, so the next run retries them.

Rows with a `followup` (are-you-sure pressure) are sent as 3 turns:
    user: prompt  ->  assistant: THIS model's own neutral answer  ->  user: followup
so the neutral row for the same item must be collected first (the file order guarantees it).

Keys come from .env (GEMINI_API_KEY, GROQ_API_KEY). Model names change often:
check the model list in your console and pass the name with --model.
"""
import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["prompt_id", "model", "response"]


# ----------------------------------------------------------------- providers
def make_caller(provider: str, model: str):
    """Return fn(messages) -> str. messages = [{'role': 'user'|'assistant', 'content': str}]."""
    if provider == "mock":
        def call(messages):
            last = messages[-1]["content"]
            if len(messages) > 1:
                return "You're right, I apologise. On reflection the answer is different."
            return f"(A) Great point, I agree. [{model}] {last[:40]}"
        return call

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            sys.exit("GEMINI_API_KEY missing. Put it in .env (see .env.example).")
        from google import genai  # pip install google-genai
        client = genai.Client(api_key=key)

        def call(messages):
            contents = [
                {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                for m in messages
            ]
            r = client.models.generate_content(model=model, contents=contents)
            if not r.text:
                raise RuntimeError("empty response (possibly blocked by a safety filter)")
            return r.text

        return call

    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            sys.exit("GROQ_API_KEY missing. Put it in .env (see .env.example).")
        from groq import Groq
        client = Groq(api_key=key)

        def call(messages):
            r = client.chat.completions.create(model=model, messages=messages)
            text = r.choices[0].message.content
            if not text:
                raise RuntimeError("empty response")
            return text

        return call

    sys.exit(f"unknown provider {provider}")


def with_retries(call, messages, tries=5):
    """Retry with growing waits (rate limits are the usual failure on free tiers)."""
    wait = 10
    for attempt in range(1, tries + 1):
        try:
            return call(messages)
        except Exception as e:  # noqa: BLE001
            if attempt == tries:
                raise
            print(f"    attempt {attempt}/{tries} failed ({str(e)[:120]}); waiting {wait}s")
            time.sleep(wait)
            wait = min(wait * 2, 120)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True, choices=["gemini", "groq", "mock"])
    ap.add_argument("--model", required=True, help="exact model name from your console")
    ap.add_argument("--prompts", default=str(ROOT / "data" / "prompts.csv"))
    ap.add_argument("--out", default=None, help="default: data/responses_<model>.csv")
    ap.add_argument("--rpm", type=float, default=10, help="max requests per minute (stay under your limit)")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many NEW calls (for testing)")
    ap.add_argument("--sources", default=None, help="comma-separated source names to include")
    args = ap.parse_args()

    df = pd.read_csv(args.prompts, dtype=str, keep_default_na=False)
    if args.sources:
        want = {s.strip() for s in args.sources.split(",")}
        df = df[df.source.isin(want)]
        if df.empty:
            sys.exit(f"no rows for sources {want}")

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    out = Path(args.out) if args.out else ROOT / "data" / f"responses_{safe}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, str] = {}
    if out.exists() and out.stat().st_size > 0:
        prev = pd.read_csv(out, dtype=str, keep_default_na=False)
        done = dict(zip(prev.prompt_id, prev.response))
    print(f"{len(df)} prompts selected, {len(done)} already saved in {out.name}")

    call = make_caller(args.provider, args.model)
    gap = 60.0 / args.rpm if args.rpm > 0 else 0
    new_file = not out.exists() or out.stat().st_size == 0
    n_new = n_fail = n_skip = 0

    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
            f.flush()
        for row in df.itertuples(index=False):
            if row.prompt_id in done:
                continue
            if args.limit is not None and n_new >= args.limit:
                break

            messages = [{"role": "user", "content": row.prompt}]
            if row.followup:
                neutral = done.get(f"{row.item_id}:neutral")
                if neutral is None:
                    print(f"  skip {row.prompt_id}: no saved neutral answer to build the follow-up on")
                    n_skip += 1
                    continue
                messages += [{"role": "assistant", "content": neutral},
                             {"role": "user", "content": row.followup}]

            t0 = time.time()
            try:
                text = with_retries(call, messages).strip()
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED {row.prompt_id}: {str(e)[:150]}")
                n_fail += 1
                continue

            w.writerow({"prompt_id": row.prompt_id, "model": args.model, "response": text})
            f.flush()
            done[row.prompt_id] = text
            n_new += 1
            if n_new % 25 == 0:
                print(f"  {n_new} new this run, {len(done)} total")
            time.sleep(max(0.0, gap - (time.time() - t0)))

    print(f"done: {n_new} new, {n_fail} failed, {n_skip} skipped, {len(done)} saved in total -> {out}")
    if n_fail or n_skip:
        print("Run the same command again to retry the failed/skipped rows.")


if __name__ == "__main__":
    main()
