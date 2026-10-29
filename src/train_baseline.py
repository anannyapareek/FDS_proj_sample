"""Quick baseline detector: TF-IDF + logistic regression on response text only (black-box).

Usage (from the repo root, after label_auto.py):
    python src/train_baseline.py                     # is it sycophantic? (0/1 score)
    python src/train_baseline.py --target family     # agreement / evaluation / none
    python src/train_baseline.py --target breed
    python src/train_baseline.py --labels data/labeled.csv   # Eesha's final labels

Split is by item_id, NOT prompt_id: the neutral and pressured versions of one item
share the same question, so splitting them across train and test would leak.
Rows with auto_status other than "labeled" are skipped (if that column exists).
The first answer letter of opinion prompts "(A)" is removed from the text so the model
cannot just read the label off the letter.

Output: printed report, results/baseline_<target>.txt, results/baseline_<target>.joblib (gitignored)
"""
import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def clean_text(s: str) -> str:
    return re.sub(r"^\s*[\(\[]?[A-Z][\)\]]?(?=[\s.:,)\-]|$)", "", str(s)).strip()


def make_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "data" / "labeled_auto.csv"))
    ap.add_argument("--prompts", default=str(ROOT / "data" / "prompts.csv"))
    ap.add_argument("--target", default="is_sycophantic",
                    choices=["is_sycophantic", "family", "breed"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-neutral", action="store_true",
                    help="drop neutral rows (they are 0 by construction and make the task easy)")
    ap.add_argument("--loso", action="store_true",
                    help="also run leave-one-source-out (train without one source, test on it)")
    args = ap.parse_args()

    lab = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    if "auto_status" in lab.columns:
        lab = lab[lab.auto_status == "labeled"]
    lab = lab[lab[args.target] != ""].copy()
    if args.target == "is_sycophantic":
        lab[args.target] = lab[args.target].astype(float).astype(int).astype(str)
    prompts = pd.read_csv(args.prompts, dtype=str, keep_default_na=False)[["prompt_id", "item_id", "source", "variant"]]
    df = lab.merge(prompts, on="prompt_id", how="inner").reset_index(drop=True)
    if args.exclude_neutral:
        df = df[df.variant != "neutral"].reset_index(drop=True)
        print("excluded neutral rows")
    if len(df) < 50:
        sys.exit(f"only {len(df)} labeled rows: collect more responses first")

    df["text"] = df["response"].map(clean_text)
    y = df[args.target].astype(str).values
    groups = df["item_id"].values
    print(f"{len(df)} labeled rows, {df.item_id.nunique()} items, target={args.target}")
    print("class counts:", dict(pd.Series(y).value_counts()))
    if len(set(y)) < 2:
        sys.exit("only one class present: need more variety (collect more rows)")

    lines = []
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed).split(df, y, groups))
    model = make_model().fit(df.text.iloc[tr], y[tr])
    pred = model.predict(df.text.iloc[te])
    rep = classification_report(y[te], pred, digits=3, zero_division=0)
    cm = confusion_matrix(y[te], pred, labels=sorted(set(y)))
    lines += [f"Held-out split (20% of items): {len(te)} rows", rep,
              f"confusion matrix (rows=true, cols=pred, order={sorted(set(y))}):", str(cm)]

    if args.target == "is_sycophantic":
        proba = model.predict_proba(df.text.iloc[te])[:, list(model.classes_).index("1")]
        lines.append(f"mean sycophancy score: sycophantic rows {proba[y[te] == '1'].mean():.2f}, "
                     f"other rows {proba[y[te] != '1'].mean():.2f}")

    # accuracy per source on the held-out split: shows whether it only learned the source
    te_df = df.iloc[te].assign(pred=pred, true=y[te])
    lines.append("\nheld-out accuracy by source:")
    for src, g in te_df.groupby("source"):
        lines.append(f"  {src:36s} n={len(g):3d}  acc={(g.pred == g.true).mean():.2f}  "
                     f"share_positive={(g.true != '0').mean() if args.target == 'is_sycophantic' else float('nan'):.2f}")

    if args.loso:
        lines.append("\nleave-one-source-out (train on all other sources, test on this one):")
        for src in sorted(df.source.unique()):
            trn, tst = df[df.source != src], df[df.source == src]
            if len(set(y[trn.index])) < 2 or len(tst) < 10:
                continue
            m = make_model().fit(trn.text, y[trn.index])
            pr = m.predict(tst.text)
            lines.append(f"  {src:36s} n={len(tst):3d}  acc={(pr == y[tst.index]).mean():.2f}  "
                         f"macro-F1={f1_score(y[tst.index], pr, average='macro', zero_division=0):.2f}")

    k = min(5, df.item_id.nunique())
    f1s = []
    for a, b in GroupKFold(n_splits=k).split(df, y, groups):
        if len(set(y[a])) < 2:
            continue
        m = make_model().fit(df.text.iloc[a], y[a])
        f1s.append(f1_score(y[b], m.predict(df.text.iloc[b]), average="macro", zero_division=0))
    lines.append(f"{k}-fold grouped CV macro-F1: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")

    final = make_model().fit(df.text, y)
    vec, clf = final.steps[0][1], final.steps[1][1]
    names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_
    lines.append("\nstrongest words per class:")
    if coefs.shape[0] == 1:
        lines.append(f"  -> {clf.classes_[1]}: {', '.join(names[np.argsort(coefs[0])[-15:][::-1]])}")
        lines.append(f"  -> {clf.classes_[0]}: {', '.join(names[np.argsort(coefs[0])[:15]])}")
    else:
        for i, c in enumerate(clf.classes_):
            lines.append(f"  -> {c}: {', '.join(names[np.argsort(coefs[i])[-15:][::-1]])}")

    report = "\n".join(lines)
    print("\n" + report)
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    (res / f"baseline_{args.target}.txt").write_text(report, encoding="utf-8")
    joblib.dump(final, res / f"baseline_{args.target}.joblib")
    print(f"\nsaved results/baseline_{args.target}.txt and .joblib")


if __name__ == "__main__":
    main()
