"""Evaluation-family text features (Eesha's part). Anannya merges these into features.py.

Usage in code:
    from eval_features import eval_features
    eval_features("Great work! One tiny thing you might consider...")  -> dict of numbers

Command line (adds the columns to a responses file):
    python src/eval_features.py data/labeled.csv data/eval_features.csv

All counts are per 100 words, so long and short responses are comparable.
Lexicons are small on purpose, easy to explain in the report, and easy to extend.
"""
import re
import sys

SUPERLATIVES = {
    "excellent", "brilliant", "perfect", "outstanding", "amazing", "fantastic", "wonderful",
    "superb", "flawless", "exceptional", "incredible", "masterpiece", "stellar", "phenomenal",
    "best", "great", "awesome", "terrific", "magnificent", "impressive", "genius", "remarkable",
}
POSITIVE = SUPERLATIVES | {
    "good", "nice", "strong", "solid", "love", "clear", "well", "effective", "compelling",
    "polished", "thoughtful", "creative", "engaging", "smart", "right", "beautiful", "ready",
}
NEGATIVE = {
    "problem", "flaw", "issue", "mistake", "wrong", "weak", "unclear", "vague", "missing",
    "lacks", "lack", "error", "bug", "fails", "risk", "risky", "unrealistic", "incorrect",
    "confusing", "cliched", "generic", "poor", "doesn't", "won't", "cannot", "unprofessional",
    "no", "not", "without", "never", "needs", "need", "fix", "fixing", "should", "must", "unfortunately",
}
HEDGES = {
    "might", "maybe", "perhaps", "possibly", "could", "somewhat", "slightly", "a bit", "kind of",
    "sort of", "arguably", "potentially", "seems", "appears", "just", "little", "tiny", "small",
}
MINIMIZERS = {"minor", "tiny", "small", "slight", "little", "just a", "only a", "nitpick", "nothing major"}
CRITICISM_CUES = {"however", "but", "although", "that said", "one thing", "consider", "could be improved",
                  "suggest", "improve", "weakness", "issue", "problem", "flaw"}
PRAISE_OPENERS = {"great", "excellent", "wonderful", "love", "fantastic", "amazing", "brilliant",
                  "congratulations", "well done", "good job", "nice", "impressive", "wow"}
INTENSIFIERS = {"really", "very", "truly", "absolutely", "incredibly", "extremely", "so", "totally"}


def _words(text):
    return re.findall(r"[a-z']+", text.lower())


def _count(text, vocab):
    """Count whole words and phrases from vocab in text."""
    t = " " + " ".join(_words(text)) + " "
    return sum(len(re.findall(rf"(?<![a-z']){re.escape(v)}(?![a-z'])", t)) for v in vocab)


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def eval_features(text):
    text = "" if text is None else str(text)
    w = _words(text)
    n = max(len(w), 1)
    per100 = lambda c: 100.0 * c / n

    sents = _sentences(text)
    # Sentences that contain a criticism cue or a negative word
    crit = [s for s in sents if _count(s, CRITICISM_CUES) or _count(s, NEGATIVE)]
    crit_text = " ".join(crit)
    crit_words = max(len(_words(crit_text)), 1)

    pos, neg = _count(text, POSITIVE), _count(text, NEGATIVE)
    first = sents[0] if sents else ""
    last = sents[-1] if sents else ""

    return {
        "n_words": len(w),
        "superlatives": per100(_count(text, SUPERLATIVES)),
        "positive_words": per100(pos),
        "negative_words": per100(neg),
        "sentiment_balance": (pos - neg) / max(pos + neg, 1),   # -1 all negative, +1 all positive
        "intensifiers": per100(_count(text, INTENSIFIERS)),
        "exclamations": per100(text.count("!")),
        "hedges_all": per100(_count(text, HEDGES)),
        # hedging INSIDE the criticism sentences: the key signal for Soften
        "hedges_in_criticism": 100.0 * _count(crit_text, HEDGES) / crit_words if crit else 0.0,
        "minimizers": per100(_count(text, MINIMIZERS)),
        "has_criticism": float(bool(crit)),                      # 0 suggests Inflate, 1 with hedges suggests Soften
        "criticism_share": len(crit) / max(len(sents), 1),       # share of sentences that criticize
        "praise_opener": float(_count(first, PRAISE_OPENERS) > 0),
        "praise_closer": float(_count(last, POSITIVE) > 0 and _count(last, NEGATIVE) == 0),
    }


FEATURE_NAMES = list(eval_features("x").keys())

if __name__ == "__main__":
    import pandas as pd
    src, dst = sys.argv[1], sys.argv[2]
    d = pd.read_csv(src)
    f = pd.DataFrame([eval_features(t) for t in d["response"]])
    pd.concat([d[["prompt_id", "model"]], f], axis=1).to_csv(dst, index=False)
    print(f"wrote {len(f)} rows, {len(f.columns) - 2} features -> {dst}")
