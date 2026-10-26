# FlatDetect

Cross-model sycophancy detection using black-box text analysis.

Team: Sanjam Bedi, Eesha B, Anannya Pareek, Bhoomika Singh.
Full plan and work split: see [PLAN.md](PLAN.md).

## Taxonomy

```
Sycophancy
├── Agreement family
│   ├── Echo     endorses the user's stated belief
│   ├── Cave     drops its correct answer after pushback
│   └── Defer    accepts a claim because of claimed credentials
└── Evaluation family
    ├── Inflate  praises beyond what it deserves
    └── Soften   buries or dilutes real criticism
```

Definitions and examples: [codebook.md](codebook.md).

## Shared column names (everyone must use these)

| File | Columns |
|---|---|
| `data/prompts.csv` | `prompt_id, source, variant, prompt, answer_key` |
| `data/responses_<model>.csv` | `prompt_id, model, response` |
| `data/labeled.csv` | `prompt_id, model, response, is_sycophantic, family, breed` |
| `data/features.csv` | `prompt_id, model, <feature columns>, is_sycophantic, family, breed` |

`variant` is `neutral` or `pressured`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your own keys
```

Get free keys: Gemini at aistudio.google.com, Groq at console.groq.com.
Check each console for current model names.

## Owners

| Person | Area | Files |
|---|---|---|
| Sanjam | Data and collection | `src/collect.py`, `data/prompts.csv` |
| Eesha | Taxonomy and labels | `codebook.md`, `src/judge.py`, `src/kappa.py` |
| Anannya | Features and models | `src/features.py`, `src/train.py` |
| Bhoomika | Testing and report | `src/transfer.py`, `results/` |

## Rules

- Work on your own branch (`name/topic`) and merge through pull requests.
- Never commit API keys or `data/raw/`.
- Do not change shared column names without telling everyone.
