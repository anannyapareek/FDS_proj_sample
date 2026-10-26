# FlatDetect: Project Plan and Work Split

Cross-model sycophancy detection using black-box text analysis.
Team: Sanjam Bedi, Eesha B, Anannya Pareek, Bhoomika Singh.

---

## 1. Goal

Detect sycophancy in LLM responses using only the output text (no weights, no activations), classify which kind of sycophancy it is, and test whether a detector trained on one model's responses works on models it has never seen.

**Contributions**
1. Text-only (black-box) sycophancy detector with a 0 to 1 sycophancy score.
2. Own taxonomy of sycophancy (two families, five breeds).
3. Cross-model transfer results, per family and per breed.

Novelty is stated as "to our knowledge". Text-only detection has appeared as a control experiment in internal-probing papers, and we found no study testing cross-model transfer of a text-only detector.

---

## 2. Taxonomy (our own)

```
Sycophancy
├── Agreement family: the AI goes along with a claim
│   ├── Echo     endorses the user's stated belief
│   ├── Cave     drops its correct answer after pushback
│   └── Defer    accepts a claim because of claimed credentials
└── Evaluation family: the AI judges the user's work or idea
    ├── Inflate  praises beyond what it deserves
    └── Soften   buries or dilutes real criticism
```

**Family rule:** if the response is about whether a claim is true, it is Agreement. If it is about how good something is, it is Evaluation.

| Breed | Family | Response does this | Text signal |
|---|---|---|---|
| Echo | Agreement | Endorses the user's stated belief | agreement phrases |
| Cave | Agreement | Reverses a correct answer after pushback | apology words, stance shift |
| Defer | Agreement | Accepts a claim because of credentials | credential references |
| Inflate | Evaluation | Over-praises weak work | superlatives, positive sentiment |
| Soften | Evaluation | Mentions the flaw but buries it | hedging, "minor", "however" |

Labels are defined by what the response does, not by which dataset it came from. Dataset source is metadata and a rough prior only. One primary breed per response; overlaps are reported as a limitation.

---

## 3. Datasets

| Dataset | Used for |
|---|---|
| Anthropic model-written-evals | Agreement prompts (opinion mirroring) |
| EleutherAI/sycophancy | Agreement prompts, independent cross-check |
| meg-tong/sycophancy-eval | Caving (`are_you_sure`), suggested answers (`answer`), feedback (`feedback`) |
| Beacon | Held out for final evaluation |
| Custom (about 100 prompts) | Mostly Evaluation, plus credential and emotional-pressure variants |

Confirm the exact meg-tong file names in the repo before relying on them.

None of these datasets contain real model responses. We send the prompts to models through their APIs and record what they say.

---

## 4. Tools

Python, pandas, scikit-learn, XGBoost, HuggingFace datasets, Gemini API (free tier), Groq API (free tier, Llama and other open models). No database server is needed. CSV files are enough.

Check each console for current model names, since older ones get retired.

---

## 5. Shared rules

- One GitHub repo with folders `data/`, `src/`, `results/`.
- Never commit API keys. Use a `.env` file.
- Shared column names:
  - `prompts.csv`: `prompt_id | source | variant | prompt | answer_key`
  - `responses.csv`: `prompt_id | model | response`
  - `labeled.csv`: `prompt_id | model | response | is_sycophantic | family | breed`
  - `features.csv`: `prompt_id | model | feature columns... | label columns`

---

## 6. Pipeline

```
datasets → prompts → responses → labels → features → classifier → cross-model test → write-up
```

1. **Setup.** Get Gemini and Groq keys, test one call on each, create the repo.
2. **Download datasets.** Anthropic, EleutherAI, meg-tong into `data/raw`. Keep Beacon aside.
3. **Write the codebook.** Definition, 2 examples and edge cases per breed.
4. **Write custom prompts.** About 100, mostly Evaluation: weak essays, flawed plans, buggy code, with emotional-pressure and credential variants.
5. **Build `prompts.csv`.** Sample about 300 prompts per Agreement breed. Keep neutral and pressured versions of Cave and Evaluation items.
6. **Fix multiple-choice prompts.** Add "Answer in 2 to 3 sentences and explain why" so the responses have text to analyze.
7. **Collect responses.** Send every prompt to 3 models (Gemini, Llama on Groq, one more on Groq). Add delays, save after every call, skip rows already done.
8. **Label responses.** LLM judge (a model not being tested) gives the first pass using the codebook. All four members hand-label the same 100 responses and report Cohen's or Fleiss' kappa. Revise the codebook if agreement is low.
9. **Extract features.** Hedging, agreement phrases, apology words, superlatives, sentiment, credential references, stance shift, response length. TF-IDF as baseline.
10. **Train in three stages.** Stage 1: sycophantic or not (`predict_proba` gives the 0 to 1 score). Stage 2: family. Stage 3: breed. Logistic regression and XGBoost. Split by `prompt_id`. Use class weights, since Agreement will be larger than Evaluation.
11. **Cross-model test.** Train on one model's responses, test on the others with no retraining. 3×3 transfer table per family and per breed. List which features survive and which break.
12. **Sanity checks.** Leave-one-source-out (train without one dataset, test on it) to check the classifier learned sycophancy and not a dataset's writing style. Then run the final classifier on Beacon.
13. **Write-up.** Problem, taxonomy, datasets, method, in-domain results, cross-model results, limitations.

---

## 7. Work split

| Person | Role | Main deliverables |
|---|---|---|
| Sanjam | Data and collection | `prompts.csv`, `collect.py`, Gemini responses |
| Eesha | Taxonomy and labels | `codebook.md`, `judge.py`, `kappa.py`, `labeled.csv` |
| Anannya | Features and models | `features.py`, TF-IDF baseline, trained classifiers |
| Bhoomika | Testing and report | `transfer.py`, sanity checks, final report |

### Sanjam: data and collection
1. Create the repo, get Gemini and Groq keys, test one call on each.
2. Download all datasets into `data/raw`, keep Beacon aside, note each format, confirm meg-tong file names.
3. Build `prompts.csv`: convert to the shared format, sample prompts, add the multiple-choice fix, keep neutral and pressured versions, merge in the custom prompts.
4. Write `collect.py` (model name argument, delay, save after every call, resume). It must work for Gemini and Groq so the others can reuse it.
5. Run Gemini collection.

Done when: `prompts.csv` is final, `collect.py` is shared, Gemini responses are complete.

### Eesha: taxonomy and labels
1. Write `codebook.md` in the first 2 days, since others are blocked on it.
2. Write 50 custom prompts, mostly Evaluation.
3. Write `judge.py`: sends each response and the codebook to an LLM judge and gets family and breed back.
4. Run group labeling, collect the sheets, compute kappa in `kappa.py`, revise the codebook if agreement is low.
5. Produce `labeled.csv` by merging judge labels, answer-key checks and hand corrections.
6. Code the Evaluation features: superlatives, positive sentiment, hedging in criticism.

Done when: codebook is final, kappa is reported, `labeled.csv` exists for all models.

### Anannya: features and models
1. Run Llama collection on Groq with Sanjam's script.
2. Write `features.py` with the Agreement features (agreement phrases, apology words, credential references, stance shift), then merge Eesha's Evaluation features. Normalize counts by response length.
3. Build the TF-IDF baseline with logistic regression.
4. Train the three stages with logistic regression and XGBoost. Split by `prompt_id`, use class weights.
5. Report accuracy, F1, confusion matrix and feature importance. Save models with `joblib` for Bhoomika.

Done when: trained model files and a results table exist for both the hand-crafted and TF-IDF versions.

### Bhoomika: testing and report
1. Write 50 custom prompts.
2. Run third-model collection on Groq, using a different model from Anannya's Llama.
3. Write `transfer.py` while waiting for trained models: train on one model, test on the others, fill the 3×3 table per family and per breed, list stable and unstable features.
4. Run leave-one-source-out.
5. Run the final classifier on Beacon. Check its format first, since it is forced-choice and may need adapting.
6. Compile the report from everyone's sections.

Done when: transfer table, sanity-check table and report draft exist.

### All four together
- Hand-label the same 100 responses for kappa (Week 2).
- Each person writes the report section for their own work.
- Everyone should be able to explain the whole pipeline in the viva, not just their own part.

### Handoffs

```
Sanjam: prompts.csv ──► everyone runs collection
Eesha:  codebook.md ──► labeling
All:    responses.csv ──► Eesha: labeled.csv ──► Anannya: features + training ──► Bhoomika: cross-model test
```

---

## 8. Timeline

| Week | Sanjam | Eesha | Anannya | Bhoomika |
|---|---|---|---|---|
| 1 | Setup, datasets, `prompts.csv`, `collect.py` | Codebook, 50 prompts | Test keys, draft features on dummy data | 50 prompts, draft `transfer.py` |
| 2 | Gemini responses | Judge labels, group labeling, kappa | Llama responses | Third-model responses |
| 3 | Support and debugging | Finish `labeled.csv`, Evaluation features | Features, training | Cross-model test |
| 4 | Write data section | Write taxonomy section | Write model section | Sanity checks, compile report |

---

## 9. Risks

- **Bottleneck on labels.** Anannya and Bhoomika cannot do real work until Eesha's labels exist, so the codebook comes first.
- **Rate limits.** Free tiers are slow and limits are per key. Each person collects with their own key.
- **Thin Evaluation data.** Anthropic, EleutherAI and most meg-tong tasks only produce Agreement responses. Evaluation comes from meg-tong `feedback` and the custom prompts, so write the custom set mostly as Evaluation.
- **Class imbalance.** Agreement will be much larger. Use class weights or even sampling.
- **Overlapping breeds.** A response can fit two breeds. The kappa shows how big the problem is.
- **Short responses give few text signals.** The multiple-choice fix in step 6 helps.
- **Source confound.** Stage 3 might learn a dataset's style instead of the behavior. Leave-one-source-out is the check.
- **Hedging is ambiguous.** Genuine uncertainty also uses hedge words, which is why we combine several features.

---

## 10. What to tell sir

> Since there is no standard taxonomy, we defined our own. Sycophancy splits into an Agreement family and an Evaluation family, with five breeds: Echo, Cave, Defer, Inflate and Soften. Each breed is defined by what the response does and maps to a measurable text signal, and we validate the labels with inter-annotator agreement. The datasets and sub-class ideas come from published work. Our contribution is the black-box setup, the cross-model transfer test, and the breed-level analysis of which text signals survive across models.
