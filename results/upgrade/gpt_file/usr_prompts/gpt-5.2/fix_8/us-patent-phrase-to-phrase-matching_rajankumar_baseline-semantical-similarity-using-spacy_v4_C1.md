# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.4194

# 6. Current score

0.34698

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.30085) has done: 'I fix the runtime blocker by removing the dependency on the unavailable `en_core_web_lg` spaCy model and replacing it with a lightweight, always-available similarity baseline using TF‑IDF + cosine similarity (keeps the same “compute similarity then optionally map to 0/0.25/…/1” core approach). I also fix the cascade of `KeyError`/`AttributeError` issues that happen because `similarity_score`/`score` columns never get created after the spaCy load failure. Finally, I ensure the predictions are aligned to the required `[0,1]` range and that a valid `submission.csv` with columns `id,score` is always written.'
- What this solution (achieved 0.34069) has done: 'Your current score (0.30085) is below the target (0.4194), so we should improve correlation with the smallest possible changes while keeping the same TF‑IDF + cosine “compute similarity then predict” core logic. The biggest avoidable issue is the hard discretization to {0,0.25,…,1}, which typically reduces Pearson correlation because the metric rewards continuous ranking; removing that step usually yields a sizable gain without changing the modeling approach. To keep predictions well-calibrated and stable, we fit a simple monotonic 1D calibration (isotonic regression) from TF‑IDF cosine similarity to the true training scores and apply it to test; this preserves the same feature extraction and similarity computation, but better matches the target scale for Pearson. Finally, we keep all outputs clipped to [0,1] and write the same `submission.csv` format.'
- What this solution (achieved 0.34069) has done: 'Your current score (0.34069) is below the target (0.4194), so we want a modest, low-risk lift while keeping the same TF‑IDF + cosine + 1D calibration core. The biggest likely issue is leakage/overfitting from fitting isotonic regression on the full training set and evaluating on the same data; switching to out-of-fold (OOF) isotonic calibration typically generalizes better to Kaggle test and improves Pearson without changing the model family. We do a 5-fold GroupKFold by `context` (to better match how the task varies by CPC code) to generate OOF calibrated predictions, then refit isotonic on all training data for final test predictions. We also keep clipping to [0,1] and the same submission merge to guarantee a valid `submission.csv`.'
- What this solution (achieved 0.35133) has done: 'To move your 0.34069 closer to the 0.4194 target (higher is better) without changing the core “TF‑IDF cosine similarity → 1D calibrator” approach, I make one low-risk improvement: calibrate per `context` rather than globally, because similarity meaning varies strongly by CPC code. Specifically, we fit an isotonic regression model for each context (with a safe fallback to the global isotonic model when a context has too few training examples or too little label variety for isotonic to behave well). This preserves the exact same features, similarity computation, and calibration family, but usually improves Pearson by reducing context-mismatch error. Submission writing and schema stay unchanged.'
- What this solution (achieved 0.3388) has done: 'We’re currently below the target, so we should make a small, low-risk improvement that preserves the same TF‑IDF cosine → isotonic calibration core. The biggest easy gain is to compute cosine similarity using both the `anchor` and `target` phrased in the same joint feature space that includes `context` (currently you fit TF‑IDF on joint text but then compute similarity from a different “context + anchor/target” view). I change similarity computation to use the same `_make_text` representation for each side (context+anchor vs context+target) while keeping the same vectorizer, isotonic models, and submission format. This typically improves correlation a bit by reducing representation mismatch without altering the approach.'
- What this solution (achieved 0.29962) has done: 'We’re below the target (0.3388 vs 0.4194), so we should make a small, low-risk lift while keeping the same TF‑IDF cosine → isotonic calibration core. The most direct improvement is to stop fitting isotonic models on “in-sample” similarities: we generate **out-of-fold (OOF) similarities** for train using a 5-fold GroupKFold by `context`, then fit the isotonic calibrators on those OOF similarities (this typically generalizes better and improves Pearson). We keep the same vectorizer family/params and same cosine similarity computation, but compute train similarities in a leakage-safe way; test similarities are computed by averaging fold models (stable and still minimal). Submission writing stays identical and still produces `submission.csv`.'
- What this solution (achieved 0.34698) has done: 'We’re below the target (0.29962 vs 0.4194), so we aim for a modest lift while keeping the same TF‑IDF cosine → isotonic calibration core. The biggest avoidable weakness is that the TF‑IDF space is being refit inside each fold (and even includes test text), which makes the “similarity_score” noisy/inconsistent across folds and can hurt Pearson. With minimal change, we fit a single TF‑IDF vectorizer on train-only text once, compute cosine similarities for train/test in that fixed space, and then keep your existing global + per-context isotonic calibration unchanged. This keeps the same model family and semantics, but typically improves correlation by making similarities comparable across all rows.'

# 9. Code solution

## === cell 0
import os
import sys
import time
import numpy as np
import pandas as pd



## === cell 1
TRAIN_FILE_PATH = "../input/us-patent-phrase-to-phrase-matching/train.csv"
TEST_FILE_PATH = "../input/us-patent-phrase-to-phrase-matching/test.csv"
SAMPLE_SUBMISSION_PATH = (
    "../input/us-patent-phrase-to-phrase-matching/sample_submission.csv"
)




## === cell 2
class config:
    PRINT_EVERY_N_WORD = 1000
    BAR_LEN = 50




## === cell 3
train_df = pd.read_csv(TRAIN_FILE_PATH)
test_df = pd.read_csv(TEST_FILE_PATH)
submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

print("train_df shape:", train_df.shape)
print("test_df shape:", test_df.shape)
print("submission_df shape:", submission_df.shape)



## === cell 4
from sklearn.feature_extraction.text import TfidfVectorizer


def _make_text_pair(df: pd.DataFrame, side: str) -> pd.Series:
    if side == "left":
        return df["context"].astype(str) + " [SEP] " + df["anchor"].astype(str)
    elif side == "right":
        return df["context"].astype(str) + " [SEP] " + df["target"].astype(str)
    raise ValueError("side must be 'left' or 'right'")


def _fit_vectorizer(text_series: pd.Series) -> TfidfVectorizer:
    v = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        lowercase=True,
    )
    v.fit(text_series)
    return v


def _cosine_from_vectorizer(v: TfidfVectorizer, df: pd.DataFrame) -> np.ndarray:
    left = _make_text_pair(df, "left").tolist()
    right = _make_text_pair(df, "right").tolist()

    A = v.transform(left)
    B = v.transform(right)

    num = A.multiply(B).sum(axis=1).A1
    denom = (
        np.sqrt(A.multiply(A).sum(axis=1)).A1 * np.sqrt(B.multiply(B).sum(axis=1)).A1
    )
    sim = np.divide(num, denom, out=np.zeros_like(num, dtype=float), where=denom != 0)
    return np.clip(sim, 0.0, 1.0)


start = time.time()
train_text_all = pd.concat(
    [_make_text_pair(train_df, "left"), _make_text_pair(train_df, "right")], axis=0
).reset_index(drop=True)
vec = _fit_vectorizer(train_text_all)

train_sim = _cosine_from_vectorizer(vec, train_df)
test_sim = _cosine_from_vectorizer(vec, test_df)

print(
    f"Computed TF-IDF cosine similarities (single train-only vectorizer) in {time.time()-start:.1f}s"
)

train_df["similarity_score"] = train_sim.astype(float)
test_df["score_raw_sim"] = test_sim.astype(float)



## === cell 5
from scipy.stats import pearsonr

corr, _ = pearsonr(train_df["score"].values, train_df["similarity_score"].values)
print("Training Pearson Correlation (continuous sim): %0.3f" % corr)



## === cell 6
from sklearn.isotonic import IsotonicRegression

X_sim = train_df["similarity_score"].values.astype(float)
y = train_df["score"].values.astype(float)

global_iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
global_iso.fit(X_sim, y)

train_pred_global = global_iso.predict(X_sim)
corr_cal, _ = pearsonr(y, train_pred_global)
print(
    "Training Pearson Correlation (isotonic calibrated, global fit): %0.3f" % corr_cal
)

MIN_CTX_SAMPLES = 80
MIN_UNIQUE_LABELS = 3

ctx_models = {}
ctx_counts = train_df["context"].astype(str).value_counts().to_dict()
train_ctx = train_df["context"].astype(str).values

for ctx, cnt in ctx_counts.items():
    if cnt < MIN_CTX_SAMPLES:
        continue
    mask = train_ctx == ctx
    y_ctx = y[mask]
    if np.unique(y_ctx).size < MIN_UNIQUE_LABELS:
        continue
    iso_ctx = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_ctx.fit(X_sim[mask], y_ctx)
    ctx_models[ctx] = iso_ctx

print(
    f"Built {len(ctx_models)} per-context isotonic models (fallback to global for others)."
)

test_ctx = test_df["context"].astype(str).values
test_sim = test_df["score_raw_sim"].values.astype(float)

test_pred = global_iso.predict(test_sim)
for ctx, model in ctx_models.items():
    m = test_ctx == ctx
    if np.any(m):
        test_pred[m] = model.predict(test_sim[m])

test_df["score"] = np.clip(test_pred.astype(float), 0.0, 1.0)



## === cell 7
sub = submission_df[["id"]].merge(test_df[["id", "score"]], on="id", how="left")
sub["score"] = sub["score"].fillna(0.0).astype(float).clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
