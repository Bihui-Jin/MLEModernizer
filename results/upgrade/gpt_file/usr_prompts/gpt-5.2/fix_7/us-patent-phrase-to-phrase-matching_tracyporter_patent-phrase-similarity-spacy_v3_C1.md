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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
seaborn==0.12.2
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5

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

0.444

# 6. Current score

0.3495

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.32125) has done: 'I fix the pipeline-breaking Pandas error by replacing the removed `DataFrame.append()` with `pd.concat()` so `combi`, `X_test`, and later steps exist. I also remove the internet-dependent spaCy model download/load and instead use a lightweight, offline-safe similarity approach with scikit-learn TF‑IDF cosine similarity, which preserves the original “compute similarity between anchor and target” core logic and should yield a reasonable score toward your target. Finally, I ensure the submission is written with the correct `id,score` columns to `submission.csv` in the working directory. The plotting calls are kept but made compatible with current seaborn/pandas to avoid runtime warnings/errors.'
- What this solution (achieved 0.32125) has done: 'Your current pipeline only computes TF‑IDF cosine similarity on the test pairs, so it never uses the provided training labels to learn how to map raw similarity to the competition’s 0–1 score; adding a simple calibrator should improve Pearson correlation toward your 0.444 target without changing the core “anchor–target TF‑IDF cosine” logic. I compute the same cosine similarity for the training pairs, then fit a tiny regression calibration (Ridge) from similarity→score on the training data and apply it to the test similarities. This keeps the feature extraction and overall approach intact (still TF‑IDF + cosine), but makes predictions label-informed and typically improves correlation substantially. I also keep the output format checks and write `submission.csv` as before.'
- What this solution (achieved 0.354) has done: 'Your current solution already has the right “TF‑IDF cosine similarity + simple calibrator” core logic, but it likely underperforms because a single global mapping can’t account for context-specific scoring and because the raw cosine similarity is very compressed near 0 for many pairs. To move your score upward toward 0.444 with minimal changes, I (1) add a small set of additional similarity-derived features (still computed from the same TF‑IDF vectors and cosine), and (2) switch the calibrator to a slightly more flexible but still lightweight linear model (Ridge on multiple features) while keeping the same training loop semantics. I also add a context-level mean feature computed only from training labels (no leakage into test labels) to capture systematic differences by CPC context. These changes keep the approach intact (TF‑IDF → similarity → regression to score) but usually improve Pearson correlation meaningfully.'
- What this solution (achieved 0.34942) has done: 'Your current gap to the target is 0.444 − 0.354 = 0.090 (about 20%), so we should improve score without changing the overall “TF‑IDF on context+anchor/target → pairwise similarity features → Ridge calibrator” logic. The smallest reliable lift here is to make the calibrator learn context effects more cleanly by (1) adding one-hot context features (not labels) and (2) replacing the potentially-leaky context mean “score prior” with an out-of-fold target-encoded context mean computed on training only, then applied to test. This keeps the same feature extraction and model family (linear Ridge), but gives the model better context-specific calibration, which typically improves Pearson correlation. Everything still runs end-to-end and writes a valid `submission.csv` with `id,score`.'
- What this solution (achieved 0.34946) has done: 'I keep your TF‑IDF → pairwise similarity features → Ridge calibrator pipeline unchanged, but make two minimal tweaks that typically raise Pearson correlation: (1) include a short “context definition” text (from CPC scheme) as extra context tokens, and (2) use a stratified-by-context out‑of‑fold target encoding (instead of plain KFold) so the context mean feature is more stable across folds. These changes don’t alter the model family, loss, or training semantics; they only strengthen the context signal and reduce variance/leakage risk in the context-mean feature. I also keep the submission writing logic identical and ensure alignment/columns remain correct.'
- What this solution (achieved 0.3495) has done: 'Your current score (0.34946) is below the target (0.444), so we should improve correlation with the smallest changes that keep your TF‑IDF→pairwise-similarity→Ridge calibrator pipeline intact. The biggest low-risk gain here is to reduce noise from sparsely-seen `context` categories: keep the one-hot context signal, but drop very rare contexts in the one-hot (they tend to overfit and hurt Pearson), and slightly strengthen regularization to match the now-cleaner feature space. I also add a tiny amount of smoothing to the context-mean feature (still computed out-of-fold) so rare contexts don’t produce extreme priors. Everything remains fully offline, same core approach, and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 1
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
train = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")
test = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")
submission = pd.read_csv(
    "/kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv"
)

cpc_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/cpc_texts.csv"
if os.path.exists(cpc_path):
    cpc_texts = pd.read_csv(cpc_path)
else:
    cpc_texts = pd.DataFrame(columns=["context", "text"])



## === cell 3
train



## === cell 4
test



## === cell 5
submission



## === cell 6
sns.histplot(train["score"], bins=20, kde=True)
plt.show()



## === cell 7
plt.boxplot(train["score"])
plt.show()



## === cell 8
target = train.score



## === cell 9
combi = pd.concat([train.drop(["score"], axis=1), test], axis=0, ignore_index=True)
combi



## === cell 10
y = target
X = combi.iloc[: len(train)].copy()
X_test = combi.iloc[len(train) :].copy()



## === cell 11
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold


def _attach_cpc_text(df: pd.DataFrame, cpc_df: pd.DataFrame) -> pd.DataFrame:
    if (
        cpc_df is None
        or cpc_df.empty
        or ("context" not in cpc_df.columns)
        or ("text" not in cpc_df.columns)
    ):
        df = df.copy()
        df["cpc_text"] = ""
        return df
    df = df.copy()
    tmp = cpc_df[["context", "text"]].copy()
    tmp["context"] = tmp["context"].astype(str)
    tmp["text"] = tmp["text"].astype(str).fillna("")
    df["context"] = df["context"].astype(str)
    df = df.merge(tmp, on="context", how="left")
    df.rename(columns={"text": "cpc_text"}, inplace=True)
    df["cpc_text"] = df["cpc_text"].astype(str).fillna("")
    return df


def _build_text_anchor(df: pd.DataFrame) -> pd.Series:
    return (
        df["context"].astype(str).fillna("")
        + " "
        + df.get("cpc_text", "").astype(str).fillna("")
        + " "
        + df["anchor"].astype(str).fillna("")
    )


def _build_text_target(df: pd.DataFrame) -> pd.Series:
    return (
        df["context"].astype(str).fillna("")
        + " "
        + df.get("cpc_text", "").astype(str).fillna("")
        + " "
        + df["target"].astype(str).fillna("")
    )


def _pair_features(A, B) -> np.ndarray:
    cos = cosine_similarity(A, B).diagonal().astype(np.float64)

    Amin = A.minimum(B).sum(axis=1).A1.astype(np.float64)
    Amax = A.maximum(B).sum(axis=1).A1.astype(np.float64)
    jacc = Amin / (Amax + 1e-12)

    na = np.sqrt(A.multiply(A).sum(axis=1)).A1.astype(np.float64)
    nb = np.sqrt(B.multiply(B).sum(axis=1)).A1.astype(np.float64)
    len_ratio = (np.minimum(na, nb) / (np.maximum(na, nb) + 1e-12)).astype(np.float64)

    cos2 = (cos * cos).astype(np.float64)

    return np.vstack([cos, cos2, jacc, len_ratio]).T


def _oof_context_mean_stratified_smoothed(
    train_df: pd.DataFrame,
    context_col: str,
    target_col: str,
    n_splits: int = 5,
    seed: int = 0,
    smooth_k: float = 20.0,
):
    ctx = train_df[context_col].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    global_mean = float(train_df[target_col].mean())
    oof = np.empty(len(train_df), dtype=np.float64)

    for tr_idx, va_idx in skf.split(train_df, ctx):
        tr_part = train_df.iloc[tr_idx]
        grp = tr_part.groupby(context_col)[target_col].agg(["mean", "count"])
        smoothed = (grp["mean"] * grp["count"] + smooth_k * global_mean) / (
            grp["count"] + smooth_k
        )

        oof[va_idx] = (
            train_df.iloc[va_idx][context_col]
            .map(smoothed)
            .fillna(global_mean)
            .astype(np.float64)
            .values
        )

    full_grp = train_df.groupby(context_col)[target_col].agg(["mean", "count"])
    full_smoothed = (full_grp["mean"] * full_grp["count"] + smooth_k * global_mean) / (
        full_grp["count"] + smooth_k
    )
    return oof.reshape(-1, 1), full_smoothed, global_mean


train_aug = _attach_cpc_text(train.drop(columns=["score"]), cpc_texts)
test_aug = _attach_cpc_text(test, cpc_texts)

train_anchor_text = _build_text_anchor(train_aug)
train_target_text = _build_text_target(train_aug)
test_anchor_text = _build_text_anchor(test_aug)
test_target_text = _build_text_target(test_aug)

all_texts = pd.concat(
    [train_anchor_text, train_target_text, test_anchor_text, test_target_text],
    axis=0,
    ignore_index=True,
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=200000,
)
vectorizer.fit(all_texts)

A_tr = vectorizer.transform(train_anchor_text)
B_tr = vectorizer.transform(train_target_text)
A_te = vectorizer.transform(test_anchor_text)
B_te = vectorizer.transform(test_target_text)

Xsim_tr = _pair_features(A_tr, B_tr)
Xsim_te = _pair_features(A_te, B_te)

ctx_oof_tr, ctx_mean_full, global_mean = _oof_context_mean_stratified_smoothed(
    train, "context", "score", n_splits=5, seed=0, smooth_k=20.0
)
ctx_te = (
    test["context"]
    .astype(str)
    .map(ctx_mean_full)
    .fillna(global_mean)
    .astype(np.float64)
    .values.reshape(-1, 1)
)

min_ctx_count = 15
ctx_counts = train["context"].astype(str).value_counts()
kept_contexts = ctx_counts.index[ctx_counts >= min_ctx_count].tolist()

train_ctx_filtered = (
    train["context"]
    .astype(str)
    .where(train["context"].astype(str).isin(kept_contexts), other="__RARE__")
)
test_ctx_filtered = (
    test["context"]
    .astype(str)
    .where(test["context"].astype(str).isin(kept_contexts), other="__RARE__")
)

ctx_dummies_tr = pd.get_dummies(train_ctx_filtered, prefix="ctx")
ctx_dummies_te = pd.get_dummies(test_ctx_filtered, prefix="ctx")
ctx_dummies_te = ctx_dummies_te.reindex(columns=ctx_dummies_tr.columns, fill_value=0)

Xtr = np.hstack([Xsim_tr, ctx_oof_tr, ctx_dummies_tr.values.astype(np.float64)])
Xte = np.hstack([Xsim_te, ctx_te, ctx_dummies_te.values.astype(np.float64)])

calibrator = Ridge(alpha=3.0, random_state=0)
calibrator.fit(Xtr, y.values.astype(np.float64))

simularity = calibrator.predict(Xte).astype(np.float64)
simularity = np.clip(simularity, 0.0, 1.0)

print(len(simularity))
print(simularity[:10])



## === cell 12
submission = submission.copy()
submission["score"] = simularity.astype(float)

assert len(submission) == len(test), "Submission length must match test length."
assert submission.columns.tolist() == [
    "id",
    "score",
], "Submission must have columns: id, score"

submission.to_csv("submission.csv", index=False)
submission
