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

0.30085

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.30085) has done: 'I fix the runtime blocker by removing the dependency on the unavailable `en_core_web_lg` spaCy model and replacing it with a lightweight, always-available similarity baseline using TF‑IDF + cosine similarity (keeps the same “compute similarity then optionally map to 0/0.25/…/1” core approach). I also fix the cascade of `KeyError`/`AttributeError` issues that happen because `similarity_score`/`score` columns never get created after the spaCy load failure. Finally, I ensure the predictions are aligned to the required `[0,1]` range and that a valid `submission.csv` with columns `id,score` is always written.'

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
from sklearn.metrics.pairwise import cosine_similarity


def _make_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["context"].astype(str)
        + " [SEP] "
        + df["anchor"].astype(str)
        + " [SEP] "
        + df["target"].astype(str)
    )


all_text = pd.concat([_make_text(train_df), _make_text(test_df)], axis=0).reset_index(
    drop=True
)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    strip_accents="unicode",
    lowercase=True,
)
X_all = vectorizer.fit_transform(all_text)

n_train = len(train_df)
X_train = X_all[:n_train]
X_test = X_all[n_train:]


def compute_pairwise_cosine(df: pd.DataFrame) -> np.ndarray:
    left = (df["context"].astype(str) + " " + df["anchor"].astype(str)).tolist()
    right = (df["context"].astype(str) + " " + df["target"].astype(str)).tolist()

    A = vectorizer.transform(left)
    B = vectorizer.transform(right)

    num = A.multiply(B).sum(axis=1).A1
    denom = (
        np.sqrt(A.multiply(A).sum(axis=1)).A1 * np.sqrt(B.multiply(B).sum(axis=1)).A1
    )
    sim = np.divide(num, denom, out=np.zeros_like(num, dtype=float), where=denom != 0)
    return sim


start = time.time()
train_sim = compute_pairwise_cosine(train_df)
test_sim = compute_pairwise_cosine(test_df)
print(f"Computed TF-IDF cosine similarities in {time.time()-start:.1f}s")

train_df["similarity_score"] = np.clip(train_sim, 0.0, 1.0)
test_df["score"] = np.clip(test_sim, 0.0, 1.0)



## === cell 5
"""
0.000 - 0.125 -> 0.00
0.125 - 0.375 -> 0.25
0.375 - 0.625 -> 0.50
0.625 - 0.875 -> 0.75
0.875 - 1.000 -> 1.00
"""
mapping = {
    0.00: (0.000, 0.125),
    0.25: (0.125, 0.375),
    0.50: (0.375, 0.625),
    0.75: (0.625, 0.875),
    1.00: (0.875, 1.0000000001),  # include 1.0
}

for key, (lo, hi) in mapping.items():
    train_df["similarity_score"] = train_df["similarity_score"].mask(
        (train_df["similarity_score"] >= lo) & (train_df["similarity_score"] < hi), key
    )



## === cell 6
from scipy.stats import pearsonr

corr, _ = pearsonr(train_df["score"].values, train_df["similarity_score"].values)
print("Training Pearson Correlation: %0.3f" % corr)



## === cell 7
for key, (lo, hi) in mapping.items():
    test_df["score"] = test_df["score"].mask(
        (test_df["score"] >= lo) & (test_df["score"] < hi), key
    )

test_df["score"] = test_df["score"].astype(float).clip(0.0, 1.0)



## === cell 8
sub = submission_df[["id"]].merge(test_df[["id", "score"]], on="id", how="left")
sub["score"] = sub["score"].fillna(0.0).astype(float).clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
