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

No external packages required in the script and installed.

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

0.8084755399348172

# 6. Current score

0.56861

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.38176) has done: 'The fix updates the Ridge regression to use a solver compatible with the available SciPy version (`solver='lsqr'`), preventing the `cg`‑related TypeError and allowing the model to train. After training, predictions are clipped to the valid 0‑1 score range to keep the submission format correct and potentially improve the Pearson score. No other logic is altered, preserving the original workflow.'
- What this solution (achieved 0.5186) has done: 'I add a complementary word‑level TF‑IDF representation and combine it with the existing character‑level TF‑IDF using a sparse hstack. This richer feature set typically captures semantic similarity better and should raise the Pearson correlation toward the target. I also slightly lower the Ridge regularisation (alpha = 0.5) to let the model fit the data more closely while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.56861) has done: 'I add two simple similarity features (token‑overlap ratio and character‑length difference) to the existing TF‑IDF representation and lower the Ridge regularisation (α = 0.2). These extra numeric cues often boost Pearson correlation without altering the core pipeline, and the smaller α lets the model utilise the richer feature set more fully, moving the score toward the target.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from scipy import sparse




## === cell 1
def locate_file(rel_path):
    """
    Locate a file that may reside in one of several common Kaggle directories.
    """
    candidates = [
        Path("data") / rel_path,
        Path("input") / rel_path,
        Path("kaggle") / "input" / rel_path,
        Path("/kaggle") / "input" / rel_path,  # absolute fallback for Kaggle kernels
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Could not locate {rel_path} in any known location.")


base_rel = Path("us-patent-phrase-to-phrase-matching")
train_path = locate_file(base_rel / "train.csv")
test_path = locate_file(base_rel / "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)




## === cell 2
def make_input(df):
    return (
        df["context"].fillna("")
        + " "
        + df["anchor"].fillna("")
        + " "
        + df["target"].fillna("")
    )


train_df["inputs"] = make_input(train_df)
test_df["inputs"] = make_input(test_df)



## === cell 3
np.random.seed(42)
unique_anchors = train_df["anchor"].unique()
np.random.shuffle(unique_anchors)
val_ratio = 0.25
val_anchors = set(unique_anchors[: int(len(unique_anchors) * val_ratio)])

is_val = train_df["anchor"].isin(val_anchors)
train_split = train_df[~is_val]
val_split = train_df[is_val]

X_train, y_train = train_split["inputs"].values, train_split["score"].values
X_val, y_val = val_split["inputs"].values, val_split["score"].values

char_vectorizer = TfidfVectorizer(
    max_features=150000,
    ngram_range=(1, 2),
    analyzer="char",
)

word_vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    analyzer="word",
    stop_words=None,
    sublinear_tf=True,
)

X_train_char = char_vectorizer.fit_transform(X_train)
X_train_word = word_vectorizer.fit_transform(X_train)

X_val_char = char_vectorizer.transform(X_val)
X_val_word = word_vectorizer.transform(X_val)


def extra_features(df):
    anchor = df["anchor"].fillna("").astype(str)
    target = df["target"].fillna("").astype(str)
    overlap = []
    char_diff = []
    for a, t in zip(anchor, target):
        a_set = set(a.split())
        t_set = set(t.split())
        union = a_set | t_set
        inter = a_set & t_set
        overlap.append(len(inter) / len(union) if union else 0.0)
        char_diff.append(abs(len(a) - len(t)))
    return np.column_stack([overlap, char_diff])


extra_train = extra_features(train_split)
extra_val = extra_features(val_split)

extra_train_sparse = sparse.csr_matrix(extra_train)
extra_val_sparse = sparse.csr_matrix(extra_val)

X_train_vec = sparse.hstack([X_train_char, X_train_word, extra_train_sparse])
X_val_vec = sparse.hstack([X_val_char, X_val_word, extra_val_sparse])

model = Ridge(alpha=0.2, solver="lsqr")
model.fit(X_train_vec, y_train)



## === cell 4
val_preds = model.predict(X_val_vec)
pearson, _ = pearsonr(y_val, val_preds)
print(f"Validation Pearson correlation: {pearson:.6f}")



## === cell 5
X_test_char = char_vectorizer.transform(test_df["inputs"].values)
X_test_word = word_vectorizer.transform(test_df["inputs"].values)

extra_test = extra_features(test_df)
extra_test_sparse = sparse.csr_matrix(extra_test)

X_test_vec = sparse.hstack([X_test_char, X_test_word, extra_test_sparse])

test_preds = model.predict(X_test_vec)
test_preds = np.clip(test_preds, 0.0, 1.0)



## === cell 6
submission = pd.DataFrame({"id": test_df["id"], "score": test_preds})
submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path.resolve()}")
