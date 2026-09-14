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

0.8069452756275215

# 6. Current score

0.65201

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.53436) has done: 'I fix the error caused by Ridge’s default sparse solver by explicitly using a solver compatible with the current SciPy version (`solver="lsqr"`). This resolves the `cg()` argument issue, allowing the model to train, produce predictions, and write a proper `submission.csv` file. No other logic is altered, preserving the original approach and keeping the validation metric unchanged.'
- What this solution (achieved 0.53442) has done: 'I add a simple linear scaling step that aligns the model’s raw predictions with the true scores on the validation split. By fitting a 1‑D linear regression (slope & intercept) on validation predictions and applying the same transformation to the test predictions, we can improve Pearson correlation without changing the core TF‑IDF + Ridge pipeline. The scaling is clipped to the [0, 1] range to keep predictions valid.'
- What this solution (achieved 0.60238) has done: 'The changes adjust the TF‑IDF vectorizer to capture a broader range of n‑grams and more features, and reduce the Ridge regularisation strength (alpha) to let the model fit the data more closely. These minor parameter tweaks keep the overall TF‑IDF + Ridge pipeline intact while expectedly raising the validation Pearson correlation, moving the score toward the target.'
- What this solution (achieved 0.65477) has done: 'I keep the overall TF‑IDF + Ridge pipeline but improve the feature representation and regularisation: add a character‑level TF‑IDF vectorizer, remove stop‑word removal, enable sublinear term frequencies, and increase the word‑level feature budget. I also lower the Ridge alpha to let the model fit the data more closely. These changes are small extensions of the existing logic and are expected to raise the Pearson correlation toward the target score.'
- What this solution (achieved 0.65201) has done: 'I keep the TF‑IDF + Ridge pipeline but tune a few hyper‑parameters that have a high impact on Pearson correlation while preserving the overall structure.  
- Increase the word and character TF‑IDF feature budgets (300 k and 100 k) to capture more informative n‑grams.  
- Try a small set of Ridge regularisation strengths (0.01, 0.1, 1.0) on the validation split, keep the model and scaling that yields the highest validation Pearson.  
- Retrain the selected model on the full training data before predicting the test set, using the same scaling factors.  

These targeted tweaks should raise the validation correlation toward the target without altering the core workflow.'

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
from scipy.sparse import hstack

base_path = Path("input/us-patent-phrase-to-phrase-matching")
if not base_path.exists():
    base_path = Path("us-patent-phrase-to-phrase-matching")
assert base_path.exists(), f"Data path not found: {base_path}"



## === cell 1
train_df = pd.read_csv(base_path / "train.csv")
test_df = pd.read_csv(base_path / "test.csv")

train_df["input"] = (
    "TEXT1: "
    + train_df["context"]
    + "; TEXT2: "
    + train_df["target"]
    + "; ANC1: "
    + train_df["anchor"]
)
test_df["input"] = (
    "TEXT1: "
    + test_df["context"]
    + "; TEXT2: "
    + test_df["target"]
    + "; ANC1: "
    + test_df["anchor"]
)



## === cell 2
train_split, val_split = train_test_split(train_df, test_size=0.25, random_state=42)

word_vectorizer = TfidfVectorizer(
    max_features=300000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    stop_words=None,
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=100000,
    sublinear_tf=True,
)

X_train_word = word_vectorizer.fit_transform(train_split["input"])
X_train_char = char_vectorizer.fit_transform(train_split["input"])
X_train = hstack([X_train_word, X_train_char])

X_val_word = word_vectorizer.transform(val_split["input"])
X_val_char = char_vectorizer.transform(val_split["input"])
X_val = hstack([X_val_word, X_val_char])

y_train = train_split["score"].values
y_val = val_split["score"].values

candidate_alphas = [0.01, 0.1, 1.0]
best_pearson = -np.inf
best_alpha = None
best_model = None
best_scale_coef = None
best_scale_intercept = None

for alpha in candidate_alphas:
    model = Ridge(alpha=alpha, solver="lsqr", random_state=42)
    model.fit(X_train, y_train)
    val_preds_raw = model.predict(X_val)
    coef, intercept = np.polyfit(val_preds_raw, y_val, 1)
    val_preds = np.clip(coef * val_preds_raw + intercept, 0.0, 1.0)
    pearson = pearsonr(y_val, val_preds)[0]
    if pearson > best_pearson:
        best_pearson = pearson
        best_alpha = alpha
        best_model = model
        best_scale_coef = coef
        best_scale_intercept = intercept

print(f"Chosen alpha: {best_alpha}, Validation Pearson (scaled): {best_pearson:.5f}")



## === cell 3
X_full_word = word_vectorizer.fit_transform(train_df["input"])
X_full_char = char_vectorizer.fit_transform(train_df["input"])
X_full = hstack([X_full_word, X_full_char])
y_full = train_df["score"].values

final_model = Ridge(alpha=best_alpha, solver="lsqr", random_state=42)
final_model.fit(X_full, y_full)

X_test_word = word_vectorizer.transform(test_df["input"])
X_test_char = char_vectorizer.transform(test_df["input"])
X_test = hstack([X_test_word, X_test_char])
test_preds_raw = final_model.predict(X_test)
test_preds = np.clip(best_scale_coef * test_preds_raw + best_scale_intercept, 0.0, 1.0)



## === cell 4
submission = pd.DataFrame({"id": test_df["id"], "score": test_preds})
submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path.resolve()}")
