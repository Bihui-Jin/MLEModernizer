# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

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



## === cell 4
vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    analyzer="char",
)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

model = Ridge(alpha=1.0)
model.fit(X_train_vec, y_train)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    128             try:
--> 129                 coefs[i], info = sp_linalg.cg(
    130                     C, y_column, maxiter=max_iter, tol=tol, atol="legacy"

TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3715084082.py in <cell line: 0>()
      8 
      9 model = Ridge(alpha=1.0)
---> 10 model.fit(X_train_vec, y_train)
     11 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
   1132             y_numeric=True,
   1133         )
-> 1134         return super().fit(X, y, sample_weight=sample_weight)
   1135 
   1136 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
    898                 params = {}
    899 
--> 900             self.coef_, self.n_iter_ = _ridge_regression(
    901                 X,
    902                 y,

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _ridge_regression(X, y, alpha, sample_weight, solver, max_iter, tol, verbose, positive, random_state, return_n_iter, return_intercept, X_scale, X_offset, check_input, fit_intercept)
    669     n_iter = None
    670     if solver == "sparse_cg":
--> 671         coef = _solve_sparse_cg(
    672             X,
    673             y,

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    132             except TypeError:
    133                 # old scipy
--> 134                 coefs[i], info = sp_linalg.cg(C, y_column, maxiter=max_iter, tol=tol)
    135 
    136         if info < 0:

TypeError: cg() got an unexpected keyword argument 'tol'

## === cell 5
val_preds = model.predict(X_val_vec)
pearson, _ = pearsonr(y_val, val_preds)
print(f"Validation Pearson correlation: {pearson:.6f}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1638288077.py in <cell line: 0>()
----> 1 val_preds = model.predict(X_val_vec)
      2 pearson, _ = pearsonr(y_val, val_preds)
      3 print(f"Validation Pearson correlation: {pearson:.6f}")
      4 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    352             Returns predicted values.
    353         """
--> 354         return self._decision_function(X)
    355 
    356     def _set_intercept(self, X_offset, y_offset, X_scale):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in _decision_function(self, X)
    336 
    337         X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
--> 338         return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
    339 
    340     def predict(self, X):

AttributeError: 'Ridge' object has no attribute 'coef_'

## === cell 6
X_test_vec = vectorizer.transform(test_df["inputs"].values)
test_preds = model.predict(X_test_vec)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1159921275.py in <cell line: 0>()
      1 X_test_vec = vectorizer.transform(test_df["inputs"].values)
----> 2 test_preds = model.predict(X_test_vec)
      3 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    352             Returns predicted values.
    353         """
--> 354         return self._decision_function(X)
    355 
    356     def _set_intercept(self, X_offset, y_offset, X_scale):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in _decision_function(self, X)
    336 
    337         X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
--> 338         return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
    339 
    340     def predict(self, X):

AttributeError: 'Ridge' object has no attribute 'coef_'

## === cell 7
submission = pd.DataFrame({"id": test_df["id"], "score": test_preds})
submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path.resolve()}")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3514684630.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"id": test_df["id"], "score": test_preds})
      2 submission_path = Path("submission.csv")
      3 submission.to_csv(submission_path, index=False)
      4 print(f"Submission saved to {submission_path.resolve()}")

NameError: name 'test_preds' is not defined
