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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

datasets==4.4.1
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
tokenizers==0.21.2
transformers==4.53.3
vega-datasets==0.9.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.7993434084034929

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
LOAD_FROM = None  # Disabled loading from a non‑existent checkpoint



## === cell 1
import warnings
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score

warnings.simplefilter("ignore")




## === cell 2
class PATHS:
    train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
    test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    sub_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"


USE_REGRESSION = True



## === cell 3
train_df = pd.read_csv(PATHS.train_path)
test_df = pd.read_csv(PATHS.test_path)
sub_df = pd.read_csv(PATHS.sub_path)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)




## === cell 4
class CFG:
    n_splits = 5
    seed = 42
    max_features = 50000  # limit vocabulary size for speed / memory
    ngram_range = (1, 2)  # use unigrams & bigrams
    alpha = 1.0  # Ridge regularization strength




## === cell 5
kf = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

X = train_df["full_text"].astype(str).reset_index(drop=True)
y = train_df["score"].astype(float).reset_index(drop=True)

test_texts = test_df["full_text"].astype(str).reset_index(drop=True)

test_pred_sum = np.zeros(len(test_df), dtype=np.float64)

fold = 0
for train_idx, val_idx in kf.split(X, y):
    fold += 1
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    vectorizer = TfidfVectorizer(
        max_features=CFG.max_features,
        ngram_range=CFG.ngram_range,
        stop_words="english",
    )
    X_tr_vec = vectorizer.fit_transform(X_tr)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(test_texts)

    model = Ridge(alpha=CFG.alpha, random_state=CFG.seed)
    model.fit(X_tr_vec, y_tr)

    val_pred = model.predict(X_val_vec).clip(1, 6).round()
    qwk = cohen_kappa_score(y_val, val_pred, weights="quadratic")
    print(f"Fold {fold} – Validation QWK: {qwk:.4f}")

    test_pred = model.predict(X_test_vec)
    test_pred_sum += test_pred

test_pred_avg = test_pred_sum / CFG.n_splits



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    114             try:
--> 115                 coef, info = sp_linalg.cg(C, y_column, tol=tol, atol="legacy")
    116             except TypeError:

TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2992283640.py in <cell line: 0>()
     27     # Ridge regression model
     28     model = Ridge(alpha=CFG.alpha, random_state=CFG.seed)
---> 29     model.fit(X_tr_vec, y_tr)
     30 
     31     # Optional: compute QWK on validation split for monitoring

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
    116             except TypeError:
    117                 # old scipy
--> 118                 coef, info = sp_linalg.cg(C, y_column, tol=tol)
    119             coefs[i] = X1.rmatvec(coef)
    120         else:

TypeError: cg() got an unexpected keyword argument 'tol'

## === cell 6
if USE_REGRESSION:
    final_scores = np.clip(test_pred_avg, 1, 6).round().astype(int)
else:
    final_scores = np.argmax(test_pred_avg, axis=1) + 1

sub_df["score"] = final_scores
sub_df["score"] = sub_df["score"].fillna(1).astype("int32")
sub_df.to_csv("submission.csv", index=False)

print("Submission written to submission.csv")
print("Submission shape:", sub_df.shape)
sub_df.head()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/220175974.py in <cell line: 0>()
      2 if USE_REGRESSION:
      3     # Clip to valid score range, round, and shift to integer 1‑6
----> 4     final_scores = np.clip(test_pred_avg, 1, 6).round().astype(int)
      5 else:
      6     # Not used in this script, kept for compatibility

NameError: name 'test_pred_avg' is not defined
