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

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
transformers==4.53.3

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

0.6623544245383161

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.46202) has done: 'I remove the dependency on the missing external `/kaggle/input/multi-scale-bert-aes` script (which is causing the pipeline to crash) and replace it with an in-notebook predictor that works in the provided Kaggle environment. To keep core logic simple and stable, I use a sentence-transformer to embed `full_text` and then fit a lightweight regressor on train, rounding/clipping predictions to the required 1–6 integer scores. This also eliminate the non-numeric “pred file parsing” issue and guarantee a valid `submission.csv` is written. The submission be aligned to `essay_id` order from `test.csv` and use the exact required column names.'

# 9. Code solution

## === cell 0
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

print("Python:", sys.version)
print("Working dir:", os.getcwd())
print("Listing /kaggle/input (top-level):", os.listdir("/kaggle/input")[:10])



## === cell 1
TEST_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
test_df = pd.read_csv(TEST_PATH)

test_df["full_text"] = (
    test_df["full_text"].astype(str).fillna("").str.replace("\n", " ", regex=False)
)

test_df.head()



## === cell 2
TRAIN_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
train_df = pd.read_csv(TRAIN_PATH)

train_df["full_text"] = (
    train_df["full_text"].astype(str).fillna("").str.replace("\n", " ", regex=False)
)

train_df["score"] = pd.to_numeric(train_df["score"], errors="coerce").astype(int)

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)
print(train_df[["essay_id", "score"]].head())



## === cell 3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

np.random.seed(42)

train_texts = train_df["full_text"].tolist()
test_texts = test_df["full_text"].tolist()
y_train = train_df["score"].values.astype(float)

tfidf = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)

X_train = tfidf.fit_transform(train_texts)
X_test = tfidf.transform(test_texts)

reg = Ridge(alpha=1.0, random_state=42)
reg.fit(X_train, y_train)

pred_cont = reg.predict(X_test).astype(float)

df_pred = pd.DataFrame({"label": np.zeros(len(test_df), dtype=int), "pred": pred_cont})
df_pred.head()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    114             try:
--> 115                 coef, info = sp_linalg.cg(C, y_column, tol=tol, atol="legacy")
    116             except TypeError:

TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_56/4255851727.py in <cell line: 0>()
     25 
     26 reg = Ridge(alpha=1.0, random_state=42)
---> 27 reg.fit(X_train, y_train)
     28 
     29 pred_cont = reg.predict(X_test).astype(float)

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

## === cell 4
df_pred["pred"] = pd.to_numeric(df_pred["pred"], errors="coerce")

if df_pred["pred"].isna().any():
    fill_val = float(np.mean(y_train))
    df_pred["pred"] = df_pred["pred"].fillna(fill_val)

df_pred["pred_1to6"] = np.rint(df_pred["pred"]).astype(int).clip(1, 6)

df_pred[["pred", "pred_1to6"]].head()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1871843818.py in <cell line: 0>()
----> 1 df_pred["pred"] = pd.to_numeric(df_pred["pred"], errors="coerce")
      2 
      3 if df_pred["pred"].isna().any():
      4     fill_val = float(np.mean(y_train))
      5     df_pred["pred"] = df_pred["pred"].fillna(fill_val)

NameError: name 'df_pred' is not defined

## === cell 5
if len(df_pred) != len(test_df):
    raise ValueError(f"Row mismatch: test={len(test_df)} preds={len(df_pred)}")

submission = pd.DataFrame(
    {"essay_id": test_df["essay_id"].values, "score": df_pred["pred_1to6"].values}
)
submission["score"] = submission["score"].clip(1, 6).astype(int)

sub_path = "/kaggle/working/submission.csv"
submission.to_csv(sub_path, index=False)

print("Wrote submission:", sub_path)
print(submission.head())
print("Submission shape:", submission.shape)
print("Score value counts:\n", submission["score"].value_counts().sort_index())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3948499443.py in <cell line: 0>()
----> 1 if len(df_pred) != len(test_df):
      2     raise ValueError(f"Row mismatch: test={len(test_df)} preds={len(df_pred)}")
      3 
      4 submission = pd.DataFrame(
      5     {"essay_id": test_df["essay_id"].values, "score": df_pred["pred_1to6"].values}

NameError: name 'df_pred' is not defined
