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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
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

0.7984215469600132

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import cohen_kappa_score
import numpy as np

DATA_ROOT = Path("/kaggle/input/learning-agency-lab-automated-essay-scoring-2")
TRAIN_PATH = DATA_ROOT / "train.csv"
TEST_PATH = DATA_ROOT / "test.csv"
SUBMISSION_PATH = "submission.csv"



## === cell 1
df_train = pd.read_csv(TRAIN_PATH)

model = make_pipeline(
    TfidfVectorizer(
        max_features=60000, ngram_range=(1, 2), stop_words="english", min_df=2
    ),
    Ridge(alpha=1.0, random_state=42),
)

X_train = df_train["full_text"].fillna("")
y_train = df_train["score"].astype(float)
model.fit(X_train, y_train)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    114             try:
--> 115                 coef, info = sp_linalg.cg(C, y_column, tol=tol, atol="legacy")
    116             except TypeError:

TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3370703715.py in <cell line: 0>()
     15 X_train = df_train["full_text"].fillna("")
     16 y_train = df_train["score"].astype(float)
---> 17 model.fit(X_train, y_train)
     18 

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit(self, X, y, **fit_params)
    403             if self._final_estimator != "passthrough":
    404                 fit_params_last_step = fit_params_steps[self.steps[-1][0]]
--> 405                 self._final_estimator.fit(Xt, y, **fit_params_last_step)
    406 
    407         return self

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

## === cell 2
df_test = pd.read_csv(TEST_PATH)

raw_preds = model.predict(df_test["full_text"].fillna(""))
int_preds = np.rint(raw_preds).astype(int)
int_preds = np.clip(int_preds, 1, 6)  # ensure valid score range

df_test["score"] = int_preds

df_test[["essay_id", "score"]].to_csv(SUBMISSION_PATH, index=False)

print(f"Submission written to {SUBMISSION_PATH}")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3140793311.py in <cell line: 0>()
      3 
      4 # Predict raw scores, then round to nearest integer in [1, 6]
----> 5 raw_preds = model.predict(df_test["full_text"].fillna(""))
      6 int_preds = np.rint(raw_preds).astype(int)
      7 int_preds = np.clip(int_preds, 1, 6)  # ensure valid score range

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in predict(self, X, **predict_params)
    479         for _, name, transform in self._iter(with_final=False):
    480             Xt = transform.transform(Xt)
--> 481         return self.steps[-1][1].predict(Xt, **predict_params)
    482 
    483     @available_if(_final_estimator_has("fit_predict"))

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
