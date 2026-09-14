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

3.14

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
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

0.7925148136739251

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
TRAIN_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
TEST_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
OUTPUT_PATH = "/kaggle/working/"

NUM_LABELS = 6
RANDOM_STATE = 42


## === cell 1
import os
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, accuracy_score

warnings.filterwarnings("ignore")


## === cell 2
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

assert {"essay_id", "full_text", "score"}.issubset(train_df.columns)
assert {"essay_id", "full_text"}.issubset(test_df.columns)


## === cell 3
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["full_text"].astype(str).values,
    train_df["score"].astype(int).values,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=train_df["score"],
)

vectorizer = TfidfVectorizer(
    max_features=50000, ngram_range=(1, 2), stop_words="english", dtype=np.float32
)

X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)

ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
ridge.fit(X_train, train_labels)

val_pred_cont = ridge.predict(X_val)
val_pred = np.rint(val_pred_cont).astype(int)
val_pred = np.clip(val_pred, 1, NUM_LABELS)

val_qwk = cohen_kappa_score(val_labels, val_pred, weights="quadratic")
val_acc = accuracy_score(val_labels, val_pred)
print(f"Validation QWK: {val_qwk:.5f}")
print(f"Validation Accuracy: {val_acc:.5f}")


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
/tmp/ipykernel_55/690026539.py in <cell line: 0>()
     18 # Ridge regression (continuous output, later rounded)
     19 ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
---> 20 ridge.fit(X_train, train_labels)
     21 
     22 # Validation predictions

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
full_texts = train_df["full_text"].astype(str).values
full_labels = train_df["score"].astype(int).values

X_full = vectorizer.fit_transform(full_texts)
ridge.fit(X_full, full_labels)


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    114             try:
--> 115                 coef, info = sp_linalg.cg(C, y_column, tol=tol, atol="legacy")
    116             except TypeError:

TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3652823380.py in <cell line: 0>()
      4 
      5 X_full = vectorizer.fit_transform(full_texts)
----> 6 ridge.fit(X_full, full_labels)

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

## === cell 5
test_texts = test_df["full_text"].astype(str).values
X_test = vectorizer.transform(test_texts)

test_pred_cont = ridge.predict(X_test)
test_pred = np.rint(test_pred_cont).astype(int)
test_pred = np.clip(test_pred, 1, NUM_LABELS)

submission_df = pd.DataFrame({"essay_id": test_df["essay_id"], "score": test_pred})


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3154217972.py in <cell line: 0>()
      3 X_test = vectorizer.transform(test_texts)
      4 
----> 5 test_pred_cont = ridge.predict(X_test)
      6 test_pred = np.rint(test_pred_cont).astype(int)
      7 test_pred = np.clip(test_pred, 1, NUM_LABELS)

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
submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/706601844.py in <cell line: 0>()
      1 # Write submission
      2 submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
----> 3 submission_df.to_csv(submission_path, index=False)
      4 print(f"Submission written to {submission_path}")

NameError: name 'submission_df' is not defined
