# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
joblib==1.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
tqdm==4.67.1
xgboost==2.0.3

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

# 5. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(train_path), f"Missing: {train_path}"
assert os.path.exists(test_path), f"Missing: {test_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"

try:
    train_df = pd.read_csv(train_path, engine="pyarrow")
    test_df = pd.read_csv(test_path, engine="pyarrow")
    sample_sub = pd.read_csv(sample_sub_path, engine="pyarrow")
except Exception:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)

print(train_df.shape, test_df.shape, sample_sub.shape)
print(train_df.columns.tolist())
print(test_df.columns.tolist())
print(sample_sub.columns.tolist())




## === cell 1
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    print("Applied sklearnex patch for acceleration.")
except Exception as e:
    print("sklearnex patch not applied:", repr(e))

import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score

NTHREAD = os.cpu_count() or 4
if NTHREAD > 16:
    NTHREAD = 16

for df, name in [(train_df, "train"), (test_df, "test")]:
    if "full_text" not in df.columns or "essay_id" not in df.columns:
        raise ValueError(f"{name} is missing required columns.")
    df["full_text"] = df["full_text"].fillna("").astype(str)
    df["essay_id"] = df["essay_id"].astype(str)

if "score" not in train_df.columns:
    raise ValueError("train is missing required column: score")

le = LabelEncoder()
y_raw = train_df["score"].astype(int).values
y = le.fit_transform(y_raw)  # classes correspond to scores 1..6

n_classes = len(le.classes_)
print("Encoded classes:", le.classes_, "n_classes:", n_classes)

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=60000,
    lowercase=True,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w+\b",
    strip_accents=None,
    dtype=np.float32,  # negligible FP diffs allowed
)

train_text = train_df["full_text"].to_numpy()
test_text = test_df["full_text"].to_numpy()

X_train = tfidf.fit_transform(train_text)
X_test = tfidf.transform(test_text)

X_train = X_train.tocsr(copy=False)
X_test = X_test.tocsr(copy=False)
X_train.sort_indices()
X_test.sort_indices()

print("TFIDF shapes:", X_train.shape, X_test.shape)

del train_text, test_text




## === cell 2
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

models = []
oof_pred = np.zeros((X_train.shape[0],), dtype=np.float32)

early_stopping_rounds = 50

params = {
    "objective": "multi:softprob",
    "num_class": n_classes,
    "eval_metric": "mlogloss",
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "tree_method": "hist",
    "seed": RANDOM_STATE,
    "nthread": NTHREAD,
    "predictor": "cpu_predictor",
}

num_boost_round = 1200
cls_idx = np.arange(n_classes, dtype=np.float32)

use_quantile = False


def _dmatrix(X, y_arr=None):
    if y_arr is None:
        return xgb.DMatrix(X, nthread=NTHREAD)
    return xgb.DMatrix(X, label=y_arr, nthread=NTHREAD)


for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
    X_tr, y_tr = X_train[tr_idx], y[tr_idx]
    X_va, y_va = X_train[va_idx], y[va_idx]

    dtr = _dmatrix(X_tr, y_tr)
    dva = _dmatrix(X_va, y_va)

    model = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=num_boost_round,
        evals=[(dva, "valid")],
        verbose_eval=200,
        early_stopping_rounds=early_stopping_rounds,
    )
    models.append(model)

    best_iter = model.best_iteration + 1
    proba_va = model.predict(dva, iteration_range=(0, best_iter))
    fold_pred = (proba_va @ cls_idx).astype(np.float32, copy=False)
    np.nan_to_num(
        fold_pred, copy=False, nan=0.0, posinf=float(n_classes - 1), neginf=0.0
    )
    oof_pred[va_idx] = fold_pred

oof_rounded = np.clip(np.rint(oof_pred), 0, n_classes - 1).astype(int)
oof_score = le.inverse_transform(oof_rounded)
qwk = cohen_kappa_score(y_raw, oof_score, weights="quadratic")
print("OOF QWK:", qwk)




## === cell 3
dtest = _dmatrix(X_test)

avg_pred = np.zeros((X_test.shape[0],), dtype=np.float32)
for m in models:
    best_iter = m.best_iteration + 1
    proba = m.predict(dtest, iteration_range=(0, best_iter))
    fold_pred = (proba @ cls_idx).astype(np.float32, copy=False)
    np.nan_to_num(
        fold_pred, copy=False, nan=0.0, posinf=float(n_classes - 1), neginf=0.0
    )
    avg_pred += fold_pred

avg_pred /= max(len(models), 1)

pred_cls = np.clip(np.rint(avg_pred), 0, n_classes - 1).astype(int)
pred_score = le.inverse_transform(pred_cls).astype(int)
pred_score = np.clip(pred_score, 1, 6)

submission_pred = pd.DataFrame(
    {"essay_id": test_df["essay_id"].to_numpy(), "score": pred_score}
)
print(submission_pred.head())
print(submission_pred.shape)




## === cell 4
submission = sample_sub[["essay_id"]].merge(submission_pred, on="essay_id", how="left")

if submission["score"].isna().any():
    fill_val = int(np.median(train_df["score"].values))
    submission["score"] = submission["score"].fillna(fill_val).astype(int)

submission["score"] = submission["score"].astype(int)
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(submission.head())
print(submission["score"].value_counts().sort_index())
