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

# 5. Target score

0.7789778763185116

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The timeout is dominated by repeatedly converting large sparse TF‑IDF folds into `DMatrix` inside the CV loop and re-materializing validation matrices; this adds heavy overhead on top of 4×1200 boosting rounds. I keep the exact same model/training logic, but speed it up by (1) building a single `DMatrix` for the full training set once and slicing it per-fold (equivalent), (2) using XGBoost’s `QuantileDMatrix` with `tree_method=hist` to avoid repeated CSR→DMatrix work and reduce memory/CPU overhead, and (3) reusing the prebuilt test matrix. These changes preserve the same algorithm, parameters, folds, and prediction math (only negligible FP differences possible), while cutting substantial per-fold setup time.'
- What this solution (achieved 0.0) has done: 'The timeout is most likely dominated by repeatedly building XGBoost QuantileDMatrix objects for each fold (and again for test), plus unnecessary overhead from the custom Python tokenizer in TF-IDF. I keep the exact same TF‑IDF features (same token regex and ngrams) but switch to scikit-learn’s built-in `token_pattern` to avoid Python-level tokenization overhead. For XGBoost, I build one QuantileDMatrix for the full training set and use `slice()` to create per-fold train/valid matrices without re-quantizing each time, which preserves the training logic and results while removing a large repeated cost. I also build the test DMatrix once as before and keep all model params, CV, boosting rounds, and prediction logic unchanged.'

# 9. Code solution

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

use_quantile = True
_has_device_qdm = hasattr(xgb, "DeviceQuantileDMatrix")


def _dmatrix(X, y_arr=None):
    if use_quantile:
        if _has_device_qdm:
            if y_arr is None:
                return xgb.DeviceQuantileDMatrix(X, nthread=NTHREAD)
            return xgb.DeviceQuantileDMatrix(X, label=y_arr, nthread=NTHREAD)
        else:
            if y_arr is None:
                return xgb.QuantileDMatrix(X, nthread=NTHREAD)
            return xgb.QuantileDMatrix(X, label=y_arr, nthread=NTHREAD)
    else:
        if y_arr is None:
            return xgb.DMatrix(X, nthread=NTHREAD)
        return xgb.DMatrix(X, label=y_arr, nthread=NTHREAD)


if use_quantile:
    dtrain_full = _dmatrix(X_train, y)
else:
    dtrain_full = None  # unused

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
    if use_quantile:
        dtr = dtrain_full.slice(tr_idx)
        dva = dtrain_full.slice(va_idx)
    else:
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

    proba_va = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
    fold_pred = (proba_va @ cls_idx).astype(np.float32, copy=False)
    np.nan_to_num(
        fold_pred, copy=False, nan=0.0, posinf=float(n_classes - 1), neginf=0.0
    )
    oof_pred[va_idx] = fold_pred

oof_rounded = np.clip(np.rint(oof_pred), 0, n_classes - 1).astype(int)
oof_score = le.inverse_transform(oof_rounded)
qwk = cohen_kappa_score(y_raw, oof_score, weights="quadratic")
print("OOF QWK:", qwk)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/829607956.py in <cell line: 0>()
     57 for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
     58     if use_quantile:
---> 59         dtr = dtrain_full.slice(tr_idx)
     60         dva = dtrain_full.slice(va_idx)
     61     else:

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in slice(self, rindex, allow_groups)
   1256         res.handle = ctypes.c_void_p()
   1257         rindex = _maybe_np_slice(rindex, dtype=np.int32)
-> 1258         _check_call(
   1259             _LIB.XGDMatrixSliceDMatrixEx(
   1260                 self.handle,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [06:57:58] /workspace/src/data/iterative_dmatrix.h:88: Slicing DMatrix is not supported for Quantile DMatrix.
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3effba) [0x7fff5ab63fba]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3ff7ab) [0x7fff5ab737ab]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGDMatrixSliceDMatrixEx+0x146) [0x7fff5a8d4206]
  [bt] (3) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (4) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (5) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]
  [bt] (6) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7ffff63bbc8e]
  [bt] (7) /usr/bin/python3(_PyObject_MakeTpCall+0x27c) [0x52f85c]
  [bt] (8) /usr/bin/python3(_PyEval_EvalFrameDefault+0x6bc) [0x53da0c]



## === cell 3
dtest = _dmatrix(X_test)

avg_pred = np.zeros((X_test.shape[0],), dtype=np.float32)
for m in models:
    proba = m.predict(dtest, iteration_range=(0, m.best_iteration + 1))
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
