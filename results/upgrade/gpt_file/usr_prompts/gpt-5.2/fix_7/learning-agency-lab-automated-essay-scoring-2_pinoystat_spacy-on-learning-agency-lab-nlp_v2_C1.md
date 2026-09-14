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

0.0108952342277427

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'The timeout is dominated by fitting TF‑IDF on 139k long documents with a very large feature space and then training 5 XGBoost models with 1200 boosting rounds each. I keep the exact same feature extraction and training logic, but reduce overhead by (1) using a faster tokenizer path in `TfidfVectorizer` (still identical token pattern), (2) prebuilding the XGBoost `QuantileDMatrix` once for the full train/test and slicing it per fold to avoid repeated sparse→DMatrix conversion work, and (3) enabling XGBoost’s built-in training cache (`cache_prefix`) to reuse binning across folds. These changes preserve the same model, parameters, and evaluation semantics while cutting repeated preprocessing costs that commonly cause the 10‑minute timeout.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_PATH), f"Missing: {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"Missing: {TEST_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"

train_df = pd.read_csv(
    TRAIN_PATH,
    usecols=["essay_id", "full_text", "score"],
    dtype={"essay_id": "string", "full_text": "string", "score": "int8"},
)
test_df = pd.read_csv(
    TEST_PATH,
    usecols=["essay_id", "full_text"],
    dtype={"essay_id": "string", "full_text": "string"},
)
sample_sub = pd.read_csv(
    SAMPLE_SUB_PATH,
    usecols=["essay_id", "score"],
    dtype={"essay_id": "string", "score": "int8"},
)

print(train_df.shape, test_df.shape, sample_sub.shape)
print(train_df.columns.tolist())
print(test_df.columns.tolist())



## === cell 1
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb

DEFAULT_NJOBS = os.cpu_count() or 4
N_JOBS = min(DEFAULT_NJOBS, 8)  # cap to avoid slowdown on shared Kaggle CPUs

os.environ.setdefault("OMP_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("MKL_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(N_JOBS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(N_JOBS))

train_df["full_text"] = train_df["full_text"].fillna("")
test_df["full_text"] = test_df["full_text"].fillna("")

X_text = train_df["full_text"].to_numpy(dtype=object, copy=False)
y = train_df["score"].to_numpy(dtype=np.int32, copy=False)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

fold_indices = list(skf.split(np.zeros(len(y), dtype=np.uint8), y))



## === cell 2
tfidf = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=200000,
    sublinear_tf=True,
)

X_train_tfidf = tfidf.fit_transform(X_text)
X_test_tfidf = tfidf.transform(test_df["full_text"].to_numpy(dtype=object, copy=False))

xgb_params = dict(
    objective="reg:squarederror",
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    tree_method="hist",
    nthread=N_JOBS,
    verbosity=0,
    seed=RANDOM_STATE,
    cache_prefix="/kaggle/working/xgb_cache",
)

NUM_BOOST_ROUND = 1200  # identical to previous n_estimators

oof_pred = np.zeros(len(train_df), dtype=np.float32)
models = []

use_quantile = True  # keep identical behavior


def _make_matrix(X, label=None):
    if use_quantile:
        try:
            return xgb.QuantileDMatrix(X, label=label)
        except Exception:
            return xgb.DMatrix(X, label=label)
    return xgb.DMatrix(X, label=label)


dtrain_full = _make_matrix(X_train_tfidf, label=y)
dtest = _make_matrix(X_test_tfidf, label=None)

for fold, (tr_idx, va_idx) in enumerate(fold_indices, 1):
    dtr = dtrain_full.slice(tr_idx)
    dva = dtrain_full.slice(va_idx)

    booster = xgb.train(
        params=xgb_params,
        dtrain=dtr,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dva, "valid")],
        verbose_eval=False,
    )
    pred_va = booster.predict(dva).astype(np.float32, copy=False)

    oof_pred[va_idx] = pred_va
    models.append(booster)

    pred_va_int = np.clip(np.rint(pred_va), 1, 6).astype(np.int32, copy=False)
    kappa = cohen_kappa_score(y[va_idx], pred_va_int, weights="quadratic")
    print(f"Fold {fold} QWK: {kappa:.5f}")

oof_pred_int = np.clip(np.rint(oof_pred), 1, 6).astype(np.int32, copy=False)
oof_qwk = cohen_kappa_score(y, oof_pred_int, weights="quadratic")
print(f"OOF QWK: {oof_qwk:.5f}")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/1702996051.py in <cell line: 0>()
     58 
     59 for fold, (tr_idx, va_idx) in enumerate(fold_indices, 1):
---> 60     dtr = dtrain_full.slice(tr_idx)
     61     dva = dtrain_full.slice(va_idx)
     62 

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

XGBoostError: [05:35:38] /workspace/src/data/iterative_dmatrix.h:88: Slicing DMatrix is not supported for Quantile DMatrix.
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3effba) [0x7f1b4d61ffba]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3ff7ab) [0x7f1b4d62f7ab]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGDMatrixSliceDMatrixEx+0x146) [0x7f1b4d390206]
  [bt] (3) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7f1bc02c4e2e]
  [bt] (4) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7f1bc02c1493]
  [bt] (5) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7f1bc02d44d8]
  [bt] (6) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7f1bc02d3c8e]
  [bt] (7) /usr/bin/python3(_PyObject_MakeTpCall+0x27c) [0x52f85c]
  [bt] (8) /usr/bin/python3(_PyEval_EvalFrameDefault+0x6bc) [0x53da0c]



## === cell 3
test_pred = np.zeros(X_test_tfidf.shape[0], dtype=np.float32)
for booster in models:
    test_pred += booster.predict(dtest).astype(np.float32, copy=False)
test_pred /= np.float32(len(models))
test_pred_int = np.clip(np.rint(test_pred), 1, 6).astype(np.int32, copy=False)



## === cell 4
sub = sample_sub[["essay_id"]].copy()

pred_s = pd.Series(
    test_pred_int, index=test_df["essay_id"].astype("string"), name="score"
)
sub["score"] = sub["essay_id"].astype("string").map(pred_s)

sub["score"] = sub["score"].fillna(3).astype(int)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {sub.shape}")
print(sub.head())
