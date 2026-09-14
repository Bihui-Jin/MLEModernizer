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

0.7860272050602427

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

os.environ.setdefault("XGBOOST_NUM_THREADS", "16")

import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("Using paths:")
print("TRAIN_PATH:", TRAIN_PATH)
print("TEST_PATH :", TEST_PATH)
print("SAMPLE   :", SAMPLE_PATH)




## === cell 1
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    print("Enabled scikit-learn-intelex acceleration (sklearnex).")
except Exception as e:
    print("sklearnex not enabled:", repr(e))

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb

train_df = pd.read_csv(
    TRAIN_PATH,
    dtype={"essay_id": "string", "full_text": "string", "score": "int16"},
    engine="pyarrow",
)
test_df = pd.read_csv(
    TEST_PATH, dtype={"essay_id": "string", "full_text": "string"}, engine="pyarrow"
)
sample_sub = pd.read_csv(
    SAMPLE_PATH, dtype={"essay_id": "string", "score": "int16"}, engine="pyarrow"
)

assert {"essay_id", "full_text", "score"}.issubset(train_df.columns)
assert {"essay_id", "full_text"}.issubset(test_df.columns)
assert {"essay_id", "score"}.issubset(sample_sub.columns)

train_df["full_text"] = train_df["full_text"].fillna("")
test_df["full_text"] = test_df["full_text"].fillna("")

y = train_df["score"].astype(np.int32).to_numpy()
X_text = train_df["full_text"].to_numpy()
X_text_test = test_df["full_text"].to_numpy()

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)
print("Score distribution:", pd.Series(y).value_counts().sort_index().to_dict())




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ArrowInvalid                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/arrow_parser_wrapper.py in read(self)
    265         try:
--> 266             table = pyarrow_csv.read_csv(
    267                 self.src,

/usr/local/lib/python3.11/dist-packages/pyarrow/_csv.pyx in pyarrow._csv.read_csv()

/usr/local/lib/python3.11/dist-packages/pyarrow/_csv.pyx in pyarrow._csv.read_csv()

/usr/local/lib/python3.11/dist-packages/pyarrow/error.pxi in pyarrow.lib.pyarrow_internal_check_status()

/usr/local/lib/python3.11/dist-packages/pyarrow/error.pxi in pyarrow.lib.check_status()

ArrowInvalid: CSV parse error: Expected 3 columns, got 1: Now with getting to Venus. We would need a ship or rover that could with stand the pressure of t ...

The above exception was the direct cause of the following exception:

ParserError                               Traceback (most recent call last)
/tmp/ipykernel_11/2642086978.py in <cell line: 0>()
     13 
     14 # Speed: use pyarrow engine (when available) for faster CSV parsing; dtype unchanged => same values.
---> 15 train_df = pd.read_csv(
     16     TRAIN_PATH,
     17     dtype={"essay_id": "string", "full_text": "string", "score": "int16"},

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    624 
    625     with parser:
--> 626         return parser.read(nrows)
    627 
    628 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read(self, nrows)
   1909             try:
   1910                 # error: "ParserBase" has no attribute "read"
-> 1911                 df = self._engine.read()  # type: ignore[attr-defined]
   1912             except Exception:
   1913                 self.close()

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/arrow_parser_wrapper.py in read(self)
    271             )
    272         except pa.ArrowInvalid as e:
--> 273             raise ParserError(e) from e
    274 
    275         dtype_backend = self.kwds["dtype_backend"]

ParserError: CSV parse error: Expected 3 columns, got 1: Now with getting to Venus. We would need a ship or rover that could with stand the pressure of t ...

## === cell 2
cpu_n = os.cpu_count() or 4

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    lowercase=True,
    sublinear_tf=True,
    max_features=120000,
    dtype=np.float32,
    token_pattern=r"(?u)\b\w\w+\b",  # explicit default
    strip_accents=None,
)

X_all = tfidf.fit_transform(X_text).tocsr()
X_all.sort_indices()

X_test = tfidf.transform(X_text_test).tocsr()
X_test.sort_indices()

print("TF-IDF shapes:", X_all.shape, X_test.shape)
print("TF-IDF dtype:", X_all.dtype)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/742361169.py in <cell line: 0>()
     14 
     15 # Speed: keep CSR and ensure indices are sorted once; this accelerates downstream row slicing and XGBoost ingestion.
---> 16 X_all = tfidf.fit_transform(X_text).tocsr()
     17 X_all.sort_indices()
     18 

NameError: name 'X_text' is not defined

## === cell 3
def qwk(y_true, y_pred_int):
    return cohen_kappa_score(y_true, y_pred_int, weights="quadratic")


skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

oof_pred = np.zeros(len(train_df), dtype=np.float32)

xgb_nthread = min(cpu_n, 16)  # cap to avoid oversubscription on Kaggle CPUs

xgb_params = dict(
    eta=0.05,  # learning_rate
    max_depth=6,
    min_child_weight=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    alpha=0.0,  # reg_alpha
    **{"lambda": 1.0},  # reg_lambda; keep value identical to intent
    objective="reg:squarederror",
    tree_method="hist",
    seed=RANDOM_STATE,
    nthread=xgb_nthread,
    verbosity=0,
)

num_boost_round = 1200
models = []


def make_dmatrix(X, label=None):
    try:
        return xgb.QuantileDMatrix(X, label=label)
    except Exception:
        return xgb.DMatrix(X, label=label)


dtest = make_dmatrix(X_test)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y), 1):
    X_tr = X_all[tr_idx]
    X_va = X_all[va_idx]

    dtr = make_dmatrix(X_tr, label=y[tr_idx])
    dva = make_dmatrix(X_va, label=y[va_idx])

    booster = xgb.train(
        params=xgb_params,
        dtrain=dtr,
        num_boost_round=num_boost_round,
        evals=[(dva, "valid")],
        verbose_eval=False,
    )
    models.append(booster)

    va_pred = booster.predict(dva).astype(np.float32, copy=False)
    oof_pred[va_idx] = va_pred

    va_pred_int = np.clip(np.rint(va_pred), 1, 6).astype(np.int32, copy=False)
    fold_qwk = qwk(y[va_idx], va_pred_int)
    print(f"Fold {fold} QWK (naive round): {fold_qwk:.5f}")

    del X_tr, X_va, dtr, dva, va_pred, va_pred_int

test_pred_sum = np.zeros(X_test.shape[0], dtype=np.float32)

for booster in models:
    try:
        pred = booster.inplace_predict(X_test)
    except Exception:
        pred = booster.predict(dtest)
    test_pred_sum += np.asarray(pred, dtype=np.float32)

test_pred_mean = test_pred_sum / np.float32(len(models))

print("OOF pred range:", float(oof_pred.min()), float(oof_pred.max()))
print("Test pred range:", float(test_pred_mean.min()), float(test_pred_mean.max()))




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4070070720.py in <cell line: 0>()
      5 skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
      6 
----> 7 oof_pred = np.zeros(len(train_df), dtype=np.float32)
      8 
      9 xgb_nthread = min(cpu_n, 16)  # cap to avoid oversubscription on Kaggle CPUs

NameError: name 'train_df' is not defined

## === cell 4
def apply_thresholds(pred, th):
    th = np.asarray(th, dtype=np.float32)
    out = np.ones_like(pred, dtype=np.int32)
    out[pred > th[0]] = 2
    out[pred > th[1]] = 3
    out[pred > th[2]] = 4
    out[pred > th[3]] = 5
    out[pred > th[4]] = 6
    return out


def optimize_thresholds(y_true, pred, n_iter=30, step=0.05):
    y_true = np.asarray(y_true, dtype=np.int32)
    pred = np.asarray(pred, dtype=np.float32)

    best_th = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
    th_try = best_th.copy()

    best_score = qwk(y_true, apply_thresholds(pred, best_th))

    offsets = np.array([-2, -1, 0, 1, 2], dtype=np.float32)

    for _ in range(n_iter):
        improved = False
        for j in range(5):
            base = best_th[j]
            candidates = base + step * offsets
            for cand in candidates:
                th_try[:] = best_th
                th_try[j] = cand
                if not (th_try[0] < th_try[1] < th_try[2] < th_try[3] < th_try[4]):
                    continue
                if th_try[0] < 1.0 or th_try[4] > 6.0:
                    continue
                score_try = qwk(y_true, apply_thresholds(pred, th_try))
                if score_try > best_score + 1e-7:
                    best_score = score_try
                    best_th = th_try.copy()
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-3:
                break

    return best_th, best_score


best_th, best_oof_qwk = optimize_thresholds(y, oof_pred, n_iter=40, step=0.1)
print("Best thresholds:", best_th)
print("OOF QWK (optimized thresholds):", float(best_oof_qwk))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1387890581.py in <cell line: 0>()
     48 
     49 
---> 50 best_th, best_oof_qwk = optimize_thresholds(y, oof_pred, n_iter=40, step=0.1)
     51 print("Best thresholds:", best_th)
     52 print("OOF QWK (optimized thresholds):", float(best_oof_qwk))

NameError: name 'y' is not defined

## === cell 5
test_pred_int = apply_thresholds(test_pred_mean, best_th).astype(int)
test_pred_int = np.clip(test_pred_int, 1, 6)

sub = pd.DataFrame({"essay_id": test_df["essay_id"].to_numpy(), "score": test_pred_int})

assert sub.shape[0] == test_df.shape[0]
assert sub["essay_id"].isna().sum() == 0
assert sub["score"].between(1, 6).all()

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Score value counts:", sub["score"].value_counts().sort_index().to_dict())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1659222999.py in <cell line: 0>()
----> 1 test_pred_int = apply_thresholds(test_pred_mean, best_th).astype(int)
      2 test_pred_int = np.clip(test_pred_int, 1, 6)
      3 
      4 sub = pd.DataFrame({"essay_id": test_df["essay_id"].to_numpy(), "score": test_pred_int})
      5 

NameError: name 'test_pred_mean' is not defined
