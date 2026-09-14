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

0.7879701644258821

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The timeout is dominated by slow Python-level feature engineering (multiple `Series.apply` over 139k long texts), repeated DMatrix construction for the test set inside each fold, and single-threaded CPU usage in both TF-IDF and XGBoost. I replace the per-row `apply` computations with a single compiled-regex pass per text (provably equivalent counts/ratios) while keeping the same numeric features, and I precompute the test `DMatrix` once and reuse it across folds. I also enable full CPU parallelism (`n_jobs`/`nthread`) for TF-IDF and XGBoost without changing the algorithm, and ensure sparse matrices use efficient dtypes where safe. These changes preserve the exact core pipeline and evaluation semantics, but remove large constant-factor overheads that cause the 10-minute timeout.'
- What this solution (achieved 0.0) has done: 'The main timeout driver is `unique_word_ratio_train_test`, which explodes every token in train+test (~150k essays) into a huge intermediate Series and then does two groupbys; this is asymptotically and memory-wise expensive. I replace that step with a provably equivalent per-document computation using `CountVectorizer` analyzers to count total tokens and unique tokens without exploding (same token pattern, lowercasing, and word definition), keeping the rest of feature extraction and model training identical. I also reduce XGBoost training overhead by using a single `QuantileDMatrix` for the full training set and slicing it per fold (same data, same algorithm), and avoid redundant DMatrix construction work. All changes are deterministic and preserve evaluation semantics (only negligible FP differences possible).'
- What this solution (achieved 0.0) has done: 'The crash comes from trying to `.slice()` an `xgb.QuantileDMatrix`, which XGBoost does not support; switching the fold matrices to regular `xgb.DMatrix` fixes execution while keeping the same model, features, and CV logic. To keep runtime reasonable and avoid extra overhead, the test `DMatrix` is still built once and reused across folds. I also set `n_jobs` on the TF-IDF vectorizer (same algorithm, just parallel tokenization) to help finish within the Kaggle time limit. The script now train all folds end-to-end and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

N_JOBS = int(os.environ.get("OMP_NUM_THREADS", "0")) or (os.cpu_count() or 4)
os.environ.setdefault("OMP_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("MKL_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(N_JOBS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(N_JOBS))

DATA_DIR_CANDIDATES = [
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
    "/kaggle/data/learning-agency-lab-automated-essay-scoring-2",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_file(filename: str) -> str:
    for base in DATA_DIR_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    for base in ["/kaggle/input", "/kaggle/data"]:
        for root, _, files in os.walk(base):
            if filename in files:
                return os.path.join(root, filename)
    raise FileNotFoundError(
        f"Could not find {filename} under /kaggle/input or /kaggle/data"
    )


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sample_path = _find_file("sample_submission.csv")

try:
    train_df = pd.read_csv(train_path, engine="pyarrow")
    test_df = pd.read_csv(test_path, engine="pyarrow")
    sample_sub = pd.read_csv(sample_path, engine="pyarrow")
except Exception:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_path)

assert {"essay_id", "full_text", "score"}.issubset(train_df.columns)
assert {"essay_id", "full_text"}.issubset(test_df.columns)
assert {"essay_id", "score"}.issubset(sample_sub.columns)

train_df["full_text"] = train_df["full_text"].fillna("")
test_df["full_text"] = test_df["full_text"].fillna("")

print("train:", train_df.shape, "test:", test_df.shape, "sample:", sample_sub.shape)
print("score distribution:\n", train_df["score"].value_counts().sort_index())



## === cell 1
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from scipy import sparse


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


_SENT_RE = re.compile(r"[.!?]+")
_PUNC_RE = re.compile(r"[.,;:!?]")
_UPPER_RE = re.compile(r"[A-Z]")
_DIGIT_RE = re.compile(r"\d")

_TOKEN_PATTERN = r"(?u)\b\w+\b"


def unique_word_ratio_train_test(text_train: pd.Series, text_test: pd.Series):
    tr = text_train.fillna("").astype(str)
    te = text_test.fillna("").astype(str)
    all_text = pd.concat([tr, te], axis=0, ignore_index=True)

    cv = CountVectorizer(
        lowercase=True,
        token_pattern=_TOKEN_PATTERN,
    )
    analyzer = cv.build_analyzer()

    n = len(all_text)
    total = np.empty(n, dtype=np.float32)
    nunique = np.empty(n, dtype=np.float32)

    for i, doc in enumerate(all_text.values):
        toks = analyzer(doc)
        total_i = len(toks)
        total[i] = total_i
        nunique[i] = len(set(toks)) if total_i else 0.0

    denom = total.copy()
    denom[denom == 0] = 1.0
    ratio = (nunique / denom).astype(np.float32, copy=False)

    n_tr = len(tr)
    return ratio[:n_tr], ratio[n_tr:]


def make_numeric_features(text_train: pd.Series, text_test: pd.Series):
    tr = text_train.fillna("").astype(str)
    te = text_test.fillna("").astype(str)

    char_len_tr = tr.str.len().to_numpy(dtype=np.float32)
    char_len_te = te.str.len().to_numpy(dtype=np.float32)

    word_count_tr = tr.str.count(r"\S+").to_numpy(dtype=np.float32)
    word_count_te = te.str.count(r"\S+").to_numpy(dtype=np.float32)

    sent_count_tr = tr.str.count(_SENT_RE).to_numpy(dtype=np.float32)
    sent_count_te = te.str.count(_SENT_RE).to_numpy(dtype=np.float32)

    denom_wc_tr = word_count_tr.copy()
    denom_wc_tr[denom_wc_tr == 0] = 1.0
    denom_wc_te = word_count_te.copy()
    denom_wc_te[denom_wc_te == 0] = 1.0

    avg_word_len_tr = (char_len_tr / denom_wc_tr).astype(np.float32, copy=False)
    avg_word_len_te = (char_len_te / denom_wc_te).astype(np.float32, copy=False)

    denom_cl_tr = char_len_tr.copy()
    denom_cl_tr[denom_cl_tr == 0] = 1.0
    denom_cl_te = char_len_te.copy()
    denom_cl_te[denom_cl_te == 0] = 1.0

    upper_ratio_tr = (
        tr.str.count(_UPPER_RE).to_numpy(dtype=np.float32) / denom_cl_tr
    ).astype(np.float32, copy=False)
    upper_ratio_te = (
        te.str.count(_UPPER_RE).to_numpy(dtype=np.float32) / denom_cl_te
    ).astype(np.float32, copy=False)

    digit_ratio_tr = (
        tr.str.count(_DIGIT_RE).to_numpy(dtype=np.float32) / denom_cl_tr
    ).astype(np.float32, copy=False)
    digit_ratio_te = (
        te.str.count(_DIGIT_RE).to_numpy(dtype=np.float32) / denom_cl_te
    ).astype(np.float32, copy=False)

    punc_ratio_tr = (
        tr.str.count(_PUNC_RE).to_numpy(dtype=np.float32) / denom_cl_tr
    ).astype(np.float32, copy=False)
    punc_ratio_te = (
        te.str.count(_PUNC_RE).to_numpy(dtype=np.float32) / denom_cl_te
    ).astype(np.float32, copy=False)

    uniq_word_ratio_tr, uniq_word_ratio_te = unique_word_ratio_train_test(tr, te)

    X_num_train = np.column_stack(
        [
            char_len_tr,
            word_count_tr,
            sent_count_tr,
            avg_word_len_tr,
            uniq_word_ratio_tr,
            upper_ratio_tr,
            digit_ratio_tr,
            punc_ratio_tr,
        ]
    ).astype(np.float32, copy=False)

    X_num_test = np.column_stack(
        [
            char_len_te,
            word_count_te,
            sent_count_te,
            avg_word_len_te,
            uniq_word_ratio_te,
            upper_ratio_te,
            digit_ratio_te,
            punc_ratio_te,
        ]
    ).astype(np.float32, copy=False)

    return X_num_train, X_num_test


y = train_df["score"].astype(np.int32).values
X_text_train = train_df["full_text"]
X_text_test = test_df["full_text"]

X_num_train, X_num_test = make_numeric_features(X_text_train, X_text_test)

num_scaler = RobustScaler()
X_num_train_sc = num_scaler.fit_transform(X_num_train).astype(np.float32, copy=False)
X_num_test_sc = num_scaler.transform(X_num_test).astype(np.float32, copy=False)

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    strip_accents="unicode",
    lowercase=True,
    sublinear_tf=True,
    max_features=60000,
    dtype=np.float32,
    n_jobs=N_JOBS,
)
X_tfidf_train = tfidf.fit_transform(X_text_train)
X_tfidf_test = tfidf.transform(X_text_test)

X_num_train_csr = sparse.csr_matrix(X_num_train_sc, dtype=np.float32)
X_num_test_csr = sparse.csr_matrix(X_num_test_sc, dtype=np.float32)

X_train = sparse.hstack(
    [X_tfidf_train, X_num_train_csr], format="csr", dtype=np.float32
)
X_test = sparse.hstack([X_tfidf_test, X_num_test_csr], format="csr", dtype=np.float32)

if not X_train.has_sorted_indices:
    X_train.sort_indices()
if not X_test.has_sorted_indices:
    X_test.sort_indices()

print("X_train:", X_train.shape, "X_test:", X_test.shape)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1088997790.py in <cell line: 0>()
    144 
    145 # Bugfix/perf: enable parallelism in TF-IDF tokenization; algorithm unchanged.
--> 146 tfidf = TfidfVectorizer(
    147     ngram_range=(1, 2),
    148     min_df=2,

TypeError: TfidfVectorizer.__init__() got an unexpected keyword argument 'n_jobs'

## === cell 2
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

oof_pred = np.zeros(train_df.shape[0], dtype=np.float32)
test_pred_folds = np.zeros((test_df.shape[0], 4), dtype=np.float32)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "seed": RANDOM_STATE,
    "nthread": N_JOBS,
    "verbosity": 0,
}

NUM_BOOST_ROUND = 1500
EARLY_STOPPING_ROUNDS = 100

fold_qwks = []

dte = xgb.DMatrix(X_test, nthread=N_JOBS)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
    dtr = xgb.DMatrix(X_train[tr_idx], label=y[tr_idx], nthread=N_JOBS)
    dva = xgb.DMatrix(X_train[va_idx], label=y[va_idx], nthread=N_JOBS)

    y_va = y[va_idx]

    bst = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=500,
    )

    best_end = bst.best_iteration + 1

    va_pred = bst.predict(dva, iteration_range=(0, best_end))
    oof_pred[va_idx] = va_pred

    va_pred_round = np.clip(np.rint(va_pred).astype(np.int32, copy=False), 1, 6)
    qwk = quadratic_weighted_kappa(y_va, va_pred_round)
    fold_qwks.append(qwk)
    print(f"Fold {fold} QWK: {qwk:.6f} (best_iteration={bst.best_iteration})")

    te_pred = bst.predict(dte, iteration_range=(0, best_end))
    test_pred_folds[:, fold - 1] = te_pred

oof_pred_round = np.clip(np.rint(oof_pred).astype(np.int32, copy=False), 1, 6)
oof_qwk = quadratic_weighted_kappa(y, oof_pred_round)
print(f"OOF QWK: {oof_qwk:.6f}")
print("Fold QWKs:", fold_qwks, "mean:", float(np.mean(fold_qwks)))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1603441901.py in <cell line: 0>()
     27 # Bugfix: QuantileDMatrix does NOT support slicing; use standard DMatrix per fold.
     28 # This preserves the same data and training API semantics, just fixes the runtime error.
---> 29 dte = xgb.DMatrix(X_test, nthread=N_JOBS)
     30 
     31 for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):

NameError: name 'X_test' is not defined

## === cell 3
test_pred_mean = test_pred_folds.mean(axis=1)
test_score = np.clip(np.rint(test_pred_mean).astype(np.int32, copy=False), 1, 6)

sub = pd.DataFrame(
    {
        "essay_id": test_df["essay_id"].astype(str).values,
        "score": test_score.astype(int, copy=False),
    }
)

sub = sub[["essay_id", "score"]]

assert sub.shape[0] == test_df.shape[0]
assert sub["essay_id"].isna().sum() == 0
assert sub["score"].between(1, 6).all()

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
