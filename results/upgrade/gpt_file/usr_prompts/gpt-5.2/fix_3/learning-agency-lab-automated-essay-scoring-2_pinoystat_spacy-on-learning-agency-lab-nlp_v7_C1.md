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

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'The timeout is dominated by slow Python-level feature engineering (multiple `Series.apply` over 139k long texts), repeated DMatrix construction for the test set inside each fold, and single-threaded CPU usage in both TF-IDF and XGBoost. I replace the per-row `apply` computations with a single compiled-regex pass per text (provably equivalent counts/ratios) while keeping the same numeric features, and I precompute the test `DMatrix` once and reuse it across folds. I also enable full CPU parallelism (`n_jobs`/`nthread`) for TF-IDF and XGBoost without changing the algorithm, and ensure sparse matrices use efficient dtypes where safe. These changes preserve the exact core pipeline and evaluation semantics, but remove large constant-factor overheads that cause the 10-minute timeout.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

N_JOBS = int(os.environ.get("OMP_NUM_THREADS", "0")) or (os.cpu_count() or 4)

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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from scipy import sparse


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


_WORD_RE = re.compile(r"\b\w+\b")
_SENT_RE = re.compile(r"[.!?]+")
_PUNC_SET = set(".,;:!?")


def make_numeric_features(text_series: pd.Series) -> np.ndarray:
    t = text_series.fillna("").astype(str).to_numpy()

    n = t.shape[0]
    char_len = np.empty(n, dtype=np.float32)
    word_count = np.empty(n, dtype=np.float32)
    sent_count = np.empty(n, dtype=np.float32)
    avg_word_len = np.empty(n, dtype=np.float32)
    uniq_word_ratio = np.empty(n, dtype=np.float32)
    upper_ratio = np.empty(n, dtype=np.float32)
    digit_ratio = np.empty(n, dtype=np.float32)
    punc_ratio = np.empty(n, dtype=np.float32)

    for i, s in enumerate(t):
        L = len(s)
        char_len[i] = float(L)

        wc = len(s.split())
        word_count[i] = float(wc)

        sc = len(_SENT_RE.findall(s))
        sent_count[i] = float(sc)

        avg_word_len[i] = float(L / (wc if wc > 0 else 1.0))

        words = _WORD_RE.findall(s.lower())
        denom = len(words) if words else 1
        uniq_word_ratio[i] = float(len(set(words)) / denom)

        if L > 0:
            up = 0
            dg = 0
            pc = 0
            for c in s:
                if c.isupper():
                    up += 1
                if c.isdigit():
                    dg += 1
                if c in _PUNC_SET:
                    pc += 1
            invL = 1.0 / L
            upper_ratio[i] = float(up * invL)
            digit_ratio[i] = float(dg * invL)
            punc_ratio[i] = float(pc * invL)
        else:
            upper_ratio[i] = 0.0
            digit_ratio[i] = 0.0
            punc_ratio[i] = 0.0

    feats = np.vstack(
        [
            char_len,
            word_count,
            sent_count,
            avg_word_len,
            uniq_word_ratio,
            upper_ratio,
            digit_ratio,
            punc_ratio,
        ]
    ).T
    return feats


y = train_df["score"].astype(int).values
X_text_train = train_df["full_text"]
X_text_test = test_df["full_text"]

X_num_train = make_numeric_features(X_text_train)
X_num_test = make_numeric_features(X_text_test)

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
    n_jobs=N_JOBS,
)
X_tfidf_train = tfidf.fit_transform(X_text_train)
X_tfidf_test = tfidf.transform(X_text_test)

X_tfidf_train = X_tfidf_train.astype(np.float32)
X_tfidf_test = X_tfidf_test.astype(np.float32)

X_train = sparse.hstack(
    [X_tfidf_train, sparse.csr_matrix(X_num_train_sc, dtype=np.float32)], format="csr"
)
X_test = sparse.hstack(
    [X_tfidf_test, sparse.csr_matrix(X_num_test_sc, dtype=np.float32)], format="csr"
)

print("X_train:", X_train.shape, "X_test:", X_test.shape)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1724997324.py in <cell line: 0>()
    101 
    102 # Speed: enable parallel tokenization in TF-IDF; preserves exact TF-IDF semantics.
--> 103 tfidf = TfidfVectorizer(
    104     ngram_range=(1, 2),
    105     min_df=2,

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
}

NUM_BOOST_ROUND = 1500
EARLY_STOPPING_ROUNDS = 100

fold_qwks = []
models = []

dte = xgb.DMatrix(X_test)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)

    bst = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=200,
    )
    models.append(bst)

    va_pred = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
    oof_pred[va_idx] = va_pred

    va_pred_round = np.rint(va_pred).astype(int)
    va_pred_round = np.clip(va_pred_round, 1, 6)
    qwk = quadratic_weighted_kappa(y_va, va_pred_round)
    fold_qwks.append(qwk)
    print(f"Fold {fold} QWK: {qwk:.6f} (best_iteration={bst.best_iteration})")

    te_pred = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))
    test_pred_folds[:, fold - 1] = te_pred

oof_pred_round = np.rint(oof_pred).astype(int)
oof_pred_round = np.clip(oof_pred_round, 1, 6)
oof_qwk = quadratic_weighted_kappa(y, oof_pred_round)
print(f"OOF QWK: {oof_qwk:.6f}")
print("Fold QWKs:", fold_qwks, "mean:", float(np.mean(fold_qwks)))




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2242113696.py in <cell line: 0>()
     27 
     28 # Speed: build test DMatrix once (it is identical for all folds); preserves predictions exactly.
---> 29 dte = xgb.DMatrix(X_test)
     30 
     31 for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):

NameError: name 'X_test' is not defined

## === cell 3
test_pred_mean = test_pred_folds.mean(axis=1)
test_score = np.rint(test_pred_mean).astype(int)
test_score = np.clip(test_score, 1, 6)

sub = pd.DataFrame(
    {
        "essay_id": test_df["essay_id"].astype(str).values,
        "score": test_score.astype(int),
    }
)

sub = sub[["essay_id", "score"]]

assert sub.shape[0] == test_df.shape[0]
assert sub["essay_id"].isna().sum() == 0
assert sub["score"].between(1, 6).all()

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
