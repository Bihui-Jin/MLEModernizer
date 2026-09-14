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
DEFAULT_NJOBS = os.cpu_count() or 4
N_JOBS = min(DEFAULT_NJOBS, 8)  # keep original cap logic

os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)

try:
    from sklearnex import patch_sklearn

    patch_sklearn(verbose=False)
except Exception:
    pass

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
import xgboost as xgb

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
    dtype=np.float32,
    token_pattern=r"(?u)\S+",
)

X_train_tfidf = tfidf.fit_transform(X_text)
X_test_tfidf = tfidf.transform(test_df["full_text"].to_numpy(dtype=object, copy=False))

X_train_tfidf = X_train_tfidf.tocsr(copy=False)
X_test_tfidf = X_test_tfidf.tocsr(copy=False)

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
    deterministic_histogram=True,
    cache_prefix="/kaggle/working/xgb_cache",
)

NUM_BOOST_ROUND = 1200  # identical to previous n_estimators

oof_pred = np.zeros(len(train_df), dtype=np.float32)
models = []

dtest = xgb.DMatrix(X_test_tfidf)

for fold, (tr_idx, va_idx) in enumerate(fold_indices, 1):
    X_tr = X_train_tfidf[tr_idx]
    y_tr = y[tr_idx]
    X_va = X_train_tfidf[va_idx]
    y_va = y[va_idx]

    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)

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
    kappa = cohen_kappa_score(y_va, pred_va_int, weights="quadratic")
    print(f"Fold {fold} QWK: {kappa:.5f}")

oof_pred_int = np.clip(np.rint(oof_pred), 1, 6).astype(np.int32, copy=False)
oof_qwk = cohen_kappa_score(y, oof_pred_int, weights="quadratic")
print(f"OOF QWK: {oof_qwk:.5f}")



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
