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

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

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

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_PATH)

assert {"essay_id", "full_text", "score"}.issubset(train_df.columns)
assert {"essay_id", "full_text"}.issubset(test_df.columns)
assert {"essay_id", "score"}.issubset(sample_sub.columns)

train_df["full_text"] = train_df["full_text"].fillna("")
test_df["full_text"] = test_df["full_text"].fillna("")

y = train_df["score"].astype(np.int32).values
X_text = train_df["full_text"].values
X_text_test = test_df["full_text"].values

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)
print("Score distribution:", pd.Series(y).value_counts().sort_index().to_dict())




## === cell 2
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    strip_accents="unicode",
    lowercase=True,
    sublinear_tf=True,
    max_features=120000,
    dtype=np.float32,
)

X_all = tfidf.fit_transform(X_text)
X_test = tfidf.transform(X_text_test)

print("TF-IDF shapes:", X_all.shape, X_test.shape)
print("TF-IDF dtype:", X_all.dtype)




## === cell 3
def qwk(y_true, y_pred_int):
    return cohen_kappa_score(y_true, y_pred_int, weights="quadratic")


skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

oof_pred = np.zeros(len(train_df), dtype=np.float32)

cpu_n = os.cpu_count() or 4
xgb_nthread = min(8, cpu_n)  # keep strong parallelism but avoid oversubscription

xgb_params = dict(
    eta=0.05,  # learning_rate
    max_depth=6,
    min_child_weight=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    alpha=0.0,  # reg_alpha
    lambda_=1.0,  # reg_lambda
    objective="reg:squarederror",
    tree_method="hist",
    seed=RANDOM_STATE,
    nthread=xgb_nthread,
    verbosity=0,
)

num_boost_round = 1200

models = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y), 1):
    X_tr, X_va = X_all[tr_idx], X_all[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)

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
    fold_qwk = qwk(y_va, va_pred_int)
    print(f"Fold {fold} QWK (naive round): {fold_qwk:.5f}")

dtest = xgb.DMatrix(X_test)
test_pred_sum = np.zeros(X_test.shape[0], dtype=np.float32)
for booster in models:
    te_pred = booster.predict(dtest).astype(np.float32, copy=False)
    test_pred_sum += te_pred
test_pred_mean = test_pred_sum / np.float32(len(models))

print("OOF pred range:", float(oof_pred.min()), float(oof_pred.max()))
print("Test pred range:", float(test_pred_mean.min()), float(test_pred_mean.max()))




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
    th = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)

    best_th = th.copy()
    best_score = qwk(y_true, apply_thresholds(pred, best_th))

    for it in range(n_iter):
        improved = False
        for j in range(5):
            candidates = best_th[j] + step * np.array(
                [-2, -1, 0, 1, 2], dtype=np.float32
            )
            for cand in candidates:
                th_try = best_th.copy()
                th_try[j] = cand
                if not (th_try[0] < th_try[1] < th_try[2] < th_try[3] < th_try[4]):
                    continue
                if th_try[0] < 1.0 or th_try[4] > 6.0:
                    continue
                score_try = qwk(y_true, apply_thresholds(pred, th_try))
                if score_try > best_score + 1e-7:
                    best_score = score_try
                    best_th = th_try
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-3:
                break

    return best_th, best_score


best_th, best_oof_qwk = optimize_thresholds(y, oof_pred, n_iter=40, step=0.1)
print("Best thresholds:", best_th)
print("OOF QWK (optimized thresholds):", float(best_oof_qwk))




## === cell 5
test_pred_int = apply_thresholds(test_pred_mean, best_th).astype(int)
test_pred_int = np.clip(test_pred_int, 1, 6)

sub = pd.DataFrame({"essay_id": test_df["essay_id"].values, "score": test_pred_int})

assert sub.shape[0] == test_df.shape[0]
assert sub["essay_id"].isna().sum() == 0
assert sub["score"].between(1, 6).all()

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Score value counts:", sub["score"].value_counts().sort_index().to_dict())
