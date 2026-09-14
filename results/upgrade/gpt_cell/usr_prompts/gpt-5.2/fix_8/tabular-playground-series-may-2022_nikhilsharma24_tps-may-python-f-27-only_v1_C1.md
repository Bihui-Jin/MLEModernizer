# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.63094

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import string
import os

_ = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", None)



## === cell 1
from sklearn import (
    preprocessing,
)  # kept (even if unused) to preserve original imports/cell structure
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier  # kept to preserve original imports/cell structure
import optuna



## === cell 2
TRAIN_PATH = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-may-2022/test.csv"

train = pd.read_csv(
    TRAIN_PATH,
    usecols=["id", "f_27", "target"],
    dtype={"id": np.int64, "f_27": "string", "target": np.int8},
)
test = pd.read_csv(
    TEST_PATH,
    usecols=["id", "f_27"],
    dtype={"id": np.int64, "f_27": "string"},
)



## === cell 3
print(train.shape)
print(test.shape)



## === cell 4
from sklearn import model_selection

train["kfold"] = -1

kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=102)
for fold, (train_indicies, valid_indicies) in enumerate(kf.split(X=train)):
    train.loc[valid_indicies, "kfold"] = fold



## === cell 5
train.head()



## === cell 6
df_tr = train[["f_27", "target", "kfold"]].copy()
df_te = test[["f_27"]].copy()

print(df_tr.shape)
print(df_te.shape)



## === cell 7
pd.crosstab(index=df_tr["target"], columns=df_tr["kfold"])



## === cell 8
df_tr.dtypes



## === cell 9
df_tr.f_27




## === cell 10
def count_alpha(df: pd.DataFrame) -> pd.DataFrame:
    s = df["f_27"].astype("string").fillna("")
    letters = string.ascii_uppercase[:20]  # A..T

    arr = s.to_numpy(dtype="U")
    b = s.str.encode("ascii").to_numpy()

    out = np.zeros((len(df), len(letters)), dtype=np.int16)
    ordA = ord("A")
    for i, bi in enumerate(b):
        if not bi:
            continue
        v = np.frombuffer(bi, dtype=np.uint8) - ordA
        v = v[(v >= 0) & (v < 20)]
        if v.size:
            out[i] = np.bincount(v, minlength=20).astype(np.int16, copy=False)

    for j, x in enumerate(letters):
        df[f"count_{x}"] = out[:, j]

    df = df.drop("f_27", axis=1)
    return df




## === cell 11
print(df_tr.shape)
print(df_te.shape)



## === cell 12
df_tr.head()



## === cell 13
use_feature = [c for c in df_tr.columns if c not in ("target", "kfold")]



## === cell 14
import xgboost as xgb

df_tr = count_alpha(df_tr.copy())
df_te = count_alpha(df_te.copy())
use_feature = [c for c in df_tr.columns if c not in ("target", "kfold")]

X_all = df_tr[use_feature].to_numpy(dtype=np.float32, copy=False)
y_all = df_tr["target"].to_numpy(dtype=np.int8, copy=False)
folds = df_tr["kfold"].to_numpy(dtype=np.int8, copy=False)
X_test = df_te[use_feature].to_numpy(dtype=np.float32, copy=False)

fold_indices = {}
for f in range(10):
    fold_indices[f] = {
        "tr": np.flatnonzero(folds != f),
        "va": np.flatnonzero(folds == f),
    }

_has_gpu = False
try:
    _cfg = xgb.get_config()
    _dev = str(_cfg.get("device", "")).lower()
    _has_gpu = "cuda" in _dev
except Exception:
    _has_gpu = False

_FIT_VERBOSE = 0

_N_JOBS = int(os.environ.get("OMP_NUM_THREADS", "0") or "0")
if _N_JOBS <= 0:
    _N_JOBS = max(1, (os.cpu_count() or 2) - 1)

_DMatrix = xgb.QuantileDMatrix if (_has_gpu) else xgb.DMatrix

_dtrain_folds = {}
_dvalid_folds = {}
for f in range(10):
    tr_idx = fold_indices[f]["tr"]
    va_idx = fold_indices[f]["va"]
    _dtrain_folds[f] = _DMatrix(X_all[tr_idx], label=y_all[tr_idx])
    _dvalid_folds[f] = _DMatrix(X_all[va_idx], label=y_all[va_idx])

_dtest = _DMatrix(X_test)

_fold_for_optuna = 0
_dtrain0 = _dtrain_folds[_fold_for_optuna]
_dvalid0 = _dvalid_folds[_fold_for_optuna]
_yvalid0 = y_all[fold_indices[_fold_for_optuna]["va"]]


def run(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 7)

    params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
        seed=42,
        nthread=_N_JOBS,
    )
    if _has_gpu:
        params.update(tree_method="gpu_hist", predictor="gpu_predictor")
    else:
        params.update(tree_method="hist")

    booster = xgb.train(
        params=params,
        dtrain=_dtrain0,
        num_boost_round=7000,
        evals=[(_dvalid0, "valid")],
        early_stopping_rounds=300,
        verbose_eval=False,
    )

    preds_valid = booster.predict(
        _dvalid0, iteration_range=(0, booster.best_iteration + 1)
    )
    AUC = roc_auc_score(_yvalid0, preds_valid)
    return AUC


optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction="maximize")
study.optimize(run, n_trials=5)

best_params = study.best_params
_params_best = dict(
    objective="binary:logistic",
    eval_metric="auc",
    seed=42,
    nthread=_N_JOBS,
    **best_params,
)
if _has_gpu:
    _params_best.update(tree_method="gpu_hist", predictor="gpu_predictor")
else:
    _params_best.update(tree_method="hist")

_best_booster = xgb.train(
    params=_params_best,
    dtrain=_dtrain0,
    num_boost_round=7000,
    evals=[(_dvalid0, "valid")],
    early_stopping_rounds=300,
    verbose_eval=False,
)
_best_num_boost_round = int(_best_booster.best_iteration + 1)



## === cell 15
final_predictions = []
scores = []

params = study.best_params  # Speed: fetch once.

for fold in range(5):
    dtrain = _dtrain_folds[fold]
    dvalid = _dvalid_folds[fold]
    yvalid = y_all[fold_indices[fold]["va"]]

    model_params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        seed=42,
        nthread=_N_JOBS,
        **params,
    )
    if _has_gpu:
        model_params.update(tree_method="gpu_hist", predictor="gpu_predictor")
    else:
        model_params.update(tree_method="hist")

    booster = xgb.train(
        params=model_params,
        dtrain=dtrain,
        num_boost_round=_best_num_boost_round,
        evals=[(dvalid, "valid")],
        verbose_eval=False,
    )

    preds_valid = booster.predict(dvalid)
    test_preds = booster.predict(_dtest)

    final_predictions.append(test_preds)
    ROC = roc_auc_score(yvalid, preds_valid)
    print(fold, ROC)
    scores.append(ROC)

print(np.mean(scores), np.std(scores))



## === cell 16
sample_submission = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)

preds = np.mean(np.column_stack(final_predictions), axis=1)

sub = sample_submission[["id"]].copy()
test_ids = test["id"].to_numpy()
pred_df = pd.DataFrame({"id": test_ids, "target": preds})
sub = sub.merge(pred_df, on="id", how="left", validate="one_to_one")

if sub["target"].isna().any():
    sub["target"] = preds[: len(sub)]

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
