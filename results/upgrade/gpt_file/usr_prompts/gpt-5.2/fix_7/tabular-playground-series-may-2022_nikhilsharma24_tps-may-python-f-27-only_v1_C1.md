# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import string
import os

np.random.seed(42)

PRINT_INPUT_TREE = False
if PRINT_INPUT_TREE:
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(os.path.join(dirname, filename))



## === cell 1
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import optuna



## === cell 2
TRAIN_PATH = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-may-2022/test.csv"

train = pd.read_csv(
    TRAIN_PATH,
    usecols=["id", "target", "f_27"],
    dtype={"id": np.int32, "target": np.int8, "f_27": "string"},
)
test = pd.read_csv(
    TEST_PATH,
    usecols=["id", "f_27"],
    dtype={"id": np.int32, "f_27": "string"},
)

print(train.shape)
print(test.shape)



## === cell 3
train["kfold"] = -1
kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=102)
for fold, (_, valid_indicies) in enumerate(kf.split(X=train)):
    train.loc[valid_indicies, "kfold"] = fold

fold_valid_idx = []
fold_train_idx = []
kfold_arr = train["kfold"].to_numpy()
all_idx = np.arange(train.shape[0], dtype=np.int32)
for f in range(10):
    v = all_idx[kfold_arr == f]
    t = all_idx[kfold_arr != f]
    fold_valid_idx.append(v)
    fold_train_idx.append(t)



## === cell 4
df_tr = train[["f_27", "target", "kfold"]].copy()
df_te = test[["f_27"]].copy()

print(df_tr.shape)
print(df_te.shape)



## === cell 5
pd.crosstab(index=df_tr["target"], columns=df_tr["kfold"])




## === cell 6
def count_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Speed: replace per-letter Python loop with a single vectorized bincount over bytes.
    This is equivalent: counts of ASCII 'A'..'T' per row in f_27.
    """
    s = df["f_27"].astype("string").fillna("")
    py = s.astype(str).to_numpy()
    n = py.shape[0]
    max_len = int(max((len(x) for x in py), default=0))

    letters = string.ascii_uppercase[:20]  # A..T
    if max_len == 0:
        counts = np.zeros((n, len(letters)), dtype=np.int16)
    else:
        b = np.asarray(py, dtype=f"S{max_len}")
        mat = b.view(np.uint8).reshape(n, max_len)

        m = mat.astype(np.int16) - ord("A")
        valid = (m >= 0) & (m < 20)
        m2 = np.where(valid, m, -1)

        row_ids = np.repeat(np.arange(n, dtype=np.int32), max_len)
        vals = m2.reshape(-1)
        mask = vals >= 0
        bins = row_ids[mask] * 20 + vals[mask].astype(np.int32)
        bc = np.bincount(bins, minlength=n * 20)
        counts = bc.reshape(n, 20).astype(np.int16, copy=False)

    feat = pd.DataFrame(
        counts, columns=[f"count_{ch}" for ch in letters], index=df.index
    )

    for col in df.columns:
        if col != "f_27":
            feat[col] = df[col].to_numpy(copy=False)
    return feat


df_tr = count_alpha(df_tr)
df_te = count_alpha(df_te)

print(df_tr.shape)
print(df_te.shape)
print(df_tr.head())



## === cell 7
use_feature = [c for c in df_tr.columns if c not in ("target", "kfold")]
print("n_features:", len(use_feature))
print("missing in test:", sorted(set(use_feature) - set(df_te.columns)))




## === cell 8
def _gpu_available() -> bool:
    return os.system("command -v nvidia-smi >/dev/null 2>&1") == 0


USE_GPU = _gpu_available()
print("GPU available:", USE_GPU)

X_all = np.asarray(df_tr[use_feature].to_numpy(dtype=np.float32, copy=False), order="C")
y_all = df_tr["target"].to_numpy(dtype=np.int8, copy=False)
X_test = np.asarray(
    df_te[use_feature].to_numpy(dtype=np.float32, copy=False), order="C"
)

tree_method = "gpu_hist" if USE_GPU else "hist"
predictor = "gpu_predictor" if USE_GPU else "auto"

EARLY_STOPPING_ROUNDS = 300


def _predict_best(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    best_it = getattr(model, "best_iteration", None)
    if best_it is None:
        return model.predict_proba(X)[:, 1]

    iteration_range = (0, int(best_it) + 1)

    booster = getattr(model, "get_booster", None)
    if booster is not None:
        booster = model.get_booster()
        try:
            p = booster.inplace_predict(X, iteration_range=iteration_range)
            return np.asarray(p, dtype=np.float32)
        except Exception:
            return model.predict_proba(X, iteration_range=iteration_range)[:, 1]
    else:
        return model.predict_proba(X, iteration_range=iteration_range)[:, 1]


def run(trial):
    fold = 0

    learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 7)

    tr_idx = fold_train_idx[fold]
    va_idx = fold_valid_idx[fold]

    xtrain = X_all[tr_idx]
    xvalid = X_all[va_idx]
    ytrain = y_all[tr_idx]
    yvalid = y_all[va_idx]

    model = XGBClassifier(
        random_state=42,
        tree_method=tree_method,
        predictor=predictor,
        n_estimators=7000,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
        eval_metric="auc",
        n_jobs=-1,
    )

    model.fit(
        xtrain,
        ytrain,
        eval_set=[(xvalid, yvalid)],
        verbose=False,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    preds_valid = _predict_best(model, xvalid)
    auc = roc_auc_score(yvalid, preds_valid)
    return auc




## === cell 9
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)

study.optimize(run, n_trials=5)

print("Best AUC (fold=0):", study.best_value)
print("Best params:", study.best_params)



## === cell 10
final_predictions = []
scores = []

params = study.best_params

for fold in range(5):
    tr_idx = fold_train_idx[fold]
    va_idx = fold_valid_idx[fold]

    xtrain = X_all[tr_idx]
    xvalid = X_all[va_idx]
    ytrain = y_all[tr_idx]
    yvalid = y_all[va_idx]

    model = XGBClassifier(
        random_state=0,
        tree_method=tree_method,
        predictor=predictor,
        n_estimators=5000,
        eval_metric="auc",
        n_jobs=-1,
        **params,
    )

    model.fit(
        xtrain,
        ytrain,
        eval_set=[(xvalid, yvalid)],
        verbose=False,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    preds_valid = _predict_best(model, xvalid)
    test_preds = _predict_best(model, X_test)

    final_predictions.append(test_preds)
    roc = roc_auc_score(yvalid, preds_valid)
    print(fold, roc)
    scores.append(roc)

print("CV mean/std:", float(np.mean(scores)), float(np.std(scores)))



## === cell 11
sample_submission = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)

if len(final_predictions) == 0:
    preds = np.full(sample_submission.shape[0], 0.5, dtype=np.float32)
else:
    preds = np.mean(np.column_stack(final_predictions), axis=1)

sample_submission["target"] = preds.astype(np.float32)
sample_submission.to_csv("submission.csv", index=False)

print(sample_submission.head())
print("Wrote submission.csv with shape:", sample_submission.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
print(
    "submission.csv size (bytes):",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)
