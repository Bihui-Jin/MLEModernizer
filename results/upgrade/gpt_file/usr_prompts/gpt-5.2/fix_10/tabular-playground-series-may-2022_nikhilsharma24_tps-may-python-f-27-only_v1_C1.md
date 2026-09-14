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

# 5. Target score

0.63094

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The timeout is dominated by repeated XGBoost training: Optuna runs `xgb.cv` up to 5 times on 800k rows with up to 7000 rounds, then you train 5 more fold models up to 5000 rounds. To keep identical learning logic while cutting wall time, the main fixes are (1) avoid building/processing DMatrix repeatedly by caching and using `QuantileDMatrix` for `hist/gpu_hist`, (2) cap thread oversubscription deterministically (XGBoost + BLAS) to reduce wasted CPU time, (3) reuse the same test matrix once and use in-place prediction everywhere, and (4) reduce CV overhead by using XGBoost’s built-in fold splitting via precomputed indices without extra pandas/numpy slicing. These changes preserve the exact model family, objective, early stopping, folds, and evaluation semantics; they only remove redundant work and speed up data handling/training.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import string
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

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
import xgboost as xgb
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
kfold_arr = np.empty(train.shape[0], dtype=np.int8)
for fold, (_, valid_indicies) in enumerate(kf.split(X=train)):
    kfold_arr[valid_indicies] = fold
train["kfold"] = kfold_arr  # keep same semantics/visibility as original


def get_fold_indices(fold: int):
    va_mask = kfold_arr == fold
    tr_mask = ~va_mask
    va_idx = np.flatnonzero(va_mask)
    tr_idx = np.flatnonzero(tr_mask)
    return tr_idx, va_idx




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


def _predict_best_booster(
    booster: xgb.Booster, X: np.ndarray, best_it: int | None
) -> np.ndarray:
    if best_it is None:
        try:
            p = booster.inplace_predict(X)
            return np.asarray(p, dtype=np.float32)
        except Exception:
            d = xgb.DMatrix(X)
            return np.asarray(booster.predict(d), dtype=np.float32)
    iteration_range = (0, int(best_it) + 1)
    try:
        p = booster.inplace_predict(X, iteration_range=iteration_range)
        return np.asarray(p, dtype=np.float32)
    except Exception:
        d = xgb.DMatrix(X)
        return np.asarray(
            booster.predict(d, iteration_range=iteration_range), dtype=np.float32
        )




## === cell 9
if tree_method in ("hist", "gpu_hist"):
    dtrain_all = xgb.QuantileDMatrix(X_all, label=y_all, max_bin=256)
else:
    dtrain_all = xgb.DMatrix(X_all, label=y_all)

tr_idx0, va_idx0 = get_fold_indices(0)
folds0 = [(tr_idx0, va_idx0)]


def run(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 7)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": tree_method,
        "predictor": predictor,
        "learning_rate": learning_rate,
        "reg_lambda": reg_lambda,
        "reg_alpha": reg_alpha,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "max_depth": max_depth,
        "seed": 42,
        "verbosity": 0,
        "nthread": 8,
    }

    cv = xgb.cv(
        params=params,
        dtrain=dtrain_all,
        num_boost_round=7000,
        folds=folds0,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
        seed=42,
    )
    return float(cv["test-auc-mean"].iloc[-1])




## === cell 10
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)

study.optimize(run, n_trials=5)

print("Best AUC (fold=0):", study.best_value)
print("Best params:", study.best_params)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/2289788581.py in <cell line: 0>()
      2 study = optuna.create_study(direction="maximize", sampler=sampler)
      3 
----> 4 study.optimize(run, n_trials=5)
      5 
      6 print("Best AUC (fold=0):", study.best_value)

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in optimize(self, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)
    488                 If nested invocation of this method occurs.
    489         """
--> 490         _optimize(
    491             study=self,
    492             func=func,

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _optimize(study, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)
     61     try:
     62         if n_jobs == 1:
---> 63             _optimize_sequential(
     64                 study,
     65                 func,

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _optimize_sequential(study, func, n_trials, timeout, catch, callbacks, gc_after_trial, reseed_sampler_rng, time_start, progress_bar)
    158 
    159         try:
--> 160             frozen_trial_id = _run_trial(study, func, catch)
    161         finally:
    162             # The following line mitigates memory problems that can be occurred in some

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _run_trial(study, func, catch)
    256         and not isinstance(func_err, catch)
    257     ):
--> 258         raise func_err
    259     return trial._trial_id
    260 

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _run_trial(study, func, catch)
    199     with get_heartbeat_thread(trial._trial_id, study._storage):
    200         try:
--> 201             value_or_values = func(trial)
    202         except exceptions.TrialPruned as e:
    203             # TODO(mamu): Handle multi-objective cases.

/tmp/ipykernel_11/2655052222.py in run(trial)
     36     }
     37 
---> 38     cv = xgb.cv(
     39         params=params,
     40         dtrain=dtrain_all,

/usr/local/lib/python3.11/dist-packages/xgboost/training.py in cv(params, dtrain, num_boost_round, nfold, stratified, folds, metrics, obj, feval, maximize, early_stopping_rounds, fpreproc, as_pandas, verbose_eval, show_stdv, seed, callbacks, shuffle, custom_metric)
    541 
    542     results: Dict[str, List[float]] = {}
--> 543     cvfolds = mknfold(
    544         dtrain, nfold, params, seed, metrics, fpreproc, stratified, folds, shuffle
    545     )

/usr/local/lib/python3.11/dist-packages/xgboost/training.py in mknfold(dall, nfold, param, seed, evals, fpreproc, stratified, folds, shuffle)
    396     for k in range(nfold):
    397         # perform the slicing using the indexes determined by the above methods
--> 398         dtrain = dall.slice(in_idset[k])
    399         dtest = dall.slice(out_idset[k])
    400         # run preprocessing on the data set if needed

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

XGBoostError: [19:58:45] /workspace/src/data/iterative_dmatrix.h:88: Slicing DMatrix is not supported for Quantile DMatrix.
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3effba) [0x7fff83903fba]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3ff7ab) [0x7fff839137ab]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGDMatrixSliceDMatrixEx+0x146) [0x7fff83674206]
  [bt] (3) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (4) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (5) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]
  [bt] (6) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7ffff63bbc8e]
  [bt] (7) /usr/bin/python3(_PyObject_MakeTpCall+0x27c) [0x52f85c]
  [bt] (8) /usr/bin/python3(_PyEval_EvalFrameDefault+0x6bc) [0x53da0c]



## === cell 11
final_predictions = []
scores = []

params = study.best_params

if tree_method in ("hist", "gpu_hist"):
    dtest = xgb.QuantileDMatrix(X_test, max_bin=256)
else:
    dtest = xgb.DMatrix(X_test)

for fold in range(5):
    tr_idx, va_idx = get_fold_indices(fold)

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
        n_jobs=8,
        **params,
    )

    model.fit(
        xtrain,
        ytrain,
        eval_set=[(xvalid, yvalid)],
        verbose=False,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    booster = model.get_booster()
    best_it = getattr(model, "best_iteration", None)

    preds_valid = _predict_best_booster(booster, xvalid, best_it)

    if best_it is None:
        test_preds = booster.predict(dtest)
    else:
        test_preds = booster.predict(dtest, iteration_range=(0, int(best_it) + 1))

    final_predictions.append(np.asarray(test_preds, dtype=np.float32))
    roc = roc_auc_score(yvalid, preds_valid)
    print(fold, roc)
    scores.append(roc)

print("CV mean/std:", float(np.mean(scores)), float(np.std(scores)))




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/426378515.py in <cell line: 0>()
      2 scores = []
      3 
----> 4 params = study.best_params
      5 
      6 # --- Speed: build test DMatrix/QuantileDMatrix once and reuse across folds.

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in best_params(self)
    118         """
    119 
--> 120         return self.best_trial.params
    121 
    122     @property

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in best_trial(self)
    154 
    155         """
--> 156         return self._get_best_trial(deepcopy=True)
    157 
    158     @property

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in _get_best_trial(self, deepcopy)
    306             )
    307 
--> 308         best_trial = self._storage.get_best_trial(self._study_id)
    309 
    310         # If the trial with the best value is infeasible, select the best trial from all feasible

/usr/local/lib/python3.11/dist-packages/optuna/storages/_in_memory.py in get_best_trial(self, study_id)
    250 
    251             if best_trial_id is None:
--> 252                 raise ValueError("No trials are completed yet.")
    253             elif len(self._studies[study_id].directions) > 1:
    254                 raise RuntimeError(

ValueError: No trials are completed yet.

## === cell 12
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
