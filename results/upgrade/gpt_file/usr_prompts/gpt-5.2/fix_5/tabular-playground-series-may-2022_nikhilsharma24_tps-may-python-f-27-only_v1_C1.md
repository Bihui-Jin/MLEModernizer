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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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

from xgboost.callback import EarlyStopping



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

        counts = np.empty((n, len(letters)), dtype=np.int16)
        for j, ch in enumerate(letters):
            counts[:, j] = (mat == ord(ch)).sum(axis=1, dtype=np.int16)

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

X_all = np.ascontiguousarray(df_tr[use_feature].to_numpy(dtype=np.float32, copy=False))
y_all = df_tr["target"].to_numpy(dtype=np.int8, copy=False)
X_test = np.ascontiguousarray(df_te[use_feature].to_numpy(dtype=np.float32, copy=False))

tree_method = "gpu_hist" if USE_GPU else "hist"
predictor = "gpu_predictor" if USE_GPU else "auto"

_es_cb = EarlyStopping(
    rounds=300, metric_name="auc", data_name="validation_0", save_best=True
)


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
        callbacks=[_es_cb],
    )

    preds_valid = model.predict_proba(xvalid)[:, 1]
    auc = roc_auc_score(yvalid, preds_valid)
    return auc




## === cell 9
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)

study.optimize(run, n_trials=5, show_progress_bar=False)

print("Best AUC (fold=0):", study.best_value)
print("Best params:", study.best_params)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1889375558.py in <cell line: 0>()
      3 
      4 # Keep same number of trials (core logic); reduce overhead by disabling progress bar already.
----> 5 study.optimize(run, n_trials=5, show_progress_bar=False)
      6 
      7 print("Best AUC (fold=0):", study.best_value)

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

/tmp/ipykernel_11/2978271220.py in run(trial)
     53     )
     54 
---> 55     model.fit(
     56         xtrain,
     57         ytrain,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in fit(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)
   1517             )
   1518 
-> 1519             self._Booster = train(
   1520                 params,
   1521                 train_dmatrix,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/training.py in train(params, dtrain, num_boost_round, evals, obj, feval, maximize, early_stopping_rounds, evals_result, verbose_eval, xgb_model, callbacks, custom_metric)
    183             break
    184 
--> 185     bst = cb_container.after_training(bst)
    186 
    187     if evals_result is not None:

/usr/local/lib/python3.11/dist-packages/xgboost/callback.py in after_training(self, model)
    168         """Function called after training."""
    169         for c in self.callbacks:
--> 170             model = c.after_training(model=model)
    171             msg = "after_training should return the model"
    172             if self.is_cv:

/usr/local/lib/python3.11/dist-packages/xgboost/callback.py in after_training(self, model)
    459 
    460         try:
--> 461             best_iteration = model.best_iteration
    462             best_score = model.best_score
    463             assert best_iteration is not None and best_score is not None

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in best_iteration(self)
   2602             return int(best)
   2603 
-> 2604         raise AttributeError(
   2605             "`best_iteration` is only defined when early stopping is used."
   2606         )

AttributeError: `best_iteration` is only defined when early stopping is used.

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
        callbacks=[_es_cb],
    )

    preds_valid = model.predict_proba(xvalid)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]

    final_predictions.append(test_preds)
    roc = roc_auc_score(yvalid, preds_valid)
    print(fold, roc)
    scores.append(roc)

print("CV mean/std:", float(np.mean(scores)), float(np.std(scores)))



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2497355154.py in <cell line: 0>()
     24     )
     25 
---> 26     model.fit(
     27         xtrain,
     28         ytrain,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in fit(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)
   1517             )
   1518 
-> 1519             self._Booster = train(
   1520                 params,
   1521                 train_dmatrix,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/training.py in train(params, dtrain, num_boost_round, evals, obj, feval, maximize, early_stopping_rounds, evals_result, verbose_eval, xgb_model, callbacks, custom_metric)
    183             break
    184 
--> 185     bst = cb_container.after_training(bst)
    186 
    187     if evals_result is not None:

/usr/local/lib/python3.11/dist-packages/xgboost/callback.py in after_training(self, model)
    168         """Function called after training."""
    169         for c in self.callbacks:
--> 170             model = c.after_training(model=model)
    171             msg = "after_training should return the model"
    172             if self.is_cv:

/usr/local/lib/python3.11/dist-packages/xgboost/callback.py in after_training(self, model)
    459 
    460         try:
--> 461             best_iteration = model.best_iteration
    462             best_score = model.best_score
    463             assert best_iteration is not None and best_score is not None

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in best_iteration(self)
   2602             return int(best)
   2603 
-> 2604         raise AttributeError(
   2605             "`best_iteration` is only defined when early stopping is used."
   2606         )

AttributeError: `best_iteration` is only defined when early stopping is used.

## === cell 11
sample_submission = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)

preds = np.mean(np.column_stack(final_predictions), axis=1)

sample_submission["target"] = preds
sample_submission.to_csv("submission.csv", index=False)

print(sample_submission.head())
print("Wrote submission.csv with shape:", sample_submission.shape)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3557526801.py in <cell line: 0>()
      3 )
      4 
----> 5 preds = np.mean(np.column_stack(final_predictions), axis=1)
      6 
      7 sample_submission["target"] = preds

/usr/local/lib/python3.11/dist-packages/numpy/lib/shape_base.py in column_stack(tup)
    650             arr = array(arr, copy=False, subok=True, ndmin=2).T
    651         arrays.append(arr)
--> 652     return _nx.concatenate(arrays, 1)
    653 
    654 

ValueError: need at least one array to concatenate
