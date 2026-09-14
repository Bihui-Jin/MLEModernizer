# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

geopandas==0.14.4
imbalanced-learn==0.13.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        input/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        working/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
```

-> data/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> data/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> input/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

import matplotlib.pyplot as plt  # kept for compatibility with original cells
import collections
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import optuna

SEED = 42
np.random.seed(SEED)

try:
    from imblearn.under_sampling import RandomUnderSampler  # optional
except ModuleNotFoundError:
    RandomUnderSampler = None



## === cell 1
TRAIN_PATH = "../input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "../input/tabular-playground-series-dec-2021/test.csv"
SUB_PATH = "../input/tabular-playground-series-dec-2021/sample_submission.csv"

df_train_og = pd.read_csv(TRAIN_PATH, nrows=5)
_train_cols = df_train_og.columns.tolist()
del df_train_og

df_test_og = pd.read_csv(TEST_PATH, nrows=5)
_test_cols = df_test_og.columns.tolist()
del df_test_og

train_dtypes = {"Id": np.int32, "Cover_Type": np.int8}
test_dtypes = {"Id": np.int32}

for c in _train_cols:
    if c not in train_dtypes:
        train_dtypes[c] = np.int16
for c in _test_cols:
    if c not in test_dtypes:
        test_dtypes[c] = np.int16

df_train_og = pd.read_csv(TRAIN_PATH, dtype=train_dtypes)
df_test_og = pd.read_csv(TEST_PATH, dtype=test_dtypes)
submission = pd.read_csv(SUB_PATH)



## === cell 2
df_train_og.shape



## === cell 3
df_train_og.head()



## === cell 4
df_train_og.nunique()




## === cell 5
def reduce_mem_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem if start_mem else 0
            )
        )
    return df




## === cell 6
df_train = df_train_og
df_test = df_test_og
del df_train_og
del df_test_og



## === cell 7
cat_count = collections.Counter(df_train["Cover_Type"])
print(cat_count)



## === cell 8
df_train = df_train[(df_train["Cover_Type"] != 4) & (df_train["Cover_Type"] != 5)]




## === cell 9
def undersample_not_minority_numpy(X_df, y_series, random_state=42):
    y = np.asarray(y_series)
    labels, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    minority_labels = labels[counts == min_count]
    minority_label = minority_labels.min()

    rng = np.random.RandomState(random_state)

    order = np.argsort(y, kind="mergesort")
    y_sorted = y[order]

    starts = np.searchsorted(y_sorted, labels, side="left")
    ends = np.searchsorted(y_sorted, labels, side="right")

    chosen_parts = []
    for lab, s, e in zip(labels, starts, ends):
        idx_block = order[s:e]
        if lab == minority_label:
            chosen = idx_block
        else:
            chosen = rng.choice(idx_block, size=min_count, replace=False)
        chosen_parts.append(chosen)

    keep_idx = np.concatenate(chosen_parts)
    rng.shuffle(keep_idx)

    X_res = X_df.to_numpy(copy=False)[keep_idx]
    X_res = pd.DataFrame(X_res, columns=X_df.columns)
    y_res = pd.Series(y[keep_idx])
    return X_res.reset_index(drop=True), y_res.reset_index(drop=True)


X = df_train.drop(columns=["Id", "Cover_Type", "Soil_Type7", "Soil_Type15"])
y = df_train["Cover_Type"]
X_res, y_res = undersample_not_minority_numpy(X, y, random_state=SEED)



## === cell 10
cat_count = collections.Counter(y_res)
print(cat_count)



## === cell 11
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k="all")
_ = selector.fit(X_res, y_res)



## === cell 12
pass



## === cell 13
pass



## === cell 14
pass



## === cell 15
pass



## === cell 16
x_train, x_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=SEED, stratify=y_res
)



## === cell 17
scaler = StandardScaler()
Xtr = scaler.fit_transform(x_train)
Xva = scaler.transform(x_test)

Xtr = np.asarray(Xtr, dtype=np.float32, order="C")
Xva = np.asarray(Xva, dtype=np.float32, order="C")

classes = np.sort(np.unique(np.asarray(y_train)))
y_train_arr = np.asarray(y_train)
y_test_arr = np.asarray(y_test)
y_train_enc = np.searchsorted(classes, y_train_arr).astype(np.int32, copy=False)
y_test_enc = np.searchsorted(classes, y_test_arr).astype(np.int32, copy=False)

try:
    import xgboost as xgb

    gpu_available = True
    try:
        with xgb.config_context(device="cuda"):
            pass
    except Exception:
        gpu_available = False
except Exception:
    gpu_available = False

TREE_METHOD = "gpu_hist" if gpu_available else "hist"



## === cell 18
import xgboost as xgb

n_cpus = os.cpu_count() or 4
n_jobs_optuna = min(4, n_cpus)
xgb_threads_per_trial = max(1, n_cpus // n_jobs_optuna)


def _fast_acc_int(y_true_int32, y_pred_int32):
    return float(np.mean(y_true_int32 == y_pred_int32))


dtrain_full = xgb.DMatrix(Xtr, label=y_train_enc)
dvalid_shared = xgb.DMatrix(Xva, label=y_test_enc)


def objective_xgb(trial):
    n_estimators = trial.suggest_int("n_estimators", 500, 4000, 100)

    params = {
        "eta": 0.01,  # learning_rate
        "tree_method": TREE_METHOD,
        "booster": "gbtree",
        "lambda": trial.suggest_int("reg_lambda", 1, 100),
        "alpha": trial.suggest_int("reg_alpha", 1, 100),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0, step=0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        "gamma": trial.suggest_float("gamma", 0, 20),
        "num_class": len(classes),
        "seed": SEED,
        "nthread": xgb_threads_per_trial,
        "objective": "multi:softmax",
        "eval_metric": "mlogloss",
        "verbosity": 0,
        "disable_default_eval_metric": 1,
    }

    if TREE_METHOD == "gpu_hist":
        params["predictor"] = "gpu_predictor"
        params["sampling_method"] = "gradient_based"
    else:
        params["predictor"] = "cpu_predictor"

    cv_res = xgb.cv(
        params=params,
        dtrain=dtrain_full,
        num_boost_round=n_estimators,
        nfold=5,
        stratified=True,
        shuffle=True,
        early_stopping_rounds=100,
        metrics=("mlogloss",),
        seed=SEED,
        verbose_eval=False,
        as_pandas=True,
        enable_categorical=False,
    )
    best_round = int(cv_res.shape[0])

    booster_full = xgb.train(
        params=params,
        dtrain=dtrain_full,
        num_boost_round=best_round,
        verbose_eval=False,
    )

    y_pred = booster_full.predict(dvalid_shared).astype(np.int32, copy=False)
    return _fast_acc_int(y_test_enc, y_pred)


sampler = optuna.samplers.TPESampler(seed=SEED)
study_xgb = optuna.create_study(direction="maximize", sampler=sampler)

study_xgb.optimize(objective_xgb, n_trials=50, n_jobs=n_jobs_optuna)



## --- ERROR in cell 18, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3986573413.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     78[0m [0mstudy_xgb[0m [0;34m=[0m [0moptuna[0m[0;34m.[0m[0mcreate_study[0m[0;34m([0m[0mdirection[0m[0;34m=[0m[0;34m"maximize"[0m[0;34m,[0m [0msampler[0m[0;34m=[0m[0msampler[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     79[0m [0;34m[0m[0m
[0;32m---> 80[0;31m [0mstudy_xgb[0m[0;34m.[0m[0moptimize[0m[0;34m([0m[0mobjective_xgb[0m[0;34m,[0m [0mn_trials[0m[0;34m=[0m[0;36m50[0m[0;34m,[0m [0mn_jobs[0m[0;34m=[0m[0mn_jobs_optuna[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     81[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/study.py[0m in [0;36moptimize[0;34m(self, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)[0m
[1;32m    488[0m                 [0mIf[0m [0mnested[0m [0minvocation[0m [0mof[0m [0mthis[0m [0mmethod[0m [0moccurs[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    489[0m         """
[0;32m--> 490[0;31m         _optimize(
[0m[1;32m    491[0m             [0mstudy[0m[0;34m=[0m[0mself[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    492[0m             [0mfunc[0m[0;34m=[0m[0mfunc[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py[0m in [0;36m_optimize[0;34m(study, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)[0m
[1;32m     98[0m                         [0;31m# Raise if exception occurred in executing the completed futures.[0m[0;34m[0m[0;34m[0m[0m
[1;32m     99[0m                         [0;32mfor[0m [0mf[0m [0;32min[0m [0mcompleted[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 100[0;31m                             [0mf[0m[0;34m.[0m[0mresult[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    101[0m [0;34m[0m[0m
[1;32m    102[0m                     futures.add(

[0;32m/usr/lib/python3.11/concurrent/futures/_base.py[0m in [0;36mresult[0;34m(self, timeout)[0m
[1;32m    447[0m                     [0;32mraise[0m [0mCancelledError[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    448[0m                 [0;32melif[0m [0mself[0m[0;34m.[0m[0m_state[0m [0;34m==[0m [0mFINISHED[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 449[0;31m                     [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m__get_result[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    450[0m [0;34m[0m[0m
[1;32m    451[0m                 [0mself[0m[0;34m.[0m[0m_condition[0m[0;34m.[0m[0mwait[0m[0;34m([0m[0mtimeout[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/concurrent/futures/_base.py[0m in [0;36m__get_result[0;34m(self)[0m
[1;32m    399[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0m_exception[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    400[0m             [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 401[0;31m                 [0;32mraise[0m [0mself[0m[0;34m.[0m[0m_exception[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    402[0m             [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    403[0m                 [0;31m# Break a reference cycle with the exception in self._exception[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/concurrent/futures/thread.py[0m in [0;36mrun[0;34m(self)[0m
[1;32m     56[0m [0;34m[0m[0m
[1;32m     57[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 58[0;31m             [0mresult[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mfn[0m[0;34m([0m[0;34m*[0m[0mself[0m[0;34m.[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mself[0m[0;34m.[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     59[0m         [0;32mexcept[0m [0mBaseException[0m [0;32mas[0m [0mexc[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     60[0m             [0mself[0m[0;34m.[0m[0mfuture[0m[0;34m.[0m[0mset_exception[0m[0;34m([0m[0mexc[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py[0m in [0;36m_optimize_sequential[0;34m(study, func, n_trials, timeout, catch, callbacks, gc_after_trial, reseed_sampler_rng, time_start, progress_bar)[0m
[1;32m    158[0m [0;34m[0m[0m
[1;32m    159[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 160[0;31m             [0mfrozen_trial_id[0m [0;34m=[0m [0m_run_trial[0m[0;34m([0m[0mstudy[0m[0;34m,[0m [0mfunc[0m[0;34m,[0m [0mcatch[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    161[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    162[0m             [0;31m# The following line mitigates memory problems that can be occurred in some[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py[0m in [0;36m_run_trial[0;34m(study, func, catch)[0m
[1;32m    256[0m         [0;32mand[0m [0;32mnot[0m [0misinstance[0m[0;34m([0m[0mfunc_err[0m[0;34m,[0m [0mcatch[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    257[0m     ):
[0;32m--> 258[0;31m         [0;32mraise[0m [0mfunc_err[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    259[0m     [0;32mreturn[0m [0mtrial[0m[0;34m.[0m[0m_trial_id[0m[0;34m[0m[0;34m[0m[0m
[1;32m    260[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py[0m in [0;36m_run_trial[0;34m(study, func, catch)[0m
[1;32m    199[0m     [0;32mwith[0m [0mget_heartbeat_thread[0m[0;34m([0m[0mtrial[0m[0;34m.[0m[0m_trial_id[0m[0;34m,[0m [0mstudy[0m[0;34m.[0m[0m_storage[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    200[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 201[0;31m             [0mvalue_or_values[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0mtrial[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    202[0m         [0;32mexcept[0m [0mexceptions[0m[0;34m.[0m[0mTrialPruned[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    203[0m             [0;31m# TODO(mamu): Handle multi-objective cases.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3986573413.py[0m in [0;36mobjective_xgb[0;34m(trial)[0m
[1;32m     47[0m [0;34m[0m[0m
[1;32m     48[0m     [0;31m# Equivalent CV semantics: stratified, 5-fold, shuffle controlled by seed.[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 49[0;31m     cv_res = xgb.cv(
[0m[1;32m     50[0m         [0mparams[0m[0;34m=[0m[0mparams[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m         [0mdtrain[0m[0;34m=[0m[0mdtrain_full[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: cv() got an unexpected keyword argument 'enable_categorical'

## === cell 19
best_params_xgb = study_xgb.best_params
best_params_xgb
