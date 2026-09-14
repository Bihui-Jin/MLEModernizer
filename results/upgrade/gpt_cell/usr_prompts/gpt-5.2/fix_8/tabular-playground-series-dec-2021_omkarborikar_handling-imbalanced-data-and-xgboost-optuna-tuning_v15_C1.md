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
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
import optuna

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass




## === cell 1
TRAIN_PATH = "../input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "../input/tabular-playground-series-dec-2021/test.csv"
SUB_PATH = "../input/tabular-playground-series-dec-2021/sample_submission.csv"

_continuous = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
_binary = [f"Wilderness_Area{i}" for i in range(1, 5)] + [
    f"Soil_Type{i}" for i in range(1, 41)
]

dtype_train = {"Id": np.int32, "Cover_Type": np.int8}
dtype_test = {"Id": np.int32}
for c in _continuous:
    dtype_train[c] = np.float32
    dtype_test[c] = np.float32
for c in _binary:
    dtype_train[c] = np.int8
    dtype_test[c] = np.int8

df_train = pd.read_csv(TRAIN_PATH, dtype=dtype_train)
df_test = pd.read_csv(TEST_PATH, dtype=dtype_test)
submission = pd.read_csv(SUB_PATH)




## === cell 2
pass




## === cell 3
pass




## === cell 4
def reduce_mem_usage(df, verbose=True):
    return df




## === cell 5
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)




## === cell 6
pass




## === cell 7
df_train = df_train[(df_train["Cover_Type"] != 4) & (df_train["Cover_Type"] != 5)]




## === cell 8
X = df_train.drop(columns=["Id", "Cover_Type", "Soil_Type7", "Soil_Type15"])
y = df_train["Cover_Type"]

class_counts = y.value_counts()
minority_class = class_counts.idxmin()
minority_n = int(class_counts.min())

rng_seed = SEED
rs = np.random.RandomState(rng_seed)

res_idx_parts = []
for cls, cnt in class_counts.items():
    cls_idx = df_train.index[df_train["Cover_Type"] == cls].to_numpy()
    if cls == minority_class:
        res_idx_parts.append(cls_idx)
    else:
        res_idx_parts.append(rs.choice(cls_idx, size=minority_n, replace=False))

res_idx = np.concatenate(res_idx_parts, axis=0)
rs.shuffle(res_idx)

df_res = df_train.loc[res_idx, X.columns.tolist() + ["Cover_Type"]].reset_index(
    drop=True
)
X_res = df_res.drop(columns=["Cover_Type"])
y_res = df_res["Cover_Type"]




## === cell 9
pass




## === cell 10
x_train, x_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=SEED
)




## === cell 11
def objective_xgb(trial):
    raise RuntimeError("Superseded by cell 13 objective_xgb")




## === cell 12
import xgboost as xgb

_le = LabelEncoder()
y_train_enc = _le.fit_transform(y_train)
y_test_enc = _le.transform(y_test)

X_train_np = x_train.to_numpy(dtype=np.float32, copy=False)
X_test_np = x_test.to_numpy(dtype=np.float32, copy=False)
y_train_np = y_train_enc.astype(np.int32, copy=False)
y_test_np = y_test_enc.astype(np.int32, copy=False)

_scaler = StandardScaler()
X_train_scaled = _scaler.fit_transform(X_train_np)
X_test_scaled = _scaler.transform(X_test_np)


def _gpu_available():
    try:
        info = xgb.build_info()
        txt = str(info).lower()
        return ("cuda" in txt) or ("gpu" in txt)
    except Exception:
        return False


_USE_GPU = _gpu_available()


def objective_xgb(trial):
    xgb_params = {
        "learning_rate": 0.03,
        "tree_method": "hist",
        "booster": "gbtree",
        "eval_metric": "mlogloss",
        "objective": "multi:softmax",
        "n_estimators": trial.suggest_int("n_estimators", 500, 1000, 100),
        "subsample": trial.suggest_float("subsample", 0.2, 0.8, step=0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "gamma": trial.suggest_float("gamma", 0, 1.0),
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if _USE_GPU:
        xgb_params["device"] = "cuda"

    clf = XGBClassifier(**xgb_params)

    pruning_cb = optuna.integration.XGBoostPruningCallback(
        trial, "validation_0-mlogloss"
    )

    clf.fit(
        X_train_scaled,
        y_train_np,
        eval_set=[(X_test_scaled, y_test_np)],
        verbose=False,
        callbacks=[pruning_cb],
    )

    y_pred = clf.predict(X_test_scaled)
    return accuracy_score(y_test_np, y_pred)


pruner = optuna.pruners.MedianPruner(
    n_startup_trials=10, n_warmup_steps=50, interval_steps=10
)
study_xgb = optuna.create_study(direction="maximize", pruner=pruner)
study_xgb.optimize(objective_xgb, n_trials=50)




## --- ERROR in cell 12, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/optuna/integration/xgboost.py[0m in [0;36m<module>[0;34m[0m
[1;32m      4[0m [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m     [0;32mfrom[0m [0moptuna_integration[0m[0;34m.[0m[0mxgboost[0m [0;32mimport[0m [0mXGBoostPruningCallback[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;32mexcept[0m [0mModuleNotFoundError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'optuna_integration'

During handling of the above exception, another exception occurred:

[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/optuna/integration/__init__.py[0m in [0;36m_get_module[0;34m(self, module_name)[0m
[1;32m    131[0m             [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 132[0;31m                 [0;32mreturn[0m [0mimportlib[0m[0;34m.[0m[0mimport_module[0m[0;34m([0m[0;34m"."[0m [0;34m+[0m [0mmodule_name[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m__name__[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    133[0m             [0;32mexcept[0m [0mModuleNotFoundError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    127[0m [0;34m[0m[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_gcd_import[0;34m(name, package, level)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_find_and_load[0;34m(name, import_)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_find_and_load_unlocked[0;34m(name, import_)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_load_unlocked[0;34m(spec)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap_external.py[0m in [0;36mexec_module[0;34m(self, module)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_call_with_frames_removed[0;34m(f, *args, **kwds)[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/integration/xgboost.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mexcept[0m [0mModuleNotFoundError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m     [0;32mraise[0m [0mModuleNotFoundError[0m[0;34m([0m[0m_INTEGRATION_IMPORT_ERROR_TEMPLATE[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0;34m"xgboost"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0;34m[0m[0m

[0;31mModuleNotFoundError[0m: 
Could not find `optuna-integration` for `xgboost`.
Please run `pip install optuna-integration[xgboost]`.

During handling of the above exception, another exception occurred:

[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1410749075.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     78[0m )
[1;32m     79[0m [0mstudy_xgb[0m [0;34m=[0m [0moptuna[0m[0;34m.[0m[0mcreate_study[0m[0;34m([0m[0mdirection[0m[0;34m=[0m[0;34m"maximize"[0m[0;34m,[0m [0mpruner[0m[0;34m=[0m[0mpruner[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 80[0;31m [0mstudy_xgb[0m[0;34m.[0m[0moptimize[0m[0;34m([0m[0mobjective_xgb[0m[0;34m,[0m [0mn_trials[0m[0;34m=[0m[0;36m50[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     81[0m [0;34m[0m[0m
[1;32m     82[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/study.py[0m in [0;36moptimize[0;34m(self, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)[0m
[1;32m    488[0m                 [0mIf[0m [0mnested[0m [0minvocation[0m [0mof[0m [0mthis[0m [0mmethod[0m [0moccurs[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    489[0m         """
[0;32m--> 490[0;31m         _optimize(
[0m[1;32m    491[0m             [0mstudy[0m[0;34m=[0m[0mself[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    492[0m             [0mfunc[0m[0;34m=[0m[0mfunc[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py[0m in [0;36m_optimize[0;34m(study, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)[0m
[1;32m     61[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     62[0m         [0;32mif[0m [0mn_jobs[0m [0;34m==[0m [0;36m1[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 63[0;31m             _optimize_sequential(
[0m[1;32m     64[0m                 [0mstudy[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     65[0m                 [0mfunc[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

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

[0;32m/tmp/ipykernel_11/1410749075.py[0m in [0;36mobjective_xgb[0;34m(trial)[0m
[1;32m     57[0m     [0;31m# Pruning callback uses evaluation on the held-out split already used for scoring.[0m[0;34m[0m[0;34m[0m[0m
[1;32m     58[0m     [0;31m# This preserves the same metric definition while cutting wasted compute.[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 59[0;31m     pruning_cb = optuna.integration.XGBoostPruningCallback(
[0m[1;32m     60[0m         [0mtrial[0m[0;34m,[0m [0;34m"validation_0-mlogloss"[0m[0;34m[0m[0;34m[0m[0m
[1;32m     61[0m     )

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/integration/__init__.py[0m in [0;36m__getattr__[0;34m(self, name)[0m
[1;32m    118[0m                 [0mvalue[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_module[0m[0;34m([0m[0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    119[0m             [0;32melif[0m [0mname[0m [0;32min[0m [0mself[0m[0;34m.[0m[0m_class_to_module[0m[0;34m.[0m[0mkeys[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 120[0;31m                 [0mmodule[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_module[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_class_to_module[0m[0;34m[[0m[0mname[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    121[0m                 [0mvalue[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mmodule[0m[0;34m,[0m [0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    122[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/integration/__init__.py[0m in [0;36m_get_module[0;34m(self, module_name)[0m
[1;32m    132[0m                 [0;32mreturn[0m [0mimportlib[0m[0;34m.[0m[0mimport_module[0m[0;34m([0m[0;34m"."[0m [0;34m+[0m [0mmodule_name[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m__name__[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    133[0m             [0;32mexcept[0m [0mModuleNotFoundError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 134[0;31m                 [0;32mraise[0m [0mModuleNotFoundError[0m[0;34m([0m[0m_INTEGRATION_IMPORT_ERROR_TEMPLATE[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mmodule_name[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    135[0m [0;34m[0m[0m
[1;32m    136[0m     [0msys[0m[0;34m.[0m[0mmodules[0m[0;34m[[0m[0m__name__[0m[0;34m][0m [0;34m=[0m [0m_IntegrationModule[0m[0;34m([0m[0m__name__[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: 
Could not find `optuna-integration` for `xgboost`.
Please run `pip install optuna-integration[xgboost]`.

## === cell 13
best_params_xgb = study_xgb.best_params
