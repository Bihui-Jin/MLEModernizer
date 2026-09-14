# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the class of a given image from a synthetic dataset.

## MetricMulti-class classification accuracy.

## Submission FormatFor each `Id` in the test set, you must predict the `Cover_Type` class. The file should contain a header and have the following format:
```
Id,Cover_Type
4000000,2
4000001,1
4000001,3
etc.
```

## Dataset 
- train.csv - the training data with the target `Cover_Type` column
- test.csv - the test set; you will be predicting the `Cover_Type` for each row in this file (the target integer class)
- sample_submission.csv - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Code solution

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

    X_res = X_df.iloc[keep_idx].reset_index(drop=True)
    y_res = pd.Series(y[keep_idx]).reset_index(drop=True)
    return X_res, y_res


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
from sklearn.model_selection import StratifiedKFold

use_quantile = TREE_METHOD in ("hist", "gpu_hist")

if use_quantile:
    dtrain_full = xgb.QuantileDMatrix(Xtr, label=y_train_enc, max_bin=256)
    dtrain_cv_base = xgb.DMatrix(Xtr, label=y_train_enc)
else:
    dtrain_full = xgb.DMatrix(Xtr, label=y_train_enc)
    dtrain_cv_base = dtrain_full

n_cpus = os.cpu_count() or 4
n_jobs_optuna = min(4, n_cpus)
xgb_threads_per_trial = max(1, n_cpus // n_jobs_optuna)


def _fast_acc_int(y_true_int32, y_pred_int32):
    return float(np.mean(y_true_int32 == y_pred_int32))


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_folds = list(skf.split(np.empty(len(y_train_enc), dtype=np.uint8), y_train_enc))

dtrain_folds = []
dvalid_folds = []
for tr_idx, va_idx in cv_folds:
    dtr = xgb.DMatrix(Xtr[tr_idx], label=y_train_enc[tr_idx])
    dva = xgb.DMatrix(Xtr[va_idx], label=y_train_enc[va_idx])
    dtrain_folds.append(dtr)
    dvalid_folds.append(dva)


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

    best_rounds = []
    for dtr, dva in zip(dtrain_folds, dvalid_folds):
        booster = xgb.train(
            params=params,
            dtrain=dtr,
            num_boost_round=n_estimators,
            evals=[(dva, "valid")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        best_rounds.append(booster.best_iteration + 1)

    best_round = int(np.round(np.mean(best_rounds)))

    booster_full = xgb.train(
        params=params,
        dtrain=dtrain_full,
        num_boost_round=best_round,
        verbose_eval=False,
    )

    y_pred = booster_full.predict(dvalid_shared).astype(np.int32, copy=False)
    return _fast_acc_int(y_test_enc, y_pred)


if use_quantile:
    dvalid_shared = xgb.QuantileDMatrix(
        Xva, label=y_test_enc, max_bin=256, ref=dtrain_full
    )
else:
    dvalid_shared = xgb.DMatrix(Xva, label=y_test_enc)

sampler = optuna.samplers.TPESampler(seed=SEED)
study_xgb = optuna.create_study(direction="maximize", sampler=sampler)
study_xgb.optimize(objective_xgb, n_trials=50, n_jobs=n_jobs_optuna)



## === cell 19
best_params_xgb = study_xgb.best_params
best_params_xgb



## === cell 20
best_params_xgb_final = dict(best_params_xgb)
best_params_xgb_final.update(
    {
        "learning_rate": 0.01,  # fixed in objective
        "tree_method": TREE_METHOD,  # fixed in objective
        "booster": "gbtree",  # fixed in objective
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
        "objective": "multi:softmax",
        "eval_metric": "mlogloss",
        "disable_default_eval_metric": 1,
    }
)

pipe = Pipeline(
    steps=[
        ("step1", StandardScaler()),
        ("step2", XGBClassifier(**best_params_xgb_final)),
    ]
)



## === cell 21
final_model = XGBClassifier(**best_params_xgb_final)
final_model.fit(Xtr, y_train_enc)

df_test = df_test.drop(columns=["Id", "Soil_Type7", "Soil_Type15"])
Xte = scaler.transform(df_test)
Xte = np.asarray(Xte, dtype=np.float32, order="C")
Final_pred_enc = final_model.predict(Xte).astype(np.int32, copy=False)

Final_pred = classes[Final_pred_enc]

submission["Cover_Type"] = Final_pred
submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", submission.shape)
