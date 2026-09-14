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

    try:
        pruning_cb = optuna.integration.XGBoostPruningCallback(
            trial, "validation_0-mlogloss"
        )
        callbacks = [pruning_cb]
    except ModuleNotFoundError:
        callbacks = []

    clf.fit(
        X_train_scaled,
        y_train_np,
        eval_set=[(X_test_scaled, y_test_np)],
        verbose=False,
        callbacks=callbacks,
    )

    y_pred = clf.predict(X_test_scaled)
    return accuracy_score(y_test_np, y_pred)


pruner = optuna.pruners.MedianPruner(
    n_startup_trials=10, n_warmup_steps=50, interval_steps=10
)
study_xgb = optuna.create_study(direction="maximize", pruner=pruner)
study_xgb.optimize(objective_xgb, n_trials=50)


## === cell 13
best_params_xgb = study_xgb.best_params




## === cell 14
final_xgb_params = {
    "learning_rate": 0.03,
    "tree_method": "hist",
    "booster": "gbtree",
    "eval_metric": "mlogloss",
    "objective": "multi:softmax",
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
    **best_params_xgb,
}
if _USE_GPU:
    final_xgb_params["device"] = "cuda"

final_clf = XGBClassifier(**final_xgb_params)
final_clf.fit(X_train_scaled, y_train_np, verbose=False)

y_pred_enc = final_clf.predict(X_test_scaled)
print(accuracy_score(y_test_np, y_pred_enc))




## === cell 15
df_test = df_test.drop(columns=["Id", "Soil_Type7", "Soil_Type15"])
X_submit_np = df_test.to_numpy(dtype=np.float32, copy=False)
X_submit_scaled = _scaler.transform(X_submit_np)

final_pred_enc = final_clf.predict(X_submit_scaled)
Final_pred = _le.inverse_transform(final_pred_enc.astype(np.int32, copy=False))




## === cell 16
submission["Cover_Type"] = Final_pred
submission.to_csv("Submission.csv", index=False)
