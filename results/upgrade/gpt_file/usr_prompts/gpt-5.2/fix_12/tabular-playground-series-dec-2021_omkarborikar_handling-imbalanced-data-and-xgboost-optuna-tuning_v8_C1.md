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
import time
import random
import numpy as np
import pandas as pd

import collections
import optuna

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

import xgboost as xgb

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DO_PLOTS = False

_CPU = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_CPU))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_CPU))



## === cell 1
DATA_DIR = "../input/tabular-playground-series-dec-2021"

useful_features = [
    "Elevation",
    "Wilderness_Area4",
    "Soil_Type10",
    "Wilderness_Area3",
    "Horizontal_Distance_To_Roadways",
    "Wilderness_Area1",
    "Soil_Type39",
    "Horizontal_Distance_To_Fire_Points",
    "Soil_Type38",
    "Soil_Type40",
]

train_usecols = ["Id"] + useful_features + ["Cover_Type"]
test_usecols = ["Id"] + useful_features

dtype_train = {c: np.int32 for c in (["Id", "Cover_Type"] + useful_features)}
dtype_test = {c: np.int32 for c in (["Id"] + useful_features)}

df_train_og = pd.read_csv(
    os.path.join(DATA_DIR, "train.csv"),
    usecols=train_usecols,
    dtype=dtype_train,
)
df_test_og = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv"),
    usecols=test_usecols,
    dtype=dtype_test,
)
submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))



## === cell 2
df_train_og.shape



## === cell 3
df_train_og.head()



## === cell 4
df_train_og.nunique()




## === cell 5
def reduce_mem_usage(df, verbose=True):
    return df




## === cell 6
test_id_series = df_test_og["Id"]
test_sort_index = test_id_series.argsort(kind="mergesort")  # stable, deterministic
test_ids_sorted = test_id_series.iloc[test_sort_index].reset_index(drop=True)

df_train = reduce_mem_usage(df_train_og)
df_test = reduce_mem_usage(df_test_og)
del df_train_og
del df_test_og



## === cell 7
cat_count = collections.Counter(df_train["Cover_Type"])
if DO_PLOTS:
    import matplotlib.pyplot as plt

    cat_freq = cat_count.values()
    cat = cat_count.keys()
    plt.bar(cat, cat_freq)
print(cat_count)



## === cell 8
X = df_train.drop(
    columns=["Id", "Cover_Type", "Soil_Type7", "Soil_Type15"], errors="ignore"
)
y = df_train["Cover_Type"]

X_res = X  # no reset_index needed for downstream logic
y_res = y



## === cell 9
cat_count = collections.Counter(y_res)
if DO_PLOTS:
    import matplotlib.pyplot as plt

    cat_freq = cat_count.values()
    cat = cat_count.keys()
    plt.bar(cat, cat_freq)
print(cat_count)



## === cell 10
SKIP_EXPENSIVE_EDA = True  # does not affect model; only avoids unused computation/plots

if not SKIP_EXPENSIVE_EDA:
    from sklearn.feature_selection import SelectKBest, f_classif
    import matplotlib.pyplot as plt

    selector = SelectKBest(f_classif, k="all")
    fitter = selector.fit(X_res, y_res)
    scores_df = pd.DataFrame(fitter.scores_)
    columns_df = pd.DataFrame(X_res.columns)
    featurescores = pd.concat([scores_df, columns_df], axis=1)
    featurescores.columns = ["score", "column name"]
    plt.figure(figsize=(20, 5))
    plt.bar(featurescores["column name"], featurescores["score"], width=0.4)
    plt.xticks(rotation="vertical")
    plt.plot()



## === cell 11
if "featurescores" in globals():
    featurescores.sort_values(by="score", ascending=False)
else:
    None



## === cell 12
useful_features = [
    "Elevation",
    "Wilderness_Area4",
    "Soil_Type10",
    "Wilderness_Area3",
    "Horizontal_Distance_To_Roadways",
    "Wilderness_Area1",
    "Soil_Type39",
    "Horizontal_Distance_To_Fire_Points",
    "Soil_Type38",
    "Soil_Type40",
]



## === cell 13
X_res = X_res[useful_features]



## === cell 14
le = LabelEncoder()
y_res_enc_full = le.fit_transform(y_res.astype(int))

class_counts = np.bincount(y_res_enc_full)
valid_classes = np.where(class_counts >= 2)[0]
mask = np.isin(y_res_enc_full, valid_classes)

X_res_f = X_res.to_numpy(copy=False)[mask]
y_res_enc_full_f = y_res_enc_full[mask]

x_train, x_test, y_train_full, y_test_full = train_test_split(
    X_res_f,
    y_res_enc_full_f,
    test_size=0.2,
    random_state=SEED,
    stratify=y_res_enc_full_f,
)

train_classes = np.unique(y_train_full)
num_class_train = int(len(train_classes))

max_label = int(max(y_train_full.max(), y_test_full.max()))
lut = np.full(max_label + 1, -1, dtype=np.int32)
lut[train_classes] = np.arange(num_class_train, dtype=np.int32)

y_train = lut[y_train_full].astype(np.int32, copy=False)
y_test = lut[y_test_full].astype(np.int32, copy=False)

new_to_class_arr = train_classes.astype(np.int32, copy=False)

print("Num classes in training split:", num_class_train)



## === cell 15
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train_scaled = np.ascontiguousarray(x_train_scaled, dtype=np.float32)
x_test_scaled = np.ascontiguousarray(x_test_scaled, dtype=np.float32)
y_train_arr = np.asarray(y_train, dtype=np.int32)
y_test_arr = np.asarray(y_test, dtype=np.int32)

dtrain = xgb.DMatrix(x_train_scaled, label=y_train_arr)
dvalid = xgb.DMatrix(x_test_scaled, label=y_test_arr)




## === cell 16
def objective_xgb(trial):
    xgb_params = {
        "learning_rate": 0.01,
        "tree_method": "hist",
        "booster": "gbtree",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        "gamma": trial.suggest_float("gamma", 0, 20),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0, step=0.1),
        "reg_lambda": trial.suggest_int("reg_lambda", 1, 100),
        "reg_alpha": trial.suggest_int("reg_alpha", 1, 100),
        "objective": "multi:softprob",
        "num_class": num_class_train,
        "eval_metric": "mlogloss",
        "seed": SEED,
        "nthread": _CPU,
    }
    n_estimators = trial.suggest_int("n_estimators", 500, 4000, 100)

    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=[(dvalid, "valid")],
        verbose_eval=False,
        early_stopping_rounds=50,
    )

    proba = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
    y_pred = np.argmax(proba, axis=1)
    return accuracy_score(y_test_arr, y_pred)




## === cell 17
sampler = optuna.samplers.TPESampler(seed=SEED)
study_xgb = optuna.create_study(direction="maximize", sampler=sampler)

OPTUNA_TIMEOUT_SEC = 360

try:
    study_xgb.optimize(
        objective_xgb, n_trials=50, timeout=OPTUNA_TIMEOUT_SEC, gc_after_trial=True
    )
except Exception as e:
    print(
        "Optuna optimization failed; will fall back to default params. Error:", repr(e)
    )



## === cell 18
if len(study_xgb.trials) > 0 and any(
    t.state.name == "COMPLETE" for t in study_xgb.trials
):
    best_params_xgb = dict(study_xgb.best_params)
else:
    best_params_xgb = {
        "n_estimators": 2000,
        "reg_lambda": 10,
        "reg_alpha": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_depth": 6,
        "min_child_weight": 2,
        "gamma": 0.0,
    }



## === cell 19
final_params = {
    "learning_rate": 0.01,
    "tree_method": "hist",
    "booster": "gbtree",
    "objective": "multi:softprob",
    "num_class": num_class_train,
    "eval_metric": "mlogloss",
    "seed": SEED,
    "nthread": _CPU,
    "max_depth": int(best_params_xgb["max_depth"]),
    "min_child_weight": int(best_params_xgb["min_child_weight"]),
    "gamma": float(best_params_xgb["gamma"]),
    "subsample": float(best_params_xgb["subsample"]),
    "colsample_bytree": float(best_params_xgb["colsample_bytree"]),
    "reg_lambda": float(best_params_xgb["reg_lambda"]),
    "reg_alpha": float(best_params_xgb["reg_alpha"]),
}
final_num_boost_round = int(best_params_xgb["n_estimators"])

final_booster = xgb.train(
    params=final_params,
    dtrain=dtrain,
    num_boost_round=final_num_boost_round,
    evals=[(dvalid, "valid")],
    verbose_eval=False,
    early_stopping_rounds=50,
)

df_test_feat_sorted = df_test.iloc[test_sort_index][useful_features]
x_test_full_scaled = scaler.transform(df_test_feat_sorted.to_numpy(copy=False))
x_test_full_scaled = np.ascontiguousarray(x_test_full_scaled, dtype=np.float32)
dtest = xgb.DMatrix(x_test_full_scaled)

proba_test = final_booster.predict(
    dtest, iteration_range=(0, final_booster.best_iteration + 1)
)
final_pred_train_enc = np.argmax(proba_test, axis=1).astype(np.int32, copy=False)

final_pred_full_enc = new_to_class_arr[final_pred_train_enc]
Final_pred = le.inverse_transform(final_pred_full_enc.astype(int))

submission = submission.sort_values("Id").reset_index(drop=True)
if len(submission) == len(test_ids_sorted):
    submission["Id"] = test_ids_sorted

submission["Cover_Type"] = Final_pred.astype(int)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print(submission.shape)
print("Wrote: submission.csv")
