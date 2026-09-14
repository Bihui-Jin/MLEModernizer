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

# 5. Target score

0.82798

# 6. Current score

0.06752

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.06752) has done: 'The timeout is dominated by repeatedly fitting heavy XGBoost models inside Optuna (50 trials with up to 4000 estimators each), plus extra overhead from pandas objects flowing through scikit-learn pipelines. To keep the core logic identical (same model family, same training loop semantics, same feature selection, same metric), the main speedups are: enabling scikit-learn-intelex for faster preprocessing, converting data once to contiguous NumPy arrays (so scaling/XGBoost avoid pandas overhead each trial), caching the scaled arrays per-trial, and using Optuna’s built-in pruning-free, deterministic sampler with a fixed seed to avoid wasted overhead. We also ensure XGBoost uses all CPU threads (and GPU if available) and avoid re-encoding labels inside every trial.'

# 9. Code solution

## === cell 0
import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

try:
    from imblearn.under_sampling import RandomUnderSampler  # Used for under sampling.
except Exception:
    RandomUnderSampler = None

import collections
import optuna

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)



## === cell 1
read_kwargs = {}
try:
    import pyarrow  # noqa: F401

    read_kwargs["engine"] = "pyarrow"
except Exception:
    pass

df_train_og = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/train.csv", **read_kwargs
)
df_test_og = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/test.csv", **read_kwargs
)
submission = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/sample_submission.csv", **read_kwargs
)



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
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
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
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )

    return df




## === cell 6
df_train = reduce_mem_usage(df_train_og)
df_test = reduce_mem_usage(df_test_og)
del df_train_og
del df_test_og
gc.collect()



## === cell 7
cat_count = collections.Counter(df_train["Cover_Type"])
cat_freq = cat_count.values()
cat = cat_count.keys()
plt.bar(cat, cat_freq)

print(cat_count)



## === cell 8
df_train = df_train[(df_train["Cover_Type"] != 4) & (df_train["Cover_Type"] != 5)]



## === cell 9
X = df_train.drop(columns=["Id", "Cover_Type", "Soil_Type7", "Soil_Type15"])
y = df_train["Cover_Type"]

if RandomUnderSampler is None:
    tmp = X.copy()
    tmp["_target_"] = y.values
    class_counts = tmp["_target_"].value_counts()
    minority_count = int(class_counts.min())

    def _downsample_group(g):
        if len(g) <= minority_count:
            return g
        return g.sample(n=minority_count, replace=False, random_state=42)

    tmp_res = (
        tmp.groupby("_target_", group_keys=False)
        .apply(_downsample_group)
        .reset_index(drop=True)
    )
    X_res = tmp_res.drop(columns=["_target_"])
    y_res = tmp_res["_target_"]
    rus = None
else:
    rus = RandomUnderSampler(sampling_strategy="not minority", random_state=42)
    X_res, y_res = rus.fit_resample(X, y)

del X, y, df_train
gc.collect()



## === cell 10
cat_count = collections.Counter(y_res)
cat_freq = cat_count.values()
cat = cat_count.keys()
plt.bar(cat, cat_freq)

print(cat_count)



## === cell 11
from sklearn.feature_selection import SelectKBest, f_classif

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



## === cell 12
featurescores.sort_values(by="score", ascending=False)



## === cell 13
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



## === cell 14
X_res = X_res[useful_features]



## === cell 15
x_train, x_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, shuffle=True
)



## === cell 16
label_classes = np.sort(pd.unique(y_res))
label_map = {c: i for i, c in enumerate(label_classes)}

y_train_enc = y_train.map(label_map).astype(np.int32)
y_test_enc = y_test.map(label_map).astype(np.int32)

scaler = StandardScaler()
X_train_np = np.ascontiguousarray(x_train.to_numpy(dtype=np.float32, copy=False))
X_test_np = np.ascontiguousarray(x_test.to_numpy(dtype=np.float32, copy=False))
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)

del X_res, y_res
gc.collect()




## === cell 17
def _xgb_has_gpu() -> bool:
    try:
        import xgboost as xgb

        X_dummy = np.array([[0.0], [1.0]], dtype=np.float32)
        y_dummy = np.array([0, 1], dtype=np.int32)
        dtrain = xgb.DMatrix(X_dummy, label=y_dummy)
        xgb.train(
            {"tree_method": "gpu_hist", "max_depth": 1, "verbosity": 0},
            dtrain,
            num_boost_round=1,
        )
        return True
    except Exception:
        return False


_TREE_METHOD = "gpu_hist" if _xgb_has_gpu() else "hist"

sampler = optuna.samplers.TPESampler(seed=42)


def objective_xgb(trial):
    xgb_params = {
        "learning_rate": 0.01,
        "tree_method": _TREE_METHOD,
        "booster": "gbtree",
        "n_estimators": trial.suggest_int("n_estimators", 500, 4000, 100),
        "reg_lambda": trial.suggest_int("reg_lambda", 1, 100),
        "reg_alpha": trial.suggest_int("reg_alpha", 1, 100),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0, step=0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        "gamma": trial.suggest_float("gamma", 0, 20),
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": 0,
    }

    model = XGBClassifier(**xgb_params)
    model.fit(X_train_scaled, y_train_enc)
    y_pred = model.predict(X_test_scaled)
    return accuracy_score(y_test_enc, y_pred)


study_xgb = optuna.create_study(direction="maximize", sampler=sampler)
study_xgb.optimize(objective_xgb, n_trials=50)



## === cell 18
best_params_xgb = study_xgb.best_params



## === cell 19
final_params = dict(best_params_xgb)
final_params.update(
    {
        "learning_rate": 0.01,
        "tree_method": _TREE_METHOD,
        "booster": "gbtree",
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": 0,
    }
)

pipe = Pipeline(
    steps=[
        ("step1", scaler),  # reuse already-fitted scaler object
        ("step2", XGBClassifier(**final_params)),
    ]
)



## === cell 20
pipe.fit(x_train, y_train_enc)



## === cell 21
df_test = df_test[useful_features]

X_test_full = np.ascontiguousarray(df_test.to_numpy(dtype=np.float32, copy=False))
Final_pred = pipe.named_steps["step2"].predict(
    pipe.named_steps["step1"].transform(X_test_full)
)



## === cell 22
submission["Cover_Type"] = Final_pred
submission.to_csv("Submission.csv", index=False)
