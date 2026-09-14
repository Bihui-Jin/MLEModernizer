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
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

import collections

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

from xgboost import XGBClassifier
import optuna

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DO_PLOTS = False



## === cell 1
base_path = "../input/tabular-playground-series-dec-2021"
if not os.path.exists(base_path):
    base_path = "/kaggle/input/tabular-playground-series-dec-2021"

train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")
sub_path = os.path.join(base_path, "sample_submission.csv")


def _build_read_dtypes(path, is_train: bool):
    dtypes = {"Id": np.int32}
    if is_train:
        dtypes["Cover_Type"] = np.int8
    return dtypes


df_train_og = pd.read_csv(train_path, dtype=_build_read_dtypes(train_path, True))
df_test_og = pd.read_csv(test_path, dtype=_build_read_dtypes(test_path, False))
submission = pd.read_csv(sub_path)



## === cell 2
_ = df_train_og.shape



## === cell 3
_ = df_train_og.head(1)



## === cell 4
if False:
    _ = df_train_og.nunique()




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
df_train = reduce_mem_usage(df_train_og, verbose=True)
df_test = reduce_mem_usage(df_test_og, verbose=True)
del df_train_og
del df_test_og



## === cell 7
cat_count = collections.Counter(df_train["Cover_Type"].values)
if DO_PLOTS:
    import matplotlib.pyplot as plt

    cat_freq = list(cat_count.values())
    cat = list(cat_count.keys())
    plt.figure(figsize=(8, 4))
    plt.bar(cat, cat_freq)
    plt.title("Cover_Type class counts (original)")
    plt.show()
print(cat_count)



## === cell 8
df_train = df_train[(df_train["Cover_Type"] != 4) & (df_train["Cover_Type"] != 5)]



## === cell 9
df_train = df_train.reset_index(drop=True)

drop_cols = [
    c
    for c in ["Id", "Cover_Type", "Soil_Type7", "Soil_Type15"]
    if c in df_train.columns
]

y = df_train["Cover_Type"]
class_counts = y.value_counts()
min_count = int(class_counts.min())

parts = []
for cls, grp_idx in df_train.groupby("Cover_Type", sort=False).groups.items():
    idx = np.fromiter(grp_idx, dtype=np.int64, count=len(grp_idx))
    if idx.size > min_count:
        rng = np.random.RandomState(RANDOM_STATE + int(cls))
        idx = rng.choice(idx, size=min_count, replace=False)
    parts.append(idx)

all_idx = np.concatenate(parts)
rng = np.random.RandomState(RANDOM_STATE)
rng.shuffle(all_idx)

df_bal = df_train.iloc[all_idx].reset_index(drop=True)

X_res = df_bal.drop(columns=drop_cols)
y_res = df_bal["Cover_Type"]

classes_sorted = np.sort(y_res.unique())
class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
y_res_enc = y_res.map(class_to_idx).astype(np.int32)



## === cell 10
cat_count = collections.Counter(y_res.values)
if DO_PLOTS:
    import matplotlib.pyplot as plt

    cat_freq = list(cat_count.values())
    cat = list(cat_count.keys())
    plt.figure(figsize=(8, 4))
    plt.bar(cat, cat_freq)
    plt.title("Cover_Type class counts (after under-sampling)")
    plt.show()
print(cat_count)



## === cell 11
X_res_np = np.ascontiguousarray(X_res.values)
y_res_enc_np = np.ascontiguousarray(y_res_enc.values)

selector = SelectKBest(f_classif, k="all")
fitter = selector.fit(X_res_np, y_res_enc_np)

featurescores = pd.DataFrame({"score": fitter.scores_, "column name": X_res.columns})

if DO_PLOTS:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 5))
    plt.bar(featurescores["column name"], featurescores["score"], width=0.4)
    plt.xticks(rotation="vertical")
    plt.title("ANOVA F-scores (all features)")
    plt.show()



## === cell 12
featurescores = featurescores.sort_values(by="score", ascending=False)



## === cell 13
useful_features = featurescores["column name"].head(20).tolist()



## === cell 14
_ = useful_features



## === cell 15
X_res = X_res[useful_features]
X_res_np = np.ascontiguousarray(X_res.values)
y_res_enc_np = np.ascontiguousarray(y_res_enc.values)



## === cell 16
x_train, x_test, y_train, y_test = train_test_split(
    X_res_np,
    y_res_enc_np,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_res_enc_np,
)

_scaler = StandardScaler()
x_train_scaled = _scaler.fit_transform(x_train)
x_test_scaled = _scaler.transform(x_test)




## === cell 17
def objective_xgb(trial):
    xgb_params = {
        "learning_rate": 0.01,
        "tree_method": "hist",  # CPU-stable on Kaggle
        "booster": "gbtree",
        "n_estimators": trial.suggest_int("n_estimators", 500, 4000, 100),
        "reg_lambda": trial.suggest_int("reg_lambda", 1, 100),
        "reg_alpha": trial.suggest_int("reg_alpha", 1, 100),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0, step=0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
        "gamma": trial.suggest_float("gamma", 0, 20),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "objective": "multi:softmax",
        "num_class": int(np.unique(y_res_enc_np).shape[0]),
        "eval_metric": "mlogloss",
    }

    model = XGBClassifier(**xgb_params)
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    return accuracy_score(y_test, y_pred)




## === cell 18
sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
study_xgb = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

study_xgb.optimize(objective_xgb, n_trials=50, catch=(Exception,))



## === cell 19
best_params_xgb = {}
try:
    if len(study_xgb.trials) > 0:
        _ = study_xgb.best_trial  # may raise if none completed
        best_params_xgb = dict(study_xgb.best_params)
except Exception:
    best_params_xgb = {}

best_params_xgb



## === cell 20
final_params = dict(best_params_xgb)
final_params.update(
    {
        "learning_rate": 0.01,
        "tree_method": "hist",
        "booster": "gbtree",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "objective": "multi:softmax",
        "num_class": int(np.unique(y_res_enc_np).shape[0]),
        "eval_metric": "mlogloss",
    }
)

if "n_estimators" not in final_params:
    final_params.update(
        {
            "n_estimators": 2000,
            "reg_lambda": 10,
            "reg_alpha": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "max_depth": 8,
            "min_child_weight": 2,
            "gamma": 0.0,
        }
    )

pipe = Pipeline(
    steps=[
        ("step1", StandardScaler()),
        ("step2", XGBClassifier(**final_params)),
    ]
)



## === cell 21
pipe.fit(x_train, y_train)



## === cell 22
test_drop = [c for c in ["Id", "Soil_Type7", "Soil_Type15"] if c in df_test.columns]
df_test_fe = df_test.drop(columns=test_drop)
df_test_fe = df_test_fe[useful_features]
df_test_np = np.ascontiguousarray(df_test_fe.values)

final_pred_enc = pipe.predict(df_test_np).astype(int)
Final_pred = pd.Series(final_pred_enc).map(idx_to_class).astype(int).values

submission = submission.copy()
submission["Cover_Type"] = Final_pred.astype(int)
submission[["Id", "Cover_Type"]].to_csv("submission.csv", index=False)

submission.head()
