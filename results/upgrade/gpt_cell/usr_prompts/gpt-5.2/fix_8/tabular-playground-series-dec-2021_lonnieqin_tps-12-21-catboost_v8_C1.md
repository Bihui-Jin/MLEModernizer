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

No external packages required in the script and installed.

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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split




## === cell 1
class Config:
    is_kaggle_platform = os.path.exists("/kaggle/input")
    dataset_name = "tabular-playground-series-dec-2021"
    data_path = "/kaggle/input/%s/" % (dataset_name) if is_kaggle_platform else ""
    submit_filename = "submission.csv"
    label_name = "Cover_Type"
    id_field = "Id"


config = Config()



## === cell 2
if not config.is_kaggle_platform:
    raise RuntimeError(
        "This optimized script is intended for Kaggle where /kaggle/input exists. "
        "Non-Kaggle download cells were removed to avoid timeouts and notebook-magics in scripts."
    )



## === cell 3
train_path = config.data_path + "train.csv"
test_path = config.data_path + "test.csv"
sub_path = config.data_path + "sample_submission.csv"

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
feature_cols = [c for c in train_cols if c not in (config.id_field, config.label_name)]

usecols_train = [config.id_field] + feature_cols + [config.label_name]
usecols_test = [config.id_field] + feature_cols

dtype_map_train = {}
dtype_map_test = {}
for c in feature_cols:
    if c.startswith("Wilderness_Area") or c.startswith("Soil_Type"):
        dtype_map_train[c] = np.uint8
        dtype_map_test[c] = np.uint8
    else:
        dtype_map_train[c] = np.int32
        dtype_map_test[c] = np.int32

dtype_map_train[config.id_field] = np.int32
dtype_map_test[config.id_field] = np.int32
dtype_map_train[config.label_name] = np.uint8

_read_kwargs = dict(low_memory=False)
try:
    train = pd.read_csv(
        train_path,
        usecols=usecols_train,
        dtype=dtype_map_train,
        engine="pyarrow",
        **_read_kwargs
    )
    test = pd.read_csv(
        test_path,
        usecols=usecols_test,
        dtype=dtype_map_test,
        engine="pyarrow",
        **_read_kwargs
    )
except Exception:
    train = pd.read_csv(
        train_path,
        usecols=usecols_train,
        dtype=dtype_map_train,
        engine="c",
        **_read_kwargs
    )
    test = pd.read_csv(
        test_path,
        usecols=usecols_test,
        dtype=dtype_map_test,
        engine="c",
        **_read_kwargs
    )

sample_submission = pd.read_csv(sub_path)



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
pass



## === cell 11
pass



## === cell 12
pass



## === cell 13
mask = train[config.label_name].to_numpy(copy=False) == 5
idx0 = int(np.flatnonzero(mask)[0])
train = train.drop(index=train.index[idx0])



## === cell 14
train.pop(config.id_field)
_ = test.pop(config.id_field)



## === cell 15
pass



## === cell 16
y_all = train[config.label_name].to_numpy(copy=False)
X_df = train.drop(columns=[config.label_name])

feature_names = X_df.columns.tolist()

n = len(X_df)
idx = np.arange(n, dtype=np.int32)
train_idx, val_idx = train_test_split(idx, test_size=0.15, random_state=42)

X_train = np.ascontiguousarray(X_df.to_numpy(dtype=np.float32, copy=False)[train_idx])
X_val = np.ascontiguousarray(X_df.to_numpy(dtype=np.float32, copy=False)[val_idx])
y_train = y_all[train_idx]
y_val = y_all[val_idx]




## === cell 17
def row_stats(x: np.ndarray):
    mean = x.mean(axis=1, dtype=np.float32)
    minv = x.min(axis=1)
    maxv = x.max(axis=1)

    n_feat = x.shape[1]
    ex2 = (x * x).mean(axis=1, dtype=np.float32)
    var_pop = ex2 - mean * mean
    np.maximum(var_pop, 0.0, out=var_pop)
    var_samp = var_pop * (n_feat / (n_feat - 1))
    std = np.sqrt(var_samp, dtype=np.float32)
    return mean, minv, maxv, std


def augment_with_row_stats(x: np.ndarray) -> np.ndarray:
    m, mi, ma, s = row_stats(x)
    n_rows, n_feat = x.shape
    out = np.empty((n_rows, n_feat + 4), dtype=np.float32)
    out[:, :n_feat] = x
    out[:, n_feat + 0] = m
    out[:, n_feat + 1] = mi
    out[:, n_feat + 2] = ma
    out[:, n_feat + 3] = s
    return np.ascontiguousarray(out)


X_train_aug = augment_with_row_stats(X_train)
X_val_aug = augment_with_row_stats(X_val)

X_test = np.ascontiguousarray(
    test[feature_names].to_numpy(dtype=np.float32, copy=False)
)
X_test_aug = augment_with_row_stats(X_test)

aug_feature_names = feature_names + ["mean", "min", "max", "std"]

train_features = X_train_aug
val_features = X_val_aug
test_features = X_test_aug

train_targets = y_train
val_targets = y_val



## === cell 18
cat_params = {
    "iterations": 15000,
    "learning_rate": 0.1,
    "od_wait": 1000,
    "depth": 7,
    "task_type": "CPU",
    "l2_leaf_reg": 3,
    "eval_metric": "Accuracy",
    "verbose": 1000,
    "random_seed": 42,
    "use_best_model": True,
}

train_pool = Pool(train_features, train_targets, feature_names=aug_feature_names)
val_pool = Pool(val_features, val_targets, feature_names=aug_feature_names)
test_pool = Pool(test_features, feature_names=aug_feature_names)

cat = CatBoostClassifier(**cat_params)
cat.fit(train_pool, eval_set=val_pool)

y_pred = cat.predict(test_pool)
sample_submission[config.label_name] = y_pred.reshape(-1)
sample_submission.to_csv(config.submit_filename, index=False)
