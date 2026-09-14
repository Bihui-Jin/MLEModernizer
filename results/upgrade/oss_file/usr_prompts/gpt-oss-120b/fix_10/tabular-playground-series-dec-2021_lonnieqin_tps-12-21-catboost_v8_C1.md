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
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
import gc




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
    try:
        import kaggle
    except Exception:
        pass  # ignore if kaggle CLI not available



## === cell 3
usecols = (
    None  # None loads all columns; we will drop id after loading to keep code simple
)
train = pd.read_csv(
    config.data_path + "train.csv",
    dtype=np.float32,
    low_memory=False,
)
test = pd.read_csv(
    config.data_path + "test.csv",
    dtype=np.float32,
    low_memory=False,
)
sample_submission = pd.read_csv(config.data_path + "sample_submission.csv")



## === cell 4
if 5 in train[config.label_name].unique():
    train = train[train[config.label_name] != 5]



## === cell 5
train.pop(config.id_field)
test.pop(config.id_field)



## === cell 6
train_split = train.sample(frac=0.85, random_state=42)
val_split = train.drop(train_split.index).reset_index(drop=True)
train_split = train_split.reset_index(drop=True)

train_targets = train_split.pop(config.label_name).astype(np.int32)
val_targets = val_split.pop(config.label_name).astype(np.int32)



## === cell 7
cols = train_split.columns
for df in (train_split, val_split, test):
    arr = df[cols].to_numpy(dtype=np.float32)
    df["mean"] = arr.mean(axis=1).astype(np.float32)
    df["min"] = arr.min(axis=1).astype(np.float32)
    df["max"] = arr.max(axis=1).astype(np.float32)
    df["std"] = arr.std(axis=1, ddof=1).astype(np.float32)
    del arr  # free memory immediately

train_X = train_split
val_X = val_split
test_X = test

del train, test, train_split, val_split, sample_submission
gc.collect()



## === cell 8
cat_params = {
    "iterations": 2000,
    "learning_rate": 0.1,
    "od_wait": 100,
    "depth": 7,
    "l2_leaf_reg": 3,
    "eval_metric": "Accuracy",
    "verbose": False,
    "thread_count": -1,
    "allow_writing_files": False,
    "random_seed": 42,
}
cat = CatBoostClassifier(**cat_params)

train_pool = Pool(data=train_X, label=train_targets.values)
val_pool = Pool(data=val_X, label=val_targets.values)

cat.fit(train_pool, eval_set=val_pool)



## === cell 9
y_pred = cat.predict(test_X)
sample_submission = pd.read_csv(config.data_path + "sample_submission.csv")
sample_submission[config.label_name] = y_pred.reshape(-1)
sample_submission.to_csv(config.submit_filename, index=False)
if not config.is_kaggle_platform:
    pass
