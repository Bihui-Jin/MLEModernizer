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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from pathlib import Path



## === cell 1
TRAIN_PATH = "../input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "../input/tabular-playground-series-dec-2021/test.csv"
SUB_PATH = "../input/tabular-playground-series-dec-2021/sample_submission.csv"

train_cols = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
test_cols = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()

feature_cols = [c for c in train_cols if c not in ("Id", "Cover_Type")]
usecols_train = ["Id"] + feature_cols + ["Cover_Type"]
usecols_test = ["Id"] + feature_cols

dtype_map = {c: np.int32 for c in feature_cols}
dtype_map.update({"Id": np.int32, "Cover_Type": np.int8})

read_csv_kwargs = dict(low_memory=False)

cache_dir = Path("../working")
cache_dir.mkdir(parents=True, exist_ok=True)
train_feather = cache_dir / "tps_dec2021_train_cached.feather"
test_feather = cache_dir / "tps_dec2021_test_cached.feather"


def _read_train_test():
    if train_feather.exists() and test_feather.exists():
        train_df = pd.read_feather(train_feather)
        test_df = pd.read_feather(test_feather)
        return train_df, test_df

    try:
        train_df = pd.read_csv(
            TRAIN_PATH,
            usecols=usecols_train,
            dtype=dtype_map,
            engine="pyarrow",
            **read_csv_kwargs,
        )
        test_df = pd.read_csv(
            TEST_PATH,
            usecols=usecols_test,
            dtype={k: v for k, v in dtype_map.items() if k != "Cover_Type"},
            engine="pyarrow",
            **read_csv_kwargs,
        )
    except Exception:
        train_df = pd.read_csv(
            TRAIN_PATH, usecols=usecols_train, dtype=dtype_map, **read_csv_kwargs
        )
        test_df = pd.read_csv(
            TEST_PATH,
            usecols=usecols_test,
            dtype={k: v for k, v in dtype_map.items() if k != "Cover_Type"},
            **read_csv_kwargs,
        )

    train_df.to_feather(train_feather)
    test_df.to_feather(test_feather)
    return train_df, test_df


train, test = _read_train_test()
sample_submission = pd.read_csv(SUB_PATH)



## === cell 2
y_train = train["Cover_Type"]
X_train = train[feature_cols]
X_test = test[feature_cols]



## === cell 3
from catboost import CatBoostClassifier, Pool

train_pool = Pool(X_train, label=y_train)
test_pool = Pool(X_test)

clf_CatBoostClassifier = CatBoostClassifier(
    verbose=0,
    task_type="CPU",
    thread_count=os.cpu_count(),
    random_seed=0,
)

clf_CatBoostClassifier.fit(train_pool)
sample_submission.iloc[:, 1] = clf_CatBoostClassifier.predict(test_pool)
sample_submission.to_csv("submission_CatBoostClassifier.csv", index=False)
sample_submission.head()



## === cell 4
train.head()
