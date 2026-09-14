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
import os, warnings, gc

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

np.random.seed(42)




## === cell 1
train_path = "../input/tabular-playground-series-dec-2021/train.csv"
test_path = "../input/tabular-playground-series-dec-2021/test.csv"
sample_sub_path = "../input/tabular-playground-series-dec-2021/sample_submission.csv"

all_cols = pd.read_csv(train_path, nrows=0).columns.tolist()

dtype_map = {c: np.float32 for c in all_cols if c not in ["Id", "Cover_Type"]}
dtype_map["Id"] = np.int32
dtype_map["Cover_Type"] = np.int32

feature_cols = [c for c in all_cols if c not in ["Id", "Cover_Type"]]

train_df = pd.read_csv(
    train_path,
    usecols=feature_cols + ["Cover_Type"],
    dtype=dtype_map,
)

test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()
test_feature_cols = [c for c in test_cols if c != "Id"]
test_df = pd.read_csv(
    test_path,
    usecols=test_feature_cols,
    dtype={c: dtype_map.get(c, np.float32) for c in test_feature_cols},
)

sample_submission = pd.read_csv(sample_sub_path)

gc.collect()




## === cell 2
train_pool = Pool(data=train_df[feature_cols], label=train_df["Cover_Type"])
test_pool = Pool(data=test_df)

del train_df, test_df
gc.collect()




## === cell 3
params = {
    "learning_rate": 0.37644647769699235,
    "depth": 10,
    "one_hot_max_size": 4,
    "l2_leaf_reg": 0.05846053355686806,
    "task_type": "CPU",
    "thread_count": os.cpu_count() or 1,  # use all available CPU cores
    "iterations": 200,
    "random_seed": 42,
    "verbose": 0,
}
clf = CatBoostClassifier(**params)




## === cell 4
clf.fit(train_pool)

sample_submission.iloc[:, 1] = clf.predict(test_pool)
sample_submission.to_csv("submission_CatBoostClassifier.csv", index=False)
sample_submission.head()
