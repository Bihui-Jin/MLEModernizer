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
import gc

if False:
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(os.path.join(dirname, filename))




## === cell 1
def _resolve_input_path(p: str) -> str:
    if os.path.exists(p):
        return p
    if p.startswith("../input/"):
        alt = p.replace("../input/", "/kaggle/input/", 1)
        if os.path.exists(alt):
            return alt
    if p.startswith("/kaggle/input/"):
        alt = p.replace("/kaggle/input/", "../input/", 1)
        if os.path.exists(alt):
            return alt
    return p


train_path = _resolve_input_path(
    "../input/tabular-playground-series-dec-2021/train.csv"
)
test_path = _resolve_input_path("../input/tabular-playground-series-dec-2021/test.csv")
sub_path = _resolve_input_path(
    "../input/tabular-playground-series-dec-2021/sample_submission.csv"
)

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
feature_cols = [c for c in train_cols if c not in ("Id", "Cover_Type")]

usecols_train = ["Cover_Type"] + feature_cols
usecols_test = feature_cols

dtype_train = {c: np.int32 for c in feature_cols}
dtype_train.update({"Cover_Type": np.int8})
dtype_test = {c: np.int32 for c in feature_cols}

sample_submission = pd.read_csv(
    sub_path,
    usecols=["Id"],
    dtype={"Id": np.int32},
)



## === cell 2
gc.collect()



## === cell 3
params = {
    "learning_rate": 0.37644647769699235,
    "depth": 10,
    "one_hot_max_size": 4,
    "l2_leaf_reg": 0.05846053355686806,
}



## === cell 4
from catboost import CatBoostClassifier, Pool

_env_tc = os.environ.get("OMP_NUM_THREADS", None)
if _env_tc is not None:
    thread_count = int(_env_tc)
else:
    thread_count = int(os.cpu_count() or 1)
thread_count = max(1, min(thread_count, 16))

os.environ["OMP_NUM_THREADS"] = str(thread_count)
os.environ["OPENBLAS_NUM_THREADS"] = str(thread_count)
os.environ["MKL_NUM_THREADS"] = str(thread_count)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(thread_count)
os.environ["NUMEXPR_NUM_THREADS"] = str(thread_count)

clf_CatBoostClassifier = CatBoostClassifier(
    **params,
    verbose=0,
    task_type="CPU",
    thread_count=thread_count,
    allow_writing_files=False,
    used_ram_limit="10gb",
    border_count=128,
    boosting_type="Plain",
    random_seed=0,
)

train_df = pd.read_csv(train_path, usecols=usecols_train, dtype=dtype_train)
X_train = train_df[feature_cols]
y_train = train_df["Cover_Type"].astype(np.int32, copy=False)

test_df = pd.read_csv(test_path, usecols=usecols_test, dtype=dtype_test)
X_test = test_df[feature_cols]

del train_df, test_df
gc.collect()

train_pool = Pool(data=X_train, label=y_train, feature_names=feature_cols)
test_pool = Pool(data=X_test, feature_names=feature_cols)

clf_CatBoostClassifier.fit(train_pool)

pred = (
    clf_CatBoostClassifier.predict(test_pool).reshape(-1).astype(np.int32, copy=False)
)

sample_submission["Cover_Type"] = pred
sample_submission.to_csv("submission_CatBoostClassifier.csv", index=False)
sample_submission.head()



## === cell 5
if False:
    train.head()



## === cell 6
if False:
    train.iloc[:, 1:11].hist(figsize=(20, 10), bins=50)



## === cell 7
if False:
    train["Cover_Type"].hist(figsize=(3, 3))



## === cell 8
if False:
    train["Cover_Type"].value_counts()



## === cell 9
if False:
    train.iloc[:, 11:25].describe()



## === cell 10
if False:
    train.iloc[:, 25:40].describe()



## === cell 11
if False:
    train.iloc[:, 40:55].describe()



## === cell 12
if False:
    test.iloc[:, 1:11].hist(figsize=(20, 10), bins=50)



## === cell 13
if False:
    test.iloc[:, 11:25].describe()



## === cell 14
if False:
    test.iloc[:, 25:40].describe()



## === cell 15
if False:
    test.iloc[:, 40:55].describe()
