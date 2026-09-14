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
import os
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split





## === cell 1
class Config:
    is_kaggle_platform = os.path.exists("/kaggle/input")
    dataset_name = "tabular-playground-series-dec-2021"
    data_path = f"/kaggle/input/{dataset_name}/" if is_kaggle_platform else ""
    submit_filename = "submission.csv"
    label_name = "Cover_Type"
    id_field = "Id"

    run_eda = False


config = Config()



## === cell 2
if not config.is_kaggle_platform:
    try:
        import kaggle  # noqa: F401
    except Exception:
        import sys, subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    if not os.path.exists("/root/.kaggle/kaggle.json"):
        raise RuntimeError(
            "kaggle.json not found; this block is intended only for non-Kaggle environments with credentials configured."
        )
    os.system(f"kaggle competitions download -c {config.dataset_name}")
    os.system("unzip -o test.csv.zip")
    os.system("unzip -o train.csv.zip")
    os.system("unzip -o sample_submission.csv.zip")



## === cell 3
train_path = config.data_path + "train.csv"
test_path = config.data_path + "test.csv"
sub_path = config.data_path + "sample_submission.csv"

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()

dtype_map_train = {}
for c in train_cols:
    if c == config.id_field or c == config.label_name:
        dtype_map_train[c] = np.int32
    else:
        dtype_map_train[c] = np.int32

dtype_map_test = {
    c: (np.int32 if c == config.id_field else np.int32) for c in test_cols
}

train = pd.read_csv(train_path, dtype=dtype_map_train, engine="c")
test = pd.read_csv(test_path, dtype=dtype_map_test, engine="c")
sample_submission = pd.read_csv(sub_path, engine="c")



## === cell 4
if config.run_eda:
    train.head()



## === cell 5
if config.run_eda:
    train.info()



## === cell 6
if config.run_eda:
    train.describe()



## === cell 7
if config.run_eda:
    corr = train.corr(numeric_only=True)
    corr



## === cell 8
if config.run_eda:
    corr.sort_values(
        ascending=False, inplace=True, by=config.label_name, key=lambda x: abs(x)
    )
    corr[config.label_name]



## === cell 9
if config.run_eda:
    correlation_score = train.corr(numeric_only=True)
    correlated_features = (
        correlation_score[config.label_name].sort_values(ascending=False).dropna()
    )
    correlated_columns = list(
        correlated_features[correlated_features.abs() > 0.05].index
    )
    if config.label_name in correlated_columns:
        correlated_columns.remove(config.label_name)
    print(correlated_columns)
else:
    correlated_columns = []



## === cell 10
if config.run_eda and correlated_columns:
    corr2 = train[correlated_columns].corr(numeric_only=True)
    corr2



## === cell 11
if config.run_eda and correlated_columns:
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 20))
    sns.heatmap(corr2, annot=False)



## === cell 12
if config.run_eda:
    import seaborn as sns

    sns.countplot(x=config.label_name, data=train)



## === cell 13
if config.run_eda:
    train[config.label_name].value_counts()



## === cell 14
train = train.loc[train[config.label_name] != 5].reset_index(drop=True)



## === cell 15
test_ids = test[config.id_field].copy()
train.pop(config.id_field)
_ = test.pop(config.id_field)



## === cell 16
null_counts_train = train.isnull().sum()
mask_train = null_counts_train.to_numpy() > 0
if mask_train.any():
    print(null_counts_train[null_counts_train > 0])

null_counts_test = test.isnull().sum()
mask_test = null_counts_test.to_numpy() > 0
if mask_test.any():
    print(null_counts_test[null_counts_test > 0])



## === cell 17
train_features, val_features = train_test_split(train, test_size=0.15, random_state=42)
train_targets = train_features.pop(config.label_name)
val_targets = val_features.pop(config.label_name)
train_features.head()



## === cell 18
cols = train_features.columns


def add_row_stats(df: pd.DataFrame, cols_index) -> None:
    x = df[cols_index].to_numpy()
    df["mean"] = x.mean(axis=1)
    df["min"] = x.min(axis=1)
    df["max"] = x.max(axis=1)
    df["std"] = x.std(axis=1, ddof=1)


for data in (train_features, val_features, test):
    add_row_stats(data, cols)



## === cell 19
cat_params = {
    "iterations": 15000,
    "learning_rate": 0.1,
    "od_wait": 1000,
    "depth": 7,
    "l2_leaf_reg": 3,
    "eval_metric": "Accuracy",
    "verbose": 1000,
    "loss_function": "MultiClass",
    "random_seed": 42,
    "task_type": "CPU",
    "thread_count": -1,
    "allow_writing_files": False,
    "use_best_model": True,
}
cat = CatBoostClassifier(**cat_params)
cat.fit(train_features, train_targets, eval_set=(val_features, val_targets))



## === cell 20
y_pred = cat.predict(test)
y_pred = np.asarray(y_pred).reshape(-1).astype(int)

submission = pd.DataFrame({config.id_field: test_ids.values, config.label_name: y_pred})
submission.to_csv(config.submit_filename, index=False)

if not config.is_kaggle_platform:
    os.system(
        f'kaggle competitions submit {config.dataset_name} -m "Submission" -f {config.submit_filename}'
    )
