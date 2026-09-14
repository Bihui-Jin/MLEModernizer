# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

catboost==1.2.8
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
import os

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 1))




## === cell 1
train_path = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
test_path = "/kaggle/input/tabular-playground-series-may-2022/test.csv"

all_feature_cols = [f"f_{i:02d}" for i in range(0, 31)]
usecols_train = ["id"] + all_feature_cols + ["target"]
usecols_test = ["id"] + all_feature_cols

categorical_feature_cols = ["f_27", "f_29", "f_30"]
numeric_feature_cols = [
    c for c in all_feature_cols if c not in categorical_feature_cols
]

dtype_train = {
    "id": "int32",
    "target": "int8",
    **{c: "float32" for c in numeric_feature_cols},
    **{c: "string" for c in categorical_feature_cols},
}
dtype_test = {
    "id": "int32",
    **{c: "float32" for c in numeric_feature_cols},
    **{c: "string" for c in categorical_feature_cols},
}

train_df = pd.read_csv(train_path, usecols=usecols_train, dtype=dtype_train)
test_df = pd.read_csv(test_path, usecols=usecols_test, dtype=dtype_test)




## === cell 2
categorical_columns = [
    f"f_{i}" if len(str(i)) == 2 else f"f_0{i}" for i in range(7, 19)
] + [
    "f_27",
    "f_29",
    "f_30",
]
real_value_columns = [f"f_0{i}" for i in range(0, 7)] + [
    f"f_{i}" for i in range(19, 29)
]
real_value_columns.pop(real_value_columns.index("f_27"))




## === cell 3
feature_cols = [c for c in train_df.columns if c not in ("id", "target")]
cat_feature_indices = [feature_cols.index(c) for c in categorical_columns]

for c in categorical_columns:
    combined = pd.concat([train_df[c], test_df[c]], axis=0, ignore_index=True)
    cats = pd.Categorical(combined, ordered=False)
    train_df[c] = pd.Categorical(
        train_df[c], categories=cats.categories, ordered=False
    ).codes.astype(np.int32, copy=False)
    test_df[c] = pd.Categorical(
        test_df[c], categories=cats.categories, ordered=False
    ).codes.astype(np.int32, copy=False)

X = train_df[feature_cols]
y_np = train_df["target"].to_numpy(copy=False)

train_pool = Pool(X, y_np, cat_features=cat_feature_indices)




## === cell 4
model = CatBoostClassifier(
    cat_features=cat_feature_indices,
    n_estimators=7000,
    learning_rate=0.10315154739037707,
    depth=2,
    l2_leaf_reg=1,
    task_type="CPU",
    verbose=300,
    random_seed=RANDOM_STATE,
    thread_count=(os.cpu_count() or 1),
    boosting_type="Ordered",
)

model.fit(train_pool)




## === cell 5
pass




## === cell 6
test_pool = Pool(test_df[feature_cols], cat_features=cat_feature_indices)

prediction = model.predict(test_pool)




## === cell 7
prediction_df = pd.DataFrame(
    {"id": test_df["id"].to_numpy(copy=False), "target": prediction.astype(int)}
)
prediction_df.to_csv("submission.csv", index=False)
