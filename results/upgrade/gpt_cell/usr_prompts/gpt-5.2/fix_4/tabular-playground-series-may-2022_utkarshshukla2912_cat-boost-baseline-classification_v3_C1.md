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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

plt.rcParams["figure.figsize"] = (20, 10)
plt.style.use("ggplot")

np.random.seed(42)



## === cell 1
if False:
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(os.path.join(dirname, filename))



## === cell 2
train_path = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
test_path = "/kaggle/input/tabular-playground-series-may-2022/test.csv"

categorical_columns = [
    f"f_{i}" if len(str(i)) == 2 else f"f_0{i}" for i in range(7, 19)
] + ["f_27", "f_29", "f_30"]
real_value_columns = [f"f_0{i}" for i in range(0, 7)] + [
    f"f_{i}" for i in range(19, 29)
]
real_value_columns.pop(real_value_columns.index("f_27"))

dtype_map_train = {"id": "int32", "target": "int8"}
dtype_map_test = {"id": "int32"}
for c in categorical_columns:
    dtype_map_train[c] = "string"
    dtype_map_test[c] = "string"
for c in real_value_columns:
    dtype_map_train[c] = "float32"
    dtype_map_test[c] = "float32"

try:
    train_df = pd.read_csv(train_path, engine="pyarrow", dtype=dtype_map_train)
    test_df = pd.read_csv(test_path, engine="pyarrow", dtype=dtype_map_test)
except Exception:
    train_df = pd.read_csv(train_path, dtype=dtype_map_train)
    test_df = pd.read_csv(test_path, dtype=dtype_map_test)



## === cell 3
for c in categorical_columns:
    train_df[c] = train_df[c].astype("category")
    test_df[c] = test_df[c].astype("category")

feature_cols = [c for c in train_df.columns if c not in ("id", "target")]
X_all = train_df[feature_cols]
y_all = train_df["target"]



## === cell 4
x_train, x_rem, y_train, y_rem = train_test_split(
    X_all,
    y_all,
    test_size=0.3,
    random_state=42,
)
x_val, x_test, y_val, y_test = train_test_split(
    x_rem,
    y_rem,
    test_size=0.3,
    random_state=42,
)



## === cell 5
train_pool = Pool(x_train, y_train, cat_features=categorical_columns)
val_pool = Pool(x_val, y_val, cat_features=categorical_columns)

model = CatBoostClassifier(
    cat_features=categorical_columns,
    n_estimators=10000,
    learning_rate=0.10315154739037707,
    depth=3,
    l2_leaf_reg=1,
    task_type="CPU",
    thread_count=-1,
    verbose=False,
    random_seed=42,
)

model.fit(train_pool, eval_set=val_pool, use_best_model=True)



## === cell 6
if False:
    print(classification_report(y_test, model.predict(x_test)))



## === cell 7
if False:
    sns.lineplot(x=model.feature_names_, y=model.feature_importances_, marker="o")



## === cell 8
X_test_final = test_df[feature_cols]
test_pool = Pool(X_test_final, cat_features=categorical_columns)

prediction = model.predict_proba(test_pool)[:, 1]



## === cell 9
prediction_df = test_df[["id"]].copy()
prediction_df["target"] = prediction



## === cell 10
prediction_df.to_csv("submission.csv", index=False)
