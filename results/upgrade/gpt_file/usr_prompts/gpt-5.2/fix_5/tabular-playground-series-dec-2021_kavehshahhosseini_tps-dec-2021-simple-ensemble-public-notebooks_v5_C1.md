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
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0

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

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

np.random.seed(42)

DATA_DIR_CANDIDATES = [
    "/kaggle/input/tabular-playground-series-dec-2021",
    "/kaggle/data/tabular-playground-series-dec-2021",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename: str) -> str:
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find {filename} in any of: {DATA_DIR_CANDIDATES}"
    )


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sample_path = find_file("sample_submission.csv")

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()

usecols_train = train_cols  # includes Cover_Type
usecols_test = test_cols

dtype_train = {c: np.float32 for c in usecols_train if c not in ("Id", "Cover_Type")}
dtype_train["Id"] = np.int64
dtype_train["Cover_Type"] = np.int64

dtype_test = {c: np.float32 for c in usecols_test if c != "Id"}
dtype_test["Id"] = np.int64

train = pd.read_csv(train_path, usecols=usecols_train, dtype=dtype_train, engine="c")
test = pd.read_csv(test_path, usecols=usecols_test, dtype=dtype_test, engine="c")
submission = pd.read_csv(
    sample_path, usecols=["Id"], dtype={"Id": np.int64}, engine="c"
)

assert "Cover_Type" in train.columns
assert "Id" in train.columns and "Id" in test.columns and "Id" in submission.columns

y_np = train["Cover_Type"].to_numpy(dtype=np.int64, copy=False)

feature_cols = [c for c in train.columns if c != "Cover_Type"]
X = train[feature_cols]
test_X = test.reindex(columns=feature_cols)

X = X.fillna(0)
test_X = test_X.fillna(0)

X_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32, copy=False))
test_X_np = np.ascontiguousarray(test_X.to_numpy(dtype=np.float32, copy=False))

del train, test, X, test_X



## === cell 1
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

TREE_N_JOBS = min(4, (os.cpu_count() or 4))

models = [
    (
        "lr",
        Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=300,
                        n_jobs=-1,  # keep core logic identical
                        multi_class="auto",
                        C=2.0,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
    ),
    ("gnb", GaussianNB()),
    (
        "rf",
        RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            n_jobs=TREE_N_JOBS,
            random_state=42,
            min_samples_split=2,
            min_samples_leaf=1,
        ),
    ),
    (
        "et",
        ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            n_jobs=TREE_N_JOBS,
            random_state=42,
            min_samples_split=2,
            min_samples_leaf=1,
        ),
    ),
]

n_test = test_X_np.shape[0]
pred_mat = np.empty((n_test, len(models)), dtype=np.int64)

for j, (name, model) in enumerate(models):
    model.fit(X_np, y_np)
    pred_mat[:, j] = model.predict(test_X_np).astype(np.int64, copy=False)

print(
    f"Built {pred_mat.shape[1]} local predictors. Each has length: {pred_mat.shape[0]}"
)



## === cell 2
results = pd.DataFrame(
    pred_mat,
    columns=[f"p{i+1}" for i in range(pred_mat.shape[1])],
    index=np.arange(len(submission)),
)

print(results.shape)
results.head()



## === cell 3
arr = pred_mat  # shape (n_samples, n_models)
arr_sorted = np.sort(arr, axis=1)

eq = arr_sorted[:, 1:] == arr_sorted[:, :-1]
bound = np.concatenate([np.ones((arr_sorted.shape[0], 1), dtype=bool), ~eq], axis=1)
run_id = np.cumsum(bound, axis=1) - 1  # 0..n_runs-1 per row

rows = np.repeat(np.arange(arr_sorted.shape[0]), arr_sorted.shape[1])
keys = rows * (arr_sorted.shape[1] + 1) + run_id.ravel()
counts = np.bincount(keys, minlength=arr_sorted.shape[0] * (arr_sorted.shape[1] + 1))
counts = counts.reshape(arr_sorted.shape[0], arr_sorted.shape[1] + 1)

best_run = counts.argmax(axis=1)
mask = run_id == best_run[:, None]
first_pos = mask.argmax(axis=1)
ensemble = arr_sorted[np.arange(arr_sorted.shape[0]), first_pos].astype(
    np.int64, copy=False
)

results["ensemble"] = ensemble
results.head()




## === cell 4
def nunique(a, axis):
    return (np.diff(np.sort(a, axis=axis), axis=axis) != 0).sum(axis=axis) + 1




## === cell 5
results["dif"] = nunique(pred_mat, 1) - 1
results.head()



## === cell 6
_ = results["dif"].value_counts()
_



## === cell 7
submission = submission.copy()
submission["Cover_Type"] = results["ensemble"].to_numpy(dtype=int, copy=False)

assert len(submission) == len(results)
submission.to_csv("submission.csv", index=False)

submission.head()



## === cell 8
if False:
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(x=submission["Cover_Type"])
    plt.title("Predictions")
    plt.xlabel("Cover Type")
    ax.bar_label(ax.containers[0])
    plt.show()
