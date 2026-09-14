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

RANDOM_SEED = 42

os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_SEED))

for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = "1"

import numpy as np
import pandas as pd

np.random.seed(RANDOM_SEED)

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

WORKDIR = "/kaggle/working"
os.makedirs(WORKDIR, exist_ok=True)




## === cell 1
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression




## === cell 2
BASE_DIR = "/kaggle/input/tabular-playground-series-dec-2021"
train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()
assert "Cover_Type" in train_cols
assert "Id" in train_cols and "Id" in test_cols

feature_cols = [c for c in train_cols if c not in ("Id", "Cover_Type")]

dtype_train = {"Id": "int64", "Cover_Type": "int32"}
dtype_test = {"Id": "int64"}
for c in feature_cols:
    dtype_train[c] = "float32"
    dtype_test[c] = "float32"

train_usecols = ["Id"] + feature_cols + ["Cover_Type"]
test_usecols = ["Id"] + feature_cols

train_df = pd.read_csv(train_path, usecols=train_usecols, dtype=dtype_train)
X_np = train_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
y = train_df["Cover_Type"].to_numpy(dtype=np.int32, copy=False).astype(int, copy=False)
del train_df

X_mm_path = os.path.join(WORKDIR, "X_train_float32.memmap")
if os.path.exists(X_mm_path):
    os.remove(X_mm_path)
X = np.memmap(X_mm_path, dtype=np.float32, mode="w+", shape=X_np.shape)
X[:] = X_np
del X_np

test_df = pd.read_csv(test_path, usecols=test_usecols, dtype=dtype_test)
test_ids = test_df["Id"].to_numpy(copy=False)
X_test_np = test_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
del test_df

Xtest_mm_path = os.path.join(WORKDIR, "X_test_float32.memmap")
if os.path.exists(Xtest_mm_path):
    os.remove(Xtest_mm_path)
X_test = np.memmap(Xtest_mm_path, dtype=np.float32, mode="w+", shape=X_test_np.shape)
X_test[:] = X_test_np
del X_test_np

submission = pd.read_csv(sample_sub_path, usecols=["Id"], dtype={"Id": "int64"})

classes = np.unique(y)
n_classes = int(classes.size)
print(
    "Train shape:", X.shape, "Test shape:", X_test.shape, "Classes:", classes.tolist()
)




## === cell 3
def build_clf(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False, copy=False)),
            (
                "lr",
                LogisticRegression(
                    solver="saga",
                    multi_class="multinomial",
                    max_iter=200,
                    n_jobs=1,  # avoid oversubscription; core logic unchanged
                    random_state=seed,
                    C=2.0,
                ),
            ),
        ]
    )


clf = build_clf(seed=0)

if threadpool_limits is not None:
    with threadpool_limits(limits=1):
        clf.fit(X, y)
else:
    clf.fit(X, y)

pred0 = clf.predict(X_test).astype(np.int32)

seeds = [
    0,
    1,
    2,
    3,
    4,
]  # kept for semantic parity with original code (though only seed=0 is used)

print("Built predictions vector:", pred0.shape, "seeds:", len(seeds))




## === cell 4
print((pred0.shape[0], len(seeds)))
print("pred0 head:", pred0[:5].tolist())




## === cell 5
def row_mode_int(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.int32)
    n, m = mat.shape
    K = 8
    idx = (np.arange(n, dtype=np.int64)[:, None] * K + mat.astype(np.int64)).ravel()
    counts = (
        np.bincount(idx, minlength=n * K).reshape(n, K).astype(np.int32, copy=False)
    )
    return counts.argmax(axis=1).astype(np.int32, copy=False)


ensemble_pred = pred0.astype(np.int32, copy=False)
print("Ensemble head:", ensemble_pred[:10])




## === cell 6
dif = np.zeros_like(ensemble_pred, dtype=np.int32)
print(pd.Series(dif[:1]).value_counts().sort_index().rename(lambda _: 0).to_string())




## === cell 7
if not (submission["Id"].to_numpy(copy=False) == test_ids).all():
    ens = pd.DataFrame({"Id": test_ids, "Cover_Type": ensemble_pred.astype(int)})
    submission2 = submission.merge(ens, on="Id", how="left")
    if submission2["Cover_Type"].isna().any():
        raise RuntimeError("Missing predictions after merge; check Id alignment.")
    submission2["Cover_Type"] = submission2["Cover_Type"].astype(int)
    submission = submission2
else:
    submission = submission.copy()
    submission["Cover_Type"] = ensemble_pred.astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)




## === cell 8
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != "Batch":
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(x=submission["Cover_Type"])
    plt.title("Predictions")
    plt.xlabel("Cover Type")
    for container in ax.containers:
        ax.bar_label(container)
    plt.show()
