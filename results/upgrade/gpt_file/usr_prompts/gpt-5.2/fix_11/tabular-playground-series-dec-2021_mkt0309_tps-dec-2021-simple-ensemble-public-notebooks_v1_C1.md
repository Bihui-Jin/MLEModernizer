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

from scipy import stats  # kept (original import)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_INPUT_CANDIDATES = [
    "/kaggle/input/tabular-playground-series-dec-2021",
    "/kaggle/data/tabular-playground-series-dec-2021",
    "../input/tabular-playground-series-dec-2021",
    "/kaggle/input",
    "/kaggle/data",
    "../input",
]


def find_file(filename: str) -> str:
    for base in BASE_INPUT_CANDIDATES:
        candidate = os.path.join(base, filename)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename} in known input locations: {BASE_INPUT_CANDIDATES}"
    )


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sub_path = find_file("sample_submission.csv")

_target_dtype = "int16"
_id_dtype = "int32"

_train_head = pd.read_csv(train_path, nrows=5)
_test_head = pd.read_csv(test_path, nrows=5)

dtype_train = {c: "int16" for c in _train_head.columns if c not in ("Id", "Cover_Type")}
dtype_train["Id"] = _id_dtype
dtype_train["Cover_Type"] = _target_dtype

dtype_test = {c: "int16" for c in _test_head.columns if c != "Id"}
dtype_test["Id"] = _id_dtype

train_df = pd.read_csv(train_path, low_memory=False, dtype=dtype_train)
test_df = pd.read_csv(test_path, low_memory=False, dtype=dtype_test)
submission = pd.read_csv(sub_path, low_memory=False, dtype={"Id": _id_dtype})

assert "Cover_Type" in train_df.columns
assert "Id" in train_df.columns and "Id" in test_df.columns
assert submission.shape[0] == test_df.shape[0]

train_df["Cover_Type"] = train_df["Cover_Type"].astype(_target_dtype, copy=False)
train_df["Id"] = train_df["Id"].astype(_id_dtype, copy=False)
test_df["Id"] = test_df["Id"].astype(_id_dtype, copy=False)

train_df.shape, test_df.shape, submission.shape



## === cell 1
from scipy import sparse

X = train_df.drop(columns=["Cover_Type"])
y_train = train_df["Cover_Type"].astype(int, copy=False)

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

numeric_transformer = "passthrough"

categorical_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop",
    sparse_threshold=1.0,
    verbose_feature_names_out=False,
)


def make_lr(seed: int):
    return LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        max_iter=200,
        n_jobs=1,
        random_state=seed,
        C=2.0,
    )


Xt_train = preprocess.fit_transform(X)
Xt_test = preprocess.transform(test_df)

del X  # free RAM early (train_df needed later only for deletion)

if sparse.issparse(Xt_train) and not sparse.isspmatrix_csr(Xt_train):
    Xt_train = Xt_train.tocsr()
if sparse.issparse(Xt_test) and not sparse.isspmatrix_csr(Xt_test):
    Xt_test = Xt_test.tocsr()

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

clf = make_lr(seeds[0])
clf.fit(Xt_train, y_train)

pred_single = clf.predict(Xt_test).astype(np.int16, copy=False)

del train_df, Xt_train  # free RAM early

pred_single.shape, pred_single[:10]



## === cell 2
results = pd.DataFrame({"ensemble": pred_single.astype(int, copy=False)})
print(results.shape)
results.head()



## === cell 3
results["ensemble"] = results["ensemble"].astype(int, copy=False)
results.head()




## === cell 4
def nunique(a, axis):
    a_sorted = np.sort(a, axis=axis)
    return (np.diff(a_sorted, axis=axis) != 0).sum(axis=axis) + 1




## === cell 5
results["dif"] = 0
results.head()



## === cell 6
results["dif"].value_counts()



## === cell 7
pred_df = pd.DataFrame(
    {
        "Id": test_df["Id"].values,
        "Cover_Type": results["ensemble"].values.astype(int, copy=False),
    }
)

submission = submission[["Id"]].copy()
submission["Cover_Type"] = pred_df["Cover_Type"].values

if submission["Cover_Type"].isna().any():
    raise ValueError(
        "Missing predictions after assigning predictions; check Id alignment."
    )

submission["Cover_Type"] = submission["Cover_Type"].astype(int, copy=False)
submission.to_csv("submission.csv", index=False)
submission.head()
