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

category_encoders==2.7.0
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
xgboost==2.0.3

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
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from category_encoders.target_encoder import TargetEncoder

warnings.filterwarnings("ignore")

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
np.random.seed(2021)

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass



## === cell 1
DATA_DIR = "../input/tabular-playground-series-dec-2021"
train_path = f"{DATA_DIR}/train.csv"
test_path = f"{DATA_DIR}/test.csv"

_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
dtype_map = {c: np.int32 for c in _cols if c not in ("Id", "Cover_Type")}
dtype_map["Id"] = np.int32
dtype_map["Cover_Type"] = np.int32

train = pd.read_csv(train_path, dtype=dtype_map)
test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()
test_dtype_map = {c: np.int32 for c in test_cols if c != "Id"}
test_dtype_map["Id"] = np.int32
test = pd.read_csv(test_path, dtype=test_dtype_map)



## === cell 2
_ = train.shape, test.shape



## === cell 3
train_X = train.drop(["Id", "Cover_Type"], axis=1)
train_y = train["Cover_Type"].astype(np.int32) - 1



## === cell 4
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    train_X, train_y, test_size=0.22, random_state=2021
)



## === cell 5
del train, train_X, train_y



## === cell 6
dtypes = X_train.dtypes
nums_cols = dtypes.index[
    dtypes.isin([np.dtype("float16"), np.dtype("float32"), np.dtype("float64")])
].tolist()
catgo_cols = [c for c in X_train.columns if c not in nums_cols]



## === cell 7
test_ids = test["Id"].copy()
test = test.drop("Id", axis=1)



## === cell 8
d_test = test

if len(catgo_cols) > 0:
    enc = TargetEncoder(cols=catgo_cols)
    X_train = enc.fit_transform(X_train, y_train)
    X_test = enc.transform(X_test)
    d_test = enc.transform(d_test)



## === cell 9
del test



## === cell 10
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

train_X = scaler.transform(X_train).astype(np.float32, copy=False)
test_X = scaler.transform(X_test).astype(np.float32, copy=False)
test = scaler.transform(d_test).astype(np.float32, copy=False)



## === cell 11
del X_train, X_test, d_test



## === cell 12
y_train = y_train.to_numpy().astype(np.int32, copy=False)
y_test = y_test.to_numpy().astype(np.int32, copy=False)



## === cell 13
from xgboost import XGBClassifier

params = {
    "objective": "multi:softmax",
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "booster": "gbtree",
    "gamma": 0.75,
    "max_depth": 7,
    "alpha": 10,
    "learning_rate": 0.007,
    "n_estimators": 2000,
}

xgb = XGBClassifier(**params, n_jobs=4, random_state=2021)

xgb.fit(
    train_X,
    y_train,
    early_stopping_rounds=200,
    eval_set=[(test_X, y_test)],
    verbose=True,
)



## === cell 14
preds_valid = (xgb.predict(test_X).astype(np.int32) + 1).astype("int")
acc = accuracy_score((y_test + 1).astype(np.int32), preds_valid)
print("accuracy score:", acc)



## === cell 15
sub = pd.read_csv("../input/tabular-playground-series-dec-2021/sample_submission.csv")

if "Id" in sub.columns and len(test_ids) == len(sub):
    if not np.array_equal(sub["Id"].to_numpy(), test_ids.to_numpy()):
        sub = sub.drop(columns=["Cover_Type"], errors="ignore")
        sub["Id"] = test_ids.to_numpy()

sub["Cover_Type"] = (xgb.predict(test).astype(np.int32) + 1).astype("int")
sub.to_csv("submission.csv", index=False)
sub.head()
