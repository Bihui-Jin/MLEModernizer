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
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

RANDOM_STATE = 2021
np.random.seed(RANDOM_STATE)



## === cell 1
train_path = "../input/tabular-playground-series-dec-2021/train.csv"
test_path = "../input/tabular-playground-series-dec-2021/test.csv"

_sample = pd.read_csv(train_path, nrows=200)
dtype_map = {}
for c in _sample.columns:
    if c == "Id":
        dtype_map[c] = np.int32
    elif c == "Cover_Type":
        dtype_map[c] = np.int8
    else:
        if pd.api.types.is_integer_dtype(_sample[c].dtype):
            dtype_map[c] = np.int32
        else:
            dtype_map[c] = np.float32

train = pd.read_csv(train_path, dtype=dtype_map)
test = pd.read_csv(
    test_path, dtype={k: v for k, v in dtype_map.items() if k != "Cover_Type"}
)



## === cell 2
_ = train.shape
_ = test.shape



## === cell 3
pass



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
pass



## === cell 11
pass



## === cell 12
pass



## === cell 13
_ = train.isnull().sum().sum()



## === cell 14
train_X = train.drop("Cover_Type", axis=1)
train_y = train["Cover_Type"]



## === cell 15
_ = train_y.iloc[:5].to_list()



## === cell 16
_ = train_X.shape



## === cell 17
from sklearn.model_selection import train_test_split

_min_class_count = train_y.value_counts().min()
_stratify = train_y if _min_class_count >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    train_X, train_y, test_size=0.22, random_state=RANDOM_STATE, stratify=_stratify
)



## === cell 18
_ = y_test.shape



## === cell 19
nums_cols = [
    col
    for col in X_train.columns
    if X_train[col].dtype in ["float16", "float32", "float64"]
]
catgo_cols = [
    col
    for col in X_train.columns
    if X_train[col].dtype not in ["float16", "float32", "float64"]
]



## === cell 20
d_test = test
catgo_cols = []



## === cell 21
del train, test, train_X, train_y



## === cell 22
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(copy=True)
scaler.fit(X_train)

train_X = scaler.transform(X_train)
test_X = scaler.transform(X_test)
test = scaler.transform(d_test)



## === cell 23
_ = train_X.shape



## === cell 24
train_X = np.asarray(train_X, dtype=np.float32)
test_X = np.asarray(test_X, dtype=np.float32)
test = np.asarray(test, dtype=np.float32)

y_train = y_train.to_numpy().astype(np.float32)
y_test = y_test.to_numpy().astype(np.float32)



## === cell 25
train_X = np.ascontiguousarray(train_X)
test_X = np.ascontiguousarray(test_X)
test = np.ascontiguousarray(test)



## === cell 26
from xgboost import XGBClassifier

y_train = (y_train - 1).astype(np.int32)
y_test = (y_test - 1).astype(np.int32)

use_gpu = False
try:
    import xgboost as _xgb

    use_gpu = (
        getattr(_xgb, "cuda", None) is not None and _xgb.cuda.get_device_count() > 0
    )
except Exception:
    use_gpu = False

params = {
    "objective": "multi:softmax",
    "num_class": 7,
    "tree_method": "gpu_hist" if use_gpu else "hist",
    "eval_metric": "mlogloss",
    "booster": "gbtree",
    "gamma": 0.75,
    "max_depth": 7,
    "alpha": 10,
    "learning_rate": 0.007,
    "n_estimators": 2000,
    "predictor": "gpu_predictor" if use_gpu else "cpu_predictor",
    "random_state": RANDOM_STATE,
}

xgb = XGBClassifier(**params)
xgb.fit(
    train_X,
    y_train,
    early_stopping_rounds=200,
    eval_set=[(test_X, y_test)],
    verbose=True,
)



## === cell 27
from sklearn.metrics import accuracy_score

preds_valid = xgb.predict(test_X).astype(np.int32) + 1
acc = accuracy_score((y_test + 1).astype(np.int32), preds_valid)
print("accuracy score:", acc)



## === cell 28
_ = preds_valid[:10]



## === cell 29
sub = pd.read_csv("../input/tabular-playground-series-dec-2021/sample_submission.csv")

sub["Cover_Type"] = xgb.predict(test).astype(np.int32) + 1

sub.to_csv("submission.csv", index=False)
sub.head()
