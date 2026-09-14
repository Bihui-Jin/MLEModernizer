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

os.environ.setdefault("PYTHONHASHSEED", "2021")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

SEED = 2021
np.random.seed(SEED)



## === cell 1
train_path = "../input/tabular-playground-series-dec-2021/train.csv"
test_path = "../input/tabular-playground-series-dec-2021/test.csv"

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()

feature_cols = [c for c in train_cols if c != "Cover_Type"]

int16_cols = [
    c
    for c in feature_cols
    if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
]
dtype_train = {"Id": "int32", "Cover_Type": "int8"}
dtype_test = {"Id": "int32"}
for c in feature_cols:
    if c in int16_cols:
        dtype_train[c] = "int16"
        dtype_test[c] = "int16"
    elif c == "Id":
        continue
    else:
        dtype_train[c] = "float32"
        dtype_test[c] = "float32"

train = pd.read_csv(train_path, dtype=dtype_train)
test = pd.read_csv(test_path, dtype=dtype_test)



## === cell 2
pass



## === cell 3
train_X = train.drop(["Cover_Type", "Id"], axis=1)
train_y = train["Cover_Type"]

test_ids = test["Id"].to_numpy()
test_X_full = test.drop(["Id"], axis=1)



## === cell 4
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    train_X, train_y, test_size=0.22, random_state=SEED
)



## === cell 5
nums_cols = [
    col
    for col in X_train.columns
    if X_train[col].dtype
    in [
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ]
]
catgo_cols = [col for col in X_train.columns if col not in nums_cols]

d_test = test_X_full

if len(catgo_cols) > 0:
    enc = TargetEncoder(cols=catgo_cols)
    X_train = enc.fit_transform(X_train, y_train)
    X_test = enc.transform(X_test)
    d_test = enc.transform(d_test)



## === cell 6
del train, test, train_X, test_X_full



## === cell 7
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

train_X = scaler.transform(X_train)
test_X = scaler.transform(X_test)
test = scaler.transform(d_test)



## === cell 8
train_X = np.asarray(train_X, dtype=np.float32, order="C")
test_X = np.asarray(test_X, dtype=np.float32, order="C")
test = np.asarray(test, dtype=np.float32, order="C")

y_train = y_train.to_numpy().astype(np.int32) - 1
y_test = y_test.to_numpy().astype(np.int32) - 1



## === cell 9
from xgboost import XGBClassifier, DMatrix


def _build_xgb_and_fit(train_X, y_train, test_X, y_test, base_params):
    try:
        xgb = XGBClassifier(
            **{**base_params, "tree_method": "gpu_hist", "predictor": "gpu_predictor"}
        )
        xgb.fit(
            train_X,
            y_train,
            early_stopping_rounds=200,
            eval_set=[(test_X, y_test)],
            verbose=True,
        )
        return xgb
    except Exception:
        xgb = XGBClassifier(
            **{**base_params, "tree_method": "hist", "predictor": "cpu_predictor"}
        )
        xgb.fit(
            train_X,
            y_train,
            early_stopping_rounds=200,
            eval_set=[(test_X, y_test)],
            verbose=True,
        )
        return xgb


params = {
    "objective": "multi:softmax",
    "num_class": 7,
    "eval_metric": "merror",
    "booster": "gbtree",
    "gamma": 0.75,
    "max_depth": 7,
    "alpha": 10,
    "learning_rate": 0.007,
    "n_estimators": 2000,
    "random_state": SEED,
}

xgb = _build_xgb_and_fit(train_X, y_train, test_X, y_test, params)



## === cell 10
dvalid = DMatrix(test_X)
preds_valid = xgb.get_booster().predict(dvalid).astype("int32")
acc = accuracy_score(y_test, preds_valid)
print("accuracy score:", acc)



## === cell 11
sub = pd.read_csv("../input/tabular-playground-series-dec-2021/sample_submission.csv")

dtest = DMatrix(test)
pred_test = xgb.get_booster().predict(dtest).astype("int32") + 1
pred_df = pd.DataFrame({"Id": test_ids, "Cover_Type": pred_test})
sub = sub.drop(columns=["Cover_Type"]).merge(pred_df, on="Id", how="left")

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("submission.csv written with shape:", sub.shape)
