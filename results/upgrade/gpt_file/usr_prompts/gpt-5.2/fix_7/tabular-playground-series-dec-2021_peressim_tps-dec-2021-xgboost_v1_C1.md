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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import random
import warnings
import gc

try:
    from sklearnex import patch_sklearn  # scikit-learn-intelex

    patch_sklearn()
except Exception:
    pass

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")




## === cell 1
seed = 47
random.seed(seed)
np.random.seed(seed)




## === cell 2
def evaluate_model(model, x, y):
    y_pred = model.predict(x)
    acc = accuracy_score(y, y_pred)
    return {"accuracy": acc}




## === cell 3
def _build_dtypes_for_tps_dec2021():
    dtypes = {"Id": np.int32, "Cover_Type": np.int8}

    num_int16 = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
    for c in num_int16:
        dtypes[c] = np.int16

    for i in range(1, 5):
        dtypes[f"Wilderness_Area{i}"] = np.int8
    for i in range(1, 41):
        dtypes[f"Soil_Type{i}"] = np.int8

    return dtypes


TRAIN_PATH = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

DTYPES_TRAIN = _build_dtypes_for_tps_dec2021()
DTYPES_TEST = {k: v for k, v in DTYPES_TRAIN.items() if k != "Cover_Type"}

train_df = pd.read_csv(TRAIN_PATH, sep=",", engine="c", dtype=DTYPES_TRAIN)




## === cell 4
def fe(df):
    hroad = df["Horizontal_Distance_To_Roadways"].to_numpy(copy=False)
    elev = df["Elevation"].to_numpy(copy=False)
    vdh = df["Vertical_Distance_To_Hydrology"].to_numpy(copy=False)
    hdh = df["Horizontal_Distance_To_Hydrology"].to_numpy(copy=False)
    hfire = df["Horizontal_Distance_To_Fire_Points"].to_numpy(copy=False)

    df["EHiElv"] = hroad * elev
    df["EViElv"] = vdh * elev

    a2 = df["Aspect"].to_numpy(copy=False)
    df["Aspect2"] = np.where(a2 + 180 > 360, a2 - 180, a2 + 180)
    df["Aspect"] = np.mod(a2, 360)

    for col in ("Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"):
        arr = df[col].to_numpy(copy=False)
        np.clip(arr, 0, 255, out=arr)

    df["Highwater"] = (vdh < 0).astype(np.int8)
    df["EVDtH"] = elev - vdh
    df["EHDtH"] = elev - hdh * 0.2
    df["Euclidean_Distance_to_Hydrolody"] = np.sqrt(hdh * hdh + vdh * vdh)
    df["Manhattan_Distance_to_Hydrolody"] = hdh + vdh
    df["Hydro_Fire_1"] = hdh + hfire
    df["Hydro_Fire_2"] = np.abs(hdh - hfire)
    df["Hydro_Road_1"] = np.abs(hdh + hroad)
    df["Hydro_Road_2"] = np.abs(hdh - hroad)
    df["Fire_Road_1"] = np.abs(hfire + hroad)
    df["Fire_Road_2"] = np.abs(hfire - hroad)
    df["Hillshade_3pm_is_zero"] = (
        df["Hillshade_3pm"].to_numpy(copy=False) == 0
    ).astype(np.int8)
    return df




## === cell 5
DROP_COLS = ["Id", "Soil_Type7", "Soil_Type15", "Cover_Type"]

X_full = train_df.drop(DROP_COLS, axis=1)
y_raw = train_df["Cover_Type"].astype(np.int8).to_numpy(copy=False)

X_full["mean"] = X_full.to_numpy(copy=False).mean(axis=1)

X_full = fe(X_full)

soil_features = [c for c in X_full.columns if c.startswith("Soil_Type")]
wilderness_features = [c for c in X_full.columns if c.startswith("Wilderness_Area")]

X_full["soil_type_count"] = X_full[soil_features].to_numpy(copy=False).sum(axis=1)
X_full["wilderness_area_count"] = (
    X_full[wilderness_features].to_numpy(copy=False).sum(axis=1)
)

idx = np.arange(len(X_full), dtype=np.int32)
idx_train, idx_valid, y_train_raw, y_valid_raw = train_test_split(
    idx, y_raw, test_size=0.2, random_state=seed, shuffle=True
)

X_full_np = np.ascontiguousarray(X_full.to_numpy(copy=False))
X_train_np = X_full_np[idx_train]
X_valid_np = X_full_np[idx_valid]

y_train_np = (y_train_raw - 1).astype(np.int8, copy=False)
y_valid_np = (y_valid_raw - 1).astype(np.int8, copy=False)




## === cell 6
params = {
    "n_estimators": 300,
    "max_depth": 18,
    "subsample": 1.0,
    "eta": 0.3,
    "colsample_bytree": 1.0,
    "gamma": 0.0,
    "min_child_weight": 1,
    "reg_alpha": 1,
}

tree_method = "hist"
device = "cpu"
try:
    _cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if _cuda_visible not in ("", "-1"):
        device = "cuda"
        tree_method = "hist"
except Exception:
    tree_method = "hist"
    device = "cpu"

n_jobs = os.cpu_count() or 1

model = XGBClassifier(
    **params,
    objective="multi:softmax",
    num_class=7,
    random_state=seed,
    tree_method=tree_method,
    device=device,
    n_jobs=n_jobs,
    verbosity=0,
)

model.fit(
    X_train_np,
    y_train_np,
    eval_set=[(X_valid_np, y_valid_np)],
    verbose=False,
)

score = evaluate_model(model, X_valid_np, y_valid_np)
print(score)




## === cell 7
del (
    train_df,
    X_full,
    X_full_np,
    idx,
    idx_train,
    idx_valid,
    y_raw,
    y_train_raw,
    y_valid_raw,
)
gc.collect()




## === cell 8
test_df = pd.read_csv(TEST_PATH, sep=",", engine="c", dtype=DTYPES_TEST)

X_test = test_df.drop(["Id", "Soil_Type7", "Soil_Type15"], axis=1)
X_test["mean"] = X_test.to_numpy(copy=False).mean(axis=1)

X_test = fe(X_test)

X_test["soil_type_count"] = X_test[soil_features].to_numpy(copy=False).sum(axis=1)
X_test["wilderness_area_count"] = (
    X_test[wilderness_features].to_numpy(copy=False).sum(axis=1)
)

X_test_np = np.ascontiguousarray(X_test.to_numpy(copy=False))

target0 = model.predict(X_test_np).astype(int).squeeze()
target = target0 + 1

ids = test_df["Id"].to_numpy(copy=False)
submission_xgboost = pd.DataFrame({"Id": ids, "Cover_Type": target.astype(np.int16)})

submission_xgboost.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_xgboost.shape)
print(submission_xgboost.head())
print(submission_xgboost.dtypes)
