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
plotly==5.24.1
plotly-express==0.4.1
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
import os
import random
import pandas as pd
import xgboost
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

random.seed(64)
np.random.seed(64)

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass




## === cell 1
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {paths}")


train_path = _first_existing(
    [
        "/kaggle/input/tabular-playground-series-dec-2021/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/tabular-playground-series-dec-2021/train.csv",
        "/kaggle/data/train.csv",
    ]
)

test_path = _first_existing(
    [
        "/kaggle/input/tabular-playground-series-dec-2021/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/tabular-playground-series-dec-2021/test.csv",
        "/kaggle/data/test.csv",
    ]
)


def _read_train_test(train_path, test_path):
    train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
    test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()

    if "Id" not in train_cols or "Cover_Type" not in train_cols:
        raise ValueError("Train must contain Id and Cover_Type.")
    if "Id" not in test_cols:
        raise ValueError("Test must contain Id.")

    feature_cols_local = [c for c in train_cols if c not in ("Id", "Cover_Type")]
    usecols_train = ["Id", "Cover_Type"] + feature_cols_local
    usecols_test = ["Id"] + feature_cols_local

    dtype_train = {c: "float32" for c in feature_cols_local}
    dtype_train["Id"] = "int64"
    dtype_train["Cover_Type"] = "int64"

    dtype_test = {c: "float32" for c in feature_cols_local}
    dtype_test["Id"] = "int64"

    read_kwargs_train = dict(
        usecols=usecols_train,
        dtype=dtype_train,
        low_memory=False,
    )
    read_kwargs_test = dict(
        usecols=usecols_test,
        dtype=dtype_test,
        low_memory=False,
    )

    try:
        df_local = pd.read_csv(train_path, engine="pyarrow", **read_kwargs_train)
        df_test_local = pd.read_csv(test_path, engine="pyarrow", **read_kwargs_test)
    except Exception:
        df_local = pd.read_csv(train_path, **read_kwargs_train)
        df_test_local = pd.read_csv(test_path, **read_kwargs_test)

    return df_local, df_test_local


df, df_test = _read_train_test(train_path, test_path)



## === cell 2
required_cols = {"Id", "Cover_Type"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Train is missing required columns: {missing}")

if "Id" not in df_test.columns:
    raise ValueError("Test is missing required column: Id")



## === cell 3
print("Train shape:", df.shape, "Test shape:", df_test.shape)
print("Train dtypes (first 10):", df.dtypes.head(10).to_dict())



## === cell 4
feature_cols = [c for c in df.columns if c not in ["Id", "Cover_Type"]]
if any(c not in df_test.columns for c in feature_cols):
    missing_in_test = [c for c in feature_cols if c not in df_test.columns]
    raise ValueError(
        f"Test is missing feature columns present in train: {missing_in_test}"
    )



## === cell 5
vc = df["Cover_Type"].value_counts(dropna=False)
rare_classes = vc[vc < 2].index.tolist()
if rare_classes:
    print(
        f"Dropping {len(rare_classes)} class(es) with <2 samples to allow stratified split: {rare_classes}"
    )
    df = df[~df["Cover_Type"].isin(rare_classes)].reset_index(drop=True)

label_encoder = LabelEncoder()
y_np = label_encoder.fit_transform(df["Cover_Type"].to_numpy())

idx = np.arange(len(df), dtype=np.int32)
try:
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx, y_np, test_size=0.2, shuffle=True, random_state=64, stratify=y_np
    )
except ValueError as e:
    print(
        "Stratified split failed after rare-class filtering; falling back to non-stratified split. Error:",
        str(e),
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx, y_np, test_size=0.2, shuffle=True, random_state=64, stratify=None
    )

work_dir = "/kaggle/working"
os.makedirs(work_dir, exist_ok=True)

X_all = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
X_train = X_all[idx_train]
X_val = X_all[idx_val]

df_test_ids = df_test["Id"].to_numpy(dtype=np.int64, copy=False)
X_test = df_test[feature_cols].to_numpy(dtype=np.float32, copy=False)

df = df[["Id", "Cover_Type"]]
df_test = df_test[["Id"]]
del y_np, idx, X_all

print(
    "Prepared features as NumPy arrays. Train rows:",
    X_train.shape[0],
    "Val rows:",
    X_val.shape[0],
    "Test rows:",
    X_test.shape[0],
    "Num features:",
    X_train.shape[1],
    "Num classes:",
    len(label_encoder.classes_),
)



## === cell 6
nthread = int(os.environ.get("OMP_NUM_THREADS", "0")) or (os.cpu_count() or 4)

sane_defaults = {
    "objective": "multi:softmax",
    "num_class": len(label_encoder.classes_),
    "tree_method": "gpu_hist",
    "sampling_method": "gradient_based",
    "subsample": 0.25,
    "max_depth": 4,
    "learning_rate": 0.10,
    "colsample_bytree": 0.5,
    "eval_metric": ["mlogloss", "merror"],
    "predictor": "gpu_predictor",
    "seed": 64,
    "nthread": nthread,
}

use_gpu = True
try:
    dtrain = xgboost.QuantileDMatrix(X_train, label=y_train, nthread=nthread)
    dval = xgboost.QuantileDMatrix(X_val, label=y_val, nthread=nthread)
except Exception:
    use_gpu = False
    dtrain = xgboost.DMatrix(X_train, label=y_train, nthread=nthread)
    dval = xgboost.DMatrix(X_val, label=y_val, nthread=nthread)

try:
    booster = xgboost.train(
        params=sane_defaults,
        dtrain=dtrain,
        num_boost_round=3000,
        early_stopping_rounds=50,
        evals=[(dval, "val")],
        verbose_eval=100,
    )
except xgboost.core.XGBoostError:
    cpu_params = dict(sane_defaults)
    cpu_params["tree_method"] = "hist"
    cpu_params.pop("sampling_method", None)  # only applicable for some GPU configs
    cpu_params["predictor"] = "auto"

    dtrain_cpu = xgboost.DMatrix(X_train, label=y_train, nthread=nthread)
    dval_cpu = xgboost.DMatrix(X_val, label=y_val, nthread=nthread)

    booster = xgboost.train(
        params=cpu_params,
        dtrain=dtrain_cpu,
        num_boost_round=3000,
        early_stopping_rounds=50,
        evals=[(dval_cpu, "val")],
        verbose_eval=100,
    )



## === cell 7
fscore = booster.get_fscore()
top10 = sorted(fscore.items(), key=lambda kv: kv[1], reverse=True)[:10]
print("Top-10 features by fscore:", top10)



## === cell 8
try:
    dtest = xgboost.QuantileDMatrix(X_test, nthread=nthread)
except Exception:
    dtest = xgboost.DMatrix(X_test, nthread=nthread)

pred = booster.predict(dtest).astype(np.int32)
cover_type = label_encoder.inverse_transform(pred)

sub = pd.DataFrame(
    {
        "Id": df_test_ids,
        "Cover_Type": cover_type.astype(np.int64, copy=False),
    }
)

sub = sub[["Id", "Cover_Type"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Unique predicted classes:", np.unique(sub["Cover_Type"]).tolist())
