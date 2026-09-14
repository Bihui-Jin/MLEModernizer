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
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass



## === cell 1
train_path = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
test_path = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()

assert "Cover_Type" in train_cols, "train.csv must contain Cover_Type"
assert "Id" in train_cols and "Id" in test_cols, "train/test must contain Id"

feature_cols = [c for c in train_cols if c not in ["Cover_Type", "Id"]]

dtype_train = {"Id": np.int32, "Cover_Type": np.int32}
dtype_test = {"Id": np.int32}
for c in feature_cols:
    if c.startswith("Wilderness_Area") or c.startswith("Soil_Type"):
        dt = np.int8
    else:
        dt = np.float32
    dtype_train[c] = dt
    dtype_test[c] = dt

usecols_train = ["Id"] + feature_cols + ["Cover_Type"]
usecols_test = ["Id"] + feature_cols

train_df = pd.read_csv(
    train_path, usecols=usecols_train, dtype=dtype_train, low_memory=False
)
test_df = pd.read_csv(
    test_path, usecols=usecols_test, dtype=dtype_test, low_memory=False
)

print("Loaded train:", train_df.shape, "test:", test_df.shape)



## === cell 2
X = train_df[feature_cols]

y_int = train_df["Cover_Type"].to_numpy(dtype=np.int32, copy=False)
classes_ = np.unique(y_int)
classes_.sort()
num_class = int(classes_.shape[0])

y_all_idx = np.searchsorted(classes_, y_int).astype(np.int32, copy=False)

assert num_class >= 2, "Need at least 2 classes."
assert np.array_equal(
    np.unique(y_all_idx), np.arange(num_class, dtype=np.int32)
), "Encoded labels must be contiguous 0..K-1"

print(
    "X shape:",
    X.shape,
    "y shape:",
    y_int.shape,
    "num_class:",
    num_class,
    "classes:",
    classes_,
)



## === cell 3
from sklearn.model_selection import train_test_split

class_counts = np.bincount(y_all_idx, minlength=num_class)
can_stratify = int(class_counts.min()) >= 2

idx = np.arange(X.shape[0], dtype=np.int32)
if can_stratify:
    idx_train, idx_val, y_train_idx, y_val_idx = train_test_split(
        idx, y_all_idx, test_size=0.25, random_state=42, stratify=y_all_idx
    )
else:
    idx_train, idx_val, y_train_idx, y_val_idx = train_test_split(
        idx, y_all_idx, test_size=0.25, random_state=42, shuffle=True
    )

x_val = X.iloc[idx_val].to_numpy(copy=False)
y_val_idx = np.asarray(y_val_idx, dtype=np.int32)

print("Train fold size:", len(idx_train), "Val fold size:", len(idx_val))
print("Unique classes in val fold:", np.unique(y_val_idx))



## === cell 4
from xgboost import XGBClassifier

params = dict(
    n_estimators=20000,
    n_jobs=4,
)

early_stopping_rounds = 50

X_fit = X.to_numpy(copy=False)
y_fit = np.asarray(y_all_idx, dtype=np.int32)


def fit_xgb(tree_method: str):
    model = XGBClassifier(
        **params,
        tree_method=tree_method,
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="merror",
        random_state=42,
    )

    model.fit(
        X_fit,
        y_fit,
        eval_set=[(x_val, y_val_idx)],
        verbose=100,
        early_stopping_rounds=early_stopping_rounds,
    )
    return model


try:
    model_xgbc = fit_xgb(tree_method="gpu_hist")
except Exception as e:
    print("GPU training failed; falling back to CPU hist. Original error:", repr(e))
    model_xgbc = fit_xgb(tree_method="hist")

print("Best iteration:", getattr(model_xgbc, "best_iteration", None))



## === cell 5
test_X = test_df[feature_cols].to_numpy(copy=False)

proba = model_xgbc.predict_proba(test_X)

if proba.ndim != 2 or proba.shape[1] != num_class:
    raise ValueError(
        f"Unexpected predict_proba shape {proba.shape}, expected (n_samples, {num_class})."
    )

y_pred_idx = np.asarray(np.argmax(proba, axis=1), dtype=np.int32)
assert (
    y_pred_idx.min() >= 0 and y_pred_idx.max() < num_class
), "Predicted class indices out of range."

y_predict_xgbc = classes_[y_pred_idx].astype(np.int32, copy=False)
pred_unique = np.unique(y_predict_xgbc)
assert set(pred_unique).issubset(set(classes_)), "Predictions include unknown classes."

print("Prediction unique classes:", pred_unique[:20], " ... total:", len(pred_unique))



## === cell 6
result = pd.DataFrame(
    {"Id": test_df["Id"].to_numpy(copy=False), "Cover_Type": y_predict_xgbc}
)

result["Id"] = result["Id"].astype(np.int32, copy=False)
result["Cover_Type"] = result["Cover_Type"].astype(np.int32, copy=False)

print(result.head())
print("Result shape:", result.shape)

out_path = "/kaggle/working/submission.csv"
result.to_csv(out_path, index=False)

print("Done. Wrote", out_path, "with columns:", list(result.columns))
print("Rows:", len(result))
print("Cover_Type value counts (sample):")
print(result["Cover_Type"].value_counts().head(10))
