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
import numpy as np
import pandas as pd



## === cell 1
train_df = pd.read_csv("/kaggle/input/tabular-playground-series-dec-2021/train.csv")
test_df = pd.read_csv("/kaggle/input/tabular-playground-series-dec-2021/test.csv")

assert "Cover_Type" in train_df.columns, "train.csv must contain Cover_Type"
assert (
    "Id" in train_df.columns and "Id" in test_df.columns
), "train/test must contain Id"



## === cell 2
feature_cols = [c for c in train_df.columns if c not in ["Cover_Type", "Id"]]

X = train_df[feature_cols]
y_raw = pd.to_numeric(train_df["Cover_Type"], errors="coerce")
if y_raw.isna().any():
    raise ValueError("Found NaNs in Cover_Type after numeric coercion.")

y_int = y_raw.astype(np.int32).to_numpy()

classes_ = np.sort(np.unique(y_int)).astype(np.int32)
num_class = int(classes_.shape[0])
class_to_idx = {int(c): int(i) for i, c in enumerate(classes_)}
y_all_idx = np.vectorize(class_to_idx.get, otypes=[np.int32])(y_int).astype(np.int32)

assert num_class >= 2, "Need at least 2 classes."
assert np.array_equal(
    np.unique(y_all_idx), np.arange(num_class, dtype=np.int32)
), "Encoded labels must be contiguous 0..K-1"

print(
    "X shape:",
    X.shape,
    "y shape:",
    y_raw.shape,
    "num_class:",
    num_class,
    "classes:",
    classes_,
)



## === cell 3
from sklearn.model_selection import train_test_split

class_counts = pd.Series(y_all_idx).value_counts()
can_stratify = int(class_counts.min()) >= 2

if can_stratify:
    x_train, x_val, y_train_idx, y_val_idx = train_test_split(
        X, y_all_idx, test_size=0.25, random_state=42, stratify=y_all_idx
    )
else:
    x_train, x_val, y_train_idx, y_val_idx = train_test_split(
        X, y_all_idx, test_size=0.25, random_state=42, shuffle=True
    )

y_train_idx = np.asarray(y_train_idx, dtype=np.int32)
y_val_idx = np.asarray(y_val_idx, dtype=np.int32)

assert np.all(
    (0 <= y_train_idx) & (y_train_idx < num_class)
), "Train labels out of range after encoding."
assert np.all(
    (0 <= y_val_idx) & (y_val_idx < num_class)
), "Val labels out of range after encoding."

print("Train fold shape:", x_train.shape, "Val fold shape:", x_val.shape)
print("Unique classes in train fold:", np.unique(y_train_idx))
print("Unique classes in val fold:", np.unique(y_val_idx))



## === cell 4
from xgboost import XGBClassifier

params = dict(
    n_estimators=20000,
    n_jobs=4,
)

early_stopping_rounds = 50

X_fit = X
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
        verbose=True,
        early_stopping_rounds=early_stopping_rounds,
    )
    return model


try:
    model_xgbc = fit_xgb(tree_method="gpu_hist")
except Exception as e:
    print("GPU training failed; falling back to CPU hist. Original error:", repr(e))
    model_xgbc = fit_xgb(tree_method="hist")



## === cell 5
test_X = test_df[feature_cols]

proba = model_xgbc.predict_proba(test_X)
if proba.ndim != 2 or proba.shape[1] != num_class:
    raise ValueError(
        f"Unexpected predict_proba shape {proba.shape}, expected (n_samples, {num_class})."
    )

y_pred_idx = np.asarray(np.argmax(proba, axis=1), dtype=np.int32)
assert (
    y_pred_idx.min() >= 0 and y_pred_idx.max() < num_class
), "Predicted class indices out of range."

y_predict_xgbc = classes_[y_pred_idx].astype(np.int32)
pred_unique = np.unique(y_predict_xgbc)
assert set(pred_unique).issubset(set(classes_)), "Predictions include unknown classes."

print("Prediction unique classes:", pred_unique[:20], " ... total:", len(pred_unique))



## === cell 6
result = pd.DataFrame({"Id": test_df["Id"].values, "Cover_Type": y_predict_xgbc})
print(result.head())
print("Result shape:", result.shape)



## === cell 7
out_path = "/kaggle/working/submission.csv"
result.to_csv(out_path, index=False)

print("Done. Wrote", out_path, "with columns:", list(result.columns))
print("Rows:", len(result))
print("Cover_Type value counts (sample):")
print(result["Cover_Type"].value_counts().head(10))
