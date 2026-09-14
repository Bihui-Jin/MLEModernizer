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

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/tabular-playground-series-dec-2021"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

CSV_ENGINE = "pyarrow"

train_head = pd.read_csv(TRAIN_PATH, nrows=5)
test_head = pd.read_csv(TEST_PATH, nrows=5)

assert "Cover_Type" in train_head.columns
assert "Id" in train_head.columns and "Id" in test_head.columns

feature_cols = [c for c in train_head.columns if c not in ("Id", "Cover_Type")]
usecols_train = ["Cover_Type"] + feature_cols
usecols_test = feature_cols

dtype_map = {}
for c in feature_cols:
    if c.startswith("Wilderness_Area") or c.startswith("Soil_Type"):
        dtype_map[c] = np.int8
    else:
        dtype_map[c] = np.int32

dtype_map_train = dict(dtype_map)
dtype_map_train["Cover_Type"] = np.int8


def _read_csv_robust(path, *, usecols=None, dtype=None, engine_preference="pyarrow"):
    if engine_preference is None:
        return pd.read_csv(
            path, usecols=usecols, dtype=dtype, low_memory=False, memory_map=True
        )
    try:
        return pd.read_csv(path, usecols=usecols, dtype=dtype, engine=engine_preference)
    except Exception as e:
        print(
            f"Falling back to default pandas engine for {os.path.basename(path)} due to: {type(e).__name__}: {e}"
        )
        return pd.read_csv(
            path, usecols=usecols, dtype=dtype, low_memory=False, memory_map=True
        )


train = _read_csv_robust(
    TRAIN_PATH,
    usecols=usecols_train,
    dtype=dtype_map_train,
    engine_preference=CSV_ENGINE,
)
test = _read_csv_robust(
    TEST_PATH, usecols=usecols_test, dtype=dtype_map, engine_preference=CSV_ENGINE
)
submission = _read_csv_robust(SAMPLE_SUB_PATH, engine_preference=CSV_ENGINE)

y = train["Cover_Type"].astype(np.int32, copy=False)

y_values = y.to_numpy(copy=False)
unique_classes = np.unique(y_values)
counts = np.bincount(y_values.astype(np.int64, copy=False))
counts = counts[counts > 0]

print(
    "Train shape:",
    (train.shape[0], len(feature_cols)),
    "Test shape:",
    (test.shape[0], len(feature_cols)),
    "Num classes:",
    int(unique_classes.size),
)
print(
    "Class counts (min/max):",
    int(counts.min()),
    int(counts.max()),
)



## === cell 1
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

models = []

models.append(
    (
        "lr",
        Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=200,
                        n_jobs=-1,
                        multi_class="auto",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    )
)

models.append(
    (
        "rf",
        RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            warm_start=False,
        ),
    )
)

models.append(
    (
        "et",
        ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            warm_start=False,
        ),
    )
)

models.append(
    ("gnb", Pipeline(steps=[("scaler", StandardScaler()), ("clf", GaussianNB())]))
)



## === cell 2
print(
    "Skipping validation fit to meet the 600s runtime constraint; training will run on full data for submission."
)



## === cell 3
X_train_int = train[feature_cols].to_numpy(copy=False)
y_np = y.to_numpy(copy=False)
X_test_int = test[feature_cols].to_numpy(copy=False)

X_train_f32 = np.asarray(X_train_int, dtype=np.float32, order="C")
X_test_f32 = np.asarray(X_test_int, dtype=np.float32, order="C")

del train, test, y, y_values, train_head, test_head

trained_models = []
pred_matrix = np.empty((X_test_int.shape[0], len(models)), dtype=np.int16)


def _fit_predict(name, clf):
    """Fit classifier and return predictions on test using the right dtype view."""
    if name in ("lr", "gnb"):
        clf.fit(X_train_f32, y_np)
        return clf.predict(X_test_f32)

    clf.fit(X_train_int, y_np)
    return clf.predict(X_test_int)


for j, (name, clf) in enumerate(models):
    y_pred = _fit_predict(name, clf)
    trained_models.append((name, clf))
    pred_matrix[:, j] = y_pred.astype(np.int16, copy=False)

print("Pred matrix shape:", pred_matrix.shape)



## === cell 4
n_rows, n_models = pred_matrix.shape

min_label = int(pred_matrix.min())
max_label = int(pred_matrix.max())
offset = -min_label if min_label < 0 else 0
K = max_label + offset + 1

pred_shifted = pred_matrix.astype(np.int32, copy=False)
if offset:
    pred_shifted = pred_shifted + offset

row_ids = np.arange(n_rows, dtype=np.int64)
flat_idx = (row_ids[:, None] * K + pred_shifted).ravel()
flat_counts = np.bincount(flat_idx, minlength=n_rows * K)
counts = flat_counts.reshape(n_rows, K)

ensemble = counts.argmax(axis=1).astype(np.int32)
if offset:
    ensemble = ensemble - offset

submission["Cover_Type"] = ensemble.astype(np.int32, copy=False)

assert submission.shape[0] == n_rows
assert list(submission.columns) == ["Id", "Cover_Type"]

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", submission.shape)
print(submission.head())

del (
    X_train_int,
    X_test_int,
    X_train_f32,
    X_test_f32,
    pred_matrix,
    flat_idx,
    flat_counts,
    counts,
    row_ids,
)
