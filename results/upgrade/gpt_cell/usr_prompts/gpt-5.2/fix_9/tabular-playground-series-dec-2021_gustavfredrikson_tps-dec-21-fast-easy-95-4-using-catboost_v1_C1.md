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

catboost==1.2.8
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
import numpy as np
import pandas as pd

try:
    import datatable as dt
except ModuleNotFoundError:
    dt = None

import sklearn.model_selection as skl_ms
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 1))




## === cell 1
train_path = "../input/tabular-playground-series-dec-2021/train.csv"
test_path = "../input/tabular-playground-series-dec-2021/test.csv"


def _make_dtypes(is_train: bool):
    base_cols = [
        "Id",
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
    wilderness = [f"Wilderness_Area{i}" for i in range(1, 5)]
    soil = [f"Soil_Type{i}" for i in range(1, 41)]
    cols = base_cols + wilderness + soil
    if is_train:
        cols = cols + ["Cover_Type"]
    dtypes = {c: np.int32 for c in cols}
    return dtypes


if dt is not None:
    train_dt = dt.fread(train_path, columns=None, fill=True)
    test_dt = dt.fread(test_path, columns=None, fill=True)

    if "Id" in train_dt.names:
        train_dt = train_dt[:, [c for c in train_dt.names if c != "Id"]]
    if "Id" in test_dt.names:
        test_dt = test_dt[:, [c for c in test_dt.names if c != "Id"]]

    train_dt = train_dt[:, [dt.as_type(dt.f[c], dt.int32) for c in train_dt.names]]
    test_dt = test_dt[:, [dt.as_type(dt.f[c], dt.int32) for c in test_dt.names]]

    ct = dt.f["Cover_Type"]
    print(f"Nr of cover_type = 5: {(train_dt[:, dt.sum(ct == 5)].to_list()[0][0])}")
    print(f"Nr of cover_type = 4: {(train_dt[:, dt.sum(ct == 4)].to_list()[0][0])}")
    train_dt = train_dt[(ct != 4) & (ct != 5), :]

    y_series = train_dt[:, "Cover_Type"].to_pandas()["Cover_Type"]
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y_series).astype(np.int32, copy=False)

    X_dt = train_dt[:, [c for c in train_dt.names if c != "Cover_Type"]]
    X_test_dt = test_dt
else:
    train_df = pd.read_csv(train_path, dtype=_make_dtypes(is_train=True))
    test_df = pd.read_csv(test_path, dtype=_make_dtypes(is_train=False))
    if "Id" in train_df.columns:
        train_df.drop(columns=["Id"], inplace=True)
    if "Id" in test_df.columns:
        test_df.drop(columns=["Id"], inplace=True)

    mask_keep = ~train_df["Cover_Type"].isin((4, 5))
    print(f"Nr of cover_type = 5: {(train_df['Cover_Type'] == 5).sum()}")
    print(f"Nr of cover_type = 4: {(train_df['Cover_Type'] == 4).sum()}")
    train_df = train_df.loc[mask_keep].reset_index(drop=True)

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(train_df["Cover_Type"]).astype(np.int32, copy=False)
    X_df = train_df.drop(columns=["Cover_Type"])
    X_test_df = test_df




## === cell 2
from catboost import Pool

test_size = 0.01  # unchanged

if dt is not None:
    n = X_dt.nrows
    idx = np.arange(n, dtype=np.int64)
    idx_train, idx_valid = skl_ms.train_test_split(
        idx, test_size=test_size, random_state=SEED, shuffle=True
    )

    X_train_dt = X_dt[idx_train.tolist(), :]
    X_valid_dt = X_dt[idx_valid.tolist(), :]
    y_train = y_enc[idx_train]
    y_valid = y_enc[idx_valid]

    train_pool = Pool(X_train_dt, y_train)
    valid_pool = Pool(X_valid_dt, y_valid)
else:
    n = len(X_df)
    idx = np.arange(n, dtype=np.int64)
    idx_train, idx_valid = skl_ms.train_test_split(
        idx, test_size=test_size, random_state=SEED, shuffle=True
    )

    X_train = X_df.iloc[idx_train]
    X_valid = X_df.iloc[idx_valid]
    y_train = y_enc[idx_train]
    y_valid = y_enc[idx_valid]

    X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.int32, copy=False))
    X_valid_np = np.ascontiguousarray(X_valid.to_numpy(dtype=np.int32, copy=False))
    y_train_np = np.ascontiguousarray(y_train)
    y_valid_np = np.ascontiguousarray(y_valid)

    train_pool = Pool(X_train_np, y_train_np)
    valid_pool = Pool(X_valid_np, y_valid_np)




## === cell 3
try:
    model = CatBoostClassifier(
        iterations=5000,
        task_type="GPU",
        devices="0:1",
        random_seed=SEED,
        verbose=False,
        use_best_model=True,
        od_type="Iter",
        od_wait=200,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=valid_pool, verbose=False)
except Exception as e:
    from catboost import CatBoostError

    if isinstance(e, CatBoostError) and "CUDA" in str(e):
        model = CatBoostClassifier(
            iterations=5000,
            task_type="CPU",
            random_seed=SEED,
            thread_count=(os.cpu_count() or -1),
            verbose=False,
            use_best_model=True,
            od_type="Iter",
            od_wait=200,
            allow_writing_files=False,
        )
        model.fit(train_pool, eval_set=valid_pool, verbose=False)
    else:
        raise




## === cell 4
accuracy = model.score(valid_pool)
print(f"Accuracy of catboost on test data: {accuracy}")

subm_df = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/sample_submission.csv"
)

if dt is not None:
    test_pool = Pool(X_test_dt)
else:
    test_np = np.ascontiguousarray(X_test_df.to_numpy(dtype=np.int32, copy=False))
    test_pool = Pool(test_np)

preds = model.predict(test_pool)
subm_df.Cover_Type = encoder.inverse_transform(preds.astype(np.int32, copy=False))
subm_df.to_csv("Submission CB.csv", index=False)
