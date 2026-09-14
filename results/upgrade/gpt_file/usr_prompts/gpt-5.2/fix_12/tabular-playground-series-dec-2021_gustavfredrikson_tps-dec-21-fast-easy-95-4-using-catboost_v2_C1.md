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
import numpy as np
import pandas as pd

import sklearn.model_selection as skl_ms
from sklearn.preprocessing import LabelEncoder

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

N_THREADS = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(N_THREADS))

from catboost import CatBoostClassifier, Pool



## === cell 1
CANDIDATE_BASES = [
    "../input/tabular-playground-series-dec-2021",  # original Kaggle Notebook path
    "/kaggle/input/tabular-playground-series-dec-2021",  # provided in this environment
    "/kaggle/data/tabular-playground-series-dec-2021",  # provided in this environment
    "/kaggle/input",  # fallback root
    "/kaggle/data",  # fallback root
]

BASE_PATH = None
for p in CANDIDATE_BASES:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        BASE_PATH = p
        break

if BASE_PATH is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv. Tried: " + ", ".join(CANDIDATE_BASES)
    )

train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sub_path = os.path.join(BASE_PATH, "sample_submission.csv")

num_cols = [
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
wilderness_cols = [f"Wilderness_Area{i}" for i in range(1, 5)]
soil_cols = [f"Soil_Type{i}" for i in range(1, 41)]

feature_cols = num_cols + wilderness_cols + soil_cols

train_usecols = ["Id"] + feature_cols + ["Cover_Type"]
test_usecols = ["Id"] + feature_cols

dtype_train = {c: np.int16 for c in num_cols}
dtype_train.update({c: np.int8 for c in wilderness_cols})
dtype_train.update({c: np.int8 for c in soil_cols})
dtype_train.update({"Id": np.int32, "Cover_Type": np.int8})

dtype_test = dtype_train.copy()
dtype_test.pop("Cover_Type", None)

train_df = pd.read_csv(
    train_path,
    usecols=train_usecols,
    dtype=dtype_train,
    engine="c",
    low_memory=False,
)
test_df = pd.read_csv(
    test_path,
    usecols=test_usecols,
    dtype=dtype_test,
    engine="c",
    low_memory=False,
)

encoder = LabelEncoder()
train_df["Cover_Type"] = encoder.fit_transform(train_df["Cover_Type"])



## === cell 2
test_size = 0.01  # keep identical intent

n = len(train_df)
idx = np.arange(n, dtype=np.int32)
idx_train, idx_valid = skl_ms.train_test_split(
    idx, test_size=test_size, random_state=RANDOM_STATE, shuffle=True
)

train_part = train_df.take(idx_train)
valid_part = train_df.take(idx_valid)

X_train = np.ascontiguousarray(train_part[feature_cols].to_numpy(copy=False))
y_train = train_part["Cover_Type"].to_numpy(copy=False)

X_valid = np.ascontiguousarray(valid_part[feature_cols].to_numpy(copy=False))
y_valid = valid_part["Cover_Type"].to_numpy(copy=False)

train_pool = Pool(
    X_train,
    y_train,
    feature_names=feature_cols,
    cat_features=[],
)
valid_pool = Pool(
    X_valid,
    y_valid,
    feature_names=feature_cols,
    cat_features=[],
)




## === cell 3
def build_model():
    params = dict(
        iterations=5000,
        random_seed=RANDOM_STATE,
        loss_function="MultiClass",
        verbose=False,
        allow_writing_files=False,
        use_best_model=True,
        od_type="Iter",
        od_wait=200,
        grow_policy="SymmetricTree",
    )
    model_cpu = CatBoostClassifier(
        **params,
        task_type="CPU",
        thread_count=N_THREADS,
    )
    model_cpu.fit(train_pool, eval_set=valid_pool, verbose=False)
    return model_cpu, "CPU"


model, used_device = build_model()
print(f"Trained CatBoost using: {used_device}")



## === cell 4
accuracy = model.score(valid_pool)
print(f"Accuracy of catboost on holdout data: {accuracy}")



## === cell 5
sample_sub = pd.read_csv(sub_path, usecols=["Id"])

test_ids = test_df["Id"].to_numpy(copy=False)
sub_ids = sample_sub["Id"].to_numpy(copy=False)

X_test = np.ascontiguousarray(test_df[feature_cols].to_numpy(copy=False))
test_pool_full = Pool(X_test, feature_names=feature_cols, cat_features=[])

preds = model.predict(test_pool_full)
preds = np.asarray(preds).reshape(-1).astype(int)
test_pred_labels = encoder.inverse_transform(preds).astype(np.int64)

out_path = "submission_cb.csv"

if len(sub_ids) == len(test_ids) and np.array_equal(sub_ids, test_ids):
    subm_df = sample_sub.copy()
    subm_df["Cover_Type"] = test_pred_labels
else:
    subm_df = sample_sub.copy()
    pred_by_id = pd.Series(test_pred_labels, index=test_ids)
    subm_df["Cover_Type"] = subm_df["Id"].map(pred_by_id).astype(np.int64)

    if subm_df["Cover_Type"].isna().any():
        missing = int(subm_df["Cover_Type"].isna().sum())
        raise RuntimeError(
            f"Submission has {missing} missing predictions after Id alignment."
        )

subm_df.to_csv(out_path, index=False)
print(
    f"Wrote submission to: {out_path} with shape {subm_df.shape} and columns {list(subm_df.columns)}"
)
