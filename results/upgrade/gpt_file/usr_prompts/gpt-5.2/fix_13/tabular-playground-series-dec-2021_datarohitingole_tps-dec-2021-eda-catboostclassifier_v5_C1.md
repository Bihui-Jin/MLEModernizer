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

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

try:
    from sklearnex import patch_sklearn  # type: ignore

    patch_sklearn()
except Exception:
    pass



## === cell 1
Base_Path = "/kaggle/input/tabular-playground-series-dec-2021/"

train_path = Base_Path + "train.csv"
test_path = Base_Path + "test.csv"

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
test_cols = [c for c in train_cols if c != "Cover_Type"]

dtype_train = {}
for c in train_cols:
    if c == "Id":
        dtype_train[c] = np.int32
    elif c == "Cover_Type":
        dtype_train[c] = np.int8
    else:
        if ("Wilderness_Area" in c) or ("Soil_Type" in c):
            dtype_train[c] = np.int8
        else:
            dtype_train[c] = np.int32

dtype_test = {
    c: (
        np.int32
        if c == "Id"
        else (np.int8 if (("Wilderness_Area" in c) or ("Soil_Type" in c)) else np.int32)
    )
    for c in test_cols
}

_read_csv_kwargs = {}
try:
    import pyarrow  # noqa: F401

    _read_csv_kwargs["engine"] = "pyarrow"
except Exception:
    _read_csv_kwargs["low_memory"] = False

train = pd.read_csv(
    train_path, dtype=dtype_train, usecols=train_cols, **_read_csv_kwargs
)
test = pd.read_csv(test_path, dtype=dtype_test, usecols=test_cols, **_read_csv_kwargs)



## === cell 2
print(
    f"""
Training Data
    Rows    : {train.shape[0]}
    Columns : {train.shape[1]}

Testing Data
    Rows    : {test.shape[0]}
    Columns : {test.shape[1]}
"""
)



## === cell 3
train_target = train["Cover_Type"]
train_features = train.drop(columns=["Cover_Type"])



## === cell 4
num_cols = train_features.select_dtypes(include=np.number).columns.tolist()
obj_cols = train_features.select_dtypes(include=["object"]).columns.tolist()

print(
    f"""
Count of Numeric Columns : {len(num_cols)}
Count of Object Columns  : {len(obj_cols)}
"""
)



## === cell 5
train_nulls = int(train_features.isna().any().sum())
test_nulls = int(test.isna().any().sum())
print(
    f"""
Count of Columns with Null Values
    Training Data : {train_nulls}
    Testing Data  : {test_nulls}
"""
)



## === cell 6
_ = None



## === cell 7
drop_cols_train = ["Id", "Soil_Type7", "Soil_Type15"]
drop_cols_test = ["Soil_Type7", "Soil_Type15"]

train_features = train_features.drop(columns=drop_cols_train)
test_features = test.drop(columns=drop_cols_test)



## === cell 8
cont_cols = train_features.columns[:10]
cate_cols = train_features.columns[10:]

print(
    f"""
List of Continious Columns :
    {cont_cols}

List of Categorical Columns :
    {cate_cols}
"""
)



## === cell 9
_ = None



## === cell 10
_ = None



## === cell 11
_ = None



## === cell 12
train_cat = np.ascontiguousarray(
    train_features.loc[:, cate_cols].to_numpy(dtype=np.int16, copy=False)
)
test_cat = np.ascontiguousarray(
    test_features.loc[:, cate_cols].to_numpy(dtype=np.int16, copy=False)
)

train_features["Cat_Sum"] = train_cat.sum(axis=1)
test_features["Cat_Sum"] = test_cat.sum(axis=1)

train_features = train_features.drop(columns=cate_cols)
test_features = test_features.drop(columns=cate_cols)

del train_cat, test_cat



## === cell 13
tr_cont = np.ascontiguousarray(
    train_features.loc[:, cont_cols].to_numpy(dtype=np.float32, copy=False)
)
te_cont = np.ascontiguousarray(
    test_features.loc[:, cont_cols].to_numpy(dtype=np.float32, copy=False)
)

train_features["mean"] = tr_cont.mean(axis=1)
train_features["std"] = tr_cont.std(axis=1)
train_features["min"] = tr_cont.min(axis=1)
train_features["max"] = tr_cont.max(axis=1)

test_features["mean"] = te_cont.mean(axis=1)
test_features["std"] = te_cont.std(axis=1)
test_features["min"] = te_cont.min(axis=1)
test_features["max"] = te_cont.max(axis=1)

del tr_cont, te_cont



## === cell 14
_ = None



## === cell 15
_ = None



## === cell 16
standardscaler = StandardScaler()

test_ids = test_features["Id"].to_numpy(copy=False)

X_train_full = train_features.to_numpy(dtype=np.float32, copy=False)
X_all = standardscaler.fit_transform(X_train_full)
X_all = np.ascontiguousarray(X_all, dtype=np.float32)

X_test_full = test_features.drop(columns=["Id"]).to_numpy(dtype=np.float32, copy=False)
X_submit = standardscaler.transform(X_test_full)
X_submit = np.ascontiguousarray(X_submit, dtype=np.float32)

y = train_target.to_numpy(copy=False)

del X_train_full, X_test_full, train_features, test_features



## === cell 17
_ = None



## === cell 18
_ = None



## === cell 19
catb_params = {
    "objective": "MultiClass",
    "task_type": "CPU",
    "silent": True,
    "random_seed": RANDOM_STATE,
    "thread_count": int(os.environ.get("OMP_NUM_THREADS", "4")),
    "allow_writing_files": True,  # enables internal caching; no effect on prediction correctness
    "grow_policy": "SymmetricTree",
}

catboostclassifier = CatBoostClassifier(
    **catb_params,
    loss_function="MultiClass",
    eval_fraction=0.2,  # identical training semantics
    use_best_model=False,  # identical training semantics
)

catboostclassifier.fit(X_all, y, verbose=False)

try:
    eval_pool_obj = catboostclassifier.get_evals()[0][1]
    val_proba = catboostclassifier.predict_proba(eval_pool_obj)
    y_test = eval_pool_obj.get_label()
    y_pred = val_proba.argmax(axis=1) + 1
    acc = float((y_pred == y_test).mean())
    print(f"Internal eval accuracy: {acc:.6f}")
    del val_proba, y_pred, y_test, eval_pool_obj
except Exception:
    pass



## === cell 20
sample_submission = pd.read_csv(Base_Path + "sample_submission.csv", **_read_csv_kwargs)



## === cell 21
pred = catboostclassifier.predict(X_submit)
pred = np.asarray(pred).reshape(-1).astype(int)

del X_submit, train, test, train_target, X_all



## === cell 22
submission_df = pd.DataFrame({"Id": test_ids, "Cover_Type": pred})

if "Id" in sample_submission.columns and len(sample_submission) == len(submission_df):
    if np.array_equal(
        sample_submission["Id"].to_numpy(copy=False),
        submission_df["Id"].to_numpy(copy=False),
    ):
        submission_df = sample_submission[["Id"]].copy()
        submission_df["Cover_Type"] = pred
    else:
        submission_df = sample_submission[["Id"]].merge(
            submission_df, on="Id", how="left"
        )

submission_df["Cover_Type"] = submission_df["Cover_Type"].astype(int)
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
print("Unique predicted classes:", np.sort(submission_df["Cover_Type"].unique()))
