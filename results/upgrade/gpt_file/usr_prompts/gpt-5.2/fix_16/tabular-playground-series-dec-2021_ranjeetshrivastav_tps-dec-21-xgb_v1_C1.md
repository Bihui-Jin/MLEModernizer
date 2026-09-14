# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.94889

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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

from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

os.environ.setdefault("PYTHONHASHSEED", "0")

RANDOM_STATE = 2021
N_SPLITS = 5

_CPU = os.cpu_count() or 4
N_JOBS = min(_CPU, 8)

os.environ.setdefault("OMP_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("MKL_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(N_JOBS))

np.random.seed(RANDOM_STATE)

XGB_CACHE_DIR = "./xgb_cache"
os.makedirs(XGB_CACHE_DIR, exist_ok=True)




## === cell 1
DATA_DIR = r"../input/tabular-playground-series-dec-2021"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()
feature_cols = [c for c in test_cols if c != "Id"]

dtype_map_train = {c: np.int32 for c in feature_cols}
dtype_map_train["Id"] = np.int32
dtype_map_train["Cover_Type"] = np.int16

dtype_map_test = {c: np.int32 for c in feature_cols}
dtype_map_test["Id"] = np.int32

_read_csv_kwargs = dict(low_memory=False)

try:
    _read_csv_kwargs_pa = dict(_read_csv_kwargs)
    _read_csv_kwargs_pa["engine"] = "pyarrow"
    train = pd.read_csv(
        train_path,
        usecols=["Cover_Type"] + feature_cols,
        dtype=dtype_map_train,
        **_read_csv_kwargs_pa,
    )
    test = pd.read_csv(
        test_path,
        usecols=["Id"] + feature_cols,
        dtype=dtype_map_test,
        **_read_csv_kwargs_pa,
    )
except Exception:
    train = pd.read_csv(
        train_path,
        usecols=["Cover_Type"] + feature_cols,
        dtype=dtype_map_train,
        **_read_csv_kwargs,
    )
    test = pd.read_csv(
        test_path,
        usecols=["Id"] + feature_cols,
        dtype=dtype_map_test,
        **_read_csv_kwargs,
    )

sample_submission = pd.read_csv(sub_path)

print(f"train set have {train.shape[0]} rows and {train.shape[1]} columns.")
print(f"test set have {test.shape[0]} rows and {test.shape[1]} columns.")
print(
    f"sample_submission set have {sample_submission.shape[0]} rows and {sample_submission.shape[1]} columns."
)




## === cell 2
train = train[train["Cover_Type"] != 5].reset_index(drop=True)




## === cell 3
y_raw = train["Cover_Type"].astype(np.int16)
X_df = train.drop(columns=["Cover_Type"])

classes_ = np.sort(y_raw.unique())  # e.g. [1,2,3,4,6,7]
class_to_idx = {c: i for i, c in enumerate(classes_)}
idx_to_class = {i: c for i, c in enumerate(classes_)}

y = y_raw.map(class_to_idx).astype(np.int16)

X = np.ascontiguousarray(X_df.to_numpy(dtype=np.float32, copy=False))
y_np = np.ascontiguousarray(y.to_numpy(dtype=np.int32, copy=False))

test_ids = test["Id"].to_numpy(copy=False)
X_test = np.ascontiguousarray(
    test.drop(columns=["Id"]).to_numpy(dtype=np.float32, copy=False)
)

folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

n_classes = len(classes_)
proba_sum = np.zeros((X_test.shape[0], n_classes), dtype=np.float32)




## === cell 4
params = dict(
    tree_method="hist",
    objective="multi:softprob",
    num_class=n_classes,
    seed=RANDOM_STATE,
    eval_metric="mlogloss",
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    nthread=N_JOBS,
)

dtest = xgb.QuantileDMatrix(
    X_test,
    cache_prefix=os.path.join(XGB_CACHE_DIR, "test"),
)

for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y_np), start=1):
    print(f"Fold: {fold}")

    dtrn = xgb.QuantileDMatrix(
        X[trn_idx],
        label=y_np[trn_idx],
        cache_prefix=os.path.join(XGB_CACHE_DIR, f"train_f{fold}"),
    )
    dval = xgb.QuantileDMatrix(
        X[val_idx],
        label=y_np[val_idx],
        cache_prefix=os.path.join(XGB_CACHE_DIR, f"val_f{fold}"),
    )

    booster = xgb.train(
        params=params,
        dtrain=dtrn,
        num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )

    best_iter = int(getattr(booster, "best_iteration", 0)) + 1

    proba_valid = booster.predict(dval, iteration_range=(0, best_iter))
    pred_valid = np.argmax(proba_valid, axis=1).astype(np.int32)

    y_val = y_np[val_idx]
    acc = float((pred_valid == y_val).mean())
    print(f" accuracy_score: {acc}")
    print("-" * 50)

    proba_test = booster.predict(dtest, iteration_range=(0, best_iter))
    proba_sum += proba_test / N_SPLITS

pred_idx = np.argmax(proba_sum, axis=1)
classes_arr = np.asarray(classes_, dtype=np.int16)
predictions = classes_arr[pred_idx].astype(int)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/750193691.py in <cell line: 0>()
     15 # Using cache_prefix enables on-disk caching of quantile cuts to avoid repeated work across folds.
     16 # Correctness is preserved for hist training/prediction; differences are at most negligible FP rounding.
---> 17 dtest = xgb.QuantileDMatrix(
     18     X_test,
     19     cache_prefix=os.path.join(XGB_CACHE_DIR, "test"),

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

TypeError: QuantileDMatrix.__init__() got an unexpected keyword argument 'cache_prefix'

## === cell 5
submission = pd.DataFrame(
    {"Id": test_ids.astype(int), "Cover_Type": predictions.astype(int)}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1264843808.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"Id": test_ids.astype(int), "Cover_Type": predictions.astype(int)}
      3 )
      4 submission.to_csv("submission.csv", index=False)
      5 

NameError: name 'predictions' is not defined
