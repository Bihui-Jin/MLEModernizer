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
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
optuna==4.5.0
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
import gc
import random
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn  # scikit-learn-intelex

    patch_sklearn()
except Exception:
    pass

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from lightgbm import LGBMClassifier  # kept (not used) to preserve original imports
import lightgbm as lgb

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 6)

random.seed(48)
np.random.seed(48)

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")




## === cell 1
TRAIN_PATH = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/test.csv"

train_cols = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
cols = [c for c in train_cols if c not in ("Id", "Cover_Type")]

continous_features = cols[:10]
categorical_features = cols[10:]

dtype_train = {"Id": np.int32, "Cover_Type": np.int8}
dtype_test = {"Id": np.int32}
for c in cols:
    dtype_train[c] = np.int16
    dtype_test[c] = np.int16

read_kwargs = {"low_memory": False}
try:
    import pyarrow  # noqa: F401

    read_kwargs = {"engine": "pyarrow"}  # pyarrow doesn't support low_memory
except Exception:
    pass

train = pd.read_csv(TRAIN_PATH, dtype=dtype_train, **read_kwargs)
test = pd.read_csv(TEST_PATH, dtype=dtype_test, **read_kwargs)

print("Loaded train/test:", train.shape, test.shape)
print("Cover_Type value counts (train):")
print(train["Cover_Type"].value_counts().sort_index())




## === cell 2
pass




## === cell 3
pass




## === cell 4
pass




## === cell 5
pass




## === cell 6
pass




## === cell 7
pass




## === cell 8
pass




## === cell 9
pass




## === cell 10
pass




## === cell 11
pass




## === cell 12
def reduce_mem_usage(df, verbose=True):
    numerics = [
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
        "int8",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ]
    start_memory = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype
        if col_type.name in numerics:
            arr = df[col].to_numpy(copy=False)
            c_min = arr.min()
            c_max = arr.max()

            if np.issubdtype(col_type, np.integer):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif (
                    c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max
                ):
                    df[col] = df[col].astype(np.int16)
                elif (
                    c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max
                ):
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min >= np.finfo(np.float16).min
                    and c_max <= np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min >= np.finfo(np.float32).min
                    and c_max <= np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_memory = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage of dataframe after reduction {end_memory:.2f} MB")
        print(f"Reduced by {100 * (start_memory - end_memory) / start_memory:.2f} % ")
    return df




## === cell 13
def _already_compact(df, feature_cols):
    dts = df[feature_cols].dtypes
    return (dts == np.int8).all() or (dts == np.int16).all()


if False and not (_already_compact(train, cols) and _already_compact(test, cols)):
    all_feat = pd.concat([train[cols], test[cols]], axis=0, ignore_index=True)
    all_feat = reduce_mem_usage(all_feat)
    train.loc[:, cols] = all_feat.iloc[: len(train)].to_numpy(copy=False)
    test.loc[:, cols] = all_feat.iloc[len(train) :].to_numpy(copy=False)
    del all_feat
    gc.collect()




## === cell 14
train = train.loc[train["Cover_Type"] != 5].reset_index(drop=True)
print("After filtering Cover_Type!=5:", train.shape)
print("Remaining classes:", sorted(train["Cover_Type"].unique().tolist()))




## === cell 15
tr_cont = train[continous_features].to_numpy(copy=False)
te_cont = test[continous_features].to_numpy(copy=False)

train_mean = tr_cont.mean(axis=1, dtype=np.float32)
train_min = tr_cont.min(axis=1)
train_max = tr_cont.max(axis=1)

test_mean = te_cont.mean(axis=1, dtype=np.float32)
test_min = te_cont.min(axis=1)
test_max = te_cont.max(axis=1)

train["mean"] = train_mean
train["min"] = train_min
train["max"] = train_max

test["mean"] = test_mean
test["min"] = test_min
test["max"] = test_max

cols = [c for c in test.columns if c != "Id"]
print("Feature columns:", len(cols))




## === cell 16
params = {
    "objective": "multiclass",
    "random_state": 48,
    "n_estimators": 20000,
    "n_jobs": -1,
    "reg_alpha": 0.0010309124257626384,
    "reg_lambda": 9.48149567512538,
    "colsample_bytree": 0.5,
    "subsample": 1,
    "learning_rate": 0.3,
    "max_depth": 100,
    "num_leaves": 142,
    "min_child_samples": 204,
    "cat_smooth": 99,
    "force_col_wise": True,
    "deterministic": True,
    "num_threads": 4,
    "verbosity": -1,
}




## === cell 17
X_all = np.ascontiguousarray(train[cols].to_numpy(copy=False)).astype(
    np.float32, copy=False
)
X_test = np.ascontiguousarray(test[cols].to_numpy(copy=False)).astype(
    np.float32, copy=False
)

y_raw = train["Cover_Type"].to_numpy(copy=False)
classes_sorted = np.array(sorted(np.unique(y_raw)), dtype=np.int16)
class_to_index = {int(c): i for i, c in enumerate(classes_sorted.tolist())}
index_to_class = classes_sorted  # index -> original label

mapper = np.zeros(int(classes_sorted.max()) + 1, dtype=np.int16)
for k, v in class_to_index.items():
    mapper[int(k)] = np.int16(v)
y_all = mapper[y_raw.astype(np.int32, copy=False)]

num_class = int(len(classes_sorted))
print("Encoded num_class:", num_class)
print("Class mapping (original -> encoded):", class_to_index)

kf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)

acc = []
vote_counts = np.zeros((X_test.shape[0], num_class), dtype=np.uint8)

lgb_params = {
    "objective": "multiclass",
    "num_class": num_class,
    "learning_rate": params["learning_rate"],
    "max_depth": params["max_depth"],
    "num_leaves": params["num_leaves"],
    "min_child_samples": params["min_child_samples"],
    "cat_smooth": params["cat_smooth"],
    "colsample_bytree": params["colsample_bytree"],
    "subsample": params["subsample"],
    "reg_alpha": params["reg_alpha"],
    "reg_lambda": params["reg_lambda"],
    "force_col_wise": params["force_col_wise"],
    "deterministic": params["deterministic"],
    "verbosity": params["verbosity"],
    "num_threads": params["num_threads"],
    "seed": params["random_state"],
    "metric": "multi_logloss",
}

n_rounds = int(params["n_estimators"])

model = None

for n, (trn_idx, val_idx) in enumerate(kf.split(X_all, y_all)):
    trn_idx = trn_idx.astype(np.int32, copy=False)
    val_idx = val_idx.astype(np.int32, copy=False)

    X_tr, X_val = X_all[trn_idx], X_all[val_idx]
    y_tr, y_val = y_all[trn_idx], y_all[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=n_rounds,
        valid_sets=[dvalid],
    )

    val_prob = model.predict(X_val, num_iteration=n_rounds)
    val_pred = np.argmax(val_prob, axis=1).astype(np.int16, copy=False)

    test_prob = model.predict(X_test, num_iteration=n_rounds)
    test_pred = np.argmax(test_prob, axis=1).astype(np.int16, copy=False)

    vote_counts[np.arange(X_test.shape[0]), test_pred] += 1

    acc_fold = accuracy_score(y_val, val_pred)
    acc.append(acc_fold)
    print(f"fold: {n+1} , accuracy: {round(acc_fold*100, 3)}")

    del (
        X_tr,
        X_val,
        y_tr,
        y_val,
        dtrain,
        dvalid,
        val_prob,
        test_prob,
        val_pred,
        test_pred,
        trn_idx,
        val_idx,
    )
    gc.collect()




## === cell 18
print(f"the mean Accuracy is : {round(np.mean(acc)*100, 3)} ")




## === cell 19
"""from sklearn.metrics import confusion_matrix, classification_report
pred_val = model.predict(X_val)
print(classification_report(y_val, model.predict(pred_val)))"""




## === cell 20
try:
    _ = lgb.plot_importance(model, max_num_features=20, figsize=(10, 10))
    plt.close()
except Exception as e:
    print("Skipping feature importance plot due to:", repr(e))




## === cell 21
SAMPLE_SUB_PATH = (
    "/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv"
)
if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"

sub = pd.read_csv(SAMPLE_SUB_PATH)

pred_encoded = vote_counts.argmax(axis=1).astype(np.int16, copy=False)
prediction = index_to_class[pred_encoded].astype(np.int16, copy=False)

if "Id" not in sub.columns or len(sub) != len(test):
    sub = pd.DataFrame({"Id": test["Id"].to_numpy(copy=False)})

sub["Id"] = test["Id"].to_numpy(copy=False)
sub["Cover_Type"] = prediction
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())




## === cell 22
sub
