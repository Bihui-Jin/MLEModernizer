# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

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
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from sklearn.metrics import roc_auc_score

np.random.seed(42)

try:
    from sklearnex import patch_sklearn  # type: ignore

    patch_sklearn()
except Exception:
    pass



## === cell 1
train_path = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
test_path = "/kaggle/input/tabular-playground-series-may-2022/test.csv"

float_feats = [f"f_{i:02d}" for i in range(31) if i != 7 and i != 27]
dtypes_train = {"id": np.int32, "target": np.int8, "f_07": np.int16, "f_27": "string"}
dtypes_train.update({c: np.float32 for c in float_feats})

dtypes_test = {"id": np.int32, "f_07": np.int16, "f_27": "string"}
dtypes_test.update({c: np.float32 for c in float_feats})

train = pd.read_csv(train_path, dtype=dtypes_train)
test = pd.read_csv(test_path, dtype=dtypes_test)

_ = train.shape, test.shape




## === cell 2
def check(df):
    col_list = df.columns.values
    rows = []
    for col in col_list:
        tmp = (
            col,
            df[col].dtype,
            df[col].isnull().sum(),
            df[col].count(),
            df[col].nunique(),
        )
        rows.append(tmp)
    df_out = pd.DataFrame(rows, columns=["feature", "dtype", "nan", "count", "nunique"])
    return df_out




## === cell 3
pass




## === cell 4
def color_negative_red(val):
    color = "red" if val < 0 else "black"
    return "color: %s" % color




## === cell 5
pass



## === cell 6
_ = test.shape



## === cell 7
pass



## === cell 8
pass



## === cell 9
_ = train["target"].value_counts()



## === cell 10
_ = train["target"].describe()



## === cell 11
pass



## === cell 12
_ = train["f_27"].value_counts().head()



## === cell 13
_ = test["f_27"].value_counts().head()



## === cell 14
from collections import OrderedDict


def encord(s: str) -> str:
    d = OrderedDict()
    for ch in s:
        d[ch] = d.get(ch, 0) + 1
    parts = []
    for k, v in d.items():
        parts.append(k)
        parts.append(str(v))
    return "".join(parts)




## === cell 15
def _encord_map_for_values(values: np.ndarray) -> dict:
    out = {}
    for u in values:
        out[u] = encord(u)
    return out


f27_train = train["f_27"]
f27_test = test["f_27"]

uniq_all = np.unique(
    np.concatenate(
        [
            f27_train.dropna().to_numpy(dtype=object, copy=False),
            f27_test.dropna().to_numpy(dtype=object, copy=False),
        ]
    )
)

_enc_map = _encord_map_for_values(uniq_all)

train["f_27_en"] = f27_train.map(_enc_map)
test["f_27_ent"] = f27_test.map(_enc_map)



## === cell 16
label_f27 = LabelEncoder()
all_f27 = np.concatenate(
    [
        train["f_27"].astype(str).to_numpy(copy=False),
        test["f_27"].astype(str).to_numpy(copy=False),
    ]
)
label_f27.fit(all_f27)

train["en_27"] = label_f27.transform(train["f_27"].astype(str).to_numpy(copy=False))

label_f27_en = LabelEncoder()
all_f27_en = np.concatenate(
    [
        train["f_27_en"].astype(str).to_numpy(copy=False),
        test["f_27_ent"].astype(str).to_numpy(copy=False),
    ]
)
label_f27_en.fit(all_f27_en)

train["f_27_enc"] = label_f27_en.transform(
    train["f_27_en"].astype(str).to_numpy(copy=False)
)
test["f_27_enc"] = label_f27_en.transform(
    test["f_27_ent"].astype(str).to_numpy(copy=False)
)

_ = (
    train["en_27"].head(1).tolist(),
    train["f_27_enc"].head(1).tolist(),
    test["f_27_enc"].head(1).tolist(),
)



## === cell 17
_ = train.shape



## === cell 18
_ = test.shape



## === cell 19
pass



## === cell 20
pass



## === cell 21
pass



## === cell 22
pass



## === cell 23
float_cols_train = train.select_dtypes(include=["float64"]).columns
int_cols_train = train.select_dtypes(include=["int64"]).columns
if len(float_cols_train) > 0:
    train[float_cols_train] = train[float_cols_train].astype(np.float32)
if len(int_cols_train) > 0:
    train[int_cols_train] = train[int_cols_train].astype(np.int32)

float_cols_test = test.select_dtypes(include=["float64"]).columns
int_cols_test = test.select_dtypes(include=["int64"]).columns
if len(float_cols_test) > 0:
    test[float_cols_test] = test[float_cols_test].astype(np.float32)
if len(int_cols_test) > 0:
    test[int_cols_test] = test[int_cols_test].astype(np.int32)



## === cell 24
_ = (train.shape, test.shape)



## === cell 25
X = train.drop(["id", "target", "f_27", "en_27", "f_27_en"], axis=1)
y = train["target"]
X_test = test.drop(["id", "f_27", "f_27_ent"], axis=1)

if "f_07" in X.columns:
    X["f_07"] = X["f_07"].astype("category")
    X_test["f_07"] = X_test["f_07"].astype("category")

del train
del test



## === cell 26
params = {
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "n_estimators": 10000,
    "colsample_bytree": 0.5,
    "subsample": 0.5,
    "learning_rate": 0.02,
    "max_depth": 6,
    "enable_categorical": True,
}



## === cell 27
splits = 5
seed = 42
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

preds = []
scores = []

X_idx = np.arange(len(X), dtype=np.int32)
y_np = y.to_numpy(dtype=np.int8, copy=False)

try:
    has_cuda = bool(xgb.core._has_cuda_support())
except Exception:
    has_cuda = False

train_params = {
    "booster": "gbtree",
    "eval_metric": "auc",
    "colsample_bytree": params["colsample_bytree"],
    "subsample": params["subsample"],
    "learning_rate": params["learning_rate"],
    "max_depth": params["max_depth"],
    "enable_categorical": params["enable_categorical"],
    "seed": seed,
    "verbosity": 0,
    "num_parallel_tree": 1,
}

if has_cuda:
    train_params["tree_method"] = "gpu_hist"
    train_params["predictor"] = "gpu_predictor"
else:
    train_params["tree_method"] = "hist"
    train_params["predictor"] = "cpu_predictor"
    train_params["nthread"] = min(4, os.cpu_count() or 1)

num_boost_round = params["n_estimators"]

dtest = xgb.DMatrix(X_test, enable_categorical=True)

for fold, (idx_train, idx_valid) in enumerate(skf.split(X_idx, y_np)):
    dtrain = xgb.DMatrix(
        X.iloc[idx_train],
        label=y_np[idx_train],
        enable_categorical=True,
    )
    dvalid = xgb.DMatrix(
        X.iloc[idx_valid],
        label=y_np[idx_valid],
        enable_categorical=True,
    )

    booster = xgb.train(
        params=train_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    pred_valid = booster.predict(
        dvalid, iteration_range=(0, booster.best_iteration + 1)
    )
    score = roc_auc_score(y_np[idx_valid], pred_valid)
    scores.append(score)

    test_preds = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    preds.append(test_preds)

    print("fold : ", fold, "score : ", score)



## === cell 28
print(scores)



## === cell 29
sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)

sub["target"] = np.mean(np.vstack(preds), axis=0)
sub.to_csv("submission.csv", index=False)
sub.head()
