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

import matplotlib.pyplot as plt
import seaborn as sns

from collections import OrderedDict

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_curve, auc

from xgboost import XGBClassifier
import xgboost as xgb

np.random.seed(42)



## === cell 1
train_path = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
test_path = "/kaggle/input/tabular-playground-series-may-2022/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)



## === cell 2
_ = train.shape, test.shape




## === cell 3
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
            df[col].unique(),
        )
        rows.append(tmp)
    df = pd.DataFrame(rows)
    df.columns = ["feature", "dtype", "nan", "count", "nunique", "unique"]
    return df




## === cell 4
pass




## === cell 5
def color_negative_red(val):
    color = "red" if val < 0 else "black"
    return "color: %s" % color




## === cell 6
pass



## === cell 7
_ = test.shape



## === cell 8
pass



## === cell 9
pass



## === cell 10
_ = train["target"].value_counts()



## === cell 11
_ = train["target"].describe()



## === cell 12
pass



## === cell 13
_ = train["f_27"].value_counts().head()



## === cell 14
_ = test["f_27"].value_counts().head()



## === cell 15
from collections import OrderedDict


def encord(input):
    dict = OrderedDict.fromkeys(input, 0)

    for ch in input:
        dict[ch] += 1

    output = ""
    for k, v in dict.items():
        output = output + k + str(v)
    return output




## === cell 16
def _encord_map_for_values(values: np.ndarray) -> dict:
    out = {}
    for u in values:
        out[u] = encord(u)
    return out


f27_train = train["f_27"]
f27_test = test["f_27"]

uniq_all = (
    pd.Index(f27_train.dropna().unique())
    .append(pd.Index(f27_test.dropna().unique()))
    .unique()
)

_enc_map = _encord_map_for_values(uniq_all.to_numpy(dtype=object))

train["f_27_en"] = f27_train.map(_enc_map)
test["f_27_ent"] = f27_test.map(_enc_map)



## === cell 17
label_f27 = LabelEncoder()
label_f27.fit(pd.concat([train["f_27"], test["f_27"]], axis=0).astype(str).to_numpy())

train["en_27"] = label_f27.transform(train["f_27"].astype(str).to_numpy())
label_f27_en = LabelEncoder()
label_f27_en.fit(
    pd.concat([train["f_27_en"], test["f_27_ent"]], axis=0).astype(str).to_numpy()
)

train["f_27_enc"] = label_f27_en.transform(train["f_27_en"].astype(str).to_numpy())
test["f_27_enc"] = label_f27_en.transform(test["f_27_ent"].astype(str).to_numpy())

_ = (
    train["en_27"].head(1).tolist(),
    train["f_27_enc"].head(1).tolist(),
    test["f_27_enc"].head(1).tolist(),
)



## === cell 18
_ = train.shape



## === cell 19
_ = test.shape



## === cell 20
pass



## === cell 21
pass



## === cell 22
pass



## === cell 23
pass



## === cell 24
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



## === cell 25
_ = (train.shape, test.shape)



## === cell 26
X = train.drop(["id", "target", "f_27", "en_27", "f_27_en"], axis=1)
y = train["target"]
X_test = test.drop(["id", "f_27", "f_27_ent"], axis=1)

del train
del test



## === cell 27
params = {
    "tree_method": "gpu_hist",
    "n_estimators": 10000,
    "colsample_bytree": 0.5,
    "subsample": 0.5,
    "learning_rate": 0.02,
    "max_depth": 6,
}



## === cell 28
splits = 5
seed = 42
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

preds = []
scores = []

X_np = np.ascontiguousarray(X.to_numpy())
y_np = y.to_numpy()
X_test_np = np.ascontiguousarray(X_test.to_numpy())

n_jobs = os.cpu_count() or 1

for fold, (idx_train, idx_valid) in enumerate(skf.split(X_np, y_np)):
    X_train, y_train = X_np[idx_train], y_np[idx_train]
    X_valid, y_valid = X_np[idx_valid], y_np[idx_valid]

    params_cpu = dict(params)
    params_cpu["tree_method"] = "hist"

    model = XGBClassifier(
        **params_cpu,
        booster="gbtree",
        eval_metric="auc",
        use_label_encoder=False,
        random_state=seed,
        n_jobs=n_jobs,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=False,
    )

    pred_valid = model.predict_proba(X_valid)[:, 1]
    fpr, tpr, _ = roc_curve(y_valid, pred_valid)
    score = auc(fpr, tpr)
    scores.append(score)

    test_preds = model.predict_proba(X_test_np)[:, 1]
    preds.append(test_preds)

    print("fold : ", fold, "score : ", score)



## === cell 29
print(scores)



## === cell 30
sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)



## === cell 31
sub["target"] = np.mean(np.vstack(preds), axis=0)
sub.to_csv("submission.csv", index=False)
sub.head()
