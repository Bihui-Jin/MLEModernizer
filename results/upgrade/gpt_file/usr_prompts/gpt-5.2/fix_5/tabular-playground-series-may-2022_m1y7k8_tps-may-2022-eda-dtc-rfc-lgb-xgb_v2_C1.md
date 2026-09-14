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

# 5. Target score

0.93489

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

import matplotlib.pyplot as plt
import seaborn as sns

from collections import OrderedDict

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
import xgboost as xgb

os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 1))

np.random.seed(42)

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass



## === cell 1
TRAIN_PATH = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-may-2022/test.csv"

float_cols = [f"f_{i:02d}" for i in range(31) if i != 27]
dtypes_train = {c: "float32" for c in float_cols}
dtypes_train.update({"id": "int32", "target": "int8", "f_27": "string"})
dtypes_test = {c: "float32" for c in float_cols}
dtypes_test.update({"id": "int32", "f_27": "string"})

train = pd.read_csv(TRAIN_PATH, dtype=dtypes_train)
test = pd.read_csv(TEST_PATH, dtype=dtypes_test)



## === cell 2
_ = train.head()




## === cell 3
def check(df):
    col_list = df.columns.values
    rows = []
    for col in col_list:
        nunique = df[col].nunique(dropna=False)
        tmp = (
            col,
            df[col].dtype,
            int(df[col].isnull().sum()),
            int(df[col].count()),
            int(nunique),
            None,  # avoid df[col].unique()
        )
        rows.append(tmp)
    df2 = pd.DataFrame(
        rows, columns=["feature", "dtype", "nan", "count", "nunique", "unique"]
    )
    return df2




## === cell 4
_ = None




## === cell 5
def color_negative_red(val):
    color = "red" if val < 0 else "black"
    return "color: %s" % color




## === cell 6
cm = sns.light_palette("green", as_cmap=True)
_ = None



## === cell 7
_ = test.head()



## === cell 8
_ = None



## === cell 9
_ = None



## === cell 10
target_count = train["target"].value_counts()



## === cell 11
_ = train["target"].describe()



## === cell 12
_ = None



## === cell 13
_ = train["f_27"].value_counts()



## === cell 14
_ = test["f_27"].value_counts()



## === cell 15
from collections import OrderedDict


def encord(input_str: str) -> str:
    if not input_str:
        return ""
    seen = OrderedDict.fromkeys(input_str)
    parts = []
    for ch in seen.keys():
        parts.append(ch)
        parts.append(str(input_str.count(ch)))
    return "".join(parts)




## === cell 16
all_f27_str = pd.concat([train["f_27"], test["f_27"]], axis=0).astype(str)

unique_f27 = all_f27_str.unique()
enc_map = {s: encord(s) for s in unique_f27}

train["f_27_en"] = train["f_27"].astype(str).map(enc_map)
_ = train["f_27_en"].value_counts()



## === cell 17
test["f_27_ent"] = test["f_27"].astype(str).map(enc_map)
_ = test["f_27_ent"].value_counts()



## === cell 18
label_f27 = LabelEncoder()
all_f27 = pd.concat([train["f_27"], test["f_27"]], axis=0).astype(str)
label_f27.fit(all_f27)

train["en_27"] = label_f27.transform(train["f_27"].astype(str))
test["en_27"] = label_f27.transform(test["f_27"].astype(str))

label_f27_en = LabelEncoder()
all_f27_en = pd.concat([train["f_27_en"], test["f_27_ent"]], axis=0).astype(str)
label_f27_en.fit(all_f27_en)

train["f_27_enc"] = label_f27_en.transform(train["f_27_en"].astype(str))
test["f_27_enc"] = label_f27_en.transform(test["f_27_ent"].astype(str))

_ = (train["en_27"].head(10), train["f_27_enc"].head(10), test["f_27_enc"].head(10))



## === cell 19
_ = train.head()



## === cell 20
_ = test.head()



## === cell 21
_ = None



## === cell 22
_ = None



## === cell 23
_ = None



## === cell 24
_ = None



## === cell 25
_ = None



## === cell 26
_ = None



## === cell 27
X = train.drop(["id", "target", "f_27", "en_27", "f_27_en"], axis=1)
y = train["target"]

X_test = test.drop(["id", "f_27", "en_27", "f_27_ent"], axis=1)
X_test = X_test.reindex(columns=X.columns)

X_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
y_np = y.to_numpy(dtype=np.int8)
X_test_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float32))

del X, y, X_test
del train, test



## === cell 28
params = {
    "n_estimators": 10000,
    "colsample_bytree": 0.5,
    "subsample": 0.5,
    "learning_rate": 0.02,
    "max_depth": 6,
}



## === cell 29
splits = 5
seed = 42
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

preds = []
scores = []

use_gpu = True
cpu_n_jobs = os.cpu_count() or 1

for fold, (idx_train, idx_valid) in enumerate(skf.split(X_np, y_np)):
    X_train, y_train = X_np[idx_train], y_np[idx_train]
    X_valid, y_valid = X_np[idx_valid], y_np[idx_valid]

    if use_gpu:
        model = XGBClassifier(
            **params,
            booster="gbtree",
            eval_metric="auc",
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            gpu_id=0,
            use_label_encoder=False,
            random_state=seed,
        )
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=100,
                verbose=False,
            )
        except xgb.core.XGBoostError as e:
            if (
                "Must have at least one device" in str(e)
                or "gpu_id" in str(e)
                or "gpu_hist" in str(e)
                or "No visible GPU" in str(e)
                or "CUDA" in str(e)
            ):
                use_gpu = False
            else:
                raise

    if not use_gpu:
        dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
        dvalid = xgb.QuantileDMatrix(X_valid, label=y_valid, ref=dtrain)

        model = XGBClassifier(
            **params,
            booster="gbtree",
            eval_metric="auc",
            tree_method="hist",
            predictor="auto",
            use_label_encoder=False,
            random_state=seed,
            n_jobs=cpu_n_jobs,
        )
        model.fit(
            dtrain,
            y_train,
            eval_set=[(dvalid, y_valid)],
            early_stopping_rounds=100,
            verbose=False,
        )

    pred_valid = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, pred_valid)
    scores.append(float(score))

    test_preds_fold = model.predict_proba(X_test_np)[:, 1]
    preds.append(test_preds_fold)

    print("fold : ", fold, "score : ", score)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3495427804.py in <cell line: 0>()
     61             n_jobs=cpu_n_jobs,
     62         )
---> 63         model.fit(
     64             dtrain,
     65             y_train,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in fit(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)
   1498                 xgb_model, eval_metric, params, early_stopping_rounds, callbacks
   1499             )
-> 1500             train_dmatrix, evals = _wrap_evaluation_matrices(
   1501                 missing=self.missing,
   1502                 X=X,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in _wrap_evaluation_matrices(missing, X, y, group, qid, sample_weight, base_margin, feature_weights, eval_set, sample_weight_eval_set, base_margin_eval_set, eval_group, eval_qid, create_dmatrix, enable_categorical, feature_types)
    519     """Convert array_like evaluation matrices into DMatrix.  Perform validation on the
    520     way."""
--> 521     train_dmatrix = create_dmatrix(
    522         data=X,
    523         label=y,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in _create_dmatrix(self, ref, **kwargs)
    961             except TypeError:  # `QuantileDMatrix` supports lesser types than DMatrix
    962                 pass
--> 963         return DMatrix(**kwargs, nthread=self.n_jobs)
    964 
    965     def _set_evaluation_result(self, evals_result: TrainingCallback.EvalsLog) -> None:

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in __init__(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)
    855             return
    856 
--> 857         handle, feature_names, feature_types = dispatch_data_backend(
    858             data,
    859             missing=self.missing,

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in dispatch_data_backend(data, missing, threads, feature_names, feature_types, enable_categorical, data_split_mode)
   1129         )
   1130 
-> 1131     raise TypeError("Not supported type for data." + str(type(data)))
   1132 
   1133 

TypeError: Not supported type for data.<class 'xgboost.core.QuantileDMatrix'>

## === cell 30
print(scores)
print("CV mean AUC:", float(np.mean(scores)), "std:", float(np.std(scores)))



## === cell 31
sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)

test_pred_mean = np.mean(np.vstack(preds), axis=0)
sub["target"] = test_pred_mean.astype(float)

sub.to_csv("submission.csv", index=False)
_ = sub.head()

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3457475652.py in <cell line: 0>()
      3 )
      4 
----> 5 test_pred_mean = np.mean(np.vstack(preds), axis=0)
      6 sub["target"] = test_pred_mean.astype(float)
      7 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in vstack(tup, dtype, casting)
    287     if not isinstance(arrs, list):
    288         arrs = [arrs]
--> 289     return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
    290 
    291 

ValueError: need at least one array to concatenate
