# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

np.random.seed(42)



## === cell 1
TRAIN_PATH = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

train_cols = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
test_cols = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()


def _dtype_for_col(c: str):
    if c == "Id":
        return np.int32
    if c == "Cover_Type":
        return np.uint8
    if c.startswith("Wilderness_Area") or c.startswith("Soil_Type"):
        return np.uint8
    return np.int32


train_dtypes = {c: _dtype_for_col(c) for c in train_cols}
test_dtypes = {c: _dtype_for_col(c) for c in test_cols}

train_df = pd.read_csv(
    TRAIN_PATH,
    dtype=train_dtypes,
    usecols=train_cols,
    engine="c",
    low_memory=False,
)
test_df = pd.read_csv(
    TEST_PATH,
    dtype=test_dtypes,
    usecols=test_cols,
    engine="c",
    low_memory=False,
)



## === cell 2
X_np = np.ascontiguousarray(
    train_df.drop("Cover_Type", axis=1).to_numpy(dtype=np.float32, copy=False)
)
Y_np = train_df["Cover_Type"].to_numpy(copy=False)

del train_df
gc.collect()

X_np.shape, Y_np.shape



## === cell 3
from sklearn.model_selection import train_test_split

VAL_SIZE = 20000  # keep identical

_class_counts = np.bincount(Y_np.astype(np.int64, copy=False))
_use_stratify = _class_counts.min() >= 2

x_train, x_val, y_train, y_val = train_test_split(
    X_np,
    Y_np,
    test_size=VAL_SIZE,
    random_state=42,
    stratify=Y_np if _use_stratify else None,
)

del X_np, Y_np
gc.collect()

x_train.shape, x_val.shape, y_train.shape, y_val.shape



## === cell 4
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, DMatrix

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)

dtrain = DMatrix(x_train, label=y_train_enc)
dval = DMatrix(x_val, label=y_val_enc)

try:
    model_xgbc = XGBClassifier(
        n_estimators=20000,
        n_jobs=4,
        learning_rate=0.01,
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        gpu_id=0,
        random_state=42,
        use_label_encoder=False,
    )
    model_xgbc.fit(
        dtrain,
        y_train_enc,
        eval_set=[(dval, y_val_enc)],
        early_stopping_rounds=5,
        verbose=True,
    )
except Exception:
    model_xgbc = XGBClassifier(
        n_estimators=20000,
        n_jobs=4,
        learning_rate=0.01,
        tree_method="hist",
        predictor="auto",
        random_state=42,
        use_label_encoder=False,
    )
    model_xgbc.fit(
        dtrain,
        y_train_enc,
        eval_set=[(dval, y_val_enc)],
        early_stopping_rounds=5,
        verbose=True,
    )

del dtrain, dval, x_train, x_val, y_train, y_val, y_train_enc, y_val_enc
gc.collect()

test_np = np.ascontiguousarray(test_df.to_numpy(dtype=np.float32, copy=False))
dtest = DMatrix(test_np)

_pred_enc = model_xgbc.predict(dtest)
y_predict_xgbc = le.inverse_transform(_pred_enc)



## --- ERROR in cell 4, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/345891275.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     23[0m     )
[0;32m---> 24[0;31m     model_xgbc.fit(
[0m[1;32m     25[0m         [0mdtrain[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)[0m
[1;32m   1499[0m             )
[0;32m-> 1500[0;31m             train_dmatrix, evals = _wrap_evaluation_matrices(
[0m[1;32m   1501[0m                 [0mmissing[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mmissing[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py[0m in [0;36m_wrap_evaluation_matrices[0;34m(missing, X, y, group, qid, sample_weight, base_margin, feature_weights, eval_set, sample_weight_eval_set, base_margin_eval_set, eval_group, eval_qid, create_dmatrix, enable_categorical, feature_types)[0m
[1;32m    520[0m     way."""
[0;32m--> 521[0;31m     train_dmatrix = create_dmatrix(
[0m[1;32m    522[0m         [0mdata[0m[0;34m=[0m[0mX[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py[0m in [0;36m_create_dmatrix[0;34m(self, ref, **kwargs)[0m
[1;32m    962[0m                 [0;32mpass[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 963[0;31m         [0;32mreturn[0m [0mDMatrix[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m,[0m [0mnthread[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mn_jobs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    964[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m__init__[0;34m(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)[0m
[1;32m    856[0m [0;34m[0m[0m
[0;32m--> 857[0;31m         handle, feature_names, feature_types = dispatch_data_backend(
[0m[1;32m    858[0m             [0mdata[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/data.py[0m in [0;36mdispatch_data_backend[0;34m(data, missing, threads, feature_names, feature_types, enable_categorical, data_split_mode)[0m
[1;32m   1130[0m [0;34m[0m[0m
[0;32m-> 1131[0;31m     [0;32mraise[0m [0mTypeError[0m[0;34m([0m[0;34m"Not supported type for data."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mtype[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1132[0m [0;34m[0m[0m

[0;31mTypeError[0m: Not supported type for data.<class 'xgboost.core.DMatrix'>

During handling of the above exception, another exception occurred:

[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/345891275.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     39[0m         [0muse_label_encoder[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     40[0m     )
[0;32m---> 41[0;31m     model_xgbc.fit(
[0m[1;32m     42[0m         [0mdtrain[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     43[0m         [0my_train_enc[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)[0m
[1;32m   1498[0m                 [0mxgb_model[0m[0;34m,[0m [0meval_metric[0m[0;34m,[0m [0mparams[0m[0;34m,[0m [0mearly_stopping_rounds[0m[0;34m,[0m [0mcallbacks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1499[0m             )
[0;32m-> 1500[0;31m             train_dmatrix, evals = _wrap_evaluation_matrices(
[0m[1;32m   1501[0m                 [0mmissing[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mmissing[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1502[0m                 [0mX[0m[0;34m=[0m[0mX[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py[0m in [0;36m_wrap_evaluation_matrices[0;34m(missing, X, y, group, qid, sample_weight, base_margin, feature_weights, eval_set, sample_weight_eval_set, base_margin_eval_set, eval_group, eval_qid, create_dmatrix, enable_categorical, feature_types)[0m
[1;32m    519[0m     """Convert array_like evaluation matrices into DMatrix.  Perform validation on the
[1;32m    520[0m     way."""
[0;32m--> 521[0;31m     train_dmatrix = create_dmatrix(
[0m[1;32m    522[0m         [0mdata[0m[0;34m=[0m[0mX[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    523[0m         [0mlabel[0m[0;34m=[0m[0my[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py[0m in [0;36m_create_dmatrix[0;34m(self, ref, **kwargs)[0m
[1;32m    961[0m             [0;32mexcept[0m [0mTypeError[0m[0;34m:[0m  [0;31m# `QuantileDMatrix` supports lesser types than DMatrix[0m[0;34m[0m[0;34m[0m[0m
[1;32m    962[0m                 [0;32mpass[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 963[0;31m         [0;32mreturn[0m [0mDMatrix[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m,[0m [0mnthread[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mn_jobs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    964[0m [0;34m[0m[0m
[1;32m    965[0m     [0;32mdef[0m [0m_set_evaluation_result[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mevals_result[0m[0;34m:[0m [0mTrainingCallback[0m[0;34m.[0m[0mEvalsLog[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m__init__[0;34m(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)[0m
[1;32m    855[0m             [0;32mreturn[0m[0;34m[0m[0;34m[0m[0m
[1;32m    856[0m [0;34m[0m[0m
[0;32m--> 857[0;31m         handle, feature_names, feature_types = dispatch_data_backend(
[0m[1;32m    858[0m             [0mdata[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    859[0m             [0mmissing[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mmissing[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/data.py[0m in [0;36mdispatch_data_backend[0;34m(data, missing, threads, feature_names, feature_types, enable_categorical, data_split_mode)[0m
[1;32m   1129[0m         )
[1;32m   1130[0m [0;34m[0m[0m
[0;32m-> 1131[0;31m     [0;32mraise[0m [0mTypeError[0m[0;34m([0m[0;34m"Not supported type for data."[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mtype[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1132[0m [0;34m[0m[0m
[1;32m   1133[0m [0;34m[0m[0m

[0;31mTypeError[0m: Not supported type for data.<class 'xgboost.core.DMatrix'>

## === cell 5
result = pd.DataFrame(
    {
        "Id": test_df["Id"].to_numpy(copy=False),
        "Cover_Type": y_predict_xgbc,
    }
)
result.head()
