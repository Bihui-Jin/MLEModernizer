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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier



## === cell 2
import warnings

warnings.filterwarnings("ignore")



## === cell 3
train = pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
test = pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")

x_data = train.drop(
    ["Id", "Cover_Type", "Soil_Type1", "Soil_Type7", "Soil_Type15"], axis=1
)
x_test = test.drop(["Id", "Soil_Type1", "Soil_Type7", "Soil_Type15"], axis=1)

y_data = train.Cover_Type



## === cell 4
x_data["mean"] = x_data.mean(axis=1)
x_data["std"] = x_data.std(axis=1)
x_data["max"] = x_data.max(axis=1)
x_data["min"] = x_data.min(axis=1)
fcols = x_data.columns[:3]
x_data["f1"] = x_data[fcols[0]] * x_data[fcols[1]] / (x_data[fcols[2]] + 1)



## === cell 5
x_test["mean"] = x_test.mean(axis=1)
x_test["std"] = x_test.std(axis=1)
x_test["max"] = x_test.max(axis=1)
x_test["min"] = x_test.min(axis=1)
fcols_t = x_test.columns[:3]
x_test["f1"] = x_test[fcols_t[0]] * x_test[fcols_t[1]] / (x_test[fcols_t[2]] + 1)



## === cell 6
class_counts = y_data.value_counts(dropna=False)
stratify_arg = y_data if (class_counts.min() >= 2) else None

x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42, stratify=stratify_arg
)


## === cell 7
from xgboost.core import XGBoostError

classes_ = np.sort(pd.unique(y_train))
class_to_idx = {c: i for i, c in enumerate(classes_)}
idx_to_class = np.array(classes_, dtype=np.int64)

y_train_xgb = y_train.map(class_to_idx).astype(np.int64)

try:
    model = XGBClassifier(tree_method="gpu_hist", predictor="gpu_predictor")
    model.fit(x_train, y_train_xgb)
except XGBoostError:
    model = XGBClassifier(tree_method="hist")
    model.fit(x_train, y_train_xgb)

y_pred_idx = model.predict(x_val).astype(np.int64)
y_pred = idx_to_class[y_pred_idx]
acc = accuracy_score(y_val, y_pred)
print(f"validation acc {acc}")

_orig_predict = model.predict


def _predict_with_decode_minus_one(X):
    pred_idx = _orig_predict(X).astype(np.int64)
    decoded = idx_to_class[pred_idx]
    return decoded - 1


model.predict = _predict_with_decode_minus_one


## --- ERROR in cell 7, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mXGBoostError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2597055490.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     14[0m     [0mmodel[0m [0;34m=[0m [0mXGBClassifier[0m[0;34m([0m[0mtree_method[0m[0;34m=[0m[0;34m"gpu_hist"[0m[0;34m,[0m [0mpredictor[0m[0;34m=[0m[0;34m"gpu_predictor"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 15[0;31m     [0mmodel[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mx_train[0m[0;34m,[0m [0my_train_xgb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     16[0m [0;32mexcept[0m [0mXGBoostError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m    957[0m             [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 958[0;31m                 return QuantileDMatrix(
[0m[1;32m    959[0m                     [0;34m**[0m[0mkwargs[0m[0;34m,[0m [0mref[0m[0;34m=[0m[0mref[0m[0;34m,[0m [0mnthread[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mn_jobs[0m[0;34m,[0m [0mmax_bin[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mmax_bin[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m__init__[0;34m(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, max_bin, ref, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)[0m
[1;32m   1528[0m [0;34m[0m[0m
[0;32m-> 1529[0;31m         self._init(
[0m[1;32m   1530[0m             [0mdata[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m_init[0;34m(self, data, ref, enable_categorical, **meta)[0m
[1;32m   1589[0m         [0;31m# delay check_call to throw intermediate exception first[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1590[0;31m         [0m_check_call[0m[0;34m([0m[0mret[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1591[0m         [0mself[0m[0;34m.[0m[0mhandle[0m [0;34m=[0m [0mhandle[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m_check_call[0;34m(ret)[0m
[1;32m    281[0m     [0;32mif[0m [0mret[0m [0;34m!=[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 282[0;31m         [0;32mraise[0m [0mXGBoostError[0m[0;34m([0m[0mpy_str[0m[0;34m([0m[0m_LIB[0m[0;34m.[0m[0mXGBGetLastError[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    283[0m [0;34m[0m[0m

[0;31mXGBoostError[0m: [08:31:20] /workspace/src/data/../common/../data/gradient_index.h:94: Check failed: valid: Input data contains `inf` or a value too large, while `missing` is not set to `inf`
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3effba) [0x7fff83903fba]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x407477) [0x7fff8391b477]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3f6316) [0x7fff8390a316]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3f8858) [0x7fff8390c858]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3a2a07) [0x7fff838b6a07]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGQuantileDMatrixCreateFromCallback+0x2b0) [0x7fff83679c40]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]



During handling of the above exception, another exception occurred:

[0;31mXGBoostError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2597055490.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     16[0m [0;32mexcept[0m [0mXGBoostError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     17[0m     [0mmodel[0m [0;34m=[0m [0mXGBClassifier[0m[0;34m([0m[0mtree_method[0m[0;34m=[0m[0;34m"hist"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 18[0;31m     [0mmodel[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mx_train[0m[0;34m,[0m [0my_train_xgb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     19[0m [0;34m[0m[0m
[1;32m     20[0m [0;31m# Decode validation predictions back to original label space for correct accuracy.[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m    956[0m         [0;32mif[0m [0m_can_use_qdm[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mtree_method[0m[0;34m)[0m [0;32mand[0m [0mself[0m[0;34m.[0m[0mbooster[0m [0;34m!=[0m [0;34m"gblinear"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    957[0m             [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 958[0;31m                 return QuantileDMatrix(
[0m[1;32m    959[0m                     [0;34m**[0m[0mkwargs[0m[0;34m,[0m [0mref[0m[0;34m=[0m[0mref[0m[0;34m,[0m [0mnthread[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mn_jobs[0m[0;34m,[0m [0mmax_bin[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mmax_bin[0m[0;34m[0m[0;34m[0m[0m
[1;32m    960[0m                 )

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m__init__[0;34m(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, max_bin, ref, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)[0m
[1;32m   1527[0m                 )
[1;32m   1528[0m [0;34m[0m[0m
[0;32m-> 1529[0;31m         self._init(
[0m[1;32m   1530[0m             [0mdata[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1531[0m             [0mref[0m[0;34m=[0m[0mref[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m_init[0;34m(self, data, ref, enable_categorical, **meta)[0m
[1;32m   1588[0m         [0mit[0m[0;34m.[0m[0mreraise[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1589[0m         [0;31m# delay check_call to throw intermediate exception first[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1590[0;31m         [0m_check_call[0m[0;34m([0m[0mret[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1591[0m         [0mself[0m[0;34m.[0m[0mhandle[0m [0;34m=[0m [0mhandle[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1592[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m_check_call[0;34m(ret)[0m
[1;32m    280[0m     """
[1;32m    281[0m     [0;32mif[0m [0mret[0m [0;34m!=[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 282[0;31m         [0;32mraise[0m [0mXGBoostError[0m[0;34m([0m[0mpy_str[0m[0;34m([0m[0m_LIB[0m[0;34m.[0m[0mXGBGetLastError[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    283[0m [0;34m[0m[0m
[1;32m    284[0m [0;34m[0m[0m

[0;31mXGBoostError[0m: [08:31:26] /workspace/src/data/../common/../data/gradient_index.h:94: Check failed: valid: Input data contains `inf` or a value too large, while `missing` is not set to `inf`
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3effba) [0x7fff83903fba]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x407477) [0x7fff8391b477]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3f6316) [0x7fff8390a316]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3f8858) [0x7fff8390c858]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3a2a07) [0x7fff838b6a07]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGQuantileDMatrixCreateFromCallback+0x2b0) [0x7fff83679c40]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]



## === cell 8
y_test = model.predict(x_test) + 1
