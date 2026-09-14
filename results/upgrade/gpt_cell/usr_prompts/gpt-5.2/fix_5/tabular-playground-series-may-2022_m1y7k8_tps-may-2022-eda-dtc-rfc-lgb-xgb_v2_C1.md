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

# 3. Data file paths

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

# 4. Code solution

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

float_feats = [f"f_{i:02d}" for i in range(31) if i != 7 and i != 27]
dtypes_train = {"id": np.int32, "target": np.int8, "f_07": np.int16, "f_27": "string"}
dtypes_train.update({c: np.float32 for c in float_feats})

dtypes_test = {"id": np.int32, "f_07": np.int16, "f_27": "string"}
dtypes_test.update({c: np.float32 for c in float_feats})

train = pd.read_csv(train_path, dtype=dtypes_train)
test = pd.read_csv(test_path, dtype=dtypes_test)



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

if "f_07" in X.columns:
    X["f_07"] = X["f_07"].astype("category")
    X_test["f_07"] = X_test["f_07"].astype("category")

del train
del test



## === cell 27
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



## === cell 28
splits = 5
seed = 42
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

preds = []
scores = []

X_idx = np.arange(len(X), dtype=np.int32)
y_np = y.to_numpy(dtype=np.int8, copy=False)

X_test_local = X_test

n_jobs = min(4, os.cpu_count() or 1)

for fold, (idx_train, idx_valid) in enumerate(skf.split(X_idx, y_np)):
    X_train, y_train = X.iloc[idx_train], y_np[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y_np[idx_valid]

    model = XGBClassifier(
        **params,
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

    test_preds = model.predict_proba(X_test_local)[:, 1]
    preds.append(test_preds)

    print("fold : ", fold, "score : ", score)



## --- ERROR in cell 28, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mXGBoostError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4152606169.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     31[0m     )
[1;32m     32[0m [0;34m[0m[0m
[0;32m---> 33[0;31m     model.fit(
[0m[1;32m     34[0m         [0mX_train[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     35[0m         [0my_train[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)[0m
[1;32m   1517[0m             )
[1;32m   1518[0m [0;34m[0m[0m
[0;32m-> 1519[0;31m             self._Booster = train(
[0m[1;32m   1520[0m                 [0mparams[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1521[0m                 [0mtrain_dmatrix[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/training.py[0m in [0;36mtrain[0;34m(params, dtrain, num_boost_round, evals, obj, feval, maximize, early_stopping_rounds, evals_result, verbose_eval, xgb_model, callbacks, custom_metric)[0m
[1;32m    179[0m         [0;32mif[0m [0mcb_container[0m[0;34m.[0m[0mbefore_iteration[0m[0;34m([0m[0mbst[0m[0;34m,[0m [0mi[0m[0;34m,[0m [0mdtrain[0m[0;34m,[0m [0mevals[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    180[0m             [0;32mbreak[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 181[0;31m         [0mbst[0m[0;34m.[0m[0mupdate[0m[0;34m([0m[0mdtrain[0m[0;34m,[0m [0mi[0m[0;34m,[0m [0mobj[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    182[0m         [0;32mif[0m [0mcb_container[0m[0;34m.[0m[0mafter_iteration[0m[0;34m([0m[0mbst[0m[0;34m,[0m [0mi[0m[0;34m,[0m [0mdtrain[0m[0;34m,[0m [0mevals[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    183[0m             [0;32mbreak[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36mupdate[0;34m(self, dtrain, iteration, fobj)[0m
[1;32m   2048[0m [0;34m[0m[0m
[1;32m   2049[0m         [0;32mif[0m [0mfobj[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2050[0;31m             _check_call(
[0m[1;32m   2051[0m                 _LIB.XGBoosterUpdateOneIter(
[1;32m   2052[0m                     [0mself[0m[0;34m.[0m[0mhandle[0m[0;34m,[0m [0mctypes[0m[0;34m.[0m[0mc_int[0m[0;34m([0m[0miteration[0m[0;34m)[0m[0;34m,[0m [0mdtrain[0m[0;34m.[0m[0mhandle[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m_check_call[0;34m(ret)[0m
[1;32m    280[0m     """
[1;32m    281[0m     [0;32mif[0m [0mret[0m [0;34m!=[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 282[0;31m         [0;32mraise[0m [0mXGBoostError[0m[0;34m([0m[0mpy_str[0m[0;34m([0m[0m_LIB[0m[0;34m.[0m[0mXGBGetLastError[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    283[0m [0;34m[0m[0m
[1;32m    284[0m [0;34m[0m[0m

[0;31mXGBoostError[0m: [09:11:15] /workspace/src/tree/updater_gpu_hist.cu:781: Exception in gpu_hist: [09:11:15] /workspace/src/tree/updater_gpu_hist.cu:787: Check failed: ctx_->gpu_id >= 0 (-1 vs. 0) : Must have at least one device
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fff832f0f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb3e95a) [0x7fff8330795a]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb483cd) [0x7fff833113cd]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fff82c29c79]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fff82c2a76c]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fff82c8e4f7]
  [bt] (6) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fff8292aef0]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (8) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]



Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fff832f0f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb485c9) [0x7fff833115c9]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fff82c29c79]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fff82c2a76c]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fff82c8e4f7]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fff8292aef0]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]



## === cell 29
print(scores)
