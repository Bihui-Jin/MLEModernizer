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

No external packages required in the script and installed.

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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import optuna


## === cell 2
train = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv")


## === cell 3
print(train.shape); print(test.shape) 


## === cell 4
from sklearn import model_selection

train["kfold"] = -1

kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=102)
for fold, (train_indicies, valid_indicies) in enumerate(kf.split(X=train)):
    train.loc[valid_indicies, "kfold"] = fold


## === cell 5
train.head()


## === cell 6
df_tr = train[['f_27', 'target', 'kfold']]
df_te = test[['f_27']]

print(df_tr.shape); print(df_te.shape) 


## === cell 7
pd.crosstab(index=df_tr['target'], columns=df_tr['kfold'])


## === cell 8
df_tr.dtypes


## === cell 9
df_tr.f_27


## === cell 10
def count_alpha(df):
    for x in string.ascii_uppercase[:20]:
        df[f'count_{x}'] = df['f_27'].str.count(x)
    
    df = df.drop('f_27', 1)

    return(df)


## === cell 11
def count_alpha(df):
    for x in string.ascii_uppercase[:20]:
        df[f"count_{x}"] = df["f_27"].str.count(x)

    df = df.drop("f_27", axis=1)

    return df


## === cell 12
print(df_tr.shape)
print(df_te.shape)


## === cell 13
df_tr.head()


## === cell 14
 use_feature = [c for c in df_tr.columns if c not in ("target", "kfold")]


## === cell 15
def run(trial):
    fold = 0
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 100.0)
    reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 100.0)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 7)

    xtrain = df_tr[df_tr.kfold != fold].reset_index(drop=True)
    xvalid = df_tr[df_tr.kfold == fold].reset_index(drop=True)

    ytrain = xtrain.target
    yvalid = xvalid.target

    xtrain = xtrain[use_feature]
    xvalid = xvalid[use_feature]

    model = XGBClassifier(
        random_state=42,
        tree_method="gpu_hist",
        gpu_id=1,
        predictor="gpu_predictor",
        n_estimators=7000,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
    )
    model.fit(xtrain, ytrain, early_stopping_rounds=300, eval_set=[(xvalid, yvalid)], verbose=1000)
    preds_valid = model.predict(xvalid)
    AUC = roc_auc_score(yvalid, preds_valid)
    return AUC


## === cell 16
study = optuna.create_study(direction="maximize")
study.optimize(run, n_trials=5)


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/851819958.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mstudy[0m [0;34m=[0m [0moptuna[0m[0;34m.[0m[0mcreate_study[0m[0;34m([0m[0mdirection[0m[0;34m=[0m[0;34m"maximize"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mstudy[0m[0;34m.[0m[0moptimize[0m[0;34m([0m[0mrun[0m[0;34m,[0m [0mn_trials[0m[0;34m=[0m[0;36m5[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/study.py[0m in [0;36moptimize[0;34m(self, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)[0m
[1;32m    488[0m                 [0mIf[0m [0mnested[0m [0minvocation[0m [0mof[0m [0mthis[0m [0mmethod[0m [0moccurs[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    489[0m         """
[0;32m--> 490[0;31m         _optimize(
[0m[1;32m    491[0m             [0mstudy[0m[0;34m=[0m[0mself[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    492[0m             [0mfunc[0m[0;34m=[0m[0mfunc[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py[0m in [0;36m_optimize[0;34m(study, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)[0m
[1;32m     61[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     62[0m         [0;32mif[0m [0mn_jobs[0m [0;34m==[0m [0;36m1[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 63[0;31m             _optimize_sequential(
[0m[1;32m     64[0m                 [0mstudy[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     65[0m                 [0mfunc[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py[0m in [0;36m_optimize_sequential[0;34m(study, func, n_trials, timeout, catch, callbacks, gc_after_trial, reseed_sampler_rng, time_start, progress_bar)[0m
[1;32m    158[0m [0;34m[0m[0m
[1;32m    159[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 160[0;31m             [0mfrozen_trial_id[0m [0;34m=[0m [0m_run_trial[0m[0;34m([0m[0mstudy[0m[0;34m,[0m [0mfunc[0m[0;34m,[0m [0mcatch[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    161[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    162[0m             [0;31m# The following line mitigates memory problems that can be occurred in some[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py[0m in [0;36m_run_trial[0;34m(study, func, catch)[0m
[1;32m    256[0m         [0;32mand[0m [0;32mnot[0m [0misinstance[0m[0;34m([0m[0mfunc_err[0m[0;34m,[0m [0mcatch[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    257[0m     ):
[0;32m--> 258[0;31m         [0;32mraise[0m [0mfunc_err[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    259[0m     [0;32mreturn[0m [0mtrial[0m[0;34m.[0m[0m_trial_id[0m[0;34m[0m[0;34m[0m[0m
[1;32m    260[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py[0m in [0;36m_run_trial[0;34m(study, func, catch)[0m
[1;32m    199[0m     [0;32mwith[0m [0mget_heartbeat_thread[0m[0;34m([0m[0mtrial[0m[0;34m.[0m[0m_trial_id[0m[0;34m,[0m [0mstudy[0m[0;34m.[0m[0m_storage[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    200[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 201[0;31m             [0mvalue_or_values[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0mtrial[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    202[0m         [0;32mexcept[0m [0mexceptions[0m[0;34m.[0m[0mTrialPruned[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    203[0m             [0;31m# TODO(mamu): Handle multi-objective cases.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1605920172.py[0m in [0;36mrun[0;34m(trial)[0m
[1;32m     30[0m         [0mmax_depth[0m[0;34m=[0m[0mmax_depth[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     31[0m     )
[0;32m---> 32[0;31m     [0mmodel[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mxtrain[0m[0;34m,[0m [0mytrain[0m[0;34m,[0m [0mearly_stopping_rounds[0m[0;34m=[0m[0;36m300[0m[0;34m,[0m [0meval_set[0m[0;34m=[0m[0;34m[[0m[0;34m([0m[0mxvalid[0m[0;34m,[0m [0myvalid[0m[0;34m)[0m[0;34m][0m[0;34m,[0m [0mverbose[0m[0;34m=[0m[0;36m1000[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     33[0m     [0mpreds_valid[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mxvalid[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     34[0m     [0mAUC[0m [0;34m=[0m [0mroc_auc_score[0m[0;34m([0m[0myvalid[0m[0;34m,[0m [0mpreds_valid[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m   1586[0m             [0mctypes[0m[0;34m.[0m[0mbyref[0m[0;34m([0m[0mhandle[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1587[0m         )
[0;32m-> 1588[0;31m         [0mit[0m[0;34m.[0m[0mreraise[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1589[0m         [0;31m# delay check_call to throw intermediate exception first[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1590[0m         [0m_check_call[0m[0;34m([0m[0mret[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36mreraise[0;34m(self)[0m
[1;32m    574[0m             [0mexc[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_exception[0m[0;34m[0m[0;34m[0m[0m
[1;32m    575[0m             [0mself[0m[0;34m.[0m[0m_exception[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 576[0;31m             [0;32mraise[0m [0mexc[0m  [0;31m# pylint: disable=raising-bad-type[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    577[0m [0;34m[0m[0m
[1;32m    578[0m     [0;32mdef[0m [0m__del__[0m[0;34m([0m[0mself[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m_handle_exception[0;34m(self, fn, dft_ret)[0m
[1;32m    555[0m [0;34m[0m[0m
[1;32m    556[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 557[0;31m             [0;32mreturn[0m [0mfn[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    558[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me[0m[0;34m:[0m  [0;31m# pylint: disable=broad-except[0m[0;34m[0m[0;34m[0m[0m
[1;32m    559[0m             [0;31m# Defer the exception in order to return 0 and stop the iteration.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m<lambda>[0;34m()[0m
[1;32m    639[0m [0;34m[0m[0m
[1;32m    640[0m         [0;31m# pylint: disable=not-callable[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 641[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_handle_exception[0m[0;34m([0m[0;32mlambda[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0mnext[0m[0;34m([0m[0minput_data[0m[0;34m)[0m[0;34m,[0m [0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    642[0m [0;34m[0m[0m
[1;32m    643[0m     [0;34m@[0m[0mabstractmethod[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/data.py[0m in [0;36mnext[0;34m(self, input_data)[0m
[1;32m   1278[0m             [0;32mreturn[0m [0;36m0[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1279[0m         [0mself[0m[0;34m.[0m[0mit[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1280[0;31m         [0minput_data[0m[0;34m([0m[0;34m**[0m[0mself[0m[0;34m.[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1281[0m         [0;32mreturn[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1282[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minput_data[0;34m(data, feature_names, feature_types, **kwargs)[0m
[1;32m    622[0m                 [0mnew[0m[0;34m,[0m [0mcat_codes[0m[0;34m,[0m [0mfeature_names[0m[0;34m,[0m [0mfeature_types[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_temporary_data[0m[0;34m[0m[0;34m[0m[0m
[1;32m    623[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 624[0;31m                 new, cat_codes, feature_names, feature_types = _proxy_transform(
[0m[1;32m    625[0m                     [0mdata[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    626[0m                     [0mfeature_names[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/data.py[0m in [0;36m_proxy_transform[0;34m(data, feature_names, feature_types, enable_categorical)[0m
[1;32m   1313[0m         [0mdata[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mDataFrame[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1314[0m     [0;32mif[0m [0m_is_pandas_df[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1315[0;31m         arr, feature_names, feature_types = _transform_pandas_df(
[0m[1;32m   1316[0m             [0mdata[0m[0;34m,[0m [0menable_categorical[0m[0;34m,[0m [0mfeature_names[0m[0;34m,[0m [0mfeature_types[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1317[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/data.py[0m in [0;36m_transform_pandas_df[0;34m(data, enable_categorical, feature_names, feature_types, meta, meta_type)[0m
[1;32m    488[0m             [0;32mor[0m [0mis_pa_ext_dtype[0m[0;34m([0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    489[0m         ):
[0;32m--> 490[0;31m             [0m_invalid_dataframe_dtype[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    491[0m         [0;32mif[0m [0mis_pa_ext_dtype[0m[0;34m([0m[0mdtype[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    492[0m             [0mpyarrow_extension[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/data.py[0m in [0;36m_invalid_dataframe_dtype[0;34m(data)[0m
[1;32m    306[0m     [0mtype_err[0m [0;34m=[0m [0;34m"DataFrame.dtypes for data must be int, float, bool or category."[0m[0;34m[0m[0;34m[0m[0m
[1;32m    307[0m     [0mmsg[0m [0;34m=[0m [0;34mf"""{type_err} {_ENABLE_CAT_ERR} {err}"""[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 308[0;31m     [0;32mraise[0m [0mValueError[0m[0;34m([0m[0mmsg[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    309[0m [0;34m[0m[0m
[1;32m    310[0m [0;34m[0m[0m

[0;31mValueError[0m: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, The experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:f_27: object

## === cell 17
final_predictions = []
scores = []

for fold in range(5):
    xtrain = df_tr[df_tr.kfold != fold].reset_index(drop=True)
    xvalid = df_tr[df_tr.kfold == fold].reset_index(drop=True)
    
    xtest = df_te[use_feature]
    
    ytrain = xtrain.target
    yvalid = xvalid.target
    
    xtrain = xtrain[use_feature]
    xvalid = xvalid[use_feature]
    
    params = study.best_params
    
    model = XGBClassifier(
        random_state=0, 
        tree_method='gpu_hist',
        predictor="gpu_predictor",
        n_estimators=5000,
         eval_metric = 'auc',
        **params
    )
    
    model.fit(xtrain, ytrain, early_stopping_rounds=300, eval_set=[(xvalid, yvalid)], verbose=1000)
    preds_valid = model.predict_proba(xvalid)
    test_preds = model.predict_proba(xtest)
    final_predictions.append(test_preds)
    ROC = roc_auc_score(yvalid, preds_valid[:,1])
    print(fold, ROC)
    scores.append(ROC)

print(np.mean(scores), np.std(scores))
