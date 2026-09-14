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
plotly==5.24.1
plotly-express==0.4.1
seaborn==0.12.2
sklearn-pandas==2.2.0
xgboost==2.0.3

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px


## === cell 2
data = pd.read_csv('../input/ventilator-pressure-prediction/train.csv')
data.head()


## === cell 3
data.shape


## === cell 4
data.isnull().sum()


## === cell 5
data.info()


## === cell 6
data.describe()


## === cell 7
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), cmap='cool')


## === cell 8
cat_col = []
num_col = []
for i in data.columns:
    if data[i].value_counts().count() > 10:
        num_col.append(i)
    else:
        cat_col.append(i)
print(f'categorical columns: {cat_col}')
print(f'numerical columns: {num_col}')


## === cell 9
fig, ax = plt.subplots(1,3,figsize=(12,5))
j=0
for i in cat_col:
    sns.countplot(data[i], palette='cool', ax=ax[j])
    j+=1
fig.suptitle('Countplot of Categorical Data')


## === cell 10
num_col = num_col[2:]
num_col


## === cell 11
fig, ax = plt.subplots(1,3, figsize=(18,5))
j=0
for i in num_col:
    sns.histplot(data[i], ax=ax[j])
    j+=1
fig.suptitle('Histplot of Numerical Data')


## === cell 12
fig, ax = plt.subplots(1,3, figsize=(18,5))
j=0
for i in num_col:
    sns.boxplot(data[i], ax=ax[j], palette='cool')
    j+=1
fig.suptitle('Boxplot of Numerical Data')


## === cell 13
train = data.copy()


## === cell 14
fig, ax = plt.subplots(1,2, figsize=(18,5))
train['u_in'] = np.where(train['u_in']>12, train['u_in'].mean(), train['u_in'])
sns.boxplot(train['u_in'], palette='cool', ax=ax[0])
sns.histplot(train['u_in'], ax=ax[1])


## === cell 15
fig, ax = plt.subplots(1,2, figsize=(12,5))
sns.distplot(train['pressure'], ax=ax[0])

train['pressure'] = np.sqrt(np.sqrt(train['pressure']))

sns.distplot(train['pressure'], ax=ax[1])


## === cell 16
train_R = pd.get_dummies(train['R'], prefix='R')
train_C = pd.get_dummies(train['C'], prefix='C')
train_u_out = pd.get_dummies(train['u_out'], prefix='u_out')


## === cell 17
train_R.head()


## === cell 18
train = pd.concat([train, train_R, train_C, train_u_out], axis=1)
train = train.drop(['R', 'C', 'u_out'], axis=1)
train.head()


## === cell 19
test_data = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')
test_data.head()


## === cell 20
fig, ax = plt.subplots(1,2, figsize=(18,5))
num_col = num_col[:2] ## Removing 'pressure' column.
test_data['u_in'] = np.where(test_data['u_in']>12, test_data['u_in'].mean(), test_data['u_in'])
sns.boxplot(test_data['u_in'], palette='cool', ax=ax[0])
sns.histplot(test_data['u_in'], ax=ax[1])


## === cell 21
test_data_R = pd.get_dummies(test_data['R'], prefix='R')
test_data_C = pd.get_dummies(test_data['C'], prefix='C')
test_data_u_out = pd.get_dummies(test_data['u_out'], prefix='u_out')


## === cell 22
test_data = pd.concat([test_data, test_data_R, test_data_C, test_data_u_out], axis=1)
test_data = test_data.drop(['R', 'C', 'u_out'], axis=1)
test_data.head()


## === cell 23
X_train = train.drop('pressure', axis=1)
y_train = train['pressure']


## === cell 24
from xgboost import XGBRegressor

xgb_params = {
    'n_estimators': 5000,
    'learning_rate': 0.1,
    'subsample': 0.95,
    'colsample_bytree': 0.11,
    'max_depth': 2,
    'booster': 'gbtree', 
    'reg_lambda': 66.1,
    'reg_alpha': 15.9,
    'random_state':42,
    'tree_method':'gpu_hist',
    'gpu_id':0,
    'predictor':'gpu_predictor'
}

model = XGBRegressor(**xgb_params)

model.fit(X_train,y_train)


## --- ERROR in cell 24, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mXGBoostError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3584466888.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     18[0m [0mmodel[0m [0;34m=[0m [0mXGBRegressor[0m[0;34m([0m[0;34m**[0m[0mxgb_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m [0;34m[0m[0m
[0;32m---> 20[0;31m [0mmodel[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX_train[0m[0;34m,[0m[0my_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)[0m
[1;32m   1053[0m         [0;32mwith[0m [0mconfig_context[0m[0;34m([0m[0mverbosity[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mverbosity[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1054[0m             [0mevals_result[0m[0;34m:[0m [0mTrainingCallback[0m[0;34m.[0m[0mEvalsLog[0m [0;34m=[0m [0;34m{[0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1055[0;31m             train_dmatrix, evals = _wrap_evaluation_matrices(
[0m[1;32m   1056[0m                 [0mmissing[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mmissing[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1057[0m                 [0mX[0m[0;34m=[0m[0mX[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m    631[0m             [0mself[0m[0;34m.[0m[0m_temporary_data[0m [0;34m=[0m [0;34m([0m[0mnew[0m[0;34m,[0m [0mcat_codes[0m[0;34m,[0m [0mfeature_names[0m[0;34m,[0m [0mfeature_types[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    632[0m             [0mdispatch_proxy_set_data[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mproxy[0m[0;34m,[0m [0mnew[0m[0;34m,[0m [0mcat_codes[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_allow_host[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 633[0;31m             self.proxy.set_info(
[0m[1;32m    634[0m                 [0mfeature_names[0m[0;34m=[0m[0mfeature_names[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    635[0m                 [0mfeature_types[0m[0;34m=[0m[0mfeature_types[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36mset_info[0;34m(self, label, weight, base_margin, group, qid, label_lower_bound, label_upper_bound, feature_names, feature_types, feature_weights)[0m
[1;32m    930[0m [0;34m[0m[0m
[1;32m    931[0m         [0;32mif[0m [0mlabel[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 932[0;31m             [0mself[0m[0;34m.[0m[0mset_label[0m[0;34m([0m[0mlabel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    933[0m         [0;32mif[0m [0mweight[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    934[0m             [0mself[0m[0;34m.[0m[0mset_weight[0m[0;34m([0m[0mweight[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36mset_label[0;34m(self, label)[0m
[1;32m   1068[0m         [0;32mfrom[0m [0;34m.[0m[0mdata[0m [0;32mimport[0m [0mdispatch_meta_backend[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1069[0m [0;34m[0m[0m
[0;32m-> 1070[0;31m         [0mdispatch_meta_backend[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mlabel[0m[0;34m,[0m [0;34m"label"[0m[0;34m,[0m [0;34m"float"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1071[0m [0;34m[0m[0m
[1;32m   1072[0m     [0;32mdef[0m [0mset_weight[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mweight[0m[0;34m:[0m [0mArrayLike[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/data.py[0m in [0;36mdispatch_meta_backend[0;34m(matrix, data, name, dtype)[0m
[1;32m   1223[0m         [0;32mreturn[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1224[0m     [0;32mif[0m [0m_is_pandas_series[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1225[0;31m         [0m_meta_from_pandas_series[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mname[0m[0;34m,[0m [0mdtype[0m[0;34m,[0m [0mhandle[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1226[0m         [0;32mreturn[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1227[0m     [0;32mif[0m [0m_is_dlpack[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/data.py[0m in [0;36m_meta_from_pandas_series[0;34m(data, name, dtype, handle)[0m
[1;32m    543[0m         [0mdata[0m [0;34m=[0m [0mdata[0m[0;34m.[0m[0mto_dense[0m[0;34m([0m[0;34m)[0m  [0;31m# type: ignore[0m[0;34m[0m[0;34m[0m[0m
[1;32m    544[0m     [0;32massert[0m [0mlen[0m[0;34m([0m[0mdata[0m[0;34m.[0m[0mshape[0m[0;34m)[0m [0;34m==[0m [0;36m1[0m [0;32mor[0m [0mdata[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m1[0m[0;34m][0m [0;34m==[0m [0;36m0[0m [0;32mor[0m [0mdata[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m1[0m[0;34m][0m [0;34m==[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 545[0;31m     [0m_meta_from_numpy[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mname[0m[0;34m,[0m [0mdtype[0m[0;34m,[0m [0mhandle[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    546[0m [0;34m[0m[0m
[1;32m    547[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/data.py[0m in [0;36m_meta_from_numpy[0;34m(data, field, dtype, handle)[0m
[1;32m   1157[0m         [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"Masked array is not supported."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1158[0m     [0minterface_str[0m [0;34m=[0m [0m_array_interface[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1159[0;31m     [0m_check_call[0m[0;34m([0m[0m_LIB[0m[0;34m.[0m[0mXGDMatrixSetInfoFromInterface[0m[0;34m([0m[0mhandle[0m[0;34m,[0m [0mc_str[0m[0;34m([0m[0mfield[0m[0;34m)[0m[0;34m,[0m [0minterface_str[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1160[0m [0;34m[0m[0m
[1;32m   1161[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m_check_call[0;34m(ret)[0m
[1;32m    280[0m     """
[1;32m    281[0m     [0;32mif[0m [0mret[0m [0;34m!=[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 282[0;31m         [0;32mraise[0m [0mXGBoostError[0m[0;34m([0m[0mpy_str[0m[0;34m([0m[0m_LIB[0m[0;34m.[0m[0mXGBGetLastError[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    283[0m [0;34m[0m[0m
[1;32m    284[0m [0;34m[0m[0m

[0;31mXGBoostError[0m: [02:17:27] /workspace/src/data/data.cc:507: Check failed: valid: Label contains NaN, infinity or a value too large.
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3588ca) [0x7ffea377a8ca]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x38a21d) [0x7ffea37ac21d]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x38ab51) [0x7ffea37acb51]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGDMatrixSetInfoFromInterface+0xb0) [0x7ffea35803a0]
  [bt] (4) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (5) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (6) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]
  [bt] (7) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7ffff63bbc8e]
  [bt] (8) /usr/bin/python3(_PyObject_MakeTpCall+0x27c) [0x52f85c]



## === cell 25
pred = model.predict(test_data)
