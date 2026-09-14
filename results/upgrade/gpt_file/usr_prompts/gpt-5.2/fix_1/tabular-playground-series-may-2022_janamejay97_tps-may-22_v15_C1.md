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

0.98462

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import string
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from xgboost  import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
print('Train data:')
train = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
print('Shape of train data: ' + str(train.shape))
print()
train.head(5)


## === cell 2
print('Test data:')
test = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')
print('Shape of test data: ' + str(test.shape))
print()
test.head(5)


## === cell 3
print(train['target'].value_counts())


## === cell 4
train.info()


## === cell 5
train.isnull().sum()


## === cell 6
train.describe()


## === cell 7
train.nunique().sort_values(ascending = True)


## === cell 8
categorical_feats = []

for col in train.columns:
    if train[col].dtype == 'int64':
        categorical_feats.append(col)

train[categorical_feats].sample(10)


## === cell 9
for feature in categorical_feats:
    print('Possible values for ' + feature)
    print(train[feature].unique())
    print()


## === cell 10
fig, ax = plt.subplots(figsize = (50, 50))
sns.heatmap(train[categorical_feats].corr(), annot = True, fmt = '.2f', ax = ax)
plt.show()


## === cell 11
float_feats = []
for col in train.columns:
    if train[col].dtype == 'float64':
        float_feats.append(col)

print(float_feats)


## === cell 12
fig, axs = plt.subplots(4, 4, figsize=(18, 18))
axs = axs.ravel()

for ind, ax in enumerate(axs):
    ax.hist(train[float_feats[ind]], density = True, bins = 100)
    ax.set_title(f'Train: {float_feats[ind]}, Std. dev: {train[float_feats[ind]].std():.1f}')
plt.show()    


## === cell 13
fig, axs = plt.subplots(4, 4, figsize=(14, 24))
axs = axs.ravel()

for ind, ax in enumerate(axs):
    ax.boxplot(train[train.target == 0][float_feats[ind]], positions = [0], widths = 0.7)
    ax.boxplot(train[train.target == 1][float_feats[ind]], positions = [1], widths = 0.7)
    ax.set_title(f'{float_feats[ind]}')
plt.show()    


## === cell 14
fig, ax = plt.subplots(figsize = (50, 50))
sns.heatmap(train[float_feats + ['target']].corr(), annot = True, fmt = '.2f', ax = ax)
plt.show()


## === cell 16
train['f_27'].str.len().value_counts()


## === cell 17
print('No. of unique strings')
print(train['f_27'].nunique())
print()

print('Difference between train and test')
print(len(set(test['f_27']).difference(set(train['f_27']))))


## === cell 18
print('Top 20 frequent strings')

train.f_27.value_counts()[:20].sort_values().plot(kind = 'barh', figsize = (15, 15), colormap = 'Paired')


## === cell 19
for charind in range(10):
    print(f'Position {charind + 1}:')
    char_group = train.groupby(train['f_27'].str.get(charind))
    char_info = pd.DataFrame({'Length': char_group.size(), 'Prob': char_group.target.mean().round(2)})
    print(char_info)
    print()


## === cell 20
for ite in range(10):
    train['char_' + str(ite)] = train['f_27'].str.get(ite).apply(ord) - ord('A')
    test['char_' + str(ite)] = test['f_27'].str.get(ite).apply(ord) - ord('A')


## === cell 21
train['unique_letters'] = train['f_27'].apply(lambda s: len(set(s)))
test['unique_letters'] = test['f_27'].apply(lambda s: len(set(s)))


## === cell 22
exclude_feats = ['id', 'f_27', 'target']
features = [feature for feature in train.columns if feature not in exclude_feats]


## === cell 23
xgb_params = {
              'n_estimators'     : 8192,
              'min_child_weight' : 96,
              'max_depth'        : 6,
              'learning_rate'    : 0.15,
              'subsample'        : 0.95,
              'colsample_bytree' : 0.95,
              'reg_lambda'       : 1.50,
              'reg_alpha'        : 1.50,
              'gamma'            : 1.50,
              'max_bin'          : 512,
              'random_state'     : 46,
              'objective'        : 'binary:logistic',
              'tree_method'      : 'gpu_hist',
             }


## === cell 24
scores, predictions = [], []

kf = KFold(n_splits = 5)

for fold, (train_ind, cv_ind) in enumerate(kf.split(train)):
    print('Train fold ' + str(fold))
    
    X_train, y_train = train.iloc[train_ind][features], train.iloc[train_ind]['target']
    X_cv, y_cv = train.iloc[cv_ind][features], train.iloc[cv_ind]['target']

    mdl = XGBClassifier(**xgb_params)
    mdl.fit(X_train, y_train, eval_set = [(X_cv, y_cv)], eval_metric = ['auc'], early_stopping_rounds = 256, verbose = 0)

    y_cv_pred = mdl.predict_proba(X_cv.values)[:, 1]
    score = roc_auc_score(y_cv, y_cv_pred)

    scores.append(score)
    print(f"Fold {fold}, AUC = {score:.3f}")
    print((''))
    
    test_pred = mdl.predict_proba(test[features])[:, 1]
    predictions.append(test_pred)

print('AUC' + str(np.mean(scores)))


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/690037924.py in <cell line: 0>()
     10 
     11     mdl = XGBClassifier(**xgb_params)
---> 12     mdl.fit(X_train, y_train, eval_set = [(X_cv, y_cv)], eval_metric = ['auc'], early_stopping_rounds = 256, verbose = 0)
     13 
     14     y_cv_pred = mdl.predict_proba(X_cv.values)[:, 1]

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in fit(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)
   1517             )
   1518 
-> 1519             self._Booster = train(
   1520                 params,
   1521                 train_dmatrix,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/training.py in train(params, dtrain, num_boost_round, evals, obj, feval, maximize, early_stopping_rounds, evals_result, verbose_eval, xgb_model, callbacks, custom_metric)
    179         if cb_container.before_iteration(bst, i, dtrain, evals):
    180             break
--> 181         bst.update(dtrain, i, obj)
    182         if cb_container.after_iteration(bst, i, dtrain, evals):
    183             break

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in update(self, dtrain, iteration, fobj)
   2048 
   2049         if fobj is None:
-> 2050             _check_call(
   2051                 _LIB.XGBoosterUpdateOneIter(
   2052                     self.handle, ctypes.c_int(iteration), dtrain.handle

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [02:21:08] /workspace/src/tree/updater_gpu_hist.cu:781: Exception in gpu_hist: [02:21:08] /workspace/src/tree/updater_gpu_hist.cu:787: Check failed: ctx_->gpu_id >= 0 (-1 vs. 0) : Must have at least one device
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fff83b77f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb3e95a) [0x7fff83b8e95a]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb483cd) [0x7fff83b983cd]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fff834b0c79]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fff834b176c]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fff835154f7]
  [bt] (6) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fff831b1ef0]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (8) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]



Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fff83b77f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb485c9) [0x7fff83b985c9]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fff834b0c79]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fff834b176c]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fff835154f7]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fff831b1ef0]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]



## === cell 25
submission = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv')
submission.head(5)


## === cell 26
submission['target'] = np.array(predictions).mean(axis = 0)
submission.to_csv('submission.csv', index = False)


## === cell 27
submission.head(5)
