# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Given readings from several seismic sensors around a volcano, estimate how long it will be until the next eruption.

## Metric
Mean absolute error (MAE) between the predicted loss and the actual loss.

## Submission Format
For every id in the test set, you should predict the time until the next eruption. The file should contain a header and have the following format:

```
segment_id,time_to_eruption
1,1
2,2
3,3
etc.
```

## Data
### Dataset Description

#### Files
**train.csv** Metadata for the train files.

- `segment_id`: ID code for the data segment. Matches the name of the associated data file.
- `time_to_eruption`: The target value, the time until the next eruption.

**[train|test]/*.csv**: the data files. Each file contains ten minutes of logs from ten different sensors arrayed around a volcano. The readings have been normalized within each segment, in part to ensure that the readings fall within the range of int16 values. If you are using the Pandas library you may find that you still need to load the data as float32 due to the presence of some nulls.

# 2. Python version

3.9

# 3. Installed packages

catboost==1.2.8
geopandas==0.14.4
lightgbm==4.6.0
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
tqdm==4.67.1
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (70 lines)
            sample_submission.csv (445 lines)
            sample_submission.csv.zip (2.8 kB)
            test.zip (514.3 MB)
            train.csv (3988 lines)
            train.csv.zip (39.1 kB)
            train.zip (4.6 GB)
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
            test/
                1003520023.csv (60002 lines)
                1004346803.csv (60002 lines)
                ... and 442 other files
                test/
            train/
                1000015382.csv (60002 lines)
                1000554676.csv (60002 lines)
                ... and 3985 other files
                train/
        input/
            description.md (70 lines)
            sample_submission.csv (445 lines)
            sample_submission.csv.zip (2.8 kB)
            test.zip (514.3 MB)
            train.csv (3988 lines)
            train.csv.zip (39.1 kB)
            train.zip (4.6 GB)
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
            test/
                1003520023.csv (60002 lines)
                1004346803.csv (60002 lines)
                ... and 442 other files
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
            train/
                1000015382.csv (60002 lines)
                1000554676.csv (60002 lines)
                ... and 3985 other files
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
        working/
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
```

-> data/predict-volcanic-eruptions-ingv-oe/sample_submission.csv has 444 rows and 2 columns.
The columns are: segment_id, time_to_eruption

-> data/predict-volcanic-eruptions-ingv-oe/test/1003520023.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1004346803.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1007996426.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1009749143.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1016956864.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1024522044.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1028325789.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> (stopped after 10 files for performance)

# 5. Target score

11757378.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import NuSVR, SVR
from sklearn.linear_model import Ridge, RidgeCV

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
%matplotlib inline


## === cell 1
segment_csvs = glob.glob("../input/predict-volcanic-eruptions-ingv-oe/train/*")
len_segment_csvs = len(segment_csvs)
len_segment_csvs


## === cell 2
test_csvs = glob.glob("../input/predict-volcanic-eruptions-ingv-oe/test/*")
len_test_csvs = test_csvs
len(len_test_csvs)


## === cell 3
sample_submission = pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')


## === cell 4
len(sample_submission)


## === cell 5
sample


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/109366192.py in <cell line: 0>()
----> 1 sample

NameError: name 'sample' is not defined

## === cell 6
segment_csvs[0]


## === cell 7
train_1 = pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/train/2037160701.csv')


## === cell 8
train_1


## === cell 9
import matplotlib.pyplot as plt

def sensor_show(df):
    f, axes = plt.subplots(10, 1)
    f.set_size_inches((16, 8)) 
    f.tight_layout() 
    plt.subplots_adjust(bottom=-0.4)
    
    for i in range(1,11):
        axes[i-1].plot(df[f'sensor_{i}'].values)
        axes[i-1].set_title('Sensor_'+str(i))
        axes[i-1].set_xlabel('time')


## === cell 10
sensor_show(train_1)


## === cell 11
!pip install datatable==0.11.0 > /dev/null

import datatable as dt


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/78358694.py in <cell line: 0>()
      2 get_ipython().system('pip install datatable==0.11.0 > /dev/null')
      3 
----> 4 import datatable as dt

ModuleNotFoundError: No module named 'datatable'

## === cell 12
df = pd.read_csv(segment_csvs[0])
df_mean = pd.DataFrame(df.mean()).T
df_mean['id'] = segment_csvs[0].split('/')[-1].split('.')[0]
segment_csvs.remove(segment_csvs[0])

for csv in tqdm(segment_csvs):
    seg_name = csv.split('/')[-1].split('.')[0]
    df = dt.fread(csv).to_jay('train.jay')
    df = dt.fread('train.jay')
    df_ = pd.DataFrame(df.mean().to_pandas()) # df.mean() for datatable
    df_['id'] = csv.split('/')[-1].split('.')[0]
    df_mean = pd.concat([df_mean,df_])
    del df
    
df_mean.head(3)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3575915135.py in <cell line: 0>()
      6 for csv in tqdm(segment_csvs):
      7     seg_name = csv.split('/')[-1].split('.')[0]
----> 8     df = dt.fread(csv).to_jay('train.jay')
      9     df = dt.fread('train.jay')
     10     df_ = pd.DataFrame(df.mean().to_pandas()) # df.mean() for datatable

NameError: name 'dt' is not defined

## === cell 13
df_train = pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/train.csv')
df_train.head(2)


## === cell 14
df_mean['id'] = df_mean['id'].astype('int64')


## === cell 15
df_mean = df_mean.join(df_train.set_index('segment_id'), on='id')
df_mean.head(3)


## === cell 16
X_train = df_mean.drop(['id','time_to_eruption'],axis=1)
y_train = df_mean['time_to_eruption']
X_train = X_train.fillna(X_train.mean())
del df_mean


## === cell 17
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)


## === cell 18
X_train_scaled


## === cell 19
df = pd.read_csv(test_csvs[0])
df_mean_test = pd.DataFrame(df.mean()).T
df_mean_test['id'] = test_csvs[0].split('/')[-1].split('.')[0]
test_csvs.remove(test_csvs[0])

for csv in tqdm(test_csvs):
    df = dt.fread(csv).to_jay('test.jay')
    df = dt.fread("test.jay")
    df_ = pd.DataFrame(df.mean().to_pandas()) # df.mean() for datatable
    df_['id'] = csv.split('/')[-1].split('.')[0]
    df_mean_test = pd.concat([df_mean_test,df_])
    del df
df_mean_test.head(3)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2693402714.py in <cell line: 0>()
      5 
      6 for csv in tqdm(test_csvs):
----> 7     df = dt.fread(csv).to_jay('test.jay')
      8     df = dt.fread("test.jay")
      9     df_ = pd.DataFrame(df.mean().to_pandas()) # df.mean() for datatable

NameError: name 'dt' is not defined

## === cell 20
X_test = df_mean_test.fillna(df_mean_test.mean())
X_test = X_test.drop(['id'],axis=1)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4203903050.py in <cell line: 0>()
----> 1 X_test = df_mean_test.fillna(df_mean_test.mean())
      2 X_test = X_test.drop(['id'],axis=1)
      3 #del df_mean_test

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in mean(self, axis, skipna, numeric_only, **kwargs)
  11691         **kwargs,
  11692     ):
> 11693         result = super().mean(axis, skipna, numeric_only, **kwargs)
  11694         if isinstance(result, Series):
  11695             result = result.__finalize__(self, method="mean")

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in mean(self, axis, skipna, numeric_only, **kwargs)
  12418         **kwargs,
  12419     ) -> Series | float:
> 12420         return self._stat_function(
  12421             "mean", nanops.nanmean, axis, skipna, numeric_only, **kwargs
  12422         )

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _stat_function(self, name, func, axis, skipna, numeric_only, **kwargs)
  12375         validate_bool_kwarg(skipna, "skipna", none_allowed=False)
  12376 
> 12377         return self._reduce(
  12378             func, name=name, axis=axis, skipna=skipna, numeric_only=numeric_only
  12379         )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _reduce(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)
  11560         # After possibly _get_data and transposing, we are now in the
  11561         #  simple case where we can use BlockManager.reduce
> 11562         res = df._mgr.reduce(blk_func)
  11563         out = df._constructor_from_mgr(res, axes=res.axes).iloc[0]
  11564         if out_dtype is not None and out.dtype != "boolean":

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in reduce(self, func)
   1498         res_blocks: list[Block] = []
   1499         for blk in self.blocks:
-> 1500             nbs = blk.reduce(func)
   1501             res_blocks.extend(nbs)
   1502 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in reduce(self, func)
    402         assert self.ndim == 2
    403 
--> 404         result = func(self.values)
    405 
    406         if self.values.ndim == 1:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in blk_func(values, axis)
  11479                     return np.array([result])
  11480             else:
> 11481                 return op(values, axis=axis, skipna=skipna, **kwds)
  11482 
  11483         def _get_data() -> DataFrame:

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in f(values, axis, skipna, **kwds)
    145                     result = alt(values, axis=axis, skipna=skipna, **kwds)
    146             else:
--> 147                 result = alt(values, axis=axis, skipna=skipna, **kwds)
    148 
    149             return result

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in new_func(values, axis, skipna, mask, **kwargs)
    402             mask = isna(values)
    403 
--> 404         result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
    405 
    406         if datetimelike:

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in nanmean(values, axis, skipna, mask)
    718     count = _get_counts(values.shape, mask, axis, dtype=dtype_count)
    719     the_sum = values.sum(axis, dtype=dtype_sum)
--> 720     the_sum = _ensure_numeric(the_sum)
    721 
    722     if axis is not None and getattr(the_sum, "ndim", False):

/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py in _ensure_numeric(x)
   1684             if inferred in ["string", "mixed"]:
   1685                 # GH#44008, GH#36703 avoid casting e.g. strings to numeric
-> 1686                 raise TypeError(f"Could not convert {x} to numeric")
   1687             try:
   1688                 x = x.astype(np.complex128)

TypeError: Could not convert ['194039587'] to numeric

## === cell 21
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1195248653.py in <cell line: 0>()
----> 1 X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

NameError: name 'X_test' is not defined

## === cell 22
X_test_scaled


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3127281192.py in <cell line: 0>()
----> 1 X_test_scaled

NameError: name 'X_test_scaled' is not defined

## === cell 23
train_set = pd.DataFrame()
train_set['segment_id'] = df_train.segment_id
train_set = train_set.set_index('segment_id')
train_set = pd.merge(train_set.reset_index(), df_train, on=['segment_id'], how='left').set_index('segment_id')

y_train = train_set['time_to_eruption']


## === cell 24
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)


## === cell 25
def train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_train, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):

    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                    verbose=10000, early_stopping_rounds=200)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_train.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_train.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_train.columns), ntree_limit=model.best_ntree_limit)
            
        if model_type == 'rcv':
            model = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), scoring='neg_mean_absolute_error', cv=3)
            model.fit(X_train, y_train)
            print(model.alpha_)

            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = mean_absolute_error(y_valid, y_pred_valid)
            print(f'Fold {fold_n}. MAE: {score:.4f}.')
            print('')
            
            y_pred = model.predict(X_test).reshape(-1,)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = mean_absolute_error(y_valid, y_pred_valid)
            print(f'Fold {fold_n}. MAE: {score:.4f}.')
            print('')
            
            y_pred = model.predict(X_test).reshape(-1,)
        
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric='MAE', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred    
        
        if model_type == 'lgb':
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
        
            return oof, prediction, feature_importance
        return oof, prediction
    
    else:
        return oof, prediction


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1473066671.py in <cell line: 0>()
----> 1 def train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_train, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):
      2 
      3     oof = np.zeros(len(X))
      4     prediction = np.zeros(len(X_test))
      5     scores = []

NameError: name 'X_test_scaled' is not defined

## === cell 26
import lightgbm as lgb


## === cell 27
params = {'num_leaves': 54,
         'min_data_in_leaf': 79,
         'objective': 'huber',
         'max_depth': -1,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "bagging_freq": 3,
         "bagging_fraction": 0.8126672064208567,
         "bagging_seed": 11,
         "metric": 'mae',
         "verbosity": -1,
         'reg_alpha': 1.1302650970728192,
         'reg_lambda': 0.3603427518866501
         }
oof_lgb, prediction_lgb, feature_importance = train_model(params=params, model_type='lgb', plot_feature_importance=True)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1200207443.py in <cell line: 0>()
     14          'reg_lambda': 0.3603427518866501
     15          }
---> 16 oof_lgb, prediction_lgb, feature_importance = train_model(params=params, model_type='lgb', plot_feature_importance=True)

NameError: name 'train_model' is not defined

## === cell 28
xgb_params = {'eta': 0.03, 'max_depth': 10, 'subsample': 0.85, #'colsample_bytree': 0.8, 
          'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': True, 'nthread': 4}
oof_xgb, prediction_xgb = train_model(params=xgb_params, model_type='xgb')


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2130110105.py in <cell line: 0>()
      1 xgb_params = {'eta': 0.03, 'max_depth': 10, 'subsample': 0.85, #'colsample_bytree': 0.8, 
      2           'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': True, 'nthread': 4}
----> 3 oof_xgb, prediction_xgb = train_model(params=xgb_params, model_type='xgb')

NameError: name 'train_model' is not defined

## === cell 29
model = NuSVR(gamma='scale', nu=0.75, C=10.0)
oof_svr, prediction_svr = train_model(params=None, model_type='sklearn', model=model)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4118201189.py in <cell line: 0>()
      1 model = NuSVR(gamma='scale', nu=0.75, C=10.0)
----> 2 oof_svr, prediction_svr = train_model(params=None, model_type='sklearn', model=model)

NameError: name 'train_model' is not defined

## === cell 30
plt.figure(figsize=(16, 8))
plt.plot(oof_lgb, color='b', label='lgb')
plt.plot(oof_xgb, color='teal', label='xgb')
plt.plot(oof_svr, color='red', label='svr')
plt.plot((oof_lgb + oof_xgb + oof_svr) / 3, color='gold', label='blend')
plt.legend();
plt.title('Predictions');


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/496966360.py in <cell line: 0>()
      1 plt.figure(figsize=(16, 8))
----> 2 plt.plot(oof_lgb, color='b', label='lgb')
      3 plt.plot(oof_xgb, color='teal', label='xgb')
      4 plt.plot(oof_svr, color='red', label='svr')
      5 plt.plot((oof_lgb + oof_xgb + oof_svr) / 3, color='gold', label='blend')

NameError: name 'oof_lgb' is not defined

## === cell 31
prediction_lgb[:10], prediction_xgb[:10], prediction_svr[:10]


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3128396531.py in <cell line: 0>()
----> 1 prediction_lgb[:10], prediction_xgb[:10], prediction_svr[:10]

NameError: name 'prediction_lgb' is not defined

## === cell 32
submission = pd.DataFrame()
submission['segment_id'] = sample_submission.segment_id
submission['time_to_eruption'] = (prediction_lgb + prediction_xgb + prediction_svr) / 3
print(submission.head())


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4080101352.py in <cell line: 0>()
      1 submission = pd.DataFrame()
      2 submission['segment_id'] = sample_submission.segment_id
----> 3 submission['time_to_eruption'] = (prediction_lgb + prediction_xgb + prediction_svr) / 3
      4 print(submission.head())

NameError: name 'prediction_lgb' is not defined

## === cell 33
submission.to_csv('submission.csv', index=False)


## --- ERROR in outputing the csv:
Invalid submission: Target column time_to_eruption not found in submissions DataFrame.
