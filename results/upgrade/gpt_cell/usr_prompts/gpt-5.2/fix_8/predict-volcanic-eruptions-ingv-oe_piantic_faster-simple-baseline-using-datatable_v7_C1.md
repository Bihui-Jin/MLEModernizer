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

3.9

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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
sample_submission


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
import pandas as _pd


class _DTMeanResult:
    def __init__(self, s: _pd.Series):
        self._s = s

    def to_pandas(self):
        return self._s.to_frame()


class _DTFrame:
    def __init__(self, df: _pd.DataFrame):
        self._df = df

    def to_jay(self, path: str):
        _DTShim._jay_cache[path] = self._df.copy()
        return path

    def mean(self):
        return _DTMeanResult(self._df.mean(numeric_only=True))


class _DTShim:
    _jay_cache = {}

    @staticmethod
    def fread(path: str):
        if path in _DTShim._jay_cache:
            return _DTFrame(_DTShim._jay_cache[path].copy())
        return _DTFrame(_pd.read_csv(path))


dt = _DTShim


## === cell 12
import os

segment_csvs = [
    p for p in segment_csvs if os.path.isfile(p) and p.lower().endswith(".csv")
]

df = pd.read_csv(segment_csvs[0])
df_mean = pd.DataFrame(df.mean()).T
df_mean["id"] = segment_csvs[0].split("/")[-1].split(".")[0]
segment_csvs.remove(segment_csvs[0])

for csv in tqdm(segment_csvs):
    seg_name = csv.split("/")[-1].split(".")[0]
    df = dt.fread(csv).to_jay("train.jay")
    df = dt.fread("train.jay")
    df_ = pd.DataFrame(df.mean().to_pandas())  # df.mean() for datatable
    df_["id"] = csv.split("/")[-1].split(".")[0]
    df_mean = pd.concat([df_mean, df_])
    del df

df_mean.head(3)


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
X_train.columns = X_train.columns.astype(str)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)


## === cell 18
X_train_scaled


## === cell 19
import os

test_csvs = [p for p in test_csvs if os.path.isfile(p) and p.lower().endswith(".csv")]
test_csvs = sorted(test_csvs)

df = pd.read_csv(test_csvs[0])
df_mean_test = pd.DataFrame(df.mean()).T
df_mean_test["id"] = test_csvs[0].split("/")[-1].split(".")[0]
test_csvs.remove(test_csvs[0])

for csv in tqdm(test_csvs):
    df = dt.fread(csv).to_jay("test.jay")
    df = dt.fread("test.jay")
    df_ = pd.DataFrame(df.mean().to_pandas())  # df.mean() for datatable
    df_["id"] = csv.split("/")[-1].split(".")[0]
    df_mean_test = pd.concat([df_mean_test, df_])
    del df
df_mean_test.head(3)


## === cell 20
X_test = df_mean_test.fillna(df_mean_test.mean(numeric_only=True))
X_test = X_test.drop(["id"], axis=1)


## === cell 21
X_test.columns = X_test.columns.astype(str)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


## === cell 22
X_test_scaled


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
[0;31m---------------------------------------------------------------------------[0m
[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_get_list_axis[0;34m(self, key, axis)[0m
[1;32m   1713[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1714[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m.[0m[0m_take_with_is_copy[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1715[0m         [0;32mexcept[0m [0mIndexError[0m [0;32mas[0m [0merr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36m_take_with_is_copy[0;34m(self, indices, axis)[0m
[1;32m   4152[0m         """
[0;32m-> 4153[0;31m         [0mresult[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mtake[0m[0;34m([0m[0mindices[0m[0;34m=[0m[0mindices[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4154[0m         [0;31m# Maybe set copy if we didn't actually change the index.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36mtake[0;34m(self, indices, axis, **kwargs)[0m
[1;32m   4132[0m [0;34m[0m[0m
[0;32m-> 4133[0;31m         new_data = self._mgr.take(
[0m[1;32m   4134[0m             [0mindices[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py[0m in [0;36mtake[0;34m(self, indexer, axis, verify)[0m
[1;32m    890[0m         [0mn[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0maxis[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 891[0;31m         [0mindexer[0m [0;34m=[0m [0mmaybe_convert_indices[0m[0;34m([0m[0mindexer[0m[0;34m,[0m [0mn[0m[0;34m,[0m [0mverify[0m[0;34m=[0m[0mverify[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    892[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexers/utils.py[0m in [0;36mmaybe_convert_indices[0;34m(indices, n, verify)[0m
[1;32m    281[0m         [0;32mif[0m [0mmask[0m[0;34m.[0m[0many[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 282[0;31m             [0;32mraise[0m [0mIndexError[0m[0;34m([0m[0;34m"indices are out-of-bounds"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    283[0m     [0;32mreturn[0m [0mindices[0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexError[0m: indices are out-of-bounds

The above exception was the direct cause of the following exception:

[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1200207443.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     14[0m          [0;34m'reg_lambda'[0m[0;34m:[0m [0;36m0.3603427518866501[0m[0;34m[0m[0;34m[0m[0m
[1;32m     15[0m          }
[0;32m---> 16[0;31m [0moof_lgb[0m[0;34m,[0m [0mprediction_lgb[0m[0;34m,[0m [0mfeature_importance[0m [0;34m=[0m [0mtrain_model[0m[0;34m([0m[0mparams[0m[0;34m=[0m[0mparams[0m[0;34m,[0m [0mmodel_type[0m[0;34m=[0m[0;34m'lgb'[0m[0;34m,[0m [0mplot_feature_importance[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/1473066671.py[0m in [0;36mtrain_model[0;34m(X, X_test, y, params, folds, model_type, plot_feature_importance, model)[0m
[1;32m      8[0m         [0mprint[0m[0;34m([0m[0;34m'Fold'[0m[0;34m,[0m [0mfold_n[0m[0;34m,[0m [0;34m'started at'[0m[0;34m,[0m [0mtime[0m[0;34m.[0m[0mctime[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m         [0mX_train[0m[0;34m,[0m [0mX_valid[0m [0;34m=[0m [0mX[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0mtrain_index[0m[0;34m][0m[0;34m,[0m [0mX[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0mvalid_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m         [0my_train[0m[0;34m,[0m [0my_valid[0m [0;34m=[0m [0my[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0mtrain_index[0m[0;34m][0m[0;34m,[0m [0my[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0mvalid_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m [0;34m[0m[0m
[1;32m     12[0m         [0;32mif[0m [0mmodel_type[0m [0;34m==[0m [0;34m'lgb'[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m__getitem__[0;34m(self, key)[0m
[1;32m   1189[0m             [0mmaybe_callable[0m [0;34m=[0m [0mcom[0m[0;34m.[0m[0mapply_if_callable[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1190[0m             [0mmaybe_callable[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_check_deprecated_callable_usage[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mmaybe_callable[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1191[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_axis[0m[0;34m([0m[0mmaybe_callable[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1192[0m [0;34m[0m[0m
[1;32m   1193[0m     [0;32mdef[0m [0m_is_scalar_access[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mkey[0m[0;34m:[0m [0mtuple[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_getitem_axis[0;34m(self, key, axis)[0m
[1;32m   1741[0m         [0;31m# a list of integers[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1742[0m         [0;32melif[0m [0mis_list_like_indexer[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1743[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_get_list_axis[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1744[0m [0;34m[0m[0m
[1;32m   1745[0m         [0;31m# a single integer[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_get_list_axis[0;34m(self, key, axis)[0m
[1;32m   1715[0m         [0;32mexcept[0m [0mIndexError[0m [0;32mas[0m [0merr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1716[0m             [0;31m# re-raise with different error message, e.g. test_getitem_ndarray_3d[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1717[0;31m             [0;32mraise[0m [0mIndexError[0m[0;34m([0m[0;34m"positional indexers are out-of-bounds"[0m[0;34m)[0m [0;32mfrom[0m [0merr[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1718[0m [0;34m[0m[0m
[1;32m   1719[0m     [0;32mdef[0m [0m_getitem_axis[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mkey[0m[0;34m,[0m [0maxis[0m[0;34m:[0m [0mAxisInt[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexError[0m: positional indexers are out-of-bounds

## === cell 28
xgb_params = {'eta': 0.03, 'max_depth': 10, 'subsample': 0.85, #'colsample_bytree': 0.8, 
          'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': True, 'nthread': 4}
oof_xgb, prediction_xgb = train_model(params=xgb_params, model_type='xgb')
