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

0.99579

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


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
%%time
import warnings
warnings.filterwarnings('ignore')


## === cell 2
%%time

DATA_ROWS = None
NROWS = 50
NCOLS = 15
BASE_PATH = '...'


## === cell 3
%%time
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', NCOLS) 
pd.set_option('display.max_rows', NROWS)


## === cell 4
%%time
trn_data = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
tst_data = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')

sub = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv')


## === cell 5
%%time
trn_data.shape


## === cell 6
%%time
trn_data.info()


## === cell 7
%%time
trn_data.head()


## === cell 8
%%time
trn_data.describe()


## === cell 9
%%time
trn_data.isnull().sum().sum()


## === cell 10
%%time
trn_data.isnull().sum()


## === cell 11
%%time
trn_data.nunique()


## === cell 12
trn_data.nunique().sort_values(ascending = True)


## === cell 13
%%time
categ_cols = ['f_29','f_30','f_13', 'f_18','f_17','f_14','f_11','f_10','f_09','f_15','f_07','f_12','f_16','f_08','f_27']
trn_data[categ_cols].sample(5)


## === cell 14
%%time
correlation = trn_data.corr()


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<timed exec> in <module>

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in corr(self, method, min_periods, numeric_only)
  11047         cols = data.columns
  11048         idx = cols.copy()
> 11049         mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
  11050 
  11051         if method == "pearson":

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in to_numpy(self, dtype, copy, na_value)
   1991         if dtype is not None:
   1992             dtype = np.dtype(dtype)
-> 1993         result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
   1994         if result.dtype is not dtype:
   1995             result = np.asarray(result, dtype=dtype)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in as_array(self, dtype, copy, na_value)
   1692                 arr.flags.writeable = False
   1693         else:
-> 1694             arr = self._interleave(dtype=dtype, na_value=na_value)
   1695             # The underlying data was copied within _interleave, so no need
   1696             # to further copy if copy=True or setting na_value

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in _interleave(self, dtype, na_value)
   1751             else:
   1752                 arr = blk.get_values(dtype)
-> 1753             result[rl.indexer] = arr
   1754             itemmask[rl.indexer] = 1
   1755 

ValueError: could not convert string to float: 'ACBADABECB'

## === cell 15
%%time
correlation


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed eval> in <module>

NameError: name 'correlation' is not defined

## === cell 16
%%time
correlation['target'].sort_values(ascending = False)[:5]


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed eval> in <module>

NameError: name 'correlation' is not defined

## === cell 17
%%time
correlation['target'].sort_values(ascending = True)[:5]


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed eval> in <module>

NameError: name 'correlation' is not defined

## === cell 18
%%time
trn_data['target'].value_counts()


## === cell 19
%%time
trn_data['target'].describe()


## === cell 20
%%time

def count_sequence(df, field):
    '''
    For each letter of the provided suquence it return new feature with the number of occurences.
    '''
    alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']    
    
    for letter in alphabet:
        df[letter + '_count'] = df[field].str.count(letter)
    
    df["unique_characters"] = df['f_27'].apply(lambda s: len(set(s)))
    return df


## === cell 21
%%time


## === cell 22
%%time
def count_chars(df, field):
    '''
    Describes something...
    '''
    
    for i in range(10):
        df[f'ch_{i}'] = df[field].str.get(i).apply(ord) - ord('A')
        
    df["unique_characters"] = df[field].apply(lambda s: len(set(s)))
    return df


## === cell 23
%%time
trn_data = count_chars(trn_data, 'f_27')
tst_data = count_chars(tst_data, 'f_27')


## === cell 24
%%time
def calculate_feat_int(df):
    df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
    df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
    i_00_01_26 = df.f_00 + df.f_01 + df.f_26
    df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)
    return df


## === cell 25
%%time
trn_data = calculate_feat_int(trn_data)
tst_data = calculate_feat_int(tst_data)


## === cell 26
%%time
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
def encode_features(df, cols = ['f_27']):
    for col in cols:
        df[col + '_enc'] = encoder.fit_transform(df[col])
    return df

trn_data = encode_features(trn_data)
tst_data = encode_features(tst_data)


## === cell 27
trn_data.head()


## === cell 28
%%time
ignore = ['id', 'target', 'f_27',  'f_27_enc'] # f_27 has been label encoded...

features = [feat for feat in trn_data.columns if feat not in ignore]
target_feature = 'target'


## === cell 29
%%time
from sklearn.model_selection import train_test_split
test_size_pct = 0.20
X_train, X_valid, y_train, y_valid = train_test_split(trn_data[features], trn_data[target_feature], test_size = test_size_pct, random_state = 42)


## === cell 30
%%time
%%script false --no-raise-error
from xgboost  import XGBClassifier


## === cell 31
%%time
%%script false --no-raise-error
xgb_params = {'n_estimators'     : 8192,
              'min_child_weight' : 96,
              'max_bin'          : 512,
              'random_state'     : 46,
              'objective'        : 'binary:logistic',
              'tree_method'      : 'gpu_hist',
             }


## === cell 32
%%time
%%script false --no-raise-error
xgb = XGBClassifier(**xgb_params)
xgb.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], eval_metric = ['auc'], early_stopping_rounds = 256, verbose = 250)


## === cell 33
%%time
%%script false --no-raise-error
from sklearn.metrics import roc_auc_score
val_preds = xgb.predict_proba(X_valid[features])[:, 1]
roc_auc_score(y_valid, val_preds)


## === cell 35
%%time
%%script false --no-raise-error
from lightgbm import LGBMClassifier


## === cell 36
%%time
%%script false --no-raise-error
lgb_params = {'n_estimators'      : 8192,
              'min_child_samples' : 96,
              'max_bins'          : 512,
              'random_state'      : 46,
             }


## === cell 37
%%time
%%script false --no-raise-error
lgb = LGBMClassifier(**lgb_params)
lgb.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], eval_metric = ['auc'], early_stopping_rounds = 256, verbose = 250)


## === cell 38
%%time
%%script false --no-raise-error
from sklearn.metrics import roc_auc_score
val_preds = lgb.predict_proba(X_valid[features])[:, 1]
roc_auc_score(y_valid, val_preds)


## === cell 39
%%time
from lightgbm import LGBMClassifier
from xgboost  import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
import math


## === cell 40
%%time
lgb_params = {'n_estimators'      : 8192, # Was 8192...
              'min_child_samples' : 96,
              'max_bins'          : 512,
              'random_state'      : 46,
             }

xgb_params = {'n_estimators'     : 8192,
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


xgb_opt_params = {'n_estimators': 798,
                  'max_depth': 8,
                  'learning_rate': 0.1228167536332915,
                  'subsample': 0.5460612275431587,
                  'colsample_bytree': 0.6250334254566884,
                  'reg_lambda': 3.9594130578291624,
                  'reg_alpha': 5.534583993468559,
                  'gamma': 2.9138421416832916,
                  'min_child_weight': 0,
                  'max_bin': 235,
                  'random_state': 46,
                  'objective': 'binary:logistic',
                  'tree_method': 'gpu_hist',}


## === cell 41
%%time

score_list   = []
predictions  = [] 
kf = KFold(n_splits = 5)

for fold, (trn_idx, val_idx) in enumerate(kf.split(trn_data)):
    print(f'Training Fold {fold} ...')
    X_train, X_valid = trn_data.iloc[trn_idx][features], trn_data.iloc[val_idx][features]
    y_train, y_valid = trn_data.iloc[trn_idx][target_feature], trn_data.iloc[val_idx][target_feature]
    
    
    model = XGBClassifier(**xgb_opt_params)
    model.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], eval_metric = ['auc'], early_stopping_rounds = 256, verbose = 0)
    
    y_valid_pred = model.predict_proba(X_valid.values)[:,1]
    score = roc_auc_score(y_valid, y_valid_pred)

    score_list.append(score)
    print(f"Fold {fold}, AUC = {score:.4f}")
    print((''))
    
    tst_pred = model.predict_proba(tst_data[features].values)[:,1]
    predictions.append(tst_pred)

print(f'OOF AUC: {np.mean(score_list):.4f}')
print('.........')


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
<timed exec> in <module>

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

XGBoostError: [02:17:39] /workspace/src/tree/updater_gpu_hist.cu:781: Exception in gpu_hist: [02:17:39] /workspace/src/tree/updater_gpu_hist.cu:787: Check failed: ctx_->gpu_id >= 0 (-1 vs. 0) : Must have at least one device
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fff46fa0f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb3e95a) [0x7fff46fb795a]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb483cd) [0x7fff46fc13cd]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fff468d9c79]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fff468da76c]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fff4693e4f7]
  [bt] (6) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fff465daef0]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (8) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]



Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fff46fa0f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb485c9) [0x7fff46fc15c9]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fff468d9c79]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fff468da76c]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fff4693e4f7]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fff465daef0]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]



## === cell 42
%%time
def plot_feature_importance(importance, names, model_type, max_features = 10):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    fi_df = fi_df.head(max_features)

    plt.figure(figsize=(8,6))
    
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


## === cell 43
%%time
import seaborn as sns
import matplotlib.pyplot as plt
plot_feature_importance(model.feature_importances_,X_train.columns,'LGBM ', max_features = 25)


## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
<timed exec> in <module>

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in feature_importances_(self)
   1312 
   1313         """
-> 1314         b: Booster = self.get_booster()
   1315 
   1316         def dft() -> str:

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in get_booster(self)
    723             from sklearn.exceptions import NotFittedError
    724 
--> 725             raise NotFittedError("need to call fit or load_model beforehand")
    726         return self._Booster
    727 

NotFittedError: need to call fit or load_model beforehand

## === cell 44
%%time
sub.head()


## === cell 45
%%time
sub['target'] = np.array(predictions).mean(axis=0)
sub.to_csv('my_submission_043022.csv', index = False)


## === cell 46
%%time
sub.head()
