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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
optuna==4.5.0
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
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

20.46829

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import optuna

import matplotlib.pyplot as plt
import seaborn as sns


## === cell 1
TRAIN_DATA_PATH = '../input/petfinder-pawpularity-score/train.csv'
TEST_DATA_PATH = '../input/petfinder-pawpularity-score/test.csv'


## === cell 2
TARGET_NAME = 'Pawpularity'
VAL_SIZE = 0.15
SEED = 5
EARLY_ROUNDS = 50


## === cell 3
def set_seed(seed=42):
    """Utility function to use for reproducibility.
    :param seed: Random seed
    :return: None
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_display():
    """Function sets display options for charts and pd.DataFrames.
    """
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = 12, 8
    plt.rcParams.update({'font.size': 14})
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.options.display.float_format = '{:.4f}'.format


def get_features(df: pd.DataFrame) -> list:
    """Function selects input features from a DataFrame.
    :param df: DataFrame containing features, Ids and possibly target values
    :return: List of input features
    """
    return [column for column in df.columns
            if column != 'Id' and column != TARGET_NAME]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Function adds new features to the DataFrame
    by summing up existing features. Uses variable "features"
    defined outside the scope of this function.
    :param df: Original DataFrame
    :return: Updated DataFrame
    """
    df['features_sum'] = df[features].sum(axis=1) / len(features)

    for i in range(len(features) - 1):
        for j in range(i + 1, len(features)):
            feature_1 = features[i]
            feature_2 = features[j]
            df[f'{feature_1}_{feature_2}'] = (df[feature_1] + df[feature_2]) / 2

    for i in range(len(features) - 2):
        for j in range(i + 1, len(features) - 1):
            for z in range(j + 1, len(features)):
                feature_1 = features[i]
                feature_2 = features[j]
                feature_3 = features[z]
                df[f'{feature_1}_{feature_2}_{feature_3}'] = (
                    df[feature_1] + df[feature_2] + df[feature_3]) / 3

    return df


def rmse(y_true, y_pred) -> float:
    """Function calculates Root Mean Squared Error
    for predicted and actual values.
    :param y_true: Actual values
    :param y_pred: Predicted values
    :return: RMSE value
    """
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def objective(trial):
    """Function performs trials of parameter optimization
    for XGBoost model.
    :param trial: optuna trial object
    :return: RMSE score
    """
    global model
    params = {
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'n_estimators': trial.suggest_int('n_estimators', 250, 10_000, 250),
        'reg_lambda': trial.suggest_int('reg_lambda', 1, 100),
        'reg_alpha': trial.suggest_int('reg_alpha', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0, step=0.1),
        'max_depth': trial.suggest_int('max_depth', 1, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 100, step=5),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.95),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0)
    }

    fit_params = dict(eval_set=[(valid_x, valid_y)], eval_metric='rmse',
                      early_stopping_rounds=EARLY_ROUNDS, verbose=False)

    pruning_callback = optuna.integration.XGBoostPruningCallback(
        trial, 'validation_0-rmse')

    model = XGBRegressor(**params)
    model.fit(train_x, train_y, **fit_params, callbacks=[pruning_callback])
    y_pred = model.predict(valid_x)
    val_rmse = rmse(valid_y, y_pred)

    return val_rmse


## === cell 4
set_seed(SEED)
set_display()


## === cell 5
data_train = pd.read_csv(TRAIN_DATA_PATH)
print(f'Train data shape: {data_train.shape}')
data_train.head()


## === cell 6
data_test = pd.read_csv(TEST_DATA_PATH)
print(f'Test data shape: {data_test.shape}')
data_test.head()


## === cell 7
print(f'Target values: {data_train[TARGET_NAME].min()} - {data_train[TARGET_NAME].max()}\n'
      f'Mean value: {data_train[TARGET_NAME].mean()}\n'
      f'Median value: {data_train[TARGET_NAME].median()}\n'
      f'Standard deviation: {data_train[TARGET_NAME].std()}')

sns.histplot(data=data_train, x=TARGET_NAME, kde=True)
plt.axvline(data_train[TARGET_NAME].mean(), c='orange', ls='-', lw=3, label='Mean')
plt.axvline(data_train[TARGET_NAME].median(), c='green', ls='-', lw=3, label='Median')
plt.legend()
plt.title('Pawpularity Score')
plt.tight_layout()
plt.show()


## === cell 8
correlation = data_train.corr()
ax = sns.heatmap(correlation, center=0, annot=True, cmap='RdBu_r', fmt='0.3f')
l, r = ax.get_ylim()
ax.set_ylim(l + 0.5, r - 0.5)
plt.yticks(rotation=0)
plt.title('Correlation Matrix')
plt.show()

correlation[TARGET_NAME].sort_values()


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1589332861.py in <cell line: 0>()
      1 # Binary features correlation with the target.
----> 2 correlation = data_train.corr()
      3 ax = sns.heatmap(correlation, center=0, annot=True, cmap='RdBu_r', fmt='0.3f')
      4 l, r = ax.get_ylim()
      5 ax.set_ylim(l + 0.5, r - 0.5)

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

ValueError: could not convert string to float: '1a8795e64a294ed0c95132e18ee198e1'

## === cell 9
neg_features = correlation[correlation[TARGET_NAME] < 0].index.to_list()
data_train[neg_features] = data_train[neg_features].apply(lambda x: (x + 1) % 2)
data_test[neg_features] = data_test[neg_features].apply(lambda x: (x + 1) % 2)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1508579879.py in <cell line: 0>()
      1 # Before crossing the features we need to transform negatively correlated features
      2 # into positively correlated. We do it by switching 0 and 1 values.
----> 3 neg_features = correlation[correlation[TARGET_NAME] < 0].index.to_list()
      4 data_train[neg_features] = data_train[neg_features].apply(lambda x: (x + 1) % 2)
      5 data_test[neg_features] = data_test[neg_features].apply(lambda x: (x + 1) % 2)

NameError: name 'correlation' is not defined

## === cell 10
features = get_features(data_train)

data_train = add_features(data_train)
data_test = add_features(data_test)


## === cell 11
correlation = data_train.corr()
correlation[TARGET_NAME].sort_values()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/464668248.py in <cell line: 0>()
      1 # Check correlation of the new features with the target.
----> 2 correlation = data_train.corr()
      3 correlation[TARGET_NAME].sort_values()

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

ValueError: could not convert string to float: '1a8795e64a294ed0c95132e18ee198e1'

## === cell 12
features = get_features(data_train)

y = data_train[TARGET_NAME]
x = data_train[features]

train_x, valid_x, train_y, valid_y = train_test_split(
    x, y, test_size=VAL_SIZE, shuffle=True, random_state=SEED)
print(f'Train data shape: {train_x.shape}\n'
      f'Validation data shape: {valid_x.shape}')


## === cell 13
xgb_model = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor',
                         objective='reg:squarederror', booster='gbtree')

xgb_model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
              eval_metric='rmse', early_stopping_rounds=EARLY_ROUNDS)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/1280580829.py in <cell line: 0>()
      3                          objective='reg:squarederror', booster='gbtree')
      4 
----> 5 xgb_model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
      6               eval_metric='rmse', early_stopping_rounds=EARLY_ROUNDS)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in fit(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)
   1088                 xgb_model, eval_metric, params, early_stopping_rounds, callbacks
   1089             )
-> 1090             self._Booster = train(
   1091                 params,
   1092                 train_dmatrix,

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

XGBoostError: [01:23:50] /workspace/src/tree/updater_gpu_hist.cu:781: Exception in gpu_hist: [01:23:50] /workspace/src/tree/updater_gpu_hist.cu:787: Check failed: ctx_->gpu_id >= 0 (-1 vs. 0) : Must have at least one device
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fe5a6a85f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb3e95a) [0x7fe5a6a9c95a]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb483cd) [0x7fe5a6aa63cd]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fe5a63bec79]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fe5a63bf76c]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fe5a64234f7]
  [bt] (6) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fe5a60bfef0]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7fe618df0e2e]
  [bt] (8) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7fe618ded493]



Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fe5a6a85f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb485c9) [0x7fe5a6aa65c9]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fe5a63bec79]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fe5a63bf76c]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fe5a64234f7]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fe5a60bfef0]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7fe618df0e2e]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7fe618ded493]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7fe618e004d8]



## === cell 14
importance = pd.DataFrame({
    'features': features,
    'importance': xgb_model.feature_importances_
})
importance.sort_values(by='importance', inplace=True)

plt.barh([i for i in range(len(importance))], importance['importance'])
plt.title('XGBoost Feature Importance')
plt.show()


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/1234322807.py in <cell line: 0>()
      2 importance = pd.DataFrame({
      3     'features': features,
----> 4     'importance': xgb_model.feature_importances_
      5 })
      6 importance.sort_values(by='importance', inplace=True)

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

## === cell 15
threshold = 0.005
importance = importance[importance['importance'] >= threshold]
plt.figure(figsize=(12, 16))
plt.barh(importance['features'], importance['importance'])
plt.title('XGBoost Feature Importance')
plt.savefig('features.png', dpi=300)
plt.show()


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2381319694.py in <cell line: 0>()
      1 # Select informative features.
      2 threshold = 0.005
----> 3 importance = importance[importance['importance'] >= threshold]
      4 plt.figure(figsize=(12, 16))
      5 plt.barh(importance['features'], importance['importance'])

NameError: name 'importance' is not defined

## === cell 16
features = importance['features'].to_list()
x = data_train[features]

train_x, valid_x, train_y, valid_y = train_test_split(
    x, y, test_size=VAL_SIZE, shuffle=True, random_state=SEED)
print(f'Train data shape: {train_x.shape}\n'
      f'Validation data shape: {valid_x.shape}')


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2058268124.py in <cell line: 0>()
      1 # Split the data using only selected features.
----> 2 features = importance['features'].to_list()
      3 x = data_train[features]
      4 
      5 train_x, valid_x, train_y, valid_y = train_test_split(

NameError: name 'importance' is not defined

## === cell 17
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(seed=SEED),
    direction='minimize',
    study_name='xgb')


## === cell 18
study.optimize(objective, n_trials=200)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/optuna/integration/xgboost.py in <module>
      4 try:
----> 5     from optuna_integration.xgboost import XGBoostPruningCallback
      6 except ModuleNotFoundError:

ModuleNotFoundError: No module named 'optuna_integration'

During handling of the above exception, another exception occurred:

ModuleNotFoundError                       Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/optuna/integration/__init__.py in _get_module(self, module_name)
    131             try:
--> 132                 return importlib.import_module("." + module_name, self.__name__)
    133             except ModuleNotFoundError:

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 

/usr/lib/python3.11/importlib/_bootstrap.py in _gcd_import(name, package, level)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load(name, import_)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load_unlocked(name, import_)

/usr/lib/python3.11/importlib/_bootstrap.py in _load_unlocked(spec)

/usr/lib/python3.11/importlib/_bootstrap_external.py in exec_module(self, module)

/usr/lib/python3.11/importlib/_bootstrap.py in _call_with_frames_removed(f, *args, **kwds)

/usr/local/lib/python3.11/dist-packages/optuna/integration/xgboost.py in <module>
      6 except ModuleNotFoundError:
----> 7     raise ModuleNotFoundError(_INTEGRATION_IMPORT_ERROR_TEMPLATE.format("xgboost"))
      8 

ModuleNotFoundError: 
Could not find `optuna-integration` for `xgboost`.
Please run `pip install optuna-integration[xgboost]`.

During handling of the above exception, another exception occurred:

ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3256500717.py in <cell line: 0>()
----> 1 study.optimize(objective, n_trials=200)

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in optimize(self, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)
    488                 If nested invocation of this method occurs.
    489         """
--> 490         _optimize(
    491             study=self,
    492             func=func,

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _optimize(study, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)
     61     try:
     62         if n_jobs == 1:
---> 63             _optimize_sequential(
     64                 study,
     65                 func,

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _optimize_sequential(study, func, n_trials, timeout, catch, callbacks, gc_after_trial, reseed_sampler_rng, time_start, progress_bar)
    158 
    159         try:
--> 160             frozen_trial_id = _run_trial(study, func, catch)
    161         finally:
    162             # The following line mitigates memory problems that can be occurred in some

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _run_trial(study, func, catch)
    256         and not isinstance(func_err, catch)
    257     ):
--> 258         raise func_err
    259     return trial._trial_id
    260 

/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py in _run_trial(study, func, catch)
    199     with get_heartbeat_thread(trial._trial_id, study._storage):
    200         try:
--> 201             value_or_values = func(trial)
    202         except exceptions.TrialPruned as e:
    203             # TODO(mamu): Handle multi-objective cases.

/tmp/ipykernel_11/3099793299.py in objective(trial)
     97                       early_stopping_rounds=EARLY_ROUNDS, verbose=False)
     98 
---> 99     pruning_callback = optuna.integration.XGBoostPruningCallback(
    100         trial, 'validation_0-rmse')
    101 

/usr/local/lib/python3.11/dist-packages/optuna/integration/__init__.py in __getattr__(self, name)
    118                 value = self._get_module(name)
    119             elif name in self._class_to_module.keys():
--> 120                 module = self._get_module(self._class_to_module[name])
    121                 value = getattr(module, name)
    122             else:

/usr/local/lib/python3.11/dist-packages/optuna/integration/__init__.py in _get_module(self, module_name)
    132                 return importlib.import_module("." + module_name, self.__name__)
    133             except ModuleNotFoundError:
--> 134                 raise ModuleNotFoundError(_INTEGRATION_IMPORT_ERROR_TEMPLATE.format(module_name))
    135 
    136     sys.modules[__name__] = _IntegrationModule(__name__)

ModuleNotFoundError: 
Could not find `optuna-integration` for `xgboost`.
Please run `pip install optuna-integration[xgboost]`.

## === cell 19
xgb_params = study.best_params
print('XGBoost best RMSE:', study.best_value)
print('Optimal parameters:')
for key, value in xgb_params.items():
    print(f'\t{key}: {value}')


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/310538607.py in <cell line: 0>()
----> 1 xgb_params = study.best_params
      2 print('XGBoost best RMSE:', study.best_value)
      3 print('Optimal parameters:')
      4 for key, value in xgb_params.items():
      5     print(f'\t{key}: {value}')

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in best_params(self)
    118         """
    119 
--> 120         return self.best_trial.params
    121 
    122     @property

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in best_trial(self)
    154 
    155         """
--> 156         return self._get_best_trial(deepcopy=True)
    157 
    158     @property

/usr/local/lib/python3.11/dist-packages/optuna/study/study.py in _get_best_trial(self, deepcopy)
    306             )
    307 
--> 308         best_trial = self._storage.get_best_trial(self._study_id)
    309 
    310         # If the trial with the best value is infeasible, select the best trial from all feasible

/usr/local/lib/python3.11/dist-packages/optuna/storages/_in_memory.py in get_best_trial(self, study_id)
    250 
    251             if best_trial_id is None:
--> 252                 raise ValueError("No trials are completed yet.")
    253             elif len(self._studies[study_id].directions) > 1:
    254                 raise RuntimeError(

ValueError: No trials are completed yet.

## === cell 20
xgb_model = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor',
                         objective='reg:squarederror', booster='gbtree',
                         **xgb_params)

xgb_model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
              eval_metric='rmse', early_stopping_rounds=EARLY_ROUNDS)

print('Validation RMSE:', xgb_model.best_score)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3424314583.py in <cell line: 0>()
      2 xgb_model = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor',
      3                          objective='reg:squarederror', booster='gbtree',
----> 4                          **xgb_params)
      5 
      6 xgb_model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],

NameError: name 'xgb_params' is not defined

## === cell 21
data_test[TARGET_NAME] = xgb_model.predict(data_test[features])
data_test[['Id', TARGET_NAME]].to_csv('submission.csv', index=False)
data_test[['Id', TARGET_NAME]].head()


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/42104644.py in <cell line: 0>()
      1 # Predict values for the test set.
----> 2 data_test[TARGET_NAME] = xgb_model.predict(data_test[features])
      3 data_test[['Id', TARGET_NAME]].to_csv('submission.csv', index=False)
      4 data_test[['Id', TARGET_NAME]].head()

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict(self, X, output_margin, validate_features, base_margin, iteration_range)
   1166             if self._can_use_inplace_predict():
   1167                 try:
-> 1168                     predts = self.get_booster().inplace_predict(
   1169                         data=X,
   1170                         iteration_range=iteration_range,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in get_booster(self)
    723             from sklearn.exceptions import NotFittedError
    724 
--> 725             raise NotFittedError("need to call fit or load_model beforehand")
    726         return self._Booster
    727 

NotFittedError: need to call fit or load_model beforehand
