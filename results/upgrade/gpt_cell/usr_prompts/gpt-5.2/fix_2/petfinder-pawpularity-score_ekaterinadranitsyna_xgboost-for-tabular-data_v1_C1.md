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

# 3. Data file paths

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

# 4. Code solution

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
correlation = data_train.corr(numeric_only=True)
ax = sns.heatmap(correlation, center=0, annot=True, cmap="RdBu_r", fmt="0.3f")
l, r = ax.get_ylim()
ax.set_ylim(l + 0.5, r - 0.5)
plt.yticks(rotation=0)
plt.title("Correlation Matrix")
plt.show()

correlation[TARGET_NAME].sort_values()


## === cell 9
neg_features = correlation[correlation[TARGET_NAME] < 0].index.to_list()
data_train[neg_features] = data_train[neg_features].apply(lambda x: (x + 1) % 2)
data_test[neg_features] = data_test[neg_features].apply(lambda x: (x + 1) % 2)


## === cell 10
features = get_features(data_train)

data_train = add_features(data_train)
data_test = add_features(data_test)


## === cell 11
correlation = data_train.corr()
correlation[TARGET_NAME].sort_values()


## --- ERROR in cell 11, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/464668248.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# Check correlation of the new features with the target.[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mcorrelation[0m [0;34m=[0m [0mdata_train[0m[0;34m.[0m[0mcorr[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0mcorrelation[0m[0;34m[[0m[0mTARGET_NAME[0m[0;34m][0m[0;34m.[0m[0msort_values[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36mcorr[0;34m(self, method, min_periods, numeric_only)[0m
[1;32m  11047[0m         [0mcols[0m [0;34m=[0m [0mdata[0m[0;34m.[0m[0mcolumns[0m[0;34m[0m[0;34m[0m[0m
[1;32m  11048[0m         [0midx[0m [0;34m=[0m [0mcols[0m[0;34m.[0m[0mcopy[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m> 11049[0;31m         [0mmat[0m [0;34m=[0m [0mdata[0m[0;34m.[0m[0mto_numpy[0m[0;34m([0m[0mdtype[0m[0;34m=[0m[0mfloat[0m[0;34m,[0m [0mna_value[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mnan[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m  11050[0m [0;34m[0m[0m
[1;32m  11051[0m         [0;32mif[0m [0mmethod[0m [0;34m==[0m [0;34m"pearson"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36mto_numpy[0;34m(self, dtype, copy, na_value)[0m
[1;32m   1991[0m         [0;32mif[0m [0mdtype[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1992[0m             [0mdtype[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mdtype[0m[0;34m([0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1993[0;31m         [0mresult[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_mgr[0m[0;34m.[0m[0mas_array[0m[0;34m([0m[0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0mcopy[0m[0;34m,[0m [0mna_value[0m[0;34m=[0m[0mna_value[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1994[0m         [0;32mif[0m [0mresult[0m[0;34m.[0m[0mdtype[0m [0;32mis[0m [0;32mnot[0m [0mdtype[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1995[0m             [0mresult[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0mresult[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py[0m in [0;36mas_array[0;34m(self, dtype, copy, na_value)[0m
[1;32m   1692[0m                 [0marr[0m[0;34m.[0m[0mflags[0m[0;34m.[0m[0mwriteable[0m [0;34m=[0m [0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1693[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1694[0;31m             [0marr[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_interleave[0m[0;34m([0m[0mdtype[0m[0;34m=[0m[0mdtype[0m[0;34m,[0m [0mna_value[0m[0;34m=[0m[0mna_value[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1695[0m             [0;31m# The underlying data was copied within _interleave, so no need[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1696[0m             [0;31m# to further copy if copy=True or setting na_value[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py[0m in [0;36m_interleave[0;34m(self, dtype, na_value)[0m
[1;32m   1751[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1752[0m                 [0marr[0m [0;34m=[0m [0mblk[0m[0;34m.[0m[0mget_values[0m[0;34m([0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1753[0;31m             [0mresult[0m[0;34m[[0m[0mrl[0m[0;34m.[0m[0mindexer[0m[0;34m][0m [0;34m=[0m [0marr[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1754[0m             [0mitemmask[0m[0;34m[[0m[0mrl[0m[0;34m.[0m[0mindexer[0m[0;34m][0m [0;34m=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1755[0m [0;34m[0m[0m

[0;31mValueError[0m: could not convert string to float: '1a8795e64a294ed0c95132e18ee198e1'

## === cell 12
features = get_features(data_train)

y = data_train[TARGET_NAME]
x = data_train[features]

train_x, valid_x, train_y, valid_y = train_test_split(
    x, y, test_size=VAL_SIZE, shuffle=True, random_state=SEED)
print(f'Train data shape: {train_x.shape}\n'
      f'Validation data shape: {valid_x.shape}')
