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
correlation = data_train.corr(numeric_only=True)
correlation[TARGET_NAME].sort_values()


## === cell 12
features = get_features(data_train)

y = data_train[TARGET_NAME]
x = data_train[features]

train_x, valid_x, train_y, valid_y = train_test_split(
    x, y, test_size=VAL_SIZE, shuffle=True, random_state=SEED)
print(f'Train data shape: {train_x.shape}\n'
      f'Validation data shape: {valid_x.shape}')


## === cell 13
xgb_model = XGBRegressor(
    tree_method="hist",
    predictor="cpu_predictor",
    objective="reg:squarederror",
    booster="gbtree",
)

xgb_model.fit(
    train_x,
    train_y,
    eval_set=[(valid_x, valid_y)],
    eval_metric="rmse",
    early_stopping_rounds=EARLY_ROUNDS,
)


## === cell 14
importance = pd.DataFrame({
    'features': features,
    'importance': xgb_model.feature_importances_
})
importance.sort_values(by='importance', inplace=True)

plt.barh([i for i in range(len(importance))], importance['importance'])
plt.title('XGBoost Feature Importance')
plt.show()


## === cell 15
threshold = 0.005
importance = importance[importance['importance'] >= threshold]
plt.figure(figsize=(12, 16))
plt.barh(importance['features'], importance['importance'])
plt.title('XGBoost Feature Importance')
plt.savefig('features.png', dpi=300)
plt.show()


## === cell 16
features = importance['features'].to_list()
x = data_train[features]

train_x, valid_x, train_y, valid_y = train_test_split(
    x, y, test_size=VAL_SIZE, shuffle=True, random_state=SEED)
print(f'Train data shape: {train_x.shape}\n'
      f'Validation data shape: {valid_x.shape}')


## === cell 17
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(seed=SEED),
    direction='minimize',
    study_name='xgb')


## === cell 18
study.optimize(objective, n_trials=200)


## --- ERROR in cell 18, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/optuna/integration/xgboost.py[0m in [0;36m<module>[0;34m[0m
[1;32m      4[0m [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m     [0;32mfrom[0m [0moptuna_integration[0m[0;34m.[0m[0mxgboost[0m [0;32mimport[0m [0mXGBoostPruningCallback[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;32mexcept[0m [0mModuleNotFoundError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'optuna_integration'

During handling of the above exception, another exception occurred:

[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/optuna/integration/__init__.py[0m in [0;36m_get_module[0;34m(self, module_name)[0m
[1;32m    131[0m             [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 132[0;31m                 [0;32mreturn[0m [0mimportlib[0m[0;34m.[0m[0mimport_module[0m[0;34m([0m[0;34m"."[0m [0;34m+[0m [0mmodule_name[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m__name__[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    133[0m             [0;32mexcept[0m [0mModuleNotFoundError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    127[0m [0;34m[0m[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_gcd_import[0;34m(name, package, level)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_find_and_load[0;34m(name, import_)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_find_and_load_unlocked[0;34m(name, import_)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_load_unlocked[0;34m(spec)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap_external.py[0m in [0;36mexec_module[0;34m(self, module)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_call_with_frames_removed[0;34m(f, *args, **kwds)[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/integration/xgboost.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mexcept[0m [0mModuleNotFoundError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m     [0;32mraise[0m [0mModuleNotFoundError[0m[0;34m([0m[0m_INTEGRATION_IMPORT_ERROR_TEMPLATE[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0;34m"xgboost"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0;34m[0m[0m

[0;31mModuleNotFoundError[0m: 
Could not find `optuna-integration` for `xgboost`.
Please run `pip install optuna-integration[xgboost]`.

During handling of the above exception, another exception occurred:

[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3256500717.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mstudy[0m[0;34m.[0m[0moptimize[0m[0;34m([0m[0mobjective[0m[0;34m,[0m [0mn_trials[0m[0;34m=[0m[0;36m200[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
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

[0;32m/tmp/ipykernel_11/3099793299.py[0m in [0;36mobjective[0;34m(trial)[0m
[1;32m     97[0m                       early_stopping_rounds=EARLY_ROUNDS, verbose=False)
[1;32m     98[0m [0;34m[0m[0m
[0;32m---> 99[0;31m     pruning_callback = optuna.integration.XGBoostPruningCallback(
[0m[1;32m    100[0m         trial, 'validation_0-rmse')
[1;32m    101[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/integration/__init__.py[0m in [0;36m__getattr__[0;34m(self, name)[0m
[1;32m    118[0m                 [0mvalue[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_module[0m[0;34m([0m[0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    119[0m             [0;32melif[0m [0mname[0m [0;32min[0m [0mself[0m[0;34m.[0m[0m_class_to_module[0m[0;34m.[0m[0mkeys[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 120[0;31m                 [0mmodule[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_module[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_class_to_module[0m[0;34m[[0m[0mname[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    121[0m                 [0mvalue[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mmodule[0m[0;34m,[0m [0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    122[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/integration/__init__.py[0m in [0;36m_get_module[0;34m(self, module_name)[0m
[1;32m    132[0m                 [0;32mreturn[0m [0mimportlib[0m[0;34m.[0m[0mimport_module[0m[0;34m([0m[0;34m"."[0m [0;34m+[0m [0mmodule_name[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m__name__[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    133[0m             [0;32mexcept[0m [0mModuleNotFoundError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 134[0;31m                 [0;32mraise[0m [0mModuleNotFoundError[0m[0;34m([0m[0m_INTEGRATION_IMPORT_ERROR_TEMPLATE[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mmodule_name[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    135[0m [0;34m[0m[0m
[1;32m    136[0m     [0msys[0m[0;34m.[0m[0mmodules[0m[0;34m[[0m[0m__name__[0m[0;34m][0m [0;34m=[0m [0m_IntegrationModule[0m[0;34m([0m[0m__name__[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: 
Could not find `optuna-integration` for `xgboost`.
Please run `pip install optuna-integration[xgboost]`.

## === cell 19
xgb_params = study.best_params
print('XGBoost best RMSE:', study.best_value)
print('Optimal parameters:')
for key, value in xgb_params.items():
    print(f'\t{key}: {value}')
