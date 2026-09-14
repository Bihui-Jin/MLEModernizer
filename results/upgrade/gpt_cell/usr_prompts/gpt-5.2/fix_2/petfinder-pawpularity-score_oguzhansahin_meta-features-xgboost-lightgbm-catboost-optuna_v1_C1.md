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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
sample_submission = pd.read_csv("../input/petfinder-pawpularity-score/sample_submission.csv")
sample_submission.head()


## === cell 2
train_df = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
train_df.head()


## === cell 3
test_df = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
test_df.head()


## === cell 4
def kfold_split(df:pd.DataFrame, n_splits:int, shuffle:bool, random_state:int):
    
    from sklearn.model_selection import StratifiedKFold
    
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    for fold, (train_idx,valid_idx) in enumerate(kf.split(X=df,y=df.Pawpularity.values)):
        
        df.loc[valid_idx,'kfold'] = fold
        
    return df 


## === cell 5
train_df = kfold_split(df=train_df, n_splits=5, shuffle=True, random_state=42)
train_df.head()


## === cell 6
import optuna
import catboost as ctb
from xgboost import XGBRegressor
import lightgbm as lgb

from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

def objective(trial,model,df):
    
    x_train = df[df.kfold != 0].reset_index(drop=True)
    x_test  = df[df.kfold  == 0].reset_index(drop=True)


    y_train = x_train.Pawpularity
    y_test = x_test.Pawpularity

    x_train = x_train.drop(columns=['Id','Pawpularity','kfold'])
    x_test  = x_test.drop(columns=['Id','Pawpularity','kfold'])
    
    if model == "catboost":
      
        param = {

            'iterations': trial.suggest_int("iterations",100,2000),
            'learning_rate': trial.suggest_float("learning_rate",1e-2,0.25, log=True),
            'subsample': trial.suggest_float("subsample",0.1,1.0),
            'depth':trial.suggest_int("depth",5,9),
            'bagging_temperature':trial.suggest_int("bagging_temperature",0,5)

        }

        catboost_regressor = ctb.CatBoostRegressor(**param,random_state = 42,verbose=0)

        catboost_regressor.fit(x_train,y_train,eval_set=[(x_test,y_test)])

        preds = catboost_regressor.predict(x_test)
        rmse = metrics.mean_squared_error(y_test,preds)

        return rmse
    
    if model == "xgboost":
        
        param = {

            'reg_lambda': trial.suggest_loguniform("reg_lambda",1e-8,100.0),
            'reg_alpha': trial.suggest_loguniform("reg_alpha", 1e-8,100.0),
            'learning_rate': trial.suggest_float("learning_rate",1e-2,0.25, log=True),
            'subsample': trial.suggest_float("subsample",0.1,1.0),
            'max_depth':trial.suggest_int("max_depth",1,7),
            'colsample_bytree':trial.suggest_float("colsample_bytree",0.1,1.0)

        }
        
        xgboost_regressor = XGBRegressor(**param,random_state=42,tree_method="gpu_hist",gpu_id=1,predictor="gpu_predictor")
        xgboost_regressor.fit(x_train,y_train,eval_set=[(x_test,y_test)])
        print("buradasın")
        preds = xgboost_regressor.predict(x_test)
        rmse  = metrics.mean_squared_error(y_test,preds)
        
        return rmse
    
    
    if model =="lightgbm":

        param = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse'},
            'n_estimators': trial.suggest_int("n_estimators", 64, 8192),
            'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.25, log=True),
            'num_leaves': trial.suggest_int("num_leaves", 4, 16),
            'max_depth': trial.suggest_int("max_depth", 4, 16),
            'feature_fraction': trial.suggest_float("feature_fraction", 0.1, 1.0),
            'lambda_l1': trial.suggest_loguniform("lambda_l1", 1e-8, 100.0),
            'lambda_l2': trial.suggest_loguniform("lambda_l2", 1e-8, 100.0),
            'seed': 42,
            'deterministic': True,
            'verbose':-1,
        }

        lgb_train = lgb.Dataset(x_train,y_train)
        lgb_val   = lgb.Dataset(x_test,y_test)

        model=lgb.train(
            param,
            lgb_train,
            num_boost_round=5000,
            valid_sets=(lgb_train, lgb_val),
            early_stopping_rounds=100,
            verbose_eval=False
            
        )
        
        preds = model.predict(x_test)
        rmse = metrics.mean_squared_error(y_test,preds)
        
        return rmse   


## === cell 7
from functools import partial


def optimize(model,n_trials):

    optimizer_func = partial(objective,model=model,df=train_df)

    study = optuna.create_study(direction="minimize")
    study.optimize(optimizer_func,n_trials=n_trials)

    return study.best_params


## === cell 8
import optuna
import catboost as ctb
from xgboost import XGBRegressor
import lightgbm as lgb

from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


def objective(trial, model, df):

    x_train = df[df.kfold != 0].reset_index(drop=True)
    x_test = df[df.kfold == 0].reset_index(drop=True)

    y_train = x_train.Pawpularity
    y_test = x_test.Pawpularity

    x_train = x_train.drop(columns=["Id", "Pawpularity", "kfold"])
    x_test = x_test.drop(columns=["Id", "Pawpularity", "kfold"])

    if model == "catboost":

        param = {
            "iterations": trial.suggest_int("iterations", 100, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "depth": trial.suggest_int("depth", 5, 9),
            "bagging_temperature": trial.suggest_int("bagging_temperature", 0, 5),
        }

        catboost_regressor = ctb.CatBoostRegressor(**param, random_state=42, verbose=0)

        catboost_regressor.fit(x_train, y_train, eval_set=[(x_test, y_test)])

        preds = catboost_regressor.predict(x_test)
        rmse = metrics.mean_squared_error(y_test, preds)

        return rmse

    if model == "xgboost":

        param = {
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 100.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 100.0),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "max_depth": trial.suggest_int("max_depth", 1, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        }

        xgboost_regressor = XGBRegressor(
            **param,
            random_state=42,
            tree_method="gpu_hist",
            gpu_id=1,
            predictor="gpu_predictor"
        )
        xgboost_regressor.fit(x_train, y_train, eval_set=[(x_test, y_test)])
        print("buradasın")
        preds = xgboost_regressor.predict(x_test)
        rmse = metrics.mean_squared_error(y_test, preds)

        return rmse

    if model == "lightgbm":

        param = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": {"rmse"},
            "n_estimators": trial.suggest_int("n_estimators", 64, 8192),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.25, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 16),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 100.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 100.0),
            "seed": 42,
            "deterministic": True,
            "verbose": -1,
        }

        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_val = lgb.Dataset(x_test, y_test)

        model = lgb.train(
            param,
            lgb_train,
            num_boost_round=5000,
            valid_sets=(lgb_train, lgb_val),
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=0),
            ],
        )

        preds = model.predict(x_test)
        rmse = metrics.mean_squared_error(y_test, preds)

        return rmse


## === cell 9
catboost_best_params = optimize(model="catboost",n_trials=100)


## === cell 10
xgboost_best_params = optimize(model="xgboost",n_trials=30)


## --- ERROR in cell 10, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mXGBoostError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/509943082.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mxgboost_best_params[0m [0;34m=[0m [0moptimize[0m[0;34m([0m[0mmodel[0m[0;34m=[0m[0;34m"xgboost"[0m[0;34m,[0m[0mn_trials[0m[0;34m=[0m[0;36m30[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/tmp/ipykernel_11/1933843560.py[0m in [0;36moptimize[0;34m(model, n_trials)[0m
[1;32m      7[0m [0;34m[0m[0m
[1;32m      8[0m     [0mstudy[0m [0;34m=[0m [0moptuna[0m[0;34m.[0m[0mcreate_study[0m[0;34m([0m[0mdirection[0m[0;34m=[0m[0;34m"minimize"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 9[0;31m     [0mstudy[0m[0;34m.[0m[0moptimize[0m[0;34m([0m[0moptimizer_func[0m[0;34m,[0m[0mn_trials[0m[0;34m=[0m[0mn_trials[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     10[0m [0;34m[0m[0m
[1;32m     11[0m     [0;32mreturn[0m [0mstudy[0m[0;34m.[0m[0mbest_params[0m[0;34m[0m[0;34m[0m[0m

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

[0;32m/tmp/ipykernel_11/3724451208.py[0m in [0;36mobjective[0;34m(trial, model, df)[0m
[1;32m     57[0m             [0mpredictor[0m[0;34m=[0m[0;34m"gpu_predictor"[0m[0;34m[0m[0;34m[0m[0m
[1;32m     58[0m         )
[0;32m---> 59[0;31m         [0mxgboost_regressor[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mx_train[0m[0;34m,[0m [0my_train[0m[0;34m,[0m [0meval_set[0m[0;34m=[0m[0;34m[[0m[0;34m([0m[0mx_test[0m[0;34m,[0m [0my_test[0m[0;34m)[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     60[0m         [0mprint[0m[0;34m([0m[0;34m"buradasın"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     61[0m         [0mpreds[0m [0;34m=[0m [0mxgboost_regressor[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mx_test[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)[0m
[1;32m   1088[0m                 [0mxgb_model[0m[0;34m,[0m [0meval_metric[0m[0;34m,[0m [0mparams[0m[0;34m,[0m [0mearly_stopping_rounds[0m[0;34m,[0m [0mcallbacks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1089[0m             )
[0;32m-> 1090[0;31m             self._Booster = train(
[0m[1;32m   1091[0m                 [0mparams[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1092[0m                 [0mtrain_dmatrix[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

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

[0;31mXGBoostError[0m: [02:07:12] /workspace/src/tree/updater_gpu_hist.cu:781: Exception in gpu_hist: [02:07:12] /workspace/src/tree/updater_gpu_hist.cu:787: Check failed: ctx_->gpu_id >= 0 (-1 vs. 0) : Must have at least one device
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fff6d033f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb3e95a) [0x7fff6d04a95a]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb483cd) [0x7fff6d0543cd]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fff6c96cc79]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fff6c96d76c]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fff6c9d14f7]
  [bt] (6) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fff6c66def0]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (8) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]



Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fff6d033f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb485c9) [0x7fff6d0545c9]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fff6c96cc79]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fff6c96d76c]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fff6c9d14f7]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fff6c66def0]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]



## === cell 11
def fit_fold_with_best_params(df,test_df,fold,model,params):
    
    fold_predictions = []
    
    for fold_idx in range(fold):

        x_train = df[df.kfold != fold_idx].reset_index(drop=True)
        x_valid  = df[df.kfold  == fold_idx].reset_index(drop=True)


        y_train = x_train.Pawpularity
        y_valid = x_valid.Pawpularity

        x_train = x_train.drop(columns=['Id','Pawpularity','kfold'])
        x_valid  = x_valid.drop(columns=['Id','Pawpularity','kfold'])


        x_test = test_df.drop(columns=['Id'])

        if model == "catboost":

            catboost_regressor = ctb.CatBoostRegressor(**params,random_state = 42,verbose=0)
            catboost_regressor.fit(x_train,y_train,eval_set=[(x_valid,y_valid)])

            valid_preds = catboost_regressor.predict(x_valid)
            rmse = np.sqrt(metrics.mean_squared_error(y_valid,valid_preds))

            print(f"RMSE of {model} for the fold {fold_idx} : {rmse}")

            test_preds = catboost_regressor.predict(x_test)

            fold_predictions.append(test_preds)
        
        elif model =="xgboost":
            
            xgboost_regressor = XGBRegressor(**params,random_state=42,tree_method="gpu_hist",gpu_id=1,predictor="gpu_predictor")
            xgboost_regressor.fit(x_train,y_train,eval_set=[(x_valid,y_valid)])
            
            valid_preds = xgboost_regressor.predict(x_valid)
            rmse = np.sqrt(metrics.mean_squared_error(y_valid,valid_preds))

            print(f"RMSE of {model} for the fold {fold_idx} : {rmse}")
            
            test_preds = xgboost_regressor.predict(x_test)
            
            fold_predictions.append(test_preds)
        
        elif model=="lightgbm":
            
            lgb_train = lgb.Dataset(x_train,y_train)
            lgb_val   = lgb.Dataset(x_valid,y_valid)

            model=lgb.train(
                params,
                lgb_train,
                num_boost_round=5000,
                verbose_eval=False

            )
            
            valid_preds = model.predict(x_valid)
            rmse = np.sqrt(metrics.mean_squared_error(y_valid,valid_preds))

            print(f"RMSE of {model} for the fold {fold_idx} : {rmse}")
            
            test_preds = model.predict(x_test)
            fold_predictions.append(test_preds)
        
    final_predictions = np.mean(np.column_stack(fold_predictions), axis=1)
       
    return final_predictions
