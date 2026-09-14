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

20.45855

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

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
lgb_best_params = optimize(model="lightgbm",n_trials=100)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3569219734.py in <cell line: 0>()
----> 1 lgb_best_params = optimize(model="lightgbm",n_trials=100)

/tmp/ipykernel_11/1933843560.py in optimize(model, n_trials)
      7 
      8     study = optuna.create_study(direction="minimize")
----> 9     study.optimize(optimizer_func,n_trials=n_trials)
     10 
     11     return study.best_params

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

/tmp/ipykernel_11/302839608.py in objective(trial, model, df)
     84         lgb_val   = lgb.Dataset(x_test,y_test)
     85 
---> 86         model=lgb.train(
     87             param,
     88             lgb_train,

TypeError: train() got an unexpected keyword argument 'early_stopping_rounds'

## === cell 9
catboost_best_params = optimize(model="catboost",n_trials=100)


## === cell 10
xgboost_best_params = optimize(model="xgboost",n_trials=30)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/509943082.py in <cell line: 0>()
----> 1 xgboost_best_params = optimize(model="xgboost",n_trials=30)

/tmp/ipykernel_11/1933843560.py in optimize(model, n_trials)
      7 
      8     study = optuna.create_study(direction="minimize")
----> 9     study.optimize(optimizer_func,n_trials=n_trials)
     10 
     11     return study.best_params

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

/tmp/ipykernel_11/302839608.py in objective(trial, model, df)
     55 
     56         xgboost_regressor = XGBRegressor(**param,random_state=42,tree_method="gpu_hist",gpu_id=1,predictor="gpu_predictor")
---> 57         xgboost_regressor.fit(x_train,y_train,eval_set=[(x_test,y_test)])
     58         print("buradasın")
     59         preds = xgboost_regressor.predict(x_test)

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

XGBoostError: [01:03:54] /workspace/src/tree/updater_gpu_hist.cu:781: Exception in gpu_hist: [01:03:54] /workspace/src/tree/updater_gpu_hist.cu:787: Check failed: ctx_->gpu_id >= 0 (-1 vs. 0) : Must have at least one device
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


## === cell 12
model_predictions = []

for model,param in zip(['catboost','xgboost','lightgbm'],[catboost_best_params,xgboost_best_params,lgb_best_params]):
    
    preds =  fit_fold_with_best_params(df=train_df,test_df=test_df,fold=5,model=model,params=param)
    
    model_predictions.append(preds)   


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2648824631.py in <cell line: 0>()
      1 model_predictions = []
      2 
----> 3 for model,param in zip(['catboost','xgboost','lightgbm'],[catboost_best_params,xgboost_best_params,lgb_best_params]):
      4 
      5     preds =  fit_fold_with_best_params(df=train_df,test_df=test_df,fold=5,model=model,params=param)

NameError: name 'xgboost_best_params' is not defined

## === cell 13
last = np.mean(np.column_stack(model_predictions), axis=1)
 


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/251325224.py in <cell line: 0>()
----> 1 last = np.mean(np.column_stack(model_predictions), axis=1)
      2 

/usr/local/lib/python3.11/dist-packages/numpy/lib/shape_base.py in column_stack(tup)
    650             arr = array(arr, copy=False, subok=True, ndmin=2).T
    651         arrays.append(arr)
--> 652     return _nx.concatenate(arrays, 1)
    653 
    654 

ValueError: need at least one array to concatenate

## === cell 14
ids = test_df['Id'].values

submission = pd.DataFrame({'Id':ids,
                            'Pawpularity':last})


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/464334331.py in <cell line: 0>()
      2 
      3 submission = pd.DataFrame({'Id':ids,
----> 4                             'Pawpularity':last})

NameError: name 'last' is not defined

## === cell 15
submission.to_csv("submission.csv",index=False)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1693054785.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv",index=False)

NameError: name 'submission' is not defined
