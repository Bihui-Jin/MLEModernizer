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
Predict values for synthetic data.

### Description
## Metric
Area under the ROC curve for each target, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict the value for the targets `EC1` and `EC2`. The file should contain a header and have the following format:

```
id,EC1,EC2
14838,0.22,0.71
14839,0.78,0.43
14840,0.53,0.11
etc.
```

## Dataset 
- **train.csv** - the training dataset; `[EC1 - EC6]` are the (binary) targets, although you are only asked to predict `EC1` and `EC2`.
- **test.csv** - the test dataset; your objective is to predict the probability of the two targets `EC1` and `EC2`
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

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
scikit-multilearn==0.2.0
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (1485 lines)
            sample_submission.csv.zip (4.4 kB)
            test.csv (1485 lines)
            test.csv.zip (147.8 kB)
            train.csv (13355 lines)
            train.csv.zip (1.4 MB)
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
        input/
            description.md (56 lines)
            sample_submission.csv (1485 lines)
            sample_submission.csv.zip (4.4 kB)
            test.csv (1485 lines)
            test.csv.zip (147.8 kB)
            train.csv (13355 lines)
            train.csv.zip (1.4 MB)
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
        working/
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
```

-> data/playground-series-s3e18/sample_submission.csv has 1484 rows and 3 columns.
The columns are: id, EC1, EC2

-> data/playground-series-s3e18/test.csv has 1484 rows and 32 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 17 more columns

-> data/playground-series-s3e18/train.csv has 13354 rows and 38 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 23 more columns

-> data/sample_submission.csv has 1484 rows and 3 columns.
The columns are: id, EC1, EC2

-> data/test.csv has 1484 rows and 32 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 17 more columns

-> data/train.csv has 13354 rows and 38 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 23 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.64882

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


## === cell 1
train = pd.read_csv('/kaggle/input/playground-series-s3e18/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e18/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s3e18/sample_submission.csv')


## === cell 2
train


## === cell 3
test


## === cell 4
Df = [train,test]
names = ['Training Data','Test Data']
print('---'*5,'Data Information','---'*5,'\n')
for df,name in zip(Df,names):
    print(name)
    print(df.info())
    print('=='*25,'\n')


## === cell 5
desc = test.describe().transpose()
desc.style.background_gradient()


## === cell 6
for df in [train,test]:
    df.drop('id',axis=1,inplace=True)
train.drop(['EC3','EC4','EC5','EC6'],axis=1,inplace=True)


## === cell 9
plt.figure(figsize=(12,8))
sns.heatmap(data=train.drop(['EC1','EC2'],axis=1).corr(),annot=True,cmap='Greens')
plt.title('Correlation Matrix for Features of Train Data');
plt.show()


## === cell 10
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, early_stopping
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score,classification_report, confusion_matrix, roc_auc_score, roc_curve #ConfusionMatrixDisplay 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


## === cell 11
target = train[['EC1','EC2']]

X = train.drop(target,axis=1)
y = target
X_test = test


## === cell 12
scaler = StandardScaler()

X = StandardScaler().fit_transform(X)
X_test = StandardScaler().fit_transform(X_test)


## === cell 13
models={
    "Logistic Regression":LogisticRegression(),
    "Decision Tree":DecisionTreeClassifier(),
    "Random Forest":RandomForestClassifier(),
    "LGBM":LGBMClassifier()
}

def Modeling(X_data,y_data,Target):
    print('For {} : '.format(Target))
    for i in range(len(list(models))):
        X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.30, random_state=101)
    
        model = list(models.values())[i]
        model.fit(X_train, y_train.values) #Train Model
    
        y_train_pred = model.predict_proba(X_train)[:,1]
        y_test_pred = model.predict_proba(X_val)[:,1]

        model_train_rocauc_score = roc_auc_score(y_train, y_train_pred)

        model_test_rocauc_score = roc_auc_score(y_val, y_test_pred)

        print(list(models.keys())[i])

        print('Model performance for training set')
        print("- Roc Auc Score: {:.4f}".format(model_train_rocauc_score))

        print('-'*30)

        print('Model performance for Test set')
        print("- Roc Auc Score: {:.4f}".format(model_test_rocauc_score))

        print('='*30)
        print('\n')


## === cell 14
Modeling(X,y.EC1,'EC1')


## === cell 15
Modeling(X,y.EC2,'EC2')


## === cell 16
!pip install optuna


## === cell 17
X_train, X_val, y_train, y_val = train_test_split(X, y.EC1, test_size=0.33, random_state=42)


## === cell 18
import optuna
def objective(trial):
    
    n_leaves = trial.suggest_int('num_leaves', 31,100)
    max_depth = trial.suggest_int("max_depth", -1,10)
    n_estimators = trial.suggest_int("n_estimators", 10,2000)
    r_alpha = trial.suggest_int("reg_alpha", 0.0, 0.1)
    r_lambda = trial.suggest_int("reg_lambda", 0.0, 0.1)
    l_rate = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
    subsample =  trial.suggest_uniform('subsample', 0.5, 1.0)
    lambda_l1 = trial.suggest_int("lambda_l1", 0.0, 4)
    lambda_l2 = trial.suggest_int("lambda_l2", 0.0, 4)
    feature_fraction =  trial.suggest_uniform('feature_fraction', 0.5, 1.0)

    lgb = LGBMClassifier(
            num_leaves =n_leaves,
            max_depth=max_depth, 
            n_estimators=n_estimators,
            reg_alpha = r_alpha,
            reg_lambda = r_lambda,
            learning_rate = l_rate,
            subsample = subsample,
            lambda_l1 = lambda_l1,
            lambda_l2 = lambda_l2,
            feature_fraction = feature_fraction
        )
    
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)

    y_pred_proba = lgb.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    return roc_auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
trial1 = study.best_trial


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4218658954.py in <cell line: 0>()
     34 
     35 study = optuna.create_study(direction="maximize")
---> 36 study.optimize(objective, n_trials=20)
     37 trial1 = study.best_trial

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

/tmp/ipykernel_11/4218658954.py in objective(trial)
     26         )
     27 
---> 28     lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
     29 
     30     y_pred_proba = lgb.predict_proba(X_val)[:, 1]

TypeError: LGBMClassifier.fit() got an unexpected keyword argument 'early_stopping_rounds'

## === cell 19
print('For EC1')
print('Auc: {}'.format(trial1.value))
print("Best hyperparameters: {}".format(trial1.params))


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3808658409.py in <cell line: 0>()
      1 print('For EC1')
----> 2 print('Auc: {}'.format(trial1.value))
      3 print("Best hyperparameters: {}".format(trial1.params))

NameError: name 'trial1' is not defined

## === cell 20
X_train, X_val, y_train, y_val = train_test_split(X, y.EC2, test_size=0.33, random_state=42)


## === cell 21
import optuna
def objective(trial):
    
    n_leaves = trial.suggest_int('num_leaves', 31,100)
    max_depth = trial.suggest_int("max_depth", -1,10)
    n_estimators = trial.suggest_int("n_estimators", 10,2000)
    r_alpha = trial.suggest_int("reg_alpha", 0.0, 0.1)
    r_lambda = trial.suggest_int("reg_lambda", 0.0, 0.1)
    l_rate = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
    subsample =  trial.suggest_uniform('subsample', 0.5, 1.0)
    lambda_l1 = trial.suggest_int("lambda_l1", 0.0, 4)
    lambda_l2 = trial.suggest_int("lambda_l2", 0.0, 4)
    feature_fraction =  trial.suggest_uniform('feature_fraction', 0.5, 1.0)

    lgb = LGBMClassifier(
            num_leaves =n_leaves,
            max_depth=max_depth, 
            n_estimators=n_estimators,
            reg_alpha = r_alpha,
            reg_lambda = r_lambda,
            learning_rate = l_rate,
            subsample = subsample,
            lambda_l1 = lambda_l1,
            lambda_l2 = lambda_l2,
            feature_fraction = feature_fraction
        )
    
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)

    y_pred_proba = lgb.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    return roc_auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
trial2 = study.best_trial


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1445424326.py in <cell line: 0>()
     34 
     35 study = optuna.create_study(direction="maximize")
---> 36 study.optimize(objective, n_trials=20)
     37 trial2 = study.best_trial

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

/tmp/ipykernel_11/1445424326.py in objective(trial)
     26         )
     27 
---> 28     lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
     29 
     30     y_pred_proba = lgb.predict_proba(X_val)[:, 1]

TypeError: LGBMClassifier.fit() got an unexpected keyword argument 'early_stopping_rounds'

## === cell 22
print('For EC2')
print('Auc: {}'.format(trial2.value))
print("Best hyperparameters: {}".format(trial2.params))


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2674090689.py in <cell line: 0>()
      1 print('For EC2')
----> 2 print('Auc: {}'.format(trial2.value))
      3 print("Best hyperparameters: {}".format(trial2.params))

NameError: name 'trial2' is not defined

## === cell 23

lgb_clf1  = LGBMClassifier(**trial1.params)
lgb_clf1.fit(X,y['EC1'].values)
pred1 = lgb_clf1.predict_proba(X_test)

lgb_clf2  = LGBMClassifier(**trial2.params)
lgb_clf2.fit(X,y['EC2'].values)
pred2 = lgb_clf2.predict_proba(X_test)
pred2[:,1]


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1780610871.py in <cell line: 0>()
      1 # For EC1
      2 
----> 3 lgb_clf1  = LGBMClassifier(**trial1.params)
      4 lgb_clf1.fit(X,y['EC1'].values)
      5 # Predicting the probabilities of the classes using the model

NameError: name 'trial1' is not defined

## === cell 24
df1 = pd.DataFrame(pred1[:,1],columns=['EC1'])
df2 = pd.DataFrame(pred2[:,1],columns=['EC2'])
df = pd.concat([df1,df2],axis=1,ignore_index=True)
df.columns = ['EC1','EC2']
df


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/885484611.py in <cell line: 0>()
----> 1 df1 = pd.DataFrame(pred1[:,1],columns=['EC1'])
      2 df2 = pd.DataFrame(pred2[:,1],columns=['EC2'])
      3 df = pd.concat([df1,df2],axis=1,ignore_index=True)
      4 df.columns = ['EC1','EC2']
      5 df

NameError: name 'pred1' is not defined

## === cell 25
sub.drop(['EC1','EC2'],axis=1,inplace=True)
sub[['EC1','EC2']]=df.copy()
sub.to_csv('sub_LGBMc.csv', index=False)
sub


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/703869043.py in <cell line: 0>()
      1 # Cappendreating the Data for the submission to competition
      2 sub.drop(['EC1','EC2'],axis=1,inplace=True)
----> 3 sub[['EC1','EC2']]=df.copy()
      4 sub.to_csv('sub_LGBMc.csv', index=False)
      5 sub

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4297             self._setitem_frame(key, value)
   4298         elif isinstance(key, (Series, np.ndarray, list, Index)):
-> 4299             self._setitem_array(key, value)
   4300         elif isinstance(value, DataFrame):
   4301             self._set_item_frame_value(key, value)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _setitem_array(self, key, value)
   4339 
   4340             if isinstance(value, DataFrame):
-> 4341                 check_key_length(self.columns, key, value)
   4342                 for k1, k2 in zip(key, value.columns):
   4343                     self[k1] = value[k2]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexers/utils.py in check_key_length(columns, key, value)
    388     if columns.is_unique:
    389         if len(value.columns) != len(key):
--> 390             raise ValueError("Columns must be same length as key")
    391     else:
    392         # Missing keys in columns are represented as -1

ValueError: Columns must be same length as key

## === cell 26
from lightgbm import plot_importance


## === cell 27
plot_importance(lgb_clf1, figsize=(10, 9));


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2404068093.py in <cell line: 0>()
----> 1 plot_importance(lgb_clf1, figsize=(10, 9));

NameError: name 'lgb_clf1' is not defined

## === cell 28
plot_importance(lgb_clf2, figsize=(10, 9));


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/690271782.py in <cell line: 0>()
----> 1 plot_importance(lgb_clf2, figsize=(10, 9));

NameError: name 'lgb_clf2' is not defined

## === cell 29
X_train, X_val, y_train, y_val = train_test_split(X, y.EC1, test_size=0.33, random_state=42)

def objective(trial):
    params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
        }
    rfc = RandomForestClassifier(**params)
    rfc.fit(X_train, y_train)

    y_pred_proba = rfc.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    return roc_auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
trial1 = study.best_trial


## === cell 30
print('For EC1')
print('Auc: {}'.format(trial1.value))
print("Best hyperparameters: {}".format(trial1.params))


## === cell 31
X_train, X_val, y_train, y_val = train_test_split(X, y.EC2, test_size=0.33, random_state=42)

def objective(trial):
    params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 4, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 60),
        }
    rfc = RandomForestClassifier(**params)
    rfc.fit(X_train, y_train)

    y_pred_proba = rfc.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    return roc_auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
trial2 = study.best_trial


## === cell 32
print('For EC2')
print('Auc: {}'.format(trial2.value))
print("Best hyperparameters: {}".format(trial2.params))


## === cell 33
rfc1  = RandomForestClassifier(**trial1.params)

rfc1.fit(X,y['EC1'].values)
pred1 = rfc1.predict_proba(X_test)

rfc2  = RandomForestClassifier(**trial2.params)
rfc2.fit(X,y['EC2'].values)
pred2 = rfc2.predict_proba(X_test)
pred2[:,1]


## === cell 34
df1 = pd.DataFrame(pred1[:,1],columns=['EC1'])
df2 = pd.DataFrame(pred2[:,1],columns=['EC2'])
df = pd.concat([df1,df2],axis=1,ignore_index=True)
df.columns = ['EC1','EC2']
df


## === cell 35
sub.drop(['EC1','EC2'],axis=1,inplace=True)
sub[['EC1','EC2']]=df.copy()
sub.to_csv('sub_RFc.csv', index=False)
sub


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3791073454.py in <cell line: 0>()
      1 # Cappendreating the Data for the submission to competition
----> 2 sub.drop(['EC1','EC2'],axis=1,inplace=True)
      3 sub[['EC1','EC2']]=df.copy()
      4 sub.to_csv('sub_RFc.csv', index=False)
      5 sub

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   5579                 weight  1.0     0.8
   5580         """
-> 5581         return super().drop(
   5582             labels=labels,
   5583             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   4786         for axis, labels in axes.items():
   4787             if labels is not None:
-> 4788                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
   4789 
   4790         if inplace:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors, only_slice)
   4828                 new_axis = axis.drop(labels, level=level, errors=errors)
   4829             else:
-> 4830                 new_axis = axis.drop(labels, errors=errors)
   4831             indexer = axis.get_indexer(new_axis)
   4832 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
   7068         if mask.any():
   7069             if errors != "ignore":
-> 7070                 raise KeyError(f"{labels[mask].tolist()} not found in axis")
   7071             indexer = indexer[~mask]
   7072         return self.delete(indexer)

KeyError: "['EC1', 'EC2'] not found in axis"

## === cell 36

feature_scores = pd.Series(rfc1.feature_importances_, index=test.columns).sort_values(ascending=False)
feature_scores


## === cell 37

f, ax = plt.subplots(figsize=(30, 24))
ax = sns.barplot(x=feature_scores, y=feature_scores.index, data=train)
ax.set_title("Visualize feature scores of the features")
ax.set_yticklabels(feature_scores.index)
ax.set_xlabel("Feature importance score")
ax.set_ylabel("Features")
plt.show()


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1052862283.py in <cell line: 0>()
      2 
      3 f, ax = plt.subplots(figsize=(30, 24))
----> 4 ax = sns.barplot(x=feature_scores, y=feature_scores.index, data=train)
      5 ax.set_title("Visualize feature scores of the features")
      6 ax.set_yticklabels(feature_scores.index)

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in barplot(data, x, y, hue, order, hue_order, estimator, errorbar, n_boot, units, seed, orient, color, palette, saturation, width, errcolor, errwidth, capsize, dodge, ci, ax, **kwargs)
   2753         estimator = "size"
   2754 
-> 2755     plotter = _BarPlotter(x, y, hue, data, order, hue_order,
   2756                           estimator, errorbar, n_boot, units, seed,
   2757                           orient, color, palette, saturation,

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in __init__(self, x, y, hue, data, order, hue_order, estimator, errorbar, n_boot, units, seed, orient, color, palette, saturation, width, errcolor, errwidth, capsize, dodge)
   1528                  errcolor, errwidth, capsize, dodge):
   1529         """Initialize the plotter."""
-> 1530         self.establish_variables(x, y, hue, data, orient,
   1531                                  order, hue_order, units)
   1532         self.establish_colors(color, palette, saturation)

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in establish_variables(self, x, y, hue, data, orient, order, hue_order, units)
    542 
    543             # Figure out the plotting orientation
--> 544             orient = infer_orient(
    545                 x, y, orient, require_numeric=self.require_numeric
    546             )

/usr/local/lib/python3.11/dist-packages/seaborn/_oldcore.py in infer_orient(x, y, orient, require_numeric)
   1583 
   1584     x_type = None if x is None else variable_type(x)
-> 1585     y_type = None if y is None else variable_type(y)
   1586 
   1587     nonnumeric_dv_error = "{} orientation requires numeric `{}` variable."

/usr/local/lib/python3.11/dist-packages/seaborn/_oldcore.py in variable_type(vector, boolean_type)
   1500 
   1501     # Special-case all-na data, which is always "numeric"
-> 1502     if pd.isna(vector).all():
   1503         return VariableType("numeric")
   1504 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __nonzero__(self)
   1575     @final
   1576     def __nonzero__(self) -> NoReturn:
-> 1577         raise ValueError(
   1578             f"The truth value of a {type(self).__name__} is ambiguous. "
   1579             "Use a.empty, a.bool(), a.item(), a.any() or a.all()."

ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
