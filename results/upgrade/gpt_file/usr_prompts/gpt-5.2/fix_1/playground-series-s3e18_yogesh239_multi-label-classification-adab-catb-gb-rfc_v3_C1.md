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
scikit-multilearn==0.2.0
seaborn==0.12.2
sklearn-pandas==2.2.0
xgboost==2.0.3

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

0.65138

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


## === cell 7
sns.set_style('darkgrid')

plt.figure(figsize=(15,100))
i = 1
for col in train.drop(['EC1','EC2'],axis=1).columns:
    plt.subplot(31,2,i)
    sns.histplot(x=train[col],color='#288BA8',kde=True,lw=1)
    plt.title("training data: distribution of '{}' feature".format(col));
   
    plt.subplot(31,2,i+1)
    sns.histplot(x=test[col],color='#B22222',kde=True,lw=1)
    plt.title("testing data: distribution of '{}' feature".format(col));
    i+=2
plt.tight_layout()


## === cell 8
for df in [train,test]:
    df.drop(['FpDensityMorgan1','FpDensityMorgan2','FpDensityMorgan3'],axis=1,inplace=True)


## === cell 9
plt.figure(figsize=(20,10))
sns.heatmap(data=train.corr(),cmap='Greens')
plt.title('Correlation Matrix for Features of Train Data');
plt.show()


## === cell 10
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
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
    "LGBM":LGBMClassifier(),
    "XGB" :XGBClassifier()
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
import optuna

X_train, X_val, y_train, y_val = train_test_split(X, y.EC1, test_size=0.33, random_state=42)

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


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3349422469.py in <cell line: 0>()
     37 
     38 study = optuna.create_study(direction="maximize")
---> 39 study.optimize(objective, n_trials=20)
     40 trial1 = study.best_trial

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

/tmp/ipykernel_11/3349422469.py in objective(trial)
     29         )
     30 
---> 31     lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
     32 
     33     y_pred_proba = lgb.predict_proba(X_val)[:, 1]

TypeError: LGBMClassifier.fit() got an unexpected keyword argument 'early_stopping_rounds'

## === cell 18
print('For EC1')
print('Auc: {}'.format(trial1.value))
print("Best hyperparameters: {}".format(trial1.params))


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3808658409.py in <cell line: 0>()
      1 print('For EC1')
----> 2 print('Auc: {}'.format(trial1.value))
      3 print("Best hyperparameters: {}".format(trial1.params))

NameError: name 'trial1' is not defined

## === cell 19
X_train, X_val, y_train, y_val = train_test_split(X, y.EC2, test_size=0.33, random_state=42)

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


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/358445483.py in <cell line: 0>()
     35 
     36 study = optuna.create_study(direction="maximize")
---> 37 study.optimize(objective, n_trials=20)
     38 trial2 = study.best_trial

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

/tmp/ipykernel_11/358445483.py in objective(trial)
     27         )
     28 
---> 29     lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
     30 
     31     y_pred_proba = lgb.predict_proba(X_val)[:, 1]

TypeError: LGBMClassifier.fit() got an unexpected keyword argument 'early_stopping_rounds'

## === cell 20
print('For EC2')
print('Auc: {}'.format(trial2.value))
print("Best hyperparameters: {}".format(trial2.params))


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2674090689.py in <cell line: 0>()
      1 print('For EC2')
----> 2 print('Auc: {}'.format(trial2.value))
      3 print("Best hyperparameters: {}".format(trial2.params))

NameError: name 'trial2' is not defined

## === cell 21

lgb_clf1  = LGBMClassifier(**trial1.params)
lgb_clf1.fit(X,y['EC1'].values)
pred1 = lgb_clf1.predict_proba(X_test)

lgb_clf2  = LGBMClassifier(**trial2.params)
lgb_clf2.fit(X,y['EC2'].values)
pred2 = lgb_clf2.predict_proba(X_test)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1430468160.py in <cell line: 0>()
      1 # For EC1
      2 
----> 3 lgb_clf1  = LGBMClassifier(**trial1.params)
      4 lgb_clf1.fit(X,y['EC1'].values)
      5 # Predicting the probabilities of the classes using the model

NameError: name 'trial1' is not defined

## === cell 22
df1 = pd.DataFrame(pred1[:,1],columns=['EC1'])
df2 = pd.DataFrame(pred2[:,1],columns=['EC2'])
df = pd.concat([df1,df2],axis=1,ignore_index=True)
df.columns = ['EC1','EC2']
df


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/885484611.py in <cell line: 0>()
----> 1 df1 = pd.DataFrame(pred1[:,1],columns=['EC1'])
      2 df2 = pd.DataFrame(pred2[:,1],columns=['EC2'])
      3 df = pd.concat([df1,df2],axis=1,ignore_index=True)
      4 df.columns = ['EC1','EC2']
      5 df

NameError: name 'pred1' is not defined

## === cell 23
sub.drop(['EC1','EC2'],axis=1,inplace=True)
sub[['EC1','EC2']]=df.copy()
sub.to_csv('sub_LGBMc.csv', index=False)
sub


## --- ERROR in cell 23, traceback:
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

## === cell 24
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


## === cell 25
print('For EC1')
print('Auc: {}'.format(trial1.value))
print("Best hyperparameters: {}".format(trial1.params))


## === cell 26
X_train, X_val, y_train, y_val = train_test_split(X, y.EC2, test_size=0.33, random_state=42)

def objective(trial):
    params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
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


## === cell 27
print('For EC2')
print('Auc: {}'.format(trial2.value))
print("Best hyperparameters: {}".format(trial2.params))


## === cell 28
rfc1  = RandomForestClassifier(**trial1.params)

rfc1.fit(X,y['EC1'].values)
pred1 = rfc1.predict_proba(X_test)

rfc2  = RandomForestClassifier(**trial2.params)
rfc2.fit(X,y['EC2'].values)
pred2 = rfc2.predict_proba(X_test)
pred2[:,1]


## === cell 29
df1 = pd.DataFrame(pred1[:,1],columns=['EC1'])
df2 = pd.DataFrame(pred2[:,1],columns=['EC2'])
df = pd.concat([df1,df2],axis=1,ignore_index=True)
df.columns = ['EC1','EC2']
df


## === cell 30
sub.drop(['EC1','EC2'],axis=1,inplace=True)
sub[['EC1','EC2']]=df.copy()
sub.to_csv('sub_RFc.csv', index=False)
sub


## --- ERROR in cell 30, traceback:
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

## === cell 31
feature_scores = pd.Series(rfc1.feature_importances_, index=test.columns).sort_values(ascending=False)
feature_scores


## === cell 32

f, ax = plt.subplots(figsize=(40, 20))
ax = sns.barplot(x=feature_scores, y=feature_scores.index)
ax.set_title("Visualize feature scores of the features",fontdict={'fontsize':30})
ax.set_yticklabels(feature_scores.index,fontdict={'fontsize':20})
ax.set_xlabel("Feature importance score",fontdict={'fontsize':30})
ax.set_ylabel("Features",fontdict={'fontsize':30})
plt.show()
