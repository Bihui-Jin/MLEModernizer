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
imbalanced-learn==0.13.0
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
scipy==1.15.3
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

0.61958

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 1

data_set = pd.read_csv("/kaggle/input/playground-series-s3e18/train.csv")
data_set


## === cell 2

data_set.isna().sum()


## === cell 3

del data_set["EC3"], data_set["EC4"],data_set["EC5"], data_set["EC6"], data_set["id"]


## === cell 4

import seaborn as sn
import matplotlib.pyplot as plt

corrmat = data_set.corr().abs()
f, ax = plt.subplots(figsize=(10, 7))
sn.heatmap(corrmat, square=True)


## === cell 5

corrmat.loc["fr_COO","fr_COO2"]


## === cell 6

corrmat.loc["FpDensityMorgan1","FpDensityMorgan2"], corrmat.loc["FpDensityMorgan1","FpDensityMorgan3"], corrmat.loc["FpDensityMorgan2","FpDensityMorgan3"] 


## === cell 7

corrmat.loc["EC1"], corrmat.loc["EC2"] 


## === cell 8

corrmat.loc["HeavyAtomMolWt"]


## === cell 9

variables_to_drop = ['FpDensityMorgan2', 'FpDensityMorgan1', 'fr_COO2', 'HeavyAtomMolWt', 'Chi1', 'Chi1v', 'Chi2v', 'BertzCT', 'Chi4n', 'Chi3v', 'Chi1n', 'Chi2n']
for i in variables_to_drop:
    del data_set[i]


## === cell 10

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler_data = scaler.fit(data_set)
data_set_scaled = pd.DataFrame(scaler_data.transform(data_set), index=data_set.index, columns=data_set.columns)


## === cell 11

y = data_set_scaled[['EC1', 'EC2']].copy() #toma el valor de 1 cuando la variable es si y 0 cuando la variable es no
del data_set_scaled['EC1'], data_set_scaled["EC2"]
X = data_set_scaled
y


## === cell 12

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


## === cell 13

print(y_train.value_counts())


## === cell 14

print(y_train.count(),X_train.count(),y_test.count(),X_train.count())


## === cell 15
X_train.head(20)


## === cell 16

corrmat = X_train.corr().abs()
f, ax = plt.subplots(figsize=(10, 7))
sn.heatmap(corrmat, square=True)


## === cell 17

from imblearn.over_sampling import SMOTE

sm = SMOTE(sampling_strategy='minority') 
X_train_n, y_train_n = sm.fit_resample(X_train, y_train["EC1"])
unique, counts = np.unique(y_train_n, return_counts=True)
dict(zip(unique, counts))


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/367559141.py in <cell line: 0>()
      1 # Oversampling para subsanar diferencias de clases
      2 
----> 3 from imblearn.over_sampling import SMOTE
      4 
      5 sm = SMOTE(sampling_strategy='minority')

/usr/local/lib/python3.11/dist-packages/imblearn/__init__.py in <module>
     50     # process, as it may not be compiled yet
     51 else:
---> 52     from . import (
     53         combine,
     54         ensemble,

/usr/local/lib/python3.11/dist-packages/imblearn/combine/__init__.py in <module>
      3 """
      4 
----> 5 from ._smote_enn import SMOTEENN
      6 from ._smote_tomek import SMOTETomek
      7 

/usr/local/lib/python3.11/dist-packages/imblearn/combine/_smote_enn.py in <module>
     10 from sklearn.utils import check_X_y
     11 
---> 12 from ..base import BaseSampler
     13 from ..over_sampling import SMOTE
     14 from ..over_sampling.base import BaseOverSampler

/usr/local/lib/python3.11/dist-packages/imblearn/base.py in <module>
     10 from sklearn.base import BaseEstimator, OneToOneFeatureMixin
     11 from sklearn.preprocessing import label_binarize
---> 12 from sklearn.utils._metadata_requests import METHODS
     13 from sklearn.utils.multiclass import check_classification_targets
     14 

ModuleNotFoundError: No module named 'sklearn.utils._metadata_requests'

## === cell 18
X_train_n.info(), y_train_n.info()


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2901045124.py in <cell line: 0>()
----> 1 X_train_n.info(), y_train_n.info()

NameError: name 'X_train_n' is not defined

## === cell 19

X_train_n = pd.DataFrame(X_train_n, columns = X_train_n.columns)
X_test = pd.DataFrame(X_test, columns = X_test.columns)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2461931231.py in <cell line: 0>()
      1 # Ajustar el formato
      2 
----> 3 X_train_n = pd.DataFrame(X_train_n, columns = X_train_n.columns)
      4 X_test = pd.DataFrame(X_test, columns = X_test.columns)

NameError: name 'X_train_n' is not defined

## === cell 20

original_data  = xgb.DMatrix(X, label=y["EC1"])
train_data = xgb.DMatrix(X_train_n, label=y_train_n)
test_data = xgb.DMatrix(X_test, label=y_test["EC1"])


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1358453755.py in <cell line: 0>()
      1 # Conversion del conjunto de datos
      2 
----> 3 original_data  = xgb.DMatrix(X, label=y["EC1"])
      4 train_data = xgb.DMatrix(X_train_n, label=y_train_n)
      5 test_data = xgb.DMatrix(X_test, label=y_test["EC1"])

NameError: name 'xgb' is not defined

## === cell 21
import xgboost as xgb 
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV

gbm_random = xgb.XGBClassifier(objective='binary:logistic',
                               eval_metric='auc',
                               n_estimators=100,
                               verbosity=0,
                               random_state=0)
grid_parametros = {
    'max_depth': [1, 3, 6, 12],
    'min_child_weight': [1, 3, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'learning_rate': [0.05, 0.1, 0.5],
    'gamma': [0, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500]
}

xgb_random = RandomizedSearchCV(gbm_random, grid_parametros, scoring='roc_auc', cv=5, n_jobs=-1)
start_1 = datetime.now()
xgb_random.fit(X_train_n, y_train_n)
stop_1 = datetime.now()


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1967067456.py in <cell line: 0>()
     23 xgb_random = RandomizedSearchCV(gbm_random, grid_parametros, scoring='roc_auc', cv=5, n_jobs=-1)
     24 start_1 = datetime.now()
---> 25 xgb_random.fit(X_train_n, y_train_n)
     26 stop_1 = datetime.now()

NameError: name 'X_train_n' is not defined

## === cell 22

execution_time_xgb = stop_1-start_1

print(xgb_random.best_params_, execution_time_xgb)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2837471651.py in <cell line: 0>()
      1 #Execution time: Cuanto se demoró en entrenar el modelo.
      2 
----> 3 execution_time_xgb = stop_1-start_1
      4 
      5 #muestro los mejores parametros

NameError: name 'stop_1' is not defined

## === cell 23
import xgboost as xgb
from sklearn.metrics import roc_auc_score

grid_parametros = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': xgb_random.best_params_['n_estimators'],
    'early_stopping_rounds': 100,
    'verbose': -10,
    'seed': 0,
    'max_depth': xgb_random.best_params_['max_depth'],
    'min_child_weight': xgb_random.best_params_['min_child_weight'],
    'subsample': xgb_random.best_params_['subsample'],
    'colsample_bytree': xgb_random.best_params_['colsample_bytree'],
    'learning_rate': xgb_random.best_params_['learning_rate'],
    'gamma': xgb_random.best_params_['gamma']}

xgb_random_best = xgb.train(grid_parametros, train_data, num_boost_round=xgb_random.best_params_['n_estimators'], 
                            evals=[(test_data, 'test')], early_stopping_rounds=100, verbose_eval=False)

print('Cantidad de rounds óptimo = %d' % xgb_random_best.best_iteration)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/754940020.py in <cell line: 0>()
      7     'objective': 'binary:logistic',
      8     'eval_metric': 'auc',
----> 9     'n_estimators': xgb_random.best_params_['n_estimators'],
     10     'early_stopping_rounds': 100,
     11     'verbose': -10,

AttributeError: 'RandomizedSearchCV' object has no attribute 'best_params_'

## === cell 24

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': xgb_random_best.best_iteration,
    'verbose': -10,
    'seed': 0,
    'max_depth': xgb_random.best_params_['max_depth'],
    'min_child_weight': xgb_random.best_params_['min_child_weight'],
    'subsample': xgb_random.best_params_['subsample'],
    'colsample_bytree': xgb_random.best_params_['colsample_bytree'],
    'learning_rate': xgb_random.best_params_['learning_rate'],
    'gamma': xgb_random.best_params_['gamma']
}

xgb_random_best1 = xgb.train(params, train_data, num_boost_round=xgb_random_best.best_iteration)

y_pred_train = xgb_random_best1.predict(train_data)
auc_xgb_random_train = roc_auc_score(y_train_n, y_pred_train)
print('AUC Train = %.2f' % (auc_xgb_random_train * 100))

y_pred_test = xgb_random_best1.predict(test_data)
auc_xgb_random_test = roc_auc_score(y_test['EC1'], y_pred_test)
print('AUC Test = %.2f' % (auc_xgb_random_test * 100))


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3598665931.py in <cell line: 0>()
      5     'objective': 'binary:logistic',
      6     'eval_metric': 'auc',
----> 7     'n_estimators': xgb_random_best.best_iteration,
      8     'verbose': -10,
      9     'seed': 0,

NameError: name 'xgb_random_best' is not defined

## === cell 25
xgb_EC1 = xgb.train(params, original_data, num_boost_round=xgb_random_best.best_iteration)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/974294239.py in <cell line: 0>()
      1 # Reentrenamos sobre todo el modelo
----> 2 xgb_EC1 = xgb.train(params, original_data, num_boost_round=xgb_random_best.best_iteration)

NameError: name 'params' is not defined

## === cell 26
EC1 = xgb_EC1.predict(original_data)
auc_xgb_random_train = roc_auc_score(y['EC1'], EC1)
print('AUC = %.2f' % (auc_xgb_random_train * 100))


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2158532096.py in <cell line: 0>()
      1 # Predecir la etiqueta EC1
----> 2 EC1 = xgb_EC1.predict(original_data)
      3 auc_xgb_random_train = roc_auc_score(y['EC1'], EC1)
      4 print('AUC = %.2f' % (auc_xgb_random_train * 100))

NameError: name 'xgb_EC1' is not defined

## === cell 27
EC1 = pd.DataFrame(EC1)
EC1 = EC1.rename(columns={0: 'EC1'})


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1622466107.py in <cell line: 0>()
      1 # Los convertimos a df para añadirlos a x
----> 2 EC1 = pd.DataFrame(EC1)
      3 EC1 = EC1.rename(columns={0: 'EC1'})

NameError: name 'EC1' is not defined

## === cell 28

X_EC1 = pd.concat([X, EC1], axis=1)
X_EC1


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2049121465.py in <cell line: 0>()
      1 # Concatenamos EC1 a X.
      2 
----> 3 X_EC1 = pd.concat([X, EC1], axis=1)
      4 X_EC1

NameError: name 'EC1' is not defined

## === cell 29
from sklearn.model_selection import train_test_split

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_EC1, y['EC2'], test_size=0.2, random_state=0)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2878874204.py in <cell line: 0>()
      2 from sklearn.model_selection import train_test_split
      3 
----> 4 X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_EC1, y['EC2'], test_size=0.2, random_state=0)

NameError: name 'X_EC1' is not defined

## === cell 30
X_train_2.info(), X_test_2.info(), y_train_2.info(), y_test_2.info()


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1855807970.py in <cell line: 0>()
----> 1 X_train_2.info(), X_test_2.info(), y_train_2.info(), y_test_2.info()

NameError: name 'X_train_2' is not defined

## === cell 31

sm = SMOTE(sampling_strategy='minority') 
X_train_n_2, y_train_n_2 = sm.fit_resample(X_train_2, y_train_2)
unique, counts = np.unique(y_train_n_2, return_counts=True)
dict(zip(unique, counts))


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2525981920.py in <cell line: 0>()
      1 # Oversampling
      2 
----> 3 sm = SMOTE(sampling_strategy='minority')
      4 X_train_n_2, y_train_n_2 = sm.fit_resample(X_train_2, y_train_2)
      5 unique, counts = np.unique(y_train_n_2, return_counts=True)

NameError: name 'SMOTE' is not defined

## === cell 32

original_data_2  = xgb.DMatrix(X_EC1, label=y["EC2"])
train_data_2 = xgb.DMatrix(X_train_n_2, label=y_train_n_2)
test_data_2 = xgb.DMatrix(X_test_2, label=y_test_2)


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/820304434.py in <cell line: 0>()
      1 # Conversion del conjunto de datos
      2 
----> 3 original_data_2  = xgb.DMatrix(X_EC1, label=y["EC2"])
      4 train_data_2 = xgb.DMatrix(X_train_n_2, label=y_train_n_2)
      5 test_data_2 = xgb.DMatrix(X_test_2, label=y_test_2)

NameError: name 'X_EC1' is not defined

## === cell 33
gbm_random = xgb.XGBClassifier(objective='binary:logistic',
                               eval_metric='auc',
                               n_estimators=100,
                               verbosity=0,
                               random_state=0)
grid_parametros = {
    'max_depth': [1, 3, 6, 12],
    'min_child_weight': [1, 3, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'learning_rate': [0.05, 0.1, 0.5],
    'gamma': [0, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500]
}

xgb_random = RandomizedSearchCV(gbm_random, grid_parametros, scoring='roc_auc', cv=5, n_jobs=-1)
start_1 = datetime.now()
xgb_random.fit(X_train_n_2, y_train_n_2)
stop_1 = datetime.now()


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3492502539.py in <cell line: 0>()
     19 xgb_random = RandomizedSearchCV(gbm_random, grid_parametros, scoring='roc_auc', cv=5, n_jobs=-1)
     20 start_1 = datetime.now()
---> 21 xgb_random.fit(X_train_n_2, y_train_n_2)
     22 stop_1 = datetime.now()

NameError: name 'X_train_n_2' is not defined

## === cell 34

execution_time_xgb = stop_1-start_1

print(xgb_random.best_params_, execution_time_xgb)


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3271453846.py in <cell line: 0>()
      1 # Execution time: Cuanto se demoró en entrenar el modelo.
      2 
----> 3 execution_time_xgb = stop_1-start_1
      4 
      5 # Muestro los mejores parametros

NameError: name 'stop_1' is not defined

## === cell 35
grid_parametros = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': xgb_random.best_params_['n_estimators'],
    'early_stopping_rounds': 100,
    'verbose': -10,
    'seed': 0,
    'max_depth': xgb_random.best_params_['max_depth'],
    'min_child_weight': xgb_random.best_params_['min_child_weight'],
    'subsample': xgb_random.best_params_['subsample'],
    'colsample_bytree': xgb_random.best_params_['colsample_bytree'],
    'learning_rate': xgb_random.best_params_['learning_rate'],
    'gamma': xgb_random.best_params_['gamma']}

xgb_random_best = xgb.train(grid_parametros, train_data_2, num_boost_round=xgb_random.best_params_['n_estimators'], 
                            evals=[(test_data_2, 'test')], early_stopping_rounds=100, verbose_eval=False)

print('Cantidad de rounds óptimo = %d' % xgb_random_best.best_iteration)


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1215703629.py in <cell line: 0>()
      4     'objective': 'binary:logistic',
      5     'eval_metric': 'auc',
----> 6     'n_estimators': xgb_random.best_params_['n_estimators'],
      7     'early_stopping_rounds': 100,
      8     'verbose': -10,

AttributeError: 'RandomizedSearchCV' object has no attribute 'best_params_'

## === cell 36

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': xgb_random_best.best_iteration,
    'verbose': -10,
    'seed': 0,
    'max_depth': xgb_random.best_params_['max_depth'],
    'min_child_weight': xgb_random.best_params_['min_child_weight'],
    'subsample': xgb_random.best_params_['subsample'],
    'colsample_bytree': xgb_random.best_params_['colsample_bytree'],
    'learning_rate': xgb_random.best_params_['learning_rate'],
    'gamma': xgb_random.best_params_['gamma']
}

xgb_EC2 = xgb.train(params, train_data_2, num_boost_round=xgb_random_best.best_iteration)

y_pred_train = xgb_EC2.predict(train_data_2)
auc_xgb_random_train = roc_auc_score(y_train_n_2, y_pred_train)
print('AUC Train = %.2f' % (auc_xgb_random_train * 100))

y_pred_test = xgb_EC2.predict(test_data_2)
auc_xgb_random_test = roc_auc_score(y_test_2, y_pred_test)
print('AUC Test = %.2f' % (auc_xgb_random_test * 100))


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3915600250.py in <cell line: 0>()
      5     'objective': 'binary:logistic',
      6     'eval_metric': 'auc',
----> 7     'n_estimators': xgb_random_best.best_iteration,
      8     'verbose': -10,
      9     'seed': 0,

NameError: name 'xgb_random_best' is not defined

## === cell 37

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime

n_estimators = [50, 100, 125]
max_features = ['sqrt']
max_depth = [5, 10, 15]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

parameters_rfrandom = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_rm = RandomForestClassifier(random_state=0) # definimos un random forest llamado rf_rm
rf_random = RandomizedSearchCV (estimator = rf_rm, param_distributions = parameters_rfrandom, cv = 4,
                                random_state = 0, n_jobs = -1)  # definimos el RandomizedSearchCV


start_1=datetime.now()
rf_random.fit(X_train_n_2, y_train_n_2) # entrenamos el modelo

print('Mejor Configuración de Parámetros')
print('n_estimators: ' +str(rf_random.best_params_['n_estimators']))
print('max_depth: ' +str(rf_random.best_params_['max_depth']))
print('min_samples_leaf: ' +str(rf_random.best_params_['min_samples_leaf']))
print('max_features: ' +str(rf_random.best_params_['max_features']))
print('min_samples_split: ' +str(rf_random.best_params_['min_samples_split']))
print('bootstrap: ' +str(rf_random.best_params_['bootstrap']))


stop_1=datetime.now()
execution_time = stop_1-start_1
print('tiempo de entrenamiento: {}'.format(execution_time))


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/703241887.py in <cell line: 0>()
     32 
     33 start_1=datetime.now()
---> 34 rf_random.fit(X_train_n_2, y_train_n_2) # entrenamos el modelo
     35 
     36 print('Mejor Configuración de Parámetros')

NameError: name 'X_train_n_2' is not defined

## === cell 38

y_pred_test_proba_rf_random = rf_random.predict_proba(X_test_2)[:,1]
auc_rf_random_test = roc_auc_score(y_test_2, y_pred_test_proba_rf_random)
print('AUC Test = %.2f' %(auc_rf_random_test*100))

y_pred_train_proba_rf_random = rf_random.predict_proba(X_train_n_2)[:,1]
auc_rf_random_train = roc_auc_score(y_train_n_2, y_pred_train_proba_rf_random)
print('AUC Train = %.2f' %(auc_rf_random_train*100))


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1247758907.py in <cell line: 0>()
      1 # Calcular el rendimiento
      2 
----> 3 y_pred_test_proba_rf_random = rf_random.predict_proba(X_test_2)[:,1]
      4 auc_rf_random_test = roc_auc_score(y_test_2, y_pred_test_proba_rf_random)
      5 print('AUC Test = %.2f' %(auc_rf_random_test*100))

NameError: name 'X_test_2' is not defined

## === cell 39

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import randint

red = MLPClassifier()

parametros = {
    'max_iter': [50, 100],
    'alpha': [0.1, 0.01, 0.001],
    'solver': ['adam', 'lbfgs'],
    'activation': ['relu', 'logistic'],
    'hidden_layer_sizes': randint(2, 7)
}

red_random = RandomizedSearchCV(red, parametros, n_iter=10, cv=10, random_state=0)
red_random.fit(X_train_n_2, y_train_n_2)

print(red_random.best_params_)


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1233508719.py in <cell line: 0>()
     19 
     20 red_random = RandomizedSearchCV(red, parametros, n_iter=10, cv=10, random_state=0)
---> 21 red_random.fit(X_train_n_2, y_train_n_2)
     22 
     23 print(red_random.best_params_)

NameError: name 'X_train_n_2' is not defined

## === cell 40

red_EC2 = red_random.best_estimator_

y_pred_test_proba_rf_random = red.predict_proba(X_test_2)[:,1]
auc_rf_random_test = roc_auc_score(y_test_2, y_pred_test_proba_rf_random)
print('AUC Test = %.2f' %(auc_rf_random_test*100))

y_pred_train_proba_rf_random = red.predict_proba(X_train_n_2)[:,1]
auc_rf_random_train = roc_auc_score(y_train_n_2, y_pred_train_proba_rf_random)
print('AUC Train = %.2f' %(auc_rf_random_train*100))


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/613013534.py in <cell line: 0>()
      1 # Calculando rendimiento
      2 
----> 3 red_EC2 = red_random.best_estimator_
      4 
      5 y_pred_test_proba_rf_random = red.predict_proba(X_test_2)[:,1]

AttributeError: 'RandomizedSearchCV' object has no attribute 'best_estimator_'

## === cell 41

red_EC2.fit(X_EC1, y['EC2'])


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1253503697.py in <cell line: 0>()
      1 # Decidimos usar este modelo. Reentrenamos sobre toda la muestra.
      2 
----> 3 red_EC2.fit(X_EC1, y['EC2'])

NameError: name 'red_EC2' is not defined

## === cell 42
EC2 = red_EC2.predict_proba(X_EC1)[:,1]
EC2


## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/302162975.py in <cell line: 0>()
      1 # Predecir EC2
----> 2 EC2 = red_EC2.predict_proba(X_EC1)[:,1]
      3 EC2

NameError: name 'red_EC2' is not defined

## === cell 43
auc_rf_random_train = roc_auc_score(y['EC2'], EC2)
print('AUC = %.2f' %(auc_rf_random_train*100))


## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2418878824.py in <cell line: 0>()
      1 # Calculando rendimiento
----> 2 auc_rf_random_train = roc_auc_score(y['EC2'], EC2)
      3 print('AUC = %.2f' %(auc_rf_random_train*100))

NameError: name 'EC2' is not defined

## === cell 44
data_test = pd.read_csv('/kaggle/input/playground-series-s3e18/test.csv')
print (data_test)


## === cell 45
id_var = data_test['id']
id_var


## === cell 46

variables_to_drop = ['id','FpDensityMorgan2', 'FpDensityMorgan1', 'fr_COO2', 'HeavyAtomMolWt', 'Chi1', 'Chi1v', 'Chi2v', 'BertzCT', 'Chi4n', 'Chi3v', 'Chi1n', 'Chi2n']
for i in variables_to_drop:
    del data_test[i]


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler_data = scaler.fit(data_test)
data_test_scaled = pd.DataFrame(scaler_data.transform(data_test), index=data_test.index, columns=data_test.columns)

data_test_scaled


## === cell 47

final_data  = xgb.DMatrix(data_test_scaled)
EC1 = xgb_EC1.predict(final_data)
EC1


## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1785061723.py in <cell line: 0>()
      2 
      3 final_data  = xgb.DMatrix(data_test_scaled)
----> 4 EC1 = xgb_EC1.predict(final_data)
      5 EC1

NameError: name 'xgb_EC1' is not defined

## === cell 48

EC1df = pd.DataFrame(EC1)
EC1df = EC1df.rename(columns={0: 'EC1'})
data_test_scaled = pd.concat([data_test_scaled, EC1df], axis=1)
                     
data_test_scaled


## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/959853787.py in <cell line: 0>()
      1 # Concatenamos EC1 al data test
      2 
----> 3 EC1df = pd.DataFrame(EC1)
      4 EC1df = EC1df.rename(columns={0: 'EC1'})
      5 data_test_scaled = pd.concat([data_test_scaled, EC1df], axis=1)

NameError: name 'EC1' is not defined

## === cell 49

EC2 = red_EC2.predict_proba(data_test_scaled)[:,1]
EC2


## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/887160106.py in <cell line: 0>()
      1 # Predecir EC2
      2 
----> 3 EC2 = red_EC2.predict_proba(data_test_scaled)[:,1]
      4 EC2

NameError: name 'red_EC2' is not defined

## === cell 50

df1 = pd.DataFrame(id_var)
df2 = pd.DataFrame(EC1)
df3 = pd.DataFrame(EC2)

df_combined = pd.concat([df1, df2, df3], axis=1, ignore_index=True)
df_combined = df_combined.rename(columns={0: 'id', 1: 'EC1', 2: 'EC2'})

df_combined.to_csv('/kaggle/working/submission.csv', index=False)

print(df_combined)


## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3252032123.py in <cell line: 0>()
      2 
      3 df1 = pd.DataFrame(id_var)
----> 4 df2 = pd.DataFrame(EC1)
      5 df3 = pd.DataFrame(EC2)
      6 

NameError: name 'EC1' is not defined
