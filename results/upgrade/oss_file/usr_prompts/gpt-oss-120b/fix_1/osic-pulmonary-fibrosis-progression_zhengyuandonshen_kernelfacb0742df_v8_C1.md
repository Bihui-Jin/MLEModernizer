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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 5. Target score

-9.591

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pydicom
import os
from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
from functools import partial
from tqdm.notebook import tqdm

%matplotlib inline


## === cell 1
im_path = "../input/osic-pulmonary-fibrosis-progressiont/"
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
print('Training data shape: ', train_df.shape)
train_df.head()


## === cell 2
train_df['Patient_Week'] = train_df['Patient'].astype(str) + '_' + train_df['Weeks'].astype(str)
output = pd.DataFrame()
gb = train_df.groupby('Patient')
tk0 = tqdm(gb, total=len(gb))
for _, usr_df in tk0:
    usr_output = pd.DataFrame()
    for week, tmp in usr_df.groupby('Weeks'):
        rename_cols = {'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Percent': 'base_Percent', 'Age': 'base_Age'}
        tmp = tmp.drop(columns='Patient_Week').rename(columns=rename_cols)
        drop_cols = ['Age', 'Sex', 'SmokingStatus', 'Percent']
        _usr_output = usr_df.drop(columns=drop_cols).rename(columns={'Weeks': 'predict_Week'}).merge(tmp, on='Patient')
        _usr_output['Week_passed'] = _usr_output['predict_Week'] - _usr_output['base_Week']
        usr_output = pd.concat([usr_output, _usr_output])
    output = pd.concat([output, usr_output])
    
train_df = output[output['Week_passed']!=0].reset_index(drop=True)
print(train_df.shape)
train_df.head()


## === cell 3
train_df = pd.get_dummies(train_df, columns=['Sex'])
train_df = pd.get_dummies(train_df, columns=['SmokingStatus'])
train_df = train_df.rename(columns={"Sex_Female": "Female", 
                                    "Sex_Male": "Male",
                                    "SmokingStatus_Currently smokes": "CurrentlySmokes",
                                    "SmokingStatus_Ex-smoker": "ExSmoker",
                                    "SmokingStatus_Never smoked": "NeverSmoked"})
train_df.head()


## === cell 4
X = train_df.drop(['Patient','FVC','base_Week','predict_Week','Patient_Week'], axis=1)
y = train_df['FVC']


## === cell 5
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle = False)

print('training data has ' + str(X_train.shape[0]) + 
      ' observation with ' + str(X_train.shape[1]) + ' features')
print('test data has ' + str(X_test.shape[0]) + 
      ' observation with ' + str(X_test.shape[1]) + ' features')


## === cell 6

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


## === cell 7
import xgboost as xgb
from xgboost import XGBRegressor
regr_XGB = XGBRegressor()


## === cell 8
regr_XGB.fit(X_train, y_train)


## === cell 9
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV


## === cell 10
regr_XGB_opt = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.8999999999999999, eta=0.01,
             gamma=0, gpu_id=-1, importance_type='gain',
             interaction_constraints='', learning_rate=0.300000012,
             max_delta_step=0, max_depth=5, min_child_weight=1, missing=None,
             monotone_constraints='()', n_estimators=100, n_jobs=0,
             num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
             scale_pos_weight=1, subsample=0.7999999999999999,
             tree_method='exact', validate_parameters=1, verbosity=None)


## === cell 11
regr_XGB_opt.fit(X_train, y_train)
y_pred = regr_XGB_opt.predict(X_test)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_10/2729883573.py in <cell line: 0>()
      1 regr_XGB_opt.fit(X_train, y_train)
----> 2 y_pred = regr_XGB_opt.predict(X_test)

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict(self, X, output_margin, validate_features, base_margin, iteration_range)
   1166             if self._can_use_inplace_predict():
   1167                 try:
-> 1168                     predts = self.get_booster().inplace_predict(
   1169                         data=X,
   1170                         iteration_range=iteration_range,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inplace_predict(self, data, iteration_range, predict_type, missing, validate_features, base_margin, strict_shape)
   2435 
   2436             data, _ = _ensure_np_dtype(data, data.dtype)
-> 2437             _check_call(
   2438                 _LIB.XGBoosterPredictFromDense(
   2439                     self.handle,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [01:09:04] /workspace/include/xgboost/json.h:630: Invalid type for: `missing`, expecting one of the: {``Number`, `Integer`}, got: `Null`
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x15c2ca) [0x7fb095df62ca]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x194354) [0x7fb095e2e354]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x1944b8) [0x7fb095e2e4b8]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x16d5d8) [0x7fb095e075d8]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterPredictFromDense+0x2ab) [0x7fb095e07dcb]
  [bt] (5) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7fb10dfe5e2e]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7fb10dfe2493]
  [bt] (7) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7fb10dff54d8]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7fb10dff4c8e]



## === cell 12
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred,color = 'r', alpha = 0.3)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)], color = 'k')
plt.xlabel('FVC$_{\mathrm{test}}$')
plt.ylabel('FVC$_{\mathrm{pred}}$')
plt.rcParams.update({'font.size': 22})


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1413745637.py in <cell line: 0>()
      1 plt.figure(figsize=(5,5))
----> 2 plt.scatter(y_test, y_pred,color = 'r', alpha = 0.3)
      3 plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)], color = 'k')
      4 plt.xlabel('FVC$_{\mathrm{test}}$')
      5 plt.ylabel('FVC$_{\mathrm{pred}}$')

NameError: name 'y_pred' is not defined

## === cell 13
from sklearn.metrics import mean_squared_error
mse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: %f" % (mse**0.5))


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3119774233.py in <cell line: 0>()
      1 from sklearn.metrics import mean_squared_error
----> 2 mse = np.sqrt(mean_squared_error(y_test, y_pred))
      3 print("RMSE: %f" % (mse**0.5))

NameError: name 'y_pred' is not defined

## === cell 14
data_dmatrix = xgb.DMatrix(data=X,label=y)
params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=10,
                    num_boost_round=200,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
print((cv_results["test-rmse-mean"]).tail(1))


## === cell 15
importances = regr_XGB.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature importance ranking by XGBoost Model:")
for ind in range(X.shape[1]):
    print ("%s : %.4f" %(X.columns[indices[ind]],importances[indices[ind]]))


## === cell 16
import seaborn as sns
fig, ax = plt.subplots(figsize=(15,13)) 
sns.heatmap(X.corr(), annot = True, fmt = '.2f')


## === cell 17
FVC_pred_train = regr_XGB_opt.predict(X_train)
train_df = pd.DataFrame(X_train, columns = X.columns)
train_df['FVC'] = y_train
train_df['FVC_pred'] = FVC_pred_train

FVC_pred_test = regr_XGB_opt.predict(X_test)
test_df = pd.DataFrame(X_test, columns = X.columns)
test_df['FVC'] = np.asarray(y_test)
test_df['FVC_pred'] = FVC_pred_test


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_10/1644774887.py in <cell line: 0>()
----> 1 FVC_pred_train = regr_XGB_opt.predict(X_train)
      2 train_df = pd.DataFrame(X_train, columns = X.columns)
      3 train_df['FVC'] = y_train
      4 train_df['FVC_pred'] = FVC_pred_train
      5 

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict(self, X, output_margin, validate_features, base_margin, iteration_range)
   1166             if self._can_use_inplace_predict():
   1167                 try:
-> 1168                     predts = self.get_booster().inplace_predict(
   1169                         data=X,
   1170                         iteration_range=iteration_range,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inplace_predict(self, data, iteration_range, predict_type, missing, validate_features, base_margin, strict_shape)
   2435 
   2436             data, _ = _ensure_np_dtype(data, data.dtype)
-> 2437             _check_call(
   2438                 _LIB.XGBoosterPredictFromDense(
   2439                     self.handle,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [01:09:06] /workspace/include/xgboost/json.h:630: Invalid type for: `missing`, expecting one of the: {``Number`, `Integer`}, got: `Null`
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x15c2ca) [0x7fb095df62ca]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x194354) [0x7fb095e2e354]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x1944b8) [0x7fb095e2e4b8]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x16d5d8) [0x7fb095e075d8]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterPredictFromDense+0x2ab) [0x7fb095e07dcb]
  [bt] (5) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7fb10dfe5e2e]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7fb10dfe2493]
  [bt] (7) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7fb10dff54d8]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7fb10dff4c8e]



## === cell 18
train_df['Confidence'] = 100
train_df['sigma_clipped'] = train_df['Confidence'].apply(lambda x: max(x, 70))
train_df['diff'] = abs(train_df['FVC'] - train_df['FVC_pred'])
train_df['delta'] = train_df['diff'].apply(lambda x: min(x, 1000))
train_df['score'] = -2**0.5*train_df['delta']/train_df['sigma_clipped'] - np.log(2**0.5*train_df['sigma_clipped'])
score = train_df['score'].mean()
print(score)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'FVC_pred'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_10/630656243.py in <cell line: 0>()
      2 train_df['Confidence'] = 100
      3 train_df['sigma_clipped'] = train_df['Confidence'].apply(lambda x: max(x, 70))
----> 4 train_df['diff'] = abs(train_df['FVC'] - train_df['FVC_pred'])
      5 train_df['delta'] = train_df['diff'].apply(lambda x: min(x, 1000))
      6 train_df['score'] = -2**0.5*train_df['delta']/train_df['sigma_clipped'] - np.log(2**0.5*train_df['sigma_clipped'])

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'FVC_pred'

## === cell 19
def loss_func(weight, row):
    confidence = weight
    sigma_clipped = max(confidence, 70)
    diff = abs(row['FVC'] - row['FVC_pred'])
    delta = min(diff, 1000)
    score = -2**0.5*delta/sigma_clipped - np.log(2**0.5*sigma_clipped)
    return -score

results = []
tk0 = tqdm(test_df.iterrows(), total=len(test_df))
for _, row in tk0:
    loss_partial = partial(loss_func, row=row)
    weight = [100]
    result = sp.optimize.minimize(loss_partial, weight, method='SLSQP')
    x = result['x']
    results.append(x[0])


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'FVC_pred'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_10/2819668656.py in <cell line: 0>()
     14     #bounds = [(70, 100)]
     15     #result = sp.optimize.minimize(loss_partial, weight, method='SLSQP', bounds=bounds)
---> 16     result = sp.optimize.minimize(loss_partial, weight, method='SLSQP')
     17     x = result['x']
     18     results.append(x[0])

/usr/local/lib/python3.11/dist-packages/scipy/optimize/_minimize.py in minimize(fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
    748                                **options)
    749     elif meth == 'slsqp':
--> 750         res = _minimize_slsqp(fun, x0, args, jac, bounds,
    751                               constraints, callback=callback, **options)
    752     elif meth == 'trust-constr':

/usr/local/lib/python3.11/dist-packages/scipy/optimize/_slsqp_py.py in _minimize_slsqp(func, x0, args, jac, bounds, constraints, maxiter, ftol, iprint, disp, eps, callback, finite_diff_rel_step, **unknown_options)
    379 
    380     # ScalarFunction provides function and gradient evaluation
--> 381     sf = _prepare_scalar_function(func, x, jac=jac, args=args, epsilon=eps,
    382                                   finite_diff_rel_step=finite_diff_rel_step,
    383                                   bounds=new_bounds)

/usr/local/lib/python3.11/dist-packages/scipy/optimize/_optimize.py in _prepare_scalar_function(fun, x0, jac, args, bounds, epsilon, finite_diff_rel_step, hess)
    289     # ScalarFunction caches. Reuse of fun(x) during grad
    290     # calculation reduces overall function evaluations.
--> 291     sf = ScalarFunction(fun, x0, args, grad, hess,
    292                         finite_diff_rel_step, bounds, epsilon=epsilon)
    293 

/usr/local/lib/python3.11/dist-packages/scipy/optimize/_differentiable_functions.py in __init__(self, fun, x0, args, grad, hess, finite_diff_rel_step, finite_diff_bounds, epsilon)
    221 
    222         # Initial function evaluation
--> 223         self._update_fun()
    224 
    225         # Initial gradient evaluation

/usr/local/lib/python3.11/dist-packages/scipy/optimize/_differentiable_functions.py in _update_fun(self)
    293     def _update_fun(self):
    294         if not self.f_updated:
--> 295             fx = self._wrapped_fun(self.x)
    296             if fx < self._lowest_f:
    297                 self._lowest_x = self.x

/usr/local/lib/python3.11/dist-packages/scipy/optimize/_differentiable_functions.py in wrapped(x)
     19         # Overwriting results in undefined behaviour because
     20         # fun(self.x) will change self.x, with the two no longer linked.
---> 21         fx = fun(np.copy(x), *args)
     22         # Make sure the function returns a true scalar
     23         if not np.isscalar(fx):

/tmp/ipykernel_10/2819668656.py in loss_func(weight, row)
      2     confidence = weight
      3     sigma_clipped = max(confidence, 70)
----> 4     diff = abs(row['FVC'] - row['FVC_pred'])
      5     delta = min(diff, 1000)
      6     score = -2**0.5*delta/sigma_clipped - np.log(2**0.5*sigma_clipped)

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1119 
   1120         elif key_is_scalar:
-> 1121             return self._get_value(key)
   1122 
   1123         # Convert generator to list before going through hashable part

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1235 
   1236         # Similar to Index.get_value, but we do not fall back to positional
-> 1237         loc = self.index.get_loc(label)
   1238 
   1239         if is_integer(loc):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'FVC_pred'

## === cell 20
test_df['Confidence'] = results
test_df['sigma_clipped'] = test_df['Confidence'].apply(lambda x: max(x, 70))
test_df['diff'] = abs(test_df['FVC'] - test_df['FVC_pred'])
test_df['delta'] = test_df['diff'].apply(lambda x: min(x, 1000))
test_df['score'] = -2**0.5*test_df['delta']/test_df['sigma_clipped'] - np.log(2**0.5*test_df['sigma_clipped'])
score = test_df['score'].mean()
print(score)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_10/2770245919.py in <cell line: 0>()
----> 1 test_df['Confidence'] = results
      2 test_df['sigma_clipped'] = test_df['Confidence'].apply(lambda x: max(x, 70))
      3 test_df['diff'] = abs(test_df['FVC'] - test_df['FVC_pred'])
      4 test_df['delta'] = test_df['diff'].apply(lambda x: min(x, 1000))
      5 test_df['score'] = -2**0.5*test_df['delta']/test_df['sigma_clipped'] - np.log(2**0.5*test_df['sigma_clipped'])

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (0) does not match length of index (18)

## === cell 21
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')\
        .rename(columns={'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Percent': 'base_Percent', 'Age': 'base_Age'})
submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
submission['Patient'] = submission['Patient_Week'].apply(lambda x: x.split('_')[0])
submission['predict_Week'] = submission['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)
test = submission.drop(columns=['FVC', 'Confidence']).merge(test, on='Patient')
test['Week_passed'] = test['predict_Week'] - test['base_Week']
print(test.shape)
test.head()


## === cell 22
test = pd.get_dummies(test, columns=['Sex'])
test = pd.get_dummies(test, columns=['SmokingStatus'])
test = test.rename(columns={"Sex_Female": "Female", 
                                    "Sex_Male": "Male",
                                    "SmokingStatus_Currently smokes": "CurrentlySmokes",
                                    "SmokingStatus_Ex-smoker": "ExSmoker",
                                    "SmokingStatus_Never smoked": "NeverSmoked"})


## === cell 23
submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
submission


## === cell 24
submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
submission['Patient'] = submission['Patient_Week'].apply(lambda x: x.split('_')[0])
submission['predict_Week'] = submission['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)
submission.head()


## === cell 25
sub = submission.drop(columns=['Patient','predict_Week','FVC', 'Confidence']).merge(test, on='Patient_Week')
sub.to_csv('submission.csv', index=False)
sub['Female'] = 0
sub['CurrentlySmokes'] = 0
sub = sub[['Patient_Week', 'Patient', 'predict_Week', 'base_Week', 'base_FVC', 'base_Percent', 'base_Age', 
           'Week_passed', 'Female', 'Male', 'CurrentlySmokes', 'ExSmoker', 'NeverSmoked']]


## === cell 26
X_test_sub = sub.iloc[:,4:]
scaler = MinMaxScaler()
scaler.fit(X_test_sub)
X_test_sub = scaler.transform(X_test_sub)
FVC_pred_sub = regr_XGB_opt.predict(X_test_sub)
sub['FVC_pred'] = FVC_pred_sub 
sub


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_10/4244899723.py in <cell line: 0>()
      3 scaler.fit(X_test_sub)
      4 X_test_sub = scaler.transform(X_test_sub)
----> 5 FVC_pred_sub = regr_XGB_opt.predict(X_test_sub)
      6 sub['FVC_pred'] = FVC_pred_sub
      7 sub

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict(self, X, output_margin, validate_features, base_margin, iteration_range)
   1166             if self._can_use_inplace_predict():
   1167                 try:
-> 1168                     predts = self.get_booster().inplace_predict(
   1169                         data=X,
   1170                         iteration_range=iteration_range,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inplace_predict(self, data, iteration_range, predict_type, missing, validate_features, base_margin, strict_shape)
   2435 
   2436             data, _ = _ensure_np_dtype(data, data.dtype)
-> 2437             _check_call(
   2438                 _LIB.XGBoosterPredictFromDense(
   2439                     self.handle,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [01:09:06] /workspace/include/xgboost/json.h:630: Invalid type for: `missing`, expecting one of the: {``Number`, `Integer`}, got: `Null`
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x15c2ca) [0x7fb095df62ca]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x194354) [0x7fb095e2e354]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x1944b8) [0x7fb095e2e4b8]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x16d5d8) [0x7fb095e075d8]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterPredictFromDense+0x2ab) [0x7fb095e07dcb]
  [bt] (5) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7fb10dfe5e2e]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7fb10dfe2493]
  [bt] (7) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7fb10dff54d8]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7fb10dff4c8e]



## === cell 27
for pid in sub['Patient'].unique():
    temp = sub[sub['Patient'] == pid]
    plt.plot(temp['predict_Week'], temp['FVC_pred'])


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'FVC_pred'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_10/996374976.py in <cell line: 0>()
      2 # pid = 'ID00426637202313170790466'
      3     temp = sub[sub['Patient'] == pid]
----> 4     plt.plot(temp['predict_Week'], temp['FVC_pred'])

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'FVC_pred'

## === cell 28
attempt1 = submission.merge(sub, on='Patient_Week')
attempt1 = attempt1.loc[:,['Patient_Week','FVC_pred','Confidence']]
attempt1.columns = ['Patient_Week','FVC','Confidence']
attempt1['Confidence'] = 10000
attempt1.to_csv('submission.csv', index=False)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_10/3805255764.py in <cell line: 0>()
      1 attempt1 = submission.merge(sub, on='Patient_Week')
----> 2 attempt1 = attempt1.loc[:,['Patient_Week','FVC_pred','Confidence']]
      3 attempt1.columns = ['Patient_Week','FVC','Confidence']
      4 attempt1['Confidence'] = 10000
      5 attempt1.to_csv('submission.csv', index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1375             return self._multi_take(tup)
   1376 
-> 1377         return self._getitem_tuple_same_dim(tup)
   1378 
   1379     def _get_label(self, label, axis: AxisInt):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple_same_dim(self, tup)
   1018                 continue
   1019 
-> 1020             retval = getattr(retval, self.name)._getitem_axis(key, axis=i)
   1021             # We should never have retval.ndim < self.ndim, as that should
   1022             #  be handled by the _getitem_lowerdim call above.

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['FVC_pred'] not in index"

## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame must have a 'FVC' column.
