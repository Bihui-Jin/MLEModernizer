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

geopandas==0.14.4
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
statsmodels==0.14.5

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

-7.0589

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 1
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn import linear_model
import statsmodels.api as sm


## === cell 2
BASE_PATH = '../input/osic-pulmonary-fibrosis-progression'

data_train_dir = f'{BASE_PATH}/train'
data_test_dir = f'{BASE_PATH}/test'

train = pd.read_csv(f'{BASE_PATH}/train.csv')
test = pd.read_csv(f'{BASE_PATH}/test.csv')


## === cell 3
sample_submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')


## === cell 4
patient_dict = {}
def init_fvc(row):
    if row['Patient'] not in patient_dict.keys():
        patient_dict[row['Patient']] = row['FVC']
        return row['FVC']
    else:
        return patient_dict[row['Patient']]

train['InitFVC'] = train.apply(lambda row: init_fvc(row), axis=1)
train.head(20)


## === cell 5
patient_dict = {}
def init_week(row):
    if row['Patient'] not in patient_dict.keys():
        patient_dict[row['Patient']] = row['Weeks']
        return row['Weeks']
    else:
        return patient_dict[row['Patient']]

train['InitWeeks'] = train.apply(lambda row: init_week(row), axis=1)
train.head(20)


## === cell 7
patient_dict = {}
def init_percent(row):
    if row['Patient'] not in patient_dict.keys():
        patient_dict[row['Patient']] = row['Percent']
        return row['Percent']
    else:
        return patient_dict[row['Patient']]

train['InitPercent'] = train.apply(lambda row: init_percent(row), axis=1)
train.head(20)


## === cell 8
train_df = pd.get_dummies(train, columns=['Sex', 'SmokingStatus'], prefix=['Sex', 'SmokingStatus'])


## === cell 9
train_df.head()


## === cell 10
test.head()


## === cell 11
data = []
for i in range(-12, 133+1):
    for index, row in test.iterrows():
        new_cols = list(test.columns)
        new_cols.append('InitWeeks')
        new_vals = [row['Patient'], i, row['FVC'], row['Percent'],row['Age'],row['Sex'],row['SmokingStatus'], row['Weeks']]
        data.append(dict(zip(new_cols, new_vals)))
test_df = pd.DataFrame(data)
test_df.head(10)


## === cell 12
X = train[['Weeks']]
Y = train['FVC']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

model = sm.OLS(Y, X).fit()
print(model.summary())


## === cell 13
sns.lmplot(x='Weeks', y='FVC', data=train.sample(frac=0.8))


## === cell 14
X_test = test[['Weeks']]
Y_test = test['FVC']
print('Predicted FVCs: \n', regr.predict(X_test))
print('Actual FVCs: \n', Y_test)


## === cell 15
X = train[['Weeks', 'InitFVC']]
Y = train['FVC']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

model = sm.OLS(Y, X).fit()
print(model.summary())


## === cell 16
sns.lmplot(x='Weeks', y='FVC', hue='InitFVC', data=train.head(98))

display(test.head())


## === cell 17
X_test = test_df[['Weeks', 'FVC']]
Y_test = test_df['FVC']

Y_pred = regr.predict(X_test)
print('Predicted FVCs: \n', Y_pred)
print('Actual FVCs: \n', Y_test)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3452345967.py in <cell line: 0>()
      2 Y_test = test_df['FVC']
      3 
----> 4 Y_pred = regr.predict(X_test)
      5 print('Predicted FVCs: \n', Y_pred)
      6 print('Actual FVCs: \n', Y_test)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    352             Returns predicted values.
    353         """
--> 354         return self._decision_function(X)
    355 
    356     def _set_intercept(self, X_offset, y_offset, X_scale):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in _decision_function(self, X)
    335         check_is_fitted(self)
    336 
--> 337         X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
    338         return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
    339 

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    546             validated.
    547         """
--> 548         self._check_feature_names(X, reset=reset)
    549 
    550         if y is None and self._get_tags()["requires_y"]:

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _check_feature_names(self, X, reset)
    479                 )
    480 
--> 481             raise ValueError(message)
    482 
    483     def _validate_data(

ValueError: The feature names should match those that were passed during fit.
Feature names unseen at fit time:
- FVC
Feature names seen at fit time, yet now missing:
- InitFVC


## === cell 18
X = train[['Weeks', 'InitFVC', 'InitWeeks']]
Y = train['FVC']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

model = sm.OLS(Y, X).fit()
print(model.summary())


## === cell 19
X_test = test_df[['Weeks', 'FVC', 'InitWeeks']]
Y_test = test_df['FVC']

Y_pred = regr.predict(X_test)
print('Predicted FVCs: \n', Y_pred)
print('Actual FVCs: \n', Y_test)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/122492117.py in <cell line: 0>()
      2 Y_test = test_df['FVC']
      3 
----> 4 Y_pred = regr.predict(X_test)
      5 print('Predicted FVCs: \n', Y_pred)
      6 print('Actual FVCs: \n', Y_test)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    352             Returns predicted values.
    353         """
--> 354         return self._decision_function(X)
    355 
    356     def _set_intercept(self, X_offset, y_offset, X_scale):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in _decision_function(self, X)
    335         check_is_fitted(self)
    336 
--> 337         X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
    338         return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
    339 

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    546             validated.
    547         """
--> 548         self._check_feature_names(X, reset=reset)
    549 
    550         if y is None and self._get_tags()["requires_y"]:

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _check_feature_names(self, X, reset)
    479                 )
    480 
--> 481             raise ValueError(message)
    482 
    483     def _validate_data(

ValueError: The feature names should match those that were passed during fit.
Feature names unseen at fit time:
- FVC
Feature names seen at fit time, yet now missing:
- InitFVC


## === cell 20
data = []
for i in range(test_df.shape[0]):
    new_cols = ['Patient', 'Weeks', 'FVC', 'Confidence']
    new_vals = [test_df.iloc[i]['Patient'], test_df.iloc[i]['Weeks'], Y_pred[i], 100]
    data.append(dict(zip(new_cols, new_vals)))
viz = pd.DataFrame(data)
sns.lmplot(x='Weeks', y='FVC', hue='Patient', data=viz)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3112393277.py in <cell line: 0>()
      3 for i in range(test_df.shape[0]):
      4     new_cols = ['Patient', 'Weeks', 'FVC', 'Confidence']
----> 5     new_vals = [test_df.iloc[i]['Patient'], test_df.iloc[i]['Weeks'], Y_pred[i], 100]
      6     data.append(dict(zip(new_cols, new_vals)))
      7 viz = pd.DataFrame(data)

NameError: name 'Y_pred' is not defined

## === cell 21
train.head()


## === cell 22
X = train[['Weeks', 'InitFVC','InitWeeks']]
Y = train['FVC']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

model = sm.OLS(Y, X).fit()
print(model.summary())


## === cell 24
X = train[['Weeks', 'InitFVC', 'InitWeeks', 'Age']]
Y = train['FVC']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

model = sm.OLS(Y, X).fit()
print(model.summary())


## === cell 25
X_test = test_df[['Weeks', 'FVC', 'InitWeeks', 'Age']]
Y_test = test_df['FVC']

Y_pred = regr.predict(X_test)
print('Predicted FVCs: \n', Y_pred)
print('Actual FVCs: \n', Y_test)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/862869069.py in <cell line: 0>()
      2 Y_test = test_df['FVC']
      3 
----> 4 Y_pred = regr.predict(X_test)
      5 print('Predicted FVCs: \n', Y_pred)
      6 print('Actual FVCs: \n', Y_test)

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in predict(self, X)
    352             Returns predicted values.
    353         """
--> 354         return self._decision_function(X)
    355 
    356     def _set_intercept(self, X_offset, y_offset, X_scale):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_base.py in _decision_function(self, X)
    335         check_is_fitted(self)
    336 
--> 337         X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
    338         return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
    339 

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    546             validated.
    547         """
--> 548         self._check_feature_names(X, reset=reset)
    549 
    550         if y is None and self._get_tags()["requires_y"]:

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _check_feature_names(self, X, reset)
    479                 )
    480 
--> 481             raise ValueError(message)
    482 
    483     def _validate_data(

ValueError: The feature names should match those that were passed during fit.
Feature names unseen at fit time:
- FVC
Feature names seen at fit time, yet now missing:
- InitFVC


## === cell 27
train_df.head()


## === cell 28
X = train_df[['Weeks', 'InitFVC', 'InitWeeks', 'Age', 'SmokingStatus_Currently smokes']]
Y = train_df['FVC']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

model = sm.OLS(Y, X).fit()
print(model.summary())


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3406231074.py in <cell line: 0>()
      9 print('Coefficients: \n', regr.coef_)
     10 
---> 11 model = sm.OLS(Y, X).fit()
     12 print(model.summary())

/usr/local/lib/python3.11/dist-packages/statsmodels/regression/linear_model.py in __init__(self, endog, exog, missing, hasconst, **kwargs)
    919                    "An exception will be raised in the next version.")
    920             warnings.warn(msg, ValueWarning)
--> 921         super().__init__(endog, exog, missing=missing,
    922                                   hasconst=hasconst, **kwargs)
    923         if "weights" in self._init_keys:

/usr/local/lib/python3.11/dist-packages/statsmodels/regression/linear_model.py in __init__(self, endog, exog, weights, missing, hasconst, **kwargs)
    744         else:
    745             weights = weights.squeeze()
--> 746         super().__init__(endog, exog, missing=missing,
    747                                   weights=weights, hasconst=hasconst, **kwargs)
    748         nobs = self.exog.shape[0]

/usr/local/lib/python3.11/dist-packages/statsmodels/regression/linear_model.py in __init__(self, endog, exog, **kwargs)
    198     """
    199     def __init__(self, endog, exog, **kwargs):
--> 200         super().__init__(endog, exog, **kwargs)
    201         self.pinv_wexog: Float64Array | None = None
    202         self._data_attr.extend(['pinv_wexog', 'wendog', 'wexog', 'weights'])

/usr/local/lib/python3.11/dist-packages/statsmodels/base/model.py in __init__(self, endog, exog, **kwargs)
    268 
    269     def __init__(self, endog, exog=None, **kwargs):
--> 270         super().__init__(endog, exog, **kwargs)
    271         self.initialize()
    272 

/usr/local/lib/python3.11/dist-packages/statsmodels/base/model.py in __init__(self, endog, exog, **kwargs)
     93         missing = kwargs.pop('missing', 'none')
     94         hasconst = kwargs.pop('hasconst', None)
---> 95         self.data = self._handle_data(endog, exog, missing, hasconst,
     96                                       **kwargs)
     97         self.k_constant = self.data.k_constant

/usr/local/lib/python3.11/dist-packages/statsmodels/base/model.py in _handle_data(self, endog, exog, missing, hasconst, **kwargs)
    133 
    134     def _handle_data(self, endog, exog, missing, hasconst, **kwargs):
--> 135         data = handle_data(endog, exog, missing, hasconst, **kwargs)
    136         # kwargs arrays could have changed, easier to just attach here
    137         for key in kwargs:

/usr/local/lib/python3.11/dist-packages/statsmodels/base/data.py in handle_data(endog, exog, missing, hasconst, **kwargs)
    673 
    674     klass = handle_data_class_factory(endog, exog)
--> 675     return klass(endog, exog=exog, missing=missing, hasconst=hasconst,
    676                  **kwargs)

/usr/local/lib/python3.11/dist-packages/statsmodels/base/data.py in __init__(self, endog, exog, missing, hasconst, **kwargs)
     82             self.orig_endog = endog
     83             self.orig_exog = exog
---> 84             self.endog, self.exog = self._convert_endog_exog(endog, exog)
     85 
     86         self.const_idx = None

/usr/local/lib/python3.11/dist-packages/statsmodels/base/data.py in _convert_endog_exog(self, endog, exog)
    507         exog = exog if exog is None else np.asarray(exog)
    508         if endog.dtype == object or exog is not None and exog.dtype == object:
--> 509             raise ValueError("Pandas data cast to numpy dtype of object. "
    510                              "Check input data with np.asarray(data).")
    511         return super()._convert_endog_exog(endog, exog)

ValueError: Pandas data cast to numpy dtype of object. Check input data with np.asarray(data).

## === cell 29
test.head()


## === cell 30
data = []
for i in range(test_df.shape[0]):
    new_cols = ['Patient_Week', 'FVC', 'Confidence']
    new_vals = [test_df.iloc[i]['Patient']+"_"+str(test_df.iloc[i]['Weeks']), Y_pred[i], 500]
    data.append(dict(zip(new_cols, new_vals)))
submission = pd.DataFrame(data)
submission.head(95)



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/156781268.py in <cell line: 0>()
      3 for i in range(test_df.shape[0]):
      4     new_cols = ['Patient_Week', 'FVC', 'Confidence']
----> 5     new_vals = [test_df.iloc[i]['Patient']+"_"+str(test_df.iloc[i]['Weeks']), Y_pred[i], 500]
      6     data.append(dict(zip(new_cols, new_vals)))
      7 submission = pd.DataFrame(data)

NameError: name 'Y_pred' is not defined

## === cell 31
submission.to_csv('submission.csv', index=False)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/26793137.py in <cell line: 0>()
----> 1 submission.to_csv('submission.csv', index=False)

NameError: name 'submission' is not defined
