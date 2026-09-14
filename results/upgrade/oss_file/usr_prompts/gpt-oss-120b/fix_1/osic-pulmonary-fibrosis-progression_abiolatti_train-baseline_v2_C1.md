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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

-6.8539

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics, model_selection, linear_model

import tensorflow as tf
from tensorflow import keras


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def laplace_likelihood(y, p):
    m, s = p
    diff = np.minimum(1000, np.abs(y - m))
    s = np.maximum(70, s)
    lik = -np.sqrt(2) * diff / s - np.log(np.sqrt(2) * s)
    return np.mean(lik)


def laplace_likelihood_bound(y, p):
    m = p
    diff = np.minimum(1000, np.abs(y - m))
    s = np.maximum(70, np.sqrt(2) * diff)
    lik = -np.sqrt(2) * diff / s - np.log(np.sqrt(2) * s)
    return np.mean(lik)


def laplace_likelihood_avg(y, p):
    m = p
    diff = np.minimum(1000, np.abs(y - m))
    s = np.sqrt(2) * metrics.mean_absolute_error(y, m)
    lik = -np.sqrt(2) * diff / s - np.log(np.sqrt(2) * s)
    return np.mean(lik)


## === cell 2
def build_folds(X, y, group=None, k=5, shuffle=False, train_mask=None, valid_mask=None):
    if isinstance(X, pd.DataFrame): X = X.values
    if isinstance(y, pd.DataFrame): y = y.values
    if isinstance(group, pd.DataFrame): group = group.values
    if isinstance(train_mask, pd.DataFrame): train_mask = train_mask.values
    if isinstance(valid_mask, pd.DataFrame): valid_mask = valid_mask.values
    if group is None:
        folds = list(model_selection.KFold(k, shuffle=True).split(X, y))
    else:
        idx = utils.shuffle(np.arange(X.shape[0]))
        if shuffle:
            Xs, ys, groups = X.copy()[idx], y.copy()[idx], group.copy()[idx]
            folds = list(model_selection.GroupKFold(k).split(np.array(Xs), np.array(ys), np.array(groups)))
        else:
            folds = list(model_selection.GroupKFold(k).split(X, y, group))
    if train_mask is not None:
        for i in range(k):
            folds[i] = (np.array([j for j in folds[i][0] if train_mask[j]]), folds[i][1])
    if valid_mask is not None:
        for i in range(k):
            folds[i] = (folds[i][0], np.array([j for j in folds[i][1] if valid_mask[j]]))
    return folds


## === cell 3
def feature_eng(df):
    df = df.copy()
    df['n_weeks'] = df['Weeks_target'] - df['Weeks_base']
    df['Sex_female'] = (df['Sex'] == 'Female').astype('float')
    df['Smoking_ex'] = (df['SmokingStatus'] == 'Ex-smoker').astype(int)
    df['Smoking_currently'] = (df['SmokingStatus'] == 'Currently smokes').astype(int)
    return df


## === cell 4
data_folder = "../input/osic-pulmonary-fibrosis-progression"


## === cell 5
df_train = pd.read_csv(os.path.join(data_folder, "train.csv")).drop_duplicates(keep=False, subset=['Patient', 'Weeks'])

df_train['n_obs'] = df_train.groupby('Patient').Weeks.cumcount()
df_train['till_last'] = df_train.groupby('Patient').Weeks.transform('count') - 1 - df_train['n_obs']

df_train = df_train.merge(df_train.drop(['Age', 'Sex', 'Percent', 'SmokingStatus'], 1), on='Patient', suffixes=['_base', '_target'])

df_train = feature_eng(df_train)
FEATURES = ['n_weeks', 'FVC_base', 'Percent', 'Age', 'Sex_female', 'Smoking_ex', 'Smoking_currently']

df_train.head(2)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3507208640.py in <cell line: 0>()
      4 df_train['till_last'] = df_train.groupby('Patient').Weeks.transform('count') - 1 - df_train['n_obs']
      5 
----> 6 df_train = df_train.merge(df_train.drop(['Age', 'Sex', 'Percent', 'SmokingStatus'], 1), on='Patient', suffixes=['_base', '_target'])
      7 
      8 df_train = feature_eng(df_train)

TypeError: DataFrame.drop() takes from 1 to 2 positional arguments but 3 were given

## === cell 6
train_mask = (df_train.n_weeks > 0) & df_train.n_obs_base.between(0, 0) & df_train.till_last_target.between(0, 100)
valid_mask = (df_train.n_weeks > 0) & df_train.n_obs_base.between(0, 0) & df_train.till_last_target.between(0, 2)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/419620541.py in <cell line: 0>()
----> 1 train_mask = (df_train.n_weeks > 0) & df_train.n_obs_base.between(0, 0) & df_train.till_last_target.between(0, 100)
      2 valid_mask = (df_train.n_weeks > 0) & df_train.n_obs_base.between(0, 0) & df_train.till_last_target.between(0, 2)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'n_weeks'

## === cell 7
df_test = pd.read_csv(os.path.join(data_folder, "test.csv")).rename(columns={'Weeks': 'Weeks_base', 'FVC': 'FVC_base'})
df_test = df_test.assign(k=0).merge(pd.DataFrame({'Weeks_target': np.arange(-12, 133 + 1), 'k': 0})).drop('k', 1)
df_test = feature_eng(df_test)

df_test.head(2)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1173881857.py in <cell line: 0>()
      1 df_test = pd.read_csv(os.path.join(data_folder, "test.csv")).rename(columns={'Weeks': 'Weeks_base', 'FVC': 'FVC_base'})
----> 2 df_test = df_test.assign(k=0).merge(pd.DataFrame({'Weeks_target': np.arange(-12, 133 + 1), 'k': 0})).drop('k', 1)
      3 df_test = feature_eng(df_test)
      4 
      5 df_test.head(2)

TypeError: DataFrame.drop() takes from 1 to 2 positional arguments but 3 were given

## === cell 8
print(100 * "#")
print("Train: %4d rows with %3d unique patients" % (df_train[train_mask].shape[0], df_train[train_mask].Patient.nunique()))
print("Valid: %4d rows with %3d unique patients" % (df_train[valid_mask].shape[0], df_train[valid_mask].Patient.nunique()))
print("Test:  %4d rows with %3d unique patients" % (df_test.shape[0], df_test.Patient.nunique()))
print(100 * "#")


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/418281989.py in <cell line: 0>()
      1 print(100 * "#")
----> 2 print("Train: %4d rows with %3d unique patients" % (df_train[train_mask].shape[0], df_train[train_mask].Patient.nunique()))
      3 print("Valid: %4d rows with %3d unique patients" % (df_train[valid_mask].shape[0], df_train[valid_mask].Patient.nunique()))
      4 print("Test:  %4d rows with %3d unique patients" % (df_test.shape[0], df_test.Patient.nunique()))
      5 print(100 * "#")

NameError: name 'train_mask' is not defined

## === cell 9
X = df_train[FEATURES].copy()
y = df_train['FVC_target'].copy()
X_test = df_test[FEATURES].copy()


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2754299126.py in <cell line: 0>()
----> 1 X = df_train[FEATURES].copy()
      2 y = df_train['FVC_target'].copy()
      3 X_test = df_test[FEATURES].copy()

NameError: name 'FEATURES' is not defined

## === cell 10
model = linear_model.LinearRegression()
model.fit(X[train_mask], y[train_mask])
pred_mean = model.predict(X)
test_mean = model.predict(X_test)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/763540247.py in <cell line: 0>()
      1 model = linear_model.LinearRegression()
----> 2 model.fit(X[train_mask], y[train_mask])
      3 pred_mean = model.predict(X)
      4 test_mean = model.predict(X_test)

NameError: name 'X' is not defined

## === cell 11
opt_sigma = np.sqrt(2) * np.clip(np.abs(y - pred_mean), 0, 1000)
plt.hist(opt_sigma, 50);


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3300497410.py in <cell line: 0>()
----> 1 opt_sigma = np.sqrt(2) * np.clip(np.abs(y - pred_mean), 0, 1000)
      2 plt.hist(opt_sigma, 50);

NameError: name 'y' is not defined

## === cell 12
model = linear_model.LinearRegression()
model.fit(X[train_mask], opt_sigma[train_mask])
pred_sigma = model.predict(X)
test_sigma = model.predict(X_test)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2274659006.py in <cell line: 0>()
      1 model = linear_model.LinearRegression()
----> 2 model.fit(X[train_mask], opt_sigma[train_mask])
      3 pred_sigma = model.predict(X)
      4 test_sigma = model.predict(X_test)

NameError: name 'X' is not defined

## === cell 13
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.scatter(y[valid_mask], pred_mean[valid_mask], alpha=0.2);
plt.xlim(min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1]))
plt.ylim(min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1]))
plt.plot(plt.xlim(), plt.ylim(), color='tab:red', alpha=0.5, linestyle='--')
plt.subplot(1, 2, 2)
plt.scatter(opt_sigma[valid_mask], pred_sigma[valid_mask], alpha=0.2)
plt.xlim(min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1]))
plt.ylim(min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1]))
plt.plot(plt.xlim(), plt.ylim(), color='tab:red', alpha=0.5, linestyle='--');


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1450568553.py in <cell line: 0>()
      1 plt.figure(figsize=(16, 6))
      2 plt.subplot(1, 2, 1)
----> 3 plt.scatter(y[valid_mask], pred_mean[valid_mask], alpha=0.2);
      4 plt.xlim(min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1]))
      5 plt.ylim(min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1]))

NameError: name 'y' is not defined

## === cell 14
plt.figure(figsize=(16, 8))
for i, pid in enumerate(df_train[valid_mask].Patient.unique()[:12]):
    plt.subplot(3, 4, i + 1)
    idx = (df_train.Patient == pid) & (df_train.n_obs_base == 0)
    plt.fill_between(
        df_train[idx].Weeks_target,
        pred_mean[idx] - 1 * pred_sigma[idx],
        pred_mean[idx] + 1 * pred_sigma[idx],
        color='tab:blue', alpha=0.1
    )
    plt.fill_between(
        df_train[idx].Weeks_target,
        pred_mean[idx] - 2 * pred_sigma[idx],
        pred_mean[idx] + 2 * pred_sigma[idx],
        color='tab:blue', alpha=0.1
    )
    plt.plot(df_train[idx].Weeks_target, pred_mean[idx], marker='x', color='tab:blue')
    plt.plot(df_train[idx].Weeks_target, df_train[idx].FVC_target, marker='o', color='tab:red')


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3838585151.py in <cell line: 0>()
      1 plt.figure(figsize=(16, 8))
----> 2 for i, pid in enumerate(df_train[valid_mask].Patient.unique()[:12]):
      3     plt.subplot(3, 4, i + 1)
      4     idx = (df_train.Patient == pid) & (df_train.n_obs_base == 0)
      5     plt.fill_between(

NameError: name 'valid_mask' is not defined

## === cell 15
submission = df_test.copy()[['Patient', 'Weeks_target']]
submission['Patient_Week'] = submission['Patient'] + "_" + submission['Weeks_target'].astype('str')
submission['FVC'] = test_mean
submission['Confidence'] = test_sigma
submission = submission.sort_values(['Weeks_target', 'Patient'])[['Patient_Week', 'FVC', 'Confidence']]
submission.head()


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1270526938.py in <cell line: 0>()
----> 1 submission = df_test.copy()[['Patient', 'Weeks_target']]
      2 submission['Patient_Week'] = submission['Patient'] + "_" + submission['Weeks_target'].astype('str')
      3 submission['FVC'] = test_mean
      4 submission['Confidence'] = test_sigma
      5 submission = submission.sort_values(['Weeks_target', 'Patient'])[['Patient_Week', 'FVC', 'Confidence']]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

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

KeyError: "['Weeks_target'] not in index"

## === cell 16
submission.to_csv("submission.csv", index=False)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1216870280.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)

NameError: name 'submission' is not defined
