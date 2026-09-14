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
pymc3==3.11.4
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

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

-6.8568

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        break



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3139328713.py in <cell line: 0>()
      5 import numpy as np # linear algebra
      6 import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
----> 7 import pymc3 as pm
      8 import seaborn as sns
      9 import matplotlib.pyplot as plt

/usr/local/lib/python3.11/dist-packages/pymc3/__init__.py in <module>
     21 
     22 import semver
---> 23 import theano
     24 
     25 _log = logging.getLogger("pymc3")

/usr/local/lib/python3.11/dist-packages/theano/__init__.py in <module>
    122 from theano.printing import pprint, pp
    123 
--> 124 from theano.scan_module import (scan, map, reduce, foldl, foldr, clone,
    125                                 scan_checkpoints)
    126 

/usr/local/lib/python3.11/dist-packages/theano/scan_module/__init__.py in <module>
     39 __contact__ = "Razvan Pascanu <r.pascanu@gmail>"
     40 
---> 41 from theano.scan_module import scan_opt
     42 from theano.scan_module.scan import scan
     43 from theano.scan_module.scan_checkpoints import scan_checkpoints

/usr/local/lib/python3.11/dist-packages/theano/scan_module/scan_opt.py in <module>
     58 
     59 import theano
---> 60 from theano import tensor, scalar
     61 from theano.tensor import opt, get_scalar_constant_value, Alloc, AllocEmpty
     62 from theano import gof

/usr/local/lib/python3.11/dist-packages/theano/tensor/__init__.py in <module>
      6 import warnings
      7 
----> 8 from theano.tensor.basic import *
      9 from theano.tensor.subtensor import *
     10 from theano.tensor.type_other import *

/usr/local/lib/python3.11/dist-packages/theano/tensor/basic.py in <module>
     18 from theano.gof.type import Generic
     19 
---> 20 from theano.scalar import int32 as int32_t
     21 from theano.tensor import elemwise
     22 from theano.tensor.var import (AsTensorError, TensorVariable,

/usr/local/lib/python3.11/dist-packages/theano/scalar/__init__.py in <module>
      1 from __future__ import absolute_import, print_function, division
      2 
----> 3 from .basic import *
      4 
      5 from .basic_scipy import *

/usr/local/lib/python3.11/dist-packages/theano/scalar/basic.py in <module>
   2368             return s
   2369 
-> 2370 convert_to_bool = Cast(bool, name='convert_to_bool')
   2371 convert_to_int8 = Cast(int8, name='convert_to_int8')
   2372 convert_to_int16 = Cast(int16, name='convert_to_int16')

/usr/local/lib/python3.11/dist-packages/theano/scalar/basic.py in __init__(self, o_type, name)
   2321         super(Cast, self).__init__(specific_out(o_type), name=name)
   2322         self.o_type = o_type
-> 2323         self.ctor = getattr(np, o_type.dtype)
   2324 
   2325     def __str__(self):

/usr/local/lib/python3.11/dist-packages/numpy/__init__.py in __getattr__(attr)
    322 
    323         if attr in __former_attrs__:
--> 324             raise AttributeError(__former_attrs__[attr])
    325 
    326         if attr == 'testing':

AttributeError: module 'numpy' has no attribute 'bool'.
`np.bool` was a deprecated alias for the builtin `bool`. To avoid this error in existing code, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations

## === cell 1
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
train_raw = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

train.drop(train[train.Patient == 'ID00197637202246865691526'].index, inplace=True)



train = pd.concat([train, test], axis=0, ignore_index=True)\
    .drop_duplicates()
le_id = LabelEncoder()
train['PatientID'] = le_id.fit_transform(train['Patient'])

train.head()


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2440061105.py in <cell line: 0>()
     14 train = pd.concat([train, test], axis=0, ignore_index=True)\
     15     .drop_duplicates()
---> 16 le_id = LabelEncoder()
     17 train['PatientID'] = le_id.fit_transform(train['Patient'])
     18 

NameError: name 'LabelEncoder' is not defined

## === cell 2
def add_baselines(data):
    aux = data[['Patient', 'Weeks']].groupby('Patient')\
        .min().reset_index()
    aux = pd.merge(aux, data[['Patient', 'Weeks', 'FVC']], how='left', 
                   on=['Patient', 'Weeks'])
    aux = aux.groupby('Patient').mean().reset_index()
    aux['Weeks'] = aux['Weeks'].astype(int)
    aux['FVC'] = aux['FVC'].astype(int)
    data = pd.merge(data, aux, how='left', on='Patient', suffixes=('', '_base'))
    return data

train = add_baselines(train)
test = add_baselines(test)
train.head()


## === cell 3
def patient_class(row):
    if row['Sex'] == 'Male':
        if row['SmokingStatus'] == 'Currently smokes':
            return 0
        elif row['SmokingStatus'] == 'Ex-smoker':
            return 1
        elif row['SmokingStatus'] == 'Never smoked':
            return 2
    else:
        if row['SmokingStatus'] == 'Currently smokes':
            return 3
        elif row['SmokingStatus'] == 'Ex-smoker':
            return 4
        elif row['SmokingStatus'] == 'Never smoked':
            return 5

train['Class'] = train.apply(patient_class, axis=1)
test['Class'] = test.apply(patient_class, axis=1)
test.head()
train.loc[train['Patient']=='ID00007637202177411956430']['Class'].max()


## === cell 4
PatientID = train['Patient'].values
fvc_b = train.groupby('Patient').first()['FVC_base']
fvc_b.values


## === cell 5
def model_fit(data, examine=True):
    n_patients = data['Patient'].nunique()
    FVC_obs = data['FVC'].values
    Weeks = data['Weeks'].values
    PatientID = data['PatientID'].values
    patient_class = data['Class'].values
    FVC_b = data.groupby('PatientID').first()['FVC_base']
    w_b = data.groupby('PatientID').first()['Weeks_base']

    with pm.Model() as model:
        FVC_obs_shared = pm.Data("FVC_obs_shared", FVC_obs)
        Weeks_shared = pm.Data('Weeks_shared', Weeks)
        PatientID_shared = pm.Data('PatientID_shared', PatientID)
        patient_class_shared = pm.Data('patient_class_shared', patient_class)
        FVC_b_shared = pm.Data('FVC_b_shared', FVC_b)
        w_b_shared= pm.Data('w_b_shared', w_b)

        mu_a_base = pm.Normal('mu_a_base', mu=0., sigma=100)
        mu_a = FVC_b_shared + mu_a_base * w_b_shared
        
        sigma_a = pm.HalfNormal('sigma_a', 1000.)
        mu_b = pm.Normal('mu_b', mu=-4., sigma=1)
        sigma_b = pm.HalfNormal('sigma_b', 5.)

        a = pm.Normal('a', mu=mu_a, sigma=sigma_a, shape=n_patients)
        b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=n_patients)

        sigma = pm.HalfNormal('sigma', 150., shape=6)
        
        FVC_est = a[PatientID_shared] + b[PatientID_shared] * Weeks_shared

        FVC_like = pm.Normal('FVC_like', mu=FVC_est,
                             sigma=sigma[patient_class_shared], observed=FVC_obs_shared)

        trace = pm.sample(4000, tune=4000, target_accept=.9, init="adapt_diag")
        
    if examine:
        with model:
            pm.traceplot(trace);
            
    return model, trace


## === cell 7
def generate_template(data):
    pred_template = []
    for i, patient in enumerate(data['Patient'].unique()):
        df = pd.DataFrame(columns=['PatientID', 'Weeks', 'Patient', 'Class'])
        df['Weeks'] = np.arange(-12, 134)
        df['Patient'] = patient
        df['Class'] = data.loc[data['Patient']==patient]['Class'].max()
        pred_template.append(df)
    pred_template = pd.concat(pred_template, ignore_index=True)
    pred_template['PatientID'] = le_id.transform(pred_template['Patient'])
    return pred_template


## === cell 8
template_train_test = generate_template(test)
template_train_test.head()


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4238986684.py in <cell line: 0>()
----> 1 template_train_test = generate_template(test)
      2 template_train_test.head()

/tmp/ipykernel_11/3640280247.py in generate_template(data)
      8         pred_template.append(df)
      9     pred_template = pd.concat(pred_template, ignore_index=True)
---> 10     pred_template['PatientID'] = le_id.transform(pred_template['Patient'])
     11     return pred_template

NameError: name 'le_id' is not defined

## === cell 10
def model_predict(model, trace, template):
    with model:
        pm.set_data({
            "PatientID_shared": template['PatientID'].values.astype(int),
            "Weeks_shared": template['Weeks'].values.astype(int),
            "FVC_obs_shared": np.zeros(len(template)).astype(int),
            "patient_class_shared": template['Class'].values.astype(int),
        })
        post_pred = pm.sample_posterior_predictive(trace)
    df = pd.DataFrame(columns=['Patient', 'Weeks', 'FVC_pred', 'sigma'])
    df['Patient'] = le_id.inverse_transform(template['PatientID'])
    df['Weeks'] = template['Weeks']
    df['FVC_pred'] = post_pred['FVC_like'].T.mean(axis=1)
    df['sigma'] = post_pred['FVC_like'].T.std(axis=1)
    df['FVC_inf'] = df['FVC_pred'] - df['sigma']
    df['FVC_sup'] = df['FVC_pred'] + df['sigma']
    df = pd.merge(df, train[['Patient', 'Weeks', 'FVC']], how='left', on=['Patient', 'Weeks'])
    df = df.rename(columns={'FVC': 'FVC_true'})
    return df


## === cell 11
def examine_predictions(data):
    n = (data['Patient'].nunique())+1
    f, axes = plt.subplots((n//3)+1, 3, figsize=(15, 5*((n//3)+1)))
    for i, patient in enumerate(data['Patient'].unique()):
        ax = axes[i//3, i%3]
        df = data[data['Patient'] == patient]
        x = df['Weeks']
        ax.set_title(patient)
        ax.plot(x, df['FVC_true'], 'o')
        ax.plot(x, df['FVC_pred'])
        ax = sns.regplot(x, df['FVC_true'], ax=ax, ci=None, line_kws={'color':'red'})
        ax.fill_between(x, df["FVC_inf"], df["FVC_sup"],alpha=0.5, color='#ffcd3c')
        ax.set_ylabel('FVC')
    axes[n//3,n%3].plot()


## === cell 12
def evaluate_predictions(df, use_only_last_3_measures=False, examine=False):
    if use_only_last_3_measures:
        y = df.dropna().groupby('Patient').tail(3)
    else:
        y = df.dropna()

    rmse = ((y['FVC_pred'] - y['FVC_true']) ** 2).mean() ** (1/2)
    mae = (y['FVC_pred'] - y['FVC_true']).abs()
    mae_mean = (np.sqrt((y['FVC_pred'] - y['FVC_true']) ** 2)).mean()
    mae_sd = (np.sqrt((y['FVC_pred'] - y['FVC_true']) ** 2)).mean()
    mae_max = (np.sqrt((y['FVC_pred'] - y['FVC_true']) ** 2)).mean()
    sigma_c = y['sigma'].values*1.5
    sigma_c[sigma_c < 70] = 70
    delta = (y['FVC_pred'] - y['FVC_true']).abs()
    delta[delta > 1000] = 1000
    lll = - np.sqrt(2) * delta / sigma_c - np.log(np.sqrt(2) * sigma_c)

    y['sigma_c'] = y['sigma']
    y['sigma_c'].values[y['sigma_c'].values < 70] = 70
    y['delta_c'] = (y['FVC_pred'] - y['FVC_true']).abs()
    y['delta_c'].values[y['delta_c'].values > 1000] = 1000
    y['main_loss'] = y['delta_c']/y['sigma_c']
    if examine:
        plt.hist(y['main_loss'], bins=100)


    return lll.mean()


## === cell 13
def evaluation_cycle(train, valid, examine_trace=True, examine_preds=True):
    print("Fit model ...")
    model,trace = model_fit(train, examine=examine_trace)
    print("")

    print("Examine true vs predictions for training data ...")
    template_train = generate_template(train)
    template_train.head()
    pred_train = model_predict(model,trace,template_train)
    if examine_preds:
        examine_predictions(pred_train)
    lll_train = evaluate_predictions(pred_train)

    if valid is not None:
        print("Examine true vs predictions for validation data ...")
        template_valid = generate_template(valid)
        pred_valid = model_predict(model,trace,template_valid)
        if examine_preds:
            examine_predictions(pred_valid)
        lll_valid = evaluate_predictions(pred_valid)
    return pred_train, pred_valid, lll_train, lll_valid


## === cell 14
for fold in range(0):
    examine_lll = True

    all_patients = train['Patient'].unique()
    validation_patients = np.random.choice(all_patients, size=20, replace=False)
    df_valid = train[train['Patient'].isin(validation_patients)]
    df_train = train[~train['Patient'].isin(validation_patients)]

    df_valid_first_readings = df_valid.groupby('Patient').head(1)
    df_train = pd.concat([df_train, df_valid_first_readings], axis=0, ignore_index=True)

    print(f'Fold: {fold}')
    pred_train, pred_valid, lll_train, lll_valid = evaluation_cycle(df_train, df_valid, examine_trace=True, examine_preds=True)

    print(f'Laplace Log Likelihoods for fold: {fold}')
    print(f'Training:     {lll_train:.4f}')
    print(f'Validation:   {lll_valid:.4f}')
    print("")

    evaluate_predictions(pred_train, use_only_last_3_measures=True, examine=examine_lll)
    evaluate_predictions(pred_valid, use_only_last_3_measures=True, examine=examine_lll)


## === cell 15
print("Fit model ...")
model,trace = model_fit(train, examine=False)
print("")

print("Make predictions for test data ...")
template_test = generate_template(test)
template_test.head()
pred_test = model_predict(model,trace,template_test)

final = pd.DataFrame(columns=['Patient_Week', 'FVC', 'Confidence'])
final['Patient_Week'] = pred_test['Patient'] + '_' + pred_test['Weeks'].astype(str)
final['FVC'] = pred_test['FVC_pred']
final['Confidence'] = pred_test['sigma']
final.head()
final.to_csv('submission.csv', index=False)
print(final.shape)
final.head()


## --- ERROR in cell 15, traceback:
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

KeyError: 'PatientID'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1740360284.py in <cell line: 0>()
      1 # Fit model using all of training data
      2 print("Fit model ...")
----> 3 model,trace = model_fit(train, examine=False)
      4 print("")
      5 

/tmp/ipykernel_11/3116741053.py in model_fit(data, examine)
      4     Weeks = data['Weeks'].values
      5     #X = train[['Weeks', 'Male', 'ExSmoker', 'CurrentlySmokes', 'Percent_base']].values
----> 6     PatientID = data['PatientID'].values
      7     patient_class = data['Class'].values
      8     FVC_b = data.groupby('PatientID').first()['FVC_base']

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

KeyError: 'PatientID'
