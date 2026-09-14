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

-6.8665

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
        print(os.path.join(dirname, filename))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2534585646.py in <cell line: 0>()
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

exclude_test_patient_data_from_trainset = False

if exclude_test_patient_data_from_trainset:
    train = train[~train['Patient'].isin(test['Patient'].unique())]

train = pd.concat([train, test], axis=0, ignore_index=True)\
    .drop_duplicates()
le_id = LabelEncoder()
train['PatientID'] = le_id.fit_transform(train['Patient'])


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3417498425.py in <cell line: 0>()
     12 train = pd.concat([train, test], axis=0, ignore_index=True)\
     13     .drop_duplicates()
---> 14 le_id = LabelEncoder()
     15 train['PatientID'] = le_id.fit_transform(train['Patient'])

NameError: name 'LabelEncoder' is not defined

## === cell 2
n_patients = train['Patient'].nunique()
FVC_obs = train['FVC'].values
Weeks = train['Weeks'].values
PatientID = train['PatientID'].values

with pm.Model() as model_a:
    FVC_obs_shared = pm.Data("FVC_obs_shared", FVC_obs)
    Weeks_shared = pm.Data('Weeks_shared', Weeks)
    PatientID_shared = pm.Data('PatientID_shared', PatientID)
    
    mu_a = pm.Normal('mu_a', mu=1700., sigma=400)
    sigma_a = pm.HalfNormal('sigma_a', 1000.)
    mu_b = pm.Normal('mu_b', mu=-4., sigma=1)
    sigma_b = pm.HalfNormal('sigma_b', 5.)

    a = pm.Normal('a', mu=mu_a, sigma=sigma_a, shape=n_patients)
    b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=n_patients)

    sigma = pm.HalfNormal('sigma', 150.)

    FVC_est = a[PatientID_shared] + b[PatientID_shared] * Weeks_shared

    FVC_like = pm.Normal('FVC_like', mu=FVC_est,
                         sigma=sigma, observed=FVC_obs_shared)
    
    trace_a = pm.sample(2000, tune=2000, target_accept=.9, init="adapt_diag")


## --- ERROR in cell 2, traceback:
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
/tmp/ipykernel_11/45350986.py in <cell line: 0>()
      2 FVC_obs = train['FVC'].values
      3 Weeks = train['Weeks'].values
----> 4 PatientID = train['PatientID'].values
      5 
      6 with pm.Model() as model_a:

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

## === cell 3
with model_a:
    pm.traceplot(trace_a);


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3197106702.py in <cell line: 0>()
----> 1 with model_a:
      2     pm.traceplot(trace_a);

NameError: name 'model_a' is not defined

## === cell 4
pred_template = []
for i in range(train['Patient'].nunique()):
    df = pd.DataFrame(columns=['PatientID', 'Weeks'])
    df['Weeks'] = np.arange(-12, 134)
    df['PatientID'] = i
    pred_template.append(df)
pred_template = pd.concat(pred_template, ignore_index=True)


## === cell 5
with model_a:
    pm.set_data({
        "PatientID_shared": pred_template['PatientID'].values.astype(int),
        "Weeks_shared": pred_template['Weeks'].values.astype(int),
        "FVC_obs_shared": np.zeros(len(pred_template)).astype(int),
    })
    post_pred = pm.sample_posterior_predictive(trace_a)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/820106220.py in <cell line: 0>()
----> 1 with model_a:
      2     pm.set_data({
      3         "PatientID_shared": pred_template['PatientID'].values.astype(int),
      4         "Weeks_shared": pred_template['Weeks'].values.astype(int),
      5         "FVC_obs_shared": np.zeros(len(pred_template)).astype(int),

NameError: name 'model_a' is not defined

## === cell 6
df = pd.DataFrame(columns=['Patient', 'Weeks', 'FVC_pred', 'sigma'])
df['Patient'] = le_id.inverse_transform(pred_template['PatientID'])
df['Weeks'] = pred_template['Weeks']
df['FVC_pred'] = post_pred['FVC_like'].T.mean(axis=1)
df['sigma'] = post_pred['FVC_like'].T.std(axis=1)
df['FVC_inf'] = df['FVC_pred'] - df['sigma']
df['FVC_sup'] = df['FVC_pred'] + df['sigma']
df = pd.merge(df, train[['Patient', 'Weeks', 'FVC']], how='left', on=['Patient', 'Weeks'])


df = df.rename(columns={'FVC': 'FVC_true'})
df.head()


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3578046108.py in <cell line: 0>()
      1 df = pd.DataFrame(columns=['Patient', 'Weeks', 'FVC_pred', 'sigma'])
----> 2 df['Patient'] = le_id.inverse_transform(pred_template['PatientID'])
      3 df['Weeks'] = pred_template['Weeks']
      4 df['FVC_pred'] = post_pred['FVC_like'].T.mean(axis=1)
      5 df['sigma'] = post_pred['FVC_like'].T.std(axis=1)

NameError: name 'le_id' is not defined

## === cell 7
def chart(patient_id, ax):
    data = df[df['Patient'] == patient_id]
    x = data['Weeks']
    ax.set_title(patient_id)
    ax.plot(x, data['FVC_true'], 'o')
    ax.plot(x, data['FVC_pred'])
    ax = sns.regplot(x, data['FVC_true'], ax=ax, ci=None, 
                     line_kws={'color':'red'})
    ax.fill_between(x, data["FVC_inf"], data["FVC_sup"],
                    alpha=0.5, color='#ffcd3c')
    ax.set_ylabel('FVC')

f, axes = plt.subplots(3, 3, figsize=(15, 10))
chart('ID00007637202177411956430', axes[0, 0])
chart('ID00009637202177434476278', axes[0, 1])
chart('ID00011637202177653955184', axes[0, 2])
chart('ID00419637202311204720264', axes[1, 0])
chart('ID00421637202311550012437', axes[1, 1])
chart('ID00422637202311677017371', axes[1, 2])
chart('ID00423637202312137826377', axes[2, 0])
chart('ID00426637202313170790466', axes[2, 1])


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3940153655.py in <cell line: 0>()
     11     ax.set_ylabel('FVC')
     12 
---> 13 f, axes = plt.subplots(3, 3, figsize=(15, 10))
     14 chart('ID00007637202177411956430', axes[0, 0])
     15 chart('ID00009637202177434476278', axes[0, 1])

NameError: name 'plt' is not defined

## === cell 8

use_only_last_3_measures = True

if use_only_last_3_measures:
    y = df.dropna().groupby('Patient').tail(3)
else:
    y = df.dropna()


rmse = ((y['FVC_pred'] - y['FVC_true']) ** 2).mean() ** (1/2)
mae = (y['FVC_pred'] - y['FVC_true']).abs()
mae_mean = (np.sqrt((y['FVC_pred'] - y['FVC_true']) ** 2)).mean()
mae_sd = (np.sqrt((y['FVC_pred'] - y['FVC_true']) ** 2)).mean()
mae_max = (np.sqrt((y['FVC_pred'] - y['FVC_true']) ** 2)).mean()
print(f'RMSE (mean): {rmse:.1f} ml')
print(f'MAE (mean): {rmse:.1f} ml')
sigma_c = y['sigma'].values
sigma_c[sigma_c < 70] = 70
delta = (y['FVC_pred'] - y['FVC_true']).abs()
delta[delta > 1000] = 1000
lll = - np.sqrt(2) * delta / sigma_c - np.log(np.sqrt(2) * sigma_c)

print(f'Laplace Log Likelihood: {lll.mean():.4f}')
y['sigma_c'] = y['sigma']
y['sigma_c'].values[y['sigma_c'].values < 70] = 70
y['delta_c'] = (y['FVC_pred'] - y['FVC_true']).abs()
y['delta_c'].values[y['delta_c'].values > 1000] = 1000
y['main_loss'] = y['delta_c']/y['sigma_c']
plt.hist(y['main_loss'], bins=100);

bad_patients = y[y['main_loss'] > 2.5]
bad_patients
bad_patients['Patient'].unique()



## --- ERROR in cell 8, traceback:
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

KeyError: 'FVC_true'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1231333093.py in <cell line: 0>()
     11 
     12 
---> 13 rmse = ((y['FVC_pred'] - y['FVC_true']) ** 2).mean() ** (1/2)
     14 mae = (y['FVC_pred'] - y['FVC_true']).abs()
     15 mae_mean = (np.sqrt((y['FVC_pred'] - y['FVC_true']) ** 2)).mean()

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

KeyError: 'FVC_true'

## === cell 9
pred_template = []
for p in test['Patient'].unique():
    df = pd.DataFrame(columns=['PatientID', 'Weeks'])
    df['Weeks'] = np.arange(-12, 134)
    df['Patient'] = p
    pred_template.append(df)
pred_template = pd.concat(pred_template, ignore_index=True)
pred_template['PatientID'] = le_id.transform(pred_template['Patient'])

with model_a:
    pm.set_data({
        "PatientID_shared": pred_template['PatientID'].values.astype(int),
        "Weeks_shared": pred_template['Weeks'].values.astype(int),
        "FVC_obs_shared": np.zeros(len(pred_template)).astype(int),
    })
    post_pred = pm.sample_posterior_predictive(trace_a)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/254973854.py in <cell line: 0>()
      6     pred_template.append(df)
      7 pred_template = pd.concat(pred_template, ignore_index=True)
----> 8 pred_template['PatientID'] = le_id.transform(pred_template['Patient'])
      9 
     10 with model_a:

NameError: name 'le_id' is not defined

## === cell 10
df = pd.DataFrame(columns=['Patient', 'Weeks', 'Patient_Week', 'FVC', 'Confidence'])
df['Patient'] = pred_template['Patient']
df['Weeks'] = pred_template['Weeks']
df['Patient_Week'] = df['Patient'] + '_' + df['Weeks'].astype(str)
df['FVC'] = post_pred['FVC_like'].T.mean(axis=1)
df['Confidence'] = post_pred['FVC_like'].T.std(axis=1)
final = df[['Patient_Week', 'FVC', 'Confidence']]
final.to_csv('submission.csv', index=False)
print(final.shape)
final.head()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/908317693.py in <cell line: 0>()
      3 df['Weeks'] = pred_template['Weeks']
      4 df['Patient_Week'] = df['Patient'] + '_' + df['Weeks'].astype(str)
----> 5 df['FVC'] = post_pred['FVC_like'].T.mean(axis=1)
      6 df['Confidence'] = post_pred['FVC_like'].T.std(axis=1)
      7 final = df[['Patient_Week', 'FVC', 'Confidence']]

NameError: name 'post_pred' is not defined
