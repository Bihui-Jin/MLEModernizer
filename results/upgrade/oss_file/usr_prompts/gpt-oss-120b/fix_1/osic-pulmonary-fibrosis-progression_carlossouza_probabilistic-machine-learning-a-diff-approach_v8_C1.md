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

arviz==0.21.0
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
Theano==1.0.5
Theano-PyMC==1.1.2

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

-6.9569

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
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')


## === cell 2
def chart(patient_id, ax):
    data = train[train['Patient'] == patient_id]
    x = data['Weeks']
    y = data['FVC']
    ax.set_title(patient_id)
    ax = sns.regplot(x, y, ax=ax, ci=None, line_kws={'color':'red'})
    

f, axes = plt.subplots(1, 3, figsize=(15, 5))
chart('ID00007637202177411956430', axes[0])
chart('ID00009637202177434476278', axes[1])
chart('ID00010637202177584971671', axes[2])


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/112568455.py in <cell line: 0>()
      8 
      9 f, axes = plt.subplots(1, 3, figsize=(15, 5))
---> 10 chart('ID00007637202177411956430', axes[0])
     11 chart('ID00009637202177434476278', axes[1])
     12 chart('ID00010637202177584971671', axes[2])

/tmp/ipykernel_11/112568455.py in chart(patient_id, ax)
      4     y = data['FVC']
      5     ax.set_title(patient_id)
----> 6     ax = sns.regplot(x, y, ax=ax, ci=None, line_kws={'color':'red'})
      7 
      8 

TypeError: regplot() takes from 0 to 1 positional arguments but 2 positional arguments (and 3 keyword-only arguments) were given

## === cell 3
import pymc3 as pm
import theano
import arviz as az
from sklearn import preprocessing


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4250646923.py in <cell line: 0>()
      1 # Kaggle, please add Pyro/PyTorch support!
----> 2 import pymc3 as pm
      3 import theano
      4 import arviz as az
      5 from sklearn import preprocessing

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

## === cell 4
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


## === cell 5
aux = train[['Patient', 'Weeks']].groupby('Patient')\
    .min().reset_index()
aux = pd.merge(aux, train[['Patient', 'Weeks', 'FVC']], how='left', 
               on=['Patient', 'Weeks'])
aux = aux.groupby('Patient').mean().reset_index()
aux['Weeks'] = aux['Weeks'].astype(int)
aux['FVC'] = aux['FVC'].astype(int)
train = pd.merge(train, aux, how='left', on='Patient', suffixes=('', '_base'))


## === cell 6
le = preprocessing.LabelEncoder()
train['PatientID'] = le.fit_transform(train['Patient'])

patients = train[['Patient', 'PatientID', 'Age', 'Class', 'Weeks_base', 'FVC_base']].drop_duplicates()
fvc_data = train[['Patient', 'PatientID', 'Weeks', 'FVC']]

patients.head()


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4160654823.py in <cell line: 0>()
      1 # Very simple pre-processing: creating patient indexes
----> 2 le = preprocessing.LabelEncoder()
      3 train['PatientID'] = le.fit_transform(train['Patient'])
      4 
      5 patients = train[['Patient', 'PatientID', 'Age', 'Class', 'Weeks_base', 'FVC_base']].drop_duplicates()

NameError: name 'preprocessing' is not defined

## === cell 7
fvc_data.head()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2355748995.py in <cell line: 0>()
----> 1 fvc_data.head()

NameError: name 'fvc_data' is not defined

## === cell 8
FVC_b = patients['FVC_base'].values
w_b = patients['Weeks_base'].values
age = patients['Age'].values
patient_class = patients['Class'].values

t = fvc_data['Weeks'].values
FVC_obs = fvc_data['FVC'].values
patient_id = fvc_data['PatientID'].values

with pm.Model() as hierarchical_model:
    beta_int = pm.Normal('beta_int', 0, sigma=100)
    sigma_int = pm.HalfNormal('sigma_int', 100)
    
    mu_alpha = FVC_b + beta_int * w_b
    alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_int, 
                      shape=train['Patient'].nunique())
    
    sigma_s = pm.HalfNormal('sigma_s', 100)
    alpha_s = pm.Normal('alpha_s', 0, sigma=100)
    beta_cs = pm.Normal('beta_cs', 0, sigma=100, shape=6)
    
    mu_beta = alpha_s + age * beta_cs[patient_class]
    beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_s,
                     shape=train['Patient'].nunique())
    
    sigma = pm.HalfNormal('sigma', 200)
    
    FVC_est = alpha[patient_id] + beta[patient_id] * t
    
    FVC_like = pm.Normal('FVC_like', mu=FVC_est,
                          sigma=sigma, observed=FVC_obs)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1496809428.py in <cell line: 0>()
----> 1 FVC_b = patients['FVC_base'].values
      2 w_b = patients['Weeks_base'].values
      3 age = patients['Age'].values
      4 patient_class = patients['Class'].values
      5 

NameError: name 'patients' is not defined

## === cell 9
with hierarchical_model:
    trace = pm.sample(2000, tune=2000, target_accept=.9)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1981886986.py in <cell line: 0>()
      1 # Inference button (TM)!
----> 2 with hierarchical_model:
      3     trace = pm.sample(2000, tune=2000, target_accept=.9)

NameError: name 'hierarchical_model' is not defined

## === cell 10
with hierarchical_model:
    pm.traceplot(trace);


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1997399996.py in <cell line: 0>()
----> 1 with hierarchical_model:
      2     pm.traceplot(trace);

NameError: name 'hierarchical_model' is not defined

## === cell 11
def chart(patient_id, ax):
    data = train[train['Patient'] == patient_id]
    x = data['Weeks']
    y = data['FVC']
    ax.set_title(patient_id)
    ax = sns.regplot(x, y, ax=ax, ci=None, line_kws={'color':'red'})
    
    x2 = np.arange(-12, 133, step=0.1)
    
    pid = patients[patients['Patient'] == patient_id]['PatientID'].values[0]
    for sample in range(100):
        alpha = trace['alpha'][sample, pid]
        beta = trace['beta'][sample, pid]
        sigma = trace['sigma'][sample]
        y2 = alpha + beta * x2
        ax.plot(x2, y2, linewidth=0.1, color='green')
        y2 = alpha + beta * x2 + sigma
        ax.plot(x2, y2, linewidth=0.1, color='yellow')
        y2 = alpha + beta * x2 - sigma
        ax.plot(x2, y2, linewidth=0.1, color='yellow')

f, axes = plt.subplots(1, 3, figsize=(15, 5))
chart('ID00007637202177411956430', axes[0])
chart('ID00009637202177434476278', axes[1])
chart('ID00010637202177584971671', axes[2])


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3908857007.py in <cell line: 0>()
     21 
     22 f, axes = plt.subplots(1, 3, figsize=(15, 5))
---> 23 chart('ID00007637202177411956430', axes[0])
     24 chart('ID00009637202177434476278', axes[1])
     25 chart('ID00010637202177584971671', axes[2])

/tmp/ipykernel_11/3908857007.py in chart(patient_id, ax)
      4     y = data['FVC']
      5     ax.set_title(patient_id)
----> 6     ax = sns.regplot(x, y, ax=ax, ci=None, line_kws={'color':'red'})
      7 
      8     x2 = np.arange(-12, 133, step=0.1)

TypeError: regplot() takes from 0 to 1 positional arguments but 2 positional arguments (and 3 keyword-only arguments) were given

## === cell 12
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

test['Class'] = test.apply(patient_class, axis=1)
test = test.rename(columns={'FVC': 'FVC_base', 'Weeks': 'Weeks_base'})
test.head()


## === cell 13
submission = []
for i, patient in enumerate(test['Patient'].unique()):
    df = pd.DataFrame(columns=['Patient', 'Weeks', 'FVC'])
    df['Weeks'] = np.arange(-12, 134)
    df['Patient'] = patient
    df['PatientID'] = i
    df['FVC'] = 0
    submission.append(df)
    
submission = pd.concat(submission).reset_index(drop=True)
submission.head()


## === cell 14
FVC_b = test['FVC_base'].values
w_b = test['Weeks_base'].values
age = test['Age'].values
patient_class = test['Class'].values
t = submission['Weeks'].values
patient_id = submission['PatientID'].values
            
with pm.Model() as new_model:
    beta_int = pm.Normal('beta_int', 
                         trace['beta_int'].mean(), 
                         sigma=trace['beta_int'].std())
    sigma_int = pm.TruncatedNormal('sigma_int', 
                                   trace['sigma_int'].mean(),
                                   sigma=trace['sigma_int'].std(),
                                   lower=0)
    
    mu_alpha = FVC_b + beta_int * w_b
    alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_int, 
                      shape=test['Patient'].nunique())
    
    sigma_s = pm.TruncatedNormal('sigma_s', 
                                 trace['sigma_s'].mean(),
                                 sigma=trace['sigma_s'].std(),
                                 lower=0)
    alpha_s = pm.Normal('alpha_s', 
                        trace['alpha_s'].mean(), 
                        sigma=trace['alpha_s'].std())
    cov = np.zeros((6, 6))
    np.fill_diagonal(cov, trace['beta_cs'].var(axis=0))
    beta_cs = pm.MvNormal('beta_cs',
                          mu=trace['beta_cs'].mean(axis=0),
                          cov=cov,
                          shape=6)
    
    mu_beta = alpha_s + age * beta_cs[patient_class]
    beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_s,
                     shape=test['Patient'].nunique())
    
    sigma = pm.TruncatedNormal('sigma', 
                               trace['sigma'].mean(),
                               sigma=trace['sigma'].std(),
                               lower=0)
    
    FVC_est = pm.Deterministic('FVC_est', alpha[patient_id] + beta[patient_id] * t)
    


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1579933768.py in <cell line: 0>()
      6 patient_id = submission['PatientID'].values
      7 
----> 8 with pm.Model() as new_model:
      9     # Hyperpriors for Alpha
     10     beta_int = pm.Normal('beta_int', 

NameError: name 'pm' is not defined

## === cell 15
with new_model:
    trace2 = pm.sample(2000, tune=2000, target_accept=.9)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3172983120.py in <cell line: 0>()
----> 1 with new_model:
      2     trace2 = pm.sample(2000, tune=2000, target_accept=.9)

NameError: name 'new_model' is not defined

## === cell 16
trace2['FVC_est'].shape


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1901454520.py in <cell line: 0>()
----> 1 trace2['FVC_est'].shape

NameError: name 'trace2' is not defined

## === cell 17
preds = pd.DataFrame(data=trace2['FVC_est'].T)
submission = pd.merge(submission, preds, left_index=True, right_index=True)
submission['Patient_Week'] = submission['Patient'] + '_' + submission['Weeks'].astype(str)
submission = submission.drop(columns=['Patient', 'Weeks', 'FVC', 'PatientID'])

FVC = submission.iloc[:, :-1].mean(axis=1)
confidence = submission.iloc[:, :-1].std(axis=1)
submission['FVC'] = FVC
submission['Confidence'] = confidence
submission = submission[['Patient_Week', 'FVC', 'Confidence']]
submission.to_csv('submission.csv', index=False)
submission.head()


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/362891469.py in <cell line: 0>()
----> 1 preds = pd.DataFrame(data=trace2['FVC_est'].T)
      2 submission = pd.merge(submission, preds, left_index=True, right_index=True)
      3 submission['Patient_Week'] = submission['Patient'] + '_' + submission['Weeks'].astype(str)
      4 submission = submission.drop(columns=['Patient', 'Weeks', 'FVC', 'PatientID'])
      5 

NameError: name 'trace2' is not defined
