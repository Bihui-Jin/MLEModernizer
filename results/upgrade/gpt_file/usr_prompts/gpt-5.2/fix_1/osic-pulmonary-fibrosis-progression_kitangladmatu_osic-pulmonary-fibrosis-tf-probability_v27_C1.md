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
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tqdm==4.67.1

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

-7.91066025917951

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
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import seaborn as sns
import pydicom
import os
import re
import time
import gzip
from tqdm import tqdm
import seaborn as sns
import sklearn.preprocessing
tfd = tfp.distributions
tfb = tfp. bijectors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import ct_scan_processing2_module as ct
import osic_utils
from tqdm import tqdm_notebook as tqdm

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
lung_stats = pd.read_csv('../usr/lib/ct_scan_processing2/lung_statistics.csv')
cols = lung_stats.columns[~pd.Series(lung_stats.columns).isin(['PatientId'])]
row, col_ = (3,2)
fig, ax = plt.subplots(row, col_, figsize = (15,9))
plt.subplots_adjust(bottom = -0.4)
for i, col in enumerate(cols):
    ridx = int(i/col_)
    cidx = i - col_ * ridx      
    sns.distplot(lung_stats[col], ax = ax[ridx][cidx])
    ax[ridx][cidx].set_xlabel('')
    ax[ridx][cidx].set_title(col)
fig.delaxes(ax[row - 1][col_ - 1])


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1105832452.py in <cell line: 0>()
----> 1 lung_stats = pd.read_csv('../usr/lib/ct_scan_processing2/lung_statistics.csv')
      2 cols = lung_stats.columns[~pd.Series(lung_stats.columns).isin(['PatientId'])]
      3 row, col_ = (3,2)
      4 fig, ax = plt.subplots(row, col_, figsize = (15,9))
      5 plt.subplots_adjust(bottom = -0.4)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../usr/lib/ct_scan_processing2/lung_statistics.csv'

## === cell 3
lung_stats.describe()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1322847697.py in <cell line: 0>()
----> 1 lung_stats.describe()

NameError: name 'lung_stats' is not defined

## === cell 4
patient_dcm_dict = {}
root_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"

remove_ids = []
for dirname, _, filenames in os.walk(root_dir):
    if 'ID' in dirname:
        dirname_ = dirname.replace(root_dir, "")
        s = [i for i in remove_ids if dirname_ == i]
        if len(s) == 0:            
            patient_dcm_dict[dirname_] = filenames
        
patient_dcm_dict = {k: i for i, k in enumerate(sorted(patient_dcm_dict.keys()))}
print("A total of", len(patient_dcm_dict.keys()), "patients")

## === cell 5
patient_dcm_dict = {k: i for i, k in enumerate(sorted(patient_dcm_dict.keys()))}
{k: v for i, (k, v) in enumerate(patient_dcm_dict.items()) if i < 5}

## === cell 7
lung_stats = lung_stats.set_index('PatientId') \
.loc[list(patient_dcm_dict.keys())] \
.reset_index()

lung_stats['lung_vol'] = lung_stats['lung_vol']/10000
lung_stats.head()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1427062542.py in <cell line: 0>()
----> 1 lung_stats = lung_stats.set_index('PatientId') \
      2 .loc[list(patient_dcm_dict.keys())] \
      3 .reset_index()
      4 
      5 lung_stats['lung_vol'] = lung_stats['lung_vol']/10000

NameError: name 'lung_stats' is not defined

## === cell 11
train_data = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
test_data = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")

oh_encoder = OneHotEncoder(sparse = False, dtype = 'int32')
oh_encoder.fit(train_data['SmokingStatus'].values.reshape(-1, 1))

age_scaler = StandardScaler()
age_scaler.fit(train_data[['Patient', 'Age']].drop_duplicates()['Age'] \
               .values.reshape(-1, 1))
week_scaler = StandardScaler()
week_scaler.fit(train_data[['Patient', 'Weeks']].drop_duplicates()['Weeks'] \
               .values.reshape(-1,1))
pct_scaler = StandardScaler()
pct_scaler.fit(train_data['Percent'].values.reshape(-1,1))
baseline_scaler = StandardScaler()
baseline_scaler.fit(train_data['FVC'].values.reshape(-1,1))

encoder_dict = {
    'Age': age_scaler,
    'Weeks': week_scaler,
    'pct_bsl': pct_scaler,
    'baseline': baseline_scaler,
    'SmokingStatus': oh_encoder
}
X, pid, lbl = osic_utils.preprocess_structured_data(
    train_data, 
    patient_dcm_dict, 
    encoder_dict)
print(X.shape)
print(pid.shape)
print(lbl.shape)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3717941491.py in <cell line: 0>()
     30     'SmokingStatus': oh_encoder
     31 }
---> 32 X, pid, lbl = osic_utils.preprocess_structured_data(
     33     train_data,
     34     patient_dcm_dict,

NameError: name 'osic_utils' is not defined

## === cell 13
lung_stats_train = tf.gather(lung_stats.iloc[:,1:].values, 
                             tf.constant(pid, dtype = tf.int32), axis = 0)
X = tf.concat([X,lung_stats_train], axis = 1)
X

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2354458772.py in <cell line: 0>()
----> 1 lung_stats_train = tf.gather(lung_stats.iloc[:,1:].values, 
      2                              tf.constant(pid, dtype = tf.int32), axis = 0)
      3 X = tf.concat([X,lung_stats_train], axis = 1)
      4 X

NameError: name 'lung_stats' is not defined

## === cell 14
holdout = osic_utils.transform_holdout_data(test_data)

patient_dcm_holdout = {k: i for i, k in enumerate(holdout.Patient.unique())}
X_holdout, pid_holdout, lbl_holdout = osic_utils.preprocess_structured_data(holdout, 
                                                         patient_dcm_holdout,
                                                         encoder_dict)
print(X_holdout.shape)
print(pid_holdout.shape)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2145816821.py in <cell line: 0>()
----> 1 holdout = osic_utils.transform_holdout_data(test_data)
      2 
      3 # there are no actual labels.
      4 #lbl_holdout is just there as placeholder for consistency with train tf dataset generation
      5 patient_dcm_holdout = {k: i for i, k in enumerate(holdout.Patient.unique())}

NameError: name 'osic_utils' is not defined

## === cell 15
dataset = osic_utils.DatasetGen(patient_dcm_dict, split = None, seed = 300, 
                         batch_size = 1, 
                         root_dir = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/',
                         shuffle = False)

"""
holdout_ds = osic_utils.DatasetGen(patient_dcm_holdout, 
                                   split = None,
                                   seed = 300, batch_size =1,
                                   root_dir = '/kaggle/input/osic-pulmonary-fibrosis-progression/test/',
                                   struct_data = X_holdout, label = lbl_holdout, 
                                   id_tensor = pid_holdout,
                                   return_stats_only = True,
                                  shuffle = False)
"""



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2080609891.py in <cell line: 0>()
      1 # create tf dataset for train
----> 2 dataset = osic_utils.DatasetGen(patient_dcm_dict, split = None, seed = 300, 
      3                          batch_size = 1,
      4                          root_dir = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/',
      5                          shuffle = False)

NameError: name 'osic_utils' is not defined

## === cell 16
def make_joint_dist_coroutine(num_ids, pids, X):
    def model():
        X_cst = tf.cast(X, tf.float32)
        Root = tfd.JointDistributionCoroutine.Root
        patient_scale = yield Root(tfd.HalfCauchy(loc = 0, scale = 5.))
        intercept = yield Root(tfd.Normal(loc = 0, scale = 10.))
        patient_prior = yield tfd.MultivariateNormalDiag(loc = tf.zeros(num_ids), 
                                                       scale_identity_multiplier = patient_scale)
        int_resp = tf.gather(patient_prior, pids, axis = -1) + intercept[...,tf.newaxis]
        
        beta_week = yield Root(tfd.Normal(loc = 0, scale = 10.))
        beta_week_scale = yield Root(tfd.HalfCauchy(loc = 0, scale = 5.))
        beta_week_prior = yield tfd.MultivariateNormalDiag(loc = tf.zeros(num_ids),
                                                          scale_identity_multiplier = beta_week_scale)
        bw_resp = (tf.gather(beta_week_prior, pids, axis = -1)  + beta_week[..., tf.newaxis]) * X_cst[:,0]
        
        betas = yield Root(tfd.MultivariateNormalDiag(loc = tf.zeros(tf.shape(X_cst)[1] - 1), 
                                                scale_identity_multiplier = 10.))
        other_vars_resp = tf.tensordot(betas, tf.transpose(X_cst[:,1:]), axes = 1)    
        total_response = int_resp + bw_resp + other_vars_resp
        
        resp_scale = yield Root(tfd.HalfCauchy(loc =0, scale = 5.))
        
        yield tfd.Normal(loc = total_response,  scale = resp_scale[...,tf.newaxis])
        
    return tfd.JointDistributionCoroutineAutoBatched(model)
    

## === cell 18
dist = make_joint_dist_coroutine(176, tf.constant(pid, tf.int32), X)
[i.shape for i in dist.sample(2)]

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/40036017.py in <cell line: 0>()
----> 1 dist = make_joint_dist_coroutine(176, tf.constant(pid, tf.int32), X)
      2 [i.shape for i in dist.sample(2)]

NameError: name 'pid' is not defined

## === cell 20
"""
# define parameters to be optimized
_init_loc = lambda name, shape = (): tf.Variable(
    tf.random.uniform(shape, name = name,  minval=-2., maxval=2.))

_init_scale =lambda name, shape = (): tfp.util.TransformedVariable(
    initial_value=tf.random.uniform(shape, minval=0.01, maxval=1.),
    bijector=tfb.Softplus(), name = name)

### SET PARAMETERS HERE ####
num_epochs = 1
print_every = 5
num_ids = len(patient_dcm_dict.keys())
num_draws = 2

# intercept & random intercepts
a0 = _init_loc(name = "alpha0")
a0_sigma = _init_scale(name = "alpha0_sigma")
patient_scale = _init_loc(name = "patient_scale")
patient_scale_sigma = _init_scale(name = "patient_scale_sigma")
a = _init_loc(name = "alphas", shape = [num_ids])
a_sigma = _init_scale(name = "alphas_sigma", shape = [num_ids])

int_vars = [patient_scale, patient_scale_sigma.trainable_variables[0],
           a0, a0_sigma.trainable_variables[0],
           a, a_sigma.trainable_variables[0]]

# random slope (week variable)
bw = _init_loc(name = "bw")
bw_sigma = _init_scale(name = "bw_sigma")
bws_scale = _init_loc(name = "bw_scale")
bws_scale_sigma = _init_scale(name = "bws_scale_sigma")
bws = _init_loc(name = "bws", shape = [num_ids])
bws_sigma = _init_scale(name = "bws_sigma", shape = [num_ids])

rnd_slope_vars  = [bw, bw_sigma.trainable_variables[0], 
                   bws_scale, bws_scale_sigma.trainable_variables[0],
                   bws, bws_sigma.trainable_variables[0]]

# other vars
b = _init_loc(name = "betas", shape = [11])
b_sigma = _init_scale(name = "betas_sigma", shape = [11])

other_vars = [b, b_sigma.trainable_variables[0]]

# response scale
resp_scale = _init_loc(name = "resp_scale")
resp_scale_sigma = _init_scale(name = "resp_scale_sigma")
response_scale_vars = [resp_scale, resp_scale_sigma.trainable_variables[0]]

trainable_vars = int_vars + rnd_slope_vars + other_vars + response_scale_vars 

optimizer = tf.optimizers.Adam(learning_rate = 0.001)
epoch_loss = []
start = time.time()
for e in range(num_epochs):
    batch_loss = []
    i = 0
    for ids, X, lbl, idx, img_stats in dataset.train:
        with tf.GradientTape() as tape:      
            lbl = tf.cast(lbl/1000, tf.float32) #reduce the magnitude of the labels
            shp = tf.shape(ids)
            # concatenate structured data and extracted image features
            X = tf.concat([tf.cast(X, tf.float32), tf.cast(img_stats, tf.float32)], axis = 1)
            jd = make_joint_dist_coroutine(shp.numpy()[0], idx, X)
        
            # create surrogate posterior dynamically
            def variational_model_fn():
                return tfd.JointDistributionSequentialAutoBatched([
                  tfb.Softplus()(tfd.Normal(patient_scale, patient_scale_sigma)),  # scale_prior
                  tfd.Normal(a0, a0_sigma),                                        # intercept
                  tfd.Normal(tf.gather(a, ids), tf.gather(a_sigma, ids)),          # patient prior
                  tfd.Normal(bw, bw_sigma),                                        # week random slope
                  tfb.Softplus()(tfd.Normal(bws_scale, bws_scale_sigma)),          # bw scale prior
                  tfd.Normal(tf.gather(bws, ids), tf.gather(bws_sigma, ids)),      # patient by week prior
                  tfd.Normal(b, b_sigma),                                          # other vars prior
                  tfb.Softplus()(tfd.Normal(resp_scale, resp_scale_sigma)),        # response scale
                    
                ])

            # create the surrogate posterior
            surrogate_pos = variational_model_fn()
            # calculate losses
            loss = tfp.vi.monte_carlo_variational_loss(
                lambda *args: jd.log_prob(*args, lbl),
                surrogate_pos,
                sample_size = num_draws,
                use_reparameterization = True
            )
            
        # compute gradients 
        grad = tape.gradient(loss, trainable_vars)        
        optimizer.apply_gradients(zip(grad, trainable_vars))
        print("epoch {}, batch {} loss: {}".format(e, i, loss))
        i += 1
        batch_loss.append(loss)
        
    epoch_loss.append(np.mean(batch_loss))
    if e % print_every == 0:
        print("epoch {} loss: {}".format(e, np.mean(batch_loss)))
end = time.time()

print("Total processing time:", str((end - start)/60), "mins")
"""

## === cell 21
num_ids = len(patient_dcm_dict.keys())
jd = make_joint_dist_coroutine(num_ids, pid, X)
def target_log_prob_fn(*args):
    return jd.log_prob(*args, lbl/1000)
    
_init_loc = lambda shape=(): tf.Variable(
    tf.random.uniform(shape, minval=-2., maxval=2.))
_init_scale = lambda shape=(): tfp.util.TransformedVariable(
    initial_value=tf.random.uniform(shape, minval=0.01, maxval=1.),
    bijector=tfb.Softplus())

num_other_vars = X.shape[1] - 1

surrogate_posterior = tfd.JointDistributionSequentialAutoBatched([
                  tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),                       # scale_prior
                  tfd.Normal(_init_loc(), _init_scale()),                                       # intercept
                  tfd.Normal(_init_loc(shape = [num_ids]), _init_scale(shape = [num_ids])),      # patient prior
                  tfd.Normal(_init_loc(), _init_scale()),                                       # week random slope
                  tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),                       # week scale prior
                  tfd.Normal(_init_loc(shape = [num_ids]), _init_scale(shape = [num_ids])),      # patient by week prior
                  tfd.Normal(_init_loc(shape = [num_other_vars]), 
                             _init_scale(shape = [num_other_vars])),                            # other vars prior
                  tfb.Softplus()(tfd.Normal(_init_loc(), _init_scale())),                       # response scale
                    
                ])


optimizer = tf.optimizers.Adam(learning_rate=1e-4)

start = time.time()
losses = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn,  
    surrogate_posterior,
    optimizer=optimizer,
    num_steps= 50000, 
    seed=42,
    sample_size= 500)
end = time.time()

print("processing time:", str((end - start)/60), "min")

(scale_prior_, 
 intercept_,  
 patient_weights,
 week_slope,
 week_scale_prior,
 week_patient_weights,
 other_vars_weights,
 response_scale), _ = surrogate_posterior.sample_distributions()


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2294223077.py in <cell line: 0>()
      1 num_ids = len(patient_dcm_dict.keys())
----> 2 jd = make_joint_dist_coroutine(num_ids, pid, X)
      3 def target_log_prob_fn(*args):
      4     return jd.log_prob(*args, lbl/1000)
      5 

NameError: name 'pid' is not defined

## === cell 23
print('intercept:', intercept_.mean())
print('week weight:', week_slope.mean())
print('other var weights', other_vars_weights.mean())

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4148748289.py in <cell line: 0>()
----> 1 print('intercept:', intercept_.mean())
      2 print('week weight:', week_slope.mean())
      3 print('other var weights', other_vars_weights.mean())

NameError: name 'intercept_' is not defined

## === cell 24
fig = plt.figure(figsize = (10, 6))
plt.plot(losses)
plt.title('ELBO loss across epochs', size = 15)
plt.show()

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3717571618.py in <cell line: 0>()
      1 fig = plt.figure(figsize = (10, 6))
----> 2 plt.plot(losses)
      3 plt.title('ELBO loss across epochs', size = 15)
      4 plt.show()

NameError: name 'losses' is not defined

## === cell 26
def predict_from_params(X):
    other_vars_weights_ = other_vars_weights.mean().numpy().reshape(-1, 1)
    intercept = intercept_.mean().numpy().reshape(-1, 1)
    week_slope_ = week_slope.mean().numpy().reshape(-1, 1)
    expected_val = (np.matmul(X[:, 1:], other_vars_weights_)
                    + week_slope_ * X[:, :1]
                    + intercept)
    return expected_val

## === cell 28
min_FVC = np.min(lbl/1000)
max_FVC = np.max(lbl/1000)
plt.subplots(1,1, figsize = (8,6))
plt.scatter(np.squeeze(predict_from_params(X)), lbl/1000, alpha =0.3)
plt.xlabel('Predicted', size = 10)
plt.ylabel('Actual', size = 10)
plt.plot([min_FVC, max_FVC], [min_FVC, max_FVC], "--")

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2375106320.py in <cell line: 0>()
----> 1 min_FVC = np.min(lbl/1000)
      2 max_FVC = np.max(lbl/1000)
      3 plt.subplots(1,1, figsize = (8,6))
      4 plt.scatter(np.squeeze(predict_from_params(X)), lbl/1000, alpha =0.3)
      5 plt.xlabel('Predicted', size = 10)

NameError: name 'lbl' is not defined

## === cell 29
preds_holdout = []
for i, (ids, idx) in enumerate(patient_dcm_holdout.items()):
    path = "/kaggle/input/osic-pulmonary-fibrosis-progression/test/" + ids
    stats = ct.image_processing_pipeline(path, return_stats_only = True)
    stats[0,0] = stats[0,0]/1000
    idxs = np.where(pid_holdout == idx)[0]
    X_h = np.take(X_holdout, idxs, axis = 0)   
    pred = predict_from_params(np.concatenate([X_h, np.repeat(stats, idxs.shape[0], axis=0)], axis=1))
    preds_holdout.append(pred)
    
    

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1230340591.py in <cell line: 0>()
      1 preds_holdout = []
----> 2 for i, (ids, idx) in enumerate(patient_dcm_holdout.items()):
      3     path = "/kaggle/input/osic-pulmonary-fibrosis-progression/test/" + ids
      4     stats = ct.image_processing_pipeline(path, return_stats_only = True)
      5     stats[0,0] = stats[0,0]/1000

NameError: name 'patient_dcm_holdout' is not defined

## === cell 30
holdout['FVC'] = np.concatenate(preds_holdout, axis = 0) * 1000
holdout['Confidence'] = np.mean(response_scale.sample(50000).numpy()) * 1000
holdout['Patient_Week'] = holdout[['Patient', 'Weeks', 'Weeks_base']] \
.apply(lambda x: '{}_{}'.format(x['Patient'], x['Weeks'] + x['Weeks_base']), axis = 1)
holdout


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4196123960.py in <cell line: 0>()
----> 1 holdout['FVC'] = np.concatenate(preds_holdout, axis = 0) * 1000
      2 holdout['Confidence'] = np.mean(response_scale.sample(50000).numpy()) * 1000
      3 holdout['Patient_Week'] = holdout[['Patient', 'Weeks', 'Weeks_base']] \
      4 .apply(lambda x: '{}_{}'.format(x['Patient'], x['Weeks'] + x['Weeks_base']), axis = 1)
      5 holdout

ValueError: need at least one array to concatenate

## === cell 31
submission = holdout[['Patient_Week',  'FVC', 'Confidence']]
submission.to_csv("submission.csv", index = False)

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1311080250.py in <cell line: 0>()
----> 1 submission = holdout[['Patient_Week',  'FVC', 'Confidence']]
      2 submission.to_csv("submission.csv", index = False)

NameError: name 'holdout' is not defined

## === cell 32
submission.head()

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined
