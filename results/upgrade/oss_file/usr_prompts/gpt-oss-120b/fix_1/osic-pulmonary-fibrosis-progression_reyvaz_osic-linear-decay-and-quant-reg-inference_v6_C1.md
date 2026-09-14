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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

-6.8638683582785145

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
import os, sys
import numpy as np
import pandas as pd

import tensorflow as tf

from IPython.display import display
pd.set_option('display.max_columns', 50)

tf_version = tf.__version__
print("\nTensorflow version " + tf_version)

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
input_path = '../input/osic-pulmonary-fibrosis-progression'
pretrained_path = '../input/osic-linear-decay-and-quant-reg-base/pretrained_weights'

## === cell 5
def height_proxy(fvc_e, age, sex):
    if sex == 'Female': h = fvc_e/(21.78-0.101*age)
    else: h = fvc_e/(27.63-0.112*age)
    return h

def process_init_week(df, train_df = False):
    if train_df:
        df['min_week'] = df.groupby('Patient')['Weeks'].transform('min')

    base = df.loc[df.Weeks == df.min_week][['Patient', 'FVC', 'Percent', 'Age', 'Sex']]
    base['FVC_init_avg']= base.groupby('Patient')['FVC'].transform('mean').astype(int)
    base['Percent_init']= base.groupby('Patient')['Percent'].transform('mean')
    base = base[['Patient', 'FVC_init_avg', 'Percent_init', 'Age', 'Sex']].drop_duplicates()
    base['FVC_expected'] = (base['FVC_init_avg'] / (base['Percent_init']/100))
    base['Height_proxy'] = base.apply(lambda x: height_proxy(x.FVC_expected, x.Age, x.Sex), axis=1)
    base = base[['Patient', 'Height_proxy', 'FVC_init_avg', 'Percent_init']]

    df = df.merge(base, on='Patient', how='left')
    df['init_week'] = df['Weeks'] - df['min_week']
    return df

## === cell 6
train = pd.read_csv(input_path + '/train.csv')
train = process_init_week(train, train_df = True)
train.drop_duplicates(keep='first', inplace=True, subset=['Patient','Weeks'])
train.head(3)

## === cell 7
sub = pd.read_csv(input_path + '/sample_submission.csv') 
test = pd.read_csv(input_path + '/test.csv')

sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]

test = test.rename(columns={'Weeks': 'min_week'})
sub = sub.merge(test, on='Patient')

sub = process_init_week(sub, train_df = False)
sub.head(3)

## === cell 8
def scale_fn(var_name):
    col = train[var_name]
    return lambda x: (x - col.min())/(col.max()- col.min())

scale_age = scale_fn('Age')
scale_height = scale_fn('Height_proxy')
scale_percent = scale_fn('Percent')
scale_fvc = scale_fn('FVC_init_avg')

scale_week = lambda x: (x - (-12))/(133-(-12))

def transform_features(df):
    df = df.assign(sex_code = np.where(df['Sex'] == 'Female', 1, 0))
    df = df.assign(ex_smoker = np.where(df['SmokingStatus'] == 'Ex-smoker', 1, 0))
    df = df.assign(never_smoked = np.where(df['SmokingStatus'] == 'Never smoked', 1, 0))
    df = df.assign(current_smoker = np.where(df['SmokingStatus'] == 'Currently smokes', 1, 0))
    df['has_smoked'] = df['ex_smoker'] + df['current_smoker']

    df['age'] = df['Age'].map(scale_age)
    df['height'] = df['Height_proxy'].map(scale_height)
    df['percent'] = df['Percent'].map(scale_percent)
    df['percent_init'] = df['Percent_init'].map(scale_percent) # scale the same as Percent 
    df['week'] = df['Weeks'].map(scale_week) # this is to original week, use init week for validation analysis
    df['fvc_init'] = df['FVC_init_avg'].map(scale_fvc) # can change 'FVC_init_avg' to 'FVC_init_first' see data exploration notes
    return df

## === cell 9
train = transform_features(train)
train.reset_index(inplace=True, drop = True)
train.head(3)

## === cell 10
sub = transform_features(sub)
sub.head(3)

## === cell 12
linear_decay_features = ['age', 'sex_code', 'has_smoked', 'current_smoker', 'height', 'percent_init', 'fvc_init']

def get_patient_tab(df): # df is either train or sub
    patients_init = df[df['init_week'] == 0].copy()
    patients_init = patients_init[['Patient']+ linear_decay_features]
    patients_init.set_index('Patient', inplace = True)
    return patients_init

patients_tab_train = get_patient_tab(train)
patients_tab_test = get_patient_tab(sub)
print(patients_tab_train.shape)
display(patients_tab_train.head(3))
patients_tab_test

## === cell 14
PREDICTIONS = sub[['Patient', 'Weeks', 'Patient_Week']].copy()
PREDICTIONS.head(5)

## === cell 16
LD_inference = pd.read_csv(pretrained_path + '/inference_linear_decay_2020Sep19.csv')
LD_inference

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2714264993.py in <cell line: 0>()
----> 1 LD_inference = pd.read_csv(pretrained_path + '/inference_linear_decay_2020Sep19.csv')
      2 LD_inference

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/osic-linear-decay-and-quant-reg-base/pretrained_weights/inference_linear_decay_2020Sep19.csv'

## === cell 17
LD_test = patients_tab_test.reset_index()
pred_cols = ['Patient', 'Weeks', 'Patient_Week', 'FVC_init_avg', 'init_week']
return_cols = ['Patient', 'Weeks', 'Patient_Week', 'FVC_hat', 'sigma']

def get_sigma_function(s_intercept, s_multiplier, s_power):
    def alt_sigma(coeff, init_week):
        coeff = abs(coeff)
        week_distance = abs(init_week)
        sigma = s_intercept + s_multiplier*coeff*(week_distance**s_power)
        return sigma
    return alt_sigma

def pred_test(model, sigma_fn):
    X = LD_test[linear_decay_features].copy()
    XID = LD_test[['Patient']].copy()
    XID['coeff_pred'] = model.predict(X, batch_size = 32)

    P = sub[pred_cols].copy()
    P = P.merge(XID, how='left', on='Patient')

    P['FVC_hat'] = P['FVC_init_avg'] + (P['coeff_pred'] * P['init_week'])
    P['sigma'] = P.apply(lambda x: sigma_fn(x.coeff_pred, x.init_week), axis = 1)
    return P[return_cols]

## === cell 18
for fold_num in range(5):
    prefix = LD_inference.loc[fold_num].prefix
    fname = '{}/{}_weights.h5'.format(pretrained_path, prefix)
    s_intercept, s_multiplier, s_power = eval(LD_inference.loc[fold_num].alt_sigma_param)
    f_sigma = get_sigma_function(s_intercept, s_multiplier, s_power)

    model = tf.keras.models.load_model(fname)
    P = pred_test(model, sigma_fn = f_sigma)
    PREDICTIONS['FVC_LD{}'.format(fold_num)] = P['FVC_hat']
    PREDICTIONS['Confidence_LD{}'.format(fold_num)] = P['sigma']
    
del P, fold_num, fname, model

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1503048431.py in <cell line: 0>()
      1 for fold_num in range(5):
----> 2     prefix = LD_inference.loc[fold_num].prefix
      3     fname = '{}/{}_weights.h5'.format(pretrained_path, prefix)
      4     s_intercept, s_multiplier, s_power = eval(LD_inference.loc[fold_num].alt_sigma_param)
      5     f_sigma = get_sigma_function(s_intercept, s_multiplier, s_power)

NameError: name 'LD_inference' is not defined

## === cell 19
PREDICTIONS

## === cell 21
QR_inference = pd.read_csv(pretrained_path + '/inference_quant_reg_2020Sep23.csv')
QR_inference

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/339188138.py in <cell line: 0>()
----> 1 QR_inference = pd.read_csv(pretrained_path + '/inference_quant_reg_2020Sep23.csv')
      2 QR_inference

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/osic-linear-decay-and-quant-reg-base/pretrained_weights/inference_quant_reg_2020Sep23.csv'

## === cell 22
qr_features8 = ['fvc_init','week', 'sex_code', 'age', 'height', 'has_smoked', 'current_smoker', 'percent_init']
qr_features7 = ['fvc_init','week', 'sex_code', 'age', 'has_smoked', 'current_smoker', 'percent_init']

## === cell 23
for fold_num in range(5):
    prefix = QR_inference.loc[fold_num].prefix
    fname = '{}/{}_weights.h5'.format(pretrained_path, prefix)
    model = tf.keras.models.load_model(fname, compile = False)
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])

    num_features = QR_inference.loc[fold_num].num_features
    if num_features == 7: features = qr_features7
    else: features = qr_features8

    X = sub[features].copy()
    preds = model.predict(X)

    PREDICTIONS['FVC_QR{}'.format(fold_num)] = preds[:,1]
    PREDICTIONS['Confidence_QR{}'.format(fold_num)] = preds[:,2] - preds[:,0]

del X, fold_num, fname, model, prefix, num_features, features, preds

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1467931428.py in <cell line: 0>()
      1 for fold_num in range(5):
----> 2     prefix = QR_inference.loc[fold_num].prefix
      3     fname = '{}/{}_weights.h5'.format(pretrained_path, prefix)
      4     model = tf.keras.models.load_model(fname, compile = False)
      5     model.compile(loss='mae', optimizer='adam', metrics=['mae'])

NameError: name 'QR_inference' is not defined

## === cell 24
PREDICTIONS

## === cell 26
to_submit = PREDICTIONS[['Patient_Week','FVC_QR0', 'Confidence_QR0']]
to_submit.columns = ['Patient_Week','FVC','Confidence']
to_submit

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1412095543.py in <cell line: 0>()
----> 1 to_submit = PREDICTIONS[['Patient_Week','FVC_QR0', 'Confidence_QR0']]
      2 to_submit.columns = ['Patient_Week','FVC','Confidence']
      3 to_submit

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

KeyError: "['FVC_QR0', 'Confidence_QR0'] not in index"

## === cell 27
to_submit.describe().T

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3748868496.py in <cell line: 0>()
----> 1 to_submit.describe().T

NameError: name 'to_submit' is not defined

## === cell 28
to_submit.to_csv('submission.csv', index=False)

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2721867826.py in <cell line: 0>()
----> 1 to_submit.to_csv('submission.csv', index=False)

NameError: name 'to_submit' is not defined

## === cell 29
!head -3 submission.csv
