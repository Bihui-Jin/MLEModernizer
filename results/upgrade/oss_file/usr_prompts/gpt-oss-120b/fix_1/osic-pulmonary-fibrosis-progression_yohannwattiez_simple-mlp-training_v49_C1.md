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

-13.890558069528478

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import random
import pickle
import time

from tensorflow import keras as K
from tensorflow.keras import layers as L

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
def seed_all(seed = 20):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_all(20)

## === cell 5


train = pd.read_csv('/kaggle/input/data-preparation-for-osic/train.csv')

raw_test = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')
X_prediction = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/795118444.py in <cell line: 0>()
      6 # list_files = pd.read_csv('/kaggle/input/prep-data/list_files.csv')
      7 
----> 8 train = pd.read_csv('/kaggle/input/data-preparation-for-osic/train.csv')
      9 
     10 raw_test = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/data-preparation-for-osic/train.csv'

## === cell 6
ID='Patient_Week'
PINBALL_QUANTILE = [0.2, 0.50, 0.8]
LAMBDA_LOSS = 0.8
EPOCH = 250
BATCH_SIZE = 128

## === cell 8
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
def score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.backend.mean(metric)
def qloss(y_true, y_pred):
    qs = PINBALL_QUANTILE
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.backend.mean(v)
def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    return loss

## === cell 9
def eval_score(y_true, y_pred):
    y_true = tf.dtypes.cast(y_true, tf.float32)*(data_prep.fvc_max-data_prep.fvc_min)+data_prep.fvc_min
    y_pred = tf.dtypes.cast(y_pred, tf.float32)*(data_prep.fvc_max-data_prep.fvc_min)+data_prep.fvc_min
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return -K.backend.mean(metric)

## === cell 10
SELECTED_COLUMNS = ['Weeks', 'Percent', 'Age', 'Sex', 'Min_week', 'Base_FVC','Base_week', '_Currently smokes', '_Ex-smoker', '_Never smoked']
def create_model(lambda_loss):
    model_input = K.Input(shape=(len(SELECTED_COLUMNS),))
    x = L.Dense(500, activation="selu", name='dense_to_freeze1')(model_input)
    x = L.Dense(100, activation="selu", name='dense_to_freeze2')(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="selu", name="p2")(x)
    FVC = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 
                         name="FVC")([p1, p2])

    model = K.Model(
        inputs=model_input,
        outputs=[FVC],
    )
    model.compile(
        optimizer=K.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False),
        loss=mloss(lambda_loss),
        metrics=score,
    )
    return model

model = create_model(LAMBDA_LOSS)
model.summary()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/879412989.py in <cell line: 0>()
     25     return model
     26 
---> 27 model = create_model(LAMBDA_LOSS)
     28 # tf.keras.utils.plot_model(model)
     29 model.summary()

/tmp/ipykernel_11/879412989.py in create_model(lambda_loss)
     19 #     optimizer=K.optimizers.Adam(learning_rate=learning_rate_fn, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
     20     model.compile(
---> 21         optimizer=K.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False),
     22         loss=mloss(lambda_loss),
     23         metrics=score,

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/adam.py in __init__(self, learning_rate, beta_1, beta_2, epsilon, amsgrad, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     60         **kwargs,
     61     ):
---> 62         super().__init__(
     63             learning_rate=learning_rate,
     64             name=name,

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/optimizer.py in __init__(self, *args, **kwargs)
     19 class TFOptimizer(KerasAutoTrackable, base_optimizer.BaseOptimizer):
     20     def __init__(self, *args, **kwargs):
---> 21         super().__init__(*args, **kwargs)
     22         self._distribution_strategy = tf.distribute.get_strategy()
     23 

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py in __init__(self, learning_rate, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     88             )
     89         if kwargs:
---> 90             raise ValueError(f"Argument(s) not recognized: {kwargs}")
     91 
     92         if name is None:

ValueError: Argument(s) not recognized: {'lr': 0.1}

## === cell 11
X_prediction['Patient'] = X_prediction['Patient_Week'].str.extract(r'(.*)_.*')
X_prediction['Weeks'] = X_prediction['Patient_Week'].str.extract(r'.*_(.*)').astype(int)
X_prediction = X_prediction[['Patient', 'Weeks', 'Patient_Week']]
rename_cols = {'Weeks_y':'Min_week', 'Weeks_x': 'Weeks', 'FVC':'Base_FVC'}
X_prediction = X_prediction.merge(raw_test, how='left', left_on='Patient', right_on='Patient').rename(columns=rename_cols)[['Patient', 'Min_week', 'Base_FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus', 'Weeks', 'Patient_Week']].reset_index(drop=True)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3753751225.py in <cell line: 0>()
----> 1 X_prediction['Patient'] = X_prediction['Patient_Week'].str.extract(r'(.*)_.*')
      2 X_prediction['Weeks'] = X_prediction['Patient_Week'].str.extract(r'.*_(.*)').astype(int)
      3 X_prediction = X_prediction[['Patient', 'Weeks', 'Patient_Week']]
      4 rename_cols = {'Weeks_y':'Min_week', 'Weeks_x': 'Weeks', 'FVC':'Base_FVC'}
      5 X_prediction = X_prediction.merge(raw_test, how='left', left_on='Patient', right_on='Patient').rename(columns=rename_cols)[['Patient', 'Min_week', 'Base_FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus', 'Weeks', 'Patient_Week']].reset_index(drop=True)

NameError: name 'X_prediction' is not defined

## === cell 12
X_prediction['Base_week'] = X_prediction['Weeks'] - X_prediction['Min_week']

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2768986138.py in <cell line: 0>()
----> 1 X_prediction['Base_week'] = X_prediction['Weeks'] - X_prediction['Min_week']

NameError: name 'X_prediction' is not defined

## === cell 14
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, categories, index='', name='', **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X, name=name, categories=categories)
        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=index)
        return d_out

    def fit_transform(self, X, categories, index, name, **kwargs):
        self.fit(X)
        return self.transform(X, categories=categories, index=index, name=name)

    def get_new_columns(self, X, name, categories):
        new_columns = []
        for j in range(len(categories)):
            new_columns.append('{}_{}'.format(name, categories[j]))
        return new_columns

from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

def standardisation(x, u, s):
    return (x-u)/s

def normalization(x, ma, mi):
    return (x-mi)/(ma-mi)

class data_preparation():
    def __init__(self, bool_normalization=True,bool_standard=False):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = OneHotEncoder()
        self.standardisation = bool_standard
        self.normalization = bool_normalization
        
        
    def __call__(self, data_untransformed):
        data = data_untransformed.copy(deep=True)
        
        try:
            data['Sex'] = self.enc_sex.transform(data['Sex'].values)
            data['SmokingStatus'] = self.enc_smok.transform(data['SmokingStatus'].values)
            data = pd.concat([data.drop(columns=['SmokingStatus']), self.onehotenc_smok.transform(data['SmokingStatus'].values.reshape(-1,1), categories=self.enc_smok.classes_, name='', index=data.index).astype(int)], axis=1)
            
            if self.standardisation:
                data['Base_week'] = standardisation(data['Base_week'],self.base_week_mean,self.base_week_std)
                data['Base_FVC'] = standardisation(data['Base_FVC'],self.base_fvc_mean,self.base_fvc_std)
                data['Base_percent'] = standardisation(data['Base_percent'],self.base_percent_mean,self.base_percent_std)
                data['Age'] = standardisation(data['Age'],self.age_mean,self.age_std)
                data['Weeks'] = standardisation(data['Weeks'],self.weeks_mean,self.weeks_std)
            
            if self.normalization:
                data['Base_week'] = normalization(data['Base_week'],self.base_week_max,self.base_week_min)
                data['Base_FVC'] = normalization(data['Base_FVC'],self.base_fvc_max,self.base_fvc_min)
                data['Percent'] = normalization(data['Percent'],self.base_percent_max,self.base_percent_min)
                data['Age'] = normalization(data['Age'],self.age_max,self.age_min)
                data['Weeks'] = normalization(data['Weeks'],self.weeks_max,self.weeks_min)
                data['Min_week'] = normalization(data['Min_week'],self.base_week_max,self.base_week_min)

        except NotFittedError:
            data['Sex'] = self.enc_sex.fit_transform(data['Sex'].values)
            data['SmokingStatus'] = self.enc_smok.fit_transform(data['SmokingStatus'].values)
            data = pd.concat([data.drop(columns=['SmokingStatus']), self.onehotenc_smok.fit_transform(data['SmokingStatus'].values.reshape(-1,1), categories=self.enc_smok.classes_, name='', index=data.index).astype(int)], axis=1)
            
            if self.standardisation:
                self.base_week_mean = data['Base_week'].mean()
                self.base_week_std = data['Base_week'].std()
                data['Base_week'] = standardisation(data['Base_week'],self.base_week_mean,self.base_week_std)

                self.base_fvc_mean = data['Base_FVC'].mean()
                self.base_fvc_std = data['Base_FVC'].std()
                data['Base_FVC'] = standardisation(data['Base_FVC'],self.base_fvc_mean,self.base_fvc_std)

                self.base_percent_mean = data['Base_percent'].mean()
                self.base_percent_std = data['Base_percent'].std()
                data['Base_percent'] = standardisation(data['Base_percent'],self.base_percent_mean,self.base_percent_std)

                self.age_mean = data['Age'].mean()
                self.age_std = data['Age'].std()
                data['Age'] = standardisation(data['Age'],self.age_mean,self.age_std)

                self.weeks_mean = data['Weeks'].mean()
                self.weeks_std = data['Weeks'].std()
                data['Weeks'] = standardisation(data['Weeks'],self.weeks_mean,self.weeks_std)

                
            if self.normalization:
                self.base_week_min = data['Base_week'].min()
                self.base_week_max = data['Base_week'].max()
                data['Base_week'] = normalization(data['Base_week'],self.base_week_max,self.base_week_min)

                self.base_fvc_min = data['Base_FVC'].min()
                self.base_fvc_max = data['Base_FVC'].max()
                data['Base_FVC'] = normalization(data['Base_FVC'],self.base_fvc_max,self.base_fvc_min)

                self.base_percent_min = data['Percent'].min()
                self.base_percent_max = data['Percent'].max()
                data['Percent'] = normalization(data['Percent'],self.base_percent_max,self.base_percent_min)

                self.age_min = data['Age'].min()
                self.age_max = data['Age'].max()
                data['Age'] = normalization(data['Age'],self.age_max,self.age_min)

                self.weeks_min = data['Weeks'].min()
                self.weeks_max = data['Weeks'].max()
                data['Weeks'] = normalization(data['Weeks'],self.weeks_max,self.weeks_min)
                
                self.base_week_min = data['Min_week'].min()
                self.base_week_max = data['Min_week'].max()
                data['Min_week'] = normalization(data['Min_week'],self.base_week_max,self.base_week_min)

            
        return data

## === cell 15
pickefile = open('/kaggle/input/data-preparation-for-osic/data_prep', 'rb')
data_prep = pickle.load(pickefile)
pickefile.close()

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2383209039.py in <cell line: 0>()
----> 1 pickefile = open('/kaggle/input/data-preparation-for-osic/data_prep', 'rb')
      2 data_prep = pickle.load(pickefile)
      3 pickefile.close()

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/data-preparation-for-osic/data_prep'

## === cell 16
X_prediction = data_prep(X_prediction)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3833085080.py in <cell line: 0>()
----> 1 X_prediction = data_prep(X_prediction)
      2 # test = train[train['Patient']=='ID00009637202177434476278']
      3 # train = train[~(train['Patient']=='ID00009637202177434476278')]

NameError: name 'data_prep' is not defined

## === cell 18
pickefile = open('/kaggle/input/simple-mlp-training/list_patient_score', 'rb')
list_patient_score = pickle.load(pickefile)
pickefile.close()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2978664924.py in <cell line: 0>()
----> 1 pickefile = open('/kaggle/input/simple-mlp-training/list_patient_score', 'rb')
      2 list_patient_score = pickle.load(pickefile)
      3 pickefile.close()

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/simple-mlp-training/list_patient_score'

## === cell 19
pickefile = open('list_patient_score', 'wb')
pickle.dump(list_patient_score, pickefile)
pickefile.close()

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3835588352.py in <cell line: 0>()
      1 pickefile = open('list_patient_score', 'wb')
----> 2 pickle.dump(list_patient_score, pickefile)
      3 pickefile.close()

NameError: name 'list_patient_score' is not defined

## === cell 20
count=0
for i in list_patient_score:
    if np.mean(i[1])<6.4:
        plt.plot(i[1])
        count+=1    
print(count)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2355414831.py in <cell line: 0>()
      1 count=0
----> 2 for i in list_patient_score:
      3     if np.mean(i[1])<6.4:
      4         plt.plot(i[1])
      5         count+=1

NameError: name 'list_patient_score' is not defined

## === cell 21
list_patient_weight = []
for i in list_patient_score:
    if 6.8>np.mean(i[1])>6.4:
        list_patient_weight.append([i[0], 3])
    elif 7.1>np.mean(i[1])>6.8:
        list_patient_weight.append([i[0], 5])
    else:
        list_patient_weight.append([i[0], 1])
        
train['Weight'] = train.Patient.map(dict(list_patient_weight))

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4088751766.py in <cell line: 0>()
      1 list_patient_weight = []
----> 2 for i in list_patient_score:
      3     if 6.8>np.mean(i[1])>6.4:
      4         list_patient_weight.append([i[0], 3])
      5     elif 7.1>np.mean(i[1])>6.8:

NameError: name 'list_patient_score' is not defined

## === cell 24
pe = np.zeros((X_prediction.shape[0], 3))
NFOLD = 8
for i in range(NFOLD):
    model = create_model(LAMBDA_LOSS)
    list_patients = train.Patient.unique()[np.random.randint(0, len(train.Patient.unique())-1, size=round(len(train.Patient.unique())/NFOLD))]
    history = model.fit(x=train[~train.Patient.isin(list_patients)][SELECTED_COLUMNS], y=train[~train.Patient.isin(list_patients)][['FVC']], validation_data=(train[train.Patient.isin(list_patients)][SELECTED_COLUMNS], train[train.Patient.isin(list_patients)][['FVC']]), epochs=EPOCH, sample_weight=train[~train.Patient.isin(list_patients)].Weight, verbose=0)
    pe += model.predict(X_prediction[SELECTED_COLUMNS], batch_size=BATCH_SIZE, verbose=0) / NFOLD
    model.save('model_' + str(NFOLD))

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1302291753.py in <cell line: 0>()
----> 1 pe = np.zeros((X_prediction.shape[0], 3))
      2 NFOLD = 8
      3 for i in range(NFOLD):
      4     model = create_model(LAMBDA_LOSS)
      5     list_patients = train.Patient.unique()[np.random.randint(0, len(train.Patient.unique())-1, size=round(len(train.Patient.unique())/NFOLD))]

NameError: name 'X_prediction' is not defined

## === cell 26
sigma_opt = mean_absolute_error(train[['FVC']], pred[:, 1])
unc = pred[:,2] - pred[:, 0]
sigma_mean = np.mean(unc)
print(sigma_opt, sigma_mean)

X_prediction['FVC1'] = 0.996*pe[:, 1]
X_prediction['Confidence1'] = pe[:, 2] - pe[:, 0]

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/876963152.py in <cell line: 0>()
----> 1 sigma_opt = mean_absolute_error(train[['FVC']], pred[:, 1])
      2 unc = pred[:,2] - pred[:, 0]
      3 sigma_mean = np.mean(unc)
      4 print(sigma_opt, sigma_mean)
      5 

NameError: name 'train' is not defined

## === cell 27
subm = X_prediction.copy()
subm['FVC'] = 3020
subm['Confidence'] = 100

subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']
if sigma_mean<70:
    subm['Confidence'] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1472805402.py in <cell line: 0>()
----> 1 subm = X_prediction.copy()
      2 subm['FVC'] = 3020
      3 subm['Confidence'] = 100
      4 
      5 subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']

NameError: name 'X_prediction' is not defined

## === cell 28
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
for i in range(len(otest)):
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1
subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4009015779.py in <cell line: 0>()
      1 otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
      2 for i in range(len(otest)):
----> 3     subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
      4     subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1
      5 subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)

NameError: name 'subm' is not defined
