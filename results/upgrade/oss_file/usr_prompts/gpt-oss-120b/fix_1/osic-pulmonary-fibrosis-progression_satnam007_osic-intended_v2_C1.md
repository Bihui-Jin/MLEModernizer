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

-6.9731

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
import seaborn as sns
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold


## === cell 1
import tensorflow as tf
import tensorflow.keras.backend as backend   #K  
import tensorflow.keras.layers as layers     #L
import tensorflow.keras.models as models     #M


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return seed


## === cell 3
path = "../input/osic-pulmonary-fibrosis-progression"

train = pd.read_csv(f"{path}/train.csv")                        #tr
test  = pd.read_csv(f"{path}/test.csv")                         #chunk
print('****training data head 1 value****\n')
print(train.head(1))
print('\n****test data head 1 value****\n')
print(test.head(1))


## === cell 4
print('train_data_shape', train.shape)
print('test_data_shape', test.shape)
print('duplicates',train.duplicated().sum())

print('duplicates',train.duplicated(subset=['Patient','Weeks']).sum())
train.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])


## === cell 5
sub = pd.read_csv(f"{path}/sample_submission.csv")

sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))


sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]


## === cell 6
submission = sub.merge(test.drop('Weeks', axis=1), on="Patient")
print('****submission_data head 1 value*****\n')
print(sub.head(1))
print('******test data head 1 value********')
print(test.head(1))

submission.head(1)


## === cell 7
train['WHERE'] = 'train'
test['WHERE'] = 'val'
submission['WHERE'] = 'test'
print('train data shape\n\n',train.shape)
print('\ntest data shape\n\n',test.shape)
print('\nsubmission data shape\n\n',submission.shape)

data = train.append([test, submission])
print('\ndata_shape',data.shape)
data.head(2)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3398039557.py in <cell line: 0>()
      8 
      9 # we append the test and submission data on to
---> 10 data = train.append([test, submission])
     11 print('\ndata_shape',data.shape)
     12 data.head(2)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 8
print(train.Patient.nunique(), data.Patient.nunique(), test.Patient.nunique(), submission.Patient.nunique())


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3483969680.py in <cell line: 0>()
      1 # we know each patient have multiple entries so we are checking the unique entries
----> 2 print(train.Patient.nunique(), data.Patient.nunique(), test.Patient.nunique(), submission.Patient.nunique())

NameError: name 'data' is not defined

## === cell 9
check_point_1 = data['min_week'] = data['Weeks']

check_point_2 = data.loc[data.WHERE=='test','min_week'] = np.nan

check_point_3 = data['min_week'] = data.groupby('Patient')['min_week'].transform('min')

print('check_point_1\n',check_point_1.head(10))
print('\ncheck_point_2\n',check_point_2)
print('\ncheck_point_3\n',check_point_3)
data.head(10)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1254376653.py in <cell line: 0>()
      1 # creating a new column min_week
      2 # min week means the first week of patient's observation
----> 3 check_point_1 = data['min_week'] = data['Weeks']
      4 
      5 # putting the min_week to NAN value

NameError: name 'data' is not defined

## === cell 10
base = data.loc[data.Weeks == data.min_week]
a = base
base = base[['Patient','FVC']].copy()
b = base

base.columns = ['Patient','min_FVC']
base['nb'] = 1
c = base

base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
d = base

base = base[base.nb==1]
e = base

base.drop('nb', axis=1, inplace=True)
f = base



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/316173221.py in <cell line: 0>()
----> 1 base = data.loc[data.Weeks == data.min_week]
      2 a = base
      3 base = base[['Patient','FVC']].copy()
      4 b = base
      5 

NameError: name 'data' is not defined

## === cell 11
data = data.merge(base, on='Patient', how='left')

data['base_week'] = data['Weeks'] - data['min_week']


data.head(10)
del base


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2041371861.py in <cell line: 0>()
----> 1 data = data.merge(base, on='Patient', how='left')
      2 
      3 data['base_week'] = data['Weeks'] - data['min_week']
      4 
      5 

NameError: name 'data' is not defined

## === cell 12
data.head(5)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1134898118.py in <cell line: 0>()
----> 1 data.head(5)

NameError: name 'data' is not defined

## === cell 13

COLS = ['Sex','SmokingStatus']
FE = []
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/728591123.py in <cell line: 0>()
      5 FE = []
      6 for col in COLS:
----> 7     for mod in data[col].unique():
      8         FE.append(mod)
      9         #print(FE)

NameError: name 'data' is not defined

## === cell 14
data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )


data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )


data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )


data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )


FE += ['age','percent','week','BASE']


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2680796611.py in <cell line: 0>()
      1 #Scaling the data
      2 ######### complex way ############## IN BUILD METHORD IS EASIRES ONE ###################
----> 3 data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )
      4 
      5 

NameError: name 'data' is not defined

## === cell 15
data


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3021797462.py in <cell line: 0>()
----> 1 data

NameError: name 'data' is not defined

## === cell 17
train = data.loc[data.WHERE=='train']
test = data.loc[data.WHERE=='val']
submission = data.loc[data.WHERE=='test']





## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1918614311.py in <cell line: 0>()
----> 1 train = data.loc[data.WHERE=='train']
      2 test = data.loc[data.WHERE=='val']
      3 submission = data.loc[data.WHERE=='test']
      4 
      5 

NameError: name 'data' is not defined

## === cell 18
SEED = seed_everything(42)
NFOLD      = 5
BATCH_SIZE = 128
EPOCHS     = 800


## === cell 19
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

print('C1 = ',C1)
print('C2 = ',C2)


## === cell 20
def Laplace_log_Likelihood_score(y_true, y_pred):
    
    tf.dtypes.cast(y_true, tf.float32)  # converting y_true in float values
    tf.dtypes.cast(y_pred, tf.float32)  # converting y_pred in float values
    sigma = y_pred[:, 2] - y_pred[:, 0] # calculating the standard deviation 
    fvc_pred = y_pred[:, 1]
    
    sigma_clip = tf.maximum(sigma, C1)  # clipping all standard deviation(sigma) the other values less than 70
    
    delta = tf.abs(y_true[:, 0] - fvc_pred) # |FVC_true - FVC_predicted| abs mean we need a +ve value as always
    delta = tf.minimum(delta, C2)           # clipping all values greater than 1000
    
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) ) # calculating sqr root of 2
    
    metric = -(delta / sigma_clip)*sq2 - tf.math.log(sigma_clip* sq2) # calculating metric as given in OSIC  evaluation
    print('Calculating Laplace_log_Likelihood_score')
    return backend.mean(metric)


## === cell 21
def qloss(y_true, y_pred):
    
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    
    print('Calculating qLoss')
    return backend.mean(v)


## === cell 22
def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*Laplace_log_Likelihood_score(y_true, y_pred)
    print('Calculating mLoss')
    return loss


## === cell 23
def make_model():                          #backend   #K layers     #L models     #M
    z = layers.Input((9,), name="Patient")
    
    x = layers.Dense(100, activation="relu", name="d1")(z)
    x = layers.Dense(100, activation="relu", name="d2")(x)
    p1 = layers.Dense(3, activation="linear", name="p1")(x)
    p2 = layers.Dense(3, activation="relu", name="p2")(x)
    
    preds = layers.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 
                     name="preds")([p1, p2])
    
    model = models.Model(z, preds, name="ANN")
    model.compile(loss=mloss(0.775), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, 
                beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[Laplace_log_Likelihood_score])
    return model


## === cell 24
model = make_model()
print(model.summary())
print(model.count_params())


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2388832127.py in <cell line: 0>()
----> 1 model = make_model()
      2 print(model.summary())
      3 print(model.count_params())

/tmp/ipykernel_11/2986177956.py in make_model()
     13     model = models.Model(z, preds, name="ANN")
     14     #model.compile(loss=qloss, optimizer="adam", metrics=[Laplace_log_Likelihood_score])
---> 15     model.compile(loss=mloss(0.775), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, 
     16                 beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[Laplace_log_Likelihood_score])
     17     return model

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

## === cell 25
y = train['FVC'].values
z = train[FE].values
ze = submission[FE].values
pe = np.zeros((ze.shape[0], 3))
pred = np.zeros((z.shape[0], 3))


## === cell 27
kf = KFold(n_splits=NFOLD)
print(kf)


## === cell 28
%%time

cnt = 0
for train_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")
    
    model.fit(z[train_idx], y[train_idx], batch_size=BATCH_SIZE, epochs= EPOCHS, 
            validation_data=(z[val_idx], y[val_idx]), verbose=0) #
    
    
    print("train", model.evaluate(z[train_idx], y[train_idx], verbose=0, batch_size=BATCH_SIZE))
    print("val", model.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))
    
    pred[val_idx] = model.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
    pe += model.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<timed exec> in <module>

NameError: name 'model' is not defined

## === cell 29
mean_absolute_error = mean_absolute_error(y, pred[:, 1]) # find  UNC 
unc = pred[:,2] - pred[:, 0]
unc_mean = np.mean(unc)


## === cell 31
stats = pd.DataFrame()
index = 0


## === cell 32
data = [[index, SEED, BATCH_SIZE, EPOCHS, mean_absolute_error, unc.min(), unc.mean(), unc.max(), (unc>=0).mean()]]
columns = ['Run Kernal','seed', 'batch_size', 'epochs', 'mean_abs_err', 'unc.min', 'unc.mean', 'unc.max', '(unc>=0).mean']
kernal_stats = pd.DataFrame(data, columns=columns)


## === cell 33
stats = pd.concat([stats, kernal_stats])
stats.to_csv('kernal.csv', index = False)
index+=1



## === cell 35
print('we are using fix seed value always to avoid RANDOMIZATION (NEED TO GET SAME RESULT)')
print('Seed value          =',SEED)
print('Number of folds     =',BATCH_SIZE)
print('Number of epochs    =',EPOCHS)

print('\nmean_absolute_error =',mean_absolute_error)

print('unc_mean            =',unc.mean())
print('unc_min             =',unc.min())
print('unc_max             =',unc.max())
print('unc_mean            =',(unc>=0).mean())


## === cell 36
idxs = np.random.randint(0, y.shape[0], 100)
plt.plot(y[idxs], label="ground truth")
plt.plot(pred[idxs, 0], label="q25")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()


## === cell 37
sns.distplot(unc)
plt.title("uncertainty in prediction")
plt.show()


## === cell 38
plt.hist(unc)
plt.title("uncertainty in prediction")
plt.show()


## === cell 39
submission.head()


## === cell 40
submission['FVC1'] = pe[:, 1]
submission['Confidence1'] = pe[:, 2] - pe[:, 0]


## === cell 41
subm = submission[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()


## === cell 42
subm.loc[~subm.FVC1.isnull()].head(10)


## === cell 43
subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']
if unc_mean<70:
    subm['Confidence'] = mean_absolute_error
else:
    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']


## === cell 44
subm.head()


## === cell 45
sns.distplot(subm.FVC)


## === cell 46
sns.distplot(subm.Confidence)


## === cell 47
subm.describe().T


## === cell 48
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
for i in range(len(otest)):
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1


## === cell 49
subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)


## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame must have a 'Patient_Week' column.
