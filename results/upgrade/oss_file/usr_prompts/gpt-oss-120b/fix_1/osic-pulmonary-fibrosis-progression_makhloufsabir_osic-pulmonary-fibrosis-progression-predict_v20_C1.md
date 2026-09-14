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

category_encoders==2.7.0
geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
pydicom==3.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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
tf_keras==2.18.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
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

-11.0734

# 6. Current score

-9.10046

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import pydicom
import os
import seaborn as sns
import seaborn as sb
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import LabelEncoder,normalize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import category_encoders as ce
from keras.optimizers import Adam

import cv2

import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn-darkgrid')


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')


## === cell 2
import torch

if torch.cuda.is_available():    

    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


## === cell 3
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')


## === cell 4
sub = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
sub = sub.merge(test.drop('Weeks', axis=1), on="Patient")


## === cell 5
sub


## === cell 6
train.head(100)


## === cell 7
test.head()


## === cell 8
print(train.shape, test.shape)


## === cell 9
print(train.isnull().sum(),'\n')
print(test.isnull().sum())


## === cell 10
train.info()


## === cell 11
plt.figure(figsize=(16,10))


## === cell 12
sns.barplot(train['Sex'].value_counts().index, train['Sex'].value_counts())


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3147645123.py in <cell line: 0>()
----> 1 sns.barplot(train['Sex'].value_counts().index, train['Sex'].value_counts())

TypeError: barplot() takes from 0 to 1 positional arguments but 2 were given

## === cell 13
sns.barplot(train['SmokingStatus'].value_counts().index, train['SmokingStatus'].value_counts())


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4098276230.py in <cell line: 0>()
----> 1 sns.barplot(train['SmokingStatus'].value_counts().index, train['SmokingStatus'].value_counts())

TypeError: barplot() takes from 0 to 1 positional arguments but 2 were given

## === cell 14
sa = pd.crosstab(train['SmokingStatus'],train['Sex'])
sa.plot(kind="bar",title='No of passengers survived')
plt.show()


## === cell 15
plt.rcParams['figure.figsize'] = 10,5
ax = train['Age'].hist(bins = 15,alpha = 0.9, color = 'green')
ax.set(xlabel = 'Age',ylabel = 'Count',title = 'Visualization of Ages')
plt.show()


## === cell 16
plt.rcParams['figure.figsize'] = 10,5
ax = train['Weeks'].hist(bins = 15,alpha = 0.9, color = 'green')
ax.set(xlabel = 'Weeks',ylabel = 'Count',title = 'Visualization of Ages')
plt.show()


## === cell 17
plt.scatter(train['Weeks'],train['FVC'])


## === cell 18
plt.scatter(train['Age'],train['FVC'])


## === cell 19
plt.rcParams['figure.figsize'] = 10,10
sb.heatmap(train.corr(),annot = True,square = True,linewidths = 2,linecolor = 'black')


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3750187939.py in <cell line: 0>()
      1 plt.rcParams['figure.figsize'] = 10,10
----> 2 sb.heatmap(train.corr(),annot = True,square = True,linewidths = 2,linecolor = 'black')

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in corr(self, method, min_periods, numeric_only)
  11047         cols = data.columns
  11048         idx = cols.copy()
> 11049         mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
  11050 
  11051         if method == "pearson":

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in to_numpy(self, dtype, copy, na_value)
   1991         if dtype is not None:
   1992             dtype = np.dtype(dtype)
-> 1993         result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
   1994         if result.dtype is not dtype:
   1995             result = np.asarray(result, dtype=dtype)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in as_array(self, dtype, copy, na_value)
   1692                 arr.flags.writeable = False
   1693         else:
-> 1694             arr = self._interleave(dtype=dtype, na_value=na_value)
   1695             # The underlying data was copied within _interleave, so no need
   1696             # to further copy if copy=True or setting na_value

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in _interleave(self, dtype, na_value)
   1751             else:
   1752                 arr = blk.get_values(dtype)
-> 1753             result[rl.indexer] = arr
   1754             itemmask[rl.indexer] = 1
   1755 

ValueError: could not convert string to float: 'ID00133637202223847701934'

## === cell 20
train[train['FVC'] == train['FVC'].max()]


## === cell 21
train[train['FVC'] == train['FVC'].min()]


## === cell 22
imdir_max = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00219637202258203123958"
imdir_min = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00225637202259339837603"

fig=plt.figure(figsize=(12, 12))

columns = 4
rows = 5

imglist = os.listdir(imdir_max)


for i in range(1, columns*rows +1):
    filename = imdir_max + "/" + str(i) + ".dcm"
    ds = pydicom.dcmread(filename)
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap='jet')
plt.show()


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/937561430.py in <cell line: 0>()
      7 rows = 5
      8 
----> 9 imglist = os.listdir(imdir_max)
     10 
     11 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00219637202258203123958'

## === cell 23
fig=plt.figure(figsize=(12, 12))

columns = 4
rows = 5

imglist = os.listdir(imdir_min)


for i in range(1, columns*rows +1):
    filename = imdir_min + "/" + str(i) + ".dcm"
    ds = pydicom.dcmread(filename)
    fig.add_subplot(rows, columns, i)
    plt.imshow(ds.pixel_array, cmap='jet')
plt.show()


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2424486576.py in <cell line: 0>()
      4 rows = 5
      5 
----> 6 imglist = os.listdir(imdir_min)
      7 
      8 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00225637202259339837603'

## === cell 24
train['Patient_Week'] = train['Patient'].astype(str) + '_' + train['Weeks'].astype(str)
train.head()


## === cell 25
test['Patient_Week'] = test['Patient'].astype(str) + '_' + test['Weeks'].astype(str)
test.head()


## === cell 26
print(train.shape)
print(test.shape)
train


## === cell 27

output = pd.DataFrame()
gb = train.groupby('Patient')
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

train = output[output['Week_passed']!=0].reset_index(drop=True)

train.head()


## === cell 28

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')\
        .rename(columns={'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Percent': 'base_Percent', 'Age': 'base_Age'})
submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
submission['Patient'] = submission['Patient_Week'].apply(lambda x: x.split('_')[0])
submission['predict_Week'] = submission['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)
test = submission.drop(columns=['FVC', 'Confidence']).merge(test, on='Patient')
test['Week_passed'] = test['predict_Week'] - test['base_Week']

test.head()


## === cell 29
print(submission.shape)
print(test.shape)
print(train.shape)


## === cell 30
print(train.isnull().sum(),'\n')
print(test.isnull().sum())


## === cell 31
train.set_index(['Patient_Week'],inplace = True)

test.set_index(['Patient_Week'],inplace = True)


## === cell 32
train


## === cell 33
y = train['FVC']
X = train.drop(['FVC'], axis=1)
X = X.drop(['Patient'], axis=1)

test_X = test.drop(['Patient'], axis=1)


## === cell 34
X


## === cell 35

enc = LabelEncoder()

X['Sex'] = enc.fit_transform(X['Sex'])

X['SmokingStatus'] = enc.fit_transform(X['SmokingStatus'])


## === cell 36
X


## === cell 37
test_X['Sex'] = enc.fit_transform(test_X['Sex'])
test_X['SmokingStatus'] = enc.fit_transform(test_X['SmokingStatus'])


## === cell 38
test_X


## === cell 39
y


## === cell 40

from sklearn.preprocessing import normalize

X = normalize(X)
test_X = normalize(test_X)


## === cell 49

from sklearn.model_selection import KFold


kf = KFold(n_splits=5, random_state=2020, shuffle=True)

for train_index, val_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", val_index)
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]


## === cell 50
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


X_train = X_train.reshape(-1, 1, 8)
X_val  = X_val.reshape(-1, 1, 8)
y_train = y_train.values #convert pd to array
y_train = y_train.reshape(-1, 1,)
y_val = y_val.values #convert pd to array
y_val = y_val.reshape(-1, 1,)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


## === cell 51
from tensorflow.keras.layers import Conv2D,LSTM,LeakyReLU, MaxPooling2D,Concatenate,Input, Dropout, Flatten, Dense, GlobalAveragePooling2D,Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model


    

input_layer = Input(shape=(1,8))
main_rnn_layer = LSTM(64, return_sequences=True, recurrent_dropout=0.2)(input_layer)

    
rnn = LSTM(32)(main_rnn_layer)
dense = Dense(128)(rnn)
dropout_c = Dropout(0.3)(dense)
classes = Dense(1, activation= LeakyReLU(alpha=0.1),name="class")(dropout_c)

model = Model(input_layer, classes)

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=1, factor=0.6),
             EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
model.compile(loss=[tf.keras.losses.MeanSquaredLogarithmicError(),tf.keras.losses.MeanSquaredLogarithmicError()], optimizer="adam")


model.summary()
history = model.fit(X_train, y_train, 
          epochs = 250, 
          batch_size = 16, 
          validation_data=(X_val,  y_val), 
          callbacks=callbacks)


## === cell 52
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss over epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()


## === cell 53
model.load_weights("best_model.h5")

test_X = test_X.reshape(-1, 1,8)


predictions = model.predict(test_X)


## === cell 54
print(predictions.shape)
print(predictions)


## === cell 55
test


## === cell 56
submis = test.copy() #for Patient_Week
submis


## === cell 57
import math

 
submis['FVC_pred'] = predictions[:,0]

submis['Confidence'] = 100
submis['sigma_clipped'] = submis['Confidence'].apply(lambda x: max(x, 70))
submis['diff'] = abs(submis['base_FVC'] - submis['FVC_pred'])
submis['delta'] = submis['diff'].apply(lambda x: min(x, 1000))
submis['score'] = -math.sqrt(2)*submis['delta']/submis['sigma_clipped'] - np.log(math.sqrt(2)*submis['sigma_clipped'])
score = submis['score'].mean()
print(score)


## === cell 58
import scipy as sp
from functools import partial

def loss_func(weight, row):
    confidence = weight
    sigma_clipped = max(confidence, 70)
    diff = abs(row['base_FVC'] - row['FVC_pred'])
    delta = min(diff, 1000)
    score = -math.sqrt(2)*delta/sigma_clipped - np.log(math.sqrt(2)*sigma_clipped)
    return -score

results = []
tk0 = tqdm(submis.iterrows(), total=len(submis))
for _, row in tk0:
    loss_partial = partial(loss_func, row=row)
    weight = [100]
    result = sp.optimize.minimize(loss_partial, weight, method='SLSQP')
    x = result['x']
    results.append(x[0])


## === cell 59
submis['Confidence'] = results
submis['sigma_clipped'] = submis['Confidence'].apply(lambda x: max(x, 70))
submis['diff'] = abs(submis['base_FVC'] - submis['FVC_pred'])
submis['delta'] = submis['diff'].apply(lambda x: min(x, 1000))
submis['score'] = -math.sqrt(2)*submis['delta']/submis['sigma_clipped'] - np.log(math.sqrt(2)*submis['sigma_clipped'])
score = submis['score'].mean()
print(score)


## === cell 60
submis=submis.reset_index()
submis


## === cell 61
submis_final =  submis[['Patient_Week', 'FVC_pred', 'Confidence']].copy()
submis_final = submis_final.rename(columns={"FVC_pred": "FVC"})
submis_final['FVC'] = submis_final['FVC']*1000
submis_final.to_csv('submission.csv', index=False)
submis_final
