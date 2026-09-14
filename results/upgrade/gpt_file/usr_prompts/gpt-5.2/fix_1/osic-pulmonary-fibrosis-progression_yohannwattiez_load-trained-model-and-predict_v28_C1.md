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

-8.658902507542287

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import random
import pickle
import time

import pydicom
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K
from tensorflow import keras as K
from tensorflow.keras import layers as L

from skimage.transform import resize
from math import ceil

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
raw_test = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')
X_prediction = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

## === cell 2
TEST_PATH = '/kaggle/input/osic-pulmonary-fibrosis-progression/test'

DESIRED_SIZE = (20,512,512)
BATCH_SIZE = 4

## === cell 3
def create_submission(value):
    if value==0:
        sub = pd.DataFrame(data={'Patient_Week':X_prediction['Patient_Week'], 'FVC':[0 for i in X_prediction.index], 'Confidence': [10000 for i in X_prediction.index]})
    elif value==1:
        sub = pd.DataFrame(data={'Patient_Week':X_prediction['Patient_Week'], 'FVC':[100 for i in X_prediction.index], 'Confidence': [5000 for i in X_prediction.index]})
    elif value==2:
        sub = pd.DataFrame(data={'Patient_Week':X_prediction['Patient_Week'], 'FVC':[500 for i in X_prediction.index], 'Confidence': [1000 for i in X_prediction.index]})
    elif value==3:
        sub = pd.DataFrame(data={'Patient_Week':X_prediction['Patient_Week'], 'FVC':[1000 for i in X_prediction.index], 'Confidence': [500 for i in X_prediction.index]})
    elif value==4:
        sub = pd.DataFrame(data={'Patient_Week':X_prediction['Patient_Week'], 'FVC':[2000 for i in X_prediction.index], 'Confidence': [100 for i in X_prediction.index]})
    return sub
        

## === cell 5
X_prediction['Patient'] = X_prediction['Patient_Week'].str.extract(r'(.*)_.*')
X_prediction['Weeks'] = X_prediction['Patient_Week'].str.extract(r'.*_(.*)').astype(int)
X_prediction = X_prediction[['Patient', 'Weeks', 'Patient_Week']]
rename_cols = {'Weeks_y':'Base_week', 'Weeks_x': 'Weeks', 'Percent':'Base_percent', 'FVC':'Base_FVC'}
X_prediction = X_prediction.merge(raw_test, how='left', left_on='Patient', right_on='Patient').rename(columns=rename_cols)[['Patient', 'Base_week', 'Base_FVC', 'Base_percent', 'Age', 'Sex', 'SmokingStatus', 'Weeks', 'Patient_Week']].reset_index(drop=True)

## === cell 6
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

class data_preparation():
    def __init__(self):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = OneHotEncoder()
        
        
    def __call__(self, data_untransformed):
        data = data_untransformed.copy(deep=True)
        
        try:
            data['Sex'] = self.enc_sex.transform(data['Sex'].values)
            data['SmokingStatus'] = self.enc_smok.transform(data['SmokingStatus'].values)
            data = pd.concat([data.drop(columns=['SmokingStatus']), self.onehotenc_smok.transform(data['SmokingStatus'].values.reshape(-1,1), categories=self.enc_smok.classes_, name='', index=data.index).astype(int)], axis=1)
            
        except NotFittedError:
            data['Sex'] = self.enc_sex.fit_transform(data['Sex'].values)
            data['SmokingStatus'] = self.enc_smok.fit_transform(data['SmokingStatus'].values)
            data = pd.concat([data.drop(columns=['SmokingStatus']), self.onehotenc_smok.fit_transform(data['SmokingStatus'].values.reshape(-1,1), categories=self.enc_smok.classes_, name='', index=data.index).astype(int)], axis=1)
            
        return data

## === cell 7
pickefile = open('/kaggle/input/prep-data/data_prep', 'rb')
data_prep = pickle.load(pickefile)
pickefile.close()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3820022542.py in <cell line: 0>()
----> 1 pickefile = open('/kaggle/input/prep-data/data_prep', 'rb')
      2 data_prep = pickle.load(pickefile)
      3 pickefile.close()

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/prep-data/data_prep'

## === cell 8
X_prediction = data_prep(X_prediction).sort_values('Patient')


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2842353034.py in <cell line: 0>()
----> 1 X_prediction = data_prep(X_prediction).sort_values('Patient')

NameError: name 'data_prep' is not defined

## === cell 12
def sort_function(x): #Get the files in the right order
    return int(x.split('.')[0])

def _read(path, patients=[], desired_size=(60,512,512)):
    X = np.empty(np.concatenate(([len(patients)], [1], np.array(desired_size))))
    i=0
    for patient in patients:
        list_patient_files = sorted([i for i in os.listdir(path+'/'+patient)], key=sort_function)
        list_patient_files = list_patient_files[::max(int(len(list_patient_files)/30),1)][:20]
        df = pd.DataFrame(list_patient_files).apply(lambda x:path+'/'+patient+'/'+x)
        df = np.array(df.iloc[:,0].apply(lambda x: resize(pydicom.dcmread(x).pixel_array, DESIRED_SIZE[1:3])).to_list())
        X[i,0,:,:,:] = df
        i+=1
            
    return X

## === cell 15
temp_SELECTED_COLUMNS = ['Weeks', 'Base_week', 'Base_FVC', 'Base_percent', 'Age', 'Sex', '_Currently smokes', '_Ex-smoker', '_Never smoked']
class DataGenerator(K.utils.Sequence):
    
    def on_epoch_end(self):#Indices=np.arange(size of dataset)
        self.indices = np.arange(len(self.list_IDs))
    
    def __len__(self):
        return int(ceil(len(self.indices) / self.batch_size))
    
    def __init__(self, train, list_IDs, batch_size=1, desired_size=(10,512,512), img_path=TEST_PATH, *args, **kwargs):
        self.train = train
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.desired_size = desired_size
        self.img_path = img_path
        self.on_epoch_end()
    
    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        
        patients = self.train.loc[list_IDs_temp, 'Patient'].unique()
        
        imgs = _read(self.img_path, patients=patients, desired_size=self.desired_size)
        
        return self.train.loc[list_IDs_temp, :].reset_index(drop=True), np.transpose(np.asarray(imgs), (0, 2, 3, 4, 1))

        


## === cell 17
model = K.models.load_model('/kaggle/input/cnn-for-latent-features/model',compile=False)
CNN = K.models.load_model('/kaggle/input/cnn-for-latent-features/CNN',compile=False)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3675939518.py in <cell line: 0>()
      1 # model = K.models.load_model('/kaggle/input/cnn-for-latent-features/model',custom_objects={'mloss':mloss(LAMBDA_LOSS)})
----> 2 model = K.models.load_model('/kaggle/input/cnn-for-latent-features/model',compile=False)
      3 CNN = K.models.load_model('/kaggle/input/cnn-for-latent-features/CNN',compile=False)

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    204         )
    205     else:
--> 206         raise ValueError(
    207             f"File format not supported: filepath={filepath}. "
    208             "Keras 3 only supports V3 `.keras` files and "

ValueError: File format not supported: filepath=/kaggle/input/cnn-for-latent-features/model. Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, use `keras.layers.TFSMLayer(/kaggle/input/cnn-for-latent-features/model, call_endpoint='serving_default')` (note that your `call_endpoint` might have a different name).

## === cell 19
pred_SELECTED_COLUMNS = ['Weeks', 'Base_week', 'Patient', 'Base_FVC', 'Base_percent', 'Age', 'Sex', '_Currently smokes', '_Ex-smoker', '_Never smoked']
pred_generator = DataGenerator(X_prediction[pred_SELECTED_COLUMNS], X_prediction.index, batch_size=BATCH_SIZE, desired_size=DESIRED_SIZE, img_path=TEST_PATH)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3455106454.py in <cell line: 0>()
      1 pred_SELECTED_COLUMNS = ['Weeks', 'Base_week', 'Patient', 'Base_FVC', 'Base_percent', 'Age', 'Sex', '_Currently smokes', '_Ex-smoker', '_Never smoked']
----> 2 pred_generator = DataGenerator(X_prediction[pred_SELECTED_COLUMNS], X_prediction.index, batch_size=BATCH_SIZE, desired_size=DESIRED_SIZE, img_path=TEST_PATH)

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

KeyError: "['_Currently smokes', '_Ex-smoker', '_Never smoked'] not in index"

## === cell 24
y_prediction = np.array([[0,0,0]])
for X1, X2 in pred_generator:
    out_imgs = CNN(X2)
    patients = X1.loc[:, 'Patient'].unique()
    X_imgs = np.empty([X1.shape[0], out_imgs.shape[1]])
    for i in range(len(patients)):
        X_imgs[np.where(X1['Patient']==patients[i])] = out_imgs.numpy()[i]
    inp_mlp = [tf.keras.layers.concatenate([0.1*X_imgs, np.asarray(X1[temp_SELECTED_COLUMNS])], axis=1)]
    sub = create_submission(2)
    sub.to_csv('submission.csv', index=False)
    for layer in model.layers[-3:]:
        inp_mlp.append(layer(inp_mlp[-1]))
    y_prediction = np.append(y_prediction, inp_mlp[-1].numpy(), axis=0)
y_prediction = y_prediction[1:]

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2081501775.py in <cell line: 0>()
      1 y_prediction = np.array([[0,0,0]])
----> 2 for X1, X2 in pred_generator:
      3     out_imgs = CNN(X2)
      4     patients = X1.loc[:, 'Patient'].unique()
      5     X_imgs = np.empty([X1.shape[0], out_imgs.shape[1]])

NameError: name 'pred_generator' is not defined

## === cell 27
sub = pd.DataFrame(data={'Patient_Week':X_prediction['Patient_Week'], 'FVC':y_prediction[:,1], 'Confidence': (y_prediction[:,2]-y_prediction[:,0])})
sub.to_csv('submission.csv', index=False)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3251589677.py in <cell line: 0>()
----> 1 sub = pd.DataFrame(data={'Patient_Week':X_prediction['Patient_Week'], 'FVC':y_prediction[:,1], 'Confidence': (y_prediction[:,2]-y_prediction[:,0])})
      2 # sub = pd.DataFrame([0])
      3 sub.to_csv('submission.csv', index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    688                     f"length {len(index)}"
    689                 )
--> 690                 raise ValueError(msg)
    691         else:
    692             index = default_index(lengths[0])

ValueError: array length 1 does not match index length 1908
