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

-6.944662466663177

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
import os
from PIL import Image
import pydicom
import matplotlib.pyplot as plt
import pylab
import cv2
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.backend as K
import tensorflow.keras.regularizers as R
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
EPOCHS = 5
NUM_IMAGES = 140
BATCH_SIZE = 4
FOLDS = 5
IMAGE_DIM = (NUM_IMAGES,60,60)
COMP_DIR = '../input/osic-pulmonary-fibrosis-progression/'
TRAIN_PATH = '../input/osic-pulmonary-fibrosis-progression/train'
TEST_PATH = '../input/osic-pulmonary-fibrosis-progression/test'
SUB_PATH = '../input/osic-pulmonary-fibrosis-progression/sample_submission.csv'

## === cell 2
comp_dir = '../input/osic-pulmonary-fibrosis-progression'

train_data = pd.read_csv(os.path.join(comp_dir,'train.csv'))
sub = pd.read_csv(os.path.join(comp_dir,'sample_submission.csv'))
test_data = pd.read_csv(os.path.join(comp_dir,'test.csv'))
train_data.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])

## === cell 3
train_data_u = train_data
train_data_u = train_data_u.drop_duplicates(subset=['Patient'])
train_data_u = train_data_u.rename(columns={'Weeks':'Base_Week','FVC':'Base_FVC','Percent':'Base_Percent'})
train_data_u['Typical_FVC'] = (train_data_u.Base_FVC.values/train_data_u.Base_Percent.values)*100
train_data = train_data.merge(train_data_u.drop(['Age','Sex','SmokingStatus'],axis=1),on='Patient',how='left')

## === cell 4
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub = sub.drop(['Confidence'],axis=1)
sub =  sub[['Patient','Weeks','Patient_Week']]

## === cell 5
test_data = test_data.rename(columns={'Weeks':'Base_Week','FVC':'Base_FVC','Percent':'Base_Percent'})
test_data['Typical_FVC'] = (test_data.Base_FVC.values/test_data.Base_Percent.values)*100
sub = sub.merge(test_data, how='left', on='Patient')

## === cell 6
train_data['Type'] = 'train'
sub['Type'] = 'test'

## === cell 7
data = train_data.append(sub)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3448103982.py in <cell line: 0>()
----> 1 data = train_data.append(sub)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 8
data

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/391604064.py in <cell line: 0>()
----> 1 data

NameError: name 'data' is not defined

## === cell 9
prediction_col = ["FVC"]
Continuos_cols = ["Weeks","Base_Week","Base_FVC","Typical_FVC","Age","Percent","Base_Percent"]
Categorical_cols = ['Sex','Smoking_status']

## === cell 10
from sklearn.preprocessing import MinMaxScaler

## === cell 11
scaler = MinMaxScaler()
conti = scaler.fit_transform(data[Continuos_cols])
data[Continuos_cols] = conti

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/853137164.py in <cell line: 0>()
      1 scaler = MinMaxScaler()
----> 2 conti = scaler.fit_transform(data[Continuos_cols])
      3 data[Continuos_cols] = conti

NameError: name 'data' is not defined

## === cell 12
print(np.mean(train_data_u.query('SmokingStatus == \'Never smoked\'').Base_Percent.values))
print(np.mean(train_data_u.query('SmokingStatus == \'Currently smokes\'').Base_Percent.values))
print(np.mean(train_data_u.query('SmokingStatus == \'Ex-smoker\'').Base_Percent.values))
print(np.mean(train_data_u.query('Sex == \'Male\'').Base_Percent.values))
print(np.mean(train_data_u.query('Sex == \'Female\'').Base_Percent.values))

## === cell 13
for i in range(len(data['Sex'].values)):
    if data['Sex'].values[i] =='Male':
        data['Sex'].values[i] = 0
    else:
        data['Sex'].values[i] = 1

for i in range(len(data['SmokingStatus'].values)):
    if data['SmokingStatus'].values[i] =='Ex-smoker':
        data['SmokingStatus'].values[i] = 0
    elif data['SmokingStatus'].values[i] =='Never smoked':
        data['SmokingStatus'].values[i] = 1     
    else:
        data['SmokingStatus'].values[i] = 2 

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/159705020.py in <cell line: 0>()
----> 1 for i in range(len(data['Sex'].values)):
      2     if data['Sex'].values[i] =='Male':
      3         data['Sex'].values[i] = 0
      4     else:
      5         data['Sex'].values[i] = 1

NameError: name 'data' is not defined

## === cell 14
x_cols = ['Weeks','Base_Week','Base_FVC','Sex','Age']

## === cell 15
x_train = data[x_cols].loc[data['Type'] == "train"].values.astype(np.float)
y_train = data[prediction_col].loc[data['Type'] == "train"].values.astype(np.float)
x_test = data[x_cols].loc[data['Type'] == "test"].values.astype(np.float)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2896732124.py in <cell line: 0>()
----> 1 x_train = data[x_cols].loc[data['Type'] == "train"].values.astype(np.float)
      2 y_train = data[prediction_col].loc[data['Type'] == "train"].values.astype(np.float)
      3 x_test = data[x_cols].loc[data['Type'] == "test"].values.astype(np.float)

NameError: name 'data' is not defined

## === cell 16
data

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/391604064.py in <cell line: 0>()
----> 1 data

NameError: name 'data' is not defined

## === cell 17
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2474616054.py in <cell line: 0>()
----> 1 x_train = x_train.astype(np.float32)
      2 y_train = y_train.astype(np.float32)
      3 x_test = x_test.astype(np.float32)

NameError: name 'x_train' is not defined

## === cell 18
x_train.shape,y_train.shape,x_test.shape

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1944284163.py in <cell line: 0>()
----> 1 x_train.shape,y_train.shape,x_test.shape

NameError: name 'x_train' is not defined

## === cell 19
type(x_train[0][3])

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/193595592.py in <cell line: 0>()
----> 1 type(x_train[0][3])

NameError: name 'x_train' is not defined

## === cell 20
type(x_train),type(y_train),type(x_test)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3218977339.py in <cell line: 0>()
----> 1 type(x_train),type(y_train),type(x_test)

NameError: name 'x_train' is not defined

## === cell 21
class Data_Generator(tf.keras.utils.Sequence):
    
    def __init__(self,batch_size,patient_ids,tab_data,dim,target=None,train=True,augment=False):
        self.batch_size = batch_size
        self.image_ids = patient_ids
        self.augment = augment
        self.dim = dim
        self.target = target
        self.indices = range(len(self.image_ids))
        self.train = train
        self.tab_data = tab_data
    
    def getimage(self,image_id):
        X1 = np.zeros((NUM_IMAGES,self.dim[1],self.dim[2], 1))
        if self.train:
            path = TRAIN_PATH
        else:
            path = TEST_PATH
        for i,dcm_i in enumerate(os.listdir(os.path.join(path,image_id))):
            try:
                im = pydicom.dcmread(os.path.join(TRAIN_PATH,f'{image_id}/{dcm_i}'))
                img = im.pixel_array/255
                img = cv2.resize(img, (self.dim[1],self.dim[2]))
                img = np.reshape(img,(IMAGE_DIM[1],IMAGE_DIM[2],1))
                X1[i,] = img
                if i>=NUM_IMAGES-1:
                    break
            except:
                continue
        if self.augment == True:
            img = self.ImageAugment(img)
            return img
        return X1
    
    def on_epoch_end(self):
        return self.indices
    
    def getdata(self, image_id_list):
        X = np.empty((self.batch_size,*self.dim, 1))
        for i, im_id in enumerate(image_id_list):
            X[i,] = self.getimage(im_id)
        
        return X
    '''
    def ImageAugment(self,image):
        augmentor = ImageAugmentor(image,axis_point=[self.dim/2,self.dim/2])
        augmentor.cutmix()
        #augmentor.zoom()
        augmentor.flip()
        augmentor.rotate()
        return augmentor.get_image()
    ''' 
    
    def __getitem__(self,index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        image_id_list = [self.image_ids[k] for k in indices]
        tab_X = np.array([self.tab_data[k] for k in indices]).astype(np.float32)
        X = self.getdata(image_id_list)
        if self.train == True:
            target_list = [self.target[k] for k in indices]
            y = np.array(target_list).astype(np.float32)
            return [X,tab_X],y
        return [X,tab_X]
    
    def __len__(self):
        return int(np.floor(len(self.indices)/self.batch_size))
    

## === cell 22
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:,2]-y_pred[:,0]
    fvc_pred = y_pred[:,1]
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:,0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)
def qloss(y_true, y_pred):
    qs = [0.2,0.50,0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e,(q-1)*e)
    return K.mean(v)
def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    return loss


## === cell 23
def build_model():
        
    inp2 = L.Input(shape=(*IMAGE_DIM,1,),name='conv_input')
    
    x = L.Conv3D(32,3,activation='relu',data_format='channels_last')(inp2)
    
    x = L.Conv3D(64,3,activation='relu',data_format='channels_last')(x)
    
    x = L.BatchNormalization()(x)
    
    x1 = L.MaxPooling3D()(x)
    
    x = L.Conv3D(64,3,activation='relu',padding='same',data_format='channels_last')(x1)
    
    x = L.Conv3D(64,3,activation='relu',padding='same',data_format='channels_last')(x)
    
    x = L.add([x,x1])
    
    x = L.Conv3D(128,5,activation='relu',data_format='channels_last')(x)
    
    x = L.Conv3D(128,5,activation='relu',data_format='channels_last')(x)
    
    x = L.BatchNormalization()(x)
    
    x2 = L.MaxPooling3D()(x)
    
    x = L.Conv3D(128,5,activation='relu',padding='same',data_format='channels_last')(x2)
    
    x = L.Conv3D(128,5,activation='relu',padding='same',data_format='channels_last')(x)
    
    x = L.add([x,x2])
    
    x = L.BatchNormalization()(x)
    
    x = L.MaxPooling3D()(x)
    
    x = L.Dropout(0.5)(x)
    
    op1 = L.Flatten()(x)
    
    tab_model = M.load_model('../input/osic-tabularmodel/model.h5',custom_objects={'loss':mloss,'score':score})
    
    tab_model.trainable = False
    
    inp = L.Input((5,),name='input_d')

    d = tab_model.get_layer(name='dense')(inp)
    
    d1 = tab_model.get_layer(name='dense_1')(d)
    
    op2 = tab_model.get_layer(name='dense_2')(d1)    
    
    '''
    inp = L.Input(shape=(4,),name='tab_input')
    
    x = L.Dense(128,activation ='relu',kernel_regularizer=R.l2(5e-4))(inp)
    
    x = L.Dense(128,activation ='relu',kernel_regularizer=R.l2(4e-4))(x)
    
    op2 = L.Dense(64,activation ='relu',kernel_regularizer=R.l2(2e-5))(x)
    '''
    x = L.Concatenate()([op1,op2])
    
    o1 = L.Dense(3,activation = 'linear',name = 'dense_f1')(x)
    o2 = L.Dense(3,activation = 'relu', name='dense_f2')(x)
    
    pred1 = L.Lambda(lambda x: (x[0]+(tf.cumsum(x[1],axis=1))),name='output')([o1,o2])    
    
    
    model = M.Model(inputs=[inp2,inp],outputs=pred1)
    
    model.compile(loss=mloss(0.8), optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=[score])
    
    return model


## === cell 25
folds = 5
KF = KFold(n_splits=folds,shuffle=True,random_state=24)

## === cell 26
x_train = data[x_cols].loc[data['Type'] == "train"].values
y_train = data[prediction_col].loc[data['Type'] == "train"].values
x_test = data[x_cols].loc[data['Type'] == "test"].values

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2813100142.py in <cell line: 0>()
----> 1 x_train = data[x_cols].loc[data['Type'] == "train"].values
      2 y_train = data[prediction_col].loc[data['Type'] == "train"].values
      3 x_test = data[x_cols].loc[data['Type'] == "test"].values

NameError: name 'data' is not defined

## === cell 27
def get_lr_callback():
    lr_start   = 0.001
    lr_max     = 0.001
    lr_min     = 0.0001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.9
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = lr_start - ( lr_start - lr_min ) / lr_ramp_ep * epoch 
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_min
            '''
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            '''
        return lr
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback

## === cell 29
'''%%time
x_train = data[x_cols].loc[data['Type'] == "train"].values
patient_ids = data['Patient'].loc[data['Type']=='train'].values
target_val = data[prediction_col].loc[data['Type'] == "train"].values
for fold,(t_idx,v_idx) in enumerate(KF.split(patient_ids)):
    print(f"################ FOLD {fold+1} #####################")
    train_gen = Data_Generator(BATCH_SIZE,patient_ids[t_idx],x_train[t_idx],IMAGE_DIM,target=target_val[t_idx],train=True)
    val_gen = Data_Generator(BATCH_SIZE,patient_ids[v_idx],x_train[v_idx],IMAGE_DIM,target=target_val[v_idx],train=True)
    total_val_train = len(patient_ids[v_idx])
    total_train = len(patient_ids[t_idx])
    history = model.fit(
        train_gen,
        steps_per_epoch=total_train//BATCH_SIZE,
        epochs=EPOCHS,
        validation_data = val_gen,
        validation_steps = total_val_train//BATCH_SIZE,
        verbose=1,
        callbacks=[get_lr_callback()]
    )
    
    plt.figure()
    plt.plot(list(range(EPOCHS)),history.history['val_loss'])
    plt.plot(list(range(EPOCHS)),history.history['loss'])
    plt.xlabel("Epochs")
    plt.ylabel('loss')
    plt.legend(['val_loss','loss'])
    plt.show()
    
    plt.figure()
    plt.plot(list(range(EPOCHS)),history.history['val_score'])
    plt.plot(list(range(EPOCHS)),history.history['score'])
    plt.xlabel("Epochs")
    plt.ylabel('score')
    plt.legend(['val_score','score'])
    plt.show()'''

## === cell 30
model = M.load_model('../input/tab-data-osic/model.h5',custom_objects={'loss':mloss,'score':score})
model.summary()

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4294446110.py in <cell line: 0>()
----> 1 model = M.load_model('../input/tab-data-osic/model.h5',custom_objects={'loss':mloss,'score':score})
      2 model.summary()

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/tab-data-osic/model.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 31
test_patient_ids = data['Patient'].loc[data['Type'] == "test"].values

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3916550940.py in <cell line: 0>()
----> 1 test_patient_ids = data['Patient'].loc[data['Type'] == "test"].values

NameError: name 'data' is not defined

## === cell 32
test_gen = Data_Generator(2,test_patient_ids,x_test,IMAGE_DIM,train=False)

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1408181201.py in <cell line: 0>()
----> 1 test_gen = Data_Generator(2,test_patient_ids,x_test,IMAGE_DIM,train=False)

NameError: name 'test_patient_ids' is not defined

## === cell 33
print("Inferencing")

## === cell 34
pred = model.predict(test_gen,verbose=1,batch_size=BATCH_SIZE)

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2687103315.py in <cell line: 0>()
----> 1 pred = model.predict(test_gen,verbose=1,batch_size=BATCH_SIZE)

NameError: name 'model' is not defined

## === cell 35
pred

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2686397780.py in <cell line: 0>()
----> 1 pred

NameError: name 'pred' is not defined

## === cell 37
conf = pred[:,2] - pred[:,0]
for i in range(len(conf)):
    conf[i] = max(conf[i],70)

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3454222270.py in <cell line: 0>()
----> 1 conf = pred[:,2] - pred[:,0]
      2 for i in range(len(conf)):
      3     conf[i] = max(conf[i],70)

NameError: name 'pred' is not defined

## === cell 38
pred[:,1]

## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1358112473.py in <cell line: 0>()
----> 1 pred[:,1]

NameError: name 'pred' is not defined

## === cell 39
pred_dict = {'FVC':pred[:,1],'Confidence':conf}
pred_df = pd.DataFrame(pred_dict)

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2381904620.py in <cell line: 0>()
----> 1 pred_dict = {'FVC':pred[:,1],'Confidence':conf}
      2 pred_df = pd.DataFrame(pred_dict)

NameError: name 'pred' is not defined

## === cell 40
sub['Confidence'] = pred_df['Confidence']
sub['FVC'] = pred_df['FVC']

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2791545372.py in <cell line: 0>()
----> 1 sub['Confidence'] = pred_df['Confidence']
      2 sub['FVC'] = pred_df['FVC']

NameError: name 'pred_df' is not defined

## === cell 41
subm = sub[['Patient_Week','FVC','Confidence']].copy()

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1029027480.py in <cell line: 0>()
----> 1 subm = sub[['Patient_Week','FVC','Confidence']].copy()

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

KeyError: "['FVC', 'Confidence'] not in index"

## === cell 42
subm.to_csv("submission.csv", index=False)

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1915734318.py in <cell line: 0>()
----> 1 subm.to_csv("submission.csv", index=False)

NameError: name 'subm' is not defined
