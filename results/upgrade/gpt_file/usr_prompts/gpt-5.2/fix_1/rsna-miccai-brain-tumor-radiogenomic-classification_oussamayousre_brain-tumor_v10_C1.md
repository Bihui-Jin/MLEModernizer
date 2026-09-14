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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

0.5278344382117967

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
import pydicom as dicom
import cv2
import os 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image.dataframe_iterator import DataFrameIterator
import matplotlib.pylab as plt
from tensorflow.keras.utils import Sequence
from keras.utils import data_utils
from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import pandas as pd
from glob import glob

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv('../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv')
test_df = pd.read_csv('../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv')
train_array = np.array(train_df['BraTS21ID'])
train_label =  np.array(train_df['MGMT_value'])
test_array = np.array(test_df['BraTS21ID'])
test_label =  np.array(test_df['MGMT_value'])
DIM = 512
NB_CHANNELS = 1
i = 0
BATCH_SIZE = 16 
a = 0
train_ds = pd.DataFrame(columns = ['path','label'] )
test_ds = pd.DataFrame(columns = ['path','label'] )

## === cell 2
len(train_df)

## === cell 3
test_df

## === cell 4
print(str(10).zfill(5))

## === cell 5
dataframe_values = []
for a,b in zip(train_df['BraTS21ID'],train_df['MGMT_value']):
    folders_list = os.listdir(f'../input/rsna-miccai-brain-tumor-radiogenomic-classification/train/{str(a).zfill(5)}')
    for folder in folders_list :  
        img_list = os.listdir(f'../input/rsna-miccai-brain-tumor-radiogenomic-classification/train/{str(a).zfill(5)}/{folder}')
        for img in img_list : 
            dataframe_values.append({'path' : f'../input/rsna-miccai-brain-tumor-radiogenomic-classification/train/{str(a).zfill(5)}/{folder}/{img}','label' : b})
train_ds  = pd.DataFrame.from_dict(dataframe_values)



## === cell 6
train_ds

## === cell 8
dataframe_values = []
for a,b in zip(test_df['BraTS21ID'],test_df['MGMT_value']):
    folders_list = os.listdir(f'../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/{str(a).zfill(5)}')
    for folder in folders_list :    
        img_list = os.listdir(f'../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/{str(a).zfill(5)}/{folder}')
        for img in img_list : 
            dataframe_values.append({'path' : f'../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/{str(a).zfill(5)}/{folder}/{img}','label' : b})
test_ds  = pd.DataFrame.from_dict(dataframe_values)


## === cell 9
test_ds

## === cell 10
"""train_augmentation_parameters = dict(
    rescale=1.0/255.0,
    rotation_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range = [0.8, 1.2])

valid_augmentation_parameters = dict(
    rescale=1.0/255.0)

# Using the training phase generators 
train_augmenter = ImageDataGenerator(**train_augmentation_parameters)
valid_augmenter = ImageDataGenerator(**valid_augmentation_parameters)"""

## === cell 12
"""class Frames_Generator(data_utils.Sequence):
    'Generates data for Keras'
    def __init__(self,Sample_array, batch_size=1, dim=(512,512), shuffle=True,  train = True ):
        'Initialization'
        self.batch_size = batch_size
        self.Sample_array = Sample_array 
        self.shuffle = shuffle
        self.dim = dim
        self.train = train
        self.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.Sample_array) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [str(self.Sample_array[k]).zfill(5) for k in indexes]
        #list_IDs_temp = [20]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp, index)
        X =  np.array([X])[0]
        y =  np.array([y])
        y = y.reshape(y.shape[1])
        randomize = np.arange(y.shape[0])
        np.random.shuffle(randomize)
        X = X[randomize]
        y = y[randomize]
        #train_generator = train_augmenter.flow(X ,y,batch_size = 32 )
        return X, y
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.Sample_array))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp,index):
        label_train  = []
        Frame_train = [] 
        global a 
        global b
        print(list_IDs_temp[0])
        if self.train :
            folders_list = os.listdir(f'../input/rsna-miccai-brain-tumor-radiogenomic-classification/train/{list_IDs_temp[0]}')
            for folder in folders_list : 
                img_list = os.listdir(f'../input/rsna-miccai-brain-tumor-radiogenomic-classification/train/{list_IDs_temp[0]}/{folder}')
                for img in img_list :
                    ds = dicom.dcmread(f'../input/rsna-miccai-brain-tumor-radiogenomic-classification/train/{list_IDs_temp[0]}/{folder}/{img}').pixel_array
                    img = cv2.resize(ds,self.dim)
                    img = img.reshape(self.dim[0],self.dim[1],1)
                    label_train.append(train_label[index])
                    Frame_train.append(img)
        else : 
            folders_list = os.listdir(f'../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/{list_IDs_temp[0]}')
            for folder in folders_list : 
                img_list = os.listdir(f'../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/{list_IDs_temp[0]}/{folder}')
                for img in img_list :
                    ds = dicom.dcmread(f'../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/{list_IDs_temp[0]}/{folder}/{img}').pixel_array
                    img = cv2.resize(ds,self.dim)
                    img = img.reshape(dim[0],dim[1],1)
                    label_train.append(test_label[index])
                    Frame_train.append(img)    
        return Frame_train, label_train"""

## === cell 13
"""# Parameters
params = {'dim': (512,512),
          'batch_size': 1,
          'shuffle': True, 
           'train' : True }
valid_params = {'dim': (512,512),
          'batch_size': 1,
          'shuffle': True,
          'train':False}
training_generator = Frames_Generator(train_array, **params)
validation_generator = Frames_Generator(test_array, **valid_params)"""


## === cell 15
class Frames_Generator(data_utils.Sequence):
    'Generates data for Keras'
    def __init__(self,Sample_df = np.arange(len(train_ds)//BATCH_SIZE-16) , batch_size=1, dim=(512,512), shuffle=True ,train_ds = train_ds,nb_steps = 0 ):
        'Initialization'
        self.batch_size = batch_size
        self.Sample_df = Sample_df
        self.shuffle = shuffle
        self.dim = dim
        self.train_ds = train_ds
        self.nb_steps = nb_steps
        self.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.Sample_df) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [str(self.Sample_df[k]).zfill(5) for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        X =  np.array([X])[0]
        y =  np.array([y])
        y = y.reshape(y.shape[1])
        randomize = np.arange(y.shape[0])
        np.random.shuffle(randomize)
        X = X[randomize]
        y = y[randomize]
        return X/255, y
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.Sample_df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        label_train  = []
        Frame_train = [] 
        i = 0
        while(i<16) : 
            ds = dicom.dcmread(self.train_ds['path'][self.nb_steps]).pixel_array
            img = cv2.resize(ds,self.dim)
            img = img.reshape(self.dim[0],self.dim[1],1)
            label = self.train_ds['label'][self.nb_steps]
            label_train.append(self.train_ds['label'][self.nb_steps])
            Frame_train.append(img)
            self.nb_steps += 1 
            i +=1
            if  self.nb_steps == len(self.train_ds) :
                self.nb_steps = 0
        return Frame_train, label_train

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1826140627.py in <cell line: 0>()
----> 1 class Frames_Generator(data_utils.Sequence):
      2     'Generates data for Keras'
      3     def __init__(self,Sample_df = np.arange(len(train_ds)//BATCH_SIZE-16) , batch_size=1, dim=(512,512), shuffle=True ,train_ds = train_ds,nb_steps = 0 ):
      4         'Initialization'
      5         self.batch_size = batch_size

NameError: name 'data_utils' is not defined

## === cell 16
len(train_ds)//BATCH_SIZE-16

## === cell 17
train_params = {'Sample_df':np.arange(len(train_ds)//BATCH_SIZE-16),
          'dim': (512,512),
          'batch_size': 1,
          'shuffle': True, 
           'train_ds' : train_ds,
         'nb_steps': 0}
test_params = {'Sample_df':np.arange(len(test_ds)//BATCH_SIZE-16),
          'dim': (512,512),
          'batch_size': 1,
          'shuffle': True, 
           'train_ds' : test_ds,
         'nb_steps': 0}
training_generator = Frames_Generator(**train_params)
validation_generator = Frames_Generator(**test_params)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1555342274.py in <cell line: 0>()
     12            'train_ds' : test_ds,
     13          'nb_steps': 0}
---> 14 training_generator = Frames_Generator(**train_params)
     15 validation_generator = Frames_Generator(**test_params)

NameError: name 'Frames_Generator' is not defined

## === cell 18
"""x , y = validation_generator.__getitem__(0)"""

## === cell 19
"""arr = np.arange(len(train_ds)//BATCH_SIZE)
len(arr)"""

## === cell 20
"""
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same",input_shape = (DIM , DIM , NB_CHANNELS)))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.2))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Dropout(0.2))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(16))
model.add(Dropout(0.2))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.build((0,512,512,3))
model.summary()"""

## === cell 21
"""from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy',
                                            patience = 2,
                                            verbose=1,
                                            factor=0.1,
                                            min_lr=0.000001)

opt = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer = opt, loss='binary_crossentropy', metrics=['accuracy'])"""

## === cell 22
"""model.fit(training_generator,epochs = 1,callbacks = [learning_rate_reduction])"""


## === cell 23
model = tf.keras.models.load_model('../input/tumor-model/model.h5')

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/533861300.py in <cell line: 0>()
----> 1 model = tf.keras.models.load_model('../input/tumor-model/model.h5')

NameError: name 'tf' is not defined

## === cell 24
result = model.predict(validation_generator, verbose = True, workers = 2)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3742455240.py in <cell line: 0>()
----> 1 result = model.predict(validation_generator, verbose = True, workers = 2)

NameError: name 'model' is not defined

## === cell 25
result.shape

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2123009515.py in <cell line: 0>()
----> 1 result.shape

NameError: name 'result' is not defined

## === cell 26
data_dir = '../input/rsna-miccai-brain-tumor-radiogenomic-classification/test'
patients_test = sorted(os.listdir(data_dir))

## === cell 27
patients_test

## === cell 28
final_result = []
i = 0
for patient in patients_test:
    arr = os.listdir('../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/' + patient)
    length = 0
    for j in arr : 
        lenn = len(glob('../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/' + patient +'/'+j+'/*.dcm'))
        length = lenn + length 
    final_result.append(result[i:i+length].sum()/length)
    i += length
    


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2569551358.py in <cell line: 0>()
      5     length = 0
      6     for j in arr :
----> 7         lenn = len(glob('../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/' + patient +'/'+j+'/*.dcm'))
      8         length = lenn + length
      9     final_result.append(result[i:i+length].sum()/length)

NameError: name 'glob' is not defined

## === cell 29
final_result

## === cell 30
test = pd.read_csv('../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv')


## === cell 31
submission = pd.DataFrame({"BraTS21ID": test['BraTS21ID'].apply(lambda x: str(x).zfill(5)), "MGMT_value": final_result})
submission


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4021862527.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"BraTS21ID": test['BraTS21ID'].apply(lambda x: str(x).zfill(5)), "MGMT_value": final_result})
      2 submission

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

ValueError: array length 0 does not match index length 59

## === cell 32
submission.to_csv('submission.csv', index = 0)

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2386182416.py in <cell line: 0>()
----> 1 submission.to_csv('submission.csv', index = 0)

NameError: name 'submission' is not defined
