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

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
protobuf==6.33.0
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.58439

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pydicom as dicom
import matplotlib.pylab as plt

image_path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/00185/T2w/Image-40.dcm'
ds = dicom.dcmread(image_path)

plt.imshow(ds.pixel_array)


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3290155313.py in <cell line: 0>()
      4 # specify your image path
      5 image_path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/00185/T2w/Image-40.dcm'
----> 6 ds = dicom.dcmread(image_path)
      7 
      8 plt.imshow(ds.pixel_array)

/usr/local/lib/python3.11/dist-packages/pydicom/filereader.py in dcmread(fp, defer_size, stop_before_pixels, force, specific_tags)
   1040         caller_owns_file = False
   1041         logger.debug(f"Reading file '{fp}'")
-> 1042         fp = open(fp, "rb")
   1043     elif (
   1044         fp is None

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/00185/T2w/Image-40.dcm'

## === cell 1
import numpy as np
import pydicom
def load_dicom(path):
    
    data = pydicom.dcmread(path)
    '''
    Returns the image data as a numpy array.
    '''  
    if np.max(data.pixel_array)==0:
        img = data.pixel_array
    else:
        img = data.pixel_array/np.max(data.pixel_array)
        img = (img * 255).astype(np.uint8)
        
    return img


## === cell 2
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

df = pd.read_csv('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv')
data = df.set_index("BraTS21ID")
data = data.drop([109, 123, 709], axis=0)
df=data.reset_index()


## === cell 3
import numpy as np
labels=np.array(df['MGMT_value'])


## === cell 4
from glob import glob
imagePatches = glob('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/*/', recursive=True)


## === cell 5
import os
files=[]
files.append ('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/00109/')
files.append ('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/00709/')
files.append ('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/00123/')
for file in files:
    imagePatches.remove(file)


## === cell 6
len(imagePatches)


## === cell 7
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


## === cell 8
imagePatches.sort(key=natural_keys)
flair_patches = []
t1w_patches=[]
t1wce_patches=[]
t2w_patches=[]
for subfolder in imagePatches:
    flair_patches.append(glob(subfolder + 'FLAIR/**/**/*.dcm', recursive=True))
    t1w_patches.append(glob(subfolder + 'T1w/**/**/*.dcm', recursive=True))
    t1wce_patches.append(glob(subfolder + 'T1wCE/**/**/*.dcm', recursive=True))
    t2w_patches.append(glob(subfolder + 'T2w/**/**/*.dcm', recursive=True))


## === cell 9
def all_slice(sequence, list_name):
    for x in range(0,len(sequence)):
        sequence[x].sort(key=natural_keys)
        list_name.append((sequence[x][0:len(sequence[x])]))
    return list_name


## === cell 10
all_flair_patches = []
all_slice(flair_patches,all_flair_patches)
all_t1w_patches =[]
all_t1w_patches = all_slice(t1w_patches,all_t1w_patches)
all_t1wce_patches=[]
all_t1wce_patches = all_slice(t1wce_patches,all_t1wce_patches)
all_t2w_patches=[]
all_t2w_patches = all_slice(t2w_patches,all_t2w_patches)


## === cell 11
import cv2
def create_input(patches):
    inputs=np.zeros((582,256,256,3))
    for i in range(0,len(patches)):
        for j in range(len(patches[i])//2,(len(patches[i])//2)+3):
            img= load_dicom(patches[i][j])
            image_array = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
            image_array = np.expand_dims(image_array, -1)
        inputs[i]=image_array
    return inputs


## === cell 12
t2w_inputs = create_input(all_t2w_patches)
flair_inputs = create_input(all_flair_patches)
t1wce_inputs = create_input(all_t1wce_patches)
t1w_inputs = create_input(all_t1w_patches)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1831883774.py in <cell line: 0>()
----> 1 t2w_inputs = create_input(all_t2w_patches)
      2 flair_inputs = create_input(all_flair_patches)
      3 t1wce_inputs = create_input(all_t1wce_patches)
      4 t1w_inputs = create_input(all_t1w_patches)

/tmp/ipykernel_11/2630136278.py in create_input(patches)
      4     for i in range(0,len(patches)):
      5         for j in range(len(patches[i])//2,(len(patches[i])//2)+3):
----> 6             img= load_dicom(patches[i][j])
      7             image_array = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
      8             image_array = np.expand_dims(image_array, -1)

IndexError: list index out of range

## === cell 13
import os
import numpy as np
import pandas as pd 
import random
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
import keras

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam, Adamax, Adagrad
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 14
flair_inputs= np.asarray(flair_inputs)
t1w_inputs= np.asarray(t1w_inputs)
t1wce_inputs= np.asarray(t1wce_inputs)
t2w_inputs = np.asarray(t2w_inputs)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1233584666.py in <cell line: 0>()
----> 1 flair_inputs= np.asarray(flair_inputs)
      2 t1w_inputs= np.asarray(t1w_inputs)
      3 t1wce_inputs= np.asarray(t1wce_inputs)
      4 t2w_inputs = np.asarray(t2w_inputs)

NameError: name 'flair_inputs' is not defined

## === cell 15
from sklearn.model_selection import train_test_split
from IPython.display import Image
from sklearn.metrics import confusion_matrix
np.random.seed(1)


## === cell 16
from tensorflow.keras import layers
def get_model(optimizer):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((256,256,3,1))
    
    

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu",padding='same')(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)


    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu",padding='same')(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3dcnn")
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model


model = get_model('Adam')


## === cell 17
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, mode='auto', verbose = 1)
callbacks = [ModelCheckpoint(filepath='/kaggle/output/models/best_model.h5', monitor='val_loss', save_best_only=True)]
x_train, x_valid, y_train, y_valid = train_test_split(flair_inputs, labels, test_size = 0.2, random_state = 1)
optimizers= ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=3, mode='min')
x_train = x_train/255
x_valid = x_valid/255
model=get_model('Adam')
history = model.fit(x_train, y_train, epochs=50, batch_size=8,shuffle=True, validation_data=(x_valid, y_valid),callbacks=[early_stop])


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1208176736.py in <cell line: 0>()
----> 1 reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, mode='auto', verbose = 1)
      2 callbacks = [ModelCheckpoint(filepath='/kaggle/output/models/best_model.h5', monitor='val_loss', save_best_only=True)]
      3 x_train, x_valid, y_train, y_valid = train_test_split(flair_inputs, labels, test_size = 0.2, random_state = 1)
      4 optimizers= ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
      5 early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=3, mode='min')

NameError: name 'ReduceLROnPlateau' is not defined

## === cell 18
model.save('/kaggle/t1winputs.hdf5')
model_t1w= keras.models.load_model('/kaggle/t1winputs.hdf5')


## === cell 19
model.save('/kaggle/t1wceinputs.hdf5')
model_t1wce= keras.models.load_model('/kaggle/t1wceinputs.hdf5')


## === cell 20
model.save('/kaggle/t2winputs.hdf5')
model_t2w= keras.models.load_model('/kaggle/t2winputs.hdf5')


## === cell 21
model.save('/kaggle/flairinputs.hdf5')
model_flair= keras.models.load_model('/kaggle/flairinputs.hdf5')


## === cell 22
preds_flair=model_flair.predict(x_valid)
preds_t1w=model_t1w.predict(x_valid)
preds_t2w=model_t2w.predict(x_valid)
preds_t1wce=model_t1wce.predict(x_valid)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2047001481.py in <cell line: 0>()
----> 1 preds_flair=model_flair.predict(x_valid)
      2 preds_t1w=model_t1w.predict(x_valid)
      3 preds_t2w=model_t2w.predict(x_valid)
      4 preds_t1wce=model_t1wce.predict(x_valid)

NameError: name 'x_valid' is not defined

## === cell 23
mean = [(g + h + a + b) / 4 for g, h, a, b in zip(preds_flair,preds_t1w, preds_t2w,preds_t1wce)]


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3665956134.py in <cell line: 0>()
----> 1 mean = [(g + h + a + b) / 4 for g, h, a, b in zip(preds_flair,preds_t1w, preds_t2w,preds_t1wce)]

NameError: name 'preds_flair' is not defined

## === cell 24
from sklearn.metrics import roc_curve,roc_auc_score

fpr , tpr , thresholds = roc_curve ( y_valid , mean)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2268048544.py in <cell line: 0>()
      1 from sklearn.metrics import roc_curve,roc_auc_score
      2 
----> 3 fpr , tpr , thresholds = roc_curve ( y_valid , mean)

NameError: name 'y_valid' is not defined

## === cell 25
def plot_roc_curve(fpr,tpr): 
  plt.plot(fpr,tpr) 
  plt.axis([0,1,0,1]) 
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate') 
  plt.show()    
  
plot_roc_curve (fpr,tpr) 


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2263744642.py in <cell line: 0>()
      6   plt.show()
      7 
----> 8 plot_roc_curve (fpr,tpr)

NameError: name 'fpr' is not defined

## === cell 26
auc_score=roc_auc_score(y_valid,mean)
print(auc_score)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2259517247.py in <cell line: 0>()
----> 1 auc_score=roc_auc_score(y_valid,mean)
      2 print(auc_score)

NameError: name 'y_valid' is not defined

## === cell 27
model_t1w= keras.models.load_model('/kaggle/t1winputs.hdf5')
model_t1wce= keras.models.load_model('/kaggle/t1wceinputs.hdf5')
model_t2w= keras.models.load_model('/kaggle/t2winputs.hdf5')
model_flair= keras.models.load_model('/kaggle/flairinputs.hdf5')


## === cell 28
import cv2
def create_input(patches, number_image):
    inputs=np.zeros((number_image,256,256,3))
    for i in range(0,len(patches)):
        for j in range(len(patches[i])//2,(len(patches[i])//2)+3):
            img= load_dicom(patches[i][j])
            image_array = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
            image_array = np.expand_dims(image_array, -1)
        inputs[i]=image_array
    return inputs


## === cell 29
testimages= glob('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test/*/', recursive=True)
testimages.sort(key=natural_keys)
flair_patches = []
t1w_patches=[]
t1wce_patches=[]
t2w_patches=[]
for subfolder in testimages:
    flair_patches.append(glob(subfolder + 'FLAIR/**/**/*.dcm', recursive=True))
    t1w_patches.append(glob(subfolder + 'T1w/**/**/*.dcm', recursive=True))
    t1wce_patches.append(glob(subfolder + 'T1wCE/**/**/*.dcm', recursive=True))
    t2w_patches.append(glob(subfolder + 'T2w/**/**/*.dcm', recursive=True))
all_flair_patches = []
all_slice(flair_patches,all_flair_patches)
all_t1w_patches =[]
all_t1w_patches = all_slice(t1w_patches,all_t1w_patches)
all_t1wce_patches=[]
all_t1wce_patches = all_slice(t1wce_patches,all_t1wce_patches)
all_t2w_patches=[]
all_t2w_patches = all_slice(t2w_patches,all_t2w_patches)
t2w_inputs = create_input(all_t2w_patches, len(testimages))
flair_inputs = create_input(all_flair_patches,len(testimages))
t1wce_inputs = create_input(all_t1wce_patches,len(testimages))
t1w_inputs = create_input(all_t1w_patches,len(testimages))
testflair_inputs= np.asarray(flair_inputs)/255
testt1w_inputs= np.asarray(t1w_inputs)/255
testt1wce_inputs= np.asarray(t1wce_inputs)/255
testt2w_inputs = np.asarray(t2w_inputs)/255
preds_flair=model_flair.predict(testflair_inputs)
preds_t1w=model_t1w.predict(testt1w_inputs)
preds_t2w=model_t2w.predict(testt2w_inputs)
preds_t1wce=model_t1wce.predict(testt1wce_inputs)
mean = [(g + h + a + b) / 4 for g, h, a, b in zip(preds_flair,preds_t1w, preds_t2w,preds_t1wce)]


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/592727933.py in <cell line: 0>()
     18 all_t2w_patches=[]
     19 all_t2w_patches = all_slice(t2w_patches,all_t2w_patches)
---> 20 t2w_inputs = create_input(all_t2w_patches, len(testimages))
     21 flair_inputs = create_input(all_flair_patches,len(testimages))
     22 t1wce_inputs = create_input(all_t1wce_patches,len(testimages))

/tmp/ipykernel_11/175905925.py in create_input(patches, number_image)
      4     for i in range(0,len(patches)):
      5         for j in range(len(patches[i])//2,(len(patches[i])//2)+3):
----> 6             img= load_dicom(patches[i][j])
      7             image_array = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
      8             image_array = np.expand_dims(image_array, -1)

IndexError: list index out of range

## === cell 30
sample=pd.read_csv("/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv",index_col="BraTS21ID")
sample["MGMT_value"] = 0
sample["MGMT_value"] = [str(a)[1:-1] for a in mean]
sample["MGMT_value"].to_csv("submission.csv")


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2293842554.py in <cell line: 0>()
      1 sample=pd.read_csv("/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv",index_col="BraTS21ID")
      2 sample["MGMT_value"] = 0
----> 3 sample["MGMT_value"] = [str(a)[1:-1] for a in mean]
      4 sample["MGMT_value"].to_csv("submission.csv")

NameError: name 'mean' is not defined

## === cell 31
sample
