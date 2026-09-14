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

3.10

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

-1.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pydicom
import pandas as pd
import numpy as np
import cv2
import math
data_dir = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train/"
patients = os.listdir(data_dir)
patients.sort()

## === cell 1
def average(l):
    return sum(l)/len(l)
    
def layers(l,n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

m2 = [22, 23, 24, 36, 37, 38, 39, 40, 50, 51, 52, 53, 54, 55, 56, 65, 66, 67, 68, 69, 70, 81, 82, 83, 84, 97, 98]
m3 = [19, 20, 21, 25, 26, 33, 34, 35, 41, 42, 49]
m4 = [18, 27, 28]
m5 = [17]

def adjuster(file):
    if len(file) in m5:
        n = 5
    elif len(file) in m4:
        n = 4
    elif len(file) in m3:
        n = 3
    elif len(file) in m2:
        n = 2
    else:
        n = 1
    new_file = []
    for i in range(len(file)):
        for j in range(n):
            new_file.append(file[i])
    return new_file       

def decider(length, layer_number):
    return math.ceil(length/layer_number)
   
def adjuster2(layers, layer_number):
    if len(layers) == layer_number - 1:
        layers.append(layers[-1])

## === cell 2
'''layer_number=16
c=1
Input_Values = []
for p in patients[:]:
    if p != '00109' and p != '00123' and p != '00709':
        scan=[]
        
        flair_list = os.listdir(data_dir + p + '/FLAIR/')
        flair_list.sort(key = lambda x : int(pydicom.read_file(data_dir + p + '/FLAIR/'+ x).ImagePositionPatient[2]))
        flair = [
            cv2.resize(pydicom.read_file(data_dir + p + '/FLAIR/'+ layer).pixel_array,(64,64)) 
            for layer in flair_list ]

        new_flair=[]
        flair = adjuster(flair)
        layer_size = decider(len(flair) , layer_number)
        for layer_chunk in layers(flair, layer_size):
            layer_chunk = list(map(average, zip(*layer_chunk)))
            new_flair.append(layer_chunk)
        adjuster2(new_flair, layer_number)

        T1w_list = os.listdir(data_dir + p + '/T1w/')
        T1w_list.sort(key = lambda x : int(pydicom.read_file(data_dir + p + '/T1w/'+ x).ImagePositionPatient[2]))
        T1w = [
            cv2.resize(pydicom.read_file(data_dir + p + '/T1w/'+ layer).pixel_array,(64,64)) 
            for layer in T1w_list]

        new_T1w = []
        T1w=adjuster(T1w)
        layer_size = decider(len(T1w) , layer_number)
        for layer_chunk in layers(T1w, layer_size):
            layer_chunk = list(map(average, zip(*layer_chunk)))
            new_T1w.append(layer_chunk)
        adjuster2(new_T1w, layer_number)

        T1wCE_list = os.listdir(data_dir + p + '/T1wCE/')
        T1wCE_list.sort(key = lambda x : int(pydicom.read_file(data_dir + p + '/T1wCE/'+ x).ImagePositionPatient[2]))
        T1wCE = [
            cv2.resize(pydicom.read_file(data_dir + p + '/T1wCE/'+ layer).pixel_array,(64,64)) 
            for layer in T1wCE_list]

        new_T1wCE = []
        T1wCE=adjuster(T1wCE)
        layer_size = decider(len(T1wCE) , layer_number)
        for layer_chunk in layers(T1wCE, layer_size):
            layer_chunk = list(map(average, zip(*layer_chunk)))
            new_T1wCE.append(layer_chunk)

        adjuster2(new_T1wCE, layer_number)

        T2w_list = os.listdir(data_dir + p + '/T2w/')
        T2w_list.sort(key = lambda x : int(pydicom.read_file(data_dir + p + '/T2w/'+ x).ImagePositionPatient[2]))
        T2w = [
            cv2.resize(pydicom.read_file(data_dir + p + '/T2w/'+ layer).pixel_array,(64,64)) 
            for layer in T2w_list]

        new_T2w=[]
        T2w=adjuster(T2w)
        layer_size = decider(len(T2w) , layer_number)
        for layer_chunk in layers(T2w, layer_size):
            layer_chunk = list(map(average, zip(*layer_chunk)))
            new_T2w.append(layer_chunk)

        adjuster2(new_T2w, layer_number)

        for i in range(0,layer_number):
            x=[]
            for j in range(64):
                y =[]
                for k in range(64):
                    channel=[]
                    channel.append(new_flair[i][j][k])
                    channel.append(new_T1w[i][j][k])
                    channel.append(new_T1wCE[i][j][k])
                    channel.append(new_T2w[i][j][k])
                    y.append(channel)
                x.append(y)
            scan.append(np.array(x))
        Input_Values.append(scan)
        print(str(c))
        c+=1

np.shape(Input_Values)'''

## === cell 3
df = pd.read_csv('../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv')
df = df.drop([71,81,488])
Output = df['MGMT_value']

Output_Values = []
for i in range(585):
    if i != 71 and i != 81 and i != 488:
        Output_Values.append(Output[i])
Output_Values = np.reshape(Output_Values,(582,1))
print(np.shape(Output_Values))

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.Int64HashTable.get_item()

KeyError: 526

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3315706778.py in <cell line: 0>()
      6 for i in range(585):
      7     if i != 71 and i != 81 and i != 488:
----> 8         Output_Values.append(Output[i])
      9 Output_Values = np.reshape(Output_Values,(582,1))
     10 print(np.shape(Output_Values))

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1119 
   1120         elif key_is_scalar:
-> 1121             return self._get_value(key)
   1122 
   1123         # Convert generator to list before going through hashable part

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1235 
   1236         # Similar to Index.get_value, but we do not fall back to positional
-> 1237         loc = self.index.get_loc(label)
   1238 
   1239         if is_integer(loc):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 526

## === cell 4
Input_Values = np.load('../input/rsna-competition-2/Input_Values.npy')
Input_Values = np.array(Input_Values)
Output_Values = np.array(Output_Values)
print(np.shape(Input_Values))
print(np.shape(Output_Values))

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3933216236.py in <cell line: 0>()
----> 1 Input_Values = np.load('../input/rsna-competition-2/Input_Values.npy')
      2 #np.save('./Input_Values',Input_Values)
      3 Input_Values = np.array(Input_Values)
      4 Output_Values = np.array(Output_Values)
      5 print(np.shape(Input_Values))

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '../input/rsna-competition-2/Input_Values.npy'

## === cell 5
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Activation, Dropout, Flatten, Dense, GlobalAveragePooling3D, LeakyReLU
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.regularizers import l2, l1
import matplotlib.pyplot as plt

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
model = Sequential()
model.add(Conv3D(32, (3, 3, 3), activation = "relu", input_shape=(16, 64, 64, 4), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling3D(pool_size=(2, 3, 3)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv3D(64, (3, 3, 3), activation = "relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(GlobalAveragePooling3D())
model.add(Dense(64, activation = "relu" ,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1, activation = "sigmoid" ))

model.summary()

## === cell 7
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile( loss='binary_crossentropy' , metrics=['accuracy'], optimizer=opt)

## === cell 8
history = model.fit(Input_Values,Output_Values, epochs=45, batch_size=32, shuffle = 'true')

print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/494860528.py in <cell line: 0>()
----> 1 history = model.fit(Input_Values,Output_Values, epochs=45, batch_size=32, shuffle = 'true')
      2 
      3 print(history.history.keys())
      4 
      5 plt.plot(history.history['accuracy'])

NameError: name 'Input_Values' is not defined

## === cell 10
test_dir = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/"
test = os.listdir(test_dir)
test.sort()

## === cell 11
layer_number=16
c=1
Test_Values = []
for p in test[:]:
    scan=[]

    flair_list = os.listdir(test_dir + p + '/FLAIR/')
    flair_list.sort(key = lambda x : int(pydicom.read_file(test_dir + p + '/FLAIR/'+ x).ImagePositionPatient[2]))
    flair = [
        cv2.resize(pydicom.read_file(test_dir + p + '/FLAIR/'+ layer).pixel_array,(64,64)) 
        for layer in flair_list ]

    new_flair=[]
    flair = adjuster(flair)
    layer_size = decider(len(flair) , layer_number)
    for layer_chunk in layers(flair, layer_size):
        layer_chunk = list(map(average, zip(*layer_chunk)))
        new_flair.append(layer_chunk)
    adjuster2(new_flair, layer_number)

    T1w_list = os.listdir(test_dir + p + '/T1w/')
    T1w_list.sort(key = lambda x : int(pydicom.read_file(test_dir + p + '/T1w/'+ x).ImagePositionPatient[2]))
    T1w = [
        cv2.resize(pydicom.read_file(test_dir + p + '/T1w/'+ layer).pixel_array,(64,64)) 
        for layer in T1w_list]

    new_T1w = []
    T1w=adjuster(T1w)
    layer_size = decider(len(T1w) , layer_number)
    for layer_chunk in layers(T1w, layer_size):
        layer_chunk = list(map(average, zip(*layer_chunk)))
        new_T1w.append(layer_chunk)
    adjuster2(new_T1w, layer_number)

    T1wCE_list = os.listdir(test_dir + p + '/T1wCE/')
    T1wCE_list.sort(key = lambda x : int(pydicom.read_file(test_dir + p + '/T1wCE/'+ x).ImagePositionPatient[2]))
    T1wCE = [
        cv2.resize(pydicom.read_file(test_dir + p + '/T1wCE/'+ layer).pixel_array,(64,64)) 
        for layer in T1wCE_list]

    new_T1wCE = []
    T1wCE=adjuster(T1wCE)
    layer_size = decider(len(T1wCE) , layer_number)
    for layer_chunk in layers(T1wCE, layer_size):
        layer_chunk = list(map(average, zip(*layer_chunk)))
        new_T1wCE.append(layer_chunk)

    adjuster2(new_T1wCE, layer_number)

    T2w_list = os.listdir(test_dir + p + '/T2w/')
    T2w_list.sort(key = lambda x : int(pydicom.read_file(test_dir + p + '/T2w/'+ x).ImagePositionPatient[2]))
    T2w = [
        cv2.resize(pydicom.read_file(test_dir + p + '/T2w/'+ layer).pixel_array,(64,64)) 
        for layer in T2w_list]

    new_T2w=[]
    T2w=adjuster(T2w)
    layer_size = decider(len(T2w) , layer_number)
    for layer_chunk in layers(T2w, layer_size):
        layer_chunk = list(map(average, zip(*layer_chunk)))
        new_T2w.append(layer_chunk)

    adjuster2(new_T2w, layer_number)

    for i in range(0,layer_number):
        x=[]
        for j in range(64):
            y =[]
            for k in range(64):
                channel=[]
                channel.append(new_flair[i][j][k])
                channel.append(new_T1w[i][j][k])
                channel.append(new_T1wCE[i][j][k])
                channel.append(new_T2w[i][j][k])
                y.append(channel)
            x.append(y)
        scan.append(np.array(x))
    Test_Values.append(scan)
    print(str(c))
    c+=1

Test_Values = np.array(Test_Values)
np.shape(Test_Values)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/371957648.py in <cell line: 0>()
      6 
      7     flair_list = os.listdir(test_dir + p + '/FLAIR/')
----> 8     flair_list.sort(key = lambda x : int(pydicom.read_file(test_dir + p + '/FLAIR/'+ x).ImagePositionPatient[2]))
      9     flair = [
     10         cv2.resize(pydicom.read_file(test_dir + p + '/FLAIR/'+ layer).pixel_array,(64,64))

/tmp/ipykernel_11/371957648.py in <lambda>(x)
      6 
      7     flair_list = os.listdir(test_dir + p + '/FLAIR/')
----> 8     flair_list.sort(key = lambda x : int(pydicom.read_file(test_dir + p + '/FLAIR/'+ x).ImagePositionPatient[2]))
      9     flair = [
     10         cv2.resize(pydicom.read_file(test_dir + p + '/FLAIR/'+ layer).pixel_array,(64,64))

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 13
Results = model.predict(Test_Values)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3436205974.py in <cell line: 0>()
----> 1 Results = model.predict(Test_Values)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/array_data_adapter.py in __init__(self, x, y, sample_weight, batch_size, steps, shuffle, class_weight)
     77 
     78         data_adapter_utils.check_data_cardinality(inputs)
---> 79         num_samples = set(i.shape[0] for i in tree.flatten(inputs)).pop()
     80         self._num_samples = num_samples
     81         self._inputs = inputs

KeyError: 'pop from an empty set'

## === cell 14
sub = pd.DataFrame(columns=['BraTS21ID','MGMT_value'])
sub.loc[:,'BraTS21ID'] = test
sub.loc[:,'MGMT_value'] = Results
sub.set_index('BraTS21ID', inplace = True)
sub.to_csv('./submission.csv')

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2239331589.py in <cell line: 0>()
      1 sub = pd.DataFrame(columns=['BraTS21ID','MGMT_value'])
      2 sub.loc[:,'BraTS21ID'] = test
----> 3 sub.loc[:,'MGMT_value'] = Results
      4 sub.set_index('BraTS21ID', inplace = True)
      5 sub.to_csv('./submission.csv')

NameError: name 'Results' is not defined
