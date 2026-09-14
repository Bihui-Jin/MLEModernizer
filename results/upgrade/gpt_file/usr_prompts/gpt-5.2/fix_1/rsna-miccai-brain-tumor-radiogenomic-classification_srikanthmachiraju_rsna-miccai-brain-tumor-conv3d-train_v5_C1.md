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

-1.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
!pip install --no-index --find-links ../input/nobrainerwheels/ -r ../input/nobrainerwheels/requirements.txt

## === cell 2
import nobrainer

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2173042771.py in <cell line: 0>()
----> 1 import nobrainer

ModuleNotFoundError: No module named 'nobrainer'

## === cell 3
import os
import re 
import glob
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
from pathlib import Path
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')
import random as rn
import matplotlib.pyplot as plt
import imageio
import pydicom
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import backend as K, optimizers, regularizers
from random import shuffle
import wandb
from wandb.keras import WandbCallback

rn.seed(30)
np.random.seed(30)
tf.compat.v1.random.set_random_seed(30)
print('W&B version: ', wandb.__version__)
from pydicom.pixel_data_handlers.util import apply_voi_lut

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
config = {
  'images_source_path' : '../input/rsna-miccai-brain-tumor-radiogenomic-classification/train',
  'test_images_source_path' : '../input/rsna-miccai-brain-tumor-radiogenomic-classification/test',
  'csv_path': '../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv',
  'data_path': '../input/rsna-miccai-brain-tumor-radiogenomic-classification',
  'output_path': './crnn/',
  'nfolds': 3,
  'global_seed': 42,
  'batch_size': 4,
  'frames_per_seq': 12,
  'img_size': 224,
  'learning_rate': 0.0001,
  'num_epochs': 10,
  'channels': 3,
  'scale' : 0.75
}

mri_types = ['T2w'] 


## === cell 5
df_data = pd.read_csv(config['csv_path'])
df_data["folder_name"] = [format(x, "05d") for x in df_data["BraTS21ID"]]
df_data["folder_path"] = [os.path.join(config['images_source_path'], x) for x in df_data["folder_name"]]
skf = StratifiedKFold(n_splits=config['nfolds'], shuffle=True, random_state=config['global_seed'])
for index, (train_index, val_index) in enumerate(skf.split(X=df_data.index, y=df_data.MGMT_value)):
    df_data.loc[val_index, 'fold'] = index


## === cell 6
class Dataset(tf.keras.utils.Sequence):
    def __init__(self,df,is_train=True,batch_size=config['batch_size'],shuffle=True):
        self.idx = df["BraTS21ID"].values
        self.paths = df["folder_path"].values
        self.y =  df["MGMT_value"].values
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.df = df
        
    def __len__(self):
        return math.ceil(len(self.idx)/self.batch_size)
   
    def __getitem__(self,ids):
        
        id_path= self.paths[ids]
        
        batch_paths = self.paths[ids * self.batch_size:(ids + 1) * self.batch_size]
        
        if self.y is not None:
            batch_y = self.y[ids * self.batch_size: (ids + 1) * self.batch_size]
        
        if self.is_train:
            list_x =  [self.load_dicom_images_3d(x,split="train") for x in batch_paths]
            batch_X = np.stack(list_x, axis=4)
            return batch_X,batch_y
        else:
            list_x =  self.load_dicom_images_3d(id_path,split="test")
            batch_X = np.stack(list_x)
            return batch_X
    
    def load_dicom_images_3d(self, scan_id, num_imgs=config['frames_per_seq'], img_size=config['img_size'], 
                             mri_type=mri_types[0], split="train", rotate=0):

        target_file_paths = self.get_img_path_3d(scan_id, mri_type)
        
        img3d = np.stack([self.read_mri(f) for f in target_file_paths]).T 
        
        if img3d.shape[-1] < num_imgs:
            n_zero = np.zeros((img_size, img_size, num_imgs - img3d.shape[-1]))
            img3d = np.concatenate((img3d,  n_zero), axis = -1)
        
        if np.min(img3d) < np.max(img3d):
            img3d = img3d - np.min(img3d)
            img3d = img3d / np.max(img3d)
            
        return np.expand_dims(img3d,0)
     
    def read_mri(self, path, voi_lut = True, fix_monochrome = True):
        dicom = pydicom.read_file(path)
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        data = cv2.resize(data, (config['img_size'], config['img_size']))
        return data

    
    def get_img_path_3d(self, scan_id, mri_type):
        modality_path = os.path.join(scan_id, mri_type)
        total_img_num = len(glob.glob(f"{modality_path}/*.dcm"))
        files = sorted(glob.glob(f"{modality_path}/*.dcm"), 
                       key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        mid_num = total_img_num // 2
        num_3d2 = config['frames_per_seq'] // 2
        start_idx = max(0, mid_num - num_3d2)
        end_idx = min(len(files), mid_num + num_3d2)
        target_file_paths = files[start_idx:end_idx]
        return target_file_paths

    def on_epoch_end(self):
        if self.shuffle and self.is_train:
            ids_y = list(zip(self.idx, self.y))
            shuffle(ids_y)
            self.idx, self.y = list(zip(*ids_y))    
    

train_dataset = Dataset(df_data,batch_size=config['batch_size'])
valid_dataset = Dataset(df_data,batch_size=config['batch_size'])

for i in range(1):
    images, label = train_dataset[i]
    print("Dimension of the CT scan is:", images.shape)
    print("label=",label)
    plt.imshow(images[0,:,:,3,0], cmap="gray")
    plt.show()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1355186389.py in <cell line: 0>()
     86 
     87 for i in range(1):
---> 88     images, label = train_dataset[i]
     89     print("Dimension of the CT scan is:", images.shape)
     90     print("label=",label)

/tmp/ipykernel_11/1355186389.py in __getitem__(self, ids)
     22 
     23         if self.is_train:
---> 24             list_x =  [self.load_dicom_images_3d(x,split="train") for x in batch_paths]
     25             batch_X = np.stack(list_x, axis=4)
     26             return batch_X,batch_y

/tmp/ipykernel_11/1355186389.py in <listcomp>(.0)
     22 
     23         if self.is_train:
---> 24             list_x =  [self.load_dicom_images_3d(x,split="train") for x in batch_paths]
     25             batch_X = np.stack(list_x, axis=4)
     26             return batch_X,batch_y

/tmp/ipykernel_11/1355186389.py in load_dicom_images_3d(self, scan_id, num_imgs, img_size, mri_type, split, rotate)
     35         target_file_paths = self.get_img_path_3d(scan_id, mri_type)
     36 
---> 37         img3d = np.stack([self.read_mri(f) for f in target_file_paths]).T
     38 
     39         if img3d.shape[-1] < num_imgs:

/tmp/ipykernel_11/1355186389.py in <listcomp>(.0)
     35         target_file_paths = self.get_img_path_3d(scan_id, mri_type)
     36 
---> 37         img3d = np.stack([self.read_mri(f) for f in target_file_paths]).T
     38 
     39         if img3d.shape[-1] < num_imgs:

/tmp/ipykernel_11/1355186389.py in read_mri(self, path, voi_lut, fix_monochrome)
     49     def read_mri(self, path, voi_lut = True, fix_monochrome = True):
     50         # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
---> 51         dicom = pydicom.read_file(path)
     52         if voi_lut:
     53             data = apply_voi_lut(dicom.pixel_array, dicom)

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 7
def get_3d_model(width=config['img_size'], height=config['img_size'], depth=config['frames_per_seq']):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))
    
    x = Conv3D(filters=32, kernel_size=3, padding='same', activation="relu")(inputs)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)
    
    x = Conv3D(filters=32, kernel_size=3, padding='same', activation="relu")(inputs)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)
    
    x = Conv3D(filters=64, kernel_size=3, padding='same', activation="relu")(inputs)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.01)(x)
    
    x = Conv3D(filters=128, kernel_size=3, padding='same', activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.02)(x)

    x = Conv3D(filters=256, kernel_size=3, padding='same', activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.03)(x)


    x = GlobalAveragePooling3D()(x)
    x = Dense(units=1024, activation="relu")(x)
    x = Dropout(0.08)(x)

    outputs = Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3dcnn")

    return model

model = get_3d_model()
model.summary()

## === cell 9
from tensorflow.keras.metrics import AUC
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=[AUC(name='auc'),"acc"],
)

model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=config['num_epochs'],
    shuffle=True,
    verbose=1
)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4078309604.py in <cell line: 0>()
     11 
     12 # Train the model, doing validation at the end of each epoch
---> 13 model.fit(
     14     train_dataset,
     15     validation_data=valid_dataset,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/1355186389.py in __getitem__(self, ids)
     22 
     23         if self.is_train:
---> 24             list_x =  [self.load_dicom_images_3d(x,split="train") for x in batch_paths]
     25             batch_X = np.stack(list_x, axis=4)
     26             return batch_X,batch_y

/tmp/ipykernel_11/1355186389.py in <listcomp>(.0)
     22 
     23         if self.is_train:
---> 24             list_x =  [self.load_dicom_images_3d(x,split="train") for x in batch_paths]
     25             batch_X = np.stack(list_x, axis=4)
     26             return batch_X,batch_y

/tmp/ipykernel_11/1355186389.py in load_dicom_images_3d(self, scan_id, num_imgs, img_size, mri_type, split, rotate)
     35         target_file_paths = self.get_img_path_3d(scan_id, mri_type)
     36 
---> 37         img3d = np.stack([self.read_mri(f) for f in target_file_paths]).T
     38 
     39         if img3d.shape[-1] < num_imgs:

/tmp/ipykernel_11/1355186389.py in <listcomp>(.0)
     35         target_file_paths = self.get_img_path_3d(scan_id, mri_type)
     36 
---> 37         img3d = np.stack([self.read_mri(f) for f in target_file_paths]).T
     38 
     39         if img3d.shape[-1] < num_imgs:

/tmp/ipykernel_11/1355186389.py in read_mri(self, path, voi_lut, fix_monochrome)
     49     def read_mri(self, path, voi_lut = True, fix_monochrome = True):
     50         # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
---> 51         dicom = pydicom.read_file(path)
     52         if voi_lut:
     53             data = apply_voi_lut(dicom.pixel_array, dicom)

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 10
sample_submission_path = os.path.join(config['data_path'], 'sample_submission.csv')
sample_df = pd.read_csv(sample_submission_path); 
test_df = sample_df.copy(); 
test_df["folder_name"] = [format(x, "05d") for x in test_df.BraTS21ID]
test_df["folder_path"] = [os.path.join(config['data_path'], 'test', x) for x in test_df["folder_name"]]
test_dataset = Dataset(test_df,is_train=False,batch_size=1)
preds = model.predict(test_dataset)
preds = preds.reshape(-1)
submission = pd.DataFrame({'BraTS21ID':sample_df['BraTS21ID'],'MGMT_value':preds})

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1046513357.py in <cell line: 0>()
      5 test_df["folder_path"] = [os.path.join(config['data_path'], 'test', x) for x in test_df["folder_name"]]
      6 test_dataset = Dataset(test_df,is_train=False,batch_size=1)
----> 7 preds = model.predict(test_dataset)
      8 preds = preds.reshape(-1)
      9 submission = pd.DataFrame({'BraTS21ID':sample_df['BraTS21ID'],'MGMT_value':preds})

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/1355186389.py in __getitem__(self, ids)
     26             return batch_X,batch_y
     27         else:
---> 28             list_x =  self.load_dicom_images_3d(id_path,split="test")
     29             batch_X = np.stack(list_x)
     30             return batch_X

/tmp/ipykernel_11/1355186389.py in load_dicom_images_3d(self, scan_id, num_imgs, img_size, mri_type, split, rotate)
     35         target_file_paths = self.get_img_path_3d(scan_id, mri_type)
     36 
---> 37         img3d = np.stack([self.read_mri(f) for f in target_file_paths]).T
     38 
     39         if img3d.shape[-1] < num_imgs:

/tmp/ipykernel_11/1355186389.py in <listcomp>(.0)
     35         target_file_paths = self.get_img_path_3d(scan_id, mri_type)
     36 
---> 37         img3d = np.stack([self.read_mri(f) for f in target_file_paths]).T
     38 
     39         if img3d.shape[-1] < num_imgs:

/tmp/ipykernel_11/1355186389.py in read_mri(self, path, voi_lut, fix_monochrome)
     49     def read_mri(self, path, voi_lut = True, fix_monochrome = True):
     50         # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
---> 51         dicom = pydicom.read_file(path)
     52         if voi_lut:
     53             data = apply_voi_lut(dicom.pixel_array, dicom)

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 11
submission.head()

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined

## === cell 12
submission.to_csv('submission.csv',index=False)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3349756476.py in <cell line: 0>()
----> 1 submission.to_csv('submission.csv',index=False)

NameError: name 'submission' is not defined
