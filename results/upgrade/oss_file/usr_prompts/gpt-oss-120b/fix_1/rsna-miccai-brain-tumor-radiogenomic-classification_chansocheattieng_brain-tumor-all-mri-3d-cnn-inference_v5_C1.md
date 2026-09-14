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
import os 
import glob
import re
import math
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from random import shuffle
from sklearn import model_selection as sk_model_selection

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import AUC

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
data_directory = '../input/rsna-miccai-brain-tumor-radiogenomic-classification'
 
mri_types_orig = ['FLAIR','T1w','T1wCE','T2w']
mri_types = ['FLAIR','T1w','T1wCE','T2w']

IMAGE_SIZE = 128
NUM_IMAGES_PER_TYPE = 32
NUM_IMAGES = NUM_IMAGES_PER_TYPE * len(mri_types)
BATCH_SIZE= 4

train_df = pd.read_csv("../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv")

to_exclude = [109, 123, 709]
train_df = train_df[~train_df['BraTS21ID'].isin(to_exclude)]

train_df['BraTS21ID5'] = [format(x, '05d') for x in train_df.BraTS21ID]
print(len(train_df))
train_df.head(3)

## === cell 4
sample_submission = pd.read_csv('../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv')
test=sample_submission
test['BraTS21ID5'] = [format(x, '05d') for x in test.BraTS21ID]
test.head(3)

## === cell 6
def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
        
    if rotate > 0:
        rot_choices = [0, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
        data = cv2.rotate(data, rot_choices[rotate])
        
    data = cv2.resize(data, (img_size, img_size))
    return data


def load_dicom_images_3d(scan_id, split = 'train', mri_type="FLAIR", num_imgs=NUM_IMAGES_PER_TYPE, img_size=IMAGE_SIZE, rotate=0):

    files = sorted(glob.glob(f"{data_directory}/{split}/{scan_id}/{mri_type}/*.dcm"), 
               key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    middle = len(files)//2
    num_imgs2 = num_imgs
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)
    img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2:2]]).T 
    
    if img3d.shape[-1] < 3*num_imgs//4:
        middle = len(files)//2
        num_imgs2 = num_imgs//2
        p1 = max(0, middle - num_imgs2)
        p2 = min(len(files), middle + num_imgs2)
        img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2]]).T 

    if img3d.shape[-1] < num_imgs:
        n_zero_front = np.zeros((img_size, img_size, (num_imgs - img3d.shape[-1])//2))
        n_zero_back = np.zeros((img_size, img_size, num_imgs - img3d.shape[-1] - n_zero_front.shape[-1]))
        img3d = np.concatenate((n_zero_front, img3d, n_zero_back), axis = -1)

    if np.min(img3d) < np.max(img3d):
        img3d = img3d - np.min(img3d)
        img3d = img3d / np.max(img3d)
            
    return np.expand_dims(img3d,0)

def load_dicom_images_3d_all(scan_id, split = 'train'):
    img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
    
    return img3d_all

a = load_dicom_images_3d_all("00001", 'test')
print(a.shape)
print(np.min(a), np.max(a), np.mean(a), np.median(a))

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3975171123.py in <cell line: 0>()
     51     return img3d_all
     52 
---> 53 a = load_dicom_images_3d_all("00001", 'test')
     54 print(a.shape)
     55 print(np.min(a), np.max(a), np.mean(a), np.median(a))

/tmp/ipykernel_11/3975171123.py in load_dicom_images_3d_all(scan_id, split)
     47 
     48 def load_dicom_images_3d_all(scan_id, split = 'train'):
---> 49     img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
     50 
     51     return img3d_all

/tmp/ipykernel_11/3975171123.py in <listcomp>(.0)
     47 
     48 def load_dicom_images_3d_all(scan_id, split = 'train'):
---> 49     img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
     50 
     51     return img3d_all

/tmp/ipykernel_11/3975171123.py in load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
     24     p1 = max(0, middle - num_imgs2)
     25     p2 = min(len(files), middle + num_imgs2)
---> 26     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2:2]]).T
     27 
     28     if img3d.shape[-1] < 3*num_imgs//4:

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    443     arrays = [asanyarray(arr) for arr in arrays]
    444     if not arrays:
--> 445         raise ValueError('need at least one array to stack')
    446 
    447     shapes = {arr.shape for arr in arrays}

ValueError: need at least one array to stack

## === cell 7
from tensorflow.keras.utils import Sequence

class Dataset(Sequence):
    def __init__(self,df,split = 'train', is_train=True,batch_size=BATCH_SIZE,shuffle=True):
        self.idx = df["BraTS21ID"].values
        self.paths = df["BraTS21ID5"].values
        self.y =  df["MGMT_value"].values
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split
        
    def __len__(self):
        return math.ceil(len(self.idx)/self.batch_size)
   
    def __getitem__(self,ids):
        id_path= self.paths[ids]
        batch_paths = self.paths[ids * self.batch_size:(ids + 1) * self.batch_size]
        split=self.split
        
        if self.y is not None:
            batch_y = self.y[ids * self.batch_size: (ids + 1) * self.batch_size]
            
        if split == 'train':
            list_x =  [load_dicom_images_3d_all(x, split) for x in batch_paths]
            batch_X = np.stack(list_x, axis=4)
        else:
            list_x =  load_dicom_images_3d_all(id_path, split)
            batch_X = np.stack(list_x)
            
        if self.is_train:
            return batch_X, batch_y
        else:
            return batch_X


## === cell 9
df_train, df_valid = sk_model_selection.train_test_split(
    train_df, 
    test_size=0.2, 
    random_state=12, 
    stratify=train_df["MGMT_value"],
)

## === cell 10
df_train

## === cell 11
train_dataset = Dataset(df_train, 'train')
valid_dataset = Dataset(df_valid, 'train')
test_dataset = Dataset(test, 'test', is_train=False)

## === cell 12
del train_df

## === cell 14
def plot_sample_all(images, label, j): 
    plt.figure(figsize=(35, 35))
    for i in range(NUM_IMAGES):
        plt.subplot(16,16,(i+1))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[0,:,:,i,j], cmap="gray")
    plt.show()

## === cell 15
i = 0
j = 0
images, label = train_dataset[i]
plot_sample_all(images, label, j)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3057939637.py in <cell line: 0>()
      1 i = 0
      2 j = 0
----> 3 images, label = train_dataset[i]
      4 plot_sample_all(images, label, j)

/tmp/ipykernel_11/2437154305.py in __getitem__(self, ids)
     23 
     24         if split == 'train':
---> 25             list_x =  [load_dicom_images_3d_all(x, split) for x in batch_paths]
     26             batch_X = np.stack(list_x, axis=4)
     27         else:

/tmp/ipykernel_11/2437154305.py in <listcomp>(.0)
     23 
     24         if split == 'train':
---> 25             list_x =  [load_dicom_images_3d_all(x, split) for x in batch_paths]
     26             batch_X = np.stack(list_x, axis=4)
     27         else:

/tmp/ipykernel_11/3975171123.py in load_dicom_images_3d_all(scan_id, split)
     47 
     48 def load_dicom_images_3d_all(scan_id, split = 'train'):
---> 49     img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
     50 
     51     return img3d_all

/tmp/ipykernel_11/3975171123.py in <listcomp>(.0)
     47 
     48 def load_dicom_images_3d_all(scan_id, split = 'train'):
---> 49     img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
     50 
     51     return img3d_all

/tmp/ipykernel_11/3975171123.py in load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
     24     p1 = max(0, middle - num_imgs2)
     25     p2 = min(len(files), middle + num_imgs2)
---> 26     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2:2]]).T
     27 
     28     if img3d.shape[-1] < 3*num_imgs//4:

/tmp/ipykernel_11/3975171123.py in <listcomp>(.0)
     24     p1 = max(0, middle - num_imgs2)
     25     p2 = min(len(files), middle + num_imgs2)
---> 26     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2:2]]).T
     27 
     28     if img3d.shape[-1] < 3*num_imgs//4:

/tmp/ipykernel_11/3975171123.py in load_dicom_image(path, img_size, voi_lut, rotate)
      1 def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
----> 2     dicom = pydicom.read_file(path)
      3     data = dicom.pixel_array
      4     if voi_lut:
      5         data = apply_voi_lut(dicom.pixel_array, dicom)

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 16
i = 0
j = 1
images, label = train_dataset[i]
plot_sample_all(images, label, j)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/637212519.py in <cell line: 0>()
      1 i = 0
      2 j = 1
----> 3 images, label = train_dataset[i]
      4 plot_sample_all(images, label, j)

/tmp/ipykernel_11/2437154305.py in __getitem__(self, ids)
     23 
     24         if split == 'train':
---> 25             list_x =  [load_dicom_images_3d_all(x, split) for x in batch_paths]
     26             batch_X = np.stack(list_x, axis=4)
     27         else:

/tmp/ipykernel_11/2437154305.py in <listcomp>(.0)
     23 
     24         if split == 'train':
---> 25             list_x =  [load_dicom_images_3d_all(x, split) for x in batch_paths]
     26             batch_X = np.stack(list_x, axis=4)
     27         else:

/tmp/ipykernel_11/3975171123.py in load_dicom_images_3d_all(scan_id, split)
     47 
     48 def load_dicom_images_3d_all(scan_id, split = 'train'):
---> 49     img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
     50 
     51     return img3d_all

/tmp/ipykernel_11/3975171123.py in <listcomp>(.0)
     47 
     48 def load_dicom_images_3d_all(scan_id, split = 'train'):
---> 49     img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
     50 
     51     return img3d_all

/tmp/ipykernel_11/3975171123.py in load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
     24     p1 = max(0, middle - num_imgs2)
     25     p2 = min(len(files), middle + num_imgs2)
---> 26     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2:2]]).T
     27 
     28     if img3d.shape[-1] < 3*num_imgs//4:

/tmp/ipykernel_11/3975171123.py in <listcomp>(.0)
     24     p1 = max(0, middle - num_imgs2)
     25     p2 = min(len(files), middle + num_imgs2)
---> 26     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2:2]]).T
     27 
     28     if img3d.shape[-1] < 3*num_imgs//4:

/tmp/ipykernel_11/3975171123.py in load_dicom_image(path, img_size, voi_lut, rotate)
      1 def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
----> 2     dicom = pydicom.read_file(path)
      3     data = dicom.pixel_array
      4     if voi_lut:
      5         data = apply_voi_lut(dicom.pixel_array, dicom)

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 17
i = 0
j = 2
images, label = train_dataset[i]
plot_sample_all(images, label, j)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2986706875.py in <cell line: 0>()
      1 i = 0
      2 j = 2
----> 3 images, label = train_dataset[i]
      4 plot_sample_all(images, label, j)

/tmp/ipykernel_11/2437154305.py in __getitem__(self, ids)
     23 
     24         if split == 'train':
---> 25             list_x =  [load_dicom_images_3d_all(x, split) for x in batch_paths]
     26             batch_X = np.stack(list_x, axis=4)
     27         else:

/tmp/ipykernel_11/2437154305.py in <listcomp>(.0)
     23 
     24         if split == 'train':
---> 25             list_x =  [load_dicom_images_3d_all(x, split) for x in batch_paths]
     26             batch_X = np.stack(list_x, axis=4)
     27         else:

/tmp/ipykernel_11/3975171123.py in load_dicom_images_3d_all(scan_id, split)
     47 
     48 def load_dicom_images_3d_all(scan_id, split = 'train'):
---> 49     img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
     50 
     51     return img3d_all

/tmp/ipykernel_11/3975171123.py in <listcomp>(.0)
     47 
     48 def load_dicom_images_3d_all(scan_id, split = 'train'):
---> 49     img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
     50 
     51     return img3d_all

/tmp/ipykernel_11/3975171123.py in load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
     24     p1 = max(0, middle - num_imgs2)
     25     p2 = min(len(files), middle + num_imgs2)
---> 26     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2:2]]).T
     27 
     28     if img3d.shape[-1] < 3*num_imgs//4:

/tmp/ipykernel_11/3975171123.py in <listcomp>(.0)
     24     p1 = max(0, middle - num_imgs2)
     25     p2 = min(len(files), middle + num_imgs2)
---> 26     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2:2]]).T
     27 
     28     if img3d.shape[-1] < 3*num_imgs//4:

/tmp/ipykernel_11/3975171123.py in load_dicom_image(path, img_size, voi_lut, rotate)
      1 def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
----> 2     dicom = pydicom.read_file(path)
      3     data = dicom.pixel_array
      4     if voi_lut:
      5         data = apply_voi_lut(dicom.pixel_array, dicom)

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 18
i = 0
j = 3
images, label = train_dataset[i]
plot_sample_all(images, label, j)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1193037913.py in <cell line: 0>()
      1 i = 0
      2 j = 3
----> 3 images, label = train_dataset[i]
      4 plot_sample_all(images, label, j)

/tmp/ipykernel_11/2437154305.py in __getitem__(self, ids)
     23 
     24         if split == 'train':
---> 25             list_x =  [load_dicom_images_3d_all(x, split) for x in batch_paths]
     26             batch_X = np.stack(list_x, axis=4)
     27         else:

/tmp/ipykernel_11/2437154305.py in <listcomp>(.0)
     23 
     24         if split == 'train':
---> 25             list_x =  [load_dicom_images_3d_all(x, split) for x in batch_paths]
     26             batch_X = np.stack(list_x, axis=4)
     27         else:

/tmp/ipykernel_11/3975171123.py in load_dicom_images_3d_all(scan_id, split)
     47 
     48 def load_dicom_images_3d_all(scan_id, split = 'train'):
---> 49     img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
     50 
     51     return img3d_all

/tmp/ipykernel_11/3975171123.py in <listcomp>(.0)
     47 
     48 def load_dicom_images_3d_all(scan_id, split = 'train'):
---> 49     img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
     50 
     51     return img3d_all

/tmp/ipykernel_11/3975171123.py in load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
     24     p1 = max(0, middle - num_imgs2)
     25     p2 = min(len(files), middle + num_imgs2)
---> 26     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2:2]]).T
     27 
     28     if img3d.shape[-1] < 3*num_imgs//4:

/tmp/ipykernel_11/3975171123.py in <listcomp>(.0)
     24     p1 = max(0, middle - num_imgs2)
     25     p2 = min(len(files), middle + num_imgs2)
---> 26     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2:2]]).T
     27 
     28     if img3d.shape[-1] < 3*num_imgs//4:

/tmp/ipykernel_11/3975171123.py in load_dicom_image(path, img_size, voi_lut, rotate)
      1 def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
----> 2     dicom = pydicom.read_file(path)
      3     data = dicom.pixel_array
      4     if voi_lut:
      5         data = apply_voi_lut(dicom.pixel_array, dicom)

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 21
def plot_sample_train(images, label): 
    plt.figure(figsize=(16, 16))
    idx_base = int(NUM_IMAGES_PER_TYPE/2)
    idx = [idx_base, idx_base*3, idx_base*5, idx_base*7]
    for i in range(len(idx)*BATCH_SIZE):
        plt.subplot(BATCH_SIZE,len(idx),i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        j = int(i/len(idx))
        plt.imshow(images[0,:,:,idx[i%len(idx)],j], cmap="gray")
        plt.xlabel(f'{idx[i%len(idx)]} {label[j]}')
    plt.show()

## === cell 22
i = 0
images, label = train_dataset[i]
print("Dimension of the CT scan is:", images.shape)
print("label=",label ,label.shape)
plot_sample_train(images, label)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1016579788.py in <cell line: 0>()
      1 i = 0
----> 2 images, label = train_dataset[i]
      3 print("Dimension of the CT scan is:", images.shape)
      4 print("label=",label ,label.shape)
      5 plot_sample_train(images, label)

/tmp/ipykernel_11/2437154305.py in __getitem__(self, ids)
     23 
     24         if split == 'train':
---> 25             list_x =  [load_dicom_images_3d_all(x, split) for x in batch_paths]
     26             batch_X = np.stack(list_x, axis=4)
     27         else:

/tmp/ipykernel_11/2437154305.py in <listcomp>(.0)
     23 
     24         if split == 'train':
---> 25             list_x =  [load_dicom_images_3d_all(x, split) for x in batch_paths]
     26             batch_X = np.stack(list_x, axis=4)
     27         else:

/tmp/ipykernel_11/3975171123.py in load_dicom_images_3d_all(scan_id, split)
     47 
     48 def load_dicom_images_3d_all(scan_id, split = 'train'):
---> 49     img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
     50 
     51     return img3d_all

/tmp/ipykernel_11/3975171123.py in <listcomp>(.0)
     47 
     48 def load_dicom_images_3d_all(scan_id, split = 'train'):
---> 49     img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
     50 
     51     return img3d_all

/tmp/ipykernel_11/3975171123.py in load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
     24     p1 = max(0, middle - num_imgs2)
     25     p2 = min(len(files), middle + num_imgs2)
---> 26     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2:2]]).T
     27 
     28     if img3d.shape[-1] < 3*num_imgs//4:

/tmp/ipykernel_11/3975171123.py in <listcomp>(.0)
     24     p1 = max(0, middle - num_imgs2)
     25     p2 = min(len(files), middle + num_imgs2)
---> 26     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2:2]]).T
     27 
     28     if img3d.shape[-1] < 3*num_imgs//4:

/tmp/ipykernel_11/3975171123.py in load_dicom_image(path, img_size, voi_lut, rotate)
      1 def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
----> 2     dicom = pydicom.read_file(path)
      3     data = dicom.pixel_array
      4     if voi_lut:
      5         data = apply_voi_lut(dicom.pixel_array, dicom)

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 24
def plot_sample_test(images): 
    plt.figure(figsize=(16, 16))
    idx_base = int(NUM_IMAGES_PER_TYPE/2)
    idx = [idx_base, idx_base*3, idx_base*5, idx_base*7]
    for i in range(len(idx)*BATCH_SIZE):
        plt.subplot(BATCH_SIZE,len(idx),i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        j = int(i/len(idx))
        plt.imshow(images[0,:,:,idx[i%len(idx)],j], cmap="gray")
        plt.xlabel(idx[i%len(idx)])
    plt.show()

## === cell 25
i = 0
image = test_dataset[i]
print("Dimension of the CT scan is:", images.shape)
plot_sample_test(images)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3825339864.py in <cell line: 0>()
      1 i = 0
----> 2 image = test_dataset[i]
      3 # print(test_dataset[0])
      4 print("Dimension of the CT scan is:", images.shape)
      5 plot_sample_test(images)

/tmp/ipykernel_11/2437154305.py in __getitem__(self, ids)
     26             batch_X = np.stack(list_x, axis=4)
     27         else:
---> 28             list_x =  load_dicom_images_3d_all(id_path, split)
     29             batch_X = np.stack(list_x)
     30 

/tmp/ipykernel_11/3975171123.py in load_dicom_images_3d_all(scan_id, split)
     47 
     48 def load_dicom_images_3d_all(scan_id, split = 'train'):
---> 49     img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
     50 
     51     return img3d_all

/tmp/ipykernel_11/3975171123.py in <listcomp>(.0)
     47 
     48 def load_dicom_images_3d_all(scan_id, split = 'train'):
---> 49     img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
     50 
     51     return img3d_all

/tmp/ipykernel_11/3975171123.py in load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
     24     p1 = max(0, middle - num_imgs2)
     25     p2 = min(len(files), middle + num_imgs2)
---> 26     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2:2]]).T
     27 
     28     if img3d.shape[-1] < 3*num_imgs//4:

/tmp/ipykernel_11/3975171123.py in <listcomp>(.0)
     24     p1 = max(0, middle - num_imgs2)
     25     p2 = min(len(files), middle + num_imgs2)
---> 26     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2:2]]).T
     27 
     28     if img3d.shape[-1] < 3*num_imgs//4:

/tmp/ipykernel_11/3975171123.py in load_dicom_image(path, img_size, voi_lut, rotate)
      1 def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
----> 2     dicom = pydicom.read_file(path)
      3     data = dicom.pixel_array
      4     if voi_lut:
      5         data = apply_voi_lut(dicom.pixel_array, dicom)

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 27
def get_model(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=NUM_IMAGES):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))
    
    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3D_CNN")

    return model

## === cell 28
model = get_model()
model.summary()

## === cell 35
model.load_weights('../input/braintumormodel7/Brain_Tumor_All_MRI_3D_CNN.h5')

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2336055097.py in <cell line: 0>()
----> 1 model.load_weights('../input/braintumormodel7/Brain_Tumor_All_MRI_3D_CNN.h5')

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/braintumormodel7/Brain_Tumor_All_MRI_3D_CNN.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 36
from tensorflow.keras.utils import Sequence

class Dataset(Sequence):
    def __init__(self,df,is_train=True,batch_size=1,shuffle=True):
        self.idx = df["BraTS21ID"].values
        self.paths = df["BraTS21ID5"].values
        self.y =  df["MGMT_value"].values
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __len__(self):
        return math.ceil(len(self.idx)/self.batch_size)
   
    def __getitem__(self,ids):
        id_path= self.paths[ids]
        batch_paths = self.paths[ids * self.batch_size:(ids + 1) * self.batch_size]
        
        if self.y is not None:
            batch_y = self.y[ids * self.batch_size: (ids + 1) * self.batch_size]
            
        list_x =  load_dicom_images_3d_all(id_path, 'test') #str(scan_id).zfill(5)
        batch_X = np.stack(list_x)
        if self.is_train:
            return batch_X,batch_y
        else:
            return batch_X
    
    def on_epoch_end(self):
        if self.shuffle and self.is_train:
            ids_y = list(zip(self.idx, self.y))
            shuffle(ids_y)
            self.idx, self.y = list(zip(*ids_y))

## === cell 37
test_dataset = Dataset(test,is_train=False)

## === cell 39
predictions = model.predict(test_dataset)
predictions = predictions.reshape(-1)

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1620014663.py in <cell line: 0>()
----> 1 predictions = model.predict(test_dataset)
      2 predictions = predictions.reshape(-1)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/2312955873.py in __getitem__(self, ids)
     19             batch_y = self.y[ids * self.batch_size: (ids + 1) * self.batch_size]
     20 
---> 21         list_x =  load_dicom_images_3d_all(id_path, 'test') #str(scan_id).zfill(5)
     22         #list_x =  [load_dicom_images_3d(x) for x in batch_paths]
     23         batch_X = np.stack(list_x)

/tmp/ipykernel_11/3975171123.py in load_dicom_images_3d_all(scan_id, split)
     47 
     48 def load_dicom_images_3d_all(scan_id, split = 'train'):
---> 49     img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
     50 
     51     return img3d_all

/tmp/ipykernel_11/3975171123.py in <listcomp>(.0)
     47 
     48 def load_dicom_images_3d_all(scan_id, split = 'train'):
---> 49     img3d_all = np.concatenate([load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types], axis = -1)
     50 
     51     return img3d_all

/tmp/ipykernel_11/3975171123.py in load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
     24     p1 = max(0, middle - num_imgs2)
     25     p2 = min(len(files), middle + num_imgs2)
---> 26     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2:2]]).T
     27 
     28     if img3d.shape[-1] < 3*num_imgs//4:

/tmp/ipykernel_11/3975171123.py in <listcomp>(.0)
     24     p1 = max(0, middle - num_imgs2)
     25     p2 = min(len(files), middle + num_imgs2)
---> 26     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2:2]]).T
     27 
     28     if img3d.shape[-1] < 3*num_imgs//4:

/tmp/ipykernel_11/3975171123.py in load_dicom_image(path, img_size, voi_lut, rotate)
      1 def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
----> 2     dicom = pydicom.read_file(path)
      3     data = dicom.pixel_array
      4     if voi_lut:
      5         data = apply_voi_lut(dicom.pixel_array, dicom)

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 41
submission = pd.DataFrame({'BraTS21ID':sample_submission['BraTS21ID'],'MGMT_value':predictions})

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/157673979.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({'BraTS21ID':sample_submission['BraTS21ID'],'MGMT_value':predictions})

NameError: name 'predictions' is not defined

## === cell 42
submission

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/493289180.py in <cell line: 0>()
----> 1 submission

NameError: name 'submission' is not defined

## === cell 43
submission['BraTS21ID'] = [format(x, '05d') for x in submission.BraTS21ID]

## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1004206381.py in <cell line: 0>()
----> 1 submission['BraTS21ID'] = [format(x, '05d') for x in submission.BraTS21ID]

NameError: name 'submission' is not defined

## === cell 44
submission

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/493289180.py in <cell line: 0>()
----> 1 submission

NameError: name 'submission' is not defined

## === cell 45
submission.to_csv('submission.csv',index=False)

## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3349756476.py in <cell line: 0>()
----> 1 submission.to_csv('submission.csv',index=False)

NameError: name 'submission' is not defined
