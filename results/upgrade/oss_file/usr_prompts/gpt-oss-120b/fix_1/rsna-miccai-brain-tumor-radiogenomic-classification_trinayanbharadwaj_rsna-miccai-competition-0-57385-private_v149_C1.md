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

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
import seaborn as sns
import os
import pydicom as dicom
from pympler import asizeof
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers.experimental import preprocessing
from keras.preprocessing.image import ImageDataGenerator
import cv2
from skimage.transform import resize
from random import randrange
from sklearn.metrics import roc_auc_score

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
model_T2 = keras.models.load_model("../input/trained-model-for-rsnamiccai/rsna_miccai_114_epochs_T2W_7k_imgs.h5")

model_T2_2 = keras.models.load_model("../input/trained-model-for-rsnamiccai/rsna_miccai_200_epochs_T2W_7k_imgs.h5")

model_T2_3 = keras.models.load_model("../input/trained-model-for-rsnamiccai/rsna_miccai_83_b600_T2W_7k_imgs.h5")

model_T2_4 = keras.models.load_model("../input/trained-model-for-rsnamiccai/rsna_miccai_28_b50_T2W_7k_imgs.h5")

model_T2_5 = keras.models.load_model("../input/trained-model-for-rsnamiccai/rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5")

model_T2_6 = keras.models.load_model("../input/trained-model-for-rsnamiccai/rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5")

model_T2_7 = keras.models.load_model("../input/trained-model-for-rsnamiccai/rsna_miccai_13_b600_T2w_7k_0.73auc_imgs.h5")

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1983043557.py in <cell line: 0>()
----> 1 model_T2 = keras.models.load_model("../input/trained-model-for-rsnamiccai/rsna_miccai_114_epochs_T2W_7k_imgs.h5")
      2 
      3 model_T2_2 = keras.models.load_model("../input/trained-model-for-rsnamiccai/rsna_miccai_200_epochs_T2W_7k_imgs.h5")
      4 
      5 model_T2_3 = keras.models.load_model("../input/trained-model-for-rsnamiccai/rsna_miccai_83_b600_T2W_7k_imgs.h5")

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/trained-model-for-rsnamiccai/rsna_miccai_114_epochs_T2W_7k_imgs.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 2
def load_test_T2W_images(path_test):
    array_1 = []  
    array_2 = []  
    array_3 = []  
    array_4 = []  
    array_5 = []
    array_6 = []
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test)])
    for i in range(len(path_cases)):
        count=0
        mri_type = sorted([f.path for f in os.scandir(path_cases[i])])
        img_path = sorted([f.path for f in os.scandir(mri_type[3])])
        for k in range(len(img_path)): 
            img = dicom.dcmread(img_path[k])
            if (img.pixel_array.sum()>100000):
                    resized_img = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                    img = np.array(resized_img)
                    stacked_img = np.stack((img,)*3, axis=-1)
                    stacked_img_normalize = stacked_img/np.max(stacked_img)
                    if stacked_img_normalize.sum()>2000:
                        if count==0:
                            array_1.append(stacked_img_normalize)
                            count+=1
                            continue
                        if count==1:
                            array_2.append(stacked_img_normalize)
                            count+=1
                            continue
                        if count==2:
                            array_3.append(stacked_img_normalize)
                            count+=1
                            continue
                        if count==3:
                            array_4.append(stacked_img_normalize)
                            count+=1
                            continue 
                        if count==4:
                            array_5.append(stacked_img_normalize)
                            count+=1
                            continue 
                        if count==5:
                            array_6.append(stacked_img_normalize)
                            count+=1
                            continue
                        if count==6:
                            break
                            
    array_1 = array_1/np.max(array_1)
    array_2 = array_2/np.max(array_2)
    array_3 = array_3/np.max(array_3)
    array_4 = array_4/np.max(array_4)
    array_5 = array_5/np.max(array_5)
    array_6 = array_6/np.max(array_6)
    
    print("Number of T2 images loaded are ", len(array_1), ",", len(array_2), ",", len(array_3),",", len(array_4)
          , ",", len(array_5), ",", len(array_6)
         )
    
    
    return array_1, array_2, array_3, array_4, array_5, array_6

## === cell 3
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"

## === cell 4
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3990769994.py in <cell line: 0>()
----> 1 pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)

/tmp/ipykernel_11/2349032727.py in load_test_T2W_images(path_test)
     15             img = dicom.dcmread(img_path[k])
     16             if (img.pixel_array.sum()>100000):
---> 17                     resized_img = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
     18                     img = np.array(resized_img)
     19                     stacked_img = np.stack((img,)*3, axis=-1)

NameError: name 'resize' is not defined

## === cell 5
preds_1 = model_T2.predict(pixels_1)
prediction_1 = preds_1[:,1]
preds_2 = model_T2.predict(pixels_2)
prediction_2 = preds_2[:,1]
preds_3 = model_T2.predict(pixels_3)
prediction_3 = preds_3[:,1]
preds_4 = model_T2.predict(pixels_4)
prediction_4 = preds_4[:,1]
preds_5 = model_T2.predict(pixels_5)
prediction_5 = preds_5[:,1]
preds_6 = model_T2.predict(pixels_6)
prediction_6 = preds_6[:,1]




preds_101 = model_T2_2.predict(pixels_1)
prediction_101 = preds_101[:,1]
preds_102 = model_T2_2.predict(pixels_2)
prediction_102 = preds_102[:,1]
preds_103 = model_T2_2.predict(pixels_3)
prediction_103 = preds_103[:,1]
preds_104 = model_T2_2.predict(pixels_4)
prediction_104 = preds_104[:,1]
preds_105 = model_T2_2.predict(pixels_5)
prediction_105 = preds_105[:,1]
preds_106 = model_T2_2.predict(pixels_6)
prediction_106 = preds_106[:,1]



preds_201 = model_T2_3.predict(pixels_1)
prediction_201 = preds_201[:,1]
preds_202 = model_T2_3.predict(pixels_2)
prediction_202 = preds_202[:,1]
preds_203 = model_T2_3.predict(pixels_3)
prediction_203 = preds_203[:,1]
preds_204 = model_T2_3.predict(pixels_4)
prediction_204 = preds_204[:,1]
preds_205 = model_T2_3.predict(pixels_5)
prediction_205 = preds_205[:,1]
preds_206 = model_T2_3.predict(pixels_6)
prediction_206 = preds_206[:,1]



preds_301 = model_T2_4.predict(pixels_1)
prediction_301 = preds_301[:,1]
preds_302 = model_T2_4.predict(pixels_2)
prediction_302 = preds_302[:,1]
preds_303 = model_T2_4.predict(pixels_3)
prediction_303 = preds_303[:,1]
preds_304 = model_T2_4.predict(pixels_4)
prediction_304 = preds_304[:,1]
preds_305 = model_T2_4.predict(pixels_5)
prediction_305 = preds_305[:,1]
preds_306 = model_T2_4.predict(pixels_6)
prediction_306 = preds_306[:,1]

preds_401 = model_T2_5.predict(pixels_1)
prediction_401 = preds_401[:,1]
preds_402 = model_T2_5.predict(pixels_2)
prediction_402 = preds_402[:,1]
preds_403 = model_T2_5.predict(pixels_3)
prediction_403 = preds_403[:,1]
preds_404 = model_T2_5.predict(pixels_4)
prediction_404 = preds_404[:,1]
preds_405 = model_T2_5.predict(pixels_5)
prediction_405 = preds_405[:,1]
preds_406 = model_T2_5.predict(pixels_6)
prediction_406 = preds_406[:,1]


preds_501 = model_T2_6.predict(pixels_1)
prediction_501 = preds_501[:,1]
preds_502 = model_T2_6.predict(pixels_2)
prediction_502 = preds_502[:,1]
preds_503 = model_T2_6.predict(pixels_3)
prediction_503 = preds_503[:,1]
preds_504 = model_T2_6.predict(pixels_4)
prediction_504 = preds_504[:,1]
preds_505 = model_T2_6.predict(pixels_5)
prediction_505 = preds_505[:,1]
preds_506 = model_T2_6.predict(pixels_6)
prediction_506 = preds_506[:,1]



preds_601 = model_T2_7.predict(pixels_1)
prediction_601 = preds_601[:,1]
preds_602 = model_T2_7.predict(pixels_2)
prediction_602 = preds_602[:,1]
preds_603 = model_T2_7.predict(pixels_3)
prediction_603 = preds_603[:,1]
preds_604 = model_T2_7.predict(pixels_4)
prediction_604 = preds_604[:,1]
preds_605 = model_T2_7.predict(pixels_5)
prediction_605 = preds_605[:,1]
preds_606 = model_T2_7.predict(pixels_6)
prediction_606 = preds_606[:,1]

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4137075447.py in <cell line: 0>()
----> 1 preds_1 = model_T2.predict(pixels_1)
      2 prediction_1 = preds_1[:,1]
      3 #-----------------------------------------
      4 preds_2 = model_T2.predict(pixels_2)
      5 prediction_2 = preds_2[:,1]

NameError: name 'model_T2' is not defined

## === cell 6
 def create_sub(path_test, 
                p1, p2, p3, p4, p5, p6,
                p101, p102, p103, p104, p105, p106, 
                p201, p202, p203, p204, p205, p206, 
                p301, p302, p303, p304, p305, p306,
                p401, p402, p403, p404, p405, p406,
                p501, p502, p503, p504, p505, p506,
                p601, p602, p603, p604, p605, p606,
               ):
    cases = []
    path_cases = sorted([f.path for f in os.scandir(path_test)])
    for i in range(len(path_cases)):
        
        case_number = path_cases[i][-5:]
        final_case_no = case_number.lstrip("0")
        cases.append(int(final_case_no))
        
        prediction = (
            
            p1.astype(float)
            +p2.astype(float)
            +p3.astype(float)
            +p4.astype(float)
            +p5.astype(float)
            +p6.astype(float)
            
            +p101.astype(float)
            +p102.astype(float)
            +p103.astype(float)
            +p104.astype(float)
            +p105.astype(float)
            +p106.astype(float)

            +p201.astype(float)
            +p202.astype(float)
            +p203.astype(float)
            +p204.astype(float)
            +p205.astype(float)
            +p206.astype(float)
            
            +p301.astype(float)
            +p302.astype(float)
            +p303.astype(float)
            +p304.astype(float)
            +p305.astype(float)
            +p306.astype(float)
            
            +p401.astype(float)
            +p402.astype(float)
            +p403.astype(float)
            +p404.astype(float)
            +p405.astype(float)
            +p406.astype(float)
            
            +p501.astype(float)
            +p502.astype(float)
            +p503.astype(float)
            +p504.astype(float)
            +p505.astype(float)
            +p506.astype(float)
            
            +p601.astype(float)
            +p602.astype(float)
            +p603.astype(float)
            +p604.astype(float)
            +p605.astype(float)
            +p606.astype(float)
        )/42
    
        
    df = pd.DataFrame({"BraTS21ID":cases, "MGMT_value":prediction})
    return df

## === cell 7
sub_df = create_sub(test,
                    
                    prediction_1,
                    prediction_2, 
                    prediction_3,
                    prediction_4,
                    prediction_5,
                    prediction_6,
                    
                    prediction_101,
                    prediction_102, 
                    prediction_103,
                    prediction_104,
                    prediction_105,
                    prediction_106,
                    
                    prediction_201,
                    prediction_202, 
                    prediction_203,
                    prediction_204,
                    prediction_205,
                    prediction_206,
                    
                    prediction_301,
                    prediction_302, 
                    prediction_303,
                    prediction_304,
                    prediction_305,
                    prediction_306,
                    
                    prediction_401,
                    prediction_402, 
                    prediction_403,
                    prediction_404,
                    prediction_405,
                    prediction_406,
                    
                    prediction_501,
                    prediction_502, 
                    prediction_503,
                    prediction_504,
                    prediction_505,
                    prediction_506,
                    
                    prediction_601,
                    prediction_602, 
                    prediction_603,
                    prediction_604,
                    prediction_605,
                    prediction_606,
                   )

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/826174274.py in <cell line: 0>()
      1 sub_df = create_sub(test,
      2 
----> 3                     prediction_1,
      4                     prediction_2,
      5                     prediction_3,

NameError: name 'prediction_1' is not defined

## === cell 8
sns.displot(sub_df.MGMT_value)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2187901442.py in <cell line: 0>()
----> 1 sns.displot(sub_df.MGMT_value)

NameError: name 'sub_df' is not defined

## === cell 9
sub_df.to_csv("submission.csv", index=False)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1864717487.py in <cell line: 0>()
----> 1 sub_df.to_csv("submission.csv", index=False)

NameError: name 'sub_df' is not defined
