# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

geopandas==0.14.4
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
pillow==11.3.0
protobuf==6.33.0
pydicom==3.0.1
Pympler==1.1
scikit-image==0.25.2
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
tf_keras==2.18.0

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = None


def _major(ver):
    try:
        return int(str(ver).split(".", 1)[0])
    except Exception:
        return None


if _pb_ver is None or (_major(_pb_ver) is not None and _major(_pb_ver) >= 5):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
    for _m in list(sys.modules):
        if _m.startswith("google.protobuf"):
            sys.modules.pop(_m, None)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pydicom as dicom
from pympler import asizeof
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers

try:
    from keras import (
        layers as preprocessing,
    )  # preprocessing layers live directly under keras.layers in Keras 3
except Exception:
    from tensorflow.keras import (
        layers as preprocessing,
    )  # fallback for TF/Keras environments

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception:
    from keras.preprocessing.image import ImageDataGenerator

import cv2
from skimage.transform import resize
from random import randrange


## === cell 1
model_1 = keras.models.load_model("../input/trained-model-for-rsnamiccai/model_rsna_miccai_100epochs.h5")

model_2 = keras.models.load_model("../input/trained-model-for-rsnamiccai/rsna_miccai_100_epochs_T1W.h5")

model_3 = keras.models.load_model("../input/trained-model-for-rsnamiccai/rsna_miccai_100_epochs_T1wCE.h5")

model_4 = keras.models.load_model("../input/trained-model-for-rsnamiccai/rsna_miccai_100_epochs_T2W.h5")


## --- ERROR in cell 1, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3804670845.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mmodel_1[0m [0;34m=[0m [0mkeras[0m[0;34m.[0m[0mmodels[0m[0;34m.[0m[0mload_model[0m[0;34m([0m[0;34m"../input/trained-model-for-rsnamiccai/model_rsna_miccai_100epochs.h5"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0mmodel_2[0m [0;34m=[0m [0mkeras[0m[0;34m.[0m[0mmodels[0m[0;34m.[0m[0mload_model[0m[0;34m([0m[0;34m"../input/trained-model-for-rsnamiccai/rsna_miccai_100_epochs_T1W.h5"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m [0mmodel_3[0m [0;34m=[0m [0mkeras[0m[0;34m.[0m[0mmodels[0m[0;34m.[0m[0mload_model[0m[0;34m([0m[0;34m"../input/trained-model-for-rsnamiccai/rsna_miccai_100_epochs_T1wCE.h5"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py[0m in [0;36mload_model[0;34m(filepath, custom_objects, compile, safe_mode)[0m
[1;32m    194[0m         )
[1;32m    195[0m     [0;32mif[0m [0mstr[0m[0;34m([0m[0mfilepath[0m[0;34m)[0m[0;34m.[0m[0mendswith[0m[0;34m([0m[0;34m([0m[0;34m".h5"[0m[0;34m,[0m [0;34m".hdf5"[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 196[0;31m         return legacy_h5_format.load_model_from_hdf5(
[0m[1;32m    197[0m             [0mfilepath[0m[0;34m,[0m [0mcustom_objects[0m[0;34m=[0m[0mcustom_objects[0m[0;34m,[0m [0mcompile[0m[0;34m=[0m[0mcompile[0m[0;34m[0m[0;34m[0m[0m
[1;32m    198[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py[0m in [0;36mload_model_from_hdf5[0;34m(filepath, custom_objects, compile)[0m
[1;32m    114[0m     [0mopened_new_file[0m [0;34m=[0m [0;32mnot[0m [0misinstance[0m[0;34m([0m[0mfilepath[0m[0;34m,[0m [0mh5py[0m[0;34m.[0m[0mFile[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    115[0m     [0;32mif[0m [0mopened_new_file[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 116[0;31m         [0mf[0m [0;34m=[0m [0mh5py[0m[0;34m.[0m[0mFile[0m[0;34m([0m[0mfilepath[0m[0;34m,[0m [0mmode[0m[0;34m=[0m[0;34m"r"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    117[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    118[0m         [0mf[0m [0;34m=[0m [0mfilepath[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py[0m in [0;36m__init__[0;34m(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)[0m
[1;32m    562[0m                                  [0mfs_persist[0m[0;34m=[0m[0mfs_persist[0m[0;34m,[0m [0mfs_threshold[0m[0;34m=[0m[0mfs_threshold[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    563[0m                                  fs_page_size=fs_page_size)
[0;32m--> 564[0;31m                 [0mfid[0m [0;34m=[0m [0mmake_fid[0m[0;34m([0m[0mname[0m[0;34m,[0m [0mmode[0m[0;34m,[0m [0muserblock_size[0m[0;34m,[0m [0mfapl[0m[0;34m,[0m [0mfcpl[0m[0;34m,[0m [0mswmr[0m[0;34m=[0m[0mswmr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    565[0m [0;34m[0m[0m
[1;32m    566[0m             [0;32mif[0m [0misinstance[0m[0;34m([0m[0mlibver[0m[0;34m,[0m [0mtuple[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py[0m in [0;36mmake_fid[0;34m(name, mode, userblock_size, fapl, fcpl, swmr)[0m
[1;32m    236[0m         [0;32mif[0m [0mswmr[0m [0;32mand[0m [0mswmr_support[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    237[0m             [0mflags[0m [0;34m|=[0m [0mh5f[0m[0;34m.[0m[0mACC_SWMR_READ[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 238[0;31m         [0mfid[0m [0;34m=[0m [0mh5f[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mname[0m[0;34m,[0m [0mflags[0m[0;34m,[0m [0mfapl[0m[0;34m=[0m[0mfapl[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    239[0m     [0;32melif[0m [0mmode[0m [0;34m==[0m [0;34m'r+'[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    240[0m         [0mfid[0m [0;34m=[0m [0mh5f[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mname[0m[0;34m,[0m [0mh5f[0m[0;34m.[0m[0mACC_RDWR[0m[0;34m,[0m [0mfapl[0m[0;34m=[0m[0mfapl[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32mh5py/_objects.pyx[0m in [0;36mh5py._objects.with_phil.wrapper[0;34m()[0m

[0;32mh5py/_objects.pyx[0m in [0;36mh5py._objects.with_phil.wrapper[0;34m()[0m

[0;32mh5py/h5f.pyx[0m in [0;36mh5py.h5f.open[0;34m()[0m

[0;31mFileNotFoundError[0m: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/trained-model-for-rsnamiccai/model_rsna_miccai_100epochs.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 2
def load_test_flair_images(path_test):
    array = []  
    IMG_PX_SIZE = 299
    path_cases = sorted([f.path for f in os.scandir(path_test)])
    for i in range(len(path_cases)):
        mri_type = sorted([f.path for f in os.scandir(path_cases[i])])
        img_path = sorted([f.path for f in os.scandir(mri_type[0])])
        for k in range(len(img_path)): 
            img = dicom.dcmread(img_path[k])
            if (img.pixel_array.sum()>100000):
                    resized_img = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                    img = np.array(resized_img)
                    stacked_img = np.stack((img,)*3, axis=-1)
                    stacked_img_normalize = stacked_img/np.max(stacked_img)
                    if stacked_img_normalize.sum()>5000:
                        array.append(stacked_img_normalize)
                        break
    array = array/np.max(array)
    print("Number of flair images loaded are ", len(array))
    return array



def load_test_T1W_images(path_test):
    array = []  
    IMG_PX_SIZE = 299
    path_cases = sorted([f.path for f in os.scandir(path_test)])
    for i in range(len(path_cases)):
        mri_type = sorted([f.path for f in os.scandir(path_cases[i])])
        img_path = sorted([f.path for f in os.scandir(mri_type[1])])
        for k in range(len(img_path)): 
            img = dicom.dcmread(img_path[k])
            if (img.pixel_array.sum()>100000):
                    resized_img = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                    img = np.array(resized_img)
                    stacked_img = np.stack((img,)*3, axis=-1)
                    stacked_img_normalize = stacked_img/np.max(stacked_img)
                    if stacked_img_normalize.sum()>5000:
                        array.append(stacked_img_normalize)
                        break
    array = array/np.max(array)
    print("Number of T1W images loaded are ", len(array))
    return array


def load_test_T1wCE_images(path_test):
    array = []  
    IMG_PX_SIZE = 299
    path_cases = sorted([f.path for f in os.scandir(path_test)])
    for i in range(len(path_cases)):
        mri_type = sorted([f.path for f in os.scandir(path_cases[i])])
        img_path = sorted([f.path for f in os.scandir(mri_type[2])])
        for k in range(len(img_path)): 
            img = dicom.dcmread(img_path[k])
            if (img.pixel_array.sum()>100000):
                    resized_img = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                    img = np.array(resized_img)
                    stacked_img = np.stack((img,)*3, axis=-1)
                    stacked_img_normalize = stacked_img/np.max(stacked_img)
                    if stacked_img_normalize.sum()>5000:
                        array.append(stacked_img_normalize)
                        break
    array = array/np.max(array)
    print("Number of T1wCE images loaded are ", len(array))
    return array


def load_test_T2W_images(path_test):
    array = []  
    IMG_PX_SIZE = 299
    path_cases = sorted([f.path for f in os.scandir(path_test)])
    for i in range(len(path_cases)):
        mri_type = sorted([f.path for f in os.scandir(path_cases[i])])
        img_path = sorted([f.path for f in os.scandir(mri_type[3])])
        for k in range(len(img_path)): 
            img = dicom.dcmread(img_path[k])
            if (img.pixel_array.sum()>100000):
                    resized_img = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                    img = np.array(resized_img)
                    stacked_img = np.stack((img,)*3, axis=-1)
                    stacked_img_normalize = stacked_img/np.max(stacked_img)
                    if stacked_img_normalize.sum()>5000:
                        array.append(stacked_img_normalize)
                        break
    array = array/np.max(array)
    print("Number of T2W images loaded are ", len(array))
    return array
