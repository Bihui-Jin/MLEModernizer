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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import sys

if "tensorflow" not in sys.modules:
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import glob
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow import keras


SEED = 42
DEBUG = False


## === cell 1
!pip install ../input/kerasapplication/Keras_Applications-1.0.8-py3-none-any.whl
!pip install ../input/efficientnet/efficientnet-1.1.1-py3-none-any.whl


## === cell 2
try:
    from efficientnet.tfkeras import EfficientNetB0  # type: ignore
except ModuleNotFoundError:
    from tensorflow.keras.applications import EfficientNetB0

weight_path = "../input/model-v17/model_v0.38.h5"
my_model = load_model(weight_path, custom_objects={"EfficientNetB0": EfficientNetB0})


## --- ERROR in cell 2, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/756911505.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      8[0m [0;34m[0m[0m
[1;32m      9[0m [0mweight_path[0m [0;34m=[0m [0;34m"../input/model-v17/model_v0.38.h5"[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m [0mmy_model[0m [0;34m=[0m [0mload_model[0m[0;34m([0m[0mweight_path[0m[0;34m,[0m [0mcustom_objects[0m[0;34m=[0m[0;34m{[0m[0;34m"EfficientNetB0"[0m[0;34m:[0m [0mEfficientNetB0[0m[0;34m}[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
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

[0;31mFileNotFoundError[0m: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/model-v17/model_v0.38.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 3
test_images = glob.glob('../input/cassava-leaf-disease-classification/test_images/*.jpg')

df_test = pd.DataFrame(test_images, columns = ['path'])



def make_test_gen( batch_size=64):
    
    my_test_idg = ImageDataGenerator()
    test_gen= my_test_idg.flow_from_dataframe(dataframe=df_test,

                                                x_col="path",
                                                y_col=None,
                                                batch_size=batch_size,
                                                seed=42,
                                                shuffle=False,
                                                class_mode=None,
                                                target_size=(512,512))
    return test_gen
