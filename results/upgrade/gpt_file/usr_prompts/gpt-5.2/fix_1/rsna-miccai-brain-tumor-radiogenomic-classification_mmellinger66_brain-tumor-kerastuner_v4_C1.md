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

geopandas==0.14.4
keras-tuner==1.4.7
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
tqdm==4.67.1

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

0.4221

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob

import pandas as pd
import numpy as np
from pathlib import Path

import random
from tqdm.notebook import tqdm
import pydicom # Handle MRI images

import cv2  # OpenCV - https://docs.opencv.org/master/d6/d00/tutorial_py_root.html

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_dir = Path('../input/rsna-miccai-brain-tumor-radiogenomic-classification/')

mri_types = ["FLAIR", "T1w", "T2w", "T1wCE"]
excluded_images = [109, 123, 709] # Bad images

train_df = pd.read_csv(data_dir / "train_labels.csv")
test_df = pd.read_csv(data_dir / "sample_submission.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

train_df = train_df[~train_df.BraTS21ID.isin(excluded_images)]


## === cell 2
def load_dicom(path, size = 224):
    ''' 
    Reads a DICOM image, standardizes so that the pixel values are between 0 and 1, then rescales to 0 and 255
    
    Not super sure if this kind of scaling is appropriate, but everyone seems to do it. 
    '''
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return cv2.resize(data, (size, size))

def get_all_image_paths(brats21id, image_type, folder='train'): 
    '''
    Returns an arry of all the images of a particular type for a particular patient ID
    '''
    assert(image_type in mri_types)
    
    patient_path = os.path.join(
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/%s/" % folder, 
        str(brats21id).zfill(5),
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")), 
        key=lambda x: int(x[:-4].split("-")[-1]),
    )
    
    num_images = len(paths)
    
    start = int(num_images * 0.25)
    end = int(num_images * 0.75)

    interval = 3
    
    if num_images < 10: 
        interval = 1
    
    return np.array(paths[start:end:interval])

def get_all_images(brats21id, image_type, folder='train', size=225):
    return [load_dicom(path, size) for path in get_all_image_paths(brats21id, image_type, folder)]

def get_all_data_for_train(image_type, image_size=32):
    global train_df
    
    X = []
    y = []
    train_ids = []

    for i in tqdm(train_df.index):
        x = train_df.loc[i]
        images = get_all_images(int(x['BraTS21ID']), image_type, 'train', image_size)
        label = x['MGMT_value']

        X += images
        y += [label] * len(images)
        train_ids += [int(x['BraTS21ID'])] * len(images)
        assert(len(X) == len(y))
    return np.array(X), np.array(y), np.array(train_ids)

def get_all_data_for_test(image_type, image_size=32):
    global test_df
    
    X = []
    test_ids = []

    for i in tqdm(test_df.index):
        x = test_df.loc[i]
        images = get_all_images(int(x['BraTS21ID']), image_type, 'test', image_size)
        X += images
        test_ids += [int(x['BraTS21ID'])] * len(images)

    return np.array(X), np.array(test_ids)


## === cell 3
X, y, trainidt = get_all_data_for_train('T1wCE', image_size=32)
X_test, testidt = get_all_data_for_test('T1wCE', image_size=32)


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3712730407.py in <cell line: 0>()
----> 1 X, y, trainidt = get_all_data_for_train('T1wCE', image_size=32)
      2 X_test, testidt = get_all_data_for_test('T1wCE', image_size=32)

/tmp/ipykernel_11/3166407343.py in get_all_data_for_train(image_type, image_size)
     54     for i in tqdm(train_df.index):
     55         x = train_df.loc[i]
---> 56         images = get_all_images(int(x['BraTS21ID']), image_type, 'train', image_size)
     57         label = x['MGMT_value']
     58 

/tmp/ipykernel_11/3166407343.py in get_all_images(brats21id, image_type, folder, size)
     43 
     44 def get_all_images(brats21id, image_type, folder='train', size=225):
---> 45     return [load_dicom(path, size) for path in get_all_image_paths(brats21id, image_type, folder)]
     46 
     47 def get_all_data_for_train(image_type, image_size=32):

/tmp/ipykernel_11/3166407343.py in <listcomp>(.0)
     43 
     44 def get_all_images(brats21id, image_type, folder='train', size=225):
---> 45     return [load_dicom(path, size) for path in get_all_image_paths(brats21id, image_type, folder)]
     46 
     47 def get_all_data_for_train(image_type, image_size=32):

/tmp/ipykernel_11/3166407343.py in load_dicom(path, size)
      5     Not super sure if this kind of scaling is appropriate, but everyone seems to do it.
      6     '''
----> 7     dicom = pydicom.read_file(path)
      8     data = dicom.pixel_array
      9     # transform data into black and white scale / grayscale

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 4
X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = train_test_split(X, y, trainidt, test_size=0.2, random_state=42)


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2260038567.py in <cell line: 0>()
----> 1 X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = train_test_split(X, y, trainidt, test_size=0.2, random_state=42)

NameError: name 'X' is not defined

## === cell 5
X_train = tf.expand_dims(X_train, axis=-1)
X_valid = tf.expand_dims(X_valid, axis=-1)
X_train.shape


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1993180187.py in <cell line: 0>()
----> 1 X_train = tf.expand_dims(X_train, axis=-1)
      2 X_valid = tf.expand_dims(X_valid, axis=-1)
      3 X_train.shape

NameError: name 'X_train' is not defined

## === cell 6
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4158199754.py in <cell line: 0>()
----> 1 y_train = to_categorical(y_train)
      2 y_valid = to_categorical(y_valid)

NameError: name 'y_train' is not defined

## === cell 7
import keras_tuner as kt


def make_model(hp):
    inputs = keras.Input(shape=X_train.shape[1:])
    
    x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)

    
    x = keras.layers.Conv2D(filters=hp.Int('units_Conv_1_' + str(0),
                                            min_value=64,
                                            max_value=256,
                                            step=32),
                            kernel_size=(4, 4),
                            activation="relu", 
                            name="Conv_1")(x)

    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = keras.layers.Conv2D(filters=hp.Int('units_conv2_' + str(1),
                                            min_value=16,
                                            max_value=128,
                                            step=16),
                            kernel_size=(2, 2),
                            activation="relu",
                            name="Conv_2")(x)

    x = keras.layers.MaxPool2D(pool_size=(1, 1))(x)
    
    


    x = layers.Dropout(
        hp.Float('dense_dropout', min_value=0., max_value=0.7)
    )(x)
    x = keras.layers.Flatten()(x)
        
    x = layers.Dense(
        units=hp.Int('num_dense_units', min_value=16, max_value=64, step=8),
        activation='relu'
    )(x)

    outputs = keras.layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    roc_auc = tf.keras.metrics.AUC(name='roc_auc', curve='ROC')

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=[roc_auc]
    )
    model.summary()
    return model


## === cell 8
tuner = kt.tuners.BayesianOptimization(
    make_model,
    objective='val_loss',
    max_trials=5,  # Set to 5 to run quicker, but need 100+ for good results
    overwrite=True)

callbacks=[keras.callbacks.EarlyStopping(monitor='val_roc_acc', mode='max', patience=3, baseline=0.9)]

tuner.search(X_train, y_train, validation_split=0.2, callbacks=callbacks, verbose=1, epochs=20)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3312546848.py in <cell line: 0>()
----> 1 tuner = kt.tuners.BayesianOptimization(
      2     make_model,
      3     objective='val_loss',
      4     max_trials=5,  # Set to 5 to run quicker, but need 100+ for good results
      5     overwrite=True)

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/tuners/bayesian.py in __init__(self, hypermodel, objective, max_trials, num_initial_points, alpha, beta, seed, hyperparameters, tune_new_entries, allow_new_entries, max_retries_per_trial, max_consecutive_failed_trials, **kwargs)
    392             max_consecutive_failed_trials=max_consecutive_failed_trials,
    393         )
--> 394         super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
    395 

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/tuner.py in __init__(self, oracle, hypermodel, max_model_size, optimizer, loss, metrics, distribution_strategy, directory, project_name, logger, tuner_id, overwrite, executions_per_trial, **kwargs)
    120             )
    121 
--> 122         super().__init__(
    123             oracle=oracle,
    124             hypermodel=hypermodel,

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/base_tuner.py in __init__(self, oracle, hypermodel, directory, project_name, overwrite, **kwargs)
    130         else:
    131             # Only populate initial space if not reloading.
--> 132             self._populate_initial_space()
    133 
    134         # Run in distributed mode.

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/base_tuner.py in _populate_initial_space(self)
    190         self.hypermodel.declare_hyperparameters(hp)
    191         self.oracle.update_space(hp)
--> 192         self._activate_all_conditions()
    193 
    194     def search(self, *fit_args, **fit_kwargs):

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/base_tuner.py in _activate_all_conditions(self)
    147         hp = self.oracle.get_space()
    148         while True:
--> 149             self.hypermodel.build(hp)
    150             self.oracle.update_space(hp)
    151 

/tmp/ipykernel_11/3953028537.py in make_model(hp)
      3 
      4 def make_model(hp):
----> 5     inputs = keras.Input(shape=X_train.shape[1:])
      6 
      7     x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)

NameError: name 'X_train' is not defined

## === cell 9
best_hp = tuner.get_best_hyperparameters()[0]
best_model = make_model(best_hp)
history = best_model.fit(X_train, y_train, validation_split=0.2, epochs=50)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3980464880.py in <cell line: 0>()
----> 1 best_hp = tuner.get_best_hyperparameters()[0]
      2 best_model = make_model(best_hp)
      3 history = best_model.fit(X_train, y_train, validation_split=0.2, epochs=50)

NameError: name 'tuner' is not defined

## === cell 10
y_pred = best_model.predict(X_valid)

pred = np.argmax(y_pred, axis=1)

result = pd.DataFrame(trainidt_valid)
result[1] = pred

result.columns = ["BraTS21ID", "MGMT_value"]
result2 = result.groupby("BraTS21ID", as_index=False).mean()

result2 = result2.merge(train_df, on="BraTS21ID")
auc = roc_auc_score(
    result2.MGMT_value_y,
    result2.MGMT_value_x,
)
print(f"Validation AUC={auc}")


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1290815832.py in <cell line: 0>()
----> 1 y_pred = best_model.predict(X_valid)
      2 
      3 pred = np.argmax(y_pred, axis=1)
      4 
      5 result = pd.DataFrame(trainidt_valid)

NameError: name 'best_model' is not defined

## === cell 11
y_pred = best_model.predict(X_test)

pred = np.argmax(y_pred, axis=1) #

result = pd.DataFrame(testidt)
result[1] = pred
pred


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1767657842.py in <cell line: 0>()
----> 1 y_pred = best_model.predict(X_test)
      2 
      3 pred = np.argmax(y_pred, axis=1) #
      4 
      5 result = pd.DataFrame(testidt)

NameError: name 'best_model' is not defined

## === cell 12
result.columns=['BraTS21ID','MGMT_value']

result2 = result.groupby('BraTS21ID',as_index=False).mean()
result2['BraTS21ID'] = sample_submission['BraTS21ID']

result2['MGMT_value'] = result2['MGMT_value'].apply(lambda x:round(x*10)/10)
result2.to_csv('submission.csv',index=False)
result2


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3026020236.py in <cell line: 0>()
----> 1 result.columns=['BraTS21ID','MGMT_value']
      2 
      3 result2 = result.groupby('BraTS21ID',as_index=False).mean()
      4 result2['BraTS21ID'] = sample_submission['BraTS21ID']
      5 

NameError: name 'result' is not defined
