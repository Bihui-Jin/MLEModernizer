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

3.10

# 2. Installed packages

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
import glob

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

import pandas as pd
import numpy as np
from pathlib import Path

import random
from tqdm.notebook import tqdm
import pydicom  # Handle MRI images

import cv2  # OpenCV - https://docs.opencv.org/master/d6/d00/tutorial_py_root.html

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import model_selection

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomUniform


## === cell 1
data_dir = Path('../input/rsna-miccai-brain-tumor-radiogenomic-classification/')

mri_types = ["FLAIR", "T1w", "T2w", "T1wCE"]
excluded_images = [109, 123, 709] # Bad images

train_df = pd.read_csv(data_dir / "train_labels.csv")
test_df = pd.read_csv(data_dir / "sample_submission.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

train_df = train_df[~train_df.BraTS21ID.isin(excluded_images)].reset_index(drop=True)


## === cell 2
def create_folds(data, num_splits):
    data["kfold"] = -1
    kf = model_selection.KFold(n_splits=num_splits, shuffle=True, random_state=42)
    for f, (t, v) in enumerate(kf.split(X=data)):
        data.loc[v, "kfold"] = f
    return data


## === cell 3
k = 5

train_df = create_folds(train_df, k)


## === cell 4
train_df.head()


## === cell 5
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


## === cell 6
def load_dicom(path, size=224):
    """
    Reads a DICOM image, standardizes so that the pixel values are between 0 and 1, then rescales to 0 and 255

    Not super sure if this kind of scaling is appropriate, but everyone seems to do it.
    """
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return cv2.resize(data, (size, size))


def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Returns an arry of all the images of a particular type for a particular patient ID
    """
    assert image_type in mri_types

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


def get_all_images(brats21id, image_type, folder="train", size=225):
    return [
        load_dicom(path, size)
        for path in get_all_image_paths(brats21id, image_type, folder)
    ]


def get_all_data_for_train(image_type, image_size=32):
    global train_df

    X = []
    y = []
    train_ids = []

    for i in tqdm(train_df.index):
        x = train_df.loc[i]
        images = get_all_images(int(x["BraTS21ID"]), image_type, "train", image_size)
        label = x["MGMT_value"]

        X += images
        y += [label] * len(images)
        train_ids += [int(x["BraTS21ID"])] * len(images)
        assert len(X) == len(y)
    return np.array(X), np.array(y), np.array(train_ids)


def get_all_data_for_test(image_type, image_size=32):
    global test_df

    X = []
    test_ids = []

    for i in tqdm(test_df.index):
        x = test_df.loc[i]
        images = get_all_images(int(x["BraTS21ID"]), image_type, "test", image_size)
        X += images
        test_ids += [int(x["BraTS21ID"])] * len(images)

    return np.array(X), np.array(test_ids)


X, y, trainidt = get_all_data_for_train("T1wCE", image_size=32)
X_test, testidt = get_all_data_for_test("T1wCE", image_size=32)


## === cell 7
X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = train_test_split(X, y, trainidt, test_size=0.2, random_state=42)


## === cell 8
X_train = tf.expand_dims(X_train, axis=-1)
X_valid = tf.expand_dims(X_valid, axis=-1)
X_train.shape


## === cell 9
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)


## === cell 10
class SineDenseLayer(keras.layers.Layer):
    
    
    
    def __init__(self, features,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.features = features
        
        if self.is_first:
            initializer = RandomUniform(-1 / self.features, 1 / self.features)   
            self.linear = keras.layers.Dense(features, kernel_initializer=initializer)
    
        else:
            initializer = RandomUniform(-np.sqrt(6 / self.features) / self.omega_0, np.sqrt(6 / self.features) / self.omega_0)
            self.linear = keras.layers.Dense(features, kernel_initializer=initializer)
     

    def call(self, input):
        return tf.math.sin(self.omega_0 * self.linear(input))
    

class SineConvLayer(keras.layers.Layer):
    
    
    
    def __init__(self, features, kernel_size,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.features = features
        
        if self.is_first:
            initializer = RandomUniform(-1 / self.features, 1 / self.features)            
            self.conv = keras.layers.Conv2D(features, kernel_size, kernel_initializer=initializer)
            
        else:
            initializer = RandomUniform(-np.sqrt(6 / self.features) / self.omega_0, np.sqrt(6 / self.features) / self.omega_0)
            self.conv = keras.layers.Conv2D(features, kernel_size, kernel_initializer=initializer)
            

    def call(self, input):
        return tf.math.sin(self.omega_0 * self.conv(input))
    


## === cell 11
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


## === cell 12
def make_model_augmented(hp):
    input_shape = (32, 32, 1)
    classes = 10

    
    data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

    shape=X_train.shape[1:]
    print(f"shape={shape}") # shape=(32, 32, 1)
    
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)

    x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    
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


## === cell 13
import keras_tuner as kt


def make_model_siren(hp):
    inputs = keras.Input(shape=X_train.shape[1:])
    
    x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)

    x = SineConvLayer(features=hp.Int('features_conv_1', min_value=64, max_value=256, step=32),
                      kernel_size=hp.Int('kernel_conv_1', min_value=2, max_value=7, step=1),
                      is_first=True, 
                      omega_0=hp.Int('omega_0_conv_1', min_value=10, max_value=50, step=5))(x)
    
    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

    x = SineConvLayer(features=hp.Int('features_conv_2', min_value=16, max_value=128, step=16),
                      kernel_size=hp.Int('kernel_conv_2', min_value=2, max_value=7, step=1),
                      is_first=False, 
                      omega_0=hp.Int('omega_0_conv_2', min_value=10, max_value=50, step=5))(x)

    x = keras.layers.MaxPool2D(pool_size=(1, 1))(x)
    
    x = layers.Dropout(
        hp.Float('dense_dropout', min_value=0., max_value=0.7)
    )(x)
    x = keras.layers.Flatten()(x)
    x = SineDenseLayer(features=hp.Int('features_dense_1', min_value=64, max_value=256, step=32),
                      is_first=False, 
                      omega_0=hp.Int('omega_0_dense_1', min_value=10, max_value=50, step=5))(x)

    outputs = keras.layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    roc_auc = tf.keras.metrics.AUC(name='roc_auc', curve='ROC')

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=[roc_auc]
    )
    model.summary()
    return model


## === cell 14
import types

if not hasattr(layers, "experimental"):
    layers.experimental = types.SimpleNamespace()
if not hasattr(layers.experimental, "preprocessing"):
    layers.experimental.preprocessing = types.SimpleNamespace()

layers.experimental.preprocessing.RandomFlip = tf.keras.layers.RandomFlip
layers.experimental.preprocessing.RandomRotation = tf.keras.layers.RandomRotation
layers.experimental.preprocessing.Rescaling = tf.keras.layers.Rescaling

tuner = kt.tuners.BayesianOptimization(
    make_model_augmented,
    objective="val_loss",
    max_trials=5,  # Set to 5 to run quicker, but need 100+ for good results
    overwrite=True,
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_roc_auc",
        mode="max",
        patience=3,
        baseline=0.9,
    )
]

tuner.search(
    X_train, y_train, validation_split=0.2, callbacks=callbacks, verbose=1, epochs=20
)


## === cell 15
best_hp = tuner.get_best_hyperparameters()[0]
best_model = make_model(best_hp)


## === cell 16
best_model.save("best_model.keras")


## === cell 17
history = best_model.fit(X_train, y_train, validation_split=0.2, epochs=50)


## === cell 18
y_pred = best_model.predict(X_valid)

pred = np.argmax(y_pred, axis=1)

result = pd.DataFrame(trainidt_valid)
result[1] = pred

result.columns = ["BraTS21ID", "MGMT_value"]
result2 = result.groupby("BraTS21ID", as_index=False).mean()
result2


## === cell 19
result2 = result2.merge(train_df, on="BraTS21ID")
result2


## === cell 20
auc = roc_auc_score(
    result2.MGMT_value_y,
    result2.MGMT_value_x,
)
print(f"Validation AUC={auc}")


## === cell 21
y_pred = best_model.predict(X_test)

pred = np.argmax(y_pred, axis=1) #

result = pd.DataFrame(testidt)
result[1] = pred
pred


## --- ERROR in cell 21, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1767657842.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0my_pred[0m [0;34m=[0m [0mbest_model[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mX_test[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0mpred[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0margmax[0m[0;34m([0m[0my_pred[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m1[0m[0;34m)[0m [0;31m#[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m [0mresult[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mDataFrame[0m[0;34m([0m[0mtestidt[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    122[0m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 124[0;31m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    125[0m [0;34m[0m[0m
[1;32m    126[0m     [0;32mreturn[0m [0merror_handler[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: as_list() is not defined on an unknown TensorShape.

## === cell 22
result.columns=['BraTS21ID','MGMT_value']

result2 = result.groupby('BraTS21ID',as_index=False).mean()
result2['BraTS21ID'] = sample_submission['BraTS21ID']

result2
