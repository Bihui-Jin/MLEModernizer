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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

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

# 5. Target score

0.1128739800543971

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!ls /kaggle/input/library4/ImageDataAugmentor-master
import sys
sys.path.append('/kaggle/input/library4/ImageDataAugmentor-master') #/ImageDataAugmentor')

from ImageDataAugmentor.image_data_augmentor import *

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1307516327.py in <cell line: 0>()
      3 sys.path.append('/kaggle/input/library4/ImageDataAugmentor-master') #/ImageDataAugmentor')
      4 
----> 5 from ImageDataAugmentor.image_data_augmentor import *

ModuleNotFoundError: No module named 'ImageDataAugmentor'

## === cell 1
SUBMISSION_MODE = 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
import os
import cv2
import sys
from pylab import rcParams
from PIL import Image
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Input, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import InceptionV3, Xception
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from sklearn.model_selection import StratifiedShuffleSplit

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy) #shortens training time by 2x

df_train = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
df_train.head()

df_train["label"] = df_train["label"].astype(str)


batch_size=32
image_size=300

input_shape = (image_size, image_size, 3)
target_size = (image_size, image_size)

img_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomCrop(image_size, image_size),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.25),
        tf.keras.layers.experimental.preprocessing.RandomZoom((-0.25, 0.25), (-0.25, 0.25)),
    ])

path = "../input/cassava-leaf-disease-classification/train_images/"
import tensorflow.keras.utils
from ImageDataAugmentor.image_data_augmentor import *
import albumentations as A

train_augmentations = A.Compose([
            A.RandomCrop(image_size, image_size, p=1),
            A.CoarseDropout(p=0.5),
            A.Cutout(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(p=0.5, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
            A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2)),
            A.ToFloat()
            ], p=1)

val_augmentations = A.Compose([
                A.CenterCrop(image_size, image_size, p=1),
                A.ToFloat()
                ], p=1)

def TFDataGenerator(train_set, val_set):
    train_generator = ImageDataAugmentor(augment=train_augmentations)
    val_generator = ImageDataAugmentor(augment=val_augmentations)
    
    train_datagen = train_generator.flow_from_dataframe(
                  dataframe = train_set,
                  directory='../input/cassava-leaf-disease-classification/train_images',
                  x_col='image_id',
                  y_col='label',
                  target_size=target_size,
                  batch_size=batch_size,
                  shuffle=True,
                  class_mode='categorical',
                  seed=2020)

    val_datagen = val_generator.flow_from_dataframe(
                dataframe = val_set,
                directory='../input/cassava-leaf-disease-classification/train_images',
                x_col='image_id',
                y_col='label',
                target_size=target_size,
                batch_size=batch_size,
                shuffle=False,
                class_mode='categorical',
                seed=2020)
    
    return train_datagen, val_datagen

train_set = df_train.iloc[:int(len(df_train)*0.8)]
val_set = df_train.iloc[-int(len(df_train)*0.2):]
train_datagen, val_datagen = TFDataGenerator(train_set, val_set)


def create_Inception():
    base_model = InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape)

    inputs = Input(shape=input_shape)

    model = base_model(inputs)
    pooling = GlobalAveragePooling2D()(model)
    dropout = Dropout(0.2)(pooling)

    outputs = Dense(5, activation="softmax", name="dense", dtype='float32')(dropout)

    inception = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2, from_logits=True)

    inception.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return inception

def create_Xception():
    base_model = Xception(include_top=False, weights="imagenet", input_shape=input_shape)

    inputs = Input(shape=input_shape)

    model = base_model(inputs)
    pooling = GlobalAveragePooling2D()(model)
    dropout = Dropout(0.2)(pooling)

    outputs = Dense(5, activation="softmax", name="dense", dtype='float32')(dropout)

    xception = Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2, from_logits=True)

    xception.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return xception

if SUBMISSION_MODE == 0:

    fold_number = 0
    n_splits = 3
    epochs = 8

    tf.keras.backend.clear_session()
    KFoldSplit = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=2020)
    for train_index, val_index in KFoldSplit.split(df_train["image_id"], df_train["label"]):
        train_set = df_train.loc[train_index]
        val_set = df_train.loc[val_index]
        train_datagen, val_datagen = TFDataGenerator(train_set, val_set)
        model = create_Inception()
        print("Training fold no.: " + str(fold_number+1))

        model_name = "inception "
        fold_name = "fold.h5"
        filepath = model_name + str(fold_number+1) + fold_name
        callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.2),
                     EarlyStopping(monitor='val_loss', patience=3),
                     ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True)]

        history = model.fit(train_datagen, epochs=epochs, validation_data=val_datagen, callbacks=callbacks)
        fold_number += 1
        if fold_number == n_splits:
            print("Training finished!")
if SUBMISSION_MODE == 1:
    model = load_model("../input/models/xception 1fold.h5")

    SampleSubmit = pd.read_csv(os.path.join('../input/cassava-leaf-disease-classification', "sample_submission.csv"))
    preds = []
    results = []

    for image_id in SampleSubmit.image_id:
        image = Image.open(os.path.join('../input/cassava-leaf-disease-classification', "test_images", image_id))
        image = image.resize((image_size, image_size))
        image = np.expand_dims(image, axis = 0)
        results.append(np.argmax(model.predict(image), axis=1).item())

    SampleSubmit['label'] = results
    SampleSubmit.to_csv(os.path.join('/kaggle/working', 'submission.csv'), index = False)

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
