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

3.9

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

0.8389241462677546

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os



## === cell 2
import tensorflow as tf
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.datasets import cifar10
from keras.engine import training
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Activation, Average
from keras.losses import categorical_crossentropy
from keras.models import Model, Input
from keras.optimizers import Adam

import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications import InceptionResNetV2

import matplotlib.pyplot as plt


from keras.utils import to_categorical
from tensorflow.python.framework.ops import Tensor
from typing import Tuple, List
import glob
import numpy as np
import os
from keras.models import load_model
from keras.utils import to_categorical
from numpy import dstack

from sklearn.metrics import classification_report

import pandas as pd

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
tf.random.set_seed(23)


IMG_SIZE_incres = 320
IMG_SIZE_effnet = 333
BATCH_SZ = 320

## === cell 4
effnet_path = "../input/effnetpp/eff.h5"
incRes_path = "../input/inceptionresnet/inceptionResNetv2_Sun_6pm.h5"

model_paths = [effnet_path, incRes_path]

list_of_models = [load_model(m_p) for m_p in model_paths]
effModel = list_of_models[0]
incResModel = list_of_models[1]

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/111953385.py in <cell line: 0>()
      4 model_paths = [effnet_path, incRes_path]
      5 
----> 6 list_of_models = [load_model(m_p) for m_p in model_paths]
      7 effModel = list_of_models[0]
      8 incResModel = list_of_models[1]

/tmp/ipykernel_11/111953385.py in <listcomp>(.0)
      4 model_paths = [effnet_path, incRes_path]
      5 
----> 6 list_of_models = [load_model(m_p) for m_p in model_paths]
      7 effModel = list_of_models[0]
      8 incResModel = list_of_models[1]

NameError: name 'load_model' is not defined

## === cell 5
train = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")
train['label'] = train['label'].astype('string')
train.head()

diseases = pd.read_json("/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json", typ='series')
diseases

## === cell 6
datagen_incres = image.ImageDataGenerator(rotation_range=360,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                brightness_range=[0.2,1.5],
                                shear_range=25,
                                zoom_range=0.3,
                                channel_shift_range=0.1,
                                horizontal_flip=True,
                                vertical_flip=True,
                                rescale=1/255,
                                validation_split=0.15)

val_datagen_incres = image.ImageDataGenerator(rescale=1/255,
                                       validation_split = 0.2)

train_generator_incres = datagen_incres.flow_from_dataframe(
    dataframe=train,
    directory="/kaggle/input/cassava-leaf-disease-classification/train_images",
    x_col='image_id',
    y_col='label',
    target_size=(IMG_SIZE_incres, IMG_SIZE_incres),
    batch_size=32,
    subset='training',
    shuffle = True,
    class_mode='categorical'
)

val_generator_incres = val_datagen_incres.flow_from_dataframe(
    dataframe=train,
    directory="/kaggle/input/cassava-leaf-disease-classification/train_images",
    x_col='image_id',
    y_col='label',
    target_size=(IMG_SIZE_incres, IMG_SIZE_incres),
    batch_size=32,
    subset='validation',
    class_mode = 'categorical',
    shuffle = True
)

test_generator_incres = val_datagen_incres.flow_from_dataframe(
    dataframe=train,
    directory="/kaggle/input/cassava-leaf-disease-classification/train_images",
    x_col='image_id',
    y_col='label',
    target_size=(IMG_SIZE_incres, IMG_SIZE_incres),
    batch_size=BATCH_SZ,
    subset='validation',
    class_mode = 'categorical',
    shuffle = False
)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3813444798.py in <cell line: 0>()
----> 1 datagen_incres = image.ImageDataGenerator(rotation_range=360,
      2                                 width_shift_range=0.1,
      3                                 height_shift_range=0.1,
      4                                 brightness_range=[0.2,1.5],
      5                                 shear_range=25,

NameError: name 'image' is not defined

## === cell 7
datagen_effnet = image.ImageDataGenerator(rotation_range=360,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                brightness_range=[0.2,1.5],
                                shear_range=25,
                                zoom_range=0.3,
                                channel_shift_range=0.1,
                                horizontal_flip=True,
                                vertical_flip=True,
                                rescale=1/255,
                                validation_split=0.15)

val_datagen_effnet = image.ImageDataGenerator(rescale=1/255,
                                       validation_split = 0.2)

train_generator_effnet = datagen_effnet.flow_from_dataframe(
    dataframe=train,
    directory="/kaggle/input/cassava-leaf-disease-classification/train_images",
    x_col='image_id',
    y_col='label',
    target_size=(IMG_SIZE_effnet, IMG_SIZE_effnet),
    batch_size=32,
    subset='training',
    shuffle = True,
    class_mode='categorical'
)

val_generator_effnet = val_datagen_effnet.flow_from_dataframe(
    dataframe=train,
    directory="/kaggle/input/cassava-leaf-disease-classification/train_images",
    x_col='image_id',
    y_col='label',
    target_size=(IMG_SIZE_effnet, IMG_SIZE_effnet),
    batch_size=32,
    subset='validation',
    class_mode = 'categorical',
    shuffle = True
)

test_generator_effnet = val_datagen_effnet.flow_from_dataframe(
    dataframe=train,
    directory="/kaggle/input/cassava-leaf-disease-classification/train_images",
    x_col='image_id',
    y_col='label',
    target_size=(IMG_SIZE_effnet, IMG_SIZE_effnet),
    batch_size=BATCH_SZ,
    subset='validation',
    class_mode = 'categorical',
    shuffle = False
)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1241495674.py in <cell line: 0>()
----> 1 datagen_effnet = image.ImageDataGenerator(rotation_range=360,
      2                                 width_shift_range=0.1,
      3                                 height_shift_range=0.1,
      4                                 brightness_range=[0.2,1.5],
      5                                 shear_range=25,

NameError: name 'image' is not defined

## === cell 20
import cv2

## === cell 21
sample_sub = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv')

def getIndividProbPred(image_str, IMG_SIZE, model):
    img = tf.keras.preprocessing.image.load_img('../input/cassava-leaf-disease-classification/test_images/' + image_str)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.preprocessing.image.smart_resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.reshape(img, (-1, IMG_SIZE, IMG_SIZE, 3))
    prediction = model.predict(img/255)
    return prediction

def getIRpred(img_id): return getIndividProbPred(img_id, IMG_SIZE_incres, incResModel)
def getEffpred(img_id): return getIndividProbPred(img_id, IMG_SIZE_effnet, effModel)

def getEnsembPredForImg(img_id):
    predictions = [getIRpred(img_id), getEffpred(img_id)]
    avg_preds = np.average(predictions, axis=0)
    class_pred = avg_preds.argmax(axis=1)
    return class_pred.item(0)


## === cell 22
ss = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')
preds = [getEnsembPredForImg(img_id) for img_id in ss.image_id]
preds

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3591423991.py in <cell line: 0>()
      1 ss = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')
----> 2 preds = [getEnsembPredForImg(img_id) for img_id in ss.image_id]
      3 preds

/tmp/ipykernel_11/3591423991.py in <listcomp>(.0)
      1 ss = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')
----> 2 preds = [getEnsembPredForImg(img_id) for img_id in ss.image_id]
      3 preds

/tmp/ipykernel_11/2173733470.py in getEnsembPredForImg(img_id)
     13 
     14 def getEnsembPredForImg(img_id):
---> 15     predictions = [getIRpred(img_id), getEffpred(img_id)]
     16     avg_preds = np.average(predictions, axis=0)
     17     class_pred = avg_preds.argmax(axis=1)

/tmp/ipykernel_11/2173733470.py in getIRpred(img_id)
      9     return prediction
     10 
---> 11 def getIRpred(img_id): return getIndividProbPred(img_id, IMG_SIZE_incres, incResModel)
     12 def getEffpred(img_id): return getIndividProbPred(img_id, IMG_SIZE_effnet, effModel)
     13 

NameError: name 'incResModel' is not defined

## === cell 23

submission = pd.DataFrame({'image_id': ss.image_id, 'label': preds})
submission

submission.to_csv('submission.csv', index = False)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2430045708.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({'image_id': ss.image_id, 'label': preds})
      2 submission
      3 
      4 submission.to_csv('submission.csv', index = False)

NameError: name 'preds' is not defined

## === cell 24
submission

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/493289180.py in <cell line: 0>()
----> 1 submission

NameError: name 'submission' is not defined
