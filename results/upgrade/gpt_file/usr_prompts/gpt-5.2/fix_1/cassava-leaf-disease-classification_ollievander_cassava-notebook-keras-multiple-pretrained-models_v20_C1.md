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

0.8445149592021759

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
INPUT_DIR = '../input/cassava-leaf-disease-classification/'
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
TRAIN_PATH = '../input/cassava-leaf-disease-classification/train_images'
TEST_PATH = '../input/cassava-leaf-disease-classification/test_images'

## === cell 1
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate
)
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras.utils import to_categorical
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from PIL import Image
import numpy as np

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2747770568.py in <cell line: 0>()
      3 from sklearn.model_selection import StratifiedKFold
      4 import albumentations as A
----> 5 from albumentations import (
      6     Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
      7     Rotate

ImportError: cannot import name 'RandomBrightness' from 'albumentations' (/usr/local/lib/python3.11/dist-packages/albumentations/__init__.py)

## === cell 2
train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
train

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2131246260.py in <cell line: 0>()
----> 1 train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
      2 train

NameError: name 'pd' is not defined

## === cell 3
import json

with open('../input/cassava-leaf-disease-classification/label_num_to_disease_map.json') as f:
    classes = json.load(f)
    
classes

## === cell 4
train['class']=train['label'].apply(lambda x:classes[str(x)])

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3466435379.py in <cell line: 0>()
----> 1 train['class']=train['label'].apply(lambda x:classes[str(x)])

NameError: name 'train' is not defined

## === cell 5
plt.figure(figsize = (15,7))
ax =sns.countplot(x=train['class'],order=train['class'].value_counts().index )
plt.show()


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4093991964.py in <cell line: 0>()
----> 1 plt.figure(figsize = (15,7))
      2 ax =sns.countplot(x=train['class'],order=train['class'].value_counts().index )
      3 plt.show()

NameError: name 'plt' is not defined

## === cell 6
train['path'] = train['image_id'].apply(lambda x:'../input/cassava-leaf-disease-classification/train_images/'+str(x))
train= train.astype('str')
train, val = train_test_split(train, test_size = 0.05, random_state = 100,
                                    stratify = train['label'].values)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2190876702.py in <cell line: 0>()
----> 1 train['path'] = train['image_id'].apply(lambda x:'../input/cassava-leaf-disease-classification/train_images/'+str(x))
      2 train= train.astype('str')
      3 train, val = train_test_split(train, test_size = 0.05, random_state = 100,
      4                                     stratify = train['label'].values)

NameError: name 'train' is not defined

## === cell 7
train

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/973257664.py in <cell line: 0>()
----> 1 train

NameError: name 'train' is not defined

## === cell 8
batch_size=4
def transform(image):
    aug = A.Compose([
        A.Flip(),
        A.Rotate(limit=40),
        A.HorizontalFlip(),
        A.Transpose(p=0.5)
        
    ])
    return aug(image=image)['image']


datagen = ImageDataGenerator(preprocessing_function=transform)\
    .flow_from_dataframe(batch_size=batch_size,
        dataframe=train,
        directory=os.path.join(INPUT_DIR, 'train_images'),
        shuffle=True,
        x_col='image_id',
        y_col='label',
        target_size=(512,512), 
        class_mode='categorical'
    )

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1821016339.py in <cell line: 0>()
     11 
     12 
---> 13 datagen = ImageDataGenerator(preprocessing_function=transform)\
     14     .flow_from_dataframe(batch_size=batch_size,
     15         dataframe=train,

NameError: name 'ImageDataGenerator' is not defined

## === cell 9
val_datagen = ImageDataGenerator()\
    .flow_from_dataframe(batch_size=batch_size,
        dataframe=val,
        directory=os.path.join(INPUT_DIR, 'train_images'),
        shuffle=True,
        x_col='image_id',
        y_col='label',
        target_size=(512,512), 
        class_mode='categorical'
    )

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1404203340.py in <cell line: 0>()
----> 1 val_datagen = ImageDataGenerator()\
      2     .flow_from_dataframe(batch_size=batch_size,
      3         dataframe=val,
      4         directory=os.path.join(INPUT_DIR, 'train_images'),
      5         shuffle=True,

NameError: name 'ImageDataGenerator' is not defined

## === cell 15
TEST_DIR = '../input/cassava-leaf-disease-classification/test_images/'
test_images = os.listdir(TEST_DIR)
predictions = []



## === cell 16
from keras.models import load_model
model = load_model("../input/test-5/weightEffnetB7_v6.h5")
model2=load_model("../input/mdpa56/initialweightInceptionResnet4.h5")

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 17
df_test=pd.DataFrame(test_images)
df_test['path']=df_test[0].apply(lambda x:'../input/cassava-leaf-disease-classification/test_images/'+str(x))
df_test

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3693978550.py in <cell line: 0>()
----> 1 df_test=pd.DataFrame(test_images)
      2 df_test['path']=df_test[0].apply(lambda x:'../input/cassava-leaf-disease-classification/test_images/'+str(x))
      3 df_test

NameError: name 'pd' is not defined

## === cell 18
testgen = ImageDataGenerator(#rescale=1. / 255.0,
                                    horizontal_flip = True, 
                                    vertical_flip = True, 
                                    height_shift_range=0.2, 
                                    width_shift_range=0.2, 
                                    rotation_range=30, 
                                    shear_range=0.2,
                                    fill_mode='nearest',
                                    zoom_range=[0.3,0.6])
test_gen2= testgen.flow_from_dataframe(dataframe=df_test,

                                                x_col="path",
                                                y_col=None,
                                                batch_size=batch_size,
                                                seed=42,
                                                shuffle=False,
                                                class_mode=None,
                                                target_size=(512,512))



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1044846325.py in <cell line: 0>()
----> 1 testgen = ImageDataGenerator(#rescale=1. / 255.0,
      2                                     horizontal_flip = True,
      3                                     vertical_flip = True,
      4                                     height_shift_range=0.2,
      5                                     width_shift_range=0.2,

NameError: name 'ImageDataGenerator' is not defined

## === cell 19
preds = []
tta=10
for i in range(tta):
               preds.append(model.predict(test_gen2)+model2.predict(test_gen2))
predbis=np.mean(preds,axis=0)
predictions = np.argmax(predbis, axis = -1)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3302835524.py in <cell line: 0>()
      2 tta=10
      3 for i in range(tta):
----> 4                preds.append(model.predict(test_gen2)+model2.predict(test_gen2))
      5 predbis=np.mean(preds,axis=0)
      6 predictions = np.argmax(predbis, axis = -1)

NameError: name 'model' is not defined

## === cell 21
submission = pd.DataFrame({'image_id': test_images, 'label': predictions})
submission.to_csv('submission.csv', index = False)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2209008323.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({'image_id': test_images, 'label': predictions})
      2 submission.to_csv('submission.csv', index = False)

NameError: name 'pd' is not defined

## === cell 22
submission

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/493289180.py in <cell line: 0>()
----> 1 submission

NameError: name 'submission' is not defined
