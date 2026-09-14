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

0.8818374131157449

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, json, random
import numpy as np, pandas as pd
from PIL import Image



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"



## === cell 2
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as f:
    map_classes = json.load(f)
label_list = [int(k) for k in map_classes.keys()]



## === cell 3
IMG_HEIGHT = 300
IMG_WIDTH = 300
BATCH_SIZE = 32
EPOCHS = 3  # short training for quick turnaround



## === cell 4
from albumentations import (
    Compose,
    HorizontalFlip,
    RandomContrast,
    RandomBrightness,
    CenterCrop,
    ShiftScaleRotate,
    ToFloat,
)

AUGMENTATIONS_TRAIN = Compose(
    [
        HorizontalFlip(p=0.5),
        RandomContrast(limit=0.2, p=0.5),
        RandomBrightness(limit=0.2, p=0.5),
        CenterCrop(always_apply=False, p=1.0, height=IMG_HEIGHT, width=IMG_WIDTH),
        ShiftScaleRotate(
            always_apply=False,
            p=0.5,
            shift_limit=0,
            scale_limit=(0.5, 1.5),
            rotate_limit=15,
            interpolation=0,
            border_mode=0,
        ),
        ToFloat(max_value=255),
    ]
)

AUGMENTATIONS_TEST = Compose([ToFloat(max_value=255)])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/4174428256.py in <cell line: 0>()
      1 # Albumentations – keep only available transforms
----> 2 from albumentations import (
      3     Compose,
      4     HorizontalFlip,
      5     RandomContrast,

ImportError: cannot import name 'RandomContrast' from 'albumentations' (/usr/local/lib/python3.11/dist-packages/albumentations/__init__.py)

## === cell 5
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
train_df["label"] = train_df["label"].astype(
    str
)  # ensure string for flow_from_dataframe


def albumentations_wrapper(augmentations):
    def func(img):
        augmented = augmentations(image=img)
        return augmented["image"]

    return func


train_datagen = ImageDataGenerator(
    preprocessing_function=albumentations_wrapper(AUGMENTATIONS_TRAIN),
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_DIR,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_DIR,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1317636983.py in <cell line: 0>()
     17 
     18 train_datagen = ImageDataGenerator(
---> 19     preprocessing_function=albumentations_wrapper(AUGMENTATIONS_TRAIN),
     20     validation_split=0.2,
     21 )

NameError: name 'AUGMENTATIONS_TRAIN' is not defined

## === cell 7
base_model = applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False  # freeze for quick training

model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(len(label_list), activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()



## === cell 8
model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator, verbose=2)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1699798959.py in <cell line: 0>()
      1 # Train the model (few epochs for speed)
----> 2 model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator, verbose=2)
      3 

NameError: name 'train_generator' is not defined

## === cell 9
test_df = pd.DataFrame({"image_id": os.listdir(TEST_DIR)})

test_datagen = ImageDataGenerator(
    preprocessing_function=albumentations_wrapper(AUGMENTATIONS_TEST)
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=TEST_DIR,
    x_col="image_id",
    y_col=None,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/437299239.py in <cell line: 0>()
      3 
      4 test_datagen = ImageDataGenerator(
----> 5     preprocessing_function=albumentations_wrapper(AUGMENTATIONS_TEST)
      6 )
      7 

NameError: name 'AUGMENTATIONS_TEST' is not defined

## === cell 10
preds = model.predict(test_generator, verbose=2)
pred_labels = np.argmax(preds, axis=1)

submission = pd.DataFrame({"image_id": test_df["image_id"], "label": pred_labels})
submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3814889105.py in <cell line: 0>()
      1 # Predict
----> 2 preds = model.predict(test_generator, verbose=2)
      3 pred_labels = np.argmax(preds, axis=1)
      4 
      5 submission = pd.DataFrame({"image_id": test_df["image_id"], "label": pred_labels})

NameError: name 'test_generator' is not defined

## === cell 11
print("Submission file created:", os.path.isfile("submission.csv"))
print(submission.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1685574671.py in <cell line: 0>()
      1 # Verify file creation
      2 print("Submission file created:", os.path.isfile("submission.csv"))
----> 3 print(submission.head())

NameError: name 'submission' is not defined
