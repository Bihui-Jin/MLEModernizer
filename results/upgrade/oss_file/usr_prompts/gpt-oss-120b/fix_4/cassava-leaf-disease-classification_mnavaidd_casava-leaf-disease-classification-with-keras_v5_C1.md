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

geopandas==0.14.4
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

0.0018

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.20927) has done: 'The changes add parallel data loading during training by enabling multiprocessing workers, and replace the per‑image prediction loop with efficient batched inference while keeping the same model, architecture, and training schedule. These tweaks drastically lower I/O and Python‑loop overhead without altering any model logic or accuracy.'

# 9. Code solution

## === cell 0
import os

from google.protobuf import message_factory

if not hasattr(message_factory.MessageFactory, "GetPrototype"):
    message_factory.MessageFactory.GetPrototype = (
        message_factory.MessageFactory.GetMessageClass
    )

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3062839733.py in <cell line: 0>()
      7 if not hasattr(message_factory.MessageFactory, "GetPrototype"):
      8     message_factory.MessageFactory.GetPrototype = (
----> 9         message_factory.MessageFactory.GetMessageClass
     10     )
     11 

AttributeError: type object 'MessageFactory' has no attribute 'GetMessageClass'

## === cell 1
data_path = "/kaggle/input/cassava-leaf-disease-classification/"
train_csv_data_path = data_path + "train.csv"
label_json_data_path = data_path + "label_num_to_disease_map.json"
images_dir_data_path = data_path + "train_images"



## === cell 2
train_csv = pd.read_csv(train_csv_data_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_data_path, orient="index")
label_class = label_class.values.flatten().tolist()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3621794643.py in <cell line: 0>()
----> 1 train_csv = pd.read_csv(train_csv_data_path)
      2 train_csv["label"] = train_csv["label"].astype("string")
      3 
      4 label_class = pd.read_json(label_json_data_path, orient="index")
      5 label_class = label_class.values.flatten().tolist()

NameError: name 'pd' is not defined

## === cell 3
train_data_label_3 = train_csv[train_csv["label"] == "3"]
train_data_label_3 = shuffle(train_data_label_3)
train_data_label_3 = train_data_label_3[:3000]

train_data_label_not_3 = train_csv[train_csv["label"] != "3"]

train_csv = pd.concat([train_data_label_3, train_data_label_not_3], ignore_index=True)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2077132043.py in <cell line: 0>()
----> 1 train_data_label_3 = train_csv[train_csv["label"] == "3"]
      2 train_data_label_3 = shuffle(train_data_label_3)
      3 train_data_label_3 = train_data_label_3[:3000]
      4 
      5 train_data_label_not_3 = train_csv[train_csv["label"] != "3"]

NameError: name 'train_csv' is not defined

## === cell 4
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2872813560.py in <cell line: 0>()
      1 print("Label names :")
----> 2 for i, label in enumerate(label_class):
      3     print(f" {i}. {label}")
      4 

NameError: name 'label_class' is not defined

## === cell 5
train_csv.head()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2594079454.py in <cell line: 0>()
----> 1 train_csv.head()
      2 

NameError: name 'train_csv' is not defined

## === cell 6
train_gen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.1, 0.9],
    shear_range=25,
    zoom_range=0.3,
    channel_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1 / 255,
    validation_split=0.15,
)

valid_gen = ImageDataGenerator(rescale=1 / 255, validation_split=0.15)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1487264624.py in <cell line: 0>()
----> 1 train_gen = ImageDataGenerator(
      2     rotation_range=360,
      3     width_shift_range=0.1,
      4     height_shift_range=0.1,
      5     brightness_range=[0.1, 0.9],

NameError: name 'ImageDataGenerator' is not defined

## === cell 7
BATCH_SIZE = 18
IMG_SIZE = 320



## === cell 8
train_generator = train_gen.flow_from_dataframe(
    dataframe=train_csv,
    directory=images_dir_data_path,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset="training",
)

valid_generator = valid_gen.flow_from_dataframe(
    dataframe=train_csv,
    directory=images_dir_data_path,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
    subset="validation",
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2767456312.py in <cell line: 0>()
----> 1 train_generator = train_gen.flow_from_dataframe(
      2     dataframe=train_csv,
      3     directory=images_dir_data_path,
      4     x_col="image_id",
      5     y_col="label",

NameError: name 'train_gen' is not defined

## === cell 9
batch = next(train_generator)
images = batch[0]
labels = batch[1]



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3600404771.py in <cell line: 0>()
----> 1 batch = next(train_generator)
      2 images = batch[0]
      3 labels = batch[1]
      4 

NameError: name 'train_generator' is not defined

## === cell 10
plt.figure(figsize=(12, 9))
for i, (img, label) in enumerate(zip(images, labels)):
    plt.subplot(2, 3, i % 6 + 1)
    plt.axis("off")
    plt.imshow(img)
    plt.title(label_class[np.argmax(label)])

    if i == 15:
        break



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/593174744.py in <cell line: 0>()
----> 1 plt.figure(figsize=(12, 9))
      2 for i, (img, label) in enumerate(zip(images, labels)):
      3     plt.subplot(2, 3, i % 6 + 1)
      4     plt.axis("off")
      5     plt.imshow(img)

NameError: name 'plt' is not defined

## === cell 11
base = tf.keras.applications.ResNet152V2(
    include_top=False, weights="imagenet", input_shape=[IMG_SIZE, IMG_SIZE, 3]
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1401534595.py in <cell line: 0>()
----> 1 base = tf.keras.applications.ResNet152V2(
      2     include_top=False, weights="imagenet", input_shape=[IMG_SIZE, IMG_SIZE, 3]
      3 )
      4 

NameError: name 'tf' is not defined

## === cell 12
base.summary()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3762650433.py in <cell line: 0>()
----> 1 base.summary()
      2 

NameError: name 'base' is not defined

## === cell 13
model = tf.keras.Sequential()
model.add(base)
model.add(BatchNormalization(axis=-1))
model.add(GlobalAveragePooling2D())
model.add(Dense(5, activation="softmax"))



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1276249332.py in <cell line: 0>()
----> 1 model = tf.keras.Sequential()
      2 model.add(base)
      3 model.add(BatchNormalization(axis=-1))
      4 model.add(GlobalAveragePooling2D())
      5 model.add(Dense(5, activation="softmax"))

NameError: name 'tf' is not defined

## === cell 14
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.01),
    metrics=["acc"],
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1855623367.py in <cell line: 0>()
----> 1 model.compile(
      2     loss=tf.keras.losses.CategoricalCrossentropy(),
      3     optimizer=tf.keras.optimizers.Adamax(learning_rate=0.01),
      4     metrics=["acc"],
      5 )

NameError: name 'model' is not defined

## === cell 15
model.summary()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1903595429.py in <cell line: 0>()
----> 1 model.summary()
      2 

NameError: name 'model' is not defined

## === cell 16
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=20,
    validation_data=valid_generator,
    verbose=1,
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3472478374.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_generator,
      3     steps_per_epoch=train_generator.samples // BATCH_SIZE,
      4     epochs=20,
      5     validation_data=valid_generator,

NameError: name 'model' is not defined

## === cell 17
model.save("model.h5")



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/365182331.py in <cell line: 0>()
----> 1 model.save("model.h5")
      2 

NameError: name 'model' is not defined

## === cell 18
model = tf.keras.models.load_model("/kaggle/working/model.h5")



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/685029229.py in <cell line: 0>()
----> 1 model = tf.keras.models.load_model("/kaggle/working/model.h5")
      2 

NameError: name 'tf' is not defined

## === cell 19
test_img_path = data_path + "test_images/2216849948.jpg"

img = cv2.imread(test_img_path)
if img is not None:
    resized_img = (
        cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255
    )

    plt.figure(figsize=(8, 4))
    plt.title("TEST IMAGE")
    plt.imshow(resized_img[0])
else:
    print(f"Warning: test image {test_img_path} not found; skipping visual check.")



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2009400050.py in <cell line: 0>()
      1 test_img_path = data_path + "test_images/2216849948.jpg"
      2 
----> 3 img = cv2.imread(test_img_path)
      4 if img is not None:
      5     resized_img = (

NameError: name 'cv2' is not defined

## === cell 20
ss = pd.read_csv(data_path + "sample_submission.csv")
test_filenames = [
    os.path.join(data_path, "test_images", img_name) for img_name in ss.image_id
]

preds = []
for i in range(0, len(test_filenames), BATCH_SIZE):
    batch_paths = test_filenames[i : i + BATCH_SIZE]
    batch_imgs = []
    for p in batch_paths:
        img = tf.keras.preprocessing.image.load_img(p)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.preprocessing.image.smart_resize(img, (IMG_SIZE, IMG_SIZE))
        batch_imgs.append(img)
    batch_array = np.stack(batch_imgs, axis=0) / 255.0
    batch_pred = model.predict(batch_array, batch_size=BATCH_SIZE, verbose=0)
    preds.extend(np.argmax(batch_pred, axis=1))

my_submission = pd.DataFrame({"image_id": ss.image_id, "label": preds})
my_submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3148206470.py in <cell line: 0>()
----> 1 ss = pd.read_csv(data_path + "sample_submission.csv")
      2 test_filenames = [
      3     os.path.join(data_path, "test_images", img_name) for img_name in ss.image_id
      4 ]
      5 

NameError: name 'pd' is not defined

## === cell 21
print("Submission File: \n---------------\n")
print(my_submission.head())

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3014216753.py in <cell line: 0>()
      1 print("Submission File: \n---------------\n")
----> 2 print(my_submission.head())

NameError: name 'my_submission' is not defined
