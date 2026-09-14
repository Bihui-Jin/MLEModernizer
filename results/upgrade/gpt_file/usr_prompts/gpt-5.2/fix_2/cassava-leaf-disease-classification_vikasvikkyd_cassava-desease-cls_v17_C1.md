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

0.6740707162284678

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import random

import numpy as np
import pandas as pd
import cv2

import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras import layers



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "../input/cassava-leaf-disease-classification"
source_dir = os.path.join(BASE_DIR, "train_images")
train_csv_path = os.path.join(BASE_DIR, "train.csv")
label_name_path = os.path.join(BASE_DIR, "label_num_to_disease_map.json")

data_label = pd.read_csv(train_csv_path)

with open(label_name_path, "r") as f:
    label_name = json.load(f)

disk_images = set(os.listdir(source_dir))
all_images = [img for img in data_label["image_id"].tolist() if img in disk_images]

random.seed(42)
random.shuffle(all_images)

print("Train images on disk:", len(disk_images))
print("Train labels in CSV:", len(data_label))
print("Usable images (intersection):", len(all_images))




## === cell 2
def load_data(source_dir, img_height, img_width, train_size=0.9):
    X_train, y_train, X_test, y_test = [], [], [], []

    split_size = int(len(all_images) * train_size)
    train_images = all_images[:split_size]
    test_images = all_images[split_size:]

    id_to_label = dict(zip(data_label["image_id"].values, data_label["label"].values))

    def _read_and_resize(path):
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)

    for img in train_images:
        x = _read_and_resize(os.path.join(source_dir, img))
        if x is None:
            continue
        X_train.append(x)
        y_train.append(id_to_label[img])

    for img in test_images:
        x = _read_and_resize(os.path.join(source_dir, img))
        if x is None:
            continue
        X_test.append(x)
        y_test.append(id_to_label[img])

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)




## === cell 3
X_train, y_train, X_test, y_test = load_data(source_dir, 100, 100, 0.9)

if len(X_train) == 0 or len(X_test) == 0:
    raise RuntimeError(
        f"Loaded empty train/test arrays: X_train={X_train.shape}, X_test={X_test.shape}"
    )

y_train = to_categorical(y_train, 5)
y_test = to_categorical(y_test, 5)

datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_generator = datagen.flow(X_train, y_train, batch_size=32, shuffle=True)

datagen_test = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = datagen_test.flow(X_test, y_test, batch_size=32, shuffle=False)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2406197753.py in <cell line: 0>()
     11 y_test = to_categorical(y_test, 5)
     12 
---> 13 datagen = ImageDataGenerator(rescale=1.0 / 255.0)
     14 train_generator = datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
     15 

NameError: name 'ImageDataGenerator' is not defined

## === cell 4
def define_directory():
    base = "/kaggle/working/Training"
    subdirs = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
    if not os.path.exists(base):
        os.mkdir(base)
    for sd in subdirs:
        p = os.path.join(base, sd)
        if not os.path.exists(p):
            os.mkdir(p)




## === cell 5
pre_trained_model = VGG16(
    include_top=False, weights="imagenet", input_shape=(100, 100, 3)
)

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer("block5_pool")
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(5, activation="softmax")(x)

model = keras.Model(pre_trained_model.input, x)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/733304480.py in <cell line: 0>()
      2 # Fix: the original code attempted to load weights from a missing Kaggle dataset path.
      3 # Using built-in ImageNet weights is the minimal stable replacement and improves accuracy vs weights=None.
----> 4 pre_trained_model = VGG16(
      5     include_top=False, weights="imagenet", input_shape=(100, 100, 3)
      6 )

NameError: name 'VGG16' is not defined

## === cell 6
history = model.fit(
    train_generator, validation_data=test_generator, epochs=10, verbose=2
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1395289239.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_generator, validation_data=test_generator, epochs=10, verbose=2
      3 )
      4 

NameError: name 'model' is not defined

## === cell 7
test_dir = os.path.join(BASE_DIR, "test_images")
test_images = os.listdir(test_dir)

sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
test_images_set = set(test_images)

ordered_test_ids = [
    img for img in sample_sub["image_id"].tolist() if img in test_images_set
]
if len(ordered_test_ids) != len(sample_sub):
    ordered_test_ids = sorted(test_images)

img_id = []
lbl = []

for img in ordered_test_ids:
    img_path = os.path.join(test_dir, img)
    x = cv2.imread(img_path)
    if x is None:
        pred_class = 0
    else:
        x = cv2.resize(x, (100, 100), interpolation=cv2.INTER_AREA)
        x = x.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        probs = model.predict(x, verbose=0)
        pred_class = int(np.argmax(probs, axis=1)[0])

    img_id.append(img)
    lbl.append(pred_class)

submission_df = pd.DataFrame({"image_id": img_id, "label": lbl})
submission_path = "./submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/747673278.py in <cell line: 0>()
     28         x = x.astype(np.float32) / 255.0
     29         x = np.expand_dims(x, axis=0)
---> 30         probs = model.predict(x, verbose=0)
     31         pred_class = int(np.argmax(probs, axis=1)[0])
     32 

NameError: name 'model' is not defined
