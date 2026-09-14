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

albumentations==2.0.8
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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_version

    _major = int(_pb_version.split(".", 1)[0])
except Exception:
    _major = None

if _major is None or _major >= 5:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])
    import importlib
    import google.protobuf  # noqa: F401

    importlib.reload(google.protobuf)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

from tensorflow.keras.layers import (
    Activation,
    Dropout,
    Flatten,
    Dense,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image
import seaborn as sns
import os, cv2, json
import matplotlib.pyplot as plt
import albumentations as aug
from keras.optimizers import SGD


## === cell 2
import warnings
warnings.simplefilter("ignore")


## === cell 3
general_path = '../input/cassava-leaf-disease-classification/'
os.listdir(general_path)


## === cell 4
with open(os.path.join(general_path, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
    map_classes = {int(k) : v for k, v in map_classes.items()}
    
    print(json.dumps(map_classes, indent=4))


## === cell 5
input_files = os.listdir(os.path.join(general_path, "train_images"))
print(f"Number of train images: {len(input_files)}")


## === cell 6
img_shapes = {}
for image_name in os.listdir(os.path.join(general_path, "train_images"))[:300]:
    image = cv2.imread(os.path.join(general_path, "train_images", image_name))
    img_shapes[image.shape] = img_shapes.get(image.shape, 0) + 1

print(img_shapes)


## === cell 7
df_train = pd.read_csv(os.path.join(general_path, "train.csv"))

df_train["class_name"] = df_train["label"].map(map_classes)

df_train


## === cell 8
plt.figure(figsize=(8, 4))
sns.countplot(y="class_name", data=df_train);


## === cell 9
def visualize_batch(image_ids, labels,class_name):
    plt.figure(figsize=(16, 12))
    
    for ind, (image_id, label,class_name) in enumerate(zip(image_ids, labels,class_name)):
        plt.subplot(3, 3, ind + 1)
        image = cv2.imread(os.path.join(general_path, "train_images", image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.title(f"Class {label}:  {class_name}", fontsize=12)
        plt.axis("off")
    
    plt.show()


## === cell 10
tmp_df = df_train[df_train["label"] == 0]
print(f"Total train images for class 0: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(6)
image_ids = tmp_df["image_id"].values
labels = tmp_df["label"].values
class_name = tmp_df["class_name"].values

visualize_batch(image_ids, labels,class_name)


## === cell 11
tmp_df = df_train[df_train["label"] == 1]
print(f"Total train images for class 1: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(6)
image_ids = tmp_df["image_id"].values
labels = tmp_df["label"].values
class_name = tmp_df["class_name"].values


visualize_batch(image_ids, labels,class_name)


## === cell 12
tmp_df = df_train[df_train["label"] == 2]
print(f"Total train images for class 2: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(6)
image_ids = tmp_df["image_id"].values
labels = tmp_df["label"].values
class_name = tmp_df["class_name"].values


visualize_batch(image_ids,labels, class_name)


## === cell 13
tmp_df = df_train[df_train["label"] == 3]
print(f"Total train images for class 2: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(6)
image_ids = tmp_df["image_id"].values
labels = tmp_df["label"].values
class_name = tmp_df["class_name"].values
visualize_batch(image_ids, labels,class_name)


## === cell 14
tmp_df = df_train[df_train["label"] == 4]
print(f"Total train images for class 4: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(6)
image_ids = tmp_df["image_id"].values
labels = tmp_df["label"].values
class_name = tmp_df["class_name"].values

visualize_batch(image_ids, labels,class_name)


## === cell 15
def plot_augmentation(image_id, transform):
    plt.figure(figsize=(12, 12))
    img = cv2.imread(os.path.join(general_path, "train_images", image_id))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("original")
    
    plt.subplot(2, 2, 2)
    x = transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")
    plt.title("Augmentation -1")

    
    plt.subplot(2, 2, 3)
    x = transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")
    plt.title("Augmentation -2")
    
    plt.subplot(2, 2, 4)
    x = transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")
    plt.title("Augmentation -3")
    
    plt.show()


## === cell 16
transform_shift_scale_rotate = aug.ShiftScaleRotate(
    p=1.0, 
    shift_limit=(-0.3, 0.3), 
    scale_limit=(-0.1, 0.1), 
    rotate_limit=(-180, 180), 
    interpolation=0, 
    border_mode=4, 
)

plot_augmentation("1003442061.jpg", transform_shift_scale_rotate)


## === cell 17
transform_coarse_dropout = aug.CoarseDropout(
    p=1.0, 
    max_holes=100, 
    max_height=50, 
    max_width=50, 
    min_holes=30, 
    min_height=20, 
    min_width=20,
)

plot_augmentation("1003442061.jpg", transform_coarse_dropout)


## === cell 18
transform_coarse_dropout = aug.HueSaturationValue(
    hue_shift_limit=0,
    sat_shift_limit=(40,80),
    val_shift_limit=(40,80)
)

plot_augmentation("5912799.jpg", transform_coarse_dropout)


## === cell 19
transform_coarse_dropout = aug.CLAHE(
 
     always_apply=False,
      p=1.0,
      clip_limit=(10, 30),
      tile_grid_size=(10, 10))
plot_augmentation("999329392.jpg", transform_coarse_dropout)


## === cell 20
transform_coarse_dropout = aug.RandomFog( p=1.0)
plot_augmentation("999329392.jpg", transform_coarse_dropout)


## === cell 21
transform_coarse_dropout = aug.RandomSunFlare( p=1.0)
plot_augmentation("999329392.jpg", transform_coarse_dropout)


## === cell 22
transform_coarse_dropout = aug.RandomBrightnessContrast(
    p=1.0,
    brightness_limit=0.2,
    contrast_limit=0.0,
)
plot_augmentation("999329392.jpg", transform_coarse_dropout)


## === cell 23
transform_coarse_dropout = aug.RandomCrop(p=1,height = 512, width = 512)
plot_augmentation("999329392.jpg", transform_coarse_dropout)


## === cell 24
transform_coarse_dropout = aug.RGBShift(p=1)
plot_augmentation("999329392.jpg", transform_coarse_dropout)


## === cell 25
transform_coarse_dropout = aug.RandomSnow(p=1)
plot_augmentation("999329392.jpg", transform_coarse_dropout)


## === cell 26
transform_coarse_dropout = aug.HorizontalFlip(p=1)
plot_augmentation("999329392.jpg", transform_coarse_dropout)


## === cell 27
transform_coarse_dropout = aug.VerticalFlip(p=1)
plot_augmentation("999329392.jpg", transform_coarse_dropout)


## === cell 28
transform_coarse_dropout = aug.RandomBrightnessContrast(
    brightness_limit=0.0, contrast_limit=0.5, p=1
)
plot_augmentation("999329392.jpg", transform_coarse_dropout)


## === cell 29
transform_coarse_dropout = aug.Cutout(p=1)
plot_augmentation("999329392.jpg", transform_coarse_dropout)


## --- ERROR in cell 29, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3188630108.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mtransform_coarse_dropout[0m [0;34m=[0m [0maug[0m[0;34m.[0m[0mCutout[0m[0;34m([0m[0mp[0m[0;34m=[0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mplot_augmentation[0m[0;34m([0m[0;34m"999329392.jpg"[0m[0;34m,[0m [0mtransform_coarse_dropout[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: module 'albumentations' has no attribute 'Cutout'

## === cell 30
transform_coarse_dropout = aug.Transpose(p=1)
plot_augmentation("999329392.jpg", transform_coarse_dropout)
