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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.6682363804247459

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        pass



## === cell 1
import tf_keras as keras
import tensorflow as tf

print("TF version:", tf.__version__)
print("tf_keras version:", keras.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
import cv2
import re
from PIL import Image
import matplotlib.pyplot as plt

from tf_keras.preprocessing.image import img_to_array, load_img
from tf_keras.applications.resnet50 import (
    preprocess_input,
    decode_predictions,
    ResNet50,
)
from tf_keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tf_keras.layers import Dropout, Flatten, Dense, Activation
from tf_keras.models import Sequential
from sklearn.model_selection import train_test_split
from tf_keras import optimizers
from tf_keras.applications.vgg16 import VGG16
from tf_keras.applications.vgg19 import VGG19
from tf_keras.models import Model
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras import backend as K

import pandas as pd
import numpy as np
import os



## === cell 3
sam_sub = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")
sam_sub.head()



## === cell 4
train_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"



## === cell 5
train = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
train.head()



## === cell 6
test_filepaths = []
test_ids = []

for dirname, _, filenames in os.walk(test_dir):
    for filename in filenames:
        test_ids.append(filename)
        test_filepaths.append(os.path.join(dirname, filename))



## === cell 7
test_df = pd.DataFrame(test_ids, columns=["image"])
test_df.head()



## === cell 8
train_datagen_sub = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)



## === cell 9
train_generator_sub = train_datagen_sub.flow_from_dataframe(
    train,
    directory=train_dir,
    x_col="image",
    y_col="labels",
    target_size=(432, 648),
    batch_size=16,
    class_mode="categorical",
)



## === cell 10
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col="image",
    target_size=(432, 648),
    batch_size=1,
    class_mode=None,
    shuffle=False,
)



## === cell 11
from tf_keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))




## === cell 12
MODEL_PATH = "../input/effnettop/pp21_top_fit_effnet10"
trained_model_sub = keras.models.load_model(MODEL_PATH, custom_objects={"f1_m": f1_m})
trained_model_sub.summary()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2055380627.py in <cell line: 0>()
      2 # Path is kept identical to the original code.
      3 MODEL_PATH = "../input/effnettop/pp21_top_fit_effnet10"
----> 4 trained_model_sub = keras.models.load_model(MODEL_PATH, custom_objects={"f1_m": f1_m})
      5 trained_model_sub.summary()
      6 

/usr/local/lib/python3.11/dist-packages/tf_keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode, **kwargs)
    260 
    261     # Legacy case.
--> 262     return legacy_sm_saving_lib.load_model(
    263         filepath, custom_objects=custom_objects, compile=compile, **kwargs
    264     )

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tf_keras/src/saving/legacy/save.py in load_model(filepath, custom_objects, compile, options)
    231                     if isinstance(filepath_str, str):
    232                         if not tf.io.gfile.exists(filepath_str):
--> 233                             raise IOError(
    234                                 f"No file or directory found at {filepath_str}"
    235                             )

OSError: No file or directory found at ../input/effnettop/pp21_top_fit_effnet10

## === cell 13
y_pred = trained_model_sub.predict(test_generator, verbose=1)
y_pred.shape



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3239975602.py in <cell line: 0>()
----> 1 y_pred = trained_model_sub.predict(test_generator, verbose=1)
      2 y_pred.shape
      3 

NameError: name 'trained_model_sub' is not defined

## === cell 14
predicted_class_indices = np.argmax(y_pred, axis=1)

labels = train_generator_sub.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

predictions[:10], len(predictions)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1137384418.py in <cell line: 0>()
----> 1 predicted_class_indices = np.argmax(y_pred, axis=1)
      2 
      3 labels = train_generator_sub.class_indices
      4 labels = dict((v, k) for k, v in labels.items())
      5 predictions = [labels[k] for k in predicted_class_indices]

NameError: name 'y_pred' is not defined

## === cell 15
sub = pd.DataFrame({"image": test_df["image"].values, "labels": predictions})

sub = sub[["image", "labels"]]
sub.head()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3980461655.py in <cell line: 0>()
      1 # FIX: Ensure submission rows align with test_generator order (which matches test_df order).
      2 # Use test_df['image'] rather than raw test_ids list (which might include duplicates or different order).
----> 3 sub = pd.DataFrame({"image": test_df["image"].values, "labels": predictions})
      4 
      5 # Safety: match sample_submission column names exactly

NameError: name 'predictions' is not defined

## === cell 16
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2737361063.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub.shape)
      3 print(sub.head())

NameError: name 'sub' is not defined
