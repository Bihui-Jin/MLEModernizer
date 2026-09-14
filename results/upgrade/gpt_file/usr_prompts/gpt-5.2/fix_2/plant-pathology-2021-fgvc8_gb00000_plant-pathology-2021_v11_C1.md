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

0.7410156971375809

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        pass



## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
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

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import (
    preprocess_input,
    decode_predictions,
    ResNet50,
)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

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

test_df = pd.DataFrame(test_ids, columns=["image"])
test_df.head()



## === cell 7
train_datagen_sub = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

train_generator_sub = train_datagen_sub.flow_from_dataframe(
    train,
    directory=train_dir,
    x_col="image",
    y_col="labels",
    target_size=(224, 336),
    batch_size=16,
    class_mode="categorical",
)



## === cell 8
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col="image",
    target_size=(224, 336),
    batch_size=1,
    class_mode=None,
    shuffle=False,
)



## === cell 9
savedmodel_path = "/kaggle/input/effnet3/pp21_effnet_sub3"

candidate_endpoints = ["serving_default", "predict", "call"]
tfsm_layer = None
last_err = None
for ep in candidate_endpoints:
    try:
        tfsm_layer = keras.layers.TFSMLayer(savedmodel_path, call_endpoint=ep)
        call_endpoint_used = ep
        break
    except Exception as e:
        last_err = e

if tfsm_layer is None:
    raise RuntimeError(
        f"Could not load SavedModel via TFSMLayer from {savedmodel_path}. Last error: {last_err}"
    )

inp = keras.Input(shape=(224, 336, 3), dtype=tf.float32, name="image")
out = tfsm_layer(inp)
trained_model_sub = keras.Model(inp, out)

print("Loaded SavedModel as TFSMLayer with endpoint:", call_endpoint_used)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/918104532.py in <cell line: 0>()
     16 
     17 if tfsm_layer is None:
---> 18     raise RuntimeError(
     19         f"Could not load SavedModel via TFSMLayer from {savedmodel_path}. Last error: {last_err}"
     20     )

RuntimeError: Could not load SavedModel via TFSMLayer from /kaggle/input/effnet3/pp21_effnet_sub3. Last error: SavedModel file does not exist at: /kaggle/input/effnet3/pp21_effnet_sub3/{saved_model.pbtxt|saved_model.pb}

## === cell 10
y_pred = trained_model_sub.predict(test_generator, verbose=1)

y_pred = np.asarray(y_pred)
print("y_pred shape:", y_pred.shape)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3612217126.py in <cell line: 0>()
      1 # Predict
----> 2 y_pred = trained_model_sub.predict(test_generator, verbose=1)
      3 
      4 # Ensure numpy array
      5 y_pred = np.asarray(y_pred)

NameError: name 'trained_model_sub' is not defined

## === cell 11
predicted_class_indices = np.argmax(y_pred, axis=1)

labels = train_generator_sub.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[int(k)] for k in predicted_class_indices]

gen_files = list(test_generator.filenames)
gen_images = [os.path.basename(f) for f in gen_files]

sub = pd.DataFrame({"image": gen_images, "labels": predictions})
sub.head()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3736759496.py in <cell line: 0>()
      1 # Class index -> label string mapping
----> 2 predicted_class_indices = np.argmax(y_pred, axis=1)
      3 
      4 labels = train_generator_sub.class_indices
      5 labels = dict((v, k) for k, v in labels.items())

NameError: name 'y_pred' is not defined

## === cell 12
assert list(sub.columns) == ["image", "labels"]
assert sub["image"].nunique() == len(sub)
assert sub["labels"].isna().sum() == 0

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1830696170.py in <cell line: 0>()
      1 # Basic validation against sample submission format
----> 2 assert list(sub.columns) == ["image", "labels"]
      3 assert sub["image"].nunique() == len(sub)
      4 assert sub["labels"].isna().sum() == 0
      5 

NameError: name 'sub' is not defined
