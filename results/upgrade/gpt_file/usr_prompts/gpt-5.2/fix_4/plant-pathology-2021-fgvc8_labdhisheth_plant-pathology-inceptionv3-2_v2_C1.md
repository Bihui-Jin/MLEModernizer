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

0.8232502308402598

# 6. Current score

0.30564

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.30607) has done: 'I fix the initial import crash by removing/guarding optional visualization and `tensorflow_addons` imports that trigger the protobuf `MessageFactory.GetPrototype` error in this environment, while keeping TensorFlow/Keras intact. I also fix the `MultiLabelBinarizer` `NameError` by ensuring it is imported (and keep that preprocessing as-is). The missing external weights file is the main blocker and also the reason for the very low score; I keep the same InceptionV3-based architecture but switch to ImageNet pretrained weights (available in Keras) so inference is meaningful without external files. Finally, I make sure the submission CSV is written with the correct columns and label formatting.'
- What this solution (achieved 0.30534) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by avoiding the Kaggle environment’s incompatible protobuf runtime: set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow (this is a common workaround in this exact failure mode). Then I keep your model architecture and inference logic identical, but ensure class-to-index mapping matches the `MultiLabelBinarizer` class order so predictions map to the correct label strings (a logic bug that can severely depress F1). Finally, I make the label string output robust (strip extra spaces) and ensure `submission.csv` is always written with the required columns.'
- What this solution (achieved 0.30564) has done: 'I fix the TensorFlow/protobuf crash by moving the environment variable setup to the very top (before any TensorFlow-related import) and also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3`, which is the common stable workaround for this Kaggle-specific `MessageFactory.GetPrototype` failure. Then I keep your model architecture and inference logic identical, but I compile the model before `predict()` to avoid runtime issues in some TF builds and ensure consistent execution. Finally, I add a small guard to ensure the submission always has valid non-empty label strings and is written as `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import gc
import re
import math
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as L

from sklearn.preprocessing import MultiLabelBinarizer

np.random.seed(0)
tf.random.set_seed(0)

import warnings

warnings.filterwarnings("ignore")

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16

IMAGE_PATH = "../input/plant-pathology-2021-fgvc8/train_images/"
TRAIN_PATH = "../input/plant-pathology-2021-fgvc8/train.csv"
SUB_PATH = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"

sub = pd.read_csv(SUB_PATH)
test_data = sub.copy()
train_data = pd.read_csv(TRAIN_PATH)

train_data["labels"] = train_data["labels"].apply(lambda string: string.split(" "))
s = list(train_data["labels"])
mlb = MultiLabelBinarizer()
trainx = pd.DataFrame(
    mlb.fit_transform(s), columns=mlb.classes_, index=train_data.index
)
print("Train one-hot shape:", trainx.shape)
print("Classes:", list(mlb.classes_))

idx_to_label = {i: c for i, c in enumerate(mlb.classes_)}
print("Index->label mapping:", idx_to_label)




## === cell 2
def format_path(st):
    return "../input/plant-pathology-2021-fgvc8/test_images/" + str(st)


def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    return image, label


def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if label is None:
        return image
    return image, label


test_paths = test_data.image.apply(format_path).values

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

print("Test samples:", len(test_paths))



## === cell 3
inputs = tf.keras.Input(shape=(512, 512, 3))
x = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
)(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(6, activation="sigmoid")(x)
model = tf.keras.models.Model(inputs, outputs)

model.compile()

model.summary()



## === cell 4
preds = model.predict(test_dataset, verbose=1)



## === cell 5
preds = np.asarray(preds)

indices = []
for pred in preds:
    temp = []
    for j, category in enumerate(pred):
        if category >= 0.30:
            temp.append(j)
    if temp:
        indices.append(temp)
    else:
        indices.append([int(np.argmax(pred))])

testlabels = []
for image_idx_list in indices:
    temp = []
    for i in image_idx_list:
        temp.append(str(idx_to_label[int(i)]))
    lbl = " ".join(temp).strip()
    if lbl == "":
        lbl = str(idx_to_label[int(np.argmax(preds[len(testlabels)]))])
    testlabels.append(lbl)

assert len(testlabels) == len(sub), (len(testlabels), len(sub))

sub["labels"] = testlabels
sub = sub[["image", "labels"]]
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
