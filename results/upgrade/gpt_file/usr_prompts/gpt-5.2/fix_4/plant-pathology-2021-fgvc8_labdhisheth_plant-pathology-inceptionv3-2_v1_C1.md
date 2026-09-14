# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import gc
import re

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

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

AUTO = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 32

IMAGE_PATH = "../input/plant-pathology-2021-fgvc8/train_images/"
TRAIN_PATH = "../input/plant-pathology-2021-fgvc8/train.csv"
SUB_PATH = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"

TEST_IMAGE_DIR = "../input/plant-pathology-2021-fgvc8/test_images/"

sub = pd.read_csv(SUB_PATH)
test_data = sub.copy()
train_data = pd.read_csv(TRAIN_PATH)

train_data["labels"] = (
    train_data["labels"].astype(str).apply(lambda string: string.split(" "))
)
s = list(train_data["labels"])
mlb = MultiLabelBinarizer()
_ = mlb.fit_transform(s)

print("Loaded train rows:", len(train_data), "test rows:", len(test_data))
print(
    "Classes from train (mlb):",
    list(mlb.classes_)[:10],
    "..." if len(mlb.classes_) > 10 else "",
)




## === cell 1
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


test_paths = (TEST_IMAGE_DIR + test_data["image"].astype(str)).to_numpy()

options = tf.data.Options()
options.experimental_deterministic = True

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 2
labels = {
    0: "complex",
    1: "frog_eye_leaf_spot",
    2: "healthy",
    3: "powdery_mildew",
    4: "rust",
    5: "scab",
}
labels



## === cell 3
inputs = tf.keras.Input(shape=(512, 512, 3))
x = tf.keras.applications.InceptionV3(include_top=False, weights=None)(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(6, activation="sigmoid")(x)
model = tf.keras.models.Model(inputs, outputs)

weights_path = "../input/trail-1-dataset/Inceptionv3.h5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded custom weights:", weights_path)
else:
    print("Custom weights not found. Falling back to ImageNet backbone weights.")
    backbone = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet", input_tensor=inputs
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    outputs = tf.keras.layers.Dense(6, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs, outputs)

model.summary()



## === cell 4
preds = model.predict(test_dataset, verbose=0)

threshold = 0.25
preds_np = np.asarray(preds)

mask = preds_np >= threshold  # [N,6] boolean
has_any = mask.any(axis=1)
argmax_idx = preds_np.argmax(axis=1)

rows = np.arange(preds_np.shape[0])
mask2 = mask.copy()
mask2[~has_any, :] = False
mask2[~has_any, argmax_idx[~has_any]] = True

label_list = [labels[i] for i in range(6)]
testlabels = [
    " ".join(np.array(label_list, dtype=object)[mask2[i]].tolist())
    for i in range(mask2.shape[0])
]

sub["labels"] = testlabels
sub.to_csv("submission.csv", index=False)
sub.head()
