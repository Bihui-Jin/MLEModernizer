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
import glob
import gc
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import tensorflow.keras.applications.resnet50 as resnet

K.set_image_data_format("channels_last")
print("keras:", keras.__version__, "tf:", tf.__version__)

os.environ.setdefault("PYTHONHASHSEED", "0")
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3

DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")




## === cell 2
imglist_test = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
len(imglist_test)




## === cell 3
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img,
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32)
    img = resnet.preprocess_input(
        img
    )  # expects 0-255 float32 RGB, converts to BGR + mean subtraction
    return img


options = tf.data.Options()
options.experimental_deterministic = False

TEST_BATCH = 64

test_ds = (
    tf.data.Dataset.from_tensor_slices(imglist_test)
    .with_options(options)
    .map(_decode_resize_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(TEST_BATCH, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

Xim_test_batch0 = next(iter(test_ds.take(1)))
print("Test batch0 shape:", tuple(Xim_test_batch0.shape))




## === cell 4
print("First pixel (preprocessed) of batch0[0]:", Xim_test_batch0[0, 0, 0, :])




## === cell 5
training_csv = pd.read_csv(TRAIN_CSV)

training_class = np.array([])
for labels in pd.unique(training_csv["labels"]):
    training_class = np.append(training_class, labels.split())
tagnames = np.unique(training_class)
NUM_CLASSES = len(tagnames)

print("Num classes:", NUM_CLASSES)
print("Classes:", tagnames)




## === cell 6
tag2idx = {t: i for i, t in enumerate(tagnames)}

labels_series = training_csv["labels"].astype(str).str.get_dummies(sep=" ")
y = labels_series.reindex(columns=tagnames, fill_value=0).to_numpy(dtype=np.float32)

MAX_TRAIN = 2048
train_df = training_csv.iloc[:MAX_TRAIN].reset_index(drop=True)
y_train = y[:MAX_TRAIN]

train_paths = [
    os.path.join(TRAIN_IMG_DIR, n) for n in train_df["image"].values.tolist()
]


def _train_map(p, yy):
    return _decode_resize_preprocess(p), yy


train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, y_train))
    .with_options(options)
    .map(_train_map, num_parallel_calls=AUTOTUNE)
    .cache()  # safe in-memory cache: only 2048 images resized to 300x300 float32
    .batch(16, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

print("Train size:", len(train_df), "Train y:", y_train.shape)




## === cell 7
base = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
)
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))
x = base(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
model_f = keras.Model(inputs, outputs)

base.trainable = False

model_f.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model_f.fit(
    train_ds,
    epochs=2,
    verbose=1,
)

X_test = model_f.predict(test_ds, verbose=1)
print(X_test.shape)




## === cell 8
print(
    "Pred stats: min/mean/max =",
    float(X_test.min()),
    float(X_test.mean()),
    float(X_test.max()),
)




## === cell 9
def class2tags(classes, tagnames):
    tagnames = np.asarray(tagnames)
    idx_rows = [np.flatnonzero(row) for row in classes]
    return [" ".join(tagnames[idx].tolist()) for idx in idx_rows]


test_predclass = X_test > 0.2
test_predtags = class2tags(test_predclass, tagnames)




## === cell 10
del test_predclass
gc.collect()




## === cell 11
sub = pd.read_csv(SAMPLE_SUB)
pred_map = {os.path.basename(p): tag for p, tag in zip(imglist_test, test_predtags)}

sub["labels"] = sub["image"].map(pred_map).fillna("")
sub.head()




## === cell 12
sub = sub[["image", "labels"]]
sub.to_csv("./submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
