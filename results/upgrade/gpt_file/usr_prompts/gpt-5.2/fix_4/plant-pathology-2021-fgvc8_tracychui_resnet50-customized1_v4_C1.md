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

from PIL import Image as PILImage

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

K.set_image_data_format("channels_last")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

print("keras:", keras.__version__, "tf:", tf.__version__)



## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3
TOTAL_INPUTS = NR_CHANNELS * IMG_HEIGHT * IMG_WIDTH



## === cell 2
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"

sample_df = pd.read_csv(SAMPLE_SUB)
imglist_test = [os.path.join(TEST_IMG_DIR, fn) for fn in sample_df["image"].tolist()]
print("num test images:", len(imglist_test))
print("first test path:", imglist_test[0])



## === cell 3
import tensorflow.keras.applications.resnet50 as resnet
from tensorflow.keras.preprocessing import image

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16


def _read_decode_x(path):
    bytes_ = tf.io.read_file(path)
    img = tf.image.decode_jpeg(bytes_, channels=3)
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = resnet.preprocess_input(img)
    return img


test_paths = tf.constant(imglist_test)
ds_test = tf.data.Dataset.from_tensor_slices(test_paths)

_test_opts = tf.data.Options()
_test_opts.experimental_deterministic = True
ds_test = ds_test.with_options(_test_opts)

ds_test = (
    ds_test.map(_read_decode_x, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

Xim_test_sample = next(iter(ds_test.take(1))).numpy()
print("Xim_test_sample:", Xim_test_sample.shape, Xim_test_sample.dtype)



## === cell 4
print(
    "Sample pixel stats:",
    float(Xim_test_sample[0].min()),
    float(Xim_test_sample[0].max()),
    float(Xim_test_sample[0].mean()),
)



## === cell 5
train_df = pd.read_csv(TRAIN_CSV)

all_tags = " ".join(train_df["labels"].astype(str).tolist()).split()
tagnames = np.unique(np.array(all_tags, dtype=object))
num_classes = len(tagnames)
print("num_classes:", num_classes)
print("classes:", tagnames)

tag2idx = {t: i for i, t in enumerate(tagnames)}


def encode_labels(label_str: str) -> np.ndarray:
    y = np.zeros((num_classes,), dtype=np.float32)
    for t in str(label_str).split():
        if t in tag2idx:
            y[tag2idx[t]] = 1.0
    return y


train_img_paths = [os.path.join(TRAIN_IMG_DIR, fn) for fn in train_df["image"].tolist()]

Y = np.stack([encode_labels(s) for s in train_df["labels"].tolist()], axis=0)

idx = np.arange(len(train_df))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
val_frac = 0.1
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

print("train size:", len(trn_idx), "val size:", len(val_idx))


def _read_decode(path, y):
    bytes_ = tf.io.read_file(path)
    img = tf.image.decode_jpeg(bytes_, channels=3)
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = resnet.preprocess_input(img)
    return img, y


trn_paths = tf.constant([train_img_paths[i] for i in trn_idx])
trn_y = tf.constant(Y[trn_idx])
val_paths = tf.constant([train_img_paths[i] for i in val_idx])
val_y = tf.constant(Y[val_idx])

_trn_opts = tf.data.Options()
_trn_opts.experimental_deterministic = True

_val_opts = tf.data.Options()
_val_opts.experimental_deterministic = True

shuffle_buf = int(min(8192, max(2048, len(trn_idx))))
ds_trn = tf.data.Dataset.from_tensor_slices((trn_paths, trn_y))
ds_trn = ds_trn.with_options(_trn_opts)
ds_trn = ds_trn.shuffle(shuffle_buf, seed=SEED, reshuffle_each_iteration=True)
ds_trn = (
    ds_trn.map(_read_decode, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

ds_val = tf.data.Dataset.from_tensor_slices((val_paths, val_y))
ds_val = ds_val.with_options(_val_opts)
ds_val = (
    ds_val.map(_read_decode, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

base = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
    pooling="avg",
)
base.trainable = False  # keeps runtime under control; still legitimate and stable

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))
x = base(inputs, training=False)
outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
model_f = keras.Model(inputs, outputs)

model_f.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

EPOCHS = 2
history = model_f.fit(ds_trn, validation_data=ds_val, epochs=EPOCHS, verbose=2)

X_test = model_f.predict(ds_test, verbose=1)
print("X_test:", X_test.shape, X_test.dtype)



## === cell 6
print("X_test sample:", X_test[0])




## === cell 7
def class2tags(classes, tagnames):
    tagnames_arr = np.asarray(tagnames)
    idxs = [np.flatnonzero(row) for row in classes]
    return [" ".join(tagnames_arr[i]) for i in idxs]




## === cell 8
test_predclass = X_test > 0.28
test_predtags = class2tags(test_predclass, tagnames)

print("Example predicted tags:", test_predtags[0])



## === cell 9
del test_predclass
gc.collect()



## === cell 10
df1 = sample_df[["image"]].copy()
print(df1.head())



## === cell 11
df2 = pd.DataFrame(test_predtags, columns=["labels"])
print(df2.head())



## === cell 12
sub_df = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)

assert list(sub_df.columns) == ["image", "labels"]
assert len(sub_df) == len(sample_df)

sub_df["labels"] = sub_df["labels"].replace("", "healthy")

print(sub_df.head())

out_path = "./submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub_df))
print("Submission columns:", sub_df.columns.tolist())
