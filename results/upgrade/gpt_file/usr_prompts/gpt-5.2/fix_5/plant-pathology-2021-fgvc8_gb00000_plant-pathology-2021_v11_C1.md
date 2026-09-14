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
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)





## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

tf.random.set_seed(42)

print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## === cell 2
import cv2
import re
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
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
test_ids = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_df = pd.DataFrame(test_ids, columns=["image"])
test_df.head()




## === cell 7
all_labels = sorted(
    {lab for s in train["labels"].fillna("").values for lab in s.split(" ") if lab}
)
label2idx = {lab: i for i, lab in enumerate(all_labels)}
idx2label = {i: lab for lab, i in label2idx.items()}
num_classes = len(all_labels)

rows = []
cols = []
for i, s in enumerate(train["labels"].fillna("").values):
    if s:
        for lab in s.split(" "):
            j = label2idx.get(lab)
            if j is not None:
                rows.append(i)
                cols.append(j)

y = np.zeros((len(train), num_classes), dtype=np.float32)
if rows:
    y[np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)] = 1.0

y_cols = [f"y_{lab}" for lab in all_labels]

train_ml = train[["image"]].copy()
for j, col in enumerate(y_cols):
    train_ml[col] = y[:, j]

train_ml.head()




## === cell 8
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (224, 336)
BATCH_SIZE = 16

train_df, val_df = train_test_split(
    train_ml, test_size=0.15, random_state=42, shuffle=True
)


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    return img  # float32 later after preprocess_input


def _augment(img, seed):
    seed = tf.cast(seed, tf.int32)
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    angle = tf.random.stateless_uniform(
        [], seed=seed + 1, minval=-20.0, maxval=20.0, dtype=tf.float32
    ) * (np.pi / 180.0)

    tx = tf.random.stateless_uniform(
        [], seed=seed + 2, minval=-0.1, maxval=0.1, dtype=tf.float32
    )
    ty = tf.random.stateless_uniform(
        [], seed=seed + 3, minval=-0.1, maxval=0.1, dtype=tf.float32
    )

    return angle, tx, ty, img


_affine_layer = keras.layers.RandomRotation(
    factor=(0.0, 0.0)
)  # placeholder; we will use ImageProjectiveTransformV3 instead.


def _apply_affine(img, angle, tx, ty):
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)

    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)

    dx = tx * w
    dy = ty * h

    a0 = cos_a
    a1 = sin_a
    a2 = cx - cos_a * cx - sin_a * cy - dx
    b0 = -sin_a
    b1 = cos_a
    b2 = cy + sin_a * cx - cos_a * cy - dy

    transforms = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0], axis=0)[tf.newaxis, :]
    img4 = img[tf.newaxis, ...]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img4,
        transforms=transforms,
        output_shape=tf.shape(img)[0:2],
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )
    return out[0]


def _make_train_ds(df):
    paths = tf.constant([os.path.join(train_dir, f) for f in df["image"].values])
    labels = tf.constant(df[y_cols].values.astype(np.float32))
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(df), seed=42, reshuffle_each_iteration=True)

    def _map(path, y, idx):
        img = _decode_resize(path)
        angle, tx, ty, img = _augment(img, seed=tf.stack([42, tf.cast(idx, tf.int32)]))
        img = _apply_affine(img, angle, tx, ty)
        return img, y

    ds = ds.enumerate()
    ds = ds.map(lambda idx, xy: _map(xy[0], xy[1], idx), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds(df):
    paths = tf.constant([os.path.join(train_dir, f) for f in df["image"].values])
    labels = tf.constant(df[y_cols].values.astype(np.float32))
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (_decode_resize(p), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_generator = _make_train_ds(train_df)
val_generator = _make_val_ds(val_df)




## === cell 9
def _make_test_ds(df):
    paths = tf.constant([os.path.join(test_dir, f) for f in df["image"].values])
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_decode_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_generator = _make_test_ds(test_df)




## === cell 10
base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 336, 3),
    pooling="avg",
)
inp = keras.Input(shape=(224, 336, 3), name="image")
x = keras.applications.efficientnet.preprocess_input(inp)
x = base(x, training=False)
out = keras.layers.Dense(num_classes, activation="sigmoid")(x)
trained_model_sub = keras.Model(inp, out)

base.trainable = False
for layer in trained_model_sub.layers:
    if isinstance(layer, keras.layers.Dense):
        layer.trainable = True

trained_model_sub.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

trained_model_sub.summary()




## === cell 11
EPOCHS = 3
_ = trained_model_sub.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1,
)




## === cell 12
y_pred = trained_model_sub.predict(test_generator, verbose=1)
y_pred = np.asarray(y_pred)
print("y_pred shape:", y_pred.shape)

threshold = 0.5

pred_label_strs = []
for row in y_pred:
    inds = np.where(row >= threshold)[0].tolist()
    if len(inds) == 0:
        inds = [int(np.argmax(row))]
    labs = [idx2label[i] for i in inds]
    pred_label_strs.append(" ".join(labs))

gen_images = test_df["image"].tolist()

sub = pd.DataFrame({"image": gen_images, "labels": pred_label_strs})
sub.head()




## === cell 13
assert list(sub.columns) == ["image", "labels"]
assert sub["image"].nunique() == len(sub)
assert sub["labels"].isna().sum() == 0

sub = sub.set_index("image").reindex(sam_sub["image"]).reset_index()

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
