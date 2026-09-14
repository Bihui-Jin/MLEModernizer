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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["PYTHONHASHSEED"] = "0"

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import random

random.seed(0)
np.random.seed(0)


def _import_tensorflow_with_protobuf_fallback():
    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except AttributeError as e:
        if "GetPrototype" not in str(e):
            raise
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
        import importlib
        import sys

        if "tensorflow" in sys.modules:
            del sys.modules["tensorflow"]
        tf = importlib.import_module("tensorflow")
        return tf


tf = _import_tensorflow_with_protobuf_fallback()

tf.random.set_seed(0)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## === cell 1
submission = pd.read_csv(
    "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv"
)
train = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")



## === cell 2
_ = train.head()



## === cell 3
_ = train.loc[:, "healthy":"scab"].sum(axis=0) / (train.shape[0] / 100)



## === cell 4
_ = test.head()



## === cell 5
_ = submission.head()



## === cell 6
IMG_SIZE = (224, 224)
IMG_DIR = "/kaggle/input/plant-pathology-2020-fgvc7/images"

train_ids = train["image_id"].astype(str).values
test_ids = test["image_id"].astype(str).values

train_paths = np.array([f"{IMG_DIR}/{iid}.jpg" for iid in train_ids], dtype=object)
test_paths = np.array([f"{IMG_DIR}/{iid}.jpg" for iid in test_ids], dtype=object)

missing = [
    p
    for p in (train_paths[:5].tolist() + test_paths[:5].tolist())
    if not os.path.exists(p)
]
if missing:
    raise FileNotFoundError(f"Could not find image(s), e.g.: {missing[0]}")


@tf.function
def _decode_resize_normalize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return img




## === cell 7
train_label = train.loc[:, "healthy":"scab"]



## === cell 8
train_label = np.asarray(train_label, dtype=np.float32)
train_label = train_label / np.clip(train_label.sum(axis=1, keepdims=True), 1.0, None)



## === cell 9
print("n_train:", len(train_paths))
print("n_test :", len(test_paths))
print("train_label shape:", train_label.shape)



## === cell 10
pass



## === cell 11
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint



## === cell 12
base_model = DenseNet201(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
)

model = Sequential()
model.add(base_model)

model.add(BatchNormalization())
model.add(Dropout(0.8))
model.add(Dense(128, activation="relu"))
model.add(Dense(4, activation="softmax"))

froze = True  # keep original behavior
if froze is True:
    base_model.trainable = False
else:
    for layer in base_model.layers[: -int(froze)]:
        layer.trainable = False

reduce_learning_rate = ReduceLROnPlateau(
    monitor="categorical_accuracy",
    factor=0.1,
    patience=2,
    cooldown=2,
    min_lr=0.0000001,
    verbose=1,
)
early_stopping = EarlyStopping(monitor="categorical_accuracy", patience=5)

check_point = ModelCheckpoint(
    filepath="resnet_50.h5", monitor="categorical_accuracy", save_best_only=True
)

callbacks = [reduce_learning_rate, early_stopping, check_point]

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)



## === cell 13
model.summary()



## === cell 14
BATCH_SIZE = 32
steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))

_rotation = tf.keras.layers.RandomRotation(
    factor=15.0 / 360.0, fill_mode="reflect", interpolation="bilinear", seed=0
)


def _stateless_rand_uniform(shape, seed, minval, maxval, dtype=tf.float32):
    return tf.random.stateless_uniform(
        shape=shape, seed=seed, minval=minval, maxval=maxval, dtype=dtype
    )


@tf.function
def _augment_tf(epoch, idx, img, label):
    idx = tf.cast(idx, tf.int32)
    epoch = tf.cast(epoch, tf.int32)

    base_seed = tf.stack([epoch, idx])  # (2,)

    flip_p = _stateless_rand_uniform(
        [], base_seed + tf.constant([1, 0], tf.int32), 0.0, 1.0
    )
    img = tf.cond(flip_p < 0.5, lambda: tf.image.flip_left_right(img), lambda: img)

    img = _rotation(img, training=True)

    zoom = _stateless_rand_uniform(
        [], base_seed + tf.constant([3, 0], tf.int32), 0.75, 1.25
    )
    new_h = tf.cast(tf.round(zoom * IMG_SIZE[0]), tf.int32)
    new_w = tf.cast(tf.round(zoom * IMG_SIZE[1]), tf.int32)
    img_zoom = tf.image.resize(img, [new_h, new_w], method="bilinear")
    img = tf.image.resize_with_crop_or_pad(img_zoom, IMG_SIZE[0], IMG_SIZE[1])

    max_dx = int(round(0.2 * IMG_SIZE[1]))
    max_dy = int(round(0.2 * IMG_SIZE[0]))
    dx = tf.cast(
        tf.round(
            _stateless_rand_uniform(
                [], base_seed + tf.constant([4, 0], tf.int32), -max_dx, max_dx
            )
        ),
        tf.int32,
    )
    dy = tf.cast(
        tf.round(
            _stateless_rand_uniform(
                [], base_seed + tf.constant([5, 0], tf.int32), -max_dy, max_dy
            )
        ),
        tf.int32,
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return img, label


options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.threading.private_threadpool_size = 8
except Exception:
    pass
try:
    options.threading.max_intra_op_parallelism = 0
    options.threading.max_inter_op_parallelism = 0
except Exception:
    pass

train_paths_tf = tf.constant(train_paths)
train_label_tf = tf.constant(train_label)
n = len(train_paths)

base_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths_tf, train_label_tf))
    .map(
        lambda p, y: (_decode_resize_normalize(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    .cache()
)

shuffle_buf = int(min(n, max(256, 8 * BATCH_SIZE)))

shuffled = base_ds.shuffle(
    buffer_size=shuffle_buf, seed=0, reshuffle_each_iteration=True
).repeat()

indexed = shuffled.enumerate()  # (idx, (img, label))


@tf.function
def _add_aug(step, idx, img, label):
    step = tf.cast(step, tf.int64)
    idx = tf.cast(idx, tf.int32)
    epoch = tf.cast(step // tf.cast(n, tf.int64), tf.int32)
    return _augment_tf(epoch, idx, img, label)


step_ds = tf.data.Dataset.counter(start=0, step=1, dtype=tf.int64).repeat()

train_ds = (
    tf.data.Dataset.zip((step_ds, indexed))
    .map(
        lambda step, pair: _add_aug(step, pair[0], pair[1][0], pair[1][1]),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(options)
)
train_ds = train_ds.apply(tf.data.experimental.ignore_errors())

start = dt.now()
history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=200,
    callbacks=callbacks,
    verbose=1,
)
print(
    "Время работы модели: {}. Количество эпох: {}.".format(
        dt.now() - start, len(history.epoch)
    )
)



## === cell 15
import gc

del train_label
gc.collect()




## === cell 16
def plot_loss(his, title):
    epoch = len(his.epoch)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history["loss"], label="train_loss")
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.close()


def plot_acc(his, title):
    epoch = len(his.epoch)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(
        np.arange(0, epoch),
        his.history.get("categorical_accuracy", []),
        label="categorical_accuracy",
    )
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    plt.close()




## === cell 17
if "history" in globals():
    plot_loss(history, "Training Dataset")
    plot_acc(history, "Training Dataset")



## === cell 18
try:
    if os.path.exists("resnet_50.h5"):
        model.load_weights("resnet_50.h5")
except Exception:
    pass

ds_test = (
    tf.data.Dataset.from_tensor_slices(tf.constant(test_paths))
    .map(_decode_resize_normalize, num_parallel_calls=AUTOTUNE)
    .batch(128)
    .prefetch(AUTOTUNE)
)
y_pred = model.predict(ds_test, verbose=1)
print(y_pred)



## === cell 19
target_cols = [c for c in submission.columns if c != "image_id"]
if y_pred.shape[1] != len(target_cols):
    raise ValueError(
        f"Prediction shape {y_pred.shape} does not match submission targets {len(target_cols)}: {target_cols}"
    )
submission.loc[:, target_cols] = y_pred



## === cell 20
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
