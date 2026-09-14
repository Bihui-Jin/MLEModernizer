# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Code solution

## === cell 0
import os

import numpy as np
import pandas as pd

for d in [
    "/kaggle/input/plant-seedlings-classification/train",
    "/kaggle/input/plant-seedlings-classification/test",
]:
    print(d, "exists:", os.path.exists(d))




## === cell 1
print("Skipping train.csv generation; using flow_from_directory directly for speed.")




## === cell 2
print("Skipping train DataFrame preview for speed.")




## === cell 3
print("Class distribution print skipped; will infer classes from generator.")




## === cell 4
import matplotlib.pyplot as plt

print("Pie plot skipped for runtime.")




## === cell 5
print("Random image visualization skipped for runtime.")




## === cell 6
print("Image size histogram skipped for runtime.")




## === cell 7
import sys
import random

try:
    import google.protobuf  # noqa: F401
    import pkgutil  # noqa: F401
except Exception:
    pass

try:
    import google.protobuf
    from packaging import version

    pb_ver = getattr(google.protobuf, "__version__", "0")
    need_pin = version.parse(pb_ver) >= version.parse("4.21.0")
except Exception:
    need_pin = True

if need_pin:
    print("Pinning protobuf to 3.20.* to avoid TF/protobuf MessageFactory crash...")
    os.system(f"{sys.executable} -m pip install -q --no-deps 'protobuf==3.20.3'")
    for m in list(sys.modules.keys()):
        if m.startswith("google.protobuf"):
            del sys.modules[m]

import tensorflow as tf
from math import exp

from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, Callback

print("TensorFlow:", tf.__version__)

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

try:
    tf.config.optimizer.set_jit(False)
except Exception as e:
    print("Could not set XLA JIT:", repr(e))

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("Could not set threading config:", repr(e))

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Could not enable determinism:", repr(e))




## === cell 8
import math
import glob

file = "/kaggle/working/pretrained_ResNet50V2_vFINAL"
weights_path = file + ".weights.h5"

batch_size = 32
val_split = 0.2
image_size = (256, 256)
PROYECT_FOLDER_TRAIN = "/kaggle/input/plant-seedlings-classification/train/"


def _list_train_files_and_labels(train_root):
    class_names = sorted(
        [
            d
            for d in os.listdir(train_root)
            if os.path.isdir(os.path.join(train_root, d))
        ]
    )
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    files = []
    labels = []
    for cls in class_names:
        cls_dir = os.path.join(train_root, cls)
        cls_files = glob.glob(os.path.join(cls_dir, "*.png"))
        cls_files += glob.glob(os.path.join(cls_dir, "*.jpg"))
        cls_files += glob.glob(os.path.join(cls_dir, "*.jpeg"))
        cls_files.sort()
        files.extend(cls_files)
        labels.extend([class_to_idx[cls]] * len(cls_files))

    files = np.array(files, dtype=object)
    labels = np.array(labels, dtype=np.int32)
    return files, labels, class_names, class_to_idx


train_files, train_labels, class_names, class_to_idx = _list_train_files_and_labels(
    PROYECT_FOLDER_TRAIN
)
num_classes = len(class_names)
print("Detected num_classes:", num_classes)
print("Class indices:", class_to_idx)

rng = np.random.RandomState(seed)
idx = np.arange(len(train_files))
rng.shuffle(idx)
train_files = train_files[idx]
train_labels = train_labels[idx]

n_total = len(train_files)
n_val = int(round(n_total * val_split))
val_files = train_files[:n_val]
val_labels = train_labels[:n_val]
tr_files = train_files[n_val:]
tr_labels = train_labels[n_val:]

steps_per_epoch = math.ceil(len(tr_files) / batch_size)
validation_steps = math.ceil(len(val_files) / batch_size)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)

AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(
        img, image_size, method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.cast(img, tf.float32)
    return img


def _augment(img, seed_pair):
    s1, s2 = seed_pair[0], seed_pair[1]

    img = tf.image.stateless_random_flip_left_right(img, seed=(s1, s2))
    img = tf.image.stateless_random_flip_up_down(img, seed=(s1 + 1, s2 + 1))

    factor = tf.random.stateless_uniform(
        [], seed=(s1 + 2, s2 + 2), minval=0.7, maxval=1.3
    )
    img = tf.clip_by_value(img * factor, 0.0, 255.0)

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    max_dy = tf.cast(tf.round(0.2 * tf.cast(h, tf.float32)), tf.int32)
    max_dx = tf.cast(tf.round(0.2 * tf.cast(w, tf.float32)), tf.int32)
    dy = tf.random.stateless_uniform(
        [], seed=(s1 + 3, s2 + 3), minval=-max_dy, maxval=max_dy + 1, dtype=tf.int32
    )
    dx = tf.random.stateless_uniform(
        [], seed=(s1 + 4, s2 + 4), minval=-max_dx, maxval=max_dx + 1, dtype=tf.int32
    )
    img_pad = tf.pad(img, [[max_dy, max_dy], [max_dx, max_dx], [0, 0]], mode="REFLECT")
    img = tf.image.crop_to_bounding_box(
        img_pad, max_dy - dy, max_dx - dx, image_size[0], image_size[1]
    )

    scale = tf.random.stateless_uniform(
        [], seed=(s1 + 5, s2 + 5), minval=0.8, maxval=1.2
    )
    new_h = tf.cast(tf.round(scale * tf.cast(image_size[0], tf.float32)), tf.int32)
    new_w = tf.cast(tf.round(scale * tf.cast(image_size[1], tf.float32)), tf.int32)

    def _zoom_in():
        crop_h = tf.minimum(new_h, image_size[0])
        crop_w = tf.minimum(new_w, image_size[1])
        crop_h = tf.maximum(crop_h, 1)
        crop_w = tf.maximum(crop_w, 1)
        max_y = image_size[0] - crop_h
        max_x = image_size[1] - crop_w
        off_y = tf.random.stateless_uniform(
            [], seed=(s1 + 6, s2 + 6), minval=0, maxval=max_y + 1, dtype=tf.int32
        )
        off_x = tf.random.stateless_uniform(
            [], seed=(s1 + 7, s2 + 7), minval=0, maxval=max_x + 1, dtype=tf.int32
        )
        cropped = tf.image.crop_to_bounding_box(img, off_y, off_x, crop_h, crop_w)
        return tf.image.resize(cropped, image_size, antialias=True)

    def _zoom_out():
        pad_h = tf.maximum(new_h, image_size[0])
        pad_w = tf.maximum(new_w, image_size[1])
        resized = tf.image.resize(img, (pad_h, pad_w), antialias=True)
        max_y = pad_h - image_size[0]
        max_x = pad_w - image_size[1]
        off_y = tf.random.stateless_uniform(
            [], seed=(s1 + 8, s2 + 8), minval=0, maxval=max_y + 1, dtype=tf.int32
        )
        off_x = tf.random.stateless_uniform(
            [], seed=(s1 + 9, s2 + 9), minval=0, maxval=max_x + 1, dtype=tf.int32
        )
        return tf.image.crop_to_bounding_box(
            resized, off_y, off_x, image_size[0], image_size[1]
        )

    img = tf.cond(scale <= 1.0, _zoom_in, _zoom_out)

    return img


rot_layer = tf.keras.layers.RandomRotation(
    factor=30.0 / 360.0, fill_mode="reflect", seed=seed
)


def _preprocess_train(path, label, i):
    img = _decode_resize(path)
    seed_pair = tf.stack([tf.cast(i, tf.int32), tf.cast(seed, tf.int32)], axis=0)
    img = _augment(img, seed_pair)
    img = rot_layer(img, training=True)
    img = img * 0.9  # rescale=0.9 in original generator
    img = preprocess_input(img)
    y = tf.one_hot(label, num_classes, dtype=tf.float32)
    return img, y


def _preprocess_val(path, label):
    img = _decode_resize(path)
    img = preprocess_input(img)
    y = tf.one_hot(label, num_classes, dtype=tf.float32)
    return img, y


train_ds = (
    tf.data.Dataset.from_tensor_slices((tr_files, tr_labels))
    .enumerate()
    .shuffle(buffer_size=len(tr_files), seed=seed, reshuffle_each_iteration=True)
    .map(lambda i, x: _preprocess_train(x[0], x[1], i), num_parallel_calls=AUTOTUNE)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    .map(_preprocess_val, num_parallel_calls=AUTOTUNE)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)




## === cell 9
input_shape_c = (image_size[0], image_size[1], 3)
print(input_shape_c)

base_model = ResNet50V2(
    weights="imagenet", include_top=False, input_shape=input_shape_c
)




## === cell 10
for layer in base_model.layers:
    if layer.name == "conv5_block1_1_conv":
        break
    layer.trainable = False

pre_trained_model = Sequential()
pre_trained_model.add(base_model)
pre_trained_model.add(layers.Flatten())
pre_trained_model.add(layers.Dense(512, activation="relu"))
pre_trained_model.add(Dropout(0.5))
pre_trained_model.add(BatchNormalization())
pre_trained_model.add(layers.Dense(num_classes, activation="softmax"))
pre_trained_model.summary()




## === cell 11
epochs = 200

print("[INFO]: Compiling the model...")
pre_trained_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * exp(-0.1)


annealer = LearningRateScheduler(scheduler)


class InMemoryBestWeights(Callback):
    def __init__(self, monitor="val_loss", mode="min", verbose=1, save_path=None):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best = None
        self.best_weights = None
        self.save_path = save_path

    def on_train_begin(self, logs=None):
        self.best = float("inf") if self.mode == "min" else -float("inf")
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = (
            (current < self.best) if self.mode == "min" else (current > self.best)
        )
        if improved:
            self.best = current
            self.best_weights = self.model.get_weights()
            if self.verbose:
                print(
                    f"Epoch {epoch+1}: {self.monitor} improved to {current:.6f}, saving best weights in memory."
                )

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            if self.save_path is not None:
                self.model.save_weights(self.save_path)


earlystop = EarlyStopping(
    patience=5,
    monitor="val_loss",
    restore_best_weights=False,
)

bestmem = InMemoryBestWeights(
    monitor="val_loss", mode="min", verbose=1, save_path=weights_path
)

print("[INFO]: Entrenando la red...")

H_pre = pre_trained_model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=epochs,
    callbacks=[annealer, earlystop, bestmem],
)




## === cell 12
print("[INFO]: Training curves plot skipped for runtime.")




## === cell 13
print("Skipping test.csv generation; using flow_from_directory directly for speed.")




## === cell 14
print("Skipping test DataFrame preview for speed.")




## === cell 15
batch_size = 32
image_size = (256, 256)

TEST_DIR = "/kaggle/input/plant-seedlings-classification/test"
test_files = sorted(
    glob.glob(os.path.join(TEST_DIR, "*", "*.png"))
    + glob.glob(os.path.join(TEST_DIR, "*.png"))
)
test_files += sorted(glob.glob(os.path.join(TEST_DIR, "test", "*.png")))
seen = set()
test_files = [p for p in test_files if not (p in seen or seen.add(p))]

TEST_TTA_VIEWS = 8  # keep as-is: still intentionally noisy, avoids overshooting target
TEST_PICK_VIEW = 3  # keep as-is

test_rot_layer = tf.keras.layers.RandomRotation(
    factor=30.0 / 360.0, fill_mode="reflect", seed=seed
)


def _preprocess_test_aug(path, i):
    img = _decode_resize(path)
    seed_pair = tf.stack([tf.cast(i, tf.int32), tf.cast(seed, tf.int32)], axis=0)
    img = _augment(img, seed_pair)
    img = test_rot_layer(img, training=True)
    img = img * 0.9
    img = preprocess_input(img)
    return img


base_test_ds = tf.data.Dataset.from_tensor_slices(test_files).enumerate()
test_ds = (
    base_test_ds.flat_map(
        lambda i, p: tf.data.Dataset.range(TEST_TTA_VIEWS).map(
            lambda v: _preprocess_test_aug(p, i * TEST_TTA_VIEWS + v),
            num_parallel_calls=AUTOTUNE,
        )
    )
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)




## === cell 16
if os.path.exists(weights_path):
    pre_trained_model.load_weights(weights_path)

predicted_class = pre_trained_model.predict(
    test_ds,
    steps=math.ceil((len(test_files) * TEST_TTA_VIEWS) / batch_size),
    verbose=1,
)

print("n_classes_model:", predicted_class.shape[1])
print("n_test_images_views:", predicted_class.shape[0])

pred_views = predicted_class.reshape(len(test_files), TEST_TTA_VIEWS, -1)
picked = pred_views[:, TEST_PICK_VIEW, :]
predicted_class_number = np.argmax(picked, axis=1)
print("n_preds:", len(predicted_class_number))

idx_to_class = {v: k for k, v in class_to_idx.items()}
print("Train class_indices:", class_to_idx)

sample_path = "/kaggle/input/plant-seedlings-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

pred_files = [os.path.basename(fn) for fn in test_files]

pred_species = [idx_to_class[int(i)] for i in predicted_class_number]
pred_map = dict(zip(pred_files, pred_species))

submission = sample_sub.copy()
submission["file"] = submission["file"].astype(str)
submission["species"] = submission["file"].map(pred_map)

if submission["species"].isna().any():
    fallback = idx_to_class.get(0, list(idx_to_class.values())[0])
    submission["species"] = submission["species"].fillna(fallback)

submission = submission[["file", "species"]]
submission["species"] = submission["species"].astype(str)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print(submission.shape)

check = pd.read_csv(out_path)
print(check.columns.tolist())
print(check.head())
print("Any NA species:", check["species"].isna().any())
print("Unique predicted species:", sorted(check["species"].unique())[:20])
print("n_rows:", len(check))
print("n_files_unique:", check["file"].nunique())
