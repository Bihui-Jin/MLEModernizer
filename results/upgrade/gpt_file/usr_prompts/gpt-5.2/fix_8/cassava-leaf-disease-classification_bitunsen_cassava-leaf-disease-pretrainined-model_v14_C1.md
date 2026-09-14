# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"



## === cell 2
import matplotlib.pyplot as plt
import json
from PIL import Image

import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 4
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 5
input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
print(f"Number of train images: {len(input_files)}")



## === cell 6
IMG_HEIGHT = 300
IMG_WIDTH = 300
batch_size = 16
PRE_TRAINED_MODEL = "../input/unionmodelv03/Cassava_Best_UnitedModel_V03.hdf5"



## === cell 7
try:
    from albumentations import (
        Compose,
        HorizontalFlip,
        VerticalFlip,
        CenterCrop,
        RandomBrightnessContrast,
        ShiftScaleRotate,
        ToFloat,
    )

    AUGMENTATIONS_TRAIN = Compose(
        [
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            CenterCrop(p=1.0, height=IMG_HEIGHT, width=IMG_WIDTH),
            ShiftScaleRotate(
                p=0.5,
                shift_limit=0.0,
                scale_limit=(0.5, 1.50),
                rotate_limit=15,
                interpolation=0,
                border_mode=0,
            ),
            ToFloat(max_value=255.0),
        ]
    )
    AUGMENTATIONS_TEST = Compose([ToFloat(max_value=255.0)])
except Exception:
    AUGMENTATIONS_TRAIN = None
    AUGMENTATIONS_TEST = None



## === cell 8
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS

from functools import lru_cache


@lru_cache(maxsize=4096)
def _load_single_image_cached(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), RESAMPLE)
        arr = np.array(img)
    return arr


def load_single_image(data_type, image_id):
    return _load_single_image_cached(data_type, image_id)




## === cell 9
from tensorflow.keras.utils import Sequence
import threading
import queue


class _PrefetchSequence(Sequence):
    def __init__(self, base_seq, prefetch=1):
        self.base = base_seq
        self.prefetch = max(1, int(prefetch))
        self._queue = queue.Queue(maxsize=self.prefetch)
        self._stop = threading.Event()
        self._thread = None
        self._epoch = 0
        self._lock = threading.Lock()

    def __len__(self):
        return len(self.base)

    def on_epoch_end(self):
        if hasattr(self.base, "on_epoch_end"):
            self.base.on_epoch_end()
        with self._lock:
            self._epoch += 1
        self._restart_thread()

    def _restart_thread(self):
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            try:
                self._thread.join(timeout=0.2)
            except Exception:
                pass
        self._stop.clear()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        i = 0
        current_epoch = None
        while not self._stop.is_set():
            with self._lock:
                current_epoch = self._epoch
            if i >= len(self.base):
                self._stop.wait(0.05)
                continue
            if self._queue.full():
                self._stop.wait(0.01)
                continue
            item = self.base[i]
            self._queue.put((current_epoch, i, item))
            i += 1

    def __getitem__(self, idx):
        if self._thread is None:
            self._restart_thread()

        while True:
            ep, i, item = self._queue.get()
            with self._lock:
                cur_ep = self._epoch
            if ep != cur_ep:
                continue
            if i != idx:
                continue
            return item


class AugmentedImageSequence(Sequence):
    def __init__(self, mode, data_set_type, x_set, y_set, batch_size, augmentations):
        self.mode = mode
        self.data_type = data_set_type
        self.x = np.array(list(x_set))
        self.y = None if y_set is None else np.array(list(y_set))
        self.batch_size = batch_size
        self.augment = augmentations

        self._rng = np.random.RandomState(
            (SEED + (0 if self.mode == "TEST" else 123) + hash(self.data_type) % 10000)
            % (2**32 - 1)
        )

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]

        if self.mode == "TEST":
            batch_y = None
        else:
            batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        if self.data_type == "TRAIN_DATA":
            do_aug = self._rng.rand(len(batch_x)) > 0.5
        elif self.data_type == "VALIDATE_DATA":
            do_aug = np.ones(len(batch_x), dtype=bool)
        else:
            do_aug = np.zeros(len(batch_x), dtype=bool)

        imgs = []
        for image_id, aug_flag in zip(batch_x, do_aug):
            img = load_single_image(self.data_type, image_id)
            if aug_flag and self.augment is not None:
                img = self.augment(image=img)["image"]
            imgs.append(img)

        img_array = np.stack(imgs, axis=0).astype(np.float32, copy=False)

        if self.mode == "TEST":
            return img_array
        return img_array, batch_y.astype(np.int32, copy=False)




## === cell 10
test_filenames = sorted(os.listdir(TEST_DIR))
test_df = pd.DataFrame({"image_id": test_filenames})
test_samples = test_df.shape[0]
test_samples



## === cell 11
AUTOTUNE = tf.data.AUTOTUNE

TRAIN_DIR_TF = tf.constant(TRAIN_DIR)  # already ends with '/'
TEST_DIR_TF = tf.constant(TEST_DIR)  # already ends with '/'


def _build_path_from_id(image_id, is_test):
    base = TEST_DIR_TF if is_test else TRAIN_DIR_TF
    return tf.strings.join([base, image_id])


def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.LANCZOS3,
        antialias=True,
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def tfa_image_rotate(image, angle):
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    cy = (h - 1.0) / 2.0
    cx = (w - 1.0) / 2.0
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    a0 = cos_a
    a1 = -sin_a
    a2 = cx - cos_a * cx + sin_a * cy
    b0 = sin_a
    b1 = cos_a
    b2 = cy - sin_a * cx - cos_a * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])
    transform = tf.expand_dims(transform, 0)
    img4 = tf.expand_dims(image, 0)
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img4,
        transforms=transform,
        output_shape=tf.shape(image)[:2],
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=0.0,
    )
    return tf.squeeze(out, 0)


@tf.function
def _train_map(image_id, label):
    path = _build_path_from_id(image_id, is_test=False)
    img = _decode_resize(path)

    hash_id = tf.cast(tf.strings.to_hash_bucket_fast(image_id, 2**31 - 1), tf.int32)
    key = tf.stack([tf.constant(SEED, tf.int32), hash_id])
    rnd = tf.random.stateless_uniform([], seed=key, dtype=tf.float32)
    do_aug = rnd > 0.5

    def _aug_fn(x):
        x = tf.image.stateless_random_flip_left_right(
            x, seed=key + tf.constant([1, 0], tf.int32)
        )
        x = tf.image.stateless_random_flip_up_down(
            x, seed=key + tf.constant([2, 0], tf.int32)
        )
        x = tf.image.stateless_random_brightness(
            x, max_delta=0.2, seed=key + tf.constant([3, 0], tf.int32)
        )
        x = tf.image.stateless_random_contrast(
            x, lower=0.8, upper=1.2, seed=key + tf.constant([4, 0], tf.int32)
        )
        angle = tf.random.stateless_uniform(
            [], seed=key + tf.constant([5, 0], tf.int32), minval=-15.0, maxval=15.0
        ) * (np.pi / 180.0)
        x = tfa_image_rotate(x, angle)
        return tf.clip_by_value(x, 0.0, 1.0)

    img = tf.cond(do_aug, lambda: _aug_fn(img), lambda: img)
    return img, tf.cast(label, tf.int32)


@tf.function
def _val_map(image_id, label):
    path = _build_path_from_id(image_id, is_test=False)
    img = _decode_resize(path)
    return img, tf.cast(label, tf.int32)


@tf.function
def _test_map(image_id):
    path = _build_path_from_id(image_id, is_test=True)
    img = _decode_resize(path)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_df["image_id"].values)
test_ds = test_ds.map(_test_map, num_parallel_calls=AUTOTUNE, deterministic=True)
test_ds = test_ds.cache()
test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)



## === cell 12
train_csv_path = os.path.join(BASE_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)

train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
val_size = int(len(train_df) * val_frac)
val_df = train_df.iloc[:val_size].reset_index(drop=True)
trn_df = train_df.iloc[val_size:].reset_index(drop=True)

num_classes = 5

train_ds = tf.data.Dataset.from_tensor_slices(
    (trn_df["image_id"].values, trn_df["label"].values)
)
train_ds = train_ds.shuffle(
    buffer_size=len(trn_df), seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.map(_train_map, num_parallel_calls=AUTOTUNE, deterministic=True)
train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_df["image_id"].values, val_df["label"].values)
)
val_ds = val_ds.map(_val_map, num_parallel_calls=AUTOTUNE, deterministic=True)
val_ds = val_ds.cache()
val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)



## === cell 13
from tensorflow.keras import layers, models

base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base.trainable = False  # keep fast; avoids large training cost

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=False,
)

model.summary()



## === cell 14
EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)

pred_probs = model.predict(
    test_ds,
    verbose=1,
)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)

sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

pred_map = dict(zip(test_df["image_id"].values, pred_labels))
sample_sub["label"] = sample_sub["image_id"].map(pred_map)

if sample_sub["label"].isna().any():
    fallback = int(train_df["label"].mode().iloc[0])
    sample_sub["label"] = sample_sub["label"].fillna(fallback)

sample_sub["label"] = sample_sub["label"].astype(int)
sample_sub.to_csv("submission.csv", index=False)
print(sample_sub.head(3))
print("Wrote submission.csv with shape:", sample_sub.shape)
