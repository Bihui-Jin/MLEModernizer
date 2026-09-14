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
import numpy as np
import pandas as pd



## === cell 1
import json
from PIL import Image

if hasattr(Image, "Resampling"):
    PIL_RESAMPLE = Image.Resampling.LANCZOS
else:
    PIL_RESAMPLE = Image.LANCZOS



## === cell 2
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"



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
PRE_TRAINED_MODEL = "../input/unionmodelv08/Cassava_Best_UnitedModel_V08.hdf5"



## === cell 7
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    RandomBrightnessContrast,
    CenterCrop,
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



## === cell 8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass



## === cell 9
from tensorflow.keras.utils import Sequence
import random


def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)

    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  # uint8
    img = tf.image.resize(
        img,
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.LANCZOS3,
        antialias=True,
    )  # float32
    return np.ascontiguousarray(img.numpy())


class AugmentedImageSequence(Sequence):
    def __init__(self, mode, data_set_type, x_set, y_set, batch_size, augmentations):
        self.mode = mode
        self.data_type = data_set_type
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]

        if self.mode == "TEST":
            batch_y = []
        else:
            batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        bs = len(batch_x)
        img_array = np.empty((bs, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)

        for i, x in enumerate(batch_x):
            if self.data_type == "TRAIN_DATA":
                random_num = random.uniform(0, 1)
                if random_num > 0.5:
                    img_data = self.augment(image=load_single_image(self.data_type, x))[
                        "image"
                    ]
                else:
                    img_data = (
                        load_single_image(self.data_type, x).astype(np.float32) / 255.0
                    )
            else:
                if self.data_type == "VALIDATE_DATA":
                    img_data = self.augment(image=load_single_image(self.data_type, x))[
                        "image"
                    ]
                else:
                    img_data = (
                        load_single_image(self.data_type, x).astype(np.float32) / 255.0
                    )

            img_data = img_data.astype(np.float32, copy=False)
            if img_data.max() > 1.5:
                img_data = img_data / 255.0
            img_array[i] = img_data

        return img_array, np.array(batch_y)




## === cell 10
test_filenames = os.listdir(TEST_DIR)
test_df = pd.DataFrame({"image_id": test_filenames})
test_samples = test_df.shape[0]
test_samples



## === cell 11
pass



## === cell 12
from tensorflow.keras.models import load_model


def build_fallback_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), n_classes=5):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = None
if os.path.exists(PRE_TRAINED_MODEL):
    model = load_model(PRE_TRAINED_MODEL)
    model.summary()
else:
    print(f"Pretrained model not found at: {PRE_TRAINED_MODEL}")
    print(
        "Training a small fallback CNN so we can generate a valid submission end-to-end."
    )
    train_csv_path = os.path.join(BASE_DIR, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    val_frac = 0.1
    n_val = int(len(train_df) * val_frac)
    val_df = train_df.iloc[:n_val].reset_index(drop=True)
    trn_df = train_df.iloc[n_val:].reset_index(drop=True)

    trn_gen = AugmentedImageSequence(
        "TRAIN",
        "TRAIN_DATA",
        trn_df["image_id"].values,
        trn_df["label"].values,
        batch_size=batch_size,
        augmentations=AUGMENTATIONS_TRAIN,
    )
    val_gen = AugmentedImageSequence(
        "VALIDATE",
        "VALIDATE_DATA",
        val_df["image_id"].values,
        val_df["label"].values,
        batch_size=batch_size,
        augmentations=AUGMENTATIONS_TEST,
    )

    model = build_fallback_model()
    model.fit(trn_gen, validation_data=val_gen, epochs=3, verbose=1)




## === cell 13
def old_predict():
    test_results_list = []
    for image_id in test_df["image_id"]:
        image_path = os.path.join(TEST_DIR, image_id)
        image_data = Image.open(image_path).convert("RGB")
        image_data = image_data.resize((IMG_HEIGHT, IMG_WIDTH), PIL_RESAMPLE)
        image_data = np.asarray(image_data).astype(np.float32) / 255.0
        image_data = np.expand_dims(image_data, axis=0)
        predict_class = model.predict(image_data, verbose=0)
        test_results_list.append(
            {"image_id": image_id, "label": int(np.argmax(predict_class))}
        )

    test_results_df = pd.DataFrame(test_results_list)
    test_results_df.to_csv("submission.csv", index=False)




## === cell 14
AUTOTUNE = tf.data.AUTOTUNE


_TEST_DIR_TF = tf.constant(TEST_DIR)
_IMG_SIZE_TF = tf.constant([IMG_HEIGHT, IMG_WIDTH], dtype=tf.int32)


def _make_test_ds(image_ids, batch):
    ids = tf.convert_to_tensor(image_ids, dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(ids)

    def _read_decode_resize(iid):
        path = tf.strings.join([_TEST_DIR_TF, iid])
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)  # uint8
        img = tf.image.resize(
            img,
            _IMG_SIZE_TF,
            method=tf.image.ResizeMethod.LANCZOS3,
            antialias=True,
        )  # float32
        img = img / 255.0
        return iid, img

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_read_decode_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


@tf.function(reduce_retracing=True)
def _rot90_batch_fast(x, k):
    k = tf.math.floormod(tf.cast(k, tf.int32), 4)

    r0 = x
    r1 = tf.image.rot90(x, k=1)
    r2 = tf.image.rot90(x, k=2)
    r3 = tf.image.rot90(x, k=3)

    stacked = tf.stack([r0, r1, r2, r3], axis=0)  # (4,N,H,W,3)
    n = tf.shape(x)[0]
    idx0 = k  # (N,)
    idx1 = tf.range(n, dtype=tf.int32)
    gather_idx = tf.stack([idx0, idx1], axis=1)  # (N,2)
    return tf.gather_nd(stacked, gather_idx)  # (N,H,W,3)


_PI_OVER_180 = tf.constant(np.pi / 180.0, dtype=tf.float32)
_ZEROS_I32 = tf.constant(0, dtype=tf.int32)
_C997 = tf.constant(997, tf.int32)
_C7919 = tf.constant(7919, tf.int32)


@tf.function(reduce_retracing=True)
def _tta_augment_5_stateless_fast(x, seed0, seed1):
    n = tf.shape(x)[0]
    x5 = tf.repeat(x, repeats=5, axis=0)  # (5N,H,W,3)

    idx = tf.range(n * 5, dtype=tf.int32)
    s0 = seed0 + idx * _C997
    s1 = seed1 + idx * _C7919
    seeds = tf.stack([s0, s1], axis=1)  # (5N,2)

    x5 = tf.image.stateless_random_flip_left_right(x5, seed=seeds)
    x5 = tf.image.stateless_random_flip_up_down(x5, seed=seeds[:, ::-1])

    k = tf.random.stateless_uniform(
        [n * 5], seed=seeds, minval=_ZEROS_I32, maxval=4, dtype=tf.int32
    )
    x5 = _rot90_batch_fast(x5, k)

    h = tf.cast(tf.shape(x5)[1], tf.float32)
    w = tf.cast(tf.shape(x5)[2], tf.float32)

    rot = tf.random.stateless_uniform(
        [n * 5], seed=seeds + 13, minval=-45.0, maxval=45.0, dtype=tf.float32
    )
    rot = rot * _PI_OVER_180

    tx = (
        tf.random.stateless_uniform(
            [n * 5], seed=seeds + 17, minval=-0.1, maxval=0.1, dtype=tf.float32
        )
        * w
    )
    ty = (
        tf.random.stateless_uniform(
            [n * 5], seed=seeds + 19, minval=-0.1, maxval=0.1, dtype=tf.float32
        )
        * h
    )

    zx = 1.0 + tf.random.stateless_uniform(
        [n * 5], seed=seeds + 23, minval=-0.4, maxval=0.4, dtype=tf.float32
    )
    zy = 1.0 + tf.random.stateless_uniform(
        [n * 5], seed=seeds + 29, minval=-0.4, maxval=0.4, dtype=tf.float32
    )

    shx = tf.random.stateless_uniform(
        [n * 5], seed=seeds + 31, minval=-0.1, maxval=0.1, dtype=tf.float32
    )
    shy = tf.random.stateless_uniform(
        [n * 5], seed=seeds + 37, minval=-0.1, maxval=0.1, dtype=tf.float32
    )

    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    cos_r = tf.cos(rot)
    sin_r = tf.sin(rot)

    a00 = 1.0 / zx
    a11 = 1.0 / zy
    a01 = tf.zeros_like(a00)
    a10 = tf.zeros_like(a00)

    a01 = a01 + shx
    a10 = a10 + shy

    r00 = cos_r
    r01 = -sin_r
    r10 = sin_r
    r11 = cos_r

    b00 = r00 * a00 + r01 * a10
    b01 = r00 * a01 + r01 * a11
    b10 = r10 * a00 + r11 * a10
    b11 = r10 * a01 + r11 * a11

    c0 = cx - b00 * cx - b01 * cy + tx
    c1 = cy - b10 * cx - b11 * cy + ty

    transforms = tf.stack(
        [b00, b01, c0, b10, b11, c1, tf.zeros_like(b00), tf.zeros_like(b00)], axis=1
    )

    x5 = tf.raw_ops.ImageProjectiveTransformV3(
        images=x5,
        transforms=transforms,
        output_shape=tf.stack([tf.shape(x5)[1], tf.shape(x5)[2]]),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return x5


@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec([None, IMG_HEIGHT, IMG_WIDTH, 3], tf.float32),
        tf.TensorSpec([], tf.int32),
        tf.TensorSpec([], tf.int32),
        tf.TensorSpec([], tf.int32),
    ],
)
def _predict_tta_mean_probs_chunked(x0, seed0, seed1, inner_bs):
    p0 = model(x0, training=False)  # (N,C)
    x5 = _tta_augment_5_stateless_fast(x0, seed0, seed1)  # (5N,H,W,3)

    total = tf.shape(x5)[0]
    p5_parts = tf.TensorArray(
        dtype=p0.dtype, size=0, dynamic_size=True, clear_after_read=False
    )
    i = tf.constant(0, tf.int32)
    j = tf.constant(0, tf.int32)

    def cond(i, j, p5_parts):
        return i < total

    def body(i, j, p5_parts):
        x_chunk = x5[i : tf.minimum(i + inner_bs, total)]
        p_chunk = model(x_chunk, training=False)
        p5_parts = p5_parts.write(j, p_chunk)
        return i + inner_bs, j + 1, p5_parts

    _, n_parts, p5_parts = tf.while_loop(
        cond, body, [i, j, p5_parts], parallel_iterations=1
    )
    p5 = p5_parts.concat()  # (5N,C)

    n = tf.shape(x0)[0]
    c = tf.shape(p0)[1]
    p5 = tf.reshape(p5, [5, n, c])  # (5,N,C)
    p5m = tf.reduce_mean(p5, axis=0)  # (N,C)
    return (p0 + 5.0 * p5m) / 6.0




## === cell 15
tta_k = 6  # 1 original + 5 augmented

PRED_GROUP = 128
INNER_PRED_BS = 64

image_ids_all = test_df["image_id"].values.astype(str)
test_ds = _make_test_ds(image_ids_all, batch=PRED_GROUP)

_dummy_x = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)
_ = _predict_tta_mean_probs_chunked(
    _dummy_x,
    tf.constant(12345, tf.int32),
    tf.constant(54321, tf.int32),
    tf.constant(INNER_PRED_BS, tf.int32),
)

all_ids = []
all_labels = []

for batch_index, (ids_tf, x0_tf) in enumerate(test_ds):
    seed0 = tf.constant(12345 + batch_index * PRED_GROUP, dtype=tf.int32)
    seed1 = tf.constant(54321 + batch_index * PRED_GROUP, dtype=tf.int32)

    probs = _predict_tta_mean_probs_chunked(
        x0_tf, seed0, seed1, tf.constant(INNER_PRED_BS, tf.int32)
    )  # (N,C)
    labels = tf.argmax(probs, axis=1, output_type=tf.int32).numpy()

    ids_np = ids_tf.numpy()
    if ids_np.dtype.kind in ("S", "O"):
        ids_np = np.char.decode(ids_np.astype("S"), "utf-8")
    else:
        ids_np = ids_np.astype(str)

    all_ids.append(ids_np)
    all_labels.append(labels)

all_ids = np.concatenate(all_ids, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

test_results_df = pd.DataFrame({"image_id": all_ids, "label": all_labels.astype(int)})

sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)
test_results_df = sample_sub[["image_id"]].merge(
    test_results_df, on="image_id", how="left"
)
test_results_df["label"] = test_results_df["label"].fillna(0).astype(int)

test_results_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_results_df.shape)
test_results_df.head()



## === cell 16
submission = pd.read_csv("submission.csv")
print(submission.head(3))
print(submission.columns)
print("Unique labels:", submission["label"].unique()[:10])
