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
import json
import random
import numpy as np
import pandas as pd

from PIL import Image

import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")
MAP_JSON = os.path.join(BASE_DIR, "label_num_to_disease_map.json")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing: {SAMPLE_SUB_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"




## === cell 2
with open(MAP_JSON, "r") as f:
    map_classes = json.load(f)

print(json.dumps(map_classes, indent=2))
label_list = sorted([int(k) for k in map_classes.keys()])
NUM_CLASSES = len(label_list)
print("NUM_CLASSES:", NUM_CLASSES)




## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

assert list(sample_sub.columns) == [
    "image_id",
    "label",
], "Unexpected submission columns"
assert set(train_df.columns) == {"image_id", "label"}, "Unexpected train.csv columns"

print("Train rows:", len(train_df), "Test rows:", len(sample_sub))




## === cell 4
IMG_HEIGHT = 400
IMG_WIDTH = 400
BATCH_SIZE = 32

_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def load_single_image_from_dir(image_dir, image_id):
    image_path = os.path.join(image_dir, image_id)
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im = im.resize((IMG_WIDTH, IMG_HEIGHT), _RESAMPLE)
        arr = np.asarray(im, dtype=np.float32)
    return arr




## === cell 5
try:
    from albumentations import (
        Compose,
        HorizontalFlip,
        RandomBrightness,
        RandomContrast,
        ToFloat,
        ShiftScaleRotate,
        CenterCrop,
    )

    _ALBU_OK = True
except Exception as e:
    print(
        "Albumentations not available or failed to import; falling back to no-aug. Error:",
        repr(e),
    )
    _ALBU_OK = False

if _ALBU_OK:
    AUGMENTATIONS_TRAIN = Compose(
        [
            HorizontalFlip(p=0.5),
            RandomContrast(limit=0.2, p=0.5),
            RandomBrightness(limit=0.2, p=0.5),
            CenterCrop(p=1.0, height=IMG_HEIGHT, width=IMG_WIDTH),
            ShiftScaleRotate(
                p=0.5,
                shift_limit=0,
                scale_limit=(0.5, 1.50),
                rotate_limit=15,
                interpolation=0,
                border_mode=0,
            ),
            ToFloat(max_value=255.0),
        ]
    )
    AUGMENTATIONS_TEST = Compose([ToFloat(max_value=255.0)])
else:
    AUGMENTATIONS_TRAIN = None
    AUGMENTATIONS_TEST = None




## === cell 6
def _tf_load_and_resize(image_dir, image_id):
    image_path = tf.strings.join([image_dir, tf.constant("/", tf.string), image_id])
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.LANCZOS3
    )
    img = tf.cast(img, tf.float32)  # 0..255 float32
    return img


def _np_albu_apply(image_f32_0_255, do_apply, is_train):
    img = np.asarray(image_f32_0_255, dtype=np.float32)
    if is_train:
        if AUGMENTATIONS_TRAIN is not None and bool(do_apply):
            out = AUGMENTATIONS_TRAIN(image=img)["image"]
        else:
            out = img / 255.0
    else:
        if AUGMENTATIONS_TEST is not None:
            out = AUGMENTATIONS_TEST(image=img)["image"]
        else:
            out = img / 255.0
    return np.asarray(out, dtype=np.float32)


def _tf_apply_train_aug(img_f32_0_255, label):
    if AUGMENTATIONS_TRAIN is None:
        return (img_f32_0_255 / 255.0), label

    do_apply = tf.random.uniform((), seed=SEED) > 0.5
    out = tf.numpy_function(
        func=_np_albu_apply,
        inp=[img_f32_0_255, do_apply, True],
        Tout=tf.float32,
    )
    out.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    return out, label


def _tf_apply_eval_aug(img_f32_0_255, label=None):
    if AUGMENTATIONS_TEST is None:
        out = img_f32_0_255 / 255.0
    else:
        out = tf.numpy_function(
            func=_np_albu_apply,
            inp=[img_f32_0_255, False, False],
            Tout=tf.float32,
        )
        out.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    if label is None:
        return out
    return out, label


def make_train_ds(image_ids, labels, batch_size):
    image_dir_const = tf.constant(TRAIN_DIR, tf.string)

    ds = tf.data.Dataset.from_tensor_slices((image_ids, labels))
    ds = ds.shuffle(
        buffer_size=len(image_ids), seed=SEED, reshuffle_each_iteration=True
    )

    ds = ds.map(
        lambda iid, y: (_tf_load_and_resize(image_dir_const, iid), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.cache()

    ds = ds.map(_tf_apply_train_aug, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(image_ids, labels, batch_size):
    image_dir_const = tf.constant(TRAIN_DIR, tf.string)

    ds = tf.data.Dataset.from_tensor_slices((image_ids, labels))
    ds = ds.map(
        lambda iid, y: (_tf_load_and_resize(image_dir_const, iid), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.cache()
    ds = ds.map(
        lambda x, y: _tf_apply_eval_aug(x, y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(image_ids, batch_size):
    image_dir_const = tf.constant(TEST_DIR, tf.string)

    ds = tf.data.Dataset.from_tensor_slices(image_ids)
    ds = ds.map(
        lambda iid: _tf_load_and_resize(image_dir_const, iid),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.cache()
    ds = ds.map(_tf_apply_eval_aug, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 7
from tensorflow.keras import layers, models

base_model = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    pooling="avg",
)
base_model.trainable = False  # phase 1: fast/stable

inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = tf.keras.applications.xception.preprocess_input(inputs * 255.0)
x = base_model(x, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 8
def stratified_split_indices(y, test_size=0.1, seed=SEED):
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    train_idx = []
    val_idx = []
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_val = max(1, int(round(len(idx_c) * test_size)))
        val_idx.append(idx_c[:n_val])
        train_idx.append(idx_c[n_val:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


train_idx, val_idx = stratified_split_indices(
    train_df["label"].values, test_size=0.1, seed=SEED
)

tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

train_ds = make_train_ds(
    tr_df["image_id"].values.astype(str),
    tr_df["label"].values.astype(np.int64),
    BATCH_SIZE,
)
val_ds = make_val_ds(
    va_df["image_id"].values.astype(str),
    va_df["label"].values.astype(np.int64),
    BATCH_SIZE,
)




## === cell 9
EPOCHS = 2
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

FINE_TUNE_EPOCHS = 1
UNFREEZE_LAST_N_LAYERS = 30

base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAST_N_LAYERS]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS, verbose=1
)




## === cell 10
test_ds = make_test_ds(sample_sub["image_id"].values.astype(str), BATCH_SIZE)

probs = model.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

submission = pd.DataFrame({"image_id": sample_sub["image_id"].values, "label": preds})

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Submission shape:", submission.shape)
assert os.path.exists(out_path) and out_path.endswith(".csv")
assert list(submission.columns) == ["image_id", "label"]
assert len(submission) == len(sample_sub)
assert submission["image_id"].iloc[0] == sample_sub["image_id"].iloc[0]
