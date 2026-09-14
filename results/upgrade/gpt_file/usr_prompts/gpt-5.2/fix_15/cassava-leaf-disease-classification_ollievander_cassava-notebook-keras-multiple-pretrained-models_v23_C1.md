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

INPUT_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train_images")
TEST_PATH = os.path.join(INPUT_DIR, "test_images")



## === cell 1
import json
import numpy as np
import pandas as pd
import tensorflow as tf

import albumentations as A
import cv2

import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models

SEED = 100
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 2
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
train.head()



## === cell 3
with open(os.path.join(INPUT_DIR, "label_num_to_disease_map.json")) as f:
    classes = json.load(f)
classes



## === cell 4
if False:
    train["class"] = train["label"].map(lambda x: classes[str(x)])
    train[["image_id", "label", "class"]].head()



## === cell 5
if False:
    plt.figure(figsize=(15, 7))
    sns.countplot(x=train["class"], order=train["class"].value_counts().index)
    plt.tight_layout()
    plt.show()



## === cell 6
train_df_full = train.copy()
train_df_full["label"] = train_df_full["label"].astype(str)

val_frac = 0.05
rng = np.random.RandomState(SEED)

val_parts = []
train_parts = []
for lab, grp in train_df_full.groupby("label", sort=True):
    idx = grp.index.values.copy()
    rng.shuffle(idx)
    n_val = max(1, int(round(len(idx) * val_frac)))
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]
    val_parts.append(train_df_full.loc[val_idx])
    train_parts.append(train_df_full.loc[trn_idx])

train_df = (
    pd.concat(train_parts, axis=0)
    .sample(frac=1.0, random_state=SEED)
    .reset_index(drop=True)
)
val_df = (
    pd.concat(val_parts, axis=0)
    .sample(frac=1.0, random_state=SEED)
    .reset_index(drop=True)
)

train_df.head()



## === cell 7
batch_size = 4

_train_aug = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=40, p=0.5),
        A.Transpose(p=0.5),
    ]
)

NUM_CLASSES = 5

TFREC_TRAIN_DIR = os.path.join(INPUT_DIR, "train_tfrecords")
TFREC_TRAIN_FILES = tf.io.gfile.glob(os.path.join(TFREC_TRAIN_DIR, "*.tfrec"))
TFREC_TRAIN_FILES = sorted(TFREC_TRAIN_FILES)

_feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_tfrecord_to_xy(example):
    ex = tf.io.parse_single_example(example, _feature_description)
    img = tf.image.decode_jpeg(ex["image"], channels=3)  # uint8
    img = tf.image.resize(img, [512, 512], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32) / 255.0
    label = tf.cast(ex["target"], tf.int32)
    y = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
    return img, y


def _albumentations_np(img):
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    out = _train_aug(image=img_u8)["image"]
    out = out.astype(np.float32) / 255.0
    return out


def _train_map_decoded(img, y):
    img = tf.numpy_function(_albumentations_np, [img], Tout=tf.float32)
    img.set_shape([512, 512, 3])
    return img, y


def _val_map_decoded(img, y):
    return img, y


_train_opts = tf.data.Options()
_train_opts.deterministic = True
_val_opts = tf.data.Options()
_val_opts.deterministic = True

_train_opts.experimental_optimization.map_parallelization = True
_train_opts.experimental_optimization.parallel_batch = True
_train_opts.experimental_optimization.map_and_batch_fusion = True
_val_opts.experimental_optimization.map_parallelization = True
_val_opts.experimental_optimization.parallel_batch = True
_val_opts.experimental_optimization.map_and_batch_fusion = True

CPU_WORKERS = max(2, (os.cpu_count() or 8) - 1)
_train_opts.threading.private_threadpool_size = CPU_WORKERS
_val_opts.threading.private_threadpool_size = max(2, CPU_WORKERS // 2)

n_val = len(val_df)
n_total = len(train_df) + len(val_df)
n_train = n_total - n_val

_SHARD_SIZE_GUESS = 1338
n_val_files = int(np.ceil(n_val / _SHARD_SIZE_GUESS))
if n_val_files >= 1 and n_val_files < len(TFREC_TRAIN_FILES):
    VAL_FILES = TFREC_TRAIN_FILES[:n_val_files]
    TRAIN_FILES = TFREC_TRAIN_FILES[n_val_files:]
    val_raw = tf.data.TFRecordDataset(VAL_FILES, num_parallel_reads=AUTOTUNE)
    train_raw = tf.data.TFRecordDataset(TRAIN_FILES, num_parallel_reads=AUTOTUNE)

    val_parsed = val_raw.map(
        _parse_tfrecord_to_xy, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    train_parsed = train_raw.map(
        _parse_tfrecord_to_xy, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    val_base = val_parsed.take(n_val)
    train_base = train_parsed.take(n_train)
else:
    tfrecord_raw = tf.data.TFRecordDataset(
        TFREC_TRAIN_FILES, num_parallel_reads=AUTOTUNE
    )
    tfrecord_ds = tfrecord_raw.map(
        _parse_tfrecord_to_xy, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    val_base = tfrecord_ds.take(n_val)
    train_base = tfrecord_ds.skip(n_val).take(n_train)

train_cache_path = os.path.join(OUTPUT_DIR, "tfdata_train_cache")
val_cache_path = os.path.join(OUTPUT_DIR, "tfdata_val_cache")

train_base = train_base.shuffle(
    buffer_size=min(8192, n_train), seed=SEED, reshuffle_each_iteration=True
)

train_ds = (
    train_base.with_options(_train_opts)
    .map(_train_map_decoded, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache(train_cache_path)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_base.with_options(_val_opts)
    .map(_val_map_decoded, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache(val_cache_path)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

steps_per_epoch = max(1, int(np.ceil(n_train / batch_size)))
val_steps = max(1, int(np.ceil(n_val / batch_size)))



## === cell 8
model = models.Sequential(
    [
        layers.Input(shape=(512, 512, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)




## === cell 9
def agg_preds(predictions, y):
    y_classes = np.argmax(y, axis=1)
    acc_hist = []
    for i in range(predictions.shape[0]):
        pred_agg = np.mean(predictions[: i + 1], axis=0)
        preds = np.argmax(pred_agg, axis=1)
        acc = preds == y_classes
        acc = np.mean(acc)
        acc_hist.append(acc)
    return acc_hist


def agg_acc(predictions, y):
    pred_agg = np.mean(predictions, axis=0)
    preds = np.argmax(pred_agg, axis=1)
    acc = np.mean(preds == y)
    return acc




## === cell 10
def _ensure_hwc(img):
    if img.ndim == 4:
        return img[0]
    return img


def _back_to_batch(img_hwc, like):
    if like.ndim == 4:
        return np.expand_dims(img_hwc, axis=0)
    return img_hwc


_tta_flip_v = A.Compose([A.VerticalFlip(p=1)])
_tta_rotate = A.Compose(
    [A.Rotate(limit=40, border_mode=cv2.BORDER_CONSTANT, value=0, p=1)]
)
_tta_flip_h = A.Compose([A.HorizontalFlip(p=1)])
_tta_dropout = A.Compose(
    [
        A.GridDropout(
            ratio=0.5,
            unit_size_min=None,
            unit_size_max=None,
            holes_number_x=None,
            holes_number_y=None,
            shift_x=0,
            shift_y=0,
            random_offset=False,
            fill_value=0,
            mask_fill_value=None,
            p=1,
        )
    ]
)
_tta_perspec = A.Compose([A.Perspective(scale=(0.02, 0.1), p=1)])


def flip_lr(image):
    img = _ensure_hwc(image)
    out = _tta_flip_v(image=img)["image"]
    return _back_to_batch(out, image)


def rotate(image):
    img = _ensure_hwc(image)
    out = _tta_rotate(image=img)["image"]
    return _back_to_batch(out, image)


def flip_hor(image):
    img = _ensure_hwc(image)
    out = _tta_flip_h(image=img)["image"]
    return _back_to_batch(out, image)


def dropout(image):
    img = _ensure_hwc(image)
    out = _tta_dropout(image=img)["image"]
    return _back_to_batch(out, image)


def perspec(image):
    img = _ensure_hwc(image)
    out = _tta_perspec(image=img)["image"]
    return _back_to_batch(out, image)




## === cell 11
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
test_images = sample_sub["image_id"].tolist()

TEST_DIR = TEST_PATH if TEST_PATH.endswith("/") else (TEST_PATH + "/")
pred_labels = []



## === cell 12
INFER_BATCH = 32

_rng = np.random.RandomState(SEED)
_h, _w = 512, 512
_tmp1 = np.ones((_h, _w, 3), dtype=np.uint8) * 255
_drop1 = _tta_dropout(image=_tmp1)["image"]
gdrop_mask = _drop1.astype(np.float32) / 255.0
_gdrop_mask_tf = tf.constant(gdrop_mask, dtype=tf.float32)


@tf.function(reduce_retracing=True)
def _decode_resize_test_tf(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [512, 512], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function(
    reduce_retracing=True,
    input_signature=[tf.TensorSpec(shape=[None, 512, 512, 3], dtype=tf.float32)],
)
def _infer_tta_batch(base_batch):
    vflip = tf.reverse(base_batch, axis=[1])
    hflip = tf.reverse(base_batch, axis=[2])
    gdrop = base_batch * _gdrop_mask_tf[None, :, :, :]

    views = tf.concat([base_batch, hflip, vflip, gdrop], axis=0)  # (4B,512,512,3)
    preds_all = model(views, training=False)  # (4B,5)

    B = tf.shape(base_batch)[0]
    pred0 = preds_all[0:B]
    pred_h = preds_all[B : 2 * B]
    pred_v = preds_all[2 * B : 3 * B]
    pred_d = preds_all[3 * B : 4 * B]
    predi = (pred0 + pred_h + pred_v + pred_d) / 4.0
    return tf.argmax(predi, axis=1, output_type=tf.int32)


test_paths = tf.constant(
    [os.path.join(TEST_PATH, image_id) for image_id in test_images]
)

test_path_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_opts = tf.data.Options()
test_opts.deterministic = True
test_opts.experimental_optimization.map_parallelization = True
test_opts.experimental_optimization.parallel_batch = True
test_opts.experimental_optimization.map_and_batch_fusion = True
test_opts.threading.private_threadpool_size = CPU_WORKERS

base_ds = (
    test_path_ds.map(
        _decode_resize_test_tf, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    .batch(INFER_BATCH, drop_remainder=False)
    .prefetch(AUTOTUNE)
).with_options(test_opts)

all_pred_classes = []
for base_batch in base_ds:
    batch_pred_classes = _infer_tta_batch(base_batch)
    all_pred_classes.append(batch_pred_classes.numpy())

all_pred_classes = np.concatenate(all_pred_classes, axis=0)
pred_labels = [int(x) for x in all_pred_classes.tolist()]



## === cell 13
submission = pd.DataFrame({"image_id": test_images, "label": pred_labels})
submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
submission.head()
