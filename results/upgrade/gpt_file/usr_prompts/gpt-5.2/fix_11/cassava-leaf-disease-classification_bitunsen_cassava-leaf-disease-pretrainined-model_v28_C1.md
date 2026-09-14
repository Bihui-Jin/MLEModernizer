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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"



## === cell 2
import matplotlib.pyplot as plt
import json
from PIL import Image

try:
    _RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    _RESAMPLE = Image.LANCZOS



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 4
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 5
train_csv_path = os.path.join(BASE_DIR, "train.csv")
_train_df_tmp = pd.read_csv(train_csv_path, usecols=["image_id"])
print(f"Number of train images (from train.csv): {len(_train_df_tmp)}")
del _train_df_tmp



## === cell 6
IMG_HEIGHT = 400
IMG_WIDTH = 400
batch_size = 32
PRE_TRAINED_MODEL = "../input/resnet50v01/Cassava_Best_ResNet50_Model_V01.hdf5"



## === cell 7
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count() or 1)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## === cell 8
pass



## === cell 9
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_df = pd.read_csv(sample_sub_path)
test_df = sample_df[["image_id"]].copy()

test_samples = test_df.shape[0]
print("Test samples:", test_samples)



## === cell 10
import glob


def resolve_model_path(preferred_path: str) -> str:
    if os.path.exists(preferred_path):
        return preferred_path

    target_name = os.path.basename(preferred_path)
    matches = glob.glob(f"/kaggle/input/**/{target_name}", recursive=True)
    return matches[0] if matches else preferred_path


resolved_model_path = resolve_model_path(PRE_TRAINED_MODEL)
print("Resolved model path:", resolved_model_path)
print("Exists:", os.path.exists(resolved_model_path))



## === cell 11
train_csv_path = os.path.join(BASE_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)
train_df["image_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(TRAIN_DIR, x).replace("\\", "/")
)

print("Train rows:", len(train_df))
print(train_df.head(2))

val_frac = 0.1
train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
val_size = int(len(train_df) * val_frac)
val_df = train_df.iloc[:val_size].copy()
trn_df = train_df.iloc[val_size:].copy()

print("Train/Val:", len(trn_df), len(val_df))



## === cell 12
trn_df = trn_df.copy()
val_df = val_df.copy()
trn_df["label"] = trn_df["label"].astype(np.int32)
val_df["label"] = val_df["label"].astype(np.int32)

AUTOTUNE = tf.data.AUTOTUNE

_train_paths = trn_df["image_path"].values
_train_labels = trn_df["label"].values
_val_paths = val_df["image_path"].values
_val_labels = val_df["label"].values

_ds_opts = tf.data.Options()
_ds_opts.deterministic = True


@tf.function(reduce_retracing=True)
def _decode_resize_train(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear", antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.cast(label, tf.int32)


@tf.function(reduce_retracing=True)
def _decode_resize_val(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear", antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.cast(label, tf.int32)


_train_aug = keras.Sequential(
    [
        keras.layers.RandomRotation(factor=25.0 / 360.0, fill_mode="nearest", seed=42),
        keras.layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="nearest",
            seed=42,
        ),
        keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=42),
        keras.layers.RandomTranslation(
            height_factor=0.1, width_factor=0.1, fill_mode="nearest", seed=42
        ),
    ],
    name="train_augment",
)


@tf.function(reduce_retracing=True)
def _apply_train_aug(img, label):
    img = _train_aug(img, training=True)
    return img, label


train_ds = tf.data.Dataset.from_tensor_slices(
    (_train_paths, _train_labels)
).with_options(_ds_opts)
train_ds = train_ds.shuffle(
    buffer_size=len(_train_paths), seed=42, reshuffle_each_iteration=True
)
train_ds = train_ds.map(
    _decode_resize_train, num_parallel_calls=AUTOTUNE, deterministic=True
)
train_ds = train_ds.map(
    _apply_train_aug, num_parallel_calls=AUTOTUNE, deterministic=True
)
train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((_val_paths, _val_labels)).with_options(
    _ds_opts
)
val_ds = val_ds.map(_decode_resize_val, num_parallel_calls=AUTOTUNE, deterministic=True)
val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)



## === cell 13
NUM_CLASSES = 5

if os.path.exists(resolved_model_path):
    model = load_model(resolved_model_path, compile=False)
    print("Loaded pretrained model.")
else:
    base = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        pooling="avg",
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base(inputs, training=False)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    steps_per_epoch = int(np.ceil(len(trn_df) / batch_size))
    val_steps = int(np.ceil(len(val_df) / batch_size))

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        verbose=2,
    )

model.summary()



## === cell 14
TEST_TTA_N = 3  # keep identical to original loop range(3)

_NEED_RESCALE = not os.path.exists(resolved_model_path)

_tta_aug_layer = keras.Sequential(
    [
        keras.layers.RandomRotation(factor=45.0 / 360.0, fill_mode="nearest", seed=42),
        keras.layers.RandomZoom(
            height_factor=(-0.4, 0.4),
            width_factor=(-0.4, 0.4),
            fill_mode="nearest",
            seed=42,
        ),
        keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=42),
        keras.layers.RandomTranslation(
            height_factor=0.1, width_factor=0.1, fill_mode="nearest", seed=42
        ),
    ],
    name="tta_augment_fast",
)


@tf.function(reduce_retracing=True)
def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method="lanczos3", antialias=True
    )
    img = tf.cast(img, tf.float32)
    if _NEED_RESCALE:
        img = img / 255.0
    return img


@tf.function(reduce_retracing=True)
def _predict_tta_mean_probs(x_batch: tf.Tensor) -> tf.Tensor:
    probs = model(x_batch, training=False)
    if TEST_TTA_N > 0:
        for _ in tf.range(TEST_TTA_N):
            x_aug = _tta_aug_layer(x_batch, training=True)
            probs += model(x_aug, training=False)
        probs = probs / tf.cast(1 + TEST_TTA_N, probs.dtype)
    return probs




## === cell 15
test_image_ids = test_df["image_id"].values
n_test = len(test_image_ids)

PRED_BATCH = 32
pred_labels = np.empty((n_test,), dtype=np.int64)

paths = np.array(
    [os.path.join(TEST_DIR, img_id) for img_id in test_image_ids], dtype=object
)
idxs = np.arange(n_test, dtype=np.int32)

ds = tf.data.Dataset.from_tensor_slices((idxs, paths)).with_options(_ds_opts)
ds = ds.map(
    lambda i, p: (i, _decode_resize(p)), num_parallel_calls=AUTOTUNE, deterministic=True
)

ds = ds.cache()

ds = ds.batch(PRED_BATCH, drop_remainder=False).prefetch(AUTOTUNE)

for id_batch, x_batch in ds:
    probs = _predict_tta_mean_probs(x_batch).numpy()  # (B,C)
    labels = np.argmax(probs, axis=1).astype(np.int64)
    pred_labels[id_batch.numpy()] = labels

test_results_df = pd.DataFrame(
    {"image_id": test_image_ids, "label": pred_labels.astype(int)}
)

submission = sample_df[["image_id"]].merge(test_results_df, on="image_id", how="left")
submission["label"] = submission["label"].fillna(0).astype(int)
submission = submission[["image_id", "label"]]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head(3))



## === cell 16
submission_check = pd.read_csv("submission.csv")
print(submission_check.head(3))
print("Columns:", list(submission_check.columns))
print("Rows:", len(submission_check))
