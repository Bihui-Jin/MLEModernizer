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

3.12

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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import efficientnet_v2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

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

print("TensorFlow:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 1
WORK_DIR = "/kaggle/input/cassava-leaf-disease-classification/"

train_csv_path = os.path.join(WORK_DIR, "train.csv")
sample_sub_path = os.path.join(WORK_DIR, "sample_submission.csv")
train_img_dir = os.path.join(WORK_DIR, "train_images")
test_img_dir = os.path.join(WORK_DIR, "test_images")

train_tfrec_dir = os.path.join(WORK_DIR, "train_tfrecords")
test_tfrec_dir = os.path.join(WORK_DIR, "test_tfrecords")

assert os.path.exists(train_csv_path), train_csv_path
assert os.path.exists(sample_sub_path), sample_sub_path
assert os.path.isdir(train_img_dir), train_img_dir
assert os.path.isdir(test_img_dir), test_img_dir

assert os.path.isdir(train_tfrec_dir), train_tfrec_dir
assert os.path.isdir(test_tfrec_dir), test_tfrec_dir

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

print("train_df:", train_df.shape, train_df.columns.tolist())
print("sample_sub:", sample_sub.shape, sample_sub.columns.tolist())




## === cell 2
class SigmoidFocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(
            1 - y_pred
        )
        weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_loss = weight * ((1 - y_pred) ** self.gamma) * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)




## === cell 3
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 6  # keep as given

from sklearn.model_selection import train_test_split

tr_df, va_df = train_test_split(
    train_df, test_size=0.15, random_state=SEED, stratify=train_df["label"]
)

print("train split:", tr_df.shape, "val split:", va_df.shape)



## === cell 4
tr_df = tr_df.copy()
va_df = va_df.copy()
tr_df["path"] = (train_img_dir + "/" + tr_df["image_id"].astype(str)).values
va_df["path"] = (train_img_dir + "/" + va_df["image_id"].astype(str)).values

aug = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.05, seed=SEED),
    ],
    name="aug",
)

_FEATURE_SPEC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_and_preprocess_image_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    img = tf.cast(img, tf.float32)
    img = efficientnet_v2.preprocess_input(img)
    return img


@tf.function
def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_SPEC)
    img = _decode_and_preprocess_image_bytes(ex["image"])
    label = tf.one_hot(tf.cast(ex["target"], tf.int32), depth=NUM_CLASSES)
    return img, label


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_SPEC)
    img = _decode_and_preprocess_image_bytes(ex["image"])
    return ex["image_name"], img


@tf.function
def apply_aug(x, y):
    return aug(x, training=True), y


def _with_ds_options(ds):
    options = tf.data.Options()
    try:
        options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass
    try:
        options.experimental_optimization.autotune = True
    except Exception:
        pass
    try:
        options.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    return ds.with_options(options)


def _list_tfrecords(folder, pattern="*.tfrec"):
    files = tf.io.gfile.glob(os.path.join(folder, pattern))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No TFRecord files found in {folder} with pattern {pattern}"
        )
    return files


def _make_tfrec_ds(tfrec_files, training: bool):
    ds_files = tf.data.Dataset.from_tensor_slices(tfrec_files)
    if training:
        ds_files = ds_files.shuffle(
            len(tfrec_files), seed=SEED, reshuffle_each_iteration=True
        )

    ds = ds_files.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=1),
        cycle_length=min(8, len(tfrec_files)),
        block_length=16,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )

    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        _parse_train_example,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )

    if training:
        ds = ds.map(apply_aug, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = _with_ds_options(ds)
    return ds


train_tfrec_files = _list_tfrecords(train_tfrec_dir, "*.tfrec")
test_tfrec_files = _list_tfrecords(test_tfrec_dir, "*.tfrec")

train_ds = _make_tfrec_ds(train_tfrec_files, training=True)

va_ids = va_df["image_id"].astype(str).to_numpy(copy=False)
va_id_set = tf.constant(va_ids)


@tf.function
def _is_in_val(image_name, img, label):
    return tf.reduce_any(tf.equal(image_name, va_id_set))


def make_val_ds_from_tfrec(tfrec_files):
    ds_files = tf.data.Dataset.from_tensor_slices(tfrec_files)
    ds = ds_files.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=1),
        cycle_length=min(8, len(tfrec_files)),
        block_length=16,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(
        lambda ex: tf.io.parse_single_example(ex, _FEATURE_SPEC),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(
        lambda ex: (
            ex["image_name"],
            _decode_and_preprocess_image_bytes(ex["image"]),
            tf.one_hot(tf.cast(ex["target"], tf.int32), depth=NUM_CLASSES),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds = ds.filter(_is_in_val)
    ds = ds.map(
        lambda image_name, img, label: (img, label),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    ds = _with_ds_options(ds)
    return ds


val_ds = make_val_ds_from_tfrec(train_tfrec_files)



## === cell 5
base = efficientnet_v2.EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

compile_kwargs = dict(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0, from_logits=False),
    metrics=["accuracy"],
)
try:
    model.compile(**compile_kwargs, jit_compile=True)
except Exception as e:
    print(
        "jit_compile=True not supported here, compiling without jit. Error was:",
        repr(e),
    )
    model.compile(**compile_kwargs)

model.summary()



## === cell 6
ckpt_path = "/kaggle/working/best_model.weights.h5"
callbacks = [
    ModelCheckpoint(
        ckpt_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1
    ),
]

steps_per_epoch = int(np.ceil(len(tr_df) / BATCH_SIZE))
validation_steps = int(np.ceil(len(va_df) / BATCH_SIZE))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)

model.load_weights(ckpt_path)



## === cell 7
test_ids = sample_sub["image_id"].to_numpy(dtype=str, copy=False)

test_ds = tf.data.Dataset.from_tensor_slices(test_tfrec_files).interleave(
    lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=1),
    cycle_length=min(8, len(test_tfrec_files)),
    block_length=16,
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=True,
)

test_ds = test_ds.map(
    _parse_test_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
)

test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
test_ds = _with_ds_options(test_ds)

all_names = []
all_preds = []

for batch in test_ds:
    names_b, imgs_b = batch
    preds_b = model(imgs_b, training=False)
    all_names.append(names_b.numpy())
    all_preds.append(preds_b.numpy())

names = np.concatenate(all_names).astype(str)
pred = np.concatenate(all_preds, axis=0)
pred_labels = np.argmax(pred, axis=1).astype(int)

pred_map = {n: l for n, l in zip(names, pred_labels)}
ordered_labels = np.array([pred_map[iid] for iid in test_ids], dtype=int)

submission = pd.DataFrame({"image_id": test_ids, "label": ordered_labels})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))
