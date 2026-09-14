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
import random
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet_v2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick
    tf.config.threading.set_inter_op_parallelism_threads(0)  # let TF pick
except Exception:
    pass




## === cell 1
WORK_DIR = "../input/cassava-leaf-disease-classification/"
TRAIN_CSV = os.path.join(WORK_DIR, "train.csv")
SAMPLE_SUB = os.path.join(WORK_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(WORK_DIR, "train_images")
TEST_IMG_DIR = os.path.join(WORK_DIR, "test_images")

for p in [TRAIN_CSV, SAMPLE_SUB, TRAIN_IMG_DIR, TEST_IMG_DIR]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required path not found: {p}")

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)
sample_sub["image_id"] = sample_sub["image_id"].astype(str)

NUM_CLASSES = int(train_df["label"].nunique())
if NUM_CLASSES != 5:
    NUM_CLASSES = 5




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


custom_objects = {"SigmoidFocalCrossEntropy": SigmoidFocalCrossEntropy}




## === cell 3
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # Keep runtime within limits; no early stopping added per constraints.


@tf.function(reduce_retracing=True)
def _decode_resize_preprocess_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = efficientnet_v2.preprocess_input(img)
    return img


@tf.function(reduce_retracing=True)
def _augment_stateless(img, path):
    h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    seed = tf.stack([tf.cast(h, tf.int32), tf.constant(SEED, tf.int32)])
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)
    img = tf.image.stateless_random_flip_up_down(
        img, seed=seed + tf.constant([1, 1], tf.int32)
    )
    return img


@tf.function(reduce_retracing=True)
def _train_decode_map(path, label):
    img = _decode_resize_preprocess_from_path(path)
    return img, path, tf.cast(label, tf.int32)


@tf.function(reduce_retracing=True)
def _train_augment_map(img, path, label):
    img = _augment_stateless(img, path)
    return img, tf.cast(label, tf.int32)


@tf.function(reduce_retracing=True)
def _val_map(path, label):
    img = _decode_resize_preprocess_from_path(path)
    return img, tf.cast(label, tf.int32)


from sklearn.model_selection import train_test_split

tr_df, va_df = train_test_split(
    train_df, test_size=0.15, random_state=SEED, stratify=train_df["label"]
)

tr_paths = np.asarray(TRAIN_IMG_DIR + "/" + tr_df["image_id"].values, dtype=np.str_)
tr_labels = np.asarray(tr_df["label"].values, dtype=np.int32)

va_paths = np.asarray(TRAIN_IMG_DIR + "/" + va_df["image_id"].values, dtype=np.str_)
va_labels = np.asarray(va_df["label"].values, dtype=np.int32)

opt = tf.data.Options()
opt.deterministic = True
try:
    opt.experimental_slack = True
except Exception:
    pass
try:
    opt.experimental_optimization.map_parallelization = True
    opt.experimental_optimization.parallel_batch = True
    opt.experimental_optimization.map_and_batch_fusion = True
    opt.experimental_optimization.autotune_buffers = True
except Exception:
    pass
try:
    opt.autotune.enabled = True
except Exception:
    pass

shuffle_buf = int(min(len(tr_df), 4096))

train_base = tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels)).with_options(opt)
train_base = train_base.shuffle(
    buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
)
train_base = train_base.map(
    _train_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True
)
train_base = train_base.cache()  # caches (img, path, label)
train_base = train_base.map(
    _train_augment_map, num_parallel_calls=AUTOTUNE, deterministic=True
)
train_base = train_base.apply(tf.data.experimental.ignore_errors())

train_ds = train_base.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

val_base = tf.data.Dataset.from_tensor_slices((va_paths, va_labels)).with_options(opt)
val_base = val_base.map(_val_map, num_parallel_calls=AUTOTUNE, deterministic=True)
val_base = val_base.cache()  # avoid re-decoding validation each epoch (already present)
val_base = val_base.apply(tf.data.experimental.ignore_errors())
val_ds = val_base.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

base = efficientnet_v2.EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
base.trainable = False  # fast, stable baseline; avoids long training

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)




## === cell 4
test_image_ids = sample_sub["image_id"].to_numpy()
test_paths = np.char.add(np.char.add(TEST_IMG_DIR, "/"), test_image_ids)

if len(test_paths) == 0 or (not os.path.exists(test_paths[0])):
    raise FileNotFoundError(
        f"Test images not found or empty path list; first path: {test_paths[:1]}"
    )


@tf.function(reduce_retracing=True)
def _test_map(path):
    return _decode_resize_preprocess_from_path(path)


test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(opt)
test_ds = test_ds.map(_test_map, num_parallel_calls=AUTOTUNE, deterministic=True)
test_ds = test_ds.cache()
test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

pred_steps = int(math.ceil(len(test_paths) / BATCH_SIZE))
pred_probs = model.predict(test_ds, verbose=0, steps=pred_steps)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)

submission = pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
