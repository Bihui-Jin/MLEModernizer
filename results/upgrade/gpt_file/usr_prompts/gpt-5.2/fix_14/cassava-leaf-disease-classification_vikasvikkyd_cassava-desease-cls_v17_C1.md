# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.6740707162284678

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.213) has done: 'The timeout is dominated by repeatedly decoding/resizing JPEGs and running VGG16 over ~18k images for 10 epochs, plus expensive input pipeline overhead per-step. I keep the exact model/loop/epochs intact, but make the tf.data pipeline cheaper and more GPU/CPU efficient by switching to `TFRecordDataset` (already provided) with parallel interleave, caching to local disk, and eliminating per-element Python list/path handling where possible. I also remove a large, unnecessary `glob()`/set membership scan of the entire test directory (uses the sample submission ordering directly) and add a few safe TensorFlow runtime settings (GPU memory growth, better autotune) that don’t change semantics. These changes are equivalent in results (same images/labels, same preprocessing, same training loop) but substantially reduce input overhead so training can finish within 600 seconds.'
- What this solution (achieved 0.213) has done: 'I fix the crash in the very first cell by removing the TensorFlow XLA JIT toggle that is triggering a protobuf/TF compatibility `MessageFactory.GetPrototype` error in this environment (score-neutral). Then I fix the `StaticHashTable` construction error by storing integer labels in the lookup tables (shape `()`), and converting them to one-hot inside the dataset map; this preserves the exact training objective while making the pipeline valid. Finally, once `train_ds/test_ds` are defined successfully, the existing training loop and submission writing run end-to-end and produce `submission.csv` in the required format, which should also improve accuracy substantially versus the previously broken/degenerate run.'
- What this solution (achieved 0.213) has done: 'I fix the TensorFlow import crash by avoiding the determinism toggle that triggers the `MessageFactory.GetPrototype` protobuf issue in this environment. Then I fix the `model.fit()` `math domain error` by explicitly setting `steps_per_epoch`/`validation_steps` to valid positive integers computed from the known split sizes (your tf.data pipelines are infinite/unknown-cardinality due to `interleave+filter`). Finally, to move accuracy toward the target without changing the model/training loop, I replace the expensive/incorrect per-element `tf.reduce_any(tf.equal(...))` membership filters with an O(1) `StaticHashTable` membership check (same semantics: only keep IDs in the split), which also prevents silently dropping/keeping wrong samples and should improve score substantially.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import json
import random

import numpy as np
import pandas as pd
import cv2  # kept (present in original), even though unused

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # kept
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input  # kept
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for _gpu in gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "../input/cassava-leaf-disease-classification"
source_dir = os.path.join(BASE_DIR, "train_images")
train_csv_path = os.path.join(BASE_DIR, "train.csv")
label_name_path = os.path.join(BASE_DIR, "label_num_to_disease_map.json")

data_label = pd.read_csv(train_csv_path)

with open(label_name_path, "r") as f:
    label_name = json.load(f)

csv_ids = data_label["image_id"].astype(str).tolist()
all_images = list(csv_ids)
random.shuffle(all_images)

print("Train labels in CSV:", len(data_label))
print("Images from CSV (assumed present):", len(all_images))



## === cell 2
IMG_H, IMG_W = 100, 100

train_size = 0.9
split_size = int(len(all_images) * train_size)
train_images = all_images[:split_size]
test_images = all_images[split_size:]

id_to_label = dict(
    zip(data_label["image_id"].astype(str).values, data_label["label"].values)
)

train_labels = np.array([id_to_label[img] for img in train_images], dtype=np.int32)
test_labels = np.array([id_to_label[img] for img in test_images], dtype=np.int32)

if len(train_images) == 0 or len(test_images) == 0:
    raise RuntimeError(
        f"Empty train/test split: train={len(train_images)}, test={len(test_images)}"
    )

y_train = to_categorical(train_labels, 5)
y_test = to_categorical(test_labels, 5)

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

tfrecord_train_dir = os.path.join(BASE_DIR, "train_tfrecords")
train_tfrecords = sorted(tf.io.gfile.glob(os.path.join(tfrecord_train_dir, "*.tfrec")))
if len(train_tfrecords) == 0:
    raise RuntimeError(f"No TFRecords found in: {tfrecord_train_dir}")

_SHUFFLE_BUFFER = min(len(train_images), 4096)

options = tf.data.Options()
options.autotune.enabled = True
options.deterministic = True

cache_dir = "/kaggle/working/tf_cache"
os.makedirs(cache_dir, exist_ok=True)
train_cache_path = os.path.join(
    cache_dir, f"train_tfrec_{IMG_H}x{IMG_W}_bs{BATCH_SIZE}.cache"
)
test_cache_path = os.path.join(
    cache_dir, f"val_tfrec_{IMG_H}x{IMG_W}_bs{BATCH_SIZE}.cache"
)

train_membership_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(train_images),
        values=tf.ones([len(train_images)], dtype=tf.int32),
    ),
    default_value=tf.constant(0, dtype=tf.int32),
)
test_membership_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(test_images),
        values=tf.ones([len(test_images)], dtype=tf.int32),
    ),
    default_value=tf.constant(0, dtype=tf.int32),
)

train_label_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(train_images),
        values=tf.constant(train_labels, dtype=tf.int32),
    ),
    default_value=tf.constant(-1, dtype=tf.int32),
)
test_label_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(test_images),
        values=tf.constant(test_labels, dtype=tf.int32),
    ),
    default_value=tf.constant(-1, dtype=tf.int32),
)

_feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
}


def _parse_tfrec(example_proto):
    ex = tf.io.parse_single_example(example_proto, _feature_description)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    image_id = ex["image_id"]
    return image_id, img


def _keep_train(image_id, img):
    return tf.equal(train_membership_table.lookup(image_id), 1)


def _keep_test(image_id, img):
    return tf.equal(test_membership_table.lookup(image_id), 1)


def _add_train_label(image_id, img):
    y = train_label_table.lookup(image_id)  # int32 scalar
    y = tf.one_hot(y, depth=5, dtype=tf.float32)
    return img, y


def _add_test_label(image_id, img):
    y = test_label_table.lookup(image_id)  # int32 scalar
    y = tf.one_hot(y, depth=5, dtype=tf.float32)
    return img, y


def _make_tfrec_ds(file_list):
    ds_files = tf.data.Dataset.from_tensor_slices(file_list)
    ds = ds_files.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTOTUNE),
        cycle_length=min(8, len(file_list)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


train_ds = _make_tfrec_ds(train_tfrecords).with_options(options)
train_ds = train_ds.filter(_keep_train)
train_ds = train_ds.map(
    _add_train_label, num_parallel_calls=AUTOTUNE, deterministic=True
)
train_ds = train_ds.cache(train_cache_path)
train_ds = train_ds.shuffle(
    buffer_size=_SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

test_ds = _make_tfrec_ds(train_tfrecords).with_options(options)
test_ds = test_ds.filter(_keep_test)
test_ds = test_ds.map(_add_test_label, num_parallel_calls=AUTOTUNE, deterministic=True)
test_ds = test_ds.cache(test_cache_path)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

steps_per_epoch = int(np.ceil(len(train_images) / BATCH_SIZE))
validation_steps = int(np.ceil(len(test_images) / BATCH_SIZE))
if steps_per_epoch <= 0 or validation_steps <= 0:
    raise RuntimeError(
        f"Non-positive steps computed: steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}"
    )

print("Train samples:", len(train_images), "Test samples:", len(test_images))
print("IMG:", (IMG_H, IMG_W), "Batch:", BATCH_SIZE, "Shuffle buffer:", _SHUFFLE_BUFFER)
print("TFRecords:", len(train_tfrecords), "Train cache:", train_cache_path)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)




## === cell 3
def define_directory():
    base = "/kaggle/working/Training"
    subdirs = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
    if not os.path.exists(base):
        os.mkdir(base)
    for sd in subdirs:
        p = os.path.join(base, sd)
        if not os.path.exists(p):
            os.mkdir(p)




## === cell 4
pre_trained_model = VGG16(
    include_top=False, weights="imagenet", input_shape=(IMG_H, IMG_W, 3)
)

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer("block5_pool")
last_output = last_layer.output

x = layers.Reshape((-1,))(last_output)
x = layers.Dense(5, activation="softmax")(x)

model = keras.Model(pre_trained_model.input, x)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)

model.summary()



## === cell 5
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10,
    verbose=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)



## === cell 6
test_dir = os.path.join(BASE_DIR, "test_images")

sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
ordered_test_ids = sample_sub["image_id"].astype(str).tolist()

tfrecord_test_dir = os.path.join(BASE_DIR, "test_tfrecords")
test_tfrecords = sorted(tf.io.gfile.glob(os.path.join(tfrecord_test_dir, "*.tfrec")))
if len(test_tfrecords) == 0:
    raise RuntimeError(f"No test TFRecords found in: {tfrecord_test_dir}")

test_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(ordered_test_ids),
        values=tf.ones([len(ordered_test_ids)], dtype=tf.int32),
    ),
    default_value=tf.constant(0, dtype=tf.int32),
)

test_index_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(ordered_test_ids),
        values=tf.range(len(ordered_test_ids), dtype=tf.int32),
    ),
    default_value=tf.constant(-1, dtype=tf.int32),
)

_test_feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
}


def _parse_test_tfrec(example_proto):
    ex = tf.io.parse_single_example(example_proto, _test_feature_description)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    image_id = ex["image_id"]
    return image_id, img


def _keep_known_test_ids(image_id, img):
    return tf.equal(test_id_table.lookup(image_id), 1)


def _attach_index(image_id, img):
    idx = test_index_table.lookup(image_id)
    return idx, img


def _make_test_tfrec_ds(file_list):
    ds_files = tf.data.Dataset.from_tensor_slices(file_list)
    ds = ds_files.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTOTUNE),
        cycle_length=min(8, len(file_list)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(_parse_test_tfrec, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.filter(_keep_known_test_ids)
    ds = ds.map(_attach_index, num_parallel_calls=AUTOTUNE, deterministic=True)
    return ds


test_ds_raw = _make_test_tfrec_ds(test_tfrecords).with_options(options)

pred_indices = []
pred_probs = []
for batch_idx, batch_img in test_ds_raw.batch(
    BATCH_SIZE, drop_remainder=False
).prefetch(AUTOTUNE):
    p = model(batch_img, training=False)
    pred_probs.append(p.numpy())
    pred_indices.append(batch_idx.numpy())

pred_probs = np.concatenate(pred_probs, axis=0)
pred_indices = np.concatenate(pred_indices, axis=0)

if pred_probs.shape[0] != len(ordered_test_ids):
    raise RuntimeError(
        f"Predicted {pred_probs.shape[0]} test items but expected {len(ordered_test_ids)}. "
        f"Check TFRecords/image_id matching."
    )

order = np.argsort(pred_indices)
pred_labels = pred_probs[order].argmax(axis=1).astype(int).tolist()

submission_df = pd.DataFrame({"image_id": ordered_test_ids, "label": pred_labels})
submission_path = "./submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())
assert submission_path.endswith(".csv")
assert submission_df.shape[0] == sample_sub.shape[0]
assert submission_df.columns.tolist() == ["image_id", "label"]

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2983922464.py in <cell line: 0>()
     81     pred_indices.append(batch_idx.numpy())
     82 
---> 83 pred_probs = np.concatenate(pred_probs, axis=0)
     84 pred_indices = np.concatenate(pred_indices, axis=0)
     85 

ValueError: need at least one array to concatenate
