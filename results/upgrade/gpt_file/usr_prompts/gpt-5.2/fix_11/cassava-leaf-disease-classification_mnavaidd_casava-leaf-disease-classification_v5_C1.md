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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for _gpu in gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_data_path = data_path + "train.csv"
label_json_data_path = data_path + "label_num_to_disease_map.json"
images_dir_data_path = data_path + "train_images/"

test_images_dir_data_path = data_path + "test_images/"
sample_submission_path = data_path + "sample_submission.csv"

train_tfrecords_dir = os.path.join(data_path, "train_tfrecords")
test_tfrecords_dir = os.path.join(data_path, "test_tfrecords")




## === cell 2
train_csv = pd.read_csv(train_csv_data_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_data_path, orient="index")
label_class = label_class.values.flatten().tolist()




## === cell 3
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")




## === cell 4
train_csv.head()




## === cell 5
BATCH_SIZE = 18
IMG_SIZE = 224
NUM_CLASSES = 5

EPOCHS = 3  # unchanged




## === cell 6
TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _decode_and_resize_from_bytes(img_bytes, label=None):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    if label is None:
        return img
    label = tf.one_hot(tf.cast(label, tf.int32), depth=NUM_CLASSES)
    return img, label


def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    return _decode_and_resize_from_bytes(ex["image"], ex["target"])


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    img = _decode_and_resize_from_bytes(ex["image"], None)
    return ex["image_name"], img


def _list_tfrecords(tfrecord_dir):
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrec"))
    if not files:
        raise FileNotFoundError(f"No TFRecord files found under: {tfrecord_dir}")
    return sorted(files)


options = tf.data.Options()
options.deterministic = True
try:
    options.experimental_deterministic = True
except Exception:
    pass
try:
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
except Exception:
    pass
try:
    options.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    options.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass

train_tfrec_files = _list_tfrecords(train_tfrecords_dir)

n_tf = len(train_tfrec_files)
n_val_tf = max(1, int(round(n_tf * 0.1)))
val_tfrec_files = train_tfrec_files[:n_val_tf]
train_tfrec_files_split = train_tfrec_files[n_val_tf:]
if not train_tfrec_files_split:
    train_tfrec_files_split = train_tfrec_files  # safety

raw_train_ds = tf.data.TFRecordDataset(
    train_tfrec_files_split, num_parallel_reads=AUTOTUNE
).with_options(options)
raw_val_ds = tf.data.TFRecordDataset(
    val_tfrec_files, num_parallel_reads=AUTOTUNE
).with_options(options)

shuffle_buf = 2048

train_ds = raw_train_ds.map(
    _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True
)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(
    buffer_size=shuffle_buf, seed=42, reshuffle_each_iteration=True
)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

val_ds = raw_val_ds.map(
    _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True
)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).cache().prefetch(AUTOTUNE)

base = applications.EfficientNetB7(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False  # transfer learning (unchanged)

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model_model = tf.keras.Model(inputs, outputs)

model_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)

history = model_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)




## === cell 7
model_model.summary()




## === cell 8
ss = pd.read_csv(sample_submission_path)
example_image_id = ss["image_id"].iloc[0]
print("Example test image_id:", example_image_id)




## === cell 9
test_tfrec_files = _list_tfrecords(test_tfrecords_dir)
raw_test_ds = tf.data.TFRecordDataset(
    test_tfrec_files, num_parallel_reads=AUTOTUNE
).with_options(options)

test_ds_named = raw_test_ds.map(
    _parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds_named = test_ds_named.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

name_batches = []
test_images_batches = []
for n, x in test_ds_named:
    name_batches.append(n.numpy())
    test_images_batches.append(x)

all_names = np.concatenate(name_batches, axis=0).astype("U")

test_images_tensor = tf.concat(test_images_batches, axis=0)
test_images_ds = tf.data.Dataset.from_tensor_slices(test_images_tensor).batch(
    BATCH_SIZE
)

probs = model_model.predict(test_images_ds, verbose=0)
preds = np.argmax(probs, axis=1).astype(int)

pred_by_id = pd.DataFrame({"image_id": all_names, "label": preds})
my_submission = ss[["image_id"]].merge(pred_by_id, on="image_id", how="left")
if my_submission["label"].isna().any():
    missing = (
        my_submission.loc[my_submission["label"].isna(), "image_id"].head(5).tolist()
    )
    raise RuntimeError(
        f"Missing predictions for some test ids (unexpected). Examples: {missing}"
    )
my_submission["label"] = my_submission["label"].astype(int)

my_submission.to_csv("submission.csv", index=False)




## === cell 10
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nWrote: submission.csv")
print("Rows:", len(my_submission), "Cols:", list(my_submission.columns))
