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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
from tensorflow.keras import Model

print("TensorFlow:", tf.__version__)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 1
data_path_candidates = [
    "../input/cassava-leaf-disease-classification/",
    "/kaggle/input/cassava-leaf-disease-classification/",
    "/kaggle/data/cassava-leaf-disease-classification/",
]
data_path = None
for p in data_path_candidates:
    if tf.io.gfile.exists(p):
        data_path = p
        break
if data_path is None:
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset folder in any of: "
        + ", ".join(data_path_candidates)
    )

train_csv_data_path = os.path.join(data_path, "train.csv")
label_json_data_path = os.path.join(data_path, "label_num_to_disease_map.json")
images_dir_data_path = os.path.join(data_path, "train_images/")
test_images_dir_data_path = os.path.join(data_path, "test_images/")
train_tfrecords_dir = os.path.join(data_path, "train_tfrecords/")
test_tfrecords_dir = os.path.join(data_path, "test_tfrecords/")

print("Using data_path:", data_path)



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
SEED = 42

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

_WORKERS = max(2, (os.cpu_count() or 4) // 2)
_USE_MPROC = True
_MAX_QSIZE = 32



## === cell 6
NUM_CLASSES = 5

df_all = train_csv.copy()
df_all["image_id"] = df_all["image_id"].astype(str)
df_all = df_all.sort_values("image_id", kind="mergesort").reset_index(drop=True)

n_total = len(df_all)
n_val = int(np.floor(0.1 * n_total))
n_train = n_total - n_val

df_train = df_all.iloc[:n_train].reset_index(drop=True)
df_val = df_all.iloc[n_train:].reset_index(drop=True)

y_train = df_train["label"].astype(np.int32).values
y_val = df_val["label"].astype(np.int32).values

x_train_ids = df_train["image_id"].values
x_val_ids = df_val["image_id"].values

AUTOTUNE = tf.data.AUTOTUNE


def _feature_description():
    return {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }


def _parse_name_img_labelint(example_proto):
    ex = tf.io.parse_single_example(example_proto, _feature_description())
    name = ex["image_name"]
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(
        img, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    label_int = tf.cast(ex["target"], tf.int32)
    return name, img, label_int


def _parse_tfrecord_test(example_proto):
    ex = tf.io.parse_single_example(example_proto, _feature_description())
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(
        img, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(
            factor=20.0 / 360.0, fill_mode="reflect", seed=SEED
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.1, width_factor=0.1, fill_mode="reflect", seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="augmenter",
)


options = tf.data.Options()
options.deterministic = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.experimental_optimization.apply_default_optimizations = True
try:
    options.threading.private_threadpool_size = _WORKERS
    options.threading.max_intra_op_parallelism = 1
except Exception:
    pass

train_tfrec_files = sorted(
    tf.io.gfile.glob(os.path.join(train_tfrecords_dir, "*.tfrec"))
)
test_tfrec_files = sorted(tf.io.gfile.glob(os.path.join(test_tfrecords_dir, "*.tfrec")))

if not train_tfrec_files:
    raise FileNotFoundError(f"No TFRecord files found in: {train_tfrecords_dir}")
if not test_tfrec_files:
    raise FileNotFoundError(f"No TFRecord files found in: {test_tfrecords_dir}")

val_pairs = df_val[["image_id", "label"]].copy()
val_pairs["image_id"] = val_pairs["image_id"].astype(str)
val_pairs["label"] = val_pairs["label"].astype(np.int32)

val_ids_tf = tf.constant(val_pairs["image_id"].values, dtype=tf.string)
val_labels_tf = tf.constant(val_pairs["label"].values, dtype=tf.int32)
val_id_label_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(val_ids_tf, val_labels_tf),
    default_value=tf.constant(-1, dtype=tf.int32),
)


def _split_batch_to_train_val(name, img, label_int):
    is_val = tf.equal(val_id_label_table.lookup(name), label_int)
    train_mask = tf.logical_not(is_val)

    train_imgs = tf.boolean_mask(img, train_mask)
    train_labels = tf.boolean_mask(label_int, train_mask)

    val_imgs = tf.boolean_mask(img, is_val)
    val_labels = tf.boolean_mask(label_int, is_val)

    return (train_imgs, train_labels), (val_imgs, val_labels)


def _nonempty_batch(x, y):
    return tf.greater(tf.shape(x)[0], 0)


full_ds = tf.data.TFRecordDataset(
    train_tfrec_files, num_parallel_reads=AUTOTUNE
).with_options(options)

full_ds = full_ds.apply(tf.data.experimental.ignore_errors())

full_ds = full_ds.map(_parse_name_img_labelint, num_parallel_calls=AUTOTUNE)
full_ds = full_ds.repeat()

full_batched = full_ds.batch(BATCH_SIZE, drop_remainder=False)
split_ds = full_batched.map(_split_batch_to_train_val, num_parallel_calls=AUTOTUNE)

train_batches = split_ds.map(lambda tr, va: tr, num_parallel_calls=AUTOTUNE).filter(
    lambda x, y: _nonempty_batch(x, y)
)
val_batches = split_ds.map(lambda tr, va: va, num_parallel_calls=AUTOTUNE).filter(
    lambda x, y: _nonempty_batch(x, y)
)

val_ds = val_batches.cache().prefetch(AUTOTUNE)

train_ds = train_batches.shuffle(
    buffer_size=2048, seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.map(
    lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

train_steps = int(np.ceil(len(x_train_ids) / BATCH_SIZE))
valid_steps = int(np.ceil(len(x_val_ids) / BATCH_SIZE))

base = applications.ResNet152(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False  # train head first; keeps runtime low and stable

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model_model = Model(inputs=base.input, outputs=outputs)

model_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
    steps_per_execution=25,
)

EPOCHS_HEAD = 3
history1 = model_model.fit(
    train_ds,
    steps_per_epoch=train_steps,
    validation_data=val_ds,
    validation_steps=valid_steps,
    epochs=EPOCHS_HEAD,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
    steps_per_execution=25,
)

EPOCHS_FT = 1
history2 = model_model.fit(
    train_ds,
    steps_per_epoch=train_steps,
    validation_data=val_ds,
    validation_steps=valid_steps,
    epochs=EPOCHS_FT,
    verbose=1,
)



## === cell 7
model_model.summary()



## === cell 8
ss_preview = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
ss = ss_preview

test_ds = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
test_ds = test_ds.with_options(options)

test_ds = test_ds.apply(tf.data.experimental.ignore_errors())

test_ds = test_ds.map(_parse_tfrecord_test, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

test_steps = int(np.ceil(len(ss) / BATCH_SIZE))

probs = model_model.predict(test_ds, steps=test_steps, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)
preds = preds[: len(ss)]

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission["label"] = my_submission["label"].astype(int)
my_submission.to_csv("submission.csv", index=False)



## === cell 9
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nSaved to: submission.csv")
print("Shape:", my_submission.shape)
print("Label value counts:\n", my_submission["label"].value_counts().sort_index())
