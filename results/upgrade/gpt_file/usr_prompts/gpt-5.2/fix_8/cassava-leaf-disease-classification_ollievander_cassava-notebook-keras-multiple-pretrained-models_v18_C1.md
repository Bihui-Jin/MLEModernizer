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
import numpy as np
import pandas as pd

INPUT_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train_images")
TEST_PATH = os.path.join(INPUT_DIR, "test_images")
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")
LABEL_MAP_JSON = os.path.join(INPUT_DIR, "label_num_to_disease_map.json")

print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB_CSV exists:", os.path.exists(SAMPLE_SUB_CSV))
print("LABEL_MAP_JSON exists:", os.path.exists(LABEL_MAP_JSON))

TRAIN_TFREC_DIR = os.path.join(INPUT_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(INPUT_DIR, "test_tfrecords")
print("TRAIN_TFREC_DIR exists:", os.path.exists(TRAIN_TFREC_DIR))
print("TEST_TFREC_DIR exists:", os.path.exists(TEST_TFREC_DIR))




## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

SEED = 100

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__ if hasattr(keras, "__version__") else "unknown")




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
with open(LABEL_MAP_JSON) as f:
    classes = json.load(f)

train_df["class"] = train_df["label"].astype(str).map(classes)
print(train_df.head())
print("train_df shape:", train_df.shape)




## === cell 3
train_df = train_df.copy()
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(str)

train_split, val_split = train_test_split(
    train_df, test_size=0.05, random_state=SEED, stratify=train_df["label"].values
)

print("Train split:", train_split.shape, "Val split:", val_split.shape)
print(train_split[["image_id", "label"]].head())




## === cell 4
IMG_SIZE = (512, 512)
BATCH_SIZE = 16
NUM_CLASSES = 5
AUTOTUNE = tf.data.AUTOTUNE

label_lookup = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant([str(i) for i in range(NUM_CLASSES)], dtype=tf.string),
        values=tf.constant(list(range(NUM_CLASSES)), dtype=tf.int32),
    ),
    default_value=-1,
)

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.experimental_deterministic = True

TRAIN_TFRECS = sorted(
    [
        os.path.join(TRAIN_TFREC_DIR, f)
        for f in os.listdir(TRAIN_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
TEST_TFRECS = sorted(
    [
        os.path.join(TEST_TFREC_DIR, f)
        for f in os.listdir(TEST_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
print("Num train tfrecs:", len(TRAIN_TFRECS), "Num test tfrecs:", len(TEST_TFRECS))


def _decode_resize_rescale_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    x = _decode_resize_rescale_bytes(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    y = tf.one_hot(y, NUM_CLASSES, dtype=tf.float32)
    return ex["image_name"], x, y


def _parse_test_example(example_proto):
    feats = dict(_TFREC_FEATURES)
    ex = tf.io.parse_single_example(example_proto, feats)
    x = _decode_resize_rescale_bytes(ex["image"])
    return ex["image_name"], x


train_ids = tf.constant(train_split["image_id"].values, dtype=tf.string)
val_ids = tf.constant(val_split["image_id"].values, dtype=tf.string)

train_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        train_ids, tf.ones_like(train_ids, dtype=tf.int32)
    ),
    default_value=0,
)
val_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(val_ids, tf.ones_like(val_ids, dtype=tf.int32)),
    default_value=0,
)


def _make_train_ds_from_tfrecs(tfrecs):
    files = tf.data.Dataset.from_tensor_slices(tf.constant(tfrecs, dtype=tf.string))
    files = files.with_options(DATASET_OPTIONS)
    ds = files.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=1),
        cycle_length=min(8, len(tfrecs)) if len(tfrecs) > 0 else 1,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.filter(
        lambda image_name, x, y: tf.equal(train_id_table.lookup(image_name), 1)
    )
    ds = ds.map(
        lambda image_name, x, y: (x, y), num_parallel_calls=AUTOTUNE, deterministic=True
    )

    SHUFFLE_BUFFER_CAP = 8192
    ds = ds.shuffle(
        buffer_size=min(int(train_split.shape[0]), SHUFFLE_BUFFER_CAP),
        seed=SEED,
        reshuffle_each_iteration=True,
    )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds_from_tfrecs(tfrecs):
    files = tf.data.Dataset.from_tensor_slices(tf.constant(tfrecs, dtype=tf.string))
    files = files.with_options(DATASET_OPTIONS)
    ds = files.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=1),
        cycle_length=min(8, len(tfrecs)) if len(tfrecs) > 0 else 1,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.filter(
        lambda image_name, x, y: tf.equal(val_id_table.lookup(image_name), 1)
    )
    ds = ds.map(
        lambda image_name, x, y: (x, y), num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_gen = _make_train_ds_from_tfrecs(TRAIN_TFRECS)
val_gen = _make_val_ds_from_tfrecs(TRAIN_TFRECS)

print("Class indices:", {str(i): i for i in range(NUM_CLASSES)})




## === cell 5
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

aug = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
    ],
    name="augmentation",
)
x = aug(inputs)

x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

ckpt_path = os.path.join(OUTPUT_DIR, "best_model.keras")
callbacks = [
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=1, verbose=1),
    EarlyStopping(
        monitor="val_accuracy", patience=2, restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
]

EPOCHS = 3  # unchanged

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

if os.path.exists(ckpt_path):
    model = keras.models.load_model(ckpt_path)
    print("Loaded best checkpoint:", ckpt_path)
else:
    print("Checkpoint not found; using last-epoch weights.")




## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
test_ids = sample_sub["image_id"].astype(str).tolist()

test_id_tensor = tf.constant(sample_sub["image_id"].astype(str).values, dtype=tf.string)

test_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        test_id_tensor, tf.ones_like(test_id_tensor, dtype=tf.int32)
    ),
    default_value=0,
)


def _make_test_ds_from_tfrecs(tfrecs):
    files = tf.data.Dataset.from_tensor_slices(tf.constant(tfrecs, dtype=tf.string))
    files = files.with_options(DATASET_OPTIONS)
    ds = files.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=1),
        cycle_length=min(8, len(tfrecs)) if len(tfrecs) > 0 else 1,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.filter(lambda image_name, x: tf.equal(test_id_table.lookup(image_name), 1))
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = _make_test_ds_from_tfrecs(TEST_TFRECS)

all_names = []
all_probs = []

for batch in test_ds:
    image_names, x = batch
    probs_batch = model(x, training=False).numpy()
    all_names.append(image_names.numpy())
    all_probs.append(probs_batch)

all_names = np.concatenate(all_names).astype("U")  # bytes->str
all_probs = np.concatenate(all_probs, axis=0)

preds_by_name = dict(zip(all_names, all_probs.argmax(axis=1).astype(int)))

preds = np.array([preds_by_name[name] for name in test_ids], dtype=int)

submission = pd.DataFrame({"image_id": test_ids, "label": preds})
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())

assert os.path.exists(submission_path), "submission.csv was not created"
sub_check = pd.read_csv(submission_path)
assert list(sub_check.columns) == ["image_id", "label"], "Wrong submission columns"
assert len(sub_check) == len(sample_sub), "Wrong number of rows in submission"
assert sub_check["label"].between(0, 4).all(), "Labels must be in [0,4]"
assert (
    sub_check["image_id"].astype(str).values
    == sample_sub["image_id"].astype(str).values
).all(), "Row order mismatch vs sample_submission"
print("Submission looks valid with shape:", sub_check.shape)
print(sub_check.tail())
