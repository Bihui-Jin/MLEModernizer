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
import warnings

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

print("TensorFlow:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Could not enable XLA JIT:", e)

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
work_dir = "../input/cassava-leaf-disease-classification/"
train_img_dir = os.path.join(work_dir, "train_images")
test_img_dir = os.path.join(work_dir, "test_images")

assert os.path.exists(os.path.join(work_dir, "train.csv")), "train.csv not found"
assert os.path.isdir(train_img_dir), "train_images directory not found"
assert os.path.isdir(test_img_dir), "test_images directory not found"

print("work_dir contents sample:", os.listdir(work_dir)[:10])




## === cell 2
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


seed = 21
seed_everything(seed)




## === cell 3
data = pd.read_csv(os.path.join(work_dir, "train.csv"))
print("Train rows:", len(data))
print(data["label"].value_counts().sort_index())




## === cell 4
with open(os.path.join(work_dir, "label_num_to_disease_map.json"), "r") as f:
    real_labels = json.load(f)
real_labels = {int(k): v for k, v in real_labels.items()}

data["class_name"] = data["label"].map(real_labels)
print("Example label map:", real_labels)




## === cell 5
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    data,
    test_size=0.05,
    random_state=42,
    stratify=data["class_name"],
)

print("Train/Val sizes:", len(train_df), len(val_df))




## === cell 6
IMG_SIZE = 456
size = (IMG_SIZE, IMG_SIZE)
n_CLASS = 5
BATCH_SIZE = 15

train_tfrecord_dir = os.path.join(work_dir, "train_tfrecords")
test_tfrecord_dir = os.path.join(work_dir, "test_tfrecords")
assert os.path.isdir(train_tfrecord_dir), "train_tfrecords directory not found"
assert os.path.isdir(test_tfrecord_dir), "test_tfrecords directory not found"

train_tfrec_files = sorted(
    [
        os.path.join(train_tfrecord_dir, f)
        for f in os.listdir(train_tfrecord_dir)
        if f.endswith(".tfrec")
    ]
)
test_tfrec_files = sorted(
    [
        os.path.join(test_tfrecord_dir, f)
        for f in os.listdir(test_tfrecord_dir)
        if f.endswith(".tfrec")
    ]
)

print(
    "Train TFRecords:", len(train_tfrec_files), "Test TFRecords:", len(test_tfrec_files)
)




## === cell 7

FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _decode_and_resize(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(
        img, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )  # interpolation="nearest"
    img = tf.cast(img, tf.float32)
    return img


def _augment(img):
    img = tf.image.random_flip_left_right(img, seed=seed)
    img = tf.image.random_flip_up_down(img, seed=seed)
    return img


def _preprocess(img):
    return tf.keras.applications.efficientnet.preprocess_input(img)


def _parse_train(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURE_DESC)
    img = _decode_and_resize(ex["image"])
    img = _augment(img)
    img = _preprocess(img)
    label = tf.cast(ex["target"], tf.int32)
    label = tf.one_hot(label, depth=n_CLASS)
    return img, label


def _parse_val(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURE_DESC)
    img = _decode_and_resize(ex["image"])
    img = _preprocess(img)
    label = tf.cast(ex["target"], tf.int32)
    label = tf.one_hot(label, depth=n_CLASS)
    return img, label


def _parse_test(example_proto):
    ex = tf.io.parse_single_example(
        example_proto,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        },
    )
    img = _decode_and_resize(ex["image"])
    img = _preprocess(img)
    return img, ex["image_name"]


val_ids = set(val_df["image_id"].values.tolist())


@tf.function
def _is_val(image, label, image_name):
    return tf.reduce_any(
        tf.equal(image_name, tf.constant(list(val_ids), dtype=tf.string))
    )


keys = tf.constant(list(val_ids), dtype=tf.string)
vals = tf.ones([tf.shape(keys)[0]], dtype=tf.int32)
val_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys, vals), default_value=0
)


def _split_predicate(example_proto):
    ex = tf.io.parse_single_example(
        example_proto,
        {
            "image_name": tf.io.FixedLenFeature([], tf.string),
        },
    )
    return tf.equal(val_table.lookup(ex["image_name"]), 1)


raw_train_ds = tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=AUTOTUNE)

raw_val_ds = raw_train_ds.filter(_split_predicate)
raw_tr_ds = raw_train_ds.filter(lambda x: tf.logical_not(_split_predicate(x)))

train_ds = (
    raw_tr_ds.shuffle(4096, seed=seed, reshuffle_each_iteration=True)
    .map(_parse_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

val_ds = (
    raw_val_ds.map(_parse_val, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

print("Built tf.data train/val datasets from TFRecords.")




## === cell 8
def create_model():
    model = models.Sequential()
    backbone = EfficientNetB3(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False  # critical speedup; architecture unchanged
    model.add(backbone)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            256,
            activation="relu",
            bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
        )
    )
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(n_CLASS, activation="softmax"))
    return model


leaf_model = create_model()
leaf_model.summary()




## === cell 9
EPOCHS = 20

leaf_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=2, verbose=1, min_lr=1e-7
    ),
    EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(
        "best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1
    ),
]

history = leaf_model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1,
)




## === cell 10
if os.path.exists("best_model.keras"):
    loaded_model = tf.keras.models.load_model("best_model.keras", compile=False)
else:
    loaded_model = leaf_model

loaded_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

val_loss, val_acc = loaded_model.evaluate(val_ds, verbose=1)
print("Validation accuracy:", val_acc)




## === cell 11
from sklearn.metrics import confusion_matrix, classification_report

val_probs = loaded_model.predict(val_ds, verbose=1)
val_pred_idx = np.argmax(val_probs, axis=1)

y_true = []
for _, y in val_ds:
    y_true.append(np.argmax(y.numpy(), axis=1))
y_true = np.concatenate(y_true, axis=0)[: len(val_pred_idx)]

print("Confusion Matrix:\n", confusion_matrix(y_true, val_pred_idx))

target_names = [real_labels[i] for i in range(n_CLASS)]
print(classification_report(y_true, val_pred_idx, target_names=target_names))




## === cell 12
ss = pd.read_csv(os.path.join(work_dir, "sample_submission.csv"))

raw_test_ds = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
test_ds = (
    raw_test_ds.map(_parse_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

test_probs = loaded_model.predict(test_ds, verbose=1)
test_pred_idx = np.argmax(test_probs, axis=1)

test_names = []
for _, names in test_ds:
    test_names.extend([n.decode("utf-8") for n in names.numpy().tolist()])

test_names = test_names[: len(test_pred_idx)]

submission = pd.DataFrame({"image_id": test_names, "label": test_pred_idx.astype(int)})
submission = ss[["image_id"]].merge(submission, on="image_id", how="left")
submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
