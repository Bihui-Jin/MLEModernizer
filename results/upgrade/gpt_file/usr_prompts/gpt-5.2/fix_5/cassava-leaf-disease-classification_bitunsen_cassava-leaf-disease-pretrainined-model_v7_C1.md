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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")




## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

train_csv_path = os.path.join(BASE_DIR, "train.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"




## === cell 2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF version:", tf.__version__)




## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))




## === cell 4
label_list = [int(key) for key in map_classes.keys()]
print("Labels:", label_list)




## === cell 5
train_df_tmp = pd.read_csv(train_csv_path, usecols=["image_id", "label"])
print(f"Number of train images: {train_df_tmp.shape[0]}")
del train_df_tmp




## === cell 6
IMG_HEIGHT = 400
IMG_WIDTH = 400
batch_size = 32

PRE_TRAINED_MODEL = "../input/xceptionv6/Cassava_Best_Xception_Model_V05.hdf5"
print(
    "Pretrained model exists?",
    os.path.exists(PRE_TRAINED_MODEL),
    "| Path:",
    PRE_TRAINED_MODEL,
)




## === cell 7
AUTOTUNE = tf.data.AUTOTUNE


def _read_decode_resize(image_path):
    img_bytes = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # uint8 [H,W,3]
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.LANCZOS3
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _aug_hflip(img, p=0.5):
    do = tf.random.uniform(()) < p
    return tf.cond(do, lambda: tf.image.flip_left_right(img), lambda: img)


def _aug_rand_brightness(img, limit=0.2, p=0.5):
    do = tf.random.uniform(()) < p

    def _apply():
        delta = tf.random.uniform((), -limit, limit, dtype=tf.float32)
        out = img + delta
        return tf.clip_by_value(out, 0.0, 1.0)

    return tf.cond(do, _apply, lambda: img)


def _aug_rand_contrast(img, limit=0.2, p=0.5):
    do = tf.random.uniform(()) < p

    def _apply():
        factor = 1.0 + tf.random.uniform((), -limit, limit, dtype=tf.float32)
        mean = tf.reduce_mean(img, axis=[0, 1], keepdims=True)
        out = (img - mean) * factor + mean
        return tf.clip_by_value(out, 0.0, 1.0)

    return tf.cond(do, _apply, lambda: img)


def _augment_train_like_original(img):
    do_full = tf.random.uniform(()) > 0.5

    def _full():
        x = img
        x = _aug_hflip(x, p=0.5)
        x = _aug_rand_contrast(x, limit=0.2, p=0.5)
        x = _aug_rand_brightness(x, limit=0.2, p=0.5)
        return x

    return tf.cond(do_full, _full, lambda: img)


def _make_dataset(image_ids, labels, data_dir, mode):
    image_paths = tf.strings.join([data_dir, image_ids])
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(image_paths)
    else:
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if mode == "TRAIN":
        buffer_size = (
            int(min(int(image_ids.shape[0]), 8192))
            if image_ids.shape[0] is not None
            else 8192
        )
        ds = ds.shuffle(
            buffer_size=buffer_size,
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def _map_train(path, y):
        img = _read_decode_resize(path)
        img = _augment_train_like_original(img)
        return img, y

    def _map_val(path, y):
        img = _read_decode_resize(path)
        return img, y

    def _map_test(path):
        img = _read_decode_resize(path)
        return img

    if mode == "TEST":
        ds = ds.map(_map_test, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.cache()
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    if mode == "TRAIN":
        ds = ds.map(_map_train, num_parallel_calls=AUTOTUNE, deterministic=False)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    ds = ds.map(_map_val, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 8
train_df = pd.read_csv(train_csv_path)
print(train_df.head())
print("Train shape:", train_df.shape)
num_classes = train_df["label"].nunique()
print("Num classes:", num_classes)

sample_sub = pd.read_csv(sample_sub_path)
test_df = sample_sub[["image_id"]].copy()
test_samples = test_df.shape[0]
print("Test samples:", test_samples)




## === cell 9
from sklearn.model_selection import train_test_split

train_ids, val_ids, train_y, val_y = train_test_split(
    train_df["image_id"].values,
    train_df["label"].values,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"].values,
)

train_ds = _make_dataset(
    image_ids=tf.constant(train_ids),
    labels=tf.constant(train_y, dtype=tf.int32),
    data_dir=tf.constant(TRAIN_DIR),
    mode="TRAIN",
)
val_ds = _make_dataset(
    image_ids=tf.constant(val_ids),
    labels=tf.constant(val_y, dtype=tf.int32),
    data_dir=tf.constant(TRAIN_DIR),
    mode="VALIDATE",
)
test_ds = _make_dataset(
    image_ids=tf.constant(test_df["image_id"].values),
    labels=None,
    data_dir=tf.constant(TEST_DIR),
    mode="TEST",
)

train_steps = int(np.ceil(len(train_ids) / batch_size))
val_steps = int(np.ceil(len(val_ids) / batch_size))
test_steps = int(np.ceil(len(test_df) / batch_size))
print("Batches - train/val/test:", train_steps, val_steps, test_steps)




## === cell 10
def build_model(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, n_classes=5):
    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = keras.applications.xception.preprocess_input(inputs * 255.0)
    base = keras.applications.Xception(
        include_top=False, weights="imagenet", input_tensor=x
    )
    base.trainable = False  # keep fast and stable

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


model = build_model(n_classes=5)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()




## === cell 11
EPOCHS = 2

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)




## === cell 12
pred_probs = model.predict(test_ds, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)

print("Pred shape:", pred_probs.shape, "Labels shape:", pred_labels.shape)
assert len(pred_labels) == len(test_df), "Prediction length mismatch with test_df"
assert np.all(
    (pred_labels >= 0) & (pred_labels <= 4)
), "Predicted labels out of range 0..4"




## === cell 13
submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": pred_labels}
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head(10))

check = pd.read_csv(submission_path)
print(check.shape)
print(check.head(3))
print("Columns:", list(check.columns))
assert list(check.columns) == ["image_id", "label"]
assert check["image_id"].nunique() == len(check), "Duplicate image_id in submission"
assert check["label"].between(0, 4).all(), "Labels out of expected range 0..4"
assert os.path.exists(submission_path) and submission_path.endswith(".csv")
