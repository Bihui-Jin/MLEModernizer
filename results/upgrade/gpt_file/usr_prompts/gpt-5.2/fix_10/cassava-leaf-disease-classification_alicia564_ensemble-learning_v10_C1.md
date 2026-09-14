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

3.13

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
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass

options = None



## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.efficientnet import preprocess_input

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)

train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)

train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"].astype(str)
)

train_csv["label_encoded"] = LabelEncoder().fit_transform(train_csv["disease"])
train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(int)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=SEED
)
train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)

classes = sorted(train_csv["disease"].unique().tolist())
class_to_index = {c: i for i, c in enumerate(classes)}
NUM_CLASSES = len(classes)

train_paths = train["path"].to_numpy()
valid_paths = valid["path"].to_numpy()

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

try:
    import tensorflow_addons as tfa  # may not exist; use only if available
except Exception:
    tfa = None

TRAIN_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords"
train_tfrecs = tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
train_tfrecs = sorted(train_tfrecs)

_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _decode_jpeg_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def _seed_from_key(key):
    h = tf.strings.to_hash_bucket_fast(key, 2**31 - 1)
    return tf.stack([tf.cast(SEED, tf.int32), tf.cast(h, tf.int32)])


@tf.function
def _augment(img, seed2):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed2)
    img = tf.image.stateless_random_flip_up_down(img, seed=seed2 + 1)

    if tfa is not None:
        angle = tf.random.stateless_uniform(
            [], seed=seed2 + 2, minval=-45.0, maxval=45.0
        ) * (np.pi / 180.0)
        img = tfa.image.rotate(
            img, angles=angle, fill_mode="nearest", interpolation="BILINEAR"
        )
        shear = tf.random.stateless_uniform([], seed=seed2 + 3, minval=-0.2, maxval=0.2)
        zoom = tf.random.stateless_uniform([], seed=seed2 + 4, minval=0.8, maxval=1.2)

        a0 = zoom
        a1 = shear
        b0 = 0.0
        b1 = zoom
        a2 = 0.0
        b2 = 0.0
        transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])
        img = tfa.image.transform(
            img, transform, fill_mode="nearest", interpolation="BILINEAR"
        )
    else:
        z = tf.random.stateless_uniform([], seed=seed2 + 4, minval=0.8, maxval=1.2)
        h = tf.cast(tf.shape(img)[0], tf.float32)
        w = tf.cast(tf.shape(img)[1], tf.float32)
        new_h = tf.cast(h / z, tf.int32)
        new_w = tf.cast(w / z, tf.int32)
        img = tf.image.resize_with_crop_or_pad(img, new_h, new_w)
        img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    max_dx = tf.cast(0.2 * tf.cast(IMG_SIZE[1], tf.float32), tf.int32)
    max_dy = tf.cast(0.2 * tf.cast(IMG_SIZE[0], tf.float32), tf.int32)
    dx = tf.random.stateless_uniform(
        [], seed=seed2 + 5, minval=-max_dx, maxval=max_dx + 1, dtype=tf.int32
    )
    dy = tf.random.stateless_uniform(
        [], seed=seed2 + 6, minval=-max_dy, maxval=max_dy + 1, dtype=tf.int32
    )

    pad_x = tf.abs(dx)
    pad_y = tf.abs(dy)
    img = tf.pad(img, [[pad_y, pad_y], [pad_x, pad_x], [0, 0]], mode="SYMMETRIC")
    start_y = pad_y - dy
    start_x = pad_x - dx
    img = tf.image.crop_to_bounding_box(img, start_y, start_x, IMG_SIZE[0], IMG_SIZE[1])
    return img


@tf.function
def _preprocess(img):
    return preprocess_input(img)


@tf.function
def _decode_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


def _parse_tfrec(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = _decode_jpeg_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    key = ex["image_name"]
    return img, label, key


def _make_label_lookup_from_df(df):
    keys = tf.constant(df["image_id"].astype(str).values)
    vals = tf.constant(df["disease"].map(class_to_index).astype(np.int32).values)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals),
        default_value=tf.constant(-1, dtype=tf.int32),
    )
    return table


_train_label_table = _make_label_lookup_from_df(train)
_valid_label_table = _make_label_lookup_from_df(valid)


def make_train_ds_from_tfrecs(tfrec_files):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.shuffle(buffer_size=2048, seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(img, label_num, key):
        y = _train_label_table.lookup(key)
        return img, y, key

    ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)
    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.filter(lambda img, y, key: y >= 0)

    def _finalize(img, y, key):
        seed2 = _seed_from_key(key)
        img = _augment(img, seed2)
        img = _preprocess(img)
        y = tf.one_hot(y, NUM_CLASSES)
        return img, y

    ds = ds.map(_finalize, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.repeat()
    return ds if options is None else ds.with_options(options)


def make_valid_ds_from_tfrecs(tfrec_files):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)

    def _map_fn(img, label_num, key):
        y = _valid_label_table.lookup(key)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.filter(lambda img, y: y >= 0)

    def _finalize(img, y):
        img = _preprocess(img)
        y = tf.one_hot(y, NUM_CLASSES)
        return img, y

    ds = ds.map(_finalize, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds if options is None else ds.with_options(options)


train_ds = make_train_ds_from_tfrecs(train_tfrecs)
valid_ds = make_valid_ds_from_tfrecs(train_tfrecs)

import math

STEPS_PER_EPOCH = int(math.ceil(len(train_paths) / BATCH_SIZE))
VALIDATION_STEPS = int(math.ceil(len(valid_paths) / BATCH_SIZE))



## === cell 2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)



## === cell 3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=outputs)

base_model.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    epochs=5,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)

base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds,
    validation_data=valid_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    epochs=3,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)

disease_to_index = class_to_index
index_to_disease = {v: k for k, v in disease_to_index.items()}
disease_to_labelnum = (
    train_csv[["disease", "label"]]
    .drop_duplicates()
    .set_index("disease")["label"]
    .to_dict()
)
index_to_labelnum = {
    i: int(disease_to_labelnum[index_to_disease[i]]) for i in range(NUM_CLASSES)
}



## === cell 4
import os
import numpy as np
import pandas as pd

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

test_df = sample_sub.copy()
test_df["path"] = image_dir.rstrip("/") + "/" + test_df["image_id"].astype(str)
test_paths = test_df["path"].to_numpy()

TEST_BATCH_SIZE = 128


def make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    @tf.function
    def _map_fn(path):
        img = _decode_image(path)
        img = _preprocess(img)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(TEST_BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds if options is None else ds.with_options(options)


test_ds = make_test_ds(test_paths)

probs = model.predict(test_ds, verbose=1)
pred_indices = np.argmax(probs, axis=1).astype(int)
pred_labels = [int(index_to_labelnum[i]) for i in pred_indices]

submission_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": pred_labels}
)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
print(submission_df.head())
print("Submission shape:", submission_df.shape)
print("Submission columns:", submission_df.columns.tolist())
