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
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

CANDIDATE_ROOTS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../data/cassava-leaf-disease-classification",
]
DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.exists(r):
        DATA_ROOT = r
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification dataset folder in expected locations."
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

print("Using DATA_ROOT:", DATA_ROOT)
print("TensorFlow:", tf.__version__)

AUTOTUNE = tf.data.AUTOTUNE
tf.config.optimizer.set_jit(False)  # keep deterministic-ish behavior closer to baseline

df_train = pd.read_csv(TRAIN_CSV)
df_train["path"] = (TRAIN_IMG_DIR + "/" + df_train["image_id"].astype(str)).astype(str)
df_train["label"] = df_train["label"].astype(np.int32)

IMG_SIZE = (300, 300)
BATCH_SIZE = 32
NUM_CLASSES = 5
classes = [str(i) for i in range(NUM_CLASSES)]


def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


def _one_hot(label):
    return tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)


def _augment(img, seed):
    s1 = tf.stack([seed, seed ^ 0x1234567])

    r = tf.random.stateless_uniform([], s1, 0, 1, dtype=tf.float32)
    img = tf.cond(r < 0.5, lambda: tf.image.flip_left_right(img), lambda: img)

    s2 = tf.stack([seed ^ 0x2345678, seed ^ 0x3456789])
    zoom = tf.random.stateless_uniform([], s2, 0.9, 1.1, dtype=tf.float32)
    new_size = tf.cast(tf.round(tf.cast(IMG_SIZE[0], tf.float32) * zoom), tf.int32)
    img = tf.image.resize(
        img, (new_size, new_size), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE[0], IMG_SIZE[1])

    s3 = tf.stack([seed ^ 0x456789A, seed ^ 0x56789AB])
    dx = tf.random.stateless_uniform([], s3, -0.05, 0.05, dtype=tf.float32) * tf.cast(
        IMG_SIZE[1], tf.float32
    )
    s4 = tf.stack([seed ^ 0x6789ABC, seed ^ 0x789ABCD])
    dy = tf.random.stateless_uniform([], s4, -0.05, 0.05, dtype=tf.float32) * tf.cast(
        IMG_SIZE[0], tf.float32
    )

    dx_i = tf.cast(tf.round(dx), tf.int32)
    dy_i = tf.cast(tf.round(dy), tf.int32)
    img = tf.roll(img, shift=[dy_i, dx_i], axis=[0, 1])

    s5 = tf.stack([seed ^ 0x89ABCDE, seed ^ 0x9ABCDEF])
    angle = tf.random.stateless_uniform([], s5, -10.0, 10.0, dtype=tf.float32) * (
        math.pi / 180.0
    )
    if hasattr(tf.image, "rotate"):
        img = tf.image.rotate(
            img, angles=angle, interpolation="BILINEAR", fill_mode="REFLECT"
        )
    return img


def _preprocess(img):
    return preprocess_input(img)


idx = np.arange(len(df_train), dtype=np.int32)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_size = int(round(0.1 * len(idx)))
val_idx = idx[:val_size]
train_idx = idx[val_size:]

train_paths = df_train["path"].to_numpy()[train_idx]
train_labels = df_train["label"].to_numpy()[train_idx]
valid_paths = df_train["path"].to_numpy()[val_idx]
valid_labels = df_train["label"].to_numpy()[val_idx]


def make_train_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    shuffle_buf = min(len(paths), 4096)
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)

    def _decode_map(path, label):
        img = _decode_and_resize(path)
        img = _preprocess(img)
        return path, img, label

    ds = ds.map(_decode_map, num_parallel_calls=AUTOTUNE)

    ds = ds.cache()

    def _aug_map(path, img, label):
        ph = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        seed = tf.cast(ph, tf.int32) ^ tf.cast(SEED, tf.int32)
        img = _augment(img, seed)
        y = _one_hot(label)
        return img, y

    ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    ds = ds.with_options(opts)

    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_valid_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(path, label):
        img = _decode_and_resize(path)
        img = _preprocess(img)
        y = _one_hot(label)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    ds = ds.with_options(opts)

    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(train_paths, train_labels, BATCH_SIZE)
valid_ds = make_valid_ds(valid_paths, valid_labels, BATCH_SIZE)

steps_per_epoch = int(math.ceil(len(train_paths) / BATCH_SIZE))
validation_steps = int(math.ceil(len(valid_paths) / BATCH_SIZE))

print("Train/Valid sizes:", len(train_paths), len(valid_paths))
print("steps_per_epoch / validation_steps:", steps_per_epoch, validation_steps)



## === cell 1
base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_tensor=Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
)
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3, seed=SEED)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs=base.input, outputs=out)

base.trainable = False

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)

EPOCHS_HEAD = 2 if DEBUG else 4
history1 = my_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_HEAD,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)

EPOCHS_FT = 1 if DEBUG else 2
history2 = my_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_FT,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 2
sample_sub = pd.read_csv(SAMPLE_SUB)
df_test = sample_sub.copy()
df_test["path"] = (TEST_IMG_DIR + "/" + df_test["image_id"].astype(str)).astype(str)

test_paths_np = df_test["path"].to_numpy()


def make_test_ds(paths, batch_size=128):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(path):
        img = _decode_and_resize(path)
        img = _preprocess(img)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    ds = ds.with_options(opts)

    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(test_paths_np, batch_size=128)



## === cell 3
pred_test = my_model.predict(test_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = pd.DataFrame(
    {"image_id": df_test["image_id"].values, "label": pred_test_labels}
)

final_csv.to_csv("submission.csv", index=False)
print(final_csv.head())
print("Wrote submission.csv with shape:", final_csv.shape)

sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == ["image_id", "label"]
assert len(sub_check) == len(pd.read_csv(SAMPLE_SUB))
assert sub_check["label"].between(0, 4).all()
sub_check.head()
