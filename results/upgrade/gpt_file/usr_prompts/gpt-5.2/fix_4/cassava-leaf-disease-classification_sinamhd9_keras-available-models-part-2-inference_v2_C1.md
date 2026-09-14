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

0.838319734058628

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.random.set_seed(SEED)

print("TF version:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

assert set(["image_id", "label"]).issubset(train_df.columns)
assert set(["image_id", "label"]).issubset(sample_df.columns)

num_classes = int(train_df["label"].nunique())
print("Train rows:", len(train_df), "Num classes:", num_classes)
print("Sample submission rows:", len(sample_df))



## === cell 3
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # keep runtime within limits

val_frac = 0.1
train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_size = int(len(train_df) * val_frac)
val_df = train_df.iloc[:val_size].reset_index(drop=True)
trn_df = train_df.iloc[val_size:].reset_index(drop=True)

print("Train split:", len(trn_df), "Val split:", len(val_df))


@tf.function
def _load_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # ensure RGB
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.image.convert_image_dtype(img, tf.float32)  # /255.0
    return img


@tf.function
def _map_train(p, y):
    x = _load_image(p)
    x = tf.image.random_flip_left_right(x, seed=SEED)
    x = tf.image.random_flip_up_down(x, seed=SEED)
    return x, y


@tf.function
def _map_eval(p, y):
    x = _load_image(p)
    return x, y


_DATA_OPTS = tf.data.Options()
_DATA_OPTS.experimental_deterministic = True
_DATA_OPTS.autotune.enabled = True
_DATA_OPTS.experimental_optimization.apply_default_optimizations = True
_DATA_OPTS.experimental_optimization.map_fusion = True
_DATA_OPTS.experimental_optimization.parallel_batch = True


def make_ds(df, training=True, cache_name=None, img_dir=TRAIN_IMG_DIR):
    image_ids = df["image_id"].astype(str).to_numpy()
    paths_np = np.char.add(img_dir + os.sep, image_ids).astype("U")
    paths = tf.convert_to_tensor(paths_np)
    labels = tf.convert_to_tensor(df["label"].to_numpy(np.int32), dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_DATA_OPTS)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(_map_train, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        if cache_name is not None:
            ds = ds.cache(cache_name)
    else:
        ds = ds.map(_map_eval, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        if cache_name is not None:
            ds = ds.cache(cache_name)

    ds = ds.batch(BATCH_SIZE, drop_remainder=training).prefetch(tf.data.AUTOTUNE)
    return ds


trn_cache_path = os.path.join("/kaggle/working", "train_cache.tfdata")
val_cache_path = os.path.join("/kaggle/working", "val_cache.tfdata")

train_ds = make_ds(
    trn_df, training=True, cache_name=trn_cache_path, img_dir=TRAIN_IMG_DIR
)
val_ds = make_ds(
    val_df, training=False, cache_name=val_cache_path, img_dir=TRAIN_IMG_DIR
)

base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False  # fast, stable

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = inputs
x = layers.Lambda(
    lambda t: keras.applications.efficientnet.preprocess_input(t * 255.0)
)(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2, seed=SEED)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3795917134.py in <cell line: 0>()
     80 val_cache_path = os.path.join("/kaggle/working", "val_cache.tfdata")
     81 
---> 82 train_ds = make_ds(
     83     trn_df, training=True, cache_name=trn_cache_path, img_dir=TRAIN_IMG_DIR
     84 )

/tmp/ipykernel_11/3795917134.py in make_ds(df, training, cache_name, img_dir)
     51     # --- Correctness: identical strings/labels.
     52     image_ids = df["image_id"].astype(str).to_numpy()
---> 53     paths_np = np.char.add(img_dir + os.sep, image_ids).astype("U")
     54     paths = tf.convert_to_tensor(paths_np)
     55     labels = tf.convert_to_tensor(df["label"].to_numpy(np.int32), dtype=tf.int32)

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U63' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 4
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    verbose=1,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/703340344.py in <cell line: 0>()
----> 1 base.trainable = True
      2 for layer in base.layers[:-20]:
      3     layer.trainable = False
      4 
      5 model.compile(

NameError: name 'base' is not defined

## === cell 5
test_image_ids = sample_df["image_id"].astype(str).tolist()

test_paths_np = np.char.add(
    (TEST_IMG_DIR + os.sep), np.asarray(test_image_ids, dtype="U")
).astype("U")
test_paths = tf.convert_to_tensor(test_paths_np)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.with_options(_DATA_OPTS)


@tf.function
def _map_test(p):
    x = _load_image(p)
    return x


test_ds = test_ds.map(
    _map_test, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
)
test_cache_path = os.path.join("/kaggle/working", "test_cache.tfdata")
test_ds = test_ds.cache(test_cache_path)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

probs = model.predict(test_ds, verbose=1)
y_preds = np.argmax(probs, axis=1).astype(int)

print("Preds:", y_preds.shape, "Expected:", len(test_image_ids))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2743286351.py in <cell line: 0>()
     25 test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
     26 
---> 27 probs = model.predict(test_ds, verbose=1)
     28 y_preds = np.argmax(probs, axis=1).astype(int)
     29 

NameError: name 'model' is not defined

## === cell 6
df_sub = pd.DataFrame({"image_id": test_image_ids, "label": y_preds})
df_sub.head()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2795901055.py in <cell line: 0>()
----> 1 df_sub = pd.DataFrame({"image_id": test_image_ids, "label": y_preds})
      2 df_sub.head()
      3 

NameError: name 'y_preds' is not defined

## === cell 7
assert len(df_sub) == len(sample_df), "Submission length must match sample_submission"
assert list(df_sub.columns) == ["image_id", "label"]
assert df_sub["label"].between(0, num_classes - 1).all()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/174739048.py in <cell line: 0>()
----> 1 assert len(df_sub) == len(sample_df), "Submission length must match sample_submission"
      2 assert list(df_sub.columns) == ["image_id", "label"]
      3 assert df_sub["label"].between(0, num_classes - 1).all()
      4 

NameError: name 'df_sub' is not defined

## === cell 8
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3924369527.py in <cell line: 0>()
----> 1 df_sub.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", df_sub.shape)
      3 print(df_sub.head())

NameError: name 'df_sub' is not defined
