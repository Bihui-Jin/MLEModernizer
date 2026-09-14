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

3.11

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

0.7446358416440012

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.27765) has done: 'I remove the `tensorflow_hub` import that is crashing under this environment and replace the missing external weight loading (`/kaggle/input/computer-vision-starter/cassava_base.h5`) with a standard built-in ResNet50-based classifier so `my_model` is always defined. This keeps the core approach the same (ResNet50 preprocessing + Keras ImageDataGenerator + `model.predict` + argmax labels) while making the notebook run end-to-end and write `submission.csv` in the required format. I also make the test image path robust to your filesystem (it exists under `/kaggle/data/...` here) so the generator actually finds images. These changes are directly required to fix the runtime errors and produce a valid submission; they should also yield a non-trivial accuracy compared to a broken pipeline.'
- What this solution (achieved 0.27765) has done: 'The crash happens before any model code runs: TensorFlow’s import triggers a protobuf incompatibility (`MessageFactory.GetPrototype`) in this Python 3.11 environment. The minimal, robust fix is to force TensorFlow to use the pure-Python protobuf implementation **before** importing `tensorflow`, which avoids the problematic compiled protobuf API. After that, the rest of your pipeline can run unchanged (same ResNet50 + ImageDataGenerator + predict + argmax). I also keep your dataset path discovery and submission formatting intact so it reliably writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os, random, glob, sys

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
DATA_ROOT = next((p for p in DATA_ROOT_CANDIDATES if os.path.exists(p)), None)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate cassava dataset directory. Tried:\n"
        + "\n".join(DATA_ROOT_CANDIDATES)
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"train.csv not found at: {TRAIN_CSV}")
if not os.path.isdir(TRAIN_IMG_DIR):
    raise FileNotFoundError(f"train_images directory not found at: {TRAIN_IMG_DIR}")
if not os.path.isdir(TEST_IMG_DIR):
    raise FileNotFoundError(f"test_images directory not found at: {TEST_IMG_DIR}")

print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV:", TRAIN_CSV)
print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("TEST_IMG_DIR:", TEST_IMG_DIR)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import matplotlib.pyplot as plt  # kept (even if unused) to preserve original intent
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

preprocess = tf.keras.applications.resnet50.preprocess_input



## === cell 2
num_classes = 5

base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(256, 256, 3),
    pooling="avg",
)

base.trainable = False

x = tfl.Dropout(0.2)(base.output)
out = tfl.Dense(num_classes, activation="softmax")(x)
my_model = Model(inputs=base.input, outputs=out)

my_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

my_model.summary()



## === cell 3
df_train = pd.read_csv(TRAIN_CSV)
if not {"image_id", "label"}.issubset(df_train.columns):
    raise ValueError(f"Unexpected train.csv columns: {df_train.columns.tolist()}")

df_train["label"] = df_train["label"].astype(np.int32)
df_train["path"] = (TRAIN_IMG_DIR.rstrip("/") + "/") + df_train["image_id"].astype(str)

paths = df_train["path"].to_numpy(dtype=object)
exists_mask = np.fromiter(
    (tf.io.gfile.exists(p) for p in paths), dtype=np.bool_, count=len(paths)
)
missing = int((~exists_mask).sum())
if missing > 0:
    raise FileNotFoundError(
        f"{missing} training images referenced in train.csv were not found under {TRAIN_IMG_DIR}"
    )

train_df, val_df = train_test_split(
    df_train[["path", "label"]],
    test_size=0.1,
    random_state=SEED,
    stratify=df_train["label"],
)

BATCH_SIZE = 16
EPOCHS = 3
IMG_SIZE = (256, 256)
AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def _read_bytes(path):
    return tf.io.read_file(path)


@tf.function
def _decode_resize_preprocess_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess(img)
    return img


@tf.function
def _decode_resize_preprocess(path):
    img_bytes = _read_bytes(path)
    return _decode_resize_preprocess_from_bytes(img_bytes)


@tf.function
def _augment(img, seed2):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed2)

    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    max_dx = tf.cast(0.05 * w, tf.int32)
    max_dy = tf.cast(0.05 * h, tf.int32)

    dx = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([11, 17], tf.int32),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([19, 23], tf.int32),
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    z = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([29, 31], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )
    z = tf.clip_by_value(z, 0.9, 1.1)
    new_h = tf.cast(tf.round(h / z), tf.int32)
    new_w = tf.cast(tf.round(w / z), tf.int32)
    new_h = tf.minimum(new_h, tf.shape(img)[0])
    new_w = tf.minimum(new_w, tf.shape(img)[1])

    img = tf.image.stateless_random_crop(
        img, size=[new_h, new_w, 3], seed=seed2 + tf.constant([37, 41], tf.int32)
    )
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    return img


def _with_fast_deterministic_options(ds):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.map_fusion = True
    opts.experimental_optimization.parallel_batch = True
    opts.experimental_optimization.autotune_buffers = True
    opts.experimental_optimization.autotune_cpu_budget = True
    opts.experimental_optimization.autotune_ram_budget = True
    return ds.with_options(opts)


def make_train_ds(df, epoch):
    path_np = df["path"].to_numpy(dtype=object)
    label_np = df["label"].to_numpy(dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((path_np, label_np))
    shuffle_buf = min(len(df), 8192)
    ds = ds.shuffle(
        buffer_size=shuffle_buf, seed=SEED + int(epoch), reshuffle_each_iteration=True
    )

    def _load_bytes(path, label):
        return path, _read_bytes(path), label

    ds = ds.map(_load_bytes, num_parallel_calls=AUTOTUNE, deterministic=True).cache()

    def _decode_and_aug(path, img_bytes, label):
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)

        seed1 = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        seed2 = tf.stack(
            [tf.cast(seed1, tf.int32), tf.cast(SEED + int(epoch), tf.int32)], axis=0
        )
        img = _augment(img, seed2)

        img = tf.cast(img, tf.float32)
        img = preprocess(img)
        return img, label

    ds = ds.map(_decode_and_aug, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _with_fast_deterministic_options(ds)
    return ds


def make_val_ds(df):
    path_np = df["path"].to_numpy(dtype=object)
    label_np = df["label"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((path_np, label_np))

    def _load(path, label):
        img = _decode_resize_preprocess(path)
        return img, label

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _with_fast_deterministic_options(ds)
    return ds


val_ds = make_val_ds(val_df)

for epoch in range(EPOCHS):
    train_ds = make_train_ds(train_df, epoch=epoch)
    my_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epoch + 1,
        initial_epoch=epoch,
        verbose=1,
    )



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2791191640.py in <cell line: 0>()
    170 
    171 
--> 172 val_ds = make_val_ds(val_df)
    173 
    174 # CHANGE (timeout fix): use the same training approach/epochs, but feed epoch-specific datasets.

/tmp/ipykernel_11/2791191640.py in make_val_ds(df)
    166     ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    167     ds = ds.prefetch(AUTOTUNE)
--> 168     ds = _with_fast_deterministic_options(ds)
    169     return ds
    170 

/tmp/ipykernel_11/2791191640.py in _with_fast_deterministic_options(ds)
    107     opts.experimental_optimization.parallel_batch = True
    108     # CHANGE (timeout fix): enable additional graph optimizations for tf.data (semantics-preserving).
--> 109     opts.experimental_optimization.autotune_buffers = True
    110     opts.experimental_optimization.autotune_cpu_budget = True
    111     opts.experimental_optimization.autotune_ram_budget = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 4
test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
if len(test_images) == 0:
    raise FileNotFoundError(f"No test images found under: {TEST_IMG_DIR}")

df_test = pd.DataFrame(test_images, columns=["path"])


def make_test_ds(paths, batch_size=16):
    ds = tf.data.Dataset.from_tensor_slices(np.asarray(paths, dtype=object))
    ds = ds.map(
        _decode_resize_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _with_fast_deterministic_options(ds)
    return ds




## === cell 5
pred_list = []

test_ds = make_test_ds(df_test["path"].to_numpy(dtype=object), batch_size=16)
for i in range(1):
    pred_test = my_model.predict(test_ds, verbose=1)
    pred_list.append(pred_test)

pred_test = np.mean(pred_list, axis=0)
pred_test_labels = np.argmax(pred_test, axis=-1)

paths_test = df_test["path"].to_numpy(dtype=object)
image_ids = np.array([os.path.basename(p) for p in paths_test], dtype=object)

final_csv = pd.DataFrame({"image_id": image_ids, "label": pred_test_labels.astype(int)})
final_csv.to_csv("submission.csv", index=False)

final_csv.head()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2073896713.py in <cell line: 0>()
      1 pred_list = []
      2 
----> 3 test_ds = make_test_ds(df_test["path"].to_numpy(dtype=object), batch_size=16)
      4 for i in range(1):
      5     pred_test = my_model.predict(test_ds, verbose=1)

/tmp/ipykernel_11/3554257801.py in make_test_ds(paths, batch_size)
     13     ds = ds.batch(batch_size, drop_remainder=False)
     14     ds = ds.prefetch(AUTOTUNE)
---> 15     ds = _with_fast_deterministic_options(ds)
     16     return ds
     17 

/tmp/ipykernel_11/2791191640.py in _with_fast_deterministic_options(ds)
    107     opts.experimental_optimization.parallel_batch = True
    108     # CHANGE (timeout fix): enable additional graph optimizations for tf.data (semantics-preserving).
--> 109     opts.experimental_optimization.autotune_buffers = True
    110     opts.experimental_optimization.autotune_cpu_budget = True
    111     opts.experimental_optimization.autotune_ram_budget = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 6
print("Submission shape:", final_csv.shape)
print("Saved to:", os.path.abspath("submission.csv"))
print(final_csv.dtypes)
final_csv.head()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/644082253.py in <cell line: 0>()
----> 1 print("Submission shape:", final_csv.shape)
      2 print("Saved to:", os.path.abspath("submission.csv"))
      3 print(final_csv.dtypes)
      4 final_csv.head()

NameError: name 'final_csv' is not defined
