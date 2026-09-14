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

0.8856149894227864

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc, random, math, re

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

print("TensorFlow version:", tf.__version__)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)




## === cell 2
BASE1 = "/kaggle/input/cassava-leaf-disease-classification"
BASE2 = "../input/cassava-leaf-disease-classification"

BASE = BASE1 if os.path.exists(BASE1) else BASE2
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

print("Using BASE:", BASE)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))
print("TRAIN_IMG_DIR exists:", os.path.exists(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))




## === cell 3
PRETRAIN_BASE1 = "/kaggle/input/train-model-cassava"
PRETRAIN_BASE2 = "../input/train-model-cassava"
PRETRAIN_BASE = PRETRAIN_BASE1 if os.path.exists(PRETRAIN_BASE1) else PRETRAIN_BASE2

dense_path = os.path.join(PRETRAIN_BASE, "densenet201.h5")
inception_path = os.path.join(PRETRAIN_BASE, "inceptionv3.h5")
eff_path = os.path.join(PRETRAIN_BASE, "efficient_netb3.h5")


def try_load_model(path, custom_objects=None, compile_flag=True):
    if os.path.exists(path):
        print("Loading model:", path)
        return tf.keras.models.load_model(
            path, custom_objects=custom_objects, compile=compile_flag
        )
    print("Model not found (will fallback):", path)
    return None


dense201 = try_load_model(dense_path, compile_flag=True)
inception = try_load_model(inception_path, compile_flag=True)
efficient_net = try_load_model(
    eff_path, custom_objects={"FixedDropout": FixedDropout}, compile_flag=False
)




## === cell 4
submission = pd.read_csv(SAMPLE_SUB)
test_image_ids = np.sort(submission.image_id.values.astype(str))
print("Test rows:", len(submission))




## === cell 5
train_df = pd.read_csv(TRAIN_CSV)
num_classes = train_df["label"].nunique()
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # keep identical to original script

AUTOTUNE = tf.data.AUTOTUNE

data_opts = tf.data.Options()
data_opts.experimental_deterministic = False

try:
    data_opts.experimental_optimization.apply_default_optimizations = True
    data_opts.experimental_optimization.autotune_buffers = True
    data_opts.experimental_optimization.autotune_cpu_budget = True
except Exception:
    pass


@tf.function
def decode_and_resize(path, label=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

train_image_ids = train_df["image_id"].astype(str).values

tr_paths_np = np.char.add(
    np.char.add(TRAIN_IMG_DIR + os.sep, train_image_ids[tr_idx]), ""
)
va_paths_np = np.char.add(
    np.char.add(TRAIN_IMG_DIR + os.sep, train_image_ids[va_idx]), ""
)
test_paths_np = np.char.add(np.char.add(TEST_IMG_DIR + os.sep, test_image_ids), "")

tr_paths = tf.constant(tr_paths_np)
tr_labels = tf.constant(train_df.iloc[tr_idx]["label"].values)
va_paths = tf.constant(va_paths_np)
va_labels = tf.constant(train_df.iloc[va_idx]["label"].values)

ds_train = (
    tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels))
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
).with_options(data_opts)

ds_val = (
    tf.data.Dataset.from_tensor_slices((va_paths, va_labels))
    .map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
).with_options(data_opts)

test_paths = tf.constant(test_paths_np)

ds_test = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .cache()  # in-memory cache
    .prefetch(AUTOTUNE)
).with_options(data_opts)

TEST_STEPS = int(math.ceil(len(submission) / BATCH_SIZE))


def build_small_cnn(seed_offset=0):
    tf.keras.utils.set_random_seed(SEED + seed_offset)
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


fallback_models = []
if dense201 is None:
    dense201 = build_small_cnn(seed_offset=1)
    fallback_models.append(("dense201", dense201))
if inception is None:
    inception = build_small_cnn(seed_offset=2)
    fallback_models.append(("inception", inception))
if efficient_net is None:
    efficient_net = build_small_cnn(seed_offset=3)
    fallback_models.append(("efficient_net", efficient_net))

for name, m in fallback_models:
    print(f"Training fallback model: {name}")
    m.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, verbose=2)

gc.collect()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/66402182.py in <cell line: 0>()
     44 # np.char.add is the safe vectorized string concatenation.
     45 tr_paths_np = np.char.add(
---> 46     np.char.add(TRAIN_IMG_DIR + os.sep, train_image_ids[tr_idx]), ""
     47 )
     48 va_paths_np = np.char.add(

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U63' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 6
def vote_in_ensemble(v1, v2, v3):
    if v1 == v2:
        return v1
    if v2 == v3:
        return v2
    if v1 == v3:
        return v3
    return v1


def predict_for_pretrained(model, ds_test, steps):
    probs = model.predict(ds_test, steps=steps, verbose=0)
    preds = np.argmax(probs, axis=-1).astype(int)
    return preds


_ = next(iter(ds_test))

dense_preds = predict_for_pretrained(dense201, ds_test, TEST_STEPS)
inception_preds = predict_for_pretrained(inception, ds_test, TEST_STEPS)
efficient_net_preds = predict_for_pretrained(efficient_net, ds_test, TEST_STEPS)

n = len(submission)
dense_preds = dense_preds[:n]
inception_preds = inception_preds[:n]
efficient_net_preds = efficient_net_preds[:n]

v1 = dense_preds
v2 = inception_preds
v3 = efficient_net_preds
result = np.where(
    v1 == v2, v1, np.where(v2 == v3, v2, np.where(v1 == v3, v3, v1))
).astype(np.int64)

submission["label"] = result
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3714261001.py in <cell line: 0>()
     16 
     17 # Warm cache once so subsequent predicts reuse cached tensors (score-neutral; speed/robustness only).
---> 18 _ = next(iter(ds_test))
     19 
     20 dense_preds = predict_for_pretrained(dense201, ds_test, TEST_STEPS)

NameError: name 'ds_test' is not defined
