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

0.1403747355696585

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
TRAIN_IMG_LOC = "/kaggle/input/cassava-leaf-disease-classification/train_images"
TRAIN_CSV = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
SAMPLE_CSV = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
TEST_IMG_LOC = "/kaggle/input/cassava-leaf-disease-classification/test_images"

MODELS_WEIGHTS = "/kaggle/input/cassavaeffentb7models/content/Models"

import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

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
model_path = os.path.join(MODELS_WEIGHTS, "effnetB7_model_sparse_44acc.h5")
model = None

if os.path.exists(model_path):
    model = keras.models.load_model(model_path, compile=False)
    print("Loaded pretrained model (compile=False):", model_path)
else:
    print("Pretrained model not found at:", model_path)
    print(
        "Falling back to training a small EfficientNet model from scratch (minimal, for runnable submission)."
    )

    IMG_SIZE = 224
    NUM_CLASSES = 5
    BATCH_SIZE = 32
    EPOCHS = (
        1  # keep runtime safe; objective is to produce a valid submission end-to-end
    )

    train_df = pd.read_csv(TRAIN_CSV)

    train_df["filepath"] = TRAIN_IMG_LOC + "/" + train_df["image_id"].astype(str)
    train_df["label"] = train_df["label"].astype(np.int32)

    train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    val_frac = 0.1
    n_val = int(len(train_df) * val_frac)
    val_df = train_df.iloc[:n_val].copy()
    trn_df = train_df.iloc[n_val:].copy()

    @tf.function
    def _load_train(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3])
        return img, tf.cast(label, tf.int32)

    def make_ds(df, training):
        paths = df["filepath"].to_numpy(dtype=str)
        labels = df["label"].to_numpy(dtype=np.int32)

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        options = tf.data.Options()
        options.experimental_deterministic = True
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.autotune_buffers = True
        options.experimental_optimization.autotune_cpu_budget = 0
        ds = ds.with_options(options)

        ds = ds.map(_load_train, num_parallel_calls=AUTOTUNE)

        if training:
            ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        if not training:
            ds = ds.cache()
        ds = ds.prefetch(AUTOTUNE)
        return ds

    trn_ds = make_ds(trn_df, training=True)
    val_ds = make_ds(val_df, training=False)

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    backbone = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
    )
    x = backbone.output
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(trn_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/39911621.py in <cell line: 0>()
     67         return ds
     68 
---> 69     trn_ds = make_ds(trn_df, training=True)
     70     val_ds = make_ds(val_df, training=False)
     71 

/tmp/ipykernel_11/39911621.py in make_ds(df, training)
     51         options.experimental_deterministic = True
     52         options.experimental_optimization.apply_default_optimizations = True
---> 53         options.experimental_optimization.autotune_buffers = True
     54         options.experimental_optimization.autotune_cpu_budget = 0
     55         ds = ds.with_options(options)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 2
pass




## === cell 3
sub = pd.read_csv(SAMPLE_CSV)
assert list(sub.columns) == ["image_id", "label"], "Unexpected submission columns"

sub["filepath"] = TEST_IMG_LOC + "/" + sub["image_id"].astype(str)

IMG_SIZE = 224
BATCH_SIZE = 64

test_paths = sub["filepath"].to_numpy(dtype=str)
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


@tf.function
def _load_test(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3])
    return img


options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.autotune_buffers = True
options.experimental_optimization.autotune_cpu_budget = 0
test_ds = test_ds.with_options(options)

test_ds = (
    test_ds.map(_load_test, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/490091720.py in <cell line: 0>()
     27 options.experimental_deterministic = True
     28 options.experimental_optimization.apply_default_optimizations = True
---> 29 options.experimental_optimization.autotune_buffers = True
     30 options.experimental_optimization.autotune_cpu_budget = 0
     31 test_ds = test_ds.with_options(options)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 4
probs = model.predict(test_ds, verbose=1)
pred_labels = np.argmax(probs, axis=1).astype(int)

sub_out = sub[["image_id"]].copy()
sub_out["label"] = pred_labels

assert len(sub_out) == len(sub), "Row count mismatch in submission"
assert sub_out["label"].between(0, 4).all(), "Predicted labels out of range 0-4"

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2358969318.py in <cell line: 0>()
      1 # Speed: predict() with compile=False-loaded model is identical for inference and avoids extra overhead.
----> 2 probs = model.predict(test_ds, verbose=1)
      3 pred_labels = np.argmax(probs, axis=1).astype(int)
      4 
      5 sub_out = sub[["image_id"]].copy()

AttributeError: 'NoneType' object has no attribute 'predict'

## === cell 5
sub_out.head(10)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3599879406.py in <cell line: 0>()
----> 1 sub_out.head(10)

NameError: name 'sub_out' is not defined
