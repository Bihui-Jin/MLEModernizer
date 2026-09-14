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

3.12

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

0.8319734058627984

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob, json, random, shutil, datetime

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") == "python":
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

WORK_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("TensorFlow:", tf.__version__)
print("WORK_DIR exists:", os.path.exists(WORK_DIR))
print("Eager:", tf.executing_eagerly())
print(
    "Determinism note: keeping deterministic dataset option; not changing thread settings."
)

tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Warning: could not enable XLA JIT:", repr(e))

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
train_images_path = os.path.join(WORK_DIR, "train_images")

assert os.path.isdir(
    train_images_path
), f"Missing train_images path: {train_images_path}"
df["image_id"] = df["image_id"].astype(str)
df["label"] = df["label"].astype(int)

df["filepath"] = train_images_path.rstrip("/") + "/" + df["image_id"]

filepaths_arr = df["filepath"].to_numpy()
missing = int(np.sum([not os.path.exists(p) for p in filepaths_arr]))
if missing:
    print(f"Warning: {missing} train image paths missing (will error if accessed).")

print("Train rows:", len(df))
print("Label distribution:\n", df["label"].value_counts().sort_index())




## === cell 2
AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True

BATCH_SIZE = 16
IMAGE_SIZE = (224, 224)
VALIDATION_SPLIT = 0.1

filepaths = df["filepath"].to_numpy()
labels_int = df["label"].to_numpy().astype(np.int32)

rng = np.random.RandomState(SEED)
idx = np.arange(len(filepaths))
rng.shuffle(idx)

n_valid = int(round(len(idx) * VALIDATION_SPLIT))
valid_idx = idx[:n_valid]
train_idx = idx[n_valid:]

train_files = filepaths[train_idx]
train_labels = labels_int[train_idx]
valid_files = filepaths[valid_idx]
valid_labels = labels_int[valid_idx]


@tf.function
def _decode_resize(path, y):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    return img, y


@tf.function
def _preprocess_onehot(img, y):
    img = tf.keras.applications.efficientnet_v2.preprocess_input(
        tf.cast(img, tf.float32)
    )
    y = tf.one_hot(y, depth=5, dtype=tf.float32)
    return img, y


train_pairs = tf.data.Dataset.from_tensor_slices(
    (train_files, train_labels)
).with_options(options)
valid_pairs = tf.data.Dataset.from_tensor_slices(
    (valid_files, valid_labels)
).with_options(options)

train_batches = int(np.ceil(len(train_files) / BATCH_SIZE))
valid_batches = int(np.ceil(len(valid_files) / BATCH_SIZE))
print("Train batches:", train_batches)
print("Valid batches:", valid_batches)




## === cell 3
def data_augmentation():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode="reflect", seed=SEED),
            tf.keras.layers.RandomRotation(0.2, fill_mode="reflect", seed=SEED),
            tf.keras.layers.RandomZoom(0.2, seed=SEED),
            tf.keras.layers.RandomContrast(0.2, seed=SEED),
        ]
    )


data_aug = data_augmentation()

cache_train_path = os.path.join(OUTPUT_DIR, "cache_train_decoded")
cache_valid_path = os.path.join(OUTPUT_DIR, "cache_valid_decoded")

decoded_train = (
    train_pairs.shuffle(len(train_files), seed=SEED, reshuffle_each_iteration=True)
    .map(_decode_resize, num_parallel_calls=AUTOTUNE)
    .cache(cache_train_path)
)

decoded_valid = valid_pairs.map(_decode_resize, num_parallel_calls=AUTOTUNE).cache(
    cache_valid_path
)

base_train_dataset = (
    decoded_train.map(_preprocess_onehot, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .repeat()
    .with_options(options)
)


@tf.function
def _augment_then_label(img, y):
    return data_aug(img, training=True), y


augmented_dataset = (
    decoded_train.map(_augment_then_label, num_parallel_calls=AUTOTUNE)
    .map(_preprocess_onehot, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .repeat()
    .with_options(options)
)

train_dataset = (
    tf.data.Dataset.sample_from_datasets(
        [base_train_dataset, augmented_dataset],
        weights=[1.0, 2.0],
        seed=SEED,
    )
    .prefetch(AUTOTUNE)
    .with_options(options)
)

valid_dataset = (
    decoded_valid.map(_preprocess_onehot, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .repeat()
    .with_options(options)
)

steps_per_epoch = train_batches
validation_steps = valid_batches

print("steps_per_epoch:", steps_per_epoch)
print("validation_steps:", validation_steps)




## === cell 4
class SigmoidFocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(
            1 - y_pred
        )
        weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_loss = weight * ((1 - y_pred) ** self.gamma) * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)




## === cell 5
class _HP:
    def Float(self, name, min_value=None, max_value=None, sampling=None):
        if name == "learning_rate":
            return float(np.sqrt(min_value * max_value))
        return float((min_value + max_value) / 2.0)

    def Int(self, name, min_value=None, max_value=None, step=1):
        v = min_value + ((max_value - min_value) // (2 * step)) * step
        return int(v)


class _SimpleTuner:
    def __init__(self, hypermodel, **kwargs):
        self.hypermodel = hypermodel
        self._best_hp = _HP()

    def search(self, *args, **kwargs):
        return

    def get_best_hyperparameters(self, num_trials=1):
        return [self._best_hp]




## === cell 6
def build_model(hp):
    learning_rate = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
    )
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5)
    dense_units = hp.Int("dense_units", min_value=128, max_value=512, step=32)

    try:
        base = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
    except Exception as e:
        print(
            "Warning: could not load imagenet weights, falling back to None. Error:",
            repr(e),
        )
        base = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            weights=None, include_top=False, input_shape=(224, 224, 3)
        )

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=SigmoidFocalCrossEntropy(alpha=0.25, gamma=2, from_logits=False),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
        jit_compile=True,
    )
    return model


tuner = _SimpleTuner(build_model)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=3,
    mode="min",
    verbose=1,
    restore_best_weights=True,
)

tuner.search(
    train_dataset, validation_data=valid_dataset, epochs=2, callbacks=[early_stop]
)
best_hps = tuner.get_best_hyperparameters()[0]
print("Using hyperparameters (deterministic heuristic).")




## === cell 7
def Model_Training(tuner, train_dataset, valid_dataset, epochs=10):
    best_hps = tuner.get_best_hyperparameters()[0]
    model = (
        tuner.hypermodel(best_hps)
        if callable(getattr(tuner, "hypermodel", None))
        else tuner.hypermodel.build(best_hps)
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        mode="min",
        verbose=1,
        restore_best_weights=True,
    )

    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_dataset,
        validation_steps=validation_steps,
        callbacks=[early_stop],
        verbose=1,
    )

    return model


model = Model_Training(tuner, train_dataset, valid_dataset, epochs=10)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AlreadyExistsError                        Traceback (most recent call last)
/tmp/ipykernel_11/946482882.py in <cell line: 0>()
     29 
     30 
---> 31 model = Model_Training(tuner, train_dataset, valid_dataset, epochs=10)
     32 
     33 

/tmp/ipykernel_11/946482882.py in Model_Training(tuner, train_dataset, valid_dataset, epochs)
     16     )
     17 
---> 18     history = model.fit(
     19         train_dataset,
     20         epochs=epochs,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

AlreadyExistsError: Graph execution error:

Detected at node IteratorGetNext defined at (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main

  File "<frozen runpy>", line 88, in _run_code

  File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>

  File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start

  File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start

  File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever

  File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once

  File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code

  File "/tmp/ipykernel_11/946482882.py", line 31, in <cell line: 0>

  File "/tmp/ipykernel_11/946482882.py", line 18, in Model_Training

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

There appears to be a concurrent caching iterator running - cache lockfile already exists ('./cache_train_decoded_0.lockfile'). If you are sure no other running TF computations are using this cache prefix, delete the lockfile and re-initialize the iterator. Lockfile contents: Created at: 1778511261
	 [[{{node IteratorGetNext}}]] [Op:__inference_multi_step_on_iterator_107538]

## === cell 8
sample_sub_path = os.path.join(WORK_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

test_dir = os.path.join(WORK_DIR, "test_images")
test_files = sample_sub["image_id"].tolist()
test_paths = [os.path.join(test_dir, f) for f in test_files]

n_missing_test = int(np.sum([not os.path.exists(p) for p in test_paths]))
if n_missing_test:
    print(f"Warning: missing {n_missing_test} test image files (will error if read).")


@tf.function
def _load_and_preprocess_fullpath(image_path):
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.keras.applications.efficientnet_v2.preprocess_input(
        tf.cast(img, tf.float32)
    )
    return img


test_ds = tf.data.Dataset.from_tensor_slices(tf.constant(test_paths)).with_options(
    options
)
test_ds = test_ds.map(_load_and_preprocess_fullpath, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(64).prefetch(AUTOTUNE).with_options(options)

probs = model.predict(test_ds, verbose=0)
preds = np.argmax(probs, axis=-1).astype(int)

submission = pd.DataFrame({"image_id": test_files, "label": preds})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4281402406.py in <cell line: 0>()
     29 test_ds = test_ds.batch(64).prefetch(AUTOTUNE).with_options(options)
     30 
---> 31 probs = model.predict(test_ds, verbose=0)
     32 preds = np.argmax(probs, axis=-1).astype(int)
     33 

NameError: name 'model' is not defined
