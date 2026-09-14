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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.8928679359323058

# 6. Current score

0.69245

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.6775) has done: 'I remove the failing external EfficientNet/KerasApplications imports that trigger a protobuf incompatibility, since they are not used for inference here and stop the notebook before it can write a submission. I also replace the missing `../input/cassava-trained/*.h5` ensemble with a built-in TensorFlow model (EfficientNetB0) trained quickly on the provided `train_images/` so the pipeline runs end-to-end without relying on unavailable files. Finally, I fix pathing to use the actual `/kaggle/input/cassava-leaf-disease-classification/` directory, ensure label indexing matches `[0..4]`, and write a valid `submission.csv` with `image_id,label` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.71861) has done: 'I remove the failing TensorFlow internal protobuf version access that crashes the very first cell, which prevents the rest of the notebook from defining `train_df/sub_df` and cascades into `NameError`s. Then I keep the same EfficientNetB0 training/inference core logic but make the pipeline deterministic and compatible with TF 2.18 by adding safe GPU memory growth and consistent Keras random seeds. Finally, I ensure the submission is written as `submission.csv` with exactly `image_id,label` aligned to `sample_submission.csv`, so Kaggle accepts it.'
- What this solution (achieved 0.71861) has done: 'The crash happens before any training because TensorFlow 2.18 + protobuf 6.x can throw a `MessageFactory.GetPrototype` AttributeError during import/initialization in some Kaggle images. The minimal fix is to force Python protobuf implementation (compatible fallback) *before* importing TensorFlow, then keep your exact same EfficientNetB0 training/fine-tuning/inference logic unchanged. I also keep determinism/memory-growth as you had, and ensure the script still writes `submission.csv` with the required `image_id,label` columns aligned to `sample_submission.csv`. This should restore end-to-end execution and, since the model can actually train and predict, move your score up toward the target instead of failing early.'
- What this solution (achieved 0.71861) has done: 'The crash happens at TensorFlow import time due to an incompatibility between TF 2.18 and protobuf 6.x in this environment; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` alone is not always sufficient because it must be applied before protobuf is imported, and some images still need the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` fallback as well. I add a small “pre-import” guard that forces the pure-Python protobuf implementation and version before any TensorFlow/protobuf-related imports, without changing your model/training/inference logic. I also keep the rest of the pipeline identical so it trains and produces `submission.csv` reliably, which should restore execution and move your score upward from the current level. No architecture, loss, or training-loop changes are introduced beyond fixing the import/runtime failure.'
- What this solution (achieved 0.71861) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before any protobuf/TensorFlow-related import happens*, and also explicitly reloading `google.protobuf` if it was imported earlier in the kernel. This is a minimal, correctness-focused change that unblocks the entire pipeline so training/inference can run and a valid `submission.csv` is written. The model architecture, training loop, augmentations, and inference logic remain unchanged. With training actually running instead of crashing, the score should at least recover and move upward from the current 0.71861 toward the target band.'
- What this solution (achieved 0.71861) has done: 'We fix the runtime crash that happens before training by enforcing the pure-Python protobuf implementation *before any TensorFlow/protobuf import*, and by preventing stale `google.protobuf` modules from being reused in the same kernel. This is the minimal change needed to unblock the pipeline so the existing EfficientNetB0 training/fine-tuning/inference logic can run end-to-end. We also keep determinism/memory-growth guards as-is and ensure the submission is written to `submission.csv` with exactly `image_id,label` aligned to `sample_submission.csv`. No model architecture, loss, augmentations, or training loop logic is changed.'
- What this solution (achieved 0.71861) has done: 'The crash is caused by forcing protobuf to use the C++ implementation (`cpp`), which is incompatible with the installed protobuf package here and prevents TensorFlow from importing; switching to the pure-Python protobuf implementation *before* importing TensorFlow fixes the root runtime error and unblocks the whole pipeline. After that, the cascading `NameError`s disappear because `train_df/sub_df/tf` get defined normally. I keep your exact model (EfficientNetB0 + dropout + dense softmax), training schedule, and inference logic unchanged, only adding a safe pre-import guard plus small submission-safety checks (path selection fallback and stable ordering) to ensure a valid `submission.csv` is always written.'
- What this solution (achieved 0.71861) has done: 'We fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before anything can import protobuf*, and by also setting the recommended `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus a safe `TF_CPP_MIN_LOG_LEVEL` guard. Then we keep your exact EfficientNetB0 training/fine-tuning/inference logic unchanged, but make the `tf.data` pipeline accept Python string paths reliably by converting path lists to `np.array(dtype=str)` (prevents occasional graph/type issues). Finally, we keep the same submission formatting but add a strict alignment check against `sample_submission.csv` and always write `submission.csv` with the required columns.'
- What this solution (achieved 0.71861) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf backend before any protobuf/TensorFlow import, and by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` unconditionally (not just as a default) to avoid a bad pre-set environment value. This is the earliest blocker preventing training/inference and thus prevents reaching the target score. I keep the model/training/inference logic intact, only adding a small, safe “pre-import” guard plus an optional fallback to `tf.keras.utils.set_random_seed` for determinism compatibility. The rest of the pipeline (data loading, EfficientNetB0 training/fine-tuning, and submission formatting) remains unchanged and still write a valid `submission.csv`.'
- What this solution (achieved 0.71861) has done: 'We fix the runtime crash happening at TensorFlow import (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf backend *before* anything imports protobuf, and by explicitly preventing the C++ implementation from being used (which triggers this error in TF 2.18 + protobuf 6.x environments). This change is execution-critical and score-neutral: it simply unblocks the exact same training/inference pipeline you already have so it can run end-to-end and write `submission.csv`. We also make the pre-import cleanup more robust by clearing any already-imported `google.protobuf` modules and invalidating import caches. No changes are made to your model architecture, augmentation, training schedule, or prediction logic.'
- What this solution (achieved 0.71861) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf backend *before any protobuf-related import occurs*, and by cleanly removing any already-imported `google.protobuf` modules from `sys.modules` to prevent stale C++ bindings from being reused. This is execution-critical and score-neutral: it unblocks the exact same EfficientNetB0 training/fine-tuning/inference pipeline you already have. We also make the environment guard stricter (override any pre-set bad values) and add a small compatibility fallback around op determinism so TF 2.18 doesn’t fail during initialization. The rest of the code (data paths, model, training schedule, and submission formatting) remains unchanged and write a valid `submission.csv`.'
- What this solution (achieved 0.69245) has done: 'We fix the TensorFlow/protobuf crash that happens at import time by forcing TensorFlow to use the legacy pure-Python protobuf runtime via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and the supported `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, and also by setting `TF_USE_LEGACY_KERAS=1` before importing TF to avoid known Keras/protobuf edge issues in TF 2.18 Kaggle images. This is the earliest blocker and is score-neutral—it simply allows your exact same EfficientNetB0 training/inference pipeline to run end-to-end and produce `submission.csv`. We also add a tiny compatibility fallback to disable XLA JIT if enabled (another occasional trigger for protobuf/graph init issues) without changing the model/training semantics. No changes are made to architecture, loss, augmentations, training schedule, or submission formatting.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")

import sys
import importlib

for m in list(sys.modules.keys()):
    if m == "google.protobuf" or m.startswith("google.protobuf."):
        del sys.modules[m]
importlib.invalidate_caches()

import random
import numpy as np
import pandas as pd

import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

print("TF:", tf.__version__)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/kaggle/data/cassava-leaf-disease-classification"

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), TRAIN_CSV
assert os.path.exists(SAMPLE_SUB), SAMPLE_SUB
assert os.path.isdir(TRAIN_IMG_DIR), TRAIN_IMG_DIR
assert os.path.isdir(TEST_IMG_DIR), TEST_IMG_DIR

train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

sub_df = sub_df[["image_id", "label"]].copy()

print(train_df.head())
print(sub_df.head())
print("Train rows:", len(train_df), "Test rows:", len(sub_df))

NUM_CLASSES = 5



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-4

train_paths = np.array(
    [os.path.join(TRAIN_IMG_DIR, fn) for fn in train_df["image_id"].values], dtype=str
)
train_labels = train_df["label"].astype(np.int32).values

x_tr, x_va, y_tr, y_va = train_test_split(
    train_paths,
    train_labels,
    test_size=0.1,
    random_state=SEED,
    stratify=train_labels,
)

print("Train split:", len(x_tr), "Val split:", len(x_va))



## === cell 2
AUTOTUNE = tf.data.AUTOTUNE


def load_image(path, label=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


def make_train_ds(paths, labels, training=True, cache=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 4096), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        aug = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal", seed=SEED),
                tf.keras.layers.RandomRotation(0.05, seed=SEED),
                tf.keras.layers.RandomZoom(0.1, seed=SEED),
            ]
        )
        ds = ds.map(
            lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE
        )
    if cache:
        ds = ds.cache()
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(x_tr, y_tr, training=True, cache=True)
val_ds = make_train_ds(x_va, y_va, training=False, cache=True)



## === cell 3
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    pooling="avg",
)
base.trainable = False  # warm-start

inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = base(inputs, training=False)
x = tf.keras.layers.Dropout(0.2, seed=SEED)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 4
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR * 0.1),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=2)

full_train_ds = make_train_ds(train_paths, train_labels, training=True, cache=True)
_ = model.fit(full_train_ds, epochs=1, verbose=2)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
UnimplementedError                        Traceback (most recent call last)
/tmp/ipykernel_55/3566904339.py in <cell line: 0>()
     11 )
     12 
---> 13 history_ft = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=2)
     14 
     15 full_train_ds = make_train_ds(train_paths, train_labels, training=True, cache=True)

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

UnimplementedError: Graph execution error:

Detected at node gradient_tape/model/efficientnetb0/top_bn/FusedBatchNormGradV3 defined at (most recent call last):
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

  File "/tmp/ipykernel_55/3566904339.py", line 13, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py", line 65, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 1804, in fit

  File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 1398, in train_function

  File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 1381, in step_function

  File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 1370, in run_step

  File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/training.py", line 1151, in train_step

  File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/optimizers/optimizer.py", line 622, in minimize

  File "/usr/local/lib/python3.11/dist-packages/tf_keras/src/optimizers/optimizer.py", line 280, in compute_gradients

A deterministic GPU implementation of fused batch-norm backprop, when training is disabled, is not currently available.
	 [[{{node gradient_tape/model/efficientnetb0/top_bn/FusedBatchNormGradV3}}]] [Op:__inference_train_function_36880]

## === cell 5
test_paths = np.array(
    [os.path.join(TEST_IMG_DIR, fn) for fn in sub_df["image_id"].values], dtype=str
)
for p in test_paths[:3]:
    assert os.path.exists(p), f"Missing test image: {p}"

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(lambda p: load_image(p, label=None), num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("Computing predictions...")
probs = model.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(np.int64)

assert len(preds) == len(sub_df), (len(preds), len(sub_df))
assert preds.min() >= 0 and preds.max() < NUM_CLASSES, (preds.min(), preds.max())

sub_out = sub_df.copy()
sub_out["label"] = preds
sub_out = sub_out[["image_id", "label"]]

sample_ids = pd.read_csv(SAMPLE_SUB)["image_id"].values
assert np.array_equal(
    sub_out["image_id"].values, sample_ids
), "Submission image_id order mismatch vs sample_submission.csv"

sub_out.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(sub_out.head())
print(sub_out["label"].value_counts().sort_index())
