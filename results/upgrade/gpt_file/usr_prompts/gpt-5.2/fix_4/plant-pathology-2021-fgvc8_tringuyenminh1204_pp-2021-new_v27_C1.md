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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.2819271863870179

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, math, re
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model

print("tf:", tf.__version__)
print("tf.keras:", tf.keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"
train = pd.read_csv(os.path.join(path, "train.csv"))
test = pd.read_csv(os.path.join(path, "sample_submission.csv"))
sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

train_images_dir = os.path.join(path, "train_images")
test_images_dir = os.path.join(path, "test_images")

print(train.shape, test.shape, sub.shape)
print(train.head())



## === cell 2
AUTO = tf.data.experimental.AUTOTUNE
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not available:", e)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("XLA JIT not available:", e)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPUs:", gpus)
except Exception as e:
    print("GPU config skipped:", e)



## === cell 3
print("Skipped matplotlib preview to save time.")



## === cell 4
import pathlib



## === cell 5
train_paths = [
    os.path.join(train_images_dir, img_id) for img_id in train["image"].tolist()
]
train_paths = sorted(train_paths)

test_paths = [
    os.path.join(test_images_dir, img_id) for img_id in test["image"].tolist()
]

print("n_train_paths:", len(train_paths))
print("n_test_paths:", len(test_paths))

sample_check = test_paths[:32]
missing_test = [p for p in sample_check if not tf.io.gfile.exists(p)]
print("missing_test_images_in_first_32:", len(missing_test))



## === cell 6
labs_series = train["labels"].astype(str).str.split()
all_labels = set()
for lst in labs_series.tolist():
    all_labels.update(lst)

classes = sorted(list(all_labels))
print("classes:", classes)
num_classes = len(classes)

class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}



## === cell 7
y = np.zeros((len(train), num_classes), dtype=np.float32)
for i, lst in enumerate(labs_series.tolist()):
    if lst:
        y[i, [class_to_idx[lab] for lab in lst]] = 1.0

new_train = pd.concat(
    [train[["image", "labels"]], pd.DataFrame(y, columns=classes)], axis=1
)
new_train.head()



## === cell 8
new_train




## === cell 9
@tf.function
def decode_image_nosize(filename, label=None):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(
        bits, channels=3, try_recover_truncated=True, acceptable_fraction=0.75
    )
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    if label is None:
        return image
    return image, label


@tf.function
def resize_only(image, label=None, image_size=(512, 512)):
    image = tf.image.resize(image, image_size, method="bilinear")
    if label is None:
        return image
    return image, label




## === cell 10
test_paths[:5]



## === cell 11
BATCH_SIZE = 32  # keep as original
IMG_SIZE = (512, 512)



## === cell 12
test_options = tf.data.Options()
test_options.deterministic = True

test_dataset = (
    tf.data.Dataset.from_tensor_slices(tf.constant(test_paths))
    .with_options(test_options)
    .map(
        lambda p: decode_image_nosize(p, None),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .map(
        lambda x: resize_only(x, None, IMG_SIZE),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .cache()
    .prefetch(AUTO)
)



## === cell 13
from tensorflow import keras



## === cell 14
train_filepaths = [
    os.path.join(train_images_dir, img_id) for img_id in train["image"].tolist()
]

sample_check = train_filepaths[:64]
missing_train = [p for p in sample_check if not tf.io.gfile.exists(p)]
print("missing_train_images_in_first_64:", len(missing_train))



## === cell 15
idx = np.arange(len(train_filepaths))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_size = int(len(idx) * val_frac)

val_idx = idx[:val_size]
trn_idx = idx[val_size:]

train_filepaths_arr = np.asarray(train_filepaths, dtype=object)
x_trn = train_filepaths_arr[trn_idx]
y_trn = y[trn_idx]
x_val = train_filepaths_arr[val_idx]
y_val = y[val_idx]

train_options = tf.data.Options()
train_options.deterministic = True

val_options = tf.data.Options()
val_options.deterministic = True

train_dataset = (
    tf.data.Dataset.from_tensor_slices((tf.constant(x_trn.tolist()), y_trn))
    .with_options(train_options)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .map(
        lambda p, lab: decode_image_nosize(p, lab),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .batch(BATCH_SIZE, drop_remainder=True)
    .map(
        lambda x, lab: resize_only(x, lab, IMG_SIZE),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .cache()
    .prefetch(AUTO)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((tf.constant(x_val.tolist()), y_val))
    .with_options(val_options)
    .map(
        lambda p, lab: decode_image_nosize(p, lab),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .map(
        lambda x, lab: resize_only(x, lab, IMG_SIZE),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .cache()
    .prefetch(AUTO)
)

print("train batches:", tf.data.experimental.cardinality(train_dataset).numpy())
print("val batches:", tf.data.experimental.cardinality(val_dataset).numpy())



## === cell 16
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)
model.summary()



## === cell 17
EPOCHS = 3
history = model.fit(
    train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=2
)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/183675962.py in <cell line: 0>()
      1 EPOCHS = 3
----> 2 history = model.fit(
      3     train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=2
      4 )
      5 

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

InvalidArgumentError: Graph execution error:

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

  File "/tmp/ipykernel_11/183675962.py", line 2, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

Cannot add tensor to the batch: number of elements does not match. Shapes are: [tensor]: [3456,5184,3], [batch]: [2672,4000,3]
	 [[{{node IteratorGetNext}}]] [Op:__inference_multi_step_on_iterator_1998]

## === cell 18
probs = model.predict(test_dataset, verbose=1)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3073497979.py in <cell line: 0>()
----> 1 probs = model.predict(test_dataset, verbose=1)
      2 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Cannot add tensor to the batch: number of elements does not match. Shapes are: [tensor]: [3000,4000,3], [batch]: [2672,4000,3] [Op:IteratorGetNext] name: 

## === cell 19
probs.shape



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4130349179.py in <cell line: 0>()
----> 1 probs.shape
      2 

NameError: name 'probs' is not defined

## === cell 20
probs[:2]



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2619598287.py in <cell line: 0>()
----> 1 probs[:2]
      2 

NameError: name 'probs' is not defined

## === cell 21
temp_probs = probs



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1438685205.py in <cell line: 0>()
----> 1 temp_probs = probs
      2 

NameError: name 'probs' is not defined

## === cell 22
temp_probs.shape



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3836011535.py in <cell line: 0>()
----> 1 temp_probs.shape
      2 

NameError: name 'temp_probs' is not defined

## === cell 23
default_thr = 0.10
thr = default_thr

pred_bool = temp_probs > thr  # (N, C)
complex_idx = class_to_idx.get("complex", None)
healthy_idx = class_to_idx.get("healthy", None)

pred_string = []
for row in pred_bool:
    chosen_idx = np.flatnonzero(row)
    chosen = [idx_to_class[int(i)] for i in chosen_idx]

    if len(chosen) >= 2 and (complex_idx is not None) and ("complex" not in chosen):
        chosen.append("complex")

    if len(chosen) == 0:
        if healthy_idx is not None:
            chosen = ["healthy"]
        else:
            chosen = [idx_to_class[int(np.argmax(temp_probs[len(pred_string)]))]]

    pred_string.append(" ".join(chosen))

test["labels"] = pred_string
submission = test[["image", "labels"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3863824128.py in <cell line: 0>()
      3 thr = default_thr
      4 
----> 5 pred_bool = temp_probs > thr  # (N, C)
      6 complex_idx = class_to_idx.get("complex", None)
      7 healthy_idx = class_to_idx.get("healthy", None)

NameError: name 'temp_probs' is not defined
