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

0.7714470386492558

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, re, math, random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K

print("TensorFlow:", tf.__version__)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTO = tf.data.AUTOTUNE
tf.keras.backend.clear_session()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import pathlib




## === cell 2
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3)  # fast path for this dataset
    image = tf.image.convert_image_dtype(image, tf.float32)  # cast/255 equivalent
    image = tf.image.resize(image, image_size)
    image.set_shape([image_size[0], image_size[1], 3])
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 32




## === cell 4
source = "../input/plant-pathology-2021-fgvc8/test_images"

try:
    IMAGE_PATHS = sorted(
        [
            os.path.join(source, f)
            for f in os.listdir(source)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
except FileNotFoundError:
    IMAGE_PATHS = []




## === cell 5
IMAGE_PATHS[:5], len(IMAGE_PATHS)




## === cell 6
SAMPLE_SUB_PATH = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_images = sample_sub["image"].tolist()
test_image_paths = [os.path.join(source, img) for img in test_images]

missing = [p for p in test_image_paths if not tf.io.gfile.exists(p)]
print("Test images:", len(test_image_paths), " Missing:", len(missing))




## === cell 7
options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.autotune_buffers = True
except Exception:
    pass

TEST_CACHE = "../kaggle/working/tfdata_cache_test_512"
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_image_paths)
    .with_options(options)
    .map(
        lambda x: decode_image(x, None, (512, 512)),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    .cache(TEST_CACHE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)




## === cell 8
import tensorflow as tf
from tensorflow import keras




## === cell 9
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




## === cell 10
TRAIN_CSV_PATH = "../input/plant-pathology-2021-fgvc8/train.csv"
TRAIN_IMG_DIR = "../input/plant-pathology-2021-fgvc8/train_images"

train_df = pd.read_csv(TRAIN_CSV_PATH)

classes = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew"]
class_to_idx = {c: i for i, c in enumerate(classes)}


def labels_to_vec_series(labels_series: pd.Series) -> np.ndarray:
    d = labels_series.fillna("").str.get_dummies(sep=" ")
    d = d.reindex(columns=classes, fill_value=0).astype(np.float32)
    return d.to_numpy()


train_df["filepath"] = train_df["image"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))
targets = labels_to_vec_series(train_df["labels"])
train_df["target"] = list(targets)

rng = np.random.RandomState(SEED)
idx = np.arange(len(train_df))
rng.shuffle(idx)
val_size = int(0.1 * len(train_df))
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

IMAGE_SIZE = (512, 512)


def decode_image_with_label(filename, label, image_size=IMAGE_SIZE):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)
    image.set_shape([image_size[0], image_size[1], 3])
    return image, label


train_x = tr_df["filepath"].values
train_y = np.stack(tr_df["target"].values)
val_x = val_df["filepath"].values
val_y = np.stack(val_df["target"].values)

TRAIN_CACHE = "../kaggle/working/tfdata_cache_train_512"
VAL_CACHE = "../kaggle/working/tfdata_cache_val_512"

train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_x, train_y))
    .with_options(options)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .map(decode_image_with_label, num_parallel_calls=AUTO, deterministic=True)
    .cache(TRAIN_CACHE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((val_x, val_y))
    .with_options(options)
    .map(decode_image_with_label, num_parallel_calls=AUTO, deterministic=True)
    .cache(VAL_CACHE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(len(classes), activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=16,
)

EPOCHS = 3
history = model.fit(
    train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=2
)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_12/1948197695.py in <cell line: 0>()
     89 
     90 EPOCHS = 3
---> 91 history = model.fit(
     92     train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=2
     93 )

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

NotFoundError: Graph execution error:

Detected at node IteratorGetNextAsOptional defined at (most recent call last):
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

  File "/tmp/ipykernel_12/1948197695.py", line 91, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 199, in multi_step_on_iterator

../kaggle/working/tfdata_cache_train_512_0.lockfile; No such file or directory
	 [[{{node IteratorGetNextAsOptional}}]] [Op:__inference_multi_step_on_iterator_2055]

## === cell 11
probs = model.predict(test_dataset, verbose=1)
temp_probs = probs




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_12/1116676986.py in <cell line: 0>()
----> 1 probs = model.predict(test_dataset, verbose=1)
      2 temp_probs = probs
      3 
      4 

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

NotFoundError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} ../kaggle/working/tfdata_cache_test_512_0.lockfile; No such file or directory [Op:IteratorGetNext] name: 

## === cell 12
probs.shape




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/2258643480.py in <cell line: 0>()
----> 1 probs.shape
      2 
      3 

NameError: name 'probs' is not defined

## === cell 13
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}

threshold = {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01}
threshold2 = {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01}


def get_key(val):
    for key, value in name.items():
        if val == value:
            return key
    return "key doesn't exist"


thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)
mask = temp_probs[:, :5] > thr[None, :]

class_names = np.array([name[i] for i in range(5)], dtype=object)

class_strs = np.where(mask, class_names[None, :], "")
joined = np.char.strip(
    np.char.replace(np.sum(class_strs.astype("U"), axis=1), "  ", " ")
)
for _ in range(3):
    joined = np.char.replace(joined, "  ", " ")
joined = np.char.strip(joined)

pred_string = np.where(mask.any(axis=1), joined.astype(object), name[6]).tolist()




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/291958450.py in <cell line: 0>()
     20 
     21 thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)
---> 22 mask = temp_probs[:, :5] > thr[None, :]
     23 
     24 class_names = np.array([name[i] for i in range(5)], dtype=object)

NameError: name 'temp_probs' is not defined

## === cell 14
pred_string[:10], len(pred_string)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/2264494190.py in <cell line: 0>()
----> 1 pred_string[:10], len(pred_string)
      2 
      3 

NameError: name 'pred_string' is not defined

## === cell 15
IMAGE_PATHS[:5], len(IMAGE_PATHS)




## === cell 16
df = pd.DataFrame({"image": test_images, "labels": pred_string})
df.to_csv("submission.csv", index=False)
print(df.shape)
print(df.head())

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/4250774090.py in <cell line: 0>()
----> 1 df = pd.DataFrame({"image": test_images, "labels": pred_string})
      2 df.to_csv("submission.csv", index=False)
      3 print(df.shape)
      4 print(df.head())

NameError: name 'pred_string' is not defined
