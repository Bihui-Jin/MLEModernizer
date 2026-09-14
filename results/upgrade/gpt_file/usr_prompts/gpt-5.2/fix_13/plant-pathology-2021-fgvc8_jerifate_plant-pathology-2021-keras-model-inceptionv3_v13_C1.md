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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
missingno==0.5.2
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
seaborn==0.12.2
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

0.15789

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

SEED = 42
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

tf.config.run_functions_eagerly(False)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
os.listdir("/kaggle/input/plant-pathology-2021-fgvc8/")



## === cell 2
train_df = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
test_df = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")
print("Dataset Shape: ", train_df.shape)
train_df.head()



## === cell 3
train_images_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
print("Train images dir:", train_images_dir)




## === cell 4
def add_link(path):
    return "/kaggle/input/plant-pathology-2021-fgvc8/train_images/" + str(path)




## === cell 5
def add_link_test(path):
    return "/kaggle/input/plant-pathology-2021-fgvc8/test_images/" + str(path)




## === cell 6
train_df["image"] = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/" + train_df[
    "image"
].astype(str)
train_df.head()



## === cell 7
test_df["image"] = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/" + test_df[
    "image"
].astype(str)

test_df.head()



## === cell 8
label_counts = train_df["labels"].value_counts()
print("Top-10 label combinations:\n", label_counts.head(10))



## === cell 9
unique_list = np.unique(train_df["labels"])
print(unique_list[:10], "...")
print("Number of unique label combinations:", train_df["labels"].value_counts().count())




## === cell 10
def read_image(path):
    gfile = tf.io.read_file(path)
    image = tf.io.decode_jpeg(gfile, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    return image




## === cell 11
def get_label(path):
    return_label = train_df[train_df["image"] == path]["labels"]
    print(return_label)
    return list(return_label)




## === cell 12
def get_label_image(path):
    label = get_label(path)
    image = read_image(path)
    return label, image




## === cell 13
try:
    pass
except Exception as e:
    print("Skipped sample visualization due to:", repr(e))



## === cell 14
INPUT_SIZE = (224, 224, 3)
BATCH_SIZE = 32

classes = sorted(train_df["labels"].unique().tolist())
CLASSES = len(classes)
print("Number of classes (unique label combinations):", CLASSES)



## === cell 15
train_data, val_data = train_test_split(
    train_df, test_size=0.2, random_state=SEED, shuffle=True
)
print("Train Data Shape: ", train_data.shape)
print("Validation Data Shape: ", val_data.shape)



## === cell 16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1 / 255.0, width_shift_range=0.3, zoom_range=0.2, horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1 / 255.0)
val_datagen = ImageDataGenerator(rescale=1 / 255.0)



## === cell 17
pre_model = DenseNet121(include_top=False, weights="imagenet", input_shape=INPUT_SIZE)
pre_model.trainable = False



## === cell 18
model = tf.keras.Sequential()
model.add(pre_model)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(CLASSES, activation="softmax"))



## === cell 19
callback = ReduceLROnPlateau(monitor="val_loss", factor=0.01, patience=3, min_lr=1e-5)



## === cell 20
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)



## === cell 21
steps_per_epoch = int(np.ceil(len(train_data) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_data) / BATCH_SIZE))

feature_shape = pre_model.output_shape[1:]  # e.g. (7, 7, 1024)
head_input = keras.Input(shape=feature_shape)
x = head_input
for lyr in model.layers[1:]:
    x = lyr(x)
head_model = keras.Model(head_input, x, name="head_model")

head_model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)

feature_extractor = keras.Model(
    pre_model.input, pre_model.output, name="feature_extractor"
)

AUTOTUNE = tf.data.AUTOTUNE

class_to_idx = {c: i for i, c in enumerate(classes)}
train_label_idx = train_data["labels"].map(class_to_idx).to_numpy(np.int32)
val_label_idx = val_data["labels"].map(class_to_idx).to_numpy(np.int32)


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(
        img, INPUT_SIZE[:2], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    return img


augmenter = keras.Sequential(
    [
        layers.Rescaling(1.0),  # image already in [0,1]; keep explicit
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomTranslation(
            height_factor=0.0, width_factor=0.3, fill_mode="reflect", seed=SEED
        ),
        layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="augmenter",
)


def _make_ds(paths, label_idx=None, training=False, augment=False, cache=False):
    paths = tf.convert_to_tensor(paths, dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(paths)

    buffer_size = (
        int(paths.shape[0])
        if paths.shape.rank == 1 and paths.shape[0] is not None
        else int(tf.size(paths).numpy())
    )

    if label_idx is not None:
        ys = tf.convert_to_tensor(label_idx, dtype=tf.int32)
        ds_y = tf.data.Dataset.from_tensor_slices(ys)
        ds = tf.data.Dataset.zip((ds, ds_y))

        if training:
            ds = ds.shuffle(
                buffer_size=buffer_size, seed=SEED, reshuffle_each_iteration=True
            )

        def _load(path, y):
            x = _decode_resize(path)
            if augment:
                x = augmenter(x, training=True)
            y_oh = tf.one_hot(y, depth=CLASSES, dtype=tf.float32)
            return x, y_oh

        ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:
        if training:
            ds = ds.shuffle(
                buffer_size=buffer_size, seed=SEED, reshuffle_each_iteration=True
            )

        def _load_x(path):
            x = _decode_resize(path)
            return x

        ds = ds.map(_load_x, num_parallel_calls=AUTOTUNE, deterministic=True)

    if cache:
        ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


@tf.function(jit_compile=True)
def _to_bottleneck(x, y=None):
    f = feature_extractor(x, training=False)
    return (f, y) if y is not None else f


train_paths = train_data["image"].to_numpy()
val_paths = val_data["image"].to_numpy()

train_ds_img = _make_ds(
    train_paths, train_label_idx, training=True, augment=True, cache=False
)
val_ds_img = _make_ds(
    val_paths, val_label_idx, training=False, augment=False, cache=True
)

train_ds_bneck = train_ds_img.map(
    lambda x, y: _to_bottleneck(x, y), num_parallel_calls=AUTOTUNE, deterministic=True
).prefetch(AUTOTUNE)
val_ds_bneck = val_ds_img.map(
    lambda x, y: _to_bottleneck(x, y), num_parallel_calls=AUTOTUNE, deterministic=True
).prefetch(AUTOTUNE)

history = head_model.fit(
    train_ds_bneck,
    epochs=25,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds_bneck,
    validation_steps=val_steps,
    callbacks=[callback],
    verbose=1,
)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3273506035.py in <cell line: 0>()
    131 ).prefetch(AUTOTUNE)
    132 
--> 133 history = head_model.fit(
    134     train_ds_bneck,
    135     epochs=25,

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

Detected at node StatefulPartitionedCall defined at (most recent call last):
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

  File "/tmp/ipykernel_11/3273506035.py", line 133, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

Detected at node StatefulPartitionedCall defined at (most recent call last):
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

  File "/tmp/ipykernel_11/3273506035.py", line 133, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

2 root error(s) found.
  (0) INVALID_ARGUMENT:  Error in user-defined function passed to ParallelMapDatasetV2:14 transformation with iterator: Iterator::Root::Prefetch::Map:  Trying to access resource conv1_conv/kernel/0 (defined @ /usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py:38) located in device /job:localhost/replica:0/task:0/device:GPU:0 from device /job:localhost/replica:0/task:0/device:CPU:0
 Cf. https://www.tensorflow.org/xla/known_issues#tfvariable_on_a_different_device
	 [[{{node StatefulPartitionedCall}}]]
	 [[IteratorGetNext]]
	 [[StatefulPartitionedCall/Shape/_8]]
  (1) INVALID_ARGUMENT:  Error in user-defined function passed to ParallelMapDatasetV2:14 transformation with iterator: Iterator::Root::Prefetch::Map:  Trying to access resource conv1_conv/kernel/0 (defined @ /usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py:38) located in device /job:localhost/replica:0/task:0/device:GPU:0 from device /job:localhost/replica:0/task:0/device:CPU:0
 Cf. https://www.tensorflow.org/xla/known_issues#tfvariable_on_a_different_device
	 [[{{node StatefulPartitionedCall}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_12740]

## === cell 22
try:
    pass
except Exception as e:
    print("Skipped training plots due to:", repr(e))



## === cell 23
submission = pd.read_csv(
    "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
)
submission.head()



## === cell 24
test_paths = test_df["image"].to_numpy()
test_n = len(test_paths)

test_ds_img = _make_ds(
    test_paths, label_idx=None, training=False, augment=False, cache=True
)

test_ds_bneck = test_ds_img.map(
    lambda x: _to_bottleneck(x, None), num_parallel_calls=AUTOTUNE, deterministic=True
).prefetch(AUTOTUNE)

preds = head_model.predict(
    test_ds_bneck,
    verbose=1,
    steps=int(np.ceil(test_n / BATCH_SIZE)),
)

idx_to_class = np.asarray(classes, dtype=object)

test_pred_idx = np.argmax(preds, axis=-1)
test_pred_labels = idx_to_class[test_pred_idx]

submission["image"] = [os.path.basename(p) for p in test_paths]
submission["labels"] = test_pred_labels
submission = submission[["image", "labels"]]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3962484701.py in <cell line: 0>()
     11 ).prefetch(AUTOTUNE)
     12 
---> 13 preds = head_model.predict(
     14     test_ds_bneck,
     15     verbose=1,

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

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:26 transformation with iterator: Iterator::Root::Prefetch::Map: {{function_node __inference_Dataset_map_lambda_17492}} Trying to access resource conv1_conv/kernel/0 (defined @ /usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py:38) located in device /job:localhost/replica:0/task:0/device:GPU:0 from device /job:localhost/replica:0/task:0/device:CPU:0
 Cf. https://www.tensorflow.org/xla/known_issues#tfvariable_on_a_different_device
	 [[{{node StatefulPartitionedCall}}]] [Op:IteratorGetNext] name:
