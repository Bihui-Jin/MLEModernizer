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

0.2239124126104717

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.27088) has done: 'The timeout is dominated by decoding and resizing 512×512 JPEGs on-the-fly for ~15k training images over multiple epochs; we keep the exact same model/training logic but make the input pipeline substantially faster and more CPU-parallel. Specifically, we enable TF graph optimizations and deterministic execution, add dataset caching (in-memory if possible, otherwise to disk) so each image is decoded/resized only once across epochs, and tune tf.data options (parallelism, prefetch, non-blocking) without changing semantics. We also replace the slow Python loop that builds submission strings with an equivalent vectorized NumPy implementation (same thresholds and “healthy” fallback). These changes preserve accuracy/evaluation behavior while cutting redundant work to fit within the 600s budget.'

# 9. Code solution

## === cell 0
import os, random, math, re

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers

print("tf:", tf.__version__)
print("keras:", tf.keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception as e:
    print("Determinism setting not available:", e)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("Thread config not available:", e)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("XLA JIT not available:", e)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"

train = pd.read_csv(os.path.join(path, "train.csv"))
sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

train_images_dir = os.path.join(path, "train_images")
test_images_dir = os.path.join(path, "test_images")

print(train.shape, sub.shape)
print(train.head())



## === cell 2
AUTO = tf.data.experimental.AUTOTUNE



## === cell 3
img_path = os.path.join(train_images_dir, train.iloc[0]["image"])
print("Example image (path only):", img_path)



## === cell 4
import pathlib



## === cell 5
train_paths = sorted(tf.io.gfile.glob(os.path.join(train_images_dir, "*.jpg")))
test_paths = sorted(tf.io.gfile.glob(os.path.join(test_images_dir, "*.jpg")))

print("n_train_images:", len(train_paths))
print("n_test_images:", len(test_paths))
print("example test path:", test_paths[0])



## === cell 6
CLASSES = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew", "healthy"]
num_classes = len(CLASSES)
print("classes:", CLASSES)



## === cell 7
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

train["filepath"] = train_images_dir.rstrip("/") + "/" + train["image"].astype(str)

_expl = train["labels"].astype(str).str.split().explode()
_expl = _expl[_expl.isin(CLASSES)]
ct = (
    pd.crosstab(_expl.index, _expl)
    .reindex(columns=CLASSES, fill_value=0)
    .astype(np.float32)
)

train_multihot = ct.to_numpy(dtype=np.float32, copy=False)
train["multihot"] = list(train_multihot)

new_train = pd.concat([train[["image"]], ct.reset_index(drop=True)], axis=1)
print(new_train.head())
print("Multihot check sum:", new_train[CLASSES].sum().to_dict())



## === cell 8
new_train




## === cell 9
@tf.function(jit_compile=True)
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.convert_image_dtype(image, tf.float32)  # stable cast/255 semantics
    image = tf.image.resize(
        image, image_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    if label is None:
        return image
    else:
        return image, label




## === cell 10
test_df = pd.DataFrame(
    {"image": [os.path.basename(p) for p in test_paths], "filepath": test_paths}
)
print(test_df.head(), test_df.shape)



## === cell 11
BATCH_SIZE = 16  # keep memory safe for 512x512 images
IMG_SIZE = (512, 512)




## === cell 12
def with_fast_deterministic_options(ds: tf.data.Dataset) -> tf.data.Dataset:
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.autotune_buffers = True
    except Exception as e:
        print("Could not set some tf.data optimization flags:", e)
    return ds.with_options(opts)




## === cell 13
def _decode_only(x):
    return decode_image(x, None, IMG_SIZE)


def _decode_with_label(x, y):
    return decode_image(x, y, IMG_SIZE)


test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_df["filepath"].values)
    .map(_decode_only, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)
test_dataset = with_fast_deterministic_options(test_dataset)



## === cell 14
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    train["filepath"].values,
    train_multihot,
    test_size=0.10,
    random_state=SEED,
    shuffle=True,
)

train_cache_path = os.path.join("/kaggle/working", "pp2021_train_cache.tfdata")
val_cache_path = os.path.join("/kaggle/working", "pp2021_val_cache.tfdata")

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .map(_decode_with_label, num_parallel_calls=AUTO, deterministic=True)
    .cache(train_cache_path)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)
train_dataset = with_fast_deterministic_options(train_dataset)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .map(_decode_with_label, num_parallel_calls=AUTO, deterministic=True)
    .cache(val_cache_path)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)
val_dataset = with_fast_deterministic_options(val_dataset)

print("train batches:", tf.data.experimental.cardinality(train_dataset).numpy())
print("val batches:", tf.data.experimental.cardinality(val_dataset).numpy())



## === cell 15
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({"swish": keras.layers.Activation(tf.nn.swish)})


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




## === cell 16
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy"
)
model.summary()



## === cell 17
EPOCHS = 2
history = model.fit(
    train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=1
)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3817637290.py in <cell line: 0>()
      1 EPOCHS = 2
----> 2 history = model.fit(
      3     train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=1
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

Detected at node filename defined at (most recent call last):
<stack traces unavailable>
Detected at node filename defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:6 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Shuffle::FileCacheImpl::ParallelMapV2: Detected unsupported operations when trying to compile graph __inference_decode_image_51[] on XLA_CPU_JIT: _Arg (No registered '_Arg' OpKernel for XLA_CPU_JIT devices compatible with node {{node filename}}
	 (OpKernel was found, but attributes didn't match) Requested Attributes: T=DT_STRING, _output_shapes=[[]], _user_specified_name="filename", index=0){{node filename}}
The op is created at: 
dummy_file_name:10:dummy_function_name
	tf2xla conversion failed while converting __inference_decode_image_51[]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
	 [[PartitionedCall]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_1840]

## === cell 18
probs = model.predict(test_dataset, verbose=1)
print("probs shape:", probs.shape)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/4111444144.py in <cell line: 0>()
----> 1 probs = model.predict(test_dataset, verbose=1)
      2 print("probs shape:", probs.shape)
      3 

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

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Detected unsupported operations when trying to compile graph __inference_decode_image_18[] on XLA_CPU_JIT: _Arg (No registered '_Arg' OpKernel for XLA_CPU_JIT devices compatible with node {{node filename}}
	 (OpKernel was found, but attributes didn't match) Requested Attributes: T=DT_STRING, _output_shapes=[[]], _user_specified_name="filename", index=0){{node filename}}
The op is created at: 
dummy_file_name:10:dummy_function_name
	tf2xla conversion failed while converting __inference_decode_image_18[]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
	 [[PartitionedCall]] [Op:IteratorGetNext] name: 

## === cell 19
temp_probs = probs



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1438685205.py in <cell line: 0>()
----> 1 temp_probs = probs
      2 

NameError: name 'probs' is not defined

## === cell 20
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.30, 1: 0.50, 2: 0.30, 3: 0.50, 4: 0.50}

thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)  # (5,)
mask = temp_probs[:, :5] > thr[None, :]  # (N,5) bool
labels5 = np.array([name[i] for i in range(5)], dtype=object)

pred_string = []
append = pred_string.append
for row in mask:
    if row.any():
        append(" ".join(labels5[row].tolist()))
    else:
        append(name[5])

submission = pd.DataFrame({"image": test_df["image"].values, "labels": pred_string})
submission = submission[["image", "labels"]]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2887086657.py in <cell line: 0>()
     13 # avoiding repeated dict lookups and preserving identical thresholding semantics.
     14 thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)  # (5,)
---> 15 mask = temp_probs[:, :5] > thr[None, :]  # (N,5) bool
     16 labels5 = np.array([name[i] for i in range(5)], dtype=object)
     17 

NameError: name 'temp_probs' is not defined
