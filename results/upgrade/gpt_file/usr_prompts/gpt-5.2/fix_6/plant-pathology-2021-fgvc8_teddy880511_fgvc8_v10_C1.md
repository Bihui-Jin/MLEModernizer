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

0.1986149584487535

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(True)  # XLA JIT (best-effort)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

BASE_PATH = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.exists(TRAIN_IMG_DIR), f"Missing train image dir: {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing test image dir: {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_CSV), f"Missing train.csv: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample_submission.csv: {SAMPLE_SUB}"

IMG_SIZE = 64

label_class = [
    "scab",
    "healthy",
    "frog_eye_leaf_spot",
    "cider_apple_rust",
    "complex",
    "powdery_mildew",
    "scab frog_eye_leaf_spot",
]
label_to_idx = {l: i for i, l in enumerate(label_class)}



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
y_train_csv = pd.read_csv(TRAIN_CSV)

labels_arr = y_train_csv["labels"].to_numpy()
label_num = np.fromiter(
    (label_to_idx.get(x, label_to_idx["complex"]) for x in labels_arr),
    dtype=np.int32,
    count=len(labels_arr),
)
y_train = keras.utils.to_categorical(label_num, num_classes=7).astype(np.float32)

train_files = y_train_csv["image"].to_numpy()

train_dir_t = tf.constant(TRAIN_IMG_DIR)


@tf.function
def _load_and_preprocess_from_name(fname, y):
    path = tf.strings.join([train_dir_t, fname], separator=os.sep)
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_and_crop_jpeg(
        img_bytes, crop_window=[0, 0, 0, 0], channels=3
    )  # no-op crop (keeps semantics)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img, y


options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True

train_ds = tf.data.Dataset.from_tensor_slices((train_files, y_train)).with_options(
    options
)

train_ds = train_ds.map(_load_and_preprocess_from_name, num_parallel_calls=AUTOTUNE)


train_ds = train_ds.batch(100, drop_remainder=False)

train_ds = train_ds.prefetch(AUTOTUNE)
try:
    if tf.config.list_physical_devices("GPU"):
        from tensorflow.data.experimental import prefetch_to_device

        train_ds = train_ds.apply(prefetch_to_device("/GPU:0"))
except Exception:
    pass



## === cell 2
from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling=None,
    classes=7,
)

model.compile(
    optimizer="SGD",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_ds, epochs=12, verbose=2)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3540087243.py in <cell line: 0>()
     16 )
     17 
---> 18 model.fit(train_ds, epochs=12, verbose=2)
     19 

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

Detected at node DecodeAndCropJpeg defined at (most recent call last):
<stack traces unavailable>
jpeg::Uncompress failed. Invalid JPEG data or crop window.
	 [[{{node DecodeAndCropJpeg}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_18988]

## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB)
test_files = sample_sub["image"].to_numpy()

test_dir_t = tf.constant(TEST_IMG_DIR)


@tf.function
def _load_test_from_name(fname):
    path = tf.strings.join([test_dir_t, fname], separator=os.sep)
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_and_crop_jpeg(img_bytes, crop_window=[0, 0, 0, 0], channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


test_options = tf.data.Options()
test_options.experimental_deterministic = True
test_options.experimental_optimization.apply_default_optimizations = True
test_options.experimental_optimization.map_parallelization = True

test_ds = tf.data.Dataset.from_tensor_slices(test_files).with_options(test_options)
test_ds = test_ds.map(_load_test_from_name, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(64, drop_remainder=False).prefetch(AUTOTUNE)

try:
    if tf.config.list_physical_devices("GPU"):
        from tensorflow.data.experimental import prefetch_to_device

        test_ds = test_ds.apply(prefetch_to_device("/GPU:0"))
except Exception:
    pass

pred = model.predict(test_ds, verbose=2)
pred_idx = np.argmax(pred, axis=1)

pred_labels = np.take(np.array(label_class, dtype=object), pred_idx)

sub = pd.DataFrame({"image": test_files, "labels": pred_labels})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/656006528.py in <cell line: 0>()
     34     pass
     35 
---> 36 pred = model.predict(test_ds, verbose=2)
     37 pred_idx = np.argmax(pred, axis=1)
     38 

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

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} jpeg::Uncompress failed. Invalid JPEG data or crop window.
	 [[{{function_node __inference__load_test_from_name_19010}}{{node DecodeAndCropJpeg}}]] [Op:IteratorGetNext] name:
