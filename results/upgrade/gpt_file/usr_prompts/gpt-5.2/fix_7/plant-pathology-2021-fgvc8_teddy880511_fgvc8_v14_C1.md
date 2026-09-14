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

0.2613481071098803

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

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

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

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "../input/plant-pathology-2021-fgvc8"
train_imgpath = os.path.join(BASE_PATH, "train_images")
train_csvpath = os.path.join(BASE_PATH, "train.csv")
test_imgpath = os.path.join(BASE_PATH, "test_images")
sample_csvpath = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.exists(train_imgpath), f"Missing: {train_imgpath}"
assert os.path.exists(train_csvpath), f"Missing: {train_csvpath}"
assert os.path.exists(test_imgpath), f"Missing: {test_imgpath}"
assert os.path.exists(sample_csvpath), f"Missing: {sample_csvpath}"

train_df = pd.read_csv(train_csvpath)
sample_df = pd.read_csv(sample_csvpath)

print(train_df.shape, sample_df.shape)
print(train_df.columns.tolist(), sample_df.columns.tolist())
train_df.head()



## === cell 2
label_class = [
    "scab",
    "healthy",
    "frog_eye_leaf_spot",
    "cider_apple_rust",
    "complex",
    "powdery_mildew",
    "scab frog_eye_leaf_spot",
    "scab frog_eye_leaf_spot complex",
    "frog_eye_leaf_spot complex",
    "rust frog_eye_leaf_spot",
    "rust complex",
    "powdery_mildew complex",
]
label_to_idx = {l: i for i, l in enumerate(label_class)}

UNKNOWN_FALLBACK = label_to_idx["complex"]

y_idx = (
    train_df["labels"]
    .map(label_to_idx)
    .fillna(UNKNOWN_FALLBACK)
    .astype(np.int32)
    .to_numpy()
)

y_train = tf.keras.utils.to_categorical(y_idx, num_classes=len(label_class)).astype(
    np.float32
)

train_df["label_num"] = y_idx
print("Label distribution (label_num):")
print(train_df["label_num"].value_counts().sort_index())



## === cell 3
IMG_SIZE = (64, 64)

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 100

train_paths_np = (train_imgpath + os.sep + train_df["image"].values.astype(str)).astype(
    str
)
train_paths_tf = tf.constant(train_paths_np)
y_train_tf = tf.constant(y_train)  # float32 one-hot


@tf.function(reduce_retracing=True)
def _load_and_preprocess(path, y):
    bytes_ = tf.io.read_file(path)
    img = tf.io.decode_jpeg(bytes_, channels=3, dct_method="INTEGER_FAST")  # uint8
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, (IMG_SIZE[0], IMG_SIZE[1], 3))
    y = tf.ensure_shape(y, (len(label_class),))
    return img, y


train_ds = tf.data.Dataset.from_tensor_slices((train_paths_tf, y_train_tf))

options = tf.data.Options()
options.deterministic = True
options.experimental_slack = True
try:
    options.threading.private_threadpool_size = max(8, (os.cpu_count() or 8))
    options.threading.max_intra_op_parallelism = 1
except Exception:
    pass
train_ds = train_ds.with_options(options)

train_ds = train_ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache()

train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

steps_per_epoch = (
    train_df.shape[0] // BATCH_SIZE
)  # matches drop_remainder=True semantics
print("Prepared train_ds with", train_df.shape[0], "samples")
print("steps_per_epoch:", steps_per_epoch)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
UFuncTypeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2042827441.py in <cell line: 0>()
      6 # Speed fix (correctness-preserving): build full file paths once in NumPy instead of tf.strings ops.
      7 # This avoids per-element string tensor joins and reduces input pipeline overhead without changing which files are read.
----> 8 train_paths_np = (train_imgpath + os.sep + train_df["image"].values.astype(str)).astype(
      9     str
     10 )

UFuncTypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U49'), dtype('<U20')) -> None

## === cell 4
from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(64, 64, 3),
    pooling=None,
    classes=len(label_class),
)

model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    epochs=20,
    steps_per_epoch=steps_per_epoch,
    verbose=2,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/385487863.py in <cell line: 0>()
     20 # it exactly matches drop_remainder=True batching above.
     21 history = model.fit(
---> 22     train_ds,
     23     epochs=20,
     24     steps_per_epoch=steps_per_epoch,

NameError: name 'train_ds' is not defined

## === cell 5
test_files = sorted(
    [
        e.name
        for e in os.scandir(test_imgpath)
        if e.is_file() and e.name.endswith(".jpg")
    ]
)

test_paths_np = (test_imgpath + os.sep + np.array(test_files, dtype=str)).astype(str)
test_paths_tf = tf.constant(test_paths_np)


@tf.function(reduce_retracing=True)
def _load_and_preprocess_test(path):
    bytes_ = tf.io.read_file(path)
    img = tf.io.decode_jpeg(bytes_, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, (IMG_SIZE[0], IMG_SIZE[1], 3))
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths_tf)
test_ds = test_ds.with_options(options)
test_ds = test_ds.map(_load_and_preprocess_test, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

pred = model.predict(test_ds, verbose=2)
pred_idx = np.argmax(pred, axis=1)
pred_labels = [label_class[j] for j in pred_idx]

sub = pd.DataFrame({"image": test_files, "labels": pred_labels})

sub = sample_df[["image"]].merge(sub, on="image", how="left")
sub["labels"] = sub["labels"].fillna("healthy")

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
sub.head()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
UFuncTypeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1837311351.py in <cell line: 0>()
      8 
      9 # Speed fix (correctness-preserving): build full file paths once in NumPy instead of tf.strings ops.
---> 10 test_paths_np = (test_imgpath + os.sep + np.array(test_files, dtype=str)).astype(str)
     11 test_paths_tf = tf.constant(test_paths_np)
     12 

UFuncTypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U48'), dtype('<U20')) -> None
