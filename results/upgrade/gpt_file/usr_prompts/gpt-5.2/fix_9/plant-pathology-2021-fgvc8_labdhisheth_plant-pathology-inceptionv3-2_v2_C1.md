# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import gc
import re
import math
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def _import_tf_with_fallback():
    try:
        import tensorflow as tf  # noqa: F401

        return tf, "python"
    except Exception as e1:
        import sys

        for k in list(sys.modules.keys()):
            if k.startswith("tensorflow"):
                sys.modules.pop(k, None)
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "upb"
        try:
            import tensorflow as tf  # noqa: F401

            return tf, "upb"
        except Exception as e2:
            raise RuntimeError(
                "TensorFlow failed to import with both protobuf implementations.\n"
                f"First error (python): {repr(e1)}\n"
                f"Second error (upb): {repr(e2)}"
            )


tf, pb_impl = _import_tf_with_fallback()
from tensorflow import keras
import tensorflow.keras.layers as L

from sklearn.preprocessing import MultiLabelBinarizer

np.random.seed(0)
tf.random.set_seed(0)

print("TF version:", tf.__version__)
print("Using protobuf implementation:", pb_impl)

try:
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception as e:
    print("Thread config set failed (continuing):", repr(e))

try:
    tf.config.optimizer.set_jit(True)
    print("XLA JIT enabled")
except Exception as e:
    print("XLA JIT enable failed (continuing):", repr(e))




## === cell 1
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16

CANDIDATE_ROOTS = [
    "../input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvcvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8",
]
DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.exists(r):
        DATA_ROOT = r
        break
if DATA_ROOT is None:
    DATA_ROOT = "../input/plant-pathology-2021-fgvc8"

TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("SUB_PATH exists:", os.path.exists(SUB_PATH))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))

sub = pd.read_csv(SUB_PATH)
test_data = sub.copy()
train_data = pd.read_csv(TRAIN_PATH)

train_data["labels"] = train_data["labels"].apply(lambda string: string.split(" "))
s = list(train_data["labels"])
mlb = MultiLabelBinarizer()
trainx = pd.DataFrame(
    mlb.fit_transform(s), columns=mlb.classes_, index=train_data.index
)

print("Train one-hot shape:", trainx.shape)
print("Classes:", list(mlb.classes_))

idx_to_label = {i: c for i, c in enumerate(mlb.classes_)}
print("Index->label mapping:", idx_to_label)
NUM_CLASSES = len(mlb.classes_)
print("NUM_CLASSES:", NUM_CLASSES)




## === cell 2
def format_path(st):
    return os.path.join(TEST_IMG_DIR, str(st))


def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    shape = tf.io.extract_jpeg_shape(bits)
    crop_window = tf.stack([0, 0, shape[0], shape[1]])
    image = tf.io.decode_and_crop_jpeg(bits, crop_window, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    return image, label


def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if label is None:
        return image
    return image, label


test_paths = test_data.image.apply(format_path).values

_missing = [p for p in test_paths[:20] if not os.path.exists(p)]
if _missing:
    raise FileNotFoundError(
        "Test image paths not found. Example missing paths:\n" + "\n".join(_missing)
    )

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

print("Test samples:", len(test_paths))




## === cell 3
inputs = tf.keras.Input(shape=(512, 512, 3))
x = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
)(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
model = tf.keras.models.Model(inputs, outputs)

model.compile(run_eagerly=False)

model.summary()




## === cell 4
preds = model.predict(test_dataset, verbose=1)




## === cell 5
preds = np.asarray(preds)

THRESH = 0.30

mask = preds >= THRESH
argmax_idx = preds.argmax(axis=1)
rows_no_hit = ~mask.any(axis=1)
mask[rows_no_hit, :] = False
mask[rows_no_hit, argmax_idx[rows_no_hit]] = True

classes = np.array([idx_to_label[i] for i in range(NUM_CLASSES)], dtype=object)

selected = [classes[row_mask].tolist() for row_mask in mask]
testlabels = [" ".join(lbls).strip() for lbls in selected]

assert len(testlabels) == len(sub), (len(testlabels), len(sub))

sub["labels"] = testlabels
sub = sub[["image", "labels"]]
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
