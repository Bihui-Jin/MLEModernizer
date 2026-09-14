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

0.7773961218836576

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, re, random, math, glob

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K

print("tf:", tf.__version__)
print("tf.keras:", tf.keras.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import pathlib

SEED = 1337
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




## === cell 2
@tf.function
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST", ratio=4)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.AREA)
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 32
IMAGE_SIZE = (512, 512)




## === cell 4
INPUT_DIR = "../input/plant-pathology-2021-fgvc8"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "../input"

TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_DIR = os.path.join(INPUT_DIR, "test_images")

if not os.path.exists(TRAIN_CSV):
    alt = "../input/plant-pathology-2021-fgvc8/train.csv"
    if os.path.exists(alt):
        TRAIN_CSV = alt
if not os.path.exists(SAMPLE_SUB):
    alt = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
    if os.path.exists(alt):
        SAMPLE_SUB = alt
if not os.path.exists(TRAIN_DIR):
    alt = "../input/plant-pathology-2021-fgvc8/train_images"
    if os.path.exists(alt):
        TRAIN_DIR = alt
if not os.path.exists(TEST_DIR):
    alt = "../input/plant-pathology-2021-fgvcvc8/test_images"
    if os.path.exists(alt):
        TEST_DIR = alt
    else:
        alt = "../input/plant-pathology-2021-fgvc8/test_images"
        if os.path.exists(alt):
            TEST_DIR = alt

if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"Could not find train.csv. Tried: {TRAIN_CSV}")
if not os.path.exists(SAMPLE_SUB):
    raise FileNotFoundError(
        f"Could not find sample_submission.csv. Tried: {SAMPLE_SUB}"
    )
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Could not find train_images dir. Tried: {TRAIN_DIR}")
if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Could not find test_images dir. Tried: {TEST_DIR}")

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

print("train_df:", train_df.shape, "sample_sub:", sample_sub.shape)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)




## === cell 5
CLASSES = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew", "healthy"]
class_to_idx = {c: i for i, c in enumerate(CLASSES)}


def labels_to_multi_hot(label_str):
    if not isinstance(label_str, str):
        label_str = ""
    parts = [p for p in label_str.strip().split(" ") if p]
    y = np.zeros((len(CLASSES),), dtype=np.float32)
    for p in parts:
        if p in class_to_idx:
            y[class_to_idx[p]] = 1.0
    return y


train_df["filepath"] = TRAIN_DIR.rstrip("/") + "/" + train_df["image"].astype(str)
train_df["target"] = train_df["labels"].map(labels_to_multi_hot)

X_paths = train_df["filepath"].to_numpy()
y = np.stack(train_df["target"].values)

print("X_paths:", X_paths.shape, "y:", y.shape, "positive rate:", y.mean(axis=0))




## === cell 6
AUTO = tf.data.experimental.AUTOTUNE

aug = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ]
)


def _ds_options_deterministic():
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.map_and_batch_fusion = True
        opts.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    return opts


def make_train_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(
        _ds_options_deterministic()
    )
    ds = ds.shuffle(
        buffer_size=min(len(paths), 16384), seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.map(
        lambda p, l: decode_image(p, l, image_size=IMAGE_SIZE), num_parallel_calls=AUTO
    )
    ds = ds.map(lambda x, l: (aug(x, training=True), l), num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


def make_valid_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(
        _ds_options_deterministic()
    )
    ds = ds.map(
        lambda p, l: decode_image(p, l, image_size=IMAGE_SIZE), num_parallel_calls=AUTO
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


idx = np.arange(len(X_paths))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_size = int(0.1 * len(idx))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

train_ds = make_train_ds(X_paths[trn_idx], y[trn_idx])
valid_ds = make_valid_ds(X_paths[val_idx], y[val_idx])

print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Valid batches:", tf.data.experimental.cardinality(valid_ds).numpy())




## === cell 7
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




## === cell 8
def find_model_file():
    preferred = [
        "../input/SMResNet50.h5",
        "../input/plant-pathology-2021-fgvc8/SMResNet50.h5",
        "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/SMResNet50.h5",
        "/kaggle/input/SMResNet50.h5",
        "/kaggle/input/plant-pathology-2021-fgvc8/SMResNet50.h5",
        "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/SMResNet50.h5",
    ]
    for p in preferred:
        if os.path.isfile(p):
            return p

    patterns = [
        "../input/*.h5",
        "../input/*/*.h5",
        "../input/*/*/*.h5",
        "/kaggle/input/*.h5",
        "/kaggle/input/*/*.h5",
        "/kaggle/input/*/*/*.h5",
        "../input/*.keras",
        "../input/*/*.keras",
        "../input/*/*/*.keras",
        "/kaggle/input/*.keras",
        "/kaggle/input/*/*.keras",
        "/kaggle/input/*/*/*.keras",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    candidates = [p for p in candidates if os.path.isfile(p)]
    bad = ("tokenizer", "embedding", "bert", "gpt")
    candidates = [
        p for p in candidates if not any(b in os.path.basename(p).lower() for b in bad)
    ]
    candidates.sort()
    return candidates[0] if len(candidates) else None


model_path = find_model_file()
model = None

if model_path is None:
    raise FileNotFoundError(
        "No external model file found (expected SMResNet50.h5 or similar in ../input). "
        "Training is disabled to ensure runtime < 600s."
    )

print("Loading model:", model_path)
model = tf.keras.models.load_model(
    model_path,
    compile=False,
    custom_objects={"FixedDropout": FixedDropout},
)

try:
    model.jit_compile = True
except Exception:
    pass




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1286077778.py in <cell line: 0>()
     45 # we fail fast instead of timing out.
     46 if model_path is None:
---> 47     raise FileNotFoundError(
     48         "No external model file found (expected SMResNet50.h5 or similar in ../input). "
     49         "Training is disabled to ensure runtime < 600s."

FileNotFoundError: No external model file found (expected SMResNet50.h5 or similar in ../input). Training is disabled to ensure runtime < 600s.

## === cell 9
test_images = sample_sub["image"].astype(str).to_numpy()
IMAGE_PATHS = (TEST_DIR.rstrip("/") + "/" + test_images).astype(object)

opts = tf.data.Options()
opts.experimental_deterministic = True
try:
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.map_and_batch_fusion = True
    opts.experimental_optimization.parallel_batch = True
except Exception:
    pass

test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .with_options(opts)
    .map(
        lambda p: decode_image(p, label=None, image_size=IMAGE_SIZE),
        num_parallel_calls=AUTO,
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
    .apply(tf.data.experimental.ignore_errors())
)

probs = model.predict(test_dataset, verbose=1)
temp_probs = np.asarray(probs)

print("Raw probs shape:", temp_probs.shape)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2224453389.py in <cell line: 0>()
     24 )
     25 
---> 26 probs = model.predict(test_dataset, verbose=1)
     27 temp_probs = np.asarray(probs)
     28 

AttributeError: 'NoneType' object has no attribute 'predict'

## === cell 10
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.27, 1: 0.5, 2: 0.3, 3: 0.5, 4: 0.5}

probs5 = (
    temp_probs[:, :5]
    if (temp_probs.ndim == 2 and temp_probs.shape[1] >= 5)
    else np.zeros((len(temp_probs), 5), dtype=np.float32)
)
thr = np.array([threshold[i] for i in range(5)], dtype=probs5.dtype)
sel = probs5 > thr  # (N,5)
count = sel.sum(axis=1)

add_complex = (count >= 2) & (~sel[:, 2])

base_names = np.array([name[i] for i in range(5)], dtype=object)

pred_string = []
for i in range(sel.shape[0]):
    parts = base_names[sel[i]].tolist()
    if add_complex[i]:
        parts.append("complex")
    if not parts:
        parts = [name[5]]
    pred_string.append(" ".join(parts))

print("Num predictions:", len(pred_string))




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2271372568.py in <cell line: 0>()
     12 probs5 = (
     13     temp_probs[:, :5]
---> 14     if (temp_probs.ndim == 2 and temp_probs.shape[1] >= 5)
     15     else np.zeros((len(temp_probs), 5), dtype=np.float32)
     16 )

NameError: name 'temp_probs' is not defined

## === cell 11
if len(pred_string) != len(sample_sub):
    raise ValueError(
        f"Prediction length mismatch: got {len(pred_string)} preds, expected {len(sample_sub)}"
    )

sub = pd.DataFrame({"image": test_images.tolist(), "labels": pred_string})

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote", sub_path)
print(sub.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1914345702.py in <cell line: 0>()
----> 1 if len(pred_string) != len(sample_sub):
      2     raise ValueError(
      3         f"Prediction length mismatch: got {len(pred_string)} preds, expected {len(sample_sub)}"
      4     )
      5 

NameError: name 'pred_string' is not defined
