# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.2924284395198524

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.21955) has done: 'I fix the import crash by removing the problematic `from sklearn import *` (it’s unnecessary here and triggers the protobuf `MessageFactory` error in Kaggle). Then I fix the test image glob/path joining so it reliably finds all `test_images/*.jpg`, and replace the missing external model (`../input/resnet50-customized1-model/abc.h5`) with an on-the-fly ResNet50 backbone + Dense head that preserves the “ResNet feature extractor → predict probabilities → threshold → space-delimited labels” core flow. Finally, I ensure the submission uses the exact `sample_submission.csv` image order and writes a valid `submission.csv` with columns `image,labels` (and avoid attaching dataframes to the pandas module). This run end-to-end and produce a valid submission file.'
- What this solution (achieved 0.24507) has done: 'Main bottlenecks are (1) `train_ds.cache()` attempting to cache thousands of decoded+resized 300x300 images in memory (very slow / may thrash) and (2) unnecessary graph retracing/extra Python overhead in the tf.data pipeline. I keep the exact same model, loss, epochs, steps, thresholds, and semantics, but change caching to a disk cache in `/tmp` (same samples, same order; just avoids recomputing decode/resize without blowing RAM) and ensure the map runs as a compiled TF graph with parallel calls and deterministic options unchanged. I also make the label-map conversion (`class2tags`) vectorized to reduce Python loop time at submission creation, without changing any predicted labels. These changes are runtime-only improvements and preserve the algorithm’s outputs up to negligible float differences.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import gc
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import tensorflow.keras.applications.resnet50 as resnet

K.set_image_data_format("channels_last")
print("keras:", keras.__version__, "tf:", tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

tfdata_opts = tf.data.Options()
tfdata_opts.experimental_deterministic = True
try:
    tfdata_opts.experimental_optimization.apply_default_optimizations = True
    tfdata_opts.experimental_optimization.map_parallelization = True
    tfdata_opts.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tfdata_opts.experimental_threading.private_threadpool_size = 16
    tfdata_opts.experimental_threading.max_intra_op_parallelism = 0
except Exception:
    pass


def _resolve_comp_root():
    candidates = [
        "../input/plant-pathology-2021-fgvc8",
        "/kaggle/input/plant-pathology-2021-fgvc8",
        "/kaggle/data/plant-pathology-2021-fgvc8",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


COMP_ROOT = _resolve_comp_root()
print("COMP_ROOT:", COMP_ROOT)



## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3
TOTAL_INPUTS = NR_CHANNELS * IMG_HEIGHT * IMG_WIDTH

BATCH_SIZE = 16
EPOCHS = 2  # keep identical

MAX_STEPS_PER_EPOCH = 256  # keep identical



## === cell 2
TEST_DIR = os.path.join(COMP_ROOT, "test_images")
imglist_test = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
print("Found test images:", len(imglist_test))
print("First test image:", imglist_test[0] if imglist_test else None)




## === cell 3
@tf.function
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(
        contents=img_bytes, channels=NR_CHANNELS, dct_method="INTEGER_FAST"
    )
    img = tf.image.resize(
        img, (IMG_HEIGHT, IMG_WIDTH), method="bilinear", antialias=False
    )
    img = tf.cast(img, tf.float32)
    img = resnet.preprocess_input(img)
    img.set_shape((IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))
    return img


test_paths_tf = tf.constant(imglist_test)

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths_tf)
    .with_options(tfdata_opts)
    .map(_decode_resize_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
print("Built test_ds for inference.")



## === cell 4
pass



## === cell 5
training_csv = pd.read_csv(os.path.join(COMP_ROOT, "train.csv"))

all_tokens = training_csv["labels"].astype(str).str.split()

tagnames = np.unique(np.concatenate(all_tokens.to_numpy()))
tagnames = np.array(sorted(list(tagnames)))  # deterministic order
n_classes = len(tagnames)
print("Number of classes:", n_classes)
print("Classes:", tagnames)

exploded = training_csv["labels"].astype(str).str.split().explode()
y_df = pd.get_dummies(exploded).groupby(level=0).max()
y_df = y_df.reindex(columns=tagnames, fill_value=0)
y_train = y_df.to_numpy(dtype=np.float32, copy=False)

TRAIN_DIR = os.path.join(COMP_ROOT, "train_images")
img_paths_train = [os.path.join(TRAIN_DIR, fn) for fn in training_csv["image"].values]

print("Prepared train paths:", len(img_paths_train), "y_train shape:", y_train.shape)



## === cell 6
train_paths_tf = tf.constant(img_paths_train)
y_train_tf = tf.constant(y_train)

train_steps = int(
    min(MAX_STEPS_PER_EPOCH, int(np.ceil(len(img_paths_train) / BATCH_SIZE)))
)
needed_examples = train_steps * BATCH_SIZE * EPOCHS
print(
    "Train steps per epoch:",
    train_steps,
    "needed_examples (upper bound):",
    needed_examples,
)

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths_tf, y_train_tf)
).with_options(tfdata_opts)

shuffle_buf = min(len(img_paths_train), 4096)
train_ds = train_ds.shuffle(
    buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.take(needed_examples)


@tf.function
def _decode_resize_preprocess_with_label(path, y):
    return _decode_resize_preprocess(path), y


cache_path = os.path.join("/tmp", "pp2021_train_cache_v1")
train_ds = (
    train_ds.map(_decode_resize_preprocess_with_label, num_parallel_calls=AUTOTUNE)
    .cache(cache_path)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)



## === cell 7
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))
base = resnet.ResNet50(
    include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
)

base.trainable = False
outputs = keras.layers.Dense(n_classes, activation="sigmoid")(base.output)
model_f = keras.Model(inputs=inputs, outputs=outputs)

model_f.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

print(model_f.summary())



## === cell 8
history = model_f.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    verbose=1,
)



## === cell 9
X_test = model_f.predict(test_ds, verbose=1)
print("X_test shape:", X_test.shape)



## === cell 10
if X_test.size:
    print(
        "X_test[0] summary: min/max/mean",
        float(X_test[0].min()),
        float(X_test[0].max()),
        float(X_test[0].mean()),
    )




## === cell 11
def class2tags(classes, tagnames):
    tagnames_arr = np.asarray(tagnames, dtype=object)
    has_healthy = np.any(tagnames_arr == "healthy")
    default = "healthy" if has_healthy else ""

    classes = np.asarray(classes, dtype=bool)
    idxs_per_row = [np.flatnonzero(r) for r in classes]

    out = []
    for idxs in idxs_per_row:
        if idxs.size == 0:
            out.append(default)
        else:
            out.append(" ".join(tagnames_arr[idxs].tolist()))
    return out




## === cell 12
test_predclass = X_test > 0.5
test_predtags = class2tags(test_predclass, tagnames)
print("Example predicted tags:", test_predtags[0] if len(test_predtags) else None)



## === cell 13
del test_predclass
gc.collect()



## === cell 14
sample_sub = pd.read_csv(os.path.join(COMP_ROOT, "sample_submission.csv"))
print(sample_sub.head())

pred_map = {os.path.basename(p): tag for p, tag in zip(imglist_test, test_predtags)}



## === cell 15
sub = sample_sub.copy()
sub["labels"] = sub["image"].map(pred_map)

default_label = "healthy" if "healthy" in tagnames else ""
sub["labels"] = sub["labels"].fillna(default_label)

print(sub.head())
print("Submission rows:", len(sub), "null labels:", sub["labels"].isna().sum())



## === cell 16
sub[["labels"]].head()



## === cell 17
sub.head()



## === cell 18
sub.to_csv("./submission.csv", index=False)
print("Wrote ./submission.csv with shape:", sub.shape)
