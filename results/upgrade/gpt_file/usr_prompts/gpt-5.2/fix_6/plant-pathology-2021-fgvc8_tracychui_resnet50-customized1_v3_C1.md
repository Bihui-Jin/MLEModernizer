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

0.4357340720221596

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.24507) has done: 'The timeout is dominated by expensive image decoding/resizing inside `tf.data` and by running ResNet50 at 300×300 with a small batch size, which increases step overhead. I keep the exact same model/fit/predict logic, but speed up the input pipeline by enabling TF data optimizations, caching decoded+preprocessed images (train/val/test) to avoid repeated work across epochs/validation, and using non-blocking prefetch and dataset options earlier. I also raise batch size (when possible) to reduce Python/TF step overhead without changing training semantics, and remove a few avoidable Python-side conversions in label processing while keeping identical outputs. These changes are provably equivalent in terms of data/labels/model math, just reducing redundant computation and overhead.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import gc

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


K.set_image_data_format("channels_last")
print("keras:", keras.__version__, "tf:", tf.__version__)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("XLA JIT not enabled:", e)

DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3

BATCH_SIZE = 32

EPOCHS = 2  # keep small for runtime; preserves core logic (train then predict)


## === cell 2
imglist_test = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
print("num test images:", len(imglist_test))
assert len(imglist_test) > 0, "No test images found - check TEST_IMG_DIR"


## === cell 3
import tensorflow.keras.applications.resnet50 as resnet


@tf.function
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize_with_pad(
        img, IMG_HEIGHT, IMG_WIDTH, method="bilinear", antialias=False
    )
    img = tf.cast(img, tf.float32)
    img = resnet.preprocess_input(img)
    return img


def _ds_options():
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.autotune_buffers = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    return opts


def make_image_dataset_from_paths(paths, batch_size: int):
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = path_ds.with_options(_ds_options())
    ds = ds.map(_decode_resize_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_image_dataset_from_paths(imglist_test, BATCH_SIZE)
print("Built test_ds for streaming inference.")


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4191000816.py in <cell line: 0>()
     40 
     41 
---> 42 test_ds = make_image_dataset_from_paths(imglist_test, BATCH_SIZE)
     43 print("Built test_ds for streaming inference.")

/tmp/ipykernel_11/4191000816.py in make_image_dataset_from_paths(paths, batch_size)
     28 def make_image_dataset_from_paths(paths, batch_size: int):
     29     path_ds = tf.data.Dataset.from_tensor_slices(paths)
---> 30     ds = path_ds.with_options(_ds_options())
     31     ds = ds.map(_decode_resize_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
     32 

/tmp/ipykernel_11/4191000816.py in _ds_options()
     20     opts.experimental_deterministic = True
     21     opts.experimental_optimization.apply_default_optimizations = True
---> 22     opts.experimental_optimization.autotune_buffers = True
     23     opts.experimental_optimization.map_parallelization = True
     24     opts.experimental_optimization.parallel_batch = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 4
first_img = _decode_resize_preprocess(tf.constant(imglist_test[0]))
first_min = float(tf.reduce_min(first_img).numpy())
first_max = float(tf.reduce_max(first_img).numpy())
print("Example preprocessed pixel stats (min/max):", first_min, first_max)


## === cell 5
training_csv = pd.read_csv(TRAIN_CSV)

training_class = []
for labels in pd.unique(training_csv["labels"]):
    training_class.extend(str(labels).split())
tagnames = np.unique(np.array(training_class))
num_classes = len(tagnames)
print("num_classes:", num_classes)
print("tagnames:", tagnames)

tag2idx = {t: i for i, t in enumerate(tagnames)}


def labels_series_to_multihot(
    labels: pd.Series, tag2idx: dict, num_classes: int
) -> np.ndarray:
    split_lists = labels.fillna("").astype(str).str.split()

    rows = []
    cols = []
    for r, tags in enumerate(split_lists):
        for t in tags:
            j = tag2idx.get(t, None)
            if j is not None:
                rows.append(r)
                cols.append(j)

    y = np.zeros((len(split_lists), num_classes), dtype=np.float32)
    if rows:
        y[np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)] = 1.0
    return y


idx = np.arange(len(training_csv))
np.random.shuffle(idx)
val_size = int(0.1 * len(idx))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

train_df = training_csv.iloc[trn_idx].reset_index(drop=True)
val_df = training_csv.iloc[val_idx].reset_index(drop=True)

print("train size:", len(train_df), "val size:", len(val_df))


def make_dataset(df: pd.DataFrame, img_dir: str, batch_size: int, training: bool):
    paths_np = np.array(
        [os.path.join(img_dir, fn) for fn in df["image"].values], dtype=object
    )
    y = labels_series_to_multihot(df["labels"], tag2idx, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((paths_np, y))
    ds = ds.with_options(_ds_options())

    @tf.function
    def _load(path, target):
        return _decode_resize_preprocess(path), target

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    ds = ds.repeat()
    return ds


train_ds = make_dataset(train_df, TRAIN_IMG_DIR, BATCH_SIZE, training=True)
val_ds = make_dataset(val_df, TRAIN_IMG_DIR, BATCH_SIZE, training=False)

steps_per_epoch = int(np.ceil(len(train_df) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

base = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
)
base.trainable = False  # feature extractor (consistent with using a pre-trained model)

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))
x = base(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
model_f = keras.Model(inputs, outputs)

model_f.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model_f.summary()

history = model_f.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=2,
)

X_test = model_f.predict(test_ds, verbose=1)
print("X_test:", X_test.shape, X_test.dtype)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3449206619.py in <cell line: 0>()
     77 
     78 
---> 79 train_ds = make_dataset(train_df, TRAIN_IMG_DIR, BATCH_SIZE, training=True)
     80 val_ds = make_dataset(val_df, TRAIN_IMG_DIR, BATCH_SIZE, training=False)
     81 

/tmp/ipykernel_11/3449206619.py in make_dataset(df, img_dir, batch_size, training)
     54 
     55     ds = tf.data.Dataset.from_tensor_slices((paths_np, y))
---> 56     ds = ds.with_options(_ds_options())
     57 
     58     @tf.function

/tmp/ipykernel_11/4191000816.py in _ds_options()
     20     opts.experimental_deterministic = True
     21     opts.experimental_optimization.apply_default_optimizations = True
---> 22     opts.experimental_optimization.autotune_buffers = True
     23     opts.experimental_optimization.map_parallelization = True
     24     opts.experimental_optimization.parallel_batch = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 6
print("Pred sample:", X_test[0])




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3979816819.py in <cell line: 0>()
----> 1 print("Pred sample:", X_test[0])
      2 
      3 

NameError: name 'X_test' is not defined

## === cell 7
def class2tags(classes, tagnames):
    tags = []
    for n in range(classes.shape[0]):
        idxs = np.flatnonzero(classes[n])
        if idxs.size == 0:
            tags.append("healthy")
        else:
            tags.append(" ".join(tagnames[idxs]))
    return tags




## === cell 8
threshold = 0.3
test_predclass = X_test > threshold
test_predtags = class2tags(test_predclass, tagnames)
print("Example tags:", test_predtags[:5])


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1673210957.py in <cell line: 0>()
      1 threshold = 0.3
----> 2 test_predclass = X_test > threshold
      3 test_predtags = class2tags(test_predclass, tagnames)
      4 print("Example tags:", test_predtags[:5])

NameError: name 'X_test' is not defined

## === cell 9
del test_predclass
gc.collect()


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/591404338.py in <cell line: 0>()
----> 1 del test_predclass
      2 gc.collect()

NameError: name 'test_predclass' is not defined

## === cell 10
sub = pd.read_csv(SAMPLE_SUB)
sub_images = sub["image"].tolist()

pred_map = {os.path.basename(p): tag for p, tag in zip(imglist_test, test_predtags)}

sub["labels"] = [pred_map.get(img, "healthy") for img in sub_images]

assert list(sub.columns) == ["image", "labels"]
assert len(sub) == len(pd.read_csv(SAMPLE_SUB))

sub.head()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/23536540.py in <cell line: 0>()
      2 sub_images = sub["image"].tolist()
      3 
----> 4 pred_map = {os.path.basename(p): tag for p, tag in zip(imglist_test, test_predtags)}
      5 
      6 sub["labels"] = [pred_map.get(img, "healthy") for img in sub_images]

NameError: name 'test_predtags' is not defined

## === cell 11
out_path = "./submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub))
print(sub.head(3).to_string(index=False))
