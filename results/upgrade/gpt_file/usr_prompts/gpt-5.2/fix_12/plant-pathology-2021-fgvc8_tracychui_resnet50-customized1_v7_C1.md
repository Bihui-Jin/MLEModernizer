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

0.4426223453370249

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.24507) has done: 'The timeout is dominated by (1) building and iterating the `tf.data` pipelines with disk-cache files, plus the extra `take(1)`/printing that forces an additional full pipeline instantiation, and (2) slower input throughput due to non-fused JPEG decode/resize and conservative dataset options. I keep the exact same preprocessing, model, training (2 epochs), and prediction semantics, but speed up I/O by switching to `tf.image.decode_and_crop_jpeg`-equivalent fast path (`tf.io.decode_jpeg` remains), enabling deterministic-safe parallelism, removing on-disk caching (which is slower than streaming here and adds filesystem overhead), and ensuring the dataset is built once without extra warmup iterations. I also avoid expensive Python-side list creation where possible and reduce redundant work/objects, while preserving identical outputs up to negligible float differences.'

# 9. Code solution

## === cell 0
import os
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

os.environ.setdefault("PYTHONHASHSEED", "0")
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3

DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")




## === cell 2
imglist_test = tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
imglist_test = sorted(imglist_test)
len(imglist_test)




## === cell 3
@tf.function
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32)
    img = resnet.preprocess_input(img)
    img.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
    return img


options = tf.data.Options()
options.experimental_deterministic = False
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_slack = True
except Exception:
    pass

TEST_BATCH = 128

test_files_ds = tf.data.Dataset.from_tensor_slices(imglist_test).with_options(options)

test_ds = (
    test_files_ds.map(
        _decode_resize_preprocess, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    .batch(TEST_BATCH, drop_remainder=False)
    .prefetch(AUTOTUNE)
)





## === cell 4
print(
    "Num test images:",
    len(imglist_test),
    "Example path:",
    imglist_test[0] if imglist_test else None,
)




## === cell 5
training_csv = pd.read_csv(TRAIN_CSV)

tagnames = np.sort(
    pd.Series(training_csv["labels"].astype(str).str.split())
    .explode()
    .dropna()
    .unique()
)
NUM_CLASSES = len(tagnames)

print("Num classes:", NUM_CLASSES)
print("Classes:", tagnames)




## === cell 6
tag2idx = {t: i for i, t in enumerate(tagnames)}

labels_series = training_csv["labels"].astype(str).str.get_dummies(sep=" ")
y = labels_series.reindex(columns=tagnames, fill_value=0).to_numpy(
    dtype=np.float32, copy=False
)

MAX_TRAIN = 2048
train_df = training_csv.iloc[:MAX_TRAIN].reset_index(drop=True)
y_train = y[:MAX_TRAIN]

train_paths = np.char.add(TRAIN_IMG_DIR + os.sep, train_df["image"].values).astype(str)


@tf.function
def _train_map(p, yy):
    return _decode_resize_preprocess(p), yy


TRAIN_BATCH = 32

train_base_ds = tf.data.Dataset.from_tensor_slices((train_paths, y_train)).with_options(
    options
)

train_ds = (
    train_base_ds.map(_train_map, num_parallel_calls=AUTOTUNE, deterministic=False)
    .batch(TRAIN_BATCH, drop_remainder=True)
    .prefetch(AUTOTUNE)
)

print("Train size:", len(train_df), "Train y:", y_train.shape)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3577070813.py in <cell line: 0>()
     12 
     13 # Speed: avoid .tolist() (extra Python overhead); use NumPy array directly.
---> 14 train_paths = np.char.add(TRAIN_IMG_DIR + os.sep, train_df["image"].values).astype(str)
     15 
     16 

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U49' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 7
base = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
)
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))
x = base(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
model_f = keras.Model(inputs, outputs)

base.trainable = False

model_f.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model_f.fit(
    train_ds,
    epochs=2,
    verbose=1,
)

X_test = model_f.predict(test_ds, verbose=1)
print(X_test.shape)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2694902447.py in <cell line: 0>()
     18 
     19 model_f.fit(
---> 20     train_ds,
     21     epochs=2,
     22     verbose=1,

NameError: name 'train_ds' is not defined

## === cell 8
print(
    "Pred stats: min/mean/max =",
    float(X_test.min()),
    float(X_test.mean()),
    float(X_test.max()),
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2476998881.py in <cell line: 0>()
      1 print(
      2     "Pred stats: min/mean/max =",
----> 3     float(X_test.min()),
      4     float(X_test.mean()),
      5     float(X_test.max()),

NameError: name 'X_test' is not defined

## === cell 9
def class2tags(classes, tagnames):
    tagnames = np.asarray(tagnames)
    idxs = [np.flatnonzero(r) for r in classes]
    return [" ".join(tagnames[i]) for i in idxs]


test_predclass = X_test > 0.2
test_predtags = class2tags(test_predclass, tagnames)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3105303851.py in <cell line: 0>()
      5 
      6 
----> 7 test_predclass = X_test > 0.2
      8 test_predtags = class2tags(test_predclass, tagnames)
      9 

NameError: name 'X_test' is not defined

## === cell 10
del test_predclass
gc.collect()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4078782309.py in <cell line: 0>()
----> 1 del test_predclass
      2 gc.collect()
      3 
      4 

NameError: name 'test_predclass' is not defined

## === cell 11
sub = pd.read_csv(SAMPLE_SUB)
pred_map = {os.path.basename(p): tag for p, tag in zip(imglist_test, test_predtags)}

sub["labels"] = sub["image"].map(pred_map).fillna("")
sub.head()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1037895330.py in <cell line: 0>()
      1 sub = pd.read_csv(SAMPLE_SUB)
----> 2 pred_map = {os.path.basename(p): tag for p, tag in zip(imglist_test, test_predtags)}
      3 
      4 sub["labels"] = sub["image"].map(pred_map).fillna("")
      5 sub.head()

NameError: name 'test_predtags' is not defined

## === cell 12
sub = sub[["image", "labels"]]
sub.to_csv("./submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
