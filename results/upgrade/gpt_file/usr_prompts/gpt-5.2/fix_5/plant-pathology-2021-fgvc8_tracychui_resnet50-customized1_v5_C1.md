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

0.4458910433979671

# 6. Current score

0.35001

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.3041) has done: 'I fix the immediate runtime failure caused by `from sklearn import *` (it triggers a protobuf-related crash in this environment) by removing that import since it isn’t used. Then I fix the missing model file issue by keeping the same ResNet50-based core idea but building the feature extractor directly from `tf.keras.applications.ResNet50` with ImageNet weights, so inference runs end-to-end without external model artifacts. Finally, I correct path handling, ensure test image order aligns exactly with `sample_submission.csv`, and fix the submission creation code (avoid `pd.df` and the broken `write_path()`), so a valid `submission.csv` is always written with the required `image,labels` columns.'
- What this solution (achieved 0.3065) has done: 'The timeout is dominated by slow, single-threaded image loading/preprocessing into a huge in-memory array, plus unnecessary Python overhead (expand_dims loop, repeated allocations) before a relatively fast single forward pass. I keep the exact same model and prediction logic, but replace the manual loop with a `tf.data` pipeline that loads/decodes/resizes/preprocesses images in parallel and streams batches directly into `model.predict`, which is equivalent but far faster. I also vectorize the tag conversion (remove nested Python loops) to cut post-processing time without changing thresholds/semantics. Finally, I keep paths and outputs identical, and add deterministic settings to avoid run-to-run drift.'
- What this solution (achieved 0.35001) has done: 'I fix the immediate runtime crash happening before any training/inference by removing the protobuf-forcing environment variable that triggers the `MessageFactory.GetPrototype` error in this Kaggle TF environment. Then I keep your exact model and tf.data inference pipeline intact, but add a minimal, score-improving step: compute per-class thresholds on a small validation split of the training set to better match mean F1 (instead of using a single hardcoded 0.26 threshold). Finally, I ensure submission formatting remains exactly `image,labels` (space-delimited labels) and that the file is always written as `submission.csv` aligned to `sample_submission.csv` order.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

K.set_image_data_format("channels_last")
print("keras:", keras.__version__, "tf:", tf.__version__)

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
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
TOTAL_INPUTS = NR_CHANNELS * IMG_HEIGHT * IMG_WIDTH



## === cell 2
BASE_INPUT = "/kaggle/input/plant-pathology-2021-fgvc8"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../input/plant-pathology-2021-fgvc8"

TRAIN_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_DIR = os.path.join(BASE_INPUT, "test_images")
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

assert os.path.exists(TEST_DIR), f"Missing TEST_DIR: {TEST_DIR}"
assert os.path.exists(TRAIN_DIR), f"Missing TRAIN_DIR: {TRAIN_DIR}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing TRAIN_CSV_PATH: {TRAIN_CSV_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing SAMPLE_SUB_PATH: {SAMPLE_SUB_PATH}"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_images = sample_sub["image"].tolist()

imglist_test = [os.path.join(TEST_DIR, fn) for fn in test_images]
missing = [p for p in imglist_test if not os.path.exists(p)]
print("test images:", len(imglist_test), "missing:", len(missing))
if len(missing) > 0:
    print("First missing:", missing[0])



## === cell 3
AUTOTUNE = tf.data.AUTOTUNE


def _load_and_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img


test_ds = (
    tf.data.Dataset.from_tensor_slices(tf.constant(imglist_test))
    .map(_load_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(32, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

_first_batch = next(iter(test_ds))
print((len(imglist_test), IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))



## === cell 4
print(_first_batch[0, :2, :2, :].numpy())



## === cell 5
training_csv = pd.read_csv(TRAIN_CSV_PATH)

training_class = np.array([])
for labels in pd.unique(training_csv["labels"]):
    training_class = np.append(training_class, str(labels).split())

tagnames = np.unique(training_class)
num_classes = len(tagnames)
print("num_classes:", num_classes)
print("tagnames:", tagnames)

base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
    pooling="avg",
)
base.trainable = False

inp = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))
x = base(inp, training=False)
out = tf.keras.layers.Dense(num_classes, activation="sigmoid", name="pred")(x)
model_f = tf.keras.Model(inp, out)




## === cell 6
def labels_to_multihot(labels_series, tagnames):
    tag_to_idx = {t: i for i, t in enumerate(tagnames)}
    y = np.zeros((len(labels_series), len(tagnames)), dtype=np.int8)
    for r, s in enumerate(labels_series.astype(str).tolist()):
        for tok in s.split():
            y[r, tag_to_idx[tok]] = 1
    return y


def build_ds_from_paths(paths, batch_size=32, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(tf.constant(paths))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 4096), seed=42, reshuffle_each_iteration=False
        )
    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def f1_per_class(y_true, y_pred_bool):
    tp = np.sum((y_true == 1) & (y_pred_bool == 1), axis=0).astype(np.float64)
    fp = np.sum((y_true == 0) & (y_pred_bool == 1), axis=0).astype(np.float64)
    fn = np.sum((y_true == 1) & (y_pred_bool == 0), axis=0).astype(np.float64)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1


val_frac = 0.10
idx = np.arange(len(training_csv))
rng = np.random.RandomState(42)
rng.shuffle(idx)
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]

val_df = training_csv.iloc[val_idx].reset_index(drop=True)
val_paths = [os.path.join(TRAIN_DIR, fn) for fn in val_df["image"].tolist()]
y_val = labels_to_multihot(val_df["labels"], tagnames)

val_ds = build_ds_from_paths(val_paths, batch_size=32, shuffle=False)
p_val = model_f.predict(val_ds, verbose=1)

thr_grid = np.linspace(0.05, 0.60, 12)
best_thr = np.full((num_classes,), 0.26, dtype=np.float32)
best_f1 = np.full((num_classes,), -1.0, dtype=np.float64)

for thr in thr_grid:
    pred_bool = p_val >= thr
    f1c = f1_per_class(y_val, pred_bool)
    better = f1c > best_f1
    best_f1[better] = f1c[better]
    best_thr[better] = thr

print(
    "Calibrated thresholds summary:",
    "min",
    float(best_thr.min()),
    "max",
    float(best_thr.max()),
    "mean",
    float(best_thr.mean()),
)

del val_ds, val_paths, val_df, y_val, p_val
gc.collect()



## === cell 7
X_test = model_f.predict(test_ds, verbose=1)
print(X_test.shape)



## === cell 8
print(X_test[:2])



## === cell 9
pass




## === cell 10
def class2tags(classes_bool, tagnames):
    idx = [np.flatnonzero(row) for row in classes_bool]
    return [" ".join(tagnames[i]) for i in idx]


def probs_to_tags_with_fallback_perclass(probs, tagnames, per_class_thresh):
    pred = probs >= per_class_thresh.reshape(1, -1)
    empty = pred.sum(axis=1) == 0
    if np.any(empty):
        top1 = np.argmax(probs[empty], axis=1)
        pred[empty, :] = False
        pred[np.where(empty)[0], top1] = True
    return class2tags(pred, tagnames)




## === cell 11
test_predtags = probs_to_tags_with_fallback_perclass(X_test, tagnames, best_thr)
print("Example tags:", test_predtags[:5])



## === cell 12
del _first_batch
gc.collect()




## === cell 13
def write_path_from_sample(sample_df):
    return sample_df["image"].values




## === cell 14
df1 = pd.DataFrame(write_path_from_sample(sample_sub), columns=["image"])
df1.head()



## === cell 15
df2 = pd.DataFrame(test_predtags, columns=["labels"])
df2.head()



## === cell 16
sub_df = pd.concat([df1, df2], axis=1)
sub_df = sub_df[["image", "labels"]]
assert sub_df.shape[0] == sample_sub.shape[0], "Submission row count mismatch."

sub_path = "./submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", sub_df.shape)
print(sub_df.columns.tolist())
print(sub_df.head())
