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
import glob
import gc
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

K.set_image_data_format("channels_last")
print("keras", keras.__version__, "tf", tf.__version__)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3




## === cell 2
DATA_DIR = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

print(train_df.shape, sub_df.shape)
train_df.head()




## === cell 3
training_class = []
for labels in pd.unique(train_df["labels"]):
    training_class.extend(str(labels).split())
tagnames = np.array(sorted(np.unique(training_class)))
num_classes = len(tagnames)
print("num_classes:", num_classes)
print("classes:", tagnames)

tag2idx = {t: i for i, t in enumerate(tagnames)}

Y = np.zeros((len(train_df), num_classes), dtype=np.int8)
labels_list = train_df["labels"].astype(str).tolist()
for i, lab in enumerate(labels_list):
    for t in lab.split():
        Y[i, tag2idx[t]] = 1

print("Y shape:", Y.shape, "positive rate:", Y.mean())




## === cell 4
from tensorflow.keras.applications import resnet50

base = resnet50.ResNet50(
    include_top=False,
    weights="imagenet",
    pooling="avg",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
)
base.trainable = False

AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, (IMG_HEIGHT, IMG_WIDTH), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = resnet50.preprocess_input(img)
    return img


def extract_features(image_paths, batch_size=64, cache_path=None):
    options = tf.data.Options()
    options.experimental_deterministic = (
        False  # higher throughput; model is deterministic given inputs
    )
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass

    paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)
    ds = ds.map(_decode_resize_preprocess, num_parallel_calls=AUTOTUNE)

    if cache_path is not None:
        ds = ds.cache(cache_path)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    feats = base.predict(ds, verbose=0)
    return feats.astype(np.float32, copy=False)


train_paths = [os.path.join(TRAIN_IMG_DIR, fn) for fn in train_df["image"].values]

assert len(train_paths) == len(train_df), "Unexpected train path construction mismatch"
assert len(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg"))) > 0, "No train images found"

Xf = extract_features(
    train_paths,
    batch_size=192,
    cache_path="/kaggle/working/cache_train_resnet50_300.tfdata",
)
print("Train features:", Xf.shape)




## === cell 5
Xf = Xf.astype(np.float32, copy=False)
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
Xn = scaler.fit_transform(Xf).astype(np.float32, copy=False)
del Xf
gc.collect()
print("Scaled train:", Xn.shape)




## === cell 6
X_tr, X_va, y_tr, y_va = train_test_split(
    Xn,
    Y,
    test_size=0.2,
    random_state=SEED,
    stratify=Y.sum(axis=1),  # approximate stratification
)

n_jobs = os.cpu_count() or 1
clf = OneVsRestClassifier(
    LogisticRegression(solver="liblinear", max_iter=1000, C=1.0, random_state=SEED),
    n_jobs=n_jobs,
)

clf.fit(X_tr, y_tr)

va_prob = clf.predict_proba(X_va)
print("Val prob:", va_prob.shape)




## === cell 7
thr_grid = np.linspace(0.05, 0.95, 19).astype(np.float32)

best_thr = np.full(num_classes, 0.5, dtype=np.float32)
for k in range(num_classes):
    yk = y_va[:, k].astype(np.int8, copy=False)
    pk = va_prob[:, k].astype(np.float32, copy=False)
    if yk.sum() == 0:
        best_thr[k] = 0.5
        continue

    preds = pk[None, :] >= thr_grid[:, None]
    y = yk[None, :].astype(bool, copy=False)

    tp = np.sum(preds & y, axis=1).astype(np.float32)
    fp = np.sum(preds & (~y), axis=1).astype(np.float32)
    fn = np.sum((~preds) & y, axis=1).astype(np.float32)

    denom = 2 * tp + fp + fn
    f1s = np.where(denom > 0, (2 * tp) / denom, 0.0)

    best_thr[k] = float(thr_grid[int(np.argmax(f1s))])

va_pred = (va_prob >= best_thr[None, :]).astype(np.int8)
mean_f1 = np.mean(
    [f1_score(y_va[:, k], va_pred[:, k], zero_division=0) for k in range(num_classes)]
)
print("Validation mean per-class F1 (proxy):", mean_f1)
print("Thresholds (first 10):", best_thr[:10])




## === cell 8
def class2tags(classes_bin, tagnames_arr):
    classes_bin = classes_bin.astype(bool, copy=False)
    idxs = [np.flatnonzero(row) for row in classes_bin]
    return [" ".join(tagnames_arr[i]) if len(i) else "" for i in idxs]




## === cell 9
test_images = sub_df["image"].astype(str).values
test_paths = [os.path.join(TEST_IMG_DIR, fn) for fn in test_images]

assert len(test_paths) == len(sub_df), "Unexpected test path construction mismatch"
assert len(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))) > 0, "No test images found"

Xf_test = extract_features(
    test_paths,
    batch_size=192,
    cache_path="/kaggle/working/cache_test_resnet50_300.tfdata",
)
Xf_test = Xf_test.astype(np.float32, copy=False)
Xn_test = scaler.transform(Xf_test).astype(np.float32, copy=False)
del Xf_test
gc.collect()

test_prob = clf.predict_proba(Xn_test)
test_bin = (test_prob >= best_thr[None, :]).astype(np.int8)
test_predtags = class2tags(test_bin, tagnames)

print("Pred tags example:", test_predtags[:5])




## === cell 10
submission = pd.DataFrame({"image": test_images, "labels": test_predtags})
submission.to_csv("./submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
submission.head()
