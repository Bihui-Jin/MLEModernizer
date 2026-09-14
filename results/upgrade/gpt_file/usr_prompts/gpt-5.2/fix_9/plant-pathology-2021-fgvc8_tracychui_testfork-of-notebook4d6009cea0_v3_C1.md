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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import glob
import gc

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

print("tf:", tf.__version__, "keras:", keras.__version__)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3

BATCH_SIZE = 128

tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
tf.config.threading.set_inter_op_parallelism_threads(2)

AUTOTUNE = tf.data.AUTOTUNE



## === cell 2
DATA_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

imglist_test = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
print("n_test_images:", len(imglist_test))
imglist_train = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg")))
print("n_train_images:", len(imglist_train))



## === cell 3
import tensorflow.keras.applications.resnet50 as resnet


def make_image_dataset(paths, batch_size=BATCH_SIZE, cache_path=None):
    paths = tf.convert_to_tensor(paths, dtype=tf.string)

    def _load(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.resize(
            img,
            size=[IMG_HEIGHT, IMG_WIDTH],
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False,
        )
        img = tf.cast(img, tf.float32)
        img = resnet.preprocess_input(img)
        return img

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_slack = True
    try:
        options.autotune.enabled = True
    except Exception:
        pass

    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)

    if cache_path is not None:
        pass

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 4
base = resnet.ResNet50(
    include_top=False, weights="imagenet", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
inp = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base(inp, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
model_f = keras.Model(inp, x)

print("Feature dim:", model_f.output_shape)



## === cell 5
training_csv = pd.read_csv(TRAIN_CSV)

all_tokens = training_csv["labels"].astype(str).str.split()
tagnames = np.unique(np.concatenate(all_tokens.values))
print("n_classes:", len(tagnames))
print("classes:", tagnames)

train_df = training_csv.copy()
train_df["image_path"] = train_df["image"].map(lambda x: os.path.join(TRAIN_IMG_DIR, x))
train_paths = train_df["image_path"].tolist()

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(classes=list(tagnames))
Y_train = mlb.fit_transform(all_tokens.values).astype(np.int8, copy=False)
print("Y_train shape:", Y_train.shape)



## === cell 6
ds_train = make_image_dataset(
    train_paths,
    batch_size=BATCH_SIZE,
    cache_path="/kaggle/working/cache_train_resnet50.tfdata",
)
Xf_train = model_f.predict(ds_train, verbose=1)
print("Xf_train:", Xf_train.shape)

del ds_train
gc.collect()



## === cell 7
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["image_path"] = sample_sub["image"].map(
    lambda x: os.path.join(TEST_IMG_DIR, x)
)

test_paths = sample_sub["image_path"].tolist()

ds_test = make_image_dataset(
    test_paths,
    batch_size=BATCH_SIZE,
    cache_path="/kaggle/working/cache_test_resnet50.tfdata",
)
Xf_test = model_f.predict(ds_test, verbose=1)
print("Xf_test:", Xf_test.shape)

del ds_test
gc.collect()



## === cell 8
scaler = MinMaxScaler(feature_range=(0, 1))
trainXn = scaler.fit_transform(Xf_train)
testXn_test = scaler.transform(Xf_test)

print("Scaled shapes:", trainXn.shape, testXn_test.shape)

del Xf_train, Xf_test
gc.collect()



## === cell 9
from sklearn.multiclass import OneVsRestClassifier

base_lr = LogisticRegression(
    solver="saga",
    penalty="l2",
    max_iter=1000,
    random_state=SEED,
    class_weight="balanced",
    n_jobs=1,  # avoid nested parallelism; OVR handles parallelism via OneVsRestClassifier(n_jobs=...)
)

n_jobs = max(1, (os.cpu_count() or 2) - 1)
clf = OneVsRestClassifier(base_lr, n_jobs=n_jobs)
clf.fit(trainXn, Y_train)

del Y_train
gc.collect()



## === cell 10
testKaggle_ppredscore1 = clf.predict_proba(testXn_test).astype(np.float32, copy=False)
print("Pred score matrix:", testKaggle_ppredscore1.shape)



## === cell 11
tagnames = np.asarray(tagnames)

test_predclass = testKaggle_ppredscore1 > 0.5
idx_lists = [np.flatnonzero(r) for r in test_predclass]
test_predtags = [" ".join(tagnames[idx]) if idx.size else "" for idx in idx_lists]

empty_mask = np.fromiter(
    (t == "" for t in test_predtags), count=len(test_predtags), dtype=bool
)
if empty_mask.any():
    argm = np.argmax(testKaggle_ppredscore1[empty_mask], axis=1)
    fill = tagnames[argm]
    it = iter(fill.tolist())
    for i, is_empty in enumerate(empty_mask.tolist()):
        if is_empty:
            test_predtags[i] = next(it)

print("Example preds:", test_predtags[:5])

submission = pd.DataFrame(
    {"image": sample_sub["image"].values, "labels": test_predtags}
)
print(submission.head())
print(submission.shape)

out_path = "./submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "size:", os.path.getsize(out_path), "bytes")
