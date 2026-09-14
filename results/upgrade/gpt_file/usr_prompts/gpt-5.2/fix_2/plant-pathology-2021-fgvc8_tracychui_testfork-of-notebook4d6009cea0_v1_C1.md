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

K.set_image_data_format("channels_last")
print("keras", keras.__version__, "tf", tf.__version__)



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
training_class = np.array([], dtype=object)
for labels in pd.unique(train_df["labels"]):
    training_class = np.append(training_class, str(labels).split())
tagnames = np.unique(training_class)
tagnames = np.array(sorted(tagnames))
num_classes = len(tagnames)
print("num_classes:", num_classes)
print("classes:", tagnames)

tag2idx = {t: i for i, t in enumerate(tagnames)}

Y = np.zeros((len(train_df), num_classes), dtype=np.int8)
for i, lab in enumerate(train_df["labels"].astype(str).values):
    for t in lab.split():
        Y[i, tag2idx[t]] = 1

print("Y shape:", Y.shape, "positive rate:", Y.mean())



## === cell 4
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50


def load_and_preprocess(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = resnet50.preprocess_input(x)
    return x




## === cell 5
base = resnet50.ResNet50(
    include_top=False,
    weights="imagenet",
    pooling="avg",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
)


def extract_features(image_paths, batch_size=32):
    feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        xb = np.zeros(
            (len(batch_paths), IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS), dtype=np.float32
        )
        for j, p in enumerate(batch_paths):
            xb[j] = load_and_preprocess(p)[0]
        fb = base.predict(xb, verbose=0)
        feats.append(fb.astype(np.float32))
        del xb, fb
        gc.collect()
    return np.vstack(feats)


train_paths = [os.path.join(TRAIN_IMG_DIR, fn) for fn in train_df["image"].values]
missing_train = [p for p in train_paths if not os.path.exists(p)]
assert (
    len(missing_train) == 0
), f"Missing {len(missing_train)} train images, e.g. {missing_train[:3]}"

Xf = extract_features(train_paths, batch_size=32)
print("Train features:", Xf.shape)



## === cell 6
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
Xn = scaler.fit_transform(Xf)
del Xf
gc.collect()
print("Scaled train:", Xn.shape)



## === cell 7
X_tr, X_va, y_tr, y_va = train_test_split(
    Xn,
    Y,
    test_size=0.2,
    random_state=SEED,
    stratify=Y.sum(axis=1),  # approximate stratification
)

clf = OneVsRestClassifier(
    LogisticRegression(solver="liblinear", max_iter=1000, C=1.0, random_state=SEED),
    n_jobs=None,
)

clf.fit(X_tr, y_tr)

va_prob = clf.predict_proba(X_va)
print("Val prob:", va_prob.shape)



## === cell 8
thr_grid = np.linspace(0.05, 0.95, 19)

best_thr = np.full(num_classes, 0.5, dtype=np.float32)
for k in range(num_classes):
    yk = y_va[:, k]
    pk = va_prob[:, k]
    if yk.sum() == 0:
        best_thr[k] = 0.5
        continue
    best_f1 = -1.0
    best_t = 0.5
    for t in thr_grid:
        predk = (pk >= t).astype(np.int8)
        f1 = f1_score(yk, predk, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    best_thr[k] = best_t

va_pred = (va_prob >= best_thr[None, :]).astype(np.int8)
mean_f1 = np.mean(
    [f1_score(y_va[:, k], va_pred[:, k], zero_division=0) for k in range(num_classes)]
)
print("Validation mean per-class F1 (proxy):", mean_f1)
print("Thresholds (first 10):", best_thr[:10])




## === cell 9
def class2tags(classes_bin, tagnames_arr):
    tags = []
    for n in range(classes_bin.shape[0]):
        tmp = []
        for i in range(classes_bin.shape[1]):
            if classes_bin[n, i]:
                tmp.append(tagnames_arr[i])
        tags.append(" ".join(tmp))
    return tags




## === cell 10
test_images = sub_df["image"].astype(str).values
test_paths = [os.path.join(TEST_IMG_DIR, fn) for fn in test_images]
missing_test = [p for p in test_paths if not os.path.exists(p)]
assert (
    len(missing_test) == 0
), f"Missing {len(missing_test)} test images, e.g. {missing_test[:3]}"

Xf_test = extract_features(test_paths, batch_size=32)
Xn_test = scaler.transform(Xf_test)
del Xf_test
gc.collect()

test_prob = clf.predict_proba(Xn_test)
test_bin = (test_prob >= best_thr[None, :]).astype(np.int8)
test_predtags = class2tags(test_bin, tagnames)

print("Pred tags example:", test_predtags[:5])



## === cell 11
submission = pd.DataFrame({"image": test_images, "labels": test_predtags})
submission.to_csv("./submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
submission.head()
