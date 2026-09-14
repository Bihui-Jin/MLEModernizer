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

from PIL import Image as PILImage

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import tensorflow.keras.applications.resnet50 as resnet
from keras.preprocessing import image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

K.set_image_data_format("channels_last")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("keras:", keras.__version__, "tf:", tf.__version__)



## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3



## === cell 2
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"
TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TRAIN_CSV_PATH = "../input/plant-pathology-2021-fgvc8/train.csv"
SAMPLE_SUB_PATH = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"

imglist_test = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
len(imglist_test)



## === cell 3
base = resnet.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
    pooling="avg",
)
base.trainable = False


def load_and_preprocess_batch(paths, batch_size=32):
    n = len(paths)
    for i in range(0, n, batch_size):
        batch_paths = paths[i : i + batch_size]
        X = np.zeros(
            (len(batch_paths), IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS), dtype=np.float32
        )
        for j, p in enumerate(batch_paths):
            img = image.load_img(p, target_size=(IMG_HEIGHT, IMG_WIDTH))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = resnet.preprocess_input(x)
            X[j] = x[0]
        yield X, batch_paths


def extract_features(paths, batch_size=32, verbose=1):
    feats = []
    for X, _ in load_and_preprocess_batch(paths, batch_size=batch_size):
        f = base.predict(X, verbose=0)
        feats.append(f)
    feats = np.vstack(feats) if len(feats) else np.zeros((0, 2048), dtype=np.float32)
    if verbose:
        print("Extracted features:", feats.shape)
    return feats




## === cell 4
training_csv = pd.read_csv(TRAIN_CSV_PATH)

all_tags = set()
for lbl in training_csv["labels"].astype(str).values:
    for t in lbl.split():
        all_tags.add(t)
tagnames = np.array(sorted(list(all_tags)))
print("Num classes:", len(tagnames))
print("Classes:", tagnames)

tag2idx = {t: i for i, t in enumerate(tagnames)}

Y = np.zeros((len(training_csv), len(tagnames)), dtype=np.int8)
for r, lbl in enumerate(training_csv["labels"].astype(str).values):
    for t in lbl.split():
        Y[r, tag2idx[t]] = 1

train_paths = [os.path.join(TRAIN_DIR, fn) for fn in training_csv["image"].values]

missing = sum([0 if os.path.exists(p) else 1 for p in train_paths])
print("Missing train images:", missing)



## === cell 5
idx = np.arange(len(train_paths))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=SEED, shuffle=True)

train_paths_tr = [train_paths[i] for i in tr_idx]
train_paths_va = [train_paths[i] for i in va_idx]
Y_tr = Y[tr_idx]
Y_va = Y[va_idx]

print("Train:", len(train_paths_tr), "Val:", len(train_paths_va))



## === cell 6
Xf_tr = extract_features(train_paths_tr, batch_size=32, verbose=1)
Xf_va = extract_features(train_paths_va, batch_size=32, verbose=1)

scaler = MinMaxScaler(feature_range=(0, 1))
Xn_tr = scaler.fit_transform(Xf_tr)
Xn_va = scaler.transform(Xf_va)

del Xf_tr, Xf_va
gc.collect()



## === cell 7
tagmodels1 = {}
val_pred_proba = np.zeros((Xn_va.shape[0], len(tagnames)), dtype=np.float32)

for i, t in enumerate(tagnames):
    y = Y_tr[:, i]
    if np.unique(y).size < 2:
        tagmodels1[t] = None
        prior = float(y.mean())
        val_pred_proba[:, i] = prior
        continue

    clf = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=SEED,
    )
    clf.fit(Xn_tr, y)
    tagmodels1[t] = clf
    val_pred_proba[:, i] = clf.predict_proba(Xn_va)[:, 1]

print(
    "Trained models:",
    sum([m is not None for m in tagmodels1.values()]),
    "/",
    len(tagnames),
)



## === cell 8
thr_grid = np.linspace(0.2, 0.9, 15)
best_thr = np.full(len(tagnames), 0.5, dtype=np.float32)

for i in range(len(tagnames)):
    y_true = Y_va[:, i]
    if y_true.sum() == 0:
        best_thr[i] = 0.5
        continue

    p = val_pred_proba[:, i]
    best_f1 = -1.0
    best_t = 0.5
    for t in thr_grid:
        y_pred = (p >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    best_thr[i] = best_t

val_pred_bin = (val_pred_proba >= best_thr.reshape(1, -1)).astype(int)
mean_f1 = np.mean(
    [
        f1_score(Y_va[:, i], val_pred_bin[:, i], zero_division=0)
        for i in range(len(tagnames))
    ]
)
print("Validation mean per-class F1 (approx proxy):", mean_f1)

del Xn_tr, Xn_va, val_pred_proba, val_pred_bin
gc.collect()



## === cell 9
Xf_test = extract_features(imglist_test, batch_size=32, verbose=1)
Xn_test = scaler.transform(Xf_test)

testKaggle_ppredscore1 = np.zeros((Xn_test.shape[0], len(tagnames)), dtype=np.float32)
for i, t in enumerate(tagnames):
    clf = tagmodels1[t]
    if clf is None:
        prior = float(Y_tr[:, i].mean())
        testKaggle_ppredscore1[:, i] = prior
    else:
        testKaggle_ppredscore1[:, i] = clf.predict_proba(Xn_test)[:, 1]

print("Test proba matrix:", testKaggle_ppredscore1.shape)

del Xf_test, Xn_test
gc.collect()




## === cell 10
def class2tags(classes, tagnames):
    tags = []
    for n in range(classes.shape[0]):
        tmp = []
        for i in range(classes.shape[1]):
            if classes[n, i]:
                tmp.append(tagnames[i])
        tags.append(" ".join(tmp))
    return tags




## === cell 11
test_predclass = testKaggle_ppredscore1 >= best_thr.reshape(1, -1)
test_predtags = class2tags(test_predclass, tagnames)

test_predtags = [t if len(t) else "healthy" for t in test_predtags]

del test_predclass, testKaggle_ppredscore1
gc.collect()



## === cell 12
sub = pd.read_csv(SAMPLE_SUB_PATH)

test_images = [os.path.basename(p) for p in imglist_test]
pred_map = dict(zip(test_images, test_predtags))

sub["labels"] = sub["image"].map(lambda x: pred_map.get(x, "healthy"))

sub["image"] = sub["image"].astype(str)
sub["labels"] = sub["labels"].astype(str)

sub.head()



## === cell 13
out_path = "./submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", sub.shape)
print(sub.head())
