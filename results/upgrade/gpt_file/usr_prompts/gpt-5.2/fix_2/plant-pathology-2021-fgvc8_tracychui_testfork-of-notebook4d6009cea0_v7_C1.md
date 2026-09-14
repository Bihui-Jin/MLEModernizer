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
import pickle
import random

import numpy as np
import pandas as pd

from PIL import Image as PILImage

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

K.set_image_data_format("channels_last")
print("keras:", keras.__version__, "tf:", tf.__version__)



## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3

DATA_ROOT = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"



## === cell 2
sub_df = pd.read_csv(SAMPLE_SUB)
test_images = sub_df["image"].astype(str).tolist()
print("sample_submission rows:", len(test_images), "example:", test_images[:3])

test_paths = [os.path.join(TEST_IMG_DIR, img) for img in test_images]
print("Test image dir exists:", os.path.isdir(TEST_IMG_DIR))
print("First path:", test_paths[0])



## === cell 3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

base = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
    pooling="avg",
)
model_f = Model(inputs=base.input, outputs=base.output)
model_f.trainable = False

print("Feature dim:", model_f.output_shape)



## === cell 4
from tensorflow.keras.preprocessing import image


def extract_features(image_paths, batch_size=32):
    n = len(image_paths)
    feats = None
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bpaths = image_paths[start:end]
        X = np.zeros(
            (len(bpaths), IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS), dtype=np.float32
        )
        for i, p in enumerate(bpaths):
            img = image.load_img(p, target_size=(IMG_HEIGHT, IMG_WIDTH))
            x = image.img_to_array(img)
            X[i] = x
        X = preprocess_input(X)
        f = model_f.predict(X, verbose=0)
        if feats is None:
            feats = np.zeros((n, f.shape[1]), dtype=np.float32)
        feats[start:end] = f.astype(np.float32)
        if (start // batch_size) % 20 == 0:
            gc.collect()
    return feats




## === cell 5
train_df = pd.read_csv(TRAIN_CSV)
train_df["image"] = train_df["image"].astype(str)
train_df["labels"] = train_df["labels"].astype(str)

training_class = np.array([])
for labels in pd.unique(train_df["labels"]):
    training_class = np.append(training_class, labels.split())
tagnames = np.unique(training_class)

print("Num train:", len(train_df), "Num tags:", len(tagnames))
print("Tags:", tagnames)



## === cell 6
train_paths = [os.path.join(TRAIN_IMG_DIR, img) for img in train_df["image"].tolist()]
for p in train_paths[:5]:
    assert os.path.exists(p), f"Missing train image: {p}"

Xf_train = extract_features(train_paths, batch_size=32)
print("Xf_train:", Xf_train.shape)



## === cell 7
missing = [p for p in test_paths[:10] if not os.path.exists(p)]
if len(missing) > 0:
    print(
        "Warning: some test files not found in this environment subset (expected in notebook preview). Example:",
        missing[0],
    )

avail_mask = [os.path.exists(p) for p in test_paths]
avail_idx = np.where(avail_mask)[0]
print("Available test images in this runtime:", len(avail_idx), "of", len(test_paths))

Xf_test = extract_features([test_paths[i] for i in avail_idx], batch_size=32)
print("Xf_test:", Xf_test.shape)



## === cell 8
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
trainXn = scaler.fit_transform(Xf_train)
testXn_avail = scaler.transform(Xf_test)

print("Scaled train/test:", trainXn.shape, testXn_avail.shape)



## === cell 9
label_sets = [set(s.split()) for s in train_df["labels"].tolist()]
Y = np.zeros((len(train_df), len(tagnames)), dtype=np.int32)
tag_to_i = {t: i for i, t in enumerate(tagnames)}
for r, s in enumerate(label_sets):
    for t in s:
        if t in tag_to_i:
            Y[r, tag_to_i[t]] = 1

tagmodels1 = {}
for i, t in enumerate(tagnames):
    y = Y[:, i]
    if y.min() == y.max():
        tagmodels1[t] = None
        print(f"Tag {t}: skipped (only one class present in y)")
        continue

    lr = LogisticRegression(
        solver="liblinear",
        max_iter=200,
        random_state=SEED,
    )
    lr.fit(trainXn, y)
    tagmodels1[t] = lr

print(
    "Trained models:",
    sum(m is not None for m in tagmodels1.values()),
    "of",
    len(tagnames),
)



## === cell 10
testKaggle_ppredscore1_avail = np.zeros(
    (testXn_avail.shape[0], len(tagnames)), dtype=np.float32
)
for i, t in enumerate(tagnames):
    mdl = tagmodels1[t]
    if mdl is None:
        continue
    testKaggle_ppredscore1_avail[:, i] = mdl.predict_proba(testXn_avail)[:, 1].astype(
        np.float32
    )

print("Pred score shape:", testKaggle_ppredscore1_avail.shape)




## === cell 11
def class2tags(classes, tagnames):
    tags = []
    for n in range(classes.shape[0]):
        tmp = []
        for i in range(classes.shape[1]):
            if classes[n, i]:
                tmp.append(tagnames[i])
        tags.append(" ".join(tmp))
    return tags




## === cell 12
TH = 0.90
test_predclass_avail = testKaggle_ppredscore1_avail > TH
test_predtags_avail = class2tags(test_predclass_avail, tagnames)

test_predtags_avail = [
    s if len(s.strip()) > 0 else "healthy" for s in test_predtags_avail
]

print("Example preds:", test_predtags_avail[:10])



## === cell 13
full_predtags = ["healthy"] * len(test_images)
for out_pos, idx in enumerate(avail_idx):
    full_predtags[idx] = test_predtags_avail[out_pos]

submission = pd.DataFrame({"image": test_images, "labels": full_predtags})
print(submission.head())
print("Submission shape:", submission.shape)



## === cell 14
out_path = "./submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "size:", os.path.getsize(out_path), "bytes")
