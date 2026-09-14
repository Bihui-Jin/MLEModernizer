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

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from concurrent.futures import ProcessPoolExecutor  # use processes for true parallelism

BASE_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
OUTPUT_DIR = "./"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

IMG_SIZE = (64, 64)  # keep small for fast training
TRAIN_SAMPLE_SIZE = None  # use all training images
THRESHOLD = 0.5  # initial default (will be tuned)
RANDOM_STATE = 42



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
train_df["label_list"] = train_df["labels"].apply(lambda x: x.split(" "))

mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(train_df["label_list"])
class_names = mlb.classes_.tolist()


def load_image(path):
    """Read and resize a single image, returning a float32 array (no normalization)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, Image.BILINEAR)
        return np.asarray(img, dtype=np.float32).ravel()


def load_images(paths):
    """Load a list of image paths in parallel using processes and pre‑allocated array."""
    n = len(paths)
    images = np.empty((n, IMG_SIZE[0] * IMG_SIZE[1] * 3), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i, img_vec in enumerate(executor.map(load_image, paths)):
            images[i] = img_vec
    images /= 255.0
    return images




## === cell 2
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)  # ensure sklearn reproducibility seeds

if TRAIN_SAMPLE_SIZE is None or TRAIN_SAMPLE_SIZE >= len(train_df):
    sampled_indices = list(range(len(train_df)))
else:
    sampled_indices = random.sample(
        range(len(train_df)), min(TRAIN_SAMPLE_SIZE, len(train_df))
    )

sampled_paths = [
    os.path.join(TRAIN_IMG_DIR, train_df.iloc[i]["image"]) for i in sampled_indices
]
sampled_labels = train_labels[sampled_indices]

X_all = load_images(sampled_paths)
y_all = sampled_labels



## === cell 3
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.10, random_state=RANDOM_STATE
)

base_clf = LogisticRegression(
    solver="lbfgs", max_iter=300, n_jobs=-1, random_state=RANDOM_STATE
)
clf = OneVsRestClassifier(base_clf)

clf.fit(X_train, y_train)

val_probs = clf.predict_proba(X_val)
best_thr = THRESHOLD
best_f1 = -1.0
for thr in np.arange(0.2, 0.51, 0.05):
    pred_binary = (val_probs >= thr).astype(int)
    no_label_mask = pred_binary.sum(axis=1) == 0
    if np.any(no_label_mask):
        argmax_idxs = np.argmax(val_probs[no_label_mask], axis=1)
        pred_binary[no_label_mask, argmax_idxs] = 1
    f1 = f1_score(y_val, pred_binary, average="macro")
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

clf.fit(X_all, y_all)



## === cell 4
test_filenames = [
    f
    for f in sorted(os.listdir(TEST_IMG_DIR))
    if os.path.isfile(os.path.join(TEST_IMG_DIR, f))
]
test_paths = [os.path.join(TEST_IMG_DIR, fname) for fname in test_filenames]

X_test = load_images(test_paths)

probs = clf.predict_proba(X_test)

pred_labels = []
for prob_vec in probs:
    idxs = np.where(prob_vec >= best_thr)[0]
    if len(idxs) == 0:
        idxs = [int(np.argmax(prob_vec))]
    labels = " ".join([class_names[i] for i in idxs])
    pred_labels.append(labels)

submission_df = pd.DataFrame({"image": test_filenames, "labels": pred_labels})
submission_df.to_csv(OUTPUT_PATH, index=False)
