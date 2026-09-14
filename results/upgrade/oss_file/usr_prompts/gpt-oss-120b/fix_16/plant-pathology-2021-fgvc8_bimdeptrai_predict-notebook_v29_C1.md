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
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from PIL import Image
import concurrent.futures  # parallel image loading (threads)
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
random.seed(42)
np.random.seed(42)

BASE_PATH = "../input/plant-pathology-2021-fgvc8"
if not os.path.isdir(BASE_PATH):
    BASE_PATH = "./input/plant-pathology-2021-fgvc8"

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_SUBMIT_CSV = os.path.join(BASE_PATH, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")




## === cell 1
train_df = pd.read_csv(TRAIN_CSV)

label_lists = train_df["labels"].apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(label_lists)




## === cell 2
IMG_SIZE = (64, 64)


def load_and_preprocess(img_path):
    """
    Load an image, resize to 64×64, flatten and scale to [0,1].
    Returns a 1‑D float32 array of length IMG_SIZE[0]*IMG_SIZE[1]*3.
    """
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
    return arr


train_cache_path = Path("train_features.npy")
train_paths = [os.path.join(TRAIN_IMG_DIR, img_name) for img_name in train_df["image"]]
num_train = len(train_paths)
feature_len = IMG_SIZE[0] * IMG_SIZE[1] * 3

if train_cache_path.is_file():
    X = np.load(train_cache_path, allow_pickle=False)
else:
    X = np.empty((num_train, feature_len), dtype=np.float32)
    max_workers = min(8, os.cpu_count() or 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, arr in enumerate(
            executor.map(load_and_preprocess, train_paths, chunksize=128)
        ):
            X[idx] = arr
    np.save(train_cache_path, X, allow_pickle=False)




## === cell 3
clf = OneVsRestClassifier(
    LogisticRegression(
        solver="lbfgs", max_iter=1000, n_jobs=-1, class_weight="balanced"
    )
)
clf.fit(X, Y)




## === cell 4
submission_df = pd.read_csv(TEST_SUBMIT_CSV)

test_paths = [
    os.path.join(TEST_IMG_DIR, img_name) for img_name in submission_df["image"]
]

num_test = len(test_paths)
test_cache_path = Path("test_features.npy")

if test_cache_path.is_file():
    test_X = np.load(test_cache_path, allow_pickle=False)
else:
    test_X = np.empty((num_test, feature_len), dtype=np.float32)
    max_workers = min(8, os.cpu_count() or 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, arr in enumerate(
            executor.map(load_and_preprocess, test_paths, chunksize=128)
        ):
            test_X[idx] = arr
    np.save(test_cache_path, test_X, allow_pickle=False)

probas = clf.predict_proba(test_X)

threshold = 0.2  # lower threshold to capture more relevant labels
pred_label_strings = []
for prob_vec in probas:
    idx = np.where(prob_vec >= threshold)[0]
    if len(idx) == 0:
        idx = [np.argmax(prob_vec)]
    pred_label_strings.append(" ".join(mlb.classes_[i] for i in idx))

submission_df["labels"] = pred_label_strings




## === cell 5
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
