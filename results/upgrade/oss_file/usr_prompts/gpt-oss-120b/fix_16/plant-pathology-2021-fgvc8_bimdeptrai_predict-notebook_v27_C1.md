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
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from PIL import Image

from concurrent.futures import (
    ThreadPoolExecutor,
)  # use threads to bypass GIL where possible

_MAX_WORKERS = min(32, (os.cpu_count() or 1) * 2)
_GLOBAL_POOL = ThreadPoolExecutor(max_workers=_MAX_WORKERS)



## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")



## === cell 2
train_idx, val_idx = train_test_split(
    np.arange(len(train)), test_size=0.2, random_state=42, shuffle=True
)

label_lists = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer().fit(label_lists)
true_bin = mlb.transform(label_lists)  # (n_samples, n_classes)




## === cell 3
def extract_features(img_path):
    """Return a colour statistic vector (12 basic stats + 48‑bin colour histogram)."""
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((32, 32), resample=Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # normalise to [0,1]

        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        mn = arr.min(axis=(0, 1))
        mx = arr.max(axis=(0, 1))
        basic = np.concatenate([mean, std, mn, mx])

        hist_feats = []
        for ch in range(3):
            hist, _ = np.histogram(
                arr[:, :, ch], bins=16, range=(0.0, 1.0), density=False
            )
            hist = hist.astype(np.float32) / (32 * 32)  # normalise by number of pixels
            hist_feats.append(hist)
        hist_feats = np.concatenate(hist_feats)

        return np.concatenate([basic, hist_feats])
    except Exception:
        return np.zeros(12 + 48, dtype=np.float32)


train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images"
test_img_dir = "../input/plant-pathology-2021-fgvc8/test_images"




## === cell 4
def build_feature_matrix(idxs):
    img_names = train["image"].values[idxs]  # NumPy array, no pandas overhead
    paths = [os.path.join(train_img_dir, name) for name in img_names]
    n = len(paths)
    feats = np.empty((n, 12 + 48), dtype=np.float32)  # updated feature size
    for i, f in enumerate(_GLOBAL_POOL.map(extract_features, paths, chunksize=64)):
        feats[i] = f
    return feats


X_train = build_feature_matrix(train_idx)
X_val = build_feature_matrix(val_idx)
y_train = true_bin[train_idx]
y_val = true_bin[val_idx]

base_clf = RandomForestClassifier(
    n_estimators=800,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)
clf = OneVsRestClassifier(base_clf)
clf.fit(X_train, y_train)

val_prob = clf.predict_proba(X_val)  # (n_val, n_classes)


def optimise_thresholds(probs, true_labels):
    """
    Find per‑class probability thresholds that maximise the
    sample‑averaged F1 score, using a finer search grid.
    """
    n_classes = probs.shape[1]
    thresholds = np.full(n_classes, 0.5)  # start with 0.5 for all
    cand_thr = np.arange(0.01, 0.991, 0.01)  # finer candidate thresholds

    for c in range(n_classes):
        best_f1 = -1.0
        best_t = 0.5
        for t in cand_thr:
            tmp_thresh = thresholds.copy()
            tmp_thresh[c] = t  # only modify class c
            pred = (probs >= tmp_thresh).astype(int)
            f1 = f1_score(true_labels, pred, average="samples")
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[c] = best_t
    return thresholds


best_thresh = optimise_thresholds(val_prob, y_val)

val_pred = (val_prob >= best_thresh).astype(int)
val_score = f1_score(y_val, val_pred, average="samples")
print(f"Validation samples‑averaged F1 (tuned thresholds): {val_score:.5f}")



## === cell 5
X_full = np.vstack([X_train, X_val])
y_full = np.vstack([y_train, y_val])
clf_full = OneVsRestClassifier(base_clf)
clf_full.fit(X_full, y_full)



## === cell 6
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")




## === cell 7
def build_test_features(image_names):
    paths = [os.path.join(test_img_dir, name) for name in image_names]
    n = len(paths)
    feats = np.empty((n, 12 + 48), dtype=np.float32)
    for i, f in enumerate(_GLOBAL_POOL.map(extract_features, paths, chunksize=64)):
        feats[i] = f
    return feats


X_test = build_test_features(submissions["image"].values)

test_prob = clf_full.predict_proba(X_test)  # (n_test, n_classes)
test_pred = (test_prob >= best_thresh).astype(int)

pred_labels = []
for row in test_pred:
    lbls = [mlb.classes_[i] for i, v in enumerate(row) if v == 1]
    if not lbls:
        lbls = [mlb.classes_[0]]
    pred_labels.append(" ".join(lbls))

submissions["labels"] = pred_labels



## === cell 8
submissions.to_csv("submission.csv", index=False)

_GLOBAL_POOL.shutdown(wait=True)
