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
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from PIL import Image



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
    """Return a 6‑dim colour statistic vector (mean + std for each RGB channel)."""
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((32, 32))
        arr = np.array(img) / 255.0  # normalise
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        return np.concatenate([mean, std])
    except Exception:
        return np.zeros(6, dtype=np.float32)


train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images"
test_img_dir = "../input/plant-pathology-2021-fgvc8/test_images"




## === cell 4
def build_feature_matrix(idxs):
    feats = []
    for i in idxs:
        img_name = train.iloc[i]["image"]
        img_path = os.path.join(train_img_dir, img_name)
        feats.append(extract_features(img_path))
    return np.vstack(feats)


X_train = build_feature_matrix(train_idx)
X_val = build_feature_matrix(val_idx)
y_train = true_bin[train_idx]
y_val = true_bin[val_idx]



## === cell 5
base_clf = LogisticRegression(solver="lbfgs", max_iter=1000, n_jobs=-1)
clf = OneVsRestClassifier(base_clf)
clf.fit(X_train, y_train)

val_prob = clf.predict_proba(X_val)
val_pred = (np.array(val_prob) >= 0.5).astype(int)
val_score = f1_score(y_val, val_pred, average="samples")
print(f"Validation samples‑averaged F1: {val_score:.5f}")



## === cell 6
X_full = np.vstack([X_train, X_val])
y_full = np.vstack([y_train, y_val])
clf_full = OneVsRestClassifier(base_clf)
clf_full.fit(X_full, y_full)



## === cell 7
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")



## === cell 8
test_feats = []
for img_name in submissions["image"]:
    img_path = os.path.join(test_img_dir, img_name)
    test_feats.append(extract_features(img_path))
X_test = np.vstack(test_feats)

test_prob = clf_full.predict_proba(X_test)
test_pred = (np.array(test_prob) >= 0.5).astype(int)

pred_labels = []
for row in test_pred:
    lbls = [mlb.classes_[i] for i, v in enumerate(row) if v == 1]
    if not lbls:
        lbls = [mlb.classes_[0]]
    pred_labels.append(" ".join(lbls))

submissions["labels"] = pred_labels



## === cell 9
submissions.to_csv("submission.csv", index=False)
