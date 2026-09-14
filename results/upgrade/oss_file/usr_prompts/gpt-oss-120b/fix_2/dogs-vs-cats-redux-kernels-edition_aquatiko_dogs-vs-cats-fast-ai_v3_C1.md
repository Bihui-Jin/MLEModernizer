# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss



## === cell 1
PATH = "../input/"

IMG_SIZE = 32  # 32x32 pixels




## === cell 2
def gather_train_data(base_path):
    """
    Walk through the train directory, collect image file paths and labels.
    Labels: 0 = cat, 1 = dog (derived from the parent folder name).
    Returns lists of relative paths (relative to PATH) and corresponding labels.
    """
    train_dir = os.path.join(base_path, "train")
    fnames, labels = [], []
    for root, _, files in os.walk(train_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                rel_path = os.path.relpath(os.path.join(root, f), base_path)
                fnames.append(rel_path)
                label = 0 if "cat" in os.path.basename(root).lower() else 1
                labels.append(label)
    return np.array(fnames), np.array(labels)


train_fnames, train_labels = gather_train_data(PATH)

print(f"Found {len(train_fnames)} training images.")




## === cell 3
def image_to_vector(path, img_size=IMG_SIZE):
    """Load an image, resize, and flatten to a 1‑D numpy array (RGB)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((img_size, img_size))
        return np.asarray(img, dtype=np.float32).ravel() / 255.0  # normalise


def build_feature_matrix(fnames, base_path):
    """Convert a list of relative filenames to a 2‑D feature matrix."""
    X = []
    for rel in fnames:
        full_path = os.path.join(base_path, rel)
        X.append(image_to_vector(full_path))
    return np.stack(X)


X_train = build_feature_matrix(train_fnames, PATH)
y_train = train_labels

print(f"Feature matrix shape: {X_train.shape}")



## === cell 4
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

clf = LogisticRegression(
    solver="liblinear",
    max_iter=200,
    class_weight="balanced",
    random_state=42,
)

clf.fit(X_tr, y_tr)

val_pred = clf.predict_proba(X_val)[:, 1]
print(f"Validation LogLoss: {log_loss(y_val, val_pred):.5f}")




## === cell 5
def gather_test_data(base_path):
    """Collect test image filenames (relative to base_path)."""
    test_dir = os.path.join(base_path, "test")
    fnames = []
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                rel_path = os.path.relpath(os.path.join(root, f), base_path)
                fnames.append(rel_path)
    return np.array(fnames)


test_fnames = gather_test_data(PATH)
print(f"Found {len(test_fnames)} test images.")



## === cell 6
X_test = build_feature_matrix(test_fnames, PATH)

test_probs = clf.predict_proba(X_test)[:, 1]



## === cell 7
test_ids = [os.path.splitext(os.path.basename(f))[0] for f in test_fnames]

assert len(test_ids) == len(test_probs)

submission = pd.DataFrame({"id": test_ids, "label": test_probs})
submission.head()



## === cell 8
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
