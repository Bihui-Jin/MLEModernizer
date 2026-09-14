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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from concurrent.futures import (
    ThreadPoolExecutor,
)  # use threads for I/O‑bound image loading




## === cell 1
_possible_paths = [
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition",
    os.path.join(os.getcwd(), "input", "dogs-vs-cats-redux-kernels-edition"),
    "./input",
]
PATH = next((p for p in _possible_paths if os.path.isdir(p)), None)
if PATH is None:
    raise FileNotFoundError(
        "Could not locate the dataset directory. Checked paths: "
        + ", ".join(_possible_paths)
    )




## === cell 2
IMG_SIZE = 64  # a slightly larger resolution for more detail
USE_COLOR = True  # use RGB channels (3‑channel)




## === cell 3
def load_image(filepath):
    """Load an image, resize, convert to RGB/gray, and flatten."""
    img = Image.open(filepath)
    if not USE_COLOR:
        img = img.convert("L")  # grayscale
    else:
        img = img.convert("RGB")  # ensure 3 channels
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)
    return arr.flatten()




## === cell 4
def _load_image_full_path(full_path):
    """Helper for ThreadPoolExecutor: load and flatten a single image."""
    return load_image(full_path)


def build_feature_matrix(filenames, base_dir):
    """Build a 2‑D feature matrix (samples × features) using parallel image loading."""
    n_samples = len(filenames)
    n_features = IMG_SIZE * IMG_SIZE * (3 if USE_COLOR else 1)
    features = np.empty((n_samples, n_features), dtype=np.float32)

    full_paths = [os.path.join(base_dir, fn) for fn in filenames]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for idx, arr in enumerate(
            executor.map(_load_image_full_path, full_paths, chunksize=100)
        ):
            features[idx] = arr

    return features




## === cell 5
train_dir = os.path.join(PATH, "train")
cat_dir = os.path.join(train_dir, "cat")
dog_dir = os.path.join(train_dir, "dog")

cat_files = [
    os.path.join("train", "cat", entry.name)
    for entry in os.scandir(cat_dir)
    if entry.is_file() and entry.name.lower().endswith(".jpg")
]
dog_files = [
    os.path.join("train", "dog", entry.name)
    for entry in os.scandir(dog_dir)
    if entry.is_file() and entry.name.lower().endswith(".jpg")
]

train_fnames = cat_files + dog_files
y_train = np.array([0] * len(cat_files) + [1] * len(dog_files), dtype=np.int8)

test_dir = os.path.join(PATH, "test", "unknown")
test_files = [
    os.path.join("test", "unknown", entry.name)
    for entry in os.scandir(test_dir)
    if entry.is_file() and entry.name.lower().endswith(".jpg")
]
test_fnames = sorted(test_files)  # keep deterministic order




## === cell 6
print("Building training feature matrix...")
X_train = build_feature_matrix(train_fnames, PATH)




## === cell 7
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)

clf = LogisticRegression(
    solver="lbfgs",
    max_iter=2000,
    C=1.0,  # weaker regularisation than before
    class_weight="balanced",
    random_state=42,
)

clf.fit(X_tr_scaled, y_tr)
val_pred = clf.predict_proba(X_val_scaled)[:, 1]
print(f"Validation LogLoss: {log_loss(y_val, val_pred):.5f}")




## === cell 8
scaler_full = StandardScaler()
X_train_scaled = scaler_full.fit_transform(X_train)

clf_full = LogisticRegression(
    solver="lbfgs",
    max_iter=2000,
    C=1.0,
    class_weight="balanced",
    random_state=42,
)
clf_full.fit(X_train_scaled, y_train)

print("Building test feature matrix...")
X_test = build_feature_matrix(test_fnames, PATH)
X_test_scaled = scaler_full.transform(X_test)

test_probs = clf_full.predict_proba(X_test_scaled)[:, 1]

submission = pd.DataFrame(
    {
        "id": [os.path.splitext(os.path.basename(f))[0] for f in test_fnames],
        "label": test_probs,
    }
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
