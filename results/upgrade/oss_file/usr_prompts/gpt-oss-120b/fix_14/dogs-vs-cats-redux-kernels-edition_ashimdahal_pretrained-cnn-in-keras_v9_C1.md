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

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
        input/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
            test/
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
            train/
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
        working/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

# 5. Code solution

## === cell 0
import os
import zipfile
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor  # use threads for OpenCV I/O



## === cell 1
possible_roots = [
    os.path.join("/kaggle", "input", "dogs-vs-cats-redux-kernels-edition"),
    os.path.join("/kaggle", "working", "dogs-vs-cats-redux-kernels-edition"),
    os.path.join("kaggle", "input", "dogs-vs-cats-redux-kernels-edition"),
    os.path.join("working", "dogs-vs-cats-redux-kernels-edition"),
    "dogs-vs-cats-redux-kernels-edition",
    os.path.join("data", "dogs-vs-cats-redux-kernels-edition"),
]
ROOT_DIR = next((p for p in possible_roots if os.path.isdir(p)), None)

if ROOT_DIR is None:
    raise FileNotFoundError(
        "Dataset root directory not found. Checked: " + ", ".join(possible_roots)
    )




## === cell 2
train_dir = os.path.join(ROOT_DIR, "train")
test_dir = os.path.join(ROOT_DIR, "test")




## === cell 3
train_images = [
    os.path.join(dp, f)
    for dp, _, filenames in os.walk(train_dir)
    for f in filenames
    if f.lower().endswith(".jpg")
]

test_images = [
    os.path.join(dp, f)
    for dp, _, filenames in os.walk(test_dir)
    for f in filenames
    if f.lower().endswith(".jpg")
]

train_images.sort()
test_images.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

print(f"Found {len(train_images)} training images and {len(test_images)} test images.")




## === cell 4
IMG_SIZE = 64


def _load_resize(path):
    """Read, resize, and normalize a single image to float32 in [0,1]."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Unable to read image {path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    return img.astype(np.float32) / 255.0


def load_and_resize(image_paths):
    """
    Parallel loading and resizing of images using threads.
    Returns a float32 array with values already in [0,1].
    """
    n = len(image_paths)
    data = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    max_workers = os.cpu_count() or 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, img in enumerate(
            executor.map(_load_resize, image_paths, chunksize=256)
        ):
            data[idx] = img
    return data




## === cell 5
train_imgs = load_and_resize(train_images)




## === cell 6
labels = np.array(
    [1 if "dog" in os.path.basename(p).lower() else 0 for p in train_images]
)




## === cell 7
X = train_imgs.reshape(len(train_imgs), -1)
X_train, X_val, y_train, y_val = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)




## === cell 8
clf = LogisticRegression(
    max_iter=1000,  # reduced from 3000; still enough given tol=1e-3
    tol=1e-3,
    solver="saga",
    n_jobs=-1,
    C=5.0,
    class_weight="balanced",
    random_state=42,
)
clf.fit(X_train, y_train)




## === cell 9
val_pred = clf.predict_proba(X_val)[:, 1]
print("Validation LogLoss:", log_loss(y_val, val_pred))




## === cell 10
test_imgs = load_and_resize(test_images)

test_X = test_imgs.reshape(len(test_imgs), -1)
test_X = scaler.transform(test_X)

test_pred = clf.predict_proba(test_X)[:, 1]

test_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in test_images]
submission = pd.DataFrame({"id": test_ids, "label": test_pred})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with", len(submission), "rows.")
