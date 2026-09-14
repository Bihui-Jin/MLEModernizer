# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.6

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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Code solution

## === cell 0
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

BASE_DIR = "/kaggle/input/plant-seedlings-classification"
train_dir = os.path.join(BASE_DIR, "train")
test_dir = os.path.join(BASE_DIR, "test")
IMG_SIZE = 224  # higher resolution for richer features
MAX_PER_CLASS = None  # use all images
K_NEIGHBORS = 5  # proven optimal neighbour count

CATEGORIES = [
    "Black-grass",
    "Charlock",
    "Cleavers",
    "Common Chickweed",
    "Common wheat",
    "Fat Hen",
    "Loose Silky-bent",
    "Maize",
    "Scentless Mayweed",
    "Shepherds Purse",
    "Small-flowered Cranesbill",
    "Sugar beet",
]
NUM_CATEGORIES = len(CATEGORIES)
print("Number of categories:", NUM_CATEGORIES)




## === cell 1
def load_image_cv2(path):
    """Read an image, resize, convert to RGB and normalise to [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return img




## === cell 2
def load_training_data():
    """Load all training images (with deterministic flips + rotation) and L2‑normalize them."""
    X = []
    y = []
    rng = np.random.default_rng(seed=42)  # deterministic sampling
    for idx, cat in enumerate(CATEGORIES):
        cat_path = os.path.join(train_dir, cat)
        if not os.path.isdir(cat_path):
            continue
        all_files = [
            f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f))
        ]
        if MAX_PER_CLASS is not None and len(all_files) > MAX_PER_CLASS:
            chosen = rng.choice(all_files, size=MAX_PER_CLASS, replace=False)
        else:
            chosen = all_files
        for fname in chosen:
            fpath = os.path.join(cat_path, fname)
            img = load_image_cv2(fpath)
            X.append(img.flatten())
            y.append(idx)

            flipped_h = cv2.flip(img, 1)
            X.append(flipped_h.flatten())
            y.append(idx)

            flipped_v = cv2.flip(img, 0)
            X.append(flipped_v.flatten())
            y.append(idx)

            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            X.append(rotated.flatten())
            y.append(idx)

    X = np.stack(X)  # shape (N, IMG_SIZE*IMG_SIZE*3)
    row_norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    X = X / row_norms
    y = np.array(y, dtype=np.int32)
    return X, y




## === cell 3
def create_test_filenames():
    """Collect test image filenames in deterministic (sorted) order."""
    test_filenames = []
    for img_name in tqdm(os.listdir(test_dir), desc="loading test"):
        path = os.path.join(test_dir, img_name)
        if os.path.isfile(path):
            test_filenames.append(img_name)
    test_filenames.sort()
    return test_filenames




## === cell 4
print("Loading training data…")
X_train, y_train = load_training_data()
print("Training samples:", X_train.shape[0])




## === cell 5
def knn_predict(X_train, y_train, X_test, k=K_NEIGHBORS):
    """k‑NN with simple majority voting (no distance weighting)."""
    preds = []
    for test_vec in tqdm(X_test, desc="predicting"):
        dists = np.sum((X_train - test_vec) ** 2, axis=1)
        nearest_idx = np.argpartition(dists, k)[:k]
        nearest_labels = y_train[nearest_idx]
        vote_counts = np.bincount(nearest_labels, minlength=NUM_CATEGORIES)
        pred = vote_counts.argmax()
        preds.append(pred)
    return np.array(preds, dtype=np.int32)




## === cell 6
test_filenames = create_test_filenames()
print("Loading test images…")
X_test = np.stack(
    [
        load_image_cv2(os.path.join(test_dir, fname)).flatten()
        for fname in test_filenames
    ]
)
row_norms_test = np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-10
X_test = X_test / row_norms_test

print("Running k‑NN prediction on test set…")
pred_indices = knn_predict(X_train, y_train, X_test, k=K_NEIGHBORS)
pred_species = [CATEGORIES[i] for i in pred_indices]




## === cell 7
submission_path = "submission.csv"
with open(submission_path, "w") as f:
    f.write("file,species\n")
    for img_name, species in zip(test_filenames, pred_species):
        f.write(f"{img_name},{species}\n")
print(f"Submission written to {submission_path}")
