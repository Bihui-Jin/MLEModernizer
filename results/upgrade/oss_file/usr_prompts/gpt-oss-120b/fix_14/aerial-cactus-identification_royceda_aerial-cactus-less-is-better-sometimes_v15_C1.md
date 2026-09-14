# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import shutil, zipfile, os, io

shutil.copy("/kaggle/input/aerial-cactus-identification/train.csv", ".")

TRAIN_ZIP_PATH = "/kaggle/input/aerial-cactus-identification/train.zip"
TEST_ZIP_PATH = "/kaggle/input/aerial-cactus-identification/test.zip"



## === cell 1
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

np.random.seed(42)
tf.random.set_seed(42)

TRAIN_ZIP = zipfile.ZipFile(TRAIN_ZIP_PATH, "r")
TEST_ZIP = zipfile.ZipFile(TEST_ZIP_PATH, "r")

print(f"Using TRAIN_ZIP = {TRAIN_ZIP_PATH}")
print(f"Using TEST_ZIP = {TEST_ZIP_PATH}")



## === cell 2
df = pd.read_csv("train.csv")
print(df.head())



## === cell 3
train_df, validate_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["has_cactus"]
)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)



## === cell 4
IMAGE_SIZE = (32, 32)

from concurrent.futures import ThreadPoolExecutor
from PIL import Image  # Pillow is lightweight and fast for small JPEGs
from io import BytesIO


def _load_one_path(args):
    """
    Load a single JPEG image from a zip archive using Pillow; returns (index, flattened float32 array).
    If loading fails, returns a zero‑filled array to preserve ordering.
    """
    idx, name, zip_obj = args
    try:
        data = zip_obj.read(name)
        with Image.open(BytesIO(data)) as img:
            img = img.convert("RGB")
            img = img.resize(IMAGE_SIZE, Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0  # (32,32,3)
            flat = arr.ravel()
    except Exception:
        flat = np.zeros(IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3, dtype=np.float32)
    return idx, flat


def load_images_once(image_ids, zip_obj):
    """
    Load every image in `image_ids` exactly once from the provided zip file using a ThreadPoolExecutor.
    Returns a NumPy array with rows ordered identically to `image_ids`.
    """
    n = len(image_ids)
    out = np.empty((n, IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3), dtype=np.float32)

    args = [(i, name, zip_obj) for i, name in enumerate(image_ids)]

    max_workers = min(os.cpu_count() or 1, 16)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, flat in executor.map(_load_one_path, args, chunksize=1024):
            out[idx] = flat
    return out


print("Loading all training images (train + validation) once...")
all_ids = pd.concat([train_df["id"], validate_df["id"]]).reset_index(drop=True)
X_all = load_images_once(all_ids.tolist(), TRAIN_ZIP)

n_train = len(train_df)
X_train = X_all[:n_train]
X_val = X_all[n_train:]

y_train = train_df["has_cactus"].values
y_val = validate_df["has_cactus"].values



## === cell 5
model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
)

model.fit(X_train, y_train)

val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {val_auc:.5f}")



## === cell 6
sample_sub = pd.read_csv("sample_submission.csv")
test_ids = sample_sub["id"].astype(str).tolist()  # preserve order

print("Loading test images...")
X_test = load_images_once(test_ids, TEST_ZIP)

assert X_test.shape[0] == len(test_ids), "Mismatch between test images and predictions"

test_pred = model.predict_proba(X_test)[:, 1]

test_df = pd.DataFrame({"id": test_ids, "has_cactus": test_pred})



## === cell 7
submission = test_df[["id", "has_cactus"]]
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} (rows: {len(submission)})")

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers should have the same number of rows
