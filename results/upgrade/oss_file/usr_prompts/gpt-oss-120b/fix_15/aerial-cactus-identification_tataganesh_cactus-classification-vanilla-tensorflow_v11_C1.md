# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        input/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> working/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> working/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

# 5. Target score

0.9938

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pathlib
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

candidate_dirs = [
    pathlib.Path("/kaggle/input/aerial-cactus-identification"),
    pathlib.Path("input/aerial-cactus-identification"),
    pathlib.Path("../input/aerial-cactus-identification"),
    pathlib.Path("working/aerial-cactus-identification"),
    pathlib.Path("./working/aerial-cactus-identification"),
]
BASE_DIR = None
for cand in candidate_dirs:
    if cand.exists():
        BASE_DIR = cand
        break
if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate the aerial-cactus-identification data directory."
    )

TRAIN_IMAGES_DIR = BASE_DIR / "train"
TEST_IMAGES_DIR = BASE_DIR / "test"

train_csv_path = BASE_DIR / "train.csv"
sample_sub_path = BASE_DIR / "sample_submission.csv"

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(sample_sub_path)

train_df["filepath"] = train_df["id"].apply(
    lambda x: str((TRAIN_IMAGES_DIR / x).resolve())
)
test_df["filepath"] = test_df["id"].apply(
    lambda x: str((TEST_IMAGES_DIR / x).resolve())
)




## === cell 1
def load_image(path):
    """Read an image with OpenCV, resize to 32×32, convert to RGB,
    normalize, and flatten to a 1‑D float32 array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read image at {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img.flatten()


train_images = []
train_labels = []
for _, row in train_df.iterrows():
    fp = row["filepath"]
    if pathlib.Path(fp).is_file():
        try:
            train_images.append(load_image(fp))
            train_labels.append(row["has_cactus"])
        except Exception:
            continue
if not train_images:
    raise RuntimeError("No training images were loaded; check image paths.")
train_X = np.stack(train_images)
train_y = np.array(train_labels, dtype=np.int32)

test_images = []
valid_test_indices = (
    []
)  # indices in test_df that correspond to successfully loaded images
for idx, row in test_df.iterrows():
    fp = row["filepath"]
    if pathlib.Path(fp).is_file():
        try:
            test_images.append(load_image(fp))
            valid_test_indices.append(idx)
        except Exception:
            continue
if not test_images:
    raise RuntimeError("No test images were loaded; check image paths.")
test_X = np.stack(test_images)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1444643877.py in <cell line: 0>()
     23             continue
     24 if not train_images:
---> 25     raise RuntimeError("No training images were loaded; check image paths.")
     26 train_X = np.stack(train_images)
     27 train_y = np.array(train_labels, dtype=np.int32)

RuntimeError: No training images were loaded; check image paths.

## === cell 2
X_tr, X_val, y_tr, y_val = train_test_split(
    train_X, train_y, test_size=0.2, random_state=42, stratify=train_y
)

rf = RandomForestClassifier(
    n_estimators=1000,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)
rf.fit(X_tr, y_tr)

val_probs = rf.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_probs)
print(f"Validation AUC: {val_auc:.5f}")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3350249620.py in <cell line: 0>()
      1 # Train / validation split
      2 X_tr, X_val, y_tr, y_val = train_test_split(
----> 3     train_X, train_y, test_size=0.2, random_state=42, stratify=train_y
      4 )
      5 

NameError: name 'train_X' is not defined

## === cell 3
test_probs = rf.predict_proba(test_X)[:, 1]

submission = pd.DataFrame({"id": test_df["id"], "has_cactus": 0.5})
submission.loc[valid_test_indices, "has_cactus"] = test_probs

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3663446787.py in <cell line: 0>()
      1 # Predict on test set
----> 2 test_probs = rf.predict_proba(test_X)[:, 1]
      3 
      4 # Build submission
      5 submission = pd.DataFrame({"id": test_df["id"], "has_cactus": 0.5})

NameError: name 'rf' is not defined
