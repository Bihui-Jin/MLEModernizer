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

3.8

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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

0.9926

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

base_dir_candidates = [
    "/kaggle/input/aerial-cactus-identification",
    "./working/aerial-cactus-identification",
    "./aerial-cactus-identification",
    ".",
]
base_dir = next((p for p in base_dir_candidates if os.path.isdir(p)), None)
if base_dir is None:
    raise FileNotFoundError(
        "Dataset base directory not found in any of the expected locations."
    )

train_path = os.path.join(base_dir, "train")
test_path = os.path.join(base_dir, "test")

train_data = pd.read_csv(os.path.join(base_dir, "train.csv"))
test_data = pd.read_csv(os.path.join(base_dir, "sample_submission.csv"))
train_data["has_cactus"] = train_data["has_cactus"].astype(int)



## === cell 1
train_count = len(os.listdir(train_path)) if os.path.isdir(train_path) else 0
test_count = len(os.listdir(test_path)) if os.path.isdir(test_path) else 0
print("Training Images:", train_count)
print("Testing Images :", test_count)




## === cell 2
def load_images(df, directory, target_size=(32, 32)):
    """Load images listed in df['id'] from directory, resize, scale to [0,1] and flatten."""
    imgs = []
    for img_name in df["id"]:
        img_path = os.path.join(directory, img_name)
        with Image.open(img_path) as im:
            im = im.convert("RGB").resize(target_size)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            imgs.append(arr.flatten())
    return np.stack(imgs)


print("Loading training images...")
X = load_images(train_data, train_path)
y = train_data["has_cactus"].values
print("Training data shape:", X.shape)

print("Loading test images...")
X_test = load_images(test_data, test_path)
print("Test data shape:", X_test.shape)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/1692497744.py in <cell line: 0>()
     13 
     14 print("Loading training images...")
---> 15 X = load_images(train_data, train_path)
     16 y = train_data["has_cactus"].values
     17 print("Training data shape:", X.shape)

/tmp/ipykernel_10/1692497744.py in load_images(df, directory, target_size)
      5         img_path = os.path.join(directory, img_name)
      6         # open, convert to RGB, resize, to array
----> 7         with Image.open(img_path) as im:
      8             im = im.convert("RGB").resize(target_size)
      9             arr = np.asarray(im, dtype=np.float32) / 255.0

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/aerial-cactus-identification/train/2de8f189f1dce439766637e75df0ee27.jpg'

## === cell 3
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1061321914.py in <cell line: 0>()
      1 # Split training set for validation
      2 X_train, X_val, y_train, y_val = train_test_split(
----> 3     X, y, test_size=0.20, stratify=y, random_state=42
      4 )
      5 

NameError: name 'X' is not defined

## === cell 4
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)
rf.fit(X_train, y_train)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/494412505.py in <cell line: 0>()
     10     class_weight="balanced",
     11 )
---> 12 rf.fit(X_train, y_train)
     13 

NameError: name 'X_train' is not defined

## === cell 5
val_pred = rf.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {val_auc:.5f}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/4140854364.py in <cell line: 0>()
      1 # Evaluate on validation set (AUC)
----> 2 val_pred = rf.predict_proba(X_val)[:, 1]
      3 val_auc = roc_auc_score(y_val, val_pred)
      4 print(f"Validation AUC: {val_auc:.5f}")
      5 

NameError: name 'X_val' is not defined

## === cell 6
test_pred = rf.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({"id": test_data["id"], "has_cactus": test_pred})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
submission.head()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3418994746.py in <cell line: 0>()
      1 # Predict on test set and create submission
----> 2 test_pred = rf.predict_proba(X_test)[:, 1]
      3 submission = pd.DataFrame({"id": test_data["id"], "has_cactus": test_pred})
      4 submission.to_csv("submission.csv", index=False)
      5 print("Submission saved to submission.csv")

NameError: name 'X_test' is not defined
