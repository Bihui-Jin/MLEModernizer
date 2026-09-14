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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.9705

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, cv2, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score


def find_data_root():
    """
    Robustly locate the directory that contains:
        - train.csv
        - a subfolder named 'train' with training images
        - a subfolder named 'test'  with test images
    Returns the absolute path to that directory.
    """
    possible_roots = [
        "/kaggle/input/aerial-cactus-identification",
        "/kaggle/input",
        "/kaggle/working",
        "/kaggle/working/aerial-cactus-identification",
        "./kaggle/working/aerial-cactus-identification",
        "./data/aerial-cactus-identification",
        "./input/aerial-cactus-identification",
        "./working/aerial-cactus-identification",
        "./aerial-cactus-identification",
        "/kaggle/data/aerial-cactus-identification",
        "./input",
        "./working",
        ".",
    ]

    for base in possible_roots:
        base = os.path.abspath(base)
        if os.path.isdir(base):
            train_csv = os.path.join(base, "train.csv")
            train_dir = os.path.join(base, "train")
            test_dir = os.path.join(base, "test")
            if (
                os.path.isfile(train_csv)
                and os.path.isdir(train_dir)
                and os.path.isdir(test_dir)
            ):
                return base
            for entry in os.listdir(base):
                sub = os.path.join(base, entry)
                if (
                    os.path.isdir(sub)
                    and os.path.isfile(os.path.join(sub, "train.csv"))
                    and os.path.isdir(os.path.join(sub, "train"))
                    and os.path.isdir(os.path.join(sub, "test"))
                ):
                    return sub

    cur = os.getcwd()
    while True:
        cand = os.path.abspath(cur)
        if (
            os.path.isfile(os.path.join(cand, "train.csv"))
            and os.path.isdir(os.path.join(cand, "train"))
            and os.path.isdir(os.path.join(cand, "test"))
        ):
            return cand
        parent = os.path.abspath(os.path.join(cur, os.pardir))
        if parent == cur:
            break
        cur = parent

    raise FileNotFoundError(
        "Dataset root not found. Checked common locations, walked up from cwd, and performed a recursive search."
    )


try:
    DATA_ROOT = find_data_root()
except FileNotFoundError:
    fallback_paths = [
        os.path.abspath("./working/aerial-cactus-identification"),
        os.path.abspath("./input/aerial-cactus-identification"),
        os.path.abspath("./aerial-cactus-identification"),
    ]
    for fp in fallback_paths:
        if (
            os.path.isfile(os.path.join(fp, "train.csv"))
            and os.path.isdir(os.path.join(fp, "train"))
            and os.path.isdir(os.path.join(fp, "test"))
        ):
            DATA_ROOT = fp
            break
    else:
        raise FileNotFoundError(
            "Unable to locate dataset root. Checked fallback paths."
        )

print("Using data root:", DATA_ROOT)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_12/1758458583.py in <cell line: 0>()
     75 try:
---> 76     DATA_ROOT = find_data_root()
     77 except FileNotFoundError:

/tmp/ipykernel_12/1758458583.py in find_data_root()
     68 
---> 69     raise FileNotFoundError(
     70         "Dataset root not found. Checked common locations, walked up from cwd, and performed a recursive search."

FileNotFoundError: Dataset root not found. Checked common locations, walked up from cwd, and performed a recursive search.

During handling of the above exception, another exception occurred:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_12/1758458583.py in <cell line: 0>()
     90             break
     91     else:
---> 92         raise FileNotFoundError(
     93             "Unable to locate dataset root. Checked fallback paths."
     94         )

FileNotFoundError: Unable to locate dataset root. Checked fallback paths.

## === cell 1
train_csv_path = os.path.join(DATA_ROOT, "train.csv")
train_df = (
    pd.read_csv(train_csv_path).sample(frac=1, random_state=42).reset_index(drop=True)
)

images = train_df["id"].tolist()
target = train_df["has_cactus"].tolist()

train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    images,
    target,
    test_size=0.1,
    random_state=42,
    stratify=target,
)

del train_df, images, target




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1739910737.py in <cell line: 0>()
----> 1 train_csv_path = os.path.join(DATA_ROOT, "train.csv")
      2 train_df = (
      3     pd.read_csv(train_csv_path).sample(frac=1, random_state=42).reset_index(drop=True)
      4 )
      5 

NameError: name 'DATA_ROOT' is not defined

## === cell 2
def load_image(imname, subfolder):
    """
    Load a 32x32 RGB image from the given subfolder ('train' or 'test')
    under DATA_ROOT.
    """
    img_path = os.path.join(DATA_ROOT, subfolder, imname)
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image {imname} not found at {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image {img_path}")
    img = cv2.resize(img, (32, 32))
    img = img.astype(np.float32) / 255.0
    return img




## === cell 3
train_X = np.stack([load_image(fname, "train").reshape(-1) for fname in train_imgs])
val_X = np.stack([load_image(fname, "train").reshape(-1) for fname in val_imgs])
train_y = np.array(train_labels, dtype=np.int32)
val_y = np.array(val_labels, dtype=np.int32)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/445825700.py in <cell line: 0>()
----> 1 train_X = np.stack([load_image(fname, "train").reshape(-1) for fname in train_imgs])
      2 val_X = np.stack([load_image(fname, "train").reshape(-1) for fname in val_imgs])
      3 train_y = np.array(train_labels, dtype=np.int32)
      4 val_y = np.array(val_labels, dtype=np.int32)
      5 

NameError: name 'train_imgs' is not defined

## === cell 4
rf = ExtraTreesClassifier(
    n_estimators=3000,  # slight increase for potential AUC gain
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)
rf.fit(train_X, train_y)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/3111998055.py in <cell line: 0>()
      9     class_weight="balanced",
     10 )
---> 11 rf.fit(train_X, train_y)
     12 
     13 

NameError: name 'train_X' is not defined

## === cell 5
val_pred = rf.predict_proba(val_X)[:, 1]
auc = roc_auc_score(val_y, val_pred)
print(f"Validation AUC: {auc:.5f}")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1852072793.py in <cell line: 0>()
----> 1 val_pred = rf.predict_proba(val_X)[:, 1]
      2 auc = roc_auc_score(val_y, val_pred)
      3 print(f"Validation AUC: {auc:.5f}")
      4 
      5 

NameError: name 'val_X' is not defined

## === cell 6
test_dir = os.path.join(DATA_ROOT, "test")
if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"Test directory not found at {test_dir}")

test_list = sorted(os.listdir(test_dir))
test_imgs = np.stack([load_image(fname, "test").reshape(-1) for fname in test_list])
test_pred = rf.predict_proba(test_imgs)[:, 1]

submission = pd.DataFrame({"id": test_list, "has_cactus": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with shape:", submission.shape)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1610743954.py in <cell line: 0>()
----> 1 test_dir = os.path.join(DATA_ROOT, "test")
      2 if not os.path.isdir(test_dir):
      3     raise FileNotFoundError(f"Test directory not found at {test_dir}")
      4 
      5 test_list = sorted(os.listdir(test_dir))

NameError: name 'DATA_ROOT' is not defined
