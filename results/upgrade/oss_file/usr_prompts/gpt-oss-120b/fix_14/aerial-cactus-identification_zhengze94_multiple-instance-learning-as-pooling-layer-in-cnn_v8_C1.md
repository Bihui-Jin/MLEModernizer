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
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

0.8572

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.99494) has done: 'I fixed the protobuf import issue, corrected the image‑loading paths (so every image is read as a 32×32×3 RGB array), repaired the custom `noisyand` layer (using the proper TensorFlow‑2 shape handling), updated the model compilation, and simplified the notebook to only the essential steps needed to train the CNN and write a valid `sample_submission.csv`. The added changes keep the original architecture and training logic while ensuring the script runs end‑to‑end and produces a proper submission file.'

# 9. Code solution

## === cell 0
import os, sys, glob
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

USE_TF = False
print("TensorFlow disabled; using sklearn fallback model.")
print(
    "Available top‑level folders in /kaggle/input:",
    os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "N/A",
)




## === cell 1
def find_base_dir():
    """
    Locate the dataset root containing `train.csv`, `sample_submission.csv`,
    and the `train` / `test` image folders. Checks common Kaggle mount points
    and also relative “data”/“input”/“working” directories.
    """
    common_paths = [
        "/kaggle/input/aerial-cactus-identification",
        "/kaggle/working/aerial-cactus-identification",
        "/kaggle/input",
        "/kaggle/working",
        os.path.abspath("aerial-cactus-identification"),
        os.path.abspath("./aerial-cactus-identification"),
        os.path.abspath("./data/aerial-cactus-identification"),
        os.path.abspath("./input/aerial-cactus-identification"),
        os.path.abspath("./working/aerial-cactus-identification"),
        os.getcwd(),
    ]

    for candidate in common_paths:
        if (
            os.path.isdir(candidate)
            and os.path.isfile(os.path.join(candidate, "train.csv"))
            and os.path.isfile(os.path.join(candidate, "sample_submission.csv"))
            and os.path.isdir(os.path.join(candidate, "train"))
            and os.path.isdir(os.path.join(candidate, "test"))
        ):
            return candidate

    for root, dirs, files in os.walk(os.getcwd()):
        if "train.csv" in files and "sample_submission.csv" in files:
            if "train" in dirs and "test" in dirs:
                return root

    raise FileNotFoundError(
        "Folder containing 'train.csv', 'sample_submission.csv', and 'train'/'test' not found."
    )


BASE_DIR = find_base_dir()
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test")

print("Using base directory:", BASE_DIR)
print("Using train images from:", TRAIN_IMG_DIR)
print("Using test images from:", TEST_IMG_DIR)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2761428362.py in <cell line: 0>()
     39 
     40 
---> 41 BASE_DIR = find_base_dir()
     42 TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
     43 TEST_IMG_DIR = os.path.join(BASE_DIR, "test")

/tmp/ipykernel_11/2761428362.py in find_base_dir()
     34                 return root
     35 
---> 36     raise FileNotFoundError(
     37         "Folder containing 'train.csv', 'sample_submission.csv', and 'train'/'test' not found."
     38     )

FileNotFoundError: Folder containing 'train.csv', 'sample_submission.csv', and 'train'/'test' not found.

## === cell 2
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Unable to read image {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
    img = cv2.resize(img, (32, 32))
    return img.astype(np.float32) / 255.0




## === cell 3
train_csv_path = os.path.join(BASE_DIR, "train.csv")
train_csv = pd.read_csv(train_csv_path)

X_train_list = []
Y_train = []

for _, row in train_csv.iterrows():
    img_path = os.path.join(TRAIN_IMG_DIR, row["id"])
    try:
        X_train_list.append(load_image(img_path))
        Y_train.append(int(row["has_cactus"]))
    except FileNotFoundError as e:
        print(e)

X_train = np.array(X_train_list)
Y_train = np.array(Y_train)
print("Training data shape:", X_train.shape, "Labels shape:", Y_train.shape)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/715156246.py in <cell line: 0>()
----> 1 train_csv_path = os.path.join(BASE_DIR, "train.csv")
      2 train_csv = pd.read_csv(train_csv_path)
      3 
      4 X_train_list = []
      5 Y_train = []

NameError: name 'BASE_DIR' is not defined

## === cell 4
if USE_TF:
    pass
else:
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    print("Using sklearn RandomForestClassifier as fallback model.")



## === cell 5
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

x_tr, x_val, y_tr, y_val = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/979777858.py in <cell line: 0>()
      3 
      4 x_tr, x_val, y_tr, y_val = train_test_split(
----> 5     X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train
      6 )
      7 

NameError: name 'X_train' is not defined

## === cell 6
if USE_TF:
    pass
else:
    X_tr_flat = x_tr.reshape((x_tr.shape[0], -1))
    X_val_flat = x_val.reshape((x_val.shape[0], -1))
    model.fit(X_tr_flat, y_tr)
    val_proba = model.predict_proba(X_val_flat)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    print(f"Validation AUC (RandomForest): {val_auc:.4f}")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3372305496.py in <cell line: 0>()
      3 else:
      4     # Flatten images for the RandomForest
----> 5     X_tr_flat = x_tr.reshape((x_tr.shape[0], -1))
      6     X_val_flat = x_val.reshape((x_val.shape[0], -1))
      7     model.fit(X_tr_flat, y_tr)

NameError: name 'x_tr' is not defined

## === cell 7
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
submission = pd.read_csv(sample_sub_path)
preds = np.empty(len(submission), dtype=np.float32)

for idx in tqdm(range(len(submission))):
    img_path = os.path.join(TEST_IMG_DIR, submission.loc[idx, "id"])
    img = load_image(img_path)  # will raise if missing
    if USE_TF:
        preds[idx] = model.predict(img.reshape(1, 32, 32, 3), verbose=0)[0][0]
    else:
        flat_img = img.reshape(1, -1)
        preds[idx] = model.predict_proba(flat_img)[0][1]

submission["has_cactus"] = preds
output_path = "sample_submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/997964315.py in <cell line: 0>()
----> 1 sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
      2 submission = pd.read_csv(sample_sub_path)
      3 preds = np.empty(len(submission), dtype=np.float32)
      4 
      5 for idx in tqdm(range(len(submission))):

NameError: name 'BASE_DIR' is not defined
