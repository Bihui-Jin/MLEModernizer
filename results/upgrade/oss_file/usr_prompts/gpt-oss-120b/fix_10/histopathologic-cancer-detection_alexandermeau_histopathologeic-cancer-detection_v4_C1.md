# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.14

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.9374697534977384

# 6. Current score

0.53924

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.53924) has done: 'I replace the TensorFlow‑based image pipeline with a lightweight NumPy/Pillow + Scikit‑Learn workflow, fixing the import errors and the misuse of symbolic tensors. The new code extracts a simple mean‑pixel feature, trains a logistic‑regression model, evaluates AUC on a validation split, and then generates a proper `submission.csv` containing probabilities for the test set.'

# 9. Code solution

## === cell 0
import os
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE_INPUT = pathlib.Path("/kaggle/input/histopathologic-cancer-detection")
TRAIN_DIR = BASE_INPUT / "train"
TEST_DIR = BASE_INPUT / "test"
TRAIN_LABELS_PATH = BASE_INPUT / "train_labels.csv"
SAMPLE_SUB_PATH = BASE_INPUT / "sample_submission.csv"

print("Base input path:", BASE_INPUT)



## === cell 1
train_labels = pd.read_csv(TRAIN_LABELS_PATH)
train_labels["train_filepath"] = train_labels["id"].apply(
    lambda x: str(TRAIN_DIR / f"{x}.tif")
)
print("Train labels shape:", train_labels.shape)
print(train_labels.head())



## === cell 2
train_df, val_df = train_test_split(
    train_labels,
    test_size=0.2,
    stratify=train_labels["label"],
    random_state=42,
)




## === cell 3
def extract_mean_red(path: str) -> float:
    """
    Load a TIFF image, resize to 96x96, and return the mean of the red channel.
    This lightweight feature works with Pillow and avoids TensorFlow.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((96, 96))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # shape (96,96,3)
    return float(arr[:, :, 0].mean())




## === cell 4
print("Extracting features for training set...")
X_train = np.array(
    [extract_mean_red(p) for p in train_df["train_filepath"].values]
).reshape(-1, 1)
y_train = train_df["label"].values

print("Extracting features for validation set...")
X_val = np.array(
    [extract_mean_red(p) for p in val_df["train_filepath"].values]
).reshape(-1, 1)
y_val = val_df["label"].values



## === cell 5
model = LogisticRegression(max_iter=1000, n_jobs=1)
model.fit(X_train, y_train)

val_preds = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_preds)
print(f"Validation AUC: {val_auc:.5f}")



## === cell 6
test_df = pd.read_csv(SAMPLE_SUB_PATH)
test_df["test_filepath"] = test_df["id"].apply(lambda x: str(TEST_DIR / f"{x}.tif"))

print("Extracting features for test set...")
X_test = np.array(
    [extract_mean_red(p) for p in test_df["test_filepath"].values]
).reshape(-1, 1)

test_probs = model.predict_proba(X_test)[:, 1]
test_df["label"] = test_probs

submission_path = "/kaggle/working/submission.csv"
test_df[["id", "label"]].to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
