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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.9699721277410228

# 6. Current score

0.55756

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the failing TPU‑based data loading and heavy model code with a simple, robust pipeline that reads the CSV files directly, computes the average label frequencies from the training set, and uses those averages as constant predictions for every test image. This removes the problematic KaggleDatasets / TPU imports, fixes undefined variables, and guarantees that a correctly‑named `submission.csv` is written with the required columns, allowing the notebook to run end‑to‑end.'
- What this solution (achieved 0.57806) has done: 'I remove the problematic TensorFlow import that caused the crash, add a lightweight image‑loading routine using Pillow, and train a simple one‑vs‑rest LogisticRegression model on resized pixel data. This provides discriminative predictions (raising ROC‑AUC far above the constant‑baseline) while keeping the original pipeline structure and still writing a correctly‑named `submission.csv`.'
- What this solution (achieved 0.57828) has done: 'I enhance the feature extraction by appending simple color statistics (mean and standard deviation for each RGB channel) to the flattened pixel vector, giving the logistic regression a richer representation while keeping the overall pipeline unchanged. I also relax regularisation a bit by increasing the inverse‑regularisation strength C to 10.0, which often improves AUC for high‑dimensional image data. These minimal tweaks should move the macro‑ROC‑AUC substantially closer to the target score.'
- What this solution (achieved 0.55756) has done: 'I add feature scaling with StandardScaler so the logistic‑regression models receive zero‑mean, unit‑variance inputs, which often boosts ROC‑AUC without altering the core model or feature set. The scaler is fit on the training split (to avoid leakage) and re‑fit on the full training data before fitting the final models and making test predictions. This small preprocessing change is expected to move the validation macro‑AUC closer to the target score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image

print("Libraries loaded successfully")




## === cell 1
BASE_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"


def format_path(st):
    """Return full path to an image given its image_id (without extension)."""
    return os.path.join(BASE_DIR, "images", f"{st}.jpg")




## === cell 2
train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission_df = pd.read_csv(sample_sub_path)

label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
train_labels = train_df[label_cols].values.astype(np.float32)




## === cell 3
def load_and_preprocess(image_id, size=(64, 64)):
    """Load an image, resize, flatten, and append channel mean/std features."""
    try:
        img = Image.open(format_path(image_id)).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize to [0,1]
        flat = arr.flatten()
        means = arr.mean(axis=(0, 1))
        stds = arr.std(axis=(0, 1))
        extra = np.concatenate([means, stds]).astype(np.float32)
        return np.concatenate([flat, extra])
    except Exception as e:
        return np.zeros(size[0] * size[1] * 3 + 6, dtype=np.float32)


print("Loading and preprocessing training images...")
X = np.stack([load_and_preprocess(img_id) for img_id in train_df["image_id"].values])
y = train_labels

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=2020
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)




## === cell 4
models = {}
for i, col in enumerate(label_cols):
    lr = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
        solver="liblinear",
        C=10.0,  # less regularisation to capture more signal
    )
    lr.fit(X_train, y_train[:, i])
    models[col] = lr

val_preds = np.column_stack(
    [models[col].predict_proba(X_val)[:, 1] for col in label_cols]
)

auc = roc_auc_score(y_val, val_preds, average="macro", multi_class="ovr")
print(f"Validation macro ROC‑AUC (LogReg enhanced): {auc:.5f}")




## === cell 5
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)

models = {}
for i, col in enumerate(label_cols):
    lr = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
        solver="liblinear",
        C=10.0,
    )
    lr.fit(X_scaled_full, y[:, i])
    models[col] = lr

print("Loading and preprocessing test images...")
X_test = np.stack(
    [load_and_preprocess(img_id) for img_id in test_df["image_id"].values]
)
X_test = scaler_full.transform(X_test)

test_preds = np.column_stack(
    [models[col].predict_proba(X_test)[:, 1] for col in label_cols]
)

submission_df.loc[:, label_cols] = test_preds
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission_df.head())
