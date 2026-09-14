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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.97926267281106

# 6. Current score

0.57802

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.18025) has done: 'The fix replaces the missing‑model loading with a lightweight, end‑to‑end training pipeline: it builds a small transfer‑learning model (MobileNetV2), trains it on the provided training images, predicts on the test set, and writes a correctly formatted `submission.csv`. All previous cells that tried to load unavailable HDF5 models are removed, and the new cells handle data loading, label mapping, model definition, training, inference, and CSV export.'
- What this solution (achieved 0.57802) has done: 'I fixed the script by removing the failing TensorFlow import and the empty data generators that couldn’t locate images in the nested class folders. Instead, I added a simple image‑loading routine using Pillow, encoded the labels, and trained a lightweight RandomForest classifier on resized 64×64 RGB pixels. The test images are processed the same way, predictions are mapped back to the original label strings, and a correctly formatted `submission.csv` is written. This resolves all runtime errors and yields a valid submission file.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



## === cell 1
BASE_INPUT = "/kaggle/input/paddy-disease-classification"
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUBMISSION = os.path.join(BASE_INPUT, "sample_submission.csv")



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)

label_list = sorted(train_df["label"].unique())
label_to_idx = {label: idx for idx, label in enumerate(label_list)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
train_df["label_idx"] = train_df["label"].map(label_to_idx)



## === cell 3
IMG_SIZE = (64, 64)  # modest size to keep memory reasonable


def load_images(df, img_root):
    """Load images referenced in `df` (expects columns `label` and `image_id`)."""
    imgs = []
    for _, row in df.iterrows():
        img_path = os.path.join(img_root, row["label"], row["image_id"])
        try:
            img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
            imgs.append(np.asarray(img, dtype=np.uint8))
        except Exception as e:
            imgs.append(np.zeros((*IMG_SIZE, 3), dtype=np.uint8))
    return np.stack(imgs)


X = load_images(train_df, TRAIN_IMG_DIR).reshape(len(train_df), -1)
y = train_df["label_idx"].values



## === cell 4
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

rf_clf = RandomForestClassifier(
    n_estimators=100, max_depth=None, random_state=42, n_jobs=-1
)
rf_clf.fit(X_train, y_train)

val_acc = rf_clf.score(X_val, y_val)
print(f"Validation accuracy (quick check): {val_acc:.4f}")



## === cell 5
test_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")])
test_df = pd.DataFrame({"image_id": test_files})


def load_test_images(df, img_root):
    imgs = []
    for _, row in df.iterrows():
        img_path = os.path.join(img_root, row["image_id"])
        try:
            img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
            imgs.append(np.asarray(img, dtype=np.uint8))
        except Exception:
            imgs.append(np.zeros((*IMG_SIZE, 3), dtype=np.uint8))
    return np.stack(imgs)


X_test = load_test_images(test_df, TEST_IMG_DIR).reshape(len(test_df), -1)

test_pred_idx = rf_clf.predict(X_test)
test_pred_labels = [idx_to_label[idx] for idx in test_pred_idx]



## === cell 6
submission = pd.DataFrame({"image_id": test_df["image_id"], "label": test_pred_labels})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")
