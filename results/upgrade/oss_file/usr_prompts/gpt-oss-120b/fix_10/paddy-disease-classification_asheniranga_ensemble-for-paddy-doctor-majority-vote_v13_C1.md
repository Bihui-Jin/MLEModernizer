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

0.9815668202764976

# 6. Current score

0.62721

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0319) has done: 'I fixed the import errors, removed the unavailable TensorFlow Hub models, and replaced the broken ensemble pipeline with a single EfficientNet B0 transfer‑learning model that trains on the provided training folders. The script now correctly builds data generators, trains a few epochs, predicts on the test set, maps the predicted class indices back to label names, and writes a proper submission file `model_submission_v21.csv` containing the required `image_id` and `label` columns.'
- What this solution (achieved 0.64643) has done: 'I replace the failing TensorFlow pipeline with a lightweight scikit‑learn solution that reads and resizes the images, trains a RandomForest classifier, and writes a correctly‑formatted CSV submission. This removes the TF import error, eliminates unsupported `workers` arguments, and guarantees a valid `model_submission_v21.csv` file while keeping the overall workflow (data loading → model training → prediction → submission) intact.'
- What this solution (achieved 0.62721) has done: 'I increase the image resolution to give the RandomForest more informative pixel features, add the normalized age column as an extra feature (training data only, zero‑filled for test), and use a slightly larger forest with balanced class weighting. These minimal tweaks keep the overall pipeline unchanged while providing extra predictive signal, which should raise the validation accuracy toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import multiprocessing

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"




## === cell 1
base_path = "/kaggle/input/paddy-disease-classification"
train_dir = os.path.join(base_path, "train_images")
train_csv_path = os.path.join(base_path, "train.csv")
test_dir = os.path.join(base_path, "test_images")
sample_sub_path = os.path.join(base_path, "sample_submission.csv")

IMG_SIZE = (96, 96)  # modest increase from 64x64
BATCH_SIZE = 32
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)

train_df = pd.read_csv(train_csv_path)

filename_to_path = {}
for root, _, files in os.walk(train_dir):
    for f in files:
        if f.lower().endswith(".jpg"):
            filename_to_path[f] = os.path.join(root, f)


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8).flatten()


train_images = []
train_labels = []
for _, row in train_df.iterrows():
    fname = row["image_id"]
    if fname in filename_to_path:
        img_path = filename_to_path[fname]
        train_images.append(load_image(img_path))
        train_labels.append(row["label"])
    else:
        continue

X_train_img = np.stack(train_images)  # shape (n_samples, height*width*3)

age_vals = train_df["age"].astype(np.float32).values
age_norm = age_vals / (age_vals.max() if age_vals.max() != 0 else 1)
age_norm = age_norm.reshape(-1, 1)

X_train = np.hstack([X_train_img, age_norm])

le = LabelEncoder()
y_train = le.fit_transform(train_labels)




## === cell 2
rf = RandomForestClassifier(
    n_estimators=500,  # more trees for better stability
    max_depth=None,
    n_jobs=NUM_WORKERS,
    random_state=42,
    class_weight="balanced",  # handle class imbalance
)
rf.fit(X_train, y_train)




## === cell 3
test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_images = []
test_paths = []
for fname in test_files:
    img_path = os.path.join(test_dir, fname)
    test_images.append(load_image(img_path))
    test_paths.append(fname)

X_test_img = np.stack(test_images)

age_test = np.zeros((X_test_img.shape[0], 1), dtype=np.float32)

X_test = np.hstack([X_test_img, age_test])

test_pred_idx = rf.predict(X_test)
test_pred_labels = le.inverse_transform(test_pred_idx)




## === cell 4
submission = pd.DataFrame({"image_id": test_paths, "label": test_pred_labels})
submission_path = "model_submission_v21.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, shape: {submission.shape}")
