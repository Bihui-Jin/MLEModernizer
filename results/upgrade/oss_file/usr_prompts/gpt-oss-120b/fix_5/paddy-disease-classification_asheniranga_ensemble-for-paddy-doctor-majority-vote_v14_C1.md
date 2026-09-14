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

0.9827188940092166

# 6. Current score

0.57494

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.57494) has done: 'The fixes address the TensorFlow import issue, replace the failing TF image pipeline with a lightweight scikit‑learn model that extracts simple image features using Pillow, and ensure the script creates a correctly‑named CSV submission. This resolves the runtime errors, produces a valid submission file, and adds a basic validation step to gauge accuracy while keeping the core logic minimal.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image_dataset_from_directory

    TF_AVAILABLE = True
except Exception as e:
    print("TensorFlow import failed:", e)
    TF_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    SKLEARN_AVAILABLE = True
except Exception as e:
    print("scikit‑learn import failed:", e)
    SKLEARN_AVAILABLE = False

try:
    from PIL import Image
except Exception as e:
    raise ImportError("Pillow is required for image handling.") from e


def _find_base_path():
    candidates = [
        os.path.join("data", "paddy-disease-classification"),
        os.path.join("input", "paddy-disease-classification"),
        "/kaggle/input/paddy-disease-classification",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("Base path for paddy-disease-classification not found.")


BASE_PATH = _find_base_path()
TRAIN_DIR = os.path.join(BASE_PATH, "train_images")
TEST_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUBMISSION = os.path.join(BASE_PATH, "sample_submission.csv")

train_meta = pd.read_csv(TRAIN_CSV)
class_names = sorted(train_meta["label"].unique())
num_classes = len(class_names)

most_common_label = train_meta["label"].mode()[0]

IMG_SIZE = (64, 64)  # small size for rapid feature extraction
BATCH_SIZE = 32
SEED = 42




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_image_as_vector(path, size=IMG_SIZE):
    """Load an image, resize, normalize and flatten to a 1‑D float vector."""
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()


def build_feature_matrix(image_paths):
    """Create a feature matrix (N x D) for a list of image file paths."""
    feats = [load_image_as_vector(p) for p in image_paths]
    return np.stack(feats)




## === cell 2
if not SKLEARN_AVAILABLE:
    raise RuntimeError("Required scikit‑learn library is not available.")

train_image_paths = []
train_labels = []
for _, row in train_meta.iterrows():
    label = row["label"]
    img_id = row["image_id"]
    img_path = os.path.join(TRAIN_DIR, label, img_id)
    if not os.path.isfile(img_path):
        possible = glob.glob(os.path.join(TRAIN_DIR, "**", img_id), recursive=True)
        if possible:
            img_path = possible[0]
        else:
            raise FileNotFoundError(f"Training image not found: {img_id}")
    train_image_paths.append(img_path)
    train_labels.append(label)

print("Extracting features for training set (this may take a few minutes)...")
X = build_feature_matrix(train_image_paths)

le = LabelEncoder()
y = le.fit_transform(train_labels)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=SEED,
)
rf_clf.fit(X_train, y_train)

val_acc = rf_clf.score(X_val, y_val)
print(f"Validation accuracy with fallback model: {val_acc:.5f}")




## === cell 3
all_test_paths = glob.glob(os.path.join(TEST_DIR, "**/*.jpg"), recursive=True)
test_files = sorted([os.path.relpath(p, TEST_DIR) for p in all_test_paths])
test_paths = [os.path.join(TEST_DIR, f) for f in test_files]

print("Extracting features for test set...")
X_test = build_feature_matrix(test_paths)

test_pred_indices = rf_clf.predict(X_test)
pred_labels = le.inverse_transform(test_pred_indices)




## === cell 4
submission = pd.DataFrame({"image_id": test_files, "label": pred_labels})
submission_path = "model_submission_v22.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
