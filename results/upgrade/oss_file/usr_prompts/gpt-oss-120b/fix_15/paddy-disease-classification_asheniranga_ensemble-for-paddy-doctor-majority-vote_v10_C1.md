# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from PIL import Image
import concurrent.futures

np.random.seed(42)




## === cell 1
BASE_INPUT = "/kaggle/input/paddy-disease-classification"
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUBMISSION = os.path.join(BASE_INPUT, "sample_submission.csv")

IMG_SIZE = (112, 112)  # was (96, 96)


def load_and_preprocess(img_path):
    """Load an image, resize, normalize to [0,1], and flatten."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize
    return arr.reshape(-1)  # flatten


train_df = pd.read_csv(TRAIN_CSV)

label_names = sorted(train_df["label"].unique())
label_to_idx = {name: idx for idx, name in enumerate(label_names)}
idx_to_label = {idx: name for name, idx in label_to_idx.items()}

variety_names = sorted(train_df["variety"].unique())
variety_encoder = OneHotEncoder(
    categories=[variety_names], sparse=False, handle_unknown="ignore"
)
variety_encoder.fit(train_df[["variety"]])

train_labels = train_df["label"].map(label_to_idx).to_numpy(dtype=np.int32)
train_variety_onehot = variety_encoder.transform(
    train_df[["variety"]]
)  # (n_samples, n_varieties)
train_ages = train_df["age"].to_numpy(dtype=np.float32).reshape(-1, 1)


def build_img_path(row):
    return os.path.join(TRAIN_IMG_DIR, row["label"], row["image_id"])


img_paths = [build_img_path(row) for _, row in train_df.iterrows()]

valid_indices = [i for i, p in enumerate(img_paths) if os.path.exists(p)]
valid_img_paths = [img_paths[i] for i in valid_indices]

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    train_images = list(executor.map(load_and_preprocess, valid_img_paths))

train_labels = train_labels[valid_indices]
train_variety_onehot = train_variety_onehot[valid_indices]
train_ages = train_ages[valid_indices]

X_img = np.stack(train_images, axis=0)  # (n_samples, H*W*3)

X = np.concatenate([X_img, train_variety_onehot, train_ages], axis=1)
y = train_labels




## === cell 2
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

scaler = StandardScaler()
mlp = MLPClassifier(
    hidden_layer_sizes=(2048, 1024, 512),  # increased capacity
    activation="relu",
    solver="adam",
    max_iter=1000,  # train longer
    early_stopping=False,  # allow full training iterations
    random_state=42,
    verbose=False,
)

model = make_pipeline(scaler, mlp)

model.fit(X_train, y_train)

val_preds = model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation accuracy: {val_acc:.4f}")




## === cell 3
test_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")])
test_paths = [os.path.join(TEST_IMG_DIR, f) for f in test_files]

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    test_images = list(executor.map(load_and_preprocess, test_paths))

test_images = np.stack(test_images, axis=0)

meta_dim = train_variety_onehot.shape[1] + train_ages.shape[1]

test_meta_zeros = np.zeros((test_images.shape[0], meta_dim), dtype=np.float32)

X_test = np.concatenate([test_images, test_meta_zeros], axis=1)




## === cell 4
test_pred_idxs = model.predict(X_test)
test_pred_labels = [idx_to_label[idx] for idx in test_pred_idxs]

submission = pd.DataFrame({"image_id": test_files, "label": test_pred_labels})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")
