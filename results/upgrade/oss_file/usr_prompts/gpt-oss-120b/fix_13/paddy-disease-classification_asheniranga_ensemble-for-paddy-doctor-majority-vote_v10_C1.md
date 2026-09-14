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

0.71522

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.18025) has done: 'The fix replaces the missing‑model loading with a lightweight, end‑to‑end training pipeline: it builds a small transfer‑learning model (MobileNetV2), trains it on the provided training images, predicts on the test set, and writes a correctly formatted `submission.csv`. All previous cells that tried to load unavailable HDF5 models are removed, and the new cells handle data loading, label mapping, model definition, training, inference, and CSV export.'
- What this solution (achieved 0.57802) has done: 'I fixed the script by removing the failing TensorFlow import and the empty data generators that couldn’t locate images in the nested class folders. Instead, I added a simple image‑loading routine using Pillow, encoded the labels, and trained a lightweight RandomForest classifier on resized 64×64 RGB pixels. The test images are processed the same way, predictions are mapped back to the original label strings, and a correctly formatted `submission.csv` is written. This resolves all runtime errors and yields a valid submission file.'
- What this solution (achieved 0.67563) has done: 'The script now removes the failing TensorFlow imports and replaces the deep‑learning pipeline with a lightweight, pure‑Python approach: it loads images with Pillow, resizes them to 64×64, flattens them, and trains a k‑Nearest Neighbors classifier (k=3). It evaluates on a held‑out validation split, prints the accuracy, predicts the test set, and writes a correctly‑formatted `submission.csv`. This fixes the runtime errors and provides a functional end‑to‑end solution while keeping the overall workflow (image loading, label mapping, training, inference, submission) intact.'
- What this solution (achieved 0.58532) has done: 'I increase the image resolution to capture more visual detail, normalize pixel values to [0, 1] for a more meaningful distance metric, and use a slightly larger k with distance weighting in the K‑Nearest Neighbours model. These modest tweaks keep the overall pipeline (flattened‑pixel KNN) unchanged while expected to raise validation accuracy closer to the target.'
- What this solution (achieved 0.55188) has done: 'I add simple metadata features (variety one‑hot and age) to the flattened image vectors and replace the K‑Nearest Neighbours model with a modest RandomForest classifier, which is allowed because the current score is far from the target. These changes keep the overall image‑loading pipeline intact while providing a stronger learner and a slight boost in validation accuracy, moving the score closer to the target.'
- What this solution (achieved 0.62567) has done: 'I replace the RandomForest with a scaled MLP (a modest neural net) while keeping the same image‑loading and metadata handling. Adding a StandardScaler before the MLP usually improves convergence on high‑dimensional pixel data, so the validation accuracy should rise toward the target without altering the overall pipeline.'
- What this solution (achieved 0.70792) has done: 'Fixed the runtime errors by defining the correct metadata dimensions for the test set (using the training metadata shapes) and ensuring `X_test` is created properly. Increased the MLP training iterations slightly to help improve validation accuracy without altering the core model architecture. The script now runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 0.71522) has done: 'I slightly increase the image resolution to capture richer visual details and give the MLP a bit more capacity and training iterations. These modest adjustments keep the overall pipeline unchanged while likely raising validation accuracy, moving the score closer to the target.'

# 9. Code solution

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

IMG_SIZE = (96, 96)  # was (64, 64)


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
    hidden_layer_sizes=(1024, 512, 256),  # larger capacity
    activation="relu",
    solver="lbfgs",  # fast convergence for this size
    max_iter=300,  # a bit more iterations for better convergence
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
