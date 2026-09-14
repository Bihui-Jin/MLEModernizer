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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8821396192203083

# 6. Current score

0.61286

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The script had several import errors, missing variable definitions, and relied on unavailable model files, which prevented it from running and creating a submission. I fixed the imports, ensured all libraries are loaded, added safeguards around model loading, and implemented a simple fallback that predicts the most common class from the training set when the pretrained models cannot be loaded. This guarantees the pipeline runs end‑to‑end and writes a valid `submission.csv` file.'
- What this solution (achieved 0.18386) has done: 'Implemented a lightweight centroid‑based classifier to replace the failing Keras model loading.  
The script now:
1. Loads training data and label map as before.  
2. Computes per‑class mean image vectors (64 × 64 RGB) from the training images.  
3. Uses these centroids for prediction: each test image is assigned the label of the nearest centroid (Euclidean distance).  
4. Falls back to the most common label only if centroid computation fails.  
5. Writes a proper `submission.csv` file.'
- What this solution (achieved 0.61697) has done: 'Implemented a lightweight logistic‑regression model on L2‑normalized flattened images (64×64 RGB).  
The script now:
1. Loads and preprocesses training images into normalized vectors.  
2. Trains a multinomial LogisticRegression classifier (fast, deterministic).  
3. Predicts test images using the trained model, falling back to the most common label if an image fails to load.  
4. Writes a valid `submission.csv` matching the required format.  

These changes keep the overall data‑handling flow while replacing the low‑accuracy centroid rule with a more expressive linear model, moving the score substantially toward the target.'
- What this solution (achieved 0.61099) has done: 'The script is slowed mainly by the very large feature matrix (128×128 images) and the overhead of a process pool for image loading. By shrinking the resize resolution to 64 × 64 we cut memory‑ and compute‑requirements ≈ 4× while keeping the exact same feature‑extraction logic. Switching to a thread pool eliminates costly process startup and memory copying, speeding up the I/O‑bound image loading without altering results.'
- What this solution (achieved 0.6151) has done: 'I replace the logistic‑regression model with a modest RandomForest classifier (more expressive for pixel‑level features) and enrich the image feature vector with per‑channel standard‑deviation values. These small, targeted changes keep the overall pipeline intact while providing a higher‑capacity model expected to raise the accuracy toward the target score.'
- What this solution (achieved 0.61286) has done: 'I keep the overall pipeline unchanged but enrich the image representation and make the RandomForest a bit more powerful. In the feature builder I add normalized per‑channel histograms (16 bins each) to capture colour distribution, which together with the existing pixel vector, mean and std gives the model more discriminative information. I also increase the forest size to 400 trees and enable `class_weight='balanced'` so minority classes are treated better. These modest, targeted tweaks are expected to raise the validation accuracy toward the target while preserving the original logic and output format.'

# 9. Code solution

## === cell 0
import os
import json
import pandas as pd
import numpy as np
import cv2
from sklearn.ensemble import (
    RandomForestClassifier,
)  # switched to a more expressive model
import concurrent.futures  # parallel processing


INPUT_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_DIR = os.path.join(INPUT_DIR, "test_images")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
LABEL_MAP_JSON = os.path.join(INPUT_DIR, "label_num_to_disease_map.json")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")




## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
print(f"Loaded {len(train_df)} training rows")




## === cell 2
with open(LABEL_MAP_JSON, "r") as f:
    label_map = json.load(f)

train_df["class_name"] = train_df["label"].astype(str).map(label_map)




## === cell 3
most_common_label = train_df["label"].mode()[0]
print(f"Most common label (fallback): {most_common_label}")




## === cell 4
IMG_SIZE = 64
HIST_BINS = 16  # additional colour‑histogram features per channel


def _load_and_features(img_path):
    """
    Load an image, resize, L2‑normalize the flattened pixels,
    concatenate mean‑RGB, std‑RGB and per‑channel histogram features.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_f = img.astype(np.float32)

    mean_rgb = img_f.mean(axis=(0, 1))  # (3,)
    std_rgb = img_f.std(axis=(0, 1))  # (3,)

    flat = img_f.reshape(-1)
    norm = np.linalg.norm(flat)
    if norm > 0:
        flat = flat / norm

    hist_features = []
    for c in range(3):
        hist, _ = np.histogram(
            img_f[:, :, c],
            bins=HIST_BINS,
            range=(0, 255),
            density=True,
        )
        hist_features.append(hist.astype(np.float32))
    hist_features = np.concatenate(hist_features)  # shape (48,)

    features = np.concatenate([flat, mean_rgb, std_rgb, hist_features])
    return features




## === cell 5
def _process_row(idx_img_label):
    """Load image and return (idx, features, label)."""
    idx, img_name, label = idx_img_label
    img_path = os.path.join(TRAIN_IMG_DIR, img_name)
    feats = _load_and_features(img_path)
    return (idx, feats, label)


rows = [(i, row.image_id, row.label) for i, row in enumerate(train_df.itertuples())]

sample_feat = _load_and_features(os.path.join(TRAIN_IMG_DIR, rows[0][1]))
if sample_feat is None:
    raise RuntimeError("Failed to load the first training image.")
feat_len = sample_feat.shape[0]

X_train = np.empty((len(rows), feat_len), dtype=np.float32)
y_train = np.empty(len(rows), dtype=np.int32)

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    for idx, feats, label in executor.map(_process_row, rows, chunksize=32):
        if feats is not None:
            X_train[idx] = feats
            y_train[idx] = label
        else:
            X_train[idx] = np.nan
            y_train[idx] = -1

valid_mask = ~np.isnan(X_train).any(axis=1)
X_train = X_train[valid_mask]
y_train = y_train[valid_mask]

print(f"Prepared training matrix with shape {X_train.shape}")




## === cell 6
rf_clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
    verbose=0,
)
rf_clf.fit(X_train, y_train)
print("RandomForest model trained.")




## === cell 7
test_images = sorted(
    [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".png"))]
)
print(f"Found {len(test_images)} test images")




## === cell 8
def predict_image(img_path):
    """Predict label using the trained RandomForest model."""
    feats = _load_and_features(img_path)
    if feats is None:
        return int(most_common_label)
    pred = rf_clf.predict(feats.reshape(1, -1))
    return int(pred[0])




## === cell 9
test_paths = [os.path.join(TEST_DIR, name) for name in test_images]

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    predicted_labels = list(executor.map(predict_image, test_paths))




## === cell 10
submission_df = pd.DataFrame({"image_id": test_images, "label": predicted_labels})
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file written to {SUBMISSION_PATH}")
