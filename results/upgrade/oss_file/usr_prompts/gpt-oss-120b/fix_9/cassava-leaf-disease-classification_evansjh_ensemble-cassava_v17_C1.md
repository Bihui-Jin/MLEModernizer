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

3.13

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

0.8964944091870656

# 6. Current score

0.6151

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I replaced the failing imports and removed references to unavailable model files. The script now reads the training labels, finds the most frequent class, and assigns that label to every test image to create a valid `submission.csv` file.'
- What this solution (achieved 0.65508) has done: 'I replace the constant‑label baseline with a tiny TensorFlow CNN that learns from the training images, then use it to predict labels for the test set. The added model is lightweight (few layers, short training) so it runs quickly, and the rest of the workflow (reading CSVs, building the submission file) stays the same, moving the accuracy closer to the target score.'
- What this solution (achieved 0.57324) has done: 'I set the protobuf implementation environment variable before importing TensorFlow to stop the `MessageFactory` error, and increase the training epochs slightly to boost accuracy while keeping the overall pipeline unchanged. The rest of the code stays the same, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.51158) has done: 'I replace the failing TensorFlow pipeline with a lightweight, pure‑Python image classifier that uses Pillow to resize images and scikit‑learn’s LogisticRegression for training. This eliminates the protobuf import error, keeps the overall workflow (loading CSVs, building a submission file) unchanged, and should raise the validation accuracy from ~0.57 toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.61472) has done: 'I increase the image resolution to capture more detail and replace the simple LogisticRegression with a lightweight multilayer perceptron (MLP) that can model non‑linear patterns while keeping the overall pipeline unchanged. These tweaks are expected to raise validation accuracy toward the target without altering the submission logic.'
- What this solution (achieved 0.62145) has done: 'I replace the low‑capacity MLP with a stronger RandomForest classifier (allowed because the performance gap exceeds 30 %). This change keeps the overall data‑loading, preprocessing, and submission logic untouched while giving the model more expressive power, which should raise validation accuracy toward the target. I also update the imports and classifier‑initialisation accordingly.'
- What this solution (achieved 0.56839) has done: 'I replace the RandomForest with a very small TensorFlow convolutional neural network, keeping the same image‑loading code and the overall workflow (train/val split, CSV handling, submission write). The CNN better captures visual patterns, so validation accuracy should improve from ~0.62 toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.6151) has done: 'I removed the TensorFlow dependency that was causing the protobuf import error and replaced the CNN with a scikit‑learn RandomForest classifier, keeping the same image loading and preprocessing steps. The data is kept as flattened float arrays, split into train/validation sets, the model is trained, validation accuracy is reported, and predictions on the test set are written to a proper `submission.csv`. This fixes the runtime crash and should raise the validation score toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
submission_path = "/kaggle/working/submission.csv"

train_df = pd.read_csv(train_csv_path)

IMG_SIZE = (96, 96)  # modest size for speed
SEED = 42
np.random.seed(SEED)




## === cell 1
def load_and_preprocess_image(img_path: str) -> np.ndarray:
    """Read an image, resize to IMG_SIZE, and return a flat float32 array."""
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize
    return arr.flatten()


train_image_paths = [
    os.path.join(train_image_dir, fname) for fname in train_df["image_id"]
]
train_labels = train_df["label"].values.astype(np.int32)

print("Loading and preprocessing training images...")
X_full = np.stack([load_and_preprocess_image(p) for p in train_image_paths])
y_full = train_labels

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.1, random_state=SEED, stratify=y_full
)

print("Training RandomForest classifier...")
rf_clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    n_jobs=-1,
    random_state=SEED,
    class_weight="balanced",
)
rf_clf.fit(X_train, y_train)

val_pred = rf_clf.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation accuracy: {val_acc:.5f}")



## === cell 2
test_image_ids = [
    fname
    for fname in os.listdir(test_image_dir)
    if fname.lower().endswith((".jpg", ".jpeg", ".png"))
]
test_paths = [os.path.join(test_image_dir, img_id) for img_id in test_image_ids]

print("Loading and preprocessing test images...")
X_test = np.stack([load_and_preprocess_image(p) for p in test_paths])

print("Predicting test labels...")
test_pred = rf_clf.predict(X_test).astype(int)

submission_df = pd.DataFrame({"image_id": test_image_ids, "label": test_pred})
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with {len(submission_df)} rows.")
