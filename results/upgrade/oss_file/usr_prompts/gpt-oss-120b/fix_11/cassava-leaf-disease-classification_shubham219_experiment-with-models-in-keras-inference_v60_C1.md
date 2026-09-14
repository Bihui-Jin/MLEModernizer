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

0.8115744938047749

# 6. Current score

0.56988

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10277) has done: 'The changes add a protobuf compatibility fix, correctly locate the dataset directory, ensure the training/validation split runs, unfreeze the EfficientNet backbone for better accuracy, and train for a few more epochs. The script now ends with a proper `submission.csv` file ready for Kaggle.'
- What this solution (achieved 0.54111) has done: 'I guard the TensorFlow import (the protobuf error) and replace the TF‑based training with a lightweight scikit‑learn model that works on down‑scaled image pixels. This fixes the import crash, resolves the datatype issue in `flow_from_dataframe`, and still provides a reasonable accuracy boost while keeping the overall pipeline (data loading, split, training, prediction, submission) intact.'
- What this solution (achieved 0.56241) has done: 'The fixes remove the problematic TensorFlow import, increase the image resolution to 64 × 64 for richer features, add feature scaling, and replace the simple Logistic Regression with a small multi‑layer perceptron (MLP) that can capture non‑linear patterns – all while keeping the original data‑handling pipeline unchanged. These changes resolve the runtime error, improve model capacity, and are expected to raise validation accuracy toward the target score. The script now reliably writes a correctly formatted `submission.csv`.'
- What this solution (achieved 0.56988) has done: 'The changes switch image loading from a process pool to a thread pool (Pillow releases the GIL, so threads are faster and avoid the heavy pickling overhead), increase the worker count to fully use the CPU, and cache the thread‑pool executor to reuse its threads. The rest of the workflow—including the exact image size, scaling, train/validation split, MLP architecture, and prediction steps—remains unchanged, preserving result accuracy while considerably cutting the total runtime.'

# 9. Code solution

## === cell 0
import os, glob, json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor  # use threads for Pillow I/O

SEED = 42
np.random.seed(SEED)




## === cell 1
candidates = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "./data/cassava-leaf-disease-classification",
    "./cassava-leaf-disease-classification",
]
for cand in candidates:
    if os.path.isdir(cand):
        base_path = cand
        break
else:
    raise FileNotFoundError("Base data directory not found among candidates.")

train_img_dir = os.path.join(base_path, "train_images")
test_img_dir = os.path.join(base_path, "test_images")
train_csv_path = os.path.join(base_path, "train.csv")
sample_submission_path = os.path.join(base_path, "sample_submission.csv")

train_df_raw = pd.read_csv(train_csv_path)
train_df_raw["path"] = train_df_raw["image_id"].apply(
    lambda x: os.path.join(train_img_dir, x)
)

test_image_paths = glob.glob(os.path.join(test_img_dir, "*.jpg"))
df_test = pd.DataFrame({"path": test_image_paths})




## === cell 2
IMG_SIZE = 128


def _load_image(p):
    """Load a single image, resize, and flatten to float32."""
    img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.float32).ravel()


def load_and_flatten(paths):
    """Load images in parallel (threads), resize, and flatten."""
    n = len(paths)
    flat = np.empty((n, IMG_SIZE * IMG_SIZE * 3), dtype=np.float32)  # pre‑allocate
    max_workers = max(1, os.cpu_count() or 4)  # fully utilize CPUs
    with ThreadPoolExecutor(max_workers=max_workers) as exec:
        for i, arr in enumerate(exec.map(_load_image, paths, chunksize=32)):
            flat[i] = arr
    return flat


all_train_paths = train_df_raw["path"].values
all_train_X = load_and_flatten(all_train_paths)
all_train_y = train_df_raw["label"].values

train_X, val_X, train_y, val_y = train_test_split(
    all_train_X,
    all_train_y,
    test_size=0.1,
    stratify=all_train_y,
    random_state=SEED,
)




## === cell 3
mean_ = train_X.mean(axis=0, keepdims=True)
std_ = train_X.std(axis=0, keepdims=True)
std_[std_ == 0] = 1.0

train_X_scaled = (train_X - mean_) / std_
val_X_scaled = (val_X - mean_) / std_

clf = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation="relu",
    solver="adam",
    batch_size=128,
    max_iter=200,
    early_stopping=True,
    n_iter_no_change=10,
    random_state=SEED,
)

clf.fit(train_X_scaled, train_y)

val_pred = clf.predict(val_X_scaled)
val_acc = accuracy_score(val_y, val_pred)
print(f"Validation accuracy: {val_acc:.4f}")




## === cell 4
test_X = load_and_flatten(df_test["path"].values)
test_X_scaled = (test_X - mean_) / std_

test_pred = clf.predict(test_X_scaled)

submission = pd.DataFrame(
    {
        "image_id": df_test["path"].apply(lambda x: os.path.basename(x)),
        "label": test_pred,
    }
)
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
