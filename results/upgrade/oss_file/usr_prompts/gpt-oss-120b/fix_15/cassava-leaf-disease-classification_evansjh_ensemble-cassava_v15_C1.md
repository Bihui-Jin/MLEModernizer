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

0.8987609549712904

# 6. Current score

0.61659

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the failing TensorFlow Hub and model‑loading code, replace it with a simple baseline that predicts the most frequent class from the training labels for every test image, and ensure the script writes a correctly‑named `submission.csv`. This fixes the import error, missing‑file errors, and undefined‑variable errors while still producing a valid Kaggle submission file.'
- What this solution (achieved 0.27317) has done: 'The script be sped up by replacing the heavyweight process‑based pools with lightweight thread‑based executors (image I/O releases the GIL, so multithreading is sufficient) and by removing the unused fallback argument. The logistic regression is run with a single thread to avoid CPU oversubscription, which is safe because the model is tiny. All changes keep the exact feature‑extraction logic and prediction pipeline, so the results remain identical.'
- What this solution (achieved 0.28662) has done: 'The changes replace heavyweight process‑based parallelism with lightweight threading (I/O‑bound image loading) and use Pillow’s native `ImageStat` which computes per‑channel statistics in C, eliminating the costly NumPy conversions. This dramatically reduces feature‑extraction time while keeping the exact 12‑dimensional statistics, so model training and predictions remain identical.'
- What this solution (achieved 0.61659) has done: 'I replace the simple logistic regression with a richer RandomForest model and expand the image features to include a small flattened RGB thumbnail (16 × 16) in addition to the existing per‑channel statistics. This adds discriminative information while keeping the overall pipeline lightweight, and it should raise the validation accuracy toward the target score. The rest of the code and file handling remain unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image, ImageStat  # use ImageStat for fast per‑channel stats
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import (
    ThreadPoolExecutor,
)  # lightweight threads for I/O‑bound work

np.random.seed(42)



## === cell 1
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
submission_path = "/kaggle/working/submission.csv"



## === cell 2
train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype(int)




## === cell 3
def _extract_features(img_path: str) -> np.ndarray:
    """Return a feature vector that combines:
    - per‑channel mean, std, min, max (12 values)
    - a flattened 16×16 RGB thumbnail (16*16*3 = 768 values)
    Total length = 780."""
    if not os.path.isfile(img_path):
        return np.zeros(780, dtype=np.float32)
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        stat = ImageStat.Stat(im)
        mean = np.array(stat.mean, dtype=np.float32)
        std = np.sqrt(np.array(stat.var, dtype=np.float32))
        mins = np.array([ext[0] for ext in stat.extrema], dtype=np.float32)
        maxs = np.array([ext[1] for ext in stat.extrema], dtype=np.float32)
        stats_feat = np.concatenate([mean, std, mins, maxs])  # shape (12,)

        thumb = im.resize((16, 16), Image.BILINEAR)
        thumb_arr = (
            np.asarray(thumb, dtype=np.float32).reshape(-1) / 255.0
        )  # shape (768,)

        return np.concatenate([stats_feat, thumb_arr])  # shape (780,)


train_image_paths = [
    os.path.join(train_image_dir, fn) for fn in train_df["image_id"].values
]

max_workers = min(32, os.cpu_count() or 1)

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    train_features = np.empty((len(train_image_paths), 780), dtype=np.float32)
    for idx, feat in enumerate(
        executor.map(_extract_features, train_image_paths, chunksize=64)
    ):
        train_features[idx] = feat

X_train = train_features  # shape (n_samples, 780)
y_train = train_df["label"].values.astype(int)



## === cell 4
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,  # utilize all cores safely
)
model.fit(X_train, y_train)




## === cell 5
def _extract_test_features(img_name: str) -> np.ndarray:
    img_path = os.path.join(test_image_dir, img_name)
    return _extract_features(img_path)


test_files = [
    entry.name
    for entry in os.scandir(test_image_dir)
    if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png"))
]

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    test_features = np.empty((len(test_files), 780), dtype=np.float32)
    for idx, feat in enumerate(
        executor.map(_extract_test_features, test_files, chunksize=64)
    ):
        test_features[idx] = feat

X_test = test_features  # shape (n_test, 780)
test_preds = model.predict(X_test)



## === cell 6
submission_df = pd.DataFrame({"image_id": test_files, "label": test_preds.astype(int)})
submission_df.to_csv(submission_path, index=False)

print("Submission file created at:", submission_path)
print(submission_df.head())
