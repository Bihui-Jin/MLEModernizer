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

3.11

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

0.7263523723179208

# 6. Current score

0.61472

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.13827) has done: 'I removed the problematic `tensorflow_hub` import, fixed the incorrect weight file path by building a ResNet‑50 model directly in the notebook, and corrected the data paths for train and test images. The script now loads the training CSV, creates image generators with proper preprocessing, trains a lightweight model, runs predictions on the test set, and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.14948) has done: 'The changes add multiprocessing workers and larger queues to the image data generators and model fitting, which speeds up disk I/O and preprocessing without altering the model architecture, training schedule, or evaluation logic.'
- What this solution (achieved 0.07698) has done: 'I added the missing `train_test_split` import, defined shared constants (`NUM_CLASSES`, `AUTOTUNE`, `preprocess_fn`) early so they are available to all cells, and moved the TensorFlow setup into the first cell. I also made the ResNet‑50 backbone trainable (a small change that usually boosts accuracy) and kept the rest of the pipeline unchanged. The script now runs end‑to‑end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.62108) has done: 'I replace the broken TensorFlow pipeline with a lightweight scikit‑learn image classifier. The new code loads and resizes images using Pillow, trains a RandomForest on the flattened pixel values, evaluates on a validation split, predicts test labels, and writes a correct `submission.csv`. This fixes the import and API errors and should raise the accuracy toward the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.61472) has done: 'The changes focus on speeding up image loading and reducing memory usage: the preprocessing now loads images in parallel using a thread pool, stores them as compact uint8 arrays (avoiding costly float scaling), and pre‑allocates the final array to eliminate list‑to‑array copying. This preserves the exact RandomForest model and training pipeline while cutting I/O and memory overhead, keeping the results identical apart from negligible floating‑point differences.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

SEED = 42
np.random.seed(SEED)

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUBMIT = os.path.join(BASE_PATH, "sample_submission.csv")

df_train = pd.read_csv(TRAIN_CSV)
df_train["label"] = df_train["label"].astype(int)
df_train["path"] = df_train["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))




## === cell 1
IMG_SIZE = (128, 128)  # increased from 64x64

import concurrent.futures


def _load_image(path):
    """Load a single image, resize, convert to RGB, and return a flattened uint8 array."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, Image.BILINEAR)
        return np.array(img, dtype=np.uint8).ravel()  # keep as uint8, no scaling


def load_and_preprocess(paths):
    """
    Load images in parallel, resize to IMG_SIZE,
    and return an array of shape (n_samples, height*width*3) of type uint8.
    """
    n_samples = len(paths)
    flat_size = IMG_SIZE[0] * IMG_SIZE[1] * 3
    result = np.empty((n_samples, flat_size), dtype=np.uint8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for idx, arr in enumerate(executor.map(_load_image, paths)):
            result[idx] = arr
    return result




## === cell 2
X = load_and_preprocess(df_train["path"].values)
y = df_train["label"].values

X = np.ascontiguousarray(X)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=SEED
)

rf = RandomForestClassifier(
    n_estimators=500,  # more trees for better performance
    max_depth=None,
    n_jobs=5,
    random_state=SEED,
    class_weight="balanced",  # help under‑represented classes
    verbose=0,
)
rf.fit(X_train, y_train)

val_pred = rf.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation accuracy: {val_acc:.5f}")




## === cell 3
test_image_paths = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
df_test = pd.DataFrame(test_image_paths, columns=["path"])

X_test = load_and_preprocess(df_test["path"].values)
X_test = np.ascontiguousarray(X_test)
test_pred = rf.predict(X_test)




## === cell 4
submission = pd.DataFrame(
    {
        "image_id": df_test["path"].apply(lambda p: os.path.basename(p)),
        "label": test_pred.astype(int),
    }
)

submission.to_csv("submission.csv", index=False)
print("Submission file saved to submission.csv")
submission.head()
