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

0.8948322756119673

# 6. Current score

0.61398

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'Implemented a lightweight fix that removes the failing TensorFlow pipeline and replaces it with a simple baseline: the model now predicts the most frequent disease label from the training set for every test image. This resolves import errors, eliminates undefined‑variable issues, and guarantees creation of a correctly formatted `submission.csv` in the expected location.'
- What this solution (achieved 0.61099) has done: 'The fix removes the heavy in‑memory cache and shrinks the shuffle buffer, which cuts RAM pressure and reduces dataset‑pipeline overhead while keeping the same image preprocessing, model, and training epochs. The batch size is increased to 128 to halve the number of training steps without affecting accuracy. These changes keep the core logic identical but make data loading and training fast enough to finish well under the 600‑second limit.'
- What this solution (achieved 0.61099) has done: 'The changes add dataset caching and increase the batch size so that images are read and processed only once and fewer training steps are required, dramatically cutting I/O and CPU time while keeping the same model, training epochs, and evaluation logic.'
- What this solution (achieved 0.61099) has done: 'We add dataset caching so that after the first epoch the already‑decoded and resized images are kept in memory, eliminating the costly file‑I/O and JPEG decoding on the second epoch. This change preserves the exact training logic (same model, epochs, batch size) while cutting the total runtime enough to stay under the 600‑second limit.'
- What this solution (achieved 0.61472) has done: 'The fix removes the failing TensorFlow import, forces the fallback RandomForest branch, and guarantees that `test_filenames` and `pred_labels` are always defined. After training or fallback, the submission DataFrame is created and written, ensuring a valid `submission.csv` is produced.'
- What this solution (achieved 0.61398) has done: 'I strengthen the RandomForest baseline by using the full training set, extracting richer 64×64 pixel features (instead of 32×32), and increasing the forest size slightly. These tweaks keep the same overall logic but give the model more informative inputs, which should raise the validation accuracy toward the target while still producing a correct `submission.csv`.'

# 9. Code solution

## === cell 0
import os, json
import pandas as pd
import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
SUBMISSION_PATH = "/kaggle/working/submission.csv"

train_df = pd.read_csv(TRAIN_CSV)

most_common_label = int(train_df["label"].mode()[0])




## === cell 1
tf_available = False

if tf_available:
    pass
else:
    from PIL import Image
    from sklearn.ensemble import RandomForestClassifier

    np.random.seed(42)

    train_sample = train_df.copy()

    train_paths = (
        train_sample["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x)).values
    )
    train_labels_np = train_sample["label"].astype(np.int32).values

    def _extract_features(path, sz=64):
        """Resize to sz×sz and flatten RGB values normalized to [0,1]."""
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((sz, sz))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return arr.ravel()
        except Exception:
            return np.zeros(sz * sz * 3, dtype=np.float32)

    print("Extracting training features …")
    train_features = np.vstack([_extract_features(p) for p in train_paths])

    print("Training RandomForest …")
    rf = RandomForestClassifier(
        n_estimators=400,  # slightly larger forest for better capacity
        max_depth=None,
        n_jobs=5,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(train_features, train_labels_np)

    test_filenames = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")]
    test_filenames.sort()
    test_filepaths = [os.path.join(TEST_IMG_DIR, f) for f in test_filenames]

    print("Extracting test features …")
    test_features = np.vstack([_extract_features(p) for p in test_filepaths])

    try:
        pred_labels = rf.predict(test_features).astype(int)
    except Exception as e:
        print("RandomForest prediction failed:", e)
        pred_labels = np.full(len(test_filenames), most_common_label, dtype=int)




## === cell 2
if "test_filenames" not in locals():
    test_filenames = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")]
    test_filenames.sort()
if "pred_labels" not in locals():
    pred_labels = np.full(len(test_filenames), most_common_label, dtype=int)

submission = pd.DataFrame({"image_id": test_filenames, "label": pred_labels})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file created at: {SUBMISSION_PATH}")
print(submission.head())
