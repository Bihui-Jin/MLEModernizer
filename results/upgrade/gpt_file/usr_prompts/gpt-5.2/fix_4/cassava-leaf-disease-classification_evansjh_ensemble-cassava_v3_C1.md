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

0.890601390148081

# 6. Current score

0.13602

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the environment import/runtime crash by avoiding TensorFlow/Keras model loading (which is currently failing due to a protobuf/TensorFlow incompatibility) and instead produce a valid submission using the provided `sample_submission.csv` as the ordered key. I also fix the missing external model paths issue by removing dependencies on `/kaggle/input/combinedmodel3/...` which are not present. To still move accuracy toward the target while keeping logic simple and stable, I generate predictions using the training-label prior (majority-class baseline), which is a legitimate, fast, and usually stronger-than-random approach for this dataset. Finally, I guarantee the submission has the exact required columns, row count, and ordering, and write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.40321) has done: 'To move accuracy up toward your target without changing the overall “no-TF, fast baseline” approach, I replace the single majority-class guess with a slightly stronger label-prior baseline that samples labels according to the training distribution. This keeps the same core logic (using only `train.csv` label statistics and `sample_submission.csv` ordering), still produces a valid `submission.csv`, and typically improves over predicting one constant class on Cassava. I also make the sampling deterministic via a fixed seed for stability across runs. No model/feature extraction/training loop is introduced, so runtime stays very low.'
- What this solution (achieved 0.13602) has done: 'Your current approach samples labels from the training prior, which has an expected accuracy around \(\sum p_k^2\) (roughly ~0.3–0.4 here), so it cannot reach the 0.89 target. To move the score upward while keeping the “no deep model / no TF” core idea, the smallest legitimate improvement is to switch from random prior-sampling to a deterministic “leak-free lookup” based on simple image statistics: compute per-class average RGB (and brightness/std) on the training images, then classify each test image by nearest class prototype. This preserves the lightweight, non-neural approach and runs fast enough by resizing images and limiting features. The submission format and ordering still be taken from `sample_submission.csv`, and we keep deterministic behavior for stability.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image



## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SUBMISSION_PATH = "/kaggle/working/submission.csv"

assert os.path.exists(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample_submission.csv at {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train_images dir at {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test_images dir at {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

required_cols_train = {"image_id", "label"}
required_cols_sub = {"image_id", "label"}
if not required_cols_train.issubset(train_df.columns):
    raise ValueError(
        f"train.csv must contain columns {required_cols_train}, got {train_df.columns.tolist()}"
    )
if not required_cols_sub.issubset(sample_df.columns):
    raise ValueError(
        f"sample_submission.csv must contain columns {required_cols_sub}, got {sample_df.columns.tolist()}"
    )

train_df["label"] = train_df["label"].astype(int)

print("train_df shape:", train_df.shape)
print("sample_df shape:", sample_df.shape)
print("train label distribution:\n", train_df["label"].value_counts().sort_index())



## === cell 2


def extract_features(image_path: str, size=(64, 64)) -> np.ndarray:
    """
    Returns a small feature vector based on global image statistics:
    mean R,G,B + std R,G,B + overall brightness mean/std.
    """
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im = im.resize(size, resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # H,W,3
    ch_mean = arr.reshape(-1, 3).mean(axis=0)
    ch_std = arr.reshape(-1, 3).std(axis=0)
    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).reshape(
        -1
    )
    g_mean = np.array([gray.mean()], dtype=np.float32)
    g_std = np.array([gray.std()], dtype=np.float32)
    feat = np.concatenate([ch_mean, ch_std, g_mean, g_std]).astype(np.float32)  # (8,)
    return feat


labels_sorted = np.sort(train_df["label"].unique())
label_to_idx = {int(l): i for i, l in enumerate(labels_sorted)}

MAX_PER_CLASS = 1200  # small but usually enough for stable prototypes

rng = np.random.default_rng(20240101)
prototypes = np.zeros((len(labels_sorted), 8), dtype=np.float32)
counts = np.zeros((len(labels_sorted),), dtype=np.int32)

for lbl in labels_sorted:
    sub = train_df[train_df["label"] == lbl]["image_id"].to_numpy()
    if len(sub) > MAX_PER_CLASS:
        sub = rng.choice(sub, size=MAX_PER_CLASS, replace=False)
    idx = label_to_idx[int(lbl)]
    acc = np.zeros((8,), dtype=np.float64)
    n = 0
    for image_id in sub:
        pth = os.path.join(TRAIN_IMG_DIR, image_id)
        if not os.path.exists(pth):
            continue
        try:
            f = extract_features(pth)
        except Exception:
            continue
        acc += f.astype(np.float64)
        n += 1
    if n == 0:
        raise RuntimeError(f"No training images could be read for label {lbl}.")
    prototypes[idx] = (acc / n).astype(np.float32)
    counts[idx] = n

print(
    "Built prototypes with per-class counts:",
    {int(l): int(counts[label_to_idx[int(l)]]) for l in labels_sorted},
)
print("Prototype matrix shape:", prototypes.shape)



## === cell 3

submission_df = sample_df[["image_id"]].copy()
preds = np.empty((len(submission_df),), dtype=np.int32)

for i, image_id in enumerate(submission_df["image_id"].to_numpy()):
    pth = os.path.join(TEST_IMG_DIR, image_id)
    if not os.path.exists(pth):
        preds[i] = int(train_df["label"].value_counts().idxmax())
        continue
    try:
        f = extract_features(pth).astype(np.float32)
        d = ((prototypes - f[None, :]) ** 2).sum(axis=1)
        preds[i] = int(labels_sorted[int(np.argmin(d))])
    except Exception:
        preds[i] = int(train_df["label"].value_counts().idxmax())

submission_df["label"] = preds.astype(int)

if submission_df.isna().any().any():
    raise ValueError("Submission contains NaNs.")
if submission_df.shape[0] != sample_df.shape[0]:
    raise ValueError("Submission row count does not match sample_submission.")
if list(submission_df.columns) != ["image_id", "label"]:
    raise ValueError(
        f"Submission columns must be ['image_id','label'], got {submission_df.columns.tolist()}"
    )

submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file saved at: {SUBMISSION_PATH}")
print(submission_df.head())



## === cell 4
submission_df.tail()
