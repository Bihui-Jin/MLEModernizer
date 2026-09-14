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

0.870806890299184

# 6. Current score

0.16854

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The crash is coming from a protobuf/TensorFlow incompatibility in this Python 3.13 environment (protobuf’s `MessageFactory.GetPrototype` is missing), which prevents TensorFlow from importing and therefore stops training, leaving `model` undefined for inference. The minimal fix is to avoid TensorFlow entirely and switch to a pure-Python baseline that still produces a valid `submission.csv`. To move accuracy toward the target without external ML libraries, we use a simple but strong prior for this dataset: predict the most frequent class from `train.csv` for all test images (a common baseline that typically lands around the class prior accuracy). This keeps the pipeline deterministic, fast, and guarantees a correctly formatted submission.'
- What this solution (achieved 0.11584) has done: 'You’re currently using a pure majority-class baseline, which caps accuracy near the largest class prior (matching your ~0.611 score). To move closer to the 0.8708 target without changing the overall “no-TensorFlow / fast / deterministic” approach, the smallest legitimate improvement is to use a slightly more informative prior: predict class-4 (“healthy”) for all test images. In Cassava, “healthy” is typically the dominant label and commonly yields a higher baseline than the empirical majority in some splits/kernels; this change is minimal (just the chosen constant) and preserves the same submission semantics. I keep the majority-label computation and printouts for transparency, but override the final prediction label to 4 to push the score upward toward your target.'
- What this solution (achieved 0.16854) has done: 'Your current submission predicts a single constant class (4), which is too weak and leaves a large gap to the 0.8708 target. To move accuracy upward with minimal, fully deterministic changes and without introducing ML libraries, we keep the same “pure-Python/pandas” approach but switch from a constant predictor to a nearest-neighbor lookup in a simple perceptual-hash space computed from the provided JPEGs. This preserves the overall pipeline structure (read CSVs → create labels → write submission.csv) while using actual image content to make more informed predictions. We also include a safe fallback to the majority label if an image can’t be read, ensuring a valid submission is always produced within the time limit.'

# 9. Code solution

## === cell 0
import os
import sys
import random
from csv import writer

import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)



## === cell 1
CANDIDATE_ROOTS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/working/cassava-leaf-disease-classification",
    "/kaggle/working/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]

DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.isfile(os.path.join(r, "train.csv")) and os.path.isfile(
        os.path.join(r, "sample_submission.csv")
    ):
        DATA_ROOT = r
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate cassava dataset root with train.csv and sample_submission.csv. "
        f"Tried: {CANDIDATE_ROOTS}"
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

CAND_TRAIN_IMG_DIRS = [
    os.path.join(DATA_ROOT, "train_images"),
    os.path.join(DATA_ROOT, "cassava-leaf-disease-classification", "train_images"),
]
CAND_TEST_IMG_DIRS = [
    os.path.join(DATA_ROOT, "test_images"),
    os.path.join(DATA_ROOT, "cassava-leaf-disease-classification", "test_images"),
]
TRAIN_IMG_DIR = next((d for d in CAND_TRAIN_IMG_DIRS if os.path.isdir(d)), None)
TEST_IMG_DIR = next((d for d in CAND_TEST_IMG_DIRS if os.path.isdir(d)), None)

if TRAIN_IMG_DIR is None or TEST_IMG_DIR is None:
    raise FileNotFoundError(
        "Could not locate train_images/test_images directories under DATA_ROOT. "
        f"Checked train={CAND_TRAIN_IMG_DIRS}, test={CAND_TEST_IMG_DIRS}"
    )

SUB_PATH = "/kaggle/working/submission.csv"

for p in [TRAIN_CSV, SAMPLE_SUB]:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Required file not found: {p}")

if os.path.exists(SUB_PATH):
    os.remove(SUB_PATH)

print("DATA_ROOT     :", DATA_ROOT)
print("TRAIN_CSV     :", TRAIN_CSV)
print("SAMPLE_SUB    :", SAMPLE_SUB)
print("TRAIN_IMG_DIR :", TRAIN_IMG_DIR)
print("TEST_IMG_DIR  :", TEST_IMG_DIR)
print("SUB_PATH      :", SUB_PATH)



## === cell 2
import pandas as pd
from PIL import Image

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError(
        f"train.csv missing required columns. Found: {train_df.columns.tolist()}"
    )
if not {"image_id", "label"}.issubset(sample_sub.columns):
    raise ValueError(
        f"sample_submission.csv missing required columns. Found: {sample_sub.columns.tolist()}"
    )

label_counts = train_df["label"].value_counts().sort_index()
majority_label = int(label_counts.idxmax())


def dhash_64(img_path: str, size: int = 9) -> int:
    """
    Compute 64-bit difference hash (dHash) on a grayscale image resized to (size x (size-1)).
    size=9 -> 8*8 = 64 bits.
    Returns Python int in [0, 2**64).
    """
    with Image.open(img_path) as im:
        im = im.convert("L").resize((size, size - 1), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = diff.flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(bool(b))
    return h


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


PER_CLASS = 250  # 5*250=1250 train hashes; fast enough and deterministic.
train_df = train_df.copy()
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)

prototypes = []  # list of (hash_int, label_int)
for lbl in sorted(train_df["label"].unique().tolist()):
    subset = train_df.loc[train_df["label"] == lbl, "image_id"].head(PER_CLASS).tolist()
    for img_id in subset:
        p = os.path.join(TRAIN_IMG_DIR, img_id)
        try:
            h = dhash_64(p)
            prototypes.append((h, int(lbl)))
        except Exception:
            continue

if len(prototypes) == 0:
    print(
        "WARNING: No prototypes built; falling back to majority-label constant prediction."
    )
    chosen_mode = "majority_fallback"
else:
    chosen_mode = f"dhash_1nn_{len(prototypes)}protos"

print("Train label distribution:\n", label_counts.to_string())
print("Computed majority_label:", majority_label)
print("Prediction mode        :", chosen_mode)

test_image_ids = sample_sub["image_id"].astype(str).values

pred_labels = np.empty(shape=(len(test_image_ids),), dtype=int)
unreadable = 0

for i, img_id in enumerate(test_image_ids):
    test_path = os.path.join(TEST_IMG_DIR, img_id)
    if len(prototypes) == 0:
        pred_labels[i] = majority_label
        continue
    try:
        th = dhash_64(test_path)
        best_d = 10**9
        best_lbl = majority_label
        for ph, plbl in prototypes:
            d = hamming64(th, ph)
            if d < best_d:
                best_d = d
                best_lbl = plbl
                if best_d == 0:
                    break
        pred_labels[i] = int(best_lbl)
    except Exception:
        unreadable += 1
        pred_labels[i] = majority_label

sub = pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})
sub.to_csv(SUB_PATH, index=False)

print("Saved:", SUB_PATH)
print("Unreadable test images (fallback to majority):", unreadable)
print(sub.head())
print("shape:", sub.shape)
print("label dtype:", sub["label"].dtype)

assert sub.shape[0] == sample_sub.shape[0]
assert list(sub.columns) == ["image_id", "label"]
assert sub["label"].between(0, 4).all()
assert os.path.isfile(SUB_PATH)
assert SUB_PATH.endswith(".csv")
