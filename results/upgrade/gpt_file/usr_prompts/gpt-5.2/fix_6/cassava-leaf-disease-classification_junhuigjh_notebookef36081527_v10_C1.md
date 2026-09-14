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

0.7893623451193714

# 6. Current score

0.13789

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the execution blockers by (1) removing the TFRecord reading path that triggers the protobuf `MessageFactory.GetPrototype` error in this environment and instead reading test images directly from `test_images/`, and (2) ensuring all constants/paths are defined in the first cell so later cells don’t crash with `NameError`. I also make the model loading robust by falling back to a simple majority-class submission if the referenced model file is not available, so a valid `submission.csv` is always produced. These changes preserve the core inference logic (Keras model predicts argmax over 5 classes) while making the pipeline run end-to-end and generate a correctly formatted submission.'
- What this solution (achieved 0.61099) has done: 'I fix the execution-blocking protobuf/Keras import issue by removing the unused `torch` import and delaying Keras imports until after basic setup, which avoids triggering the `MessageFactory.GetPrototype` error in this Kaggle environment. I also correct the cell numbering to start at 1 (your current script starts at cell 0), so it matches the expected notebook-like format and executes cleanly. To move the score toward your target (since 0.61099 is well below 0.7893), I keep the same “load Keras model → preprocess → predict argmax” core logic but add a small, score-improving test-time augmentation (original + horizontal flip averaged) and enable safe mixed-precision inference if available. Finally, I keep the same submission merge logic and always write a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'I fix the execution blocker caused by the protobuf/Keras import crash (`MessageFactory.GetPrototype`) by removing the hard dependency on Keras in this environment and switching to a stable TensorFlow/Keras import path only if it can be imported successfully. If TensorFlow/Keras cannot be imported (likely under Python 3.13 here), the script still run end-to-end and generate a valid `submission.csv` using the majority-class fallback (so you always get a valid file). I also correct the cell numbering to start at 1 as required and keep the rest of the inference logic (preprocess → optional TTA → argmax over 5 classes) unchanged. This primarily fixes runtime stability; score only improve if the environment can actually load the provided `.keras` model.'
- What this solution (achieved 0.13789) has done: 'I fix the execution blocker caused by the protobuf/TensorFlow import crash by avoiding TensorFlow/Keras entirely in this Python 3.13 environment and using a stable, end-to-end classical image baseline instead (so the script always runs and writes `submission.csv`). To move the score upward from 0.61099 toward the 0.789 target, I replace the current majority-class fallback with a lightweight nearest-centroid classifier on simple color features computed from the provided `train_images/`, then apply it to `test_images/`. This keeps the overall “read images → compute features → predict class id → write submission.csv” inference semantics, but removes the broken dependency and should legitimately improve accuracy beyond the majority-class. I also fix the cell numbering to start at 1 and keep all paths consistent with the provided dataset layout.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
rng = np.random.default_rng(SEED)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = f"{DATA_ROOT}/train.csv"
SAMPLE_SUB_PATH = f"{DATA_ROOT}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_ROOT}/train_images"
TEST_IMG_DIR = f"{DATA_ROOT}/test_images"

print("Python:", os.sys.version)
print("DATA_ROOT exists:", os.path.exists(DATA_ROOT))
print("TRAIN_CSV_PATH exists:", os.path.exists(TRAIN_CSV_PATH))
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))
print("TRAIN_IMG_DIR exists:", os.path.exists(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))

print("Working dir:", os.getcwd())



## === cell 1


def _read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def _compute_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Simple global color statistics + coarse spatial stats.
    Produces a small fixed-length feature vector.
    """
    x = img_rgb.astype(np.float32) / 255.0  # HWC in [0,1]
    mean = x.reshape(-1, 3).mean(axis=0)
    std = x.reshape(-1, 3).std(axis=0)

    small = Image.fromarray((x * 255.0).astype(np.uint8), mode="RGB").resize(
        (32, 32), resample=Image.BILINEAR
    )
    s = np.asarray(small, dtype=np.float32) / 255.0
    s_mean = s.reshape(-1, 3).mean(axis=0)

    r, g, b = mean
    greenness = g - (r + b) / 2.0
    yellowness = (r + g) / 2.0 - b

    feat = np.concatenate(
        [mean, std, s_mean, np.array([greenness, yellowness], dtype=np.float32)], axis=0
    )
    return feat.astype(np.float32)


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)




## === cell 2

train_df = pd.read_csv(TRAIN_CSV_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_ids = sample_sub["image_id"].tolist()

PER_CLASS_CAP = 1200  # 5 * 1200 = up to 6000 training images

train_df = train_df.copy()
train_df["label"] = train_df["label"].astype(int)

selected_idx = []
for c in sorted(train_df["label"].unique()):
    idx = train_df.index[train_df["label"] == c].to_numpy()
    if len(idx) > PER_CLASS_CAP:
        idx = rng.choice(idx, size=PER_CLASS_CAP, replace=False)
    selected_idx.append(idx)
selected_idx = np.concatenate(selected_idx, axis=0)
train_sub = train_df.loc[selected_idx].reset_index(drop=True)

print("Train total:", len(train_df), "Train used:", len(train_sub))
print("Train used per class:", train_sub["label"].value_counts().sort_index().to_dict())

X_list = []
y_list = []
missing_train = 0

for image_id, label in zip(train_sub["image_id"].tolist(), train_sub["label"].tolist()):
    p = os.path.join(TRAIN_IMG_DIR, image_id)
    if not os.path.exists(p):
        missing_train += 1
        continue
    img = _read_rgb(p)
    feat = _compute_features(img)
    X_list.append(feat)
    y_list.append(label)

if missing_train:
    print("WARNING: missing train images:", missing_train)

X_train = np.stack(X_list, axis=0)
y_train = np.array(y_list, dtype=np.int64)

X_train = _l2_normalize(X_train)

num_classes = 5
centroids = np.zeros((num_classes, X_train.shape[1]), dtype=np.float32)
counts = np.zeros((num_classes,), dtype=np.int64)

for c in range(num_classes):
    mask = y_train == c
    if mask.any():
        centroids[c] = X_train[mask].mean(axis=0)
        counts[c] = int(mask.sum())
    else:
        centroids[c] = X_train.mean(axis=0)
        counts[c] = 0

centroids = _l2_normalize(centroids)
print("Centroid counts:", counts.tolist())

majority_label = int(train_df["label"].mode().iloc[0])
print("Majority label:", majority_label)



## === cell 3

image_ids_out = []
preds_out = []
missing_test = 0
unreadable_test = 0

for image_name in test_image_ids:
    img_path = os.path.join(TEST_IMG_DIR, image_name)
    if not os.path.exists(img_path):
        missing_test += 1
        image_ids_out.append(image_name)
        preds_out.append(majority_label)
        continue
    try:
        img = _read_rgb(img_path)
        feat = _compute_features(img)
        feat = _l2_normalize(feat[None, :])[0]
        sims = centroids @ feat
        pred = int(np.argmax(sims))
    except Exception:
        unreadable_test += 1
        pred = majority_label

    image_ids_out.append(image_name)
    preds_out.append(pred)

if missing_test:
    print(
        f"WARNING: {missing_test} test images listed in sample_submission not found on disk."
    )
if unreadable_test:
    print(
        f"WARNING: {unreadable_test} test images could not be read; used majority fallback."
    )

pred_df = pd.DataFrame(
    {"image_id": image_ids_out, "label": np.array(preds_out, dtype=np.int64)}
)

submission = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission["label"].isna().any():
    missing = int(submission["label"].isna().sum())
    print(
        f"WARNING: {missing} missing predictions after merge; filling with majority_label={majority_label}"
    )
    submission["label"] = submission["label"].fillna(majority_label)

submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("submission.csv saved at:", os.path.abspath("submission.csv"))
