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

0.891054699304926

# 6. Current score

0.23767

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'We fix the environment-breaking TensorFlow import issue by avoiding TensorFlow/Keras entirely (the current error originates from protobuf/TensorFlow incompatibility in this runtime), and instead produce a valid submission using a deterministic, label-frequency baseline derived from `train.csv`. We also fix the missing external model files issue by removing those paths and any `load_model` usage, since they are not present under `/kaggle/input/`. Finally, we guarantee the submission format matches `sample_submission.csv` exactly (same `image_id` order and integer `label`) and write `/kaggle/working/submission.csv` end-to-end without runtime errors.'
- What this solution (achieved 0.13042) has done: 'We keep your “no TensorFlow/Keras” constraint and still improve accuracy by replacing the pure majority-class guess with a simple, deterministic image-based lookup: compute the mean RGB color of each training image and predict each test image by nearest (prototype) mean color per class. This preserves the same overall approach (no training loop/model files), runs fast enough under the time limit by downsampling images when computing means, and remains fully deterministic. We also keep the submission aligned exactly to `sample_submission.csv` and include a safe fallback to the majority label if an image can’t be read. This should move your score upward toward the 0.891 target without changing any Kaggle I/O paths.'
- What this solution (achieved 0.33296) has done: 'Your current mean-RGB nearest-prototype baseline is too weak for cassava and is driving the low score; to move toward the 0.891 target we need a stronger image signal while keeping the same “no TF/Keras, no training loop” core approach. I keep the exact pipeline structure (compute per-class prototypes from train images, then nearest-prototype for test), but switch the feature from 3D mean RGB to a compact downsampled grayscale “thumbnail vector” that captures leaf shape/texture and is still deterministic and fast. I also normalize the feature vectors (unit-length) and use cosine distance for robustness to brightness, which typically improves accuracy without changing the overall method. Everything still write `/kaggle/working/submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.27167) has done: 'Your current nearest-prototype method is being held back by using only one global prototype per class (it collapses intra-class variation) and by using a raw thumbnail without simple invariances. Keeping the exact same overall approach (extract deterministic thumbnail features from train images, build class prototypes, then nearest-prototype for test), I (1) switch the feature to a small HOG-like gradient-orientation histogram computed from the same grayscale thumbnail, and (2) use a small fixed number of prototypes per class built by deterministic “farthest-point” selection in feature space (no training loop, no randomness). Prediction stays nearest-prototype with cosine similarity, just over more prototypes, which should move accuracy substantially upward toward your target without changing I/O or relying on TensorFlow/Keras. The submission is still aligned exactly to `sample_submission.csv` and written to `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.23767) has done: 'Your current pipeline is likely underperforming because the HOG implementation “hard-assigns” orientations to bins, which is noisy and loses information; a minimal, safe improvement is to use standard bilinear vote interpolation between the two nearest orientation bins while keeping the exact same feature type and nearest-prototype logic. I also make prototype selection more stable by selecting farthest points using a deterministic distance to the *set* of selected prototypes (same approach you already use, just computed explicitly and robustly). Finally, I keep I/O paths and submission alignment identical, and keep everything deterministic with no extra packages or training loops, aiming to move accuracy upward toward your 0.891 target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
OUT_PATH = "/kaggle/working/submission.csv"

assert os.path.exists(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample_submission.csv at {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train_images dir at {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test_images dir at {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

assert {"image_id", "label"}.issubset(train_df.columns)
assert {"image_id", "label"}.issubset(sample_df.columns)
assert len(sample_df) > 0

label_counts = train_df["label"].value_counts()
majority_label = int(label_counts.idxmax())

print("Train rows:", len(train_df), "Test rows:", len(sample_df))
print("Train label distribution:\n", train_df["label"].value_counts().sort_index())
print("Majority label:", majority_label)



## === cell 1
from PIL import Image


def l2_normalize(v, eps=1e-12):
    n = float(np.sqrt(np.sum(v * v)))
    if n < eps:
        return v * 0.0
    return v / n


def hog_feature_from_path(path, size=64, cell=8, nbins=9):
    """
    Deterministic HOG-like feature, same core idea as before, but with a minimal
    accuracy-oriented fix: bilinear interpolation between neighboring orientation bins
    (instead of hard binning). This usually improves discriminative power without
    changing the overall method/pipeline.
    """
    with Image.open(path) as im:
        im = im.convert("L")
        im = im.resize((size, size), resample=Image.BILINEAR)
        img = np.asarray(im, dtype=np.float32) / 255.0

    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]

    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx) + np.pi  # [0, 2pi)
    ang = np.where(ang >= np.pi, ang - np.pi, ang)  # unsigned: [0, pi)

    bin_f = ang * (nbins / np.pi)  # [0, nbins)
    b0 = np.floor(bin_f).astype(np.int32)
    b0 = np.clip(b0, 0, nbins - 1)
    b1 = b0 + 1
    b1 = np.where(b1 >= nbins, 0, b1)  # wrap around at pi
    w1 = (bin_f - b0.astype(np.float32)).astype(np.float32)
    w0 = (1.0 - w1).astype(np.float32)

    ncell = size // cell
    feat = np.zeros((ncell, ncell, nbins), dtype=np.float32)

    for cy in range(ncell):
        y0 = cy * cell
        y1 = y0 + cell
        for cx in range(ncell):
            x0 = cx * cell
            x1 = x0 + cell

            m = mag[y0:y1, x0:x1].reshape(-1)
            bb0 = b0[y0:y1, x0:x1].reshape(-1)
            bb1 = b1[y0:y1, x0:x1].reshape(-1)
            ww0 = w0[y0:y1, x0:x1].reshape(-1)
            ww1 = w1[y0:y1, x0:x1].reshape(-1)

            h = np.zeros((nbins,), dtype=np.float32)
            h += np.bincount(bb0, weights=(m * ww0), minlength=nbins).astype(np.float32)
            h += np.bincount(bb1, weights=(m * ww1), minlength=nbins).astype(np.float32)
            feat[cy, cx, :] = h

    eps = 1e-6
    blocks = []
    for by in range(ncell - 1):
        for bx in range(ncell - 1):
            blk = feat[by : by + 2, bx : bx + 2, :].reshape(-1)
            blk = blk / np.sqrt(np.sum(blk * blk) + eps)
            blocks.append(blk)

    out = feat.reshape(-1) if len(blocks) == 0 else np.concatenate(blocks, axis=0)
    return out.astype(np.float32)


num_classes = 5
thumb_size = 64
cell = 8
nbins = 9
K_PER_CLASS = 12  # keep same intended budget/speed

train_features = []
train_labels = []
missing_train = 0

for image_id, label in zip(train_df["image_id"].values, train_df["label"].values):
    path = os.path.join(TRAIN_IMG_DIR, image_id)
    try:
        feat = hog_feature_from_path(path, size=thumb_size, cell=cell, nbins=nbins)
        feat = l2_normalize(feat.astype(np.float64)).astype(np.float32)
        train_features.append(feat)
        train_labels.append(int(label))
    except Exception:
        missing_train += 1

train_features = np.asarray(train_features, dtype=np.float32)
train_labels = np.asarray(train_labels, dtype=np.int64)

assert train_features.ndim == 2 and train_features.shape[0] == train_labels.shape[0]
dim = train_features.shape[1]

print("Missing/unreadable train images:", missing_train)
print("Train feats shape:", train_features.shape)

global_mean = (
    train_features.mean(axis=0)
    if train_features.shape[0]
    else np.zeros((dim,), dtype=np.float32)
)
global_mean = l2_normalize(global_mean.astype(np.float64)).astype(np.float32)

prototypes_list = []
proto_labels_list = []

for c in range(num_classes):
    idx = np.where(train_labels == c)[0]
    if idx.size == 0:
        prototypes_list.append(global_mean.copy())
        proto_labels_list.append(c)
        continue

    X = train_features[idx]  # (Nc, dim)
    mu = X.mean(axis=0)
    mu = l2_normalize(mu.astype(np.float64)).astype(np.float32)

    sims_to_mu = X @ mu
    first = int(np.argmax(sims_to_mu))
    selected = [first]

    k_target = min(K_PER_CLASS, X.shape[0])
    min_dist = 1.0 - (X @ X[first])  # cosine distance to first selected

    for _ in range(1, k_target):
        next_i = int(np.argmax(min_dist))
        if next_i in selected:
            break
        selected.append(next_i)
        dist_to_new = 1.0 - (X @ X[next_i])
        min_dist = np.minimum(min_dist, dist_to_new)

    for si in selected:
        prototypes_list.append(X[si].copy())
        proto_labels_list.append(c)

prototypes = np.asarray(prototypes_list, dtype=np.float32)  # (P, dim)
proto_labels = np.asarray(proto_labels_list, dtype=np.int64)  # (P,)

print("Total prototypes:", prototypes.shape[0], "Feature dim:", dim)
print(
    "Prototypes per class:",
    {c: int(np.sum(proto_labels == c)) for c in range(num_classes)},
)



## === cell 2
pred_labels = np.empty((len(sample_df),), dtype=np.int64)
missing_test = 0

for i, image_id in enumerate(sample_df["image_id"].values):
    path = os.path.join(TEST_IMG_DIR, image_id)
    try:
        feat = hog_feature_from_path(path, size=thumb_size, cell=cell, nbins=nbins)
        feat = l2_normalize(feat.astype(np.float64)).astype(np.float32)

        sims = prototypes @ feat  # (P,)
        best = np.full((num_classes,), -1e9, dtype=np.float32)
        for s, c in zip(sims, proto_labels):
            if s > best[c]:
                best[c] = s
        pred = int(np.argmax(best))
    except Exception:
        missing_test += 1
        pred = majority_label
    pred_labels[i] = pred

submission_df = sample_df[["image_id"]].copy()
submission_df["label"] = pred_labels.astype(int)

assert submission_df.shape[0] == sample_df.shape[0]
assert submission_df["image_id"].equals(sample_df["image_id"])
assert submission_df["label"].between(0, 4).all()

submission_df.to_csv(OUT_PATH, index=False)

print("Missing/unreadable test images:", missing_test)
print("Wrote:", OUT_PATH)
print("Rows:", len(submission_df))
print("Predicted label distribution:")
print(submission_df["label"].value_counts().sort_index())

submission_df.head()
