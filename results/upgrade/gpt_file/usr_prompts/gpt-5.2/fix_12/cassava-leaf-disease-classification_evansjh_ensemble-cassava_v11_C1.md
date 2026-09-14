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

0.26383

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'We fix the environment-breaking TensorFlow import issue by avoiding TensorFlow/Keras entirely (the current error originates from protobuf/TensorFlow incompatibility in this runtime), and instead produce a valid submission using a deterministic, label-frequency baseline derived from `train.csv`. We also fix the missing external model files issue by removing those paths and any `load_model` usage, since they are not present under `/kaggle/input/`. Finally, we guarantee the submission format matches `sample_submission.csv` exactly (same `image_id` order and integer `label`) and write `/kaggle/working/submission.csv` end-to-end without runtime errors.'
- What this solution (achieved 0.13042) has done: 'We keep your “no TensorFlow/Keras” constraint and still improve accuracy by replacing the pure majority-class guess with a simple, deterministic image-based lookup: compute the mean RGB color of each training image and predict each test image by nearest (prototype) mean color per class. This preserves the same overall approach (no training loop/model files), runs fast enough under the time limit by downsampling images when computing means, and remains fully deterministic. We also keep the submission aligned exactly to `sample_submission.csv` and include a safe fallback to the majority label if an image can’t be read. This should move your score upward toward the 0.891 target without changing any Kaggle I/O paths.'
- What this solution (achieved 0.33296) has done: 'Your current mean-RGB nearest-prototype baseline is too weak for cassava and is driving the low score; to move toward the 0.891 target we need a stronger image signal while keeping the same “no TF/Keras, no training loop” core approach. I keep the exact pipeline structure (compute per-class prototypes from train images, then nearest-prototype for test), but switch the feature from 3D mean RGB to a compact downsampled grayscale “thumbnail vector” that captures leaf shape/texture and is still deterministic and fast. I also normalize the feature vectors (unit-length) and use cosine distance for robustness to brightness, which typically improves accuracy without changing the overall method. Everything still write `/kaggle/working/submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.27167) has done: 'Your current nearest-prototype method is being held back by using only one global prototype per class (it collapses intra-class variation) and by using a raw thumbnail without simple invariances. Keeping the exact same overall approach (extract deterministic thumbnail features from train images, build class prototypes, then nearest-prototype for test), I (1) switch the feature to a small HOG-like gradient-orientation histogram computed from the same grayscale thumbnail, and (2) use a small fixed number of prototypes per class built by deterministic “farthest-point” selection in feature space (no training loop, no randomness). Prediction stays nearest-prototype with cosine similarity, just over more prototypes, which should move accuracy substantially upward toward your target without changing I/O or relying on TensorFlow/Keras. The submission is still aligned exactly to `sample_submission.csv` and written to `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.23767) has done: 'Your current pipeline is likely underperforming because the HOG implementation “hard-assigns” orientations to bins, which is noisy and loses information; a minimal, safe improvement is to use standard bilinear vote interpolation between the two nearest orientation bins while keeping the exact same feature type and nearest-prototype logic. I also make prototype selection more stable by selecting farthest points using a deterministic distance to the *set* of selected prototypes (same approach you already use, just computed explicitly and robustly). Finally, I keep I/O paths and submission alignment identical, and keep everything deterministic with no extra packages or training loops, aiming to move accuracy upward toward your 0.891 target.'
- What this solution (achieved 0.2429) has done: 'Your current score (0.23767) is far below the target (0.89105), so we should improve accuracy with the smallest changes that keep the same core pipeline (deterministic HOG-like features → per-class prototype selection → nearest-prototype prediction). The most likely issue is that the HOG implementation is missing two standard, high-impact pieces: spatial (bilinear) interpolation into neighboring cells and simple per-cell magnitude normalization (to reduce brightness/contrast sensitivity). I add those two pieces while keeping the same feature dimensionality and the same prototype/prediction logic, so runtime and semantics remain essentially unchanged. I also keep the submission alignment checks and write `/kaggle/working/submission.csv` exactly as required.'
- What this solution (achieved 0.23842) has done: 'Your current gap to the target is large (0.2429 vs 0.8911), so we need a meaningful accuracy lift while keeping the same core pipeline: deterministic HOG-like features → per-class prototype selection → nearest-prototype by cosine similarity. The biggest issue in your current HOG is that the “spatial bilinear interpolation” is implemented via expensive/incorrect masking against clipped cell indices, which effectively mis-assigns many pixels near borders and blurs cell structure; I replace it with a standard, deterministic per-pixel scatter-add into the 4 neighboring cells (still the same HOG concept and same feature dimensionality). I also switch your magnitude normalization from a global mean (which can distort edge contrast) to a simple per-image normalization (sqrt + mean) that is commonly more stable for HOG-like descriptors, without changing the model logic. Everything else (prototype selection, cosine similarity, submission alignment/path) stays the same and it still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.28139) has done: 'Your current score (0.238) is far below the target (0.891), so we need a meaningful accuracy lift while preserving the same core pipeline (deterministic HOG-like features → per-class prototype selection → nearest-prototype by cosine similarity). The smallest high-impact change is to make the HOG descriptor more standard by adding per-block “clipping” (Hys-style) after L2 normalization, which reduces bursty gradients and typically improves classification with cosine similarity without changing the overall method. I also add a tiny uniform smoothing term to each cell histogram before block normalization to stabilize empty/near-empty cells (helps robustness) while keeping feature dimensionality identical. Everything else (image resizing, gradient computation, bilinear orientation+spatial voting, prototype selection, submission alignment/path) remains the same and it still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.31951) has done: 'Your current pipeline is conceptually fine but is likely losing a lot of accuracy due to (1) using unsigned gradients (0..π) which discards edge direction information that is helpful for leaf texture, and (2) using a very low number of prototypes per class for a highly multi-modal dataset. I keep the same core approach (deterministic HOG-like features → deterministic farthest-point prototype selection per class → nearest-prototype by cosine similarity), but make two minimal, targeted changes: switch to signed orientations (0..2π) with double bins to preserve information, and modestly increase `K_PER_CLASS` so intra-class variation is represented better. These changes are legitimate, deterministic, preserve evaluation semantics, and should move accuracy upward toward your target while still producing the same required `submission.csv` format.'
- What this solution (achieved 0.32138) has done: 'We keep your exact pipeline (HOG-like deterministic features → farthest-point prototype selection per class → cosine-sim nearest-prototype voting) but fix two high-impact implementation issues that currently waste information and hurt accuracy. First, we make the HOG block normalization standard by using **overlapping blocks with stride 1 cell** and concatenating all blocks (instead of effectively collapsing to a single block per image), which preserves spatial structure without changing the approach. Second, we remove the very expensive inner loop over bins in `add_votes` by using direct `np.add.at` on flattened indices; this keeps semantics the same but speeds up enough to allow a modest increase in prototypes per class (still deterministic) to better cover intra-class variation. These are minimal changes, still deterministic, still no TF/Keras, and still produce `/kaggle/working/submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.26383) has done: 'Your current score (0.32138) is far below the target (0.89105), so we need a real accuracy lift while keeping the same core pipeline (deterministic HOG-like features → farthest-point prototype selection → cosine-sim nearest-prototype classification). The smallest high-impact change is to keep HOG exactly as-is but improve the *classifier aggregation*: instead of “max similarity per class” over prototypes (very sensitive to outliers), use a deterministic “top‑k average similarity per class” which is a standard robustification for multi-prototype nearest-neighbor and usually improves accuracy on multi-modal classes. This does not change the model family or training approach (still prototype NN on the same features), and it is cheap enough to keep within the time limit. I also slightly increase `K_PER_CLASS` (prototypes per class) to better cover intra-class variation; this remains the same deterministic farthest-point selection, just with a modestly larger K.'

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


def hog_feature_from_path(path, size=64, cell=8, nbins=18):
    """
    Core pipeline preserved:
      grayscale resize -> gradients -> signed orientations -> soft orientation binning ->
      bilinear spatial accumulation into cell histograms -> block normalization -> concatenate.
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

    mag = np.sqrt(mag)
    mag = mag / (float(mag.mean()) + 1e-6)

    bin_f = ang * (nbins / (2.0 * np.pi))  # [0, nbins)
    b0 = np.floor(bin_f).astype(np.int32)
    b0 = np.clip(b0, 0, nbins - 1)
    b1 = b0 + 1
    b1 = np.where(b1 >= nbins, 0, b1)  # wrap
    w1 = (bin_f - b0.astype(np.float32)).astype(np.float32)
    w0 = (1.0 - w1).astype(np.float32)

    ncell = size // cell
    feat = np.zeros((ncell, ncell, nbins), dtype=np.float32)

    ys = (np.arange(size, dtype=np.float32) + 0.5) / cell - 0.5
    xs = (np.arange(size, dtype=np.float32) + 0.5) / cell - 0.5
    yy, xx = np.meshgrid(ys, xs, indexing="ij")  # (size, size)

    cy0 = np.floor(yy).astype(np.int32)
    cx0 = np.floor(xx).astype(np.int32)
    wy1 = (yy - cy0.astype(np.float32)).astype(np.float32)
    wx1 = (xx - cx0.astype(np.float32)).astype(np.float32)
    wy0 = (1.0 - wy1).astype(np.float32)
    wx0 = (1.0 - wx1).astype(np.float32)
    cy1 = cy0 + 1
    cx1 = cx0 + 1

    def add_votes(mask, cy, cx, wcell):
        if not np.any(mask):
            return
        cyv = cy[mask].reshape(-1)
        cxv = cx[mask].reshape(-1)
        wc = wcell[mask].reshape(-1).astype(np.float32)

        mg = (mag[mask].reshape(-1) * wc).astype(np.float32)

        bb0 = b0[mask].reshape(-1)
        bb1 = b1[mask].reshape(-1)
        ww0 = w0[mask].reshape(-1).astype(np.float32)
        ww1 = w1[mask].reshape(-1).astype(np.float32)

        lin = (cyv * ncell + cxv).astype(np.int64)
        feat2 = feat.reshape(-1, nbins)

        np.add.at(feat2, (lin, bb0.astype(np.int64)), mg * ww0)
        np.add.at(feat2, (lin, bb1.astype(np.int64)), mg * ww1)

    m00 = (cy0 >= 0) & (cy0 < ncell) & (cx0 >= 0) & (cx0 < ncell)
    m01 = (cy0 >= 0) & (cy0 < ncell) & (cx1 >= 0) & (cx1 < ncell)
    m10 = (cy1 >= 0) & (cy1 < ncell) & (cx0 >= 0) & (cx0 < ncell)
    m11 = (cy1 >= 0) & (cy1 < ncell) & (cx1 >= 0) & (cx1 < ncell)

    add_votes(m00, cy0, cx0, wy0 * wx0)
    add_votes(m01, cy0, cx1, wy0 * wx1)
    add_votes(m10, cy1, cx0, wy1 * wx0)
    add_votes(m11, cy1, cx1, wy1 * wx1)

    feat = feat + 1e-4

    eps = 1e-6
    clip_val = 0.2
    blocks = []
    for by in range(ncell - 1):
        for bx in range(ncell - 1):
            blk = feat[by : by + 2, bx : bx + 2, :].reshape(-1)
            blk = blk / np.sqrt(np.sum(blk * blk) + eps)
            blk = np.minimum(blk, clip_val)
            blk = blk / np.sqrt(np.sum(blk * blk) + eps)
            blocks.append(blk)

    out = np.concatenate(blocks, axis=0) if len(blocks) else feat.reshape(-1)
    return out.astype(np.float32)


num_classes = 5
thumb_size = 64
cell = 8
nbins = 18

K_PER_CLASS = 80

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
TOPK_PER_CLASS = 5

pred_labels = np.empty((len(sample_df),), dtype=np.int64)
missing_test = 0

proto_idx_by_class = [np.where(proto_labels == c)[0] for c in range(num_classes)]

for i, image_id in enumerate(sample_df["image_id"].values):
    path = os.path.join(TEST_IMG_DIR, image_id)
    try:
        feat = hog_feature_from_path(path, size=thumb_size, cell=cell, nbins=nbins)
        feat = l2_normalize(feat.astype(np.float64)).astype(np.float32)

        sims_all = prototypes @ feat  # (P,)

        class_score = np.full((num_classes,), -1e9, dtype=np.float32)
        for c in range(num_classes):
            idxs = proto_idx_by_class[c]
            if idxs.size == 0:
                continue
            sims_c = sims_all[idxs]
            k = TOPK_PER_CLASS if sims_c.size >= TOPK_PER_CLASS else int(sims_c.size)
            topk = np.partition(sims_c, sims_c.size - k)[-k:]
            class_score[c] = float(topk.mean())

        pred = int(np.argmax(class_score))
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
