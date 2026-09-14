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

2.7

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

0.8842550619522515

# 6. Current score

0.31839

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the immediate import/runtime crash by avoiding the problematic TensorFlow/protobuf stack (your environment says Python 2.7, which is incompatible with the TF2 code you’re importing). Since the three referenced SavedModel directories are not present in your provided `/kaggle/input` tree, I add a robust fallback that trains a simple baseline classifier from `train.csv` label priors and uses it to generate a valid `submission.csv` matching the required format. This keeps the pipeline end-to-end, produces a valid `.csv`, and provides a reasonable (though not SOTA) score instead of “Not yielded”. All file paths remain within the provided dataset structure, and the output is written to `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.13266) has done: 'Your current score (0.61099) is far below the target (0.8843), and the main reason is that the fallback predicts a single majority class for every test image. To move the score upward with minimal changes and without introducing any new ML libraries, I keep your “no TensorFlow” constraint but replace the constant predictor with a simple color-based nearest-centroid classifier computed from the training images. This preserves the overall lightweight baseline approach (no deep model, no new dependencies) while using real image signal, which should substantially improve accuracy toward the target. I also keep the same submission construction from `sample_submission.csv` to guarantee correct ordering/row count and still write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.13378) has done: 'Your current gap to the target is large (0.13266 → 0.8843), and the main bottleneck is that the “mean RGB centroid” feature is too weak and your per-class training cap plus 64×64 resizing can add noise. I keep the same lightweight nearest-centroid core logic, but switch the feature to a slightly richer color representation (mean + standard deviation in RGB, 6D total) while still using PIL only. I also remove the per-class cap so centroids are computed from all available training images (still fast because we only resize and compute simple stats), and I compute features at a slightly higher resize (96) for more stable statistics. These are minimal changes that use more of the available signal without changing the overall approach or requiring TensorFlow/external models, and they should move accuracy upward toward the target.'
- What this solution (achieved 0.14462) has done: 'Your current score (0.13378) is far below the target (0.8843), and the main issue is that the feature is too weak: global color mean/std nearest-centroid cannot separate cassava diseases well. To move the score upward with minimal changes while preserving your same “PIL feature extraction + nearest-centroid classification” core logic, I switch the feature from 6D (RGB mean/std) to a slightly richer but still very cheap descriptor: a small RGB histogram (8 bins/channel = 24D) plus the original mean/std (total 30D). I also normalize the feature vectors (L2) before distance computation so images with different lighting don’t dominate by magnitude, which is a small, safe change that typically improves nearest-centroid behavior. The submission creation stays anchored to `sample_submission.csv` for correct ordering/row count and still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.22384) has done: 'Your current score (0.14462) is far below the target (0.8843), so we should legitimately improve accuracy while keeping your “PIL feature extraction + nearest-centroid classification” core logic unchanged. The biggest gain with minimal risk is to make the distance metric less brittle by standardizing features per-dimension (z-score) using training statistics, instead of relying only on L2 normalization; this often helps histogram/mean/std features. To further stabilize centroids without changing the approach, we compute centroids as the mean of *already standardized* features and use the same standardized space at test time. Submission writing stays identical and still uses `sample_submission.csv` ordering.'
- What this solution (achieved 0.23954) has done: 'The timeout is dominated by scanning and decoding all 18.7k training JPGs twice (once for standardization stats and again for centroids) using PIL in pure-Python loops. To preserve identical feature logic and predictions while cutting runtime, I compute each train image feature exactly once, streaming it to accumulate mean/variance and also caching the feature vectors in a preallocated memmap on disk; then I do a fast second pass over the cached features (no image I/O) to build centroids. I also remove per-row pandas `iterrows()` overhead by iterating over NumPy arrays, and I micro-optimize histogram computation by using `np.bincount` on uint8 values (exactly equivalent bins) instead of `np.histogram` with linspace edges. All paths, feature definitions (mean/std + per-channel hist on 2x2 grid), standardization, L2-normalization, centroid matching, and output semantics remain unchanged.'
- What this solution (achieved 0.26868) has done: 'Your current score (0.23954) is far below the target (0.8843), so we should legitimately improve accuracy while keeping your same “PIL feature extraction + nearest-centroid classification” core logic. The smallest high-impact change is to replace the single centroid per class with a few centroids per class (mini “prototype” centroids) computed by lightweight k-means in the same standardized+L2 space; inference still does nearest-centroid matching, just against more representatives, which usually boosts accuracy a lot for multimodal classes. To avoid introducing instability or new dependencies, the k-means is deterministic (fixed seed), uses a small fixed K per class, and runs on the already-cached features so runtime stays within budget. Submission format, ordering (anchored to `sample_submission.csv`), feature definition, standardization, and distance metric remain unchanged.'
- What this solution (achieved 0.31726) has done: 'Your current gap to the target is large (0.26868 → 0.88426), and the limiting factor is that global color histograms (even with per-class k-means prototypes) don’t capture disease texture/shape well. Keeping the exact same “PIL feature extraction → standardize → L2 → k-means prototypes → nearest prototype” core logic, the smallest high-impact change is to enrich the feature with lightweight edge/texture statistics computed from a grayscale image (no new dependencies). Concretely, we add per-block gradient-magnitude histogram + mean/std (computed via simple finite differences in NumPy), then recompute centroids and inference in the same standardized+L2 space. This preserves the training/inference approach and output semantics while using more discriminative signal that typically improves accuracy substantially.'
- What this solution (achieved 0.31839) has done: 'Your current score (0.31726) is far below the target (0.8843), so we should legitimately increase accuracy while keeping the exact same baseline pipeline (PIL feature extraction → standardize → L2 → per-class k-means prototypes → nearest prototype). The main issue is a subtle train/test mismatch: you L2-normalize the raw feature **before** standardization when building caches, which changes feature scaling and can hurt the standardized space; we keep caching identical data paths but cache the **raw (non-L2)** feature and only apply L2 after standardization for both train and test. I also fix the gradient-magnitude histogram range to be computed from the actual `resize_to` (it was a hard-coded value), so edge histograms don’t saturate incorrectly. These are minimal, deterministic changes that preserve the overall approach and should move accuracy upward toward the target without changing the “core logic” structure.'

# 9. Code solution

## === cell 0
from __future__ import print_function
import os
import json
import numpy as np
import pandas as pd

np.random.seed(42)

DATA_DIR = "../input/cassava-leaf-disease-classification"
if not os.path.isdir(DATA_DIR):
    alt = "/kaggle/input/cassava-leaf-disease-classification"
    if os.path.isdir(alt):
        DATA_DIR = alt

TEST_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

if not os.path.isdir(TEST_DIR):
    raise RuntimeError("Test directory not found at: {}".format(TEST_DIR))
if not os.path.isdir(TRAIN_DIR):
    raise RuntimeError("Train images directory not found at: {}".format(TRAIN_DIR))
if not os.path.isfile(TRAIN_CSV):
    raise RuntimeError("train.csv not found at: {}".format(TRAIN_CSV))
if not os.path.isfile(SAMPLE_SUB):
    raise RuntimeError("sample_submission.csv not found at: {}".format(SAMPLE_SUB))

test_jpgs = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
print("Data dir:", DATA_DIR)
print("Train images dir:", TRAIN_DIR)
print("Test images:", len(test_jpgs))



## === cell 1
"""
Original cell defined custom losses/metrics for TF training/inference.
Kept as a no-op placeholder because TensorFlow is intentionally not imported
to avoid environment crashes; the end-to-end submission is produced via a
robust baseline model below.
"""
pass



## === cell 2
"""
Original cell attempted to load external SavedModels from other Kaggle datasets:
 - ../input/only-xception-with-cropping/...
 - ../input/efficientnet-with-cropping/...
 - ../input/gambler-s-loss-cassava/...

These paths are not present in the provided file tree, so inference cannot run.
We keep the detection and use a baseline fallback that does not require unavailable assets.
"""


def _dir_exists(p):
    try:
        return p and os.path.isdir(p)
    except Exception:
        return False


missing_models = []
for p in [
    "../input/only-xception-with-cropping/saved-model-11-0.879",
    "../input/efficientnet-with-cropping/saved-model-06-0.88",
    "../input/gambler-s-loss-cassava/saved-model-10-0.843",
]:
    if not _dir_exists(p) and not _dir_exists("/kaggle/input/" + p[len("../input/") :]):
        missing_models.append(p)

if missing_models:
    print("SavedModel assets not found; will use baseline fallback. Missing:")
    for p in missing_models:
        print(" -", p)
else:
    print("SavedModel assets appear present (unexpected in provided tree).")



## === cell 3
"""
Original random crop generator used for image augmentation.
Not used in the baseline fallback, but kept to preserve cell structure.
"""


def random_crop(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y : (y + dy), x : (x + dx), :]


def crop_generator(batches, crop_length):
    while True:
        batch_x = next(batches)
        batch_crops = np.zeros(
            (batch_x.shape[0], crop_length, crop_length, 3), dtype=batch_x.dtype
        )
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield batch_crops




## === cell 4
train_df = pd.read_csv(TRAIN_CSV)
if "label" not in train_df.columns or "image_id" not in train_df.columns:
    raise RuntimeError("train.csv must have columns: image_id, label")

label_counts = train_df["label"].value_counts().sort_index()
counts = np.zeros((5,), dtype=np.float64)
for k, v in label_counts.items():
    if 0 <= int(k) < 5:
        counts[int(k)] = float(v)

alpha = 1.0
priors = (counts + alpha) / (counts.sum() + alpha * len(counts))
majority_label = int(np.argmax(priors))

print("Train label counts:", counts.astype(int).tolist())
print("Smoothed priors:", priors.tolist())
print("Majority label (fallback):", majority_label)

sample_df = pd.read_csv(SAMPLE_SUB)
if "image_id" not in sample_df.columns:
    raise RuntimeError("sample_submission.csv missing 'image_id' column")

test_df = sample_df[["image_id"]].copy()
print("Test rows (from sample_submission):", len(test_df))




## === cell 5
def _try_import_pil():
    try:
        from PIL import Image  # noqa: F401

        return True
    except Exception:
        return False


def _l2_normalize(v, eps=1e-12):
    n = float(np.sqrt((v * v).sum()))
    if n < eps:
        return v
    return v / n


def _hist_u8_fixed_bins(channel_u8, hist_bins):
    idx = (channel_u8.astype(np.uint16) * np.uint16(hist_bins)) >> np.uint16(8)
    h = np.bincount(idx, minlength=hist_bins).astype(np.float32)
    denom = float(h.sum()) if float(h.sum()) > 0.0 else 1.0
    return h / denom


def _hist_f32_fixed_bins(x_f32, hist_bins, vmin, vmax):
    if vmax <= vmin:
        return np.zeros((hist_bins,), dtype=np.float64)
    x = x_f32
    t = (x - vmin) / (vmax - vmin)
    t = np.clip(t, 0.0, 1.0)
    idx = np.floor(t * float(hist_bins)).astype(np.int32)
    idx[idx == hist_bins] = hist_bins - 1
    h = np.bincount(idx.ravel(), minlength=hist_bins).astype(np.float64)
    denom = float(h.sum()) if float(h.sum()) > 0.0 else 1.0
    return h / denom


def _block_feat(arr_u8_flat_3, hist_bins):
    arr_f = arr_u8_flat_3.astype(np.float32)
    m = arr_f.mean(axis=0)
    s = arr_f.std(axis=0)

    h0 = _hist_u8_fixed_bins(arr_u8_flat_3[:, 0], hist_bins)
    h1 = _hist_u8_fixed_bins(arr_u8_flat_3[:, 1], hist_bins)
    h2 = _hist_u8_fixed_bins(arr_u8_flat_3[:, 2], hist_bins)
    h = np.concatenate([h0, h1, h2], axis=0)
    return np.concatenate([m, s, h], axis=0).astype(np.float64)


def _block_edge_feat(gray_u8_block, edge_bins=8, mag_max=360.62445):
    g = gray_u8_block.astype(np.float32)

    gx = g[:, 1:] - g[:, :-1]
    gy = g[1:, :] - g[:-1, :]
    h = min(gx.shape[0], gy.shape[0])
    w = min(gx.shape[1], gy.shape[1])
    if h <= 0 or w <= 0:
        mag = np.zeros((1, 1), dtype=np.float32)
    else:
        gx2 = gx[:h, :w]
        gy2 = gy[:h, :w]
        mag = np.sqrt(gx2 * gx2 + gy2 * gy2)

    mag_mean = np.array([float(mag.mean())], dtype=np.float64)
    mag_std = np.array([float(mag.std())], dtype=np.float64)
    mag_hist = _hist_f32_fixed_bins(
        mag, hist_bins=edge_bins, vmin=0.0, vmax=float(mag_max)
    )
    return np.concatenate([mag_mean, mag_std, mag_hist], axis=0).astype(np.float64)


def _rgb_hist_mean_std_from_jpg_raw(
    path, resize_to=160, hist_bins=8, grid=2, edge_bins=8
):
    """
    Change (score-up): return RAW (non-L2-normalized) features.
    We will standardize first and only then apply L2 normalization, consistently for train+test.
    This fixes a train/test mismatch where cached train features were L2-normalized before standardization.
    """
    try:
        from PIL import Image

        im = Image.open(path).convert("RGB")
        if resize_to is not None:
            im = im.resize((resize_to, resize_to))
        arr = np.asarray(im, dtype=np.uint8)
        hh, ww, _ = arr.shape

        gray = (
            0.299 * arr[:, :, 0].astype(np.float32)
            + 0.587 * arr[:, :, 1].astype(np.float32)
            + 0.114 * arr[:, :, 2].astype(np.float32)
        )
        gray_u8 = np.clip(gray, 0.0, 255.0).astype(np.uint8)

        mag_max = float(np.sqrt(2.0) * 255.0)

        feats = []
        if grid is None or int(grid) <= 1:
            feats.append(_block_feat(arr.reshape(-1, 3), hist_bins))
            feats.append(
                _block_edge_feat(gray_u8, edge_bins=edge_bins, mag_max=mag_max)
            )
        else:
            g = int(grid)
            ys = np.linspace(0, hh, g + 1).astype(int)
            xs = np.linspace(0, ww, g + 1).astype(int)
            for yi in range(g):
                y0, y1 = ys[yi], ys[yi + 1]
                for xi in range(g):
                    x0, x1 = xs[xi], xs[xi + 1]
                    block = arr[y0:y1, x0:x1, :]
                    feats.append(_block_feat(block.reshape(-1, 3), hist_bins))
                    feats.append(
                        _block_edge_feat(
                            gray_u8[y0:y1, x0:x1], edge_bins=edge_bins, mag_max=mag_max
                        )
                    )

        feat = np.concatenate(feats, axis=0).astype(np.float64)
        return feat
    except Exception:
        return None


use_pil = _try_import_pil()
print("PIL available:", bool(use_pil))

resize_to = 160
hist_bins = 8
grid = 2  # 2x2 blocks
edge_bins = 8

block_dim_color = 3 + 3 + 3 * hist_bins  # 30
block_dim_edge = 1 + 1 + edge_bins  # 10
block_dim = block_dim_color + block_dim_edge  # 40 per block
feat_dim = (grid * grid) * block_dim  # 4*40=160

train_feat_sum = np.zeros((feat_dim,), dtype=np.float64)
train_feat_sum2 = np.zeros((feat_dim,), dtype=np.float64)
train_feat_n = 0

cache_tag = "RAW_r{0}_hb{1}_g{2}_eb{3}_fd{4}".format(
    resize_to, hist_bins, grid, edge_bins, feat_dim
)
cache_path = "/kaggle/working/train_feats_cache_{0}.dat".format(cache_tag)
cache_labels_path = "/kaggle/working/train_labels_cache_{0}.npy".format(cache_tag)
cache_exists = os.path.exists(cache_path) and os.path.exists(cache_labels_path)

feats_mm = None
ok_mask = None
train_labels = None
n_train = 0

if use_pil:
    train_sorted = train_df.sort_values("image_id").reset_index(drop=True)
    train_img_ids = train_sorted["image_id"].astype(str).values
    train_labels = train_sorted["label"].astype(np.int64).values
    n_train = int(len(train_sorted))

    expected_size = n_train * feat_dim * 8
    if (not cache_exists) or (os.path.getsize(cache_path) != expected_size):
        feats_mm = np.memmap(
            cache_path, dtype="float64", mode="w+", shape=(n_train, feat_dim)
        )
        ok_mask = np.zeros((n_train,), dtype=np.uint8)

        for i in range(n_train):
            img_path = os.path.join(TRAIN_DIR, train_img_ids[i])
            if not os.path.exists(img_path):
                continue
            f = _rgb_hist_mean_std_from_jpg_raw(
                img_path,
                resize_to=resize_to,
                hist_bins=hist_bins,
                grid=grid,
                edge_bins=edge_bins,
            )
            if f is None:
                continue
            feats_mm[i, :] = f
            ok_mask[i] = 1
            train_feat_sum += f
            train_feat_sum2 += f * f
            train_feat_n += 1

        feats_mm.flush()
        np.save(cache_labels_path, np.vstack([train_labels, ok_mask]).astype(np.int64))
    else:
        feats_mm = np.memmap(
            cache_path, dtype="float64", mode="r", shape=(n_train, feat_dim)
        )
        lab_ok = np.load(cache_labels_path)
        train_labels = lab_ok[0].astype(np.int64)
        ok_mask = lab_ok[1].astype(np.uint8)

        ok_idx = np.where(ok_mask == 1)[0]
        train_feat_n = int(len(ok_idx))
        if train_feat_n:
            chunk = 4096
            for s in range(0, train_feat_n, chunk):
                idx = ok_idx[s : s + chunk]
                X = np.asarray(feats_mm[idx, :], dtype=np.float64)
                train_feat_sum += X.sum(axis=0)
                train_feat_sum2 += (X * X).sum(axis=0)

if train_feat_n > 1:
    feat_mean = train_feat_sum / float(train_feat_n)
    feat_var = train_feat_sum2 / float(train_feat_n) - feat_mean * feat_mean
    feat_var = np.maximum(feat_var, 1e-12)
    feat_std = np.sqrt(feat_var)
else:
    feat_mean = np.zeros((feat_dim,), dtype=np.float64)
    feat_std = np.ones((feat_dim,), dtype=np.float64)

print("Train features used for standardization:", int(train_feat_n))


def _standardize_feat(f):
    return (f - feat_mean) / feat_std


def _kmeans_prototypes(X, k, n_iter=12, seed=42):
    n = int(X.shape[0])
    if n <= 0:
        return None
    if n <= k:
        C = X.copy()
        for i in range(C.shape[0]):
            C[i] = _l2_normalize(C[i])
        return C

    rs = np.random.RandomState(seed)
    init_idx = rs.choice(n, size=k, replace=False)
    C = X[init_idx, :].copy()

    for _ in range(int(n_iter)):
        x2 = (X * X).sum(axis=1).reshape(n, 1)
        c2 = (C * C).sum(axis=1).reshape(1, k)
        d2 = x2 + c2 - 2.0 * np.dot(X, C.T)
        assign = np.argmin(d2, axis=1)

        C_new = np.zeros_like(C)
        for j in range(k):
            idxj = np.where(assign == j)[0]
            if idxj.size > 0:
                C_new[j] = X[idxj].mean(axis=0)
            else:
                ridx = int(rs.randint(0, n))
                C_new[j] = X[ridx]
        C = C_new

    for j in range(k):
        C[j] = _l2_normalize(C[j])
    return C


proto_per_class = 6  # unchanged core setting
prototypes = []  # list of (class_id, centroid_vec)
if use_pil and (feats_mm is not None) and (ok_mask is not None) and train_feat_n > 0:
    for c in range(5):
        idx_c = np.where((ok_mask == 1) & (train_labels == c))[0]
        if idx_c.size == 0:
            continue

        Xc = np.asarray(feats_mm[idx_c, :], dtype=np.float64)

        Xc = (Xc - feat_mean.reshape(1, feat_dim)) / feat_std.reshape(1, feat_dim)
        for i in range(Xc.shape[0]):
            Xc[i] = _l2_normalize(Xc[i])

        k = int(min(proto_per_class, max(1, int(np.sqrt(float(idx_c.size))))))
        Cc = _kmeans_prototypes(Xc, k=k, n_iter=12, seed=42 + c)
        if Cc is None:
            continue
        for j in range(Cc.shape[0]):
            prototypes.append((int(c), Cc[j].astype(np.float64)))

print("Prototype centroids total:", int(len(prototypes)))
proto_counts = {}
for cls, _ in prototypes:
    proto_counts[cls] = proto_counts.get(cls, 0) + 1
print("Prototype centroids per class:", proto_counts)



## === cell 6
pred_labels = np.full((len(test_df),), majority_label, dtype=np.int64)

if use_pil and len(prototypes) > 0:
    proto_classes = np.array([p[0] for p in prototypes], dtype=np.int64)
    proto_mat = np.vstack([p[1] for p in prototypes]).astype(np.float64)  # (m,d)

    test_img_ids = test_df["image_id"].astype(str).values
    for i in range(len(test_img_ids)):
        img_path = os.path.join(TEST_DIR, test_img_ids[i])
        if not os.path.exists(img_path):
            continue
        f = _rgb_hist_mean_std_from_jpg_raw(
            img_path,
            resize_to=resize_to,
            hist_bins=hist_bins,
            grid=grid,
            edge_bins=edge_bins,
        )
        if f is None:
            continue

        fz = _standardize_feat(f)
        fz = _l2_normalize(fz)

        d = proto_mat - fz.reshape(1, feat_dim)
        dist2 = (d * d).sum(axis=1)
        pred_labels[i] = int(proto_classes[int(np.argmin(dist2))])

missing = 0
for img_id in test_df["image_id"].values[:50]:
    if not os.path.exists(os.path.join(TEST_DIR, str(img_id))):
        missing += 1
if missing:
    print("Warning: some test images not found in first 50 checked:", missing)

print(
    "Pred label distribution:",
    pd.Series(pred_labels).value_counts().sort_index().to_dict(),
)



## === cell 7
results_new = pd.DataFrame(
    {
        "image_id": test_df["image_id"].astype(str).values,
        "label": pred_labels.astype(int),
    }
)
results_new = results_new[["image_id", "label"]]

out_path = "/kaggle/working/submission.csv"
results_new.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(results_new.head())
print("Rows:", len(results_new), "Cols:", list(results_new.columns))
print("Unique labels:", np.unique(results_new["label"].values).tolist())
