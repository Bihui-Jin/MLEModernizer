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

0.34791

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.15433) has done: 'I remove the failing TensorFlow Hub dependency (it’s triggering the `MessageFactory.GetPrototype` protobuf error in this environment and also relies on an unavailable cached input folder), and replace it with a stable, built-in TensorFlow/Keras model available offline. I also fix the missing-path issues by resolving the correct competition input directory and ensuring `test_image_dir`/`sample` are always defined. Because your external `.h5` models are not present in the provided filesystem, I keep the ensemble/voting logic but load a single lightweight baseline model (MobileNetV2) so the notebook runs end-to-end and writes a valid `submission.csv` with the correct columns. Finally, I ensure predictions align exactly to `sample_submission.csv` order to avoid submission formatting/index mismatches.'
- What this solution (achieved 0.61099) has done: 'The crash happens before any data/model code runs because TensorFlow’s import triggers a protobuf API mismatch (`MessageFactory.GetPrototype`) in this environment. To make the notebook run end-to-end and produce a valid `submission.csv`, I remove the TensorFlow/Keras dependency entirely and replace the failing image-model section with a stable, offline baseline: a simple majority-class predictor learned from `train.csv`. This is minimal, fixes the runtime error deterministically, preserves correct submission ordering/format, and should substantially improve accuracy over near-random predictions (moving your score toward the target). Paths are kept compatible with both `/kaggle/input/...` and `/kaggle/data/...` layouts.'
- What this solution (achieved 0.17937) has done: 'To move accuracy upward from the current majority-class baseline (0.61099) toward your target (~0.8988), the smallest legitimate improvement is to stop predicting a single class for every image and instead use the provided test images with a lightweight, offline classifier. Since TensorFlow currently fails to import in this environment, this patch keeps the same simple “train once → predict labels” approach but upgrades the model to a CPU-friendly nearest-centroid classifier using raw pixel features (downsampled grayscale), implemented only with NumPy/Pandas (no new dependencies). This should materially increase accuracy versus majority class while staying simple and deterministic. The submission is still written in exactly `sample_submission.csv` order with the correct columns and `submission.csv` filename.'
- What this solution (achieved 0.17676) has done: 'Your current nearest-centroid baseline is likely underperforming because it uses only the first 6000 training images (sorted by `image_id`), which can bias class coverage and centroid quality. To move accuracy upward toward the 0.8988 target while preserving the same “nearest centroid on downsampled grayscale pixels” core logic, I (1) change the training subset selection to a deterministic **stratified** per-class sample so every class contributes, and (2) increase the number of training images used (still bounded for runtime) to produce more reliable centroids. I also compute centroids for all 5 classes (falling back to global mean if a class has zero loaded images) to avoid missing-class behavior. Submission formatting and ordering remain exactly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.32549) has done: 'We keep your nearest-centroid-on-downsampled-grayscale core logic unchanged, but fix the main accuracy bottleneck: raw grayscale pixels are very sensitive to lighting/background, so we add a tiny, deterministic normalization step (per-image standardization) before flattening, which usually improves centroid separability without changing the model family. We also compute centroids in a numerically stable, streaming way (sums/counts) to avoid any subtle bias from stacking order and reduce memory pressure, while still using the same subset and distance rule. Finally, we slightly increase `IMG_SIZE` (still lightweight) to retain more leaf structure, which should move the score upward toward your target without altering the overall approach or runtime envelope.'
- What this solution (achieved 0.26457) has done: 'Your current gap to the target is large (0.32549 → 0.89876), so we need a meaningful but still minimal, same-family improvement. We keep the exact “nearest-centroid on downsampled grayscale pixels + L2 distance” core logic, but fix the biggest accuracy limiter: using a single global centroid per class is too crude. The smallest upgrade within the same centroid framework is to use multiple centroids per class (a tiny, fixed k-means per class implemented in NumPy), then predict by nearest centroid among all class-centroids. This remains deterministic, lightweight, uses no new packages, keeps the same submission semantics/order, and should move accuracy upward toward the target.'
- What this solution (achieved 0.24103) has done: 'We need to move accuracy up from 0.26457 toward 0.89876, so we keep your exact “grayscale → standardize → (per-class) k-means centroids → nearest-centroid by L2” pipeline, but make the smallest changes that typically yield a large accuracy jump: (1) use a leaf-focused center-crop before resizing to reduce background noise, and (2) add a simple multi-scale feature concatenation (two resolutions) while still using the same centroid+distance logic. These keep the same model family and prediction rule, just improve the feature representation quality. To keep runtime bounded, we slightly reduce `KMEANS_ITERS` (same algorithm, same deterministic loop) and cap feature extraction cost via smaller sizes.'
- What this solution (achieved 0.28774) has done: 'We need to move accuracy up from 0.241 toward 0.898, so we keep your exact centroid/k-means + nearest-centroid core logic but make the features less sensitive to background/pose using a minimal, deterministic augmentation at inference: average predictions over a few fixed crops (multi-crop TTA) while still using the same distance-to-centroids rule. To make those centroids more reliable without changing the model family, we also compute centroids from the full stratified budget (same MAX_TRAIN_IMAGES) but use mild, deterministic horizontal-flip augmentation only when building centroids (effectively denoising class prototypes). Finally, we keep submission ordering identical to `sample_submission.csv` and preserve all paths/I/O, while adding small safeguards to ensure consistent RGB decode and reduce PIL edge-case failures.'
- What this solution (achieved 0.44021) has done: 'We need to raise accuracy from 0.28774 toward 0.89876, so we keep your exact “grayscale → standardize → per-class k-means centroids → nearest-centroid by L2 with multi-crop TTA” core logic and only strengthen the representation in a deterministic, lightweight way. The smallest high-impact change is to add simple, non-learned edge/texture information (Sobel gradients) alongside your existing multi-scale grayscale features; this often helps disease-spot patterns without changing the classifier family. To keep runtime within budget, we moderately reduce `MAX_TRAIN_IMAGES` to offset the added feature cost while keeping stratification and k-means unchanged. We also keep submission ordering identical to `sample_submission.csv` and preserve all paths/I/O.'
- What this solution (achieved 0.39798) has done: 'Your current gap to the target is large (0.44021 → 0.89876), so we need a meaningful accuracy lift but still keep the exact same “feature extraction → per-class k-means centroids → nearest-centroid with TTA” pipeline. The smallest high-impact change within that same logic is to make the hand-crafted features more disease-spot sensitive by adding a compact Local Binary Pattern (LBP) histogram (texture) and simple color statistics, while keeping all existing grayscale+Sobel features intact. To avoid degrading performance via inconsistent feature scaling between concatenated blocks, we also apply a single final per-image standardization to the concatenated feature vector (does not change the classifier family, just stabilizes distance computations). Finally, to keep runtime under control, we slightly reduce TTA crops and modestly increase training images to improve centroid quality (still stratified, still k-means, same prediction rule).'
- What this solution (achieved 0.33483) has done: 'Your current score (0.39798) is far below the target (0.89876), so we should push accuracy upward while keeping the exact same “hand-crafted feature extraction → per-class k-means centroids → nearest-centroid with TTA” pipeline. The biggest minimal win here is to fix feature scaling across blocks: concatenating huge flattened pixel/edge vectors with tiny LBP/color vectors makes the distance dominated by the largest block, so we normalize each feature block to comparable scale before concatenation (still the same features and same distance rule). We also make the LBP histogram higher-resolution (more bins) to better capture disease textures, and slightly increase training coverage and centroid count per class to improve prototype quality without changing the algorithm family. All paths/I/O and submission ordering remain identical, and the script still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.34268) has done: 'The timeout is dominated by (1) very expensive per-image feature extraction (Sobel via many array ops + multi-crop TTA) repeated thousands of times and (2) Python-loop-heavy KMeans assignment/updates. I keep the exact same features, TTA semantics, augmentation, and KMeans logic, but eliminate redundant work by caching image reads, vectorizing Sobel with slicing (same kernels, same output), and making KMeans updates/empty-cluster handling vectorized while preserving identical Lloyd iterations. I also reduce overhead by using PIL’s context manager to close files promptly and by parallelizing independent feature extraction with a deterministic thread pool (order preserved), which is safe because the work is I/O + NumPy (releases GIL) and does not change computations. All paths, hyperparameters, and algorithmic steps remain the same; only implementation details are optimized for runtime.'
- What this solution (achieved 0.34791) has done: 'Your current score (0.34268) is far below the target (0.89876), so we should increase accuracy while keeping the same “hand-crafted features → per-class k-means centroids → nearest-centroid with TTA” core logic. The biggest minimal win here is to fix a mismatch between training-time features and inference-time features: you compute dataset-level mean/std using *pre-normalized* features, but you apply it later to both train/test, which can distort distances; we instead compute mean/std on the exact features that be used by the classifier (after all per-image/block/final normalizations). Second, we make the per-class stratified sampling proportional to class frequency (instead of equal per class) to better match the true distribution while still ensuring each class is represented. These changes preserve architecture/approach and evaluation semantics (still nearest centroid by L2 with the same features and TTA), but should move the score upward toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(42)

BASE_INPUT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input",  # fallback (some environments mount files directly here)
    "/kaggle/data",
]


def _resolve_base():
    for p in BASE_INPUT_CANDIDATES:
        if os.path.isdir(p):
            if os.path.isfile(os.path.join(p, "sample_submission.csv")):
                return p
            nested = os.path.join(p, "cassava-leaf-disease-classification")
            if os.path.isfile(os.path.join(nested, "sample_submission.csv")):
                return nested
    return None


BASE_INPUT = _resolve_base()
if BASE_INPUT is None:
    raise FileNotFoundError(
        "Could not find competition directory containing sample_submission.csv in: "
        + ", ".join(BASE_INPUT_CANDIDATES)
    )

train_image_dir = os.path.join(BASE_INPUT, "train_images")
test_image_dir = os.path.join(BASE_INPUT, "test_images")
sample = os.path.join(BASE_INPUT, "sample_submission.csv")
train_csv_path = os.path.join(BASE_INPUT, "train.csv")

if not os.path.isfile(sample):
    raise FileNotFoundError(f"Missing sample submission: {sample}")
if not os.path.isfile(train_csv_path):
    raise FileNotFoundError(f"Missing train.csv: {train_csv_path}")
if not os.path.isdir(train_image_dir):
    raise FileNotFoundError(f"Missing train_images directory: {train_image_dir}")
if not os.path.isdir(test_image_dir):
    raise FileNotFoundError(f"Missing test_images directory: {test_image_dir}")

print("Resolved paths:")
print("BASE_INPUT:", BASE_INPUT)
print("train_image_dir:", train_image_dir)
print("test_image_dir:", test_image_dir)
print("sample:", sample)
print("train_csv:", train_csv_path)



## === cell 1
sample_csv = pd.read_csv(sample)
assert list(sample_csv.columns) == [
    "image_id",
    "label",
], "Unexpected sample_submission.csv columns"
print(sample_csv.head())
print("Test rows:", len(sample_csv))

train_df = pd.read_csv(train_csv_path)
assert list(train_df.columns) == ["image_id", "label"], "Unexpected train.csv columns"
print(train_df.head())
print("Train rows:", len(train_df))

label_counts = train_df["label"].value_counts()
majority_label = int(label_counts.idxmax())
print("Train label distribution:")
print(label_counts.sort_index())
print("Majority label:", majority_label, "count:", int(label_counts.max()))




## === cell 2
def _try_import_pil():
    try:
        from PIL import Image, ImageOps  # type: ignore

        return Image, ImageOps
    except Exception:
        return None, None


Image, ImageOps = _try_import_pil()
if Image is None:
    print(
        "WARNING: Pillow (PIL) not available; falling back to majority-class predictions."
    )
    USE_IMAGES = False
else:
    USE_IMAGES = True
    print("Pillow available; using image-based nearest-centroid classifier.")

IMG_SIZE_1 = 96
IMG_SIZE_2 = 64

CENTER_CROP_FRAC = 0.90

MAX_TRAIN_IMAGES = 17000
N_CLASSES = 5

K_PER_CLASS = 8
KMEANS_ITERS = 5

TTA_CROPS = ("center", "tl", "tr")

AUG_FLIP_IN_TRAIN = True

USE_SOBEL_FEATURES = True

USE_LBP_FEATURES = True
LBP_BINS = 32
USE_COLOR_STATS = True

PER_BLOCK_L2_NORMALIZE = True

FINAL_FEATURE_STANDARDIZE = True

USE_DATASET_FEATURE_NORMALIZATION = True


def _crop_box(w, h, frac, mode="center"):
    side = int(min(w, h) * float(frac))
    if side <= 1:
        return (0, 0, w, h)
    if mode == "center":
        left = (w - side) // 2
        top = (h - side) // 2
    elif mode == "tl":
        left, top = 0, 0
    elif mode == "tr":
        left, top = w - side, 0
    elif mode == "bl":
        left, top = 0, h - side
    elif mode == "br":
        left, top = w - side, h - side
    else:
        left = (w - side) // 2
        top = (h - side) // 2
    left = int(max(0, min(left, w - side)))
    top = int(max(0, min(top, h - side)))
    return (left, top, left + side, top + side)


def _to_gray_standardized(arr_rgb):
    arr = np.asarray(arr_rgb, dtype=np.float32)
    gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    gray = gray / 255.0
    m = float(gray.mean())
    s = float(gray.std())
    if s < 1e-6:
        s = 1e-6
    gray = (gray - m) / s
    return gray


def _sobel_features(gray2d: np.ndarray) -> np.ndarray:
    """
    Deterministic, non-learned feature to capture edges/texture.
    Uses simple Sobel kernels; output is standardized similarly to gray.
    """
    g = gray2d.astype(np.float32, copy=False)
    gp = np.pad(g, ((1, 1), (1, 1)), mode="edge")

    a00 = gp[:-2, :-2]
    a01 = gp[:-2, 1:-1]
    a02 = gp[:-2, 2:]
    a10 = gp[1:-1, :-2]
    a11 = gp[1:-1, 1:-1]
    a12 = gp[1:-1, 2:]
    a20 = gp[2:, :-2]
    a21 = gp[2:, 1:-1]
    a22 = gp[2:, 2:]

    gx = (-1.0 * a00) + (0.0 * a01) + (1.0 * a02)
    gx += (-2.0 * a10) + (0.0 * a11) + (2.0 * a12)
    gx += (-1.0 * a20) + (0.0 * a21) + (1.0 * a22)

    gy = (-1.0 * a00) + (-2.0 * a01) + (-1.0 * a02)
    gy += (0.0 * a10) + (0.0 * a11) + (0.0 * a12)
    gy += (1.0 * a20) + (2.0 * a21) + (1.0 * a22)

    mag = np.sqrt(gx * gx + gy * gy + 1e-12).astype(np.float32)

    m = float(mag.mean())
    s = float(mag.std())
    if s < 1e-6:
        s = 1e-6
    mag = (mag - m) / s
    return np.concatenate([gx.reshape(-1), gy.reshape(-1), mag.reshape(-1)], axis=0)


def _lbp_hist(gray2d: np.ndarray, bins: int = LBP_BINS) -> np.ndarray:
    """
    Deterministic texture descriptor: basic 8-neighbor LBP code histogram.
    Uses the standardized gray input; only relative comparisons matter.
    Returns L1-normalized histogram.
    """
    g = gray2d.astype(np.float32, copy=False)
    gp = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    c = gp[1:-1, 1:-1]

    code = np.zeros_like(c, dtype=np.uint8)
    code |= (gp[0:-2, 0:-2] > c).astype(np.uint8) << 0
    code |= (gp[0:-2, 1:-1] > c).astype(np.uint8) << 1
    code |= (gp[0:-2, 2:] > c).astype(np.uint8) << 2
    code |= (gp[1:-1, 2:] > c).astype(np.uint8) << 3
    code |= (gp[2:, 2:] > c).astype(np.uint8) << 4
    code |= (gp[2:, 1:-1] > c).astype(np.uint8) << 5
    code |= (gp[2:, 0:-2] > c).astype(np.uint8) << 6
    code |= (gp[1:-1, 0:-2] > c).astype(np.uint8) << 7

    bin_idx = (code.astype(np.int32) * bins) // 256
    hist = np.bincount(bin_idx.reshape(-1), minlength=bins).astype(np.float32)
    s = float(hist.sum())
    if s > 0:
        hist /= s
    return hist


def _color_stats(arr_rgb: np.ndarray) -> np.ndarray:
    """
    Simple global color stats (mean/std per channel) to separate healthy vs diseased discoloration.
    """
    x = np.asarray(arr_rgb, dtype=np.float32) / 255.0
    mu = x.reshape(-1, 3).mean(axis=0)
    sd = x.reshape(-1, 3).std(axis=0)
    return np.concatenate([mu, sd], axis=0).astype(np.float32)


def _final_standardize(v: np.ndarray) -> np.ndarray:
    m = float(v.mean())
    s = float(v.std())
    if s < 1e-6:
        s = 1e-6
    return ((v - m) / s).astype(np.float32, copy=False)


def _l2_normalize_block(v: np.ndarray) -> np.ndarray:
    n = float(np.sqrt((v * v).sum()))
    if n < 1e-12:
        return v.astype(np.float32, copy=False)
    return (v / n).astype(np.float32, copy=False)


def load_image_feature_from_pil(img, img_size1=IMG_SIZE_1, img_size2=IMG_SIZE_2):
    img1 = img.resize((img_size1, img_size1))
    img2 = img.resize((img_size2, img_size2))

    g1 = _to_gray_standardized(img1)
    g2 = _to_gray_standardized(img2)

    f_g1 = g1.reshape(-1).astype(np.float32, copy=False)
    f_g2 = g2.reshape(-1).astype(np.float32, copy=False)
    if PER_BLOCK_L2_NORMALIZE:
        f_g1 = _l2_normalize_block(f_g1)
        f_g2 = _l2_normalize_block(f_g2)

    feats = [f_g1, f_g2]

    if USE_SOBEL_FEATURES:
        s1 = _sobel_features(g1).astype(np.float32, copy=False)
        s2 = _sobel_features(g2).astype(np.float32, copy=False)
        if PER_BLOCK_L2_NORMALIZE:
            s1 = _l2_normalize_block(s1)
            s2 = _l2_normalize_block(s2)
        feats.extend([s1, s2])

    if USE_LBP_FEATURES:
        h1 = _lbp_hist(g1, bins=LBP_BINS).astype(np.float32, copy=False)
        h2 = _lbp_hist(g2, bins=LBP_BINS).astype(np.float32, copy=False)
        if PER_BLOCK_L2_NORMALIZE:
            h1 = _l2_normalize_block(h1)
            h2 = _l2_normalize_block(h2)
        feats.extend([h1, h2])

    if USE_COLOR_STATS:
        cs = _color_stats(np.asarray(img1)).astype(np.float32, copy=False)
        if PER_BLOCK_L2_NORMALIZE:
            cs = _l2_normalize_block(cs)
        feats.append(cs)

    v = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    if FINAL_FEATURE_STANDARDIZE:
        v = _final_standardize(v)
    return v


def load_image_feature(path, img_size1=IMG_SIZE_1, img_size2=IMG_SIZE_2):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        box = _crop_box(w, h, CENTER_CROP_FRAC, mode="center")
        img = img.crop(box)
        return load_image_feature_from_pil(
            img, img_size1=img_size1, img_size2=img_size2
        )


def load_image_feature_tta(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size

        feats = []
        for mode in TTA_CROPS:
            box = _crop_box(w, h, CENTER_CROP_FRAC, mode=mode)
            crop = img.crop(box)
            feats.append(load_image_feature_from_pil(crop))
        return np.mean(np.stack(feats, axis=0).astype(np.float32), axis=0)


def _stratified_subset(df, max_images, seed=42):
    rng = np.random.RandomState(seed)
    max_images = int(max_images)

    counts = df["label"].value_counts().sort_index()
    total = int(counts.sum())
    targets = {}
    remaining_budget = max_images

    for cls in range(N_CLASSES):
        n_avail = int(counts.get(cls, 0))
        if n_avail > 0 and remaining_budget > 0:
            targets[cls] = 1
            remaining_budget -= 1
        else:
            targets[cls] = 0

    if remaining_budget > 0 and total > 0:
        raw = {}
        for cls in range(N_CLASSES):
            n_avail = int(counts.get(cls, 0))
            raw[cls] = (n_avail / total) * remaining_budget

        floors = {cls: int(np.floor(raw[cls])) for cls in range(N_CLASSES)}
        fracs = {cls: raw[cls] - floors[cls] for cls in range(N_CLASSES)}
        for cls in range(N_CLASSES):
            targets[cls] += floors[cls]
        used = sum(floors.values())
        left = remaining_budget - used
        if left > 0:
            order = sorted(fracs.keys(), key=lambda c: fracs[c], reverse=True)
            for cls in order[:left]:
                targets[cls] += 1

    parts = []
    for cls in range(N_CLASSES):
        cls_df = df[df["label"] == cls]
        if len(cls_df) == 0 or targets[cls] <= 0:
            continue
        n = min(int(targets[cls]), len(cls_df))
        idx = rng.choice(cls_df.index.values, size=n, replace=False)
        parts.append(df.loc[idx])

    if len(parts) == 0:
        return df.head(0)

    sub = pd.concat(parts, axis=0)

    if len(sub) < max_images:
        remaining = df.drop(index=sub.index)
        need = min(max_images - len(sub), len(remaining))
        if need > 0:
            perm = rng.permutation(remaining.index.values)
            fill_idx = perm[:need]
            sub = pd.concat([sub, df.loc[fill_idx]], axis=0)

    return sub.sort_values("image_id").reset_index(drop=True)


def _kmeans_lloyd(X, k, iters, seed=42):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32)

    k_eff = int(min(k, n))
    init_idx = rng.choice(n, size=k_eff, replace=False)
    C = X[init_idx].astype(np.float32, copy=True)

    for _ in range(int(iters)):
        C_norm = (C * C).sum(axis=1, keepdims=True)  # (k,1)
        X_norm = (X * X).sum(axis=1, keepdims=True)  # (n,1)
        d2 = X_norm + C_norm.T - 2.0 * (X @ C.T)  # (n,k)
        a = np.argmin(d2, axis=1).astype(np.int64, copy=False)

        sums = np.zeros((k_eff, X.shape[1]), dtype=np.float32)
        np.add.at(sums, a, X)
        counts = np.bincount(a, minlength=k_eff).astype(np.int64, copy=False)

        newC = C.copy()
        nonempty = counts > 0
        if np.any(nonempty):
            newC[nonempty] = sums[nonempty] / counts[nonempty, None].astype(np.float32)

        if not np.all(nonempty):
            min_d2 = d2.min(axis=1)
            order = np.argsort(min_d2)[::-1]  # farthest first
            used = np.zeros((n,), dtype=bool)
            for j in range(k_eff):
                if nonempty[j]:
                    continue
                for idx in order:
                    if not used[idx]:
                        used[idx] = True
                        newC[j] = X[int(idx)]
                        break

        C = newC

    return C.astype(np.float32)


def _compute_dataset_norm_stats_from_stack(X: np.ndarray):
    mu = X.mean(axis=0).astype(np.float32, copy=False)
    sd = X.std(axis=0).astype(np.float32, copy=False)
    sd = np.where(sd < 1e-6, 1e-6, sd).astype(np.float32, copy=False)
    return mu, sd


def _apply_dataset_norm(x, mu, sd):
    return ((x - mu) / sd).astype(np.float32, copy=False)


def build_centroids(train_df, train_dir, max_images=MAX_TRAIN_IMAGES):
    from concurrent.futures import ThreadPoolExecutor

    sub = _stratified_subset(train_df, max_images=max_images, seed=42)

    img_ids = sub["image_id"].values
    labels = sub["label"].values.astype(np.int64, copy=False)

    def _process_one(img_id, y):
        p = os.path.join(train_dir, img_id)
        with Image.open(p) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            w, h = img.size
            box = _crop_box(w, h, CENTER_CROP_FRAC, mode="center")
            img_c = img.crop(box)

            x = load_image_feature_from_pil(img_c).astype(np.float32, copy=False)
            xs = [x]
            if AUG_FLIP_IN_TRAIN and ImageOps is not None:
                img_f = ImageOps.mirror(img_c)
                xf = load_image_feature_from_pil(img_f).astype(np.float32, copy=False)
                xs.append(xf)
            return int(y), xs

    max_workers = min(8, (os.cpu_count() or 4))
    results = []
    failures = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for img_id, y in zip(img_ids, labels):
            results.append(ex.submit(_process_one, img_id, int(y)))

        feats_by_class = {c: [] for c in range(N_CLASSES)}
        global_sum = None
        global_count = 0
        feat_dim = None
        all_train_feats_for_norm = []

        for fut in results:
            try:
                cls, xs = fut.result()
            except Exception:
                failures += 1
                continue

            for x_i in xs:
                if feat_dim is None:
                    feat_dim = int(x_i.shape[0])
                    global_sum = np.zeros((feat_dim,), dtype=np.float64)

                all_train_feats_for_norm.append(x_i)
                feats_by_class[int(cls)].append(x_i)
                global_sum += x_i.astype(np.float64, copy=False)
                global_count += 1

    if global_count == 0 or global_sum is None or feat_dim is None:
        raise RuntimeError("Could not load any training images to build centroids.")

    global_mean = (global_sum / max(1, global_count)).astype(np.float32)

    if USE_DATASET_FEATURE_NORMALIZATION:
        Xnorm = np.stack(all_train_feats_for_norm, axis=0).astype(
            np.float32, copy=False
        )
        feat_mu, feat_sd = _compute_dataset_norm_stats_from_stack(Xnorm)
    else:
        feat_mu, feat_sd = None, None

    if USE_DATASET_FEATURE_NORMALIZATION:
        for cls in range(N_CLASSES):
            if len(feats_by_class[cls]) > 0:
                feats_by_class[cls] = [
                    _apply_dataset_norm(xi, feat_mu, feat_sd)
                    for xi in feats_by_class[cls]
                ]
        global_mean = _apply_dataset_norm(global_mean, feat_mu, feat_sd)

    centroids = {}
    per_class_counts = []
    for cls in range(N_CLASSES):
        X_list = feats_by_class.get(cls, [])
        per_class_counts.append(len(X_list))
        if len(X_list) == 0:
            centroids[int(cls)] = global_mean.reshape(1, -1)
            continue
        X = np.stack(X_list, axis=0)
        C = _kmeans_lloyd(X, k=K_PER_CLASS, iters=KMEANS_ITERS, seed=42 + cls)
        if C.shape[0] == 0:
            C = global_mean.reshape(1, -1)
        centroids[int(cls)] = C

    total_centroids = int(sum(centroids[c].shape[0] for c in centroids))
    print(
        f"Built centroids using {int(global_count)} augmented samples (failures={failures}). "
        f"Feature dim={feat_dim}. Total centroids={total_centroids} ({K_PER_CLASS}/class target)."
    )
    print("Per-class augmented samples used:", per_class_counts)
    print(
        "Per-class centroids:",
        {c: int(centroids[c].shape[0]) for c in sorted(centroids)},
    )

    if USE_DATASET_FEATURE_NORMALIZATION:
        print(
            "Using dataset-level feature normalization (mean/std) computed from training subset."
        )
    else:
        print("Dataset-level feature normalization disabled.")

    return centroids, feat_mu, feat_sd


def predict_nearest_centroid(
    centroids, test_ids, test_dir, feat_mu, feat_sd, fallback_label=majority_label
):
    from concurrent.futures import ThreadPoolExecutor

    classes = sorted(centroids.keys())
    C_list = []
    L_list = []
    for c in classes:
        Ci = centroids[c]
        C_list.append(Ci)
        L_list.extend([c] * Ci.shape[0])

    C = np.vstack(C_list).astype(np.float32)
    L = np.asarray(L_list, dtype=np.int64)
    C_norm = (C * C).sum(axis=1)

    def _predict_one(img_id):
        p = os.path.join(test_dir, img_id)
        x = load_image_feature_tta(p).astype(np.float32, copy=False)

        if (
            USE_DATASET_FEATURE_NORMALIZATION
            and feat_mu is not None
            and feat_sd is not None
        ):
            x = _apply_dataset_norm(x, feat_mu, feat_sd)

        x_norm = float((x * x).sum())
        dots = C @ x
        d2 = x_norm + C_norm - 2.0 * dots
        return int(L[int(np.argmin(d2))])

    max_workers = min(8, (os.cpu_count() or 4))
    preds = [int(fallback_label)] * len(test_ids)
    failures = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_predict_one, img_id) for img_id in test_ids]
        for i, fut in enumerate(futs):
            try:
                preds[i] = int(fut.result())
            except Exception:
                failures += 1
                preds[i] = int(fallback_label)

    print(
        f"Predicted {len(preds)} test images (failures={failures}, used fallback={fallback_label} for those)."
    )
    return preds


if USE_IMAGES:
    centroids, feat_mu, feat_sd = build_centroids(
        train_df, train_image_dir, max_images=MAX_TRAIN_IMAGES
    )
    pred_labels = predict_nearest_centroid(
        centroids=centroids,
        test_ids=sample_csv["image_id"].values.tolist(),
        test_dir=test_image_dir,
        feat_mu=feat_mu,
        feat_sd=feat_sd,
        fallback_label=majority_label,
    )
else:
    pred_labels = np.full(
        shape=(len(sample_csv),), fill_value=majority_label, dtype=np.int64
    ).tolist()

print("Predictions:", len(pred_labels))



## === cell 3
submission_df = sample_csv.copy()
submission_df["label"] = pred_labels

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)

assert submission_df.shape[0] == sample_csv.shape[0]
assert list(submission_df.columns) == ["image_id", "label"]
assert submission_path.endswith(".csv")
