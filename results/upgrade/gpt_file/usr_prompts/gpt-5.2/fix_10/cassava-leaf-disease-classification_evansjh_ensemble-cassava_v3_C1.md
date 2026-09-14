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

0.35501

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the environment import/runtime crash by avoiding TensorFlow/Keras model loading (which is currently failing due to a protobuf/TensorFlow incompatibility) and instead produce a valid submission using the provided `sample_submission.csv` as the ordered key. I also fix the missing external model paths issue by removing dependencies on `/kaggle/input/combinedmodel3/...` which are not present. To still move accuracy toward the target while keeping logic simple and stable, I generate predictions using the training-label prior (majority-class baseline), which is a legitimate, fast, and usually stronger-than-random approach for this dataset. Finally, I guarantee the submission has the exact required columns, row count, and ordering, and write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.40321) has done: 'To move accuracy up toward your target without changing the overall “no-TF, fast baseline” approach, I replace the single majority-class guess with a slightly stronger label-prior baseline that samples labels according to the training distribution. This keeps the same core logic (using only `train.csv` label statistics and `sample_submission.csv` ordering), still produces a valid `submission.csv`, and typically improves over predicting one constant class on Cassava. I also make the sampling deterministic via a fixed seed for stability across runs. No model/feature extraction/training loop is introduced, so runtime stays very low.'
- What this solution (achieved 0.13602) has done: 'Your current approach samples labels from the training prior, which has an expected accuracy around \(\sum p_k^2\) (roughly ~0.3–0.4 here), so it cannot reach the 0.89 target. To move the score upward while keeping the “no deep model / no TF” core idea, the smallest legitimate improvement is to switch from random prior-sampling to a deterministic “leak-free lookup” based on simple image statistics: compute per-class average RGB (and brightness/std) on the training images, then classify each test image by nearest class prototype. This preserves the lightweight, non-neural approach and runs fast enough by resizing images and limiting features. The submission format and ordering still be taken from `sample_submission.csv`, and we keep deterministic behavior for stability.'
- What this solution (achieved 0.19432) has done: 'Your current prototype classifier is likely underperforming because raw mean/std RGB features are weak and also not normalized, so distance is dominated by high-variance dimensions and class imbalance. I keep the exact same “compute per-class prototypes on train then nearest-prototype on test” core logic, but (1) add per-feature standardization (z-score) computed from the training features you already extract, and (2) use class-count weighted shrinkage toward the global mean prototype to reduce noise for smaller classes. These are minimal, fast changes that usually improve nearest-centroid accuracy without introducing any new modeling framework. The submission generation, ordering (from `sample_submission.csv`), and file path remain unchanged.'
- What this solution (achieved 0.23804) has done: 'Your current nearest-prototype pipeline is likely missing the biggest accuracy lever for Cassava: keeping spatial structure. I keep the same “extract features → compute per-class prototypes → nearest prototype on test → write submission” core logic, but change the feature vector to be a slightly richer (still lightweight) grid-pooled RGB mean/std descriptor so disease patterns aren’t averaged away. I also switch the distance to cosine distance in standardized space, which is a minimal change that often improves nearest-centroid classification when magnitude varies. Everything else (sampling per class, shrinkage, ordering via `sample_submission.csv`, deterministic seed, and writing `/kaggle/working/submission.csv`) stays the same.'
- What this solution (achieved 0.38864) has done: 'Your current score (0.23804) is far below the target (0.8906), so we need a real accuracy lift while keeping the same overall “extract features → compute per-class prototypes → nearest prototype on test → submission.csv” pipeline. The minimal high-impact change is to replace the weak grid RGB-stat features with a compact HOG-like gradient-orientation histogram descriptor (still just hand-crafted features + nearest-prototype) which captures leaf texture/lesion patterns much better without changing the modeling approach. I also switch cosine similarity to standardized Euclidean distance (in the same z-scored space you already compute), which is typically a better fit for histogram-style descriptors and remains the same nearest-prototype semantics. All paths, submission ordering (from sample_submission.csv), shrinkage, and deterministic sampling remain.'
- What this solution (achieved 0.36547) has done: 'I keep your exact pipeline (hand-crafted features → per-class prototypes with shrinkage → z-score → nearest-prototype) but make two minimal, high-signal corrections that typically improve accuracy: (1) compute feature standardization (mean/std) from the same training features actually used to build prototypes (instead of using a global mean over a class-balanced subsample), and (2) add a tiny “power normalization + L2” on the HOG part to reduce bursty bins and make distances more stable. These are small, metric-aligned changes that don’t alter the overall approach, don’t add any new dependencies, and should move accuracy upward from 0.3886 toward your target. The submission ordering/format and output path remain unchanged, and the script still runs end-to-end producing `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.34492) has done: 'Your current gap to the target is large (0.36547 vs 0.8906), so we need a real lift while keeping the same overall pipeline (hand-crafted feature extraction → per-class prototypes → z-score → nearest-prototype). The most likely blocker is that the gradient/HOG part is computed on raw RGB-to-gray without any contrast normalization, making it sensitive to lighting and camera exposure; I add a very small, standard HOG pre-step (local contrast normalization on the grayscale) that keeps the same descriptor structure but improves invariance. I also make the HOG histogram “soft-binned” between neighboring angle bins (still the same bins/grid, just less quantization error), which usually gives a noticeable accuracy bump for nearest-centroid classifiers. Everything else (sampling per class, shrinkage, z-scoring, nearest-prototype, submission format/path) remains unchanged.'
- What this solution (achieved 0.35501) has done: 'Your current score (0.34492) is far below the target (0.8906), so we should increase accuracy with the smallest, safest changes that keep the same pipeline (hand-crafted features → per-class prototypes → z-score → nearest prototype). The biggest likely issue is that the current HOG-like descriptor uses unsigned gradients poorly and is very sensitive to noise; I make a minimal HOG correction by switching to the standard **unsigned** orientation range \([0,\pi)\) and adding a tiny **magnitude smoothing** (box blur) before histogramming, which usually stabilizes nearest-centroid decisions. I also add a light **per-cell normalization with small epsilon** (keeping your existing power+L2) to reduce instability from small/flat regions. All paths, sampling/shrinkage, z-scoring, nearest-prototype logic, and submission writing remain unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image

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




## === cell 1
def _box_blur_2d(x: np.ndarray, k: int = 7) -> np.ndarray:
    """
    Used for local contrast normalization and (minimal score-improving change)
    a small magnitude smoothing before orientation histogramming to reduce noise sensitivity.
    """
    k = int(k)
    if k <= 1:
        return x
    r = k // 2
    xp = np.pad(x, ((r, r), (r, r)), mode="reflect").astype(np.float32)
    ii = np.cumsum(np.cumsum(xp, axis=0), axis=1)
    ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    H, W = x.shape
    y0, x0 = 0, 0
    y1, x1 = k, k
    s = (
        ii[y1 : y1 + H, x1 : x1 + W]
        - ii[y0 : y0 + H, x1 : x1 + W]
        - ii[y1 : y1 + H, x0 : x0 + W]
        + ii[y0 : y0 + H, x0 : x0 + W]
    )
    return s / float(k * k)


def extract_features(
    image_path: str, size=(128, 128), grid=(4, 4), bins=9
) -> np.ndarray:
    """
    Same core pipeline: HOG-like grid histogram + extras -> z-score -> nearest-prototype.
    Minimal changes to increase accuracy toward target:
    - Use standard *unsigned* gradient orientations [0, pi) (more appropriate for edges/textures).
    - Lightly smooth gradient magnitude before histogramming (reduces speckle/noise effects).
    - Keep existing soft-binning, per-cell normalization, and power+L2 on HOG.
    """
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im = im.resize(size, resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0

    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(
        np.float32
    )

    mu = _box_blur_2d(gray, k=7)
    mu2 = _box_blur_2d(gray * gray, k=7)
    var = np.maximum(mu2 - mu * mu, 1e-6)
    gray_n = (gray - mu) / np.sqrt(var)

    H, W = gray_n.shape

    gx = gray_n[:, 2:] - gray_n[:, :-2]  # H, W-2
    gy = gray_n[2:, :] - gray_n[:-2, :]  # H-2, W

    gx = gx[1:-1, :]  # H-2, W-2
    gy = gy[:, 1:-1]  # H-2, W-2

    mag = np.sqrt(gx * gx + gy * gy) + 1e-8

    mag = _box_blur_2d(mag.astype(np.float32), k=3)

    ang = np.mod(np.arctan2(gy, gx), np.pi).astype(np.float32)  # [0, pi)

    ang_f = ang * (bins / np.pi)  # [0,bins)
    b0 = np.floor(ang_f).astype(np.int32)
    b0 = np.clip(b0, 0, bins - 1)
    b1 = b0 + 1
    b1 = np.where(b1 >= bins, 0, b1)  # wrap-around
    w1 = (ang_f - b0.astype(np.float32)).astype(np.float32)
    w0 = (1.0 - w1).astype(np.float32)

    gh, gw = grid
    h_step = (H - 2) // gh
    w_step = (W - 2) // gw

    hists = []
    for r in range(gh):
        for c in range(gw):
            y0 = r * h_step
            y1 = (r + 1) * h_step if r < gh - 1 else (H - 2)
            x0 = c * w_step
            x1 = (c + 1) * w_step if c < gw - 1 else (W - 2)

            b0p = b0[y0:y1, x0:x1].ravel()
            b1p = b1[y0:y1, x0:x1].ravel()
            m = mag[y0:y1, x0:x1].ravel()
            w0p = w0[y0:y1, x0:x1].ravel()
            w1p = w1[y0:y1, x0:x1].ravel()

            hist = np.zeros((bins,), dtype=np.float32)
            np.add.at(hist, b0p, m * w0p)
            np.add.at(hist, b1p, m * w1p)

            hist /= max(float(np.linalg.norm(hist)), 1e-6)
            hists.append(hist)

    hog = np.concatenate(hists).astype(np.float32)

    hog = np.sqrt(np.maximum(hog, 0.0)).astype(np.float32)
    hog /= max(float(np.linalg.norm(hog)), 1e-8)

    extra = np.array([float(gray_n.mean()), float(gray_n.std())], dtype=np.float32)
    return np.concatenate([hog, extra]).astype(np.float32)


labels_sorted = np.sort(train_df["label"].unique())
label_to_idx = {int(l): i for i, l in enumerate(labels_sorted)}

MAX_PER_CLASS = 1200  # keep same sampling approach
rng = np.random.default_rng(20240101)

tmp_dim = extract_features(
    os.path.join(TRAIN_IMG_DIR, train_df["image_id"].iloc[0])
).shape[0]

prototypes = np.zeros((len(labels_sorted), tmp_dim), dtype=np.float32)
counts = np.zeros((len(labels_sorted),), dtype=np.int32)

per_class_sum = np.zeros((len(labels_sorted), tmp_dim), dtype=np.float64)
per_class_sumsq = np.zeros((len(labels_sorted), tmp_dim), dtype=np.float64)
per_class_n = np.zeros((len(labels_sorted),), dtype=np.int64)

for lbl in labels_sorted:
    sub = train_df[train_df["label"] == lbl]["image_id"].to_numpy()
    if len(sub) > MAX_PER_CLASS:
        sub = rng.choice(sub, size=MAX_PER_CLASS, replace=False)

    idx = label_to_idx[int(lbl)]
    acc = np.zeros((tmp_dim,), dtype=np.float64)
    n = 0
    for image_id in sub:
        pth = os.path.join(TRAIN_IMG_DIR, image_id)
        if not os.path.exists(pth):
            continue
        try:
            f = extract_features(pth).astype(np.float64)
        except Exception:
            continue

        acc += f
        n += 1

        per_class_sum[idx] += f
        per_class_sumsq[idx] += f * f
        per_class_n[idx] += 1

    if n == 0:
        raise RuntimeError(f"No training images could be read for label {lbl}.")
    prototypes[idx] = (acc / n).astype(np.float32)
    counts[idx] = n

if int(per_class_n.sum()) == 0:
    raise RuntimeError("No training images could be read to compute normalization.")

train_counts_full = train_df["label"].value_counts().to_dict()
weights = np.array(
    [float(train_counts_full.get(int(l), 0)) for l in labels_sorted], dtype=np.float64
)
weights = np.maximum(weights, 1.0)  # safety
class_mean = per_class_sum / np.maximum(per_class_n[:, None].astype(np.float64), 1.0)
class_var = (
    per_class_sumsq / np.maximum(per_class_n[:, None].astype(np.float64), 1.0)
    - class_mean * class_mean
)
class_var = np.maximum(class_var, 1e-8)

w_sum = float(weights.sum())
global_mean = (weights[:, None] * class_mean).sum(axis=0) / w_sum
global_var = (weights[:, None] * class_var).sum(axis=0) / w_sum + (
    (weights[:, None] * (class_mean - global_mean) ** 2).sum(axis=0) / w_sum
)
global_std = np.sqrt(np.maximum(global_var, 1e-8)).astype(np.float32)
global_mean = global_mean.astype(np.float32)

tau = 200.0  # keep shrinkage idea unchanged
alpha = (counts.astype(np.float32) / (counts.astype(np.float32) + tau))[:, None]
prototypes = alpha * prototypes + (1.0 - alpha) * global_mean[None, :]

prototypes_z = (prototypes - global_mean[None, :]) / global_std[None, :]

print(
    "Built prototypes with per-class counts:",
    {int(l): int(counts[label_to_idx[int(l)]]) for l in labels_sorted},
)
print("Prototype matrix shape:", prototypes.shape)
print(
    "Using HOG-like grid descriptor (unsigned orientations) + local contrast norm + magnitude smoothing + soft-binning + power+L2 on HOG + feature standardization (freq-weighted) + prototype shrinkage (tau=200) + nearest-prototype."
)



## === cell 2
submission_df = sample_df[["image_id"]].copy()
preds = np.empty((len(submission_df),), dtype=np.int32)

fallback_label = int(train_df["label"].value_counts().idxmax())

for i, image_id in enumerate(submission_df["image_id"].to_numpy()):
    pth = os.path.join(TEST_IMG_DIR, image_id)
    if not os.path.exists(pth):
        preds[i] = fallback_label
        continue
    try:
        f = extract_features(pth).astype(np.float32)
        fz = (f - global_mean) / global_std

        diffs = prototypes_z - fz[None, :]
        d2 = np.sum(diffs * diffs, axis=1)
        preds[i] = int(labels_sorted[int(np.argmin(d2))])
    except Exception:
        preds[i] = fallback_label

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



## === cell 3
submission_df.tail()
