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

0.42937

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
- What this solution (achieved 0.35015) has done: 'Your current gap to the target is large (0.35501 vs 0.89060), so we need to increase accuracy, but with minimal changes that keep your exact pipeline (hand-crafted features → per-class prototypes with shrinkage → z-score → nearest-prototype). The smallest high-impact adjustment is to make the descriptor more robust to illumination/color shifts by adding a simple per-image color normalization (channel-wise standardization) and appending low-cost color features (mean/std in RGB and HSV) while keeping your HOG-like part unchanged. To reduce noise in prototypes without changing the training approach, I also switch the prototype computation from a plain mean to a per-feature **median** (robust centroid) over the same sampled features, which is still “prototype + nearest centroid” but typically more stable. Everything else (sampling per class, shrinkage with tau=200, z-scoring from training features, nearest-prototype prediction, submission ordering/format/path) remains unchanged.'
- What this solution (achieved 0.49589) has done: 'Your current nearest-prototype pipeline is already coherent, but it’s likely leaving accuracy on the table due to prototype estimation being inconsistent with the later z-scoring/shrinkage and due to class imbalance not being accounted for at prediction time. I keep the exact core logic (hand-crafted feature extraction → per-class prototype → z-score → nearest-prototype) and make two minimal, metric-aligned changes: (1) compute prototypes as the **mean of the same shrinked representation you actually use** (instead of median-before-shrink), and (2) apply a small **class-prior correction** in the distance (MAP under equal-covariance Gaussian), which often improves accuracy for imbalanced multi-class classification without changing the model family. I also keep everything deterministic and preserve the submission ordering from `sample_submission.csv` and the same output path.'
- What this solution (achieved 0.50336) has done: 'I keep your exact pipeline (hand-crafted feature extraction → per-class prototypes with shrinkage → z-score → nearest-prototype with class-prior correction) and only make small, score-relevant fixes that typically lift accuracy: (1) align the prototype computation with the exact representation you use at inference by building prototypes in z-scored space (instead of shrinking in raw space then z-scoring), and (2) add a tiny per-feature whitening floor and a slightly adjusted shrinkage strength so prototypes are less noisy without changing the approach. I also switch the training feature aggregation to be numerically consistent (float64 accumulation, float32 storage stays) and make inference use the same standardized prototype space directly (no semantic change). These changes are minimal, deterministic, and should move the score upward from 0.4959 toward your 0.8906 target while still finishing quickly and writing a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.52578) has done: 'Your current score (0.50336) is far below the target (0.89060), so we should improve accuracy with the smallest changes that keep the same overall pipeline: hand-crafted features → per-class prototypes with shrinkage → z-score → nearest-prototype. The most likely “easy” gain without changing the approach is to (1) add a tiny amount of rotation/flip augmentation when computing class means (prototypes become more invariant), and (2) tune the prior correction strength and shrinkage strength slightly using a small deterministic validation split from the already-extracted training features (no change to the inference method, just choosing better constants). This keeps the same model family and semantics, but makes prototypes and the MAP-like distance correction better aligned with held-out accuracy. The submission format, ordering (from `sample_submission.csv`), and output path remain unchanged, and runtime stays within limits by reusing extracted features and only augmenting a small subset per class.'
- What this solution (achieved 0.52317) has done: 'Your current score (0.52578) is far below the 0.8906 target, so we should push accuracy up with the smallest changes that keep your same pipeline: hand-crafted features → per-class prototypes (with shrinkage) → z-score → nearest-prototype (+ prior correction). The most likely low-risk gain is fixing a bug in prototype shrinkage: right now you multiply the *z-scored* class means by `alpha` but never add the `(1-alpha)*global_mean` term (in raw space), which unintentionally shrinks prototypes toward the zero vector rather than toward the global mean—this can noticeably hurt nearest-prototype accuracy. I keep your feature extractor, sampling, validation tuning, and scoring intact, but compute prototypes by shrinking in raw feature space toward the global mean, then mapping into the same z-space used at inference. I also make the augmentation choice include the identity (0/1/2) deterministically, since your comment says 3 options but the code only sampled 0..1.'
- What this solution (achieved 0.42937) has done: 'Your current score (0.523) is far below the target (0.8906), so we should increase accuracy with the smallest safe changes that keep the same “hand-crafted features → per-class prototypes → z-score → nearest-prototype (+ prior correction)” pipeline. The main low-risk lever is to make prototype estimation less noisy and less sensitive to the single random augmentation choice by (1) averaging multiple deterministic augmentations per training image when building class means (still the same features, just more stable prototypes), and (2) slightly increasing per-class sample size since runtime is dominated by test inference and you’re currently leaving training signal unused. I keep the feature extractor, distance, prior correction, and submission format/paths unchanged. I also make the augmentation selection deterministic (identity/flip/rotate90 averaged) so results are stable and typically improve accuracy.'

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
    Used for local contrast normalization and magnitude smoothing before histogramming.
    """
    k = int(k)
    if k <= 1:
        return x
    r = k // 2
    xp = np.pad(x, ((r, r), (r, r)), mode="reflect").astype(np.float32)
    ii = np.cumsum(np.cumsum(xp, axis=0), axis=1)
    ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    H, W = x.shape
    s = (
        ii[k : k + H, k : k + W]
        - ii[0:H, k : k + W]
        - ii[k : k + H, 0:W]
        + ii[0:H, 0:W]
    )
    return s / float(k * k)


def _rgb_to_hsv_fast(rgb01: np.ndarray) -> np.ndarray:
    """
    Minimal, dependency-free RGB->HSV (all in [0,1]).
    rgb01: float32 array (H,W,3) in [0,1]
    """
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    cmax = np.maximum(r, np.maximum(g, b))
    cmin = np.minimum(r, np.minimum(g, b))
    delta = cmax - cmin

    h = np.zeros_like(cmax, dtype=np.float32)
    nonzero = delta > 1e-8

    mask = nonzero & (cmax == r)
    h[mask] = ((g[mask] - b[mask]) / (delta[mask] + 1e-12)) % 6.0
    mask = nonzero & (cmax == g)
    h[mask] = ((b[mask] - r[mask]) / (delta[mask] + 1e-12)) + 2.0
    mask = nonzero & (cmax == b)
    h[mask] = ((r[mask] - g[mask]) / (delta[mask] + 1e-12)) + 4.0
    h = (h / 6.0).astype(np.float32)

    s = np.zeros_like(cmax, dtype=np.float32)
    s[cmax > 1e-8] = (delta[cmax > 1e-8] / (cmax[cmax > 1e-8] + 1e-12)).astype(
        np.float32
    )

    v = cmax.astype(np.float32)

    return np.stack([h, s, v], axis=-1).astype(np.float32)


def extract_features(
    image_path: str, size=(128, 128), grid=(4, 4), bins=9
) -> np.ndarray:
    """
    Same core pipeline: HOG-like grid histogram + extras -> z-score -> nearest-prototype.
    """
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im = im.resize(size, resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)

    mu_c = arr.reshape(-1, 3).mean(axis=0).astype(np.float32)
    std_c = arr.reshape(-1, 3).std(axis=0).astype(np.float32)
    arr_n = (arr - mu_c[None, None, :]) / (std_c[None, None, :] + 1e-6)

    gray = (
        0.2989 * arr_n[..., 0] + 0.5870 * arr_n[..., 1] + 0.1140 * arr_n[..., 2]
    ).astype(np.float32)

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

    rgb_mean = arr.reshape(-1, 3).mean(axis=0).astype(np.float32)
    rgb_std = arr.reshape(-1, 3).std(axis=0).astype(np.float32)
    hsv = _rgb_to_hsv_fast(arr)
    hsv_mean = hsv.reshape(-1, 3).mean(axis=0).astype(np.float32)
    hsv_std = hsv.reshape(-1, 3).std(axis=0).astype(np.float32)

    extra_gray = np.array([float(gray_n.mean()), float(gray_n.std())], dtype=np.float32)

    extra = np.concatenate([extra_gray, rgb_mean, rgb_std, hsv_mean, hsv_std]).astype(
        np.float32
    )
    return np.concatenate([hog, extra]).astype(np.float32)


def _extract_features_aug(image_path: str, aug_code: int) -> np.ndarray:
    """
    Same extractor, with tiny geometric aug options (used only for prototype building).
    aug_code:
      0: identity
      1: horizontal flip
      2: rotate 90
    """
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        if aug_code == 1:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        elif aug_code == 2:
            im = im.transpose(Image.ROTATE_90)
        im = im.resize((128, 128), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0

    mu_c = arr.reshape(-1, 3).mean(axis=0).astype(np.float32)
    std_c = arr.reshape(-1, 3).std(axis=0).astype(np.float32)
    arr_n = (arr - mu_c[None, None, :]) / (std_c[None, None, :] + 1e-6)

    gray = (
        0.2989 * arr_n[..., 0] + 0.5870 * arr_n[..., 1] + 0.1140 * arr_n[..., 2]
    ).astype(np.float32)

    mu = _box_blur_2d(gray, k=7)
    mu2 = _box_blur_2d(gray * gray, k=7)
    var = np.maximum(mu2 - mu * mu, 1e-6)
    gray_n = (gray - mu) / np.sqrt(var)

    H, W = gray_n.shape
    gx = gray_n[:, 2:] - gray_n[:, :-2]
    gy = gray_n[2:, :] - gray_n[:-2, :]
    gx = gx[1:-1, :]
    gy = gy[:, 1:-1]

    mag = np.sqrt(gx * gx + gy * gy) + 1e-8
    mag = _box_blur_2d(mag.astype(np.float32), k=3)
    ang = np.mod(np.arctan2(gy, gx), np.pi).astype(np.float32)

    bins = 9
    ang_f = ang * (bins / np.pi)
    b0 = np.floor(ang_f).astype(np.int32)
    b0 = np.clip(b0, 0, bins - 1)
    b1 = b0 + 1
    b1 = np.where(b1 >= bins, 0, b1)
    w1 = (ang_f - b0.astype(np.float32)).astype(np.float32)
    w0 = (1.0 - w1).astype(np.float32)

    gh, gw = (4, 4)
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

    rgb_mean = arr.reshape(-1, 3).mean(axis=0).astype(np.float32)
    rgb_std = arr.reshape(-1, 3).std(axis=0).astype(np.float32)
    hsv = _rgb_to_hsv_fast(arr)
    hsv_mean = hsv.reshape(-1, 3).mean(axis=0).astype(np.float32)
    hsv_std = hsv.reshape(-1, 3).std(axis=0).astype(np.float32)
    extra_gray = np.array([float(gray_n.mean()), float(gray_n.std())], dtype=np.float32)
    extra = np.concatenate([extra_gray, rgb_mean, rgb_std, hsv_mean, hsv_std]).astype(
        np.float32
    )
    return np.concatenate([hog, extra]).astype(np.float32)


labels_sorted = np.sort(train_df["label"].unique())
label_to_idx = {int(l): i for i, l in enumerate(labels_sorted)}

MAX_PER_CLASS = 1600

rng = np.random.default_rng(20240101)

tmp_dim = extract_features(
    os.path.join(TRAIN_IMG_DIR, train_df["image_id"].iloc[0])
).shape[0]

counts = np.zeros((len(labels_sorted),), dtype=np.int32)
per_class_sum = np.zeros((len(labels_sorted), tmp_dim), dtype=np.float64)
per_class_sumsq = np.zeros((len(labels_sorted), tmp_dim), dtype=np.float64)
per_class_n = np.zeros((len(labels_sorted),), dtype=np.int64)

VAL_FRACTION = 0.10
val_features = []
val_labels = []

AUG_CODES_FOR_PROTOTYPE = (0, 1, 2)

for lbl in labels_sorted:
    sub = train_df[train_df["label"] == lbl]["image_id"].to_numpy()
    if len(sub) > MAX_PER_CLASS:
        sub = rng.choice(sub, size=MAX_PER_CLASS, replace=False)

    n_sub = len(sub)
    n_val = max(1, int(round(n_sub * VAL_FRACTION)))
    perm = rng.permutation(n_sub)
    val_ids = sub[perm[:n_val]]
    tr_ids = sub[perm[n_val:]]

    idx = label_to_idx[int(lbl)]

    for image_id in tr_ids:
        pth = os.path.join(TRAIN_IMG_DIR, image_id)
        if not os.path.exists(pth):
            continue
        try:
            f_acc = None
            for ac in AUG_CODES_FOR_PROTOTYPE:
                f = _extract_features_aug(pth, aug_code=int(ac)).astype(np.float64)
                f_acc = f if f_acc is None else (f_acc + f)
            f64 = (f_acc / float(len(AUG_CODES_FOR_PROTOTYPE))).astype(np.float64)
        except Exception:
            continue

        per_class_sum[idx] += f64
        per_class_sumsq[idx] += f64 * f64
        per_class_n[idx] += 1

    for image_id in val_ids:
        pth = os.path.join(TRAIN_IMG_DIR, image_id)
        if not os.path.exists(pth):
            continue
        try:
            f = extract_features(pth).astype(np.float32)
        except Exception:
            continue
        val_features.append(f)
        val_labels.append(int(lbl))

    if int(per_class_n[idx]) == 0:
        raise RuntimeError(f"No training images could be read for label {lbl}.")
    counts[idx] = int(per_class_n[idx])

if int(per_class_n.sum()) == 0:
    raise RuntimeError("No training images could be read to compute normalization.")

val_features = np.asarray(val_features, dtype=np.float32)
val_labels = np.asarray(val_labels, dtype=np.int32)
print("Validation set size (for tuning constants):", val_features.shape[0])

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

global_std = np.sqrt(np.maximum(global_var, 1e-6)).astype(np.float32)
global_mean = global_mean.astype(np.float32)

class_mean_f32 = class_mean.astype(np.float32)

priors = (weights / float(weights.sum())).astype(np.float32)
log_priors = np.log(np.maximum(priors, 1e-12)).astype(np.float32)


def _predict_from_prototypes(
    val_f: np.ndarray, prototypes_z: np.ndarray, prior_lambda: float
) -> np.ndarray:
    fz = (val_f - global_mean[None, :]) / global_std[None, :]
    d2 = ((prototypes_z[None, :, :] - fz[:, None, :]) ** 2).sum(axis=2)
    score = d2 - (2.0 * float(prior_lambda) * log_priors[None, :])
    return labels_sorted[np.argmin(score, axis=1)].astype(np.int32)


TAU_CANDIDATES = [250.0, 350.0, 500.0]
PRIOR_LAMBDA_CANDIDATES = [0.0, 0.5, 1.0, 1.5]

best_acc = -1.0
best_tau = 350.0
best_pl = 1.0

if val_features.shape[0] >= 50:
    for tau in TAU_CANDIDATES:
        alpha = (counts.astype(np.float32) / (counts.astype(np.float32) + float(tau)))[
            :, None
        ]
        proto_raw = alpha * class_mean_f32 + (1.0 - alpha) * global_mean[None, :]
        prototypes_z_c = (proto_raw - global_mean[None, :]) / global_std[None, :]

        for pl in PRIOR_LAMBDA_CANDIDATES:
            pred = _predict_from_prototypes(
                val_features, prototypes_z_c, prior_lambda=pl
            )
            acc = float((pred == val_labels).mean())
            if acc > best_acc:
                best_acc = acc
                best_tau = float(tau)
                best_pl = float(pl)
    print(
        f"Tuned on val: best_acc={best_acc:.4f}, tau={best_tau}, PRIOR_LAMBDA={best_pl}"
    )
else:
    print("Val set too small; using defaults tau=350, PRIOR_LAMBDA=1.0")

tau = best_tau
alpha = (counts.astype(np.float32) / (counts.astype(np.float32) + float(tau)))[:, None]
proto_raw = alpha * class_mean_f32 + (1.0 - alpha) * global_mean[None, :]
prototypes_z = (proto_raw - global_mean[None, :]) / global_std[None, :]

PRIOR_LAMBDA = best_pl

print(
    "Built prototypes with per-class counts:",
    {int(l): int(counts[label_to_idx[int(l)]]) for l in labels_sorted},
)
print("Prototype matrix shape (z-space):", prototypes_z.shape)
print(
    f"Using HOG-like grid descriptor + local contrast norm + magnitude smoothing + soft-binning + power+L2 on HOG + added RGB/HSV stats + feature standardization (freq-weighted) + prototype shrinkage toward global mean (tau={tau}) + nearest-prototype (+ class-prior correction PRIOR_LAMBDA={PRIOR_LAMBDA})."
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

        score = d2 - (2.0 * float(PRIOR_LAMBDA) * log_priors)
        preds[i] = int(labels_sorted[int(np.argmin(score))])
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
