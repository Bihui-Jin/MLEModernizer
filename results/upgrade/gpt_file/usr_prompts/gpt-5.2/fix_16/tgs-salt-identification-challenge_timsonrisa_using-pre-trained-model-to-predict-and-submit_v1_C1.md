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
Segment regions of salt in seismic images.

## Metric
Mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:

$$\text{IoU}(A, B)=\frac{A \cap B}{A \cup B}$$

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: `(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)`. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value 𝑡t, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

$$\frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold:

$$\frac{1}{\mid \text { thresholds } \mid} \sum_t \frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

## Submission Format
Use run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed\
and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.

The file should contain a header and have the following format. Each row in your submission represents a single predicted salt segmentation for the given image.

```
id,rle_mask
3e06571ef3,1 1
a51b08d882,1 1
c32590b06f,1 1
etc.
```

## Dataset
The data is a set of images chosen at various locations chosen at random in the subsurface. The images are 101 x 101 pixels and each pixel is classified as either salt or sediment. In addition to the seismic images, the depth of the imaged location is provided for each image.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
        input/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
        working/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
```

-> data/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/tgs-salt-identification-challenge/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> data/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> input/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> (stopped after 10 files for performance)

# 5. Target score

0.3265599307659023

# 6. Current score

0.0458

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.026) has done: 'I fix the environment-breaking TensorFlow/Keras import conflict (the protobuf `MessageFactory` error) by using `tf.keras` consistently and setting a safe protobuf implementation before importing TensorFlow. Next, since the referenced external model directory doesn’t exist in your provided `/kaggle/input` paths, I keep the same overall “load model → predict → threshold → RLE” pipeline but replace the missing model with a tiny, deterministic baseline segmentation (derived directly from the input image) so the notebook runs end-to-end and writes a valid `submission_01.csv`. I also make the paths robust to the two common Kaggle dataset locations shown in your tree and ensure the submission is aligned to `sample_submission.csv` IDs. These changes are minimal, unblock execution, and should yield a non-empty, valid submission (and typically scores above the absolute floor).'
- What this solution (achieved 0.026) has done: 'I fix the TensorFlow/protobuf crash by removing the unnecessary TensorFlow/Keras imports (they aren’t used for your current baseline inference) and by rewriting the unused `mean_iou` helper so it no longer relies on deprecated `tf.compat.v1` session APIs. Then I keep the exact same end-to-end pipeline (load test images → simple threshold mask → resize back → RLE → write CSV) but make one minimal scoring-oriented correction: use the competition’s native 101×101 resolution instead of resizing to 128×128, which avoids interpolation artifacts and typically improves IoU-based metrics. Finally, I keep submission ID alignment with `sample_submission.csv` and ensure the output `submission_01.csv` is always written with the required columns.'
- What this solution (achieved 0.0679) has done: 'Your current score (0.026) is far below the target (0.3266), so we should make a small, legitimate improvement without changing the overall “load test → predict mask → RLE → write CSV” pipeline. The biggest issue is that the baseline thresholding is not tuned to the competition’s IoU-swept metric; a minimal, safe improvement is to (1) switch to Otsu thresholding per-image (a standard deterministic choice) and (2) apply tiny morphological cleanup (remove very small objects/holes) to reduce spurious speckle that hurts precision across IoU thresholds. These changes keep the same core logic (simple image-derived binary masks) while typically increasing mean AP substantially compared to a raw mean+std threshold. All paths and submission formatting are preserved, and the script still writes `submission_01.csv`.'
- What this solution (achieved 0.0501) has done: 'Your current 0.0679 is far below the 0.3266 target, so we should make a small but meaningful inference-only improvement without changing the overall “read test images → derive binary mask from image → light cleanup → RLE → write CSV” pipeline. The most direct gain is to tune the binarization step for salt-segmentation: keep Otsu, but apply it to a lightly denoised grayscale image and add a small bias toward predicting salt (salt is typically darker), which tends to improve recall and thus mAP under IoU sweeps. To avoid harming precision with speckle, we keep the same morphological cleanup but make it slightly more consistent by also applying a gentle binary closing/opening with a tiny disk. All paths, output format, and runtime remain within Kaggle constraints, and it still writes `submission_01.csv`.'
- What this solution (achieved 0.0427) has done: 'We keep your exact “image → Otsu threshold → tiny morphology cleanup → RLE” pipeline, but adjust two inference-only knobs that commonly move this baseline meaningfully upward on TGS: (1) apply the threshold on a lightly contrast-normalized grayscale (robust to varying illumination), and (2) add a very small depth-based bias using `depths.csv` (salt likelihood correlates with depth), which nudges the binarization without changing the method. These are minimal semantic changes (still deterministic thresholding + morphology) and should improve recall/precision balance vs a fixed global bias, moving the score toward your 0.3266 target. Submission formatting, ID alignment with `sample_submission.csv`, and runtime constraints are preserved.'
- What this solution (achieved 0.0439) has done: 'Your score is far below the target, so we keep your exact “Otsu threshold → small morphology cleanup → RLE” core pipeline and only adjust a couple of inference knobs that typically improve mAP on TGS without changing the approach. First, we reduce systematic false-positives by adding a simple “empty-mask gate” based on the predicted salt coverage fraction (many test images contain no salt), which usually boosts precision across IoU thresholds. Second, we very slightly retune the bias and morphology thresholds to better balance recall vs precision after adding that gate, while keeping everything deterministic and fast. The submission writing and ID alignment stay identical and it still produce `submission_01.csv` end-to-end.'
- What this solution (achieved 0.0445) has done: 'We keep your exact inference pipeline (grayscale → normalize → Gaussian → Otsu(+bias,+depth) → morphology → empty-mask gate → RLE) and only tune the smallest knobs that most directly affect mAP: the empty-mask gate and the binarization bias. Your current score is far below target, so we cautiously increase recall (without blowing up false positives) by slightly relaxing the empty-mask gate and very slightly shifting the threshold bias while keeping the same semantics. To avoid regressions from the depth nudge, we clamp its influence more tightly (same logic, just prevents extreme shifts). The script still runs end-to-end and writes a valid `submission_01.csv` with correct ID alignment.'
- What this solution (achieved 0.0445) has done: 'Your current score (0.0445) is far below the target (0.3266), so we keep the exact same deterministic inference pipeline (normalize → Gaussian → Otsu(+bias,+depth) → morphology → empty-mask gate → RLE) and only make two small, metric-relevant adjustments aimed at improving IoU-based mAP. First, we add a single test-time augmentation (horizontal flip) and average the two probability masks before thresholding; this usually improves boundary stability without changing the approach. Second, we replace the hard “empty coverage” cutoff with a slightly more robust combined gate using both coverage and number of connected components (to drop speckle false-positives while keeping legitimate thin salt), and we very lightly retune the gate threshold accordingly. Everything still runs end-to-end, preserves paths, and writes a valid `submission_01.csv`.'
- What this solution (achieved 0.0445) has done: 'We keep your exact inference pipeline (normalize → Gaussian → Otsu(+bias,+depth) → morphology → empty-mask gate → RLE) and only adjust the smallest knobs that most directly affect the IoU-swept mAP. Specifically, we (1) slightly relax the “empty-mask” gate to improve recall (your current score is far below target, so we need more true positives), and (2) reduce the aggressiveness of speckle suppression that may be incorrectly zeroing legitimate thin/fragmented salt masks. These changes don’t alter the core approach or semantics; they only retune the post-threshold gating criteria that currently likely discards too many positives. Submission formatting and ID alignment remain unchanged and it still write `submission_01.csv`.'
- What this solution (achieved 0.0445) has done: 'We keep your exact deterministic “normalize → Gaussian → Otsu(+bias,+depth) → morphology → empty-mask gate → RLE” pipeline, but fix two small issues that are likely suppressing too many true positives and thus keeping mAP very low. First, the connected-components gate currently counts background as a component (because it uses `label(mask).max()`), which can incorrectly zero out valid masks; we switch to counting only foreground components via `np.unique(labels[mask]).size`, preserving the same gating idea but making it correct. Second, we slightly relax the speckle/emptiness gates (coverage threshold and max-components) to improve recall while still controlling false positives, which should move your score upward toward the 0.3266 target without changing the core approach. All paths and the `submission_01.csv` output format remain unchanged.'
- What this solution (achieved 0.0446) has done: 'We keep your exact inference pipeline (normalize → Gaussian → Otsu(+bias,+depth) → morphology → empty-mask gate → RLE) and only make two small, metric-relevant adjustments to improve mAP toward the target by increasing true positives without letting false positives explode. First, we make the empty-mask gate a bit smarter by allowing borderline-low coverage masks through if they have a coherent (low-fragmentation) connected-component structure, which helps recall on thin salt. Second, we slightly relax the small-object removal threshold (MIN_OBJ) so legitimate small/fragmented salt regions aren’t deleted before IoU is computed. Everything remains deterministic, runs end-to-end, and still writes a valid `submission_01.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.0446) has done: 'Your current score (0.0446) is far below the target (0.3266), so we should improve recall/precision in the same “Otsu-threshold + tiny morphology + gating + RLE” inference pipeline without changing the overall approach. The biggest low-risk gain here is to add a second simple TTA (vertical flip) and average (now 4-way with existing hflip) to stabilize masks, then slightly relax the “empty-mask” gate so more borderline true salt cases are kept. To avoid a false-positive explosion, we also tighten the fragmentation check specifically for low-coverage masks (keep coherent thin masks, drop speckle). These are small, metric-relevant knob changes that preserve semantics and should move mAP upward toward the target while keeping runtime under limits and writing a valid `submission_01.csv`.'
- What this solution (achieved 0.0447) has done: 'We keep your exact deterministic pipeline (normalize → Gaussian → Otsu(+bias,+depth) → morphology → gating → RLE) but retune two small knobs that are currently suppressing too many true positives, since your score (0.0446) is far below the target (0.3266). First, we relax the “empty-mask” gating slightly by lowering the coverage threshold and allowing a bit more fragmentation for very-low-coverage masks, which should increase recall without changing the approach. Second, we make the morphology a touch less destructive for thin salt by reducing the minimum hole fill threshold and switching the opening/closing order to favor preserving narrow structures (still the same operations). These are minimal inference-only adjustments aimed at moving mAP upward while keeping runtime and submission format unchanged.'
- What this solution (achieved 0.0452) has done: 'Your current score (0.0447) is far below the target (0.3266), so we should improve the mask quality while keeping your exact “image-derived thresholding → morphology cleanup → gating → RLE” pipeline unchanged. The most direct minimal gain is to add a single global calibration step: estimate a better `BASE_THRESH_BIAS` and `EMPTY_COVERAGE_THRESH` on the training set by maximizing the competition’s IoU-sweep mAP (computed on a small deterministic validation split), then reuse those tuned scalars for test inference. This preserves your core logic and operations (still Otsu + bias + depth bias + same morphology + same gating structure), but replaces hand-tuned constants with data-driven ones, which typically yields a meaningful jump for this kind of baseline. The script still runs end-to-end, stays within time by tuning over a small grid and a capped validation set, and writes `submission_01.csv` with the correct format and ID alignment.'
- What this solution (achieved 0.0458) has done: 'Your score is far below the target, so we keep the exact same inference pipeline (normalize → Gaussian → Otsu(+bias,+depth) → morphology → gating → RLE) and only make two calibration changes that directly affect mask quality under the IoU-swept mAP metric. First, we fix the calibration split so it’s deterministic and stratified by empty/non-empty masks, because random sampling can heavily bias the tuned thresholds toward predicting empty masks (hurting recall). Second, we expand the calibration grid slightly (still small) and reuse the already-loaded train masks to tune both `BASE_THRESH_BIAS` and `EMPTY_COVERAGE_THRESH` more reliably, without changing any core operations. Everything else (TTA, morphology, gating logic, submission formatting/paths) stays the same and the script still writes `submission_01.csv` end-to-end within the time limit.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

print("Python:", sys.version)



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/tgs-salt-identification-challenge",
    "../input/tgs-salt-identification-challenge",
    "/kaggle/data/tgs-salt-identification-challenge",  # fallback if mounted differently
]
BASE = next((p for p in BASE_CANDIDATES if os.path.exists(p)), None)
if BASE is None:
    raise FileNotFoundError(f"Could not find dataset base in: {BASE_CANDIDATES}")

test_path = os.path.join(BASE, "test")
train_path = os.path.join(BASE, "train")
sample_path = os.path.join(BASE, "sample_submission.csv")
depths_path = os.path.join(BASE, "depths.csv")

print("Using BASE:", BASE)
print("Train path exists:", os.path.exists(train_path))
print("Test path exists:", os.path.exists(test_path))
print("Sample submission exists:", os.path.exists(sample_path))
print("Depths exists:", os.path.exists(depths_path))

sample_sub = pd.read_csv(sample_path)
print(sample_sub.head())
print("Sample rows:", len(sample_sub))

depths_df = None
if os.path.exists(depths_path):
    depths_df = pd.read_csv(depths_path)
    depths_df["id"] = depths_df["id"].astype(str)
    z_mean = float(depths_df["z"].mean())
    z_std = float(depths_df["z"].std(ddof=0) + 1e-9)
    z_map = dict(zip(depths_df["id"].values, depths_df["z"].values))
    print("Loaded depths.csv:", depths_df.shape, "z_mean:", z_mean, "z_std:", z_std)
else:
    z_mean, z_std, z_map = 0.0, 1.0, {}



## === cell 2
test_img_dir = os.path.join(test_path, "images")
test_files = sorted(next(os.walk(test_img_dir))[2])

sample_ids = sample_sub["id"].astype(str).tolist()
test_id_set = set([f[:-4] for f in test_files])

missing = [i for i in sample_ids if i not in test_id_set]
extra = [f[:-4] for f in test_files if f[:-4] not in set(sample_ids)]
print("Missing ids in filesystem (should be 0):", len(missing))
print("Extra ids in filesystem (should be 0):", len(extra))

test_ids = [i + ".png" for i in sample_ids]
print(f"# of Test images: {len(test_ids)}")

TARGET_H, TARGET_W = 101, 101

X_test = np.zeros((len(test_ids), TARGET_H, TARGET_W, 3), dtype=np.uint8)
sizes_test = []

print("Getting and resizing test images ...")
sys.stdout.flush()
for n, fn in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = imread(os.path.join(test_img_dir, fn))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        img = img[:, :, :3]
    sizes_test.append([img.shape[0], img.shape[1]])

    img_r = resize(
        img,
        (TARGET_H, TARGET_W),
        mode="constant",
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)
    X_test[n] = img_r

print("Done! X_test:", X_test.shape, X_test.dtype)
print("Unique original sizes (first 10):", list({tuple(s) for s in sizes_test})[:10])




## === cell 3
def mean_iou(y_true, y_pred):
    raise NotImplementedError("mean_iou is not used in this inference-only script.")




## === cell 4
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    binary_opening,
    disk,
)
from skimage.measure import label

TRAIN_CALIBRATE = True
CALIB_SEED = 123

CALIB_VAL_N = 800  # slight increase; still safe for time with small grid

CALIB_GRID_BIAS = [
    -0.040,
    -0.030,
    -0.022,
    -0.016,
    -0.012,
    -0.008,
    -0.005,
    -0.002,
    0.000,
    0.003,
    0.006,
]
CALIB_GRID_EMPTY = [
    0.00015,
    0.00020,
    0.00030,
    0.00045,
    0.00060,
    0.00080,
    0.00100,
    0.00130,
]


def rle_decode(rle, shape=(101, 101)):
    s = str(rle).strip()
    if s == "" or s == "nan":
        return np.zeros(shape[0] * shape[1], dtype=np.uint8).reshape(shape, order="F")
    parts = np.asarray(s.split(), dtype=int)
    starts = parts[0::2] - 1
    lengths = parts[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def fast_iou_sweep_map(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou = 0.0 if union == 0 else inter / union
    thresholds = np.arange(0.5, 1.0, 0.05)
    if y_true.sum() == 0 and y_pred.sum() == 0:
        return 1.0
    if y_true.sum() == 0 and y_pred.sum() > 0:
        return 0.0
    if y_true.sum() > 0 and y_pred.sum() == 0:
        return 0.0
    return float(np.mean(iou > thresholds))


def predict_mask_from_gray(
    g,
    img_id,
    GAUSS_SIGMA,
    BASE_THRESH_BIAS,
    DEPTH_BIAS_SCALE,
    DEPTH_BIAS_CLIP,
    MIN_OBJ,
    MIN_HOLE,
    SELEM,
    USE_HFLIP_TTA,
    USE_VFLIP_TTA,
    EMPTY_COVERAGE_THRESH,
    MAX_COMPONENTS_FOR_NONEMPTY,
    LOW_COV_ALLOW,
    MAX_COMPONENTS_FOR_LOW_COV,
):
    g_min = float(g.min())
    g_max = float(g.max())
    if (g_max - g_min) < 1e-6:
        return np.zeros_like(g, dtype=np.uint8)

    lo = float(np.percentile(g, 2.0))
    hi = float(np.percentile(g, 98.0))
    if (hi - lo) < 1e-6:
        lo, hi = g_min, g_max

    g_norm = (g - lo) / (hi - lo + 1e-9)
    g_norm = np.clip(g_norm, 0.0, 1.0)

    g_s = gaussian(g_norm, sigma=GAUSS_SIGMA, preserve_range=True)

    if USE_HFLIP_TTA or USE_VFLIP_TTA:
        acc = g_s.copy()
        cnt = 1

        if USE_HFLIP_TTA:
            g_h = np.fliplr(g_norm)
            g_hs = gaussian(g_h, sigma=GAUSS_SIGMA, preserve_range=True)
            acc += np.fliplr(g_hs)
            cnt += 1

        if USE_VFLIP_TTA:
            g_v = np.flipud(g_norm)
            g_vs = gaussian(g_v, sigma=GAUSS_SIGMA, preserve_range=True)
            acc += np.flipud(g_vs)
            cnt += 1

        if USE_HFLIP_TTA and USE_VFLIP_TTA:
            g_hv = np.flipud(np.fliplr(g_norm))
            g_hvs = gaussian(g_hv, sigma=GAUSS_SIGMA, preserve_range=True)
            acc += np.flipud(np.fliplr(g_hvs))
            cnt += 1

        g_s = acc / float(cnt)

    z = float(z_map.get(img_id, z_mean))
    z_norm = (z - z_mean) / z_std
    depth_bias = DEPTH_BIAS_SCALE * z_norm
    depth_bias = float(np.clip(depth_bias, -DEPTH_BIAS_CLIP, DEPTH_BIAS_CLIP))

    t = float(threshold_otsu(g_s) + BASE_THRESH_BIAS + depth_bias)
    t = min(1.0, max(0.0, t))
    mask = g_s < t  # salt darker => below threshold

    mask = remove_small_objects(mask, min_size=MIN_OBJ)
    mask = remove_small_holes(mask, area_threshold=MIN_HOLE)

    mask = binary_closing(mask, SELEM)
    mask = binary_opening(mask, SELEM)

    cov = float(mask.mean())
    if not mask.any():
        return np.zeros_like(g, dtype=np.uint8)

    lab = label(mask, connectivity=1)
    n_comp_fg = int(np.unique(lab[mask]).size)

    if cov < EMPTY_COVERAGE_THRESH:
        if (cov >= LOW_COV_ALLOW) and (n_comp_fg <= MAX_COMPONENTS_FOR_LOW_COV):
            return mask.astype(np.uint8)
        return np.zeros_like(g, dtype=np.uint8)
    else:
        if n_comp_fg > MAX_COMPONENTS_FOR_NONEMPTY:
            return np.zeros_like(g, dtype=np.uint8)
        return mask.astype(np.uint8)


GAUSS_SIGMA = 0.8
BASE_THRESH_BIAS = -0.005
DEPTH_BIAS_SCALE = -0.010
DEPTH_BIAS_CLIP = 0.02
MIN_OBJ = 12
MIN_HOLE = 20
SELEM = disk(1)
USE_HFLIP_TTA = True
USE_VFLIP_TTA = True
EMPTY_COVERAGE_THRESH = 0.00045
MAX_COMPONENTS_FOR_NONEMPTY = 300
LOW_COV_ALLOW = 0.00020
MAX_COMPONENTS_FOR_LOW_COV = 40

if TRAIN_CALIBRATE and os.path.exists(train_path):
    train_csv_path = os.path.join(BASE, "train.csv")
    if os.path.exists(train_csv_path):
        train_df = pd.read_csv(train_csv_path)
        train_df["id"] = train_df["id"].astype(str)

        train_df["_is_empty"] = (
            train_df["rle_mask"].fillna("").astype(str).str.strip().eq("")
        )
        empty_ids = train_df.loc[train_df["_is_empty"], "id"].values
        nonempty_ids = train_df.loc[~train_df["_is_empty"], "id"].values

        rng = np.random.RandomState(CALIB_SEED)
        rng.shuffle(empty_ids)
        rng.shuffle(nonempty_ids)

        n_total = min(CALIB_VAL_N, len(train_df))
        n_nonempty = min(len(nonempty_ids), max(1, int(0.60 * n_total)))
        n_empty = min(len(empty_ids), n_total - n_nonempty)
        if n_empty + n_nonempty < n_total:
            remaining = n_total - (n_empty + n_nonempty)
            if len(nonempty_ids) - n_nonempty >= remaining:
                n_nonempty += remaining
            else:
                n_empty = min(len(empty_ids), n_empty + remaining)

        val_ids = np.concatenate([nonempty_ids[:n_nonempty], empty_ids[:n_empty]])
        rng.shuffle(val_ids)

        train_img_dir = os.path.join(train_path, "images")
        train_mask_dir = os.path.join(train_path, "masks")

        val_gray = []
        val_gt = []
        val_id_list = []
        for img_id in tqdm(val_ids, total=len(val_ids), desc="Calib load val"):
            img_path = os.path.join(train_img_dir, img_id + ".png")
            m_path = os.path.join(train_mask_dir, img_id + ".png")
            if not (os.path.exists(img_path) and os.path.exists(m_path)):
                continue
            img = imread(img_path)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[-1] == 4:
                img = img[:, :, :3]
            img = resize(
                img,
                (TARGET_H, TARGET_W),
                mode="constant",
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.uint8)
            Xf = img.astype(np.float32) / 255.0
            g = 0.2989 * Xf[..., 0] + 0.5870 * Xf[..., 1] + 0.1140 * Xf[..., 2]

            gt = imread(m_path)
            if gt.ndim == 3:
                gt = gt[..., 0]
            gt = (gt > 127).astype(np.uint8)
            if gt.shape != (TARGET_H, TARGET_W):
                gt = resize(
                    gt,
                    (TARGET_H, TARGET_W),
                    mode="constant",
                    preserve_range=True,
                    anti_aliasing=False,
                )
                gt = (gt > 0.5).astype(np.uint8)

            val_gray.append(g.astype(np.float32))
            val_gt.append(gt.astype(np.uint8))
            val_id_list.append(img_id)

        if len(val_gray) > 0:
            val_gray = np.stack(val_gray, axis=0)
            val_gt = np.stack(val_gt, axis=0)

            best = (-1.0, BASE_THRESH_BIAS, EMPTY_COVERAGE_THRESH)
            for b in CALIB_GRID_BIAS:
                for e in CALIB_GRID_EMPTY:
                    scores = []
                    for j in range(val_gray.shape[0]):
                        pm = predict_mask_from_gray(
                            val_gray[j],
                            val_id_list[j],
                            GAUSS_SIGMA,
                            float(b),
                            DEPTH_BIAS_SCALE,
                            DEPTH_BIAS_CLIP,
                            MIN_OBJ,
                            MIN_HOLE,
                            SELEM,
                            USE_HFLIP_TTA,
                            USE_VFLIP_TTA,
                            float(e),
                            MAX_COMPONENTS_FOR_NONEMPTY,
                            LOW_COV_ALLOW,
                            MAX_COMPONENTS_FOR_LOW_COV,
                        )
                        scores.append(fast_iou_sweep_map(val_gt[j], pm))
                    m = float(np.mean(scores))
                    if m > best[0]:
                        best = (m, float(b), float(e))

            _, BASE_THRESH_BIAS, EMPTY_COVERAGE_THRESH = best
            print("Calib best val mAP:", best[0])
            print(
                "Using tuned BASE_THRESH_BIAS:",
                BASE_THRESH_BIAS,
                "EMPTY_COVERAGE_THRESH:",
                EMPTY_COVERAGE_THRESH,
            )
        else:
            print("Calibration skipped: no validation data loaded.")
    else:
        print("Calibration skipped: train.csv not found at", train_csv_path)
else:
    print("Calibration disabled or train path missing; using existing constants.")

Xf = X_test.astype(np.float32) / 255.0
gray = 0.2989 * Xf[..., 0] + 0.5870 * Xf[..., 1] + 0.1140 * Xf[..., 2]

preds_test_t = np.zeros((gray.shape[0], TARGET_H, TARGET_W), dtype=np.uint8)

for i in range(gray.shape[0]):
    pm = predict_mask_from_gray(
        gray[i],
        sample_ids[i],
        GAUSS_SIGMA,
        BASE_THRESH_BIAS,
        DEPTH_BIAS_SCALE,
        DEPTH_BIAS_CLIP,
        MIN_OBJ,
        MIN_HOLE,
        SELEM,
        USE_HFLIP_TTA,
        USE_VFLIP_TTA,
        EMPTY_COVERAGE_THRESH,
        MAX_COMPONENTS_FOR_NONEMPTY,
        LOW_COV_ALLOW,
        MAX_COMPONENTS_FOR_LOW_COV,
    )
    preds_test_t[i] = pm

preds_test_upsampled = []
for i in range(len(preds_test_t)):
    up = resize(
        preds_test_t[i],
        (sizes_test[i][0], sizes_test[i][1]),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )
    preds_test_upsampled.append((up > 0.5).astype(np.uint8))

print(
    "Prepared upsampled predictions:",
    len(preds_test_upsampled),
    preds_test_upsampled[0].shape,
)




## === cell 5
def RLenc(img, order="F", format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes_ = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []
    r = 0
    pos = 1
    for c in bytes_:
        if c == 0:
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        if len(runs) == 0:
            return ""
        return " ".join([f"{rr[0]} {rr[1]}" for rr in runs])
    else:
        return runs


pred_dict = {
    fn[:-4]: RLenc(preds_test_upsampled[i])
    for i, fn in tqdm(enumerate(test_ids), total=len(test_ids))
}
print("Example RLE:", next(iter(pred_dict.items())))

sub = pd.DataFrame({"id": sample_ids})
sub["rle_mask"] = sub["id"].map(pred_dict).fillna("")
sub.to_csv("submission_01.csv", index=False)

print("Wrote submission_01.csv")
print(sub.head())
print("Non-empty masks:", (sub["rle_mask"].str.len() > 0).sum(), "out of", len(sub))
