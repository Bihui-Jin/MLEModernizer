# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-image==0.25.2
sklearn-pandas==2.2.0
tqdm==4.67.1

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

0.7943487667676323

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `pydensecrf` (not installed in this environment) by making CRF an optional step and safely falling back to “no-CRF” behavior so the notebook runs end-to-end. I also remove the Jupyter-only `%matplotlib inline` and ensure all required imports (`pd`, `plt`, `tqdm`) are available where used, fixing the `NameError`s. Because the referenced external submission (`../input/u-net-bn-aug-strat/submission.csv`) is not available, I instead start from the provided `sample_submission.csv` and create a valid `submission.csv` output with correct columns and `.csv` suffix. This yield a valid submission file (score be low vs target because there is no model here, but it execute correctly without missing-file/module errors).'
- What this solution (achieved 0.0382) has done: 'Your current 0.0 score comes from submitting essentially empty masks (starting from `sample_submission.csv`), so the smallest legitimate improvement is to generate a non-empty prediction per test image without changing the overall “read images → make mask → RLE → write submission.csv” flow. I keep your RLE/CRF code intact, but replace the “base submission = sample submission” dependency with a simple, deterministic baseline segmenter (Otsu threshold + light morphological cleanup) that produces plausible salt masks from the image content. CRF remain optional and refine these baseline masks when `pydensecrf` is available, otherwise it safely fall back as before. This should move the score upward toward your target without introducing new packages or altering I/O paths.'
- What this solution (achieved 0.0829) has done: 'I keep your exact “image → baseline mask → optional CRF → RLE → submission.csv” pipeline, but fix two score-limiting issues that are still present: (1) your CRF currently uses a *flattened mask* as “labels”, which is not a proper unary for CRF and makes CRF (when available) behave incorrectly; I replace it with a correct probability-based unary built from the baseline mask (still the same CRF step, just correct inputs). (2) your baseline segmentation is too aggressive and produces many false positives; I add a very small, depth-free calibration step that chooses a single global binarization threshold on a small deterministic subset of the *training* images by maximizing mean IoU (this preserves the core logic—thresholding + morphology—but tunes the threshold to the metric, which should move you toward the target). These are minimal changes, avoid new packages, keep runtime under the limit, and still write a valid `submission.csv`.'
- What this solution (achieved 0.0869) has done: 'Your current score is far below the target, so we should make a small, legitimate improvement to the existing “threshold + morphology (+ optional CRF) → RLE” pipeline without changing its core approach. The biggest low-risk gain here is to tune not just the Otsu offset but also the binarization direction (<= vs >=) and the post-processing hyperparameters (closing radius, min object size, min hole size) using a small deterministic subset of training images, optimizing mean IoU as a proxy aligned with the leaderboard metric. This preserves your architecture and evaluation semantics, but reduces systematic false positives/negatives that are currently limiting score. I also keep everything deterministic and within time by using a small grid and the same calibration subset size.'
- What this solution (achieved 0.0989) has done: 'Your current approach is limited because the competition metric is instance/IoU-threshold based and is very sensitive to false positives; your tuned Otsu+morphology still tends to over-predict on many images. I keep your exact pipeline (Otsu threshold per-image → morphology → optional CRF → RLE), but adjust the calibration objective to better match the leaderboard by optimizing mean precision over IoU thresholds (the same thresholds as the competition) on the deterministic training subset instead of plain mean IoU. I also calibrate the final probability-to-mask cutoff (a single global “mask density” gate) that suppresses predictions on images likely to be empty, which typically gives a large jump from low baselines without changing the core segmentation logic. These are minimal changes confined to the calibration step and the final post-binarization decision, and still produce a valid `submission.csv`.'
- What this solution (achieved 0.1004) has done: 'I keep your existing “per-image Otsu threshold → morphology → optional CRF → RLE” pipeline intact, but make two minimal score-relevant fixes that should move you upward toward the target. First, I correct the competition metric proxy used in calibration: when both masks are empty, average precision should be 1.0 (perfect), but your current proxy returns 0.0 due to a zero-denominator branch—this biases calibration toward over-predicting and increases false positives. Second, I modestly expand the calibration search space in a controlled way (slightly larger calibration subset + a few more empty-mask gates) so the empty-mask suppression is tuned better, which is usually the biggest lever for this competition when using simple thresholding. These changes don’t alter the model/approach, don’t add packages, remain deterministic, and still produce a valid `submission.csv`.'
- What this solution (achieved 0.0984) has done: 'The timeout is dominated by the calibration grid-search doing thousands of full-image morphology passes in Python loops, plus repeated Otsu computations and per-row DataFrame writes during test inference. I keep the exact same segmentation pipeline and metric semantics, but make it faster by caching structuring elements, vectorizing AP computation over the validation set, precomputing baseline predictions once per (direction, border, radius, min_obj, min_hole, offset) combo, and avoiding per-image list builds inside the inner loops. For inference, I avoid `df.loc` in a loop (slow) by collecting RLE strings in a list and assigning once, and I pre-list existing test image ids to avoid repeated `os.path.exists` calls. These changes are provably equivalent to the original logic (same operations, same thresholds, same selection criteria) and should bring runtime under 600 seconds.'
- What this solution (achieved 0.0) has done: 'The timeout is dominated by calibration: it repeatedly performs expensive morphology (closing + small object/hole removal) for every parameter combination and every image, even though only the gating step depends on `(mask_fraction, depth)` once a baseline mask is computed. I keep the exact same segmentation pipeline, parameter grids, scoring semantics, and depth-aware gating, but restructure calibration to (1) precompute Otsu thresholds and images once, (2) compute baseline predictions once per `(direction, clear_border, closing_radius, min_obj, min_hole, offset)` combo, and (3) evaluate all `(z_split, gate_shallow, gate_deep)` candidates via vectorized numpy using cached `iou` and `mask_fraction` arrays. I also ensure `disk()` footprints are cached and avoid extra conversions/reshapes in the hot loop. The prediction/submission phase is already mostly fine; only minimal equivalent speed tweaks are kept.'
- What this solution (achieved 0.103) has done: 'I fix the calibration IndexError by correcting the boolean indexing/broadcasting when filling the AP matrix (the current `passed[:, None][both_present]` shape doesn’t match). That allow calibration to complete so `best_params` is defined, which in turn fixes the downstream `NameError` in visualization, inference, and submission-writing cells. I also add a safe fallback in case calibration returns `None` (e.g., no images loaded) so the notebook always runs end-to-end and writes `submission.csv`. These changes preserve your existing segmentation + gating + optional CRF pipeline and should restore a non-zero score by enabling the calibrated parameters to be used.'
- What this solution (achieved 0.103) has done: 'Your score is far below the target (0.103 vs 0.794), so we should improve the *existing* threshold+morphology+depth-gating pipeline without changing its nature. The largest likely gain with minimal risk is to calibrate the empty-mask gating using the *true competition metric proxy* (mean precision over IoU thresholds) rather than the current shortcut `passed = mean(iou>t)`, which over-rewards predictions when GT is present but under-penalizes false positives/false negatives. I replace that gating score computation with an exact vectorized AP computation for the single-object case (including correct handling for both-empty, pred-only, gt-only, and both-present-but-low-IoU), while keeping your segmentation, parameter grids, and inference code intact. This should reduce systematic overprediction and move the score upward toward the target while staying deterministic and within the runtime budget.'
- What this solution (achieved 0.103) has done: 'Your current 0.103 score is far below the 0.794 target, so the smallest legitimate move upward is to fix the calibration objective so it matches the competition’s mean-precision-over-IoU-thresholds metric (instead of the current “mean(iou>t)” shortcut that rewards many false positives). I keep your exact segmentation pipeline (Otsu±offset → closing → small obj/hole removal → optional clear-border → depth-gated empty suppression → optional CRF → RLE), but change calibration scoring to the exact single-object AP formula in a vectorized way. This directly tunes the empty-mask gates and threshold direction/offset toward what the leaderboard measures, without altering inference semantics. I also keep runtime bounded by reusing the already-computed IoU and mask fractions and only changing the AP computation.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we should increase performance while keeping your existing “Otsu±offset → morphology → depth-gated empty suppression → optional CRF → RLE” pipeline intact. The biggest issue is that the calibration AP computation is currently wrong: it adds a term that gives nonzero credit even when GT is empty but a prediction exists, which encourages false positives and hurts this metric. I replace that calibration scoring with the exact single-object mean-precision-over-IoU-thresholds formula (vectorized), keeping the same parameter grids and using the same cached IoU/mask-fraction arrays. This should reduce false positives via better-tuned gating and move your score upward toward the target, without changing inference semantics or adding dependencies.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    disk,
)

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    PYDENSECRF_AVAILABLE = True
except ModuleNotFoundError:
    PYDENSECRF_AVAILABLE = False

RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)




## === cell 1
def rle_decode(rle_mask, shape=(101, 101)):
    """
    rle_mask: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if rle_mask is None or (isinstance(rle_mask, float) and np.isnan(rle_mask)):
        return np.zeros(shape[0] * shape[1], dtype=np.uint8).reshape(shape)

    s = str(rle_mask).split()
    if len(s) == 0:
        return np.zeros(shape[0] * shape[1], dtype=np.uint8).reshape(shape)

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        lo = max(lo, 0)
        hi = min(hi, img.shape[0])
        if lo < hi:
            img[lo:hi] = 1
    return img.reshape(shape)




## === cell 2
"""
Reading sample submission ONLY to get the correct 'id' list and output schema.
"""
base_submission_path = (
    "../input/tgs-salt-identification-challenge/sample_submission.csv"
)
if not os.path.exists(base_submission_path):
    base_submission_path = "../input/sample_submission.csv"

df = pd.read_csv(base_submission_path)

assert (
    "id" in df.columns and "rle_mask" in df.columns
), f"Unexpected submission columns: {df.columns.tolist()}"

plt.figure(figsize=(6, 1.5))
plt.text(0.01, 0.5, f"Loaded sample_submission with {len(df)} test ids.", fontsize=12)
plt.axis("off")
plt.show()



## === cell 3
"""
CRF post-processing.

Score-relevant fix (kept): when CRF is available, use a proper unary constructed from
(mask -> foreground probability) instead of passing flattened hard labels.
"""


def crf(original_image, mask_img, fg_prob=0.70):
    if mask_img.ndim == 3:
        mask2d = mask_img[:, :, 0]
    else:
        mask2d = mask_img
    mask2d = (mask2d > 0).astype(np.uint8)

    if not PYDENSECRF_AVAILABLE:
        return mask2d

    h, w = original_image.shape[:2]

    p_fg = np.where(mask2d == 1, fg_prob, 1.0 - fg_prob).astype(np.float32)
    p_bg = 1.0 - p_fg
    probs = np.stack([p_bg, p_fg], axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(
        sxy=(3, 3),
        compat=3,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    Q = d.inference(10)
    MAP = np.argmax(Q, axis=0).reshape((h, w)).astype(np.uint8)
    return MAP




## === cell 4
test_path = "../input/tgs-salt-identification-challenge/test/images/"
if not os.path.exists(test_path):
    test_path = "../input/test/images/"
assert os.path.exists(test_path), f"Test images path not found: {test_path}"

train_img_path = "../input/tgs-salt-identification-challenge/train/images/"
train_mask_path = "../input/tgs-salt-identification-challenge/train/masks/"
if not os.path.exists(train_img_path):
    train_img_path = "../input/train/images/"
    train_mask_path = "../input/train/masks/"
assert os.path.exists(train_img_path) and os.path.exists(
    train_mask_path
), "Train images/masks path not found."

train_csv_path = "../input/tgs-salt-identification-challenge/train.csv"
if not os.path.exists(train_csv_path):
    train_csv_path = "../input/train.csv"
train_df = pd.read_csv(train_csv_path)

depths_path = "../input/tgs-salt-identification-challenge/depths.csv"
if not os.path.exists(depths_path):
    depths_path = "../input/depths.csv"
depths_df = pd.read_csv(depths_path)
depths_map = dict(zip(depths_df["id"].values, depths_df["z"].values))



## === cell 5
"""
Baseline segmentation + calibration.

Score-relevant fix (this patch): the previous calibration AP computation was incorrect because it
implicitly gave partial credit to false positives/false negatives via an unconditional
`both_present * mean_prec` add. We replace it with the exact single-object competition AP:
- both empty -> 1
- pred empty, gt non-empty -> 0
- pred non-empty, gt empty -> 0
- both non-empty -> mean_{t in thresholds} 1[iou > t]
This better tunes empty-mask gating to suppress false positives and should move score upward.
"""

IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05, dtype=np.float32)
_DISK_CACHE = {}


def _to_gray(img):
    if img.ndim == 3:
        return img[:, :, 0].astype(np.float32)
    return img.astype(np.float32)


def _clear_border_1px(mask01):
    m = mask01.copy()
    m[0, :] = 0
    m[-1, :] = 0
    m[:, 0] = 0
    m[:, -1] = 0
    return m


def _get_disk(radius):
    r = int(radius)
    d = _DISK_CACHE.get(r)
    if d is None:
        d = disk(r)
        _DISK_CACHE[r] = d
    return d


def baseline_mask_from_image(
    img,
    thr,
    direction="le",  # "le" means img <= thr, "ge" means img >= thr
    closing_radius=1,
    min_obj=30,
    min_hole=30,
    clear_border=False,
):
    img2 = _to_gray(img)
    if direction == "ge":
        mask = img2 >= thr
    else:
        mask = img2 <= thr

    mask = binary_closing(mask, footprint=_get_disk(int(closing_radius)))
    mask = remove_small_objects(mask, min_size=int(min_obj))
    mask = remove_small_holes(mask, area_threshold=int(min_hole))
    mask = mask.astype(np.uint8)

    if clear_border:
        mask = _clear_border_1px(mask)

    return mask.astype(np.uint8)


def calibrate_baseline_params(
    train_df,
    depths_map,
    n_calib=700,
    val_frac=0.25,
    offsets=(-24, -18, -12, -8, -5, -3, -1, 0, 1, 3, 5, 8, 12, 18, 24),
    directions=("le", "ge"),
    closing_radii=(0, 1, 2),
    min_objs=(15, 30, 60),
    min_holes=(15, 30, 60),
    empty_gates=(0.0, 0.001, 0.002, 0.0035, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03),
    z_splits=(650, 700, 750, 800, 850, 900),
    clear_borders=(False, True),
):
    rng = np.random.default_rng(RANDOM_SEED)

    ids_all = train_df["id"].values
    if len(ids_all) > n_calib:
        ids_all = rng.choice(ids_all, size=n_calib, replace=False)

    ims = []
    gts = []
    otsus = []
    zs = []
    for img_id in ids_all:
        img_fp = os.path.join(train_img_path, f"{img_id}.png")
        msk_fp = os.path.join(train_mask_path, f"{img_id}.png")
        if not (os.path.exists(img_fp) and os.path.exists(msk_fp)):
            continue
        z = depths_map.get(img_id, None)
        if z is None:
            continue

        im = imread(img_fp)
        gt = imread(msk_fp)
        im2 = _to_gray(im)
        ims.append(im2)
        gts.append((gt > 127).astype(np.uint8))
        zs.append(float(z))
        try:
            otsus.append(float(threshold_otsu(im2)))
        except ValueError:
            otsus.append(float(np.mean(im2)))

    n = len(ims)
    if n == 0:
        return None

    ims = np.asarray(ims, dtype=np.float32)
    gts = np.asarray(gts, dtype=np.uint8)
    otsus = np.asarray(otsus, dtype=np.float32)
    zs = np.asarray(zs, dtype=np.float32)

    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = idx[:n_val]

    gts_val = gts[val_idx]
    zs_val = zs[val_idx]
    gts_val_flat = gts_val.reshape(len(val_idx), -1).astype(np.uint8, copy=False)
    y_true_has_val = gts_val_flat.sum(axis=1) > 0

    best = None
    best_score = -1.0

    empty_gates_arr = np.asarray(empty_gates, dtype=np.float32)
    z_splits_arr = np.asarray(z_splits, dtype=np.float32)
    shallow_masks = {float(zsplt): (zs_val <= float(zsplt)) for zsplt in z_splits_arr}

    for direction in directions:
        for clear_border in clear_borders:
            for closing_radius in closing_radii:
                for min_obj in min_objs:
                    for min_hole in min_holes:
                        for off in offsets:
                            thr_vec = (otsus + float(off)).astype(np.float32)

                            preds_val_flat = np.empty(
                                (len(val_idx), 101 * 101), dtype=np.uint8
                            )
                            fracs_val = np.empty((len(val_idx),), dtype=np.float32)

                            for j, k in enumerate(val_idx):
                                pred = baseline_mask_from_image(
                                    ims[k],
                                    thr=float(thr_vec[k]),
                                    direction=direction,
                                    closing_radius=closing_radius,
                                    min_obj=min_obj,
                                    min_hole=min_hole,
                                    clear_border=clear_border,
                                )
                                pv = pred.reshape(-1).astype(np.uint8, copy=False)
                                preds_val_flat[j] = pv
                                fracs_val[j] = float(pv.mean())

                            y_pred_has_raw = preds_val_flat.sum(axis=1) > 0

                            inter = (
                                (gts_val_flat & preds_val_flat)
                                .sum(axis=1)
                                .astype(np.float32)
                            )
                            union = (
                                (gts_val_flat | preds_val_flat)
                                .sum(axis=1)
                                .astype(np.float32)
                            )
                            iou_raw = np.where(union == 0.0, 1.0, inter / union).astype(
                                np.float32
                            )

                            mean_prec_if_both_present = (
                                (iou_raw[:, None] > IOU_THRESHOLDS[None, :]).mean(
                                    axis=1
                                )
                            ).astype(
                                np.float32
                            )  # (N,)

                            for z_split in z_splits_arr:
                                shallow = shallow_masks[float(z_split)]
                                deep = ~shallow

                                for gate_shallow in empty_gates_arr:
                                    keep = np.empty(
                                        (len(val_idx), len(empty_gates_arr)), dtype=bool
                                    )

                                    keep_shallow_vec = fracs_val >= float(gate_shallow)
                                    keep[shallow, :] = keep_shallow_vec[shallow, None]
                                    keep[deep, :] = (
                                        fracs_val[deep, None]
                                        >= empty_gates_arr[None, :]
                                    )

                                    y_pred_has = y_pred_has_raw[:, None] & keep  # (N,K)

                                    both_empty = (~y_true_has_val)[:, None] & (
                                        ~y_pred_has
                                    )
                                    both_present = (y_true_has_val)[:, None] & (
                                        y_pred_has
                                    )

                                    ap = np.zeros(
                                        (len(val_idx), len(empty_gates_arr)),
                                        dtype=np.float32,
                                    )
                                    ap[both_empty] = 1.0
                                    ap[both_present] = mean_prec_if_both_present[
                                        :, None
                                    ][both_present]

                                    scores = ap.mean(axis=0)
                                    best_deep_idx = int(np.argmax(scores))
                                    score_val = float(scores[best_deep_idx])

                                    if score_val > best_score:
                                        best_score = score_val
                                        best = dict(
                                            offset=float(off),
                                            direction=direction,
                                            closing_radius=int(closing_radius),
                                            min_obj=int(min_obj),
                                            min_hole=int(min_hole),
                                            gate_shallow=float(gate_shallow),
                                            gate_deep=float(
                                                empty_gates_arr[best_deep_idx]
                                            ),
                                            z_split=float(z_split),
                                            empty_gate=float(gate_shallow),
                                            clear_border=bool(clear_border),
                                            calib_val_mean_ap=float(best_score),
                                            calib_n=int(n),
                                            calib_val_n=int(len(val_idx)),
                                        )

    return best


best_params = calibrate_baseline_params(train_df, depths_map=depths_map)

if best_params is None:
    best_params = dict(
        offset=0.0,
        direction="le",
        closing_radius=1,
        min_obj=30,
        min_hole=30,
        gate_shallow=0.0,
        gate_deep=0.0,
        z_split=800.0,
        empty_gate=0.0,
        clear_border=False,
        calib_val_mean_ap=0.0,
        calib_n=0,
        calib_val_n=0,
    )

thr_offset = best_params["offset"]
best_direction = best_params["direction"]
best_closing_radius = best_params["closing_radius"]
best_min_obj = best_params["min_obj"]
best_min_hole = best_params["min_hole"]
best_clear_border = best_params["clear_border"]

best_gate_shallow = best_params["gate_shallow"]
best_gate_deep = best_params["gate_deep"]
best_z_split = best_params["z_split"]

print("Calibrated baseline params:", best_params)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3102822748.py in <cell line: 0>()
    249 
    250 
--> 251 best_params = calibrate_baseline_params(train_df, depths_map=depths_map)
    252 
    253 if best_params is None:

/tmp/ipykernel_11/3102822748.py in calibrate_baseline_params(train_df, depths_map, n_calib, val_frac, offsets, directions, closing_radii, min_objs, min_holes, empty_gates, z_splits, clear_borders)
    218                                     )
    219                                     ap[both_empty] = 1.0
--> 220                                     ap[both_present] = mean_prec_if_both_present[
    221                                         :, None
    222                                     ][both_present]

IndexError: boolean index did not match indexed array along dimension 1; dimension is 1 but corresponding boolean dimension is 10

## === cell 6
"""
Visualizing baseline vs CRF for a few test images (unchanged intent).
"""
nImgs = 3
rng = np.random.default_rng(RANDOM_SEED)
start_i = int(rng.integers(0, len(df)))

plt.figure(figsize=(12, 9))
plt.subplots_adjust(wspace=0.2, hspace=0.2)

i = start_i
shown = 0
while i < len(df) and shown < nImgs:
    img_id = df.loc[i, "id"]
    img_fp = os.path.join(test_path, f"{img_id}.png")
    if os.path.exists(img_fp):
        orig_img = imread(img_fp)
        img2 = _to_gray(orig_img)
        try:
            th = float(threshold_otsu(img2))
        except ValueError:
            th = float(np.mean(img2))

        pred_mask = baseline_mask_from_image(
            orig_img,
            thr=th + thr_offset,
            direction=best_direction,
            closing_radius=best_closing_radius,
            min_obj=best_min_obj,
            min_hole=best_min_hole,
            clear_border=best_clear_border,
        )

        z = depths_map.get(img_id, best_z_split)
        gate = best_gate_shallow if float(z) <= float(best_z_split) else best_gate_deep
        if float(pred_mask.mean()) < float(gate):
            pred_mask = np.zeros_like(pred_mask, dtype=np.uint8)

        crf_output = crf(orig_img, pred_mask)

        plt.subplot(nImgs, 3, 3 * shown + 1)
        plt.imshow(orig_img, cmap="gray")
        plt.title("Original image")
        plt.axis("off")

        plt.subplot(nImgs, 3, 3 * shown + 2)
        plt.imshow(pred_mask, cmap="gray")
        plt.title("Baseline mask (calibrated)")
        plt.axis("off")

        plt.subplot(nImgs, 3, 3 * shown + 3)
        plt.imshow(crf_output, cmap="gray")
        plt.title("After CRF" if PYDENSECRF_AVAILABLE else "After CRF (fallback)")
        plt.axis("off")

        shown += 1
    i += 1

plt.show()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/921305080.py in <cell line: 0>()
     24         pred_mask = baseline_mask_from_image(
     25             orig_img,
---> 26             thr=th + thr_offset,
     27             direction=best_direction,
     28             closing_radius=best_closing_radius,

NameError: name 'thr_offset' is not defined

## === cell 7
"""
RLE encode.

Competition-correct: flatten in Fortran order (column-major), return "" for empty.
"""


def rle_encode(im):
    im = (im > 0).astype(np.uint8)
    if im.sum() == 0:
        return ""

    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 8
"""
Generate predictions for each test image, optionally refine with CRF, then encode to RLE.

Performance fixes (equivalent):
- Avoid per-row df.loc writes by collecting RLE strings into a list and assigning once.
- Avoid repeated os.path.exists by precomputing the set of filenames in the test folder.
Core segmentation + optional CRF logic is unchanged.
"""
try:
    test_files = os.listdir(test_path)
    test_ids_available = set(
        os.path.splitext(f)[0] for f in test_files if f.endswith(".png")
    )
except Exception:
    test_ids_available = None

rles = []
for i in tqdm(range(df.shape[0]), desc="Predict + Post-process (CRF optional)"):
    img_id = df.loc[i, "id"]
    img_fp = os.path.join(test_path, f"{img_id}.png")

    exists = (
        (img_id in test_ids_available)
        if test_ids_available is not None
        else os.path.exists(img_fp)
    )

    if exists:
        orig_img = imread(img_fp)
        img2 = _to_gray(orig_img)
        try:
            th = float(threshold_otsu(img2))
        except ValueError:
            th = float(np.mean(img2))

        pred_mask = baseline_mask_from_image(
            orig_img,
            thr=th + thr_offset,
            direction=best_direction,
            closing_radius=best_closing_radius,
            min_obj=best_min_obj,
            min_hole=best_min_hole,
            clear_border=best_clear_border,
        )

        z = depths_map.get(img_id, best_z_split)
        gate = best_gate_shallow if float(z) <= float(best_z_split) else best_gate_deep
        if float(pred_mask.mean()) < float(gate):
            pred_mask = np.zeros_like(pred_mask, dtype=np.uint8)

        post_mask = crf(orig_img, pred_mask)
    else:
        post_mask = np.zeros((101, 101), dtype=np.uint8)

    rles.append(rle_encode(post_mask))

df["rle_mask"] = rles



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1248800604.py in <cell line: 0>()
     36         pred_mask = baseline_mask_from_image(
     37             orig_img,
---> 38             thr=th + thr_offset,
     39             direction=best_direction,
     40             closing_radius=best_closing_radius,

NameError: name 'thr_offset' is not defined

## === cell 9
submission_path = "submission.csv"
df.to_csv(submission_path, index=False)
print(f"Wrote submission to: {submission_path}")
print(df.head())
print(
    "Non-empty masks:",
    int((df["rle_mask"].astype(str).str.len() > 0).sum()),
    "out of",
    len(df),
)
print("PYDENSECRF_AVAILABLE:", PYDENSECRF_AVAILABLE)
print(
    "Depth-aware gates:",
    {
        "z_split": best_z_split,
        "gate_shallow": best_gate_shallow,
        "gate_deep": best_gate_deep,
    },
)
print("Clear border used:", best_clear_border)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1417569861.py in <cell line: 0>()
     13     "Depth-aware gates:",
     14     {
---> 15         "z_split": best_z_split,
     16         "gate_shallow": best_gate_shallow,
     17         "gate_deep": best_gate_deep,

NameError: name 'best_z_split' is not defined
