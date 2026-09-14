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

0.8166594547814794

# 6. Current score

0.1113

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the notebook runnable in this Kaggle environment by (1) removing the hard dependency on `pydensecrf` (it’s not installed) and replacing the CRF step with a safe no-op fallback that preserves the pipeline structure. I also fix the missing/incorrect imports and the Jupyter-only `%matplotlib inline` so the script runs as a normal `.py`. Finally, I load `sample_submission.csv` from the provided competition input path (instead of a non-existent custom dataset path) and ensure we always write a valid `submission.csv` with the required `id,rle_mask` columns and `.csv` suffix.'
- What this solution (achieved 0.0311) has done: 'Your current 0.0 score comes from generating predictions by post-processing the *sample_submission* masks, which are essentially dummy placeholders and not model outputs; even with CRF, this stays near-empty and performs terribly. To move toward the 0.8167 target with minimal disruption, I keep your pipeline structure (load test ids → create a mask → optional CRF → RLE encode → write submission) but replace the input masks with a simple image-derived heuristic mask computed directly from each test image (no training, no new model). I also ensure every id gets a prediction (no skipped rows), because missing/blank masks across the board can crater the score. The CRF fallback remains intact and still acts as optional post-processing when available.'
- What this solution (achieved 0.0728) has done: 'Your score is far below the target, and the main reason is that the current “heuristic from test image intensity + optional CRF” produces masks that are essentially unrelated to salt regions. To move toward the target with minimal disruption, I keep the same overall pipeline (read test image → create a mask → optional CRF → RLE → write CSV) but make the heuristic more aligned with typical salt masks by (1) using Otsu thresholding (per-image, robust) instead of a fixed percentile threshold, (2) choosing the foreground side (bright vs dark) via a simple center-prior (salt tends to be more central), and (3) applying small connected-component cleanup (remove tiny components, fill small holes) to reduce false positives/fragmentation which strongly hurts mean AP. These are lightweight, deterministic post-processing changes that should increase IoU consistency without changing your core approach or adding any training. The submission writing and RLE encoding remain unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 0.0438) has done: 'Your current score is far below the target, so we should cautiously increase it without changing the overall “test-image heuristic → optional CRF → RLE” pipeline. The biggest low-risk gain here is to make the heuristic closer to typical salt-mask morphology: suppress border noise (salt rarely touches borders), smooth before thresholding, and do stronger but still simple connected-component cleanup (keep only the largest component and do a small closing) to reduce false positives that heavily hurt mean AP across IoU thresholds. I also add a single global fallback threshold (computed from a small subset of test images) used only when Otsu becomes unstable, keeping the same core logic but improving robustness. Submission writing and RLE encoding remain unchanged and a valid `submission.csv` is always produced.'
- What this solution (achieved 0.0763) has done: 'Your score is far below the target, so we should improve the mask quality while keeping the same “test-image heuristic → optional CRF → RLE → submission.csv” pipeline. The most impactful minimal change here is to tune the post-processing for this metric by (1) selecting a binarization threshold that maximizes mean IoU on a small validation split from the provided training masks (no model/training loop added), and (2) applying a very light test-time augmentation (horizontal flip ensembling) to stabilize predictions. I also fix a subtle but important bug: your `rle_decode` reshapes in C-order while encoding uses Fortran order; making decode consistent improves any train-mask validation we add. All changes are deterministic, fast (<600s), and keep the core approach intact.'
- What this solution (achieved 0.0772) has done: 'Your current score (0.0763) is far below the target (0.8167), so we should improve mask quality without changing the overall heuristic→(optional CRF)→RLE pipeline. The biggest low-risk issue is that the current flip “ensemble” uses a union (`>=1`), which strongly increases false positives and hurts mean AP; switching to a strict consensus (`==2`) typically improves precision and IoU stability. Next, calibrating the binarization threshold using plain mean IoU is misaligned with the competition’s mean AP over IoU thresholds, so we keep the same calibration structure but optimize the threshold against an AP-style metric computed on a small validation set. Finally, we keep CRF as-is (no-op fallback) and ensure every id still gets a valid RLE string in `submission.csv`.'
- What this solution (achieved 0.0912) has done: 'Your score is far below the target (0.0772 vs 0.8167), so we should improve mask quality while keeping the same overall “heuristic mask from image → optional CRF → RLE → submission.csv” pipeline. The biggest minimal gain available without changing core logic is to fix the post-processing mismatch with the competition metric: your validation metric currently treats the whole mask as one object, but Kaggle scores connected components as separate objects, so we compute AP using connected components on the validation split and calibrate the threshold against that (same calibration loop, just correct scoring). Next, we stop forcing “largest connected component only”, which suppresses real multi-island salt masks and hurts object-level AP; instead we keep all components after cleanup. Finally, we slightly relax border suppression (still present) and retune min-size thresholds modestly so we don’t delete valid small salt objects that matter for AP at higher IoUs.'
- What this solution (achieved 0.0884) has done: 'Your current score (0.0912) is far below the target (0.8167), so we should increase it with the smallest changes that keep your exact pipeline (image→heuristic mask→flip-consensus→optional CRF→RLE→CSV). The most likely reason you’re capped so low is a mask orientation mismatch: Kaggle TGS masks are encoded/decoded in column-major (“Fortran”) order, but your heuristic operates in normal image coordinates; fixing orientation at the RLE boundary is a minimal, high-impact correction that doesn’t change the heuristic itself. Concretely, we ensure the mask passed into `rle_encode` matches Kaggle’s expected orientation by applying a transpose consistently (and likewise when computing validation AP so calibration selects the right threshold). Everything else (threshold calibration loop, consensus logic, CRF fallback, and submission writing) remains unchanged.'
- What this solution (achieved 0.0884) has done: 'Your current score is far below the target, so we should improve mask quality while keeping the same pipeline (heuristic mask → flip-consensus → optional CRF → RLE → submission). The smallest high-impact fix is to incorporate the provided `depths.csv` as an additional, lightweight prior to choose which side of the Otsu threshold is “salt” (salt frequency correlates with depth), without changing the thresholding/morphology core. Next, we calibrate two post-processing hyperparameters (`min_size`, `hole_area`) alongside the existing threshold calibration using the same instancewise AP metric you already use, so the changes are metric-aligned but still minimal. Finally, we keep the transpose-at-RLE boundary consistent and ensure the submission rows remain aligned with `sample_submission.csv` ids.'
- What this solution (achieved 0.116) has done: 'The timeout is dominated by the exhaustive calibration grid in the validation loop (17×3×3×3×3=1377 parameter sets × 250 images) plus repeated disk I/O and repeated connected-component labeling/IoU work. To keep the exact same core logic and evaluation semantics, the main speedups are: (1) load the validation images/masks once and reuse them across the grid, (2) precompute the expensive per-image normalized+smoothed image and reuse it inside the heuristic for every parameter set, and (3) replace the slow DataFrame row-by-row `.loc` writes with preallocated arrays/lists while keeping identical predictions and RLE encoding. These changes are provably equivalent (they only avoid recomputation and Python overhead) and should bring runtime under 600 seconds.'
- What this solution (achieved 0.1113) has done: 'To move your score up toward 0.8167 without changing the core “heuristic mask → (optional CRF) → RLE” approach, I fix a key semantic mismatch with the competition metric: it evaluates *instance-level* matches but the dataset is effectively *single-object* per image, so we should score/calibrate as a single object (mask IoU-based mAP), not connected-components matching. I keep your exact grid-search structure and parameter grids, but change only the validation scorer to the correct “single-mask mAP over IoU thresholds,” which should select much better thresholds for your existing heuristic and typically yields a large jump from ~0.1. I also make the threshold grid “Otsu-adjustment” rather than an absolute threshold (minimal change but much better aligned with your per-image Otsu logic), so calibration generalizes instead of forcing a global constant threshold. Everything else (preprocessing, depth bias, consensus, CRF fallback, RLE transpose, and submission writing) remains the same and still produces `submission.csv`.'
- What this solution (achieved 0.1113) has done: 'I keep your exact heuristic→(optional CRF)→RLE pipeline, but fix a key evaluation mismatch: the TGS metric is *instance-based* on connected components, so calibrating with a single-mask IoU proxy can select thresholds that look good in validation but score poorly on Kaggle. I change only the validation scorer back to a fast, correct instance-mAP implementation (connected components matching) while reusing your existing grid and precomputed `val_data` so runtime stays under 600s. I also make the transpose handling consistent inside the validation metric (so we don’t “double transpose” between GT/pred and accidentally calibrate the wrong orientation). Everything else (thresholding, depth bias, flip-consensus, CRF fallback, and submission writing) remains unchanged and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from skimage.color import gray2rgb
import matplotlib.pyplot as plt

np.random.seed(100)

BASE_PATH = "/kaggle/input/tgs-salt-identification-challenge"
TEST_IMG_DIR = os.path.join(BASE_PATH, "test", "images")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train", "images")
TRAIN_MASK_DIR = os.path.join(BASE_PATH, "train", "masks")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
DEPTHS_PATH = os.path.join(BASE_PATH, "depths.csv")

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample submission at {SAMPLE_SUB_PATH}"
assert os.path.exists(TEST_IMG_DIR), f"Missing test image dir at {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train image dir at {TRAIN_IMG_DIR}"
assert os.path.exists(TRAIN_MASK_DIR), f"Missing train mask dir at {TRAIN_MASK_DIR}"
assert os.path.exists(DEPTHS_PATH), f"Missing depths.csv at {DEPTHS_PATH}"




## === cell 1
_HAS_DCRF = False
try:
    import pydensecrf.densecrf as dcrf  # type: ignore
    from pydensecrf.utils import unary_from_labels  # type: ignore

    _HAS_DCRF = True
except Exception:
    _HAS_DCRF = False




## === cell 2
def rle_decode(rle_mask):
    """
    rle_mask: run-length as string formatted (start length)
    Returns numpy array mask of shape (101, 101), 1 - mask, 0 - background

    NOTE: decode matches Kaggle's column-major convention (order='F'),
    consistent with rle_encode below.
    """
    s = str(rle_mask).split()
    if len(s) == 0:
        return np.zeros((101, 101), dtype=np.uint8)

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((101, 101), order="F")




## === cell 3
def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.flatten(order="F")  # Kaggle TGS expects column-major
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 4
df = pd.read_csv(SAMPLE_SUB_PATH)

expected_cols = {"id", "rle_mask"}
missing = expected_cols - set(df.columns)
assert not missing, f"Sample submission missing columns: {missing}"

depths = pd.read_csv(DEPTHS_PATH)
depths["id"] = depths["id"].astype(str)
depths["z"] = depths["z"].astype(np.float32)
z_map = dict(zip(depths["id"].values, depths["z"].values))
z_vals = depths["z"].values
_Z_MED = float(np.median(z_vals))
_Z_P10 = float(np.quantile(z_vals, 0.10))
_Z_P90 = float(np.quantile(z_vals, 0.90))




## === cell 5
def crf(original_image, mask_img):
    """
    Applies DenseCRF post-processing if pydensecrf is available.
    If not available, returns the input mask unchanged (no-op fallback).
    """
    if mask_img.ndim == 3:
        mask_2d = mask_img[:, :, 0]
    else:
        mask_2d = mask_img

    if not _HAS_DCRF:
        return (mask_2d > 0).astype(np.uint8)

    if len(mask_img.shape) < 3:
        mask_rgb = gray2rgb(mask_img)
    else:
        mask_rgb = mask_img

    annotated_label = (
        mask_rgb[:, :, 0].astype(np.int32)
        + (mask_rgb[:, :, 1].astype(np.int32) << 8)
        + (mask_rgb[:, :, 2].astype(np.int32) << 16)
    )
    _, labels = np.unique(annotated_label, return_inverse=True)
    n_labels = 2

    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(
        sxy=(3, 3),
        compat=3,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    Q = d.inference(10)
    MAP = np.argmax(Q, axis=0).astype(np.uint8)
    return MAP.reshape((original_image.shape[0], original_image.shape[1]))




## === cell 6
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    disk,
)
from skimage.measure import label


def _compute_global_fallback_thr(ids, max_images=200):
    vals = []
    for img_id in ids[:max_images]:
        img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")
        if not os.path.exists(img_path):
            continue
        img = imread(img_path)
        if img.ndim == 3:
            img = img[:, :, 0]
        img = img.astype(np.float32)
        mn, mx = float(np.min(img)), float(np.max(img))
        denom = (mx - mn) if (mx - mn) > 1e-6 else 1.0
        x = (img - mn) / denom
        h, w = x.shape
        y0, y1 = int(h * 0.2), int(h * 0.8)
        x0, x1 = int(w * 0.2), int(w * 0.8)
        vals.append(np.median(x[y0:y1, x0:x1]))
    if len(vals) == 0:
        return 0.5
    return float(np.median(vals))


_GLOBAL_FALLBACK_THR = _compute_global_fallback_thr(df["id"].astype(str).tolist())


def _preprocess_image(orig_img):
    img = orig_img
    if img.ndim == 3:
        img = img[:, :, 0]
    img = img.astype(np.float32)

    mn, mx = float(np.min(img)), float(np.max(img))
    denom = (mx - mn) if (mx - mn) > 1e-6 else 1.0
    x = (img - mn) / denom
    x_s = gaussian(x, sigma=0.8, preserve_range=True)
    return x_s


def heuristic_mask_from_image(
    orig_img,
    override_thr=None,
    img_id=None,
    min_size=30,
    hole_area=40,
    border=4,
    close_r=2,
    _x_s=None,  # precomputed smoothed/normalized image (optional)
):
    """
    Heuristic (core logic preserved):
    - Normalize and lightly smooth to stabilize thresholding.
    - Otsu threshold per image; fallback to a global threshold if Otsu is unstable.
    - Optionally override threshold (used for validation-calibrated global adjustment).
    - Choose foreground side using a center prior, with a small depth-based bias.
    - Suppress a thin border region to reduce noisy FP.
    - Cleanup: remove small objects/holes, close small gaps.
    """
    x_s = _x_s if _x_s is not None else _preprocess_image(orig_img)

    if override_thr is not None:
        thr = float(override_thr)
    else:
        try:
            thr = float(threshold_otsu(x_s))
        except Exception:
            thr = float(_GLOBAL_FALLBACK_THR)

        if float(np.std(x_s)) < 0.03:
            thr = float(_GLOBAL_FALLBACK_THR)

    m_high = x_s > thr
    m_low = ~m_high

    h, w = x_s.shape
    y0, y1 = int(h * 0.25), int(h * 0.75)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    center = (slice(y0, y1), slice(x0, x1))

    score_high = float(np.mean(m_high[center]))
    score_low = float(np.mean(m_low[center]))

    z = None
    if img_id is not None:
        z = z_map.get(str(img_id), None)
    bias = 0.0
    if z is not None:
        zn = (float(z) - _Z_MED) / max((_Z_P90 - _Z_P10) / 2.0, 1e-6)
        zn = float(np.clip(zn, -1.0, 1.0))
        bias = 0.06 * zn

    frac_high = float(np.mean(m_high))
    frac_low = float(np.mean(m_low))
    if frac_high >= frac_low:
        score_high_adj = score_high + bias
        score_low_adj = score_low
    else:
        score_high_adj = score_high
        score_low_adj = score_low + bias

    mask = m_high if score_high_adj >= score_low_adj else m_low

    b = int(border)
    if b > 0:
        mask = mask.copy()  # avoid mutating views (equivalent; only needed if reused)
        mask[:b, :] = False
        mask[-b:, :] = False
        mask[:, :b] = False
        mask[:, -b:] = False

    mask = remove_small_objects(mask.astype(bool), min_size=int(min_size))
    mask = remove_small_holes(mask, area_threshold=int(hole_area))
    if close_r and int(close_r) > 0:
        mask = binary_closing(mask, footprint=disk(int(close_r)))

    return mask.astype(np.uint8)


DO_PLOT = False
if DO_PLOT:
    sample_ids = df["id"].head(6).astype(str).tolist()
    plt.figure(figsize=(18, 6))
    for j, img_id in enumerate(sample_ids):
        img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")
        orig_img = imread(img_path)
        m = heuristic_mask_from_image(orig_img, img_id=img_id)
        plt.subplot(2, 6, j + 1)
        plt.imshow(orig_img, cmap="gray")
        plt.axis("off")
        plt.title(img_id)
        plt.subplot(2, 6, 6 + j + 1)
        plt.imshow(m, cmap="gray")
        plt.axis("off")
        plt.title("mask")
    plt.tight_layout()
    plt.show()




## === cell 7
def _compute_iou_matrix(true_masks, pred_masks):
    if len(true_masks) == 0 or len(pred_masks) == 0:
        return np.zeros((len(true_masks), len(pred_masks)), dtype=np.float32)
    ious = np.zeros((len(true_masks), len(pred_masks)), dtype=np.float32)
    for i, t in enumerate(true_masks):
        t_sum = int(t.sum())
        for j, p in enumerate(pred_masks):
            inter = int(np.sum(t & p))
            union = t_sum + int(p.sum()) - inter
            ious[i, j] = 0.0 if union == 0 else (inter / union)
    return ious


def _extract_instances(bin_mask):
    bin_mask = (bin_mask > 0).astype(np.uint8)
    lab = label(bin_mask)
    inst = []
    for k in range(1, lab.max() + 1):
        m = lab == k
        if m.any():
            inst.append(m.astype(np.uint8))
    return inst


def _ap_instances(y_true, y_pred, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.05)

    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    true_inst = _extract_instances(y_true)
    pred_inst = _extract_instances(y_pred)

    if len(true_inst) == 0 and len(pred_inst) == 0:
        return 1.0

    iou_mat = _compute_iou_matrix(true_inst, pred_inst)

    precisions = []
    for t in thresholds:
        matched_true = set()
        matched_pred = set()

        if iou_mat.size > 0:
            pairs = np.argwhere(iou_mat > t)
            if pairs.size > 0:
                order = np.argsort(iou_mat[pairs[:, 0], pairs[:, 1]])[::-1]
                for idx in order:
                    ti, pj = int(pairs[idx, 0]), int(pairs[idx, 1])
                    if ti in matched_true or pj in matched_pred:
                        continue
                    matched_true.add(ti)
                    matched_pred.add(pj)

        tp = len(matched_true)
        fp = len(pred_inst) - len(matched_pred)
        fn = len(true_inst) - len(matched_true)
        denom = tp + fp + fn
        precisions.append(0.0 if denom == 0 else (tp / denom))

    return float(np.mean(precisions))


train_ids = sorted(
    [f.split(".")[0] for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(".png")]
)
rs = np.random.RandomState(100)
rs.shuffle(train_ids)

n_val = 250
val_ids = train_ids[:n_val]

thr_adj_grid = np.linspace(-0.12, 0.12, 17)

min_size_grid = [20, 30, 40]
hole_area_grid = [20, 40, 60]

border_grid = [0, 2, 4]
close_r_grid = [0, 1, 2]

val_data = []
for img_id in val_ids:
    img_path = os.path.join(TRAIN_IMG_DIR, f"{img_id}.png")
    m_path = os.path.join(TRAIN_MASK_DIR, f"{img_id}.png")
    if (not os.path.exists(img_path)) or (not os.path.exists(m_path)):
        continue
    img = imread(img_path)
    gt = imread(m_path)
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    gt = (gt > 127).astype(np.uint8)

    x_s = _preprocess_image(img)
    try:
        otsu_thr = float(threshold_otsu(x_s))
    except Exception:
        otsu_thr = float(_GLOBAL_FALLBACK_THR)
    if float(np.std(x_s)) < 0.03:
        otsu_thr = float(_GLOBAL_FALLBACK_THR)

    val_data.append((img_id, gt, x_s, otsu_thr))

best_params = None
best_score = -1.0

for thr_adj in thr_adj_grid:
    for min_sz in min_size_grid:
        for hole_ar in hole_area_grid:
            for border in border_grid:
                for close_r in close_r_grid:
                    scores = []
                    for img_id, gt, x_s, otsu_thr in val_data:
                        thr = float(np.clip(otsu_thr + float(thr_adj), 0.0, 1.0))
                        pred = heuristic_mask_from_image(
                            None,
                            override_thr=thr,
                            img_id=img_id,
                            min_size=int(min_sz),
                            hole_area=int(hole_ar),
                            border=int(border),
                            close_r=int(close_r),
                            _x_s=x_s,
                        )
                        scores.append(_ap_instances(gt.T, pred.T))

                    if len(scores) == 0:
                        continue
                    s = float(np.mean(scores))
                    if s > best_score:
                        best_score = s
                        best_params = (
                            float(thr_adj),
                            int(min_sz),
                            int(hole_ar),
                            int(border),
                            int(close_r),
                        )

(
    _CALIBRATED_THR_ADJ,
    _CALIBRATED_MIN_SIZE,
    _CALIBRATED_HOLE_AREA,
    _CALIBRATED_BORDER,
    _CALIBRATED_CLOSE_R,
) = best_params

print(
    "Calibrated params (instance-mAP-like): "
    f"thr_adj={_CALIBRATED_THR_ADJ:+.4f}, min_size={_CALIBRATED_MIN_SIZE}, hole_area={_CALIBRATED_HOLE_AREA}, "
    f"border={_CALIBRATED_BORDER}, close_r={_CALIBRATED_CLOSE_R} "
    f"(val={best_score:.4f})"
)




## === cell 8
_EMPTY_FRACTION_THRESHOLD = 0.0015  # ~15 pixels out of 10201

ids = df["id"].astype(str).values
rles = [""] * len(ids)

for i in tqdm(range(len(ids))):
    img_id = ids[i]
    img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")

    if not os.path.exists(img_path):
        rles[i] = ""
        continue

    orig_img = imread(img_path)
    x_s = _preprocess_image(orig_img)

    try:
        otsu_thr = float(threshold_otsu(x_s))
    except Exception:
        otsu_thr = float(_GLOBAL_FALLBACK_THR)
    if float(np.std(x_s)) < 0.03:
        otsu_thr = float(_GLOBAL_FALLBACK_THR)

    thr = float(np.clip(otsu_thr + float(_CALIBRATED_THR_ADJ), 0.0, 1.0))

    m1 = heuristic_mask_from_image(
        orig_img,
        override_thr=thr,
        img_id=img_id,
        min_size=_CALIBRATED_MIN_SIZE,
        hole_area=_CALIBRATED_HOLE_AREA,
        border=_CALIBRATED_BORDER,
        close_r=_CALIBRATED_CLOSE_R,
    )

    orig_flip = np.fliplr(orig_img)
    x_s_flip = _preprocess_image(orig_flip)
    try:
        otsu_thr_flip = float(threshold_otsu(x_s_flip))
    except Exception:
        otsu_thr_flip = float(_GLOBAL_FALLBACK_THR)
    if float(np.std(x_s_flip)) < 0.03:
        otsu_thr_flip = float(_GLOBAL_FALLBACK_THR)

    thr_flip = float(np.clip(otsu_thr_flip + float(_CALIBRATED_THR_ADJ), 0.0, 1.0))

    m2 = heuristic_mask_from_image(
        orig_flip,
        override_thr=thr_flip,
        img_id=img_id,
        min_size=_CALIBRATED_MIN_SIZE,
        hole_area=_CALIBRATED_HOLE_AREA,
        border=_CALIBRATED_BORDER,
        close_r=_CALIBRATED_CLOSE_R,
    )
    m2 = np.fliplr(m2)

    strict = ((m1 + m2) == 2).astype(np.uint8)  # strict consensus
    if float(np.mean(strict)) < _EMPTY_FRACTION_THRESHOLD:
        decoded_mask = ((m1 + m2) >= 1).astype(np.uint8)  # soft consensus fallback
    else:
        decoded_mask = strict

    crf_output = crf(orig_img, decoded_mask)

    rles[i] = rle_encode(crf_output.T)

df["id"] = ids
df["rle_mask"] = pd.Series(rles, index=df.index).fillna("").astype(str)

out_path = "/kaggle/working/submission.csv"
df.to_csv(out_path, index=False)

print(f"Wrote submission to: {out_path}")
print(df.head())
