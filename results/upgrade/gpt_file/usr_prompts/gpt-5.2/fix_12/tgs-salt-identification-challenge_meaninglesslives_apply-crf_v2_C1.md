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

0.7702985720467324

# 6. Current score

0.1125

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I remove the hard dependency on `pydensecrf` (not installed) by adding a safe fallback that keeps the pipeline running end-to-end: if CRF isn’t available, we apply a small, deterministic morphological post-processing instead. I also remove notebook-only magic (`%matplotlib inline`) and fix the data input path: your script currently reads a submission file that doesn’t exist in this environment, so it instead start from the provided `sample_submission.csv`. Finally, I make the NaN/empty-mask checks correct and ensure the script always writes a valid `crf_correction.csv` with the required `id,rle_mask` columns.'
- What this solution (achieved 0.5221) has done: 'Your current script only post-processes masks that already exist in `sample_submission.csv`, but that file is intentionally empty/blank for this competition, so you end up submitting essentially all-empty masks and capping the score. To move toward the target, the minimal fix is to generate a reasonable baseline mask from each test image (simple deterministic thresholding) and then apply your existing CRF/fallback cleanup on top, keeping the same overall pipeline structure. I also fix the RLE flatten order to the competition’s required top-to-bottom then left-to-right convention (Fortran order), which can materially improve IoU without changing the modeling approach. These are small, deterministic changes that should raise the score substantially toward your 0.77 target.'
- What this solution (achieved 0.0736) has done: 'Your score is capped because you’re generating masks from pure Otsu thresholding, which is a weak proxy for “salt vs sediment” on these seismic images; we can improve toward the 0.77 target with minimal, fully deterministic changes that keep the same pipeline (initial mask → CRF/fallback cleanup → RLE). I add a tiny calibration step that learns (on the provided training masks) which thresholding direction (bright-is-salt vs dark-is-salt) and which quantile-based threshold works best under the competition’s IoU-sweep metric, then apply that same rule to all test images. This preserves the core logic (still simple thresholding + morphological cleanup + same CRF function) while aligning the binarization to the actual data distribution, which should materially increase IoU. I also ensure we only use the training masks for calibration (no leakage from test) and keep runtime well under the limit by evaluating a small grid.'
- What this solution (achieved 0.0795) has done: 'Your current pipeline is already structurally sound (calibrate a global thresholding rule on train → generate test masks → cleanup → RLE), but the score suggests the post-processing is harming masks and/or the calibration objective is misaligned with the competition metric (instance-style precision over IoU thresholds). I make two minimal, metric-aligned adjustments: (1) calibrate using the official “empty mask” handling and per-image AP definition used in this competition, and (2) make the morphological fallback less destructive by using hole-filling + remove-small-objects rather than open+close, which often erases thin salt structures. These changes preserve your core logic (still thresholding + deterministic cleanup + same RLE and CRF hook) while making the chosen rule and cleanup better match what mAP@IoU rewards. The rest (paths, I/O, submission format) is kept identical so it still runs end-to-end and writes `crf_correction.csv`.'
- What this solution (achieved 0.0636) has done: 'Your score is far below the target, and the most likely cause (without changing your core “threshold → cleanup/CRF → RLE” logic) is that the current morphological step inside `initial_mask_from_image()` is overly destructive and can erase thin/fragmented salt structures before the later cleanup runs. I keep the same pipeline but (1) remove the opening/closing from the initial mask (so calibration and inference start from a less-biased mask) and (2) slightly relax the fallback cleanup thresholds in `crf()` so it doesn’t delete small true salt regions. These are minimal, deterministic changes that should materially improve IoU/mAP while preserving your overall approach and submission semantics. Everything else (paths, calibration loop, metric function, RLE order, output filename) stays the same and still produces `crf_correction.csv`.'
- What this solution (achieved 0.0809) has done: 'Your current score is far below the target, so we should improve (not just tweak) while keeping the same core pipeline: global threshold rule calibration on train → initial binary mask on test → CRF/fallback cleanup → RLE. The biggest likely issue is that your CRF is being run on the binary mask as the “image” (it never uses the true grayscale intensities), which makes the CRF essentially blind and often harmful; we keep the CRF step but feed it a proper 3‑channel image derived from the real grayscale input. To better match the competition’s mask characteristics without changing the approach, we also add a tiny, deterministic depth-aware bias term during calibration/inference (still just thresholding, but with one extra scalar feature) and slightly widen the small grid search; this should move the score materially toward the 0.77 target. Everything else (paths, metric, loops, RLE order, output file) stays the same and the script still writes a valid `crf_correction.csv`.'
- What this solution (achieved 0.1005) has done: 'Your current 0.0809 score is far below the 0.7703 target, so we should improve with the smallest changes that keep the same “calibrate global threshold rule on train → predict on test → CRF/fallback cleanup → RLE” pipeline. The most likely blocker is that this code is unintentionally “training” against the wrong ground truth: it reads `train/masks/*.png` as if they were binary masks, but in this dataset they are encoded as RLE in `train.csv`, so your calibration is effectively optimizing against garbage and picks a bad rule. I fix calibration to decode true masks from `train.csv` via your existing `rle_decode()` (preserving your core logic), and also extend the bias grid slightly (still tiny) so the calibrated rule can actually find a reasonable threshold. Everything else (paths, CRF hook, morphological fallback, RLE order, output file) stays the same and it still writes `crf_correction.csv`.'
- What this solution (achieved 0.1015) has done: 'The timeout is dominated by `calibrate_threshold_rule()`, which currently runs an expensive grid search (hundreds of parameter combos) over up to 900 training images and also runs CRF inside that inner loop—this is orders of magnitude too slow. To preserve the exact algorithm/semantics while making it fast, the optimized version caches all training images and decoded masks once, precomputes per-image quantities (z-norm and threshold values needed for Otsu/quantile), and evaluates all candidate rules using vectorized NumPy without repeatedly loading images or re-running thresholding. It also memoizes CRF results per (image, initial-mask) during calibration to avoid redundant CRF calls across candidates, and speeds up test-time by iterating with `itertuples()` and writing results to a preallocated list instead of `df.loc` in a Python loop.'
- What this solution (achieved 0.1247) has done: 'I keep your exact pipeline (global threshold rule calibration on train → initial binary mask on test → CRF/fallback cleanup → RLE) but fix the main scoring blocker: the fallback cleanup is too weak and leaves lots of speckle/fragmentation, which hurts the IoU-sweep mAP. With DenseCRF not installed here, the fallback becomes your “model”, so we make it slightly more mask-appropriate by adding a deterministic closing + hole-filling step while still retaining your existing remove-small-* filters. I also add a tiny, deterministic “connected-component keep-largest” option during calibration/inference (only if it improves the calibrated objective), which often helps this dataset by suppressing scattered false positives without changing the overall approach. These are minimal changes directly aimed at improving mAP toward the 0.77 target while preserving architecture/loops and keeping runtime under the limit.'
- What this solution (achieved 0.1125) has done: 'Your current score (0.1247) is far below the target (0.7703), so we should improve it while keeping the same pipeline (global threshold calibration → initial mask → CRF/fallback cleanup → RLE). The biggest low-risk gain without changing the “model” is to align the test-time prediction distribution with what the calibration expects: right now calibration uses CRF in-loop and learns `empty_cut` and `keep_largest`, but the fallback cleanup can still create too many false positives/fragmentation. I make two minimal, metric-relevant adjustments: (1) add a tiny, deterministic test-time probability calibration by selecting a single global binarization threshold on the *refined* masks using the same training metric (this keeps core logic: still threshold→cleanup→RLE, but adds one final scalar threshold), and (2) make `keep_largest` conditional on predicted mask area (so we don’t destroy legitimate multi-lobe salt). These changes are small, deterministic, and aimed specifically at improving mAP@IoU by reducing false positives while preserving true positives.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    disk,
)
from skimage.filters import threshold_otsu
from skimage.measure import label

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels

    HAS_DCRF = True
except Exception:
    HAS_DCRF = False

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")




## === cell 1
def rle_decode(mask_rle):
    """
    mask_rle: run-length as string formatted (start length)
    Returns numpy array, 1 - mask, 0 - background, shape (101, 101)

    NOTE: Decode follows Kaggle's convention used in this competition:
    pixels are 1-indexed and ordered top-to-bottom, then left-to-right,
    which corresponds to flatten(order="F").
    """
    if mask_rle is None or (isinstance(mask_rle, float) and np.isnan(mask_rle)):
        return np.zeros((101, 101), dtype=np.uint8)
    s = str(mask_rle).strip()
    if s == "" or s.lower() == "nan":
        return np.zeros((101, 101), dtype=np.uint8)

    s = s.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((101, 101), order="F")




## === cell 2
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted

    NOTE: Encode follows Kaggle's convention used in this competition:
    flatten(order="F") (top-to-bottom, then left-to-right).
    """
    img = (img > 0).astype(np.uint8)
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
"""
Function which returns the labelled image after applying CRF.
If DenseCRF is not installed, apply a deterministic morphological cleanup fallback.
"""


def _to_3ch_uint8_from_gray(orig_img):
    """
    DenseCRF pairwise terms need the true image appearance; deterministically convert
    the real grayscale image to a 3-channel uint8 image so CRF can work as intended.
    """
    if orig_img.ndim == 3:
        g = orig_img[:, :, 0].astype(np.float32)
    else:
        g = orig_img.astype(np.float32)
    g = g - g.min()
    denom = float(g.max()) if float(g.max()) > 0 else 1.0
    g = (255.0 * (g / denom)).clip(0, 255).astype(np.uint8)
    return gray2rgb(g)


def _keep_largest_component(mask_bool):
    lab = label(mask_bool, connectivity=1)
    if lab.max() == 0:
        return mask_bool
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = counts.argmax()
    return lab == keep


def crf(original_image, annotated_image, use_2d=True, keep_largest=False):
    if annotated_image is None:
        mask = np.zeros(
            (original_image.shape[0], original_image.shape[1]), dtype=np.uint8
        )
    else:
        mask = (annotated_image > 0).astype(np.uint8)

    if HAS_DCRF and use_2d:
        img_rgb = _to_3ch_uint8_from_gray(original_image)

        if len(mask.shape) < 3:
            annotated_rgb = gray2rgb(mask)
        else:
            annotated_rgb = mask

        annotated_label = (
            annotated_rgb[:, :, 0].astype(np.uint32)
            + (annotated_rgb[:, :, 1].astype(np.uint32) << 8)
            + (annotated_rgb[:, :, 2].astype(np.uint32) << 16)
        )
        _, labels = np.unique(annotated_label, return_inverse=True)

        n_labels = 2
        d = dcrf.DenseCRF2D(img_rgb.shape[1], img_rgb.shape[0], n_labels)

        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(
            sxy=(3, 3),
            compat=3,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC,
        )

        d.addPairwiseBilateral(
            sxy=(8, 8),
            srgb=(13, 13, 13),
            rgbim=img_rgb,
            compat=5,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC,
        )

        Q = d.inference(10)
        MAP = np.argmax(Q, axis=0).astype(np.uint8)
        out = MAP.reshape((img_rgb.shape[0], img_rgb.shape[1])).astype(bool)
        if keep_largest:
            out = _keep_largest_component(out)
        return out.astype(np.uint8)

    m = mask.astype(bool)
    m = binary_closing(m, footprint=disk(1))
    m = remove_small_holes(m, area_threshold=24)
    m = remove_small_objects(m, min_size=20)
    if keep_largest:
        m = _keep_largest_component(m)
    return m.astype(np.uint8)




## === cell 4
def iou_binary(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0  # both empty
    return inter / union


def tgs_iou_metric(y_true, y_pred, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.05)

    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    true_sum = int(y_true.sum())
    pred_sum = int(y_pred.sum())

    if true_sum == 0 and pred_sum == 0:
        return 1.0
    if true_sum == 0 and pred_sum > 0:
        return 0.0
    if true_sum > 0 and pred_sum == 0:
        return 0.0

    iou = iou_binary(y_true, y_pred)
    return float(np.mean(iou > thresholds))


def _img_to_gray_float(orig_img):
    if orig_img.ndim == 3:
        img = orig_img[:, :, 0]
    else:
        img = orig_img
    return img.astype(np.float32)


def initial_mask_from_image(orig_img, method="otsu", q=0.5, invert=False, bias=0.0):
    """
    Global thresholding to produce an initial binary mask, with a small additive bias.
    """
    img = _img_to_gray_float(orig_img)

    if method == "otsu":
        t = float(threshold_otsu(img))
    elif method == "quantile":
        t = float(np.quantile(img, q))
    else:
        raise ValueError("method must be 'otsu' or 'quantile'")

    t = float(t + bias)

    if invert:
        mask = (img < t).astype(np.uint8)
    else:
        mask = (img > t).astype(np.uint8)

    return mask.astype(np.uint8)




## === cell 5
BASE = "../input/tgs-salt-identification-challenge"

test_path = os.path.join(BASE, "test", "images")
train_img_path = os.path.join(BASE, "train", "images")
train_mask_path = os.path.join(BASE, "train", "masks")

sub_path = os.path.join(BASE, "sample_submission.csv")
df = pd.read_csv(sub_path)

assert (
    "id" in df.columns and "rle_mask" in df.columns
), "Submission template must have columns: id, rle_mask"

depths_path = os.path.join(BASE, "depths.csv")
depths_df = pd.read_csv(depths_path)
depth_map = dict(
    zip(depths_df["id"].astype(str).values, depths_df["z"].astype(float).values)
)
z_mean = float(depths_df["z"].mean())
z_std = (
    float(depths_df["z"].std(ddof=0)) if float(depths_df["z"].std(ddof=0)) > 0 else 1.0
)


def _z_norm(img_id):
    z = float(depth_map.get(str(img_id), z_mean))
    return (z - z_mean) / z_std


train_csv_path = os.path.join(BASE, "train.csv")
train_df = pd.read_csv(train_csv_path)
train_rle_map = dict(
    zip(train_df["id"].astype(str).values, train_df["rle_mask"].values)
)




## === cell 6
def calibrate_threshold_rule(max_train=900, seed=1337):
    rng = np.random.RandomState(seed)

    ids_all = train_df["id"].astype(str).values.tolist()
    if len(ids_all) == 0:
        return {
            "method": "otsu",
            "q": 0.5,
            "invert": False,
            "bias_k": 0.0,
            "empty_cut": 0.0,
            "keep_largest": False,
        }

    if len(ids_all) > max_train:
        ids = list(rng.choice(ids_all, size=max_train, replace=False))
    else:
        ids = ids_all

    candidate_q = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    candidate_bias_k = [-12.0, -6.0, -3.0, -1.5, 0.0, 1.5, 3.0, 6.0, 12.0]
    candidate_empty_cut = [0.0, 0.002, 0.005, 0.01, 0.02]
    candidate_keep_largest = [False, True]

    candidates = []
    for q in candidate_q:
        for invert in (False, True):
            for k in candidate_bias_k:
                for ecut in candidate_empty_cut:
                    for kl in candidate_keep_largest:
                        candidates.append(
                            ("quantile", q, invert, float(k), float(ecut), bool(kl))
                        )
    for invert in (False, True):
        for k in candidate_bias_k:
            for ecut in candidate_empty_cut:
                for kl in candidate_keep_largest:
                    candidates.append(
                        ("otsu", 0.5, invert, float(k), float(ecut), bool(kl))
                    )

    items = []
    needed_q = sorted(set(candidate_q))
    for img_id in ids:
        img_fp = os.path.join(train_img_path, f"{img_id}.png")
        if not os.path.exists(img_fp):
            continue
        img = imread(img_fp)
        img_gray = _img_to_gray_float(img)  # float32
        y_true = rle_decode(train_rle_map.get(str(img_id), ""))

        otsu_t = float(threshold_otsu(img_gray))
        quant_ts = {q: float(np.quantile(img_gray, q)) for q in needed_q}

        items.append(
            {
                "id": str(img_id),
                "img": img,
                "gray": img_gray,
                "y_true": y_true,
                "z": float(_z_norm(img_id)),
                "otsu_t": otsu_t,
                "quant_ts": quant_ts,
            }
        )

    if len(items) == 0:
        return {
            "method": "otsu",
            "q": 0.5,
            "invert": False,
            "bias_k": 0.0,
            "empty_cut": 0.0,
            "keep_largest": False,
        }

    crf_cache = {}  # (img_id, mask_bytes, keep_largest) -> refined_mask_uint8
    thresholds = np.arange(0.5, 1.0, 0.05)

    best = None
    best_score = -1.0

    for method, q, invert, bias_k, empty_cut, keep_largest in candidates:
        total = 0.0
        n = 0
        for it in items:
            bias = float(bias_k * it["z"])
            if method == "otsu":
                t = float(it["otsu_t"] + bias)
            else:
                t = float(it["quant_ts"][float(q)] + bias)

            if invert:
                y0 = (it["gray"] < t).astype(np.uint8)
            else:
                y0 = (it["gray"] > t).astype(np.uint8)

            if float(y0.mean()) < float(empty_cut):
                y0 = np.zeros_like(y0, dtype=np.uint8)

            key = (it["id"], y0.tobytes(), bool(keep_largest))
            y_pred = crf_cache.get(key)
            if y_pred is None:
                y_pred = crf(it["img"], y0, keep_largest=bool(keep_largest))
                crf_cache[key] = y_pred

            y_true = it["y_true"]
            true_sum = int(y_true.sum())
            pred_sum = int(y_pred.sum())
            if true_sum == 0 and pred_sum == 0:
                score = 1.0
            elif true_sum == 0 and pred_sum > 0:
                score = 0.0
            elif true_sum > 0 and pred_sum == 0:
                score = 0.0
            else:
                inter = np.logical_and(y_true.astype(bool), y_pred.astype(bool)).sum()
                union = np.logical_or(y_true.astype(bool), y_pred.astype(bool)).sum()
                iou = 1.0 if union == 0 else (inter / union)
                score = float(np.mean(iou > thresholds))

            total += score
            n += 1

        if n == 0:
            continue
        s = float(total / n)
        if s > best_score:
            best_score = s
            best = {
                "method": method,
                "q": float(q),
                "invert": bool(invert),
                "bias_k": float(bias_k),
                "empty_cut": float(empty_cut),
                "keep_largest": bool(keep_largest),
            }

    if best is None:
        return {
            "method": "otsu",
            "q": 0.5,
            "invert": False,
            "bias_k": 0.0,
            "empty_cut": 0.0,
            "keep_largest": False,
        }
    return best


rule = calibrate_threshold_rule(max_train=900, seed=1337)
print(f"Calibrated rule: {rule} (DenseCRF available: {HAS_DCRF})")




## === cell 7
def calibrate_refined_cut(rule, max_train=600, seed=1337):
    rng = np.random.RandomState(seed)
    ids_all = train_df["id"].astype(str).values.tolist()
    if len(ids_all) == 0:
        return 0.5

    if len(ids_all) > max_train:
        ids = list(rng.choice(ids_all, size=max_train, replace=False))
    else:
        ids = ids_all

    candidate_area_cut = [
        0.0,
        0.001,
        0.002,
        0.004,
        0.006,
        0.008,
        0.01,
        0.015,
        0.02,
        0.03,
    ]

    thresholds = np.arange(0.5, 1.0, 0.05)
    best_cut = 0.0
    best_score = -1.0

    for area_cut in candidate_area_cut:
        total = 0.0
        n = 0
        for img_id in ids:
            img_fp = os.path.join(train_img_path, f"{img_id}.png")
            if not os.path.exists(img_fp):
                continue
            img = imread(img_fp)
            gray = _img_to_gray_float(img)
            y_true = rle_decode(train_rle_map.get(str(img_id), ""))

            bias = float(rule.get("bias_k", 0.0) * _z_norm(img_id))
            y0 = initial_mask_from_image(
                img,
                method=rule["method"],
                q=rule["q"],
                invert=rule["invert"],
                bias=bias,
            )
            if float(y0.mean()) < float(rule.get("empty_cut", 0.0)):
                y0 = np.zeros_like(y0, dtype=np.uint8)

            y_ref = crf(img, y0, keep_largest=False)
            if (
                float(y_ref.mean()) > 0
                and float(y_ref.mean()) < 0.10
                and bool(rule.get("keep_largest", False))
            ):
                y_ref = crf(img, y0, keep_largest=True)

            if float(y_ref.mean()) < float(area_cut):
                y_ref = np.zeros_like(y_ref, dtype=np.uint8)

            true_sum = int(y_true.sum())
            pred_sum = int(y_ref.sum())
            if true_sum == 0 and pred_sum == 0:
                score = 1.0
            elif true_sum == 0 and pred_sum > 0:
                score = 0.0
            elif true_sum > 0 and pred_sum == 0:
                score = 0.0
            else:
                inter = np.logical_and(y_true.astype(bool), y_ref.astype(bool)).sum()
                union = np.logical_or(y_true.astype(bool), y_ref.astype(bool)).sum()
                iou = 1.0 if union == 0 else (inter / union)
                score = float(np.mean(iou > thresholds))

            total += score
            n += 1

        if n == 0:
            continue
        s = float(total / n)
        if s > best_score:
            best_score = s
            best_cut = float(area_cut)

    return float(best_cut)


refined_area_cut = calibrate_refined_cut(rule, max_train=600, seed=1337)
print(f"Calibrated refined_area_cut: {refined_area_cut}")



## === cell 8
rles = [None] * df.shape[0]
for idx, img_id in tqdm(
    list(enumerate(df["id"].astype(str).values)), total=df.shape[0]
):
    img_path = os.path.join(test_path, f"{img_id}.png")
    if not os.path.exists(img_path):
        rles[idx] = df.loc[idx, "rle_mask"]
        continue

    orig_img = imread(img_path)

    bias = float(rule.get("bias_k", 0.0) * _z_norm(img_id))
    decoded_mask = initial_mask_from_image(
        orig_img, method=rule["method"], q=rule["q"], invert=rule["invert"], bias=bias
    )

    if float(decoded_mask.mean()) < float(rule.get("empty_cut", 0.0)):
        decoded_mask = np.zeros_like(decoded_mask, dtype=np.uint8)

    crf_output = crf(orig_img, decoded_mask, keep_largest=False)
    if (
        float(crf_output.mean()) > 0
        and float(crf_output.mean()) < 0.10
        and bool(rule.get("keep_largest", False))
    ):
        crf_output = crf(orig_img, decoded_mask, keep_largest=True)

    if float(crf_output.mean()) < float(refined_area_cut):
        crf_output = np.zeros_like(crf_output, dtype=np.uint8)

    rles[idx] = rle_encode(crf_output)

df["rle_mask"] = rles



## === cell 9
out_path = "crf_correction.csv"
df.to_csv(out_path, index=False)

assert out_path.endswith(".csv")
chk = pd.read_csv(out_path)
assert list(chk.columns) == ["id", "rle_mask"]
assert chk.shape[0] == df.shape[0]
print(f"Wrote {out_path} with shape {chk.shape}. DenseCRF available: {HAS_DCRF}")
