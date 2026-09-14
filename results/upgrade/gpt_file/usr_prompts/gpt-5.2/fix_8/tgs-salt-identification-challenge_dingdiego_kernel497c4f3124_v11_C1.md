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

0.8218693206404148

# 6. Current score

0.125

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `pydensecrf` (not installed in this Kaggle environment) by adding a safe fallback that keeps the pipeline running end-to-end while preserving the same “CRF post-process” call sites. I also fix notebook-only syntax (`%matplotlib inline`, `ls`) and missing imports that caused `NameError`s. Because the referenced `../input/k-fold-merger/submission.csv` dataset is not available, I instead read `sample_submission.csv` and generate a valid submission by applying the same loop logic over all test ids (leaving empty masks empty). Finally, I ensure the output is written as a proper `.csv` with the required `id,rle_mask` columns.'
- What this solution (achieved 0.0338) has done: 'Your current 0.0 score comes from submitting (almost) all-empty masks because the code starts from `sample_submission.csv`, which contains blank `rle_mask` values; CRF post-processing cannot create salt where none is predicted. To move the score up toward your target while preserving the existing CRF loop and RLE semantics, the minimal fix is to generate a simple, legitimate baseline mask from each test image itself (Otsu threshold on normalized grayscale) and then run your same CRF/fallback + RLE encode. This keeps the same overall pipeline structure (decode → post-process → encode → write CSV) and only replaces the empty upstream “prediction source” with a deterministic, data-derived baseline so the submission is non-trivial. I also ensure the output stays aligned to `sample_submission` ids and remains a valid `id,rle_mask` CSV.'
- What this solution (achieved 0.0657) has done: 'Your current score is far below the target, so we should improve quality without changing the overall pipeline (baseline mask → optional CRF/fallback → RLE → CSV). The smallest high-impact fix is to correct the main modeling error: salt regions are typically *brighter* than sediment, so the Otsu rule should select `img_n > thr` rather than `<= thr` (your current baseline inverts masks). To further nudge the score upward with minimal risk, we also add a tiny post-process that removes isolated noise and fills small holes (common in this competition) before encoding, while keeping the same training-free approach and same CRF call sites. These changes preserve architecture/loops/metric semantics and should move the score substantially closer to your target.'
- What this solution (achieved 0.028) has done: 'Your score is far below the target (0.0657 vs 0.8219), so we should improve the mask quality while keeping the same no-training pipeline (baseline mask → light cleanup → CRF/fallback → RLE → CSV). The biggest remaining issue is that a single global Otsu threshold per image is too brittle; a minimal, deterministic improvement is to apply Otsu on a lightly denoised image and then use a small “dual-threshold” rule (choose the smaller foreground between `>` and `<`) to avoid catastrophic inversions on different contrast regimes. To further move toward the target without changing the overall approach, we add a standard test-time augmentation that averages predictions from the original and horizontally-flipped image, then thresholds at 0.5 to produce a binary mask. These changes keep your core logic intact (still pure post-processing on test images + same CRF call sites + same RLE format) and typically yield a large jump on this competition.'
- What this solution (achieved 0.0898) has done: 'Your current score (0.028) is far below the target (0.8219), so we should improve mask quality while keeping your same overall pipeline (per-image heuristic mask → light morphology → CRF/fallback → RLE → CSV). The largest low-risk gain is to replace the brittle “binary Otsu then TTA-average” with a slightly richer but still deterministic *probability map* built from the image intensity (centered + scaled) and then threshold it once; this keeps the same approach but yields smoother, more consistent masks. We also sweep a small fixed set of thresholds and pick the best one using a tiny internal validation split built from `train/` masks (no architecture/training changes; just calibrating the final binarization), which usually moves IoU-mAP up substantially. Finally, we keep your CRF call sites and RLE semantics unchanged and still write `crf_correction.csv`.'
- What this solution (achieved 0.1023) has done: 'Your gap to the target is still very large (0.0898 vs 0.8219), so we need a real quality jump while keeping your current “image-only heuristic prob map + TTA + threshold calibration + light cleanup + CRF/fallback + RLE” pipeline intact. The biggest low-risk improvement is to make the heuristic probability map less brittle by adding edge/texture evidence (image gradients) and a mild depth prior (salt likelihood changes with depth), both without introducing any training or changing post-processing semantics. Next, we calibrate the threshold on a larger, stratified subset of training images (still just picking a single global cutoff) to better match the Kaggle metric. Finally, we slightly strengthen the mask cleanup (still deterministic morphology) to reduce speckle false positives that hurt mAP at higher IoUs.'
- What this solution (achieved 0.125) has done: 'Your score (0.1023) is far below the target (0.8219), so we need a real uplift while keeping the same overall pipeline (heuristic prob map + TTA + single global threshold calibration + cleanup + CRF/fallback + RLE). The biggest issue is that the current heuristic is still too weak to approximate a real segmenter; a minimal, metric-aligned improvement is to add a *train-derived* calibration of the probability map itself (a single affine logit scaling) plus a slightly better depth prior learned from train (still no training loop/model—just two scalars). Then we keep your existing threshold sweep, but expand it modestly and fix the metric tie at IoU exactly equal to threshold (`>=`), which better matches common implementations for this competition and usually improves validation selection. These changes preserve the core logic and should move the leaderboard score materially upward without changing the fundamental approach or outputs format.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.color import gray2rgb
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import median, sobel
from skimage.morphology import disk

import matplotlib.pyplot as plt
from tqdm import tqdm

_HAS_DCRF = False
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels

    _HAS_DCRF = True
except ModuleNotFoundError:
    _HAS_DCRF = False

np.random.seed(42)




## === cell 1
def rle_decode(rle_mask):
    """
    rle_mask: run-length as string formatted (start length)
    Returns numpy array shape (101,101), 1 - mask, 0 - background
    """
    if rle_mask is None:
        return np.zeros((101, 101), dtype=np.uint8)
    if isinstance(rle_mask, float) and np.isnan(rle_mask):
        return np.zeros((101, 101), dtype=np.uint8)

    rle_mask = str(rle_mask).strip()
    if rle_mask == "" or rle_mask.lower() == "nan":
        return np.zeros((101, 101), dtype=np.uint8)

    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101)




## === cell 2
BASE_INPUT = "../input/tgs-salt-identification-challenge"
test_path = os.path.join(BASE_INPUT, "test", "images")

sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = "../input/sample_submission.csv"

df = pd.read_csv(sample_path)
assert "id" in df.columns and "rle_mask" in df.columns, "Unexpected submission format."
df["rle_mask"] = df["rle_mask"].astype(object)

depths_path = os.path.join(BASE_INPUT, "depths.csv")
if not os.path.exists(depths_path):
    depths_path = "../input/depths.csv"
depths_df = None
depth_map = {}
if os.path.exists(depths_path):
    depths_df = pd.read_csv(depths_path)
    if "id" in depths_df.columns and "z" in depths_df.columns:
        depth_map = dict(zip(depths_df["id"].astype(str).values, depths_df["z"].values))




## === cell 3
def crf(original_image, mask_img):
    """
    Return labelled image after applying CRF.
    If pydensecrf is not available, return the input mask unchanged (score-neutral vs not running).
    """
    if len(mask_img.shape) < 3:
        mask_img_rgb = gray2rgb(mask_img)
    else:
        mask_img_rgb = mask_img

    if not _HAS_DCRF:
        if len(mask_img_rgb.shape) == 3:
            out = (mask_img_rgb[:, :, 0] > 0).astype(np.uint8)
        else:
            out = (mask_img_rgb > 0).astype(np.uint8)
        return out

    annotated_label = (
        mask_img_rgb[:, :, 0].astype(np.uint32)
        + (mask_img_rgb[:, :, 1].astype(np.uint32) << 8)
        + (mask_img_rgb[:, :, 2].astype(np.uint32) << 16)
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




## === cell 4
def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.

    Note: Competition expects 1-indexed runs in column-major order (top->bottom, left->right).
    Flattening in Fortran order satisfies that.
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 5
def _normalize_01(img2d):
    img2d = img2d.astype(np.float32)
    mn = float(img2d.min())
    mx = float(img2d.max())
    if mx > mn:
        return (img2d - mn) / (mx - mn)
    return np.zeros_like(img2d, dtype=np.float32)


def _sigmoid(x):
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


DEPTH_PRIOR_SLOPE = 0.0
DEPTH_PRIOR_BIAS = 0.0
LOGIT_SCALE = 1.0
LOGIT_BIAS = 0.0


def _depth_prior(img_id):
    """
    Mild, fixed prior based on depth z.
    """
    if not depth_map:
        return 0.0
    z = depth_map.get(str(img_id), None)
    if z is None:
        return 0.0
    zn = (float(z) - 500.0) / 500.0  # roughly centered/scaled
    return float(np.clip(DEPTH_PRIOR_SLOPE * zn + DEPTH_PRIOR_BIAS, -0.25, 0.25))


def baseline_prob_from_image(img, img_id=None):
    """
    Produce a soft salt probability map from intensity + gradients + (learned) depth prior.
    """
    if img.ndim == 3:
        img = img[:, :, 0]

    img_n = _normalize_01(img)
    img_f = median(img_n, footprint=disk(1))

    mu = float(np.mean(img_f))
    sd = float(np.std(img_f) + 1e-6)
    z_int = (img_f - mu) / sd

    g = sobel(img_f).astype(np.float32)
    g = _normalize_01(g)
    z_g = (g - float(np.mean(g))) / (float(np.std(g)) + 1e-6)

    shift = _depth_prior(img_id)

    logit_raw = 1.15 * z_int + 0.35 * z_g + shift
    logit = LOGIT_SCALE * logit_raw + LOGIT_BIAS

    p = _sigmoid(logit)
    return p.astype(np.float32)


def predict_prob_with_tta(orig_img, img_id=None):
    """
    Same TTA concept as before, on probabilities.
    """
    p1 = baseline_prob_from_image(orig_img, img_id=img_id)

    img_flip = np.fliplr(orig_img)
    p2 = baseline_prob_from_image(img_flip, img_id=img_id)
    p2 = np.fliplr(p2)

    p = 0.5 * (p1 + p2)
    return p.astype(np.float32)


def cleanup_mask(mask):
    """
    Slightly stronger deterministic cleanup.
    """
    m = mask > 0
    m = remove_small_objects(m, min_size=30)
    m = remove_small_holes(m, area_threshold=30)
    return m.astype(np.uint8)




## === cell 6
def iou_score(y_true, y_pred):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def mean_ap_iou(y_true, y_pred):
    """
    TGS metric for a single image with one object: average precision across IoU thresholds.

    Change (score-improving, minimal): use >= instead of > at the threshold boundary.
    This aligns with common Kaggle implementations and stabilizes calibration selection.
    """
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    if y_true.sum() == 0 and y_pred.sum() == 0:
        return 1.0

    iou = iou_score(y_true, y_pred)
    thresholds = np.arange(0.5, 1.0, 0.05)
    return float(np.mean([1.0 if iou >= t else 0.0 for t in thresholds]))


def find_train_base_dir():
    c1 = "../input/tgs-salt-identification-challenge/train"
    c2 = "../input/train"
    if os.path.exists(c1):
        return c1
    if os.path.exists(c2):
        return c2
    return None


def _fit_depth_prior_from_train(max_images=1200, seed=42):
    """
    Change (score-improving, still minimal/no training loop): estimate a weak linear prior
    P(salt | depth) from train masks (two scalars), then use it as an additive logit shift.
    """
    base = find_train_base_dir()
    if base is None or not depth_map:
        return 0.0, 0.0

    img_dir = os.path.join(base, "images")
    msk_dir = os.path.join(base, "masks")
    if not (os.path.exists(img_dir) and os.path.exists(msk_dir)):
        return 0.0, 0.0

    ids_all = [fn[:-4] for fn in os.listdir(img_dir) if fn.endswith(".png")]
    if len(ids_all) == 0:
        return 0.0, 0.0

    rng = np.random.RandomState(seed)
    rng.shuffle(ids_all)
    ids_all = ids_all[: min(max_images, len(ids_all))]

    xs = []
    ys = []
    for img_id in ids_all:
        z = depth_map.get(str(img_id), None)
        if z is None:
            continue
        msk_path = os.path.join(msk_dir, f"{img_id}.png")
        if not os.path.exists(msk_path):
            continue
        msk = imread(msk_path)
        if msk.ndim == 3:
            msk = msk[:, :, 0]
        y = 1.0 if (msk > 0).any() else 0.0  # presence prior (safe, weak signal)
        zn = (float(z) - 500.0) / 500.0
        xs.append(zn)
        ys.append(y)

    if len(xs) < 50:
        return 0.0, 0.0

    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)

    y_mean = float(np.clip(y.mean(), 1e-3, 1.0 - 1e-3))
    b = float(np.log(y_mean / (1.0 - y_mean)))
    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    var = float(np.var(x) + 1e-6)
    a = float(np.clip(4.0 * cov / var, -0.20, 0.20))  # weak slope; prevents overuse

    b = float(np.clip(b, -0.20, 0.20))  # weak bias
    return a, b


def _calibrate_logit_affine(max_images=800, seed=42):
    """
    Change (score-improving, minimal): calibrate a single affine transformation of the heuristic
    logit so that probabilities are better separated for the final thresholding step.
    We select (scale, bias) by maximizing mean AP@IoU on a train subset.
    """
    base = find_train_base_dir()
    if base is None:
        return 1.0, 0.0

    img_dir = os.path.join(base, "images")
    msk_dir = os.path.join(base, "masks")
    if not (os.path.exists(img_dir) and os.path.exists(msk_dir)):
        return 1.0, 0.0

    ids_all = [fn[:-4] for fn in os.listdir(img_dir) if fn.endswith(".png")]
    if len(ids_all) == 0:
        return 1.0, 0.0

    rng = np.random.RandomState(seed)
    rng.shuffle(ids_all)
    ids = ids_all[: min(max_images, len(ids_all))]

    scales = [0.9, 1.0, 1.1, 1.25]
    biases = [-0.25, -0.10, 0.0, 0.10, 0.25]

    thr0 = 0.50

    best = (1.0, 0.0)
    best_score = -1.0

    for s in scales:
        for b in biases:
            vals = []
            for img_id in ids:
                img_path = os.path.join(img_dir, f"{img_id}.png")
                msk_path = os.path.join(msk_dir, f"{img_id}.png")
                if not (os.path.exists(img_path) and os.path.exists(msk_path)):
                    continue

                img = imread(img_path)
                msk = imread(msk_path)
                if msk.ndim == 3:
                    msk = msk[:, :, 0]
                y_true = (msk > 0).astype(np.uint8)

                if img.ndim == 3:
                    img2 = img[:, :, 0]
                else:
                    img2 = img
                img_n = _normalize_01(img2)
                img_f = median(img_n, footprint=disk(1))
                mu = float(np.mean(img_f))
                sd = float(np.std(img_f) + 1e-6)
                z_int = (img_f - mu) / sd
                g = sobel(img_f).astype(np.float32)
                g = _normalize_01(g)
                z_g = (g - float(np.mean(g))) / (float(np.std(g)) + 1e-6)
                shift = _depth_prior(img_id)
                logit_raw = 1.15 * z_int + 0.35 * z_g + shift
                p = _sigmoid(s * logit_raw + b)

                y_pred = (p >= thr0).astype(np.uint8)
                y_pred = cleanup_mask(y_pred)

                vals.append(mean_ap_iou(y_true, y_pred))

            if len(vals) == 0:
                continue
            sc = float(np.mean(vals))
            if sc > best_score:
                best_score = sc
                best = (float(s), float(b))

    return best


def calibrate_threshold(max_images=1000, seed=42):
    """
    Calibrate the single global threshold on a larger, depth-stratified sample.

    Change (score-improving, minimal): slightly expand the threshold grid to allow a better
    match after probability calibration.
    """
    base = find_train_base_dir()
    if base is None:
        return 0.5

    img_dir = os.path.join(base, "images")
    msk_dir = os.path.join(base, "masks")
    if not (os.path.exists(img_dir) and os.path.exists(msk_dir)):
        return 0.5

    ids_all = [fn[:-4] for fn in os.listdir(img_dir) if fn.endswith(".png")]
    if len(ids_all) == 0:
        return 0.5

    rng = np.random.RandomState(seed)
    ids = np.array(ids_all, dtype=object)

    if depth_map:
        zs = np.array([depth_map.get(str(i), np.nan) for i in ids], dtype=np.float32)
        ok = ~np.isnan(zs)
        ids_ok = ids[ok]
        zs_ok = zs[ok]
        if len(ids_ok) > 0:
            q = np.quantile(zs_ok, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            chosen = []
            per_bin = max(1, int(max_images / 5))
            for b in range(5):
                lo, hi = q[b], q[b + 1]
                inb = ids_ok[(zs_ok >= lo) & (zs_ok <= hi)]
                if len(inb) == 0:
                    continue
                rng.shuffle(inb)
                chosen.extend(list(inb[:per_bin]))
            if len(chosen) > 0:
                ids = np.array(chosen, dtype=object)

    rng.shuffle(ids)
    ids = ids[: min(max_images, len(ids))]

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    scores = {t: [] for t in thresholds}

    for img_id in ids:
        img_path = os.path.join(img_dir, f"{img_id}.png")
        msk_path = os.path.join(msk_dir, f"{img_id}.png")
        if not (os.path.exists(img_path) and os.path.exists(msk_path)):
            continue

        img = imread(img_path)
        msk = imread(msk_path)
        if msk.ndim == 3:
            msk = msk[:, :, 0]
        y_true = (msk > 0).astype(np.uint8)

        p = predict_prob_with_tta(img, img_id=img_id)

        for t in thresholds:
            y_pred = (p >= t).astype(np.uint8)
            y_pred = cleanup_mask(y_pred)
            scores[t].append(mean_ap_iou(y_true, y_pred))

    valid = {t: v for t, v in scores.items() if len(v) > 0}
    if not valid:
        return 0.5

    mean_scores = {t: float(np.mean(v)) for t, v in valid.items()}
    best_t = max(mean_scores, key=lambda x: mean_scores[x])
    return float(best_t)


DEPTH_PRIOR_SLOPE, DEPTH_PRIOR_BIAS = _fit_depth_prior_from_train(
    max_images=1200, seed=42
)
LOGIT_SCALE, LOGIT_BIAS = _calibrate_logit_affine(max_images=800, seed=42)

BEST_THR = calibrate_threshold(max_images=1000, seed=42)
print(f"Fitted depth prior slope/bias: {DEPTH_PRIOR_SLOPE:.3f}, {DEPTH_PRIOR_BIAS:.3f}")
print(f"Calibrated logit scale/bias: {LOGIT_SCALE:.3f}, {LOGIT_BIAS:.3f}")
print(f"Calibrated probability threshold: {BEST_THR:.2f}")



## === cell 7
try:
    nImgs = 3
    start_i = np.random.randint(0, len(df))
    shown = 0
    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    i = start_i
    while shown < nImgs and i < len(df):
        img_id = df.loc[i, "id"]
        img_file = os.path.join(test_path, f"{img_id}.png")
        if os.path.exists(img_file):
            orig_img = imread(img_file)

            prob = predict_prob_with_tta(orig_img, img_id=img_id)
            base_mask = (prob >= BEST_THR).astype(np.uint8)
            base_mask = cleanup_mask(base_mask)
            crf_output = crf(orig_img, base_mask)

            plt.subplot(nImgs, 4, 4 * shown + 1)
            plt.imshow(orig_img, cmap="gray")
            plt.title(f"Original: {img_id}")
            plt.axis("off")

            plt.subplot(nImgs, 4, 4 * shown + 2)
            plt.imshow(prob, cmap="viridis", vmin=0, vmax=1)
            plt.title("Prob (heuristic+TTA)")
            plt.axis("off")

            plt.subplot(nImgs, 4, 4 * shown + 3)
            plt.imshow(base_mask, cmap="gray")
            plt.title(f"Mask @ {BEST_THR:.2f}")
            plt.axis("off")

            plt.subplot(nImgs, 4, 4 * shown + 4)
            plt.imshow(crf_output, cmap="gray")
            plt.title("After CRF (or fallback)")
            plt.axis("off")

            shown += 1
        i += 1
    plt.close()
except Exception:
    pass



## === cell 8
"""
Applying CRF on the predicted mask.

Pipeline preserved:
- deterministic probability map per test image (denoise + intensity + gradients + weak depth prior)
- average with horizontal flip TTA
- threshold with a globally calibrated cutoff (from train masks)
- lightly clean it
- apply the same CRF/fallback + RLE encoding
"""
for i in tqdm(range(df.shape[0]), desc="Predict + post-process masks"):
    img_id = df.loc[i, "id"]
    img_file = os.path.join(test_path, f"{img_id}.png")

    if not os.path.exists(img_file):
        df.loc[i, "rle_mask"] = ""
        continue

    orig_img = imread(img_file)

    prob = predict_prob_with_tta(orig_img, img_id=img_id)
    base_mask = (prob >= BEST_THR).astype(np.uint8)
    base_mask = cleanup_mask(base_mask)

    crf_output = crf(orig_img, base_mask)
    df.loc[i, "rle_mask"] = rle_encode(crf_output)



## === cell 9
out_path = "crf_correction.csv"
df.to_csv(out_path, index=False)

assert out_path.endswith(".csv")
assert df.shape[0] > 0
assert list(df.columns) == ["id", "rle_mask"]
print(f"Wrote submission: {out_path} (rows={len(df)})")
