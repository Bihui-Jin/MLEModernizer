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

0.0772

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the notebook runnable in this Kaggle environment by (1) removing the hard dependency on `pydensecrf` (it’s not installed) and replacing the CRF step with a safe no-op fallback that preserves the pipeline structure. I also fix the missing/incorrect imports and the Jupyter-only `%matplotlib inline` so the script runs as a normal `.py`. Finally, I load `sample_submission.csv` from the provided competition input path (instead of a non-existent custom dataset path) and ensure we always write a valid `submission.csv` with the required `id,rle_mask` columns and `.csv` suffix.'
- What this solution (achieved 0.0311) has done: 'Your current 0.0 score comes from generating predictions by post-processing the *sample_submission* masks, which are essentially dummy placeholders and not model outputs; even with CRF, this stays near-empty and performs terribly. To move toward the 0.8167 target with minimal disruption, I keep your pipeline structure (load test ids → create a mask → optional CRF → RLE encode → write submission) but replace the input masks with a simple image-derived heuristic mask computed directly from each test image (no training, no new model). I also ensure every id gets a prediction (no skipped rows), because missing/blank masks across the board can crater the score. The CRF fallback remains intact and still acts as optional post-processing when available.'
- What this solution (achieved 0.0728) has done: 'Your score is far below the target, and the main reason is that the current “heuristic from test image intensity + optional CRF” produces masks that are essentially unrelated to salt regions. To move toward the target with minimal disruption, I keep the same overall pipeline (read test image → create a mask → optional CRF → RLE → write CSV) but make the heuristic more aligned with typical salt masks by (1) using Otsu thresholding (per-image, robust) instead of a fixed percentile threshold, (2) choosing the foreground side (bright vs dark) via a simple center-prior (salt tends to be more central), and (3) applying small connected-component cleanup (remove tiny components, fill small holes) to reduce false positives/fragmentation which strongly hurts mean AP. These are lightweight, deterministic post-processing changes that should increase IoU consistency without changing your core approach or adding any training. The submission writing and RLE encoding remain unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 0.0438) has done: 'Your current score is far below the target, so we should cautiously increase it without changing the overall “test-image heuristic → optional CRF → RLE” pipeline. The biggest low-risk gain here is to make the heuristic closer to typical salt-mask morphology: suppress border noise (salt rarely touches borders), smooth before thresholding, and do stronger but still simple connected-component cleanup (keep only the largest component and do a small closing) to reduce false positives that heavily hurt mean AP across IoU thresholds. I also add a single global fallback threshold (computed from a small subset of test images) used only when Otsu becomes unstable, keeping the same core logic but improving robustness. Submission writing and RLE encoding remain unchanged and a valid `submission.csv` is always produced.'
- What this solution (achieved 0.0763) has done: 'Your score is far below the target, so we should improve the mask quality while keeping the same “test-image heuristic → optional CRF → RLE → submission.csv” pipeline. The most impactful minimal change here is to tune the post-processing for this metric by (1) selecting a binarization threshold that maximizes mean IoU on a small validation split from the provided training masks (no model/training loop added), and (2) applying a very light test-time augmentation (horizontal flip ensembling) to stabilize predictions. I also fix a subtle but important bug: your `rle_decode` reshapes in C-order while encoding uses Fortran order; making decode consistent improves any train-mask validation we add. All changes are deterministic, fast (<600s), and keep the core approach intact.'
- What this solution (achieved 0.0772) has done: 'Your current score (0.0763) is far below the target (0.8167), so we should improve mask quality without changing the overall heuristic→(optional CRF)→RLE pipeline. The biggest low-risk issue is that the current flip “ensemble” uses a union (`>=1`), which strongly increases false positives and hurts mean AP; switching to a strict consensus (`==2`) typically improves precision and IoU stability. Next, calibrating the binarization threshold using plain mean IoU is misaligned with the competition’s mean AP over IoU thresholds, so we keep the same calibration structure but optimize the threshold against an AP-style metric computed on a small validation set. Finally, we keep CRF as-is (no-op fallback) and ensure every id still gets a valid RLE string in `submission.csv`.'

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

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample submission at {SAMPLE_SUB_PATH}"
assert os.path.exists(TEST_IMG_DIR), f"Missing test image dir at {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train image dir at {TRAIN_IMG_DIR}"
assert os.path.exists(TRAIN_MASK_DIR), f"Missing train mask dir at {TRAIN_MASK_DIR}"




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


def heuristic_mask_from_image(orig_img, override_thr=None):
    """
    Heuristic (core logic preserved):
    - Normalize and lightly smooth to stabilize thresholding.
    - Otsu threshold per image; fallback to a global threshold if Otsu is unstable.
    - Optionally override threshold (used for validation-calibrated global adjustment).
    - Choose foreground side using a center prior.
    - Suppress a thin border region to reduce noisy FP.
    - Cleanup: remove small objects/holes, close small gaps, and keep only the largest CC.
    """
    img = orig_img
    if img.ndim == 3:
        img = img[:, :, 0]
    img = img.astype(np.float32)

    mn, mx = float(np.min(img)), float(np.max(img))
    denom = (mx - mn) if (mx - mn) > 1e-6 else 1.0
    x = (img - mn) / denom

    x_s = gaussian(x, sigma=0.8, preserve_range=True)

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
    mask = m_high if score_high >= score_low else m_low

    b = 6
    mask[:b, :] = False
    mask[-b:, :] = False
    mask[:, :b] = False
    mask[:, -b:] = False

    mask = remove_small_objects(mask.astype(bool), min_size=60)
    mask = remove_small_holes(mask, area_threshold=60)
    mask = binary_closing(mask, footprint=disk(2))

    lab = label(mask)
    if lab.max() > 0:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        largest = int(np.argmax(counts))
        mask = lab == largest

    return mask.astype(np.uint8)


DO_PLOT = False

if DO_PLOT:
    sample_ids = df["id"].head(6).astype(str).tolist()
    plt.figure(figsize=(18, 6))
    for j, img_id in enumerate(sample_ids):
        img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")
        orig_img = imread(img_path)
        m = heuristic_mask_from_image(orig_img)
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
def _iou(y_true, y_pred):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    inter = int(np.sum(y_true & y_pred))
    union = int(np.sum(y_true | y_pred))
    if union == 0:
        return 1.0  # both empty
    return inter / union


def _ap_like_single_mask(y_true, y_pred, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.05)
    iou = _iou(y_true, y_pred)
    if (y_true.sum() == 0) and (y_pred.sum() == 0):
        return 1.0
    if (y_true.sum() == 0) and (y_pred.sum() > 0):
        return 0.0
    if (y_true.sum() > 0) and (y_pred.sum() == 0):
        return 0.0
    return float(np.mean(iou > thresholds))


train_ids = sorted(
    [f.split(".")[0] for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(".png")]
)
rs = np.random.RandomState(100)
rs.shuffle(train_ids)

n_val = 250  # small bump for stability; still fast and keeps same calibration approach
val_ids = train_ids[:n_val]

thr_grid = np.linspace(0.30, 0.70, 17)

best_thr = None
best_score = -1.0

for thr in thr_grid:
    scores = []
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

        pred = heuristic_mask_from_image(img, override_thr=thr)
        scores.append(_ap_like_single_mask(gt, pred))
    if len(scores) == 0:
        continue
    s = float(np.mean(scores))
    if s > best_score:
        best_score = s
        best_thr = float(thr)

_CALIBRATED_THR = best_thr
print(
    f"Calibrated threshold override (AP-like): {_CALIBRATED_THR} (val={best_score:.4f})"
)




## === cell 8
for i in tqdm(range(df.shape[0])):
    img_id = str(df.loc[i, "id"])
    img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")

    if not os.path.exists(img_path):
        df.loc[i, "rle_mask"] = ""
        continue

    orig_img = imread(img_path)

    m1 = heuristic_mask_from_image(orig_img, override_thr=_CALIBRATED_THR)
    orig_flip = np.fliplr(orig_img)
    m2 = heuristic_mask_from_image(orig_flip, override_thr=_CALIBRATED_THR)
    m2 = np.fliplr(m2)

    decoded_mask = ((m1 + m2) == 2).astype(np.uint8)  # consensus

    crf_output = crf(orig_img, decoded_mask)

    df.loc[i, "rle_mask"] = rle_encode(crf_output)




## === cell 9
df["id"] = df["id"].astype(str)
df["rle_mask"] = df["rle_mask"].fillna("").astype(str)

out_path = "/kaggle/working/submission.csv"
df.to_csv(out_path, index=False)

print(f"Wrote submission to: {out_path}")
print(df.head())
