# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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


def heuristic_mask_from_image(
    orig_img,
    override_thr=None,
    img_id=None,
    min_size=30,
    hole_area=40,
    border=4,
    close_r=2,
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
def _iou(y_true, y_pred):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    inter = int(np.sum(y_true & y_pred))
    union = int(np.sum(y_true | y_pred))
    if union == 0:
        return 1.0  # both empty
    return inter / union


def _ap_instancewise(y_true, y_pred, thresholds=None):
    """
    Connected-components AP proxy (kept as-is).
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.05)

    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    t_lab = label(y_true)
    p_lab = label(y_pred)

    true_ids = list(range(1, int(t_lab.max()) + 1))
    pred_ids = list(range(1, int(p_lab.max()) + 1))

    if len(true_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(true_ids) == 0 and len(pred_ids) > 0:
        return 0.0
    if len(true_ids) > 0 and len(pred_ids) == 0:
        return 0.0

    true_masks = [(t_lab == i) for i in true_ids]
    pred_masks = [(p_lab == i) for i in pred_ids]

    ious = np.zeros((len(true_masks), len(pred_masks)), dtype=np.float32)
    for i, tm in enumerate(true_masks):
        tm_u8 = tm.astype(np.uint8)
        for j, pm in enumerate(pred_masks):
            pm_u8 = pm.astype(np.uint8)
            ious[i, j] = _iou(tm_u8, pm_u8)

    precisions = []
    for t in thresholds:
        matched_true = set()
        matched_pred = set()

        pairs = np.argwhere(ious > t)
        if pairs.size > 0:
            pair_ious = ious[pairs[:, 0], pairs[:, 1]]
            order = np.argsort(-pair_ious)
            for k in order:
                ti = int(pairs[k, 0])
                pj = int(pairs[k, 1])
                if (ti in matched_true) or (pj in matched_pred):
                    continue
                matched_true.add(ti)
                matched_pred.add(pj)

        tp = len(matched_true)
        fp = len(pred_ids) - len(matched_pred)
        fn = len(true_ids) - len(matched_true)
        denom = tp + fp + fn
        precisions.append(tp / denom if denom > 0 else 0.0)

    return float(np.mean(precisions))


train_ids = sorted(
    [f.split(".")[0] for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(".png")]
)
rs = np.random.RandomState(100)
rs.shuffle(train_ids)

n_val = 250
val_ids = train_ids[:n_val]

thr_grid = np.linspace(0.30, 0.70, 17)
min_size_grid = [20, 30, 40]
hole_area_grid = [20, 40, 60]

border_grid = [0, 2, 4]
close_r_grid = [0, 1, 2]

best_params = None
best_score = -1.0

for thr in thr_grid:
    for min_sz in min_size_grid:
        for hole_ar in hole_area_grid:
            for border in border_grid:
                for close_r in close_r_grid:
                    scores = []
                    for img_id in val_ids:
                        img_path = os.path.join(TRAIN_IMG_DIR, f"{img_id}.png")
                        m_path = os.path.join(TRAIN_MASK_DIR, f"{img_id}.png")
                        if (not os.path.exists(img_path)) or (
                            not os.path.exists(m_path)
                        ):
                            continue
                        img = imread(img_path)
                        gt = imread(m_path)
                        if gt.ndim == 3:
                            gt = gt[:, :, 0]
                        gt = (gt > 127).astype(np.uint8)

                        pred = heuristic_mask_from_image(
                            img,
                            override_thr=float(thr),
                            img_id=img_id,
                            min_size=int(min_sz),
                            hole_area=int(hole_ar),
                            border=int(border),
                            close_r=int(close_r),
                        )

                        scores.append(_ap_instancewise(gt.T, pred.T))

                    if len(scores) == 0:
                        continue
                    s = float(np.mean(scores))
                    if s > best_score:
                        best_score = s
                        best_params = (
                            float(thr),
                            int(min_sz),
                            int(hole_ar),
                            int(border),
                            int(close_r),
                        )

(
    _CALIBRATED_THR,
    _CALIBRATED_MIN_SIZE,
    _CALIBRATED_HOLE_AREA,
    _CALIBRATED_BORDER,
    _CALIBRATED_CLOSE_R,
) = best_params

print(
    "Calibrated params (instance-AP-like): "
    f"thr={_CALIBRATED_THR}, min_size={_CALIBRATED_MIN_SIZE}, hole_area={_CALIBRATED_HOLE_AREA}, "
    f"border={_CALIBRATED_BORDER}, close_r={_CALIBRATED_CLOSE_R} "
    f"(val={best_score:.4f})"
)




## === cell 8
_EMPTY_FRACTION_THRESHOLD = 0.0015  # ~15 pixels out of 10201

for i in tqdm(range(df.shape[0])):
    img_id = str(df.loc[i, "id"])
    img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")

    if not os.path.exists(img_path):
        df.loc[i, "rle_mask"] = ""
        continue

    orig_img = imread(img_path)

    m1 = heuristic_mask_from_image(
        orig_img,
        override_thr=_CALIBRATED_THR,
        img_id=img_id,
        min_size=_CALIBRATED_MIN_SIZE,
        hole_area=_CALIBRATED_HOLE_AREA,
        border=_CALIBRATED_BORDER,
        close_r=_CALIBRATED_CLOSE_R,
    )
    orig_flip = np.fliplr(orig_img)
    m2 = heuristic_mask_from_image(
        orig_flip,
        override_thr=_CALIBRATED_THR,
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

    df.loc[i, "rle_mask"] = rle_encode(crf_output.T)




## === cell 9
df["id"] = df["id"].astype(str)
df["rle_mask"] = df["rle_mask"].fillna("").astype(str)

out_path = "/kaggle/working/submission.csv"
df.to_csv(out_path, index=False)

print(f"Wrote submission to: {out_path}")
print(df.head())
