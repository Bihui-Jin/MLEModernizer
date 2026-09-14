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

0.82226

# 6. Current score

0.296

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0631) has done: 'I remove the hard dependency on `pydensecrf` (not installed in your Kaggle environment) and replace the CRF refinement with a small, deterministic post-processing step using only `scikit-image`, so the pipeline runs end-to-end. I also fix the input paths: your code references a non-existent `../input/candsub/blend_ez.csv`, so I instead start from the provided `sample_submission.csv` and keep any existing masks (none in the sample) while generating valid RLE strings. Finally, I ensure the submission has exactly the required columns (`id`, `rle_mask`) and is written with a `.csv` suffix. This should yield a valid (though likely low-scoring) submission rather than failing at import/path errors.'
- What this solution (achieved 0.1259) has done: 'Your current score is far below the target (0.0631 vs 0.82226), so we should improve performance while keeping the same “unsupervised threshold + morphology” core logic. The biggest win with minimal semantic change is to (1) pick the threshold direction (salt is typically darker, but not always) and (2) tune that direction and the binarization threshold using a small internal validation split from the provided training masks, then apply the best setting to test. This preserves your approach (no learning model, still threshold+morphology), but aligns the only free knob (thresholding) to the competition metric. I also add a lightweight IoU-sweep metric implementation to select the best threshold/direction without changing the submission format or paths.'
- What this solution (achieved 0.248) has done: 'We keep your exact “threshold + morphology” approach, but fix a key evaluation mismatch: the official metric is instance-based and treats empty masks specially, so the common improvement is to choose the binarization threshold based on validation mAP (not just IoU) and then apply a small “empty-mask gate” to avoid false positives on empty images. To minimize changes, I (1) expand the threshold grid slightly (still just threshold tuning), (2) tune the morphology size parameters very lightly (same operations, just small integers), and (3) add a single scalar cutoff on predicted mask area (validated) to decide when to output an empty RLE, which typically boosts mAP substantially for this competition. All paths and submission formatting remain unchanged, and it still writes `submission.csv` end-to-end.'
- What this solution (achieved 0.296) has done: 'The timeout is dominated by the validation grid-search in cell 4: it repeatedly runs expensive morphology operations per image inside deeply nested Python loops, leading to millions of small-object/hole computations. I keep the exact same search space, scoring, and morphology logic, but make it much faster by (1) precomputing the opening+closing result once per (thr/invert/pad,selem) and reusing it across hole/object grids, and (2) caching the results of `remove_small_holes` for each (ha) so `remove_small_objects` runs on an already-computed boolean mask. I also replace repeated `skimage.io.imread` calls in Python loops with a thread pool for I/O (deterministic ordering preserved) and avoid repeated dtype conversions/reshapes inside tight loops. The test-time prediction loop is kept identical logically, but uses a small thread pool to overlap PNG decoding and CPU work without changing outputs.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_opening,
    binary_closing,
    disk,
)

np.random.seed(42)

from concurrent.futures import ThreadPoolExecutor




## === cell 1
def rle_decode(mask_rle, shape=(101, 101)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if (
        mask_rle is None
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
        or str(mask_rle).strip() == ""
    ):
        return np.zeros(shape, dtype=np.uint8)

    s = str(mask_rle).split()
    starts = np.asarray(s[0:][::2], dtype=np.int64)
    lengths = np.asarray(s[1:][::2], dtype=np.int64)
    starts -= 1
    ends = starts + lengths

    n = shape[0] * shape[1]
    diff = np.zeros(n + 1, dtype=np.int32)
    starts = np.clip(starts, 0, n)
    ends = np.clip(ends, 0, n)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    img = (np.cumsum(diff[:-1]) > 0).astype(np.uint8)

    return img.reshape(shape, order="F")  # column-major per Kaggle TGS convention


def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    if im is None:
        return ""
    im = (im > 0).astype(np.uint8)

    pixels = im.reshape(-1, order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 2
BASE = "/kaggle/input/tgs-salt-identification-challenge"
TEST_IMG_DIR = os.path.join(BASE, "test", "images")
SAMPLE_SUB_PATH = os.path.join(BASE, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(BASE, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train", "images")
TRAIN_MASK_DIR = os.path.join(BASE, "train", "masks")

if not os.path.exists(SAMPLE_SUB_PATH):
    BASE = "/kaggle/data/tgs-salt-identification-challenge"
    TEST_IMG_DIR = os.path.join(BASE, "test", "images")
    SAMPLE_SUB_PATH = os.path.join(BASE, "sample_submission.csv")
    TRAIN_CSV_PATH = os.path.join(BASE, "train.csv")
    TRAIN_IMG_DIR = os.path.join(BASE, "train", "images")
    TRAIN_MASK_DIR = os.path.join(BASE, "train", "masks")

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"sample_submission.csv not found at {SAMPLE_SUB_PATH}"
assert os.path.isdir(TEST_IMG_DIR), f"Test image dir not found at {TEST_IMG_DIR}"

sub = pd.read_csv(SAMPLE_SUB_PATH)
sub = sub[["id", "rle_mask"]].copy()




## === cell 3
_DISK_CACHE = {}


def _get_disk(radius: int):
    radius = int(radius)
    se = _DISK_CACHE.get(radius)
    if se is None:
        se = disk(radius)
        _DISK_CACHE[radius] = se
    return se


def refine_mask_from_image(
    orig_img,
    thr=None,
    invert=False,
    selem_radius=1,
    hole_area=20,
    obj_min_size=20,
    pad=0,
):
    """
    orig_img: 2D array (101x101) grayscale
    thr: float in [0,1] or None (use Otsu)
    invert: if True, uses img > thr instead of img < thr
    selem_radius/hole_area/obj_min_size: same morphology ops, lightly tunable ints
    pad: small symmetric padding (then unpad) to reduce edge artifacts during morphology
    returns: binary mask (101x101) uint8
    """
    img = orig_img.astype(np.float32)

    mn, mx = float(np.min(img)), float(np.max(img))
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    pad = int(pad)
    if pad > 0:
        img = np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")

    if thr is None:
        try:
            thr = float(threshold_otsu(img))
        except Exception:
            thr = 0.5
    else:
        thr = float(thr)

    if invert:
        mask = img > thr
    else:
        mask = img < thr

    selem = _get_disk(int(selem_radius))
    mask = binary_opening(mask, selem)
    mask = binary_closing(mask, selem)
    mask = remove_small_holes(mask, area_threshold=int(hole_area))
    mask = remove_small_objects(mask, min_size=int(obj_min_size))

    if pad > 0:
        mask = mask[pad:-pad, pad:-pad]

    return mask.astype(np.uint8)


def iou_np(y_true, y_pred):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0  # both empty => perfect IoU
    return inter / union


def kaggle_map_iou(y_true, y_pred, thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    For TGS salt, masks are effectively single-instance; metric reduces to IoU hit/miss
    plus explicit empty-mask handling.
    """
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    true_empty = y_true.sum() == 0
    pred_empty = y_pred.sum() == 0
    if true_empty and pred_empty:
        return 1.0
    if true_empty and (not pred_empty):
        return 0.0
    if (not true_empty) and pred_empty:
        return 0.0

    iou = iou_np(y_true, y_pred)
    precisions = [(1.0 if iou > t else 0.0) for t in thresholds]
    return float(np.mean(precisions))




## === cell 4
best_params = {
    "thr": None,
    "invert": False,
    "selem_radius": 1,
    "hole_area": 20,
    "obj_min_size": 20,
    "empty_area_cut": 0,  # if predicted area <= cut, output empty
    "pad": 0,
}
best_score = -1.0

if (
    os.path.exists(TRAIN_CSV_PATH)
    and os.path.isdir(TRAIN_IMG_DIR)
    and os.path.isdir(TRAIN_MASK_DIR)
):
    train_df = pd.read_csv(TRAIN_CSV_PATH)[["id", "rle_mask"]].copy()

    ids = train_df["id"].values
    n_val = min(600, len(ids))

    is_empty = train_df["rle_mask"].isna() | (
        train_df["rle_mask"].astype(str).str.strip() == ""
    )
    empty_ids = train_df.loc[is_empty, "id"].values
    nonempty_ids = train_df.loc[~is_empty, "id"].values

    rng = np.random.RandomState(42)
    n_empty = min(
        len(empty_ids), int(round(n_val * (len(empty_ids) / max(1, len(ids)))))
    )
    n_nonempty = n_val - n_empty
    val_ids = np.concatenate(
        [
            (
                rng.choice(empty_ids, size=n_empty, replace=False)
                if n_empty > 0
                else np.array([], dtype=ids.dtype)
            ),
            (
                rng.choice(
                    nonempty_ids, size=min(n_nonempty, len(nonempty_ids)), replace=False
                )
                if n_nonempty > 0
                else np.array([], dtype=ids.dtype)
            ),
        ]
    )
    rng.shuffle(val_ids)

    def _load_train_pair(img_id):
        img_path = os.path.join(TRAIN_IMG_DIR, f"{img_id}.png")
        msk_path = os.path.join(TRAIN_MASK_DIR, f"{img_id}.png")
        if not (os.path.exists(img_path) and os.path.exists(msk_path)):
            return None
        img = imread(img_path)
        if img.ndim == 3:
            img = img.mean(axis=2)
        msk = imread(msk_path)
        if msk.ndim == 3:
            msk = msk[..., 0]
        msk = (msk > 0).astype(np.uint8)
        return img.astype(np.float32, copy=False), msk

    val_imgs = []
    val_masks = []
    max_workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in tqdm(
            ex.map(_load_train_pair, val_ids), total=len(val_ids), desc="Loading val"
        ):
            if res is None:
                continue
            img, msk = res
            val_imgs.append(img)
            val_masks.append(msk)

    if len(val_imgs) > 0:
        val_imgs = np.stack(val_imgs, axis=0)  # (N,H,W)
        val_masks = np.stack(val_masks, axis=0).astype(np.uint8, copy=False)
    else:
        val_imgs = np.empty((0, 101, 101), dtype=np.float32)
        val_masks = np.empty((0, 101, 101), dtype=np.uint8)

    thr_grid = [None, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68]
    inv_grid = [False, True]

    selem_grid = [1, 2]
    hole_grid = [10, 20, 40]
    obj_grid = [10, 20, 40]

    empty_area_grid = [0, 5, 10, 20, 40, 80, 120]
    pad_grid = [0, 2, 4]

    thresholds = np.arange(0.5, 1.0, 0.05).astype(np.float32)

    N = val_imgs.shape[0]
    true_area = val_masks.reshape(N, -1).sum(axis=1).astype(np.int32, copy=False)
    true_empty = true_area == 0

    if N:
        mn = val_imgs.reshape(N, -1).min(axis=1).astype(np.float32, copy=False)
        mx = val_imgs.reshape(N, -1).max(axis=1).astype(np.float32, copy=False)
        denom = mx - mn
        val_norm = np.zeros_like(val_imgs, dtype=np.float32)
        good = denom > 0
        if np.any(good):
            val_norm[good] = (val_imgs[good] - mn[good, None, None]) / denom[
                good, None, None
            ]

        val_norm_pad = {}
        for pad in pad_grid:
            pad = int(pad)
            if pad > 0:
                val_norm_pad[pad] = np.pad(
                    val_norm, ((0, 0), (pad, pad), (pad, pad)), mode="reflect"
                )
            else:
                val_norm_pad[pad] = val_norm

        otsu_thr_pad = {}
        for pad in pad_grid:
            imgs_p = np.ascontiguousarray(val_norm_pad[pad])
            thrs = np.empty(N, dtype=np.float32)
            for i in range(N):
                try:
                    thrs[i] = float(threshold_otsu(imgs_p[i]))
                except Exception:
                    thrs[i] = 0.5
            otsu_thr_pad[pad] = thrs

    for _r in selem_grid:
        _get_disk(int(_r))

    for inv in inv_grid:
        for thr in thr_grid:
            for pad in pad_grid:
                pad = int(pad)
                if N == 0:
                    continue

                imgs_p = val_norm_pad[pad]
                if thr is None:
                    thr_vec = otsu_thr_pad[pad]
                else:
                    thr_vec = np.full(N, float(thr), dtype=np.float32)

                if inv:
                    init_mask = imgs_p > thr_vec[:, None, None]
                else:
                    init_mask = imgs_p < thr_vec[:, None, None]

                for se in selem_grid:
                    selem = _get_disk(int(se))

                    morph_base = [None] * N
                    for i in range(N):
                        m = init_mask[i]
                        m = binary_opening(m, selem)
                        m = binary_closing(m, selem)
                        morph_base[i] = m  # bool

                    holes_cache = {}
                    for ha in hole_grid:
                        ha = int(ha)
                        filled = [None] * N
                        for i in range(N):
                            filled[i] = remove_small_holes(
                                morph_base[i], area_threshold=ha
                            )
                        holes_cache[ha] = filled

                    for ha in hole_grid:
                        ha = int(ha)
                        filled = holes_cache[ha]

                        for ms in obj_grid:
                            ms = int(ms)

                            pred_masks = np.empty((N, 101, 101), dtype=np.uint8)
                            pred_area = np.empty(N, dtype=np.int32)

                            for i in range(N):
                                m = remove_small_objects(filled[i], min_size=ms)
                                if pad > 0:
                                    m = m[pad:-pad, pad:-pad]
                                m_u8 = m.astype(np.uint8, copy=False)
                                pred_masks[i] = m_u8
                                pred_area[i] = int(m_u8.sum())

                            pred_empty0 = pred_area == 0
                            both_empty = true_empty & pred_empty0
                            one_empty = true_empty ^ pred_empty0

                            iou = np.zeros(N, dtype=np.float32)
                            iou[both_empty] = 1.0
                            nz = ~(both_empty | one_empty)
                            if np.any(nz):
                                gt = (
                                    val_masks[nz]
                                    .astype(bool, copy=False)
                                    .reshape(-1, 10201)
                                )
                                pr = (
                                    pred_masks[nz]
                                    .astype(bool, copy=False)
                                    .reshape(-1, 10201)
                                )
                                inter = (
                                    np.logical_and(gt, pr)
                                    .sum(axis=1, dtype=np.int32)
                                    .astype(np.float32, copy=False)
                                )
                                union = (
                                    np.logical_or(gt, pr)
                                    .sum(axis=1, dtype=np.int32)
                                    .astype(np.float32, copy=False)
                                )
                                iou[nz] = np.where(
                                    union > 0, inter / union, 1.0
                                ).astype(np.float32, copy=False)

                            base_map = (
                                (iou[:, None] > thresholds[None, :])
                                .mean(axis=1)
                                .astype(np.float32, copy=False)
                            )

                            for area_cut in empty_area_grid:
                                area_cut = int(area_cut)
                                pred_empty_cut = pred_area <= area_cut

                                score_per_image = np.where(
                                    true_empty & pred_empty_cut,
                                    1.0,
                                    np.where(
                                        true_empty ^ pred_empty_cut, 0.0, base_map
                                    ),
                                )
                                mean_score = (
                                    float(score_per_image.mean()) if N else -1.0
                                )

                                if mean_score > best_score:
                                    best_score = mean_score
                                    best_params = {
                                        "thr": thr,
                                        "invert": inv,
                                        "selem_radius": se,
                                        "hole_area": ha,
                                        "obj_min_size": ms,
                                        "empty_area_cut": area_cut,
                                        "pad": pad,
                                    }

    print(
        "Chosen params from val:",
        f"thr={best_params['thr']}, invert={best_params['invert']}, "
        f"selem_radius={best_params['selem_radius']}, hole_area={best_params['hole_area']}, "
        f"obj_min_size={best_params['obj_min_size']}, pad={best_params['pad']}, "
        f"empty_area_cut={best_params['empty_area_cut']}, "
        f"val_mAP={best_score:.4f}",
    )
else:
    print("Train data not found; using original Otsu thresholding without tuning.")
    best_params["thr"] = None
    best_params["invert"] = False
    best_params["selem_radius"] = 1
    best_params["hole_area"] = 20
    best_params["obj_min_size"] = 20
    best_params["empty_area_cut"] = 0
    best_params["pad"] = 0




## === cell 5
pred_rles = []

thr = best_params["thr"]
invert = best_params["invert"]
selem_radius = best_params["selem_radius"]
hole_area = best_params["hole_area"]
obj_min_size = best_params["obj_min_size"]
pad = best_params["pad"]
empty_area_cut = int(best_params["empty_area_cut"])

test_ids = sub["id"].values
test_paths = [os.path.join(TEST_IMG_DIR, f"{img_id}.png") for img_id in test_ids]


def _predict_one(args):
    img_id, img_path = args
    if not os.path.exists(img_path):
        return ""
    orig = imread(img_path)
    if orig.ndim == 3:
        orig = orig.mean(axis=2)
    mask = refine_mask_from_image(
        orig,
        thr=thr,
        invert=invert,
        selem_radius=selem_radius,
        hole_area=hole_area,
        obj_min_size=obj_min_size,
        pad=pad,
    )
    if int(mask.sum()) <= empty_area_cut:
        return ""
    return rle_encode(mask)


max_workers = min(8, (os.cpu_count() or 2))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    pred_rles = list(
        tqdm(
            ex.map(_predict_one, zip(test_ids, test_paths)),
            total=len(sub),
            desc="Predict test",
        )
    )

sub["rle_mask"] = pred_rles

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote submission:", out_path)
print(sub.head())
