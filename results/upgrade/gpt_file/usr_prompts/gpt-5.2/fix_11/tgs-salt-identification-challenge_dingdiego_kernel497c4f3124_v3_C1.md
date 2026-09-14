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

try:
    import imageio.v2 as imageio

    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

from skimage.io import imread as sk_imread
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
    binary_closing,
    disk,
)
from skimage.measure import label

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

INPUT_DIR = "../input/tgs-salt-identification-challenge"
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test", "images")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(INPUT_DIR, "train", "masks")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
DEPTHS_PATH = os.path.join(INPUT_DIR, "depths.csv")
TRAIN_CSV_PATH = os.path.join(INPUT_DIR, "train.csv")

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample submission at {SAMPLE_SUB_PATH}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test image directory at {TEST_IMG_DIR}"
assert os.path.exists(DEPTHS_PATH), f"Missing depths.csv at {DEPTHS_PATH}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv at {TRAIN_CSV_PATH}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train image directory at {TRAIN_IMG_DIR}"
assert os.path.isdir(
    TRAIN_MASK_DIR
), f"Missing train mask directory at {TRAIN_MASK_DIR}"

np.random.seed(0)


def _read_png(path):
    if _HAS_IMAGEIO:
        arr = imageio.imread(path)
    else:
        arr = sk_imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr




## === cell 1
def rle_decode(rle_mask):
    """
    rle_mask: run-length as string formatted (start length)
    Returns numpy array, 1 - mask, 0 - background
    """
    if rle_mask is None:
        return np.zeros((101, 101), dtype=np.uint8)
    if not isinstance(rle_mask, str) or rle_mask.strip() == "":
        return np.zeros((101, 101), dtype=np.uint8)

    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101)


def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.flatten(order="F")  # column-major as required by competition
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return ""
    return " ".join(str(x) for x in runs)




## === cell 2
def iou_score(y_true, y_pred):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0  # both empty
    return inter / union


def _prep_instances(mask_uint8):
    """Return (lab, ids(>0), areas for those ids)."""
    m = (mask_uint8 > 0).astype(np.uint8, copy=False)
    lab = label(m, connectivity=1)
    ids = np.unique(lab)
    ids = ids[ids != 0]
    if ids.size == 0:
        return lab, ids, np.zeros((0,), dtype=np.int32)
    bc = np.bincount(lab.ravel())
    areas = bc[ids].astype(np.int32, copy=False)
    return lab, ids, areas


_THRESHOLDS = np.arange(0.5, 1.0, 0.05).astype(np.float32)


def _tp_from_iou_threshold(iou_mat, thr):
    matches = iou_mat > thr
    if not np.any(matches):
        return 0
    nt, npred = matches.shape
    if not _HAS_SCIPY:
        cand = np.argwhere(matches)
        cand_ious = iou_mat[cand[:, 0], cand[:, 1]]
        order = np.argsort(-cand_ious, kind="mergesort")
        cand = cand[order]
        matched_true = np.zeros((nt,), dtype=bool)
        matched_pred = np.zeros((npred,), dtype=bool)
        tp = 0
        for ti, pj in cand:
            if matched_true[ti] or matched_pred[pj]:
                continue
            matched_true[ti] = True
            matched_pred[pj] = True
            tp += 1
        return int(tp)

    rows, cols = np.nonzero(matches)
    g = csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(nt, npred))
    m = maximum_bipartite_matching(g, perm_type="row")
    return int(np.count_nonzero(m != -1))


def mean_ap_iou_kaggle_prepped(
    true_lab, true_ids, true_areas, pred_lab, pred_ids, pred_areas
):
    nt = int(true_ids.size)
    npred = int(pred_ids.size)

    if nt == 0 and npred == 0:
        return 1.0
    if nt == 0 and npred > 0:
        return 0.0
    if nt > 0 and npred == 0:
        return 0.0

    t = true_lab.ravel()
    p = pred_lab.ravel()
    p_max = int(p.max())
    pair = t.astype(np.int64, copy=False) * (p_max + 1) + p.astype(np.int64, copy=False)
    inter_flat = np.bincount(pair, minlength=(int(true_lab.max()) + 1) * (p_max + 1))

    idx = (
        true_ids.astype(np.int64, copy=False)[:, None] * (p_max + 1)
        + pred_ids.astype(np.int64, copy=False)[None, :]
    )
    inter = inter_flat[idx].astype(np.float32, copy=False)

    unions = (
        true_areas.astype(np.float32, copy=False)[:, None]
        + pred_areas.astype(np.float32, copy=False)[None, :]
        - inter
    )
    iou_mat = np.divide(
        inter, unions, out=np.zeros_like(inter, dtype=np.float32), where=unions > 0
    )

    precisions = np.empty((_THRESHOLDS.size,), dtype=np.float32)
    for k, tthr in enumerate(_THRESHOLDS):
        tp = _tp_from_iou_threshold(iou_mat, float(tthr))
        fp = npred - tp
        fn = nt - tp
        denom = tp + fp + fn
        precisions[k] = (tp / denom) if denom > 0 else 0.0

    return float(precisions.mean())


def mean_ap_iou_kaggle(y_true, y_pred):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    true_lab, true_ids, true_areas = _prep_instances(y_true)
    pred_lab, pred_ids, pred_areas = _prep_instances(y_pred)
    return mean_ap_iou_kaggle_prepped(
        true_lab, true_ids, true_areas, pred_lab, pred_ids, pred_areas
    )


def crf(original_image, mask_img, close_radius=1, min_obj=20, min_hole=20):
    """
    Kept same overall "refine mask" intent.
    """
    m = mask_img > 0
    m = binary_closing(m, footprint=disk(int(close_radius)))
    m = remove_small_objects(m, min_size=int(min_obj))
    m = remove_small_holes(m, area_threshold=int(min_hole))
    return m.astype(np.uint8)


def baseline_mask_from_image(img_2d, thr, polarity="gt"):
    img_2d = img_2d.astype(np.uint8, copy=False)
    if polarity == "lt":
        return (img_2d < thr).astype(np.uint8)
    return (img_2d > thr).astype(np.uint8)


def apply_post_threshold(mask_uint8, post_thr=0.5):
    return ((mask_uint8.astype(np.float32, copy=False)) >= float(post_thr)).astype(
        np.uint8
    )




## === cell 3
sub_df = pd.read_csv(SAMPLE_SUB_PATH)
if "id" not in sub_df.columns or "rle_mask" not in sub_df.columns:
    raise ValueError(f"Unexpected sample submission columns: {sub_df.columns.tolist()}")

depths_df = pd.read_csv(DEPTHS_PATH)
if "id" not in depths_df.columns or "z" not in depths_df.columns:
    raise ValueError(f"Unexpected depths columns: {depths_df.columns.tolist()}")

train_df = pd.read_csv(TRAIN_CSV_PATH)
if "id" not in train_df.columns or "rle_mask" not in train_df.columns:
    raise ValueError(f"Unexpected train.csv columns: {train_df.columns.tolist()}")

train_df = train_df.merge(depths_df, on="id", how="left")
sub_df = sub_df.merge(depths_df, on="id", how="left")

assert train_df["z"].isna().sum() == 0, "Missing depths for some train ids"
if sub_df["z"].isna().sum() > 0:
    fill_z = float(train_df["z"].median())
    sub_df["z"] = sub_df["z"].fillna(fill_z)




## === cell 4
n_train_total = len(train_df)
n_calib = min(1200, n_train_total)
calib_df = train_df.sample(n=n_calib, random_state=0).reset_index(drop=True)


def load_train_pair(img_id):
    img_path = os.path.join(TRAIN_IMG_DIR, f"{img_id}.png")
    msk_path = os.path.join(TRAIN_MASK_DIR, f"{img_id}.png")
    img = _read_png(img_path).astype(np.uint8, copy=False)
    msk = _read_png(msk_path)
    msk = (msk > 127).astype(np.uint8)
    return img, msk


calib_z = calib_df["z"].values.astype(np.float32, copy=False)
n_cal = len(calib_df)
calib_imgs = np.empty((n_cal, 101, 101), dtype=np.uint8)
calib_msks = np.empty((n_cal, 101, 101), dtype=np.uint8)

for j, img_id in enumerate(
    tqdm(calib_df["id"].values, total=n_cal, desc="Loading calibration set")
):
    img, msk = load_train_pair(img_id)
    calib_imgs[j] = img
    calib_msks[j] = msk

bin_edges = np.quantile(calib_z, [0.0, 0.33, 0.66, 1.0]).astype(np.float32)
bin_edges[0] -= 1e-6
bin_edges[-1] += 1e-6


def depth_bin_index(z):
    if (
        z is None
        or (isinstance(z, float) and np.isnan(z))
        or (isinstance(z, np.floating) and np.isnan(z))
    ):
        z = float(np.median(calib_z))
    return int(np.clip(np.searchsorted(bin_edges, float(z), side="right") - 1, 0, 2))


calib_bins = np.clip(
    np.searchsorted(bin_edges, calib_z.astype(np.float64), side="right") - 1, 0, 2
).astype(np.int64)

rng = np.random.RandomState(0)
perm = rng.permutation(n_cal)
n_val = max(200, int(0.25 * n_cal))
val_idx = perm[:n_val]
fit_idx = perm[n_val:]


calib_true_prepped = [None] * n_cal
for i in range(n_cal):
    calib_true_prepped[i] = _prep_instances(calib_msks[i])




## === cell 5
thr_grid = list(range(80, 201, 10))
close_grid = [1, 2]
min_obj_grid = [10, 20, 40]
min_hole_grid = [10, 20, 40]
polarity_grid = ["gt", "lt"]
post_thr_grid = [0.35, 0.45, 0.5, 0.55, 0.65]

best_params = None
best_score = -1.0

bin_indices_fit_full = [
    fit_idx[np.where(calib_bins[fit_idx] == b)[0]] for b in range(3)
]

max_thr_fit_per_bin = 250
bin_indices_fit_thrsel = []
for b in range(3):
    idxs = bin_indices_fit_full[b]
    if len(idxs) > max_thr_fit_per_bin:
        idxs = idxs[:max_thr_fit_per_bin]
    bin_indices_fit_thrsel.append(idxs)


def _precompute_baselines_for_indices(indices, polarity):
    base = {}
    for i in indices:
        img = calib_imgs[i]
        if polarity == "lt":
            for thr in thr_grid:
                base[(int(i), int(thr))] = (img < thr).astype(np.uint8, copy=False)
        else:
            for thr in thr_grid:
                base[(int(i), int(thr))] = (img > thr).astype(np.uint8, copy=False)
    return base


def _build_predprep_cache(
    indices, close_radius, min_obj, min_hole, post_thr, baseline_cache
):
    cache = {}
    for i in indices:
        img = calib_imgs[i]
        for thr in thr_grid:
            base = baseline_cache[(int(i), int(thr))]
            refined = crf(
                img,
                base,
                close_radius=close_radius,
                min_obj=min_obj,
                min_hole=min_hole,
            )
            refined = apply_post_threshold(refined, post_thr=post_thr)
            cache[(int(i), int(thr))] = _prep_instances(refined)
    return cache


def eval_params_on_from_cache(idxs, thr_by_bin, predprep_cache):
    scores = np.empty((len(idxs),), dtype=np.float32)
    for k, i in enumerate(idxs):
        thr = int(thr_by_bin[int(calib_bins[i])])
        pred_lab, pred_ids, pred_areas = predprep_cache[(int(i), thr)]
        true_lab, true_ids, true_areas = calib_true_prepped[i]
        scores[k] = mean_ap_iou_kaggle_prepped(
            true_lab, true_ids, true_areas, pred_lab, pred_ids, pred_areas
        )
    return float(scores.mean())


base_close, base_min_obj, base_min_hole = 1, 20, 20
polarity_probe_thr = 128
probe_scores = {}

for pol in polarity_grid:
    s = np.empty((len(fit_idx),), dtype=np.float32)
    for kk, i in enumerate(fit_idx):
        base = baseline_mask_from_image(
            calib_imgs[i], thr=polarity_probe_thr, polarity=pol
        )
        refined = crf(
            calib_imgs[i],
            base,
            close_radius=base_close,
            min_obj=base_min_obj,
            min_hole=base_min_hole,
        )
        refined = apply_post_threshold(refined, post_thr=0.5)
        pred_lab, pred_ids, pred_areas = _prep_instances(refined)
        true_lab, true_ids, true_areas = calib_true_prepped[i]
        s[kk] = mean_ap_iou_kaggle_prepped(
            true_lab, true_ids, true_areas, pred_lab, pred_ids, pred_areas
        )
    probe_scores[pol] = float(s.mean())

chosen_polarity = max(probe_scores, key=probe_scores.get)

cache_indices = np.unique(
    np.concatenate(
        [np.concatenate(bin_indices_fit_thrsel), val_idx.astype(np.int64, copy=False)]
    )
).astype(np.int64, copy=False)

baseline_cache = _precompute_baselines_for_indices(cache_indices, chosen_polarity)


for close_radius in close_grid:
    for min_obj in min_obj_grid:
        for min_hole in min_hole_grid:
            for post_thr in post_thr_grid:
                predprep_cache = _build_predprep_cache(
                    cache_indices,
                    close_radius,
                    min_obj,
                    min_hole,
                    post_thr,
                    baseline_cache,
                )

                thr_by_bin = [140, 140, 140]

                for b in range(3):
                    idxs_thr = bin_indices_fit_thrsel[b]
                    if len(idxs_thr) == 0:
                        continue

                    best_thr_b = thr_by_bin[b]
                    best_b_score = -1.0

                    true_prepped = calib_true_prepped
                    idxs_list = [int(x) for x in idxs_thr]

                    for thr in thr_grid:
                        acc = 0.0
                        for i in idxs_list:
                            pred_lab, pred_ids, pred_areas = predprep_cache[
                                (i, int(thr))
                            ]
                            true_lab, true_ids, true_areas = true_prepped[i]
                            acc += mean_ap_iou_kaggle_prepped(
                                true_lab,
                                true_ids,
                                true_areas,
                                pred_lab,
                                pred_ids,
                                pred_areas,
                            )
                        m = acc / float(len(idxs_list))
                        if m > best_b_score:
                            best_b_score = m
                            best_thr_b = thr

                    thr_by_bin[b] = best_thr_b

                val_sc = eval_params_on_from_cache(
                    val_idx, thr_by_bin=thr_by_bin, predprep_cache=predprep_cache
                )
                if val_sc > best_score:
                    best_score = val_sc
                    best_params = {
                        "thr_by_bin": thr_by_bin.copy(),
                        "close_radius": close_radius,
                        "min_obj": min_obj,
                        "min_hole": min_hole,
                        "polarity": chosen_polarity,
                        "post_thr": float(post_thr),
                    }

print("Polarity probe scores (fit):", probe_scores, "=> chosen:", chosen_polarity)
print("Validation best mean mAP:", best_score)
print("Chosen params:", best_params)




## === cell 6
pred_rles = []
thr_by_bin = best_params["thr_by_bin"]
close_radius = best_params["close_radius"]
min_obj = best_params["min_obj"]
min_hole = best_params["min_hole"]
polarity = best_params["polarity"]
post_thr = best_params["post_thr"]

test_ids = sub_df["id"].values
test_z = sub_df["z"].values.astype(np.float32, copy=False)

for j in tqdm(range(len(test_ids)), total=len(test_ids), desc="Predicting"):
    img_id = test_ids[j]
    z = float(test_z[j])

    img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")
    if not os.path.exists(img_path):
        pred_rles.append("")
        continue

    img = _read_png(img_path).astype(np.uint8, copy=False)

    b = depth_bin_index(z)
    thr = int(thr_by_bin[b])
    base_mask = baseline_mask_from_image(img, thr=thr, polarity=polarity)
    refined = crf(
        img, base_mask, close_radius=close_radius, min_obj=min_obj, min_hole=min_hole
    )
    refined = apply_post_threshold(refined, post_thr=post_thr)
    pred_rles.append(rle_encode(refined))

sub_out = sub_df[["id"]].copy()
sub_out["rle_mask"] = pred_rles




## === cell 7
out_path = "submission.csv"
sub_out.to_csv(out_path, index=False)

print(sub_out.head())
print(f"Wrote submission to: {out_path} (rows={len(sub_out)})")
assert out_path.endswith(".csv") and os.path.exists(out_path)
assert list(sub_out.columns) == ["id", "rle_mask"]
assert len(sub_out) == len(pd.read_csv(SAMPLE_SUB_PATH))
