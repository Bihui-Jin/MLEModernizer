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

0.7695456512332313

# 6. Current score

0.1295

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.052) has done: 'I remove the unavailable `pydensecrf` dependency (it is not installed in this environment) and replace the CRF step with a small, deterministic morphological post-processing that keeps the same overall intent (mask refinement) and lets the notebook run end-to-end. I also remove IPython-only syntax (`%matplotlib inline`) and fix the missing imports/undefined variables by ensuring all needed modules are imported in the first cell and by making the pipeline self-contained (no external submission read from another Kaggle dataset). Finally, I read `sample_submission.csv` and test images from the provided `../input/tgs-salt-identification-challenge/` path and always write a valid `submission.csv` with columns `id,rle_mask`.'
- What this solution (achieved 0.1281) has done: 'Your current score is far below the target, and the main issue is that the solution is effectively a fixed intensity-threshold baseline that ignores the provided training masks and depth prior, so it cannot generalize to the metric. To move the score toward the target with minimal core-logic disruption, I keep your “baseline mask → refine → RLE” pipeline, but (1) learn the threshold from the training set instead of hardcoding 140, (2) use the depth (`z`) feature to pick a better threshold per depth bin (still the same thresholding logic), and (3) tune the existing morphological refinement sizes using training mAP as the objective. This keeps the same model semantics (a deterministic threshold + morphology “CRF-like” refinement) while making parameters data-driven, which should improve score substantially toward your target without introducing new dependencies. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.1281) has done: 'I fix the immediate runtime failure caused by missing depth values for some test IDs by safely filling missing `z` values rather than asserting, keeping the rest of the pipeline unchanged. I also make the depth-bin computation robust to NaNs so prediction can proceed deterministically. This is a correctness/stability fix (not a core-logic change) and should also improve score vs. emitting empty/incorrect masks for affected test rows. Finally, I keep the submission format identical and ensure `submission.csv` is written end-to-end.'
- What this solution (achieved 0.124) has done: 'Your current score is far below the target, so the most likely issue is a semantic mismatch between how masks are thresholded (salt is typically darker in this dataset) and how the pipeline currently predicts (img > thr). I keep your exact “per-depth-bin threshold → morphology refine → RLE” core logic, but (1) automatically choose the threshold *polarity* (either `img > thr` or `img < thr`) from the calibration set, and (2) calibrate the final probability-to-mask cutoff by selecting a single global post-threshold (applied after refinement) that maximizes the same mean-AP-IoU proxy you already use. These are minimal, metric-aligned parameter calibrations that should increase the score substantially without changing the overall approach or adding dependencies, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.1295) has done: 'Your current score is far below the target, and with this “threshold + morphology” core logic the biggest safe win is to align the calibration objective more closely to the competition metric by choosing the **final mask cutoff** using a validation-like split, instead of optimizing only the coarse IoU>t proxy on the same calibration set. I keep your exact pipeline (per-depth-bin threshold → morphology refine → global cutoff → RLE) but (1) implement a lightweight **true mean-AP IoU metric** via connected-component matching (single-object case reduces to IoU, multi-object works correctly), and (2) tune only the **post-threshold** (and keep your existing parameter grids) on a held-out fold to reduce overfitting. This is a minimal change that should move the score upward toward your target without changing model semantics or adding dependencies, and it still write a valid `submission.csv` end-to-end.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
    binary_closing,
    disk,
)
from skimage.measure import label

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


def mean_ap_iou_kaggle(y_true, y_pred):
    """
    Mean average precision over IoU thresholds for instance segmentation.
    For this dataset, masks are usually single-component, but this implementation
    also handles multiple components correctly (no overlaps).
    """
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    true_lab = label(y_true, connectivity=1)
    pred_lab = label(y_pred, connectivity=1)

    true_ids = np.unique(true_lab)
    pred_ids = np.unique(pred_lab)
    true_ids = true_ids[true_ids != 0]
    pred_ids = pred_ids[pred_ids != 0]

    if len(true_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(true_ids) == 0 and len(pred_ids) > 0:
        return 0.0
    if len(true_ids) > 0 and len(pred_ids) == 0:
        return 0.0

    iou_mat = np.zeros((len(true_ids), len(pred_ids)), dtype=np.float32)
    for i, t_id in enumerate(true_ids):
        t_mask = true_lab == t_id
        t_sum = t_mask.sum()
        for j, p_id in enumerate(pred_ids):
            p_mask = pred_lab == p_id
            inter = np.logical_and(t_mask, p_mask).sum()
            if inter == 0:
                continue
            union = t_sum + p_mask.sum() - inter
            if union > 0:
                iou_mat[i, j] = inter / union

    thresholds = np.arange(0.5, 1.0, 0.05)
    precisions = []
    for t in thresholds:
        matches = iou_mat > t

        matched_true = set()
        matched_pred = set()

        cand = np.argwhere(matches)
        if cand.size > 0:
            cand_ious = iou_mat[cand[:, 0], cand[:, 1]]
            order = np.argsort(-cand_ious, kind="mergesort")
            cand = cand[order]
            for ti, pj in cand:
                if ti in matched_true or pj in matched_pred:
                    continue
                matched_true.add(int(ti))
                matched_pred.add(int(pj))

        tp = len(matched_true)
        fp = len(pred_ids) - tp
        fn = len(true_ids) - tp
        denom = tp + fp + fn
        precisions.append(tp / denom if denom > 0 else 0.0)

    return float(np.mean(precisions))


def crf(original_image, mask_img, close_radius=1, min_obj=20, min_hole=20):
    """
    Kept same overall "refine mask" intent; we only allow small parameter tuning using training metric.
    """
    m = mask_img > 0
    m = binary_closing(m, footprint=disk(int(close_radius)))
    m = remove_small_objects(m, min_size=int(min_obj))
    m = remove_small_holes(m, area_threshold=int(min_hole))
    return m.astype(np.uint8)


def baseline_mask_from_image(img_2d, thr, polarity="gt"):
    img_2d = img_2d.astype(np.uint8)
    if polarity == "lt":
        return (img_2d < thr).astype(np.uint8)
    return (img_2d > thr).astype(np.uint8)


def apply_post_threshold(mask_uint8, post_thr=0.5):
    return ((mask_uint8.astype(np.float32)) >= float(post_thr)).astype(np.uint8)




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
n_calib = min(
    1200, n_train_total
)  # Change: slightly larger calibration pool, still fast, improves parameter stability.
calib_df = train_df.sample(n=n_calib, random_state=0).reset_index(drop=True)


def load_train_pair(img_id):
    img_path = os.path.join(TRAIN_IMG_DIR, f"{img_id}.png")
    msk_path = os.path.join(TRAIN_MASK_DIR, f"{img_id}.png")
    img = imread(img_path)
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.uint8)
    msk = imread(msk_path)
    if msk.ndim == 3:
        msk = msk[..., 0]
    msk = (msk > 127).astype(np.uint8)
    return img, msk


calib_imgs = []
calib_msks = []
calib_z = calib_df["z"].values.astype(np.float32)
for img_id in tqdm(
    calib_df["id"].values, total=len(calib_df), desc="Loading calibration set"
):
    img, msk = load_train_pair(img_id)
    calib_imgs.append(img)
    calib_msks.append(msk)
calib_imgs = np.stack(calib_imgs, axis=0)
calib_msks = np.stack(calib_msks, axis=0)

bin_edges = np.quantile(calib_z, [0.0, 0.33, 0.66, 1.0])
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


calib_bins = np.array([depth_bin_index(z) for z in calib_z], dtype=np.int64)

rng = np.random.RandomState(0)
perm = rng.permutation(len(calib_imgs))
n_val = max(200, int(0.25 * len(calib_imgs)))
val_idx = perm[:n_val]
fit_idx = perm[n_val:]



## === cell 5
thr_grid = list(
    range(80, 201, 10)
)  # Change: slightly wider but still small grid; helps find better bin thresholds.
close_grid = [1, 2]
min_obj_grid = [10, 20, 40]
min_hole_grid = [10, 20, 40]

polarity_grid = ["gt", "lt"]

post_thr_grid = [0.35, 0.45, 0.5, 0.55, 0.65]

best_params = None
best_score = -1.0

bin_indices_fit = [fit_idx[np.where(calib_bins[fit_idx] == b)[0]] for b in range(3)]


def eval_params_on(
    idxs, thr_by_bin, close_radius, min_obj, min_hole, polarity, post_thr
):
    scores = []
    for i in idxs:
        thr = thr_by_bin[calib_bins[i]]
        base = baseline_mask_from_image(calib_imgs[i], thr=thr, polarity=polarity)
        refined = crf(
            calib_imgs[i],
            base,
            close_radius=close_radius,
            min_obj=min_obj,
            min_hole=min_hole,
        )
        refined = apply_post_threshold(refined, post_thr=post_thr)
        scores.append(mean_ap_iou_kaggle(calib_msks[i], refined))
    return float(np.mean(scores))


base_close, base_min_obj, base_min_hole = 1, 20, 20

polarity_probe_thr = 128
probe_scores = {}
for pol in polarity_grid:
    s = []
    for i in fit_idx:
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
        s.append(mean_ap_iou_kaggle(calib_msks[i], refined))
    probe_scores[pol] = float(np.mean(s))
chosen_polarity = max(probe_scores, key=probe_scores.get)

thr_by_bin = [140, 140, 140]
for b in range(3):
    best_thr_b = thr_by_bin[b]
    best_b_score = -1.0
    idxs = bin_indices_fit[b]
    if len(idxs) == 0:
        continue
    for thr in thr_grid:
        s = []
        for i in idxs:
            base = baseline_mask_from_image(
                calib_imgs[i], thr=thr, polarity=chosen_polarity
            )
            refined = crf(
                calib_imgs[i],
                base,
                close_radius=base_close,
                min_obj=base_min_obj,
                min_hole=base_min_hole,
            )
            refined = apply_post_threshold(refined, post_thr=0.5)
            s.append(mean_ap_iou_kaggle(calib_msks[i], refined))
        m = float(np.mean(s))
        if m > best_b_score:
            best_b_score = m
            best_thr_b = thr
    thr_by_bin[b] = best_thr_b

for close_radius in close_grid:
    for min_obj in min_obj_grid:
        for min_hole in min_hole_grid:
            for post_thr in post_thr_grid:
                _ = eval_params_on(
                    fit_idx,
                    thr_by_bin=thr_by_bin,
                    close_radius=close_radius,
                    min_obj=min_obj,
                    min_hole=min_hole,
                    polarity=chosen_polarity,
                    post_thr=post_thr,
                )
                val_sc = eval_params_on(
                    val_idx,
                    thr_by_bin=thr_by_bin,
                    close_radius=close_radius,
                    min_obj=min_obj,
                    min_hole=min_hole,
                    polarity=chosen_polarity,
                    post_thr=post_thr,
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

for img_id, z in tqdm(
    sub_df[["id", "z"]].itertuples(index=False), total=len(sub_df), desc="Predicting"
):
    img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")
    if not os.path.exists(img_path):
        pred_rles.append("")
        continue

    img = imread(img_path)
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.uint8)

    b = depth_bin_index(float(z))
    thr = thr_by_bin[b]
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
