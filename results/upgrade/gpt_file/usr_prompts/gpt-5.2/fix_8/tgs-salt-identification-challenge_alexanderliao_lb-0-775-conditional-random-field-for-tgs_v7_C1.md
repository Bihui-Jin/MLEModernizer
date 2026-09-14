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



## === cell 5
"""
Baseline segmentation + calibration.

Score-relevant improvements (minimal, preserves core pipeline):
- Use a small deterministic train/val split inside calibration to reduce overfitting of
  hyperparameters to the calibration images themselves (better generalization -> higher LB AP).
- Add a tiny morphology-consistent option to clear a 1px border, which often removes
  edge artifacts (false positives) without changing the core approach.
"""

IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05, dtype=np.float32)


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

    mask = binary_closing(mask, footprint=disk(int(closing_radius)))
    mask = remove_small_objects(mask, min_size=int(min_obj))
    mask = remove_small_holes(mask, area_threshold=int(min_hole))
    mask = mask.astype(np.uint8)

    if clear_border:
        mask = _clear_border_1px(mask)

    return mask.astype(np.uint8)


def iou_binary(y_true, y_pred):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    inter = (y_true & y_pred).sum()
    union = (y_true | y_pred).sum()
    if union == 0:
        return 1.0  # both empty
    return float(inter) / float(union)


def competition_ap_single(y_true, y_pred, thresholds=IOU_THRESHOLDS):
    """
    For this dataset each image has a single binary mask (one object vs background).
    The TGS metric reduces to averaging precision over IoU thresholds.

    When both GT and pred are empty, this is a perfect prediction, AP=1.0.
    """
    iou = iou_binary(y_true, y_pred)
    y_true_has = y_true.sum() > 0
    y_pred_has = y_pred.sum() > 0

    aps = []
    for t in thresholds:
        if (not y_true_has) and (not y_pred_has):
            aps.append(1.0)
            continue

        if y_true_has and y_pred_has:
            if iou > float(t):
                tp, fp, fn = 1, 0, 0
            else:
                tp, fp, fn = 0, 1, 1
        elif (not y_true_has) and y_pred_has:
            tp, fp, fn = 0, 1, 0
        else:  # y_true_has and not y_pred_has
            tp, fp, fn = 0, 0, 1

        denom = tp + fp + fn
        aps.append(0.0 if denom == 0 else tp / denom)
    return float(np.mean(aps))


def calibrate_baseline_params(
    train_df,
    n_calib=600,
    val_frac=0.25,
    offsets=(-24, -18, -12, -8, -5, -3, -1, 0, 1, 3, 5, 8, 12, 18, 24),
    directions=("le", "ge"),
    closing_radii=(0, 1, 2),
    min_objs=(15, 30, 60),
    min_holes=(15, 30, 60),
    empty_gates=(0.0, 0.001, 0.002, 0.0035, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03),
    clear_borders=(False, True),
):
    rng = np.random.default_rng(RANDOM_SEED)

    ids_all = train_df["id"].values
    if len(ids_all) > n_calib:
        ids_all = rng.choice(ids_all, size=n_calib, replace=False)

    ims = []
    gts = []
    otsus = []
    for img_id in ids_all:
        img_fp = os.path.join(train_img_path, f"{img_id}.png")
        msk_fp = os.path.join(train_mask_path, f"{img_id}.png")
        if not (os.path.exists(img_fp) and os.path.exists(msk_fp)):
            continue
        im = imread(img_fp)
        gt = imread(msk_fp)
        im2 = _to_gray(im)
        ims.append(im2)
        gts.append((gt > 127).astype(np.uint8))
        try:
            otsus.append(float(threshold_otsu(im2)))
        except ValueError:
            otsus.append(float(np.mean(im2)))

    n = len(ims)
    if n == 0:
        return dict(
            offset=0.0,
            direction="le",
            closing_radius=1,
            min_obj=30,
            min_hole=30,
            empty_gate=0.0,
            clear_border=False,
            calib_val_mean_ap=0.0,
        )

    ims = np.asarray(ims, dtype=np.float32)
    gts = np.asarray(gts, dtype=np.uint8)
    otsus = np.asarray(otsus, dtype=np.float32)

    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    best = None
    best_score = -1.0

    for direction in directions:
        for clear_border in clear_borders:
            for closing_radius in closing_radii:
                for min_obj in min_objs:
                    for min_hole in min_holes:
                        for off in offsets:
                            preds = []
                            fracs = []
                            for im2, th in zip(ims, otsus):
                                pred = baseline_mask_from_image(
                                    im2,
                                    thr=float(th + off),
                                    direction=direction,
                                    closing_radius=closing_radius,
                                    min_obj=min_obj,
                                    min_hole=min_hole,
                                    clear_border=clear_border,
                                )
                                preds.append(pred)
                                fracs.append(float(pred.mean()))
                            preds = np.asarray(preds, dtype=np.uint8)
                            fracs = np.asarray(fracs, dtype=np.float32)

                            for gate in empty_gates:
                                for_eval = []
                                for j in val_idx:
                                    pred = preds[j]
                                    frac = fracs[j]
                                    pred2 = (
                                        pred
                                        if frac >= gate
                                        else np.zeros_like(pred, dtype=np.uint8)
                                    )
                                    for_eval.append(
                                        competition_ap_single(gts[j], pred2)
                                    )
                                score_val = (
                                    float(np.mean(for_eval)) if len(for_eval) else -1.0
                                )

                                if score_val > best_score:
                                    best_score = score_val
                                    best = dict(
                                        offset=float(off),
                                        direction=direction,
                                        closing_radius=int(closing_radius),
                                        min_obj=int(min_obj),
                                        min_hole=int(min_hole),
                                        empty_gate=float(gate),
                                        clear_border=bool(clear_border),
                                        calib_val_mean_ap=float(best_score),
                                        calib_n=n,
                                        calib_val_n=int(len(val_idx)),
                                    )

    return best


best_params = calibrate_baseline_params(train_df)
thr_offset = best_params["offset"]
best_direction = best_params["direction"]
best_closing_radius = best_params["closing_radius"]
best_min_obj = best_params["min_obj"]
best_min_hole = best_params["min_hole"]
best_empty_gate = best_params["empty_gate"]
best_clear_border = best_params["clear_border"]

print("Calibrated baseline params:", best_params)



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

        if float(pred_mask.mean()) < best_empty_gate:
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

Core pipeline unchanged; uses calibrated baseline params and a calibrated empty-mask gate
to reduce FP/FN and move score upward.
"""
for i in tqdm(range(df.shape[0]), desc="Predict + Post-process (CRF optional)"):
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

        if float(pred_mask.mean()) < best_empty_gate:
            pred_mask = np.zeros_like(pred_mask, dtype=np.uint8)

        post_mask = crf(orig_img, pred_mask)
    else:
        post_mask = np.zeros((101, 101), dtype=np.uint8)

    df.loc[i, "rle_mask"] = rle_encode(post_mask)



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
print("Empty gate used:", best_empty_gate)
print("Clear border used:", best_clear_border)
