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
from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_opening,
    binary_closing,
    disk,
)

np.random.seed(42)




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
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        if lo < 0:
            lo = 0
        if hi > img.size:
            hi = img.size
        img[lo:hi] = 1
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

    mask = binary_opening(mask, disk(int(selem_radius)))
    mask = binary_closing(mask, disk(int(selem_radius)))
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

    val_imgs = []
    val_masks = []
    for img_id in tqdm(val_ids, total=len(val_ids), desc="Loading val"):
        img_path = os.path.join(TRAIN_IMG_DIR, f"{img_id}.png")
        msk_path = os.path.join(TRAIN_MASK_DIR, f"{img_id}.png")
        if not (os.path.exists(img_path) and os.path.exists(msk_path)):
            continue
        img = imread(img_path)
        if img.ndim == 3:
            img = img.mean(axis=2)
        msk = imread(msk_path)
        if msk.ndim == 3:
            msk = msk[..., 0]
        msk = (msk > 0).astype(np.uint8)
        val_imgs.append(img)
        val_masks.append(msk)

    thr_grid = [None, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68]
    inv_grid = [False, True]

    selem_grid = [1, 2]
    hole_grid = [10, 20, 40]
    obj_grid = [10, 20, 40]

    empty_area_grid = [0, 5, 10, 20, 40, 80, 120]
    pad_grid = [0, 2, 4]

    for inv in inv_grid:
        for thr in thr_grid:
            for se in selem_grid:
                for ha in hole_grid:
                    for ms in obj_grid:
                        for pad in pad_grid:
                            preds = []
                            areas = []
                            for img in val_imgs:
                                pr = refine_mask_from_image(
                                    img,
                                    thr=thr,
                                    invert=inv,
                                    selem_radius=se,
                                    hole_area=ha,
                                    obj_min_size=ms,
                                    pad=pad,
                                )
                                preds.append(pr)
                                areas.append(int(pr.sum()))

                            for area_cut in empty_area_grid:
                                scores = []
                                for gt, pr, a in zip(val_masks, preds, areas):
                                    pr2 = pr if a > area_cut else np.zeros_like(pr)
                                    scores.append(kaggle_map_iou(gt, pr2))
                                mean_score = (
                                    float(np.mean(scores)) if len(scores) else -1.0
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
for img_id in tqdm(sub["id"].values, total=len(sub), desc="Predict test"):
    img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")
    if not os.path.exists(img_path):
        pred_rles.append("")
        continue

    orig = imread(img_path)
    if orig.ndim == 3:
        orig = orig.mean(axis=2)

    mask = refine_mask_from_image(
        orig,
        thr=best_params["thr"],
        invert=best_params["invert"],
        selem_radius=best_params["selem_radius"],
        hole_area=best_params["hole_area"],
        obj_min_size=best_params["obj_min_size"],
        pad=best_params["pad"],
    )

    if int(mask.sum()) <= int(best_params["empty_area_cut"]):
        pred_rles.append("")
    else:
        pred_rles.append(rle_encode(mask))

sub["rle_mask"] = pred_rles

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote submission:", out_path)
print(sub.head())
