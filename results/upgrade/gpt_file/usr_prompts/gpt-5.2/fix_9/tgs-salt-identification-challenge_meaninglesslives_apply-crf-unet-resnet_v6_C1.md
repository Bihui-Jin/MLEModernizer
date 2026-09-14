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
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_opening,
    binary_closing,
    square,
)

RANDOM_SEED = 100
np.random.seed(RANDOM_SEED)




## === cell 1
def rle_decode(rle_mask):
    """
    rle_mask: run-length as string formatted (start length)
    Returns numpy array mask of shape (101, 101), 1 - mask, 0 - background
    """
    if rle_mask is None or (isinstance(rle_mask, float) and np.isnan(rle_mask)):
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
def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
BASE_INPUT = "/kaggle/input/tgs-salt-identification-challenge"
test_path = os.path.join(BASE_INPUT, "test", "images")
train_img_path = os.path.join(BASE_INPUT, "train", "images")
train_mask_path = os.path.join(BASE_INPUT, "train", "masks")

sample_sub_path = os.path.join(BASE_INPUT, "sample_submission.csv")
if not os.path.exists(sample_sub_path):
    sample_sub_path = "/kaggle/input/sample_submission.csv"

df = pd.read_csv(sample_sub_path)
assert {"id", "rle_mask"}.issubset(
    df.columns
), "sample_submission.csv must have columns: id, rle_mask"

df["rle_mask"] = ""
df.head()




## === cell 4
def _read_grayscale_uint8(png_path):
    """
    Minimal robustness fix: ensure we always return (H,W) uint8 grayscale.
    This keeps your pipeline intact while avoiding channel/dtype issues.
    """
    orig_img = imread(png_path)
    if orig_img.ndim == 3:
        if orig_img.shape[-1] == 4:
            orig_img = rgba2rgb(orig_img)  # (H,W,3), float in [0,1]
        orig_img = (rgb2gray(orig_img) * 255.0).astype(np.uint8)
    else:
        if orig_img.dtype != np.uint8:
            orig_img = np.clip(orig_img, 0, 255).astype(np.uint8)
    return orig_img


def crf(original_image, mask_img, min_obj_size=12, min_hole_size=12, morph_ksize=3):
    """
    original_image: (H,W) uint8
    mask_img: (H,W) binary {0,1} or grayscale
    Returns refined binary mask (H,W) {0,1}
    """
    if mask_img is None:
        return np.zeros_like(original_image, dtype=np.uint8)

    if len(mask_img.shape) == 3:
        mask = mask_img[..., 0]
    else:
        mask = mask_img

    mask = (mask > 0).astype(bool)

    mask = binary_closing(mask, square(morph_ksize))
    mask = binary_opening(mask, square(morph_ksize))
    mask = remove_small_objects(mask, min_size=int(min_obj_size))
    mask = remove_small_holes(mask, area_threshold=int(min_hole_size))

    return mask.astype(np.uint8)




## === cell 5
def _robust_normalize_to_unit(img_uint8):
    x = img_uint8.astype(np.float32) / 255.0
    lo = float(np.percentile(x, 2.0))
    hi = float(np.percentile(x, 98.0))
    if hi > lo + 1e-6:
        x = (x - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return x


def predict_mask_from_image(
    img,
    q=0.80,
    min_pred_pixels=15,
    min_obj_size=12,
    min_hole_size=12,
    morph_ksize=3,
    invert=False,
    thr_scale=1.00,
    _x_norm=None,
):
    """
    img: (101,101) grayscale uint8
    returns: binary mask (101,101) uint8
    """
    x = _x_norm if _x_norm is not None else _robust_normalize_to_unit(img)

    thr = float(np.quantile(x, q)) * float(thr_scale)
    thr = float(np.clip(thr, 0.0, 1.0))

    if bool(invert):
        raw = (x < thr).astype(np.uint8)
    else:
        raw = (x > thr).astype(np.uint8)

    refined = crf(
        img,
        raw,
        min_obj_size=min_obj_size,
        min_hole_size=min_hole_size,
        morph_ksize=morph_ksize,
    )

    if int(refined.sum()) < int(min_pred_pixels):
        refined[:] = 0

    return refined.astype(np.uint8)




## === cell 6
_THRESHOLDS = np.arange(0.5, 1.0, 0.05).astype(np.float32)


def iou_score(y_true, y_pred):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    inter = int((y_true & y_pred).sum())
    union = int((y_true | y_pred).sum())
    if union == 0:
        return 1.0  # both empty
    return inter / union


def iou_map_metric(y_true, y_pred, thresholds=_THRESHOLDS):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)

    true_empty = y_true.sum() == 0
    pred_empty = y_pred.sum() == 0

    if true_empty and pred_empty:
        return 1.0
    if true_empty != pred_empty:
        return 0.0

    iou = iou_score(y_true, y_pred)
    return float((iou > thresholds).mean())


def tune_params_on_train(max_images=1500, seed=RANDOM_SEED):
    q_grid = [0.55, 0.60, 0.65, 0.70, 0.74, 0.78, 0.82]
    thr_scale_grid = [0.92, 0.96, 1.00, 1.04, 1.08]
    min_pred_pixels_grid = [0, 10, 20, 40]
    min_obj_size_grid = [8, 12, 20]
    min_hole_size_grid = [8, 12, 20]
    morph_ksize_grid = [3]
    invert_grid = [False, True]

    train_ids = sorted(
        [fn.split(".")[0] for fn in os.listdir(train_img_path) if fn.endswith(".png")]
    )
    if len(train_ids) == 0:
        return dict(
            q=0.80,
            min_pred_pixels=15,
            min_obj_size=12,
            min_hole_size=12,
            morph_ksize=3,
            invert=False,
            thr_scale=1.00,
        )

    rng = np.random.RandomState(seed)
    rng.shuffle(train_ids)
    use_ids = train_ids[: min(max_images, len(train_ids))]

    empty_ids, nonempty_ids = [], []
    for img_id in tqdm(use_ids, desc="Scanning masks for stratified split"):
        msk_fp = os.path.join(train_mask_path, f"{img_id}.png")
        if not os.path.exists(msk_fp):
            continue
        msk = imread(msk_fp)
        if msk.ndim == 3:
            msk = msk[..., 0]
        is_nonempty = bool((msk > 127).any())
        (nonempty_ids if is_nonempty else empty_ids).append(img_id)

    rng.shuffle(empty_ids)
    rng.shuffle(nonempty_ids)

    n_val_total = max(300, int(0.25 * (len(empty_ids) + len(nonempty_ids))))
    n_val_empty = int(
        round(
            n_val_total
            * (len(empty_ids) / max(1, (len(empty_ids) + len(nonempty_ids))))
        )
    )
    n_val_nonempty = n_val_total - n_val_empty

    val_ids = empty_ids[:n_val_empty] + nonempty_ids[:n_val_nonempty]
    if len(val_ids) < 200:
        val_ids = use_ids[: max(250, int(0.25 * len(use_ids)))]

    val_imgs = []
    val_xnorm = []
    val_true = []
    for img_id in tqdm(val_ids, desc="Loading val data"):
        img_fp = os.path.join(train_img_path, f"{img_id}.png")
        msk_fp = os.path.join(train_mask_path, f"{img_id}.png")
        if (not os.path.exists(img_fp)) or (not os.path.exists(msk_fp)):
            continue
        img = _read_grayscale_uint8(img_fp)
        msk = imread(msk_fp)
        if msk.ndim == 3:
            msk = msk[..., 0]
        msk = (msk > 127).astype(np.uint8)

        val_imgs.append(img)
        val_xnorm.append(_robust_normalize_to_unit(img))
        val_true.append(msk)

    if len(val_imgs) == 0:
        return dict(
            q=0.80,
            min_pred_pixels=15,
            min_obj_size=12,
            min_hole_size=12,
            morph_ksize=3,
            invert=False,
            thr_scale=1.00,
        )

    quant_cache = {}
    raw_cache = {}

    def _get_raw_mask(i, q, thr_scale, invert):
        key = (i, q, thr_scale, invert)
        m = raw_cache.get(key)
        if m is not None:
            return m
        qkey = (i, q)
        thr0 = quant_cache.get(qkey)
        if thr0 is None:
            thr0 = float(np.quantile(val_xnorm[i], q))
            quant_cache[qkey] = thr0
        thr = float(np.clip(thr0 * float(thr_scale), 0.0, 1.0))
        if invert:
            m = (val_xnorm[i] < thr).astype(np.uint8)
        else:
            m = (val_xnorm[i] > thr).astype(np.uint8)
        raw_cache[key] = m
        return m

    best = None
    best_score = -1.0

    for invert in invert_grid:
        for q in q_grid:
            for thr_scale in thr_scale_grid:
                raws = [
                    _get_raw_mask(i, q, thr_scale, invert) for i in range(len(val_imgs))
                ]

                for min_pred_pixels in min_pred_pixels_grid:
                    for min_obj_size in min_obj_size_grid:
                        for min_hole_size in min_hole_size_grid:
                            for morph_ksize in morph_ksize_grid:
                                scores = []
                                for i in range(len(val_imgs)):
                                    refined = crf(
                                        val_imgs[i],
                                        raws[i],
                                        min_obj_size=min_obj_size,
                                        min_hole_size=min_hole_size,
                                        morph_ksize=morph_ksize,
                                    )
                                    if int(refined.sum()) < int(min_pred_pixels):
                                        refined = refined.copy()
                                        refined[:] = 0
                                    scores.append(iou_map_metric(val_true[i], refined))
                                mean_score = float(np.mean(scores))
                                if mean_score > best_score:
                                    best_score = mean_score
                                    best = dict(
                                        q=float(q),
                                        thr_scale=float(thr_scale),
                                        min_pred_pixels=int(min_pred_pixels),
                                        min_obj_size=int(min_obj_size),
                                        min_hole_size=int(min_hole_size),
                                        morph_ksize=int(morph_ksize),
                                        invert=bool(invert),
                                    )

    print(f"Tuned params on val mAP={best_score:.4f} (n_val={len(val_imgs)}) -> {best}")
    return best


BEST_PARAMS = tune_params_on_train(max_images=1500)



## === cell 7
missing = 0
rles = []

for img_id in tqdm(df["id"].tolist(), desc="Generating masks"):
    img_fp = os.path.join(test_path, f"{img_id}.png")
    if not os.path.exists(img_fp):
        missing += 1
        rles.append("")
        continue

    orig_img = _read_grayscale_uint8(img_fp)
    pred_mask = predict_mask_from_image(orig_img, **BEST_PARAMS)
    rles.append(rle_encode(pred_mask))

df["rle_mask"] = rles

print("Missing images:", missing)
df.head()



## === cell 8
np.random.seed(RANDOM_SEED)
nImgs = 6
idxs = np.random.choice(df.index.values, size=nImgs, replace=False)

plt.figure(figsize=(18, 10))
for j, idx in enumerate(idxs, start=1):
    img_id = df.loc[idx, "id"]
    img_fp = os.path.join(test_path, f"{img_id}.png")
    orig_img = imread(img_fp)

    disp_img = orig_img
    if disp_img.ndim == 3:
        if disp_img.shape[-1] == 4:
            disp_img = rgba2rgb(disp_img)
        disp_img = rgb2gray(disp_img)

    mask = rle_decode(df.loc[idx, "rle_mask"])

    plt.subplot(2, nImgs, j)
    plt.imshow(disp_img, cmap="gray")
    plt.title(f"{img_id}\nimage")
    plt.axis("off")

    plt.subplot(2, nImgs, nImgs + j)
    plt.imshow(mask, cmap="gray")
    plt.title("mask")
    plt.axis("off")
plt.tight_layout()



## === cell 9
out_path1 = "/kaggle/working/crf_correction.csv"
out_path2 = "/kaggle/working/submission.csv"

df[["id", "rle_mask"]].to_csv(out_path1, index=False)
df[["id", "rle_mask"]].to_csv(out_path2, index=False)

print("Wrote:", out_path1)
print("Wrote:", out_path2)
print(df.shape)
print("BEST_PARAMS:", BEST_PARAMS)
