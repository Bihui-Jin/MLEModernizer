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

try:
    import pydensecrf.densecrf as dcrf  # noqa: F401
    from pydensecrf.utils import unary_from_labels  # noqa: F401
    from skimage.color import gray2rgb  # noqa: F401

    _HAS_DCRF = True
except ModuleNotFoundError:
    _HAS_DCRF = False




## === cell 1
def rle_decode(rle_mask, shape=(101, 101)):
    """
    rle_mask: run-length as string formatted (start length)
    Returns numpy array, 1 - mask, 0 - background, fixed shape (101, 101)

    Score-relevant correctness fix:
    - Decode using Fortran order to match Kaggle's column-major convention,
      consistent with rle_encode(order="F").
    """
    s = str(rle_mask).split()
    if len(s) == 0:
        return np.zeros(shape, dtype=np.uint8)

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")




## === cell 2
DATA_ROOT = "../input/tgs-salt-identification-challenge"
test_path = os.path.join(DATA_ROOT, "test", "images") + os.sep
train_img_path = os.path.join(DATA_ROOT, "train", "images") + os.sep
train_mask_path = os.path.join(DATA_ROOT, "train", "masks") + os.sep

sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
if not os.path.exists(sample_sub_path):
    sample_sub_path = "../input/sample_submission.csv"

sample_df = pd.read_csv(sample_sub_path)
if "id" not in sample_df.columns or "rle_mask" not in sample_df.columns:
    raise ValueError(
        f"Unexpected submission format in {sample_sub_path}. Columns: {sample_df.columns.tolist()}"
    )
sample_df["id"] = sample_df["id"].astype(str)

depths_path = os.path.join(DATA_ROOT, "depths.csv")
if not os.path.exists(depths_path):
    depths_path = "../input/depths.csv"
depths = pd.read_csv(depths_path)
depths["id"] = depths["id"].astype(str)

df = sample_df.copy()



## === cell 3
"""
reading and decoding the submission (visual sanity check)
Uses sample_submission by default so the notebook runs without external inputs.
"""
i = 0
j = 0
plt.figure(figsize=(30, 15))
plt.subplots_adjust(bottom=0.2, top=0.8, hspace=0.2)

while i < len(df) and j < 6:
    rle = df.loc[i, "rle_mask"]
    if isinstance(rle, str) and rle.strip() != "":
        decoded_mask = rle_decode(rle)
        plt.subplot(1, 6, j + 1)
        plt.imshow(decoded_mask, cmap="gray")
        plt.title("ID: " + df.loc[i, "id"])
        j += 1
    i += 1
plt.close()




## === cell 4
def crf(original_image, mask_img):
    mask_img = (mask_img > 0).astype(np.uint8)

    if not _HAS_DCRF:
        return mask_img

    from skimage.color import gray2rgb
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels

    if len(mask_img.shape) < 3:
        mask_rgb = gray2rgb(mask_img)
    else:
        mask_rgb = mask_img

    annotated_label = (
        mask_rgb[:, :, 0] + (mask_rgb[:, :, 1] << 8) + (mask_rgb[:, :, 2] << 16)
    )
    colors, labels = np.unique(annotated_label, return_inverse=True)

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
    MAP = np.argmax(Q, axis=0)
    return MAP.reshape((original_image.shape[0], original_image.shape[1])).astype(
        np.uint8
    )




## === cell 5
"""
used for converting the decoded image to rle mask
"""


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




## === cell 6
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_opening,
    disk,
)
from skimage.measure import label, regionprops


def keep_topk_components(mask, k=1):
    """
    Score improvement:
    - Largest-only can under-segment when salt appears split; allowing top-2 can improve IoU
      while still controlling false positives vs keeping everything.
    """
    m = mask.astype(bool)
    if m.sum() == 0:
        return mask.astype(np.uint8)
    lab = label(m)
    if lab.max() == 0:
        return mask.astype(np.uint8)
    props = regionprops(lab)
    props = sorted(props, key=lambda r: r.area, reverse=True)
    k = int(k)
    k = 1 if k < 1 else k
    keep_labels = {p.label for p in props[:k]}
    return np.isin(lab, list(keep_labels)).astype(np.uint8)


def heuristic_prob_from_image(img, z=None, depth_alpha=0.03):
    """
    Score improvement (minimal change):
    - Keep same heuristic (invert normalized grayscale) but tune depth shift strength (depth_alpha).
    """
    if img.ndim == 3:
        img = img[..., 0]
    x = img.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx > mn:
        x = (x - mn) / (mx - mn)
    else:
        x = np.zeros_like(x, dtype=np.float32)

    prob = 1.0 - x

    if z is not None and np.isfinite(z):
        z = float(z)
        z_norm = (z - 500.0) / 500.0
        prob = np.clip(prob + float(depth_alpha) * z_norm, 0.0, 1.0)

    return prob


def mask_from_prob(
    prob, cutoff, min_obj_size=25, hole_size=25, topk_cc=1, open_radius=1
):
    """
    Score improvement (minimal change):
    - Same morphology steps, tuned parameters.
    """
    mask = prob >= float(cutoff)

    r = int(open_radius)
    r = 0 if r < 0 else r
    if r > 0:
        mask = binary_opening(mask, footprint=disk(r))

    mask = remove_small_objects(mask, min_size=int(min_obj_size))
    mask = remove_small_holes(mask, area_threshold=int(hole_size))

    mask = keep_topk_components(mask, k=int(topk_cc))
    return (mask > 0).astype(np.uint8)




## === cell 7
def iou_binary(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0
    return inter / union


def mean_ap_iou_thresholds(y_true, y_pred):
    thresholds = np.arange(0.5, 1.0, 0.05)
    iou = iou_binary(y_true, y_pred)
    return float(np.mean([iou > t for t in thresholds]))


train_csv_path = os.path.join(DATA_ROOT, "train.csv")
if not os.path.exists(train_csv_path):
    train_csv_path = "../input/train.csv"
train_df = pd.read_csv(train_csv_path)
train_df["id"] = train_df["id"].astype(str)
train_df = train_df.merge(depths, on="id", how="left")

train_df = train_df.copy()
train_df["z_bin"] = pd.qcut(
    train_df["z"].fillna(train_df["z"].median()), q=10, duplicates="drop"
)

subset_n = 1200  # keep as-is for time; enough to stabilize tuning
per_bin = max(1, subset_n // train_df["z_bin"].nunique())
sampled = (
    train_df.groupby("z_bin", group_keys=False)
    .apply(lambda g: g.sample(n=min(per_bin, len(g)), random_state=1337))
    .reset_index(drop=True)
)
if len(sampled) < subset_n:
    remaining = train_df.loc[~train_df["id"].isin(sampled["id"])].copy()
    if len(remaining) > 0:
        topup = remaining.sample(
            n=min(subset_n - len(sampled), len(remaining)), random_state=1337
        )
        sampled = pd.concat([sampled, topup], axis=0, ignore_index=True)

train_ids = sampled["id"].values[:subset_n]

cutoffs = np.linspace(0.30, 0.80, 21)
min_obj_sizes = [15, 25, 40]
hole_sizes = [15, 25, 40]
topk_ccs = [1, 2]
open_radii = [0, 1, 2]
depth_alphas = [0.00, 0.02, 0.03, 0.05]

grid = []
for da in depth_alphas:
    for c in cutoffs:
        for mo in min_obj_sizes:
            for hs in hole_sizes:
                for tk in topk_ccs:
                    for rr in open_radii:
                        grid.append(
                            (float(da), float(c), int(mo), int(hs), int(tk), int(rr))
                        )

scores = np.zeros(len(grid), dtype=np.float64)

cache = []
used = 0
for img_id in train_ids:
    img_fp = train_img_path + img_id + ".png"
    mask_fp = train_mask_path + img_id + ".png"
    if not (os.path.exists(img_fp) and os.path.exists(mask_fp)):
        continue

    img = imread(img_fp)
    true_mask = (imread(mask_fp) > 0).astype(np.uint8)
    z = sampled.loc[sampled["id"] == img_id, "z"].values
    z = float(z[0]) if len(z) else None

    cache.append((img, z, true_mask))
    used += 1

den = max(1, used)

for img, z, true_mask in cache:
    base_prob = heuristic_prob_from_image(img, z=None, depth_alpha=0.0)

    for k, (da, c, mo, hs, tk, rr) in enumerate(grid):
        if z is not None and np.isfinite(z) and da != 0.0:
            z_norm = (float(z) - 500.0) / 500.0
            prob = np.clip(base_prob + float(da) * z_norm, 0.0, 1.0)
        else:
            prob = base_prob

        pred_mask = mask_from_prob(
            prob, cutoff=c, min_obj_size=mo, hole_size=hs, topk_cc=tk, open_radius=rr
        )
        scores[k] += mean_ap_iou_thresholds(true_mask, pred_mask)

scores = scores / den
best_idx = int(np.argmax(scores))
best_depth_alpha, best_cutoff, best_min_obj, best_hole, best_topk, best_open_r = grid[
    best_idx
]

pred_coverages = []
for img, z, _true_mask in cache[: min(len(cache), 200)]:
    prob = heuristic_prob_from_image(img, z=z, depth_alpha=best_depth_alpha)
    pm = mask_from_prob(
        prob,
        cutoff=best_cutoff,
        min_obj_size=best_min_obj,
        hole_size=best_hole,
        topk_cc=best_topk,
        open_radius=best_open_r,
    )
    pred_coverages.append(pm.mean())
avg_cov = float(np.mean(pred_coverages)) if len(pred_coverages) else 0.0

fallback_cutoff = 0.45
if avg_cov < 0.002:
    best_cutoff = fallback_cutoff

print("Tuned params on train subset:")
print("  depth_alpha =", best_depth_alpha)
print("  cutoff =", best_cutoff)
print("  min_obj_size =", best_min_obj)
print("  hole_size =", best_hole)
print("  topk_cc =", best_topk)
print("  open_radius =", best_open_r)
print("Used images for tuning =", used)
print("Best approx mean AP =", float(scores[best_idx]))
print("Avg predicted coverage (sanity) =", avg_cov)



## === cell 8
"""
visualizing the effect of applying CRF / heuristic (if there are any non-empty masks to show)
"""
np.random.seed(100)
nImgs = 4
i = np.random.randint(len(df))
j = 1
plt.figure(figsize=(20, 20))

shown = 0
df_with_z = df.merge(depths, on="id", how="left")

while i < len(df_with_z) and shown < nImgs:
    img_path = test_path + df_with_z.loc[i, "id"] + ".png"
    if os.path.exists(img_path):
        orig_img = imread(img_path)
        z = df_with_z.loc[i, "z"] if "z" in df_with_z.columns else None
        prob = heuristic_prob_from_image(orig_img, z=z, depth_alpha=best_depth_alpha)
        base_mask = mask_from_prob(
            prob,
            cutoff=best_cutoff,
            min_obj_size=best_min_obj,
            hole_size=best_hole,
            topk_cc=best_topk,
            open_radius=best_open_r,
        )
        crf_output = crf(orig_img, base_mask)

        plt.subplot(nImgs, 4, 4 * j - 3)
        plt.imshow(orig_img, cmap="gray")
        plt.title("Original image")

        plt.subplot(nImgs, 4, 4 * j - 2)
        plt.imshow(base_mask, cmap="gray")
        plt.title(
            f"Heuristic Mask (da={best_depth_alpha:.3f}, c={best_cutoff:.3f}, mo={best_min_obj}, hs={best_hole}, k={best_topk}, r={best_open_r})"
        )

        plt.subplot(nImgs, 4, 4 * j - 1)
        plt.imshow(crf_output, cmap="gray")
        plt.title("After CRF (if available)")

        plt.subplot(nImgs, 4, 4 * j)
        plt.imshow(base_mask ^ crf_output, cmap="gray")
        plt.title("Diff (heuristic xor crf)")

        j += 1
        shown += 1
    i += 1

plt.close()



## === cell 9
"""
Applying heuristic mask (tuned params) + (optional) CRF on the predicted mask, then encoding to RLE.

Score improvement:
- Use tuned depth_alpha + tuned cutoff + tuned morphology + top-k CC + tuned opening radius.
- Keep the same final interface (id, rle_mask) and still use the CRF refinement step.
- Ensure we preserve the sample_submission id order exactly (merge can reorder).
"""
work = sample_df[["id"]].copy()
work = work.merge(depths, on="id", how="left")

rles = []
for i in tqdm(range(work.shape[0])):
    img_id = work.loc[i, "id"]
    img_path = test_path + img_id + ".png"
    if os.path.exists(img_path):
        orig_img = imread(img_path)
        z = work.loc[i, "z"] if "z" in work.columns else None

        prob = heuristic_prob_from_image(orig_img, z=z, depth_alpha=best_depth_alpha)
        base_mask = mask_from_prob(
            prob,
            cutoff=best_cutoff,
            min_obj_size=best_min_obj,
            hole_size=best_hole,
            topk_cc=best_topk,
            open_radius=best_open_r,
        )

        refined = crf(orig_img, base_mask)
        rles.append(rle_encode(refined))
    else:
        rles.append("")

sub = sample_df.copy()
sub["rle_mask"] = rles
sub = sub[["id", "rle_mask"]]



## === cell 10
out_path = "crf_correction.csv"
sub.to_csv(out_path, index=False)

print(
    f"Wrote submission: {out_path} with shape {sub.shape} and columns {sub.columns.tolist()}"
)
print(sub.head())
print("Non-empty masks:", int((sub["rle_mask"].astype(str).str.len() > 0).sum()))
