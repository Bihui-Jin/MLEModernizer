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

0.8076157507572471

# 6. Current score

0.0407

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0678) has done: 'I remove the hard dependency on `pydensecrf` (not installed) by providing a minimal fallback “CRF-like” post-processing using `skimage` morphology so the notebook runs end-to-end without import errors. I also eliminate notebook-only syntax (`%matplotlib inline`) and remove reliance on the missing external blend file (`../input/rle-masks-blend/blend_ez.csv`) by instead loading `sample_submission.csv` and producing a valid submission from it. The core pipeline remains “decode RLE → refine mask → encode RLE”; the fallback refinement is deterministic and lightweight to stay within Kaggle time limits. Finally, I ensure the output file is written with the required columns and a `.csv` suffix.'
- What this solution (achieved 0.029) has done: 'Your current score is far below the target, so we should make a small but meaningful improvement without changing the overall “predict mask → refine → RLE” pipeline. The biggest issue is that the prediction step is essentially arbitrary thresholding at 0.5 on per-image min-max normalized pixels, which yields near-random masks; we can greatly improve by using a data-driven threshold learned from the provided training RLE masks while keeping the same simple thresholding core logic. Concretely, we (1) compute the global positive pixel fraction from `train.csv` RLEs, (2) for each test image choose a threshold so the predicted mask matches that fraction (quantile threshold), and (3) keep your existing morphology “CRF fallback” refinement unchanged. This keeps the architecture/training approach unchanged (there is none) and only calibrates the threshold to something consistent with the dataset, which should move the score much closer to the target.'
- What this solution (achieved 0.0) has done: 'Your current score (0.029) is far below the target (0.8076), so we need a meaningful but still “same-core-logic” improvement: keep the existing per-image thresholding + morphology refine + RLE pipeline, but calibrate it using the provided training data. The main fix is to learn a **depth-conditioned expected positive fraction** (salt coverage varies strongly with depth), then for each test image choose the quantile threshold that matches the expected fraction for its depth; this is still just thresholding, but data-driven instead of global. I also fix the RLE orientation to the competition’s required **Fortran order** (top-to-bottom then left-to-right), which commonly causes very low scores if incorrect. These are minimal, metric-relevant changes and should move the score substantially toward the target without changing the overall approach.'
- What this solution (achieved 0.0407) has done: 'I fix the crash in the depth-calibration step by handling the fact that `depths.csv` contains only train IDs (so test depths are legitimately missing). To preserve your core “depth-conditioned expected fraction → per-image quantile threshold → morphology refine → RLE” pipeline, I estimate the depth→fraction mapping from train depths, then for test images infer a pseudo-depth from the image itself using the same `z = mean pixel intensity` proxy used in many baseline solutions. I also fix the cell numbering to start at 1 (your provided script started at cell 0) so it runs cleanly in your cell-formatted runner, and ensure a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.io import imread
from skimage.color import gray2rgb
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
    binary_opening,
    binary_closing,
    disk,
)

import matplotlib.pyplot as plt

np.random.seed(100)



## === cell 1
try:
    import pydensecrf.densecrf as dcrf  # noqa: F401
    from pydensecrf.utils import unary_from_labels  # noqa: F401

    PYDENSECRF_AVAILABLE = True
except ModuleNotFoundError:
    PYDENSECRF_AVAILABLE = False

print("PYDENSECRF_AVAILABLE:", PYDENSECRF_AVAILABLE)




## === cell 2
def rle_decode(rle_mask, shape=(101, 101)):
    """
    rle_mask: run-length as string formatted (start length)
    Returns numpy array, 1 - mask, 0 - background, shape (101, 101)
    """
    if rle_mask is None:
        return np.zeros(shape, dtype=np.uint8)
    if isinstance(rle_mask, float) and np.isnan(rle_mask):
        return np.zeros(shape, dtype=np.uint8)
    rle_mask = str(rle_mask).strip()
    if rle_mask == "" or rle_mask.lower() == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
BASE = "../input/tgs-salt-identification-challenge"
test_path = os.path.join(BASE, "test", "images")
sample_path = os.path.join(BASE, "sample_submission.csv")
train_csv_path = os.path.join(BASE, "train.csv")
depths_path = os.path.join(BASE, "depths.csv")

assert os.path.exists(sample_path), f"Missing sample_submission at {sample_path}"
assert os.path.isdir(test_path), f"Missing test images dir at {test_path}"
assert os.path.exists(train_csv_path), f"Missing train.csv at {train_csv_path}"
assert os.path.exists(depths_path), f"Missing depths.csv at {depths_path}"

df = pd.read_csv(sample_path)
if "id" not in df.columns or "rle_mask" not in df.columns:
    raise ValueError(f"Unexpected submission columns: {df.columns.tolist()}")

print("Loaded sample_submission:", df.shape)
print(df.head())



## === cell 4
train_df = pd.read_csv(train_csv_path)
if "id" not in train_df.columns or "rle_mask" not in train_df.columns:
    raise ValueError(f"Unexpected train.csv columns: {train_df.columns.tolist()}")

depths_df = pd.read_csv(depths_path)
if "id" not in depths_df.columns or "z" not in depths_df.columns:
    raise ValueError(f"Unexpected depths.csv columns: {depths_df.columns.tolist()}")

train_merged = train_df.merge(depths_df, on="id", how="left")
if train_merged["z"].isna().any():
    raise ValueError(
        "Some train ids are missing depths; cannot depth-calibrate safely."
    )

pos_fracs = np.zeros(len(train_merged), dtype=np.float32)
for i, rle in enumerate(
    tqdm(train_merged["rle_mask"].values, desc="Compute train positive fractions")
):
    m = rle_decode(rle, shape=(101, 101))
    pos_fracs[i] = m.mean()

GLOBAL_POS_FRAC = float(np.clip(pos_fracs.mean(), 1e-4, 0.9999))
print("Estimated GLOBAL_POS_FRAC from train:", GLOBAL_POS_FRAC)

z = train_merged["z"].values.astype(np.float32)
bin_edges = np.quantile(z, np.linspace(0, 1, 11))
bin_edges = np.unique(bin_edges)
if len(bin_edges) < 3:
    bin_edges = np.array([z.min(), z.max()], dtype=np.float32)

train_bins = np.digitize(z, bin_edges[1:-1], right=True)

bin_to_frac = {}
for b in np.unique(train_bins):
    vals = pos_fracs[train_bins == b]
    if len(vals) == 0:
        continue
    bin_to_frac[int(b)] = float(np.clip(np.median(vals), 1e-4, 0.9999))

print("Depth bins:", len(bin_edges) - 1, "Computed bin fracs:", len(bin_to_frac))


def expected_pos_frac_from_depth(z_value):
    b = int(
        np.digitize(np.array([z_value], dtype=np.float32), bin_edges[1:-1], right=True)[
            0
        ]
    )
    return bin_to_frac.get(b, GLOBAL_POS_FRAC)


def pseudo_depth_from_image(img):
    img = img.astype(np.float32)
    if img.ndim == 3:
        img = img[:, :, 0]
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        x = (img - mn) / (mx - mn)
    else:
        x = img * 0.0
    z_min, z_max = float(z.min()), float(z.max())
    return z_min + float(x.mean()) * (z_max - z_min)




## === cell 5
def crf(original_image, mask_img):
    """
    If pydensecrf is available, use DenseCRF refinement (original intent).
    Otherwise, apply a deterministic morphology-based refinement that keeps
    the decode->refine->encode semantics.
    """
    if mask_img is None:
        mask_img = np.zeros((101, 101), dtype=np.uint8)
    if mask_img.ndim == 3:
        mask2d = mask_img[:, :, 0]
    else:
        mask2d = mask_img

    mask2d = (mask2d > 0).astype(np.uint8)

    if PYDENSECRF_AVAILABLE:
        if len(mask_img.shape) < 3:
            mask_img_rgb = gray2rgb(mask2d)
        else:
            mask_img_rgb = mask_img

        annotated_label = (
            mask_img_rgb[:, :, 0]
            + (mask_img_rgb[:, :, 1] << 8)
            + (mask_img_rgb[:, :, 2] << 16)
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
        MAP = np.argmax(Q, axis=0)
        return MAP.reshape((original_image.shape[0], original_image.shape[1])).astype(
            np.uint8
        )

    m = mask2d.astype(bool)
    se = disk(1)
    m = binary_opening(m, se)
    m = binary_closing(m, se)
    m = remove_small_objects(m, min_size=15)
    m = remove_small_holes(m, area_threshold=15)
    return m.astype(np.uint8)




## === cell 6
def predict_mask_from_image(img, expected_pos_frac=GLOBAL_POS_FRAC):
    img = img.astype(np.float32)
    if img.ndim == 3:
        img = img[:, :, 0]
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        x = (img - mn) / (mx - mn)
    else:
        x = img * 0.0

    q = float(np.clip(1.0 - expected_pos_frac, 0.0, 1.0))
    thr = float(np.quantile(x, q))

    m = (x > thr).astype(np.uint8)
    return m




## === cell 7
pred_rles = []
for image_id in tqdm(df["id"].values, desc="Predict+Refine"):
    img_path = os.path.join(test_path, f"{image_id}.png")
    img = imread(img_path)

    z_val = float(pseudo_depth_from_image(img))
    exp_frac = expected_pos_frac_from_depth(z_val)

    raw_mask = predict_mask_from_image(img, expected_pos_frac=exp_frac)
    refined = crf(img, raw_mask)

    rle = rle_encode(refined)
    if rle.strip() == "":
        rle = ""
    pred_rles.append(rle)

df["rle_mask"] = pred_rles



## === cell 8
assert df.shape[0] == 1000, "Expected 1000 test rows"
assert (df["id"].astype(str).str.len() > 0).all(), "Empty ids found"
assert list(df.columns) == [
    "id",
    "rle_mask",
], f"Column order mismatch: {df.columns.tolist()}"

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote submission:", out_path, "shape:", df.shape)
print(df.head())
