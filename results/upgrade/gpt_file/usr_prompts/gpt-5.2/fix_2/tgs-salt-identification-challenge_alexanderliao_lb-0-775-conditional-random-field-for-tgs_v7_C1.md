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

0.7943487667676323

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `pydensecrf` (not installed in this environment) by making CRF an optional step and safely falling back to “no-CRF” behavior so the notebook runs end-to-end. I also remove the Jupyter-only `%matplotlib inline` and ensure all required imports (`pd`, `plt`, `tqdm`) are available where used, fixing the `NameError`s. Because the referenced external submission (`../input/u-net-bn-aug-strat/submission.csv`) is not available, I instead start from the provided `sample_submission.csv` and create a valid `submission.csv` output with correct columns and `.csv` suffix. This yield a valid submission file (score be low vs target because there is no model here, but it execute correctly without missing-file/module errors).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import gray2rgb

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels

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
Reading a base submission.

The original notebook referenced an external dataset submission:
  ../input/u-net-bn-aug-strat/submission.csv
which is not present here, so we use the competition-provided sample_submission.csv
to ensure we can produce a valid output file end-to-end.
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

i = 0
j = 0
plt.figure(figsize=(18, 4))
plt.subplots_adjust(bottom=0.2, top=0.8, hspace=0.2)
while i < len(df) and j < 6:
    if pd.notna(df.loc[i, "rle_mask"]) and str(df.loc[i, "rle_mask"]).strip() != "":
        decoded_mask = rle_decode(df.loc[i, "rle_mask"])
        plt.subplot(1, 6, j + 1)
        plt.imshow(decoded_mask, cmap="gray")
        plt.title("ID: " + df.loc[i, "id"])
        j += 1
    i += 1
plt.show()



## === cell 3
"""
Function which returns the labelled image after applying CRF.

Bugfix: pydensecrf is not installed in this Kaggle environment, so we provide a safe fallback.
If CRF is unavailable, we return the input mask unchanged (identity post-processing).
"""


def crf(original_image, mask_img):
    if mask_img.ndim == 3:
        mask2d = mask_img[:, :, 0]
    else:
        mask2d = mask_img
    mask2d = (mask2d > 0).astype(np.uint8)

    if not PYDENSECRF_AVAILABLE:
        return mask2d

    labels = mask2d.flatten().astype(np.int32)

    h, w = original_image.shape[:2]
    d = dcrf.DenseCRF2D(w, h, 2)

    U = unary_from_labels(labels, 2, gt_prob=0.7, zero_unsure=False)
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



## === cell 5
"""
Visualizing the effect of applying CRF on a few images.

Bugfix: ensure plotting works without relying on prior failed imports and handle empty RLE rows.
"""
nImgs = 3
rng = np.random.default_rng(RANDOM_SEED)
start_i = int(rng.integers(0, len(df)))

j = 1
plt.figure(figsize=(12, 9))
plt.subplots_adjust(wspace=0.2, hspace=0.2)

i = start_i
shown = 0
while i < len(df) and shown < nImgs:
    rle = df.loc[i, "rle_mask"]
    decoded_mask = rle_decode(rle)

    img_id = df.loc[i, "id"]
    img_fp = os.path.join(test_path, f"{img_id}.png")
    if os.path.exists(img_fp):
        orig_img = imread(img_fp)
        crf_output = crf(orig_img, decoded_mask)

        plt.subplot(nImgs, 3, 3 * shown + 1)
        plt.imshow(orig_img, cmap="gray")
        plt.title("Original image")
        plt.axis("off")

        plt.subplot(nImgs, 3, 3 * shown + 2)
        plt.imshow(decoded_mask, cmap="gray")
        plt.title("Input Mask")
        plt.axis("off")

        plt.subplot(nImgs, 3, 3 * shown + 3)
        plt.imshow(crf_output, cmap="gray")
        plt.title("After CRF" if PYDENSECRF_AVAILABLE else "After CRF (fallback)")
        plt.axis("off")

        shown += 1
    i += 1

plt.show()



## === cell 6
"""
Used for converting the decoded image to rle mask.

Bugfix: enforce uint8/bool input and return empty string for empty mask (common in this competition).
"""


def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    im = (im > 0).astype(np.uint8)
    if im.sum() == 0:
        return ""

    pixels = im.flatten(
        order="F"
    )  # IMPORTANT: For this competition, RLE is typically column-major
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 7
"""
Applying CRF on the predicted mask.

Bugfix: works even if CRF dependency is missing; also uses robust image path joins.
Note: starting from sample_submission means masks are empty; this step is effectively a no-op.
"""
for i in tqdm(range(df.shape[0]), desc="Post-process (CRF)"):
    rle = df.loc[i, "rle_mask"]
    decoded_mask = rle_decode(rle)

    img_id = df.loc[i, "id"]
    img_fp = os.path.join(test_path, f"{img_id}.png")
    if os.path.exists(img_fp):
        orig_img = imread(img_fp)
        crf_output = crf(orig_img, decoded_mask)
    else:
        crf_output = decoded_mask

    df.loc[i, "rle_mask"] = rle_encode(crf_output)



## === cell 8
submission_path = "submission.csv"
df.to_csv(submission_path, index=False)
print(f"Wrote submission to: {submission_path}")
print(df.head())
