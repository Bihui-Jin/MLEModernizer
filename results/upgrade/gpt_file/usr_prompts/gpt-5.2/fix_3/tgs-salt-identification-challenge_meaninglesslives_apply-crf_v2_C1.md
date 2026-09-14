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

0.7702985720467324

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I remove the hard dependency on `pydensecrf` (not installed) by adding a safe fallback that keeps the pipeline running end-to-end: if CRF isn’t available, we apply a small, deterministic morphological post-processing instead. I also remove notebook-only magic (`%matplotlib inline`) and fix the data input path: your script currently reads a submission file that doesn’t exist in this environment, so it instead start from the provided `sample_submission.csv`. Finally, I make the NaN/empty-mask checks correct and ensure the script always writes a valid `crf_correction.csv` with the required `id,rle_mask` columns.'
- What this solution (achieved 0.5221) has done: 'Your current script only post-processes masks that already exist in `sample_submission.csv`, but that file is intentionally empty/blank for this competition, so you end up submitting essentially all-empty masks and capping the score. To move toward the target, the minimal fix is to generate a reasonable baseline mask from each test image (simple deterministic thresholding) and then apply your existing CRF/fallback cleanup on top, keeping the same overall pipeline structure. I also fix the RLE flatten order to the competition’s required top-to-bottom then left-to-right convention (Fortran order), which can materially improve IoU without changing the modeling approach. These are small, deterministic changes that should raise the score substantially toward your 0.77 target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.morphology import binary_opening, binary_closing, square
from skimage.filters import threshold_otsu

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels

    HAS_DCRF = True
except Exception:
    HAS_DCRF = False




## === cell 1
def rle_decode(mask_rle):
    """
    mask_rle: run-length as string formatted (start length)
    Returns numpy array, 1 - mask, 0 - background, shape (101, 101)

    NOTE: Decode follows Kaggle's convention used in this competition:
    pixels are 1-indexed and ordered top-to-bottom, then left-to-right,
    which corresponds to flatten(order="F").
    """
    if mask_rle is None or (isinstance(mask_rle, float) and np.isnan(mask_rle)):
        return np.zeros((101, 101), dtype=np.uint8)
    s = str(mask_rle).strip()
    if s == "" or s.lower() == "nan":
        return np.zeros((101, 101), dtype=np.uint8)

    s = s.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((101, 101), order="F")




## === cell 2
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted

    NOTE: Encode follows Kaggle's convention used in this competition:
    flatten(order="F") (top-to-bottom, then left-to-right).
    """
    img = (img > 0).astype(np.uint8)
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
"""
Function which returns the labelled image after applying CRF.
If DenseCRF is not installed, apply a small morphological cleanup as a safe fallback.
"""


def crf(original_image, annotated_image, use_2d=True):
    if annotated_image is None:
        mask = np.zeros(
            (original_image.shape[0], original_image.shape[1]), dtype=np.uint8
        )
    else:
        mask = (annotated_image > 0).astype(np.uint8)

    if HAS_DCRF and use_2d:
        if len(mask.shape) < 3:
            annotated_rgb = gray2rgb(mask)
        else:
            annotated_rgb = mask

        annotated_label = (
            annotated_rgb[:, :, 0].astype(np.uint32)
            + (annotated_rgb[:, :, 1].astype(np.uint32) << 8)
            + (annotated_rgb[:, :, 2].astype(np.uint32) << 16)
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

    cleaned = binary_opening(mask.astype(bool), square(3))
    cleaned = binary_closing(cleaned, square(3))
    return cleaned.astype(np.uint8)




## === cell 4
def initial_mask_from_image(orig_img):
    if orig_img.ndim == 3:
        img = orig_img[:, :, 0]
    else:
        img = orig_img
    img = img.astype(np.float32)

    t = threshold_otsu(img)
    mask = (img > t).astype(np.uint8)

    mask = binary_opening(mask.astype(bool), square(3))
    mask = binary_closing(mask, square(3))
    return mask.astype(np.uint8)




## === cell 5
BASE = "../input/tgs-salt-identification-challenge"
test_path = os.path.join(BASE, "test", "images")

sub_path = os.path.join(BASE, "sample_submission.csv")
df = pd.read_csv(sub_path)

assert (
    "id" in df.columns and "rle_mask" in df.columns
), "Submission template must have columns: id, rle_mask"



## === cell 6
for i in tqdm(range(df.shape[0])):
    img_id = df.loc[i, "id"]
    img_path = os.path.join(test_path, f"{img_id}.png")
    if not os.path.exists(img_path):
        continue

    orig_img = imread(img_path)

    rle = df.loc[i, "rle_mask"]
    if pd.isna(rle) or str(rle).strip() == "":
        decoded_mask = initial_mask_from_image(orig_img)
    else:
        decoded_mask = rle_decode(rle)

    crf_output = crf(orig_img, decoded_mask)
    df.loc[i, "rle_mask"] = rle_encode(crf_output)



## === cell 7
out_path = "crf_correction.csv"
df.to_csv(out_path, index=False)

assert out_path.endswith(".csv")
chk = pd.read_csv(out_path)
assert list(chk.columns) == ["id", "rle_mask"]
assert chk.shape[0] == df.shape[0]
print(f"Wrote {out_path} with shape {chk.shape}. DenseCRF available: {HAS_DCRF}")
