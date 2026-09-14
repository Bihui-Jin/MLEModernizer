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

0.8086888792730411

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The changes add safe imports, handle the missing `pydensecrf` library by providing a fallback CRF that returns the original mask, correctly locate and read an existing sample submission file, and ensure all required libraries are imported before use. The processing loop now decodes each mask, optionally refines it (no‑op when CRF is unavailable), re‑encodes it, and finally writes a proper `submission.csv` file, guaranteeing a valid Kaggle‑ready CSV.'
- What this solution (achieved 0.0603) has done: 'We add a lightweight image‑based prediction: for rows that contain the placeholder mask (“1 1”, empty or NaN) we compute a binary mask by Otsu thresholding the test image’s grayscale values, then optionally refine it with the existing CRF step. This keeps the original architecture, encoding/decoding utilities, and CRF fallback, but replaces the never‑changing sample submission masks with data‑driven masks, yielding a non‑zero score and moving the result toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_dilation,
    binary_closing,
    disk,
)
from skimage.measure import label  # added for post‑processing
from skimage import exposure  # new import for contrast enhancement

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral

    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False




## === cell 1
def rle_decode(rle_mask):
    """
    Decode a run‑length encoding string to a 101x101 binary mask.
    """
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101)




## === cell 2
def rle_encode(im):
    """
    Encode a binary mask (numpy array) to run‑length format string.
    """
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
def crf(original_image, mask_img):
    """
    Refine mask using DenseCRF if the library is available.
    If not, return the original mask unchanged.
    """
    if not CRF_AVAILABLE:
        return mask_img
    if mask_img.ndim < 3:
        mask_img = np.stack([mask_img] * 3, axis=-1)
    annotated_label = (
        mask_img[:, :, 0] + (mask_img[:, :, 1] << 8) + (mask_img[:, :, 2] << 16)
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
    return MAP.reshape((original_image.shape[0], original_image.shape[1]))




## === cell 4
possible_paths = [
    "../input/tgs-salt-identification-challenge/sample_submission.csv",
    "../input/sample_submission.csv",
    "sample_submission.csv",
    "../input/u-net-with-simple-resnet-blocks-forked/submission.csv",
]
submission_path = None
for p in possible_paths:
    if os.path.exists(p):
        submission_path = p
        break
if submission_path is None:
    raise FileNotFoundError("No submission CSV found in expected locations.")
df = pd.read_csv(submission_path)
test_path = "../input/tgs-salt-identification-challenge/test/images/"




## === cell 5
def simple_threshold_mask(image):
    """
    Produce a binary mask by:
    1. Converting to grayscale.
    2. Applying stronger Gaussian smoothing.
    3. Enhancing contrast with adaptive histogram equalization (CLAHE).
    4. Otsu thresholding.
    5. Removing tiny objects/holes.
    6. Closing small gaps to improve continuity.
    """
    gray = image.mean(axis=2) if image.ndim == 3 else image.astype(np.float32)

    gray_smooth = gaussian(gray, sigma=1.5)

    gray_norm = np.clip(gray_smooth, 0, 1)
    gray_eq = exposure.equalize_adapthist(gray_norm, clip_limit=0.03)

    thresh = threshold_otsu(gray_eq)
    mask = (gray_eq > thresh).astype(np.uint8)

    mask_bool = mask.astype(bool)
    mask_bool = remove_small_objects(mask_bool, min_size=10)
    mask_bool = remove_small_holes(mask_bool, area_threshold=10)

    mask_bool = binary_closing(mask_bool, selem=disk(2))

    return mask_bool.astype(np.uint8).reshape(101, 101)




## === cell 6
def clean_mask(mask):
    """
    Keep only the largest connected component and fill small holes.
    This reduces spurious isolated predictions, then dilate slightly to improve recall.
    """
    labeled = label(mask, connectivity=1)
    if labeled.max() == 0:  # no objects
        return mask
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    largest_label = component_sizes.argmax()
    cleaned = (labeled == largest_label).astype(np.uint8)
    cleaned = remove_small_holes(cleaned.astype(bool), area_threshold=30).astype(
        np.uint8
    )
    dilated = binary_dilation(cleaned, selem=disk(2)).astype(np.uint8)
    return dilated




## === cell 7
for i in tqdm(range(df.shape[0]), desc="Refining masks"):
    rle = str(df.loc[i, "rle_mask"])
    img_path = os.path.join(test_path, f"{df.loc[i, 'id']}.png")
    if not os.path.exists(img_path):
        continue
    orig_img = plt.imread(img_path)

    if rle.strip().lower() in ("nan", "", "1 1"):
        decoded_mask = simple_threshold_mask(orig_img)
    else:
        decoded_mask = rle_decode(rle)

    refined = crf(orig_img, decoded_mask)
    refined = clean_mask(refined)  # post‑process to improve quality
    df.loc[i, "rle_mask"] = rle_encode(refined)




## === cell 8
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
