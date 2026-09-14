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

No external packages required in the script and installed.

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

0.8134746862829938

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The script now safely handles the missing pydensecrf package by falling back to a no‑op CRF, corrects the data paths, ensures all imported symbols are available, and adds robust checks for missing masks. It reads the provided sample submission, decodes/encodes masks, optionally visualises a few examples, applies the (fallback) CRF, and finally writes a valid `submission.csv` file.'
- What this solution (achieved 0.0) has done: 'I add a simple Otsu‑based thresholding step to generate a baseline mask when the provided mask is empty, keeping the original CRF post‑processing. This tiny change creates non‑trivial predictions, moving the score upward toward the target while preserving all existing logic.'
- What this solution (achieved 0.0) has done: 'I ensure that every test image receives a predicted mask by generating an Otsu‑threshold mask for rows where the original submission is missing or empty, then (optionally) refine it with a small‑object removal step before applying the optional CRF. This guarantees a non‑NaN RLE for all rows, turning the zero score into a meaningful baseline that moves toward the target.'
- What this solution (achieved 0.0) has done: 'The script failed because it could not locate the sample‑submission file or the test image folder, and it also missed the `rle_decode` helper used later. I added a robust file‑search that walks the directory tree, implemented `rle_decode`, and re‑ordered the cells so that the data are loaded before they are used. No core modelling logic was changed – the fixes only ensure the pipeline runs end‑to‑end and writes a proper `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import gray2rgb
from skimage import filters, morphology  # Otsu and small‑object removal
from tqdm import tqdm

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels
except ImportError:  # pragma: no cover
    dcrf = None
    unary_from_labels = None


def rle_decode(rle_str, shape=(101, 101)):
    """
    Decode a run‑length encoded string into a binary mask.
    """
    if pd.isna(rle_str) or rle_str == "":
        return np.zeros(shape, dtype=np.uint8)

    s = list(map(int, rle_str.strip().split()))
    starts, lengths = s[0::2], s[1::2]
    starts = np.array(starts) - 1  # convert to zero‑based indexing
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rle_encode(im):
    """
    Encode a binary mask array (1 = mask, 0 = background) to run‑length format.
    Returns a string; for an empty mask returns a minimal placeholder.
    """
    if im.sum() == 0:
        return "1 1"
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 1
def _find_file(possible_paths, fallback_name=None):
    """
    Return the first existing path from ``possible_paths``.
    If none exist and ``fallback_name`` is provided, walk the filesystem
    from the current directory to locate a file with that basename.
    """
    for p in possible_paths:
        if os.path.exists(p):
            return p
    if fallback_name:
        for root, _, files in os.walk("."):
            if fallback_name in files:
                return os.path.join(root, fallback_name)
    raise FileNotFoundError(
        f"File not found. Searched locations: {possible_paths}"
        + (f" (fallback search for {fallback_name})" if fallback_name else "")
    )


def _find_dir(possible_dirs):
    """
    Return the first existing directory from ``possible_dirs``.
    Raises FileNotFoundError if none are found.
    """
    for d in possible_dirs:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"Directory not found. Searched locations: {possible_dirs}")




## === cell 2
submission_path = _find_file(
    [
        os.path.join("input", "sample_submission.csv"),
        os.path.join("data", "sample_submission.csv"),
        os.path.join(
            "input", "tgs-salt-identification-challenge", "sample_submission.csv"
        ),
        os.path.join("kaggle", "data", "sample_submission.csv"),
    ],
    fallback_name="sample_submission.csv",
)
df = pd.read_csv(submission_path)




## === cell 3
def crf(original_image, mask_img):
    """
    Apply DenseCRF if the library is available; otherwise return the mask unchanged.
    """
    if dcrf is None:
        return mask_img

    if mask_img.ndim < 3:
        mask_img = gray2rgb(mask_img)

    annotated_label = (
        mask_img[:, :, 0].astype(np.uint32)
        + (mask_img[:, :, 1].astype(np.uint32) << 8)
        + (mask_img[:, :, 2].astype(np.uint32) << 16)
    )
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2  # background / salt
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
test_path = _find_dir(
    [
        os.path.join("input", "tgs-salt-identification-challenge", "test", "images"),
        os.path.join("data", "test", "images"),
        os.path.join("test", "images"),
        os.path.join("kaggle", "working", "test", "images"),
    ]
)




## === cell 5
def generate_baseline_mask(image):
    """
    Produce a simple binary mask using Otsu thresholding and remove tiny objects.
    """
    if image.ndim == 3:
        gray = image[:, :, 0]
    else:
        gray = image
    thresh = filters.threshold_otsu(gray)
    mask = (gray > thresh).astype(np.uint8)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=30)
    return mask.astype(np.uint8)




## === cell 6
for i in tqdm(range(df.shape[0]), desc="Processing masks"):
    img_path = os.path.join(test_path, df.loc[i, "id"] + ".png")
    orig_img = imread(img_path)

    decoded_mask = rle_decode(df.loc[i, "rle_mask"])
    if decoded_mask.sum() <= 1:  # placeholder "1 1" or empty
        decoded_mask = generate_baseline_mask(orig_img)

    crf_output = crf(orig_img, decoded_mask)

    if crf_output.sum() == 0:
        crf_output = generate_baseline_mask(orig_img)

    df.loc[i, "rle_mask"] = rle_encode(crf_output)




## === cell 7
submission_df = df[["id", "rle_mask"]].copy()
submission_df.to_csv("submission.csv", index=False)
