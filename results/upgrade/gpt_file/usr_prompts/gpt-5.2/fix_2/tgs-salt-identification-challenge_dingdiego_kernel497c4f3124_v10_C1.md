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

0.8219558632626561

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5221) has done: 'I remove the unavailable `pydensecrf` dependency (it’s not installed in this environment) and replace the CRF step with a tiny, deterministic post-processing that keeps the same “refine predicted mask then re-encode to RLE” core workflow. I also remove notebook-only magics (`%matplotlib inline`) and the external input dependency on `../input/k-fold-merger/submission.csv` by instead reading the competition’s `sample_submission.csv` as the base (so the pipeline always produces a valid submission). Finally, I fix the NaN checks and missing imports so the script runs end-to-end and writes a proper `.csv` submission file with the required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
    binary_opening,
    binary_closing,
    disk,
)
from tqdm import tqdm

import matplotlib.pyplot as plt

np.random.seed(42)




## === cell 1
def rle_decode(rle_mask):
    """
    rle_mask: run-length as string formatted (start length)
    Returns numpy array mask of shape (101, 101), 1 - mask, 0 - background
    """
    if (
        rle_mask is None
        or (isinstance(rle_mask, float) and np.isnan(rle_mask))
        or str(rle_mask).strip() == ""
    ):
        return np.zeros((101, 101), dtype=np.uint8)

    s = str(rle_mask).split()
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
    Returns run length as string formatted
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.flatten(order="C")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 2
BASE = "../input/tgs-salt-identification-challenge"
if not os.path.exists(BASE):
    BASE = "../input"

sample_path = os.path.join(BASE, "sample_submission.csv")
test_img_dir = os.path.join(BASE, "test", "images")

if not os.path.exists(sample_path):
    alt_base = os.path.join(BASE, "tgs-salt-identification-challenge")
    sample_path = os.path.join(alt_base, "sample_submission.csv")
    test_img_dir = os.path.join(alt_base, "test", "images")

assert os.path.exists(
    sample_path
), f"Could not find sample_submission.csv at {sample_path}"
assert os.path.exists(
    test_img_dir
), f"Could not find test images directory at {test_img_dir}"

df = pd.read_csv(sample_path)
assert (
    "id" in df.columns and "rle_mask" in df.columns
), "sample_submission missing required columns"



## === cell 3
"""
Function which returns the refined mask.

Bug fix / environment fix:
- The original notebook used pydensecrf which is unavailable (ModuleNotFoundError).
- To keep the same pipeline semantics (refine mask then RLE encode) without changing model/training
  (there is none here), we apply a small deterministic morphological refinement.
This is score-neutral relative to having no CRF, but avoids runtime failure and produces a valid submission.
"""


def crf(original_image, mask_img):
    if len(mask_img.shape) >= 3:
        mask = mask_img[..., 0] > 0
    else:
        mask = mask_img > 0

    mask = remove_small_holes(mask, area_threshold=16)
    mask = remove_small_objects(mask, min_size=16)
    se = disk(1)
    mask = binary_closing(mask, footprint=se)
    mask = binary_opening(mask, footprint=se)

    return mask.astype(np.uint8)




## === cell 4
"""
(Optional) quick visualization sanity check on a few images.
This is not required for submission, but kept close to original intent.
"""
nImgs = 3
idxs = np.random.choice(df.index.values, size=min(nImgs, len(df)), replace=False)

plt.figure(figsize=(12, 8))
for j, i in enumerate(idxs, start=1):
    img_id = df.loc[i, "id"]
    img_path = os.path.join(test_img_dir, f"{img_id}.png")
    orig_img = imread(img_path)

    decoded_mask = rle_decode(df.loc[i, "rle_mask"])
    refined = crf(orig_img, decoded_mask)

    plt.subplot(nImgs, 3, 3 * j - 2)
    plt.imshow(orig_img, cmap="gray")
    plt.title(f"{img_id} image")
    plt.axis("off")

    plt.subplot(nImgs, 3, 3 * j - 1)
    plt.imshow(decoded_mask, cmap="gray")
    plt.title("input mask")
    plt.axis("off")

    plt.subplot(nImgs, 3, 3 * j)
    plt.imshow(refined, cmap="gray")
    plt.title("refined mask")
    plt.axis("off")

plt.tight_layout()



## === cell 5
"""
Applying refinement on the predicted mask and writing a valid submission.

Logic fix:
- The original notebook expected an external submission at ../input/k-fold-merger/submission.csv.
  That dataset isn't present here, so we base off sample_submission.csv to guarantee output.
- Robust NaN/empty handling via pandas isna.
"""
for i in tqdm(range(df.shape[0])):
    rle = df.loc[i, "rle_mask"]
    if pd.isna(rle) or str(rle).strip() == "":
        decoded_mask = np.zeros((101, 101), dtype=np.uint8)
    else:
        decoded_mask = rle_decode(rle)

    img_id = df.loc[i, "id"]
    img_path = os.path.join(test_img_dir, f"{img_id}.png")
    orig_img = imread(img_path)

    refined_mask = crf(orig_img, decoded_mask)
    df.loc[i, "rle_mask"] = rle_encode(refined_mask)



## === cell 6
df = df[["id", "rle_mask"]]
out_path = "crf_correction.csv"
df.to_csv(out_path, index=False)

print(f"Wrote submission: {out_path}")
print(df.head())
