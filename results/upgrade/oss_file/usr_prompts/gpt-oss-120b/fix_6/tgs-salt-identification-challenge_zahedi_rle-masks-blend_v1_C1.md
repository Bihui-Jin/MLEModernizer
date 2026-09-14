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

0.81087840761575

# 6. Current score

0.087

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I replace the missing submission‑file reads with a safe load of the provided sample submission (which always exists), drop the faulty merging logic, and directly write this dataframe out as the final submission CSV. This removes the FileNotFoundError and NameError chain, ensuring the notebook runs end‑to‑end and produces a valid `blend_ez.csv` file.'
- What this solution (achieved 0.1079) has done: 'I replace the final cell so that, instead of just copying the sample submission, it builds a real prediction for each test image by loading the image, applying a simple intensity‑based threshold, encoding the mask with the existing RLE functions, and then writing the resulting DataFrame. This adds a minimal yet valid model‑like step, which should raise the score from 0 toward the target while keeping the original utility functions unchanged.'
- What this solution (achieved 0.0603) has done: 'I replace the naive fixed‑intensity threshold with an Otsu‑based threshold per image and drop tiny isolated predictions, which should raise the mean‑average‑precision score toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.087) has done: 'The fix updates the morphology calls to use the correct `footprint` argument and simplifies the thresholding to rely on Otsu’s value when it can be computed, which improves mask quality and avoids the TypeError that stopped the script. The cells are renumbered starting at 1 and the corrected functions ensure a valid `blend_ez.csv` submission is written.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Top‑level contents:", os.listdir("../input"))




## === cell 1
def rle_encode(im):
    """
    Encode a binary mask (numpy array) to run‑length encoding.
    im: 2‑D array of 0/1 values.
    Returns a space‑separated string.
    """
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(rle_mask, shape=(101, 101)):
    """
    Decode a run‑length string back to a binary mask.
    Returns a 2‑D numpy array of shape (101,101).
    """
    if not rle_mask or pd.isna(rle_mask):
        return np.zeros(shape, dtype=np.uint8)
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)




## === cell 2
def merge_dataframes(dfs, merge_keys):
    """Utility to merge a list of dataframes on given keys."""
    from functools import reduce

    return reduce(lambda left, right: pd.merge(left, right, on=merge_keys), dfs)




## === cell 3
paths_to_try = [
    "../input/sample_submission.csv",
    "../../input/sample_submission.csv",
    "../input/tgs-salt-identification-challenge/sample_submission.csv",
    "../../input/tgs-salt-identification-challenge/sample_submission.csv",
]

df_sub = None
base_input_dir = None
for p in paths_to_try:
    if os.path.exists(p):
        df_sub = pd.read_csv(p)
        print(f"Loaded sample submission from: {p}")
        base_input_dir = os.path.abspath(os.path.join(p, ".."))
        break

if df_sub is None:
    raise FileNotFoundError(
        "Unable to locate sample_submission.csv in expected directories."
    )

assert (
    "id" in df_sub.columns and "rle_mask" in df_sub.columns
), "Sample submission missing required columns."

train_csv_path = os.path.join(base_input_dir, "train.csv")
train_img_dir = os.path.join(base_input_dir, "train", "images")
if not os.path.isfile(train_csv_path) or not os.path.isdir(train_img_dir):
    raise FileNotFoundError("Training CSV or image folder not found for calibration.")

train_df = pd.read_csv(train_csv_path)

fg_sum, fg_cnt = 0.0, 0
bg_sum, bg_cnt = 0.0, 0
from skimage.io import imread

for _, row in train_df.iterrows():
    img_id = row["id"]
    mask_rle = row["rle_mask"]
    mask = rle_decode(mask_rle)
    img_path = os.path.join(train_img_dir, f"{img_id}.png")
    if not os.path.exists(img_path):
        continue
    img = imread(img_path)
    if img.ndim == 3:
        img_gray = img.mean(axis=2)
    else:
        img_gray = img.astype(np.float32)

    fg_pixels = img_gray[mask == 1]
    bg_pixels = img_gray[mask == 0]
    fg_sum += fg_pixels.sum()
    fg_cnt += fg_pixels.size
    bg_sum += bg_pixels.sum()
    bg_cnt += bg_pixels.size

global_thr = 127.0
if fg_cnt > 0 and bg_cnt > 0:
    mean_fg = fg_sum / fg_cnt
    mean_bg = bg_sum / bg_cnt
    global_thr = (mean_fg + mean_bg) / 2.0
print(f"Computed global intensity threshold from training data: {global_thr:.2f}")




## === cell 4
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    binary_opening,
    binary_closing,
    disk,
)

test_img_dir = os.path.join(base_input_dir, "test", "images")
if not os.path.isdir(test_img_dir):
    test_img_dir = os.path.join(
        base_input_dir, "tgs-salt-identification-challenge", "test", "images"
    )
    if not os.path.isdir(test_img_dir):
        raise FileNotFoundError(f"Test image directory not found in {test_img_dir}")

print(f"Using test images from: {test_img_dir}")


def simple_mask_from_image(img):
    """
    Produce a binary mask using Otsu (fallback to global) and clean it with
    small‑object removal and mild morphological opening/closing.
    """
    if img.ndim == 3:
        img_gray = img.mean(axis=2)
    else:
        img_gray = img

    thr = global_thr
    try:
        thr = threshold_otsu(img_gray)
    except Exception:
        pass

    mask = (img_gray > thr).astype(np.uint8)

    mask = remove_small_objects(mask.astype(bool), min_size=25)

    se = disk(2)
    mask = binary_opening(mask, footprint=se)
    mask = binary_closing(mask, footprint=se)

    return mask.astype(np.uint8)


pred_rles = []
for _, row in df_sub.iterrows():
    img_id = row["id"]
    img_path = os.path.join(test_img_dir, f"{img_id}.png")
    if os.path.exists(img_path):
        img = imread(img_path)
        mask = simple_mask_from_image(img)
    else:
        mask = np.zeros((101, 101), dtype=np.uint8)
    pred_rles.append(rle_encode(mask))

df_sub["rle_mask"] = pred_rles

output_path = "blend_ez.csv"
df_sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
