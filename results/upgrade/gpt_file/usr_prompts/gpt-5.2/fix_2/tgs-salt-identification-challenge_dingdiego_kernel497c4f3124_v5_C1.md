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

0.7782604932929458

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5221) has done: 'I make the notebook run in the Kaggle environment by (1) removing the unavailable `pydensecrf` dependency and the Jupyter-only `%matplotlib inline`, and (2) fixing the NaN checks and missing imports that currently prevent `df` from being created and used. To preserve the core intent (post-processing a precomputed submission with a “CRF-like correction”), I keep the same I/O flow and mask encode/decode logic, but implement a safe fallback post-processing step using only installed libraries (a small morphological cleanup) when CRF isn’t available. I also ensure the script doesn’t rely on an external dataset submission path that may not exist by falling back to `sample_submission.csv` if the baseline submission file is missing. Finally, the code always write a valid `crf_correction.csv` with the required `id,rle_mask` columns.'

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

import matplotlib.pyplot as plt
from tqdm import tqdm





## === cell 1
def rle_decode(rle_mask, shape=(101, 101)):
    """
    rle_mask: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
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
    return img.reshape(shape)




## === cell 2
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




## === cell 3
BASE_INPUT = "../input/tgs-salt-identification-challenge"
test_path = os.path.join(BASE_INPUT, "test", "images")

baseline_submission_path = (
    "../input/baseline-0-731-0-158-5-and-more-fold/submission.csv"
)
sample_submission_path = os.path.join(BASE_INPUT, "sample_submission.csv")

if os.path.exists(baseline_submission_path):
    df = pd.read_csv(baseline_submission_path)
else:
    df = pd.read_csv(sample_submission_path)

if "id" not in df.columns or "rle_mask" not in df.columns:
    raise ValueError(
        f"Submission file must contain columns ['id','rle_mask'], got {df.columns.tolist()}"
    )

df["rle_mask"] = df["rle_mask"].where(df["rle_mask"].notna(), "")

df.head()



## === cell 4
"""
Function which returns the labelled image after applying CRF.

pydensecrf is not installed here; to keep the pipeline working and preserve the
"post-process mask using the original image" intent, we provide a safe fallback:
- If the input mask is binary already (decoded), perform small morphological cleanup.
This is score-neutral-ish compared to failing, and avoids changing the model logic
(which is external in the baseline submission).
"""


def crf(original_image, mask_img):
    mask = mask_img
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > 0).astype(bool)

    se = disk(1)
    mask = binary_closing(mask, se)
    mask = binary_opening(mask, se)
    mask = remove_small_objects(mask, min_size=10)
    mask = remove_small_holes(mask, area_threshold=10)

    return mask.astype(np.uint8)




## === cell 5
"""
Optional visualization: safely show a few decoded masks and post-processed outputs.
This cell is non-essential for submission; keep it lightweight.
"""
non_empty_idx = df.index[df["rle_mask"].astype(str).str.len() > 0].tolist()
if len(non_empty_idx) == 0:
    vis_idx = list(
        np.random.choice(df.index.values, size=min(3, len(df)), replace=False)
    )
else:
    vis_idx = non_empty_idx[:3]

plt.figure(figsize=(12, 9))
for k, i in enumerate(vis_idx, start=1):
    img_id = df.loc[i, "id"]
    rle = df.loc[i, "rle_mask"]
    decoded_mask = rle_decode(rle)

    img_path = os.path.join(test_path, f"{img_id}.png")
    orig_img = (
        imread(img_path)
        if os.path.exists(img_path)
        else np.zeros((101, 101), dtype=np.uint8)
    )

    post = crf(orig_img, decoded_mask)

    plt.subplot(len(vis_idx), 3, 3 * (k - 1) + 1)
    plt.imshow(orig_img, cmap="gray")
    plt.title(f"{img_id} image")
    plt.axis("off")

    plt.subplot(len(vis_idx), 3, 3 * (k - 1) + 2)
    plt.imshow(decoded_mask, cmap="gray")
    plt.title("input mask")
    plt.axis("off")

    plt.subplot(len(vis_idx), 3, 3 * (k - 1) + 3)
    plt.imshow(post, cmap="gray")
    plt.title("post-processed")
    plt.axis("off")

plt.tight_layout()



## === cell 6
"""
Applying CRF (fallback post-processing) on the predicted masks.
Fixes: proper NaN/empty checks and ensures decode/encode is consistent.
"""
for i in tqdm(range(df.shape[0])):
    rle = df.loc[i, "rle_mask"]
    if isinstance(rle, float) and np.isnan(rle):
        continue
    rle_str = str(rle).strip()
    if rle_str == "" or rle_str.lower() == "nan":
        continue

    decoded_mask = rle_decode(rle_str)

    img_id = df.loc[i, "id"]
    img_path = os.path.join(test_path, f"{img_id}.png")
    orig_img = (
        imread(img_path)
        if os.path.exists(img_path)
        else np.zeros((101, 101), dtype=np.uint8)
    )

    crf_output = crf(orig_img, decoded_mask)
    df.loc[i, "rle_mask"] = rle_encode(crf_output)



## === cell 7
df = df[["id", "rle_mask"]].copy()
df["rle_mask"] = df["rle_mask"].fillna("").astype(str)

out_path = "crf_correction.csv"
df.to_csv(out_path, index=False)

print(f"Wrote submission to: {out_path}")
print(df.head())
