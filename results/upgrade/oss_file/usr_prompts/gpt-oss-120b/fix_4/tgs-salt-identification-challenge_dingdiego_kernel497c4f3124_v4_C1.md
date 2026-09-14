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

0.7821289485071389

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I guard the unavailable `pydensecrf` import, replace the non‑Python `ls` line with a harmless placeholder, point the submission read to the actual `sample_submission.csv`, and tweak the CRF function to fall back to a no‑op when the library is missing. This resolves all NameError and ModuleNotFound errors and ensures a proper CSV file is written at the end.'
- What this solution (achieved 0.0305) has done: 'Implemented a robust test‑image lookup, added a simple grayscale‑threshold mask fallback (so we generate meaningful masks even when the original placeholder is empty), and safely handle missing/NaN RLE entries. The CRF step is still used when available; otherwise the new heuristic mask is encoded. These minimal changes keep the original workflow intact while producing a non‑trivial submission that should move the score upward toward the target.'
- What this solution (achieved 0.5221) has done: 'I compute a global pixel‑wise salt probability from the training masks and use this mask (or a CRF‑refined version when possible) for every test image instead of the naive per‑image threshold. This adds a single inexpensive preprocessing step and replaces the low‑quality threshold mask, which should raise the mean‑average‑precision toward the target while keeping the original workflow intact.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral

    DENSECRF_AVAILABLE = True
except ModuleNotFoundError:
    DENSECRF_AVAILABLE = False




## === cell 1
pass




## === cell 2
"""
Reading the baseline submission (sample submission) for further processing.
"""
submission_path = os.path.join("..", "input", "sample_submission.csv")
df = pd.read_csv(submission_path)




## === cell 3
def rle_decode(rle_mask):
    """
    rle_mask: run-length as string formatted (start length)
    Returns numpy array of shape (101,101) with 1‑mask, 0‑background.
    """
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101)




## === cell 4
def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as space‑separated string.
    """
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 5
def crf(original_image, mask_img):
    """
    Apply DenseCRF if the library is available; otherwise return the input mask.
    """
    if not DENSECRF_AVAILABLE:
        return mask_img.astype(np.uint8)

    if mask_img.ndim < 3:
        mask_img = np.stack([mask_img] * 3, axis=-1)

    annotated_label = (
        mask_img[:, :, 0] + (mask_img[:, :, 1] << 8) + (mask_img[:, :, 2] << 16)
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
    return MAP.reshape((original_image.shape[0], original_image.shape[1]))


train_candidate_paths = [
    os.path.join("..", "input", "tgs-salt-identification-challenge", "train", "masks"),
    os.path.join("..", "input", "train", "masks"),
    os.path.join("train", "masks"),
    os.path.join("input", "train", "masks"),
]
train_mask_dir = None
for p in train_candidate_paths:
    if os.path.isdir(p):
        train_mask_dir = p
        break
if train_mask_dir is None:
    raise RuntimeError("Unable to locate training masks directory.")

mask_files = [
    os.path.join(train_mask_dir, f)
    for f in os.listdir(train_mask_dir)
    if f.lower().endswith(".png")
]
if not mask_files:
    raise RuntimeError("No mask files found in training masks directory.")

global_sum = np.zeros((101, 101), dtype=np.float32)
for mf in mask_files:
    m = plt.imread(mf)
    if m.ndim == 3:
        m = m[..., 0]
    binary = (m > 0).astype(np.float32)
    global_sum += binary
global_prob = global_sum / len(mask_files)
global_mask = (global_prob > 0.5).astype(np.uint8)




## === cell 6
candidate_paths = [
    os.path.join("..", "input", "tgs-salt-identification-challenge", "test", "images"),
    os.path.join("..", "input", "test", "images"),
    os.path.join("test", "images"),
    os.path.join("input", "test", "images"),
]
test_path = None
for p in candidate_paths:
    if os.path.isdir(p):
        test_path = p
        break
if test_path is None:
    raise RuntimeError("Unable to locate test images directory.")




## === cell 7
for idx in tqdm(range(df.shape[0]), desc="Processing masks"):
    rle = df.loc[idx, "rle_mask"]
    if pd.isna(rle) or str(rle).strip() == "":
        mask = np.zeros((101, 101), dtype=np.uint8)
    else:
        decoded_mask = rle_decode(str(rle))

        img_path = os.path.join(test_path, df.loc[idx, "id"] + ".png")
        if os.path.exists(img_path):
            orig_img = plt.imread(img_path)
        else:
            orig_img = None

        if orig_img is not None:
            if DENSECRF_AVAILABLE:
                mask = crf(orig_img, global_mask)
            else:
                mask = global_mask
        else:
            mask = decoded_mask

    df.loc[idx, "rle_mask"] = rle_encode(mask)




## === cell 8
output_path = "crf_correction.csv"
df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
