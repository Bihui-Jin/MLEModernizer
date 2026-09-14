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

3.9

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

0.80965

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pathlib
import pandas as pd
import numpy as np
from PIL import Image

base_input = pathlib.Path("input/tgs-salt-identification-challenge")


class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    path_train = str(base_input / "train")
    path_test = str(base_input / "test")




## === cell 1
train_image_dir = os.path.join(config.path_train, "images")
train_mask_dir = os.path.join(config.path_train, "masks")
test_image_dir = os.path.join(config.path_test, "images")

if os.path.isdir(train_image_dir):
    train_ids = sorted(next(os.walk(train_image_dir))[2])
else:
    train_ids = []
    print("Train image directory not found; proceeding without training data.")

if os.path.isdir(test_image_dir):
    test_ids = sorted(next(os.walk(test_image_dir))[2])
else:
    test_ids = []
    print(
        "Test image directory not found; attempting to get IDs from sample submission."
    )
    sample_path = (
        pathlib.Path("input")
        / "tgs-salt-identification-challenge"
        / "sample_submission.csv"
    )
    if sample_path.is_file():
        try:
            sample_df = pd.read_csv(sample_path)
            test_ids = sample_df["id"].astype(str).tolist()
            print(f"Recovered {len(test_ids)} test IDs from sample_submission.csv.")
        except Exception as e:
            print("Failed to read sample submission for test IDs:", e)
    else:
        print("Sample submission not found; test set will be empty.")




## === cell 2
salt_vals = []
non_salt_vals = []

for img_name in train_ids:
    img_path = os.path.join(train_image_dir, img_name)
    mask_path = os.path.join(train_mask_dir, img_name)  # masks share the same filename
    if not (os.path.isfile(img_path) and os.path.isfile(mask_path)):
        continue
    img = np.array(Image.open(img_path).convert("L")).astype(np.float32) / 255.0
    mask = np.array(Image.open(mask_path).convert("L"))
    salt_vals.append(img[mask > 127])
    non_salt_vals.append(img[mask <= 127])

if salt_vals and non_salt_vals:
    salt_vals = np.concatenate(salt_vals)
    non_salt_vals = np.concatenate(non_salt_vals)
    mean_salt = salt_vals.mean()
    mean_non = non_salt_vals.mean()
    thresh = (mean_salt + mean_non) / 2.0
    print(f"Computed global threshold (fallback): {thresh:.4f}")
else:
    thresh = 0.5
    print("Could not compute threshold from training data; using default 0.5")




## === cell 3
def rle_encode(mask: np.ndarray) -> str:
    """
    Encode a binary mask (1 == salt) using run‑length encoding.
    The mask is flattened in column‑major order (Fortran order) as required by the competition.
    Returns an empty string if the mask has no positive pixels.
    """
    pixels = mask.T.flatten()  # transpose -> column‑major
    runs = []
    pos = np.where(pixels == 1)[0]
    if len(pos) == 0:
        return ""
    pos = pos + 1
    prev = -2
    for p in pos:
        if p > prev + 1:
            runs.append(p)
            runs.append(0)  # placeholder for length
        runs[-1] += 1
        prev = p
    return " ".join(map(str, runs))


def otsu_threshold(img: np.ndarray) -> float:
    """
    Compute Otsu's threshold for a single‑channel image.
    The input image is assumed to be normalized to [0, 1].
    Returns a threshold in the same [0, 1] range.
    """
    img_uint8 = (img * 255).astype(np.uint8)
    hist = np.bincount(img_uint8.ravel(), minlength=256)
    total = img_uint8.size
    sum_total = np.dot(np.arange(256), hist)

    sumB = 0.0
    wB = 0
    max_var = 0.0
    thresh = 0

    for t in range(256):
        wB += hist[t]
        if wB == 0 or wB == total:
            continue
        wF = total - wB
        sumB += t * hist[t]

        mB = sumB / wB
        mF = (sum_total - sumB) / wF

        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var:
            max_var = var_between
            thresh = t

    return thresh / 255.0


pred_dict = {}

for img_name in test_ids:
    img_path = os.path.join(test_image_dir, img_name)
    if not os.path.isfile(img_path):
        pred_dict[os.path.splitext(img_name)[0]] = "1 1"
        continue

    img = np.array(Image.open(img_path).convert("L")).astype(np.float32) / 255.0

    try:
        img_thresh = otsu_threshold(img)
    except Exception:
        img_thresh = thresh

    img_thresh = min(img_thresh, thresh)

    pred_mask = (img > img_thresh).astype(np.uint8)
    rle = rle_encode(pred_mask)
    if rle == "":
        rle = "1 1"
    pred_dict[os.path.splitext(img_name)[0]] = rle

sub = pd.DataFrame.from_dict(pred_dict, orient="index", columns=["rle_mask"])
sub.index.name = "id"
sub.to_csv("submission.csv", index=True)
print("Submission file 'submission.csv' written with", sub.shape[0], "rows.")
