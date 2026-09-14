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

0.73727

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, zipfile, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
from PIL import Image, ImageFilter

warnings.filterwarnings("ignore")




## === cell 1
class config:
    im_width = 128
    im_height = 128
    im_chan = 1  # grayscale
    path_train = "data/train/"
    path_test = "data/test/"




## === cell 2
def maybe_unzip(zip_path, dest):
    """
    Extract zip_path into dest if dest does not already exist.
    Silently skip if zip_path is missing (useful when data is already present).
    """
    if not Path(dest).exists():
        if Path(zip_path).exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest)


base_input = Path("..") / "input" / "tgs-salt-identification-challenge"
maybe_unzip(base_input / "train.zip", config.path_train)
maybe_unzip(base_input / "test.zip", config.path_test)




## === cell 3
def locate_image_dir(default_path):
    """
    Try several common locations for the image directory.
    Returns a Path if found, otherwise raises FileNotFoundError.
    """
    candidates = [
        Path(default_path) / "images",  # e.g. data/train/images
        Path("train") / "images",  # fallback to project root
        Path("test") / "images",
        Path("..") / "input" / "tgs-salt-identification-challenge" / "train" / "images",
        Path("..") / "input" / "tgs-salt-identification-challenge" / "test" / "images",
        Path("kaggle")
        / "input"
        / "tgs-salt-identification-challenge"
        / "train"
        / "images",
        Path("kaggle")
        / "input"
        / "tgs-salt-identification-challenge"
        / "test"
        / "images",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"Image directory not found among candidates: {candidates}")


train_dir = locate_image_dir(config.path_train)
test_dir = locate_image_dir(config.path_test)

train_ids = sorted(
    [
        f.name
        for f in train_dir.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
)
test_ids = sorted(
    [
        f.name
        for f in test_dir.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
)

print(f"Found {len(train_ids)} training images and {len(test_ids)} test images.")


def otsu_threshold(img):
    """
    Compute Otsu's threshold for a grayscale image (0‑255 uint8).
    """
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    total = img.size
    sum_total = np.dot(np.arange(256), hist)

    weight_background = 0
    sum_foreground = 0
    current_max, threshold = 0, 0

    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_foreground += t * hist[t]

        mean_background = sum_foreground / weight_background
        mean_foreground = (sum_total - sum_foreground) / weight_foreground

        var_between = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )

        if var_between > current_max:
            current_max = var_between
            threshold = t

    return threshold


preds_test_upsampled = []  # store binary masks at original resolution
for fn in tqdm(test_ids, desc="Predicting test masks"):
    img_path = test_dir / fn
    with Image.open(img_path) as im:
        im_gray = im.convert("L")
        img_arr = np.array(im_gray)

        pil_blur = Image.fromarray(img_arr).filter(ImageFilter.GaussianBlur(radius=1))
        img_arr_blur = np.array(pil_blur)

        thresh = otsu_threshold(img_arr_blur)
        mask = (img_arr_blur > thresh).astype(np.uint8)

        mask_img = Image.fromarray(mask * 255)
        mask_img = mask_img.filter(ImageFilter.MinFilter(3))
        mask_img = mask_img.filter(ImageFilter.MaxFilter(3))

        mask_img = mask_img.filter(ImageFilter.MedianFilter(size=3))

        mask = (np.array(mask_img) > 127).astype(np.uint8)

        if mask.sum() < 10:
            mask = np.zeros_like(mask)

        preds_test_upsampled.append(mask)




## === cell 4
def RLenc(img, order="F", format=True):
    """
    Run‑length encoding.
    img – binary mask (2‑D numpy array)
    order – 'F' for column‑wise (Fortran) order required by Kaggle
    format – if True returns a string, else a list of (start, length) tuples
    """
    bytes_ = img.reshape(-1, order=order)
    runs = []
    r = 0
    pos = 1  # 1‑based indexing
    for c in bytes_:
        if c == 0:
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1
    if r != 0:
        runs.append((pos, r))

    if format:
        return " ".join(f"{s} {l}" for s, l in runs)
    return runs




## === cell 5
pred_dict = {
    fn[:-4]: RLenc(mask, format=True)
    for fn, mask in zip(test_ids, preds_test_upsampled)
}

sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.name = "id"
sub.columns = ["rle_mask"]
sub_path = "submission.csv"
sub.to_csv(sub_path, index=True)

print(f"Submission file written to {sub_path}")
