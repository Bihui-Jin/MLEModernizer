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

0.82226

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0623) has done: 'I added handling for PNG images that contain an alpha channel (4 channels) by dropping the extra channel before converting to grayscale, and I included a small median filter to denoise the image before applying Otsu’s threshold. These minimal changes fix the runtime error and should improve the segmentation quality enough to move the score toward the target.'
- What this solution (achieved 0.0475) has done: 'I keep the overall pipeline unchanged but improve the Otsu‑based mask generation: after thresholding the image I smooth it, remove tiny objects, keep only the largest connected component (salt appears as a single blob), and fill holes. These modest post‑processing steps are known to raise the mean‑average‑precision for this competition while preserving the original simple‑threshold logic.'
- What this solution (achieved 0.0445) has done: 'The fix adds a lightweight fallback that switches to an adaptive local threshold when the Otsu mask is too empty (which was causing the very low score). This keeps the original simple‑threshold pipeline but improves segmentation on difficult images, moving the metric closer to the target while preserving the core logic.'
- What this solution (achieved 0.026) has done: 'The update keeps the original Otsu‑based pipeline but makes the preprocessing and fallback logic a little more tolerant: a slightly larger median filter, a larger structuring element for opening/closing, a lower size threshold for removing tiny objects, and a clearer rule that switches to an adaptive local threshold when the Otsu result is too empty (or too full). These modest tweaks stay within the existing workflow while expected to raise the foreground‑mask quality and move the score closer to the target.'
- What this solution (achieved 0.0) has done: 'I enhance the Otsu‑based mask generator with a few low‑cost image‑processing tweaks that stay within the original “simple threshold” workflow but improve contrast, reduce noise, and apply a stricter post‑processing pipeline. These changes (CLAHE contrast equalisation, larger median and morphological kernels, a combined Otsu + adaptive mask, higher size filtering, and keeping only the biggest component) are expected to raise the mean‑average‑precision substantially and move the score much closer to the target while preserving the overall pipeline.'
- What this solution (achieved 0.0) has done: 'The update adds a tiny fallback mask when the processing pipeline produces an empty result, ensuring every image gets a non‑empty RLE string and thus a valid submission that can score above zero. This minimal change keeps the original Otsu‑based workflow untouched while preventing the all‑zero masks that caused the 0.0 score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import (
    threshold_otsu,
    median,
    threshold_local,
)
from skimage.morphology import (
    disk,
    binary_opening,
    binary_closing,
    remove_small_objects,
)
from skimage.measure import label
from scipy.ndimage import binary_fill_holes
from skimage.exposure import equalize_adapthist  # added for contrast enhancement


def rle_decode(mask_rle):
    """
    mask_rle: run‑length as string formatted (start length)
    Returns a 101x101 binary mask (1‑mask, 0‑background)
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101)


def rle_encode(im):
    """
    im: 2‑D binary numpy array (1‑mask, 0‑background)
    Returns run‑length as a space‑separated string
    """
    pixels = im.flatten()
    if pixels.sum() == 0:
        return "1 1"
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def generate_mask_otsu(image):
    """
    Produce a binary mask from an image using Otsu's threshold.
    In addition to the original logic we:
    • Apply CLAHE (adaptive histogram equalisation) to boost contrast.
    • Use a larger median filter (disk radius 5) to suppress noise.
    • Combine Otsu mask with a local adaptive mask (AND) for robustness.
    • Employ larger structuring elements (disk radius 5) for opening/closing.
    • Remove very small objects (min_size=200) and keep only the largest component.
    """
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]

    if image.ndim == 2:
        gray = (
            image.astype(np.float32) / 255.0
            if image.max() > 1
            else image.astype(np.float32)
        )
    else:
        gray = rgb2gray(image)

    gray = equalize_adapthist(gray, clip_limit=0.03)

    gray = median(gray, disk(5))

    otsu_thresh = threshold_otsu(gray)
    otsu_mask = (gray > otsu_thresh).astype(np.uint8)

    local_mask = (gray > threshold_local(gray, block_size=101, offset=0)).astype(
        np.uint8
    )

    combined = (otsu_mask & local_mask).astype(np.uint8)

    if combined.mean() < 0.02 or combined.mean() > 0.98:
        combined = local_mask

    if combined.mean() > 0.5:
        combined = 1 - combined

    combined = binary_opening(combined, disk(5))
    combined = binary_closing(combined, disk(5))

    combined = remove_small_objects(combined.astype(bool), min_size=200).astype(
        np.uint8
    )

    labeled = label(combined)
    if labeled.max() > 0:
        counts = np.bincount(labeled.flat)
        largest_label = np.argmax(counts[1:]) + 1
        combined = (labeled == largest_label).astype(np.uint8)

    combined = binary_fill_holes(combined).astype(np.uint8)

    if combined.sum() == 0:
        centre = (combined.shape[0] // 2, combined.shape[1] // 2)
        combined[centre] = 1

    return combined


candidate_dirs = [
    os.path.join("kaggle", "data", "test", "images"),
    os.path.join("input", "test", "images"),
    os.path.join("input", "tgs-salt-identification-challenge", "test", "images"),
    os.path.join(
        "kaggle", "input", "tgs-salt-identification-challenge", "test", "images"
    ),
    os.path.join("test", "images"),
]

TEST_IMAGES_DIR = None
for cand in candidate_dirs:
    if os.path.isdir(cand) and any(
        fname.lower().endswith(".png") for fname in os.listdir(cand)
    ):
        TEST_IMAGES_DIR = cand
        break

if TEST_IMAGES_DIR is None:
    for root, dirs, files in os.walk("."):
        if os.path.basename(root) == "images":
            parent = os.path.basename(os.path.dirname(root))
            if parent == "test" and any(f.lower().endswith(".png") for f in files):
                TEST_IMAGES_DIR = root
                break

if TEST_IMAGES_DIR is None:
    raise FileNotFoundError(
        "Test images directory not found. Checked paths: "
        + ", ".join(candidate_dirs)
        + " (also performed recursive search)."
    )

test_ids = [
    os.path.splitext(fname)[0]
    for fname in os.listdir(TEST_IMAGES_DIR)
    if fname.lower().endswith(".png")
]

submission_records = []
for img_id in tqdm(test_ids, desc="Generating masks"):
    img_path = os.path.join(TEST_IMAGES_DIR, f"{img_id}.png")
    img = imread(img_path)
    mask = generate_mask_otsu(img)
    rle = rle_encode(mask)
    submission_records.append({"id": img_id, "rle_mask": rle})

submission_df = pd.DataFrame(submission_records, columns=["id", "rle_mask"])
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path} (rows: {len(submission_df)})")
