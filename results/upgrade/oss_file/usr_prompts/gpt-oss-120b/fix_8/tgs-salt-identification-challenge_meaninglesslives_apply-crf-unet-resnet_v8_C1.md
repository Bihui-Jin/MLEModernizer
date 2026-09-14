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

0.8039723063608821

# 6. Current score

0.0576

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The changes remove the unavailable `pydensecrf` dependency, replace the CRF function with a simple pass‑through (so the pipeline runs), fix missing imports, and correct the submission file path. The script now loads the sample submission, decodes each mask, optionally visualises a few examples, re‑encodes (unchanged) masks, and writes a valid `crf_correction.csv` submission file.'
- What this solution (achieved 0.0341) has done: 'I adjust the script to (1) reliably locate the input files using a unified `INPUT_DIR` so the CSV and test images are read correctly, (2) replace the placeholder CRF with a lightweight Otsu‑thresholding heuristic that creates a plausible mask from each test image, and (3) keep the rest of the pipeline unchanged. These minimal changes fix the missing‑file issue and generate non‑trivial masks, moving the Kaggle score toward the target without altering the core workflow.'
- What this solution (achieved 0.0352) has done: 'I improve the mask generation by (1) applying a Gaussian blur before Otsu, then cleaning the binary mask with opening/closing and removing tiny objects, and (2) ensuring every test image gets a prediction (even when the original sample‑submission entry is NaN) by starting from an empty mask. These changes keep the overall pipeline intact while producing more realistic masks, which should raise the mean‑average‑precision score toward the target.'
- What this solution (achieved 0.1301) has done: 'I enhance the heuristic `crf` function by applying adaptive histogram equalization before thresholding, using both Otsu‑derived foreground and background masks, and relaxing the small‑object removal (min_size = 16). These tweaks keep the overall pipeline unchanged while producing richer binary masks that should raise the mean‑average‑precision toward the target score.'
- What this solution (achieved 0.0125) has done: 'I replace the overly aggressive “OR‑both‑foreground‑and‑background” logic in the `crf` function with a single Otsu‑based foreground mask, add hole‑filling, and keep the original mask via logical OR. This yields more realistic binary masks (avoiding the previous all‑ones mask) and should raise the mean‑average‑precision toward the target while preserving the existing pipeline.'
- What this solution (achieved 0.0) has done: 'I cleanly load the depth information (in case it can be useful later) and strengthen the heuristic segmentation: use both global Otsu and adaptive local thresholds, raise the minimum object size, and ensure holes are filled. These modest but focused tweaks keep the original pipeline intact while producing much more realistic masks, which should raise the mean‑average‑precision toward the target score.'
- What this solution (achieved 0.0576) has done: 'The fix adds the missing `rle_decode` function, ensures any remaining NaN masks are replaced with an empty mask, and slightly relaxes the small‑object removal in the heuristic `crf` function to improve segmentation quality while keeping the original workflow unchanged. These minimal changes resolve the runtime errors and produce a valid submission CSV, moving the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread
from skimage import filters, morphology, exposure
from skimage.morphology import remove_small_holes
from tqdm import tqdm

INPUT_DIR = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path("./input")



## === cell 1
sample_path = INPUT_DIR / "sample_submission.csv"
df = pd.read_csv(sample_path)
assert "id" in df.columns and "rle_mask" in df.columns

depth_path_candidates = [
    INPUT_DIR / "depths.csv",
    INPUT_DIR / "tgs-salt-identification-challenge" / "depths.csv",
    INPUT_DIR / "train" / "depths.csv",
]
depth_dict = {}
for dpath in depth_path_candidates:
    if dpath.exists():
        depths_df = pd.read_csv(dpath)
        depth_dict = dict(zip(depths_df["id"].astype(str), depths_df["z"]))
        break




## === cell 2
def rle_decode(mask_rle):
    """
    Decode a run‑length encoded string into a binary mask (101x101).
    The encoding is 1‑indexed and column‑major (Fortran order).
    """
    if pd.isna(mask_rle):
        return np.zeros((101, 101), dtype=np.uint8)
    s = mask_rle.strip().split()
    starts = np.asarray(s[0::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((101, 101), order="F")




## === cell 3
plt.figure(figsize=(30, 5))
for idx in range(min(6, len(df))):
    if pd.isna(df.loc[idx, "rle_mask"]):
        continue
    mask = rle_decode(df.loc[idx, "rle_mask"])
    plt.subplot(1, 6, idx + 1)
    plt.imshow(mask, cmap="gray")
    plt.title(df.loc[idx, "id"])
plt.tight_layout()
plt.show()




## === cell 4
def crf(original_image, mask_img):
    """
    Refined heuristic segmentation:
    1. Grayscale conversion.
    2. Adaptive histogram equalisation.
    3. Gaussian blur.
    4. Combine global Otsu and adaptive local thresholds.
    5. Morphological opening & closing.
    6. Remove small objects (min_size=30).
    7. Fill small holes (area_threshold=30).
    8. Logical OR with the supplied mask.
    """
    if original_image.ndim == 3:
        gray = original_image.mean(axis=2).astype(np.uint8)
    else:
        gray = original_image.astype(np.uint8)
    gray_eq = exposure.equalize_adapthist(gray, clip_limit=0.03)
    gray_eq = (gray_eq * 255).astype(np.uint8)
    blurred = filters.gaussian(gray_eq, sigma=1.0, preserve_range=True).astype(np.uint8)

    otsu_thresh = filters.threshold_otsu(blurred)
    otsu_mask = blurred > otsu_thresh

    local_thresh = filters.threshold_local(blurred, block_size=35, offset=10)
    local_mask = blurred > local_thresh

    combined_thresh = np.logical_or(otsu_mask, local_mask).astype(np.uint8)

    selem = morphology.square(3)
    opened = morphology.opening(combined_thresh, selem)
    closed = morphology.closing(opened, selem)

    cleaned = morphology.remove_small_objects(closed.astype(bool), min_size=30)
    cleaned = remove_small_holes(cleaned, area_threshold=30)

    final_mask = np.logical_or(mask_img.astype(bool), cleaned).astype(np.uint8)
    return final_mask




## === cell 5
test_path = INPUT_DIR / "tgs-salt-identification-challenge" / "test" / "images"




## === cell 6
def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run‑length string.
    """
    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 7
for i in tqdm(range(df.shape[0]), desc="Processing masks"):
    img_id = df.loc[i, "id"]
    img_path = os.path.join(str(test_path), f"{img_id}.png")
    try:
        orig_img = imread(img_path)
    except FileNotFoundError:
        continue

    rle = df.loc[i, "rle_mask"]
    decoded_mask = rle_decode(rle)

    corrected_mask = crf(orig_img, decoded_mask)
    df.loc[i, "rle_mask"] = rle_encode(corrected_mask)

empty_rle = rle_encode(np.zeros((101, 101), dtype=np.uint8))
df["rle_mask"] = df["rle_mask"].fillna(empty_rle)



## === cell 8
df.to_csv("crf_correction.csv", index=False)
print("Submission saved to crf_correction.csv")
