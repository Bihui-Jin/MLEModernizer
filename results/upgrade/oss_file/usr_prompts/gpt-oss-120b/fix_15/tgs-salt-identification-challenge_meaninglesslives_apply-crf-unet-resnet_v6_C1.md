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

0.8217654694937253

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The changes remove the unavailable pydensecrf dependency, add the missing imports, load the official sample submission instead of a non‑existent file, replace the CRF step with a no‑op that simply returns the original mask, and correctly write the resulting DataFrame to a CSV named submission.csv. This fixes all runtime errors and ensures a valid submission file is produced while preserving the original logic flow.'
- What this solution (achieved 0.5221) has done: 'We compute a simple baseline mask by averaging all training masks and applying this average (threshold 0.5) to every test image. This replaces the empty placeholders in the original submission, keeps the existing CRF stub unchanged, and writes a valid `submission.csv`. The change is minimal, adds only the necessary imports and baseline computation, and is expected to raise the score from 0.0 toward the target.'
- What this solution (achieved 0.0599) has done: 'I replace the single average mask with a per‑image Otsu‑based threshold, keeping the overall pipeline unchanged. By computing a mask from each test image’s own pixel intensities we obtain more realistic segmentations, which should raise the mean‑average‑precision from ~0.52 toward the target while preserving the existing structure and placeholder CRF.'
- What this solution (achieved 0.0) has done: 'The changes replace the overly‑aggressive per‑image Otsu masks with the previously successful global‑average baseline mask, then apply light morphological smoothing (binary closing and removal of tiny objects) to reduce spurious predictions. This keeps the original pipeline structure while improving mask quality, which is expected to raise the mean‑average‑precision toward the target score. Additionally, the required morphology imports are added.'
- What this solution (achieved 0.0) has done: 'I fix the error caused by using the outdated `selem` argument in `binary_closing` and replace the simplistic global‑average mask with a per‑image Otsu‑based threshold (followed by light morphological cleanup). This keeps the original pipeline while providing a more meaningful mask for each test image, allowing the submission to run and moving the score toward the target.'
- What this solution (achieved 0.5221) has done: 'The fix addresses the crash caused by 4‑channel PNGs by discarding the alpha channel before converting to gray, and improves the prediction quality by intersecting the per‑image Otsu mask with the global‑average baseline mask (which performed well earlier). This small logical change keeps the original pipeline intact while raising the expected score toward the target.'
- What this solution (achieved 0.0805) has done: 'I replace the overly‑restrictive intersection of the Otsu mask with the global‑average baseline by a more permissive strategy: use the Otsu mask alone, but fall back to the baseline when Otsu finds no foreground. This should increase recall and lift the mean‑average‑precision toward the target while keeping the rest of the pipeline unchanged. Minor tweaks to the morphological closing radius are also added to smooth masks without over‑pruning.'
- What this solution (achieved 0.0873) has done: 'I keep the overall pipeline unchanged but improve the mask generation by combining the per‑image Otsu mask with the global‑average baseline (union) instead of using only one of them, and I relax the small‑object removal threshold to keep more plausible salt regions. These minimal adjustments are expected to raise the mean‑average‑precision toward the target while preserving the original logic.'
- What this solution (achieved 0.0873) has done: 'I add a depth‑conditioned baseline mask: for each test image I look up its depth, select the nearest‑depth training masks, average them and threshold to obtain a more relevant mask than the global average. This mask is then combined with the Otsu mask (or falls back to the global baseline) before the tiny‑object cleanup and CSV writing. The changes are confined to the mask‑loading cell and the prediction loop, preserving the original pipeline while providing a stronger, depth‑aware prior that should raise the score toward the target.'
- What this solution (achieved 0.5221) has done: 'I replace the per‑image Otsu step with a depth‑conditioned mask only (using a larger neighbour pool) and make the morphological post‑processing milder. This keeps the overall pipeline intact while providing a stronger prior that should raise the mean‑average‑precision toward the target score.'
- What this solution (achieved 0.0805) has done: 'I raise the score by strengthening the mask generation: increase the depth‑neighbor pool to 300, add an Otsu‑based mask for each test image and union it with the depth‑averaged mask, and apply slightly stronger morphological smoothing (disk 3) while removing very small objects (min size 20). These tweaks keep the original pipeline intact (same CRF stub, same data handling) but provide richer, more accurate predictions, moving the metric closer to the target.'
- What this solution (achieved 0.5221) has done: 'I tighten the mask generation by intersecting the depth‑based prior with the Otsu threshold (instead of a permissive union) and only fall back to the depth mask when the intersection is empty. This reduces many false positives that were hurting the mean‑average‑precision, while keeping the same data loading, depth‑neighbor logic, and CRF stub. Small adjustments to the morphological cleaning parameters keep the processing lightweight and stay within the original pipeline design.'
- What this solution (achieved 0.0805) has done: 'I adjust the mask‑combination step to use a union of the depth‑based prior and the Otsu mask (instead of the stricter intersection) and make the post‑processing a bit more permissive: a slightly larger closing disk and a lower `min_size` for small‑object removal. These modest changes are expected to increase recall and lift the MAP score toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.5221) has done: 'I simplify the mask creation to rely solely on the depth‑based prior (which previously gave ~0.52 MAP) and drop the per‑image Otsu step that was hurting precision. A small fallback to the global baseline is kept for images where the depth‑average mask is empty, and the morphological cleanup is made a bit milder (larger closing disk, keep very small objects) to retain more true salt regions. These minimal adjustments are expected to raise the score substantially toward the target while preserving the original pipeline structure.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.color import gray2rgb, rgb2gray
from skimage.filters import threshold_otsu  # kept for possible future tweaks
from skimage.morphology import (
    binary_closing,
    disk,
    remove_small_objects,
)  # used for mask refinement




## === cell 1
def rle_decode(rle_mask):
    """
    rle_mask: run‑length as string formatted "start length ..."
    Returns a (101,101) binary mask (1 – salt, 0 – background).
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
    im: binary numpy array (1 – mask, 0 – background)
    Returns run‑length encoding as a space‑separated string.
    """
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
submission_path = "../input/tgs-salt-identification-challenge/sample_submission.csv"
df = pd.read_csv(submission_path)

train_mask_dir = "../input/tgs-salt-identification-challenge/train/masks/"
mask_files = [f for f in os.listdir(train_mask_dir) if f.lower().endswith(".png")]
num_masks = len(mask_files)
avg_mask = np.zeros((101, 101), dtype=np.float32)

train_masks_dict = {}  # id -> binary mask (uint8)
train_ids = []  # list of ids in the same order as train_masks_array
train_masks_array = []  # list of masks as 2‑D arrays for fast averaging

for mf in tqdm(mask_files, desc="Loading training masks"):
    mask_path = os.path.join(train_mask_dir, mf)
    mask_img = imread(mask_path, as_gray=True)
    binary = (mask_img > 127).astype(np.uint8)
    avg_mask += binary

    img_id = os.path.splitext(mf)[0]
    train_masks_dict[img_id] = binary
    train_ids.append(img_id)
    train_masks_array.append(binary)

avg_mask /= max(num_masks, 1)  # safeguard division
baseline_mask = (avg_mask >= 0.5).astype(np.uint8)  # fallback global baseline

depths_path = "../input/tgs-salt-identification-challenge/depths.csv"
depths_df = pd.read_csv(depths_path)  # columns: id, z
depth_series = depths_df.set_index("id")["z"]

train_depths = np.array([depth_series.get(tid, np.nan) for tid in train_ids])




## === cell 4
def crf(original_image, mask_img):
    """
    Placeholder CRF function. The original code relied on pydensecrf,
    which is not available in the environment. To keep the pipeline
    functional we simply return the input mask unchanged.
    """
    if mask_img.ndim < 3:
        mask_img = gray2rgb(mask_img)
    return mask_img[:, :, 0] if mask_img.ndim == 3 else mask_img




## === cell 5
test_path = "../input/tgs-salt-identification-challenge/test/images/"

K_NEIGHBORS = 300  # enlarged neighbor pool for a richer depth prior

for idx in tqdm(range(df.shape[0]), desc="Refining masks"):
    img_id = df.loc[idx, "id"]
    img_path = f"{test_path}{img_id}.png"
    try:
        orig_img = imread(img_path)  # shape (101,101) or (101,101,3) or (101,101,4)
    except FileNotFoundError:
        continue

    if orig_img.ndim == 3 and orig_img.shape[-1] == 4:
        orig_img = orig_img[..., :3]

    test_depth = depth_series.get(img_id, np.nan)
    if np.isnan(test_depth):
        depth_mask = baseline_mask.copy()
    else:
        diffs = np.abs(train_depths - test_depth)
        nearest_idx = np.argpartition(diffs, K_NEIGHBORS)[:K_NEIGHBORS]
        selected_masks = np.stack([train_masks_array[i] for i in nearest_idx], axis=0)
        depth_avg = selected_masks.mean(axis=0)
        depth_mask = (depth_avg >= 0.5).astype(np.uint8)

    if depth_mask.sum() == 0:
        mask_img = baseline_mask.copy()
    else:
        mask_img = depth_mask

    mask_img = binary_closing(mask_img, footprint=disk(5)).astype(np.uint8)
    mask_img = remove_small_objects(mask_img.astype(bool), min_size=1).astype(np.uint8)

    refined_mask = crf(orig_img, mask_img)

    df.loc[idx, "rle_mask"] = rle_encode(refined_mask.astype(np.uint8))



## === cell 6
df.to_csv("submission.csv", index=False)
