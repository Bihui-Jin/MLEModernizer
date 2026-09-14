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

0.8018520121159664

# 6. Current score

0.0613

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I correct the file paths so the script can locate the sample submission and test images (using the Kaggle `/kaggle/input` root), ensure the dataframe is loaded before it is used, and keep the rest of the logic unchanged. This fixes the `FileNotFoundError` and the `NameError`, allowing the code to run end‑to‑end and produce a valid `submission.csv` file.'
- What this solution (achieved 0.0658) has done: 'I fix the image‑loading bug by handling RGBA PNGs (dropping the alpha channel) and add a tiny post‑processing step (fill holes and remove very small objects) to make the Otsu masks a bit more realistic, which should raise the score toward the target. All other logic and file paths remain unchanged, and the script still writes a proper `submission.csv`.'
- What this solution (achieved 0.1177) has done: 'I added a focused post‑processing step that keeps only the largest connected salt region (the competition masks are typically a single contiguous object) and strengthened the morphological cleaning (larger min‑size, closing). These adjustments stay within the original workflow but should raise the average‑precision score toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I add a simple nearest‑depth mask lookup: for each test image I load the training masks, match the test depth to the closest training depth, and combine that mask with the Otsu‑based mask (logical OR) before optional CRF. This leverages the known train labels and should raise the mean‑average‑precision toward the target while keeping the original workflow intact. The script now loads the training data, builds a depth‑to‑mask map, and uses it during processing, then writes a proper `submission.csv`.'
- What this solution (achieved 0.0463) has done: 'I fixed the runtime crashes by making the RLE decoder tolerant of missing/empty masks, added safe handling for test depths that are not present in the depths file, and corrected the Otsu thresholding logic (salt is darker, so we now use gray < threshold). These changes keep the original workflow intact while producing a valid submission.csv and should improve the evaluation score toward the target.'
- What this solution (achieved 0.0378) has done: 'I tighten the mask combination by using the intersection (logical AND) of the Otsu mask and the nearest‑depth training mask, which reduces false positives and should raise the mean‑average‑precision toward the target. I also lower the small‑object removal threshold (min_size = 100) to keep legitimate salt regions that might have been discarded. These minimal, targeted tweaks keep the overall pipeline unchanged while improving the quality of the submitted masks.'
- What this solution (achieved 0.0517) has done: 'I loosen the overly strict mask combination (switching from logical AND to logical OR) and reduce the small‑object removal threshold in the Otsu‑based mask, which together should raise recall without dramatically increasing false positives. These minimal tweaks keep the overall pipeline, CRF step and file handling unchanged while moving the score toward the target.'
- What this solution (achieved 0.052) has done: 'I adjust the Otsu‑based mask to use the correct “brighter‑salt” condition and increase the small‑object removal size, then combine the Otsu mask with the nearest‑depth training mask using an **intersection** (logical AND) rather than a union. These minimal tweaks keep the overall workflow unchanged while expectedly raising precision and moving the score much closer to the target.'
- What this solution (achieved 0.0613) has done: 'The changes lower the small‑object removal threshold, use a logical OR instead of AND to merge the Otsu mask with the nearest‑depth training mask (boosting recall), and fill any holes after merging before keeping the largest component. These minimal tweaks keep the overall workflow intact while expectedly raising the public‑score toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.color import gray2rgb, rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing, square
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
except ImportError:
    dcrf = None
    unary_from_labels = None
    create_pairwise_bilateral = None


def rle_decode(rle_mask):
    """Decode an RLE string to a binary mask (101x101). Handles NaN / empty strings."""
    if not isinstance(rle_mask, str) or rle_mask.strip() == "":
        return np.zeros((101, 101), dtype=np.uint8)
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101)


def rle_encode(im):
    """Encode a binary mask (101x101) to RLE string."""
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def keep_largest_component(mask):
    labeled = label(mask)
    if labeled.max() == 0:
        return mask
    regions = regionprops(labeled)
    largest_region = max(regions, key=lambda r: r.area)
    return (labeled == largest_region.label).astype(np.uint8)


def simple_mask(image):
    """Create a binary salt mask using Otsu.
    Salt appears brighter in this dataset, so we use > threshold.
    Reduce min_size to retain more plausible salt regions."""
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]  # drop alpha channel
    if image.ndim == 3:
        gray = rgb2gray(image)
    else:
        gray = image.astype(float) / 255.0
    thresh = threshold_otsu(gray)
    mask = (gray > thresh).astype(np.uint8)
    mask_filled = binary_fill_holes(mask).astype(np.uint8)
    mask_clean = remove_small_objects(mask_filled.astype(bool), min_size=100).astype(
        np.uint8
    )
    mask_closed = closing(mask_clean, square(3)).astype(np.uint8)
    return keep_largest_component(mask_closed)


def crf(original_image, mask_img):
    if dcrf is None:
        return mask_img
    if mask_img.ndim < 3:
        mask_img = gray2rgb(mask_img)
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


def load_train_masks(train_csv_path):
    df = pd.read_csv(train_csv_path)
    masks = [rle_decode(rle) for rle in df["rle_mask"]]
    return masks, df["id"].values


def build_depth_lookup(depths_path, ids):
    df = pd.read_csv(depths_path)
    df = df.set_index("id")
    depths = df.reindex(ids)["z"].values
    return depths


def nearest_train_mask(test_depth, train_depths, train_masks):
    idx = np.argmin(np.abs(train_depths - test_depth))
    return train_masks[idx]




## === cell 1
BASE_PATH = os.path.join("/", "kaggle", "input", "tgs-salt-identification-challenge")
sample_submission_path = os.path.join(BASE_PATH, "sample_submission.csv")
df = pd.read_csv(sample_submission_path)

test_path = os.path.join(BASE_PATH, "test", "images")
depths_path = os.path.join(BASE_PATH, "depths.csv")
train_csv_path = os.path.join(BASE_PATH, "train.csv")

train_masks, train_ids = load_train_masks(train_csv_path)
train_depths = build_depth_lookup(depths_path, train_ids)

test_depths_df = pd.read_csv(depths_path).set_index("id")
median_train_depth = np.median(train_depths)




## === cell 2
for i in tqdm(range(df.shape[0]), desc="Processing masks"):
    img_id = df.loc[i, "id"]
    img_path = os.path.join(test_path, f"{img_id}.png")
    orig_img = imread(img_path)

    otsu_mask = simple_mask(orig_img)

    try:
        test_depth = test_depths_df.loc[img_id, "z"]
    except KeyError:
        test_depth = median_train_depth

    nearest_mask = nearest_train_mask(test_depth, train_depths, train_masks)

    combined_mask = np.logical_or(otsu_mask, nearest_mask).astype(np.uint8)
    combined_mask = binary_fill_holes(combined_mask).astype(np.uint8)
    combined_mask = keep_largest_component(combined_mask)

    refined_mask = crf(orig_img, combined_mask)

    df.loc[i, "rle_mask"] = rle_encode(refined_mask)




## === cell 3
output_path = "submission.csv"
df.to_csv(output_path, index=False)
