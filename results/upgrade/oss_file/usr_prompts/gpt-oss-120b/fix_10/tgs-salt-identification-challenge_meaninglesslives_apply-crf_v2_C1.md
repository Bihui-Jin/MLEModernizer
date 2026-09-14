# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    closing,
    disk,
)
from skimage.measure import label, regionprops
from skimage import exposure  # added for adaptive contrast equalization

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
except ModuleNotFoundError:
    dcrf = None




## === cell 1
def rle_decode(mask_rle):
    """Decode a run‑length encoded mask string into a 101×101 binary mask."""
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101)


def rle_encode(img):
    """Encode a 101×101 binary mask to run‑length encoding string."""
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 2
train_img_path = "../input/tgs-salt-identification-challenge/train/images/"
train_mask_path = "../input/tgs-salt-identification-challenge/train/masks/"
test_img_path = "../input/tgs-salt-identification-challenge/test/images/"
sample_sub_path = "../input/sample_submission.csv"

submission_df = pd.read_csv(sample_sub_path)




## === cell 3
def _contrast_stretch(image, low=2, high=98):
    """
    Simple contrast stretching based on low/high percentiles.
    Works on 0‑255 uint8 images.
    """
    if image.ndim == 3:
        out = np.empty_like(image, dtype=np.float32)
        for c in range(3):
            channel = image[..., c]
            p_low, p_high = np.percentile(channel, (low, high))
            if p_high > p_low:
                out[..., c] = np.clip((channel - p_low) / (p_high - p_low), 0, 1)
            else:
                out[..., c] = channel / 255.0
        return out
    else:
        p_low, p_high = np.percentile(image, (low, high))
        if p_high > p_low:
            return np.clip((image - p_low) / (p_high - p_low), 0, 1)
        else:
            return image / 255.0


def crf(original_image, annotated_image):
    """Fallback CRF – returns the mask unchanged if pydensecrf is unavailable."""
    if dcrf is None:
        return annotated_image
    if annotated_image.ndim < 3:
        annotated_image = gray2rgb(annotated_image)
    annotated_label = (
        annotated_image[:, :, 0]
        + (annotated_image[:, :, 1] << 8)
        + (annotated_image[:, :, 2] << 16)
    )
    colors, labels = np.unique(annotated_label, return_inverse=True)
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


def _process_one(image, sigma, min_size, perc):
    """Core pipeline – contrast stretch, adaptive equalisation,
    Gaussian blur, percentile‑based threshold (fallback to Otsu), and morphology."""
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]

    img_cs = _contrast_stretch(image)
    img_eq = exposure.equalize_adapthist(img_cs, clip_limit=0.03)
    gray = rgb2gray(img_eq) if image.ndim == 3 else img_eq

    gray_blur = gaussian(gray, sigma=sigma)

    if perc is not None:
        thresh = np.percentile(gray_blur, perc)
    else:
        thresh = threshold_otsu(gray_blur)

    mask = (gray_blur > thresh).astype(np.uint8)

    mask = remove_small_objects(mask.astype(bool), min_size=min_size).astype(np.uint8)
    mask = remove_small_holes(mask.astype(bool), area_threshold=min_size).astype(
        np.uint8
    )

    labeled = label(mask)
    if labeled.max() > 0:
        regions = regionprops(labeled)
        largest_region = max(regions, key=lambda r: r.area)
        mask = (labeled == largest_region.label).astype(np.uint8)

    mask = closing(mask, disk(3)).astype(np.uint8)

    mask = crf(image, mask)
    return mask.astype(np.uint8)


def process_mask(image, sigma=1.0, min_size=100, perc=80):
    """
    Apply the pipeline to the original image and its horizontal flip,
    then merge the two masks with a logical OR.
    """
    mask_orig = _process_one(image, sigma, min_size, perc)
    image_flipped = np.fliplr(image)
    mask_flip = _process_one(image_flipped, sigma, min_size, perc)
    mask_flip = np.fliplr(mask_flip)  # revert flip
    combined = np.logical_or(mask_orig, mask_flip).astype(np.uint8)
    return combined


def iou_score(pred, true):
    """Intersection‑over‑Union for two binary masks."""
    if pred.sum() == 0 and true.sum() == 0:
        return 1.0
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return intersection / union if union != 0 else 0.0




## === cell 4
np.random.seed(42)

train_df = pd.read_csv("../input/tgs-salt-identification-challenge/train.csv")
val_ids = train_df["id"].values

subset_size = min(300, len(val_ids))
subset_ids = np.random.choice(val_ids, size=subset_size, replace=False)

candidate_sigmas = [0.5, 1.0, 1.5, 2.0]
candidate_min_sizes = [50, 100, 150, 200]
candidate_percents = [70, 75, 80, 85, 90]

best_mean_iou = -1.0
best_params = {"sigma": 1.0, "min_size": 100, "perc": 80}

for sigma in candidate_sigmas:
    for min_sz in candidate_min_sizes:
        for perc in candidate_percents:
            ious = []
            for vid in tqdm(
                subset_ids,
                desc=f"Eval sigma={sigma} min={min_sz} perc={perc}",
                leave=False,
            ):
                img_path = f"{train_img_path}{vid}.png"
                mask_path = f"{train_mask_path}{vid}.png"
                img = imread(img_path)
                true_mask = imread(mask_path)
                true_binary = (true_mask > 127).astype(np.uint8)
                pred_binary = process_mask(img, sigma=sigma, min_size=min_sz, perc=perc)
                ious.append(iou_score(pred_binary, true_binary))
            mean_iou = np.mean(ious)
            if mean_iou > best_mean_iou:
                best_mean_iou = mean_iou
                best_params = {"sigma": sigma, "min_size": min_sz, "perc": perc}

print(
    f"Selected parameters -> sigma: {best_params['sigma']}, "
    f"min_size: {best_params['min_size']}, perc: {best_params['perc']}"
)
print(f"Validation mean IoU (proxy for competition metric): {best_mean_iou:.4f}")



## === cell 5
sigma_opt = best_params["sigma"]
min_sz_opt = best_params["min_size"]
perc_opt = best_params["perc"]

for idx in tqdm(range(submission_df.shape[0]), desc="Processing test masks"):
    img_id = submission_df.loc[idx, "id"]
    img_path = f"{test_img_path}{img_id}.png"
    img = imread(img_path)
    pred_mask = process_mask(img, sigma=sigma_opt, min_size=min_sz_opt, perc=perc_opt)
    submission_df.at[idx, "rle_mask"] = rle_encode(pred_mask)



## === cell 6
output_path = "crf_correction.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")
