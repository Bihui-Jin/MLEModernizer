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

0.7702985720467324

# 6. Current score

0.0832

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The fix adds safe handling for the missing pydensecrf package, ensures pandas and other imports run, reads the provided sample_submission.csv as the base dataframe, replaces the unavailable CRF with a simple pass‑through (or optional Gaussian smoothing), generates a basic mask via Otsu thresholding when the original RLE is missing, and finally writes a valid crf_correction.csv submission file.'
- What this solution (achieved 0.0) has done: 'The fix replaces the placeholder masks from the sample submission with a simple but more realistic segmentation: each test image is converted to grayscale, Otsu‑thresholded, and tiny isolated regions are removed. This produces much more meaningful masks while keeping the original CRF fallback unchanged, moving the validation score toward the target without altering the core model logic. The script now consistently writes a valid `.csv` submission.'
- What this solution (achieved 0.0607) has done: 'I adjust the image loading to correctly handle RGBA PNGs by dropping the alpha channel before converting to grayscale, and I add a small hole‑filling step to improve mask quality. These changes fix the runtime error and should raise the validation score toward the target while keeping the original workflow intact.'
- What this solution (achieved 0.066) has done: 'To boost the validation score while keeping the original workflow, we add a light Gaussian blur, keep only the largest connected component (the main salt region), and apply a modest morphological closing. These steps improve mask quality without changing the core Otsu‑thresholding or CRF fallback logic.'
- What this solution (achieved 0.0832) has done: 'I add a lightweight validation step that tries a few sensible values for the Gaussian blur sigma and the minimum object size, picks the combination that gives the highest mean IoU on a small held‑out subset of the training data, and then uses those parameters to generate the final test masks. The core Otsu‑thresholding, morphological cleaning and optional CRF fallback remain unchanged, so the overall logic is preserved while improving the segmentation quality and moving the score toward the target.'

# 9. Code solution

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


def process_mask(image, sigma=1.0, min_size=100):
    """Apply the original Otsu‑threshold + morphology pipeline with tunable params."""
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    gray = rgb2gray(image) if image.ndim == 3 else image / 255.0
    gray_blur = gaussian(gray, sigma=sigma)
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


def iou_score(pred, true):
    """Intersection‑over‑Union for two binary masks."""
    if pred.sum() == 0 and true.sum() == 0:
        return 1.0
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return intersection / union if union != 0 else 0.0




## === cell 4
train_df = pd.read_csv("../input/tgs-salt-identification-challenge/train.csv")
val_ids = train_df["id"].values[:200]

candidate_sigmas = [0.5, 1.0, 1.5]
candidate_min_sizes = [50, 100, 200]

best_mean_iou = -1.0
best_params = {"sigma": 1.0, "min_size": 100}

for sigma in candidate_sigmas:
    for min_sz in candidate_min_sizes:
        ious = []
        for vid in tqdm(val_ids, desc=f"Eval sigma={sigma} min={min_sz}", leave=False):
            img_path = f"{train_img_path}{vid}.png"
            mask_path = f"{train_mask_path}{vid}.png"
            img = imread(img_path)
            true_mask = imread(mask_path)
            true_binary = (true_mask > 127).astype(np.uint8)
            pred_binary = process_mask(img, sigma=sigma, min_size=min_sz)
            ious.append(iou_score(pred_binary, true_binary))
        mean_iou = np.mean(ious)
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_params = {"sigma": sigma, "min_size": min_sz}

print(
    f"Selected parameters -> sigma: {best_params['sigma']}, min_size: {best_params['min_size']}"
)
print(f"Validation mean IoU (proxy for competition metric): {best_mean_iou:.4f}")




## === cell 5
sigma_opt = best_params["sigma"]
min_sz_opt = best_params["min_size"]

for idx in tqdm(range(submission_df.shape[0]), desc="Processing test masks"):
    img_id = submission_df.loc[idx, "id"]
    img_path = f"{test_img_path}{img_id}.png"
    img = imread(img_path)
    pred_mask = process_mask(img, sigma=sigma_opt, min_size=min_sz_opt)
    submission_df.at[idx, "rle_mask"] = rle_encode(pred_mask)



## === cell 6
output_path = "crf_correction.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")
