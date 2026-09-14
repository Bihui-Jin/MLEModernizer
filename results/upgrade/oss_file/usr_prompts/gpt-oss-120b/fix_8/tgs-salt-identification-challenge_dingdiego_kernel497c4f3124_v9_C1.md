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

0.8144958892254429

# 6. Current score

0.0617

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The script now safely imports optional packages, uses a fallback CRF implementation when `pydensecrf` is unavailable, correctly loads the sample submission, and ensures all required libraries (`pandas`, `tqdm`, `matplotlib`, etc.) are imported so the notebook runs without errors and writes a valid `crf_correction.csv` file.'
- What this solution (achieved 0.0305) has done: 'I add a lightweight fallback segmentation that creates a binary mask from the raw test image using a simple intensity‑mean threshold. If the original RLE mask is empty or trivially small, the script replace it with this mask before applying the existing CRF refinement. This keeps the core CRF‑based logic intact while providing a much more informative prediction than the placeholder “1 1” masks, moving the score from 0 toward the target.'
- What this solution (achieved 0.0377) has done: 'Implemented a more informative mask generation: added an Otsu‑based threshold and removed tiny isolated regions to produce cleaner binary masks. Imported the needed `remove_small_objects` routine and updated the `simple_mask` function accordingly. These changes keep the overall CRF‑based pipeline intact while delivering masks that better resemble the true salt regions, moving the validation score toward the target.'
- What this solution (achieved 0.0437) has done: 'The changes add a few morphological steps to the fallback mask generation: keeping only the largest connected component, filling interior holes, and tightening small‑object removal. These operations clean the Otsu‑based masks, which should raise the IoU‑based score toward the target while preserving the existing CRF pipeline.'
- What this solution (achieved 0.0) has done: 'The update forces the pipeline to always generate a mask from the raw test image (ignoring the placeholder masks), lowers the small‑object removal threshold, and adds a mild dilation step to better capture salt regions while preserving the existing CRF refinement.'
- What this solution (achieved 0.0481) has done: 'The fix updates the morphological dilation call to use the current `skimage` API (`footprint=` instead of the removed `selem=` argument). This resolves the TypeError that halted execution, allowing mask generation, CRF refinement, and CSV submission to run, moving the solution from a score of 0.0 toward the target.'
- What this solution (achieved 0.0617) has done: 'I keep the overall pipeline unchanged but improve the fallback mask generation: remove the “keep only the largest component” step (which drops many true salt regions), add a mild Gaussian blur before Otsu to stabilise the threshold, and replace the single dilation with a closing operation that smooths edges while preserving all sizable components. These small, targeted tweaks are expected to raise the IoU‑based score toward the target without altering the core CRF logic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.morphology import (
    remove_small_objects,
    disk,
    binary_dilation,
    binary_closing,
)
from skimage.measure import label
from scipy.ndimage import binary_fill_holes, gaussian_filter

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral

    HAVE_DENSECRF = True
except ModuleNotFoundError:
    HAVE_DENSECRF = False




## === cell 1
def rle_decode(rle_mask):
    """
    Decode a run-length encoded mask into a 101x101 binary numpy array.
    """
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101)




## === cell 2
df = pd.read_csv("../input/sample_submission.csv")
print(f"Loaded {len(df)} rows from sample submission.")




## === cell 3
def crf(original_image, mask_img):
    """
    Apply DenseCRF to refine a binary mask.
    If pydensecrf is not installed, fall back to returning the original mask.
    """
    if not HAVE_DENSECRF:
        return mask_img

    if mask_img.ndim < 3:
        mask_img = gray2rgb(mask_img)

    annotated_label = (
        mask_img[:, :, 0] + (mask_img[:, :, 1] << 8) + (mask_img[:, :, 2] << 16)
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




## === cell 4
test_path = "../input/tgs-salt-identification-challenge/test/images/"




## === cell 5
def rle_encode(im):
    """
    Encode a binary mask (numpy array) to run-length encoding string.
    """
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 6
def otsu_threshold(gray):
    """
    Compute Otsu's threshold for a grayscale image.
    """
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)

    sumB = 0.0
    wB = 0.0
    max_between = 0.0
    thresh = 0

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between > max_between:
            max_between = between
            thresh = t
    return thresh


def simple_mask(image):
    """
    Generate a binary mask from the raw image using a blurred Otsu threshold,
    remove tiny regions (min_size=30), keep all remaining components, apply
    a closing operation to smooth edges, and fill holes.
    """
    if image.ndim == 3:
        gray = image.mean(axis=2).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)

    blurred = gaussian_filter(gray, sigma=1)

    thresh = otsu_threshold(blurred)
    mask = (blurred > thresh).astype(np.uint8)

    mask = remove_small_objects(mask.astype(bool), min_size=30)

    mask = binary_closing(mask, footprint=disk(3))

    mask = binary_fill_holes(mask)

    return mask.astype(np.uint8)




## === cell 7
for i in tqdm(range(df.shape[0]), desc="Refining masks"):
    img_path = f"{test_path}{df.loc[i, 'id']}.png"
    if not os.path.exists(img_path):
        continue  # skip missing images
    orig_img = imread(img_path)

    decoded_mask = simple_mask(orig_img)

    refined_mask = crf(orig_img, decoded_mask)
    df.loc[i, "rle_mask"] = rle_encode(refined_mask)




## === cell 8
df.to_csv("crf_correction.csv", index=False)
print("Submission saved to crf_correction.csv")
