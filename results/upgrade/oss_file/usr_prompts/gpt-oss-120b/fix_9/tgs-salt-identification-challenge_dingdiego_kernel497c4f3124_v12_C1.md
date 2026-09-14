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

0.772669839896148

# 6. Current score

0.042

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'We remove the unavailable `pydensecrf` dependency, ensure all required libraries (`pandas`, `tqdm`, `matplotlib`) are imported, use the provided `sample_submission.csv` as the starting file, and replace the CRF step with a no‑op that simply returns the original mask. This fixes the runtime errors, guarantees a valid `crf_correction.csv` submission, and keeps the core logic unchanged (the CRF step is effectively bypassed).'
- What this solution (achieved 0.0097) has done: 'I replace the placeholder masks from the sample submission with a simple image‑based threshold segmentation before the (optional) CRF step. This keeps the overall pipeline intact, adds a lightweight mask generation that should give a non‑zero score, and still writes the final CSV file.'
- What this solution (achieved 0.0118) has done: 'I replace the naïve mean‑based threshold with Otsu’s adaptive threshold (and fall back to the mean if Otsu fails). This small change keeps the overall pipeline intact while producing masks that better match the true salt regions, which should raise the mean‑average‑precision toward the target score. The rest of the code—including the optional CRF step and CSV handling—remains unchanged.'
- What this solution (achieved 0.0) has done: 'Implemented a lightweight preprocessing step before thresholding: Gaussian blur to reduce noise, followed by Otsu (fallback to mean) and small‑object removal plus binary closing to clean the mask. Added handling for empty masks in the RLE encoder to ensure valid output. Imported the necessary skimage utilities while preserving the original pipeline and CRF fallback. These modest refinements are expected to raise the mean‑average‑precision toward the target score without altering the core modelling logic.'
- What this solution (achieved 0.0) has done: 'I fixed the runtime error by replacing the outdated `selem` argument with the correct `footprint` parameter for `binary_closing`. I also added a lightweight hole‑filling step (`binary_fill_holes`) to slightly improve mask quality without altering the core pipeline. These minimal changes unblock execution, produce a valid `crf_correction.csv`, and should raise the score from zero toward the target.'
- What this solution (achieved 0.0738) has done: 'I fixed the import error by pulling `binary_fill_holes` from `scipy.ndimage` and added the missing `tqdm` import (now guaranteed to load). I also improved mask quality modestly by keeping only the largest connected component after morphological cleaning, which typically matches the single salt region and should lift the mean‑average‑precision toward the target while preserving the original workflow. The script now runs end‑to‑end and writes a valid `crf_correction.csv` submission.'
- What this solution (achieved 0.0422) has done: 'I improve the preprocessing pipeline by enhancing contrast with adaptive histogram equalization, using a stronger Gaussian blur, adding a small opening step, increasing the size filter for small objects, and ensuring holes are filled after keeping the largest component. These changes keep the overall workflow intact while producing cleaner masks, which should raise the mean‑average‑precision toward the target score.'
- What this solution (achieved 0.042) has done: 'I load the image depth information and use the median depth to adapt the Otsu threshold per‑image (deeper images get a slightly lower threshold to capture more salt). I also tighten the morphological cleaning by raising the minimum object size, which helps suppress spurious noise while keeping the core pipeline unchanged. These modest, data‑driven tweaks are expected to raise the mean‑average‑precision toward the target without altering the overall workflow.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.color import gray2rgb
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import (
    remove_small_objects,
    binary_closing,
    binary_opening,
    disk,
)
from skimage.exposure import equalize_adapthist
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from tqdm import tqdm

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral

    _HAS_CRf = True
except ImportError:  # pydensecrf not installed in the environment
    _HAS_CRf = False




## === cell 1
def rle_decode(rle_mask):
    """
    Decode a run‑length encoded mask string into a 101x101 binary array.
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
try:
    df = pd.read_csv("../input/baseline-v4/submission.csv")
except FileNotFoundError:
    df = pd.read_csv("../input/sample_submission.csv")
assert {"id", "rle_mask"}.issubset(
    df.columns
), "Submission file must contain 'id' and 'rle_mask' columns."

depth_paths = [
    "../input/depths.csv",
    "../input/tgs-salt-identification-challenge/depths.csv",
    "../input/train/depths.csv",
    "../input/input/depths.csv",
]
depth_df = None
for p in depth_paths:
    try:
        depth_df = pd.read_csv(p)
        break
    except FileNotFoundError:
        continue
if depth_df is None:
    depth_df = pd.DataFrame(columns=["id", "z"])

depth_map = dict(zip(depth_df["id"].astype(str), depth_df["z"]))
if len(depth_df) > 0:
    median_depth = depth_df["z"].median()
else:
    median_depth = 1.0  # neutral fallback




## === cell 3
def crf(original_image, mask_img):
    """
    Apply DenseCRF to refine a binary mask.
    If pydensecrf is unavailable, return the original mask unchanged.
    """
    if not _HAS_CRf:
        return mask_img.astype(np.uint8)

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
    return MAP.reshape((original_image.shape[0], original_image.shape[1])).astype(
        np.uint8
    )




## === cell 4
test_path = "../input/tgs-salt-identification-challenge/test/images/"




## === cell 5
def _visual_check():
    n_imgs = 3
    i = np.random.randint(len(df))
    j = 1
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(wspace=0.2, hspace=0.1)
    while j <= n_imgs and i < len(df):
        rle = df.loc[i, "rle_mask"]
        if isinstance(rle, str) and rle.strip():
            decoded_mask = rle_decode(rle)
            orig_img = imread(test_path + df.loc[i, "id"] + ".png")
            crf_out = crf(orig_img, decoded_mask)
            plt.subplot(n_imgs, 4, 4 * j - 3)
            plt.imshow(orig_img, cmap="gray")
            plt.title("Original")
            plt.subplot(n_imgs, 4, 4 * j - 2)
            plt.imshow(decoded_mask, cmap="gray")
            plt.title("Decoded")
            plt.subplot(n_imgs, 4, 4 * j - 1)
            plt.imshow(crf_out, cmap="gray")
            plt.title("CRF")
            j += 1
        i += 1
    plt.show()




## === cell 6
def rle_encode(im):
    """
    Encode a binary mask (numpy array) to run‑length format string.
    Returns an empty string for a completely empty mask.
    """
    if im.sum() == 0:
        return ""
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 7
for idx in tqdm(range(df.shape[0]), desc="Refining masks"):
    img_id = df.loc[idx, "id"]
    orig_img = imread(test_path + img_id + ".png")
    if orig_img.ndim == 3:
        orig_img = orig_img[..., 0]

    contrast_img = equalize_adapthist(orig_img / 255.0) * 255.0

    blurred = gaussian(contrast_img, sigma=2.0, preserve_range=True)

    try:
        thresh = threshold_otsu(blurred)
    except Exception:
        thresh = blurred.mean()

    depth = depth_map.get(str(img_id), median_depth)
    scale = median_depth / depth if depth != 0 else 1.0
    thresh = np.clip(thresh * scale, blurred.min(), blurred.max())

    simple_mask = (blurred > thresh).astype(np.uint8)

    opened = binary_opening(simple_mask.astype(bool), footprint=disk(1))
    cleaned_mask = remove_small_objects(opened, min_size=200)  # raised min_size
    cleaned_mask = binary_closing(cleaned_mask, footprint=disk(2))
    cleaned_mask = binary_fill_holes(cleaned_mask)

    lbl = label(cleaned_mask)
    if lbl.max() > 0:
        regions = regionprops(lbl)
        largest = max(regions, key=lambda r: r.area)
        cleaned_mask = (lbl == largest.label).astype(np.uint8)

    refined_mask = crf(orig_img, cleaned_mask)

    df.at[idx, "rle_mask"] = rle_encode(refined_mask)




## === cell 8
df.to_csv("crf_correction.csv", index=False)
