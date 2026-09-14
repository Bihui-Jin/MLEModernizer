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

0.8218693206404148

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I add safe imports, fallback when pydensecrf is missing, correctly load the sample submission, import the missing libraries, and ensure the CRF function simply returns the original mask when the CRF library is unavailable. This fixes the runtime errors, creates a proper DataFrame, processes each mask, and writes a valid crf_correction.csv submission file.'
- What this solution (achieved 0.0) has done: 'Implemented robust path discovery using absolute Kaggle directories, added a fallback that builds the submission DataFrame from test image IDs when no sample file exists, and introduced an Otsu‑based adaptive threshold for the simple mask generation. These changes eliminate the FileNotFound and NameError issues, ensure a valid `crf_correction.csv` is written, and modestly improve mask quality toward the target score while keeping the original workflow intact.'
- What this solution (achieved 0.0) has done: 'We tighten the mask generation: convert colour images to grayscale before Otsu, apply a small‑object filter to discard noise, and ensure any image‑load failures fall back to an empty mask so the script always writes a complete CSV. These minor tweaks keep the original workflow but give a more realistic segmentation, moving the validation score toward the target while still producing a valid `crf_correction.csv` file.'
- What this solution (achieved 0.0) has done: 'I adjust the simple_mask routine to produce slightly larger salt regions, which should raise the validation score toward the target. The changes lower the Otsu threshold by a factor (making the mask more inclusive) and reduce the minimum object size for removal, while adding a small binary closing to fill tiny gaps. These tweaks are minimal, keep the original workflow intact, and aim to increase true positives without over‑hauling the model.'
- What this solution (achieved 0.0) has done: 'I make the mask generator more permissive so that predicted regions cover more salt, which raises the mean‑average‑precision toward the target without changing the overall workflow. In `simple_mask` I lower the Otsu‑based threshold multiplier to 0.5, skip aggressive small‑object removal, add a binary closing followed by a small dilation to fill gaps, and keep the same input handling. These minimal tweaks keep the original pipeline intact while producing larger, more complete masks, which should increase the score from 0 toward the target 0.8218. The rest of the script is unchanged except for renumbering cells to start at 1 for consistency.'
- What this solution (achieved 0.0) has done: 'I make the mask generator a little more permissive so that it predicts more salt pixels, which should raise the mean‑average‑precision from the current 0.0 toward the target 0.8218. The changes keep the overall workflow unchanged: they only adjust the Otsu‑derived threshold factor and add a fallback that forces a non‑empty mask when the result would be all zeros.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread
from skimage.color import gray2rgb, rgb2gray

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral

    _CRF_AVAILABLE = True
except Exception:
    _CRF_AVAILABLE = False


def _first_existing_path(candidates):
    """
    Return the first existing path from *candidates*.
    If none exist, raise FileNotFoundError.
    """
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")




## === cell 1
def rle_decode(rle_mask):
    """
    Decode a run‑length encoded mask string into a 101x101 binary numpy array.
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
test_path = _first_existing_path(
    [
        "/kaggle/input/tgs-salt-identification-challenge/test/images/",
        "/kaggle/input/test/images/",
        "./input/tgs-salt-identification-challenge/test/images/",
        "./kaggle/input/tgs-salt-identification-challenge/test/images/",
        "./kaggle/input/test/images/",
        "./input/test/images/",
        "./test/images/",
    ]
)

sample_candidates = [
    "/kaggle/input/tgs-salt-identification-challenge/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "./input/tgs-salt-identification-challenge/sample_submission.csv",
    "./kaggle/input/tgs-salt-identification-challenge/sample_submission.csv",
    "./kaggle/input/sample_submission.csv",
    "./input/sample_submission.csv",
    "./data/sample_submission.csv",
    "./sample_submission.csv",
]
try:
    submission_path = _first_existing_path(sample_candidates)
    df = pd.read_csv(submission_path)
except FileNotFoundError:
    ids = [
        os.path.splitext(f)[0]
        for f in os.listdir(test_path)
        if f.lower().endswith(".png")
    ]
    df = pd.DataFrame({"id": ids, "rle_mask": pd.NA})




## === cell 3
def crf(original_image, mask_img):
    """
    Apply DenseCRF refinement if the library is available.
    If not, return the original mask unchanged.
    """
    if not _CRF_AVAILABLE:
        return mask_img.astype(np.uint8)

    if mask_img.ndim < 3:
        mask_img = gray2rgb(mask_img)

    annotated_label = (
        mask_img[:, :, 0].astype(np.uint32)
        + (mask_img[:, :, 1].astype(np.uint32) << 8)
        + (mask_img[:, :, 2].astype(np.uint32) << 16)
    )

    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2  # background & salt
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
def rle_encode(im):
    """
    Encode a binary mask (numpy array) into run‑length format string.
    """
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 5
def _otsu_threshold(image):
    """
    Compute Otsu's threshold for a uint8 image.
    """
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    total = image.size
    sum_total = np.dot(np.arange(256), hist)
    sumB = 0.0
    wB = 0
    max_var = 0.0
    thresh = 0
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > max_var:
            max_var = var_between
            thresh = i
    return thresh


def simple_mask(image, thresh=None):
    """
    Light fallback mask: pixels brighter than `thresh` (or Otsu‑derived) are considered salt.
    Made more permissive:
      * Use a lower Otsu multiplier (0.3) to include more bright pixels.
      * Keep small objects (no size filtering).
      * Apply closing + dilation to fill gaps.
      * If the resulting mask is empty, fall back to an all‑ones mask so we never output a completely empty prediction.
    """
    from skimage.morphology import binary_closing, binary_dilation, disk

    if image.ndim == 3:
        image = (rgb2gray(image) * 255).astype(np.uint8)

    if image.dtype != np.uint8:
        image = (
            (image * 255).astype(np.uint8)
            if image.max() <= 1
            else image.astype(np.uint8)
        )

    if thresh is None:
        otsu_thresh = _otsu_threshold(image)
        thresh = max(1, int(otsu_thresh * 0.3))

    mask = (image > thresh).astype(np.uint8)

    mask = binary_closing(mask, disk(1)).astype(np.uint8)
    mask = binary_dilation(mask, disk(2)).astype(np.uint8)

    if mask.sum() == 0:
        mask = np.ones_like(mask, dtype=np.uint8)

    return mask




## === cell 6
for idx in tqdm(range(df.shape[0]), desc="Refining masks"):
    rle_val = df.loc[idx, "rle_mask"]
    img_path = os.path.join(test_path, df.loc[idx, "id"] + ".png")
    try:
        orig_img = imread(img_path)
    except Exception:
        decoded_mask = np.zeros((101, 101), dtype=np.uint8)
        refined_mask = decoded_mask
        df.loc[idx, "rle_mask"] = rle_encode(refined_mask)
        continue

    if pd.isna(rle_val):
        decoded_mask = simple_mask(orig_img)
    else:
        decoded_mask = rle_decode(str(rle_val))
        if decoded_mask.sum() == 0:
            decoded_mask = simple_mask(orig_img)

    refined_mask = crf(orig_img, decoded_mask)
    df.loc[idx, "rle_mask"] = rle_encode(refined_mask)




## === cell 7
df.to_csv("crf_correction.csv", index=False)
