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

0.77029

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Diagnosis: Cell 3 crashes because it tries to load a submission file from a hard-coded path (`../input/subnr1/submission.csv`) that does not exist in this environment. The available submission template is `../input/sample_submission.csv` (and also under `../input/tgs-salt-identification-challenge/`).  
Patch summary: Modify only cell 3 to load the first existing submission CSV from a small list of known candidate paths, preserving the downstream logic and the `df` variable name. This makes the notebook deterministic and prevents the FileNotFoundError without changing any model/metric logic.  
Updated cells: Only cell 3 is updated.  
Compatibility notes for cell k+1: No interface changes; `df` remains a pandas DataFrame with `id` and `rle_mask` columns, so later cells are unaffected.  
Assumptions: A valid submission-format CSV is available at one of the provided dataset paths (at minimum `../input/sample_submission.csv`).'
- What this solution (achieved 0.0) has done: 'Diagnosis: Cell 6 crashes inside `crf()` because `pydensecrf` is not installed in this environment, so in cell 1 the guarded import sets `dcrf = None`. When cell 6 calls `crf()`, it tries to access `dcrf.DenseCRF2D`, causing `AttributeError: 'NoneType' object has no attribute 'DenseCRF2D'`. The minimal fix is to make cell 6 skip CRF inference when `dcrf` (and its helpers) are unavailable, while keeping the visualization loop and outputs consistent.

Patch summary: Modify only cell 6 to conditionally bypass CRF computation when `pydensecrf` is missing by setting `crf_output = decoded_mask` in that case. This preserves the loop’s behavior and avoids changing the model/CRF logic itself. No other cells or functions are changed.

Updated cells: (cell 6 only)

Compatibility notes for cell k+1: Cell 7 is unaffected; variables created in cell 6 (`decoded_mask`, `orig_img`, `crf_output`) still exist with compatible shapes/types for plotting and any downstream use.

Assumptions: When `pydensecrf` is unavailable, falling back to the unrefined decoded mask is acceptable for visualization and prevents the runtime crash.'
- What this solution (achieved 0.0) has done: 'Diagnosis: The crash happens because `pydensecrf` is not installed, so in cell 1 `dcrf` is set to `None`. In cell 8, the loop still calls `crf(orig_img, decoded_mask)` unconditionally, and `crf()` tries to access `dcrf.DenseCRF2D`, causing `AttributeError: 'NoneType' object has no attribute 'DenseCRF2D'`.  
Patch summary: Update cell 8 to mirror the guard already used in cell 6: when `dcrf`/`unary_from_labels` are unavailable, skip CRF and keep the decoded mask as-is, then re-encode. This preserves the existing semantics when CRF is available and prevents the crash when it is not.  
Updated cells: Only cell 8 is modified.  
Compatibility notes for cell k+1: `df` remains a DataFrame with the same columns (`id`, `rle_mask`) and is still fully populated so cell 9 (`df.to_csv(...)`) works unchanged.  
Assumptions: If CRF dependencies are missing, the intended safe fallback is to keep the original mask (as already done in cell 6).'
- What this solution (achieved 0.5221) has done: 'Your current 0.0 score is because the code never produces a meaningful prediction: it starts from `sample_submission.csv` (all-empty masks) and then “refines” those empties, resulting in an all-empty final submission. To move toward the 0.77029 target without changing the core CRF/loop logic, the minimal legitimate improvement is to start from a real baseline prediction file when available (e.g., `train.csv` in this environment contains non-empty RLEs) and only fall back to `sample_submission.csv` if nothing else exists. I keep your decoding/encoding and CRF behavior identical (including the “skip CRF if not installed” safeguard), but adjust cell 3 to prefer a non-empty-mask CSV and align it to the sample submission `id` list so the produced `crf_correction.csv` is a valid submission. This should raise the score above 0.0 (likely still below strong model scores, which is fine since we’re aiming toward—not necessarily at—the target band with minimal change).'
- What this solution (achieved 0.5221) has done: 'Diagnosis: Cell 6 iterates with `while True` and increments `i` without any bound check, so when `i` reaches `df.shape[0]` (1000 for test ids), `df.loc[i, ...]` raises `KeyError: 1000`. This happens especially when many/most `rle_mask` values are NaN, causing the loop to skip `j` increments and keep increasing `i` until it runs off the end.  
Patch summary: Add a deterministic bounds guard at the top of the loop (`if i >= df.shape[0]: break`) to prevent out-of-range `.loc` access, preserving the existing visualization and CRF logic. No other logic is changed.  
Updated cells: Only cell 6 is modified.  
Compatibility notes for cell k+1: No variables, types, or outputs are changed; `df`, `test_path`, and all functions remain the same, so cell 7 and later cells remain compatible.  
Assumptions: `df` uses the default `RangeIndex(0..n-1)` as created in cell 3, so checking `i >= df.shape[0]` is the correct termination condition.'
- What this solution (achieved 0.5221) has done: 'We keep your decode → (optional) CRF → encode pipeline unchanged, but make the baseline masks less noisy and better aligned to the test domain to move the score up toward 0.77029 from 0.5221. The minimal lever here is post-processing: (1) use an explicit NaN check (your current string comparison mis-detects NaNs), (2) binarize CRF output (or decoded mask) to 0/1 before encoding, and (3) add a tiny hole-filling + small-object removal cleanup on each predicted mask before RLE encoding (this typically improves mAP for this competition without changing model/CRF). We also ensure that missing predictions stay empty strings (as expected by the submission format) and keep file paths unchanged. These changes are lightweight, deterministic, and should improve IoU consistency across thresholds without altering your overall approach.'
- What this solution (achieved 0.5221) has done: 'We keep your existing “use a baseline CSV → decode → (optional) CRF → morph cleanup → RLE → write submission” pipeline intact, but make two minimal tweaks that typically move TGS mAP up from ~0.52 without changing the core approach. First, we make the RLE encoding column-major (Fortran order) which matches this competition’s expected pixel ordering; your current row-major flattening can severely depress score even when masks look correct. Second, we make the decode/encode pair consistent (both Fortran order) so any subsequent CRF/morphology operates on the correctly oriented mask, improving IoU alignment and thus mAP toward the 0.77029 target.'
- What this solution (achieved 0.5221) has done: 'We keep your exact “baseline RLE → decode → (optional) CRF → encode” pipeline, but make two small post-processing changes that typically lift mAP from the ~0.52 range toward your 0.77029 target without changing architecture/training (since there is none here). First, we add a very light `binary_closing` on the boolean mask before hole-filling/object-removal to connect thin salt regions that otherwise fragment and hurt IoU at higher thresholds. Second, we add a conservative per-image thresholding rule for CRF outputs: if CRF is unavailable we keep the decoded mask, but if CRF is available we still binarize and then apply the same cleanup consistently; this stabilizes mask topology and usually improves mean AP. All file paths and submission schema remain unchanged, and we still write `crf_correction.csv`.'
- What this solution (achieved 0.5221) has done: 'We keep your decode → (optional) CRF → morph cleanup → RLE pipeline intact, but make two minimal post-processing tweaks that usually improve IoU consistency across thresholds for this competition: apply a conservative probability threshold (0.5) when CRF is available (instead of any-nonzero), and apply the same light morphology cleanup to the non-CRF fallback path so behavior is consistent. We also add an ultra-light “empty mask guard” to avoid morphological ops/encoding overhead on truly empty predictions (stability + speed, no semantic change). These changes are small, deterministic, and aimed at lifting score from 0.5221 toward 0.77029 without altering your overall approach or data paths. The script still write a valid `crf_correction.csv` with `id,rle_mask`.'
- What this solution (achieved 0.5221) has done: 'We keep your exact “baseline RLE → decode → (optional) CRF → morph cleanup → RLE” pipeline, but make two small changes that typically increase TGS mAP from the ~0.52 range without altering the core approach. First, we stop using `train.csv` as a baseline for **test** ids (it’s mismatched and mostly NaN after mapping), and instead build a simple, deterministic per-image baseline mask directly from each test image (Otsu threshold on the 101×101 grayscale) only when the baseline CSV has no mask for that id—this gives non-empty, image-aligned starting masks for CRF/morphology. Second, we add a single global threshold knob (`BASE_THR_SCALE`) to conservatively control mask size so we move upward toward the 0.77029 target without aggressive over-segmentation. Output schema/paths remain unchanged and the script still writes `crf_correction.csv`.'
- What this solution (achieved 0.5221) has done: 'We keep your exact pipeline (baseline mask → optional CRF → same morphology cleanup → RLE) but make two minimal, score-relevant fixes to move from 0.5221 toward 0.77029. First, the image-derived fallback mask is currently inverted for this dataset (salt is typically darker), so we switch the Otsu-based fallback to `g < t` (with the same `BASE_THR_SCALE` knob) to reduce systematic false positives and raise IoU. Second, we apply a tiny “empty-or-tiny mask” guard after post-processing (set to empty if area is extremely small) to reduce false positives that heavily hurt mean AP across IoU thresholds. Everything else (CRF usage/skip behavior, morphology ops, RLE Fortran order, paths, output CSV) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



## === cell 1
import numpy as np

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
except ModuleNotFoundError:
    dcrf = None
    unary_from_labels = None
    create_pairwise_bilateral = None

from skimage.io import imread, imsave
from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    square,
)

from skimage.filters import threshold_otsu

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass




## === cell 2
def rle_decode(rle_mask):
    """
    rle_mask: run-length as string formatted (start length)
    Returns numpy array, 1 - mask, 0 - background

    Why this helps toward the target:
    - TGS Salt expects pixels numbered top-to-bottom then left-to-right (column-major / Fortran order).
      Decoding with row-major order rotates/permutes the mask relative to the image and can heavily
      reduce IoU/mAP. We decode in Fortran order to match the competition definition.
    """
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((101, 101), order="F")




## === cell 3
"""
reading and decoding the submission

Why this helps toward the target:
- The previous candidate order preferred train.csv, which is not a valid baseline for test ids and
  maps to mostly-NaN masks, effectively reverting to empty predictions.
- We instead prefer an actual submission-format baseline if present; otherwise we start from the
  sample_submission and later create an image-derived baseline mask for each empty/NaN case.
- Keep NaN detection robustly (pd.isna) so we don't accidentally try to decode NaNs or skip strings.
- Ensure empty predictions are empty strings (competition expects blanks for no-mask).
"""
candidate_paths = [
    "../input/subnr1/submission.csv",
    "../input/tgs-salt-identification-challenge/sample_submission.csv",
    "../input/sample_submission.csv",
]

submission_path = None
for p in candidate_paths:
    if os.path.exists(p):
        submission_path = p
        break

if submission_path is None:
    raise FileNotFoundError("No CSV found. Tried: " + ", ".join(candidate_paths))

sample_paths = [
    "../input/tgs-salt-identification-challenge/sample_submission.csv",
    "../input/sample_submission.csv",
]
sample_path = None
for p in sample_paths:
    if os.path.exists(p):
        sample_path = p
        break
if sample_path is None:
    raise FileNotFoundError(
        "No sample_submission.csv found. Tried: " + ", ".join(sample_paths)
    )

df_ids = pd.read_csv(sample_path)  # has correct test ids/order
df_src = pd.read_csv(submission_path)

if ("id" in df_src.columns) and ("rle_mask" in df_src.columns):
    src_map = df_src.set_index("id")["rle_mask"]
    df = df_ids.copy()
    df["rle_mask"] = df["id"].map(src_map)
else:
    df = df_ids.copy()

df["rle_mask"] = df["rle_mask"].replace("", np.nan)

i = 0
j = 0
plt.figure(figsize=(30, 15))
plt.subplots_adjust(bottom=0.2, top=0.8, hspace=0.2)
while True:
    if i >= df.shape[0]:
        break
    if not pd.isna(df.loc[i, "rle_mask"]):
        decoded_mask = rle_decode(df.loc[i, "rle_mask"])
        plt.subplot(1, 6, j + 1)
        plt.imshow(decoded_mask)
        plt.title("ID: " + df.loc[i, "id"])
        j = j + 1
        if j > 5:
            break
    i = i + 1



## === cell 4
"""
Function which returns the labelled image after applying CRF
"""


def crf(original_image, mask_img):

    if len(mask_img.shape) < 3:
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




## === cell 5
test_path = "../input/tgs-salt-identification-challenge/test/images/"



## === cell 6
"""
visualizing the effect of applying CRF
"""
nImgs = 3
i = np.random.randint(1000)
j = 1
plt.figure(figsize=(15, 15))
plt.subplots_adjust(wspace=0.2, hspace=0.1)
while True:
    if i >= df.shape[0]:
        break

    if not pd.isna(df.loc[i, "rle_mask"]):
        decoded_mask = rle_decode(df.loc[i, "rle_mask"])
        orig_img = imread(test_path + df.loc[i, "id"] + ".png")

        if dcrf is None or unary_from_labels is None:
            crf_output = decoded_mask.astype(bool)
        else:
            crf_output = crf(orig_img, decoded_mask).astype(bool)

        crf_output = crf_output.astype(np.uint8)

        plt.subplot(nImgs, 4, 4 * j - 3)
        plt.imshow(orig_img)
        plt.title("Original image")
        plt.subplot(nImgs, 4, 4 * j - 2)
        plt.imshow(decoded_mask)
        plt.title("Original Mask")
        plt.subplot(nImgs, 4, 4 * j - 1)
        plt.imshow(crf_output)
        plt.title("Mask after CRF")
        if j == nImgs:
            break
        else:
            j = j + 1
    i = i + 1



## === cell 7
"""
used for converting the decoded image to rle mask
"""


def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted

    Why this helps toward the target:
    - The competition expects column-major (Fortran-order) pixel indexing.
      Encoding with row-major (C-order) scrambles masks and reduces IoU/mAP.
    """
    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 8
"""
Applying CRF on the predicted mask

Why this helps toward the target:
- Main fix to move score up from ~0.52: when the baseline CSV provides NaN/empty for a test id,
  we generate a simple image-derived baseline mask (Otsu on grayscale, with conservative scaling).
  This keeps the same pipeline structure (baseline mask -> optional CRF -> morphology -> RLE),
  but avoids producing mostly-empty submissions.
- Score fix #1 (minimal but impactful): for TGS salt, salt regions are typically darker; using (g > t)
  tends to invert the mask and creates large false positives. Switching to (g < t) aligns the fallback
  with the expected foreground and should increase IoU/mAP toward the target.
- Score fix #2 (minimal): after post-processing, drop tiny predicted blobs to empty to reduce FP,
  which improves mean AP across IoU thresholds without changing the pipeline.
- We keep your CRF availability behavior unchanged (skip if pydensecrf isn't installed).
- We keep your morphology cleanup unchanged and deterministic.
"""
BASE_THR_SCALE = 1.05
TINY_AREA_TO_EMPTY = (
    8  # conservative FP-reduction; keeps pipeline semantics, just stabilizes output
)

for i in tqdm(range(df.shape[0])):
    img_id = df.loc[i, "id"]
    orig_img = imread(test_path + img_id + ".png")

    if pd.isna(df.loc[i, "rle_mask"]):
        g = orig_img.astype(np.float32)
        if g.ndim == 3:
            g = rgb2gray(g)
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)

        t = threshold_otsu(g) * BASE_THR_SCALE

        decoded_mask = (g < t).astype(np.uint8)
    else:
        decoded_mask = rle_decode(df.loc[i, "rle_mask"]).astype(np.uint8)

    if decoded_mask.sum() == 0:
        df.loc[i, "rle_mask"] = ""
        continue

    if dcrf is None or unary_from_labels is None:
        bin_mask = decoded_mask.astype(bool)
    else:
        crf_map = crf(orig_img, decoded_mask)
        bin_mask = crf_map == 1

    bin_mask = binary_closing(bin_mask, footprint=square(3))
    bin_mask = remove_small_holes(bin_mask, area_threshold=16, connectivity=1)
    bin_mask = remove_small_objects(bin_mask, min_size=16, connectivity=1)

    if bin_mask.sum() < TINY_AREA_TO_EMPTY:
        df.loc[i, "rle_mask"] = ""
    else:
        df.loc[i, "rle_mask"] = rle_encode(bin_mask.astype(np.uint8))

df["rle_mask"] = df["rle_mask"].fillna("")



## === cell 9
df.to_csv("crf_correction.csv", index=False)
df.head()
