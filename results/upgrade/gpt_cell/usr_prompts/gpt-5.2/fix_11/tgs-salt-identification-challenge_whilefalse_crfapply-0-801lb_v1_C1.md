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

- What this solution (achieved 0.0) has done: 'Diagnosis: Cell 0 crashes immediately because it tries to read a submission blend file from `../input/candsub/blend_ez.csv`, but that path does not exist in the provided filesystem. The notebook otherwise expects a dataframe with columns `id` and `rle_mask`, which is already available via the competition’s `sample_submission.csv`. Using the sample submission preserves the downstream interface and allows the CRF loop to run deterministically without changing any model/CRF logic.  

Patch summary: Replace the hard-coded missing input path with a small fallback that loads `../input/tgs-salt-identification-challenge/sample_submission.csv` (or `../input/sample_submission.csv`) when the blend file is absent, keeping `df` schema unchanged.  

Updated cells: Only cell 0 is modified.  

Compatibility notes for cell k+1: No changes to variable names or types; `df` remains a pandas DataFrame with `id` and `rle_mask` columns, so any later cells expecting `df` remain compatible.  

Assumptions: It is acceptable to proceed without the external blended submission and instead apply CRF to an available submission template; required test images exist at `../input/tgs-salt-identification-challenge/test/images/`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is because the pipeline is applying CRF to the *sample submission*, which has essentially empty/placeholder masks, so CRF cannot “invent” salt regions and you submit almost-all-empty masks. To move the score up toward the 0.82226 target while keeping the same CRF-based core logic, the smallest valid fix is to load an actual prediction submission (your `blend_ez.csv`) and only fall back to `sample_submission.csv` if it’s truly the only option. I also make the CRF loop robust to missing test images and ensure the output file has a `.csv` suffix (Kaggle requires it). This preserves the same CRF function, RLE logic, and loop structure; it just ensures the correct input submission is used so the method can work.'
- What this solution (achieved 0.0) has done: 'Diagnosis: Cell 0 crashes because it hard-codes expected paths for an upstream prediction file (`blend_ez.csv`) that is not present in this environment. However, the environment does contain a valid CSV at `../input/sample_submission.csv` (and also under `../input/tgs-salt-identification-challenge/`), so the strict refusal to run prevents any execution. The minimal fix is to extend the candidate path list to include the available submission CSVs as a fallback, keeping the rest of the CRF post-processing logic unchanged.

Patch summary: Update `candidate_paths` in cell 0 to also search for `../input/sample_submission.csv` and `../input/tgs-salt-identification-challenge/sample_submission.csv`. Keep the existing validation for required columns and all subsequent processing identical.

Updated cells: Only cell 0 is modified.

Compatibility notes for cell k+1: No variables, outputs, or interfaces are changed; `df`, `used_path`, and the written output CSV remain the same.

Assumptions: Using `sample_submission.csv` as input is acceptable in this environment to unblock execution (even if it yields low-quality masks), since no other prediction CSV is available.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is driven by the pipeline refusing to run unless a non-existent `blend_ez.csv` is found; the only available input then becomes `sample_submission.csv`, which yields essentially empty masks. To move the score up toward the 0.82226 target while keeping the same CRF + RLE core logic, I (1) allow running with `sample_submission.csv` when no prediction CSV exists, and (2) in that fallback case, generate a simple mask proposal from each test image (Otsu thresholding) and then run your existing CRF refinement on it. This preserves your loop structure, CRF function, and encoding/decoding semantics, but ensures non-empty reasonable masks so the submission is no longer near-all-empty. I also make the test image directory selection robust (choose the first existing path) so the loop actually processes images in this filesystem.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from using `sample_submission.csv` as the “prediction” input (nearly all-empty masks), so even with CRF you submit almost-all-empty masks. To move the score upward while preserving the same CRF + RLE + loop core logic, I keep the current fallback behavior but make the fallback initial mask materially better using depth-informed Otsu (same simple thresholding family) and a tiny amount of morphology to remove speckle/holes before CRF. I also ensure every `id` in `sample_submission.csv` is processed in-order (no silent `continue` that leaves placeholders untouched) by explicitly checking image existence and filling safely, which avoids accidental empty rows. These are minimal, local changes: no new model, no new training, same output schema and file path.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with generating weak/empty masks (either because the input submission is the sample submission or because the CRF unary is being built from the wrong label set), so the smallest safe move upward is to (1) make sure the CRF unary is computed from a proper 2-class label mask (background vs salt) rather than from “unique colors” that can collapse to 1 label, and (2) slightly strengthen the fallback initial mask so it’s not dominated by noise (still the same Otsu+morphology approach you already use). These changes keep the same pipeline structure (load CSV → decode/initialize mask → CRF refine → RLE encode → write submission) and do not introduce training or new models. They should materially increase non-empty, coherent masks and thus move the score toward the 0.82226 target. The output path and submission schema stay identical.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with producing mostly-empty masks because the code only generates meaningful masks when it finds a real prediction CSV (which is missing here), and the current fallback uses a simple Otsu mask that is not well-aligned with the TGS salt appearance. To move the score upward toward 0.82226 while preserving the same pipeline (load CSV → decode/initialize mask → CRF refine → RLE encode → write submission), I only strengthen the fallback initialization when `sample_submission.csv` is used: invert the Otsu decision (salt is typically darker), and adjust the depth bias direction accordingly. Everything else (CRF settings, loop, RLE encode/decode, I/O paths, output schema/filename) stays the same, so behavior changes only when the missing `blend_ez.csv` forces fallback.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is because the notebook can’t access the intended predicted masks (`blend_ez.csv`) and falls back to `sample_submission.csv`, which is mostly empty; CRF cannot recover meaningful masks from that. To move the score upward toward 0.82226 without changing the CRF/RLE pipeline, I keep the exact same flow but improve only the fallback initialization: a stronger, still-simple threshold-based mask (percentile-based, with optional depth bias) plus the same lightweight morphology you already use. I also remove the `continue` that leaves rows empty when an image is missing, ensuring every id gets a deterministic (empty) mask rather than potentially inconsistent output. Everything else (CRF settings, inference iterations, RLE encode/decode, file name and columns) is preserved.'

# 9. Code solution

## === cell 0
import numpy as np

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral

    _HAS_DENSECRF = True
except ModuleNotFoundError:
    dcrf = None
    unary_from_labels = None
    create_pairwise_bilateral = None
    _HAS_DENSECRF = False

from skimage.io import imread
from skimage.color import gray2rgb
from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_opening,
    binary_closing,
    disk,
)
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os


def rle_decode(mask_rle, shape=(101, 101)):
    """
    mask_rle: run-length as string formatted (start length), 1-indexed, column-major.
    Returns numpy array, 1 - mask, 0 - background (shape 101x101)
    """
    if mask_rle is None:
        return np.zeros(shape, dtype=np.uint8)
    if not isinstance(mask_rle, str):
        return np.zeros(shape, dtype=np.uint8)
    mask_rle = mask_rle.strip()
    if mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted, column-major, 1-indexed.
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def crf(original_image, mask_img):
    if not _HAS_DENSECRF:
        if len(mask_img.shape) >= 3:
            mask2d = mask_img[:, :, 0]
        else:
            mask2d = mask_img
        return (mask2d > 0).astype(np.uint8)

    if mask_img.ndim == 3:
        mask2d = mask_img[:, :, 0]
    else:
        mask2d = mask_img

    labels = (mask2d > 0).astype(np.int32)

    n_labels = 2
    h, w = original_image.shape[0], original_image.shape[1]
    d = dcrf.DenseCRF2D(w, h, n_labels)

    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(
        sxy=(3, 3),
        compat=3,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    Q = d.inference(10)
    MAP = np.argmax(Q, axis=0).astype(np.uint8)
    return MAP.reshape((h, w))


test_img_dir_candidates = [
    "../input/tgs-salt-identification-challenge/test/images/",
    "../input/test/images/",
    "/kaggle/input/tgs-salt-identification-challenge/test/images/",
    "/kaggle/input/test/images/",
]
test_img_dir = None
for p in test_img_dir_candidates:
    if os.path.isdir(p):
        test_img_dir = p
        break
if test_img_dir is None:
    raise FileNotFoundError(
        "Could not find test images directory. Tried:\n"
        + "\n".join(test_img_dir_candidates)
    )
print(f"Using test image dir: {test_img_dir}")


candidate_paths = [
    "../input/candsub/blend_ez.csv",  # original intended input
    "../input/tgs-salt-identification-challenge/blend_ez.csv",
    "../input/blend_ez.csv",
]
fallback_paths = [
    "../input/tgs-salt-identification-challenge/sample_submission.csv",
    "../input/sample_submission.csv",
]

df = None
used_path = None
for p in candidate_paths:
    if os.path.exists(p):
        df = pd.read_csv(p)
        used_path = p
        break

used_fallback_sample = False
if df is None:
    for p in fallback_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            used_path = p
            used_fallback_sample = True
            break

if df is None:
    raise FileNotFoundError(
        "No submission CSV found. Tried prediction paths:\n"
        + "\n".join(candidate_paths)
        + "\n\nAnd fallback paths:\n"
        + "\n".join(fallback_paths)
    )

required_cols = {"id", "rle_mask"}
if not required_cols.issubset(df.columns):
    raise ValueError(
        f"Submission input must have columns {required_cols}, got {set(df.columns)}"
    )

print(
    f"Loaded candidate submission: {used_path} (rows={len(df)}), fallback_sample={used_fallback_sample}"
)

depth_paths = [
    "../input/tgs-salt-identification-challenge/depths.csv",
    "../input/depths.csv",
    "/kaggle/input/tgs-salt-identification-challenge/depths.csv",
    "/kaggle/input/depths.csv",
]
depth_df = None
for p in depth_paths:
    if os.path.exists(p):
        depth_df = pd.read_csv(p)
        break
if depth_df is not None and {"id", "z"}.issubset(depth_df.columns):
    depth_map = dict(zip(depth_df["id"].astype(str).values, depth_df["z"].values))
else:
    depth_map = {}


def initial_mask_from_image_otsu(orig_img, img_id=None):
    if orig_img.ndim == 3:
        img = orig_img[:, :, 0]
    else:
        img = orig_img
    img = img.astype(np.float32)

    vmin = float(np.min(img))
    vmax = float(np.max(img))
    denom = (vmax - vmin) if (vmax > vmin) else 1.0
    img01 = (img - vmin) / denom

    thr = float(np.percentile(img01, 35.0))

    if img_id is not None and img_id in depth_map:
        z = float(depth_map[img_id])
        bias = np.clip((z - 500.0) / 500.0, -1.0, 1.0) * 0.02
        thr = float(np.clip(thr - bias, 0.0, 1.0))

    mask = (img01 < thr).astype(np.uint8)

    m = mask.astype(bool)
    m = binary_opening(m, disk(1))
    m = binary_closing(m, disk(1))
    m = remove_small_objects(m, min_size=50)
    m = remove_small_holes(m, area_threshold=50)
    return m.astype(np.uint8)


i = 0
j = 0
plt.figure(figsize=(30, 15))
plt.subplots_adjust(bottom=0.2, top=0.8, hspace=0.2)

max_scan = min(df.shape[0], 200)
while i < max_scan and j < 6:
    rle = df.loc[i, "rle_mask"]
    if isinstance(rle, str) and len(rle.strip()) > 0:
        decoded_mask = rle_decode(rle)
        plt.subplot(1, 6, j + 1)
        plt.imshow(decoded_mask)
        plt.title("ID: " + str(df.loc[i, "id"]))
        j += 1
    i += 1


for i in tqdm(range(df.shape[0])):
    img_id = str(df.loc[i, "id"])
    img_path = os.path.join(test_img_dir, img_id + ".png")

    if not os.path.exists(img_path):
        df.loc[i, "rle_mask"] = ""
        continue

    orig_img = imread(img_path)
    rle = df.loc[i, "rle_mask"]

    if isinstance(rle, str) and len(rle.strip()) > 0:
        decoded_mask = rle_decode(rle)
    else:
        if used_fallback_sample:
            decoded_mask = initial_mask_from_image_otsu(orig_img, img_id=img_id)
        else:
            decoded_mask = np.zeros((101, 101), dtype=np.uint8)

    crf_output = crf(orig_img, decoded_mask)
    df.loc[i, "rle_mask"] = rle_encode(crf_output)

out_path = "crf_correction_unet_bn_diceloss.csv"
df.to_csv(out_path, index=False)
print(f"Wrote submission: {out_path} (rows={len(df)})")
