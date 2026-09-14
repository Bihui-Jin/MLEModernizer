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

0.8202596278667235

# 6. Current score

0.03

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I remove the hard dependency on `pydensecrf` (not installed in this environment) by providing a lightweight fallback CRF-like refinement using only `numpy` and `skimage` morphology; this keeps the pipeline running end-to-end and produces a valid `.csv` submission. I also remove notebook-only magic (`%matplotlib inline`) and replace shell `ls` with a Python directory listing so the script runs as a plain Kaggle notebook/script. The code currently fails because `df`, `pd`, `plt`, and `tqdm` never get defined after the first import error; I fix the import cell and make the input submission come from the provided `sample_submission.csv` (since the referenced `../input/submit-802/...` file does not exist here). Finally, I ensure NaN/empty masks are handled correctly and always write `crf_correction.csv` with columns `id,rle_mask`.'
- What this solution (achieved 0.0634) has done: 'Your current pipeline is only post-processing the `sample_submission.csv`, which contains dummy masks (“1 1”) and therefore cannot reach the target score; to move toward 0.82 we must generate real predictions first, then (optionally) apply your existing CRF-like refinement and RLE encoding. To keep core logic minimal and within constraints, I add a lightweight, classical segmentation baseline: read each test image, normalize, compute an Otsu threshold, and clean the mask with the same morphology utilities you already use (this is still “mask refinement/post-processing” and stays simple, fast, and dependency-free). I also fix the RLE encode order to the competition’s expected top-to-bottom then left-to-right indexing (Fortran order), which typically yields a large score jump versus C-order encoding. The script still write `crf_correction.csv` with the correct `id,rle_mask` columns and finish within the time limit.'
- What this solution (achieved 0.0681) has done: 'Your score is far below the target, so we should improve prediction quality while keeping the same “classical threshold → morphology cleanup → (optional) CRF/fallback → RLE” core flow intact. The biggest safe gain with minimal logic change is to tune the initial mask creation: in this competition salt is generally brighter than sediment, so using `img > threshold` (instead of `<`) and a tiny amount of denoising before Otsu typically fixes inverted masks and raises IoU substantially. I also make RLE decode consistent with the required Fortran (column-major) convention to avoid any subtle orientation mismatches, and I keep the same refinement and CSV-writing behavior. These changes are small, fast, and directly targeted at moving the score toward 0.82.'
- What this solution (achieved 0.0006) has done: 'Your current score (0.0681) is far below the target (0.8203), so we should make small, metric-aligned improvements that meaningfully increase IoU without changing the overall “threshold → morphology → refinement → RLE” flow. The biggest low-risk gain is to tune the binarization threshold: using a fixed Otsu split can be brittle, so we keep Otsu but apply a small bias and automatically choose the bias that yields a reasonable mask area (avoids near-empty/all-full masks). We also add a very small hole-filling step (still just morphology) which often boosts IoU for salt blobs. Finally, we ensure strict RLE correctness (already Fortran-order) and keep the same CSV output.'
- What this solution (achieved 0.0525) has done: 'I fix the runtime error by guaranteeing `initial_predict_mask()` always returns a valid (all-zero) mask even if no candidate threshold is selected, and by making the threshold-candidate loop robust to any unexpected morphology failures. I also add a small safety fallback for any weird/constant images to avoid NaNs and ensure `best_m` is never `None`. These changes are minimal, keep the same “Otsu → small bias search → morphology cleanup → CRF/fallback → RLE” core logic, and should restore end-to-end execution and produce a valid `crf_correction.csv` submission. With the code actually running and producing non-empty masks where appropriate, the score should increase from ~0.0006 toward the target band.'
- What this solution (achieved 0.03) has done: 'Your current score is far below the target, so the most direct way to move it upward without changing the overall “threshold → morphology cleanup → CRF/fallback → RLE” pipeline is to make the initial binarization less brittle and more salt-specific. I keep the same Otsu+bias sweep and morphology steps, but (1) add a second polarity option (salt can appear darker in some tiles), (2) replace the fixed “aim for 25% mask” heuristic with a small, depth-aware expected-area prior using the provided `depths.csv` (still just choosing among your existing candidates), and (3) add a tiny border-crop before scoring/cleanup to reduce edge artifacts from padding/contrast. These are minimal, fast, dependency-free changes that typically raise IoU materially while preserving your core logic and submission semantics. The script still run end-to-end and write `crf_correction.csv` with `id,rle_mask`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.morphology import (
    binary_opening,
    binary_closing,
    remove_small_objects,
    disk,
)
from skimage.filters import threshold_otsu, gaussian
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm

import matplotlib.pyplot as plt

_HAS_DCRF = False
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels
    from skimage.color import gray2rgb

    _HAS_DCRF = True
except Exception:
    _HAS_DCRF = False

RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)




## === cell 1
def rle_decode(rle_mask, shape=(101, 101)):
    """
    rle_mask: run-length as string formatted (start length)
    Returns numpy array, 1 - mask, 0 - background

    Reshape uses Fortran order to match the competition's
    top-to-bottom then left-to-right indexing convention.
    """
    if rle_mask is None:
        return np.zeros(shape, dtype=np.uint8)
    if isinstance(rle_mask, float) and np.isnan(rle_mask):
        return np.zeros(shape, dtype=np.uint8)

    s = str(rle_mask).strip()
    if s == "" or s.lower() == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = s.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        if lo < 0:
            lo = 0
        if hi > img.size:
            hi = img.size
        img[lo:hi] = 1
    return img.reshape(shape, order="F")




## === cell 2
def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted

    Use Fortran order flattening to match Kaggle TGS Salt submission convention.
    """
    if im is None:
        return ""
    im = (im > 0).astype(np.uint8)
    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if runs.size == 0:
        return ""
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
BASE_INPUT = "../input/tgs-salt-identification-challenge"
test_path = os.path.join(BASE_INPUT, "test", "images")
sample_sub_path = os.path.join(BASE_INPUT, "sample_submission.csv")
depths_path = os.path.join(BASE_INPUT, "depths.csv")

assert os.path.exists(
    sample_sub_path
), f"Missing sample submission at: {sample_sub_path}"
assert os.path.isdir(test_path), f"Missing test images dir at: {test_path}"
assert os.path.exists(depths_path), f"Missing depths at: {depths_path}"

print("Sample submission:", sample_sub_path)
print("Depths:", depths_path)
print("Test images dir:", test_path)
print("Num test images:", len([f for f in os.listdir(test_path) if f.endswith(".png")]))



## === cell 4
df = pd.read_csv(sample_sub_path)

if "id" not in df.columns or "rle_mask" not in df.columns:
    raise ValueError(
        f"Submission must have columns ['id','rle_mask'], got {df.columns.tolist()}"
    )

df["id"] = df["id"].astype(str)

depths_df = pd.read_csv(depths_path)
depths_df["id"] = depths_df["id"].astype(str)
depth_map = dict(zip(depths_df["id"].values, depths_df["z"].values))

_z_vals = depths_df["z"].values.astype(np.float32)
z_min, z_max = float(np.min(_z_vals)), float(np.max(_z_vals))
if z_max <= z_min:
    z_min, z_max = 0.0, 1.0

print(df.head())



## === cell 5
"""
Function which returns the labelled image after applying CRF.

We keep your original CRF/fallback refinement logic. This is applied to an
initial prediction mask (generated below).
"""


def crf(original_image, mask_img):
    if mask_img is None:
        return np.zeros_like(
            original_image[..., 0] if original_image.ndim == 3 else original_image,
            dtype=np.uint8,
        )
    mask = (mask_img > 0).astype(np.uint8)

    if _HAS_DCRF:
        if original_image.ndim == 2:
            rgb = np.stack([original_image] * 3, axis=-1)
        else:
            rgb = original_image

        if mask.ndim == 2:
            mask_rgb = gray2rgb(mask.astype(np.uint8) * 255)
        else:
            mask_rgb = mask

        annotated_label = (
            mask_rgb[:, :, 0] + (mask_rgb[:, :, 1] << 8) + (mask_rgb[:, :, 2] << 16)
        )
        _, labels = np.unique(annotated_label, return_inverse=True)

        n_labels = 2
        d = dcrf.DenseCRF2D(rgb.shape[1], rgb.shape[0], n_labels)

        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(
            sxy=(3, 3),
            compat=3,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC,
        )

        Q = d.inference(10)
        MAP = np.argmax(Q, axis=0).reshape((rgb.shape[0], rgb.shape[1]))
        return MAP.astype(np.uint8)

    m = mask.astype(bool)
    m = binary_opening(m, disk(1))
    m = binary_closing(m, disk(1))
    m = binary_fill_holes(m)
    m = remove_small_objects(m, min_size=10)
    return m.astype(np.uint8)




## === cell 6
print("Listing ../input (first 50 entries):")
try:
    print(sorted(os.listdir("../input"))[:50])
except Exception as e:
    print("Could not list ../input:", e)



## === cell 7
"""
Score improvement with minimal core-logic change:
- Keep Otsu -> small bias sweep -> morphology cleanup.
- Also test both polarities (img > t and img < t) and pick the best candidate.
- Use depths.csv to set an expected mask fraction prior, only for ranking candidates.
- Crop a 1px border during scoring/cleanup to reduce edge artifacts; paste back.
"""


def _expected_frac_from_depth(img_id):
    z = depth_map.get(str(img_id), None)
    if z is None:
        return 0.20  # safe default prior
    zn = (float(z) - z_min) / (z_max - z_min + 1e-12)
    zn = float(np.clip(zn, 0.0, 1.0))
    return float(0.28 - 0.18 * zn)


def _post_morph(m_bool):
    m_bool = binary_opening(m_bool, disk(1))
    m_bool = binary_closing(m_bool, disk(1))
    m_bool = binary_fill_holes(m_bool)
    m_bool = remove_small_objects(m_bool, min_size=10)
    return m_bool


def initial_predict_mask(orig_img, img_id=None):
    if orig_img.ndim == 3:
        img = orig_img[..., 0]
    else:
        img = orig_img

    img = img.astype(np.float32)

    mn = float(np.min(img))
    mx = float(np.max(img))
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        return np.zeros_like(img, dtype=np.uint8)

    img_smooth = gaussian(img, sigma=0.75, preserve_range=True)

    try:
        t0 = float(threshold_otsu(img_smooth))
        if not np.isfinite(t0):
            t0 = 0.5
    except Exception:
        t0 = 0.5

    candidates_t = [t0 - 0.05, t0 - 0.02, t0, t0 + 0.02, t0 + 0.05]
    expected_frac = _expected_frac_from_depth(img_id) if img_id is not None else 0.20

    best_m = None
    best_score = -1e18

    H, W = img_smooth.shape
    y0, y1 = 1, H - 1
    x0, x1 = 1, W - 1
    if y1 <= y0 or x1 <= x0:
        y0, y1, x0, x1 = 0, H, 0, W

    for t in candidates_t:
        t = float(np.clip(t, 0.0, 1.0))

        for polarity in (1, -1):
            if polarity == 1:
                m = img_smooth > t
            else:
                m = img_smooth < t

            try:
                mc = m[y0:y1, x0:x1]
                mc = _post_morph(mc)
                m2 = np.zeros_like(m, dtype=bool)
                m2[y0:y1, x0:x1] = mc
            except Exception:
                continue

            frac = float(m2.mean())

            score = -abs(frac - expected_frac)
            if frac < 0.003 or frac > 0.997:
                score -= 10.0

            if score > best_score:
                best_score = score
                best_m = m2

    if best_m is None:
        return np.zeros_like(img, dtype=np.uint8)

    return best_m.astype(np.uint8)


for i in tqdm(range(df.shape[0]), total=df.shape[0]):
    img_id = df.at[i, "id"]
    img_path = os.path.join(test_path, f"{img_id}.png")
    if not os.path.exists(img_path):
        df.at[i, "rle_mask"] = ""
        continue

    orig_img = imread(img_path)

    init_mask = initial_predict_mask(orig_img, img_id=img_id)
    if init_mask.sum() == 0:
        df.at[i, "rle_mask"] = ""
        continue

    refined = crf(orig_img, init_mask)
    df.at[i, "rle_mask"] = rle_encode(refined)



## === cell 8
df["rle_mask"] = df["rle_mask"].fillna("").astype(str)
df = df[["id", "rle_mask"]]

out_path = "crf_correction.csv"
df.to_csv(out_path, index=False)
print("Wrote submission:", out_path)
print(df.head())
print("Rows:", len(df), "Empty masks:", (df["rle_mask"] == "").sum())
