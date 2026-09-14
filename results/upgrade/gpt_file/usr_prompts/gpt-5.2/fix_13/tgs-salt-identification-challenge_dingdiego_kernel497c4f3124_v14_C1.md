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

0.0321

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I remove the hard dependency on `pydensecrf` (not installed in this environment) by providing a lightweight fallback CRF-like refinement using only `numpy` and `skimage` morphology; this keeps the pipeline running end-to-end and produces a valid `.csv` submission. I also remove notebook-only magic (`%matplotlib inline`) and replace shell `ls` with a Python directory listing so the script runs as a plain Kaggle notebook/script. The code currently fails because `df`, `pd`, `plt`, and `tqdm` never get defined after the first import error; I fix the import cell and make the input submission come from the provided `sample_submission.csv` (since the referenced `../input/submit-802/...` file does not exist here). Finally, I ensure NaN/empty masks are handled correctly and always write `crf_correction.csv` with columns `id,rle_mask`.'
- What this solution (achieved 0.0634) has done: 'Your current pipeline is only post-processing the `sample_submission.csv`, which contains dummy masks (“1 1”) and therefore cannot reach the target score; to move toward 0.82 we must generate real predictions first, then (optionally) apply your existing CRF-like refinement and RLE encoding. To keep core logic minimal and within constraints, I add a lightweight, classical segmentation baseline: read each test image, normalize, compute an Otsu threshold, and clean the mask with the same morphology utilities you already use (this is still “mask refinement/post-processing” and stays simple, fast, and dependency-free). I also fix the RLE encode order to the competition’s expected top-to-bottom then left-to-right indexing (Fortran order), which typically yields a large score jump versus C-order encoding. The script still write `crf_correction.csv` with the correct `id,rle_mask` columns and finish within the time limit.'
- What this solution (achieved 0.0681) has done: 'Your score is far below the target, so we should improve prediction quality while keeping the same “classical threshold → morphology cleanup → (optional) CRF/fallback → RLE” core flow intact. The biggest safe gain with minimal logic change is to tune the initial mask creation: in this competition salt is generally brighter than sediment, so using `img > threshold` (instead of `<`) and a tiny amount of denoising before Otsu typically fixes inverted masks and raises IoU substantially. I also make RLE decode consistent with the required Fortran (column-major) convention to avoid any subtle orientation mismatches, and I keep the same refinement and CSV-writing behavior. These changes are small, fast, and directly targeted at moving the score toward 0.82.'
- What this solution (achieved 0.0006) has done: 'Your current score (0.0681) is far below the target (0.8203), so we should make small, metric-aligned improvements that meaningfully increase IoU without changing the overall “threshold → morphology → refinement → RLE” flow. The biggest low-risk gain is to tune the binarization threshold: using a fixed Otsu split can be brittle, so we keep Otsu but apply a small bias and automatically choose the bias that yields a reasonable mask area (avoids near-empty/all-full masks). We also add a very small hole-filling step (still just morphology) which often boosts IoU for salt blobs. Finally, we ensure strict RLE correctness (already Fortran-order) and keep the same CSV output.'
- What this solution (achieved 0.0525) has done: 'I fix the runtime error by guaranteeing `initial_predict_mask()` always returns a valid (all-zero) mask even if no candidate threshold is selected, and by making the threshold-candidate loop robust to any unexpected morphology failures. I also add a small safety fallback for any weird/constant images to avoid NaNs and ensure `best_m` is never `None`. These changes are minimal, keep the same “Otsu → small bias search → morphology cleanup → CRF/fallback → RLE” core logic, and should restore end-to-end execution and produce a valid `crf_correction.csv` submission. With the code actually running and producing non-empty masks where appropriate, the score should increase from ~0.0006 toward the target band.'
- What this solution (achieved 0.03) has done: 'Your current score is far below the target, so the most direct way to move it upward without changing the overall “threshold → morphology cleanup → CRF/fallback → RLE” pipeline is to make the initial binarization less brittle and more salt-specific. I keep the same Otsu+bias sweep and morphology steps, but (1) add a second polarity option (salt can appear darker in some tiles), (2) replace the fixed “aim for 25% mask” heuristic with a small, depth-aware expected-area prior using the provided `depths.csv` (still just choosing among your existing candidates), and (3) add a tiny border-crop before scoring/cleanup to reduce edge artifacts from padding/contrast. These are minimal, fast, dependency-free changes that typically raise IoU materially while preserving your core logic and submission semantics. The script still run end-to-end and write `crf_correction.csv` with `id,rle_mask`.'
- What this solution (achieved 0.028) has done: 'Your current 0.03 score suggests the masks are still largely mis-calibrated (often too empty/full) and the candidate selection prior is not well aligned with this dataset, so we make small, metric-aligned adjustments without changing the overall “Otsu → small bias sweep → morphology cleanup → refinement → RLE” pipeline. I keep the same structure but (1) expand the Otsu bias sweep slightly and score candidates using a weak confidence term (how far pixels are from threshold) plus the same depth-based area prior, which better selects plausible masks while staying within your existing logic. (2) I make the morphology min_size depth-aware (deeper images tend to have larger contiguous salt), which is a minimal tweak to an existing step. (3) I add a tiny test-time augmentation (horizontal flip) and average predictions before binarization; this keeps the same thresholding/morphology/refinement approach but typically improves IoU materially for segmentation competitions.'
- What this solution (achieved 0.0321) has done: 'Your current score (0.028) is far below the target (0.8203), so we should make small but high-impact fixes that improve true IoU without changing the overall “classical threshold → morphology cleanup → refinement → RLE” pipeline. The biggest likely issue is that the current “flip-TTA” is effectively a no-op and the polarity/area prior can still select badly calibrated masks; we keep the same candidate-sweep logic but score candidates using a more metric-aligned heuristic (favor fewer fragmented components and more confident separation) while keeping the same Otsu+bias approach. We also add a very small, safe test-time augmentation that actually changes the input (90° rotations) and averages soft probabilities before thresholding (still the same thresholding pipeline). Finally, we add a single global threshold calibration on a small training subset to pick one binarization threshold multiplier for the whole run (no architecture/training; just choosing a threshold scalar), which typically provides a big jump versus untuned Otsu on this competition.'
- What this solution (achieved 0.032) has done: 'Your score is far below the target, so we should make a small, high-impact correction that improves mask orientation/geometry without changing your core “Otsu(+bias) → morphology → refinement → RLE” pipeline. The biggest likely silent bug is a mismatch between mask coordinate conventions and the evaluation’s RLE convention; we enforce a single, consistent Fortran-order convention end-to-end and add a minimal sanity check on a few training samples to auto-select whether a transpose is needed (this is a tiny post-processing choice, not a new model). We also calibrate the final binarization threshold used for encoding by searching a small set of global probability cutoffs on a training subset (still no training; just choosing a cutoff for the same predicted soft map). These changes keep your architecture/approach intact, run fast, and are directly aimed at raising IoU/precision toward the 0.82 target.'
- What this solution (achieved 0.0321) has done: 'Your very low score (0.032 vs target 0.8203) strongly suggests a fundamental mismatch rather than just “weak model”: the most common culprit in TGS is an RLE orientation/indexing inconsistency (transpose and/or flatten order) between train-mask convention and what’s encoded for submission. I keep your exact “Otsu(+bias sweep) → morphology → CRF/fallback → RLE” pipeline, but (1) replace the current train-IoU-based transpose/cutoff calibration with a metric-aligned mAP@IoU-thresholds calibration on a small train subset (still just choosing among identity/transpose and a cutoff), and (2) add a single optional flip (left-right) decision in that same calibration (again only a post-processing alignment choice). This is a minimal change that directly targets the Kaggle evaluation semantics and should move the score sharply upward without changing your core prediction logic or adding dependencies. The script still runs end-to-end and writes `crf_correction.csv` with `id,rle_mask`.'
- What this solution (achieved 0.0321) has done: 'Your current score is far below the target, so the most likely issue is a fundamental train/test geometry mismatch that the current calibration doesn’t fully cover. I keep your exact “Otsu(+scale) → bias sweep → morphology → refinement → RLE” flow, but extend the alignment calibration to also consider vertical flip and 180° rotation (still just post-processing alignment choices). I also fix a small but impactful bug where the Otsu scale calibration claims to optimize mAP but actually optimizes plain IoU; switching it to the competition mAP improves parameter selection without changing the prediction core. These are minimal changes aimed at moving the score sharply upward toward the 0.82 target while keeping runtime under the limit and producing the same `crf_correction.csv` format.'
- What this solution (achieved 0.0321) has done: 'Your current score (0.0321) is far below the target (0.8203), which strongly indicates a fundamental issue in the prediction→encoding pipeline rather than “small tuning.” I keep your exact Otsu(+scale)+bias-sweep → morphology → CRF/fallback → alignment → RLE flow, but fix two high-impact, minimal bugs: (1) ensure the submission encoding uses the *post-alignment* mask shape/convention consistently (including an optional transpose/rotation that can silently mismatch), and (2) calibrate the *final cutoff* on the refined **soft** map (not on a mostly-binary mask), so the cutoff search actually has meaning. I also make the calibration objective consistently use the competition’s mAP@IoU-thresholds everywhere (you already added it, but we apply it to the same post-processing path used at test time). These are small, metric-aligned corrections that should move the score sharply upward if the main issue is convention/cutoff miscalibration.'

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
from skimage.measure import label
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

train_images_path = os.path.join(BASE_INPUT, "train", "images")
train_masks_path = os.path.join(BASE_INPUT, "train", "masks")
train_csv_path = os.path.join(BASE_INPUT, "train.csv")

assert os.path.exists(
    sample_sub_path
), f"Missing sample submission at: {sample_sub_path}"
assert os.path.isdir(test_path), f"Missing test images dir at: {test_path}"
assert os.path.exists(depths_path), f"Missing depths at: {depths_path}"

print("Sample submission:", sample_sub_path)
print("Depths:", depths_path)
print("Test images dir:", test_path)
print("Num test images:", len([f for f in os.listdir(test_path) if f.endswith(".png")]))
print(
    "Has train images:",
    os.path.isdir(train_images_path),
    "Has train masks:",
    os.path.isdir(train_masks_path),
)



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
Minimal, score-relevant fixes (keep core logic: Otsu -> bias sweep -> morphology -> refinement -> RLE):

Fix 1 (high-impact, minimal): Calibrate the final encode cutoff on a *soft* refined map.
- Previously, `crf()` fallback returns a near-binary mask, so searching cutoffs like 0.35..0.65 is
  effectively meaningless (most cutoffs yield the same binary result). We keep the same CRF/fallback
  call, but for calibration/prediction we convert the refined mask into a soft map via a tiny Gaussian
  blur (still post-processing), so cutoff calibration and final binarization are actually effective.

Fix 2 (consistency): Use the exact same post-processing path in calibration as in test prediction.
- We ensure alignment is applied before cutoff and before scoring (already mostly true), and we keep
  the mAP@IoU-thresholds objective.

These do not change the overall approach; they correct cutoff calibration and make it operative.
"""


def _z_norm(img_id):
    z = depth_map.get(str(img_id), None)
    if z is None:
        return 0.5
    zn = (float(z) - z_min) / (z_max - z_min + 1e-12)
    return float(np.clip(zn, 0.0, 1.0))


def _expected_frac_from_depth(img_id):
    zn = _z_norm(img_id)
    return float(0.30 - 0.16 * zn)  # ~[0.14..0.30]


def _min_size_from_depth(img_id):
    zn = _z_norm(img_id)
    return int(np.clip(10 + 18 * zn, 10, 28))


def _post_morph(m_bool, img_id=None):
    m_bool = binary_opening(m_bool, disk(1))
    m_bool = binary_closing(m_bool, disk(1))
    m_bool = binary_fill_holes(m_bool)
    min_sz = _min_size_from_depth(img_id) if img_id is not None else 10
    m_bool = remove_small_objects(m_bool, min_size=min_sz)
    return m_bool


def _normalize_img(orig_img):
    if orig_img.ndim == 3:
        img = orig_img[..., 0]
    else:
        img = orig_img
    img = img.astype(np.float32)
    mn = float(np.min(img))
    mx = float(np.max(img))
    if mx > mn:
        img = (img - mn) / (mx - mn)
        return img
    return None


def _rot90(img, k):
    return np.rot90(img, k=k)


def _inv_rot90(img, k):
    return np.rot90(img, k=(4 - (k % 4)) % 4)


def _soft_from_img(img_norm):
    return gaussian(img_norm, sigma=0.75, preserve_range=True).astype(np.float32)


def _tta_soft_map(img_norm):
    acc = np.zeros_like(img_norm, dtype=np.float32)
    for k in (0, 1, 2, 3):
        x = _rot90(img_norm, k)
        s = _soft_from_img(x)
        acc += _inv_rot90(s, k)
    return (acc / 4.0).astype(np.float32)


def _candidate_score(m2, frac, expected_frac, conf, img_id):
    if m2.any():
        n_comp = int(label(m2, connectivity=1).max())
    else:
        n_comp = 0
    frag_pen = 0.03 * max(0, n_comp - 1)
    score = -abs(frac - expected_frac) + 0.20 * conf - frag_pen
    if frac < 0.002 or frac > 0.998:
        score -= 10.0
    return float(score)


def iou_np(y_true, y_pred):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    inter = np.sum((y_true == 1) & (y_pred == 1))
    union = np.sum((y_true == 1) | (y_pred == 1))
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def ap_iou_thresholds_single(y_true, y_pred):
    """
    TGS Salt metric for a single image with a single object mask:
    precision at each threshold is 1 if IoU > t else 0, averaged over thresholds.
    """
    iou = iou_np(y_true, y_pred)
    thresholds = np.arange(0.5, 1.0, 0.05)
    return float(np.mean(iou > thresholds))


def _calibrate_otsu_scale(n_samples=200):
    if not (
        os.path.isdir(train_images_path)
        and os.path.isdir(train_masks_path)
        and os.path.exists(train_csv_path)
    ):
        print("Calibration skipped (missing train files). Using scale=1.0")
        return 1.0

    train_df = pd.read_csv(train_csv_path)
    train_df["id"] = train_df["id"].astype(str)
    ids = train_df["id"].values
    if len(ids) == 0:
        print("Calibration skipped (empty train.csv). Using scale=1.0")
        return 1.0

    idx = np.arange(len(ids))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(idx)
    ids_sub = ids[idx[: min(n_samples, len(ids))]]

    scales = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
    scores = {s: [] for s in scales}

    for img_id in tqdm(
        ids_sub, desc="Calibrating Otsu scale (mAP)", total=len(ids_sub)
    ):
        ip = os.path.join(train_images_path, f"{img_id}.png")
        mp = os.path.join(train_masks_path, f"{img_id}.png")
        if (not os.path.exists(ip)) or (not os.path.exists(mp)):
            continue

        img = imread(ip)
        mgt = imread(mp)
        img_n = _normalize_img(img)
        if img_n is None:
            continue

        soft = _tta_soft_map(img_n)
        try:
            t0 = float(threshold_otsu(soft))
            if not np.isfinite(t0):
                t0 = 0.5
        except Exception:
            t0 = 0.5

        y_true = (mgt > 127).astype(np.uint8)

        for s in scales:
            t = float(np.clip(s * t0, 0.0, 1.0))
            y_pred = (soft > t).astype(bool)
            y_pred = _post_morph(y_pred, img_id=img_id).astype(np.uint8)
            scores[s].append(ap_iou_thresholds_single(y_true, y_pred))

    best_s = 1.0
    best_mean = -1.0
    best_n = 0
    for s in scales:
        if len(scores[s]) == 0:
            continue
        m = float(np.mean(scores[s]))
        if m > best_mean:
            best_mean = m
            best_s = s
            best_n = len(scores[s])

    print(
        f"Chosen Otsu scale={best_s} (mean mAP on subset={best_mean:.4f}, n={best_n})"
    )
    return float(best_s)


OTSU_SCALE = _calibrate_otsu_scale(n_samples=200)


def initial_predict_mask(orig_img, img_id=None):
    img = _normalize_img(orig_img)
    if img is None:
        return np.zeros((101, 101), dtype=np.uint8)

    soft = _tta_soft_map(img)

    try:
        t0 = float(threshold_otsu(soft))
        if not np.isfinite(t0):
            t0 = 0.5
    except Exception:
        t0 = 0.5

    t0 = float(np.clip(OTSU_SCALE * t0, 0.0, 1.0))

    candidates_t = [t0 + b for b in (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06)]
    expected_frac = _expected_frac_from_depth(img_id) if img_id is not None else 0.20

    best_m = None
    best_score = -1e18

    H, W = soft.shape
    y0, y1 = 1, H - 1
    x0, x1 = 1, W - 1
    if y1 <= y0 or x1 <= x0:
        y0, y1, x0, x1 = 0, H, 0, W

    for t in candidates_t:
        t = float(np.clip(t, 0.0, 1.0))

        for polarity in (1, -1):
            if polarity == 1:
                m = soft > t
            else:
                m = soft < t

            try:
                mc = m[y0:y1, x0:x1]
                mc = _post_morph(mc, img_id=img_id)
                m2 = np.zeros_like(m, dtype=bool)
                m2[y0:y1, x0:x1] = mc
            except Exception:
                continue

            frac = float(m2.mean())

            if m2.any():
                if polarity == 1:
                    conf = float(np.mean(np.clip(soft[m2] - t, 0.0, 1.0)))
                else:
                    conf = float(np.mean(np.clip(t - soft[m2], 0.0, 1.0)))
            else:
                conf = 0.0

            score = _candidate_score(m2, frac, expected_frac, conf, img_id)

            if score > best_score:
                best_score = score
                best_m = m2

    if best_m is None:
        return np.zeros_like(soft, dtype=np.uint8)

    return best_m.astype(np.uint8)


def _apply_alignment(
    mask2d, transpose=False, flip_lr=False, flip_ud=False, rot180=False
):
    m = mask2d.T if transpose else mask2d
    if rot180:
        m = np.rot90(m, 2)
    if flip_lr:
        m = np.fliplr(m)
    if flip_ud:
        m = np.flipud(m)
    return m


def _refined_soft_map(orig_img, init_mask):
    """
    Keep using your CRF/fallback, but turn its output into a soft map so that
    cutoff calibration is meaningful and stable.
    """
    refined = crf(orig_img, init_mask).astype(np.float32)
    refined_soft = gaussian(refined, sigma=0.6, preserve_range=True).astype(np.float32)
    mn, mx = float(refined_soft.min()), float(refined_soft.max())
    if mx > mn:
        refined_soft = (refined_soft - mn) / (mx - mn)
    else:
        refined_soft = np.clip(refined_soft, 0.0, 1.0)
    return refined_soft


def _choose_alignment_and_encode_cutoff_map(n_samples=120):
    """
    Calibrate using competition-like mAP@IoU thresholds.
    Considers transpose + flips + 180-degree rotation as small alignment choices.
    Calibrates cutoff on a soft refined map (so cutoffs are not degenerate).
    """
    if not (
        os.path.isdir(train_images_path)
        and os.path.isdir(train_masks_path)
        and os.path.exists(train_csv_path)
    ):
        print(
            "Alignment/cutoff calibration skipped (missing train files). Using transpose=False, flip_lr=False, flip_ud=False, rot180=False, cutoff=0.5"
        )
        return False, False, False, False, 0.5

    train_df = pd.read_csv(train_csv_path)
    train_df["id"] = train_df["id"].astype(str)
    ids = train_df["id"].values
    if len(ids) == 0:
        print(
            "Alignment/cutoff calibration skipped (empty train.csv). Using transpose=False, flip_lr=False, flip_ud=False, rot180=False, cutoff=0.5"
        )
        return False, False, False, False, 0.5

    idx = np.arange(len(ids))
    np.random.seed(RANDOM_SEED + 1)
    np.random.shuffle(idx)
    ids_sub = ids[idx[: min(n_samples, len(ids))]]

    cutoffs = [0.30, 0.40, 0.50, 0.60, 0.70]
    opts = [
        (tr, flr, fud, r180, c)
        for tr in (False, True)
        for flr in (False, True)
        for fud in (False, True)
        for r180 in (False, True)
        for c in cutoffs
    ]
    scores = {opt: [] for opt in opts}

    for img_id in tqdm(
        ids_sub, desc="Calibrating alignment+cutoff (mAP)", total=len(ids_sub)
    ):
        ip = os.path.join(train_images_path, f"{img_id}.png")
        mp = os.path.join(train_masks_path, f"{img_id}.png")
        if (not os.path.exists(ip)) or (not os.path.exists(mp)):
            continue

        orig_img = imread(ip)
        y_true = (imread(mp) > 127).astype(np.uint8)

        init_mask = initial_predict_mask(orig_img, img_id=img_id)
        refined_soft = _refined_soft_map(orig_img, init_mask)

        for tr, flr, fud, r180, c in opts:
            pred0 = _apply_alignment(
                refined_soft, transpose=tr, flip_lr=flr, flip_ud=fud, rot180=r180
            )
            y_pred = (pred0 > c).astype(np.uint8)
            scores[(tr, flr, fud, r180, c)].append(
                ap_iou_thresholds_single(y_true, y_pred)
            )

    best = (False, False, False, False, 0.5)
    best_mean = -1.0
    best_n = 0
    for key, vals in scores.items():
        if len(vals) == 0:
            continue
        m = float(np.mean(vals))
        if m > best_mean:
            best_mean = m
            best = key
            best_n = len(vals)

    print(
        f"Chosen transpose={best[0]}, flip_lr={best[1]}, flip_ud={best[2]}, rot180={best[3]}, encode_cutoff={best[4]} "
        f"(mean mAP on subset={best_mean:.4f}, n={best_n})"
    )
    return bool(best[0]), bool(best[1]), bool(best[2]), bool(best[3]), float(best[4])


NEED_TRANSPOSE, NEED_FLIP_LR, NEED_FLIP_UD, NEED_ROT180, ENCODE_CUTOFF = (
    _choose_alignment_and_encode_cutoff_map(n_samples=120)
)

for i in tqdm(range(df.shape[0]), total=df.shape[0], desc="Predicting test masks"):
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

    refined_soft = _refined_soft_map(orig_img, init_mask)

    refined_soft = _apply_alignment(
        refined_soft,
        transpose=NEED_TRANSPOSE,
        flip_lr=NEED_FLIP_LR,
        flip_ud=NEED_FLIP_UD,
        rot180=NEED_ROT180,
    )

    refined_bin = (refined_soft > ENCODE_CUTOFF).astype(np.uint8)
    df.at[i, "rle_mask"] = rle_encode(refined_bin)



## === cell 8
df["rle_mask"] = df["rle_mask"].fillna("").astype(str)
df = df[["id", "rle_mask"]]

out_path = "crf_correction.csv"
df.to_csv(out_path, index=False)
print("Wrote submission:", out_path)
print(df.head())
print("Rows:", len(df), "Empty masks:", (df["rle_mask"] == "").sum())
