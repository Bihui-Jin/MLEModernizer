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

0.7821289485071389

# 6. Current score

0.076

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0673) has done: 'I remove the unavailable `pydensecrf` dependency (it is not installed in this environment) and replace the CRF step with a lightweight, installed-package post-processing that preserves the intent: refining predicted masks using the input image. I also fix the notebook-only magics (`%matplotlib inline`) and remove the dependency on an external dataset submission (`../input/baseline-.../submission.csv`) by starting from the competition’s `sample_submission.csv` so the pipeline always produces a valid `.csv` file. Finally, I make the code robust to empty/NaN RLE masks and ensure the output columns and filename match Kaggle’s required submission format.'
- What this solution (achieved 0.0673) has done: 'Your current score is far below the target, so we need a small but meaningful improvement without changing the overall “no-training, image-thresholding + morphology + RLE” core approach. The biggest issue is that `refine_mask_with_image()` currently does not refine at all (the `combined` expression always equals `m`), so I make it actually use the image prior to suppress obvious background while keeping morphology as the same style of post-processing. To better match the competition metric (IoU-sweep), I also add a very small “empty mask” gate so we don’t submit lots of tiny false positives, which usually tanks precision. These are minimal, deterministic changes and should move the score substantially toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.0703) has done: 'Your current score is far below the target, so we keep the same “no-training, threshold + morphology + RLE” pipeline but fix two score-critical issues that create many false positives/false negatives. First, we stop intersecting the predicted mask with the image Otsu foreground (which often erases true salt) and instead use the image only to *remove very obvious background* via a conservative prior. Second, we tune the “empty mask” gate to be adaptive (relative to image size) and slightly stronger at suppressing tiny predicted blobs, which typically improves mAP by reducing false positives across IoU thresholds. These changes preserve the core approach and are deterministic, while aiming to move the score significantly closer to the 0.78 target.'
- What this solution (achieved 0.0703) has done: 'Your current score (0.0703) is far below the target (0.7821), so we need a meaningful but still “same core logic” improvement: keep the no-training threshold+morphology pipeline, but fix a metric-critical mistake—your `rle_decode` is using C-order while `rle_encode` uses the competition’s Fortran-order, which is inconsistent and can devastate IoU/precision once you do any mask-based operations. I make `rle_decode` match Kaggle’s top-to-bottom-then-left-to-right (Fortran) convention to preserve evaluation semantics. Additionally, I make the “empty mask” gate slightly more robust by applying it after final cleanup and using a tiny absolute floor, which reduces tiny false positives without changing the overall method. These are minimal, deterministic changes that should move the score substantially upward toward your target without changing the modeling approach.'
- What this solution (achieved 0.0703) has done: 'Your current score (0.0703) is far below the target (0.7821), so we need a meaningful improvement while keeping the same “no-training Otsu threshold + morphology + RLE” core pipeline. The biggest score issue left is that the predicted masks are effectively segmenting generic bright regions rather than salt; a minimal but impactful fix is to incorporate the provided depth prior as a simple, deterministic adjustment to the threshold (deeper images tend to have less salt), without changing the overall approach. I also make the background-suppression step slightly less aggressive to avoid deleting true positives, and keep the existing empty-mask gating to reduce false positives across IoU thresholds. The output format and paths remain the same and a valid `submission.csv` is always written.'
- What this solution (achieved 0.0703) has done: 'Your score is far below the target, so we keep your exact no-training Otsu+depth-adjusted threshold + morphology pipeline, but fix two evaluation-critical details that can severely depress mAP. First, we ensure “empty mask” predictions are encoded as an empty string (not a non-empty RLE like `"1 0"`), because Kaggle treats any non-empty string as a predicted object and that creates lots of false positives. Second, we make RLE encoding robust by returning `""` when the mask has zero pixels, and we add a final safety cleanup to guarantee strictly valid, sorted runs (without changing the mask itself). These are minimal, deterministic changes that preserve your core logic while aiming to move the score materially upward toward the target.'
- What this solution (achieved 0.0827) has done: 'Your score is far below the target, so we keep the exact same “no-training Otsu + depth-adjusted threshold + morphology + RLE” pipeline and only make two small, score-relevant fixes. First, we replace the current background suppression in `refine_mask_with_image()` with a more appropriate *salt-likelihood prior* that keeps darker regions (salt is typically darker) instead of removing them, which should reduce systematic false negatives. Second, we add a tiny, deterministic threshold calibration (a constant offset plus a slightly stronger depth effect) to move the mask density toward better mAP without changing the overall method. All paths/output format remain the same and a valid `submission.csv` is always written.'
- What this solution (achieved 0.0794) has done: 'We keep your exact no-training Otsu+depth-threshold+morphology pipeline, but fix one score-critical inconsistency: `predict_mask_from_image()` thresholds bright pixels (`img_s > th_adj`) while `refine_mask_with_image()` keeps dark pixels (`img_s <= th*...`), so refinement often deletes what the predictor just found. I make `refine_mask_with_image()` consistent with the predictor by using the image only as a conservative “remove obvious background” gate (keep pixels that are not too dark), which should reduce systematic false negatives and move mAP upward toward your target. I also make the depth influence slightly weaker (same logic, tiny calibration) because the current +0.075*zc can over-shift thresholds and create widespread under/over-segmentation across depths. All paths, output format, and the rest of the core logic remain unchanged, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.0257) has done: 'Your score is far below the target, so we keep the exact same no-training Otsu+depth-adjusted threshold+morphology pipeline and only make two small, score-relevant calibrations. First, we correct an inversion mismatch: the predictor currently segments bright pixels, but salt in this dataset is typically darker—so we flip the threshold direction in `predict_mask_from_image()` to segment darker-than-threshold pixels (all morphology/RLE logic unchanged). Second, we make the refinement gate consistent with that direction (remove only very bright “obvious background” pixels instead of removing dark pixels), reducing systematic false negatives. These are minimal, deterministic changes that preserve the overall approach and should move mAP substantially upward toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.076) has done: 'Your current score is far below the target, so we keep your exact “no-training Otsu + depth-adjusted threshold + morphology + RLE” pipeline and make the smallest changes that most directly affect mAP: (1) fix the depth-threshold direction so deeper images become slightly *harder* to predict as salt (your current `+ 0.055 * zc` tends to over-predict salt at depth, creating many false positives), and (2) add a tiny, deterministic per-image calibration that chooses between two very close thresholds by minimizing a simple proxy for “too empty vs too full” masks (this changes only a constant threshold offset, not the segmentation method). We also keep refinement consistent and add a strict sanity clamp to avoid pathological all-ones predictions. These changes are minimal, deterministic, keep the same core logic, and should move the score upward toward your target.'
- What this solution (achieved 0.076) has done: 'Your score is still far below the target, so we keep the exact same no-training Otsu+depth-adjusted threshold + morphology + RLE pipeline and only fix a likely score-killer: the threshold calibration currently prefers a fixed 10% mask area on every image, which tends to overpredict salt and create many false positives (hurting mAP across IoU thresholds). I replace that fixed target with a tiny depth-aware target area (shallower images slightly higher expected salt coverage, deeper slightly lower) while keeping the same two-candidate threshold selection and the same morphology/refinement logic. I also add a minimal, deterministic clamp so the refinement gate can’t accidentally erase nearly everything when Otsu is unstable, which should improve consistency without changing the approach. Output paths/format remain unchanged and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.076) has done: 'Your current score (0.076) is far below the target (0.782), so we should make a small but high-impact adjustment without changing your core “Otsu + depth-adjusted threshold + morphology + RLE” approach. The biggest likely score killer now is threshold direction/mismatch: you segment “dark as salt” in `predict_mask_from_image()`, but your refinement step is still tuned around Otsu-derived brightness and can over-suppress true salt; I make refinement use the *same* threshold value as prediction (no second Otsu) so the pipeline is self-consistent. I also make the depth effect slightly less aggressive (tiny calibration, same formula) to reduce systematic under/over-segmentation across the test set. Finally, I keep your empty-mask gating but make it apply to the pre-refinement mask too, preventing refinement from “rescuing” tiny noisy blobs that hurt mAP.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_opening,
    binary_closing,
    disk,
)

from tqdm import tqdm

np.random.seed(42)




## === cell 1
def rle_decode(rle_mask, shape=(101, 101)):
    """
    rle_mask: run-length as string formatted (start length)
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    Decode in Kaggle TGS ordering (Fortran-style): top-to-bottom, then left-to-right.
    """
    if rle_mask is None:
        return np.zeros(shape, dtype=np.uint8)

    rle_mask = str(rle_mask)
    if rle_mask.strip() == "" or rle_mask.lower() == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape((shape[1], shape[0])).T.astype(np.uint8)


def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.
    Matches Kaggle TGS ordering: pixels are 1-indexed and numbered
    top-to-bottom, then left-to-right (Fortran order).

    Return "" for empty masks to avoid emitting non-empty RLE like "1 0".
    """
    im = (im > 0).astype(np.uint8)
    if im.sum() == 0:
        return ""

    pixels = im.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    if len(runs) == 0:
        return ""
    runs = runs.reshape(-1, 2)
    runs = runs[runs[:, 1] > 0]
    if runs.shape[0] == 0:
        return ""

    runs = runs.flatten()
    return " ".join(str(int(x)) for x in runs)




## === cell 2
DATA_ROOT = "../input/tgs-salt-identification-challenge"
test_path = os.path.join(DATA_ROOT, "test", "images")

sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
if not os.path.exists(sample_sub_path):
    sample_sub_path = "../input/sample_submission.csv"

df = pd.read_csv(sample_sub_path)
assert {"id", "rle_mask"}.issubset(
    df.columns
), "sample_submission.csv must have columns: id,rle_mask"

df["id"] = df["id"].astype(str)
df["rle_mask"] = df["rle_mask"].astype(object)

depths_path = os.path.join(DATA_ROOT, "depths.csv")
if not os.path.exists(depths_path):
    depths_path = "../input/depths.csv"
depths = pd.read_csv(depths_path)
depths["id"] = depths["id"].astype(str)

z = depths.set_index("id")["z"]
z_mean, z_std = float(z.mean()), float(z.std() if z.std() > 0 else 1.0)

print("Loaded sample_submission:", df.shape)
print("Loaded depths:", depths.shape)
print("Test images dir exists:", os.path.isdir(test_path))




## === cell 3
def refine_mask_with_image(original_image, mask_img, th_adj_for_pred=None):
    """
    original_image: 2D uint8 image (101x101)
    mask_img: 2D uint8/bool mask (101x101)
    th_adj_for_pred: (score-relevant, minimal) use the SAME threshold used for prediction
                     to build a conservative brightness gate; avoids a second Otsu that can
                     contradict prediction and erase true positives.

    Returns a refined binary mask (0/1).
    """
    img = original_image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    img_s = gaussian(img, sigma=1.0, preserve_range=True)

    if th_adj_for_pred is None:
        try:
            th_gate = float(threshold_otsu(img_s))
        except ValueError:
            th_gate = 0.5
    else:
        th_gate = float(th_adj_for_pred)

    gate_th = float(np.clip(th_gate * 1.25, 0.55, 0.97))
    not_too_bright = img_s <= gate_th

    m = mask_img > 0
    combined = m & not_too_bright

    combined = binary_opening(combined, footprint=disk(1))
    combined = binary_closing(combined, footprint=disk(1))
    combined = remove_small_holes(combined, area_threshold=16)
    combined = remove_small_objects(combined, min_size=16)

    return combined.astype(np.uint8)




## === cell 4
def _postprocess_binary(init_bool):
    """
    Minimal helper to keep core morphology identical while reusing it for tiny threshold calibration.
    """
    m = binary_opening(init_bool, footprint=disk(1))
    m = binary_closing(m, footprint=disk(1))
    m = remove_small_holes(m, area_threshold=16)
    m = remove_small_objects(m, min_size=16)
    return m.astype(np.uint8)


def predict_mask_from_image(original_image, z_norm=0.0):
    img = original_image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    img_s = gaussian(img, sigma=1.0, preserve_range=True)

    try:
        th = threshold_otsu(img_s)
    except ValueError:
        th = 0.5

    zc = float(np.clip(z_norm, -2.5, 2.5))

    base_th_adj = float(np.clip((th - 0.015) - 0.038 * zc, 0.05, 0.95))

    candidates = [base_th_adj - 0.010, base_th_adj + 0.010]
    candidates = [float(np.clip(c, 0.05, 0.95)) for c in candidates]

    target_area = float(np.clip(0.10 - 0.02 * zc, 0.04, 0.16))

    best = None
    best_score = None
    best_th = None
    for th_adj in candidates:
        init = _postprocess_binary((img_s < th_adj))
        area_ratio = float(init.mean())  # proxy for FP/FN balance without labels
        score = abs(area_ratio - target_area)
        if (best_score is None) or (score < best_score):
            best_score = score
            best = init
            best_th = th_adj

    init = best.astype(np.uint8)

    min_area = max(8, int(0.004 * init.size))
    if init.sum() < min_area:
        init[:] = 0

    refined = refine_mask_with_image(original_image, init, th_adj_for_pred=best_th)

    refined = remove_small_holes(refined.astype(bool), area_threshold=16)
    refined = remove_small_objects(refined.astype(bool), min_size=16)
    refined = refined.astype(np.uint8)

    if refined.mean() > 0.80:
        refined[:] = 0

    min_area2 = max(8, int(0.004 * refined.size))
    if refined.sum() < min_area2:
        refined[:] = 0

    return refined




## === cell 5
rles = []
missing = 0

for img_id in tqdm(df["id"].values, total=len(df)):
    img_file = os.path.join(test_path, f"{img_id}.png")
    if not os.path.exists(img_file):
        img_file = os.path.join("../input/test/images", f"{img_id}.png")

    if not os.path.exists(img_file):
        missing += 1
        rles.append("")
        continue

    img = imread(img_file)
    if img.ndim == 3:
        img = img[..., 0]

    zi = float(z.get(img_id, z_mean))
    z_norm = (zi - z_mean) / z_std

    pred_mask = predict_mask_from_image(img, z_norm=z_norm)

    rles.append(rle_encode(pred_mask))

print("Missing images:", missing)
df["rle_mask"] = rles



## === cell 6
out_path = "submission.csv"
df[["id", "rle_mask"]].to_csv(out_path, index=False)

print("Wrote:", out_path)
print(df.head())
print(
    "Empty masks:",
    int((df["rle_mask"].fillna("").astype(str).str.len() == 0).sum()),
    "of",
    len(df),
)
