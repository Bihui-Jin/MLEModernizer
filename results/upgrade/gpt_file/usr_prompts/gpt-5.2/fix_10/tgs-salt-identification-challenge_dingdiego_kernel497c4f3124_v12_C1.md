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

0.128

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `pydensecrf` (not installed in this environment) by providing a safe fallback that leaves masks unchanged when CRF is unavailable, so the notebook runs end-to-end. I also remove notebook-only magic (`%matplotlib inline`) and any references to a non-existent input path (`../input/baseline-v4/submission.csv`) by instead reading `sample_submission.csv` from the provided competition dataset folder. Finally, I ensure all required imports exist in the execution cells, avoid NameErrors from earlier failed cells, and always write a valid `submission.csv` (with `.csv` suffix and the correct `id,rle_mask` columns).'
- What this solution (achieved 0.0261) has done: 'Your current 0.0 score is because the code never produces meaningful masks: it only re-encodes the *sample_submission* masks (which are placeholders), so predictions are effectively empty/garbage. To move the score upward toward your target without changing the “core approach” (still a no-model post-processing pipeline that outputs one mask per test image), the smallest legitimate improvement is to generate a basic foreground mask from each test image itself and RLE-encode it (instead of decoding/CRF’ing the placeholder). I keep your RLE helpers and optional CRF hook intact, but switch the loop to: load image → normalize → simple threshold-based mask → optional CRF → encode → write submission. This produce non-empty, image-dependent predictions and should increase the score from 0.0 toward your target band while staying lightweight and within time.'
- What this solution (achieved 0.0461) has done: 'Your current score is very far below the target, and the biggest issue is that the prediction masks are essentially arbitrary (global Otsu + optional CRF), which won’t correlate well with salt regions and yields near-random mAP. To move toward your target without changing the “core approach” (still a lightweight, no-training heuristic mask from the test image itself), I make the mask generation slightly more faithful to the competition’s typical baseline: normalize/contrast-stabilize the image, use an intensity threshold plus a small post-processing cleanup (remove tiny speckles + fill tiny holes), and also choose between “salt=dark” vs “salt=bright” using a simple center-prior heuristic. These are minimal, deterministic changes that keep the pipeline structure intact (load → basic mask → optional CRF → RLE) but should materially increase overlap quality and raise the score toward your target band. The submission writing, paths, and RLE format remain unchanged.'
- What this solution (achieved 0.0) has done: 'Your current gap to the target is large (>30%), so the minimal way to move the score upward is to keep your same “no-training heuristic” pipeline but make the binary mask more aligned with typical salt structure: add a very lightweight depth-dependent prior (salt likelihood changes with depth), and add a second simple edge/texture cue (local contrast) to avoid segmenting smooth background. I also fix the RLE encoding flatten order to the competition’s expected top-to-bottom then left-to-right (Fortran order), which often has a big impact on score even if masks look fine visually. These changes preserve your overall logic (load image → build heuristic mask → optional CRF → RLE → write submission) and stay deterministic and fast. The script still write a valid `submission.csv` with `id,rle_mask`.'
- What this solution (achieved 0.0516) has done: 'I fix the depth merge failure by loading depths and sample submission from the same root and by making the merge robust (falling back to global median depth if any IDs are missing). This also fixes the downstream `NameError` for `z_min/z_denom` by ensuring those globals are always defined before mask generation runs. The rest of your heuristic pipeline (load image → threshold/cleanup → optional CRF → RLE in Fortran order → write CSV) is kept intact so behavior changes are only what’s needed to run end-to-end and produce meaningful predictions.'
- What this solution (achieved 0.0541) has done: 'Your current score (0.0516) is far below the target (0.7727), so we should improve overlap quality without changing the overall “no-training heuristic” pipeline. The biggest low-risk gain here is to fix the RLE decode/encode convention mismatch: you already encode in Fortran order (correct), but your decoder reshapes in C order (incorrect), and aligning both to the competition’s convention helps ensure any internal mask operations are consistent. Next, we add a tiny, deterministic “largest connected component” cleanup after your existing morphology to reduce false positives (a common mAP killer) while preserving your threshold+morphology core logic. Finally, we keep CRF as a no-op fallback and keep all paths/output format unchanged while making the per-image loop slightly safer and faster via `.at` assignments.'
- What this solution (achieved 0.0765) has done: 'Your current score is far below the target, so we should improve mask quality while keeping your same overall heuristic pipeline (load image → threshold/morphology → optional CRF → RLE). The lowest-risk, score-relevant fix is to make post-processing less aggressive: your “contrast gate” and “keep only largest connected component” are likely deleting true salt regions and causing many false negatives. I (1) soften the contrast gate (use it only to suppress obvious flat background, not as a hard AND), (2) only apply “keep largest CC” when the prediction is clearly too fragmented, and (3) add a light border-clear step to reduce edge noise without erasing interior salt. These are minimal changes that preserve your core logic and should move mAP upward toward the target band.'
- What this solution (achieved 0.0476) has done: 'Your current score is far below the target, so we should improve mask quality while keeping the same overall heuristic pipeline (load test image → Otsu-based mask → light morphology → optional CRF → RLE). The smallest high-impact fix is to correct the CRF input: your `crf()` currently expects a 3‑channel “annotated label image”, but you pass a single-channel binary mask, so CRF (when available) would be operating on essentially arbitrary labels; we instead convert the binary mask into a proper 2-color RGB annotation before calling CRF. Next, we make the contrast suppression less destructive by applying it after morphology and only as a weak cleanup, reducing false negatives. These changes preserve the core logic and evaluation semantics while typically increasing overlap quality and mAP.'
- What this solution (achieved 0.128) has done: 'Your current score (0.0476) is far below the target (0.7727), so we should improve mask overlap while keeping the same overall heuristic pipeline (normalize → Otsu threshold → light morphology → optional CRF → RLE). The most score-relevant minimal fix is to use the provided `train/` masks to calibrate a single global binarization threshold and polarity (salt-as-bright vs salt-as-dark) on a small held-out split, rather than guessing per-image; this preserves the same “threshold+morphology” core logic but makes it data-aligned. Then we apply that calibrated threshold to test images (still Otsu-like thresholding logic, just with a learned constant), leaving your morphology, depth prior, CRF hook, and RLE encoding unchanged. This should materially increase mAP and move you toward the target band without changing the model/training architecture (there is still no ML model, just threshold calibration).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

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
from skimage.filters.rank import gradient
from skimage.morphology import square
from skimage.measure import label

import matplotlib.pyplot as plt
from tqdm import tqdm

CRF_AVAILABLE = False
try:
    import pydensecrf.densecrf as dcrf  # type: ignore
    from pydensecrf.utils import unary_from_labels  # type: ignore

    CRF_AVAILABLE = True
except Exception:
    CRF_AVAILABLE = False

BASE = "/kaggle/input/tgs-salt-identification-challenge"
if not os.path.exists(BASE):
    BASE = "../input/tgs-salt-identification-challenge"

TEST_IMG_DIR = os.path.join(BASE, "test", "images")
TRAIN_IMG_DIR = os.path.join(BASE, "train", "images")
TRAIN_MASK_DIR = os.path.join(BASE, "train", "masks")

SAMPLE_SUB_PATH = os.path.join(BASE, "sample_submission.csv")
DEPTHS_PATH = os.path.join(BASE, "depths.csv")

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission at {SAMPLE_SUB_PATH}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test image dir at {TEST_IMG_DIR}"
assert os.path.exists(DEPTHS_PATH), f"Missing depths at {DEPTHS_PATH}"

HAS_TRAIN = os.path.isdir(TRAIN_IMG_DIR) and os.path.isdir(TRAIN_MASK_DIR)




## === cell 1
def rle_decode(rle_mask, shape=(101, 101)):
    """
    Decode reshapes in Fortran order to match Kaggle's pixel ordering
    (top-to-bottom, then left-to-right) and our encoder.
    """
    if rle_mask is None:
        return np.zeros(shape, dtype=np.uint8)
    if isinstance(rle_mask, float) and np.isnan(rle_mask):
        return np.zeros(shape, dtype=np.uint8)
    rle_mask = str(rle_mask).strip()
    if rle_mask == "" or rle_mask.lower() == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")




## === cell 2
def rle_encode(im):
    """
    Kaggle TGS expects pixels ordered top-to-bottom then left-to-right => Fortran order flatten.
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
def crf(original_image, mask_img):
    """
    Return labelled image after applying CRF.
    If pydensecrf isn't available, return the input mask unchanged.
    """
    if not CRF_AVAILABLE:
        return (mask_img > 0).astype(np.uint8)

    if mask_img.ndim == 2:
        m = (mask_img > 0).astype(np.uint8)
        mask_rgb = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
        mask_rgb[m == 1] = np.array([255, 255, 255], dtype=np.uint8)
        mask_img = mask_rgb
    elif mask_img.shape[-1] != 3:
        mask_img = gray2rgb(mask_img)

    annotated_label = (
        mask_img[:, :, 0] + (mask_img[:, :, 1] << 8) + (mask_img[:, :, 2] << 16)
    )
    _, labels = np.unique(annotated_label, return_inverse=True)

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
df = pd.read_csv(SAMPLE_SUB_PATH)
depths = pd.read_csv(DEPTHS_PATH)

assert (
    "id" in df.columns and "rle_mask" in df.columns
), f"Unexpected submission columns: {df.columns.tolist()}"
df["rle_mask"] = df["rle_mask"].astype(object)

df = df.merge(depths, on="id", how="left")

if not df["z"].notnull().all():
    med_z = float(depths["z"].median())
    df["z"] = df["z"].fillna(med_z)

missing_imgs = []
for _id in df["id"].head(5).tolist():
    if not os.path.exists(os.path.join(TEST_IMG_DIR, f"{_id}.png")):
        missing_imgs.append(_id)
if missing_imgs:
    raise FileNotFoundError(f"Some sample ids not found in test images: {missing_imgs}")

z_min, z_max = float(depths["z"].min()), float(depths["z"].max())
z_denom = (z_max - z_min) if (z_max > z_min) else 1.0




## === cell 5
def _norm_image(g):
    g = g.astype(np.float32)
    if g.max() > 1.0:
        g = g / 255.0
    lo, hi = np.percentile(g, (2, 98))
    if hi > lo:
        g = (g - lo) / (hi - lo)
    return np.clip(g, 0.0, 1.0)


def _keep_largest_cc(mask_bool):
    lab = label(mask_bool, connectivity=1)
    if lab.max() == 0:
        return mask_bool
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    largest = counts.argmax()
    return lab == largest


def _clear_border(mask_bool, border=1):
    if border <= 0:
        return mask_bool
    m = mask_bool.copy()
    m[:border, :] = False
    m[-border:, :] = False
    m[:, :border] = False
    m[:, -border:] = False
    return m


def _iou_bin(pred, true):
    pred = pred.astype(bool)
    true = true.astype(bool)
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def _mean_ap_iou(pred, true):
    iou = _iou_bin(pred, true)
    thresholds = np.arange(0.5, 1.0, 0.05)
    return float(np.mean(iou > thresholds))


def _cleanup_mask_from_binary(m, very_flat=None):
    m = _clear_border(m, border=1)
    se = disk(1)

    m = binary_opening(m, se)
    m = binary_closing(m, se)

    m = remove_small_objects(m, min_size=14)
    m = remove_small_holes(m, area_threshold=14)

    if very_flat is not None:
        m2 = m & (~very_flat)
        if m2.sum() >= int(0.85 * m.sum()):
            m = m2

    lab = label(m, connectivity=1)
    n_comp = int(lab.max())
    if n_comp >= 6:
        m = _keep_largest_cc(m)

    return m


CALIB_T = None
CALIB_POLARITY = None  # "bright" or "dark"
if HAS_TRAIN:
    train_ids = [fn[:-4] for fn in os.listdir(TRAIN_IMG_DIR) if fn.endswith(".png")]
    train_ids = sorted(train_ids)
    if len(train_ids) > 0:
        rs = np.random.RandomState(1337)
        calib_n = min(160, len(train_ids))
        calib_ids = rs.choice(train_ids, size=calib_n, replace=False).tolist()

        t_grid = np.linspace(0.30, 0.70, 17, dtype=np.float32)

        best = (-1.0, None, None)  # (score, t, pol)
        for t in t_grid:
            for pol in ("bright", "dark"):
                scores = []
                for _id in calib_ids:
                    img = imread(os.path.join(TRAIN_IMG_DIR, f"{_id}.png"))
                    msk = imread(os.path.join(TRAIN_MASK_DIR, f"{_id}.png"))

                    g = img[:, :, 0] if img.ndim == 3 else img
                    g = _norm_image(g)

                    g8 = (g * 255.0).astype(np.uint8)
                    local_contrast = gradient(g8, square(3))
                    very_flat = local_contrast <= 2

                    raw = (g > t) if pol == "bright" else (g < t)
                    pred = _cleanup_mask_from_binary(raw, very_flat=very_flat).astype(
                        np.uint8
                    )
                    true = (msk > 0).astype(np.uint8)
                    scores.append(_mean_ap_iou(pred, true))
                s = float(np.mean(scores)) if scores else -1.0
                if s > best[0]:
                    best = (s, float(t), pol)

        CALIB_T = best[1]
        CALIB_POLARITY = best[2]




## === cell 6
def make_basic_mask_from_image(orig_img, z_value):
    """
    Core logic preserved: threshold-based mask + small morphology + optional depth prior.
    Change (score-relevant, minimal): if calibration is available, use a single calibrated
    threshold & polarity rather than per-image polarity guessing, which is unstable and
    scores poorly.
    """
    if orig_img.ndim == 3:
        g = orig_img[:, :, 0]
    else:
        g = orig_img

    g = _norm_image(g)

    g8 = (g * 255.0).astype(np.uint8)
    local_contrast = gradient(g8, square(3))
    very_flat = local_contrast <= 2

    if (CALIB_T is not None) and (CALIB_POLARITY is not None):
        z_norm = (float(z_value) - z_min) / z_denom
        t_eff = float(np.clip(CALIB_T + 0.03 * (z_norm - 0.5), 0.0, 1.0))
        raw = (g > t_eff) if CALIB_POLARITY == "bright" else (g < t_eff)
        chosen = _cleanup_mask_from_binary(raw, very_flat=very_flat)
        return chosen.astype(np.uint8)

    t = threshold_otsu(g)
    bright = g > t
    dark = ~bright

    bright_c = _cleanup_mask_from_binary(bright, very_flat=very_flat)
    dark_c = _cleanup_mask_from_binary(dark, very_flat=very_flat)

    c0, c1 = 101 // 2, 101 // 2
    r = 18
    center = (slice(c0 - r, c0 + r + 1), slice(c1 - r, c1 + r + 1))
    score_b = float(bright_c[center].mean())
    score_d = float(dark_c[center].mean())

    z_norm = (float(z_value) - z_min) / z_denom
    score_d += 0.10 * z_norm
    score_b += 0.10 * (1.0 - z_norm)

    chosen = bright_c if score_b >= score_d else dark_c
    return chosen.astype(np.uint8)




## === cell 7
try:
    print(
        f"HAS_TRAIN={HAS_TRAIN}, CALIB_T={CALIB_T}, CALIB_POLARITY={CALIB_POLARITY}, CRF_AVAILABLE={CRF_AVAILABLE}"
    )
    plt.figure(figsize=(12, 3))
    for k in range(min(4, len(df))):
        _id = df.loc[k, "id"]
        z = df.loc[k, "z"]
        img_path = os.path.join(TEST_IMG_DIR, f"{_id}.png")
        orig_img = imread(img_path)
        basic_mask = make_basic_mask_from_image(orig_img, z)
        plt.subplot(1, 4, k + 1)
        plt.imshow(basic_mask, cmap="gray")
        plt.axis("off")
        plt.title(f"{_id}\nz={z:.1f}")
    plt.tight_layout()
except Exception:
    pass



## === cell 8
for i in tqdm(range(df.shape[0])):
    _id = df.at[i, "id"]
    z = df.at[i, "z"]
    img_path = os.path.join(TEST_IMG_DIR, f"{_id}.png")
    orig_img = imread(img_path)

    basic_mask = make_basic_mask_from_image(orig_img, z)
    crf_output = crf(orig_img, basic_mask)
    df.at[i, "rle_mask"] = rle_encode(crf_output)



## === cell 9
out_path = "submission.csv"
df[["id", "rle_mask"]].to_csv(out_path, index=False)

sub = pd.read_csv(out_path)
assert list(sub.columns) == ["id", "rle_mask"]
assert len(sub) == len(df)
print(
    f"Wrote {out_path} with shape {sub.shape}. CALIB_T={CALIB_T}, CALIB_POLARITY={CALIB_POLARITY}, CRF_AVAILABLE={CRF_AVAILABLE}"
)
