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

0.7726698398961486

# 6. Current score

0.0335

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `pydensecrf` (not installed in this Kaggle environment) by providing a safe fallback that keeps the pipeline running and still produces a valid submission CSV. I also remove notebook-only magic (`%matplotlib inline`) and fix the `pd/plt/tqdm` “not defined” errors by ensuring imports execute and by making the baseline submission path robust (use `sample_submission.csv` if the referenced `baseline-v4/submission.csv` doesn’t exist). Finally, I fix the NaN/empty-mask checks so we don’t try to decode invalid RLE strings, and I always write a `.csv` with the required `id,rle_mask` columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is coming from an invalid submission encoding: this competition’s RLE is **column-major (Fortran order)**, but your `rle_encode`/`rle_decode` use row-major, which makes masks decode incorrectly and tanks IoU/AP. I minimally fix both functions to use Fortran order and add a safety guard for truly empty masks to emit an empty string, matching common TGS conventions. This preserves your core approach (post-processing an existing submission with optional CRF) while making the produced CSV semantically correct for the metric. Everything else (paths, loop, CRF fallback) stays the same so it runs end-to-end and writes `crf_correction.csv`.'
- What this solution (achieved 0.0996) has done: 'Your 0.0 score is consistent with producing a valid-looking CSV that is nevertheless *semantically wrong for this competition*: the sample submission contains placeholder masks (“1 1”), and your script only modifies rows where `rle_mask` is non-empty—so you end up submitting mostly (or entirely) those placeholders, which score ~0. I minimally switch the input source from `sample_submission.csv` to a real baseline if present, and if not present I instead run a simple, deterministic segmentation directly from the test images (Otsu thresholding) so every row gets a plausible mask. This keeps your core flow (read ids → decode/produce mask → optional CRF → RLE encode → write CSV) and preserves the CRF fallback behavior, but avoids the guaranteed-bad placeholder submission. I also make the test-image directory robust by checking alternate provided paths so we don’t silently skip all images.'
- What this solution (achieved 0.1007) has done: 'Your current score is far below the target, so the smallest safe way to improve is to keep your exact pipeline (Otsu → optional CRF fallback → RLE) but make the initial mask less noisy and more “salt-shaped” using lightweight, deterministic morphology on the Otsu mask. This doesn’t change the model/approach (still pure image-threshold segmentation), but it reduces scattered pixels and fills small holes, which typically increases IoU and AP for this competition. I also ensure every row is processed even if an image is missing by explicitly writing an empty mask for those ids (prevents stale placeholders from remaining). Finally, I keep your Fortran-order RLE and output format unchanged.'
- What this solution (achieved 0.1958) has done: 'Your current score (0.1007) is far below the target (0.7727), so we should improve the *same* Otsu→(light morphology)→CRF-fallback→RLE pipeline rather than change approaches. The most score-critical minimal fix here is to stop the “choose the bigger side of Otsu” heuristic, which often predicts the background as salt; instead we consistently take the darker-than-threshold side (salt is typically darker in this dataset) and add a very light border-clear step to remove frequent frame artifacts. To keep changes minimal and deterministic, I’m only adjusting the mask selection and adding `clear_border`, leaving CRF fallback, RLE (Fortran order), paths, and submission writing intact. This should move IoU/AP substantially upward toward your target without changing the overall logic or adding any new dependencies beyond scikit-image you already use.'
- What this solution (achieved 0.2124) has done: 'Your current score (0.1958) is far below the target (0.7727), so we should improve the same Otsu→(light morphology)→CRF-fallback→RLE pipeline with the smallest score-relevant fixes. The biggest issue remaining is that Otsu on raw intensity alone is weak; we can minimally add the provided depth feature to adapt the threshold slightly by depth (still just thresholding, no model/loop change) which often reduces systematic over/under-segmentation. We also make the morphology a touch more robust by switching to `binary_closing` with a slightly larger disk and adding a final `remove_small_objects` after closing (still the same cleanup idea, just better ordering). Finally, we guarantee consistent image normalization and keep Fortran-order RLE and submission formatting unchanged.'
- What this solution (achieved 0.2038) has done: 'Your current score (0.2124) is far below the target (0.7727), so we should keep your exact Otsu→(morphology)→CRF-fallback→RLE pipeline but make the thresholding a bit more discriminative without changing the approach. The smallest high-impact tweak is to pre-smooth the image slightly (Gaussian blur) before Otsu so the threshold is driven by regions rather than pixel noise, and to apply a very light contrast normalization (rescale intensity by percentiles) to reduce per-image brightness variation. I also make the depth-adaptive shift slightly stronger but still small, and add a final “empty mask guard” (predict empty if predicted salt coverage is implausibly large) which reduces catastrophic false positives that hurt mAP across IoU thresholds. All paths, CRF fallback behavior, Fortran-order RLE, and the submission schema remain unchanged, and it still runs end-to-end producing `crf_correction.csv`.'
- What this solution (achieved 0.298) has done: 'Your current gap to the target is large (0.2038 → 0.7727), so we should improve the same Otsu→morphology→CRF-fallback→RLE pipeline with the smallest changes that reliably increase mAP. The most impactful minimal adjustment is to tune the post-processing to better match typical salt mask structure: slightly stronger closing to connect regions, plus removing tiny connected components after closing, and a tighter “too-big mask” guard to reduce massive false positives that heavily hurt AP across IoU thresholds. I’m keeping your Fortran-order RLE, depth-adaptive thresholding, Gaussian+percentile normalization, and CRF fallback exactly as-is. I also switch to `tqdm(range(...), total=...)` to avoid any subtle tqdm issues, without changing logic.'
- What this solution (achieved 0.1658) has done: 'Your score (0.298) is far below the target (0.7727), so we should keep the exact same Otsu→(morphology)→CRF-fallback→RLE pipeline but make the binary mask a bit closer to typical salt shapes with minimal, deterministic post-processing changes. I add a lightweight per-image threshold calibration step (try a few small offsets around the Otsu/depth-adjusted threshold and pick the one with the “best-looking” coverage/connectedness score) while keeping the thresholding approach unchanged. I also add a tiny opening step before closing (to remove thin noise) and switch the closing footprint to a slightly more connecting shape, without changing the overall logic. Finally, I keep the Fortran-order RLE and submission schema identical and still write `crf_correction.csv`.'
- What this solution (achieved 0.1658) has done: 'Your current score (0.1658) is far below the target (0.7727), so we should improve score while preserving your exact pipeline (Otsu on preprocessed image → small deterministic morphology → optional CRF fallback → Fortran-order RLE). The biggest minimal win left is fixing a metric-mismatch: this competition’s scoring is instance-style across IoU thresholds, and your current post-processing tends to produce many fragmented components; we instead keep only the single largest connected component after morphology (a standard TGS trick) to reduce FP components without changing the approach. To avoid over-suppressing true salt, we apply that “largest component” rule only when there are multiple components and predicted coverage is not tiny. Everything else (paths, preprocessing, depth-shift, threshold-offset calibration, CRF fallback, and CSV schema) stays the same.'
- What this solution (achieved 0.033) has done: 'Your current score (0.1658) is far below the target (0.7727), so we should improve it while keeping your exact segmentation pipeline (preprocess → Otsu+depth shift+small offsets → morphology → optional LCC → CRF fallback → Fortran RLE). The smallest high-impact fix is to stop using a *single global* binarization direction (`img < t`) and instead choose the better of `(img < t)` vs `(img > t)` per-image, per-threshold-offset using your existing deterministic mask quality score; this preserves the thresholding approach but prevents catastrophic “background-as-salt” inversions. I also make a tiny, metric-aligned safeguard: apply `clear_border` only when the predicted salt touches the border a lot (so we don’t erase real salt that legitimately reaches edges), and I slightly relax the “too big mask” cutoff (0.65→0.72) to reduce false-emptying. Everything else (paths, CRF behavior, Fortran-order RLE, submission writing) stays the same and still produces `crf_correction.csv`.'
- What this solution (achieved 0.0499) has done: 'Your current score (0.033) is far below the target (0.7727), so we should improve performance while keeping your exact pipeline (preprocess → Otsu+depth shift+offsets → morphology → optional LCC → CRF-fallback → Fortran-order RLE). The largest minimal bug hurting mAP here is that the submission must represent a **single object mask per image**, and this competition’s metric is instance-style: if you predict “salt everywhere” or too many scattered pixels, AP collapses across IoU thresholds. I keep your existing selection logic but adjust the deterministic mask scoring to penalize over-coverage more strongly and reward compactness, and I slightly tune the morphology thresholds to reduce fragmentation while keeping semantics unchanged. Finally, I ensure the sample-submission placeholder logic always generates masks from images (never reuses “1 1”), which prevents reverting to guaranteed-bad masks.'
- What this solution (achieved 0.0345) has done: 'Your score is far below the target (0.0499 vs 0.7727, higher-is-better), so we should improve it with the smallest changes that keep your exact pipeline (preprocess → Otsu+depth shift+offsets+direction choice → morphology → optional LCC → CRF-fallback → Fortran RLE). The biggest likely issue is that your masks are still very poorly calibrated (coverage heuristic too strict and “empty by default”), so I (a) relax the hard “too big → empty” guard and the “empty is always terrible” scoring so the selector can choose non-empty masks when appropriate, and (b) add a tiny, deterministic post-step that fills small holes after closing to reduce under-segmentation without changing the overall approach. I also ensure `disk` footprints are created with small, safe radii and keep runtime bounded. Everything else (paths, CRF fallback behavior, output CSV schema/name) stays the same.'
- What this solution (achieved 0.0335) has done: 'Your current score (0.0345) is far below the target (0.7727), so we should improve it while keeping your same deterministic pipeline (preprocess → Otsu+depth shift+offsets+direction choice → morphology → optional LCC → CRF-fallback → Fortran RLE). The biggest minimal bug-like issue is that test-time segmentation from raw intensity alone is too weak for TGS; we can add the provided depth feature as an *extra channel-like prior* by applying a tiny depth-dependent bias field (still just thresholding) and slightly strengthen the “single object” constraint to reduce false positives. Concretely, I (1) add a very small center-weighting (salt is often central) and depth bias to the mask quality score used for selection (doesn’t change architecture/training—only selects among your existing candidates), and (2) add a simple “if multiple components, keep the largest unless coverage is extremely small” rule that is less aggressive than always-LCC. These are small, deterministic, metric-aligned tweaks that usually move mAP up without changing your core approach or adding dependencies, and the script still run end-to-end and write `crf_correction.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.color import gray2rgb
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity

from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    disk,
    binary_closing,
    binary_opening,
)
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops

from tqdm import tqdm

_HAS_DCRF = False
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels

    _HAS_DCRF = True
except Exception:
    _HAS_DCRF = False

BASE_INPUT = "../input/tgs-salt-identification-challenge"

_TEST_DIR_CANDIDATES = [
    os.path.join(BASE_INPUT, "test", "images"),
    "../input/test/images",
    "../input/tgs-salt-identification-challenge/test/images",
    "../input/kaggle/input/tgs-salt-identification-challenge/test/images",
    "../kaggle/data/tgs-salt-identification-challenge/test/images",
]
TEST_IMG_DIR = next(
    (p for p in _TEST_DIR_CANDIDATES if os.path.exists(p)), _TEST_DIR_CANDIDATES[0]
)

_DEPTHS_CANDIDATES = [
    os.path.join(BASE_INPUT, "depths.csv"),
    "../input/depths.csv",
    "../kaggle/data/depths.csv",
]
DEPTHS_PATH = next(
    (p for p in _DEPTHS_CANDIDATES if os.path.exists(p)), _DEPTHS_CANDIDATES[0]
)

_H, _W = 101, 101
yy, xx = np.mgrid[0:_H, 0:_W].astype(np.float32)
yy = (yy - (_H - 1) / 2.0) / ((_H - 1) / 2.0)
xx = (xx - (_W - 1) / 2.0) / ((_W - 1) / 2.0)
_r2 = xx * xx + yy * yy
CENTER_WEIGHT = np.exp(-_r2 / (2.0 * (0.75**2))).astype(np.float32)  # smooth, mild




## === cell 1
def rle_decode(rle_mask, shape=(101, 101)):
    """
    Kaggle TGS Salt uses column-major (Fortran) order for RLE.
    """
    if rle_mask is None:
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
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def rle_encode(im):
    """
    Kaggle TGS Salt uses column-major (Fortran) order for RLE.
    """
    im = (im > 0).astype(np.uint8)

    if im.sum() == 0:
        return ""

    pixels = im.reshape(-1, order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def _is_valid_rle(x):
    if x is None:
        return False
    if isinstance(x, float) and np.isnan(x):
        return False
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return False
    return True


def _is_placeholder_rle(x):
    s = str(x).strip()
    return s == "1 1"


def _mask_quality_score(mask_bool, z_norm=0.0):
    """
    Score-relevant (minimal): keep your existing heuristic structure, but add
    a mild centrality prior and a tiny depth-aware target coverage shift.
    This only selects among the existing candidate masks you already generate.
    """
    m = mask_bool.astype(bool)
    frac = float(m.mean())

    if frac <= 1e-8:
        return -3.0

    if frac >= 0.75:
        return -3.0 - 10.0 * (frac - 0.75)

    try:
        cc = label(m, connectivity=1)
        n_cc = int(cc.max())
    except Exception:
        n_cc = 9999

    target = float(np.clip(0.18 + 0.015 * np.tanh(z_norm), 0.10, 0.26))
    coverage_term = -(abs(frac - target) / (target + 1e-9))

    frag_term = -0.10 * max(0, n_cc - 1)

    compact_term = 0.0
    try:
        if n_cc >= 1:
            props = regionprops(cc.astype(np.int32))
            if len(props) > 0:
                p = max(props, key=lambda r: r.area)
                minr, minc, maxr, maxc = p.bbox
                bbox_area = float((maxr - minr) * (maxc - minc) + 1e-9)
                fill = float(p.area) / bbox_area
                compact_term = 0.12 * fill
    except Exception:
        compact_term = 0.0

    central_term = 0.0
    try:
        central_term = 0.08 * float(
            (m.astype(np.float32) * CENTER_WEIGHT).mean() / (frac + 1e-9)
        )
    except Exception:
        central_term = 0.0

    return coverage_term + frag_term + compact_term + central_term


def _keep_largest_component(mask_bool):
    m = mask_bool.astype(bool)
    if m.sum() == 0:
        return m

    cc = label(m, connectivity=1)
    n_cc = int(cc.max())
    if n_cc <= 1:
        return m

    counts = np.bincount(cc.ravel())
    if counts.shape[0] <= 1:
        return m
    counts[0] = 0
    keep = int(np.argmax(counts))
    return cc == keep


def _touches_border(mask_bool):
    m = mask_bool.astype(bool)
    if m.sum() == 0:
        return 0.0
    b = np.zeros_like(m, dtype=bool)
    b[0, :] = True
    b[-1, :] = True
    b[:, 0] = True
    b[:, -1] = True
    return float((m & b).sum()) / float(m.sum() + 1e-9)




## === cell 2
baseline_path = "../input/baseline-v4/submission.csv"
sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")

if os.path.exists(baseline_path):
    df = pd.read_csv(baseline_path)
    _INPUT_SOURCE = "baseline-v4"
else:
    df = pd.read_csv(sample_path)
    _INPUT_SOURCE = "sample_submission"

if "id" not in df.columns or "rle_mask" not in df.columns:
    raise ValueError(
        f"Submission must have columns ['id','rle_mask'], got: {df.columns.tolist()}"
    )

df["id"] = df["id"].astype(str)

depths_df = None
depth_map = {}
try:
    depths_df = pd.read_csv(DEPTHS_PATH)
    if "id" in depths_df.columns and "z" in depths_df.columns:
        depths_df["id"] = depths_df["id"].astype(str)
        depth_map = dict(zip(depths_df["id"].values, depths_df["z"].values))
except Exception:
    depth_map = {}

if len(depth_map) > 0:
    z_vals = np.asarray(list(depth_map.values()), dtype=np.float32)
    z_median = float(np.median(z_vals))
    z_iqr = float(np.percentile(z_vals, 75) - np.percentile(z_vals, 25)) or 1.0
else:
    z_median = 0.0
    z_iqr = 1.0




## === cell 3
"""
Function which returns the labelled image after applying CRF.
If pydensecrf is unavailable, return the input mask unchanged (keeps pipeline running).
"""


def crf(original_image, mask_img):
    if not _HAS_DCRF:
        return (mask_img > 0).astype(np.uint8)

    if len(mask_img.shape) < 3:
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
try:
    n_show = 6
    shown = 0
    plt.figure(figsize=(18, 4))
    for idx in range(len(df)):
        if shown >= n_show:
            break
        rle = df.loc[idx, "rle_mask"]
        if _is_valid_rle(rle) and not _is_placeholder_rle(rle):
            m = rle_decode(rle)
            plt.subplot(1, n_show, shown + 1)
            plt.imshow(m, cmap="gray")
            plt.title(df.loc[idx, "id"])
            plt.axis("off")
            shown += 1
    plt.close()
except Exception:
    pass




## === cell 5
"""
Applying CRF on the predicted mask.

Minimal score-relevant changes while preserving the same core pipeline:
- Keep: preprocess (percentile rescale + Gaussian) -> Otsu -> depth shift -> try small threshold offsets
  -> morphology -> optional LCC -> CRF-fallback -> Fortran-order RLE.
- Change (score-relevant, minimal):
  (a) Use depth-normalized value inside mask selection scoring (tiny calibration),
  (b) Apply LCC only when it helps (multiple components and not ultra-tiny coverage),
  (c) Keep all runtime bounded and deterministic.
"""
missing_imgs = 0
processed = 0
generated_from_image = 0
used_existing_rle = 0

_ignore_placeholders_globally = _INPUT_SOURCE == "sample_submission"

_MIN_OBJ = 30
_MIN_HOLE = 30

_CLOSING_RADIUS = 4
selem_close = disk(_CLOSING_RADIUS)

_OPENING_RADIUS = 1
selem_open = disk(_OPENING_RADIUS)

_MIN_OBJ_POST = 80

_DEPTH_ALPHA = 0.060

_GAUSS_SIGMA = 0.75

_MAX_SALT_FRAC = 0.75

_T_OFFSETS = (-0.04, -0.02, 0.0, 0.02, 0.04)

_LCC_MIN_FRAC = 0.002
_LCC_APPLY_MIN_FRAC = 0.008  # only force LCC when mask has enough area to justify it

_BORDER_TOUCH_FRAC_FOR_CLEAR = 0.12

_FINAL_HOLE_FILL = 120

for i in tqdm(range(df.shape[0]), total=df.shape[0]):
    img_id = df.loc[i, "id"]
    img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")
    if not os.path.exists(img_path):
        df.loc[i, "rle_mask"] = ""
        missing_imgs += 1
        continue

    orig_img = imread(img_path)

    rle = df.loc[i, "rle_mask"]
    can_use_rle = (
        _is_valid_rle(rle)
        and (not _is_placeholder_rle(rle))
        and (not _ignore_placeholders_globally)
    )

    if can_use_rle:
        decoded_mask = rle_decode(rle)
        used_existing_rle += 1
    else:
        img = orig_img.astype(np.float32)
        if img.ndim == 3:
            img = img[..., 0]

        if img.max() > 1.5:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)

        try:
            p2, p98 = np.percentile(img, (2, 98))
            if float(p98 - p2) > 1e-6:
                img_proc = rescale_intensity(
                    img, in_range=(p2, p98), out_range=(0.0, 1.0)
                )
            else:
                img_proc = img
        except Exception:
            img_proc = img

        try:
            img_proc = gaussian(img_proc, sigma=_GAUSS_SIGMA, preserve_range=True)
            img_proc = np.clip(img_proc, 0.0, 1.0)
        except Exception:
            img_proc = img

        try:
            t = float(threshold_otsu(img_proc))
        except Exception:
            t = float(np.mean(img_proc))

        z = depth_map.get(img_id, z_median)
        z_norm = float((z - z_median) / z_iqr)

        z_norm_clip = float(np.clip(z_norm, -2.0, 2.0))
        t_adj = np.clip(t + _DEPTH_ALPHA * z_norm_clip * t, 0.0, 1.0)

        best_mask = None
        best_score = -1e18

        for off in _T_OFFSETS:
            tt = float(np.clip(t_adj + off, 0.0, 1.0))

            for direction in (0, 1):
                m = (img_proc < tt) if direction == 0 else (img_proc > tt)

                try:
                    if _touches_border(m) >= _BORDER_TOUCH_FRAC_FOR_CLEAR:
                        m = clear_border(m)

                    m = remove_small_objects(m, min_size=_MIN_OBJ)
                    m = remove_small_holes(m, area_threshold=_MIN_HOLE)

                    m = binary_opening(m, selem_open)
                    m = binary_closing(m, selem_close)

                    m = remove_small_holes(m, area_threshold=_FINAL_HOLE_FILL)

                    m = remove_small_objects(m, min_size=_MIN_OBJ_POST)

                    if float(m.mean()) >= _LCC_MIN_FRAC:
                        cc = label(m, connectivity=1)
                        if int(cc.max()) > 1 and float(m.mean()) >= _LCC_APPLY_MIN_FRAC:
                            m = _keep_largest_component(m)
                except Exception:
                    pass

                try:
                    if m.mean() > _MAX_SALT_FRAC:
                        m = np.zeros_like(m, dtype=bool)
                except Exception:
                    pass

                sc = _mask_quality_score(m, z_norm=z_norm_clip)
                if sc > best_score:
                    best_score = sc
                    best_mask = m

        decoded_mask = (
            best_mask.astype(np.uint8)
            if best_mask is not None
            else np.zeros((101, 101), dtype=np.uint8)
        )
        generated_from_image += 1

    crf_output = crf(orig_img, decoded_mask)
    df.loc[i, "rle_mask"] = rle_encode(crf_output)
    processed += 1




## === cell 6
out_path = "crf_correction.csv"
df[["id", "rle_mask"]].to_csv(out_path, index=False)

print(f"Wrote submission: {out_path}")
print(
    f"Rows: {len(df)}, processed: {processed}, generated_from_image: {generated_from_image}, "
    f"used_existing_rle: {used_existing_rle}, missing_images: {missing_imgs}, "
    f"has_dcrf: {_HAS_DCRF}, input_source: {_INPUT_SOURCE}, test_img_dir: {TEST_IMG_DIR}, "
    f"depths_loaded: {len(depth_map)}"
)
