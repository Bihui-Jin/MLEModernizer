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

0.799385547382085

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'We remove the unavailable `pydensecrf` dependency (it isn’t installed in your environment) and replace the CRF step with a lightweight, deterministic post-processing that keeps the same “refine predicted mask then re-encode to RLE” core intent. We also convert notebook-only commands (`%matplotlib inline`, `ls`) into Python-safe code so the script runs end-to-end. The code robustly locate a baseline `submission.csv` if present; otherwise it fall back to `sample_submission.csv` to still produce a valid `.csv` file. Finally, we fix NaN handling, ensure masks are binary `0/1`, and write the required `id,rle_mask` submission as `crf_correction.csv`.'
- What this solution (achieved 0.5221) has done: 'Your current score is far below the target (0.5221 vs 0.7994), so we should improve the post-processing in the smallest possible way without changing the overall “decode → refine → encode” pipeline. The biggest likely issue is that your RLE decode/encode uses row-major (`order="C"`) flattening, but this competition’s RLE is column-major (top-to-bottom then left-to-right), so the masks you output are effectively transposed in encoding space, heavily hurting IoU/AP. I fix `rle_decode`/`rle_encode` to the correct column-major convention (Fortran order) while keeping everything else the same. Additionally, I make `refine_mask` slightly safer by removing tiny connected components (a common TGS Salt cleanup) without changing the model logic—still deterministic post-processing.'
- What this solution (achieved 0.5221) has done: 'Your score gap to the target is large (0.5221 vs 0.7994), so the most likely low-risk gain is fixing submission correctness issues rather than changing modeling. The current script only refines non-empty masks from the baseline file; if the baseline is weak/sparse, most test rows remain empty and score stays low. I keep the exact same “decode → refine → encode” logic, but apply it to every test id by generating a mask from the actual test image intensity when the baseline RLE is empty, then running the same refinement and encoding. This stays within the same post-processing semantics while making predictions non-trivial for all images, which should move mAP upward toward the target.'
- What this solution (achieved 0.5221) has done: 'Your current score (0.5221) is far below the target (0.7994), so we should improve predictions without changing the overall “decode → refine → encode” post-processing pipeline. The biggest low-risk gain is fixing the fallback mask generation: a fixed threshold of 0.5 is usually wrong for these grayscale seismic images, producing near-empty or noisy masks. I keep your same refinement logic, but change `initial_mask_from_image` to use a deterministic per-image Otsu threshold and then pass it through the same `refine_mask`. Additionally, I calibrate the final binarization using a small fixed probability threshold (still deterministic) and adjust `min_size` slightly to reduce false positives, which typically improves mAP toward your target.'
- What this solution (achieved 0.0) has done: 'Your current score (0.5221) is far below the target (0.7994), so we should make small, legitimate post-processing changes that usually yield a meaningful uplift without changing your overall decode→refine→encode pipeline. The main issue is that the “empty baseline” path currently produces very noisy masks (Otsu on raw images often segments geology textures as salt), which creates many false positives and drags mAP down; we keep Otsu but constrain it with a conservative prior: if the Otsu mask coverage is implausibly large/small, fall back to an inverted threshold or empty mask. We also tune `refine_mask` minimally by filling small holes and using a slightly more robust connectivity for removing small objects—both are deterministic and keep the same core semantics. Finally, we ensure we iterate in the exact sample_submission order and never change ids/row alignment while writing `crf_correction.csv`.'
- What this solution (achieved 0.5221) has done: 'I fix the import/runtime errors by replacing `binary_fill_holes` with the correct location (`scipy.ndimage.binary_fill_holes`) and by making `tqdm` and `plt` optional so missing/failed imports don’t crash later cells. I also make the code robust to environments where `scikit-image`/`tqdm` aren’t installed by providing small no-op fallbacks (this is score-neutral but ensures end-to-end execution). Finally, I keep your existing decode→refine→encode logic unchanged and ensure the script always writes a valid `crf_correction.csv` with the required `id,rle_mask` columns.'
- What this solution (achieved 0.5221) has done: 'Your score is far below the target (0.5221 vs 0.7994), so the most likely minimal, legitimate uplift is to fix prediction semantics that strongly affect mAP without changing the overall decode→refine→encode pipeline. The biggest issue is that when the baseline RLE is empty, the current Otsu-on-image fallback is essentially guessing and can create many false positives; instead we should use the provided `depths.csv` as a deterministic prior to decide when to predict empty masks (deep images are often no-salt) while keeping the same morphological refinement. Additionally, we should avoid encoding fully-empty masks as an empty string only after verifying emptiness post-refine (to reduce spurious tiny predictions). These changes keep the core logic intact (still: baseline decode or deterministic init → refine_mask → encode), but reduce false positives and typically moves mAP upward toward your target.'
- What this solution (achieved 0.5221) has done: 'Your current score is far below the target (0.5221 vs 0.7994), so we should aim for a sizeable but still minimal, post-processing-only uplift without changing the overall decode → (init if empty) → refine → encode pipeline. The biggest score drag in this script is the “empty baseline” branch: Otsu-on-image is a weak heuristic and creates many false positives/negatives; we keep the same deterministic image-based initialization, but make it more TGS-appropriate by using a small depth-conditioned intensity shift before Otsu and by selecting the better of (x>t) vs (x<t) based on a depth prior for expected salt coverage. We also make the depth prior less aggressive (your `z>800` likely forces too many empties), and we make the final “empty mask” decision more robust by dropping tiny predicted coverages (common FP source) after refine. These are small, deterministic changes that preserve core semantics and should move mAP upward toward your target.'
- What this solution (achieved 0.5221) has done: 'Your current score (0.5221) is far below the target (0.7994), so we should focus on minimal, high-impact correctness fixes rather than changing the overall decode→refine→encode pipeline. The biggest likely score killer remaining is a mask orientation mismatch: TGS images are stored in standard (row-major) PNG layout, but the RLE convention is column-major; your current `rle_decode`/`rle_encode` is correct for RLE, but when you generate masks from images (Otsu) you never transpose into the RLE coordinate system, so those predictions are effectively rotated in submission space and lose IoU. I keep your exact approach, but align image-derived masks to the same orientation used by RLE (via transpose before refine/encode, and inverse transpose only where needed), and I also ensure the “tiny prediction drop” check is applied consistently in the same orientation. This is a small, deterministic change that should substantially increase mAP toward the target without altering core logic.'
- What this solution (achieved 0.5221) has done: 'Your current score (0.5221) is far below the target (0.7994), so the most likely minimal uplift is to reduce systematic false positives coming from the “empty baseline → Otsu guess” branch while keeping the same decode→init→refine→encode pipeline. I keep your exact morphology/refinement core logic, but make the image-derived initialization choose between (x>t) and (x<t) using the baseline’s global predicted coverage as a dataset prior (instead of a depth-only prior), which is a small, deterministic calibration step. I also make the “force empty by depth” less aggressive (it can zero-out true salt and cap score), and tighten the tiny-prediction drop slightly to suppress speckle FPs that hurt mAP across IoU thresholds. These are small post-processing calibrations and should move the score upward toward the target without changing architecture/training (none exists here) or the overall semantics.'
- What this solution (achieved 0.5221) has done: 'Your score gap to the target is large (0.5221 → 0.7994), so the most likely minimal uplift is to fix a remaining correctness issue in the image-derived fallback path rather than changing the overall pipeline. Right now, for empty-baseline rows you transpose the image-derived mask into “RLE space”, but then you apply morphology in that transposed space using disk-shaped structuring elements; this is inconsistent with how morphology should operate on the original image geometry and can distort objects, hurting IoU. I keep the exact decode→(init if empty)→refine→encode logic, but refine the image-derived masks in image space first and only then transpose for RLE encoding, ensuring morphology is applied consistently. Additionally, I make the tiny-prediction drop threshold slightly less aggressive to recover some recall (without changing the model/metric semantics).'
- What this solution (achieved 0.5221) has done: 'Your gap to the target is large (0.5221 → 0.7994), so the most likely minimal uplift is reducing systematic false positives/false negatives coming from the “empty baseline → Otsu guess” branch without changing the overall decode→init→refine→encode pipeline. I keep your exact morphology/refinement and RLE conventions, but make the image-derived initialization choose the Otsu polarity using a dataset-driven prior computed from the baseline (instead of relying mostly on depth), which is a small deterministic calibration step. I also make the “force empty by depth” slightly less aggressive (to recover recall that can matter across IoU thresholds) and apply the tiny-mask drop in the same way but with a slightly safer threshold to suppress speckle FPs. These changes preserve your core logic and should move the score upward toward the target.'
- What this solution (achieved 0.5221) has done: 'We keep your exact decode → (init if empty) → refine → encode pipeline, but reduce a likely systematic score drag: refining baseline masks in “RLE space” instead of true image geometry can distort shapes because morphology is geometry-dependent. The minimal fix is to convert baseline-decoded masks back to image space before `refine_mask`, then convert to RLE space only right before encoding (the empty-baseline branch already does this correctly). Additionally, we slightly relax the tiny-prediction drop threshold to recover recall (too-aggressive dropping can hurt mAP across IoU thresholds), while keeping determinism and the same refinement operations. These changes should move the score upward toward your target without changing any modeling/training (none) or the overall post-processing intent.'
- What this solution (achieved 0.5221) has done: 'Your score is far below the target, so the smallest likely uplift is to make the *image-based fallback* less error-prone without changing the overall decode→(init if empty)→refine→encode pipeline. I keep your morphology/refinement and RLE conventions intact, but adjust the “force empty by depth” rule to be less aggressive (it can wipe out true positives and cap mAP). I also make the empty-baseline initialization slightly more conservative by adding a small “near-empty” check before refinement to prevent large false-positive masks from Otsu, which hurts AP across IoU thresholds. These tweaks are deterministic, fast, and keep your core logic unchanged while aiming to move the score upward toward the target.'
- What this solution (achieved 0.5221) has done: 'Your gap to the target is large (0.5221 → 0.7994), so the most plausible minimal uplift is to fix post-processing calibration rather than change the overall decode→(init if empty)→refine→encode pipeline. I keep your exact morphology/refinement steps, but tune the two rules that most directly trade off false positives vs recall for mAP: (1) make the “force empty by depth” rule slightly less aggressive to avoid wiping true salt, and (2) make the “drop tiny predictions” threshold slightly less aggressive to recover small-but-real masks (important at higher IoU thresholds). I also make the final binarization threshold a touch more conservative (0.55 vs 0.5) to reduce boundary noise and speckle FPs, which typically improves mean AP across IoU thresholds without changing any core logic. All changes are deterministic, fast, and preserve the same evaluation semantics and submission format.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

try:
    from skimage.io import imread
    from skimage.morphology import (
        binary_opening,
        binary_closing,
        disk,
        remove_small_objects,
    )
    from skimage.filters import threshold_otsu
except Exception as e:
    imread = None
    binary_opening = None
    binary_closing = None
    disk = None
    remove_small_objects = None
    threshold_otsu = None
    _SKIMAGE_IMPORT_ERROR = e

try:
    from scipy.ndimage import binary_fill_holes
except Exception:
    binary_fill_holes = None

try:
    from tqdm import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

np.random.seed(42)




## === cell 1
def rle_decode(rle_mask, shape=(101, 101)):
    """
    Competition-correct RLE decode for TGS Salt:
    Pixels are 1-indexed and enumerated top-to-bottom, then left-to-right,
    which corresponds to column-major (Fortran) flattening.
    """
    if rle_mask is None:
        return np.zeros(shape, dtype=np.uint8)
    if not isinstance(rle_mask, str) or rle_mask.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def rle_encode(im):
    """
    Competition-correct RLE encode for TGS Salt:
    Use column-major (Fortran) flattening.
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def imgmask_to_rlespace(mask):
    return np.ascontiguousarray(mask.T)


def rlespace_to_imgmask(mask):
    return np.ascontiguousarray(mask.T)




## === cell 2
TEST_IMG_DIR = "../input/tgs-salt-identification-challenge/test/images/"
SAMPLE_SUB_PATH = "../input/tgs-salt-identification-challenge/sample_submission.csv"
DEPTHS_PATH = "../input/tgs-salt-identification-challenge/depths.csv"

baseline_candidates = sorted(glob.glob("../input/*/submission.csv"))
baseline_path = baseline_candidates[0] if len(baseline_candidates) > 0 else None

if baseline_path is not None and os.path.exists(baseline_path):
    sub_in_path = baseline_path
else:
    sub_in_path = SAMPLE_SUB_PATH

df = pd.read_csv(sub_in_path)

if "id" not in df.columns:
    raise ValueError(f"Input submission file missing 'id' column: {sub_in_path}")
if "rle_mask" not in df.columns:
    possible = [c for c in df.columns if "rle" in c.lower()]
    if possible:
        df = df.rename(columns={possible[0]: "rle_mask"})
    else:
        df["rle_mask"] = ""

df["rle_mask"] = df["rle_mask"].fillna("").astype(str)

sample_df = pd.read_csv(SAMPLE_SUB_PATH)[["id"]]
df = sample_df.merge(df[["id", "rle_mask"]], on="id", how="left")
df["rle_mask"] = df["rle_mask"].fillna("").astype(str)

depths_df = None
if os.path.exists(DEPTHS_PATH):
    depths_df = pd.read_csv(DEPTHS_PATH)
    if "id" in depths_df.columns and "z" in depths_df.columns:
        depths_df = depths_df[["id", "z"]]
    else:
        depths_df = None

if depths_df is not None:
    df = df.merge(depths_df, on="id", how="left")
else:
    df["z"] = np.nan


def _safe_baseline_coverage_mean(df_in, max_rows=400):
    coverages = []
    n = 0
    for rle in df_in["rle_mask"].values:
        if isinstance(rle, str) and rle.strip() != "":
            m = rle_decode(rle)
            coverages.append(float(m.mean()))
            n += 1
            if n >= max_rows:
                break
    if len(coverages) == 0:
        return 0.12
    return float(np.clip(np.mean(coverages), 0.01, 0.35))


def _safe_baseline_nonempty_rate(df_in, max_rows=800):
    seen = 0
    nonempty = 0
    for rle in df_in["rle_mask"].values:
        if not isinstance(rle, str):
            continue
        if rle.strip() != "":
            nonempty += 1
        seen += 1
        if seen >= max_rows:
            break
    if seen == 0:
        return 0.5
    return float(np.clip(nonempty / float(seen), 0.05, 0.95))


BASELINE_COV_PRIOR = _safe_baseline_coverage_mean(df)
BASELINE_NONEMPTY_PRIOR = _safe_baseline_nonempty_rate(df)

print(f"Loaded {len(df)} rows from: {sub_in_path} (aligned to sample_submission order)")
print(f"Test image dir exists: {os.path.isdir(TEST_IMG_DIR)} ({TEST_IMG_DIR})")
print(
    f"Baseline coverage prior (mean over non-empty baseline masks): {BASELINE_COV_PRIOR:.4f}"
)
print(f"Baseline non-empty prior (rate): {BASELINE_NONEMPTY_PRIOR:.4f}")
print(df.head())



## === cell 3
"""
Replacement for CRF refinement (pydensecrf not available in this environment).

We keep the same pipeline semantics:
- decode baseline predicted RLE -> binary mask (or deterministic init when empty)
- refine mask using deterministic post-processing
- encode back to RLE
"""


def _ensure_skimage_available():
    if (
        imread is None
        or binary_opening is None
        or binary_closing is None
        or disk is None
        or remove_small_objects is None
    ):
        raise ImportError(
            "Required scikit-image components are unavailable in this environment. "
            "Install scikit-image or use an environment that includes it."
        )
    if binary_fill_holes is None:
        raise ImportError(
            "Required scipy.ndimage.binary_fill_holes is unavailable in this environment. "
            "Install scipy or use an environment that includes it."
        )


def refine_mask(mask, radius_open=1, radius_close=2, min_size=30, fill_holes=True):
    """
    Simple morphological refinement:
    - closing to fill small gaps
    - opening to remove small speckles
    - optionally fill holes
    - remove very small connected components
    """
    _ensure_skimage_available()

    m = mask > 0
    if radius_close and radius_close > 0:
        m = binary_closing(m, disk(radius_close))
    if radius_open and radius_open > 0:
        m = binary_opening(m, disk(radius_open))
    if fill_holes:
        m = binary_fill_holes(m)

    if min_size and min_size > 0:
        m = remove_small_objects(m.astype(bool), min_size=min_size, connectivity=2)

    return m.astype(np.uint8)


def expected_coverage_from_depth(z_value):
    """
    Heuristic prior: deeper images tend to have less salt.
    Returns a plausible target mask fraction in [0.02, 0.35].
    """
    try:
        z = float(z_value)
    except Exception:
        z = np.nan
    if not np.isfinite(z):
        return 0.12
    zn = (z - 0.0) / 1000.0
    zn = float(np.clip(zn, 0.0, 1.0))
    return float(0.30 - 0.20 * zn)


def blended_coverage_prior(z_value, baseline_prior):
    cov_d = expected_coverage_from_depth(z_value)
    return float(np.clip(0.35 * cov_d + 0.65 * float(baseline_prior), 0.01, 0.35))


def initial_mask_from_image(
    img,
    z_value=np.nan,
    baseline_cov_prior=0.12,
    baseline_nonempty_prior=0.5,
):
    """
    Deterministic initial mask when baseline is empty.

    Keep Otsu + polarity choice, but:
    - choose polarity closest to a blended (depth + baseline) expected coverage
    - reject masks that are extremely implausible in coverage to reduce FP bursts
      (score-relevant: suppresses large false positives that kill AP across IoUs)
    """
    _ensure_skimage_available()

    if img.ndim == 3:
        img = img[..., 0]
    x = img.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0

    cov_target = blended_coverage_prior(z_value, baseline_cov_prior)

    try:
        z = float(z_value)
    except Exception:
        z = np.nan
    if np.isfinite(z):
        shift = float(np.clip((z - 500.0) / 500.0, -1.0, 1.0)) * 0.03
        x = np.clip(x + shift, 0.0, 1.0)

    try:
        t = float(threshold_otsu(x))
    except Exception:
        t = 0.5

    m_hi = (x > t).astype(np.uint8)
    m_lo = (x < t).astype(np.uint8)
    cov_hi = float(m_hi.mean())
    cov_lo = float(m_lo.mean())

    candidates = [("hi", m_hi, cov_hi), ("lo", m_lo, cov_lo)]
    plausible = [c for c in candidates if 0.003 <= c[2] <= 0.65]
    if len(plausible) == 0:
        return np.zeros_like(m_hi, dtype=np.uint8)

    plausible.sort(key=lambda c: abs(c[2] - cov_target))
    best_name, best_mask, best_cov = plausible[0]

    nonempty = float(np.clip(baseline_nonempty_prior, 0.05, 0.95))
    max_dev = 0.22 * nonempty + 0.08 * (1.0 - nonempty)
    if abs(best_cov - cov_target) > max_dev:
        return np.zeros_like(best_mask, dtype=np.uint8)

    if best_cov > 0.55:
        return np.zeros_like(best_mask, dtype=np.uint8)

    return best_mask


def final_binarize(mask, thr=0.55):
    """
    CHANGE (score-relevant, minimal): slightly more conservative final threshold.
    This tends to reduce boundary noise/speckle false positives, improving mean AP.
    """
    return (mask > thr).astype(np.uint8)


def should_force_empty_by_depth(z_value):
    """
    Depth prior to reduce false positives on very-deep (often empty) cases.

    CHANGE (score-relevant, minimal): make it less aggressive to recover recall.
    """
    try:
        z = float(z_value)
    except Exception:
        return False
    return z > 1700.0


def should_drop_tiny_prediction(mask, min_frac=0.0012):
    """
    If predicted positive area is extremely tiny, treat as empty.

    CHANGE (score-relevant, minimal): slightly less aggressive drop to keep small
    true salt patches that can contribute to mAP at multiple IoU thresholds.
    """
    frac = float(mask.mean())
    return frac < float(min_frac)




## === cell 4
if plt is not None:
    try:
        n_show = min(6, len(df))
        idxs = [i for i in range(len(df)) if str(df.loc[i, "rle_mask"]).strip() != ""]
        idxs = idxs[:n_show] if len(idxs) > 0 else list(range(n_show))

        plt.figure(figsize=(18, 6))
        for j, i in enumerate(idxs):
            decoded = rle_decode(df.loc[i, "rle_mask"])
            refined = imgmask_to_rlespace(refine_mask(rlespace_to_imgmask(decoded)))
            plt.subplot(2, n_show, j + 1)
            plt.imshow(decoded, cmap="gray")
            plt.title(f"orig {df.loc[i,'id']}")
            plt.axis("off")
            plt.subplot(2, n_show, n_show + j + 1)
            plt.imshow(refined, cmap="gray")
            plt.title("refined")
            plt.axis("off")
        plt.tight_layout()
    except Exception as e:
        print(f"Skipping visualization due to error: {e}")



## === cell 5
if imread is None:
    raise ImportError(
        "skimage.io.imread is unavailable in this environment. "
        "This script needs it to read test images."
    )

missing_img_count = 0
empty_baseline_filled = 0
forced_empty_by_depth = 0
emptied_after_refine = 0
dropped_tiny_preds = 0

for i in tqdm(range(df.shape[0]), desc="Refining masks"):
    img_id = df.at[i, "id"]
    img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")

    rle = df.at[i, "rle_mask"]
    has_rle = isinstance(rle, str) and rle.strip() != ""

    if (not has_rle) and should_force_empty_by_depth(df.at[i, "z"]):
        df.at[i, "rle_mask"] = ""
        forced_empty_by_depth += 1
        continue

    if not os.path.exists(img_path):
        missing_img_count += 1
        if has_rle:
            decoded_mask = rle_decode(rle)
            refined_img = refine_mask(rlespace_to_imgmask(decoded_mask))
            refined_img = final_binarize(refined_img, thr=0.55)
            refined_rle = imgmask_to_rlespace(refined_img)

            if refined_rle.sum() == 0 or should_drop_tiny_prediction(refined_rle):
                df.at[i, "rle_mask"] = ""
                emptied_after_refine += 1
                if refined_rle.sum() > 0:
                    dropped_tiny_preds += 1
            else:
                df.at[i, "rle_mask"] = rle_encode(refined_rle)
        else:
            df.at[i, "rle_mask"] = ""
        continue

    img = imread(img_path)

    if has_rle:
        decoded_mask_rle = rle_decode(rle)  # mask in RLE space
        decoded_mask_img = rlespace_to_imgmask(decoded_mask_rle)
        refined_img = refine_mask(decoded_mask_img)
        refined_img = final_binarize(refined_img, thr=0.55)
        refined_final_rle = imgmask_to_rlespace(refined_img)
    else:
        init_img_space = initial_mask_from_image(
            img,
            z_value=df.at[i, "z"],
            baseline_cov_prior=BASELINE_COV_PRIOR,
            baseline_nonempty_prior=BASELINE_NONEMPTY_PRIOR,
        )

        if float(init_img_space.mean()) < 0.001:
            refined_img = np.zeros_like(init_img_space, dtype=np.uint8)
        else:
            refined_img = refine_mask(init_img_space)

        refined_img = final_binarize(refined_img, thr=0.55)
        refined_final_rle = imgmask_to_rlespace(refined_img)
        empty_baseline_filled += 1

    if refined_final_rle.sum() == 0 or should_drop_tiny_prediction(refined_final_rle):
        df.at[i, "rle_mask"] = ""
        emptied_after_refine += 1
        if refined_final_rle.sum() > 0:
            dropped_tiny_preds += 1
    else:
        df.at[i, "rle_mask"] = rle_encode(refined_final_rle)

print(f"Missing test images encountered: {missing_img_count}")
print(f"Empty baseline rows replaced with image-based init: {empty_baseline_filled}")
print(f"Forced empty by depth prior: {forced_empty_by_depth}")
print(f"Emptied after refine: {emptied_after_refine}")
print(f"Dropped tiny predictions: {dropped_tiny_preds}")



## === cell 6
out_path = "crf_correction.csv"
df[["id", "rle_mask"]].to_csv(out_path, index=False)

print(f"Wrote submission: {out_path}")
print(df[["id", "rle_mask"]].head())
