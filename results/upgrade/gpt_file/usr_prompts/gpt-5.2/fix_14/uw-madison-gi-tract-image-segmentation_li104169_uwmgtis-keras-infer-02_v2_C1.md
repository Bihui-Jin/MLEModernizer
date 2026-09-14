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
Create a model to automatically segment the stomach and intestines on MRI scans.

## Metric
Mean Dice coefficient and 3D Hausdorff distance. 

The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by:

$$
\frac{2 \cdot |X \cap Y|}{|X| + |Y|}
$$

where $X$ is the predicted set of pixels and $Y$ is the ground truth. The Dice coefficient is defined to be 0 when both $X$ and $Y$ are empty. 

Hausdorff distance is a method for calculating the distance between segmentation objects A and B, by calculating the furthest point on object A from the nearest point on object B. For 3D Hausdorff, we construct 3D volumes by combining each 2D segmentation with slice depth as the Z coordinate and then find the Hausdorff distance between them. (Here the slice depth for all scans is set to 1). The expected / predicted pixel locations are normalized by image size to create a bounded 0-1 score.

The two metrics are combined, with a weight of 0.4 for the Dice metric and 0.6 for the Hausdorff distance.

## Submission Format
Use run-length encoding on the pixel values.  Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the mask should be binary, meaning the masks for all objects in an image are joined into a single large mask. A value of 0 should indicate pixels that are not masked, and a value of 1 will indicate pixels that are masked.

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The file should contain a header and have the following format:

```
id,class,predicted
1,large_bowel,1 1 5 1
1,small_bowel,1 1
1,stomach,1 1
2,large_bowel,1 5 2 17
etc.
```

## Dataset
Each case is represented by multiple sets of scan slices (each set is identified by the day the scan took place). Some cases are split by time (early days are in train, later days are in test) while some cases are split by case - the entirety of the case is in train or test. The goal is to be able to generalize to both partially and wholly unseen cases.

### Files
- train.csv - IDs and masks for all training objects.
- sample_submission.csv - a sample submission file in the correct format
- train - a folder of case/day folders, each containing slice images for a particular case on a given day.

Note that the image filenames include 4 numbers (ex. 276_276_1.63_1.63.png). These four numbers are slice width / height (integers in pixels) and width/height pixel spacing (floating points in mm). The first two defines the resolution of the slide. The last two record the physical size of each pixel.

Physical pixel thickness in superior-inferior direction is 3mm.

### Columns
- `id` - unique identifier for object
- `class` - the predicted class for the object
- `segmentation` - RLE-encoded pixels for the identified object

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (126 lines)
            sample_submission.csv (20401 lines)
            sample_submission.csv.zip (57.4 kB)
            test.csv (20401 lines)
            test.csv.zip (55.2 kB)
            test.zip (432.9 MB)
            train.csv (95089 lines)
            train.csv.zip (6.7 MB)
            train.zip (2.0 GB)
            test/
                case110/
                    case110_day12/
                        scans/
                            ... (max depth reached)
                    case110_day16/
                        scans/
                            ... (max depth reached)
                case113/
                    case113_day22/
                        scans/
                            ... (max depth reached)
                ... and 27 other folders
            train/
                case101/
                    case101_day20/
                        scans/
                            ... (max depth reached)
                    case101_day22/
                        scans/
                            ... (max depth reached)
                    case101_day26/
                        scans/
                            ... (max depth reached)
                    case101_day32/
                        scans/
                            ... (max depth reached)
                case102/
                    case102_day0/
                        scans/
                            ... (max depth reached)
                ... and 75 other folders
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
        input/
            description.md (126 lines)
            sample_submission.csv (20401 lines)
            sample_submission.csv.zip (57.4 kB)
            test.csv (20401 lines)
            test.csv.zip (55.2 kB)
            test.zip (432.9 MB)
            train.csv (95089 lines)
            train.csv.zip (6.7 MB)
            train.zip (2.0 GB)
            test/
                case110/
                    case110_day12/
                        scans/
                            ... (max depth reached)
                    case110_day16/
                        scans/
                            ... (max depth reached)
                case113/
                    case113_day22/
                        scans/
                            ... (max depth reached)
                ... and 27 other folders
            train/
                case101/
                    case101_day20/
                        scans/
                            ... (max depth reached)
                    case101_day22/
                        scans/
                            ... (max depth reached)
                    case101_day26/
                        scans/
                            ... (max depth reached)
                    case101_day32/
                        scans/
                            ... (max depth reached)
                case102/
                    case102_day0/
                        scans/
                            ... (max depth reached)
                ... and 75 other folders
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
        working/
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
```

-> data/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> data/test.csv has 20400 rows and 2 columns.
The columns are: id, class

-> data/train.csv has 95088 rows and 3 columns.
The columns are: id, class, segmentation

-> data/uw-madison-gi-tract-image-segmentation/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> data/uw-madison-gi-tract-image-segmentation/test.csv has 20400 rows and 2 columns.
The columns are: id, class

-> data/uw-madison-gi-tract-image-segmentation/train.csv has 95088 rows and 3 columns.
The columns are: id, class, segmentation

-> input/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> (stopped after 10 files for performance)

# 5. Target score

0.7308678564746787

# 6. Current score

0.01842

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.34714) has done: 'I fix the environment-breaking TensorFlow/Keras import issue by avoiding protobuf-backed TF entirely (it’s triggering the `MessageFactory.GetPrototype` crash) and switching to a lightweight, deterministic, CPU-only baseline that still produces valid RLE masks. Then I remove the unsupported `workers/use_multiprocessing` arguments that break on the current Keras API, ensuring training/inference run end-to-end. Finally, to yield a reasonable non-empty segmentation (and thus a better score than an all-empty submission), I generate class masks from the MRI intensities using a per-slice thresholding heuristic and output them in the exact `id,class,predicted` format.'
- What this solution (achieved 0.01555) has done: 'Your current score is far below the target, so we should cautiously improve segmentation quality while keeping your non-TF heuristic pipeline intact. The biggest low-risk gain is to stop using a single global quantile threshold and instead use a per-slice robust z-score normalization plus Otsu thresholding (with a safe fallback), which better adapts across scans. Then we keep your morphology/connected-components cleanup but make it slightly more stable by filling holes and using organ-specific center priors (still purely heuristic, same overall approach). Finally, we ensure the output remains correctly aligned with `test.csv`/`sample_submission.csv` and still writes a valid `submission.csv`.'
- What this solution (achieved 0.0161) has done: 'We keep your pure-OpenCV heuristic pipeline intact but fix the biggest scoring issue: your RLE encoding is currently row-major, while this competition expects column-major (top-to-bottom then left-to-right), which can destroy Dice/Hausdorff and explains the very low score. I change only the RLE encode/decode to the correct Fortran-order convention and ensure masks are encoded as contiguous runs without duplicates. To avoid any accidental misalignment, I also preserve the exact `sample_submission.csv` row order by filling predictions via a keyed merge as you already do. These minimal changes should significantly increase the score toward your target without changing the heuristic segmentation logic.'
- What this solution (achieved 0.07916) has done: 'Your current score (0.0161) is far below the target (0.7309), so we should improve segmentation quality while keeping your heuristic (non-TF) pipeline intact. The minimal high-impact fix is to stop producing three disjoint organ masks from a single “base” foreground via argmax spatial priors, and instead generate three independent masks using organ-specific spatial weighting + adaptive (per-slice) thresholds—this preserves the overall heuristic approach (thresholding + morphology) but removes a major failure mode where correct regions get suppressed. I also add a tiny, deterministic intensity smoothing before thresholding to reduce speckle and stabilize connected-components cleanup without changing the pipeline type. Submission writing, paths, and RLE (already corrected to Fortran order) remain the same.'
- What this solution (achieved 0.01669) has done: 'Your current score is far below the target, so we make the smallest changes that improve mask quality without changing the overall heuristic approach (read → normalize → threshold → morphology → RLE). The main issue is that the current organ masks are derived from a single “base” mask, so true organ regions can be missed; we keep the same pipeline but relax the gating by using organ-specific seeds plus geodesic-like reconstruction (iterative dilations constrained by a permissive candidate mask). We also make the candidate mask slightly more stable by combining Otsu with a high-quantile fallback and light edge suppression, which typically reduces spurious blobs and improves both Dice and Hausdorff. Submission formatting, row alignment, paths, and the Fortran-order RLE remain unchanged.'
- What this solution (achieved 0.01668) has done: 'Your score is far below the target, so the smallest safe move is to improve mask quality without changing the overall heuristic pipeline (read → normalize → threshold → morphology → RLE). The biggest low-risk issue is that inference is done per-slice independently, which creates noisy 3D volumes and hurts the 3D Hausdorff component; we add a light, deterministic 1D smoothing of masks along the slice axis within each (case, day, class) stack. To keep semantics intact, we won’t change the core per-slice mask generation, only add a post-step that unions each slice with its immediate neighbors (a common stability trick for volumetric metrics). Finally, we keep the corrected Fortran-order RLE and the exact sample_submission row alignment, still writing `submission.csv`.'
- What this solution (achieved 0.01735) has done: 'Your score is far below the target, so we should make the smallest changes that plausibly improve mask quality without changing the overall heuristic pipeline (read → normalize → threshold/seed → morphology → RLE). The most likely issue is that the current Otsu thresholding is applied to an 8-bit quantized image (after percentile scaling), which can be unstable on these scans; switching to Otsu on a robust z-scored image (still per-slice, still Otsu) typically yields a much better candidate foreground. Then we keep your existing geodesic reconstruction and postprocessing, but make the candidate mask slightly safer by removing very thin edge artifacts before connected-components. Submission formatting, alignment, and Fortran-order RLE remain unchanged.'
- What this solution (achieved 0.01842) has done: 'Your current score (0.01735) is far below the target (0.7309), so we should make small, safe improvements that increase real segmentation quality without changing your overall heuristic pipeline (read → normalize → threshold/seed → morphology → RLE → submission). The most likely weak point is over-growing masks via (a) slice-neighbor union smoothing and (b) unconstrained geodesic dilation, which can hurt Hausdorff heavily; we replace the union-only z-smoothing with a majority vote across adjacent slices and add a tiny, deterministic “area cap” during reconstruction to prevent runaway growth. These are minimal, local changes: same inputs/outputs, same class priors, same RLE, same submission alignment. Everything still runs end-to-end and writes `submission.csv`.'

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

import cv2

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 200)


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


set_seed(42)



## === cell 1
repertory = "/kaggle/input/"

DIR = repertory + "uw-madison-gi-tract-image-segmentation/"
TRAIN_DIR = DIR + "train"
TEST_DIR = DIR + "test"
train_csv = DIR + "train.csv"
sample_sub = DIR + "sample_submission.csv"
test_csv = DIR + "test.csv"

df_train = pd.read_csv(train_csv)
df_train.head(10)




## === cell 2
class CFG:
    BATCH_SIZE = 8
    img_size = (256, 256, 1)
    n_fold = 5
    fold_selected = 1
    epochs = 3  # kept as provided (not used in this no-TF baseline)
    seed = 42


set_seed(CFG.seed)



## === cell 3
_SCAN_INDEX_CACHE = {}  # (subset_key, case_str, day_str) -> {slice_str: full_path}
_SCAN_META_CACHE = {}  # full_path -> (width,height,pxh,pxw) parsed from filename


def _build_scan_index(base_dir, case_str, day_str, subset_key):
    key = (subset_key, case_str, day_str)
    m = _SCAN_INDEX_CACHE.get(key)
    if m is not None:
        return m

    scans_dir = os.path.join(
        base_dir, f"case{case_str}", f"case{case_str}_day{day_str}", "scans"
    )
    idx = {}
    with os.scandir(scans_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            name = entry.name
            if not name.endswith(".png") or not name.startswith("slice_"):
                continue
            parts = name.split("_", 2)
            if len(parts) < 3:
                continue
            slice_str = parts[1]
            idx[slice_str] = entry.path
    if not idx:
        raise FileNotFoundError(f"No scans found in {scans_dir}")
    _SCAN_INDEX_CACHE[key] = idx
    return idx


def _resolve_scan_path_fast(base_dir, case_str, day_str, slice_str, subset_key):
    idx = _build_scan_index(base_dir, case_str, day_str, subset_key)
    p = idx.get(slice_str)
    if p is None:
        scans_dir = os.path.join(
            base_dir, f"case{case_str}", f"case{case_str}_day{day_str}", "scans"
        )
        pattern = os.path.join(scans_dir, f"slice_{slice_str}_*.png")
        matches = glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No scan found for pattern: {pattern}")
        matches.sort()
        p = matches[0]
        idx[slice_str] = p
    return p


def _parse_meta_from_path(p):
    m = _SCAN_META_CACHE.get(p)
    if m is not None:
        return m
    stem = os.path.basename(p)[:-4]  # strip .png
    toks = stem.rsplit("_", 4)
    w = np.int32(toks[1])
    h = np.int32(toks[2])
    pxh = np.float32(toks[3])
    pxw = np.float32(toks[4])
    m = (w, h, pxh, pxw)
    _SCAN_META_CACHE[p] = m
    return m


def preprocessing(df, subset="train"):
    df = df.copy()
    parts = df["id"].str.split("_", expand=True)
    df["case"] = parts[0].str.replace("case", "", regex=False).astype(np.int32)
    df["day"] = parts[1].str.replace("day", "", regex=False).astype(np.int32)
    df["slice"] = parts[3]

    base_dir = TRAIN_DIR if subset == "train" else TEST_DIR
    subset_key = "train" if subset == "train" else "test"

    case_s = df["case"].astype(str).to_numpy()
    day_s = df["day"].astype(str).to_numpy()
    slice_s = df["slice"].astype(str).to_numpy()

    paths = np.empty(len(df), dtype=object)
    for i, (cs, ds, ss) in enumerate(zip(case_s, day_s, slice_s)):
        paths[i] = _resolve_scan_path_fast(base_dir, cs, ds, ss, subset_key)
    df["path"] = paths

    widths = np.empty(len(df), dtype=np.int32)
    heights = np.empty(len(df), dtype=np.int32)
    pxh = np.empty(len(df), dtype=np.float32)
    pxw = np.empty(len(df), dtype=np.float32)
    for i, p in enumerate(paths):
        w, h, ph, pw = _parse_meta_from_path(p)
        widths[i] = w
        heights[i] = h
        pxh[i] = ph
        pxw[i] = pw
    df["width"] = widths
    df["height"] = heights
    df["px_spacing_h"] = pxh
    df["px_spacing_w"] = pxw
    return df




## === cell 4
train_df = preprocessing(df_train, subset="train")
train_df.head()




## === cell 5
def rle_decode(mask_rle, shape):
    """
    Competition convention: pixels numbered top-to-bottom, then left-to-right.
    That corresponds to flattening in Fortran order (column-major) for a (H,W) mask.
    """
    if (
        mask_rle is None
        or mask_rle == ""
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
    ):
        return np.zeros(shape, dtype=np.uint8)

    s = np.fromstring(mask_rle, sep=" ", dtype=np.int64)
    if s.size == 0:
        return np.zeros(shape, dtype=np.uint8)

    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    n = shape[0] * shape[1]
    diff = np.zeros(n + 1, dtype=np.int32)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    img = (np.cumsum(diff[:-1]) > 0).astype(np.uint8)

    return img.reshape(shape, order="F")


def rle_encode(img):
    """
    Competition convention: flatten in Fortran order (column-major).
    Ensures runs are sorted, positive, and non-overlapping by construction.
    """
    if img.ndim != 2:
        img = img.squeeze()
    img = (img > 0).astype(np.uint8)
    if img.size == 0:
        return ""

    pixels = img.reshape(-1, order="F")
    pad = np.concatenate(([0], pixels, [0])).astype(np.uint8)
    changes = np.flatnonzero(pad[1:] != pad[:-1]) + 1
    if changes.size == 0:
        return ""
    changes[1::2] -= changes[::2]
    return " ".join(map(str, changes.tolist()))




## === cell 6
def restructure(df, subset="train"):
    df_out = pd.DataFrame({"id": df["id"][::3].values})

    if subset == "train":
        df_out["large_bowel"] = df["segmentation"][::3].values
        df_out["small_bowel"] = df["segmentation"][1::3].values
        df_out["stomach"] = df["segmentation"][2::3].values

    df_out["path"] = df["path"][::3].values
    df_out["case"] = df["case"][::3].values
    df_out["day"] = df["day"][::3].values
    df_out["slice"] = df["slice"][::3].values
    df_out["width"] = df["width"][::3].values
    df_out["height"] = df["height"][::3].values

    df_out = df_out.reset_index(drop=True).fillna("")
    if subset == "train":
        df_out["count"] = np.sum(df_out.iloc[:, 1:4] != "", axis=1).values
    return df_out


DF_train = restructure(train_df, subset="train")
DF_train.head()




## === cell 7
def _read_norm_resized_gray(img_path, out_hw=(256, 256)):
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    img = cv2.resize(img, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    p5 = float(np.percentile(img, 5))
    p95 = float(np.percentile(img, 95))
    if p95 > p5:
        img = (img - p5) / (p95 - p5)
    else:
        mn, mx = float(img.min()), float(img.max())
        if mx > mn:
            img = (img - mn) / (mx - mn)
        else:
            img[:] = 0.0
    img = np.clip(img, 0.0, 1.0)
    return img


def _fill_holes(binary_mask):
    m = (binary_mask > 0).astype(np.uint8)
    if m.max() == 0:
        return m
    h, w = m.shape
    flood = m.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, seedPoint=(0, 0), newVal=1)
    holes = (flood == 0).astype(np.uint8)
    return (m | holes).astype(np.uint8)


def _postprocess_mask(m, min_area=80):
    m = (m > 0).astype(np.uint8)
    if m.max() == 0:
        return m
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)

    m = _fill_holes(m)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    out = np.zeros_like(m)
    for k in range(1, num):
        if stats[k, cv2.CC_STAT_AREA] >= min_area:
            out[labels == k] = 1
    return out


def _geodesic_reconstruct(seed, mask, iters=18, max_area_ratio=0.32):
    """
    Change (score-directed, minimal): add a conservative area cap to prevent runaway dilation.
    Overgrown masks are especially harmful for the 3D Hausdorff component, so we stop if the
    reconstruction exceeds a fixed fraction of the candidate mask area.
    """
    seed = (seed > 0).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)
    if seed.max() == 0 or mask.max() == 0:
        return np.zeros_like(seed, dtype=np.uint8)

    cur = (seed & mask).astype(np.uint8)
    if cur.max() == 0:
        return np.zeros_like(seed, dtype=np.uint8)

    cand_area = int(mask.sum())
    if cand_area <= 0:
        return np.zeros_like(seed, dtype=np.uint8)
    max_area = max(1, int(max_area_ratio * cand_area))

    kernel = np.ones((3, 3), np.uint8)
    for _ in range(iters):
        nxt = cv2.dilate(cur, kernel, iterations=1)
        nxt = (nxt & mask).astype(np.uint8)
        if int(nxt.sum()) > max_area:
            break
        if np.array_equal(nxt, cur):
            break
        cur = nxt
    return cur


def heuristic_predict_masks_from_image(img_256):
    """
    Produce 3 class masks (large_bowel, small_bowel, stomach) from a normalized 256x256 image.
    """
    H, W = img_256.shape

    med = float(np.median(img_256))
    mad = float(np.median(np.abs(img_256 - med)))
    if mad < 1e-6:
        z = (img_256 - med).astype(np.float32)
    else:
        z = ((img_256 - med) / (1.4826 * mad)).astype(np.float32)
    z = np.clip(z, -3.0, 6.0)
    z01 = (z - (-3.0)) / (6.0 - (-3.0))
    img8 = (z01 * 255.0).astype(np.uint8)

    otsu_thr, _ = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    q_thr = float(np.quantile(z, 0.85))
    cand_q = (z >= q_thr).astype(np.uint8)
    cand_o = (
        (img8 > otsu_thr).astype(np.uint8)
        if otsu_thr > 0
        else np.zeros((H, W), np.uint8)
    )

    candidate = ((cand_o | cand_q) > 0).astype(np.uint8)

    border = 6
    candidate[:border, :] = 0
    candidate[-border:, :] = 0
    candidate[:, :border] = 0
    candidate[:, -border:] = 0

    yy, xx = np.ogrid[:H, :W]
    cy, cx = H / 2.0, W / 2.0
    r2 = ((yy - cy) ** 2) / (0.65 * H) ** 2 + ((xx - cx) ** 2) / (0.70 * W) ** 2
    candidate = (candidate & (r2 <= 1.0)).astype(np.uint8)

    candidate = _postprocess_mask(candidate, min_area=30)
    if candidate.max() == 0:
        z0 = np.zeros((H, W), np.uint8)
        return z0, z0.copy(), z0.copy()

    y = np.arange(H, dtype=np.float32)[:, None]
    x = np.arange(W, dtype=np.float32)[None, :]

    def gauss2d(y0, x0, sy, sx):
        return np.exp(
            -(((y - y0) ** 2) / (2 * sy * sy) + ((x - x0) ** 2) / (2 * sx * sx))
        ).astype(np.float32)

    w_stom = gauss2d(0.30 * H, 0.50 * W, 0.20 * H, 0.28 * W)
    w_small = gauss2d(0.55 * H, 0.50 * W, 0.22 * H, 0.30 * W)
    w_large = gauss2d(0.78 * H, 0.50 * W, 0.18 * H, 0.35 * W)

    def make_mask(weight_map, q_seed, recon_iters, min_area, max_area_ratio):
        score = (z01 * (0.25 + 0.75 * weight_map)).astype(np.float32)

        svals = score[candidate > 0]
        thr_seed = (
            float(np.quantile(svals, q_seed))
            if svals.size
            else float(np.quantile(score, q_seed))
        )
        seed = ((score >= thr_seed) & (candidate > 0)).astype(np.uint8)

        recon = _geodesic_reconstruct(
            seed, candidate, iters=recon_iters, max_area_ratio=max_area_ratio
        )
        recon = _postprocess_mask(recon, min_area=min_area)
        return recon

    stomach = make_mask(
        w_stom, q_seed=0.80, recon_iters=16, min_area=35, max_area_ratio=0.28
    )
    small = make_mask(
        w_small, q_seed=0.79, recon_iters=18, min_area=35, max_area_ratio=0.30
    )
    large = make_mask(
        w_large, q_seed=0.79, recon_iters=18, min_area=35, max_area_ratio=0.34
    )

    return large, small, stomach




## === cell 8
sub_df = pd.read_csv(sample_sub)

df_test = pd.read_csv(test_csv)
test_df = preprocessing(df_test, subset="test")
test_df.head(5)




## === cell 9
def _smooth_masks_along_slices(rows, masks_by_id, class_name):
    """
    Change (score-directed, minimal): replace neighbor-union with a 3-slice majority vote.
    Union tends to systematically inflate masks and can worsen 3D Hausdorff; majority voting
    still reduces flicker but is less prone to overgrowth.
    """
    if len(rows) <= 1:
        return

    ms = [masks_by_id[r["id"]][class_name] for r in rows]
    if len(ms) <= 1:
        return
    out = []
    for i in range(len(ms)):
        m0 = ms[i]
        if 0 < i < len(ms) - 1:
            m = (
                (
                    ms[i - 1].astype(np.uint8)
                    + m0.astype(np.uint8)
                    + ms[i + 1].astype(np.uint8)
                )
                >= 2
            ).astype(np.uint8)
        elif i == 0:
            m = ((m0.astype(np.uint8) + ms[i + 1].astype(np.uint8)) >= 1).astype(
                np.uint8
            )
        else:
            m = ((ms[i - 1].astype(np.uint8) + m0.astype(np.uint8)) >= 1).astype(
                np.uint8
            )
        out.append(m)
    for r, m in zip(rows, out):
        masks_by_id[r["id"]][class_name] = m


def infer_heuristic(test_df):
    pred_ids, pred_classes, pred_rle = [], [], []

    uniq = test_df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    masks_by_id = {}
    uniq_records = []

    for i in range(len(uniq)):
        id_ = uniq.loc[i, "id"]
        p = uniq.loc[i, "path"]
        w = int(uniq.loc[i, "width"])
        h = int(uniq.loc[i, "height"])
        case = int(uniq.loc[i, "case"])
        day = int(uniq.loc[i, "day"])
        slice_str = str(uniq.loc[i, "slice"])
        try:
            slice_int = int(slice_str)
        except Exception:
            slice_int = 0

        img_256 = _read_norm_resized_gray(p, out_hw=(CFG.img_size[0], CFG.img_size[1]))
        m_large_256, m_small_256, m_stom_256 = heuristic_predict_masks_from_image(
            img_256
        )

        m_large = cv2.resize(m_large_256, (w, h), interpolation=cv2.INTER_NEAREST)
        m_small = cv2.resize(m_small_256, (w, h), interpolation=cv2.INTER_NEAREST)
        m_stom = cv2.resize(m_stom_256, (w, h), interpolation=cv2.INTER_NEAREST)

        masks_by_id[id_] = {
            "large_bowel": (m_large > 0).astype(np.uint8),
            "small_bowel": (m_small > 0).astype(np.uint8),
            "stomach": (m_stom > 0).astype(np.uint8),
        }
        uniq_records.append(
            {"id": id_, "case": case, "day": day, "slice_int": slice_int}
        )

    uniq_meta = pd.DataFrame(uniq_records)
    uniq_meta = uniq_meta.sort_values(["case", "day", "slice_int"]).reset_index(
        drop=True
    )

    for (case, day), g in uniq_meta.groupby(["case", "day"], sort=False):
        rows = g.to_dict("records")
        _smooth_masks_along_slices(rows, masks_by_id, "large_bowel")
        _smooth_masks_along_slices(rows, masks_by_id, "small_bowel")
        _smooth_masks_along_slices(rows, masks_by_id, "stomach")

    for i in range(len(test_df)):
        id_ = test_df.loc[i, "id"]
        cls = test_df.loc[i, "class"]
        m = masks_by_id[id_][cls]
        pred_ids.append(id_)
        pred_classes.append(cls)
        pred_rle.append(rle_encode(m))

    return pred_rle, pred_ids, pred_classes


pred_rle, pred_ids, pred_classes = infer_heuristic(test_df)



## === cell 10
submission = pd.DataFrame(
    {"id": pred_ids, "class": pred_classes, "predicted": pred_rle}
)

sub_df = pd.read_csv(sample_sub)
sub_df = sub_df.drop(columns=["predicted"])
sub_df = sub_df.merge(submission, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("")

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)

print("Saved:", sub_path, "rows:", len(sub_df), "cols:", list(sub_df.columns))
sub_df.head()
