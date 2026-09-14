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

0.6435472688166154

# 6. Current score

0.0163

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The failures come from relying on the `kaggle_imsegm` and `flash` packages, which are not available/installed in this runtime, so imports and all downstream training/prediction cells crash. To keep the pipeline runnable end-to-end with minimal changes and still produce a valid `.csv` submission, I removed those unavailable dependencies and replaced them with small local utilities for (1) parsing IDs to image paths, and (2) RLE encoding. Since training a full segmentation model is not feasible here without the missing libraries, the script generates an all-empty mask submission (valid format, score be low but non-erroring); once the environment supports the original packages again, you can re-enable the original modeling cells.'
- What this solution (achieved 0.03132) has done: 'Your current 0.0 score comes from predicting empty masks for every (id, class), which is valid format but yields near-zero Dice for non-empty ground truths. To move toward the target score with minimal core-logic change (still no external ML packages), I replace the always-empty prediction with a simple, legitimate image-derived threshold mask built directly from each test PNG, then RLE-encode it. This keeps the pipeline end-to-end, uses only installed libraries, preserves evaluation semantics (binary masks per class), and should increase score above 0.0 without introducing any new modeling framework. I also ensure we read from the correct `test/` folder (not `train/`) when building test-time masks and keep the submission schema identical.'
- What this solution (achieved 0.03262) has done: 'The timeout is dominated by repeated disk scans (`glob` over the scans directory) and an extremely slow pure-Python connected-component BFS that runs per image. I make `extract_tract_details_local` O(1) amortized by caching a per-(split,case,day) mapping from slice index → png path/shape so we never `glob` the same folder more than once. I also replace the Python-loop BFS with an equivalent connected-component routine based on `cv2.connectedComponentsWithStats` (available in Kaggle), preserving the “keep largest 4-connected component” logic exactly but running in optimized C++. Finally, I avoid repeated percentile computations by using OpenCV min/max + histogram percentile (still exact percentiles) and keep all I/O paths and segmentation logic the same.'
- What this solution (achieved 0.01506) has done: 'Your current score is far below the target, so we should improve predictions with the smallest change that preserves your existing heuristic pipeline. The biggest issue is that you currently generate one binary mask per `id` and reuse it for all 3 classes, which mis-specifies the submission and hurts the per-class metric; we instead derive three class-specific masks from the same image using different (but still simple) thresholds while keeping the same normalization, border clearing, and “keep largest 4-connected component” post-processing. We also ensure the `id`→mask mapping is built from the actual `test.csv` IDs (not `sample_submission`), and keep submission formatting identical. These minimal adjustments should legitimately increase the score toward the target without changing the overall approach or adding new dependencies.'
- What this solution (achieved 0.00715) has done: 'Your current gap to the target is very large, so we should legitimately improve mask quality while keeping your heuristic (single-image thresholding + border clearing + largest CC) intact. The biggest low-risk gain is to make the threshold class-specific *and* image-adaptive using Otsu’s method on the same normalized image, which stays within the same core “thresholding” approach but is far better calibrated across scans than fixed thresholds. To avoid over-segmenting (which hurts Hausdorff), we add a minimal per-class morphological cleanup (open/close) before the existing “keep largest component” step—this is still post-processing of the binary mask, not a new model. Finally, we keep submission formatting identical and ensure (id,class) alignment remains correct.'
- What this solution (achieved 0.01536) has done: 'Your current score is far below the target, so the smallest safe way to move upward is to keep your exact heuristic pipeline (normalize → threshold → morphology → border clear → largest CC → RLE) but make two calibration fixes that reduce obvious false positives/negatives without changing the approach. First, compute Otsu on a central crop (avoids background/border dominating the threshold) and clamp thresholds to stable per-class ranges; this usually improves Dice while not exploding Hausdorff. Second, add a tiny “area floor” per class after morphology (before largest CC) so we don’t submit tiny specks that can severely hurt Hausdorff. Finally, keep submission alignment identical but build `df_sub` from `sample_submission` only if present; otherwise fall back to `test.csv` to avoid any misalignment edge cases.'
- What this solution (achieved 0.01534) has done: 'Your current score (0.01536) is far below the target (0.64355), so we should make a small, legitimate improvement to the existing heuristic without changing its core structure (normalize → threshold → morphology → border clear → largest CC → RLE). The biggest low-risk issue is that `keep_largest_connected_component()` currently collapses each class mask to only one component, which often deletes true multi-part anatomy (especially bowel loops) and can severely hurt Dice; we keep the same post-processing but change it to “keep all sufficiently-large components” (size-filter) while preserving the “remove specks” intent. To avoid worsening Hausdorff with scattered false positives, we add a conservative per-class minimum component area threshold, and we keep the existing global “area floor” check. All paths, submission schema, and the rest of the pipeline remain unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.00737) has done: 'Your score is far below the target (need to increase), so the smallest likely gain without changing the overall heuristic pipeline is to (1) fix a key image–slice mismatch bug: your fallback `pngs_sorted[slc]` indexing is wrong because `slc` is a 1-based slice number while list indices are 0-based, which can completely desynchronize masks from the correct images. Then, still keeping the same normalize→threshold→morphology→border-clear→CC-cleanup flow, we make the thresholding slightly more robust by computing Otsu on a more central crop (reduces background dominance) and by adding a “fill holes” step (floodfill-based) that improves Dice for hollow organs without expanding boundaries too aggressively (helps Hausdorff). Finally, we keep submission formatting identical and still write `submission.csv`.'
- What this solution (achieved 0.0163) has done: 'Your current score is far below the target, so we should make a small, legitimate quality improvement while keeping the exact heuristic pipeline (normalize → adaptive threshold → morphology → border clear → hole fill → CC area filter → RLE) intact. The biggest likely issue now is that a single global Otsu threshold on a crop is still too brittle across slices, so we keep Otsu but also compute a second adaptive threshold (triangle) and blend them conservatively per class to stabilize across intensity distributions. Next, we add a very small per-class “soft shrink” option using one extra erosion for classes that tend to over-segment, which can improve the Hausdorff part without changing the overall method. Finally, we keep the same I/O paths and submission schema, and we keep runtime bounded by doing only O(1) extra OpenCV ops per image.'

# 9. Code solution

## === cell 0
import os, glob, re
import numpy as np
import pandas as pd

DATASET_FOLDER = "/kaggle/input/uw-madison-gi-tract-image-segmentation"

df_train = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))
df_ssub = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
WITH_SUBMISSION = not df_ssub.empty

print(
    "train:",
    df_train.shape,
    "sample_submission:",
    df_ssub.shape,
    "WITH_SUBMISSION:",
    WITH_SUBMISSION,
)



## === cell 1
_ID_RE = re.compile(r"^(case\d+)_day(\d+)_slice_(\d+)$")

_SCAN_INDEX_CACHE = {}


def _build_scan_index(dataset_dir: str, split: str, case: str, day: int):
    scan_dir = os.path.join(dataset_dir, split, case, f"{case}_day{day}", "scans")
    if not os.path.isdir(scan_dir):
        return None, {}

    pngs = glob.glob(os.path.join(scan_dir, "*.png"))
    idx = {}
    for p in pngs:
        stem = os.path.splitext(os.path.basename(p))[0]
        parts = stem.split("_")
        if len(parts) >= 3 and parts[2].isdigit():
            slc = int(parts[2])
            h = int(parts[0]) if len(parts) >= 2 and parts[0].isdigit() else np.nan
            w = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else np.nan
            if slc not in idx:
                idx[slc] = (p, h, w)

    pngs_sorted = sorted(pngs)
    return scan_dir, {"by_slice": idx, "sorted": pngs_sorted}


def extract_tract_details_local(id_str: str, dataset_dir: str, split: str = "train"):
    """
    Returns (Case, Day, Slice, image, image_path, height, width).
    """
    m = _ID_RE.match(id_str)
    if not m:
        raise ValueError(f"Unexpected id format: {id_str}")
    case, day, slc = m.group(1), int(m.group(2)), int(m.group(3))

    cache_key = (dataset_dir, split, case, day)
    if cache_key not in _SCAN_INDEX_CACHE:
        _SCAN_INDEX_CACHE[cache_key] = _build_scan_index(dataset_dir, split, case, day)

    scan_dir, data = _SCAN_INDEX_CACHE[cache_key]
    if scan_dir is None:
        image_path = os.path.join(
            split, case, f"{case}_day{day}", "scans", "MISSING.png"
        )
        return case, day, slc, os.path.basename(image_path), image_path, np.nan, np.nan

    by_slice = data["by_slice"]
    if slc in by_slice:
        chosen, h, w = by_slice[slc]
        image = os.path.basename(chosen)
        rel_path = os.path.relpath(chosen, dataset_dir)
        return case, day, slc, image, rel_path, h, w

    pngs_sorted = data["sorted"]
    idx0 = slc - 1
    if idx0 < 0 or idx0 >= len(pngs_sorted):
        image_path = os.path.join(split, case, f"{case}_day{day}", "scans", "OOR.png")
        return case, day, slc, os.path.basename(image_path), image_path, np.nan, np.nan

    chosen = pngs_sorted[idx0]
    stem = os.path.splitext(os.path.basename(chosen))[0]
    parts = stem.split("_")
    h = int(parts[0]) if len(parts) >= 2 and parts[0].isdigit() else np.nan
    w = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else np.nan
    image = os.path.basename(chosen)
    rel_path = os.path.relpath(chosen, dataset_dir)
    return case, day, slc, image, rel_path, h, w


rows = [
    extract_tract_details_local(x, DATASET_FOLDER, split="train")
    for x in df_train["id"].values
]
tmp = pd.DataFrame(
    rows, columns=["Case", "Day", "Slice", "image", "image_path", "height", "width"]
)
df_train = pd.concat([df_train, tmp], axis=1)
print(df_train.head())




## === cell 2
def rle_encode(mask: np.ndarray) -> str:
    """
    mask: 2D numpy array of 0/1 values.
    Kaggle RLE: flatten in Fortran order (column-major).
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs = changes[::2]
    lengths = changes[1::2] - changes[::2]
    if len(runs) == 0:
        return ""
    out = " ".join(str(x) for pair in zip(runs, lengths) for x in pair)
    return out


assert rle_encode(np.zeros((5, 7), np.uint8)) == ""



## === cell 3
df_test = pd.read_csv(os.path.join(DATASET_FOLDER, "test.csv"))

df_sub = df_ssub.copy() if WITH_SUBMISSION else df_test.assign(predicted="")

print("test:", df_test.shape, "sub:", df_sub.shape)
print(df_sub.head())



## === cell 4
try:
    from PIL import Image
except Exception as e:
    raise RuntimeError(
        "PIL is required to read PNGs in this minimal baseline. "
        "It is typically available in Kaggle Python images."
    ) from e

import cv2


def load_grayscale_uint8(png_path: str) -> np.ndarray:
    img = Image.open(png_path).convert("L")
    return np.asarray(img, dtype=np.uint8)


def keep_components_by_area(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    if mask_u8.sum() == 0:
        return mask_u8
    m = (mask_u8 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    if num_labels <= 1:
        return mask_u8

    areas = stats[:, cv2.CC_STAT_AREA]
    keep = np.zeros(num_labels, dtype=bool)
    keep[0] = False
    for lab in range(1, num_labels):
        if int(areas[lab]) >= int(min_area):
            keep[lab] = True

    out = keep[labels].astype(np.uint8)
    return out


def _percentile_u8_from_hist(hist: np.ndarray, p: float, total: int) -> int:
    target = p * total
    cdf = 0
    for v in range(256):
        cdf += int(hist[v])
        if cdf >= target:
            return v
    return 255


def _normalized_image(img_u8: np.ndarray) -> np.ndarray:
    """
    Keep the same 5/95 normalization pipeline (core logic preserved).
    """
    hist = np.bincount(img_u8.ravel(), minlength=256)
    total = int(img_u8.size)
    lo = _percentile_u8_from_hist(hist, 0.05, total)
    hi = _percentile_u8_from_hist(hist, 0.95, total)
    if hi <= lo:
        return None
    norm = (img_u8.astype(np.float32) - float(lo)) / float(hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    return norm


def _fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    """
    Keep: hole filling (post-processing) to improve Dice for hollow organs.
    """
    if mask_u8.sum() == 0:
        return mask_u8
    m = (mask_u8 > 0).astype(np.uint8) * 255
    h, w = m.shape
    ff = m.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_mask, seedPoint=(0, 0), newVal=255)
    inv = cv2.bitwise_not(ff)
    filled = cv2.bitwise_or(m, inv)
    return (filled > 0).astype(np.uint8)


def _postprocess_mask(m: np.ndarray, min_cc_area: int, fill_holes: bool) -> np.ndarray:
    """
    Same border clearing, then optional hole-filling, then CC cleanup (area-filter).
    """
    m = m.astype(np.uint8)
    if m.shape[0] > 8 and m.shape[1] > 8:
        m[:2, :] = 0
        m[-2:, :] = 0
        m[:, :2] = 0
        m[:, -2:] = 0
    if fill_holes:
        m = _fill_holes(m)
    m = keep_components_by_area(m, min_area=min_cc_area)
    return m


def simple_foreground_mask(img_u8: np.ndarray) -> np.ndarray:
    norm = _normalized_image(img_u8)
    if norm is None:
        return np.zeros_like(img_u8, dtype=np.uint8)
    m = (norm > 0.50).astype(np.uint8)
    return _postprocess_mask(m, min_cc_area=200, fill_holes=False)


def _center_crop(norm01: np.ndarray, frac: float = 0.60) -> np.ndarray:
    h, w = norm01.shape
    ch = max(1, int(h * frac))
    cw = max(1, int(w * frac))
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    return norm01[y0 : y0 + ch, x0 : x0 + cw]


def _otsu_threshold_on_norm(norm01: np.ndarray) -> float:
    n8 = (norm01 * 255.0 + 0.5).astype(np.uint8)
    thr_u8, _ = cv2.threshold(n8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr_u8) / 255.0


def _triangle_threshold_on_norm(norm01: np.ndarray) -> float:
    n8 = (norm01 * 255.0 + 0.5).astype(np.uint8)
    thr_u8, _ = cv2.threshold(n8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return float(thr_u8) / 255.0


def simple_class_mask(img_u8: np.ndarray, cls: str) -> np.ndarray:
    """
    Keep: normalize -> adaptive threshold -> morphology -> border clear -> hole fill -> CC cleanup -> RLE.
    """
    norm = _normalized_image(img_u8)
    if norm is None:
        return np.zeros_like(img_u8, dtype=np.uint8)

    crop = _center_crop(norm, frac=0.60)

    base_otsu = _otsu_threshold_on_norm(crop)
    base_tri = _triangle_threshold_on_norm(crop)
    base_thr = 0.7 * base_otsu + 0.3 * base_tri

    if cls == "large_bowel":
        delta = -0.03
        k_open, k_close = 3, 5
        thr_min, thr_max = 0.18, 0.75
        min_area = 250
        min_cc_area = 650
        fill_holes = True
        extra_erode = (
            1  # Change (Hausdorff-oriented): slight shrink to reduce boundary FP
        )
    elif cls == "small_bowel":
        delta = +0.02
        k_open, k_close = 3, 3
        thr_min, thr_max = 0.22, 0.80
        min_area = 180
        min_cc_area = 500
        fill_holes = True
        extra_erode = 0
    elif cls == "stomach":
        delta = -0.01
        k_open, k_close = 3, 5
        thr_min, thr_max = 0.16, 0.75
        min_area = 220
        min_cc_area = 600
        fill_holes = True
        extra_erode = 1  # Change (Hausdorff-oriented): reduce common overfill
    else:
        delta = 0.0
        k_open, k_close = 3, 3
        thr_min, thr_max = 0.15, 0.85
        min_area = 200
        min_cc_area = 550
        fill_holes = False
        extra_erode = 0

    thr = float(np.clip(base_thr + delta, thr_min, thr_max))
    m = (norm > thr).astype(np.uint8)

    if m.any():
        ker_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        ker_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker_open, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker_close, iterations=1)

        if extra_erode > 0 and m.any():
            ker_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            m = cv2.erode(m, ker_e, iterations=extra_erode)

        if int(m.sum()) < int(min_area):
            m[:] = 0

    return _postprocess_mask(m, min_cc_area=min_cc_area, fill_holes=fill_holes)


unique_test_ids = df_test["id"].unique().tolist()

missing = 0
case_day = set()
for id_str in unique_test_ids:
    m = _ID_RE.match(id_str)
    if m:
        case_day.add((m.group(1), int(m.group(2))))
for case, day in sorted(case_day):
    cache_key = (DATASET_FOLDER, "test", case, day)
    if cache_key not in _SCAN_INDEX_CACHE:
        _SCAN_INDEX_CACHE[cache_key] = _build_scan_index(
            DATASET_FOLDER, "test", case, day
        )

id_class_to_rle = {}

for id_str in unique_test_ids:
    case, day, slc, image, rel_path, h, w = extract_tract_details_local(
        id_str, DATASET_FOLDER, split="test"
    )
    full_path = os.path.join(DATASET_FOLDER, rel_path)
    if not os.path.exists(full_path):
        for cls in ["large_bowel", "small_bowel", "stomach"]:
            id_class_to_rle[(id_str, cls)] = ""
        missing += 1
        continue

    img_u8 = load_grayscale_uint8(full_path)
    for cls in ["large_bowel", "small_bowel", "stomach"]:
        mask = simple_class_mask(img_u8, cls)
        id_class_to_rle[(id_str, cls)] = rle_encode(mask)

print("Built masks for ids:", len(unique_test_ids), "missing_pngs:", missing)
print("Example RLE (first non-empty if exists):")
for (id_str, cls), rle in list(id_class_to_rle.items())[:120]:
    if rle != "":
        print(id_str, cls, rle[:80], "...")
        break



## === cell 5
df_sub["predicted"] = [
    id_class_to_rle.get((i, c), "")
    for i, c in zip(df_sub["id"].values, df_sub["class"].values)
]

assert list(df_sub.columns) == ["id", "class", "predicted"]
assert len(df_sub) == 20400

out_path = "submission.csv"
df_sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(pd.read_csv(out_path).head())
print("Non-empty predicted rows:", int((df_sub["predicted"] != "").sum()))
