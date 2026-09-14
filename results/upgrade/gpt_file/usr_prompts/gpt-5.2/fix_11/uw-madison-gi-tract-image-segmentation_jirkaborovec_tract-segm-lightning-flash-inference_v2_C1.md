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

0.3812780953400622

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `lightning-flash` and `kaggle_imsegm` (which are not importable in your environment) by replacing them with small, local equivalents for (1) parsing `id` to image paths, (2) loading/resizing images, and (3) RLE encoding. To keep the pipeline end-to-end and stable within the time limit, I generate a valid submission by predicting empty masks for all rows (this is score-low but guarantees a correct `.csv` submission while preserving the “no extra packages” constraint). I also fix missing/undefined variables (`DATASET_IMAGES`, `trainer`, `SemanticSegmentation`, etc.) by removing the broken Flash-based inference path and writing the submission directly from `test.csv`/`sample_submission.csv` schema. The result run end-to-end and write `submission.csv` with the required `id,class,predicted` columns.'
- What this solution (achieved 0.06335) has done: 'Your current 0.0 score comes from predicting empty masks for every (id, class), which is valid but yields near-worst Dice/Hausdorff. To move toward the target score with minimal core-logic change, I keep the same submission-building pipeline but replace the always-empty prediction with a fast, deterministic, image-based heuristic segmentation (simple intensity thresholding + light morphology) per slice. I derive each test slice’s image path from the `id` string, load the PNG with PIL (no extra packages), produce a binary mask, and RLE-encode it with your existing encoder. This should increase the score above 0.0 while still running within the time limit and writing a correct `submission.csv` in the required format.'
- What this solution (achieved 0.28356) has done: 'The timeout is dominated by repeated disk I/O (opening ~6,800 PNGs) and very slow Python-level morphology/connected-components loops executed per slice and per class. I keep the exact heuristic logic, but make it run fast by (1) decoding each PNG with `cv2` (much faster than PIL here), (2) vectorizing the 3×3 dilation/erosion neighborhood using precomputed shifted views (no nested dy/dx loops), (3) replacing the Python BFS connected-component search with OpenCV’s equivalent connected-components (same 4-neighbor semantics), and (4) removing redundant work by computing class masks/RLE once per unique id and filling the submission via a vectorized merge instead of Python loops. These changes preserve algorithm semantics (same thresholds, same morphology iteration counts, same “largest component” definition) while cutting constant factors enough to fit 600s.'
- What this solution (achieved 0.28347) has done: 'I keep your current heuristic segmentation core logic intact, but fix two issues that are likely holding your score back versus the target: (1) the dilation/erosion implemented via `np.roll` is currently “wrapping” at image borders, creating false connections; I switch these ops to OpenCV morphology with the same 3×3 kernel and iteration counts (same semantics, no wrap-around artifacts). (2) I add a minimal, deterministic per-class gating that suppresses tiny, noisy masks (which typically hurts both Dice and Hausdorff) without changing your thresholds or overall approach. These are small, local changes that should improve mean Dice and reduce extreme Hausdorff outliers, moving the score upward toward your target while staying within the 600s budget and still producing a valid `submission.csv`.'
- What this solution (achieved 0.32633) has done: 'I keep your heuristic segmentation core logic (same thresholds, same morphology iteration counts, same “largest component” idea) but fix a key semantic mismatch with the competition requirement: you currently generate three different class masks, while evaluation expects the same *binary* mask for all classes within an image (union/joined mask at encoding time). I therefore compute one final binary mask per slice (the foreground mask after your close/open/border-clear), RLE-encode it once, and reuse that same RLE for stomach/large_bowel/small_bowel rows. This should increase your score toward the target by avoiding systematic penalties from class-wise mask disagreement without adding any new modeling or extra packages. I also remove the per-class min-area gating (since it can wrongly zero out union masks) and instead apply a single small minimum-area filter to the unified mask to reduce tiny-noise outliers in Hausdorff.'
- What this solution (achieved 0.14655) has done: 'Your current score (0.32633) is below the target (0.38128), so we should cautiously improve while keeping your heuristic pipeline intact. The most direct minimal gain is to reduce systematic false-positives (which hurt both Dice and Hausdorff) by masking out obvious “outside body” background using a very cheap body-mask derived from low-intensity thresholding, then keeping your existing percentile-normalize + threshold + close/open + border-clear logic unchanged on the remaining region. I also replace the slow per-image quantile computation with OpenCV percentiles (histogram-based) to stabilize thresholds and reduce slice-to-slice noise without changing the overall method. Finally, I keep the union-mask submission semantics exactly as you already corrected, and still write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current heuristic is already producing a valid submission but is underperforming the target, so the smallest safe improvement is to reduce systematic false positives and keep the mask closer to plausible anatomy. I keep your exact pipeline structure (percentile normalize → threshold → close/open → border-clear → union mask reused for all classes), but (1) make the “body mask” more reliable by selecting the largest connected component after thresholding so the outside/background doesn’t leak in, and (2) replace the fixed `x > 0.55` cut with an Otsu threshold computed on the normalized image *within the body mask*, blended conservatively with your existing fixed threshold to avoid over-shifting behavior. These two changes typically improve both Dice and Hausdorff by removing spurious islands while not changing the overall approach or adding dependencies. The output format/path stays identical and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import re
import glob
import sys
import numpy as np
import pandas as pd

DATASET_FOLDER = "/kaggle/input/uw-madison-gi-tract-image-segmentation"

train_csv_path = os.path.join(DATASET_FOLDER, "train.csv")
test_csv_path = os.path.join(DATASET_FOLDER, "test.csv")
sample_sub_path = os.path.join(DATASET_FOLDER, "sample_submission.csv")

assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
assert os.path.exists(test_csv_path), f"Missing: {test_csv_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"

df_train = pd.read_csv(train_csv_path)
LABELS = sorted(df_train["class"].unique().tolist())

print("Python:", sys.version)
print("Labels:", LABELS)
print("train:", df_train.shape)



## === cell 1
df_test = pd.read_csv(test_csv_path)
df_ssub = pd.read_csv(sample_sub_path)

print("test:", df_test.shape, "columns:", df_test.columns.tolist())
print("sample_submission:", df_ssub.shape, "columns:", df_ssub.columns.tolist())

if set(["id", "class"]).issubset(df_test.columns):
    base = df_test[["id", "class"]].copy()
else:
    base = df_ssub[["id", "class"]].copy()

print("base:", base.shape)




## === cell 2
def rle_encode(mask: np.ndarray) -> str:
    """
    Kaggle RLE (1-indexed), pixels read top-to-bottom then left-to-right.
    mask: 2D array of {0,1} with shape (H,W).
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask.shape}")
    pixels = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    if pixels.size == 0:
        return ""
    padded = np.empty(pixels.size + 2, dtype=np.uint8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = pixels
    runs = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    runs[1::2] -= runs[::2]
    if runs.size == 0:
        return ""
    return " ".join(map(str, runs.tolist()))


m = np.zeros((3, 4), dtype=np.uint8)
m[0, 0] = 1
m[1, 0] = 1
assert rle_encode(m) == "1 2"



## === cell 3
import cv2

TEST_FOLDER = os.path.join(DATASET_FOLDER, "test")

_id_pat = re.compile(r"^case(\d+)_day(\d+)_slice_(\d+)$")

_scans_cache = {}  # (case_int, day_int) -> list[str] sorted png paths


def _get_sorted_pngs_for_case_day(case_i: int, day_i: int) -> list:
    key = (case_i, day_i)
    pngs = _scans_cache.get(key)
    if pngs is not None:
        return pngs
    scans_dir = os.path.join(
        TEST_FOLDER, f"case{case_i}", f"case{case_i}_day{day_i}", "scans"
    )
    pngs = sorted(glob.glob(os.path.join(scans_dir, "*.png")))
    if not pngs:
        raise FileNotFoundError(f"No PNGs found in: {scans_dir}")
    _scans_cache[key] = pngs
    return pngs


def id_to_image_path(image_id: str) -> str:
    """
    Map 'case###_day##_slice_####' -> corresponding PNG under test folder.

    Slice ids are 1-based in this competition (slice_0001...), while Python lists are 0-based.
    We convert slice_n -> slice_n-1 to avoid slice misalignment.
    """
    m = _id_pat.match(image_id)
    if m is None:
        raise ValueError(f"Unexpected id format: {image_id}")
    case_n, day_n, slice_n_str = m.group(1), m.group(2), m.group(3)
    case_i = int(case_n)
    day_i = int(day_n)
    slice_n = int(slice_n_str)

    pngs = _get_sorted_pngs_for_case_day(case_i, day_i)

    idx = slice_n - 1
    if idx < 0 or idx >= len(pngs):
        raise IndexError(
            f"Slice index out of range after 1-based->0-based conversion: "
            f"id={image_id} slice={slice_n} idx={idx} n_slices={len(pngs)}"
        )
    return pngs[idx]


_K3 = np.ones((3, 3), dtype=np.uint8)


def _binary_dilate(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    m = (mask.astype(np.uint8, copy=False) > 0).astype(np.uint8, copy=False)
    if iters <= 0:
        return m
    return cv2.dilate(m, _K3, iterations=int(iters))


def _binary_erode(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    m = (mask.astype(np.uint8, copy=False) > 0).astype(np.uint8, copy=False)
    if iters <= 0:
        return m
    return cv2.erode(m, _K3, iterations=int(iters))


def _binary_open(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    return _binary_dilate(_binary_erode(mask, iters=iters), iters=iters)


def _binary_close(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    return _binary_erode(_binary_dilate(mask, iters=iters), iters=iters)


def _fast_percentiles_u8(
    img_u8: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0
) -> tuple[float, float]:
    """
    Keep: histogram-based percentiles for speed/stability (no semantic change vs previous cell).
    """
    hist = (
        cv2.calcHist([img_u8], [0], None, [256], [0, 256])
        .reshape(-1)
        .astype(np.float64)
    )
    cdf = np.cumsum(hist)
    total = cdf[-1]
    if total <= 0:
        return 0.0, 0.0
    lo_t = total * (p_lo / 100.0)
    hi_t = total * (p_hi / 100.0)
    lo = float(np.searchsorted(cdf, lo_t, side="left"))
    hi = float(np.searchsorted(cdf, hi_t, side="left"))
    return lo, hi


def _largest_cc(mask_u8: np.ndarray, min_area: int = 32) -> np.ndarray:
    """
    Change (score-improving, minimal): keep only the largest connected component in a binary mask.
    This reduces scattered false positives, improving both Dice and Hausdorff stability.
    """
    m = (mask_u8 > 0).astype(np.uint8, copy=False)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    if int(stats[k, cv2.CC_STAT_AREA]) < int(min_area):
        return np.zeros_like(m, dtype=np.uint8)
    return (lab == k).astype(np.uint8)


def _body_mask_from_image(img_u8: np.ndarray) -> np.ndarray:
    """
    Change (score-improving, but still same heuristic family): make body mask more reliable by
    keeping the largest component after thresholding+closing, preventing background leaks.
    """
    body = (img_u8 > 5).astype(np.uint8)
    body = _binary_close(body, iters=2)
    body = _largest_cc(
        body, min_area=256
    )  # suppress multiple regions / background islands
    body = _binary_open(body, iters=1)

    body[:2, :] = 0
    body[-2:, :] = 0
    body[:, :2] = 0
    body[:, -2:] = 0
    return body


def heuristic_foreground_from_image(img_arr: np.ndarray) -> np.ndarray:
    """
    Core logic preserved: percentile normalization + threshold + close/open + border clear.
    Minimal improvement: compute Otsu threshold on normalized image within body region, and blend
    it conservatively with the existing fixed threshold to avoid major behavior changes.
    """
    img_u8 = img_arr.astype(np.uint8, copy=False)

    lo, hi = _fast_percentiles_u8(img_u8, 2.0, 98.0)
    if hi <= lo + 1e-6:
        return np.zeros_like(img_u8, dtype=np.uint8)

    x = img_u8.astype(np.float32, copy=False)
    x = (x - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)

    body = _body_mask_from_image(img_u8)

    x_u8 = (x * 255.0 + 0.5).astype(np.uint8)
    vals = x_u8[body > 0]
    if vals.size >= 64:
        _, t = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t01 = float(t) / 255.0
        thr = float(np.clip(0.85 * 0.55 + 0.15 * t01, 0.45, 0.70))
    else:
        thr = 0.55

    mask = (x > thr).astype(np.uint8)

    mask = _binary_close(mask, iters=1)
    mask = _binary_open(mask, iters=1)

    mask = (mask & body).astype(np.uint8)

    mask[:2, :] = 0
    mask[-2:, :] = 0
    mask[:, :2] = 0
    mask[:, -2:] = 0

    return mask


_fg_cache = {}  # image_id -> np.ndarray uint8 mask


def _get_foreground_for_id(image_id: str) -> np.ndarray:
    fg = _fg_cache.get(image_id)
    if fg is not None:
        return fg
    p = id_to_image_path(image_id)
    arr = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        fg = np.zeros((1, 1), dtype=np.uint8)
    else:
        fg = heuristic_foreground_from_image(arr)
    _fg_cache[image_id] = fg
    return fg


def _min_area_for_union_mask(h: int, w: int) -> int:
    area = int(h) * int(w)
    return max(64, int(area * 0.0006))


def predict_union_rle_for_id(image_id: str) -> str:
    fg = _get_foreground_for_id(image_id)
    if fg is None or fg.sum() == 0:
        return ""
    h, w = fg.shape[:2]
    if int(fg.sum()) < _min_area_for_union_mask(h, w):
        return ""
    return rle_encode(fg)


print("Example test id:", base.iloc[0]["id"], "class:", base.iloc[0]["class"])



## === cell 4
pred = base.copy()
unique_ids = pred["id"].unique()

for j, image_id in enumerate(unique_ids):
    if j % 2000 == 0:
        print(f"Loading+FG {j}/{len(unique_ids)}")
    try:
        _get_foreground_for_id(image_id)
    except Exception:
        _fg_cache[image_id] = np.zeros((1, 1), dtype=np.uint8)

rows = []
for j, image_id in enumerate(unique_ids):
    if j % 2000 == 0:
        print(f"Union RLE {j}/{len(unique_ids)}")
    try:
        r = predict_union_rle_for_id(image_id)
        rows.extend(
            [
                (image_id, "stomach", r),
                (image_id, "large_bowel", r),
                (image_id, "small_bowel", r),
            ]
        )
    except Exception:
        rows.extend(
            [
                (image_id, "stomach", ""),
                (image_id, "large_bowel", ""),
                (image_id, "small_bowel", ""),
            ]
        )

rle_df = pd.DataFrame(rows, columns=["id", "class", "predicted"])
pred = df_ssub[["id", "class"]].merge(rle_df, on=["id", "class"], how="left")
pred["predicted"] = pred["predicted"].fillna("")

print(pred.head())
print("non-empty predicted rows:", (pred["predicted"] != "").sum())



## === cell 5
out_path = "submission.csv"
pred[["id", "class", "predicted"]].to_csv(out_path, index=False)

sub_check = pd.read_csv(out_path)
assert (
    sub_check.shape[0] == df_ssub.shape[0]
), "Submission row count mismatch vs sample_submission"
assert sub_check.columns.tolist() == [
    "id",
    "class",
    "predicted",
], "Submission columns mismatch"
print("Wrote", out_path, "with shape", sub_check.shape)

with open(out_path, "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))
