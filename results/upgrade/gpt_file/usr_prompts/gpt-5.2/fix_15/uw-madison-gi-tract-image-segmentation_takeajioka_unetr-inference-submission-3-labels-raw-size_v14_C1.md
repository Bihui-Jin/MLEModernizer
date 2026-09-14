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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nibabel==5.3.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

0.8484889039601035

# 6. Current score

0.35067

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Your notebook fails because it depends on MONAI (not installed in this Kaggle environment), which cascades into undefined transforms/model/inference objects and prevents `pred_df` and the final `submission.csv` from being created. I keep your overall flow (read sample_submission → build per-case/day volumes → generate per-slice per-class RLE) but replace the MONAI-specific inference with a minimal, deterministic fallback that produces valid binary masks (empty masks) so the pipeline runs end-to-end and writes a correctly formatted `submission.csv`. This is score-poor but it satisfies the “valid submission” requirement; once you provide/enable the intended model dependency or checkpoint environment, we can restore real inference with minimal changes. I also fix the cell numbering (start at 1) and ensure the correct CSV is read (`test.csv`/`sample_submission.csv`) rather than mistakenly treating sample submission as training data.'
- What this solution (achieved 0.35175) has done: 'I fix the shape-mismatch bug in `predict_heuristic_masks_for_volume` that comes from inconsistent per-slice PNG sizes within a volume (e.g., 265×265 vs 266×266), by padding/cropping each slice to the volume’s target `(H,W)` before inserting into `segm`. This unblocks cell 9 so `pred_df` is created, which then fixes the downstream `NameError` and `KeyError` in cells 10–11 and ensures a valid `submission.csv` is written. The heuristic logic itself is kept the same (blur → percentile threshold → ROI cleanup), only made robust to differing slice dimensions. I also add a small safety clamp for slice indices to avoid out-of-range indexing if any IDs are irregular.'
- What this solution (achieved 0.18168) has done: 'The timeout is dominated by per-slice Python work: repeated percentile computations, repeated largest-connected-component flood fills, and per-row RLE encoding loops. To keep the exact same heuristic logic and thresholds, the main speedups are (1) caching per-slice results so each slice is processed once and reused across the 3 class-rows for that slice, (2) replacing the pure-Python connected-component flood fill with OpenCV’s equivalent (same 4-connectivity semantics) when available, and (3) using a faster RLE encoder that avoids `tolist()/map(str, ...)` overhead. These changes preserve identical evaluation semantics (same masks per slice given the same blurred/thresholded arrays) while dramatically reducing Python overhead and bringing runtime under the 600s limit.'
- What this solution (achieved 0.18172) has done: 'Your current heuristic is producing masks that are likely too large/noisy, which heavily hurts the Hausdorff component (60% weight) and drags the overall score down. To move the score upward toward your target with minimal logic change, I keep the exact same per-slice pipeline (blur → percentile threshold → ROI cleanup → largest CC) but add one small, deterministic post-processing step: a light 3×3 morphological opening on each predicted class mask to remove thin spurs and isolated pixels that inflate Hausdorff distance. I also add a tiny “minimum area gate” per class (set very low) to drop extremely small components that are almost always false positives; this tends to improve Hausdorff without materially changing the core approach. These changes are fast, keep your architecture/training unchanged (still heuristic), and should improve the metric rather than just runtime.'
- What this solution (achieved 0.30089) has done: 'I keep your exact heuristic pipeline (blur → percentile thresholds → ROI cleanup → largest CC → morph open) but make two minimal, metric-relevant post-processing adjustments aimed specifically at reducing the Hausdorff penalty that’s dominating your score. First, I add a very light, deterministic hole-filling step (closing) after opening to avoid “ring” shapes and small gaps that create far boundary errors. Second, I add a per-class cap on how much of the slice can be predicted (max area ratio), which suppresses occasional huge false-positive masks that massively hurt Hausdorff, without changing your per-slice thresholds or core inference flow. The submission writing and ID alignment remain unchanged and it still run end-to-end under the time limit.'
- What this solution (achieved 0.35067) has done: 'Your current heuristic tends to produce occasional large/noisy masks that get punished heavily by the Hausdorff component; to move the score up toward the target with minimal risk, I keep the exact same pipeline but (1) slightly tighten the per-class maximum-area caps and (2) add a very small, deterministic “erode-by-1px” after close to shrink boundaries and reduce far-out boundary outliers. These are purely post-processing tweaks (no architecture/training changes) and are aimed at improving Hausdorff without rewriting the core inference logic. I also keep all I/O and submission alignment unchanged so it still runs end-to-end and writes a valid `submission.csv`. The changes are small enough to stay within the “minimal changes” constraint while plausibly improving the metric from 0.30089 upward.'

# 9. Code solution

## === cell 0
import os, glob, gc
import numpy as np
import pandas as pd
from PIL import Image
import random

random.seed(0)
np.random.seed(0)



## === cell 1
DATASET_FOLDER = "../input/uw-madison-gi-tract-image-segmentation"

sample_path = os.path.join(DATASET_FOLDER, "sample_submission.csv")
test_csv_path = os.path.join(DATASET_FOLDER, "test.csv")
train_csv_path = os.path.join(DATASET_FOLDER, "train.csv")

sub_df = pd.read_csv(sample_path)
test_df = pd.read_csv(test_csv_path)

print("sample_submission:", sub_df.shape, list(sub_df.columns))
print("test.csv:", test_df.shape, list(test_df.columns))

df_work = test_df.copy()




## === cell 2
def extract_details(id_):
    id_fields = id_.split("_")
    case = id_fields[0].replace("case", "")
    day = id_fields[1].replace("day", "")
    slice_id = id_fields[3]
    return {"Case": int(case), "Day": int(day), "Slice": slice_id}


parts = df_work["id"].str.split("_", expand=True)
df_work["Case"] = parts[0].str.replace("case", "", regex=False).astype(np.int32)
df_work["Day"] = parts[1].str.replace("day", "", regex=False).astype(np.int32)
df_work["Slice"] = parts[3]
print(df_work.head())



## === cell 3
df_overview = (
    df_work.groupby(["Case", "Day"], sort=True)
    .size()
    .rename("Slices")
    .reset_index()
    .sort_values(["Case", "Day"])
    .reset_index(drop=True)
)
print("df_overview:", df_overview.shape)
print(df_overview.head())



## === cell 4
_PNG_CACHE = {}
_FIRST_SLICE_INFO = {}


def _list_pngs_cached(img_dir):
    p = _PNG_CACHE.get(img_dir)
    if p is None:
        p = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        _PNG_CACHE[img_dir] = p
    return p


def _get_first_slice_hw(img_dir):
    info = _FIRST_SLICE_INFO.get(img_dir)
    if info is not None:
        return info
    imgs = _list_pngs_cached(img_dir)
    if len(imgs) == 0:
        raise FileNotFoundError(f"No PNGs found under: {img_dir}")
    with Image.open(imgs[0]) as im0:
        im0 = im0.convert("L")
        h, w = np.asarray(im0).shape
    _FIRST_SLICE_INFO[img_dir] = (h, w)
    return h, w


def load_image_volume(img_dir, quant=0.01):
    """
    Loads a ZxHxW uint8 volume from grayscale PNGs.
    Bugfix: percentile bounds are float; clip must occur in float dtype (not out=uint8).
    Robustness: handle occasional per-slice H/W mismatch by pad/crop to first slice.
    """
    imgs = _list_pngs_cached(img_dir)
    if len(imgs) == 0:
        raise FileNotFoundError(f"No PNGs found under: {img_dir}")

    h, w = _get_first_slice_hw(img_dir)
    z = len(imgs)

    vol = np.empty((z, h, w), dtype=np.float32)

    for i, fp in enumerate(imgs):
        with Image.open(fp) as im:
            a = np.asarray(im.convert("L"))
        if a.shape != (h, w):
            hh, ww = a.shape
            a2 = np.zeros((h, w), dtype=a.dtype)
            h0 = min(h, hh)
            w0 = min(w, ww)
            a2[:h0, :w0] = a[:h0, :w0]
            a = a2
        vol[i] = a  # cast happens implicitly to float32

    if quant:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)

    v_min = float(vol.min())
    v_max = float(vol.max())
    denom = (v_max - v_min) if (v_max - v_min) != 0 else 1.0
    vol = (vol - v_min) / denom
    vol = (vol * 255.0).astype(np.uint8, copy=False)

    return vol




## === cell 5
def rle_encode(img):
    """
    img: 2D numpy array of {0,1}
    Returns run length as string formatted: start length start length ...
    """
    if img.ndim != 2:
        raise ValueError(f"rle_encode expects 2D array, got shape {img.shape}")

    if not np.any(img):
        return ""

    pixels = img.reshape(-1, order="F").astype(np.uint8, copy=False)
    padded = np.empty(pixels.size + 2, dtype=np.uint8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = pixels

    runs = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    runs[1::2] -= runs[::2]

    return " ".join(runs.astype(str))




## === cell 6
CLASSES = ["large_bowel", "small_bowel", "stomach"]
class_to_idx = {c: i for i, c in enumerate(CLASSES)}

try:
    import cv2  # noqa: F401

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def _box_blur2d(img2d_u8, k=9):
    k = int(k)
    if k < 1:
        return img2d_u8.astype(np.float32)
    if k % 2 == 0:
        k += 1
    r = k // 2

    a = img2d_u8.astype(np.float32, copy=False)
    pad = np.pad(a, ((0, 0), (r, r)), mode="edge")
    c = np.cumsum(pad, axis=1)
    h = (c[:, k:] - c[:, :-k]) / k
    pad = np.pad(h, ((r, r), (0, 0)), mode="edge")
    c = np.cumsum(pad, axis=0)
    v = (c[k:, :] - c[:-k, :]) / k
    return v


def _pad_crop_to_hw(img2d, target_h, target_w):
    h, w = img2d.shape
    out = np.zeros((target_h, target_w), dtype=img2d.dtype)
    h0 = min(h, target_h)
    w0 = min(w, target_w)
    out[:h0, :w0] = img2d[:h0, :w0]
    return out


def _largest_cc(mask_u8):
    m = mask_u8.astype(np.uint8, copy=False)
    if not np.any(m):
        return np.zeros_like(m, dtype=np.uint8)

    if _HAS_CV2:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        if num <= 1:
            return m.astype(np.uint8, copy=False)
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        return (labels == best).astype(np.uint8)

    m = m != 0
    h, w = m.shape
    visited = np.zeros((h, w), dtype=np.uint8)

    best_count = 0
    best_seed = None

    ys, xs = np.nonzero(m)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        stack_y = [int(y0)]
        stack_x = [int(x0)]
        visited[y0, x0] = 1
        count = 0
        while stack_y:
            y = stack_y.pop()
            x = stack_x.pop()
            count += 1

            yn = y - 1
            if yn >= 0 and m[yn, x] and not visited[yn, x]:
                visited[yn, x] = 1
                stack_y.append(yn)
                stack_x.append(x)
            yn = y + 1
            if yn < h and m[yn, x] and not visited[yn, x]:
                visited[yn, x] = 1
                stack_y.append(yn)
                stack_x.append(x)
            xn = x - 1
            if xn >= 0 and m[y, xn] and not visited[y, xn]:
                visited[y, xn] = 1
                stack_y.append(y)
                stack_x.append(xn)
            xn = x + 1
            if xn < w and m[y, xn] and not visited[y, xn]:
                visited[y, xn] = 1
                stack_y.append(y)
                stack_x.append(xn)

        if count > best_count:
            best_count = count
            best_seed = (int(y0), int(x0))

    out = np.zeros((h, w), dtype=np.uint8)
    if best_seed is None:
        return out

    y0, x0 = best_seed
    stack_y = [y0]
    stack_x = [x0]
    out[y0, x0] = 1
    while stack_y:
        y = stack_y.pop()
        x = stack_x.pop()

        yn = y - 1
        if yn >= 0 and m[yn, x] and not out[yn, x]:
            out[yn, x] = 1
            stack_y.append(yn)
            stack_x.append(x)
        yn = y + 1
        if yn < h and m[yn, x] and not out[yn, x]:
            out[yn, x] = 1
            stack_y.append(yn)
            stack_x.append(x)
        xn = x - 1
        if xn >= 0 and m[y, xn] and not out[y, xn]:
            out[y, xn] = 1
            stack_y.append(y)
            stack_x.append(xn)
        xn = x + 1
        if xn < w and m[y, xn] and not out[y, xn]:
            out[y, xn] = 1
            stack_y.append(y)
            stack_x.append(xn)

    return out


if _HAS_CV2:
    _MORPH_KERNEL_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
else:
    _MORPH_KERNEL_3 = None


def _morph_open_u8(mask_u8):
    if not np.any(mask_u8):
        return mask_u8.astype(np.uint8, copy=False)
    if _HAS_CV2:
        return cv2.morphologyEx(
            mask_u8.astype(np.uint8, copy=False), cv2.MORPH_OPEN, _MORPH_KERNEL_3
        )
    return mask_u8.astype(np.uint8, copy=False)


def _morph_close_u8(mask_u8):
    if not np.any(mask_u8):
        return mask_u8.astype(np.uint8, copy=False)
    if _HAS_CV2:
        return cv2.morphologyEx(
            mask_u8.astype(np.uint8, copy=False), cv2.MORPH_CLOSE, _MORPH_KERNEL_3
        )
    return mask_u8.astype(np.uint8, copy=False)


def _erode1_u8(mask_u8):
    if not np.any(mask_u8):
        return mask_u8.astype(np.uint8, copy=False)
    if _HAS_CV2:
        return cv2.erode(
            mask_u8.astype(np.uint8, copy=False), _MORPH_KERNEL_3, iterations=1
        )
    return mask_u8.astype(np.uint8, copy=False)


_MIN_AREA_BY_CLASS = {
    "large_bowel": 25,
    "small_bowel": 25,
    "stomach": 25,
}

_MAX_AREA_RATIO_BY_CLASS = {
    "large_bowel": 0.18,  # was 0.22
    "small_bowel": 0.15,  # was 0.18
    "stomach": 0.17,  # was 0.20
}


def predict_heuristic_masks_for_volume(vol_u8):
    z, h, w = vol_u8.shape
    segm = np.zeros((len(CLASSES), z, h, w), dtype=np.uint8)

    by0, by1 = int(h * 0.05), int(h * 0.95)
    bx0, bx1 = int(w * 0.05), int(w * 0.95)

    st_y0, st_y1 = int(h * 0.10), int(h * 0.60)
    st_x0, st_x1 = int(w * 0.10), int(w * 0.70)

    sb_y0, sb_y1 = int(h * 0.35), int(h * 0.90)
    sb_x0, sb_x1 = int(w * 0.15), int(w * 0.85)

    lb_y0, lb_y1 = int(h * 0.20), int(h * 0.95)
    lb_x0, lb_x1 = int(w * 0.05), int(w * 0.95)

    cx0, cx1 = int(w * 0.25), int(w * 0.75)
    cy0, cy1 = int(h * 0.35), int(h * 0.85)

    max_area_lb = int(_MAX_AREA_RATIO_BY_CLASS["large_bowel"] * h * w)
    max_area_sb = int(_MAX_AREA_RATIO_BY_CLASS["small_bowel"] * h * w)
    max_area_st = int(_MAX_AREA_RATIO_BY_CLASS["stomach"] * h * w)

    for zi in range(z):
        sl = vol_u8[zi]
        sl_blur = _box_blur2d(sl, k=9)

        roi_body = sl_blur[by0:by1, bx0:bx1]
        thr_body = float(np.percentile(roi_body, 40.0))
        body = (sl_blur >= thr_body).astype(np.uint8)
        body[:by0, :] = 0
        body[by1:, :] = 0
        body[:, :bx0] = 0
        body[:, bx1:] = 0
        body = _largest_cc(body)

        if body.sum() == 0:
            cy, cx = h // 2, w // 2
            ry, rx = max(2, h // 30), max(2, w // 30)
            body[cy - ry : cy + ry, cx - rx : cx + rx] = 1

        roi_st = sl_blur[st_y0:st_y1, st_x0:st_x1]
        thr_st = float(np.percentile(roi_st, 78.0))
        m_st = ((sl_blur >= thr_st).astype(np.uint8)) & body
        m_st[:st_y0, :] = 0
        m_st[st_y1:, :] = 0
        m_st[:, :st_x0] = 0
        m_st[:, st_x1:] = 0
        m_st = _largest_cc(m_st.astype(np.uint8))

        roi_sb = sl_blur[sb_y0:sb_y1, sb_x0:sb_x1]
        thr_sb = float(np.percentile(roi_sb, 76.0))
        m_sb = ((sl_blur >= thr_sb).astype(np.uint8)) & body
        m_sb[:sb_y0, :] = 0
        m_sb[sb_y1:, :] = 0
        m_sb[:, :sb_x0] = 0
        m_sb[:, sb_x1:] = 0
        m_sb = _largest_cc(m_sb.astype(np.uint8))

        roi_lb = sl_blur[lb_y0:lb_y1, lb_x0:lb_x1]
        thr_lb = float(np.percentile(roi_lb, 74.0))
        m_lb = ((sl_blur >= thr_lb).astype(np.uint8)) & body
        m_lb[:lb_y0, :] = 0
        m_lb[lb_y1:, :] = 0
        m_lb[:, :lb_x0] = 0
        m_lb[:, lb_x1:] = 0
        m_lb[cy0:cy1, cx0:cx1] = 0
        m_lb = _largest_cc(m_lb.astype(np.uint8))

        m_lb = _morph_close_u8(_morph_open_u8(m_lb.astype(np.uint8, copy=False)))
        m_sb = _morph_close_u8(_morph_open_u8(m_sb.astype(np.uint8, copy=False)))
        m_st = _morph_close_u8(_morph_open_u8(m_st.astype(np.uint8, copy=False)))

        m_lb = _erode1_u8(m_lb)
        m_sb = _erode1_u8(m_sb)
        m_st = _erode1_u8(m_st)

        if int(m_lb.sum()) < _MIN_AREA_BY_CLASS["large_bowel"]:
            m_lb[:] = 0
        if int(m_sb.sum()) < _MIN_AREA_BY_CLASS["small_bowel"]:
            m_sb[:] = 0
        if int(m_st.sum()) < _MIN_AREA_BY_CLASS["stomach"]:
            m_st[:] = 0

        if int(m_lb.sum()) > max_area_lb:
            m_lb[:] = 0
        if int(m_sb.sum()) > max_area_sb:
            m_sb[:] = 0
        if int(m_st.sum()) > max_area_st:
            m_st[:] = 0

        m_lb = _pad_crop_to_hw(m_lb.astype(np.uint8, copy=False), h, w)
        m_sb = _pad_crop_to_hw(m_sb.astype(np.uint8, copy=False), h, w)
        m_st = _pad_crop_to_hw(m_st.astype(np.uint8, copy=False), h, w)

        segm[class_to_idx["large_bowel"], zi, :, :] = m_lb
        segm[class_to_idx["small_bowel"], zi, :, :] = m_sb
        segm[class_to_idx["stomach"], zi, :, :] = m_st

    return segm




## === cell 7
def segm_rle(segm, df_vol):
    """
    segm: (C, Z, H, W)
    df_vol: rows for a single (Case, Day), includes columns id, class, Slice
    Returns dataframe with id,class,predicted for those rows
    """
    if df_vol.shape[0] == 0:
        return pd.DataFrame(columns=["id", "class", "predicted"])

    z = segm.shape[1]
    ids = df_vol["id"].to_numpy()
    classes = df_vol["class"].to_numpy()
    slices = df_vol["Slice"].astype(np.int32).to_numpy()

    idx = slices - 1
    idx[idx < 0] = 0
    idx[idx >= z] = z - 1

    cls_idx = np.empty(classes.shape[0], dtype=np.int8)
    cls_idx[classes == "large_bowel"] = class_to_idx["large_bowel"]
    cls_idx[classes == "small_bowel"] = class_to_idx["small_bowel"]
    cls_idx[classes == "stomach"] = class_to_idx["stomach"]

    predicted = np.empty(df_vol.shape[0], dtype=object)
    cache = {}
    for i in range(df_vol.shape[0]):
        ci = int(cls_idx[i])
        zi = int(idx[i])
        key = (ci, zi)
        s = cache.get(key)
        if s is None:
            s = rle_encode(segm[ci, zi, :, :])
            cache[key] = s
        predicted[i] = s

    return pd.DataFrame({"id": ids, "class": classes, "predicted": predicted})




## === cell 8
image_folders = []
vol_shapes = []

for row in df_overview.itertuples(index=False):
    CASE = int(row.Case)
    DAY = int(row.Day)
    image_folder = os.path.join(
        DATASET_FOLDER, "test", f"case{CASE}", f"case{CASE}_day{DAY}", "scans"
    )
    image_folders.append(image_folder)

    imgs = _list_pngs_cached(image_folder)
    if len(imgs) == 0:
        raise FileNotFoundError(f"No PNGs found under: {image_folder}")

    h, w = _get_first_slice_hw(image_folder)
    z = len(imgs)
    vol_shapes.append(str((z, h, w)))
    print("Indexed", (CASE, DAY), "volume shape:", (z, h, w))

df_overview["image_folder"] = image_folders
df_overview["vol_shape"] = vol_shapes
df_overview.head()



## === cell 9
pred_parts = []

groups = df_work.groupby(["Case", "Day"], sort=False).indices

for row in df_overview.itertuples(index=False):
    CASE = int(row.Case)
    DAY = int(row.Day)
    image_folder = row.image_folder

    vol = load_image_volume(image_folder)
    segm = predict_heuristic_masks_for_volume(vol)

    idxs = groups.get((CASE, DAY))
    if idxs is None:
        df_cd = df_work.iloc[0:0].copy()
    else:
        df_cd = df_work.iloc[idxs]

    pred_parts.append(segm_rle(segm, df_cd))

    del vol, segm, df_cd, idxs
    gc.collect()

pred_df = (
    pd.concat(pred_parts, axis=0, ignore_index=True)
    if len(pred_parts)
    else pd.DataFrame(columns=["id", "class", "predicted"])
)
print("pred_df:", pred_df.shape)
print(pred_df.head())



## === cell 10
sub_df2 = sub_df.drop(columns=["predicted"], errors="ignore")
sub_df2 = sub_df2.merge(pred_df, on=["id", "class"], how="left")

sub_df2["predicted"] = sub_df2["predicted"].fillna("")
sub_df2 = sub_df2[["id", "class", "predicted"]]

assert sub_df2.shape[0] == sub_df.shape[0], "Row count mismatch vs sample_submission"
assert list(sub_df2.columns) == [
    "id",
    "class",
    "predicted",
], "Submission columns incorrect"

sub_df2.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df2.shape)
print(sub_df2.head())



## === cell 11
print("Unique predicted strings (sample):", sub_df2["predicted"].unique()[:5])
