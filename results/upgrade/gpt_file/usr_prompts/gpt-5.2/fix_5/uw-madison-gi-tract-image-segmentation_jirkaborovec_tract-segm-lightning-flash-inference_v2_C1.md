# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import re
import glob
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

print("Labels:", LABELS)
print("train:", df_train.shape)



## === cell 1
import sys

print("Python:", sys.version)



## === cell 2
df_test = pd.read_csv(test_csv_path)
df_ssub = pd.read_csv(sample_sub_path)

print("test:", df_test.shape, "columns:", df_test.columns.tolist())
print("sample_submission:", df_ssub.shape, "columns:", df_ssub.columns.tolist())

if set(["id", "class"]).issubset(df_test.columns):
    base = df_test[["id", "class"]].copy()
else:
    base = df_ssub[["id", "class"]].copy()

print("base:", base.shape)




## === cell 3
def rle_encode(mask: np.ndarray) -> str:
    """
    Kaggle RLE (1-indexed), pixels read top-to-bottom then left-to-right.
    mask: 2D array of {0,1} with shape (H,W).
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={mask.shape}")
    pixels = mask.T.flatten()  # top-to-bottom then left-to-right
    pixels = np.concatenate([[0], pixels.astype(np.uint8), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


m = np.zeros((3, 4), dtype=np.uint8)
m[0, 0] = 1
m[1, 0] = 1
assert rle_encode(m) == "1 2"



## === cell 4
from PIL import Image

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


def _binary_dilate(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    m = mask.astype(bool)
    for _ in range(iters):
        neigh = m.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neigh |= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
        m = neigh
    return m.astype(np.uint8)


def _binary_erode(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    m = mask.astype(bool)
    for _ in range(iters):
        neigh = m.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neigh &= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
        m = neigh
    return m.astype(np.uint8)


def _binary_open(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    return _binary_dilate(_binary_erode(mask, iters=iters), iters=iters)


def _binary_close(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    return _binary_erode(_binary_dilate(mask, iters=iters), iters=iters)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Simple 4-neighbor connected-components to keep the largest blob.
    """
    H, W = mask.shape
    m = mask.astype(np.uint8)
    visited = np.zeros((H, W), dtype=np.uint8)

    best_count = 0
    best_label = 0
    labels = np.zeros((H, W), dtype=np.int32)
    current_label = 0

    stack_y = np.empty(H * W, dtype=np.int32)
    stack_x = np.empty(H * W, dtype=np.int32)

    for y0 in range(H):
        row_m = m[y0]
        row_v = visited[y0]
        for x0 in range(W):
            if row_m[x0] == 0 or row_v[x0]:
                continue
            current_label += 1
            top = 0
            stack_y[top] = y0
            stack_x[top] = x0
            visited[y0, x0] = 1
            labels[y0, x0] = current_label
            count = 0

            while top >= 0:
                y = int(stack_y[top])
                x = int(stack_x[top])
                top -= 1
                count += 1

                ny = y - 1
                if ny >= 0 and m[ny, x] and not visited[ny, x]:
                    visited[ny, x] = 1
                    labels[ny, x] = current_label
                    top += 1
                    stack_y[top] = ny
                    stack_x[top] = x
                ny = y + 1
                if ny < H and m[ny, x] and not visited[ny, x]:
                    visited[ny, x] = 1
                    labels[ny, x] = current_label
                    top += 1
                    stack_y[top] = ny
                    stack_x[top] = x
                nx = x - 1
                if nx >= 0 and m[y, nx] and not visited[y, nx]:
                    visited[y, nx] = 1
                    labels[y, nx] = current_label
                    top += 1
                    stack_y[top] = y
                    stack_x[top] = nx
                nx = x + 1
                if nx < W and m[y, nx] and not visited[y, nx]:
                    visited[y, nx] = 1
                    labels[y, nx] = current_label
                    top += 1
                    stack_y[top] = y
                    stack_x[top] = nx

            if count > best_count:
                best_count = count
                best_label = current_label

    out = (
        (labels == best_label).astype(np.uint8)
        if best_label != 0
        else np.zeros_like(m, dtype=np.uint8)
    )
    return out


def heuristic_foreground_from_image(img_arr: np.ndarray) -> np.ndarray:
    """
    Base foreground heuristic from a slice image.

    Kept as the same core logic (percentile normalization + threshold + close).
    """
    x = img_arr.astype(np.float32)

    lo = np.percentile(x, 2.0)
    hi = np.percentile(x, 98.0)
    if hi <= lo + 1e-6:
        return np.zeros_like(img_arr, dtype=np.uint8)
    x = (x - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)

    mask = (x > 0.55).astype(np.uint8)

    mask = _binary_close(mask, iters=1)
    mask = _binary_open(mask, iters=1)

    mask[:2, :] = 0
    mask[-2:, :] = 0
    mask[:, :2] = 0
    mask[:, -2:] = 0

    return mask


def class_specific_mask(fg: np.ndarray, cls: str) -> np.ndarray:
    """
    Build different masks per class from the same base fg mask.
    """
    if fg.sum() == 0:
        return fg

    if cls == "stomach":
        m = _keep_largest_component(fg)
        m = _binary_dilate(m, iters=1)
        m = _binary_close(m, iters=1)
        return m.astype(np.uint8)

    if cls == "large_bowel":
        m = _keep_largest_component(fg)
        m = _binary_erode(m, iters=1)
        m = _binary_close(m, iters=1)
        return m.astype(np.uint8)

    if cls == "small_bowel":
        largest = _keep_largest_component(fg)
        m = (fg.astype(np.uint8) & (1 - largest.astype(np.uint8))).astype(np.uint8)
        m = _binary_close(m, iters=1)
        if m.sum() == 0:
            m = fg
        return m.astype(np.uint8)

    return fg.astype(np.uint8)


_fg_cache = {}  # image_id -> np.ndarray uint8 mask


def _get_foreground_for_id(image_id: str) -> np.ndarray:
    fg = _fg_cache.get(image_id)
    if fg is not None:
        return fg
    p = id_to_image_path(image_id)
    img = Image.open(p).convert("L")
    arr = np.array(img, dtype=np.uint8)
    fg = heuristic_foreground_from_image(arr)
    _fg_cache[image_id] = fg
    return fg


def predict_rle_for_id_and_class(image_id: str, cls: str) -> str:
    fg = _get_foreground_for_id(image_id)
    mask = class_specific_mask(fg, cls)
    if mask.sum() == 0:
        return ""
    return rle_encode(mask)


print("Example test id:", base.iloc[0]["id"], "class:", base.iloc[0]["class"])



## === cell 5
pred = base.copy()
pred["predicted"] = ""

n = len(pred)
for i, (image_id, cls) in enumerate(zip(pred["id"].values, pred["class"].values)):
    if i % 2000 == 0:
        print(f"Predicting {i}/{n}")
    try:
        pred.at[i, "predicted"] = predict_rle_for_id_and_class(image_id, cls)
    except Exception:
        pred.at[i, "predicted"] = ""

pred = df_ssub[["id", "class"]].merge(pred, on=["id", "class"], how="left")
pred["predicted"] = pred["predicted"].fillna("")

print(pred.head())
print("non-empty predicted rows:", (pred["predicted"] != "").sum())



## === cell 6
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
