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


def extract_tract_details_local(id_str: str, dataset_dir: str, split: str = "train"):
    """
    Returns (Case, Day, Slice, image, image_path, height, width).

    Change (score-relevant, minimal): instead of mapping slice index to sorted PNG list position
    (often wrong), we resolve the exact PNG by matching the true slice number in filenames:
    scans/*_*_*_*.png where the 3rd token is the slice index.
    This directly improves mask↔image alignment, which should increase Dice/Hausdorff score.
    """
    m = _ID_RE.match(id_str)
    if not m:
        raise ValueError(f"Unexpected id format: {id_str}")
    case, day, slc = m.group(1), int(m.group(2)), int(m.group(3))

    scan_dir = os.path.join(dataset_dir, split, case, f"{case}_day{day}", "scans")
    if not os.path.isdir(scan_dir):
        image_path = os.path.join(
            split, case, f"{case}_day{day}", "scans", "MISSING.png"
        )
        return case, day, slc, os.path.basename(image_path), image_path, np.nan, np.nan

    pngs = glob.glob(os.path.join(scan_dir, "*.png"))
    chosen = None
    chosen_hw = (np.nan, np.nan)

    for p in pngs:
        stem = os.path.splitext(os.path.basename(p))[0]
        parts = stem.split("_")
        if len(parts) >= 3 and parts[2].isdigit() and int(parts[2]) == slc:
            chosen = p
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                chosen_hw = (int(parts[0]), int(parts[1]))
            break

    if chosen is None:
        pngs_sorted = sorted(pngs)
        if slc < 0 or slc >= len(pngs_sorted):
            image_path = os.path.join(
                split, case, f"{case}_day{day}", "scans", "OOR.png"
            )
            return (
                case,
                day,
                slc,
                os.path.basename(image_path),
                image_path,
                np.nan,
                np.nan,
            )
        chosen = pngs_sorted[slc]
        stem = os.path.splitext(os.path.basename(chosen))[0]
        parts = stem.split("_")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            chosen_hw = (int(parts[0]), int(parts[1]))

    image = os.path.basename(chosen)
    rel_path = os.path.relpath(chosen, dataset_dir)
    h, w = chosen_hw
    return case, day, slc, image, rel_path, h, w


tmp = df_train["id"].apply(
    lambda x: pd.Series(extract_tract_details_local(x, DATASET_FOLDER, split="train"))
)
tmp.columns = ["Case", "Day", "Slice", "image", "image_path", "height", "width"]
df_train = pd.concat([df_train, tmp], axis=1)
print(df_train.head())




## === cell 2
def rle_encode(mask: np.ndarray) -> str:
    """
    mask: 2D numpy array of 0/1 values.
    Kaggle RLE: flatten in Fortran order (column-major) which matches the competition's expected indexing.
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
df_sub = df_ssub.copy()

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


def load_grayscale_uint8(png_path: str) -> np.ndarray:
    img = Image.open(png_path).convert("L")
    return np.asarray(img, dtype=np.uint8)


def keep_largest_connected_component(mask_u8: np.ndarray) -> np.ndarray:
    """
    Change (score-relevant, still minimal heuristic): keep only the largest 4-connected component.
    This tends to reduce scattered false positives from intensity thresholding, improving Dice/HD.
    Implemented without external packages to fit the environment constraint.
    """
    h, w = mask_u8.shape
    visited = np.zeros((h, w), dtype=np.uint8)

    best_size = 0
    best_coords = None

    if mask_u8.sum() == 0:
        return mask_u8

    for i in range(h):
        for j in range(w):
            if mask_u8[i, j] == 0 or visited[i, j] == 1:
                continue
            q = [(i, j)]
            visited[i, j] = 1
            coords = [(i, j)]
            while q:
                x, y = q.pop()
                nx = x - 1
                if nx >= 0 and mask_u8[nx, y] == 1 and visited[nx, y] == 0:
                    visited[nx, y] = 1
                    q.append((nx, y))
                    coords.append((nx, y))
                nx = x + 1
                if nx < h and mask_u8[nx, y] == 1 and visited[nx, y] == 0:
                    visited[nx, y] = 1
                    q.append((nx, y))
                    coords.append((nx, y))
                ny = y - 1
                if ny >= 0 and mask_u8[x, ny] == 1 and visited[x, ny] == 0:
                    visited[x, ny] = 1
                    q.append((x, ny))
                    coords.append((x, ny))
                ny = y + 1
                if ny < w and mask_u8[x, ny] == 1 and visited[x, ny] == 0:
                    visited[x, ny] = 1
                    q.append((x, ny))
                    coords.append((x, ny))

            if len(coords) > best_size:
                best_size = len(coords)
                best_coords = coords

    out = np.zeros_like(mask_u8, dtype=np.uint8)
    if best_coords is not None:
        for x, y in best_coords:
            out[x, y] = 1
    return out


def simple_foreground_mask(img_u8: np.ndarray) -> np.ndarray:
    """
    Minimal heuristic segmentation (still same core approach: threshold-based from image).
    Change (score-relevant): slightly more inclusive threshold + largest-component filtering
    to better match organ-shaped regions and reduce empty/scattered masks.
    """
    lo = np.percentile(img_u8, 5)
    hi = np.percentile(img_u8, 95)
    if hi <= lo:
        return np.zeros_like(img_u8, dtype=np.uint8)

    norm = (img_u8.astype(np.float32) - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)

    m = (norm > 0.50).astype(np.uint8)

    if m.shape[0] > 8 and m.shape[1] > 8:
        m[:2, :] = 0
        m[-2:, :] = 0
        m[:, :2] = 0
        m[:, -2:] = 0

    m = keep_largest_connected_component(m)
    return m


id_to_rle = {}

unique_ids = df_sub["id"].unique().tolist()
missing = 0

for id_str in unique_ids:
    case, day, slc, image, rel_path, h, w = extract_tract_details_local(
        id_str, DATASET_FOLDER, split="test"
    )
    full_path = os.path.join(DATASET_FOLDER, rel_path)
    if not os.path.exists(full_path):
        id_to_rle[id_str] = ""
        missing += 1
        continue

    img_u8 = load_grayscale_uint8(full_path)
    mask = simple_foreground_mask(img_u8)
    id_to_rle[id_str] = rle_encode(mask)

print("Built masks for ids:", len(unique_ids), "missing_pngs:", missing)
print("Example RLE (first non-empty if exists):")
for k in unique_ids[:20]:
    if id_to_rle.get(k, "") != "":
        print(k, id_to_rle[k][:80], "...")
        break



## === cell 5
df_sub["predicted"] = df_sub["id"].map(id_to_rle).fillna("")

assert list(df_sub.columns) == ["id", "class", "predicted"]
assert len(df_sub) == 20400

out_path = "submission.csv"
df_sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(pd.read_csv(out_path).head())
print("Non-empty predicted rows:", int((df_sub["predicted"] != "").sum()))
