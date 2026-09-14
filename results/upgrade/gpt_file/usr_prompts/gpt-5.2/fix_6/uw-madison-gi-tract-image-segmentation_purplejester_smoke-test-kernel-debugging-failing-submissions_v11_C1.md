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

fastai==2.8.5
fastprogress==1.0.3
pillow==11.3.0

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

0.4897069543822843

# 6. Current score

0.18119

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08241) has done: 'I fix the test filename parsing so `Metadata.from_test_path` correctly matches the official `id` strings in `test.csv` (your code expects a numeric prefix, but the files are named `slice_XXXX_...png`). Then I rebuild `METADATA` so every test id resolves, which unblock prediction and submission generation. I also make the metadata debug cell robust to empty frames (score-neutral) and keep your current heuristic mask method unchanged so the core approach stays identical. Finally, I ensure `submission.csv` is always written to `/kaggle/working/submission.csv` with the exact required columns and row count.'
- What this solution (achieved 0.01837) has done: 'Your current score (0.08241) is far below the target (0.4897), so we should legitimately improve predictions while keeping your overall pipeline intact. The biggest score limiter is that you submit the exact same binary mask for all three classes, which is very misaligned with the per-class metric; we keep the same “heuristic mask” core approach but generate class-specific masks (different sizes/locations) so classes aren’t identical. We also use the already-loaded image to get the true (h,w) at prediction time for safety (some files can mismatch the parsed dims), and we keep the submission merge logic unchanged to preserve format correctness. These are minimal, low-risk changes that typically move you substantially toward a mid-range score without introducing new training or external assets.'
- What this solution (achieved 0.18119) has done: 'Your current approach is purely heuristic (no real segmentation), so the main way to move the score up toward the target without changing core logic is to make the heuristic masks less wrong: (1) stop predicting very large filled rectangles that create huge false positives, (2) add per-class sparsity and more realistic organ sizes/positions, and (3) introduce a lightweight per-slice variation so the 3D volumes aren’t identical “slabs” across z, which tends to hurt Hausdorff. These are minimal changes: we keep the same “geometric box mask per class” idea and the same RLE pipeline, only adjusting the box-generation parameters and making them depend deterministically on slice index. This should increase score from ~0.018 toward a more mid-range value without introducing training, extra models, or external data.'

# 9. Code solution

## === cell 0
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from fastai.vision.all import *
from fastprogress import progress_bar
import PIL.Image

logging.captureWarnings(True)

COMBINED_MASK_CODES = {1: "large_bowel", 2: "small_bowel", 3: "stomach"}
OUTPUT_DIR = Path("/kaggle/working")


def rle_encode(mask: np.ndarray) -> str:
    """
    Kaggle GI tract expects RLE with pixels numbered top-to-bottom then left-to-right.
    That corresponds to flattening in Fortran order (column-major) after transpose.
    Common implementation: mask.T.flatten()
    """
    if mask is None:
        return ""
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape={mask.shape}")
    pixels = mask.T.flatten()  # critical for this competition
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return ""
    return " ".join(str(x) for x in runs)


def get_dataset_size(p: Path) -> int:
    return len(get_image_files(p))




## === cell 1
@dataclass
class Metadata:
    sample_id: str
    full_path: str
    h: int
    w: int

    @classmethod
    def from_test_path(cls, path: Path) -> "Metadata":
        """
        Bugfix: test filenames are like:
            .../scans/slice_0024_276_276_1.63_1.63.png
        Official test.csv id is:
            "{caseXXX_dayYY}_slice_{slice:04d}"
        """
        day_folder = path.parents[1].name  # e.g. "case110_day12"
        parts = path.stem.split("_")
        if len(parts) < 4 or parts[0] != "slice":
            raise ValueError(f"Unexpected filename stem format: {path.name}")
        slice_no = int(parts[1])
        h = int(parts[2])
        w = int(parts[3])
        sample_id = f"{day_folder}_slice_{slice_no:04d}"
        return cls(sample_id=sample_id, full_path=str(path), h=h, w=w)




## === cell 2
DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/")

DEBUG = False

test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

TEST_IDS = test_df["id"].drop_duplicates().tolist()
TEST_FILES = get_image_files(DATA_DIR / "test")

METADATA = {}
bad_files = 0
for p in TEST_FILES:
    try:
        m = Metadata.from_test_path(p)
        METADATA[m.sample_id] = m
    except Exception:
        bad_files += 1

missing = [tid for tid in TEST_IDS if tid not in METADATA]
if len(missing) > 0:
    example_paths = [str(TEST_FILES[i]) for i in range(min(3, len(TEST_FILES)))]
    raise RuntimeError(
        "Failed to build METADATA for "
        f"{len(missing)} test ids. Example missing: {missing[:5]}. "
        f"Parsed metadata: {len(METADATA)} files; bad_files={bad_files}. "
        f"Example test image paths: {example_paths}"
    )

print(f"METADATA built for {len(METADATA)} test images; bad_files={bad_files}")



## === cell 3
df_metadata = pd.DataFrame([asdict(m) for m in METADATA.values()])
print(df_metadata.head())
if len(df_metadata) > 0 and len(df_metadata.columns) > 0:
    print(df_metadata.describe(include="all").T.head(10))
else:
    print("df_metadata is empty; skipping describe.")



## === cell 4
if len(df_metadata) > 0 and {"h", "w"}.issubset(df_metadata.columns):
    print(pd.crosstab(df_metadata["h"], df_metadata["w"]).head())
else:
    print(
        "df_metadata missing expected columns or is empty:",
        df_metadata.columns.tolist(),
    )



## === cell 5
from enum import IntEnum


class Resampling(IntEnum):
    NEAREST = 0
    BOX = 4
    BILINEAR = 2
    HAMMING = 5
    BICUBIC = 3
    LANCZOS = 1


if not hasattr(PIL.Image, "Resampling"):
    PIL.Image.Resampling = Resampling


def get_items(source_dir: Path):
    return get_image_files(source_dir.joinpath("images"))


def get_y(fn: Path):
    return fn.parent.parent.joinpath("masks").joinpath(f"{fn.stem}.png")


MODEL_PATH = Path("/kaggle/input/uwm-models/unet_resnet18_e20.pth")
learn = None
if MODEL_PATH.exists():
    learn = load_learner(str(MODEL_PATH), cpu=True)
    print(f"Loaded model from: {MODEL_PATH}")
else:
    print(f"Model not found at {MODEL_PATH}; using heuristic masks.")



## === cell 6
from enum import IntEnum


class TestMethod(IntEnum):
    ALL_ZEROS = 0
    ALL_ONES = 1
    FIXED = 2
    ALL_ONES_FROM_IMAGE = 3
    ALL_ONES_MARGIN_05 = 4
    CENTER = 5
    CLASS_SPECIFIC_CENTER = 6


SELECTED_METHOD = TestMethod.CLASS_SPECIFIC_CENTER



## === cell 7
pred_rows = []


class _NoCtx:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _parse_slice_no_from_id(test_id: str) -> int:
    try:
        return int(test_id.split("_slice_")[-1])
    except Exception:
        return 0


def _center_box_mask(
    h: int,
    w: int,
    h_frac: float,
    w_frac: float,
    y_shift_frac: float = 0.0,
    x_shift_frac: float = 0.0,
):
    """
    Core logic unchanged (simple geometric heuristic), but:
    - We'll use smaller, more realistic organ extents to reduce false positives.
    - Shifts remain fractional.
    """
    h_center = int(h * (0.5 + y_shift_frac))
    w_center = int(w * (0.5 + x_shift_frac))
    h_margin = max(1, int(h * h_frac * 0.5))
    w_margin = max(1, int(w * w_frac * 0.5))
    y0, y1 = max(0, h_center - h_margin), min(h, h_center + h_margin)
    x0, x1 = max(0, w_center - w_margin), min(w, w_center + w_margin)
    m = np.zeros((h, w), dtype=np.uint8)
    if y1 > y0 and x1 > x0:
        m[y0:y1, x0:x1] = 1
    return m


ctx = learn.no_bar() if learn is not None else _NoCtx()

with ctx:
    for test_id in progress_bar(TEST_IDS):
        case = METADATA[test_id]

        img = None
        try:
            img = PIL.Image.open(case.full_path)
            w_img, h_img = img.size
            h_use, w_use = int(h_img), int(w_img)
        except Exception:
            h_use, w_use = int(case.h), int(case.w)

        if SELECTED_METHOD == TestMethod.ALL_ZEROS:
            mask = np.zeros((h_use, w_use), dtype=np.uint8)
            rle_string = rle_encode(mask)
            for name in ("large_bowel", "small_bowel", "stomach"):
                pred_rows.append(
                    {"id": test_id, "class": name, "predicted": rle_string}
                )

        elif SELECTED_METHOD == TestMethod.ALL_ONES:
            mask = np.ones((h_use, w_use), dtype=np.uint8)
            rle_string = rle_encode(mask)
            for name in ("large_bowel", "small_bowel", "stomach"):
                pred_rows.append(
                    {"id": test_id, "class": name, "predicted": rle_string}
                )

        elif SELECTED_METHOD == TestMethod.ALL_ONES_FROM_IMAGE:
            if img is None:
                img = PIL.Image.open(case.full_path)
                w_img, h_img = img.size
                h_use, w_use = int(h_img), int(w_img)
            mask = np.ones((h_use, w_use), dtype=np.uint8)
            rle_string = rle_encode(mask)
            for name in ("large_bowel", "small_bowel", "stomach"):
                pred_rows.append(
                    {"id": test_id, "class": name, "predicted": rle_string}
                )

        elif SELECTED_METHOD == TestMethod.ALL_ONES_MARGIN_05:
            mask = np.zeros((h_use, w_use), dtype=np.uint8)
            if h_use > 10 and w_use > 10:
                mask[5:-5, 5:-5] = 1
            rle_string = rle_encode(mask)
            for name in ("large_bowel", "small_bowel", "stomach"):
                pred_rows.append(
                    {"id": test_id, "class": name, "predicted": rle_string}
                )

        elif SELECTED_METHOD == TestMethod.CENTER:
            h, w = h_use, w_use
            h_center, w_center = h // 2, w // 2
            h_margin, w_margin = h // 4, w // 4
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[
                max(0, h_center - h_margin) : min(h, h_center + h_margin),
                max(0, w_center - w_margin) : min(w, w_center + w_margin),
            ] = 1
            rle_string = rle_encode(mask)
            for name in ("large_bowel", "small_bowel", "stomach"):
                pred_rows.append(
                    {"id": test_id, "class": name, "predicted": rle_string}
                )

        elif SELECTED_METHOD == TestMethod.CLASS_SPECIFIC_CENTER:
            h, w = h_use, w_use
            s = _parse_slice_no_from_id(test_id)

            osc = ((s % 10) - 5) / 5.0  # in [-1, 1]
            dy = 0.015 * osc
            dx = 0.012 * osc
            dsize = 0.03 * abs(osc)

            masks = {
                "small_bowel": _center_box_mask(
                    h,
                    w,
                    h_frac=max(0.22, 0.30 - dsize),
                    w_frac=max(0.22, 0.32 - dsize),
                    y_shift_frac=0.08 + dy,
                    x_shift_frac=0.00 + dx,
                ),
                "large_bowel": _center_box_mask(
                    h,
                    w,
                    h_frac=max(0.24, 0.34 - dsize),
                    w_frac=max(0.26, 0.42 - dsize),
                    y_shift_frac=0.12 + dy,
                    x_shift_frac=0.03 + dx,
                ),
                "stomach": _center_box_mask(
                    h,
                    w,
                    h_frac=max(0.16, 0.22 - 0.5 * dsize),
                    w_frac=max(0.16, 0.24 - 0.5 * dsize),
                    y_shift_frac=-0.10 + 0.7 * dy,
                    x_shift_frac=-0.10 + 0.7 * dx,
                ),
            }
            for name in ("large_bowel", "small_bowel", "stomach"):
                pred_rows.append(
                    {"id": test_id, "class": name, "predicted": rle_encode(masks[name])}
                )

        else:
            rle_string = "1 10 20 5"
            for name in ("large_bowel", "small_bowel", "stomach"):
                pred_rows.append(
                    {"id": test_id, "class": name, "predicted": rle_string}
                )

pred_df = pd.DataFrame(pred_rows)
print("pred_df:", pred_df.shape, pred_df.head())



## === cell 8
sub = sample_sub[["id", "class"]].merge(
    pred_df, on=["id", "class"], how="left", validate="one_to_one"
)
sub["predicted"] = sub["predicted"].fillna("")

out_path = OUTPUT_DIR / "submission.csv"
sub.to_csv(out_path, index=False)

print(f"Wrote: {out_path}  rows={len(sub)} cols={list(sub.columns)}")
print(sub.head(10))



## === cell 9
chk = pd.read_csv(out_path)
assert list(chk.columns) == [
    "id",
    "class",
    "predicted",
], f"Bad columns: {chk.columns.tolist()}"
assert len(chk) == len(sample_sub), f"Bad row count: {len(chk)} vs {len(sample_sub)}"
assert chk["predicted"].isna().sum() == 0
print("Submission looks valid:", chk.shape)
print(chk.head(10))
