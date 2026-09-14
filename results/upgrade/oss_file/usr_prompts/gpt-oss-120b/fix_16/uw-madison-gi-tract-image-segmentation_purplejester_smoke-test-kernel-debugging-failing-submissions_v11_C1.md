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

0.00507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The fix ensures a submission CSV is always created by generating predictions for every test ID without relying on missing metadata. The loop now directly creates an empty RLE string for each required class, guaranteeing the DataFrame has the expected columns and rows, which resolves the KeyError and produces a valid `submission.csv` file.'
- What this solution (achieved 0.0) has done: 'I add a lightweight, deterministic mask generation step that thresholds each test image at its mean intensity and encodes the resulting binary mask with RLE. This yields non‑empty predictions for every class, moving the Dice‑based score from 0 toward the target while preserving the existing pipeline and model‑free approach.'
- What this solution (achieved 0.0) has done: 'I make the mask generation deterministic but slightly more robust: instead of leaving rows empty when metadata lookup fails, I match any file whose sample_id starts with the test id, guaranteeing a mask for every entry. I also use a modestly higher threshold (mean + 0.5 × std) rather than the raw mean, which typically yields a more balanced binary mask and improves Dice without altering the overall pipeline. These small changes keep the core logic intact while moving the score toward the target.'
- What this solution (achieved 0.0) has done: 'I make the prediction loop more robust by iterating over every test ID from `test.csv` and explicitly finding the corresponding image metadata (falling back to a substring match when the exact key is missing). This ensures that each required row gets a real mask instead of an empty string, which should improve the Dice‑based portion of the score while keeping the core thresholding logic unchanged.'
- What this solution (achieved 0.0) has done: 'I adjust the prediction loop to respect the original test CSV rows, generating a mask only for the specific organ listed in each row rather than always outputting all three organs for every case. This prevents unnecessary false‑positive masks, which should improve both Dice and Hausdorff components and move the combined score toward the target.'
- What this solution (achieved 0.00507) has done: 'I make the metadata lookup robust by normalising the “slice” token so that test IDs correctly match the stored image paths. This change ensures most rows receive a real mask instead of an empty prediction, moving the Dice‑based score upward toward the target while keeping the rest of the pipeline untouched.'

# 9. Code solution

## === cell 0
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from fastai.vision.all import get_image_files
from PIL import Image

logging.captureWarnings(True)


def rle_encode(mask: np.ndarray) -> str:
    """Run‑length encode a 2‑D binary mask in column‑major order."""
    pixels = mask.T.flatten()
    padded = np.concatenate([[0], pixels, [0]])
    runs = np.where(padded[1:] != padded[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(rle: str, h: int, w: int) -> np.ndarray:
    """Decode a RLE string back to a binary mask (column‑major)."""
    if not rle:
        return np.zeros((h, w), dtype=np.uint8)
    s = list(map(int, rle.split()))
    starts, lengths = s[0::2], s[1::2]
    starts = np.array(starts) - 1
    ends = starts + lengths
    flat = np.zeros(h * w, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        flat[lo:hi] = 1
    return flat.reshape((w, h)).T  # transpose back to original orientation


def dice_score(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Dice coefficient for two binary masks."""
    intersection = np.sum(mask1 & mask2)
    size_sum = np.sum(mask1) + np.sum(mask2)
    if size_sum == 0:
        return 1.0
    return 2.0 * intersection / size_sum




## === cell 1
@dataclass
class Metadata:
    sample_id: str
    full_path: str
    h: int
    w: int

    @classmethod
    def extract(cls, path: Path) -> "Metadata":
        case_and_day = path.parents[1].stem
        sample_id = f"{case_and_day}_slice_{path.stem}"
        with Image.open(path) as img:
            w, h = img.size
        return Metadata(sample_id, str(path), h, w)




## === cell 2
import random

DEFAULT_OFFSETS = {
    "large_bowel": 0.10,
    "small_bowel": 0.00,
    "stomach": -0.10,
}

try:
    train_files = get_image_files(
        Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/") / "train"
    )
    train_meta = {m.sample_id: m for m in map(Metadata.extract, train_files)}

    all_ids = list(train_meta.keys())
    random.seed(42)
    sample_ids = random.sample(all_ids, min(5000, len(all_ids)))

    train_df = pd.read_csv(
        Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/") / "train.csv"
    )
    train_masks = {}
    needed = set(sample_ids)
    for _, row in train_df.iterrows():
        sid, organ, rle = row["id"], row["class"], row["segmentation"]
        if sid in needed:
            meta = train_meta[sid]
            mask = rle_decode(rle, meta.h, meta.w)
            train_masks[(sid, organ)] = mask

    img_cache = {}
    for sid in sample_ids:
        meta = train_meta[sid]
        with Image.open(meta.full_path).convert("L") as img:
            arr = np.array(img)
        median = np.median(arr)
        std = arr.std()
        if std == 0:
            std = 1e-6
        img_cache[sid] = (arr, median, std)

    offset_grid = np.arange(-0.2, 0.21, 0.02)  # -0.2 to 0.2 step 0.02
    best_offsets = {}

    for organ in ("large_bowel", "small_bowel", "stomach"):
        best_score = -1.0
        best_offset = DEFAULT_OFFSETS[organ]

        for offset in offset_grid:
            scores = []
            for sid in sample_ids:
                arr, median, std = img_cache[sid]
                thresh = median + offset * std
                pred_mask = (arr > thresh).astype(np.uint8)

                true_mask = train_masks.get((sid, organ))
                if true_mask is None:
                    continue
                scores.append(dice_score(pred_mask, true_mask))

            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_offset = float(offset)

        fine_grid = np.arange(best_offset - 0.02, best_offset + 0.021, 0.005)
        for offset in fine_grid:
            scores = []
            for sid in sample_ids:
                arr, median, std = img_cache[sid]
                thresh = median + offset * std
                pred_mask = (arr > thresh).astype(np.uint8)

                true_mask = train_masks.get((sid, organ))
                if true_mask is None:
                    continue
                scores.append(dice_score(pred_mask, true_mask))

            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_offset = float(offset)

        best_offsets[organ] = best_offset

    OFFSET_FACTORS = best_offsets
except Exception as e:
    print(f"Calibration failed ({e}); using default offsets.")
    OFFSET_FACTORS = DEFAULT_OFFSETS




## === cell 3
from enum import IntEnum


class TestMethod(IntEnum):
    CENTER = 5  # only method we keep for a deterministic baseline


SELECTED_METHOD = TestMethod.CENTER




## === cell 4
preds = []

DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/")
test_df = pd.read_csv(DATA_DIR / "test.csv")  # contains id and class columns
TEST_FILES = get_image_files(DATA_DIR / "test")
METADATA = {m.sample_id: m for m in map(Metadata.extract, TEST_FILES)}

METADATA_CLEAN = {key.replace("_slice_", "_"): val for key, val in METADATA.items()}


def get_meta_for_id(test_id: str):
    """Return metadata matching the full id or a partial match (ignoring the slice token)."""
    if test_id in METADATA:
        return METADATA[test_id]
    if test_id in METADATA_CLEAN:
        return METADATA_CLEAN[test_id]
    for clean_key, meta in METADATA_CLEAN.items():
        if test_id in clean_key:
            return meta
    for key, meta in METADATA.items():
        if test_id in key:
            return meta
    return None


for _, row in test_df.iterrows():
    test_id = row["id"]
    organ = row["class"]
    meta = get_meta_for_id(test_id)

    if meta is None:
        preds.append({"id": test_id, "class": organ, "predicted": ""})
        continue

    with Image.open(meta.full_path).convert("L") as img:
        img_arr = np.array(img)

    median = np.median(img_arr)
    std = img_arr.std()
    if std == 0:
        std = 1e-6

    offset = OFFSET_FACTORS.get(organ, 0.0)
    thresh = median + offset * std
    mask = (img_arr > thresh).astype(np.uint8)
    rle_string = rle_encode(mask)
    preds.append({"id": test_id, "class": organ, "predicted": rle_string})




## === cell 5
submission_path = "submission.csv"
pd.DataFrame(preds)[["id", "class", "predicted"]].to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, rows:", len(preds))
