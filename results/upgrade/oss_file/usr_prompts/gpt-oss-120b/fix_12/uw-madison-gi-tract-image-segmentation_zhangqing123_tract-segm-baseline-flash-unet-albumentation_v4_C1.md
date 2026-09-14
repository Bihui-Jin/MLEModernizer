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
import pandas as pd
from collections import Counter




## === cell 1
DATASET_FOLDER = "/kaggle/input/uw-madison-gi-tract-image-segmentation"

train_csv = os.path.join(DATASET_FOLDER, "train.csv")
test_csv = os.path.join(DATASET_FOLDER, "test.csv")
sample_sub_csv = os.path.join(DATASET_FOLDER, "sample_submission.csv")

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)
df_sample_sub = pd.read_csv(sample_sub_csv)

print("Train rows:", len(df_train))
print("Test rows :", len(df_test))
print("Sample submission rows:", len(df_sample_sub))




## === cell 2
df_pred = df_sample_sub[["id", "class"]].copy()


def rle_to_positions(rle):
    """Convert an RLE string to a list of 1‑based pixel indices."""
    if not isinstance(rle, str) or rle == "":
        return []
    nums = list(map(int, rle.split()))
    positions = []
    for start, length in zip(nums[0::2], nums[1::2]):
        positions.extend(range(start, start + length))
    return positions


def positions_to_rle(pos_list):
    """Encode a sorted list of 1‑based pixel indices to an RLE string."""
    if not pos_list:
        return ""
    runs = []
    prev = pos_list[0]
    start = prev
    length = 1
    for p in pos_list[1:]:
        if p == prev + 1:
            length += 1
        else:
            runs.append(str(start))
            runs.append(str(length))
            start = p
            length = 1
        prev = p
    runs.append(str(start))
    runs.append(str(length))
    return " ".join(runs)


def rle_length(rle):
    """Return total number of pixels encoded in an RLE string."""
    if not isinstance(rle, str) or rle == "":
        return 0
    nums = list(map(int, rle.split()))
    return sum(nums[i] for i in range(1, len(nums), 2))


def rle_to_set(rle):
    """Return a frozenset of pixel indices for fast Dice computation."""
    return frozenset(rle_to_positions(rle))


def dice_between_sets(a, b):
    """Dice coefficient for two sets of pixel indices."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return 2 * inter / (len(a) + len(b))


id_to_seg = dict(zip(df_train["id"], df_train["segmentation"]))

unique_segs = df_train["segmentation"].dropna().unique()
seg_to_set = {
    seg: rle_to_set(seg) for seg in unique_segs if isinstance(seg, str) and seg
}

class_prototype = {}
TOP_K = None  # use all frequent masks for a richer prototype search

for cls, group in df_train.groupby("class"):
    seg_counter = Counter(
        [seg for seg in group["segmentation"] if isinstance(seg, str) and seg]
    )
    if not seg_counter:
        class_prototype[cls] = ""
        continue

    candidates = [seg for seg, _ in seg_counter.most_common(TOP_K)]

    best_seg = ""
    best_score = -1.0

    for cand in candidates:
        if not cand:
            continue
        cand_set = seg_to_set.get(cand, frozenset())
        total_score = 0.0
        total_weight = 0
        for seg, cnt in seg_counter.items():
            seg_set = seg_to_set.get(seg, frozenset())
            dice_val = dice_between_sets(cand_set, seg_set)
            total_score += dice_val * cnt
            total_weight += cnt
        avg_score = total_score / total_weight if total_weight else 0.0
        if avg_score > best_score:
            best_score = avg_score
            best_seg = cand

    if not best_seg and candidates:
        best_seg = candidates[0]

    class_prototype[cls] = best_seg

global_counter = Counter(
    [seg for seg in df_train["segmentation"] if isinstance(seg, str) and seg]
)
global_most_common = ""
if global_counter:
    global_most_common = global_counter.most_common(1)[0][0]

for cls, proto in class_prototype.items():
    if not proto:  # empty prototype -> replace with global most common non‑empty mask
        class_prototype[cls] = global_most_common


def get_prediction(row):
    if row["id"] in id_to_seg:
        return id_to_seg[row["id"]]
    return class_prototype.get(row["class"], "")


df_pred["predicted"] = df_pred.apply(get_prediction, axis=1)

print("Prepared prediction DataFrame:", df_pred.shape)




## === cell 3
submission_path = "submission.csv"
df_pred.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")




## === cell 4
df_pred.head()
