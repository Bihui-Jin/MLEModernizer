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

0.4505897321783137

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00459) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation early, which resolves the `MessageFactory.GetPrototype` error in many Kaggle images. Then I remove the missing external model dependency (`../input/uwmgi-unet-keras/model.h5`) by replacing it with a tiny, deterministic fallback Keras model that preserves the same I/O semantics (128×128×3 → 128×128×3 sigmoid) so the pipeline runs end-to-end. Finally, I fix the submission-length mismatch by generating predictions for the full `sample_submission` rows (not every 3rd row) and by writing RLEs aligned exactly to the sample ordering, producing a valid `submission.csv`.'
- What this solution (achieved 0.00378) has done: 'I fix the TensorFlow/protobuf crash by importing TensorFlow only after forcing the pure-Python protobuf implementation and by defensively retrying the import with a clean protobuf state. Then, to increase score toward your target (your current 0.00459 is far below 0.4506), I minimally change the fallback behavior so the model is trained on the provided `train.csv` masks (same 128×128×3 → 128×128×3 sigmoid semantics, same loss) and then used for test inference, instead of using random untrained weights. Finally, I keep submission generation aligned exactly to `sample_submission.csv` and ensure RLE encoding is correct and stable.'
- What this solution (achieved 0.0) has done: 'I remove the brittle TensorFlow import workaround that is currently still crashing with a protobuf `GetPrototype` error, and instead run the same pipeline using a deterministic, lightweight non-TF fallback that always produces a valid submission. I also fix the OpenCV resize crashes by ensuring every row has a valid (width,height) from the image filename; when missing, we infer it from the actual PNG. Finally, I fix the submission-length mismatch by generating exactly one RLE per `sample_submission` row (in the same order) and writing `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your current pipeline always predicts empty masks (`PROBS` is all zeros), which yields a near-zero Dice and therefore a near-zero combined score; to move toward your 0.4506 target we need non-trivial, valid segmentations while keeping the overall approach (2D masks → RLE submission) intact. The smallest legitimate change is to replace the “always empty” predictor with a lightweight heuristic segmenter that uses the provided MRI slice intensities (no external models, no extra packages) to produce plausible organ masks. I keep the same per-row loop, the same resize→threshold→RLE logic, and the same submission alignment to `sample_submission.csv`, only changing how `PROBS` is populated. This should materially increase the score from 0.0 while staying deterministic and fast (<600s).'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with the submission effectively predicting mostly-empty or badly-calibrated masks; the smallest safe way to move toward your 0.4506 target (higher-is-better) without changing the overall “heuristic mask → per-class channel → resize → threshold → RLE” pipeline is to make the heuristic produce more plausible non-empty organ regions and tune the binarization threshold slightly. I keep the same core approach and data flow, but (1) build the mask from a combination of Otsu foreground and edge-derived regions (better at capturing bowel boundaries), (2) keep multiple large connected components instead of only the single largest (organs can be disjoint across slices), and (3) lower the final threshold from 0.5 to 0.35 to avoid collapsing to empty masks. This should materially increase Dice (and not explode Hausdorff) versus the current near-empty outputs, while staying deterministic and fast and still writing a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with predicting nearly-empty masks and/or masks that are too noisy so Dice stays near zero and Hausdorff is heavily penalized; to move toward your 0.4506 target (higher-is-better) with minimal change, we only adjust the heuristic mask generation and the binarization calibration. Specifically, we keep the same pipeline (read slice → build per-row PROBS → resize → threshold → RLE → align to sample_submission), but (1) replace the edge-heavy mask with a more organ-like “mid-intensity body region” mask derived from Otsu + distance-transform fill (reduces spurious edges and improves Hausdorff), and (2) make the final threshold slightly more conservative and class-specific to reduce false positives. These are small, deterministic changes that typically increase Dice without exploding Hausdorff, while still finishing fast and writing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is consistent with generating masks that are either too often empty or too noisy/edge-like, which tanks Dice and especially hurts the 3D Hausdorff component; to move toward your 0.4506 target (higher-is-better) without changing the overall “heuristic → per-class channel → resize → threshold → RLE → submission” pipeline, we minimally improve the heuristic to produce more compact organ-like blobs. Concretely, we keep the same flow but (1) replace the Otsu-on-blur foreground with a more stable “mid-intensity band-pass” region (often aligns with soft tissue) plus morphological cleanup, and (2) add a light convex-hull fill step per connected component to reduce holes/fragmentation (typically improves Hausdorff). Finally, we make thresholds slightly less conservative (class-specific) so masks are not systematically empty, while still avoiding runaway false positives. The script still runs end-to-end, stays deterministic, and writes a valid `submission.csv` aligned exactly to `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with the metric rejecting many rows due to invalid RLEs (unsorted/duplicated runs) and/or masks being too often empty after resizing+thresholding. I make two minimal, score-relevant changes: (1) replace the RLE encoder with a known-correct “Kaggle-standard” implementation that guarantees sorted, non-overlapping runs in Fortran order, and (2) slightly stabilize the heuristic mask so it doesn’t collapse to empty (light adaptive thresholding + gentle dilation after component selection) while keeping your overall heuristic→per-class-channel→resize→threshold→RLE pipeline intact. Everything else (data reading, PROBS shape/semantics, per-row loop, class channels, thresholds, and submission alignment to `sample_submission.csv`) stays the same. This should move your score upward toward the 0.4506 target without changing the core approach.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with the evaluator rejecting many rows due to invalid/degenerate masks (too often empty after resize/threshold) and/or overly noisy masks that explode the Hausdorff term. I keep your exact pipeline (read slice → heuristic probability map → class-channel scaling → resize → threshold → RLE → merge with sample_submission), but make two minimal score-directed tweaks: (1) make the heuristic produce slightly more compact, organ-like blobs by adding a body-mask constraint and a small hole-fill, and (2) calibrate thresholds to reduce all-empty predictions without making masks overly large (which would hurt Hausdorff). These are small, deterministic changes that should move the score up toward your 0.4506 target while staying within the 600s runtime and still producing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score (higher-is-better; target 0.4506) is consistent with the metric treating many predictions as effectively empty and/or having very poor 3D consistency, so we make the smallest change that increases overlap while keeping masks compact to not blow up Hausdorff. I keep your exact pipeline (slice → heuristic PROBS → resize → threshold → RLE → merge with sample_submission), but add a per-(case,day,slice) “shared base mask” cache so all three classes for the same slice are spatially consistent (important for 3D reconstruction and Hausdorff), and I add a very small class-specific area guard that prevents systematically empty outputs by gently lowering the threshold only when the predicted area is near-zero. I not change the RLE semantics, the output columns, or the submission alignment; the script still run end-to-end and write `submission.csv`. These changes are deterministic, fast, and directly aimed at moving the score upward from 0.0 toward your target without rewriting the core approach.'

# 9. Code solution

## === cell 0
import os, random, gc, warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PYTHONHASHSEED", "42")
random.seed(42)

import numpy as np

np.random.seed(42)



## === cell 1
import pandas as pd
import cv2
from tqdm import tqdm

try:
    cv2.setNumThreads(0)
except Exception:
    pass



## === cell 2
BATCH_SIZE = 16
EPOCHS = 30
n_splits = 5
fold_selected = 1  # 1..5



## === cell 3
DATA_DIR = "../input/uw-madison-gi-tract-image-segmentation"
df = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
DEBUG = False  # keep semantics; no debug sampling



## === cell 4
df.rename(columns={"class": "class_name"}, inplace=True)

parts = df["id"].str.split("_", expand=True)
df["case"] = parts[0].str.replace("case", "", regex=False).astype(np.int32)
df["day"] = parts[1].str.replace("day", "", regex=False).astype(np.int32)
df["slice"] = parts[3]  # like '0000.png'

TEST_DIR = f"{DATA_DIR}/test"


def _build_global_scan_index(root_dir: str):
    idx = {}  # (case_int, day_int) -> (scans_dir, {slice_idx: (filename, w, h)})
    if not os.path.isdir(root_dir):
        return idx
    for case_name in os.listdir(root_dir):
        if not case_name.startswith("case"):
            continue
        case_path = os.path.join(root_dir, case_name)
        if not os.path.isdir(case_path):
            continue
        try:
            case_int = int(case_name.replace("case", ""))
        except Exception:
            continue

        for day_name in os.listdir(case_path):
            if not day_name.startswith(f"{case_name}_day"):
                continue
            try:
                day_int = int(day_name.split("_day", 1)[1])
            except Exception:
                continue

            scans_dir = os.path.join(case_path, day_name, "scans")
            if not os.path.isdir(scans_dir):
                continue

            m = {}
            for fn in os.listdir(scans_dir):
                if (not fn.startswith("slice_")) or (not fn.endswith(".png")):
                    continue
                stem = fn[:-4]
                p = stem.split("_")
                if len(p) < 6:
                    continue
                slice_idx = p[1]  # '0000'
                try:
                    w = int(p[2])
                    h = int(p[3])
                except Exception:
                    continue
                m[slice_idx] = (fn, w, h)
            idx[(case_int, day_int)] = (scans_dir, m)
    return idx


_GLOBAL_SCAN_INDEX = _build_global_scan_index(TEST_DIR)


def _get_path_w_h(case_int: int, day_int: int, slice_png: str):
    slice_idx = slice_png[:-4]  # '0000'
    key = (int(case_int), int(day_int))
    v = _GLOBAL_SCAN_INDEX.get(key)
    if v is None:
        return "", 0, 0
    scans_dir, m = v
    hit = m.get(slice_idx)
    if hit is None:
        return "", 0, 0
    fn, w, h = hit
    return os.path.join(scans_dir, fn), w, h


case_arr = df["case"].to_numpy(np.int32, copy=False)
day_arr = df["day"].to_numpy(np.int32, copy=False)
slice_arr = df["slice"].to_numpy(copy=False)

paths = np.empty(len(df), dtype=object)
widths = np.empty(len(df), dtype=np.int32)
heights = np.empty(len(df), dtype=np.int32)

for i in range(len(df)):
    p, w, h = _get_path_w_h(case_arr[i], day_arr[i], slice_arr[i])
    paths[i] = p
    widths[i] = w
    heights[i] = h

df["path"] = paths
df["width"] = widths
df["height"] = heights

del parts, case_arr, day_arr, slice_arr, paths, widths, heights
df.head(3)



## === cell 5
df_test = df[
    ["id", "class_name", "path", "case", "day", "slice", "width", "height"]
].copy()
del df
df_test.reset_index(drop=True, inplace=True)
df_test.fillna("", inplace=True)

print("df_test:", df_test.shape)
df_test.head(3)



## === cell 6
gc.collect()




## === cell 7
def rle_encode(img):
    """
    Kaggle-standard RLE encoder in Fortran order to ensure sorted, non-overlapping runs.
    Invalid RLEs can lead to 0.0 score, so we keep this strict and deterministic.
    img: (H,W) binary {0,1} or {False,True}
    """
    if img.ndim != 2:
        img = img.squeeze()
    img = (img > 0).astype(np.uint8, copy=False)

    pixels = img.reshape(-1, order="F")
    pixels = np.concatenate([[0], pixels, [0]]).astype(np.uint8, copy=False)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if runs.size == 0:
        return ""
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 8
def _infer_hw_from_png(path: str):
    if not path or (not os.path.exists(path)):
        return 0, 0
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        return 0, 0
    h, w = img.shape[:2]
    return int(h), int(w)


missing = (df_test["width"].to_numpy() <= 0) | (df_test["height"].to_numpy() <= 0)
if missing.any():
    idxs = np.where(missing)[0]
    new_w = df_test["width"].to_numpy(np.int32, copy=True)
    new_h = df_test["height"].to_numpy(np.int32, copy=True)
    for i in idxs:
        path = df_test.at[i, "path"]
        hh, ww = _infer_hw_from_png(path)
        new_w[i] = ww
        new_h[i] = hh
    df_test["width"] = new_w
    df_test["height"] = new_h

print("Missing width after fix:", int((df_test["width"] <= 0).sum()))
print("Missing height after fix:", int((df_test["height"] <= 0).sum()))




## === cell 9
def _normalize_to_uint8(img16):
    if img16 is None:
        return None
    img = img16.astype(np.float32, copy=False)
    lo, hi = np.percentile(img, (1.0, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi <= lo + 1e-6):
        lo, hi = float(img.min()), float(img.max() + 1e-6)
    img = (img - lo) / (hi - lo + 1e-6)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def _keep_topk_components(bin_mask_u8: np.ndarray, k: int = 2, min_area: int = 120):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_mask_u8, connectivity=8
    )
    if num <= 1:
        return bin_mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas)  # descending
    keep = []
    for j in order[: max(1, k)]:
        if int(areas[j]) >= int(min_area):
            keep.append(1 + int(j))
    if not keep:
        best = 1 + int(order[0])
        return (labels == best).astype(np.uint8)
    out = np.isin(labels, keep).astype(np.uint8)
    return out


def _fill_components_convex(bin_u8: np.ndarray, max_components: int = 2):
    """
    Convex-hull fill reduces holes/fragmentation (often improves Hausdorff),
    applied only to the largest components to avoid runaway false positives.
    """
    if bin_u8 is None or bin_u8.sum() == 0:
        return bin_u8
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_u8, connectivity=8)
    if num <= 1:
        return bin_u8

    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas)

    out = np.zeros_like(bin_u8)
    kept = 0
    for j in order:
        lab = 1 + int(j)
        if kept >= int(max_components):
            break
        mask = (labels == lab).astype(np.uint8) * 255
        if int(mask.sum()) == 0:
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 100:
            continue
        hull = cv2.convexHull(c)
        cv2.fillPoly(out, [hull], 1)
        kept += 1
    if out.sum() == 0:
        return bin_u8
    return out.astype(np.uint8)


def _fill_small_holes(bin_u8: np.ndarray):
    """
    Light hole-fill improves compactness and reduces boundary fragmentation,
    which typically helps the Hausdorff component without greatly enlarging masks.
    """
    if bin_u8 is None or bin_u8.sum() == 0:
        return bin_u8
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(bin_u8.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=1)


def _heuristic_mask_from_path(path: str):
    """
    Returns a (128,128) float32 probability map in [0,1].

    Uses a coarse "body mask" constraint and light hole-fill to keep masks compact,
    improving Hausdorff stability while avoiding all-empty predictions.
    """
    if (not path) or (not os.path.exists(path)):
        return np.zeros((128, 128), dtype=np.float32)

    img16 = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if img16 is None:
        return np.zeros((128, 128), dtype=np.float32)

    img8 = _normalize_to_uint8(img16)
    img8s = cv2.resize(img8, (128, 128), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(img8s, (5, 5), 0)

    _, body = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    body = (body > 0).astype(np.uint8)
    body = cv2.morphologyEx(
        body, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), 1
    )
    body = _keep_topk_components(body, k=1, min_area=800)

    p20, p80 = np.percentile(blur, (20.0, 80.0))
    lo = int(max(0, p20 - 5))
    hi = int(min(255, p80 + 10))
    band = ((blur >= lo) & (blur <= hi)).astype(np.uint8)

    if int(band.sum()) < 80:
        band = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -2
        )
        band = (band > 0).astype(np.uint8)

    h, w = band.shape
    center = np.zeros_like(band, dtype=np.uint8)
    y0, y1 = int(0.06 * h), int(0.98 * h)
    x0, x1 = int(0.06 * w), int(0.94 * w)
    center[y0:y1, x0:x1] = 1

    hard0 = (band & body & center).astype(np.uint8)

    hard0 = cv2.morphologyEx(
        hard0, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1
    )

    if hard0.sum() == 0:
        return np.zeros((128, 128), dtype=np.float32)

    hard = _keep_topk_components(hard0, k=2, min_area=160)
    hard = _fill_components_convex(hard, max_components=2)
    hard = _fill_small_holes(hard)

    hard = cv2.dilate(
        hard.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ).astype(np.uint8)

    if hard.sum() == 0:
        return np.zeros((128, 128), dtype=np.float32)

    dist = cv2.distanceTransform(hard.astype(np.uint8), cv2.DIST_L2, 3)
    if dist.max() > 1e-6:
        dist = dist / (dist.max() + 1e-6)

    prob = (0.10 + 0.90 * dist).astype(np.float32)
    prob *= hard.astype(np.float32)
    return np.clip(prob, 0.0, 1.0)


n = len(df_test)
PROBS = np.zeros((n, 128, 128, 3), dtype=np.float32)

paths = df_test["path"].to_numpy(dtype=object, copy=False)
class_names = df_test["class_name"].to_numpy(dtype=object, copy=False)

case_arr = df_test["case"].to_numpy(np.int32, copy=False)
day_arr = df_test["day"].to_numpy(np.int32, copy=False)
slice_arr = df_test["slice"].to_numpy(dtype=object, copy=False)

_base_cache = {}  # (case,day,slice_png) -> (128,128) float32

for i in tqdm(range(n), total=n):
    key = (int(case_arr[i]), int(day_arr[i]), str(slice_arr[i]))
    base = _base_cache.get(key)
    if base is None:
        base = _heuristic_mask_from_path(paths[i])  # (128,128) in [0,1]
        _base_cache[key] = base

    cls = class_names[i]
    if cls == "large_bowel":
        scales = (1.00, 0.60, 0.55)
    elif cls == "small_bowel":
        scales = (0.60, 1.00, 0.55)
    else:  # stomach
        scales = (0.55, 0.55, 1.00)

    PROBS[i, :, :, 0] = np.clip(base * scales[0], 0.0, 1.0)
    PROBS[i, :, :, 1] = np.clip(base * scales[1], 0.0, 1.0)
    PROBS[i, :, :, 2] = np.clip(base * scales[2], 0.0, 1.0)

del _base_cache
gc.collect()

print(
    "PROBS:",
    PROBS.shape,
    PROBS.dtype,
    "min/max:",
    float(PROBS.min()),
    float(PROBS.max()),
)



## === cell 10
class_to_ch = {"large_bowel": 0, "small_bowel": 1, "stomach": 2}

heights = df_test["height"].to_numpy(np.int32, copy=False)
widths = df_test["width"].to_numpy(np.int32, copy=False)
classes = df_test["class_name"].to_numpy(copy=False)

BIN_THRESH_BY_CLASS = {"large_bowel": 0.37, "small_bowel": 0.37, "stomach": 0.38}

MIN_AREA_FRAC_BY_CLASS = {
    "large_bowel": 0.0009,
    "small_bowel": 0.0010,
    "stomach": 0.0007,
}
THR_FALLBACK_DELTA = 0.06  # gentle, only when needed

rles = []
rles_append = rles.append

for i in tqdm(range(df_test.shape[0]), total=df_test.shape[0]):
    h = int(heights[i])
    w = int(widths[i])
    if h <= 0 or w <= 0:
        rles_append("")
        continue

    cls = classes[i]
    ch = class_to_ch.get(cls, 0)
    pred = PROBS[i, :, :, ch]

    pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    thr0 = float(BIN_THRESH_BY_CLASS.get(cls, 0.37))
    pred_bin = (pred_resized >= thr0).astype(np.uint8, copy=False)

    min_area = int(max(1, round(MIN_AREA_FRAC_BY_CLASS.get(cls, 0.001) * (h * w))))
    if int(pred_bin.sum()) < min_area:
        thr1 = max(0.05, thr0 - THR_FALLBACK_DELTA)
        pred_bin = (pred_resized >= thr1).astype(np.uint8, copy=False)

    rles_append(rle_encode(pred_bin))

gc.collect()
print("RLEs:", len(rles))



## === cell 11
sub = pd.DataFrame(
    {
        "id": df_test["id"].values,
        "class": df_test["class_name"].values,
        "predicted": np.asarray(rles, dtype=object),
    }
)

sample = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
sub = sample[["id", "class"]].merge(sub, on=["id", "class"], how="left")
sub["predicted"] = sub["predicted"].fillna("")

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("Null predicted:", int(sub["predicted"].isna().sum()))
print("Empty predicted:", int((sub["predicted"] == "").sum()))



## === cell 12
sub.head()
