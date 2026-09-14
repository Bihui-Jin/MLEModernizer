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
    img: numpy array (H,W) with 1 - mask, 0 - background
    Competition expects flattening in column-major order (Fortran order).
    """
    if img.ndim != 2:
        img = img.squeeze()
    pixels = img.reshape(-1, order="F")
    if pixels.dtype != np.uint8:
        pixels = pixels.astype(np.uint8, copy=False)

    padded = np.empty(pixels.size + 2, dtype=np.uint8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = pixels

    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    if changes.size == 0:
        return ""
    changes[1::2] -= changes[::2]
    return " ".join(map(str, changes.tolist()))




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


def _heuristic_mask_from_path(path: str):
    """
    Returns a (128,128) float32 probability map in [0,1].
    Heuristic: find "tissue" region by adaptive thresholding on normalized intensity,
    restrict to central abdomen region, and clean with morphology.
    """
    if (not path) or (not os.path.exists(path)):
        return np.zeros((128, 128), dtype=np.float32)

    img16 = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if img16 is None:
        return np.zeros((128, 128), dtype=np.float32)

    img8 = _normalize_to_uint8(img16)
    img8s = cv2.resize(img8, (128, 128), interpolation=cv2.INTER_AREA)

    blur = cv2.GaussianBlur(img8s, (5, 5), 0)

    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = th.shape
    center = np.zeros_like(th)
    y0, y1 = int(0.12 * h), int(0.92 * h)
    x0, x1 = int(0.12 * w), int(0.88 * w)
    center[y0:y1, x0:x1] = 255
    th = cv2.bitwise_and(th, center)

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k1, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k2, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (th > 0).astype(np.uint8), connectivity=8
    )
    if num <= 1:
        mask = (th > 0).astype(np.uint8)
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        mask = (labels == best).astype(np.uint8)

    prob = (mask * 0.85).astype(np.float32)
    return prob


n = len(df_test)
PROBS = np.zeros((n, 128, 128, 3), dtype=np.float32)

paths = df_test["path"].to_numpy(dtype=object, copy=False)
class_names = df_test["class_name"].to_numpy(dtype=object, copy=False)

for i in tqdm(range(n), total=n):
    base = _heuristic_mask_from_path(paths[i])  # (128,128) in [0,0.85]
    cls = class_names[i]
    if cls == "large_bowel":
        scales = (1.00, 0.70, 0.55)
    elif cls == "small_bowel":
        scales = (0.70, 1.00, 0.60)
    else:  # stomach
        scales = (0.55, 0.60, 1.00)

    PROBS[i, :, :, 0] = np.clip(base * scales[0], 0.0, 1.0)
    PROBS[i, :, :, 1] = np.clip(base * scales[1], 0.0, 1.0)
    PROBS[i, :, :, 2] = np.clip(base * scales[2], 0.0, 1.0)

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

rles = []
rles_append = rles.append

for i in tqdm(range(df_test.shape[0]), total=df_test.shape[0]):
    h = int(heights[i])
    w = int(widths[i])
    if h <= 0 or w <= 0:
        rles_append("")
        continue

    ch = class_to_ch.get(classes[i], 0)
    pred = PROBS[i, :, :, ch]

    pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    pred_bin = (pred_resized >= 0.5).astype(np.uint8, copy=False)
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
