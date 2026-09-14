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

0.3504678333702701

# 6. Current score

0.00707

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12005) has done: 'I fix the TensorFlow import crash by avoiding TensorFlow entirely (it is failing in this environment due to a protobuf incompatibility) and replace the missing external pretrained model load with a safe, deterministic fallback that still produces a valid submission. I also fix the dataframe construction so masks are generated per-image (not per-row) and so the RLE encoding matches the competition’s required column-major (Fortran) order. Finally, I ensure the script always writes a `submission.csv` with exactly the same row order/shape as `sample_submission.csv`, preventing length mismatches and Kaggle format rejections.'
- What this solution (achieved 0.03154) has done: 'I keep your fallback (non-TensorFlow) approach and only adjust the mask post-processing to better match GI-tract organ shapes, which should increase Dice/Hausdorff toward your target without changing the overall pipeline. Specifically, I (1) use a more robust per-slice normalization (percentile-based) to stabilize thresholds across scans, and (2) replace the current “open” morphology (which erodes away organs and hurts Dice) with a small “close then fill holes” style cleanup that tends to produce contiguous blobs and improves Hausdorff. I also make thresholds slightly less aggressive so the masks are less empty (empty predictions strongly penalize Dice), while keeping everything deterministic and lightweight. The submission writing/order and RLE Fortran-order encoding are preserved.'
- What this solution (achieved 0.03154) has done: 'Your current score is far below the target (0.03154 vs 0.35047; higher-is-better), so we should make small, legitimate changes that increase overlap and reduce empty masks without changing the overall heuristic pipeline. The biggest low-risk gain here is to (1) ensure empty masks encode to an empty string (not “0”), because invalid/degenerate RLE can get scored very poorly, and (2) use class-specific thresholds so at least one organ isn’t systematically wiped out by an overly strict global cutoff. I keep your same preprocessing, same fallback predictor, and same morphological “close + largest component” logic, only adjusting post-processing thresholds and making RLE encoding robust to empty masks. The submission row order/merge logic and Fortran-order encoding are preserved.'
- What this solution (achieved 0.01577) has done: 'Your current score (0.03154) is far below the target (0.35047; higher-is-better), so we should make minimal post-processing changes that reduce empty/fragmented masks without changing the overall fallback heuristic pipeline. I keep your exact data loading, RLE encoding (Fortran order), and the same heuristic “threshold from blurred intensity quantiles” core, but (1) switch resizing of probability maps to linear interpolation (less blocky boundaries) and (2) add a very small per-class cleanup (close + light open + fill holes) after thresholding to produce more contiguous organ shapes, which tends to improve Dice and Hausdorff. I also slightly relax the quantile thresholds used to form the three class maps so predictions are less often empty (empty masks heavily hurt Dice). All changes are deterministic, fast, and keep the end-to-end submission writing unchanged.'
- What this solution (achieved 0.01594) has done: 'Your score is far below the target (0.01577 vs 0.35047; higher-is-better), so we should make small, safe post-processing changes that reduce empty masks and better match organ morphology without changing the overall heuristic pipeline. I keep your same fallback predictor and RLE encoding, but (1) make the three probability maps nested (stomach ⊂ small_bowel ⊂ large_bowel) to reduce contradictory class shapes and stabilize Hausdorff, and (2) slightly relax thresholds while using smaller, less destructive morphology kernels to preserve thin bowel structures. I also add a tiny “area guard” that prevents returning empty masks when there is a plausible foreground signal (still deterministic), which typically improves Dice versus empty predictions. Submission formatting and row order remain identical to `sample_submission.csv`.'
- What this solution (achieved 0.0158) has done: 'Your current score (0.01594) is far below the target (0.35047; higher-is-better), so we should make the smallest changes that plausibly increase overlap without changing your heuristic “quantile-threshold + morphology + largest-component” core. The main low-risk issue is that forcing “largest connected component” for every class tends to delete the many disconnected small-bowel loops, heavily hurting Dice and Hausdorff, so we keep LCC for stomach/large_bowel but not for small_bowel. We also keep your nested-map design but use a slightly less aggressive quantile for the broadest mask to reduce empty/too-small predictions (empty masks are very costly), while keeping everything deterministic and fast. Submission order/format and the Fortran-order RLE encoding remain unchanged and a valid `submission.csv` is always produced.'
- What this solution (achieved 0.00707) has done: 'Your score (0.0158) is far below the target (0.3505; higher-is-better), so we should make the smallest changes that plausibly increase overlap while keeping your exact heuristic core (quantile-threshold → morphology → RLE) intact. The biggest likely issue is that `predict_masks_fallback()` currently forces the *largest connected component for all three classes*, which is especially harmful for small_bowel (often many disconnected loops) and can also be harmful for large_bowel; we only keep LCC inside `predict_masks_fallback()` for the stomach-like tightest mask and let the downstream per-class logic decide LCC. Additionally, we slightly relax the broad-mask quantile threshold (t1) to reduce empty/too-small predictions (empty masks are heavily penalized), without changing the architecture/looping. Everything else (I/O paths, resizing, per-class post-processing, RLE Fortran-order, submission row order) remains unchanged.'
- What this solution (achieved 0.00707) has done: 'Your current score (0.00707) is far below the target (0.35047), so we should make the smallest changes that legitimately increase overlap without changing the overall heuristic pipeline. The lowest-risk, high-impact fix here is to correct the `slice` parsing bug: right now it uses `x.split("_")[3]` which is out of range for ids like `case123_day20_slice_0001`, causing incorrect/failed path matching and effectively garbage predictions. With correct slice extraction (and stripping the `slice_` prefix), the image-path merge aligns properly and your masks be generated for the correct scans, which should move the score substantially toward the target. I also add a defensive assertion after the merge to fail fast if any ids still don’t match to a PNG, preventing silent submission corruption.'

# 9. Code solution

## === cell 0
import os, random, gc, warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
import cv2
from glob import glob
from tqdm import tqdm

print("Using OpenCV version:", cv2.__version__)



## === cell 2
BATCH_SIZE = 16
EPOCHS = 30
n_splits = 5
fold_selected = 1  # 1..5



## === cell 3
df = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
DEBUG = False
if df.shape[0] == 0:
    DEBUG = True

if DEBUG:
    df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
    df.pop("segmentation")
    df["predicted"] = ""



## === cell 4
df.rename(columns={"class": "class_name"}, inplace=True)
df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))

df["slice"] = df["id"].apply(lambda x: x.split("_")[3])

if DEBUG:
    TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train"
else:
    TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/test"

all_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
if len(all_images) == 0:
    raise FileNotFoundError(f"No .png images found under {TRAIN_DIR}")

root = all_images[0].rsplit("/", 4)[
    0
]  # ../input/uw-madison-gi-tract-image-segmentation/{train|test}

path_partial = []
for i in range(df.shape[0]):
    path_partial.append(
        os.path.join(
            root,
            "case" + str(df["case"].values[i]),
            "case" + str(df["case"].values[i]) + "_" + "day" + str(df["day"].values[i]),
            "scans",
            "slice_" + str(df["slice"].values[i]),
        )
    )
df["path_partial"] = path_partial

img_partials = [str(p.rsplit("_", 4)[0]) for p in all_images]
tmp_df = pd.DataFrame({"path_partial": img_partials, "path": all_images})

df = df.merge(tmp_df, on="path_partial").drop(columns=["path_partial"])

if df.shape[0] == 0:
    raise RuntimeError(
        "After merging ids to png paths, got 0 rows. Check id->path logic."
    )
if df["path"].isna().any():
    raise RuntimeError(
        "Some rows did not resolve to an image path after merge (NaNs present)."
    )

df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))

del root, path_partial, img_partials, tmp_df
df.head(5)



## === cell 5
df_img = (
    df[["id", "path", "case", "day", "slice", "width", "height"]]
    .drop_duplicates("id")
    .reset_index(drop=True)
)

del df
gc.collect()

df_img.head(5)



## === cell 6
print("Images:", df_img.shape)
if DEBUG:
    df_img = df_img.sample(frac=0.05, random_state=SEED).reset_index(drop=True)
print("Images (after DEBUG sampling if any):", df_img.shape)



## === cell 7
gc.collect()




## === cell 8
def rle_encode(img):
    """
    img: 2D numpy array, 1 - mask, 0 - background
    Returns run length as string formatted

    FIX (score-impacting correctness): If mask is empty, return "" (not "0"),
    matching competition expectations and avoiding degenerate RLE strings.
    Kaggle expects pixels numbered top-to-bottom, then left-to-right,
    which corresponds to flatten(order='F') for a (H,W) array.
    """
    if img.ndim != 2:
        raise ValueError(f"rle_encode expects 2D array, got shape={img.shape}")

    img = (img > 0).astype(np.uint8)
    if img.sum() == 0:
        return ""

    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)




## === cell 9
def load_grayscale_128(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)

    p1 = float(np.percentile(img, 1))
    p99 = float(np.percentile(img, 99))
    if p99 > p1:
        img = (img - p1) / (p99 - p1)
    else:
        mx = float(np.max(img))
        if mx > 0:
            img = img / mx

    img = np.clip(img, 0.0, 1.0)
    return img  # (128,128) float32 in [0,1]


def _largest_connected_component(bin_mask_u8):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask_u8, connectivity=8)
    if n <= 1:
        return bin_mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    out = (labels == k).astype(np.uint8)
    return out


def _fill_holes_u8(bin_mask_u8):
    if bin_mask_u8.dtype != np.uint8:
        bin_mask_u8 = bin_mask_u8.astype(np.uint8)
    if bin_mask_u8.sum() == 0:
        return bin_mask_u8

    h, w = bin_mask_u8.shape
    flood = bin_mask_u8.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, seedPoint=(0, 0), newVal=1)
    holes = (flood == 0).astype(np.uint8)
    filled = (bin_mask_u8 | holes).astype(np.uint8)
    return filled


def predict_masks_fallback(img_128):
    """
    Deterministic heuristic segmentation producing 3 class probability maps.
    Output: (128,128,3) in [0,1]
    """
    x = img_128
    x_blur = cv2.GaussianBlur(x, (5, 5), 0)

    t1 = float(np.quantile(x_blur, 0.54))  # was 0.58
    t2 = float(np.quantile(x_blur, 0.72))
    t3 = float(np.quantile(x_blur, 0.80))

    m1 = (x_blur > t1).astype(np.uint8)
    m2 = (x_blur > t2).astype(np.uint8)
    m3 = (x_blur > t3).astype(np.uint8)

    m2 = (m2 & m1).astype(np.uint8)
    m3 = (m3 & m2).astype(np.uint8)

    k = np.ones((5, 5), np.uint8)
    m1 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, k, iterations=1)
    m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, k, iterations=1)
    m3 = cv2.morphologyEx(m3, cv2.MORPH_CLOSE, k, iterations=1)

    m3 = _largest_connected_component(m3)

    out = np.stack([m1, m2, m3], axis=-1).astype(np.float32)
    return out.clip(0.0, 1.0)




## === cell 10
gc.collect()



## === cell 11
lbs, sbs, sts = [], [], []

THRESH = {
    0: 0.24,  # large_bowel
    1: 0.22,  # small_bowel
    2: 0.26,  # stomach
}

K_CLOSE = np.ones((5, 5), np.uint8)
K_OPEN = np.ones((3, 3), np.uint8)

for idx in tqdm(range(df_img.shape[0]), total=df_img.shape[0]):
    img_path = df_img.loc[idx, "path"]
    h = int(df_img.loc[idx, "height"])
    w = int(df_img.loc[idx, "width"])

    img_128 = load_grayscale_128(img_path)
    probs = predict_masks_fallback(img_128)  # (128,128,3)

    for c, store in zip([0, 1, 2], [lbs, sbs, sts]):
        m = probs[:, :, c]
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

        binm = (m >= THRESH[c]).astype(np.uint8)

        if binm.sum() == 0:
            alt = (m >= max(0.12, THRESH[c] * 0.5)).astype(np.uint8)
            if alt.sum() > 0:
                binm = alt

        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, K_CLOSE, iterations=1)
        binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, K_OPEN, iterations=1)
        binm = _fill_holes_u8(binm)

        if c in (0, 2):  # large_bowel, stomach
            binm = _largest_connected_component(binm)

        store.append(rle_encode(binm))

gc.collect()
print("Generated RLEs:", len(lbs), len(sbs), len(sts))



## === cell 12
ids, classes, rles = [], [], []
for i in range(df_img.shape[0]):
    image_id = df_img.loc[i, "id"]
    ids.extend([image_id, image_id, image_id])
    classes.extend(["large_bowel", "small_bowel", "stomach"])
    rles.extend([lbs[i], sbs[i], sts[i]])

sub_pred = pd.DataFrame({"id": ids, "class": classes, "predicted": rles})

sample = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
sub = sample[["id", "class"]].merge(sub_pred, on=["id", "class"], how="left")
sub["predicted"] = sub["predicted"].fillna("")
sub["predicted"] = sub["predicted"].astype(str)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
