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

0.8300169771350062

# 6. Current score

0.39185

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The notebook fails because MONAI isn’t installed/available, so all downstream imports, transforms, model construction, and inference crash before a submission can be written. I remove the dependency on MONAI by replacing the inference section with a minimal, deterministic baseline that still produces a valid RLE submission: predict empty masks for every id/class (legal format, end-to-end stable). I also fix Pandas 2.x issues (deprecated `DataFrame.append`) and correct the JSON writing logic (it was double-writing a JSON string). The result run within the time limit and always write `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.12132) has done: 'Your current 0.0 score is because the submission predicts empty masks for every slice/class, which yields near-zero Dice. To move toward the 0.83 target without changing the overall pipeline structure, I keep your data reading and submission-writing flow but replace the “always empty” prediction with a deterministic, lightweight image-based heuristic: threshold each slice and do small morphological cleanups, then reuse the same binary mask for all three classes (still valid format). I also fix the RLE orientation to match the competition convention (column-major/Fortran order), which is a common source of “looks valid but scores ~0” submissions. These are minimal, dependency-free changes that should substantially increase score while still running within the time limit and producing a valid `submission.csv`.'
- What this solution (achieved 0.31608) has done: 'Your current heuristic is held back mainly by (1) re-scanning the directory listing for every single row (slow and can cause timeouts/partial runs) and (2) using the exact same binary mask for all three classes, which hurts Dice/HD because classes are evaluated separately. I keep your heuristic approach (threshold + simple morphology + RLE) but make it case/day/slice aware by caching the sorted PNG list per case/day and computing the mask once per id, then deriving three slightly different class masks via simple, deterministic intensity-based splits (no new model/training). This is a minimal change that should increase the score substantially toward your 0.83 target while staying dependency-free and within the runtime budget. I also remove an accidental inefficiency in `load_image_volume` (using `.tolist()`), which doesn’t change semantics but reduces memory/time.'
- What this solution (achieved 0.4156) has done: 'The timeout is dominated by per-slice percentile/quantile computations, repeated morphological filtering implemented via huge strided windows, and an O(H·W) Python BFS connected-component pass for every class on every slice. The refactor keeps the exact heuristic logic and thresholds, but replaces expensive operations with provably equivalent faster primitives: `np.partition` for quantiles/percentiles, OpenCV’s highly-optimized morphology for the same open+close, and a vectorized/compiled connected-component extraction for “largest CC”. It also reduces Python overhead by caching per-slice computations (normalized image, `fg`, and its pixel values) and by avoiding repeated dict lookups in the final assignment loop. All I/O paths, submission semantics, and outputs remain the same.'
- What this solution (achieved 0.41526) has done: 'I keep your per-slice cached heuristic pipeline intact, but make two minimal changes that typically move this competition’s score upward without changing the overall approach: (1) enforce a disjoint class assignment (so pixels don’t get counted multiple times across classes) by allocating intensity tiers in a fixed order, and (2) add a tiny “fill small holes” step (close with a slightly larger kernel) before largest-CC extraction to reduce Hausdorff penalties from small gaps. These are deterministic, fast (OpenCV), and preserve your current inference flow and RLE encoding semantics. Everything still runs end-to-end and writes `submission.csv` with the required columns/row count.'
- What this solution (achieved 0.39185) has done: 'Your current score (0.41526) is far below the target (0.8300), so we should cautiously improve without changing the core heuristic pipeline. The biggest low-risk gain here is to fix the connected-component connectivity from 4 to 8 (medical masks are typically diagonally connected), which reduces fragmentation and usually improves both Dice and Hausdorff. Second, we make the “fill small holes” step slightly more consistent by applying the same close kernel but with 2 iterations (still the same morphology step, just a tiny strength increase), which tends to reduce Hausdorff penalties from small gaps without changing the overall approach. Everything else (data flow, thresholds, disjoint class tiers, RLE Fortran order, caching, and submission writing) is kept intact.'

# 9. Code solution

## === cell 0
import os, glob, gc
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



## === cell 1
sub_df = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
if len(sub_df):
    sub = True
else:
    sub = False



## === cell 2
DATASET_FOLDER = "../input/uw-madison-gi-tract-image-segmentation"
if sub == True:
    path_csv = os.path.join(DATASET_FOLDER, "sample_submission.csv")
    df_train = pd.read_csv(path_csv)
    folder = "test"
else:
    path_csv = os.path.join(DATASET_FOLDER, "train.csv")
    df_train = pd.read_csv(path_csv)[:1000]
    df_train = df_train.rename(columns={"segmentation": "predicted"})
    folder = "train"

try:
    display(df_train.head())
except NameError:
    print(df_train.head())




## === cell 3
def extract_details(id_):
    id_fields = id_.split("_")
    case = id_fields[0].replace("case", "")
    day = id_fields[1].replace("day", "")
    slice_id = id_fields[3]
    return {
        "Case": int(case),
        "Day": int(day),
        "Slice": slice_id,
    }




## === cell 4
parts = df_train["id"].str.split("_", expand=True)
df_train["Case"] = parts[0].str.replace("case", "", regex=False).astype(int)
df_train["Day"] = parts[1].str.replace("day", "", regex=False).astype(int)
df_train["Slice"] = parts[3]
del parts

try:
    display(df_train.head())
except NameError:
    print(df_train.head())



## === cell 5
df_train_overview = (
    df_train.groupby(["Case", "Day"], sort=False).size().rename("Slices").reset_index()
)
try:
    display(df_train_overview.head())
except NameError:
    print(df_train_overview.head())



## === cell 6
from PIL import Image


def load_image_volume(img_dir, quant=0.01):
    """
    Kept for compatibility with original notebook structure.
    NOTE: In the optimized script we will not call this (the heuristic works on PNG slices directly).
    """
    imgs = sorted(glob.glob(os.path.join(img_dir, f"*.png")))
    imgs = [np.array(Image.open(p)) for p in imgs]
    vol = np.stack(imgs, axis=0)
    if quant:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)
    v_min, v_max = np.min(vol), np.max(vol)
    if v_max > v_min:
        vol = (vol - v_min) / (v_max - v_min)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)
    vol = (vol * 255).astype(np.uint8)
    del imgs
    return vol




## === cell 7
import nibabel as nib



## === cell 8
df_train_overview["vol_path"] = ""
df_train_overview



## === cell 9
SKIP_NIFTI_BUILD = True

if not SKIP_NIFTI_BUILD:
    for i in range(len(df_train_overview)):
        CASE = df_train_overview["Case"][i]
        DAY = df_train_overview["Day"][i]
        IMAGE_FOLDER = os.path.join(
            "../input/uw-madison-gi-tract-image-segmentation/",
            folder,
            f"case{CASE}",
            f"case{CASE}_day{DAY}",
            "scans",
        )
        vol = load_image_volume(img_dir=IMAGE_FOLDER)
        print(vol.shape)
        nii1 = nib.Nifti1Image(vol, affine=np.eye(4))
        nii_path = f"./{CASE}_{DAY}_vol.nii.gz"
        df_train_overview.loc[i, "vol_path"] = nii_path
        nib.save(nii1, nii_path)



## === cell 10
df_train_overview



## === cell 11
test_data = []
if not SKIP_NIFTI_BUILD:
    for i in range(len(df_train_overview)):
        CASE = df_train_overview["Case"][i]
        DAY = df_train_overview["Day"][i]
        path = {"image": f"./{CASE}_{DAY}_vol.nii.gz"}
        test_data.append(path)



## === cell 12
import json

data1 = {
    "description": "UWM",
    "labels": {
        "0": "background",
        "1": "large_bowel",
        "2": "small_bowel",
        "3": "stomach",
    },
    "test": test_data,
}

json_string = json.dumps(data1)
print(json_string[:500] + ("..." if len(json_string) > 500 else ""))



## === cell 13
if not SKIP_NIFTI_BUILD:
    with open("json_data.json", "w") as outfile:
        json.dump(data1, outfile)



## === cell 14
import sys, subprocess, textwrap

print("Skipping MONAI/einops offline installs: not required for baseline submission.")



## === cell 15
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

print("Torch version:", torch.__version__)

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))



## === cell 16
sz = (80, 144, 192)



## === cell 17
test_transforms = None



## === cell 18
model = None



## === cell 19
pass




## === cell 20
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    if len(s) == 0:
        return np.zeros(shape, dtype=np.uint8)
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = np.asarray(img, dtype=np.uint8).flatten(order="F")
    if pixels.size == 0:
        return ""
    p = np.empty(pixels.size + 2, dtype=np.uint8)
    p[0] = 0
    p[-1] = 0
    p[1:-1] = pixels
    changes = np.flatnonzero(p[1:] != p[:-1]) + 1
    if changes.size == 0:
        return ""
    changes[1::2] -= changes[::2]
    return " ".join(map(str, changes.tolist()))


def show_img(img, mask=None):
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img, cmap="bone")
        if mask is not None:
            plt.imshow(mask[i, :, :], alpha=0.5)
        plt.axis("off")
    plt.show()




## === cell 21
def segm_rle(segm, df_vol):
    """
    Kept for compatibility with original code, but not used in this heuristic submission.
    BUGFIX: replace deprecated DataFrame.append with list accumulation + concat.
    """
    rows = []
    df_vol = df_vol.replace(np.nan, "")
    lbs = sorted(df_vol["class"].unique())
    for idx_, dfg in df_vol.groupby("Slice"):
        idx = int(idx_) - 1
        for i, lb in dfg[["class"]].iterrows():
            lb = lbs.index(lb.item())
            mask = segm[lb, idx, :, :]
            dfg.loc[i, "predicted"] = rle_encode((mask > 0.5).astype(np.uint8))
        rows.append(dfg.loc[:, ["id", "class", "predicted"]])
    del segm
    if len(rows) == 0:
        return pd.DataFrame(columns=["id", "class", "predicted"])
    return pd.concat(rows, axis=0, ignore_index=True)




## === cell 22
datasets = "./json_data.json"
datalist = None
test_ds = None



## === cell 23
from PIL import Image

import cv2


def _quantile_partition_1d(x: np.ndarray, q: float) -> float:
    """Exact numpy-quantile (linear interpolation) using O(n) partition instead of full sort."""
    x = np.asarray(x, dtype=np.float32).ravel()
    n = x.size
    if n == 0:
        return float("nan")
    if n == 1:
        return float(x[0])
    q = float(q)
    if q <= 0.0:
        return float(np.min(x))
    if q >= 1.0:
        return float(np.max(x))
    pos = (n - 1) * q
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(np.partition(x, lo)[lo])
    x2 = np.partition(x, (lo, hi))
    v_lo = float(x2[lo])
    v_hi = float(x2[hi])
    w = pos - lo
    return v_lo + (v_hi - v_lo) * w


def _percentile_partition(x: np.ndarray, p: float) -> float:
    return _quantile_partition_1d(x, float(p) / 100.0)


def _binary_open_close(mask: np.ndarray, k: int = 3, iters: int = 1) -> np.ndarray:
    m = (mask > 0).astype(np.uint8, copy=False)
    kernel = np.ones((k, k), dtype=np.uint8)
    for _ in range(iters):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return m


def _binary_close(mask: np.ndarray, k: int = 5, iters: int = 1) -> np.ndarray:
    m = (mask > 0).astype(np.uint8, copy=False)
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=iters)


def _normalize01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    p1 = _percentile_partition(img, 1.0)
    p99 = _percentile_partition(img, 99.0)
    if p99 > p1:
        img = np.clip(img, p1, p99)
        img = (img - p1) / (p99 - p1)
    else:
        img = img * 0.0
    return img


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8, copy=False)
    if m.sum() == 0:
        return m.astype(np.uint8, copy=False)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    if num <= 1:
        return m
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def heuristic_masks_from_png(png_path: str) -> dict:
    img0 = np.array(Image.open(png_path))
    img = _normalize01(img0)

    thr_fg = _quantile_partition_1d(img, 0.80)
    fg = (img > thr_fg).astype(np.uint8)
    fg = _binary_open_close(fg, k=3, iters=1)

    if fg.sum() < 80:
        thr_fg2 = _quantile_partition_1d(img, 0.72)
        fg2 = (img > thr_fg2).astype(np.uint8)
        fg2 = _binary_open_close(fg2, k=3, iters=1)
        if fg2.sum() > fg.sum():
            fg = fg2

    if fg.sum() < 20:
        z = np.zeros_like(fg)
        return {"large_bowel": z, "small_bowel": z.copy(), "stomach": z.copy()}

    vals = img[fg.astype(bool)]
    q_lo = _quantile_partition_1d(vals, 0.30) if vals.size else 0.0
    q_hi = _quantile_partition_1d(vals, 0.65) if vals.size else 1.0

    stomach = (fg & (img >= q_hi)).astype(np.uint8)
    small_bowel = (fg & (img >= q_lo) & (img < q_hi)).astype(np.uint8)
    large_bowel = (fg & (img < q_lo)).astype(np.uint8)

    small_bowel = (small_bowel & (1 - stomach)).astype(np.uint8)
    large_bowel = (large_bowel & (1 - stomach) & (1 - small_bowel)).astype(np.uint8)

    stomach = _binary_open_close(stomach, k=3, iters=1)
    small_bowel = _binary_open_close(small_bowel, k=3, iters=1)
    large_bowel = _binary_open_close(large_bowel, k=3, iters=1)

    stomach = _binary_close(stomach, k=5, iters=2) if stomach.sum() > 0 else stomach
    small_bowel = (
        _binary_close(small_bowel, k=5, iters=2)
        if small_bowel.sum() > 0
        else small_bowel
    )
    large_bowel = (
        _binary_close(large_bowel, k=5, iters=2)
        if large_bowel.sum() > 0
        else large_bowel
    )

    stomach = _largest_cc(stomach) if stomach.sum() > 0 else stomach
    small_bowel = _largest_cc(small_bowel) if small_bowel.sum() > 0 else small_bowel
    large_bowel = _largest_cc(large_bowel) if large_bowel.sum() > 0 else large_bowel

    if stomach.sum() < 20:
        stomach[:] = 0
    if small_bowel.sum() < 20:
        small_bowel[:] = 0
    if large_bowel.sum() < 20:
        large_bowel[:] = 0

    return {"large_bowel": large_bowel, "small_bowel": small_bowel, "stomach": stomach}




## === cell 24
if sub == True:
    pred_df = pd.read_csv(os.path.join(DATASET_FOLDER, "test.csv"))[
        ["id", "class"]
    ].copy()
else:
    pred_df = df_train[["id", "class"]].copy()

unique_ids = pred_df["id"].unique()
id_parts = pd.Series(unique_ids, copy=False).str.split("_", expand=True)
cases = id_parts[0].to_numpy()
days = id_parts[1].to_numpy()
slice_nums = id_parts[3].astype(np.int32).to_numpy()
del id_parts

case_day_df = pd.DataFrame({"case": cases, "day": days}).drop_duplicates(
    ignore_index=True
)

_scans_cache = {}
for case, day in case_day_df.itertuples(index=False):
    img_dir = os.path.join(DATASET_FOLDER, folder, case, f"{case}_{day}", "scans")
    _scans_cache[(case, day)] = sorted(glob.glob(os.path.join(img_dir, "*.png")))

id_to_png = {}
for id_str, case, day, sl in zip(unique_ids, cases, days, slice_nums):
    scans = _scans_cache.get((case, day), ())
    si = int(sl) - 1
    id_to_png[id_str] = scans[si] if 0 <= si < len(scans) else ""

png_paths = [p for p in id_to_png.values() if p]
unique_pngs = pd.unique(np.asarray(png_paths, dtype=object))

_png_to_rles = {}  # png_path -> {class: rle}
for p in tqdm(unique_pngs, desc="Heuristic inference (per-slice, cached)"):
    masks = heuristic_masks_from_png(p)
    _png_to_rles[p] = {
        k: (rle_encode(v) if v.sum() > 0 else "") for k, v in masks.items()
    }

id_arr = pred_df["id"].to_numpy()
class_arr = pred_df["class"].to_numpy()
pred_out = np.empty(len(pred_df), dtype=object)

png_per_row = np.fromiter(
    (id_to_png.get(x, "") for x in id_arr), dtype=object, count=len(id_arr)
)

for i in range(len(pred_df)):
    png = png_per_row[i]
    if not png:
        pred_out[i] = ""
    else:
        pred_out[i] = _png_to_rles[png].get(class_arr[i], "")

pred_df["predicted"] = pred_out



## === cell 25
if "predicted" not in df_train.columns:
    df_train["predicted"] = ""
if "predicted" in pred_df.columns and len(pred_df) == len(df_train):
    df_train["predicted"] = pred_df["predicted"].to_numpy()
df_train



## === cell 26
pred_df



## === cell 27
if sub == True:
    sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
else:
    sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))[:1000]
    sub_df = sub_df.rename(columns={"segmentation": "predicted"})



## === cell 28
sub_df = sub_df[["id", "class"]].copy()
sub_df = sub_df.merge(
    pred_df[["id", "class", "predicted"]], on=["id", "class"], how="left"
)
sub_df["predicted"] = sub_df["predicted"].fillna("").astype(str)
sub_df = sub_df[["id", "class", "predicted"]]
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())



## === cell 29
sub_df
