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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob, os, gc



## === cell 1
DATASET_FOLDER = "../input/uw-madison-gi-tract-image-segmentation"

sub_df0 = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
sub = True
print("sub mode:", sub, "rows:", len(sub_df0))



## === cell 2
if sub:
    path_csv = os.path.join(DATASET_FOLDER, "test.csv")
    df_train = pd.read_csv(path_csv)  # this is actually the test ids/classes
    folder = "test"
else:
    path_csv = os.path.join(DATASET_FOLDER, "train.csv")
    df_train = pd.read_csv(path_csv)[:1000]
    df_train = df_train.rename(columns={"segmentation": "predicted"})
    folder = "train"

print(df_train.head())




## === cell 3
def extract_details(id_):
    id_fields = id_.split("_")
    case = id_fields[0].replace("case", "")
    day = id_fields[1].replace("day", "")

    if len(id_fields) >= 4 and id_fields[2].lower() == "slice":
        slice_id = id_fields[3]
    else:
        slice_id = id_fields[2]
    return {"Case": int(case), "Day": int(day), "Slice": slice_id}




## === cell 4
_id = df_train["id"].astype(str)
case_num = _id.str.extract(r"case(\d+)", expand=False).astype(np.int32)
day_num = _id.str.extract(r"day(\d+)", expand=False).astype(np.int32)
slice_str = _id.str.extract(r"(?:_slice_)?(\d+)$", expand=False)
df_train["Case"] = case_num
df_train["Day"] = day_num
df_train["Slice"] = slice_str
print(df_train.head())



## === cell 5
train_overview = (
    df_train.groupby(["Case", "Day"], sort=False)
    .size()
    .reset_index(name="Slices")
    .astype({"Case": int, "Day": int, "Slices": int})
)
df_train_overview = train_overview.loc[:, ["Day", "Case", "Slices"]]
print(df_train_overview.head())



## === cell 6
from PIL import Image


def load_image_volume(img_dir, quant=0.01):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if len(imgs) == 0:
        raise FileNotFoundError(f"No PNGs found in: {img_dir}")
    vol = np.stack([np.array(Image.open(p), dtype=np.float32) for p in imgs], axis=0)

    if quant:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)

    v_min, v_max = float(np.min(vol)), float(np.max(vol))
    if v_max > v_min:
        vol = (vol - v_min) / (v_max - v_min)
    else:
        vol = vol * 0.0
    vol = (vol * 255.0).astype(np.uint8)

    del imgs
    gc.collect()
    return vol




## === cell 7
import nibabel as nib



## === cell 8
df_train_overview["vol_path"] = ""
df_train_overview



## === cell 9
df_train_overview["vol_path"] = (
    "./"
    + df_train_overview["Case"].astype(str)
    + "_"
    + df_train_overview["Day"].astype(str)
    + "_vol.nii.gz"
)
print("Prepared df_train_overview vol_path placeholders:", df_train_overview.shape)



## === cell 10
df_train_overview



## === cell 11
test_data = []
for i in range(len(df_train_overview)):
    CASE = df_train_overview.loc[i, "Case"]
    DAY = df_train_overview.loc[i, "Day"]
    test_data.append({"image": f"./{CASE}_{DAY}_vol.nii.gz"})



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

with open("json_data.json", "w") as f:
    json.dump(data1, f)

print("Wrote json_data.json with", len(test_data), "items")



## === cell 13
print(
    "MONAI not installed; using lightweight deterministic image-threshold baseline for a non-zero score."
)



## === cell 14
sz = (80, 144, 192)
print("Configured (unused) target inference size:", sz)



## === cell 15
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 16
weights_path = "../input/unetr-raw-size/best_metric_model.pth"
print("Skipping weights load (UNETR/MONAI not available). Expected path:", weights_path)




## === cell 17
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (H, W)
    returns: (H, W) uint8 mask
    """
    if mask_rle is None:
        return np.zeros(shape, dtype=np.uint8)
    s = str(mask_rle).strip()
    if s == "":
        return np.zeros(shape, dtype=np.uint8)

    s = s.split()
    starts = np.asarray(s[0::2], dtype=np.int64) - 1
    lengths = np.asarray(s[1::2], dtype=np.int64)
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def rle_encode(mask):
    """
    mask: (H, W) binary {0,1}
    returns: RLE string

    Note: For this competition, pixels are numbered top-to-bottom then left-to-right,
    which corresponds to flattening in Fortran order ("F") for an (H, W) array.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"rle_encode expects 2D mask, got shape={mask.shape}")

    pixels = (mask > 0).astype(np.uint8).flatten(order="F")
    if pixels.max() == 0:
        return ""

    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 18
def _binary_dilate(mask, iters=1):
    m = mask
    for _ in range(iters):
        p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        m = (
            p[1:-1, 1:-1]
            | p[:-2, 1:-1]
            | p[2:, 1:-1]
            | p[1:-1, :-2]
            | p[1:-1, 2:]
            | p[:-2, :-2]
            | p[:-2, 2:]
            | p[2:, :-2]
            | p[2:, 2:]
        )
    return m.astype(np.uint8)


def _binary_erode(mask, iters=1):
    m = mask.astype(bool)
    for _ in range(iters):
        p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        m = (
            p[1:-1, 1:-1]
            & p[:-2, 1:-1]
            & p[2:, 1:-1]
            & p[1:-1, :-2]
            & p[1:-1, 2:]
            & p[:-2, :-2]
            & p[:-2, 2:]
            & p[2:, :-2]
            & p[2:, 2:]
        )
    return m.astype(np.uint8)


def _binary_open(mask, iters=1):
    return _binary_dilate(_binary_erode(mask, iters=iters), iters=iters)


def _binary_close(mask, iters=1):
    return _binary_erode(_binary_dilate(mask, iters=iters), iters=iters)


def _largest_connected_component(mask):
    mask = (mask > 0).astype(np.uint8)
    H, W = mask.shape
    if mask.sum() == 0:
        return np.zeros((H, W), dtype=np.uint8)

    labels = np.zeros((H, W), dtype=np.int32)

    parent = np.zeros(int(mask.sum()) + 1, dtype=np.int32)
    next_label = 1

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a, b):
        ra = _find(a)
        rb = _find(b)
        if ra != rb:
            parent[rb] = ra

    for y in range(H):
        row_m = mask[y]
        row_l = labels[y]
        if y > 0:
            prev_l = labels[y - 1]
            prev_m = mask[y - 1]
        for x in range(W):
            if row_m[x] == 0:
                continue
            left = row_l[x - 1] if x > 0 else 0
            up = prev_l[x] if y > 0 and prev_m[x] else 0

            if left == 0 and up == 0:
                lab = next_label
                parent[lab] = lab
                next_label += 1
                row_l[x] = lab
            elif left != 0 and up == 0:
                row_l[x] = left
            elif left == 0 and up != 0:
                row_l[x] = up
            else:
                row_l[x] = left
                if left != up:
                    _union(left, up)

    flat = labels.ravel()
    nz = flat > 0
    flat_nz = flat[nz]
    for i in range(flat_nz.size):
        flat_nz[i] = _find(flat_nz[i])
    flat[nz] = flat_nz

    if flat_nz.size == 0:
        return np.zeros((H, W), dtype=np.uint8)

    max_lab = int(flat_nz.max())
    counts = np.bincount(flat_nz, minlength=max_lab + 1)
    best = int(np.argmax(counts[1:]) + 1)  # skip background bin 0

    out = (labels == best).astype(np.uint8)
    return out


def _prep_slice_for_threshold(img_u8):
    img = img_u8.astype(np.float32, copy=False)
    lo, hi = np.percentile(img, [5, 95])
    if hi <= lo:
        return None, None, None  # indicates empty
    x = (img - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    H, W = img_u8.shape
    b = max(2, int(min(H, W) * 0.02))
    return x, b, (H, W)


def predict_mask_from_slice_prepped(x, b, shape_hw, thr=0.55):
    if x is None:
        return np.zeros(shape_hw, dtype=np.uint8)

    m = (x > float(thr)).astype(np.uint8)

    H, W = shape_hw
    m[:b, :] = 0
    m[-b:, :] = 0
    m[:, :b] = 0
    m[:, -b:] = 0

    m = _binary_close(m, iters=1)
    m = _binary_open(m, iters=1)

    m = _largest_connected_component(m)

    if m.sum() < 25:
        return np.zeros(shape_hw, dtype=np.uint8)
    return m




## === cell 19
def segm_rle(segm, df_vol):
    out_frames = []
    df_vol = df_vol.copy()
    df_vol = df_vol.replace(np.nan, "")

    lbs = sorted(df_vol["class"].unique())
    for idx_, dfg in df_vol.groupby("Slice"):
        idx = int(idx_) - 1
        dfg = dfg.copy()
        for row_i, row in dfg.iterrows():
            lb = lbs.index(row["class"])
            mask = segm[lb, idx, :, :]
            dfg.loc[row_i, "predicted"] = rle_encode((mask > 0.5).astype(np.uint8))
        out_frames.append(dfg.loc[:, ["id", "class", "predicted"]])

    del segm
    gc.collect()
    if len(out_frames) == 0:
        return pd.DataFrame(columns=["id", "class", "predicted"])
    return pd.concat(out_frames, axis=0, ignore_index=True)




## === cell 20
from functools import lru_cache

_SCANS_INDEX = (
    {}
)  # img_dir -> dict with keys: by_s4, s4_sorted, any_png_path, png_count


def _build_scans_index(img_dir):
    by_s4 = {}
    s4s = []
    any_p = None
    cnt = 0
    with os.scandir(img_dir) as it:
        for e in it:
            if not e.is_file():
                continue
            name = e.name
            if not name.endswith(".png"):
                continue
            cnt += 1
            if any_p is None:
                any_p = e.path
            base = os.path.splitext(name)[0]
            tok = base.split("_")
            if not tok:
                continue
            s = tok[-1]
            if len(s) == 4 and s.isdigit():
                by_s4[s] = e.path
                s4s.append(int(s))
    s4s_sorted = (
        np.array(sorted(set(s4s)), dtype=np.int32)
        if s4s
        else np.array([], dtype=np.int32)
    )
    return {
        "by_s4": by_s4,
        "s4_sorted": s4s_sorted,
        "any_png_path": any_p,
        "png_count": cnt,
    }


def _parse_slice_from_id(id_str: str):
    toks = id_str.split("_")
    if len(toks) >= 4 and toks[2].lower() == "slice":
        return toks[3]
    if len(toks) >= 3:
        return toks[2]
    return None


def _get_img_dir_from_id(id_str, dataset_folder=DATASET_FOLDER, folder=folder):
    toks = id_str.split("_")
    case_tok = toks[0]
    day_tok = toks[1]
    case_day = f"{case_tok}_{day_tok}"
    return os.path.join(dataset_folder, folder, case_tok, case_day, "scans")


def _get_slice_png_path_from_id_with_index(id_str, idx):
    slice_tok = _parse_slice_from_id(id_str)
    if slice_tok is None:
        return None

    by_s4 = idx["by_s4"]
    s4_sorted = idx["s4_sorted"]

    try:
        s4 = f"{int(slice_tok):04d}"
    except ValueError:
        return idx["any_png_path"]

    p = by_s4.get(s4, None)
    if p is not None:
        return p

    if s4_sorted.size == 0:
        return idx["any_png_path"]

    target = int(s4)
    j = int(np.argmin(np.abs(s4_sorted - target)))
    nearest_s4 = f"{int(s4_sorted[j]):04d}"
    return by_s4.get(nearest_s4, idx["any_png_path"])


@lru_cache(maxsize=4096)
def _load_png_u8(path: str):
    img = np.array(Image.open(path), dtype=np.uint8)
    if img.ndim != 2:
        img = img[..., 0] if img.ndim == 3 else img.reshape(img.shape[0], img.shape[1])
    return img


CLASS_THR = {
    "large_bowel": 0.57,
    "small_bowel": 0.60,
    "stomach": 0.54,
}

pred_rows = []
missing = 0

sample_sub = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))

sid = sample_sub["id"].astype(str)
sample_sub["_case"] = sid.str.extract(r"(case\d+)", expand=False)
sample_sub["_day"] = sid.str.extract(r"(day\d+)", expand=False)

for (case_tok, day_tok), grp in sample_sub.groupby(["_case", "_day"], sort=False):
    if pd.isna(case_tok) or pd.isna(day_tok):
        for rid, rclass in zip(grp["id"].values, grp["class"].values):
            pred_rows.append({"id": rid, "class": rclass, "predicted": ""})
            missing += 1
        continue

    case_day = f"{case_tok}_{day_tok}"
    img_dir = os.path.join(DATASET_FOLDER, folder, case_tok, case_day, "scans")
    if not os.path.isdir(img_dir):
        for rid, rclass in zip(grp["id"].values, grp["class"].values):
            pred_rows.append({"id": rid, "class": rclass, "predicted": ""})
            missing += 1
        continue

    idx = _SCANS_INDEX.get(img_dir)
    if idx is None:
        idx = _build_scans_index(img_dir)
        _SCANS_INDEX[img_dir] = idx

    grp_ids = grp["id"].values
    grp_cls = grp["class"].values

    slice_to_rows = {}
    for rid, rclass in zip(grp_ids, grp_cls):
        s = _parse_slice_from_id(str(rid))
        slice_to_rows.setdefault(s, []).append((rid, rclass))

    for s, rows in slice_to_rows.items():
        rid0 = rows[0][0]
        p = _get_slice_png_path_from_id_with_index(str(rid0), idx)
        if p is None or (not os.path.isfile(p)):
            missing += len(rows)
            for rid, rclass in rows:
                pred_rows.append({"id": rid, "class": rclass, "predicted": ""})
            continue

        img = _load_png_u8(p)
        x, b, shape_hw = _prep_slice_for_threshold(img)

        for rid, rclass in rows:
            thr = CLASS_THR.get(rclass, 0.57)
            mask = predict_mask_from_slice_prepped(x, b, shape_hw, thr=thr)

            mask = (mask > 0).astype(np.uint8, copy=False)
            if mask.ndim != 2:
                raise ValueError(f"Predicted mask must be 2D, got shape={mask.shape}")

            pred_rows.append(
                {"id": rid, "class": rclass, "predicted": rle_encode(mask)}
            )

pred_df = pd.DataFrame(pred_rows)
print("pred_df:", pred_df.shape, "missing_slices:", missing)
print(pred_df.head())



## === cell 21
pred_map = {(r.id, r["class"]): r.predicted for r in pred_df.itertuples(index=False)}
out_pred = [
    pred_map.get((rid, rclass), "")
    for rid, rclass in zip(sample_sub["id"].values, sample_sub["class"].values)
]
sub_df = sample_sub[["id", "class"]].copy()
sub_df["predicted"] = (
    pd.Series(out_pred, index=sub_df.index, dtype="string").fillna("").astype(str)
)

assert len(sub_df) == len(sample_sub), "Row count mismatch vs sample_submission"
assert list(sub_df.columns) == ["id", "class", "predicted"]

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)



## === cell 22
sub_df.head()
