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

0.84244779784096

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the MONAI dependency (it’s not installed in your environment) and replace the failing MONAI UNETR inference with a minimal, deterministic baseline that still produces a valid Kaggle submission. I keep your existing data parsing and per-slice grouping logic, but generate empty (all-background) masks for every id/class so the pipeline runs end-to-end and writes `submission.csv` with the correct columns and row count. I also fix the “sub mode” detection so it doesn’t accidentally run the training branch, and I make the volume-building step optional (skipped) since it’s not needed for producing a valid submission and would waste time. This is score-conservative (likely low) but satisfies the primary requirement: no runtime errors and a valid `.csv` submission.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is because the submission predicts empty masks everywhere; the smallest legitimate way to move toward the 0.842 target is to generate non-empty masks using information already present in this competition’s data (the slice PNG filenames encode the original image height/width). I keep your overall structure (read test.csv → parse ids → loop per case/day/slice → write submission.csv) and only add minimal inference: create a simple threshold-based foreground mask per slice (with light morphology and largest-component cleanup for stability) and use it for all three classes. I also fix the RLE encoding to follow the competition’s required column-major order (top-to-bottom then left-to-right), which is critical for any non-empty mask to score correctly. This stays within the constraints (no MONAI, no architecture/training changes) while moving the score upward from 0.0 toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with a submission that is effectively empty/incorrectly-aligned for the metric; the smallest legitimate push toward your 0.842 target is to (1) make sure IDs map to the correct slice PNGs, (2) ensure the RLE encoding order exactly matches the competition’s definition, and (3) reduce over/under-segmentation with a tiny, deterministic per-slice calibration that still keeps your simple threshold baseline (no new models, no training). I keep your current pipeline (sample_submission → per-id PNG load → threshold mask → RLE → submission.csv) but fix the slice-index parsing (the slice is the 3rd token, not the last), and I use class-specific thresholds (stomach tends to be larger/clearer than bowel) while keeping the same morphology/CC cleanup. These changes are minimal, deterministic, and directly aimed at moving the score up from 0.0 without altering the overall approach or adding dependencies.'
- What this solution (achieved 0.0) has done: 'I fix the failing slice parsing in `get_slice_png_path_from_id`: in this competition IDs look like `case123_day20_slice_0001`, so the slice number is the 4th token, not the 3rd. I also make the parser robust to both `..._slice_0001` and `..._0001` forms to avoid crashing and reduce missing PNG matches. With that fixed, cell 20 produce `pred_df`, and cells 21–22 successfully merge into the required `submission.csv` with correct columns and row count. The core “threshold + morphology + RLE” inference logic is unchanged, so this is purely a correctness/runtime fix (and should improve score vs empty/missing predictions).'

# 9. Code solution

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
df_train[["Case", "Day", "Slice"]] = df_train["id"].apply(
    lambda x: pd.Series(extract_details(x))
)
print(df_train.head())



## === cell 5
train_overview = []
for (case, day), dfg in df_train.groupby(["Case", "Day"]):
    train_overview.append({"Day": int(day), "Case": int(case), "Slices": len(dfg)})
df_train_overview = pd.DataFrame(train_overview)
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
for i in range(len(df_train_overview)):
    CASE = df_train_overview.loc[i, "Case"]
    DAY = df_train_overview.loc[i, "Day"]
    df_train_overview.loc[i, "vol_path"] = f"./{CASE}_{DAY}_vol.nii.gz"

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
    s = mask_rle.split()
    if len(s) == 0:
        return np.zeros(shape, dtype=np.uint8)
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0])).T


def rle_encode(img):
    pixels = img.flatten(order="F").astype(np.uint8)
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
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=np.uint8)
    best = []
    best_len = 0

    ys, xs = np.where(mask > 0)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = 1
        comp = [(y0, x0)]
        while stack:
            y, x = stack.pop()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= ny < H
                    and 0 <= nx < W
                    and (mask[ny, nx] > 0)
                    and (visited[ny, nx] == 0)
                ):
                    visited[ny, nx] = 1
                    stack.append((ny, nx))
                    comp.append((ny, nx))
        if len(comp) > best_len:
            best_len = len(comp)
            best = comp

    out = np.zeros((H, W), dtype=np.uint8)
    if best_len > 0:
        yy, xx = zip(*best)
        out[np.array(yy), np.array(xx)] = 1
    return out


def predict_mask_from_slice(img_u8, thr=0.55):
    img = img_u8.astype(np.float32)
    lo, hi = np.percentile(img, [5, 95])
    if hi <= lo:
        return np.zeros_like(img_u8, dtype=np.uint8)
    x = (img - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)

    m = (x > float(thr)).astype(np.uint8)

    H, W = m.shape
    b = max(2, int(min(H, W) * 0.02))
    m[:b, :] = 0
    m[-b:, :] = 0
    m[:, :b] = 0
    m[:, -b:] = 0

    m = _binary_close(m, iters=1)
    m = _binary_open(m, iters=1)

    m = _largest_connected_component(m)

    if m.sum() < 25:
        return np.zeros_like(img_u8, dtype=np.uint8)
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
def get_slice_png_path_from_id(id_str, dataset_folder=DATASET_FOLDER, folder=folder):
    toks = id_str.split("_")
    case_tok = toks[0]  # e.g. case110
    day_tok = toks[1]  # e.g. day12
    if len(toks) >= 4 and toks[2].lower() == "slice":
        slice_idx = toks[3]
    else:
        slice_idx = toks[2]
    case_day = f"{case_tok}_{day_tok}"

    img_dir = os.path.join(dataset_folder, folder, case_tok, case_day, "scans")
    if not os.path.isdir(img_dir):
        return None

    try:
        s4 = f"{int(slice_idx):04d}"
    except ValueError:
        return None

    pngs = sorted(glob.glob(os.path.join(img_dir, f"*_{s4}.png")))
    if len(pngs) == 0:
        pngs = sorted(glob.glob(os.path.join(img_dir, f"*{s4}.png")))
    return pngs[0] if len(pngs) > 0 else None


CLASS_THR = {
    "large_bowel": 0.57,
    "small_bowel": 0.60,
    "stomach": 0.54,
}

pred_rows = []
missing = 0

sample_sub = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))

for rid, rclass in zip(sample_sub["id"].values, sample_sub["class"].values):
    p = get_slice_png_path_from_id(rid)
    if p is None:
        missing += 1
        pred_rows.append({"id": rid, "class": rclass, "predicted": ""})
        continue

    img = np.array(Image.open(p), dtype=np.uint8)

    if img.ndim != 2:
        img = img[..., 0] if img.ndim == 3 else img.reshape(img.shape[0], img.shape[1])

    thr = CLASS_THR.get(rclass, 0.57)
    mask = predict_mask_from_slice(img, thr=thr)

    pred_rows.append({"id": rid, "class": rclass, "predicted": rle_encode(mask)})

pred_df = pd.DataFrame(pred_rows)
print("pred_df:", pred_df.shape, "missing_slices:", missing)
print(pred_df.head())



## === cell 21
sub_df = sample_sub.merge(
    pred_df, on=["id", "class"], how="left", suffixes=("", "_pred")
)
sub_df["predicted"] = sub_df["predicted_pred"].fillna("").astype(str)
sub_df = sub_df[["id", "class", "predicted"]]

assert len(sub_df) == len(sample_sub), "Row count mismatch vs sample_submission"
assert list(sub_df.columns) == ["id", "class", "predicted"]

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)



## === cell 22
sub_df.head()
