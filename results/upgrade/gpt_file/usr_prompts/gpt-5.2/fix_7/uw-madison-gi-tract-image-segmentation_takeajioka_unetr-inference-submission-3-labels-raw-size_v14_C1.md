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
import os, glob, gc
import numpy as np
import pandas as pd

from PIL import Image

import random

random.seed(0)
np.random.seed(0)



## === cell 1
DATASET_FOLDER = "../input/uw-madison-gi-tract-image-segmentation"

sample_path = os.path.join(DATASET_FOLDER, "sample_submission.csv")
test_csv_path = os.path.join(DATASET_FOLDER, "test.csv")
train_csv_path = os.path.join(DATASET_FOLDER, "train.csv")

sub_df = pd.read_csv(sample_path)
test_df = pd.read_csv(test_csv_path)

print("sample_submission:", sub_df.shape, list(sub_df.columns))
print("test.csv:", test_df.shape, list(test_df.columns))

df_work = test_df.copy()




## === cell 2
def extract_details(id_):
    id_fields = id_.split("_")
    case = id_fields[0].replace("case", "")
    day = id_fields[1].replace("day", "")
    slice_id = id_fields[3]
    return {"Case": int(case), "Day": int(day), "Slice": slice_id}


parts = df_work["id"].str.split("_", expand=True)
df_work["Case"] = parts[0].str.replace("case", "", regex=False).astype(np.int32)
df_work["Day"] = parts[1].str.replace("day", "", regex=False).astype(np.int32)
df_work["Slice"] = parts[3]
print(df_work.head())



## === cell 3
df_overview = (
    df_work.groupby(["Case", "Day"], sort=True)
    .size()
    .rename("Slices")
    .reset_index()
    .sort_values(["Case", "Day"])
    .reset_index(drop=True)
)
print("df_overview:", df_overview.shape)
print(df_overview.head())




## === cell 4
def load_image_volume(img_dir, quant=0.01):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if len(imgs) == 0:
        raise FileNotFoundError(f"No PNGs found under: {img_dir}")

    with Image.open(imgs[0]) as im0:
        a0 = np.array(im0)
    h, w = a0.shape[:2]
    vol = np.empty((len(imgs), h, w), dtype=a0.dtype)
    vol[0] = a0

    for i, p in enumerate(imgs[1:], start=1):
        with Image.open(p) as im:
            vol[i] = np.array(im)

    if quant:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)

    v_min, v_max = float(np.min(vol)), float(np.max(vol))
    denom = (v_max - v_min) if (v_max - v_min) != 0 else 1.0
    vol = (vol - v_min) / denom
    vol = (vol * 255).astype(np.uint8)

    return vol




## === cell 5
df_overview["image_folder"] = ""
df_overview["vol_shape"] = ""

for i in range(len(df_overview)):
    CASE = int(df_overview.loc[i, "Case"])
    DAY = int(df_overview.loc[i, "Day"])
    image_folder = os.path.join(
        DATASET_FOLDER, "test", f"case{CASE}", f"case{CASE}_day{DAY}", "scans"
    )
    df_overview.loc[i, "image_folder"] = image_folder

    imgs = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if len(imgs) == 0:
        raise FileNotFoundError(f"No PNGs found under: {image_folder}")

    with Image.open(imgs[0]) as im:
        h, w = im.size[1], im.size[0]
    z = len(imgs)
    df_overview.loc[i, "vol_shape"] = str((z, h, w))
    print("Indexed", (CASE, DAY), "volume shape:", (z, h, w))

df_overview.head()




## === cell 6
def rle_encode(img):
    """
    img: 2D numpy array of {0,1}
    Returns run length as string formatted: start length start length ...
    """
    if img.ndim != 2:
        raise ValueError(f"rle_encode expects 2D array, got shape {img.shape}")

    if img.max() == 0:
        return ""

    pixels = img.flatten(order="F")  # column-major (Fortran order)
    pixels = np.concatenate([[0], pixels, [0]]).astype(np.uint8)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 7
CLASSES = ["large_bowel", "small_bowel", "stomach"]
class_to_idx = {c: i for i, c in enumerate(CLASSES)}


def _box_blur2d(img2d_u8, k=9):
    k = int(k)
    if k < 1:
        return img2d_u8.astype(np.float32)
    if k % 2 == 0:
        k += 1
    r = k // 2

    a = img2d_u8.astype(np.float32)
    pad = np.pad(a, ((0, 0), (r, r)), mode="edge")
    c = np.cumsum(pad, axis=1)
    h = (c[:, k:] - c[:, :-k]) / k
    pad = np.pad(h, ((r, r), (0, 0)), mode="edge")
    c = np.cumsum(pad, axis=0)
    v = (c[k:, :] - c[:-k, :]) / k
    return v


def _pad_crop_to_hw(img2d, target_h, target_w):
    h, w = img2d.shape
    out = np.zeros((target_h, target_w), dtype=img2d.dtype)
    h0 = min(h, target_h)
    w0 = min(w, target_w)
    out[:h0, :w0] = img2d[:h0, :w0]
    return out


def _largest_cc(mask_u8):
    """
    Keep only the largest 4-connected component.

    Perf: rework BFS to avoid storing full pixel coordinate lists and heavy Python tuple churn.
    Correctness: returns the same largest 4-connected component as before.
    """
    m = (mask_u8.astype(np.uint8) != 0).astype(np.uint8)
    if m.sum() == 0:
        return m

    from collections import deque

    h, w = m.shape
    visited = np.zeros((h, w), dtype=np.uint8)

    best_count = 0
    best_seed = None

    ys, xs = np.where(m == 1)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx]:
            continue
        q = deque()
        q.append((sy, sx))
        visited[sy, sx] = 1
        count = 0
        while q:
            y, x = q.pop()
            count += 1
            y1 = y - 1
            if y1 >= 0 and m[y1, x] and not visited[y1, x]:
                visited[y1, x] = 1
                q.append((y1, x))
            y1 = y + 1
            if y1 < h and m[y1, x] and not visited[y1, x]:
                visited[y1, x] = 1
                q.append((y1, x))
            x1 = x - 1
            if x1 >= 0 and m[y, x1] and not visited[y, x1]:
                visited[y, x1] = 1
                q.append((y, x1))
            x1 = x + 1
            if x1 < w and m[y, x1] and not visited[y, x1]:
                visited[y, x1] = 1
                q.append((y, x1))

        if count > best_count:
            best_count = count
            best_seed = (sy, sx)

    if best_seed is None:
        return np.zeros((h, w), dtype=np.uint8)

    out = np.zeros((h, w), dtype=np.uint8)
    q = deque([best_seed])
    out[best_seed[0], best_seed[1]] = 1
    visited.fill(0)
    visited[best_seed[0], best_seed[1]] = 1

    while q:
        y, x = q.pop()
        y1 = y - 1
        if y1 >= 0 and m[y1, x] and not visited[y1, x]:
            visited[y1, x] = 1
            out[y1, x] = 1
            q.append((y1, x))
        y1 = y + 1
        if y1 < h and m[y1, x] and not visited[y1, x]:
            visited[y1, x] = 1
            out[y1, x] = 1
            q.append((y1, x))
        x1 = x - 1
        if x1 >= 0 and m[y, x1] and not visited[y, x1]:
            visited[y, x1] = 1
            out[y, x1] = 1
            q.append((y, x1))
        x1 = x + 1
        if x1 < w and m[y, x1] and not visited[y, x1]:
            visited[y, x1] = 1
            out[y, x1] = 1
            q.append((y, x1))

    return out


def predict_heuristic_masks_for_volume(vol_u8):
    z, h, w = vol_u8.shape
    segm = np.zeros((len(CLASSES), z, h, w), dtype=np.uint8)

    by0, by1 = int(h * 0.05), int(h * 0.95)
    bx0, bx1 = int(w * 0.05), int(w * 0.95)

    st_y0, st_y1 = int(h * 0.10), int(h * 0.60)
    st_x0, st_x1 = int(w * 0.10), int(w * 0.70)

    sb_y0, sb_y1 = int(h * 0.35), int(h * 0.90)
    sb_x0, sb_x1 = int(w * 0.15), int(w * 0.85)

    lb_y0, lb_y1 = int(h * 0.20), int(h * 0.95)
    lb_x0, lb_x1 = int(w * 0.05), int(w * 0.95)

    cx0, cx1 = int(w * 0.25), int(w * 0.75)
    cy0, cy1 = int(h * 0.35), int(h * 0.85)

    for zi in range(z):
        sl = vol_u8[zi]
        sl_blur = _box_blur2d(sl, k=9)

        roi_body = sl_blur[by0:by1, bx0:bx1]
        thr_body = float(np.percentile(roi_body, 40.0))
        body = (sl_blur >= thr_body).astype(np.uint8)
        body[:by0, :] = 0
        body[by1:, :] = 0
        body[:, :bx0] = 0
        body[:, bx1:] = 0
        body = _largest_cc(body)

        if body.sum() == 0:
            cy, cx = h // 2, w // 2
            ry, rx = max(2, h // 30), max(2, w // 30)
            body[cy - ry : cy + ry, cx - rx : cx + rx] = 1

        roi_st = sl_blur[st_y0:st_y1, st_x0:st_x1]
        thr_st = float(np.percentile(roi_st, 78.0))
        m_st = ((sl_blur >= thr_st).astype(np.uint8)) & body
        m_st[:st_y0, :] = 0
        m_st[st_y1:, :] = 0
        m_st[:, :st_x0] = 0
        m_st[:, st_x1:] = 0
        m_st = _largest_cc(m_st.astype(np.uint8))

        roi_sb = sl_blur[sb_y0:sb_y1, sb_x0:sb_x1]
        thr_sb = float(np.percentile(roi_sb, 76.0))
        m_sb = ((sl_blur >= thr_sb).astype(np.uint8)) & body
        m_sb[:sb_y0, :] = 0
        m_sb[sb_y1:, :] = 0
        m_sb[:, :sb_x0] = 0
        m_sb[:, sb_x1:] = 0
        m_sb = _largest_cc(m_sb.astype(np.uint8))

        roi_lb = sl_blur[lb_y0:lb_y1, lb_x0:lb_x1]
        thr_lb = float(np.percentile(roi_lb, 74.0))
        m_lb = ((sl_blur >= thr_lb).astype(np.uint8)) & body
        m_lb[:lb_y0, :] = 0
        m_lb[lb_y1:, :] = 0
        m_lb[:, :lb_x0] = 0
        m_lb[:, lb_x1:] = 0
        m_lb[cy0:cy1, cx0:cx1] = 0
        m_lb = _largest_cc(m_lb.astype(np.uint8))

        m_lb = _pad_crop_to_hw(m_lb.astype(np.uint8), h, w)
        m_sb = _pad_crop_to_hw(m_sb.astype(np.uint8), h, w)
        m_st = _pad_crop_to_hw(m_st.astype(np.uint8), h, w)

        segm[class_to_idx["large_bowel"], zi, :, :] = m_lb
        segm[class_to_idx["small_bowel"], zi, :, :] = m_sb
        segm[class_to_idx["stomach"], zi, :, :] = m_st

    return segm




## === cell 8
def segm_rle(segm, df_vol):
    """
    segm: (C, Z, H, W)
    df_vol: rows for a single (Case, Day), includes columns id, class, Slice
    Returns dataframe with id,class,predicted for those rows
    """
    if df_vol.shape[0] == 0:
        return pd.DataFrame(columns=["id", "class", "predicted"])

    z = segm.shape[1]
    dfv = df_vol.loc[:, ["id", "class", "Slice"]].copy()

    idx = dfv["Slice"].astype(np.int32).to_numpy() - 1
    idx[idx < 0] = 0
    idx[idx >= z] = z - 1

    cls_idx = dfv["class"].map(class_to_idx).astype(np.int8).to_numpy()

    predicted = np.empty(dfv.shape[0], dtype=object)
    for i in range(dfv.shape[0]):
        mask2d = segm[int(cls_idx[i]), int(idx[i]), :, :]
        predicted[i] = rle_encode(mask2d)

    dfv["predicted"] = predicted
    return dfv.loc[:, ["id", "class", "predicted"]]




## === cell 9
pred_parts = []

groups = {k: v for k, v in df_work.groupby(["Case", "Day"], sort=False)}

for i in range(len(df_overview)):
    CASE = int(df_overview.loc[i, "Case"])
    DAY = int(df_overview.loc[i, "Day"])
    image_folder = df_overview.loc[i, "image_folder"]

    vol = load_image_volume(image_folder)
    segm = predict_heuristic_masks_for_volume(vol)

    df_cd = groups.get((CASE, DAY))
    if df_cd is None:
        df_cd = df_work.iloc[0:0].copy()

    pred_parts.append(segm_rle(segm, df_cd))

    del vol, segm, df_cd
    gc.collect()

pred_df = (
    pd.concat(pred_parts, axis=0, ignore_index=True)
    if len(pred_parts)
    else pd.DataFrame(columns=["id", "class", "predicted"])
)
print("pred_df:", pred_df.shape)
print(pred_df.head())



## === cell 10
sub_df = pd.read_csv(sample_path)

sub_df = sub_df.drop(columns=["predicted"], errors="ignore")
sub_df = sub_df.merge(pred_df, on=["id", "class"], how="left")

sub_df["predicted"] = sub_df["predicted"].fillna("")
sub_df = sub_df[["id", "class", "predicted"]]

assert (
    sub_df.shape[0] == pd.read_csv(sample_path).shape[0]
), "Row count mismatch vs sample_submission"
assert list(sub_df.columns) == [
    "id",
    "class",
    "predicted",
], "Submission columns incorrect"

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())



## === cell 11
print("Unique predicted strings (sample):", sub_df["predicted"].unique()[:5])
