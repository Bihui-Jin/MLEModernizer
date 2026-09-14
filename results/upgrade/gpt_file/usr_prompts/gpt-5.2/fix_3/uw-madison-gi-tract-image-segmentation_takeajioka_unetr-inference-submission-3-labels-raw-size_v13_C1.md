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

0.8428073683343751

# 6. Current score

0.0158

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by removing the dependency on `monai_weekly` (not available) and instead using the already-installed `monai` package for UNETR/inference. I also fix Pandas 2.x incompatibilities (`DataFrame.append` removal) by switching to list-accumulation + `pd.concat`, and correct JSON writing so `load_decathlon_datalist` reads a proper dict (not a quoted string). Finally, I make file/path handling robust (choose the correct dataset root, safe normalization when volume is constant, CPU fallback if CUDA is unavailable) and ensure a valid `submission.csv` with the required columns and ordering is always produced.'
- What this solution (achieved 0.0158) has done: 'I fix the end-to-end failures by removing the hard dependency on MONAI (not installed in this environment) and the missing external pretrained weights path, while keeping the same overall pipeline structure (build volumes → iterate per case/day → produce RLE per slice → merge into sample_submission). To move the score up from 0.0 toward the target, I generate a simple, valid binary segmentation per slice using robust intensity-thresholding (Otsu on the 8-bit normalized volume you already create), which is a minimal “model replacement” necessary only because the intended model cannot run here. I also correct a shape/axis issue in `segm_rle` (your volumes are `(Z,H,W)` but `segm_rle` expects `(C,Z,H,W)`) and make RLE encoding match the competition’s required ordering (column-major/Fortran order). Finally, the script always write `submission.csv` with the required `id,class,predicted` columns and correct row alignment to the provided `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os, glob, gc, json
import numpy as np
import pandas as pd



## === cell 1
DATASET_FOLDER = "../input/uw-madison-gi-tract-image-segmentation"
sub_path = os.path.join(DATASET_FOLDER, "sample_submission.csv")
sub_df = pd.read_csv(sub_path)
sub = len(sub_df) > 0
sub



## === cell 2
if sub:
    df_train = pd.read_csv(os.path.join(DATASET_FOLDER, "test.csv"))
    folder = "test"
    df_train["predicted"] = ""
else:
    df_train = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))[:1000]
    df_train = df_train.rename(columns={"segmentation": "predicted"})
    folder = "train"

df_train.head()




## === cell 3
def extract_details(id_):
    id_fields = id_.split("_")
    case = id_fields[0].replace("case", "")
    day = id_fields[1].replace("day", "")
    slice_id = id_fields[3]
    return {"Case": int(case), "Day": int(day), "Slice": slice_id}




## === cell 4
df_train[["Case", "Day", "Slice"]] = df_train["id"].apply(
    lambda x: pd.Series(extract_details(x))
)
df_train.head()



## === cell 5
train_overview = []
for (case, day), dfg in df_train.groupby(["Case", "Day"]):
    train_overview.append({"Day": int(day), "Case": int(case), "Slices": len(dfg)})
df_train_overview = pd.DataFrame(train_overview)
df_train_overview.head()



## === cell 6
from PIL import Image


def load_image_volume(img_dir, quant=0.01):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if len(imgs) == 0:
        raise FileNotFoundError(f"No PNG slices found in: {img_dir}")

    vol = np.stack([np.array(Image.open(p)) for p in imgs], axis=0).astype(np.float32)

    if quant is not None and quant > 0:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)

    v_min, v_max = float(np.min(vol)), float(np.max(vol))
    denom = v_max - v_min
    if denom < 1e-8:
        vol = np.zeros_like(vol, dtype=np.uint8)
    else:
        vol = (vol - v_min) / denom
        vol = (vol * 255.0).astype(np.uint8)

    gc.collect()
    return vol




## === cell 7
import nibabel as nib

df_train_overview["vol_path"] = ""
df_train_overview.head()



## === cell 8
for i in range(len(df_train_overview)):
    CASE = int(df_train_overview.loc[i, "Case"])
    DAY = int(df_train_overview.loc[i, "Day"])
    IMAGE_FOLDER = os.path.join(
        DATASET_FOLDER, folder, f"case{CASE}", f"case{CASE}_day{DAY}", "scans"
    )
    vol = load_image_volume(img_dir=IMAGE_FOLDER)
    print(f"case{CASE}_day{DAY} vol shape:", vol.shape)

    nii1 = nib.Nifti1Image(vol, affine=np.eye(4))
    nii_path = f"./{CASE}_{DAY}_vol.nii.gz"
    df_train_overview.loc[i, "vol_path"] = nii_path
    nib.save(nii1, nii_path)

    del vol, nii1
    gc.collect()



## === cell 9
df_train_overview.head()



## === cell 10
test_data = []
for i in range(len(df_train_overview)):
    CASE = int(df_train_overview.loc[i, "Case"])
    DAY = int(df_train_overview.loc[i, "Day"])
    test_data.append({"image": f"./{CASE}_{DAY}_vol.nii.gz"})
test_data[:2], len(test_data)



## === cell 11
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
print(json.dumps(data1)[:200] + " ...")



## === cell 12
with open("json_data.json", "w") as f:
    json.dump(data1, f)



## === cell 13
pass



## === cell 14
pass



## === cell 15


def otsu_threshold_uint8(img2d_uint8: np.ndarray) -> int:
    """Compute Otsu threshold for a single uint8 2D image without external deps."""
    hist = np.bincount(img2d_uint8.ravel(), minlength=256).astype(np.float64)
    total = img2d_uint8.size
    if total == 0:
        return 0

    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom

    t = int(np.nanargmax(sigma_b2))
    if not np.isfinite(t):
        t = 0
    return t




## === cell 16
sz = (80, 144, 192)




## === cell 17
def rle_decode(mask_rle, shape):
    if (
        mask_rle is None
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
        or str(mask_rle).strip() == ""
    ):
        return np.zeros(shape, dtype=np.uint8)
    s = str(mask_rle).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def rle_encode(img):
    if img is None:
        return ""
    img = (img > 0).astype(np.uint8)
    if img.sum() == 0:
        return ""
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 18
def segm_rle(segm, df_vol):
    """
    segm expected shape: (C, Z, H, W), where C aligns to sorted(df_vol['class'].unique()).
    """
    df_vol = df_vol.copy()
    df_vol["predicted"] = df_vol["predicted"].replace(np.nan, "")

    lbs = sorted(df_vol["class"].unique())
    out_rows = []

    for idx_, dfg in df_vol.groupby("Slice"):
        idx = int(idx_) - 1
        dfg = dfg.copy()
        for row_idx, row in dfg.iterrows():
            lb = lbs.index(row["class"])
            mask = segm[lb, idx, :, :]
            dfg.loc[row_idx, "predicted"] = rle_encode((mask > 0.5).astype(np.uint8))
        out_rows.append(dfg.loc[:, ["id", "class", "predicted"]])

    del segm
    gc.collect()
    return (
        pd.concat(out_rows, axis=0, ignore_index=True)
        if out_rows
        else pd.DataFrame(columns=["id", "class", "predicted"])
    )




## === cell 19
def simple_area_filter(mask2d: np.ndarray, min_area: int = 64) -> np.ndarray:
    if mask2d.sum() < min_area:
        return np.zeros_like(mask2d, dtype=np.uint8)
    return mask2d.astype(np.uint8)


def predict_volume_to_segm(vol_zyx_uint8: np.ndarray, classes: list[str]) -> np.ndarray:
    Z, H, W = vol_zyx_uint8.shape
    C = len(classes)
    segm = np.zeros((C, Z, H, W), dtype=np.float32)
    for z in range(Z):
        sl = vol_zyx_uint8[z]
        t = otsu_threshold_uint8(sl)
        m = (sl > t).astype(np.uint8)

        m = simple_area_filter(m, min_area=max(64, (H * W) // 500))

        if m.any():
            for _ in range(2):
                p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=0)
                m = (
                    p[0:-2, 0:-2]
                    | p[0:-2, 1:-1]
                    | p[0:-2, 2:]
                    | p[1:-1, 0:-2]
                    | p[1:-1, 1:-1]
                    | p[1:-1, 2:]
                    | p[2:, 0:-2]
                    | p[2:, 1:-1]
                    | p[2:, 2:]
                ).astype(np.uint8)

        for c in range(C):
            segm[c, z] = m
    return segm




## === cell 20
pred_parts = []

for i in range(len(df_train_overview)):
    CASE = int(df_train_overview.loc[i, "Case"])
    DAY = int(df_train_overview.loc[i, "Day"])
    IMAGE_FOLDER = os.path.join(
        DATASET_FOLDER, folder, f"case{CASE}", f"case{CASE}_day{DAY}", "scans"
    )

    vol = load_image_volume(img_dir=IMAGE_FOLDER)  # (Z,H,W) uint8
    print("heuristic inputs:", tuple(vol.shape), "->", f"case{CASE}_day{DAY}")

    df_ = df_train[(df_train["Case"] == CASE) & (df_train["Day"] == DAY)]
    classes = sorted(df_["class"].unique())
    segm = predict_volume_to_segm(vol, classes=classes)  # (C,Z,H,W)

    pred_parts.append(segm_rle(segm, df_))

    del vol, segm, df_
    gc.collect()

pred_df = (
    pd.concat(pred_parts, axis=0, ignore_index=True)
    if pred_parts
    else pd.DataFrame(columns=["id", "class", "predicted"])
)
pred_df.head(), len(pred_df)



## === cell 21
df_train = df_train.drop(columns=["predicted"], errors="ignore").merge(
    pred_df, on=["id", "class"], how="left"
)
df_train["predicted"] = df_train["predicted"].fillna("")
df_train.head()



## === cell 22
pred_df.head()



## === cell 23
sub_template = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
sub_template = sub_template.drop(columns=["predicted"], errors="ignore")
sub_df = sub_template.merge(
    df_train[["id", "class", "predicted"]], on=["id", "class"], how="left"
)
sub_df["predicted"] = sub_df["predicted"].fillna("")
sub_df.head()



## === cell 24
sub_df = sub_df.loc[:, ["id", "class", "predicted"]]
sub_df["predicted"] = sub_df["predicted"].astype(str).replace("nan", "")
sub_df.head()



## === cell 25
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)



## === cell 26
sub_df



## === cell 27
assert os.path.exists("submission.csv")
assert list(sub_df.columns) == ["id", "class", "predicted"]
assert len(sub_df) == len(
    pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
)
print("submission.csv looks valid.")
