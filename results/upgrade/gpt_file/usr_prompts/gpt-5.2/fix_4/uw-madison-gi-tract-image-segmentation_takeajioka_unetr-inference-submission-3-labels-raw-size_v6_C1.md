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

0.8146188749264773

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'The crash comes from `extract_details()` returning an empty string for `Slice`, because the IDs in this competition are like `case123_day20_slice_0001` (the slice number is in the 4th token, not embedded in the 3rd). I fix `extract_details()` to robustly parse both common ID patterns and ensure `Slice` is always an integer, then make `segm_rle()` use a fixed class order mapping (instead of deriving indices from `unique()`), which prevents wrong channel indexing. These changes are score-neutral (they just unblock correct formatting/alignment) and let the notebook complete and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os, glob, gc, sys, json, subprocess
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



## === cell 1
DATASET_FOLDER = "../input/uw-madison-gi-tract-image-segmentation"
if not os.path.exists(DATASET_FOLDER):
    DATASET_FOLDER = "/kaggle/input/uw-madison-gi-tract-image-segmentation"

sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
sub = len(sub_df) > 0
print("DATASET_FOLDER:", DATASET_FOLDER, "| sub mode:", sub)



## === cell 2
if sub:
    path_csv = os.path.join(DATASET_FOLDER, "test.csv")
    df_train = pd.read_csv(path_csv)
    df_train["predicted"] = ""
    folder = "test"
else:
    path_csv = os.path.join(DATASET_FOLDER, "train.csv")
    df_train = pd.read_csv(path_csv)[:1000]
    df_train = df_train.rename(columns={"segmentation": "predicted"})
    folder = "train"

print(df_train.head())




## === cell 3
def extract_details(id_):
    parts = str(id_).split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected id format: {id_}")

    case = parts[0].replace("case", "")
    day = parts[1].replace("day", "")

    slice_id = None
    if "slice" in parts:
        k = parts.index("slice")
        if k + 1 < len(parts):
            slice_id = parts[k + 1]
        elif k > 0:
            slice_id = parts[k].replace("slice", "")
    if slice_id is None:
        for p in parts:
            if p.startswith("slice"):
                slice_id = p.replace("slice", "")
                break
    if slice_id is None:
        raise ValueError(f"Could not parse slice from id: {id_}")

    slice_int = int(slice_id)
    return {"Case": int(case), "Day": int(day), "Slice": slice_int}




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
        raise FileNotFoundError(f"No .png slices found in: {img_dir}")
    vol = np.stack([np.array(Image.open(p)) for p in imgs], axis=0)

    if quant:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)
    v_min, v_max = np.min(vol), np.max(vol)
    if v_max > v_min:
        vol = (vol - v_min) / (v_max - v_min)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)
    vol = (vol * 255).astype(np.uint8)

    gc.collect()
    return vol




## === cell 7
import nibabel as nib



## === cell 8
df_train_overview["vol_path"] = ""
print(df_train_overview.head())



## === cell 9
for i in range(len(df_train_overview)):
    CASE = int(df_train_overview.loc[i, "Case"])
    DAY = int(df_train_overview.loc[i, "Day"])
    IMAGE_FOLDER = os.path.join(
        DATASET_FOLDER, folder, f"case{CASE}", f"case{CASE}_day{DAY}", "scans"
    )
    vol = load_image_volume(img_dir=IMAGE_FOLDER)
    print("volume shape:", vol.shape, "|", "CASE/DAY:", CASE, DAY)
    nii1 = nib.Nifti1Image(vol, affine=np.eye(4, dtype=np.float32))
    nii_path = f"./{CASE}_{DAY}_vol.nii.gz"
    df_train_overview.loc[i, "vol_path"] = nii_path
    nib.save(nii1, nii_path)
    del vol, nii1
    gc.collect()



## === cell 10
print(df_train_overview.head())



## === cell 11
test_data = []
for i in range(len(df_train_overview)):
    CASE = int(df_train_overview.loc[i, "Case"])
    DAY = int(df_train_overview.loc[i, "Day"])
    test_data.append({"image": f"./{CASE}_{DAY}_vol.nii.gz"})



## === cell 12
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
USE_MONAI = False
print("USE_MONAI:", USE_MONAI)



## === cell 14
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

print("torch:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 15
sz = (80, 144, 192)




## === cell 16
def load_nifti_as_tensor(path):
    vol = nib.load(path).get_fdata()  # float64
    vol = vol.astype(np.float32)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)
    x = torch.from_numpy(vol)[None, ...]  # [1, D, H, W]
    return x




## === cell 17
model = None
state_path = "../input/unetr-raw-size/best_metric_model.pth"
print("Planned weights path (unused in fallback):", state_path)




## === cell 18
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def show_img(img, mask=None):
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img, cmap="bone")
        if mask is not None:
            plt.imshow(mask[i, :, :], alpha=0.5)
        plt.axis("off")
    plt.show()




## === cell 19
def segm_rle(segm, df_vol, class_to_channel):
    out_parts = []
    df_vol = df_vol.copy()
    df_vol = df_vol.replace(np.nan, "")

    for slice_idx, dfg in df_vol.groupby("Slice", sort=True):
        idx = int(slice_idx) - 1  # convert to 0-based depth index
        dfg = dfg.copy()
        for row_i, row in dfg.iterrows():
            cls = row["class"]
            if cls not in class_to_channel:
                dfg.loc[row_i, "predicted"] = ""
                continue
            ch = class_to_channel[cls]
            mask = segm[ch, idx, :, :]
            dfg.loc[row_i, "predicted"] = rle_encode((mask > 0.5).astype(np.uint8))
        out_parts.append(dfg.loc[:, ["id", "class", "predicted"]])

    del segm
    gc.collect()
    return (
        pd.concat(out_parts, ignore_index=True)
        if len(out_parts)
        else pd.DataFrame(columns=["id", "class", "predicted"])
    )




## === cell 20
pred_parts = []
classes_sorted = ["large_bowel", "small_bowel", "stomach"]
class_to_channel = {c: i for i, c in enumerate(classes_sorted)}

for i in range(len(df_train_overview)):
    CASE = int(df_train_overview.loc[i, "Case"])
    DAY = int(df_train_overview.loc[i, "Day"])
    vol_path = df_train_overview.loc[i, "vol_path"]

    x = load_nifti_as_tensor(vol_path)  # [1, D, H, W]
    _, D, H, W = x.shape
    print("inputs:", tuple(x.shape), "CASE/DAY:", CASE, DAY)

    segm = np.zeros((3, D, H, W), dtype=np.float32)

    df_ = df_train[(df_train["Case"] == CASE) & (df_train["Day"] == DAY)].copy()
    df_ = df_[df_["class"].isin(classes_sorted)].copy()
    pred_parts.append(segm_rle(segm, df_, class_to_channel))

    del x, segm
    gc.collect()

pred_df = (
    pd.concat(pred_parts, ignore_index=True)
    if len(pred_parts)
    else pd.DataFrame(columns=["id", "class", "predicted"])
)
print("pred_df:", pred_df.shape)



## === cell 21
print(pred_df.head())



## === cell 22
print(pred_df.tail())



## === cell 23
if sub:
    sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
else:
    sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))[:1000]
    sub_df = sub_df.rename(columns={"segmentation": "predicted"})



## === cell 24
sub_df = sub_df.drop(columns=["predicted"], errors="ignore")
sub_df = sub_df.merge(pred_df, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("").astype(str)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote", sub_path, "rows:", len(sub_df), "cols:", list(sub_df.columns))



## === cell 25
print(sub_df.head())
print(sub_df.tail())
print(
    "Submission file exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else -1,
)
