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

0.8359853040468126

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the root cause of the failure by ensuring MONAI is importable without relying on unavailable pip wheels, and I update the MONAI transform names to the current API (Torch 2.6/Kaggle images often have MONAI preinstalled but with the “d” transforms renamed). I also correct the decathlon JSON writing (it was being double-written incorrectly), fix pandas `DataFrame.append` usage (removed in pandas 2.x), and make the inference-to-RLE mapping robust by iterating classes explicitly and aligning slice indices correctly. Finally, I ensure a valid `submission.csv` with the required columns is always produced (fallback to empty masks if any volume/model issue occurs), so you get a submit-ready file end-to-end.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely coming from an invalid/near-empty submission caused by a silent failure to load the external pretrained weights and/or incorrect slice indexing (the test `id` is `case_day_slice` where `slice` is **not** 1..Z), so your current `z=int(slice_str)-1` mapping usually points to the wrong plane. I keep your overall pipeline (NIfTI creation → MONAI UNETR sliding-window inference → sigmoid → per-slice RLE) but make two minimal fixes: (1) robustly locate the pretrained `best_metric_model.pth` under Kaggle inputs (fallback to empty masks only if truly missing), and (2) map each `Slice` to the correct z-index by sorting the slice identifiers within each (Case, Day) volume rather than assuming they are contiguous integers. This should move the score upward toward your target without changing architecture, loss, or inference semantics. The submission writing remains the same and always produces a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with effectively empty/incorrect masks, and the most likely root cause here is that your NIfTI volume is being saved in `(Z,H,W)` order while MONAI expects spatial dims `(H,W,D)` (or after `EnsureChannelFirstd`, `(1,H,W,D)`), so UNETR inference is operating on mis-ordered axes and the per-slice mapping becomes wrong. I keep your exact pipeline (PNG→NIfTI→MONAI UNETR sliding-window inference→sigmoid→0.5 threshold→RLE) but make two minimal, directly score-relevant fixes: (1) save NIfTI volumes as `(H,W,Z)` and (2) make `segm_to_pred_rows` robust to MONAI output being `(3,H,W,Z)` by transposing to `(3,Z,H,W)` before slice-wise RLE. These changes should move the score up substantially toward the target without altering the model, loss, or inference semantics, and the script still always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a “valid-but-wrong/empty” submission caused by either (a) the UNETR weights not being found/loaded (so you fall back to all-empty masks), or (b) a mismatch between the model’s expected input size and the real volume size (so inference produces misaligned outputs). I keep your exact pipeline (PNG→NIfTI→MONAI UNETR sliding-window inference→sigmoid→0.5 threshold→RLE), but make two minimal score-critical fixes: (1) force volumes to be resampled to the fixed `sz=(80,144,192)` your UNETR was built for (using MONAI `Resized`) so the model sees correctly-shaped data, and (2) tighten weight-file discovery to prefer the intended `best_metric_model.pth` and validate it has compatible keys before inference to avoid silent “loaded but wrong” states. These changes should move you off 0.0 toward the target while preserving architecture and inference semantics, and the script still always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import glob, os, gc

DATASET_FOLDER = "../input/uw-madison-gi-tract-image-segmentation"
WORKDIR = "."



## === cell 1
sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
sub = True if len(sub_df) else False
sub, sub_df.shape



## === cell 2
if sub:
    path_csv = os.path.join(DATASET_FOLDER, "test.csv")
    df_base = pd.read_csv(path_csv)  # has id,class only
    folder = "test"
else:
    path_csv = os.path.join(DATASET_FOLDER, "train.csv")
    df_base = pd.read_csv(path_csv)[:1000]
    df_base = df_base.rename(columns={"segmentation": "predicted"})
    folder = "train"

df_base.head()




## === cell 3
def extract_details(id_):
    id_fields = id_.split("_")
    case = id_fields[0].replace("case", "")
    day = id_fields[1].replace("day", "")
    slice_id = id_fields[-1]
    return {"Case": int(case), "Day": int(day), "Slice": slice_id}


df_base[["Case", "Day", "Slice"]] = df_base["id"].apply(
    lambda x: pd.Series(extract_details(x))
)
df_base.head()



## === cell 4
train_overview = []
for (case, day), dfg in df_base.groupby(["Case", "Day"]):
    train_overview.append({"Day": int(day), "Case": int(case), "Slices": len(dfg)})

df_overview = (
    pd.DataFrame(train_overview).sort_values(["Case", "Day"]).reset_index(drop=True)
)
df_overview.head(), len(df_overview)



## === cell 5
from PIL import Image


def load_image_volume(img_dir, quant=0.01):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if len(imgs) == 0:
        raise FileNotFoundError(f"No png slices found in {img_dir}")
    imgs = [np.array(Image.open(p)).tolist() for p in imgs]
    vol = np.array(imgs)
    if quant:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)
    v_min, v_max = np.min(vol), np.max(vol)
    if v_max - v_min < 1e-8:
        vol = np.zeros_like(vol, dtype=np.uint8)
    else:
        vol = (vol - v_min) / (v_max - v_min)
        vol = (vol * 255).astype(np.uint8)
    del imgs
    gc.collect()
    return vol




## === cell 6
import nibabel as nib

df_overview["vol_path"] = ""

for i in range(len(df_overview)):
    CASE = df_overview.loc[i, "Case"]
    DAY = df_overview.loc[i, "Day"]
    IMAGE_FOLDER = os.path.join(
        DATASET_FOLDER, folder, f"case{CASE}", f"case{CASE}_day{DAY}", "scans"
    )
    vol = load_image_volume(img_dir=IMAGE_FOLDER)  # (Z,H,W)

    vol_hwz = np.transpose(vol, (1, 2, 0))  # (H,W,Z)

    nii = nib.Nifti1Image(vol_hwz, affine=np.eye(4))
    nii_path = os.path.join(WORKDIR, f"{CASE}_{DAY}_vol.nii.gz")
    df_overview.loc[i, "vol_path"] = nii_path
    nib.save(nii, nii_path)
    del vol, vol_hwz, nii
    gc.collect()

df_overview.head()



## === cell 7
test_data = [{"image": p} for p in df_overview["vol_path"].tolist()]

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

with open("json_data.json", "r") as f:
    _tmp = json.load(f)
len(_tmp["test"]), _tmp["test"][0]



## === cell 8
monai_available = True
try:
    import monai
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose,
        LoadImaged,
        NormalizeIntensityd,
        EnsureChannelFirstd,
        EnsureTyped,
        Resized,
    )
    from monai.networks.nets import UNETR
    from monai.data import CacheDataset, load_decathlon_datalist
except Exception as e:
    monai_available = False
    monai_import_error = repr(e)

monai_available, (monai_import_error if not monai_available else "ok")



## === cell 9
sz = (80, 144, 192)

if monai_available:
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            NormalizeIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=sz, mode="area"),
            EnsureTyped(keys=["image"]),
        ]
    )



## === cell 10
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
model_loaded = False


def _find_weight_file():
    preferred = [
        "../input/unetr-raw-size/best_metric_model.pth",
        os.path.join(DATASET_FOLDER, "best_metric_model.pth"),
    ]
    for p in preferred:
        if os.path.exists(p) and os.path.isfile(p):
            return p
    candidates = glob.glob("../input/**/best_metric_model.pth", recursive=True)
    if candidates:
        return sorted(candidates)[0]
    candidates = glob.glob("../input/**/*.pth", recursive=True)
    for p in sorted(candidates):
        if os.path.exists(p) and os.path.isfile(p):
            return p
    return None


model_path = _find_weight_file()

if monai_available and (model_path is not None):
    try:
        model = UNETR(
            in_channels=1,
            out_channels=3,
            img_size=sz,
            pos_embed="perceptron",
            dropout_rate=0.2,
        ).to(device)
        state = torch.load(model_path, map_location=device)

        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(unexpected) > 0:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:10]}")
        model.eval()
        model_loaded = True
        model_load_error = None
    except Exception as e:
        model_loaded = False
        model_load_error = repr(e)
else:
    model_loaded = False
    model_load_error = (
        "weights_not_found" if model_path is None else "monai_not_available"
    )

model_loaded, (model_load_error if (not model_loaded) else f"ok:{model_path}")




## === cell 11
def rle_encode(img):
    pixels = img.flatten(order="F")  # column-major: top-to-bottom then left-to-right
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")




## === cell 12
CLASSES = ["large_bowel", "small_bowel", "stomach"]


def segm_to_pred_rows(segm, df_vol):
    """
    segm: expected model output after sigmoid, typically (3,H,W,Z) from MONAI.
          We convert to (3,Z,H,W) for slice-wise RLE using z-index.
    df_vol: rows for a specific (Case,Day), contains id,class,Slice.
    """
    df_vol = df_vol.copy()
    rows = []

    if segm.ndim != 4:
        raise ValueError(f"Unexpected segm shape: {segm.shape}")
    if segm.shape[0] != 3:
        raise ValueError(f"Unexpected channels: {segm.shape}")

    if (
        segm.shape[-1] == df_vol["Slice"].nunique()
        and segm.shape[1] != df_vol["Slice"].nunique()
    ):
        segm = np.transpose(segm, (0, 3, 1, 2))

    z_dim = segm.shape[1]

    slice_vals = df_vol["Slice"].unique().tolist()
    try:
        slice_order = sorted(slice_vals, key=lambda x: int(x))
    except Exception:
        slice_order = sorted(slice_vals)
    slice_to_z = {s: i for i, s in enumerate(slice_order)}

    for slice_str, dfg in df_vol.groupby("Slice"):
        z = slice_to_z.get(slice_str, None)
        if z is None:
            continue

        for cls in CLASSES:
            drow = dfg[dfg["class"] == cls].iloc[0]
            c = CLASSES.index(cls)
            if z < 0 or z >= z_dim:
                rows.append({"id": drow["id"], "class": cls, "predicted": ""})
                continue
            mask2d = (segm[c, z, :, :] > 0.5).astype(np.uint8)
            pred = rle_encode(mask2d) if mask2d.any() else ""
            rows.append({"id": drow["id"], "class": cls, "predicted": pred})

    return pd.DataFrame(rows, columns=["id", "class", "predicted"])




## === cell 13
pred_df = None

if monai_available and model_loaded:
    datasets = "./json_data.json"
    datalist = load_decathlon_datalist(datasets, True, "test")
    test_ds = CacheDataset(
        data=datalist,
        transform=test_transforms,
        cache_num=min(16, len(datalist)),
        cache_rate=1.0,
        num_workers=0,
    )

    pred_parts = []
    for i in range(len(df_overview)):
        x_data = test_ds[i]
        vol_path = x_data["image_meta_dict"]["filename_or_obj"]
        x_df = df_overview[
            df_overview["vol_path"].apply(lambda p: os.path.basename(p))
            == os.path.basename(vol_path)
        ]
        if len(x_df) != 1:
            x_df = df_overview[df_overview["vol_path"] == vol_path]
        CASE = int(x_df["Case"].iloc[0])
        DAY = int(x_df["Day"].iloc[0])

        inputs = torch.unsqueeze(x_data["image"], 0).to(device)
        with torch.no_grad():
            output = sliding_window_inference(inputs, sz, 1, model, overlap=0.8)
        output = output.float().sigmoid().cpu()

        segm = np.squeeze(output.numpy())  # typically (3,H,W,Z) in MONAI

        df_vol = df_base[(df_base["Case"] == CASE) & (df_base["Day"] == DAY)].copy()
        pred_parts.append(segm_to_pred_rows(segm, df_vol))

        del inputs, output, segm, df_vol
        gc.collect()

    pred_df = pd.concat(pred_parts, ignore_index=True)
else:
    pred_df = df_base[["id", "class"]].copy()
    pred_df["predicted"] = ""

pred_df.head(), pred_df.shape



## === cell 14
sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
sub_df = sub_df.drop(columns=["predicted"])
sub_df = sub_df.merge(pred_df, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("")
sub_df.to_csv("submission.csv", index=False)

sub_df.head(), sub_df.shape, os.path.exists("submission.csv")
