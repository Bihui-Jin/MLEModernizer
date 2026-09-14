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
ipywidgets==8.1.5
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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
Your code didn’t yield a score mainly because the submission generation is broken: it uses `df_ssub.empty` incorrectly (so it predicts on train in some contexts) and, more importantly, it encodes RLE with 0-based indices and wrong pixel order (should be 1-based and column-major “top-to-bottom then left-to-right”). I’ll make minimal fixes to RLE decode/encode to match the competition’s definition, and adjust submission building to start from `sample_submission.csv` and fill it deterministically. These changes preserve your core “average volume per day per class + resize + argmax” logic, but make the produced CSV valid for evaluation and typically improves score substantially versus malformed RLE.

```python


## === cell 1
import os, glob
import pandas as pd
import matplotlib.pyplot as plt

DATASET_FOLDER = "/kaggle/input/uw-madison-gi-tract-image-segmentation"



## === cell 2
df_train = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))
print(f"size: {len(df_train)}")
display(df_train.head(3))

df_ssub = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
print(f"size: {len(df_ssub)}")
display(df_ssub.head(3))



## === cell 3
def enrich_data(df, sdir="train"):
    imgs = glob.glob(os.path.join(DATASET_FOLDER, sdir, "case*", "case*_day*", "scans", "*.png"))
    img_folders = [os.path.dirname(p).split(os.path.sep) for p in imgs]
    img_names = [os.path.splitext(os.path.basename(p))[0].split("_") for p in imgs]
    img_keys = [f"{f[-2]}_slice_{n[1]}" for f, n in zip(img_folders, img_names)]

    print(img_keys[:5])
    df["img_path"] = df["id"].map({k: p for k, p in zip(img_keys, imgs)})
    df["Case_Day"] = df["id"].map({k: f[-2] for k, f in zip(img_keys, img_folders)})
    df["Case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["Day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["Slice"] = df["id"].map({k: int(n[1]) for k, n in zip(img_keys, img_names)})
    df["width"] = df["id"].map({k: int(n[2]) for k, n in zip(img_keys, img_names)})
    df["height"] = df["id"].map({k: int(n[3]) for k, n in zip(img_keys, img_names)})
    df["spacing1"] = df["id"].map({k: float(n[4]) for k, n in zip(img_keys, img_names)})
    df["spacing2"] = df["id"].map({k: float(n[5]) for k, n in zip(img_keys, img_names)})



## === cell 4
enrich_data(df_train, "train")
display(df_train.head())

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
df_train.drop_duplicates("Case_Day")[["height", "width"]].value_counts().plot.bar(ax=axes[0], grid=True)
df_train.drop_duplicates("Case_Day")[["spacing1", "spacing2"]].value_counts().plot.bar(ax=axes[1], grid=True)
df_train[df_train["class"] == "stomach"].groupby("Case_Day").size().value_counts().plot.pie(ax=axes[2])



## === cell 5
enrich_data(df_ssub, "test")
display(df_ssub.head())



## === cell 6
import numpy as np

def rle_decode(rle, img=None, label=1):
    if img is None:
        raise ValueError("img must be provided to infer shape")
    if rle is None or (isinstance(rle, float) and np.isnan(rle)) or (isinstance(rle, str) and len(rle.strip()) == 0):
        return img

    s = rle.split()
    starts = np.asarray(s[0::2], dtype=int) - 1  # 1-based -> 0-based
    lengths = np.asarray(s[1::2], dtype=int)
    ends = starts + lengths

    h, w = img.shape
    flat = img.reshape(-1, order="F")
    for lo, hi in zip(starts, ends):
        flat[lo:hi] = label
    return flat.reshape((h, w), order="F")



## === cell 7
import torch.nn.functional as F

def interpolate_volume(volume, vol_size):
    vol_shape = tuple(volume.shape)
    if not vol_size:
        d_new = min(vol_shape[:2])
        vol_size = (vol_shape[0], vol_shape[1], d_new)
    if vol_shape == vol_size:
        return volume
    vol = F.interpolate(volume.unsqueeze(0).unsqueeze(0), size=vol_size, mode="nearest")
    return vol[0, 0]



## === cell 8
import torch
from tqdm.auto import tqdm

AVG_VOLUME_SIZE = (144, 266, 266)
segms, counts = {}, {}

for lb, dfg in df_train.groupby("class"):
    print(lb)
    vol_acc, nb_acc = {}, {}
    for cd, dfgg in tqdm(dfg.groupby("Case_Day")):
        day = dfgg["Day"].iloc[0]
        h, w = dfgg[["height", "width"]].iloc[0]
        vol = np.zeros((len(dfgg), h, w), dtype=np.uint8)
        for _, row in dfgg.iterrows():
            idx = int(row["Slice"]) - 1
            rle = row["segmentation"]
            if not rle or not isinstance(rle, str):
                continue
            vol[idx, :, :] = rle_decode(rle, img=vol[idx, :, :], label=1)
        vol = interpolate_volume(torch.tensor(vol), vol_size=AVG_VOLUME_SIZE).numpy()
        vol_acc[day] = vol_acc.get(day, np.zeros(AVG_VOLUME_SIZE, dtype=np.float32)) + vol
        nb_acc[day] = nb_acc.get(day, 0) + 1
    for d in vol_acc:
        vol_acc[d] = vol_acc[d] / float(nb_acc[d])
    segms[lb] = vol_acc
    counts[lb] = nb_acc



## === cell 9
days = sorted(segms["stomach"].keys())
print(days)



## === cell 10
from ipywidgets import interact, IntSlider

def show_volumes(segms, counts, z, y, x, day=10, fig_size=(12, 12)):
    day = sorted(segms["stomach"])[day]
    fig, axarr = plt.subplots(nrows=3, ncols=3, figsize=fig_size)
    for i, lb in enumerate(segms):
        for j in range(3):
            axarr[i, j].set_title(f"{lb};\n day:{day} with cases:{counts[lb][day]}\n sum over axis {j}")
            im = axarr[i, j].imshow(np.sum(segms[lb][day], axis=j) / segms[lb][day].shape[j])
            fig.colorbar(im, ax=axarr[i, j])

def interactive_show_cum(segms, counts):
    interact(
        lambda z, y, x, day: plt.show(show_volumes(segms, counts, z, y, x, day)),
        z=IntSlider(min=0, max=AVG_VOLUME_SIZE[0], step=5, value=int(AVG_VOLUME_SIZE[0] / 2)),
        y=IntSlider(min=0, max=AVG_VOLUME_SIZE[1], step=5, value=int(AVG_VOLUME_SIZE[1] / 2)),
        x=IntSlider(min=0, max=AVG_VOLUME_SIZE[2], step=5, value=int(AVG_VOLUME_SIZE[2] / 2)),
        day=IntSlider(min=0, max=len(days), step=1, value=0),
    )



## === cell 11
interactive_show_cum(segms, counts)



## === cell 12
from ipywidgets import interact, IntSlider, FloatSlider

LABELS = sorted(df_train["class"].unique())

def show_volume(segms, z, y, x, day=0, thr=0.05, fig_size=(14, 14)):
    day = sorted(segms["stomach"])[day]
    fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))

    segms_lb = [np.ones(AVG_VOLUME_SIZE) * thr]
    segms_lb += [segms[lb][day] for lb in LABELS]
    segm = np.argmax(segms_lb, axis=0)

    _imshow_args = dict(cmap="jet", interpolation="antialiased", interpolation_stage="rgb", vmin=0, vmax=len(segms))
    axarr[0, 0].imshow(segm[z, :, :], **_imshow_args)
    axarr[0, 1].set_title(f"day:{day} (example counts shown per class in counts dict)")
    axarr[0, 1].imshow(segm[:, :, x], **_imshow_args)
    axarr[1, 0].set_title(f"day:{day} (example counts shown per class in counts dict)")
    axarr[1, 0].imshow(segm[:, y, :], **_imshow_args)
    axarr[1, 1].set_axis_off()
    fig.tight_layout()

def interactive_show_segm(segms):
    interact(
        lambda z, y, x, day, thr: plt.show(show_volume(segms, z, y, x, day, thr)),
        z=IntSlider(min=0, max=AVG_VOLUME_SIZE[0], step=5, value=int(AVG_VOLUME_SIZE[0] / 2)),
        y=IntSlider(min=0, max=AVG_VOLUME_SIZE[1], step=5, value=int(AVG_VOLUME_SIZE[1] / 2)),
        x=IntSlider(min=0, max=AVG_VOLUME_SIZE[2], step=5, value=int(AVG_VOLUME_SIZE[2] / 2)),
        day=IntSlider(min=0, max=len(days), step=1, value=0),
        thr=FloatSlider(min=0, max=1, step=0.05, value=0.1),
    )



## === cell 13
interactive_show_segm(segms)



## === cell 14
def rle_encode(mask, bg=0) -> str:
    pixels = mask.reshape(-1, order="F")
    pixels = np.asarray(pixels, dtype=np.uint8)

    pads = np.concatenate([[bg], pixels, [bg]])
    changes = np.where(pads[1:] != pads[:-1])[0]  # positions where value changes
    runs = changes + 1  # 1-based starts in the padded indexing
    runs[1::2] = runs[1::2] - runs[0::2]  # convert to lengths

    if pixels.sum() == 0:
        return ""
    return " ".join(str(x) for x in runs)



## === cell 15
def _interpolate_day(segm_lb, day):
    if day in segm_lb:
        return segm_lb[day]
    days_last = [d for d in segm_lb.keys() if d < day]
    day_last = max(days_last)
    days_next = [d for d in segm_lb.keys() if d > day]
    if not days_next:
        return segm_lb[day_last]
    day_next = min(days_next)
    dn = (day - day_last) / (day_next - day_last)
    dl = (day_next - day) / (day_next - day_last)
    return dl * segm_lb[day_last] + dn * segm_lb[day_next]

def _process_vol(dfgg, segm, thr, lb):
    day = int(dfgg[["Day"]].iloc[0])
    h, w = dfgg[["height", "width"]].iloc[0]
    vol = interpolate_volume(
        torch.tensor(_interpolate_day(segm, day) > thr, dtype=torch.float32),
        vol_size=(len(dfgg), h, w),
    ).numpy().astype(np.uint8)
    rows = []
    for _, row in dfgg.iterrows():
        idx = int(row["Slice"]) - 1
        mask = vol[idx, :, :]
        rle = rle_encode(mask) if np.sum(mask) > 0 else ""
        rows.append({"id": row["id"], "class": lb, "predicted": rle})
    return rows



## === cell 16
segm_thr = 0.05

df_pred_base = df_ssub.copy()
preds = []

for _, dfg in tqdm(df_pred_base.groupby("Case_Day")):
    day = int(dfg[["Day"]].iloc[0])

    segms_lb = [np.ones(AVG_VOLUME_SIZE, dtype=np.float32) * segm_thr]
    segms_lb += [_interpolate_day(segms[lb], day).astype(np.float32) for lb in LABELS]
    segm_sc = np.argmax(segms_lb, axis=0).astype(np.uint8)  # 0=bg, 1..3=classes

    h, w = dfg[["height", "width"]].iloc[0]

    dfgg = dfg[dfg["class"] == LABELS[0]].copy()
    vol = interpolate_volume(torch.tensor(segm_sc, dtype=torch.float32), vol_size=(len(dfgg), h, w)).numpy().astype(np.uint8)

    for _, row in dfgg.iterrows():
        idx = int(row["Slice"]) - 1
        mask_lbl = vol[idx, :, :]  # values in {0,1,2,3}

        for class_idx, lb in enumerate(LABELS, start=1):
            mask = (mask_lbl == class_idx).astype(np.uint8)
            rle = rle_encode(mask) if mask.sum() > 0 else ""
            preds.append({"id": row["id"], "class": lb, "predicted": rle})

df_pred = pd.DataFrame(preds)
display(df_pred[df_pred["predicted"] != ""].head(10))

sub = df_ssub.drop(columns=["predicted"]).merge(df_pred, on=["id", "class"], how="left")
sub["predicted"] = sub["predicted"].fillna("")
sub[["id", "class", "predicted"]].to_csv("submission.csv", index=False)

!head submission.csv
print("submission rows:", len(sub), "non-empty:", (sub["predicted"] != "").sum())
```
