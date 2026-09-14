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

# 5. Target score

0.55248

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob
import pandas as pd
import matplotlib.pyplot as plt

DATASET_FOLDER = "/kaggle/input/uw-madison-gi-tract-image-segmentation"


## === cell 1
df_train = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))
print(f"size: {len(df_train)}")
display(df_train.head(3))

df_ssub = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
print(f"size: {len(df_ssub)}")
display(df_ssub.head(3))


## === cell 2
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


## === cell 3
enrich_data(df_train, "train")
display(df_train.head())

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
df_train.drop_duplicates("Case_Day")[["height", "width"]].value_counts().plot.bar(ax=axes[0], grid=True)
df_train.drop_duplicates("Case_Day")[["spacing1", "spacing2"]].value_counts().plot.bar(ax=axes[1], grid=True)
df_train[df_train["class"] == "stomach"].groupby("Case_Day").size().value_counts().plot.pie(ax=axes[2])


## === cell 4
enrich_data(df_ssub, "test")
display(df_ssub.head())


## === cell 5
import numpy as np

def rle_decode(rle, img = None, label = 1):
    seq = rle.split()
    starts = np.array(list(map(int, seq[0::2])))
    lengths = np.array(list(map(int, seq[1::2])))
    ends = starts + lengths
    img_shape = img.shape
    img = img.flatten()
    for begin, end in zip(starts, ends):
        img[begin:end] = label
    return img.reshape(img_shape)


## === cell 6
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


## === cell 7
import torch
import numpy as np
from tqdm.auto import tqdm

AVG_VOLUME_SIZE = (144, 266, 266)
segms, counts = {}, {}

for lb, dfg in df_train.groupby("class"):
    print(lb)
    vol_acc, nb_acc = {}, {}
    for cd, dfgg in tqdm(dfg.groupby("Case_Day")):
        day = dfgg["Day"].iloc[0]
        h, w = dfgg[["height", "width"]].iloc[0]
        vol = np.zeros((len(dfgg), h, w))
        for _, row in dfgg.iterrows():
            idx = int(row['Slice']) - 1
            rle = row["segmentation"]
            if not rle or not isinstance(rle, str):
                continue
            vol[idx, :, :] = rle_decode(rle, img=vol[idx, :, :], label=1)
        vol = interpolate_volume(torch.tensor(vol), vol_size=AVG_VOLUME_SIZE).numpy()
        vol_acc[day] = vol_acc.get(day, np.zeros(AVG_VOLUME_SIZE, dtype=np.uint16)) + vol
        nb_acc[day] = nb_acc.get(day, 0) + 1
    for d in vol_acc:
        vol_acc[d] = vol_acc[d] / float(nb_acc[d])
    segms[lb] = vol_acc
    counts[lb] = nb_acc


## === cell 8
days = sorted(segms["stomach"].keys())
print(days)


## === cell 9
from ipywidgets import interact, IntSlider

def show_volumes(segms, counts, z, y, x, day=10, fig_size=(12, 12)):
    day = sorted(segms["stomach"])[day]
    fig, axarr = plt.subplots(nrows=3, ncols=3, figsize=fig_size)
    for i, lb in enumerate(segms):
        for j in range(3):
            axarr[i, j].set_title(f"{lb};\n day:{day} with cases:{counts[lb][day]}\n sum over axis {j}")
            im = axarr[i, j].imshow(np.sum(segms[lb][day], axis=j) / segms[lb][day].shape[j])
            fig.colorbar(im, ax= axarr[i, j])

def interactive_show_cum(segms, counts):
    interact(
        lambda z, y, x, day: plt.show(show_volumes(segms, counts, z, y, x, day)),
        z=IntSlider(min=0, max=AVG_VOLUME_SIZE[0], step=5, value=int(AVG_VOLUME_SIZE[0] / 2)),
        y=IntSlider(min=0, max=AVG_VOLUME_SIZE[1], step=5, value=int(AVG_VOLUME_SIZE[1] / 2)),
        x=IntSlider(min=0, max=AVG_VOLUME_SIZE[2], step=5, value=int(AVG_VOLUME_SIZE[2] / 2)),
        day=IntSlider(min=0, max=len(days), step=1, value=0),
    )


## === cell 10
interactive_show_cum(segms, counts)


## === cell 11
from ipywidgets import interact, IntSlider, FloatSlider

LABELS = sorted(df_train['class'].unique())

def show_volume(segms, z, y, x, day=0, thr=0.05, fig_size=(14, 14)):
    day = sorted(segms["stomach"])[day]
    fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
    
    segms_lb = [np.ones(AVG_VOLUME_SIZE) * thr]
    segms_lb += [segms[lb][day] for lb in LABELS]
    segm = np.argmax(segms_lb, axis=0)
    
    _imshow_args = dict(cmap="jet", interpolation="antialiased", interpolation_stage="rgb", vmin=0, vmax=len(segms))
    axarr[0, 0].imshow(segm[z, :, :], **_imshow_args)
    axarr[0, 1].set_title(f"day:{day} with cases:{counts[lb][day]}")
    axarr[0, 1].imshow(segm[:, :, x], **_imshow_args)
    axarr[1, 0].set_title(f"day:{day} with cases:{counts[lb][day]}")
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


## === cell 12
interactive_show_segm(segms)


## === cell 13
def rle_encode(mask, bg = 0) -> dict:
    vec = mask.flatten()
    nb = len(vec)
    where = np.flatnonzero
    starts = np.r_[0, where(~np.isclose(vec[1:], vec[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, nb])
    values = vec[starts]
    assert len(starts) == len(lengths) == len(values)
    rle = {}
    for start, length, val in zip(starts, lengths, values):
        if val == bg:
            continue
        rle[val] = rle.get(val, []) + [str(start), length]
    rle = {lb: " ".join(map(str, id_lens)) for lb, id_lens in rle.items()}
    return rle


## === cell 14


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
        torch.tensor(_interpolate_day(segm, day) > thr, dtype=float),
        vol_size=(len(dfgg), h, w),
    ).numpy().astype(np.uint8)
    rows = []
    for _, row in dfgg.iterrows():
        idx = int(row['Slice']) - 1
        mask = vol[idx, :, :]
        rle = rle_encode(mask)[1] if np.sum(mask) > 1 else ""
        rows.append({"id": row["id"], "class": lb, "predicted": rle})
    return rows


## === cell 15
segm_thr = 0.05
df_pred = df_train if df_ssub.empty else df_ssub

preds = []
for _, dfg in tqdm(df_pred.groupby("Case_Day")):
    day = int(dfg[["Day"]].iloc[0])
    segms_lb = [np.ones(AVG_VOLUME_SIZE) * segm_thr]
    segms_lb += [_interpolate_day(segms[lb], day) for lb in LABELS]
    segm_sc = np.argmax(segms_lb, axis=0)
    h, w = dfg[["height", "width"]].iloc[0]
    dfgg = dfg[dfg["class"] == LABELS[0]]
    vol = interpolate_volume(torch.tensor(segm_sc, dtype=float), vol_size=(len(dfgg), h, w)).numpy().astype(np.uint8)
    for _, row in dfgg.iterrows():
        idx = int(row['Slice']) - 1
        mask = vol[idx, :, :]
        rle = rle_encode(mask) if np.sum(mask) > 1 else {}
        preds += [{"id": row["id"], "class": lb, "predicted": rle.get(i + 1, "")} for i, lb in enumerate(LABELS)]


df_pred = pd.DataFrame(preds)
display(df_pred[df_pred['predicted'] != ""])


## === cell 16
del df_ssub['predicted']
df_pred = df_ssub.merge(df_pred, on=['id','class'])

df_pred[['id', 'class', 'predicted']].to_csv("submission.csv", index=False)

!head submission.csv
