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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False



## === cell 1
DATASET_FOLDER = "../input/uw-madison-gi-tract-image-segmentation"
SAMPLE_SUB_PATH = os.path.join(DATASET_FOLDER, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(DATASET_FOLDER, "test.csv")
TRAIN_CSV_PATH = os.path.join(DATASET_FOLDER, "train.csv")
TEST_FOLDER = os.path.join(DATASET_FOLDER, "test")
TRAIN_FOLDER = os.path.join(DATASET_FOLDER, "train")

sub_df = pd.read_csv(SAMPLE_SUB_PATH)
df_test = pd.read_csv(TEST_CSV_PATH)
df_train = pd.read_csv(TRAIN_CSV_PATH)

print(
    "sample_submission:",
    sub_df.shape,
    "test.csv:",
    df_test.shape,
    "train.csv:",
    df_train.shape,
)
print(df_test.head())
print(df_train.head())




## === cell 2
def add_id_fields(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["id"].str.split("_", expand=True)
    df = df.copy()
    df["Case"] = parts[0].str.replace("case", "", regex=False).astype(np.int32)
    df["Day"] = parts[1].str.replace("day", "", regex=False).astype(np.int32)
    df["Slice"] = parts[3]
    df["SliceInt"] = df["Slice"].astype(np.int32)
    return df


df_test = add_id_fields(df_test)
df_train = add_id_fields(df_train)

print(df_test.head())
print(df_train.head())



## === cell 3
df_overview = (
    df_test.groupby(["Case", "Day"], sort=True)
    .size()
    .rename("Slices")
    .reset_index()
    .sort_values(["Case", "Day"])
    .reset_index(drop=True)
)
print(df_overview.head(), "num test volumes:", len(df_overview))

df_train_overview = (
    df_train.groupby(["Case", "Day"], sort=True)
    .size()
    .rename("Slices")
    .reset_index()
    .sort_values(["Case", "Day"])
    .reset_index(drop=True)
)
print(df_train_overview.head(), "num train volumes:", len(df_train_overview))



## === cell 4
_scan_list_cache = {}

try:
    from torchvision.io import read_image as tv_read_image  # (C,H,W) uint8/uint16

    _HAS_TVIO = True
except Exception:
    tv_read_image = None
    _HAS_TVIO = False


def _read_png_grayscale_float32(path: str) -> np.ndarray:
    if _HAS_TVIO:
        x = tv_read_image(path)  # torch uint8/uint16, (C,H,W)
        if x.ndim == 3:
            x = x[0]  # grayscale channel
        return x.numpy().astype(np.float32, copy=False)
    return np.array(Image.open(path)).astype(np.float32, copy=False)


def build_scan_list(img_dir):
    scans = _scan_list_cache.get(img_dir)
    if scans is None:
        scans = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if len(scans) == 0:
            raise FileNotFoundError(f"No png files in {img_dir}")
        _scan_list_cache[img_dir] = scans
    return scans


def attach_slice_idx(df, root_folder):
    keys = df[["Case", "Day"]].drop_duplicates().sort_values(["Case", "Day"])
    mappers = []
    for case, day in keys.itertuples(index=False):
        img_dir = os.path.join(
            root_folder, f"case{int(case)}", f"case{int(case)}_day{int(day)}", "scans"
        )
        scans = build_scan_list(img_dir)
        n = len(scans)
        mappers.append(
            pd.DataFrame(
                {
                    "Case": np.int32(case),
                    "Day": np.int32(day),
                    "SliceInt": np.arange(1, n + 1, dtype=np.int32),
                    "slice_idx": np.arange(0, n, dtype=np.int32),
                    "scan_path": scans,
                }
            )
        )
    mapper = (
        pd.concat(mappers, axis=0, ignore_index=True) if mappers else pd.DataFrame()
    )
    out = df.merge(mapper, on=["Case", "Day", "SliceInt"], how="left", copy=False)
    return out


df_test = attach_slice_idx(df_test, TEST_FOLDER)
df_train = attach_slice_idx(df_train, TRAIN_FOLDER)

print("df_test w/ slice_idx null fraction:", df_test["slice_idx"].isna().mean())
print("df_train w/ slice_idx null fraction:", df_train["slice_idx"].isna().mean())
print(df_test.head())




## === cell 5
def load_volume_from_scans(scans, quant=0.01):
    arrs = [_read_png_grayscale_float32(p) for p in scans]
    vol = np.stack(arrs, axis=0).astype(np.float32, copy=False)  # (z,h,w)
    if quant:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)
    v_min, v_max = float(np.min(vol)), float(np.max(vol))
    if v_max > v_min:
        vol = (vol - v_min) / (v_max - v_min)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)
    return vol.astype(np.float32, copy=False), (v_min, v_max)


def load_image_volume_with_paths(img_dir, quant=0.01):
    scans = build_scan_list(img_dir)
    vol, _ = load_volume_from_scans(scans, quant=quant)
    return vol, scans


def load_image_volume(img_dir, quant=0.01):
    vol, _ = load_image_volume_with_paths(img_dir, quant=quant)
    return vol




## === cell 6
def rle_encode(mask2d: np.ndarray) -> str:
    """
    mask2d: 2D boolean/0-1 array (H,W). Kaggle GI Tract uses RLE over pixels in column-major order
    by flattening the transpose.
    """
    if mask2d.dtype != np.uint8:
        mask2d = mask2d.astype(np.uint8)
    pixels = mask2d.T.reshape(-1)
    pixels = np.ascontiguousarray(pixels)
    pixels = np.concatenate(([0], pixels, [0]))
    runs = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    runs[1::2] -= runs[::2]
    if runs.size == 0:
        return ""
    return " ".join(map(str, runs.tolist()))


def rle_decode(rle: str, shape):
    """
    rle: run-length string (start length ...)
    shape: (H,W)
    Returns uint8 mask (H,W) with 0/1.
    """
    h, w = shape
    if rle is None or (isinstance(rle, float) and np.isnan(rle)) or rle == "":
        return np.zeros((h, w), dtype=np.uint8)
    s = np.fromstring(rle, sep=" ", dtype=np.int64)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(h * w, dtype=np.uint8)
    img[starts] = 1
    img[ends[ends < img.size]] -= 1
    img = np.cumsum(img, dtype=np.int32) > 0
    return img.astype(np.uint8).reshape((w, h)).T




## === cell 7
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(
                d3, size=e3.shape[-2:], mode="bilinear", align_corners=False
            )
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(
                d2, size=e2.shape[-2:], mode="bilinear", align_corners=False
            )
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(
                d1, size=e1.shape[-2:], mode="bilinear", align_corners=False
            )
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.head(d1)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(
                out, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet2D(in_ch=1, out_ch=3, base=32).to(device)

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

print("Using device:", device)



## === cell 8
CLASSES = ["large_bowel", "small_bowel", "stomach"]
class_to_ch = {c: i for i, c in enumerate(CLASSES)}


class TrainSliceDataset(Dataset):
    def __init__(self, df_train, quant=0.01, max_volumes=90):
        """
        Speed-critical changes (correctness-preserving):
        - Cache the already-normalized, quantile-clipped volume once per (Case,Day) and serve slices from RAM.
          This avoids re-reading PNGs in __getitem__ while keeping identical slice values to the original,
          because both training and inference use load_volume_from_scans(scans, quant) -> clip+minmax.
        - Remove the redundant per-slice percentile pass (q_lows/q_highs) and per-item re-normalization, which
          previously produced *different* inputs than vol_raw (it clipped per-slice but normalized per-volume).
          Using vol_raw directly is consistent with the pipeline used at inference and preserves core logic
          (quantile clip + minmax normalization) without approximation.
        """
        self.items = []
        self.quant = float(quant)

        vols = df_train.groupby(["Case", "Day"], sort=True)
        sel = list(vols.groups.keys())[:max_volumes]

        for case, day in sel:
            dfv = df_train[(df_train["Case"] == case) & (df_train["Day"] == day)].copy()
            dfv = dfv[dfv["scan_path"].notna()].copy()
            if len(dfv) == 0:
                continue

            slice_map = (
                dfv[["SliceInt", "slice_idx", "scan_path"]]
                .drop_duplicates()
                .sort_values("slice_idx")
                .reset_index(drop=True)
            )
            scans = slice_map["scan_path"].tolist()

            vol_norm, _ = load_volume_from_scans(
                scans, quant=self.quant
            )  # (z,h,w) float32 in [0,1]

            dfv = dfv.sort_values(["SliceInt", "class"])
            for sid, dfs in dfv.groupby("SliceInt", sort=True):
                row0 = slice_map[slice_map["SliceInt"] == sid]
                if row0.empty:
                    continue
                zidx = int(row0["slice_idx"].iloc[0])

                img_slice = vol_norm[zidx]  # normalized float32 (h,w)
                h, w = int(img_slice.shape[0]), int(img_slice.shape[1])

                mask = np.zeros((3, h, w), dtype=np.uint8)
                for _, r in dfs.iterrows():
                    ch = class_to_ch[r["class"]]
                    mask[ch] = rle_decode(r["segmentation"], (h, w))

                self.items.append((img_slice, mask))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_slice, mask = self.items[i]
        x = torch.from_numpy(img_slice).unsqueeze(0)  # (1,h,w)
        y = torch.from_numpy(mask.astype(np.float32, copy=False))  # (3,h,w)
        return x, y


def pad_collate(batch):
    xs, ys = zip(*batch)
    max_h = max(x.shape[-2] for x in xs)
    max_w = max(x.shape[-1] for x in xs)

    xb, yb, mb = [], [], []
    for x, y in zip(xs, ys):
        _, h, w = x.shape
        pad_h = max_h - h
        pad_w = max_w - w

        xb.append(F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0))
        yb.append(F.pad(y, (0, pad_w, 0, pad_h), mode="constant", value=0.0))

        m = torch.ones((1, h, w), dtype=torch.float32)
        m = F.pad(m, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        mb.append(m)

    xb = torch.stack(xb, 0)
    yb = torch.stack(yb, 0)
    mb = torch.stack(mb, 0)

    if torch.cuda.is_available():
        xb = xb.contiguous(memory_format=torch.channels_last)
        yb = yb.contiguous()
        mb = mb.contiguous()
    return xb, yb, mb




## === cell 9
train_ds = TrainSliceDataset(df_train, quant=0.01, max_volumes=90)
print("Train slices:", len(train_ds))

n = len(train_ds)
idx = np.arange(n)
rng = np.random.default_rng(0)
rng.shuffle(idx)
val_n = max(256, int(0.05 * n)) if n > 0 else 0
val_idx = idx[:val_n]
tr_idx = idx[val_n:]

train_subset = torch.utils.data.Subset(train_ds, tr_idx.tolist())
val_subset = torch.utils.data.Subset(train_ds, val_idx.tolist())

num_workers = min(4, (os.cpu_count() or 2))
train_dl = DataLoader(
    train_subset,
    batch_size=8,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    collate_fn=pad_collate,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)
val_dl = DataLoader(
    val_subset,
    batch_size=8,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    collate_fn=pad_collate,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss(reduction="none")

model.train()
epochs = 5  # unchanged
for ep in range(epochs):
    losses = []
    for xb, yb, mb in train_dl:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        mb = mb.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)

        per_pix = criterion(logits, yb)
        per_pix = per_pix * mb
        denom = mb.sum() * yb.shape[1] + 1e-6
        loss = per_pix.sum() / denom

        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    print(f"epoch {ep+1}/{epochs} loss={np.mean(losses):.4f}")

model.eval()
gc.collect()




## === cell 10
def dice_coef(pred, true):
    pred = pred.astype(np.uint8)
    true = true.astype(np.uint8)
    inter = (pred & true).sum()
    denom = pred.sum() + true.sum()
    if denom == 0:
        return 0.0
    return (2.0 * inter) / float(denom)


def find_best_thresholds(val_dl, grid):
    best_thr = [0.25, 0.25, 0.25]
    best_score = [-1.0, -1.0, -1.0]

    max_batches = 24
    probs_list, y_list, m_list = [], [], []
    with torch.no_grad():
        for bi, (xb, yb, mb) in enumerate(val_dl):
            if bi >= max_batches:
                break
            xb = xb.to(device, non_blocking=True).float()
            logits = model(xb).cpu()
            probs = torch.sigmoid(logits).numpy()  # (b,3,H,W)
            probs_list.append(probs)
            y_list.append(yb.numpy().astype(np.uint8, copy=False))
            m_list.append(mb.numpy())  # (b,1,H,W)

    if not probs_list:
        return tuple(best_thr)

    probs = np.concatenate(probs_list, axis=0)  # (N,3,H,W)
    y = np.concatenate(y_list, axis=0)  # (N,3,H,W) uint8
    m = np.concatenate(m_list, axis=0)[:, 0]  # (N,H,W) float
    mm = (m > 0).astype(np.uint8)  # (N,H,W)

    grid = np.asarray(list(grid), dtype=np.float32)  # (T,)
    T = grid.shape[0]
    N = probs.shape[0]

    for ch in range(3):
        pr = probs[:, ch]
        gt = y[:, ch]
        gt = (gt * mm).astype(np.uint8, copy=False)

        pred = (pr[None, ...] > grid[:, None, None, None]).astype(np.uint8)
        pred = pred * mm[None, ...]

        inter = (pred & gt[None, ...]).sum(axis=(2, 3))  # (T,N)
        denom = pred.sum(axis=(2, 3)) + gt[None, ...].sum(axis=(2, 3))  # (T,N)

        dice = np.zeros((T, N), dtype=np.float32)
        nz = denom != 0
        dice[nz] = (2.0 * inter[nz]) / denom[nz].astype(np.float32)

        sc = dice.mean(axis=1)  # (T,)
        bi = int(np.argmax(sc))
        best_score[ch] = float(sc[bi])
        best_thr[ch] = float(grid[bi])

    print("Calibrated thresholds:", best_thr, "val dice per class:", best_score)
    return tuple(best_thr)


thr_grid = [
    0.10,
    0.12,
    0.14,
    0.16,
    0.18,
    0.20,
    0.22,
    0.24,
    0.26,
    0.28,
    0.30,
    0.32,
    0.35,
    0.38,
    0.40,
]
CAL_THR = find_best_thresholds(val_dl, thr_grid)

FALLBACK_THR = tuple(max(0.05, t - 0.06) for t in CAL_THR)




## === cell 11
class SliceDataset(Dataset):
    def __init__(self, vol: np.ndarray):
        self.vol = vol

    def __len__(self):
        return self.vol.shape[0]

    def __getitem__(self, idx):
        img = self.vol[idx]
        img = torch.from_numpy(img).unsqueeze(0)
        return img, idx


def infer_volume_to_masks(vol: np.ndarray, batch_size=8, thr=(0.25, 0.25, 0.25)):
    if isinstance(thr, (float, int)):
        thr = (float(thr), float(thr), float(thr))
    thr = np.asarray(thr, dtype=np.float32).reshape(1, 3, 1, 1)

    ds = SliceDataset(vol)
    inf_workers = min(2, (os.cpu_count() or 2) - 1)
    inf_workers = max(0, inf_workers)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=inf_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(inf_workers > 0),
        prefetch_factor=2 if inf_workers > 0 else None,
    )
    z, h, w = vol.shape
    out = np.zeros((3, z, h, w), dtype=np.uint8)

    with torch.no_grad():
        for xb, idxb in dl:
            xb = xb.to(device, non_blocking=True).float()
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            probs = torch.sigmoid(model(xb)).cpu().numpy()  # (b,3,h,w)
            preds = (probs > thr).astype(np.uint8)  # (b,3,h,w)
            zi = idxb.numpy().astype(np.int64, copy=False)
            out[:, zi, :, :] = np.transpose(preds, (1, 0, 2, 3))
    return out




## === cell 12
pred_rows = []

DEFAULT_THR = CAL_THR

test_groups = {
    k: v.sort_values(["slice_idx", "class"]).reset_index(drop=True)
    for k, v in df_test.groupby(["Case", "Day"], sort=True)
}

for i in range(len(df_overview)):
    CASE = int(df_overview.loc[i, "Case"])
    DAY = int(df_overview.loc[i, "Day"])
    img_dir = os.path.join(TEST_FOLDER, f"case{CASE}", f"case{CASE}_day{DAY}", "scans")

    scans = build_scan_list(img_dir)
    vol, _ = load_volume_from_scans(scans, quant=0.01)
    z, h, w = vol.shape
    print(f"Volume case{CASE}_day{DAY} shape:", vol.shape)

    segm = infer_volume_to_masks(vol, batch_size=8, thr=DEFAULT_THR)

    per_class_nonempty = segm.reshape(3, -1).sum(axis=1) > 0
    if not bool(per_class_nonempty.all()):
        segm_fb = infer_volume_to_masks(vol, batch_size=8, thr=FALLBACK_THR)
        for ch in range(3):
            if not per_class_nonempty[ch]:
                segm[ch] = segm_fb[ch]
        del segm_fb

    df_vol = test_groups.get((CASE, DAY))
    if df_vol is None:
        del vol, segm
        gc.collect()
        continue

    ids = df_vol["id"].to_numpy()
    classes = df_vol["class"].to_numpy()
    slice_idx_arr = df_vol["slice_idx"].to_numpy()

    for cls in CLASSES:
        ch = class_to_ch[cls]
        sel = classes == cls
        if not np.any(sel):
            continue
        ids_c = ids[sel]
        sidx_c = slice_idx_arr[sel]

        rles = []
        for sidx in sidx_c:
            if pd.isna(sidx):
                mask = np.zeros((h, w), dtype=np.uint8)
            else:
                si = int(sidx)
                if 0 <= si < z:
                    mask = segm[ch, si]
                else:
                    mask = np.zeros((h, w), dtype=np.uint8)
            rles.append(rle_encode(mask))

        pred_rows.extend(
            {"id": i0, "class": cls, "predicted": r0} for i0, r0 in zip(ids_c, rles)
        )

    del vol, segm
    gc.collect()

pred_df = pd.DataFrame(pred_rows, columns=["id", "class", "predicted"])
print("pred_df:", pred_df.shape)
print(pred_df.head())

non_empty = (pred_df["predicted"].astype(str).str.len() > 0).mean()
print(f"Non-empty predicted RLE fraction: {non_empty:.4f}")

assert pred_df.shape[0] == df_test.shape[0], (pred_df.shape, df_test.shape)



## === cell 13
sub_df = pd.read_csv(SAMPLE_SUB_PATH)
sub_df = sub_df[["id", "class"]].merge(pred_df, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("")
assert sub_df.shape[0] == pd.read_csv(SAMPLE_SUB_PATH).shape[0]

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", sub_df.shape)
print(sub_df.head())
