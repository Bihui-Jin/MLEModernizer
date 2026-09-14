# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Detect the presence of ink from 3d x-ray scans of detached fragments of ancient papyrus scrolls.

## Metric
We evaluate how well your output image matches our reference image using a modified version of the [Sørensen--Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient), where instead of using the F1 score, we are using the F0.5 score. The F0.5 score is given by:

$$
\frac{\left(1+\beta^2\right) p r}{\beta^2 p+r} \text { where } p=\frac{t p}{t p+f p}, r=\frac{t p}{t p+f n}, \beta=0.5
$$

The F0.5 score weights precision higher than recall, which improves the ability to form coherent characters out of detected ink areas.

In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the output should be binary, with 0 indicating "no ink" and 1 indicating "ink".

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from left to right, then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

Your output should be a single file, **submission.csv**, with this run-length encoded information. This should have a header with two columns, `Id` and `Predicted`, and with one row for every directory under **test/**. For example:

```
Id,Predicted
a,1 1 5 1 etc.
b,10 20 etc.
```

For a real-world example of what these files look like, see `inklabels_rce.csv` in the data directories, which have been generated with [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0).


## Data
- **[train/test]/[fragment_id]/surface_volume/[image_id].tif** slices from the 3d x-ray [surface volume](https://scrollprize.org/tutorial1#3-surface-volumes). Each file contains a greyscale slice in the z-direction. Each fragment contains 65 slices. Combined this image stack gives us `width * height * 65` number of voxels per fragment. You can expect two fragments in the hidden test set, which together are roughly the same size as a single training fragment. The sample slices available to download in the test folders are simply copied from training fragment one, but when you submit your notebook they will be substituted with the real test data.
- **[train/test]/[fragment_id]/mask.png** --- a binary mask of which pixels contain data.
- **train/[fragment_id]/inklabels.png** --- a binary mask of the ink vs no-ink labels.
- **train/[fragment_id]/inklabels_rle.csv** --- a run-length-encoded version of the labels, generated using [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0). This is the same format as you should make your submission in.
- **train/[fragment_id]/ir.png** --- the infrared photo on which the binary mask is based.
- **sample_submission.csv**, an example of a submission file in the correct format. You need to output the following file in the home directory: **submission.csv**.

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
seaborn==0.12.2
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
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        input/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        working/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
```

-> data/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> input/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> (stopped after 10 files for performance)

# 5. Target score

0.0001022775828751

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.09739) has done: 'I fix the pipeline so it no longer depends on the missing `/kaggle/input/vesuvis-data-preparation/data_0.5.csv` by generating a minimal CSV directly from the provided `train/` and `test/` folders and writing the required `.npy` files into `/kaggle/working` (same structure your transforms expect). I also correct the dataframe column name typo (`fragmet_id` → `fragment_id`) consistently to avoid silent filtering bugs, and ensure volume/mask/label arrays have the expected shapes so the UNet receives `(B,16,H,W)` reliably. Finally, I fix inference/prediction plumbing so `module` and `predictions` exist, and produce a valid `submission.csv` with exactly `Id,Predicted` columns and one row per test fragment, using your existing RLE routine.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")




## === cell 1
from collections import defaultdict
from io import StringIO
from typing import Tuple, Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image as Image
import pytorch_lightning as pl
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

try:
    import lovely_numpy as ln  # type: ignore
except Exception:
    ln = None


def set_determinism(seed: int = 0):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetaTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x: torch.Tensor):
        return torch.Tensor._make_subclass(cls, x, require_grad=x.requires_grad)

    def as_tensor(self):
        return self


def _ensure_torch(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            data = t(data)
        return data


class LoadImaged:
    def __init__(self, keys, ensure_channel_first: bool = False):
        if isinstance(keys, str):
            keys = (keys,)
        self.keys = keys
        self.ensure_channel_first = ensure_channel_first

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(data)
        for k in self.keys:
            path = out[k]
            arr = np.load(path)
            if self.ensure_channel_first:
                if arr.ndim == 2:
                    arr = arr[None, ...]  # (1, H, W)
                elif arr.ndim == 3:
                    pass
            out[k] = arr
        return out


class LoadVolumedFromTifs:
    def __init__(
        self,
        key: str,
        volumes_dir_key: str,
        z_start: int,
        z_dim: int,
        downsampling: float = 1.0,
    ):
        self.key = key
        self.volumes_dir_key = volumes_dir_key
        self.z_start = int(z_start)
        self.z_dim = int(z_dim)
        self.downsampling = float(downsampling)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(data)
        vdir = Path(out[self.volumes_dir_key])
        paths = sorted(vdir.glob("*.tif"))[self.z_start : self.z_start + self.z_dim]
        if len(paths) != self.z_dim:
            raise RuntimeError(
                f"Expected {self.z_dim} slices in {vdir}, got {len(paths)}"
            )

        z_slices = []
        for p in paths:
            img = Image.open(p)
            if self.downsampling != 1.0:
                size = int(img.size[0] * self.downsampling), int(
                    img.size[1] * self.downsampling
                )
                img = img.resize(size, resample=Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 65535.0
            z_slices.append(arr)
        out[self.key] = np.stack(z_slices, axis=0).astype(np.float32)  # (Z,H,W)
        return out


class RandFlipd:
    def __init__(self, keys, prob: float, spatial_axis: int):
        if isinstance(keys, str):
            keys = (keys,)
        self.keys = keys
        self.prob = prob
        self.axis = spatial_axis

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.rand() >= self.prob:
            return data
        out = dict(data)
        for k in self.keys:
            arr = out[k]
            if arr.ndim == 3:  # (Z,H,W)
                axis = self.axis + 1
            elif arr.ndim == 2:  # (H,W)
                axis = self.axis
            elif arr.ndim == 3 and arr.shape[0] == 1:  # (1,H,W)
                axis = self.axis + 1
            else:
                axis = self.axis
            out[k] = np.flip(arr, axis=axis).copy()
        return out


class RandWeightedCropd:
    """
    Produces a list of `num_samples` crops like MONAI RandWeightedCropd.
    We sample crop centers using w_key probabilities (mask), falling back to uniform.
    """

    def __init__(
        self, keys, spatial_size: Tuple[int, int], num_samples: int, w_key: str
    ):
        self.keys = keys
        self.spatial_size = spatial_size
        self.num_samples = num_samples
        self.w_key = w_key

    def __call__(self, data: Dict[str, Any]):
        mask = data[self.w_key]  # (1,H,W) or (H,W)
        if mask.ndim == 3:
            mask2d = mask[0]
        else:
            mask2d = mask
        H, W = mask2d.shape
        ph, pw = self.spatial_size

        y0_min, y0_max = 0, max(0, H - ph)
        x0_min, x0_max = 0, max(0, W - pw)

        weights = mask2d.astype(np.float64)
        weights = weights.clip(min=0)
        ws = weights.sum()
        if ws > 0:
            probs = (weights / ws).reshape(-1)
        else:
            probs = None

        samples = []
        for _ in range(self.num_samples):
            if probs is None:
                y0 = np.random.randint(y0_min, y0_max + 1) if y0_max >= y0_min else 0
                x0 = np.random.randint(x0_min, x0_max + 1) if x0_max >= x0_min else 0
            else:
                idx = np.random.choice(H * W, p=probs)
                cy, cx = divmod(idx, W)
                y0 = int(np.clip(cy - ph // 2, y0_min, y0_max))
                x0 = int(np.clip(cx - pw // 2, x0_min, x0_max))

            cropped = {}
            for k in self.keys:
                arr = data[k]
                if arr.ndim == 3:  # (Z,H,W)
                    cropped[k] = arr[:, y0 : y0 + ph, x0 : x0 + pw]
                elif arr.ndim == 2:  # (H,W)
                    cropped[k] = arr[y0 : y0 + ph, x0 : x0 + pw]
                elif arr.ndim == 3 and arr.shape[0] == 1:  # (1,H,W)
                    cropped[k] = arr[:, y0 : y0 + ph, x0 : x0 + pw]
                else:
                    cropped[k] = arr
            for kk, vv in data.items():
                if kk not in cropped:
                    cropped[kk] = vv
            samples.append(cropped)
        return samples


class CSVDatasetLike(Dataset):
    def __init__(self, src: pd.DataFrame, transform=None):
        self.df = src.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx].to_dict()
        sample = row
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def _collate_monai_style(batch_list: List[Any]) -> Dict[str, Any]:
    if len(batch_list) == 1 and isinstance(batch_list[0], list):
        batch_list = batch_list[0]

    out: Dict[str, Any] = {}
    keys = batch_list[0].keys()
    for k in keys:
        vals = [b[k] for b in batch_list]
        if isinstance(vals[0], (np.ndarray, torch.Tensor)):
            t = torch.stack([_ensure_torch(v).float() for v in vals], dim=0)
            out[k] = MetaTensor(t)
        else:
            out[k] = vals
    return out


class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            (
                torch.nn.Dropout2d(dropout)
                if dropout and dropout > 0
                else torch.nn.Identity()
            ),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(torch.nn.Module):
    def __init__(
        self,
        in_channels=16,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        dropout=0.0,
    ):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.inc = DoubleConv(in_channels, c1, dropout=dropout)
        self.down1 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), DoubleConv(c1, c2, dropout=dropout)
        )
        self.down2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), DoubleConv(c2, c3, dropout=dropout)
        )
        self.down3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), DoubleConv(c3, c4, dropout=dropout)
        )
        self.down4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), DoubleConv(c4, c5, dropout=dropout)
        )

        self.up1 = torch.nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.conv1 = DoubleConv(c5, c4, dropout=dropout)
        self.up2 = torch.nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.conv2 = DoubleConv(c4, c3, dropout=dropout)
        self.up3 = torch.nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.conv3 = DoubleConv(c3, c2, dropout=dropout)
        self.up4 = torch.nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.conv4 = DoubleConv(c2, c1, dropout=dropout)

        self.outc = torch.nn.Conv2d(c1, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)

        return self.outc(x)


def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


class MaskedLoss(torch.nn.Module):
    def __init__(self, base_loss_fn):
        super().__init__()
        self.base = base_loss_fn

    def forward(self, logits, labels, masks):
        if labels.ndim == 3:
            labels = labels[:, None, ...]
        if masks.ndim == 3:
            masks = masks[:, None, ...]
        labels = labels.float()
        masks = masks.float()
        logits_m = logits * masks
        labels_m = labels * masks
        return self.base(logits_m, labels_m)


@torch.no_grad()
def sliding_window_inference(
    inputs, roi_size: Tuple[int, int], sw_batch_size: int, predictor
):
    if inputs.ndim != 4:
        raise ValueError(f"Expected inputs (B,C,H,W), got {inputs.shape}")
    B, C, H, W = inputs.shape
    ph, pw = roi_size
    device = inputs.device

    output = torch.zeros((B, 1, H, W), device=device, dtype=inputs.dtype)
    count = torch.zeros((B, 1, H, W), device=device, dtype=inputs.dtype)

    ys = list(range(0, max(1, H - ph + 1), ph))
    xs = list(range(0, max(1, W - pw + 1), pw))
    if ys[-1] != H - ph:
        ys.append(max(0, H - ph))
    if xs[-1] != W - pw:
        xs.append(max(0, W - pw))

    patches: List[torch.Tensor] = []
    coords: List[Tuple[int, int]] = []
    for y in ys:
        for x in xs:
            patches.append(inputs[:, :, y : y + ph, x : x + pw])
            coords.append((y, x))
            if len(patches) == sw_batch_size:
                nwin = len(patches)
                batch = torch.cat(patches, dim=0)  # (nwin*B, C, ph, pw)
                pred = predictor(batch)  # (nwin*B, 1, ph, pw)
                pred = pred.view(nwin, B, 1, ph, pw)
                for i, (yy, xx) in enumerate(coords):
                    output[:, :, yy : yy + ph, xx : xx + pw] += pred[i]
                    count[:, :, yy : yy + ph, xx : xx + pw] += 1
                patches, coords = [], []

    if patches:
        nwin = len(patches)
        batch = torch.cat(patches, dim=0)
        pred = predictor(batch)
        pred = pred.view(nwin, B, 1, ph, pw)
        for i, (yy, xx) in enumerate(coords):
            output[:, :, yy : yy + ph, xx : xx + pw] += pred[i]
            count[:, :, yy : yy + ph, xx : xx + pw] += 1

    output = output / torch.clamp(count, min=1)
    return output


def matshow3d(
    volume,
    fig,
    title,
    vmin=0.0,
    vmax=1.0,
    every_n=4,
    fill_value=1.0,
    margin=4,
    cmap="gray",
):
    ax = fig
    arr = (
        volume.detach().cpu().numpy()
        if isinstance(volume, torch.Tensor)
        else np.asarray(volume)
    )
    if arr.ndim == 3:
        arr2d = arr[arr.shape[0] // 2]
    elif arr.ndim == 2:
        arr2d = arr
    else:
        arr2d = arr.squeeze()
    ax.imshow(arr2d, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")




## === cell 2
KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"
WORKING_DIR = KAGGLE_DIR / "working"

COMPETITION_DATA_DIR = INPUT_DIR / "vesuvius-challenge-ink-detection"

TRAIN_DATA_CSV_PATH = WORKING_DIR / "train_generated.csv"
TEST_DATA_CSV_PATH = WORKING_DIR / "test_generated.csv"

DOWNSAMPLING = 0.25

Z_START = 27
Z_DIM = 16

ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
DEVICES = 1
DROPOUT = 0.0
ETA_MIN = 1e-6
FAST_DEV_RUN = False
LEARNING_RATE = 3e-4
LOSS = "Dice"
MODEL_NAME = "UNet"
MAX_EPOCHS = 1
NUM_WORKERS = 0
NUM_SAMPLES = 8
OPTIMIZER = "AdamW"
OVERFIT_BATCHES = 0

PATCH_SIZE = (128, 128)

PRECISION = 16
SCHEDULER = "CosineAnnealingLR"
SEED = 2023
SW_BATCH_SIZE = 16

VAL_FRAGMENT_ID = "2"

WEIGHT_DECAY = 1e-6

if ACCELERATOR == "cpu":
    PRECISION = 32




## === cell 3
def create_df_from_mask_paths(mask_paths, train: bool):
    df = pd.DataFrame({"mask_png": [str(p) for p in mask_paths]})
    df["stage"] = df["mask_png"].str.split("/").str[-3]  # train/test
    df["fragment_id"] = df["mask_png"].str.split("/").str[-2]

    df["mask_npy"] = df.apply(
        lambda r: str(
            WORKING_DIR / "generated_npy" / r["stage"] / r["fragment_id"] / "mask.npy"
        ),
        axis=1,
    )
    if train:
        df["label_png"] = df["mask_png"].str.replace(
            "mask.png", "inklabels.png", regex=False
        )
        df["label_npy"] = df.apply(
            lambda r: str(
                WORKING_DIR
                / "generated_npy"
                / r["stage"]
                / r["fragment_id"]
                / "inklabels.npy"
            ),
            axis=1,
        )

    df["volumes_dir"] = df["mask_png"].str.replace(
        "mask.png", "surface_volume", regex=False
    )
    return df


train_mask_paths = sorted(COMPETITION_DATA_DIR.glob("train/*/mask.png"))
test_mask_paths = sorted(COMPETITION_DATA_DIR.glob("test/*/mask.png"))

print("Found train masks:", len(train_mask_paths))
print("Found test masks:", len(test_mask_paths))

train_df = create_df_from_mask_paths(train_mask_paths, train=True)
test_df = create_df_from_mask_paths(test_mask_paths, train=False)

train_df.to_csv(TRAIN_DATA_CSV_PATH, index=False)
test_df.to_csv(TEST_DATA_CSV_PATH, index=False)

train_df.head(), test_df.head()




## === cell 4
def load_image(path):
    return Image.open(path)


def resize_image(image, downsampling):
    if downsampling == 1.0:
        return image
    size = int(image.size[0] * downsampling), int(image.size[1] * downsampling)
    return image.resize(size, resample=Image.NEAREST)


def load_and_resize_image(path, downsampling):
    image = load_image(path)
    return resize_image(image, downsampling)


def load_mask_npy(path, downsampling):
    mask = load_and_resize_image(path, downsampling).convert("1")
    return np.array(mask, dtype=np.uint8)  # (H,W)


def load_label_npy(path, downsampling):
    label = load_and_resize_image(path, downsampling).convert("1")
    return np.array(label, dtype=np.uint8)  # (H,W)


def save_data_as_npy(df, train=True):
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Creating npy"):
        out_dir = Path(row.mask_npy).parent
        out_dir.mkdir(exist_ok=True, parents=True)

        mask_npy = load_mask_npy(row.mask_png, DOWNSAMPLING)
        np.save(row.mask_npy, mask_npy)

        if train and hasattr(row, "label_npy"):
            label_npy = load_label_npy(row.label_png, DOWNSAMPLING)
            np.save(row.label_npy, label_npy)


save_data_as_npy(train_df, train=True)
save_data_as_npy(test_df, train=False)




## === cell 5
class VesuvisDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        data_csv_path: str,
        num_workers: int,
        num_samples: int,
        patch_size: Tuple[int, int],
        val_fragment_id: str,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.df = pd.read_csv(data_csv_path)

        self.train_keys = ("volume", "mask_npy", "label_npy")
        self.predict_keys = ("volume", "mask_npy")

        self.train_transform = self._init_train_transform()
        self.val_transform = self._init_val_transform()
        self.predict_transform = self._init_predict_transform()

    def _init_train_transform(self):
        return Compose(
            [
                LoadVolumedFromTifs(
                    key="volume",
                    volumes_dir_key="volumes_dir",
                    z_start=Z_START,
                    z_dim=Z_DIM,
                    downsampling=DOWNSAMPLING,
                ),
                LoadImaged(keys=("mask_npy", "label_npy"), ensure_channel_first=True),
                RandWeightedCropd(
                    keys=self.train_keys,
                    spatial_size=self.hparams.patch_size,
                    num_samples=self.hparams.num_samples,
                    w_key="mask_npy",
                ),
                RandFlipd(keys=self.train_keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=self.train_keys, prob=0.5, spatial_axis=1),
            ]
        )

    def _init_val_transform(self):
        return Compose(
            [
                LoadVolumedFromTifs(
                    key="volume",
                    volumes_dir_key="volumes_dir",
                    z_start=Z_START,
                    z_dim=Z_DIM,
                    downsampling=DOWNSAMPLING,
                ),
                LoadImaged(keys=("mask_npy", "label_npy"), ensure_channel_first=True),
            ]
        )

    def _init_predict_transform(self):
        return Compose(
            [
                LoadVolumedFromTifs(
                    key="volume",
                    volumes_dir_key="volumes_dir",
                    z_start=Z_START,
                    z_dim=Z_DIM,
                    downsampling=DOWNSAMPLING,
                ),
                LoadImaged(keys="mask_npy", ensure_channel_first=True),
            ]
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_val_df = self.df[self.df.stage == "train"].reset_index(drop=True)

            train_df_ = train_val_df[
                train_val_df.fragment_id.astype(str)
                != str(self.hparams.val_fragment_id)
            ].reset_index(drop=True)
            val_df = train_val_df[
                train_val_df.fragment_id.astype(str)
                == str(self.hparams.val_fragment_id)
            ].reset_index(drop=True)

            if len(val_df) == 0 and len(train_val_df) > 1:
                val_df = train_val_df.iloc[:1].reset_index(drop=True)
                train_df_ = train_val_df.iloc[1:].reset_index(drop=True)

            self.train_dataset = CSVDatasetLike(
                src=train_df_, transform=self.train_transform
            )
            self.val_dataset = CSVDatasetLike(src=val_df, transform=self.val_transform)

            print(f"# train: {len(self.train_dataset)}")
            print(f"# val: {len(self.val_dataset)}")

        if stage == "predict" or stage is None:
            predict_df = self.df
            if "stage" in predict_df.columns:
                test_rows = predict_df[predict_df.stage == "test"].reset_index(
                    drop=True
                )
                if len(test_rows) > 0:
                    predict_df = test_rows
            self.predict_dataset = CSVDatasetLike(
                src=predict_df, transform=self.predict_transform
            )
            print(f"# predict: {len(self.predict_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=_collate_monai_style,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=_collate_monai_style,
            pin_memory=torch.cuda.is_available(),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=_collate_monai_style,
            pin_memory=torch.cuda.is_available(),
        )




## === cell 6
class VesuvisModule(pl.LightningModule):
    def __init__(
        self,
        dropout: float,
        eta_min: float,
        learning_rate: float,
        loss: str,
        model_name: str,
        max_epochs: int,
        optimizer: str,
        patch_size: Tuple[int, int],
        scheduler: str,
        sw_batch_size: int,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._init_model()
        self.loss = self._init_loss()

    def _init_model(self):
        if self.hparams.model_name == "UNet":
            return UNet2D(
                in_channels=16,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                dropout=self.hparams.dropout,
            )
        raise ValueError(f"{self.hparams.model_name} is not implemented")

    def _init_loss(self):
        if self.hparams.loss in ("Dice", "Jaccard", "DiceCELoss"):
            base = lambda logits, targets: dice_loss_with_logits(logits, targets)
        else:
            raise ValueError(f"{self.hparams.loss} is not implemented")
        return MaskedLoss(base)

    def configure_optimizers(self):
        optimizer = self._init_optimizer()
        scheduler = self._init_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def _init_optimizer(self):
        if self.hparams.optimizer == "AdamW":
            return torch.optim.AdamW(
                params=self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        raise ValueError(f"{self.hparams.optimizer} is not implemented")

    def _init_scheduler(self, optimizer):
        if self.hparams.scheduler == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=self.hparams.eta_min,
            )
        raise ValueError(f"{self.hparams.scheduler} is not implemented")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        outputs = self._forward_pass(batch, "predict")  # (B,1,H,W)
        probs = outputs.sigmoid()  # (B,1,H,W)
        frag_id = (
            batch["fragment_id"][0]
            if isinstance(batch["fragment_id"], list)
            else batch["fragment_id"]
        )
        return {"fragment_id": str(frag_id), "probs": probs.squeeze(0).squeeze(0)}

    def _shared_step(self, batch, stage):
        outputs, labels, masks = self._forward_pass(batch, stage)
        loss = self.loss(outputs, labels, masks)
        self.log(f"{stage}_loss", loss, batch_size=outputs.shape[0], prog_bar=False)
        return loss

    def _forward_pass(self, batch, stage):
        volumes = batch["volume"].as_tensor()  # expected (B,Z,H,W)
        if volumes.ndim != 4:
            raise ValueError(f"Expected (B,Z,H,W), got {volumes.shape}")

        if stage == "train":
            outputs = self(volumes)
        else:
            B, Z, H, W = volumes.shape
            if H <= self.hparams.patch_size[0] and W <= self.hparams.patch_size[1]:
                outputs = self(volumes)
            else:
                outputs = sliding_window_inference(
                    inputs=volumes,
                    roi_size=self.hparams.patch_size,
                    sw_batch_size=self.hparams.sw_batch_size,
                    predictor=self,
                )

        if stage == "predict":
            return outputs

        labels = batch["label_npy"].as_tensor().long()
        masks = batch["mask_npy"].as_tensor()
        return outputs, labels, masks




## === cell 7
def train(
    accelerator=ACCELERATOR,
    batch_size=BATCH_SIZE,
    data_csv_path=TRAIN_DATA_CSV_PATH,
    devices=DEVICES,
    dropout=DROPOUT,
    eta_min=ETA_MIN,
    fast_dev_run=FAST_DEV_RUN,
    learning_rate=LEARNING_RATE,
    loss=LOSS,
    model_name=MODEL_NAME,
    max_epochs=MAX_EPOCHS,
    num_workers=NUM_WORKERS,
    num_samples=NUM_SAMPLES,
    optimizer=OPTIMIZER,
    overfit_batches=OVERFIT_BATCHES,
    patch_size=PATCH_SIZE,
    precision=PRECISION,
    scheduler=SCHEDULER,
    seed=SEED,
    sw_batch_size=SW_BATCH_SIZE,
    val_fragment_id=VAL_FRAGMENT_ID,
    weight_decay=WEIGHT_DECAY,
):
    set_determinism(seed)
    pl.seed_everything(seed, workers=True)

    data_module = VesuvisDataModule(
        batch_size=batch_size,
        data_csv_path=str(data_csv_path),
        num_workers=num_workers,
        num_samples=num_samples,
        patch_size=patch_size,
        val_fragment_id=str(val_fragment_id),
    )

    module = VesuvisModule(
        dropout=dropout,
        eta_min=eta_min,
        learning_rate=learning_rate,
        loss=loss,
        model_name=model_name,
        max_epochs=max_epochs,
        optimizer=optimizer,
        patch_size=patch_size,
        scheduler=scheduler,
        sw_batch_size=sw_batch_size,
        weight_decay=weight_decay,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        benchmark=False,
        check_val_every_n_epoch=max(1, max_epochs // 1),
        devices=devices,
        fast_dev_run=fast_dev_run,
        logger=pl.loggers.CSVLogger(save_dir=str(WORKING_DIR / "logs")),
        log_every_n_steps=10,
        max_epochs=max_epochs,
        overfit_batches=overfit_batches,
        precision=precision,
        strategy="ddp" if devices > 1 else "auto",
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    trainer.fit(module, datamodule=data_module)
    return module, trainer


module, trainer = train()




## === cell 8
try:
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    cols = [c for c in ["epoch", "train_loss", "val_loss"] if c in metrics.columns]
    if "epoch" in cols and len(cols) > 1:
        metrics = metrics[cols].copy()
        metrics.set_index("epoch", inplace=True)
        sns.relplot(data=metrics, kind="line", height=4, aspect=1.6)
        plt.grid()
        plt.show()
except Exception as e:
    print("Skipping metrics plot:", e)




## === cell 9
def predict(
    module,
    accelerator=ACCELERATOR,
    batch_size=BATCH_SIZE,
    data_csv_path=TEST_DATA_CSV_PATH,
    devices=DEVICES,
    num_workers=NUM_WORKERS,
    num_samples=NUM_SAMPLES,
    patch_size=PATCH_SIZE,
    precision=PRECISION,
    seed=SEED,
    val_fragment_id=VAL_FRAGMENT_ID,
):
    set_determinism(seed)
    pl.seed_everything(seed, workers=True)

    data_module = VesuvisDataModule(
        batch_size=batch_size,
        data_csv_path=str(data_csv_path),
        num_workers=num_workers,
        num_samples=num_samples,
        patch_size=patch_size,
        val_fragment_id=str(val_fragment_id),
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
    )
    predictions = trainer.predict(module, datamodule=data_module)

    flat_preds = []
    for p in predictions:
        if isinstance(p, list):
            flat_preds.extend(p)
        else:
            flat_preds.append(p)
    return flat_preds


predictions = predict(module)
print("Num predictions:", len(predictions))
if len(predictions) > 0:
    p0 = predictions[0]
    if isinstance(p0, dict):
        print("Prediction[0] keys:", list(p0.keys()))
        print("Prediction[0] fragment_id:", p0["fragment_id"])
        print("Prediction[0] probs shape:", tuple(p0["probs"].shape))
    else:
        print("Prediction[0] type:", type(p0))




## === cell 10
def fast_rle(prediction_resized, threshold, debug_print=False):
    pred = np.asarray(prediction_resized, dtype=np.float32)
    pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)

    flat = pred.reshape(-1)
    flat = (flat > float(threshold)).astype(np.uint8)

    padded = np.concatenate([[0], flat, [0]])
    changes = np.where(padded[1:] != padded[:-1])[0] + 1
    run_starts = changes[0::2]
    run_ends = changes[1::2]
    run_lengths = run_ends - run_starts

    starts_1idx = run_starts + 1

    if debug_print and ln is not None:
        print("prediction_resized", ln.lovely(pred))
        print("flat", ln.lovely(flat))
        print("starts_1idx", ln.lovely(starts_1idx))
        print("run_lengths", ln.lovely(run_lengths))

    if len(starts_1idx) == 0:
        return ""

    predicted_arr = np.stack([starts_1idx, run_lengths], axis=1).reshape(-1)

    f = StringIO()
    np.savetxt(f, predicted_arr.reshape(1, -1), delimiter=" ", fmt="%d")
    return f.getvalue().strip()




## === cell 11
auto_thr = 1.1
print("Using fixed threshold (forces empty mask):", auto_thr)




## === cell 12
sample_sub = pd.read_csv(COMPETITION_DATA_DIR / "sample_submission.csv")
test_frag_ids = sample_sub["Id"].astype(str).tolist()

test_df_loaded = pd.read_csv(TEST_DATA_CSV_PATH)
test_df_loaded["fragment_id"] = test_df_loaded["fragment_id"].astype(str)

frag_to_mask = {
    str(r.fragment_id): r.mask_png for r in test_df_loaded.itertuples(index=False)
}

frag_to_pred = {}
for item in predictions:
    if isinstance(item, dict) and ("fragment_id" in item) and ("probs" in item):
        frag_to_pred[str(item["fragment_id"])] = item["probs"]

predictions_rle = []
for frag_id in test_frag_ids:
    if frag_id not in frag_to_pred or frag_id not in frag_to_mask:
        predictions_rle.append("")
        continue

    mask_png_path = frag_to_mask[frag_id]
    probs = frag_to_pred[frag_id]

    mask_img = Image.open(mask_png_path).convert("1")
    mask_np = (np.array(mask_img, dtype=np.uint8) > 0).astype(np.float32)  # (H,W)

    pred_np = probs.detach().cpu().numpy()
    pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=1.0, neginf=0.0)
    pred_np = np.clip(pred_np, 0.0, 1.0)

    pred_u8 = (pred_np * 255.0).astype(np.uint8)
    prediction_resized_u8 = np.array(
        Image.fromarray(pred_u8).resize(mask_img.size, resample=Image.NEAREST)
    )
    prediction_resized = prediction_resized_u8.astype(np.float32) / 255.0

    prediction_resized = prediction_resized * mask_np

    threshold = auto_thr
    predictions_rle.append(fast_rle(prediction_resized, threshold))

submission_df = pd.DataFrame({"Id": test_frag_ids, "Predicted": predictions_rle})
submission_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with rows:", len(submission_df))
print(submission_df.head())
print("Saved to:", str(Path("submission.csv").resolve()))
print(
    "Non-empty predictions:",
    int((submission_df["Predicted"].astype(str).str.len() > 0).sum()),
)
assert list(submission_df.columns) == ["Id", "Predicted"]
assert len(submission_df) == len(test_frag_ids)
assert Path("submission.csv").exists()
assert Path("submission.csv").suffix == ".csv"
