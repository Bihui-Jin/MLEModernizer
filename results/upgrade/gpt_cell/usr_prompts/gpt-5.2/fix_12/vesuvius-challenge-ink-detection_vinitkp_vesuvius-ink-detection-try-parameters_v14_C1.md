# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.11

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

_CANDIDATE_ROOTS = [
    "/kaggle/input/vesuvius-challenge-ink-detection/",
    "/kaggle/data/vesuvius-challenge-ink-detection/",
    "/kaggle/input/vesuvius-challenge/",
    "/kaggle/data/vesuvius-challenge/",
]
DATA_ROOT = next(
    (p for p in _CANDIDATE_ROOTS if os.path.exists(p)), _CANDIDATE_ROOTS[0]
)
if not DATA_ROOT.endswith("/"):
    DATA_ROOT += "/"

BUFFER = 30  # Buffer size in x and y direction
Z_START = 27  # First slice in the z direction to use
Z_DIM = 10  # Number of slices in the z direction
TRAINING_STEPS = 20000
LEARNING_RATE = 0.05
BATCH_SIZE = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WORKERS = min(4, os.cpu_count() or 1)

rect = (1100, 3500, 700, 950)

print("DATA_ROOT =", DATA_ROOT)
print("DEVICE =", DEVICE)
print("NUM_WORKERS =", NUM_WORKERS)




## === cell 1
def load_image_stack(
    fragment_dir: str,
    z_start: int = Z_START,
    z_dim: int = Z_DIM,
    device: torch.device | None = None,
) -> torch.Tensor:
    tif_files = sorted(glob.glob(os.path.join(fragment_dir, "surface_volume", "*.tif")))
    if len(tif_files) == 0:
        raise FileNotFoundError(
            f"No .tif files found under {fragment_dir}/surface_volume/"
        )
    tif_files = tif_files[z_start : z_start + z_dim]
    if len(tif_files) != z_dim:
        raise ValueError(
            f"Expected {z_dim} slices but found {len(tif_files)} in {fragment_dir}"
        )
    images = [np.array(Image.open(fn), dtype=np.float32) / 65535.0 for fn in tif_files]
    stack = torch.stack([torch.from_numpy(im) for im in images], dim=0)  # (Z,H,W)
    if device is not None:
        stack = stack.to(device)
    return stack


def load_mask(fragment_dir: str) -> np.ndarray:
    mpath = os.path.join(fragment_dir, "mask.png")
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"Missing mask.png at {mpath}")
    return np.array(Image.open(mpath).convert("1"))


def load_label(fragment_dir: str, device: torch.device | None = None) -> torch.Tensor:
    lpath = os.path.join(fragment_dir, "inklabels.png")
    if not os.path.exists(lpath):
        raise FileNotFoundError(f"Missing inklabels.png at {lpath}")
    t = torch.from_numpy(np.array(Image.open(lpath))).gt(0).float()
    if device is not None:
        t = t.to(device)
    return t


PREFIX = os.path.join(DATA_ROOT, "train", "1") + "/"
_ir_path = os.path.join(PREFIX, "ir.png")
if os.path.exists(_ir_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(Image.open(_ir_path), cmap="gray")
    plt.title("train/1 ir.png")
    plt.axis("off")
    plt.show()



## === cell 2
mask = load_mask(PREFIX)
label = load_label(PREFIX, device=None)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title("mask.png")
ax1.imshow(mask, cmap="gray")
ax1.axis("off")
ax2.set_title("inklabels.png")
ax2.imshow(label, cmap="gray")
ax2.axis("off")
plt.tight_layout()
plt.show()



## === cell 3
image_stack = load_image_stack(PREFIX, Z_START, Z_DIM, device=None)

images_np = [image_stack[z].detach().cpu().numpy() for z in range(image_stack.shape[0])]
fig, axes = plt.subplots(1, len(images_np), figsize=(15, 3))
for image, ax in zip(images_np, axes):
    im_small = np.array(
        Image.fromarray(image).resize((image.shape[1] // 20, image.shape[0] // 20))
    )
    ax.imshow(im_small, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.show()



## === cell 4
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(label.cpu(), cmap="gray")
patch = patches.Rectangle(
    (rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor="r", facecolor="none"
)
ax.add_patch(patch)
ax.set_title("train/1 label with rect")
ax.axis("off")
plt.show()




## === cell 5
def extract_all_patches_2d(
    image_stack: torch.Tensor, buffer: int = BUFFER
) -> torch.Tensor:
    """
    image_stack: (Z,H,W) float32 (on DEVICE)
    returns patches: (H_valid, W_valid, Z, K, K) on same device
    where K = 2*buffer+1 and (H_valid = H-2*buffer, W_valid = W-2*buffer)
    """
    Z, H, W = image_stack.shape
    K = buffer * 2 + 1
    patches = image_stack.unfold(1, K, 1).unfold(2, K, 1)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    return patches


class PatchDataset(data.Dataset):
    """
    Dataset returning (subvolume, inklabel) for a list of pixels.
    Uses precomputed patches to avoid expensive Python slicing.
    """

    def __init__(
        self,
        patches_hwzkk: torch.Tensor,
        label: torch.Tensor,
        pixels: np.ndarray,
        buffer: int = BUFFER,
    ):
        self.patches = patches_hwzkk  # (H_valid,W_valid,Z,K,K) on DEVICE
        self.label = label  # (H,W) on DEVICE (for training) or CPU dummy
        self.pixels = pixels  # Nx2 of (y,x) in full coords (CPU numpy)
        self.buffer = buffer

    def __len__(self):
        return int(self.pixels.shape[0])

    def __getitem__(self, index):
        y, x = self.pixels[index]
        yy = int(y) - self.buffer
        xx = int(x) - self.buffer
        subvolume = self.patches[yy, xx].unsqueeze(0)
        inklabel = self.label[int(y), int(x)].view(1)
        return subvolume, inklabel


model = nn.Sequential(
    nn.Conv3d(1, 16, 3, 1, 1),
    nn.MaxPool3d(2, 2),
    nn.Conv3d(16, 32, 3, 1, 1),
    nn.MaxPool3d(2, 2),
    nn.Conv3d(32, 64, 3, 1, 1),
    nn.MaxPool3d(2, 2),
    nn.Flatten(start_dim=1),
    nn.LazyLinear(128),
    nn.ReLU(),
    nn.LazyLinear(1),
    nn.Sigmoid(),
).to(DEVICE)




## === cell 6
def extract_all_patches_2d(
    image_stack: torch.Tensor, buffer: int = BUFFER
) -> torch.Tensor:
    """
    Return a lightweight handle for patch extraction.
    Previously returned a huge (H_valid,W_valid,Z,K,K) tensor; now we keep the original
    (Z,H,W) stack and let PatchDataset slice patches on-the-fly to avoid OOM.
    """
    return image_stack


class PatchDataset(data.Dataset):
    """
    Dataset returning (subvolume, inklabel) for a list of pixels.
    Extracts patches lazily from the provided (Z,H,W) image_stack to avoid OOM.
    """

    def __init__(
        self,
        patches_hwzkk: torch.Tensor,
        label: torch.Tensor,
        pixels: np.ndarray,
        buffer: int = BUFFER,
    ):
        self.stack = patches_hwzkk
        self.label = label  # (H,W) on DEVICE (for training) or CPU dummy
        self.pixels = pixels  # Nx2 of (y,x) in full coords (CPU numpy)
        self.buffer = buffer

    def __len__(self):
        return int(self.pixels.shape[0])

    def __getitem__(self, index):
        y, x = self.pixels[index]
        y = int(y)
        x = int(x)
        b = self.buffer
        subvolume = self.stack[:, y - b : y + b + 1, x - b : x + b + 1].unsqueeze(0)
        inklabel = self.label[y, x].view(1)
        return subvolume, inklabel


def build_pixels_for_training(fragment_dir: str) -> np.ndarray:
    m = load_mask(fragment_dir)
    nb = np.zeros(m.shape, dtype=bool)
    nb[BUFFER : m.shape[0] - BUFFER, BUFFER : m.shape[1] - BUFFER] = True
    arr_mask = np.array(m) * nb
    return np.argwhere(arr_mask)


train_fragment_dirs = [
    os.path.join(DATA_ROOT, "train", "1"),
    os.path.join(DATA_ROOT, "train", "2"),
]

print("Generating pixel lists (train/1 + train/2)...")
train_datasets = []
for fdir in train_fragment_dirs:
    stack = load_image_stack(fdir, Z_START, Z_DIM, device=DEVICE)
    lbl = load_label(fdir, device=DEVICE)
    pixels = build_pixels_for_training(fdir)

    patches = extract_all_patches_2d(stack, BUFFER)

    train_datasets.append(PatchDataset(patches, lbl, pixels, buffer=BUFFER))

train_dataset = data.ConcatDataset(train_datasets)

_NUM_WORKERS_SAFE = 0 if DEVICE.type == "cuda" else NUM_WORKERS
_PERSISTENT_WORKERS_SAFE = _NUM_WORKERS_SAFE > 0

train_loader = data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
    num_workers=_NUM_WORKERS_SAFE,
    persistent_workers=_PERSISTENT_WORKERS_SAFE,
)

print("Training...")
criterion = nn.BCELoss()
optimizer = optim.ASGD(model.parameters(), lr=LEARNING_RATE)

scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=2)
scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
lmbda = lambda epoch: 0.95
scheduler4 = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
scheduler5 = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.5, total_iters=4
)
scheduler6 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=30, T_mult=1
)
scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [scheduler1, scheduler2, scheduler3, scheduler4, scheduler5, scheduler6]
)

model.train()
for i, (subvolumes, inklabels) in tqdm(enumerate(train_loader), total=TRAINING_STEPS):
    if i >= TRAINING_STEPS:
        break
    optimizer.zero_grad(set_to_none=True)
    outputs = model(subvolumes)
    loss = criterion(outputs, inklabels)
    loss.backward()
    optimizer.step()
    scheduler.step()


## --- ERROR in cell 6, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1251551731.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    101[0m [0;34m[0m[0m
[1;32m    102[0m [0mmodel[0m[0;34m.[0m[0mtrain[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 103[0;31m [0;32mfor[0m [0mi[0m[0;34m,[0m [0;34m([0m[0msubvolumes[0m[0;34m,[0m [0minklabels[0m[0;34m)[0m [0;32min[0m [0mtqdm[0m[0;34m([0m[0menumerate[0m[0;34m([0m[0mtrain_loader[0m[0;34m)[0m[0;34m,[0m [0mtotal[0m[0;34m=[0m[0mTRAINING_STEPS[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    104[0m     [0;32mif[0m [0mi[0m [0;34m>=[0m [0mTRAINING_STEPS[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    105[0m         [0;32mbreak[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tqdm/std.py[0m in [0;36m__iter__[0;34m(self)[0m
[1;32m   1179[0m [0;34m[0m[0m
[1;32m   1180[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1181[0;31m             [0;32mfor[0m [0mobj[0m [0;32min[0m [0miterable[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1182[0m                 [0;32myield[0m [0mobj[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1183[0m                 [0;31m# Update and possibly print the progressbar.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m    706[0m                 [0;31m# TODO(https://github.com/pytorch/pytorch/issues/76750)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    707[0m                 [0mself[0m[0;34m.[0m[0m_reset[0m[0;34m([0m[0;34m)[0m  [0;31m# type: ignore[call-arg][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 708[0;31m             [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_next_data[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    709[0m             [0mself[0m[0;34m.[0m[0m_num_yielded[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m    710[0m             if (

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_next_data[0;34m(self)[0m
[1;32m    764[0m         [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_dataset_fetcher[0m[0;34m.[0m[0mfetch[0m[0;34m([0m[0mindex[0m[0;34m)[0m  [0;31m# may raise StopIteration[0m[0;34m[0m[0;34m[0m[0m
[1;32m    765[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0m_pin_memory[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 766[0;31m             [0mdata[0m [0;34m=[0m [0m_utils[0m[0;34m.[0m[0mpin_memory[0m[0;34m.[0m[0mpin_memory[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_pin_memory_device[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    767[0m         [0;32mreturn[0m [0mdata[0m[0;34m[0m[0;34m[0m[0m
[1;32m    768[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py[0m in [0;36mpin_memory[0;34m(data, device)[0m
[1;32m     96[0m                 [0mclone[0m [0;34m=[0m [0mcopy[0m[0;34m.[0m[0mcopy[0m[0;34m([0m[0mdata[0m[0;34m)[0m  [0;31m# type: ignore[arg-type][0m[0;34m[0m[0;34m[0m[0m
[1;32m     97[0m                 [0;32mfor[0m [0mi[0m[0;34m,[0m [0mitem[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 98[0;31m                     [0mclone[0m[0;34m[[0m[0mi[0m[0;34m][0m [0;34m=[0m [0mpin_memory[0m[0;34m([0m[0mitem[0m[0;34m,[0m [0mdevice[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     99[0m                 [0;32mreturn[0m [0mclone[0m[0;34m[0m[0;34m[0m[0m
[1;32m    100[0m             [0;32mreturn[0m [0mtype[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m([0m[0;34m[[0m[0mpin_memory[0m[0;34m([0m[0msample[0m[0;34m,[0m [0mdevice[0m[0;34m)[0m [0;32mfor[0m [0msample[0m [0;32min[0m [0mdata[0m[0;34m][0m[0;34m)[0m  [0;31m# type: ignore[call-arg][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py[0m in [0;36mpin_memory[0;34m(data, device)[0m
[1;32m     62[0m [0;32mdef[0m [0mpin_memory[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mdevice[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     63[0m     [0;32mif[0m [0misinstance[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mtorch[0m[0;34m.[0m[0mTensor[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 64[0;31m         [0;32mreturn[0m [0mdata[0m[0;34m.[0m[0mpin_memory[0m[0;34m([0m[0mdevice[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     65[0m     [0;32melif[0m [0misinstance[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0;34m([0m[0mstr[0m[0;34m,[0m [0mbytes[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     66[0m         [0;32mreturn[0m [0mdata[0m[0;34m[0m[0;34m[0m[0m

[0;31mRuntimeError[0m: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned

## === cell 7
print("Evaluating on train/1 inside rect (sanity-check)...")
m1 = load_mask(os.path.join(DATA_ROOT, "train", "1"))
lbl1_cpu = load_label(os.path.join(DATA_ROOT, "train", "1"), device=None)
stack1 = load_image_stack(
    os.path.join(DATA_ROOT, "train", "1"), Z_START, Z_DIM, device=DEVICE
)

not_border = np.zeros(m1.shape, dtype=bool)
not_border[BUFFER : m1.shape[0] - BUFFER, BUFFER : m1.shape[1] - BUFFER] = True
arr_mask = np.array(m1).astype(bool) & not_border

inside_rect = np.zeros(m1.shape, dtype=bool)
inside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = True
inside_rect = inside_rect & arr_mask

pixels_inside_rect = np.argwhere(inside_rect)

patches1 = extract_all_patches_2d(stack1, BUFFER)
dummy_label = torch.zeros(m1.shape, dtype=torch.float32, device=DEVICE)
eval_dataset = PatchDataset(patches1, dummy_label, pixels_inside_rect, buffer=BUFFER)

eval_loader = data.DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=NUM_WORKERS,
    persistent_workers=(NUM_WORKERS > 0),
)

output = torch.zeros_like(lbl1_cpu).float()
flat_output = output.view(-1)
idx_flat = torch.from_numpy(
    (pixels_inside_rect[:, 0] * m1.shape[1] + pixels_inside_rect[:, 1]).astype(np.int64)
)

model.eval()
with torch.no_grad():
    offset = 0
    for subvolumes, _ in tqdm(eval_loader):
        preds = model(subvolumes).view(-1).detach().cpu()
        end = offset + preds.numel()
        flat_output[idx_flat[offset:end]] = preds
        offset = end

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(output.cpu(), cmap="gray")
ax1.set_title("train/1 output (inside rect)")
ax1.axis("off")
ax2.imshow(lbl1_cpu.cpu(), cmap="gray")
ax2.set_title("train/1 label")
ax2.axis("off")
plt.tight_layout()
plt.show()
