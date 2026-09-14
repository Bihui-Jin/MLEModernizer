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

_STACK_CPU_CACHE: dict[tuple[str, int, int], torch.Tensor] = {}
_MASK_CACHE: dict[str, np.ndarray] = {}
_LABEL_CPU_CACHE: dict[str, torch.Tensor] = {}

USE_GPU = DEVICE.type == "cuda"




## === cell 1


def load_image_stack(
    fragment_dir: str,
    z_start: int = Z_START,
    z_dim: int = Z_DIM,
    device: torch.device | None = None,
) -> torch.Tensor:
    key = (fragment_dir, int(z_start), int(z_dim))
    stack_cpu = _STACK_CPU_CACHE.get(key)
    if stack_cpu is None:
        tif_files = sorted(
            glob.glob(os.path.join(fragment_dir, "surface_volume", "*.tif"))
        )
        if len(tif_files) == 0:
            raise FileNotFoundError(
                f"No .tif files found under {fragment_dir}/surface_volume/"
            )
        tif_files = tif_files[z_start : z_start + z_dim]
        if len(tif_files) != z_dim:
            raise ValueError(
                f"Expected {z_dim} slices but found {len(tif_files)} in {fragment_dir}"
            )

        imgs = []
        for fn in tif_files:
            imgs.append(np.asarray(Image.open(fn), dtype=np.float32) / 65535.0)
        stack_cpu = torch.from_numpy(np.stack(imgs, axis=0))  # (Z,H,W) float32 CPU
        _STACK_CPU_CACHE[key] = stack_cpu

    if device is not None:
        return stack_cpu.to(device, non_blocking=True)
    return stack_cpu


def load_mask(fragment_dir: str) -> np.ndarray:
    m = _MASK_CACHE.get(fragment_dir)
    if m is None:
        mpath = os.path.join(fragment_dir, "mask.png")
        if not os.path.exists(mpath):
            raise FileNotFoundError(f"Missing mask.png at {mpath}")
        m = np.array(Image.open(mpath).convert("1"))
        _MASK_CACHE[fragment_dir] = m
    return m


def load_label(fragment_dir: str, device: torch.device | None = None) -> torch.Tensor:
    t_cpu = _LABEL_CPU_CACHE.get(fragment_dir)
    if t_cpu is None:
        lpath = os.path.join(fragment_dir, "inklabels.png")
        if not os.path.exists(lpath):
            raise FileNotFoundError(f"Missing inklabels.png at {lpath}")
        t_cpu = torch.from_numpy(np.array(Image.open(lpath))).gt(0).float()
        _LABEL_CPU_CACHE[fragment_dir] = t_cpu
    if device is not None:
        return t_cpu.to(device, non_blocking=True)
    return t_cpu


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
    image_stack: (Z,H,W) float32 CPU or GPU
    returns patches view: (H_valid, W_valid, Z, K, K) on same device/dtype
    """
    Z, H, W = image_stack.shape
    K = buffer * 2 + 1
    patches = image_stack.unfold(1, K, 1).unfold(2, K, 1).permute(1, 2, 0, 3, 4)
    return patches.contiguous()


class PatchDataset(data.Dataset):
    """
    Dataset for ALL valid pixels in a fragment in row-major order without storing Nx2 coords.

    patches: (H_valid,W_valid,Z,K,K)
    label: (H,W) float32 (CPU or GPU)
    mask_valid: (H_valid,W_valid) bool (CPU)
    """

    def __init__(
        self,
        patches_hwzkk: torch.Tensor,
        label: torch.Tensor,
        mask_valid_hw: np.ndarray,
        buffer: int = BUFFER,
    ):
        self.patches = patches_hwzkk
        self.label = label
        self.mask_valid = mask_valid_hw  # bool, shape (H_valid,W_valid)
        self.buffer = buffer

        self.valid_flat = np.flatnonzero(self.mask_valid.reshape(-1)).astype(np.int64)
        self.Hv, self.Wv = self.mask_valid.shape

    def __len__(self):
        return int(self.valid_flat.shape[0])

    def __getitem__(self, index):
        flat = int(self.valid_flat[index])
        yy = flat // self.Wv  # in [0..H_valid-1]
        xx = flat - yy * self.Wv

        subvolume = self.patches[yy, xx].unsqueeze(0)  # (1,Z,K,K)
        y = yy + self.buffer
        x = xx + self.buffer
        inklabel = self.label[y, x].view(1)
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


def build_valid_mask_hw(fragment_dir: str) -> np.ndarray:
    m = load_mask(fragment_dir).astype(bool)
    H, W = m.shape
    not_border = np.zeros((H, W), dtype=bool)
    not_border[BUFFER : H - BUFFER, BUFFER : W - BUFFER] = True
    valid = m & not_border
    return valid[BUFFER : H - BUFFER, BUFFER : W - BUFFER]


train_fragment_dirs = [
    os.path.join(DATA_ROOT, "train", "1"),
    os.path.join(DATA_ROOT, "train", "2"),
]

print("Preparing train datasets (train/1 + train/2)...")
train_datasets = []
for fdir in train_fragment_dirs:
    stack_cpu = load_image_stack(fdir, Z_START, Z_DIM, device=None)
    lbl_cpu = load_label(fdir, device=None)

    patches_cpu = extract_all_patches_2d(stack_cpu, BUFFER)  # CPU patches view
    valid_hw = build_valid_mask_hw(fdir)
    train_datasets.append(PatchDataset(patches_cpu, lbl_cpu, valid_hw, buffer=BUFFER))

train_dataset = data.ConcatDataset(train_datasets)

_NUM_WORKERS_SAFE = NUM_WORKERS
_PERSISTENT_WORKERS_SAFE = _NUM_WORKERS_SAFE > 0
_PIN_MEMORY_SAFE = torch.cuda.is_available()

train_loader = data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=_PIN_MEMORY_SAFE,
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
it = iter(train_loader)
for i in tqdm(range(TRAINING_STEPS), total=TRAINING_STEPS):
    try:
        subvolumes, inklabels = next(it)
    except StopIteration:
        it = iter(train_loader)
        subvolumes, inklabels = next(it)

    if USE_GPU:
        subvolumes = subvolumes.to(DEVICE, non_blocking=True)
        inklabels = inklabels.to(DEVICE, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(subvolumes)
    loss = criterion(outputs, inklabels)
    loss.backward()
    optimizer.step()
    scheduler.step()




## --- ERROR in cell 6, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2680358116.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     25[0m     [0mlbl_cpu[0m [0;34m=[0m [0mload_label[0m[0;34m([0m[0mfdir[0m[0;34m,[0m [0mdevice[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     26[0m [0;34m[0m[0m
[0;32m---> 27[0;31m     [0mpatches_cpu[0m [0;34m=[0m [0mextract_all_patches_2d[0m[0;34m([0m[0mstack_cpu[0m[0;34m,[0m [0mBUFFER[0m[0;34m)[0m  [0;31m# CPU patches view[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     28[0m     [0mvalid_hw[0m [0;34m=[0m [0mbuild_valid_mask_hw[0m[0;34m([0m[0mfdir[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     29[0m     [0mtrain_datasets[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mPatchDataset[0m[0;34m([0m[0mpatches_cpu[0m[0;34m,[0m [0mlbl_cpu[0m[0;34m,[0m [0mvalid_hw[0m[0;34m,[0m [0mbuffer[0m[0;34m=[0m[0mBUFFER[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2062436954.py[0m in [0;36mextract_all_patches_2d[0;34m(image_stack, buffer)[0m
[1;32m     16[0m     [0mpatches[0m [0;34m=[0m [0mimage_stack[0m[0;34m.[0m[0munfold[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0mK[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m.[0m[0munfold[0m[0;34m([0m[0;36m2[0m[0;34m,[0m [0mK[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m.[0m[0mpermute[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0;36m2[0m[0;34m,[0m [0;36m0[0m[0;34m,[0m [0;36m3[0m[0;34m,[0m [0;36m4[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     17[0m     [0;31m# Keep as contiguous for faster indexed gathers (one-time cost).[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 18[0;31m     [0;32mreturn[0m [0mpatches[0m[0;34m.[0m[0mcontiguous[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     19[0m [0;34m[0m[0m
[1;32m     20[0m [0;34m[0m[0m

[0;31mRuntimeError[0m: [enforce fail at alloc_cpu.cpp:118] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 7578734842800 bytes. Error code 12 (Cannot allocate memory)

## === cell 7
print("Evaluating on train/1 inside rect (sanity-check)...")
train1_dir = os.path.join(DATA_ROOT, "train", "1")
m1 = load_mask(train1_dir)
lbl1_cpu = load_label(train1_dir, device=None)
stack1_cpu = load_image_stack(train1_dir, Z_START, Z_DIM, device=None)
patches1_cpu = extract_all_patches_2d(stack1_cpu, BUFFER)

H, W = m1.shape
not_border = np.zeros((H, W), dtype=bool)
not_border[BUFFER : H - BUFFER, BUFFER : W - BUFFER] = True
arr_mask = m1.astype(bool) & not_border

inside_rect = np.zeros((H, W), dtype=bool)
inside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = True
inside_rect = inside_rect & arr_mask

inside_rect_hw = inside_rect[BUFFER : H - BUFFER, BUFFER : W - BUFFER]

dummy_label = torch.zeros((H, W), dtype=torch.float32)  # CPU
eval_dataset = PatchDataset(patches1_cpu, dummy_label, inside_rect_hw, buffer=BUFFER)

eval_loader = data.DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=_PIN_MEMORY_SAFE,
    num_workers=_NUM_WORKERS_SAFE,
    persistent_workers=_PERSISTENT_WORKERS_SAFE,
)

output = torch.zeros_like(lbl1_cpu).float()
flat_output = output.view(-1)

Hv, Wv = inside_rect_hw.shape
valid_flat = eval_dataset.valid_flat
yy = valid_flat // Wv
xx = valid_flat - yy * Wv
y_full = yy + BUFFER
x_full = xx + BUFFER
idx_flat = torch.from_numpy((y_full * W + x_full).astype(np.int64))

model.eval()
with torch.no_grad():
    offset = 0
    for subvolumes, _ in tqdm(eval_loader, leave=False):
        if USE_GPU:
            subvolumes = subvolumes.to(DEVICE, non_blocking=True)
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
