# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import glob
import math
import random
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


def resolve_base_dir():
    candidates = [
        "/kaggle/input/vesuvius-challenge-ink-detection",
        "/kaggle/data/vesuvius-challenge-ink-detection",
        "/kaggle/input/vesuvius-challenge",
        "/kaggle/data/vesuvius-challenge",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    for root in ["/kaggle/input", "/kaggle/data"]:
        if os.path.isdir(root):
            for name in os.listdir(root):
                p = os.path.join(root, name)
                if os.path.isdir(p) and os.path.isfile(
                    os.path.join(p, "sample_submission.csv")
                ):
                    return p
    raise FileNotFoundError(
        "Could not resolve Kaggle dataset directory containing sample_submission.csv"
    )


def seed_everything(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(1234)

BASE_DIR = resolve_base_dir()
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

PREFIX = os.path.join(TRAIN_DIR, "1") + "/"

BUFFER = 30  # Buffer size in x and y direction
Z_START = 27  # First slice in the z direction to use
Z_DIM = 10  # Number of slices in the z direction
TRAINING_STEPS = 20000
LEARNING_RATE = 0.05
BATCH_SIZE = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CPU_COUNT = os.cpu_count() or 2

DL_WORKERS = min(4, max(1, (CPU_COUNT // 2)))

THRESHOLD = 0.4

print("BASE_DIR:", BASE_DIR)
print("PREFIX:", PREFIX)
print("DEVICE:", DEVICE)
print("DL_WORKERS:", DL_WORKERS)

ir_path = os.path.join(PREFIX, "ir.png")
if os.path.isfile(ir_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(Image.open(ir_path), cmap="gray")
    plt.title("ir.png")
    plt.axis("off")
    plt.show()
else:
    print("Warning: ir.png not found at", ir_path)


def seed_worker(worker_id: int):
    worker_seed = 1234 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


_DL_GENERATOR = torch.Generator()
_DL_GENERATOR.manual_seed(1234)

_IO_EX = ThreadPoolExecutor(max_workers=min(8, CPU_COUNT))


def _read_tif_norm(fn: str) -> np.ndarray:
    return np.array(Image.open(fn), dtype=np.float32) / 65535.0




## === cell 1
mask_path = os.path.join(PREFIX, "mask.png")
label_path = os.path.join(PREFIX, "inklabels.png")

mask = np.array(Image.open(mask_path).convert("1"), dtype=bool)
label = torch.from_numpy(np.array(Image.open(label_path), dtype=np.uint8)).gt(0).float()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title("mask.png")
ax1.imshow(mask, cmap="gray")
ax1.axis("off")
ax2.set_title("inklabels.png")
ax2.imshow(label.numpy(), cmap="gray")
ax2.axis("off")
plt.tight_layout()
plt.show()

print(
    "mask shape:",
    mask.shape,
    "label shape:",
    tuple(label.shape),
    "label device:",
    label.device,
)




## === cell 2
slice_files = sorted(glob.glob(os.path.join(PREFIX, "surface_volume", "*.tif")))
if len(slice_files) == 0:
    raise FileNotFoundError(
        f"No .tif slices found under {os.path.join(PREFIX, 'surface_volume')}"
    )

z0 = max(0, min(Z_START, len(slice_files) - 1))
z1 = max(z0 + 1, min(z0 + Z_DIM, len(slice_files)))
sel_files = slice_files[z0:z1]
if len(sel_files) < Z_DIM:
    print(
        f"Warning: requested Z_DIM={Z_DIM} slices but only got {len(sel_files)} from z={z0}:{z1}"
    )

images = list(
    tqdm(
        _IO_EX.map(_read_tif_norm, sel_files),
        total=len(sel_files),
        desc="Loading slices",
    )
)
image_stack = torch.from_numpy(np.stack(images, axis=0))  # (Z, H, W) CPU float32

fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
if len(images) == 1:
    axes = [axes]
for im, ax in zip(images, axes):
    small = np.array(
        Image.fromarray((im * 255).astype(np.uint8)).resize(
            (max(1, im.shape[1] // 20), max(1, im.shape[0] // 20))
        )
    )
    ax.imshow(small, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

print(
    "image_stack:",
    tuple(image_stack.shape),
    "dtype:",
    image_stack.dtype,
    "device:",
    image_stack.device,
)




## === cell 3
H, W = mask.shape
orig_rect = (1100, 3500, 700, 950)  # (x, y, w, h)
if (orig_rect[0] + orig_rect[2] <= W) and (orig_rect[1] + orig_rect[3] <= H):
    rect = orig_rect
else:
    w = min(700, W - 2 * BUFFER - 1)
    h = min(950, H - 2 * BUFFER - 1)
    x = max(BUFFER, (W - w) // 2)
    y = max(BUFFER, (H - h) // 2)
    rect = (x, y, w, h)
    print("Adjusted rect to fit:", rect)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(label.numpy(), cmap="gray")
patch = patches.Rectangle(
    (rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor="r", facecolor="none"
)
ax.add_patch(patch)
ax.set_title("Training/eval region rect (red)")
ax.axis("off")
plt.show()




## === cell 4
pass




## === cell 5
pass




## === cell 6
def _make_unfolded_views(stack_zhw: torch.Tensor, buffer: int):
    """
    stack_zhw: CPU float32 (Z,H,W)
    returns:
      patches_view: (Z, H-2b, W-2b, K, K) view (no copy) via unfold
      y0_offset, x0_offset: offsets so that patch centered at (y,x) maps to
                            top-left index (y-b, x-b) in original image, and to
                            (y-b, x-b) in unfold index space.
    """
    b = int(buffer)
    K = 2 * b + 1
    patches_view = stack_zhw.unfold(1, K, 1).unfold(2, K, 1)
    return patches_view, b, b  # offsets in original coordinates


def extract_patches_batch(
    patches_view: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor, buffer: int
) -> torch.Tensor:
    """
    patches_view: (Z, H-2b, W-2b, K, K) from _make_unfolded_views
    ys/xs: CPU int64 tensors of shape (B,) giving center pixels in original coords
    returns: CPU float32 (B,1,Z,K,K)
    """
    b = int(buffer)
    ys = ys.to(dtype=torch.long, device="cpu")
    xs = xs.to(dtype=torch.long, device="cpu")
    iy = ys - b
    ix = xs - b
    sub = patches_view[:, iy, ix, :, :].permute(1, 0, 2, 3).unsqueeze(1).contiguous()
    return sub


class PixelDataset(data.Dataset):
    def __init__(
        self,
        stack_zhw: torch.Tensor,
        pixels_yx: np.ndarray,
        labels: torch.Tensor | None,
        buffer: int,
    ):
        self.stack = stack_zhw  # CPU
        self.pixels = pixels_yx.astype(np.int64, copy=False)  # (N,2) y,x
        self.labels = labels  # CPU float (N,1) or None
        self.buffer = int(buffer)

    def __len__(self):
        return self.pixels.shape[0]

    def __getitem__(self, idx: int):
        y = int(self.pixels[idx, 0])
        x = int(self.pixels[idx, 1])
        if self.labels is None:
            return y, x
        return y, x, float(self.labels[idx, 0].item())


_train_patches_view, _, _ = _make_unfolded_views(image_stack, BUFFER)


def collate_train(batch):
    ys = torch.as_tensor([b[0] for b in batch], dtype=torch.long)
    xs = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
    labs = torch.as_tensor([b[2] for b in batch], dtype=torch.float32).view(-1, 1)
    sub = extract_patches_batch(_train_patches_view, ys, xs, BUFFER)
    return sub, labs


def collate_eval(batch):
    ys = torch.as_tensor([b[0] for b in batch], dtype=torch.long)
    xs = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
    sub = extract_patches_batch(_train_patches_view, ys, xs, BUFFER)
    return sub, ys, xs


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

print("Generating pixel lists...")
not_border = np.zeros(mask.shape, dtype=bool)
not_border[BUFFER : mask.shape[0] - BUFFER, BUFFER : mask.shape[1] - BUFFER] = True

arr_mask = mask & not_border

inside_rect = np.zeros(mask.shape, dtype=bool)
inside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = True
inside_rect = inside_rect & arr_mask

outside_rect = arr_mask & (~inside_rect)

pixels_inside_rect = np.argwhere(inside_rect).astype(np.int32, copy=False)
pixels_outside_rect = np.argwhere(outside_rect).astype(np.int32, copy=False)

print(
    "pixels_inside_rect:",
    len(pixels_inside_rect),
    "pixels_outside_rect:",
    len(pixels_outside_rect),
)

train_labels_cpu = (
    label[
        torch.from_numpy(pixels_outside_rect[:, 0].astype(np.int64, copy=False)),
        torch.from_numpy(pixels_outside_rect[:, 1].astype(np.int64, copy=False)),
    ]
    .view(-1, 1)
    .float()
)

train_dataset = PixelDataset(image_stack, pixels_outside_rect, train_labels_cpu, BUFFER)
eval_dataset = PixelDataset(image_stack, pixels_inside_rect, None, BUFFER)

train_loader = data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=DL_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=(DL_WORKERS > 0),
    worker_init_fn=seed_worker,
    generator=_DL_GENERATOR,
    collate_fn=collate_train,
    prefetch_factor=4 if DL_WORKERS > 0 else None,
)

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

print("Training...")
model.train()
pbar = tqdm(total=TRAINING_STEPS, desc="Train iters", miniters=200, mininterval=0.5)
it = 0
while it < TRAINING_STEPS:
    for subvolumes, inklabels in train_loader:
        if it >= TRAINING_STEPS:
            break
        optimizer.zero_grad(set_to_none=True)
        subvolumes = subvolumes.to(DEVICE, non_blocking=True)
        inklabels = inklabels.to(DEVICE, non_blocking=True)
        outputs = model(subvolumes)
        loss = criterion(outputs, inklabels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        it += 1
        pbar.update(1)
pbar.close()

print("Done training.")




## === cell 7
eval_loader = data.DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=DL_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=(DL_WORKERS > 0),
    worker_init_fn=seed_worker,
    collate_fn=collate_eval,
    prefetch_factor=4 if DL_WORKERS > 0 else None,
)

output = torch.zeros_like(label).float()

model.eval()
with torch.inference_mode():
    for subvolumes, ys, xs in tqdm(eval_loader, desc="Eval rect"):
        preds = model(subvolumes.to(DEVICE, non_blocking=True)).view(-1).detach().cpu()
        output[ys, xs] = preds

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title("model output (rect area filled)")
ax1.imshow(output.numpy(), cmap="gray")
ax1.axis("off")
ax2.set_title("label")
ax2.imshow(label.numpy(), cmap="gray")
ax2.axis("off")
plt.tight_layout()
plt.show()




## === cell 8
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title(f"output > {THRESHOLD}")
ax1.imshow(output.gt(THRESHOLD).numpy(), cmap="gray")
ax1.axis("off")
ax2.set_title("label")
ax2.imshow(label.numpy(), cmap="gray")
ax2.axis("off")
plt.tight_layout()
plt.show()




## === cell 9
def rle_from_binary_mask(binary_mask_2d: np.ndarray) -> str:
    pixels = binary_mask_2d.astype(np.uint8, copy=False).reshape(-1)
    pixels = np.concatenate(([0], pixels, [0]))
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if runs.size == 0:
        return ""
    return " ".join(map(str, runs.tolist()))


def load_image_stack(fragment_dir: str, z_start: int, z_dim: int) -> torch.Tensor:
    sv_dir = os.path.join(fragment_dir, "surface_volume")
    files = sorted(glob.glob(os.path.join(sv_dir, "*.tif")))
    if len(files) == 0:
        raise FileNotFoundError(f"No .tif slices found in {sv_dir}")
    z0 = max(0, min(z_start, len(files) - 1))
    z1 = max(z0 + 1, min(z0 + z_dim, len(files)))
    sel = files[z0:z1]
    ims = list(_IO_EX.map(_read_tif_norm, sel))
    st = torch.from_numpy(np.stack(ims, axis=0))  # CPU (Z,H,W) float32
    return st


def predict_fragment(fragment_id: str) -> str:
    frag_dir = os.path.join(TEST_DIR, fragment_id)
    mpath = os.path.join(frag_dir, "mask.png")
    if not os.path.isfile(mpath):
        raise FileNotFoundError(
            f"Missing mask.png for test fragment {fragment_id}: {mpath}"
        )

    frag_mask = np.array(Image.open(mpath).convert("1"), dtype=bool)
    Ht, Wt = frag_mask.shape

    frag_stack = load_image_stack(frag_dir, Z_START, Z_DIM)

    frag_patches_view, _, _ = _make_unfolded_views(frag_stack, BUFFER)

    not_border = np.zeros((Ht, Wt), dtype=bool)
    not_border[BUFFER : Ht - BUFFER, BUFFER : Wt - BUFFER] = True
    valid = frag_mask & not_border
    valid_pixels = np.argwhere(valid).astype(np.int64, copy=False)  # (N,2) y,x

    def collate_test(batch):
        ys = torch.as_tensor([b[0] for b in batch], dtype=torch.long)
        xs = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
        sub = extract_patches_batch(frag_patches_view, ys, xs, BUFFER)
        return sub, ys, xs

    test_ds = PixelDataset(
        frag_stack, valid_pixels.astype(np.int32, copy=False), None, BUFFER
    )
    test_dl = data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DL_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(DL_WORKERS > 0),
        worker_init_fn=seed_worker,
        collate_fn=collate_test,
        prefetch_factor=4 if DL_WORKERS > 0 else None,
    )

    out = torch.zeros((Ht, Wt), dtype=torch.float32)  # CPU
    model.eval()
    with torch.inference_mode():
        for subvols, bys, bxs in tqdm(test_dl, desc=f"Predict test {fragment_id}"):
            preds = model(subvols.to(DEVICE, non_blocking=True)).view(-1).detach().cpu()
            out[bys, bxs] = preds

    bin_mask = (out.numpy() > THRESHOLD) & frag_mask
    return rle_from_binary_mask(bin_mask)


sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample = pd.read_csv(sample_path)
test_ids = sample["Id"].astype(str).tolist()

preds = []
for fid in test_ids:
    preds.append(predict_fragment(fid))

sub = pd.DataFrame({"Id": test_ids, "Predicted": preds})
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("submission.csv path:", os.path.abspath("submission.csv"))
