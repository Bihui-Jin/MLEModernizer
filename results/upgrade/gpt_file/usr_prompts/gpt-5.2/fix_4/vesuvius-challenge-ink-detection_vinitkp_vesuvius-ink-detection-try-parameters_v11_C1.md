# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.007643

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm


def resolve_dataset_root():
    candidates = [
        "/kaggle/input/vesuvius-challenge-ink-detection",
        "/kaggle/input/vesuvius-challenge",
        "/kaggle/data/vesuvius-challenge-ink-detection",
        "/kaggle/data/vesuvius-challenge",
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "train")) and os.path.isdir(
            os.path.join(c, "test")
        ):
            return c
    base = "/kaggle/input"
    if os.path.isdir(base):
        for name in os.listdir(base):
            c = os.path.join(base, name)
            if os.path.isdir(os.path.join(c, "train")) and os.path.isdir(
                os.path.join(c, "test")
            ):
                return c
    raise FileNotFoundError(
        "Could not find dataset root containing train/ and test/ under /kaggle/input or /kaggle/data"
    )


SEED = 1337
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

DATA_ROOT = resolve_dataset_root()
TRAIN_ROOT = os.path.join(DATA_ROOT, "train")
TEST_ROOT = os.path.join(DATA_ROOT, "test")

print("DATA_ROOT =", DATA_ROOT)
print(
    "Train fragments:",
    sorted(
        [
            d
            for d in os.listdir(TRAIN_ROOT)
            if os.path.isdir(os.path.join(TRAIN_ROOT, d))
        ]
    )[:10],
)
print(
    "Test fragments:",
    sorted(
        [d for d in os.listdir(TEST_ROOT) if os.path.isdir(os.path.join(TEST_ROOT, d))]
    )[:10],
)

BUFFER = 30  # Buffer size in x and y direction
Z_START = 27  # First slice in the z direction to use
Z_DIM = 10  # Number of slices in the z direction
TRAINING_STEPS = 20000
LEARNING_RATE = 0.05
BATCH_SIZE = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CPU_COUNT = os.cpu_count() or 2
NUM_WORKERS = min(4, max(1, CPU_COUNT // 2))
PIN_MEMORY = DEVICE.type == "cuda"

train_fragment_id = (
    "1"
    if os.path.isdir(os.path.join(TRAIN_ROOT, "1"))
    else sorted(
        [
            d
            for d in os.listdir(TRAIN_ROOT)
            if os.path.isdir(os.path.join(TRAIN_ROOT, d))
        ]
    )[0]
)
PREFIX = os.path.join(TRAIN_ROOT, train_fragment_id) + os.sep
print("Using training fragment:", train_fragment_id, "PREFIX=", PREFIX)

ir_path = os.path.join(PREFIX, "ir.png")
if os.path.exists(ir_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(Image.open(ir_path), cmap="gray")
    plt.title(f"ir.png (train/{train_fragment_id})")
    plt.axis("off")
    plt.show()



## === cell 1
mask_path = os.path.join(PREFIX, "mask.png")
label_path = os.path.join(PREFIX, "inklabels.png")

mask = np.array(Image.open(mask_path).convert("1"), dtype=np.uint8)  # 0/1
label_np = (np.array(Image.open(label_path)) > 0).astype(np.float32)  # 0/1 float

label = torch.from_numpy(label_np)  # CPU tensor

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title("mask.png")
ax1.imshow(mask, cmap="gray")
ax1.axis("off")
ax2.set_title("inklabels.png")
ax2.imshow(label.numpy(), cmap="gray")
ax2.axis("off")
plt.tight_layout()
plt.show()



## === cell 2
_SORTED_TIF_CACHE = {}


def sorted_tif_files(folder):
    if folder in _SORTED_TIF_CACHE:
        return _SORTED_TIF_CACHE[folder]
    files = glob.glob(os.path.join(folder, "*.tif"))

    def key_fn(p):
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            return int(base)
        except ValueError:
            return base

    out = sorted(files, key=key_fn)
    _SORTED_TIF_CACHE[folder] = out
    return out


surface_dir = os.path.join(PREFIX, "surface_volume")
tif_files = sorted_tif_files(surface_dir)
if len(tif_files) == 0:
    raise FileNotFoundError(f"No .tif files found in {surface_dir}")

z_end = Z_START + Z_DIM
if z_end > len(tif_files):
    raise ValueError(
        f"Requested Z slices [{Z_START}:{z_end}] but only {len(tif_files)} tif files available"
    )

images = [
    np.array(Image.open(fn), dtype=np.float32) / 65535.0
    for fn in tqdm(tif_files[Z_START:z_end], desc="Loading train z-slices")
]
image_stack = torch.stack(
    [torch.from_numpy(im) for im in images], dim=0
)  # (Z, H, W) CPU
print(
    "image_stack:",
    tuple(image_stack.shape),
    "dtype:",
    image_stack.dtype,
    "device:",
    image_stack.device,
)

fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
for im, ax in zip(images, axes):
    thumb = np.array(
        Image.fromarray((im * 255).astype(np.uint8)).resize(
            (im.shape[1] // 20, im.shape[0] // 20)
        )
    )
    ax.imshow(thumb, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()



## === cell 3
rect = (1100, 3500, 700, 950)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(label.numpy(), cmap="gray")
patch = patches.Rectangle(
    (rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor="r", facecolor="none"
)
ax.add_patch(patch)
ax.set_title("Training fragment label with eval rect")
ax.axis("off")
plt.show()




## === cell 4
class SubvolumeDataset(data.Dataset):
    def __init__(self, image_stack_cpu: torch.Tensor, label_cpu: torch.Tensor, pixels):
        self.image_stack = image_stack_cpu.contiguous()  # (Z,H,W) CPU
        self.label = label_cpu  # (H,W) CPU
        self.pixels = np.asarray(pixels, dtype=np.int64)

        self.Z, self.H, self.W = self.image_stack.shape
        self.kernel = BUFFER * 2 + 1
        self._grid_w = self.W - 2 * BUFFER  # valid x positions count
        self._label_flat = self.label.contiguous().view(-1)

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        y, x = self.pixels[index]
        inklabel = self.label[y, x].view(1)  # CPU tensor shape (1,)
        return int(y), int(x), inklabel


def _batch_to_subvolumes(
    image_stack_cpu: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor
) -> torch.Tensor:
    """
    Build (B,1,Z_DIM,K,K) subvolumes for centers at (ys,xs), exact match to original slicing.
    Uses unfold per z-slice, then gathers patch columns for requested centers.
    """
    Z, H, W = image_stack_cpu.shape
    k = BUFFER * 2 + 1
    grid_w = W - 2 * BUFFER
    cols = (ys - BUFFER) * grid_w + (xs - BUFFER)  # (B,)
    cols = cols.to(torch.long)

    patches_z = []
    for z in range(Z):
        img = image_stack_cpu[z].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        uf = F.unfold(img, kernel_size=(k, k), padding=0, stride=1)  # (1, k*k, L)
        sel = uf.index_select(dim=2, index=cols)  # (1, k*k, B)
        sel = sel.permute(2, 1, 0).contiguous().view(cols.numel(), k * k)  # (B, k*k)
        patches_z.append(sel.view(cols.numel(), 1, k, k))  # (B,1,k,k)
    x = torch.stack(patches_z, dim=1)  # (B, Z, 1, k, k)
    x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B,1,Z,k,k)
    return x


def make_collate_fn(image_stack_cpu: torch.Tensor):
    def _collate(batch):
        ys = torch.tensor([b[0] for b in batch], dtype=torch.long)
        xs = torch.tensor([b[1] for b in batch], dtype=torch.long)
        inklabels = torch.stack([b[2] for b in batch], dim=0)  # (B,1)
        subvolumes = _batch_to_subvolumes(image_stack_cpu, ys, xs)  # (B,1,Z,k,k)
        return subvolumes, inklabels

    return _collate


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



## === cell 5
print("Generating pixel lists...")
not_border = np.zeros(mask.shape, dtype=bool)
not_border[BUFFER : mask.shape[0] - BUFFER, BUFFER : mask.shape[1] - BUFFER] = True

arr_mask = mask.astype(bool) & not_border

inside_rect = np.zeros(mask.shape, dtype=bool)
inside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = True
inside_rect = inside_rect & arr_mask

outside_rect = arr_mask.copy()
outside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = False

pixels_inside_rect = np.argwhere(inside_rect)
pixels_outside_rect = np.argwhere(outside_rect)

print(
    "pixels_inside_rect:",
    len(pixels_inside_rect),
    "pixels_outside_rect:",
    len(pixels_outside_rect),
)

print("Training...")
train_dataset = SubvolumeDataset(image_stack, label, pixels_outside_rect)

g = torch.Generator()
g.manual_seed(SEED)
train_loader = data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
    generator=g,
    collate_fn=make_collate_fn(image_stack),
)

criterion = nn.BCELoss()
optimizer = optim.ASGD(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=4)

model.train()
for i, (subvolumes, inklabels) in tqdm(enumerate(train_loader), total=TRAINING_STEPS):
    if i >= TRAINING_STEPS:
        break
    optimizer.zero_grad(set_to_none=True)
    subvolumes = subvolumes.to(DEVICE, non_blocking=True)
    inklabels = inklabels.to(DEVICE, non_blocking=True)
    outputs = model(subvolumes)
    loss = criterion(outputs, inklabels)
    loss.backward()
    optimizer.step()
    scheduler.step()

print("Training done.")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2880642627.py in <cell line: 0>()
     45 
     46 model.train()
---> 47 for i, (subvolumes, inklabels) in tqdm(enumerate(train_loader), total=TRAINING_STEPS):
     48     if i >= TRAINING_STEPS:
     49         break

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/903999269.py", line 65, in _collate
    subvolumes = _batch_to_subvolumes(image_stack_cpu, ys, xs)  # (B,1,Z,k,k)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/903999269.py", line 49, in _batch_to_subvolumes
    uf = F.unfold(img, kernel_size=(k, k), padding=0, stride=1)  # (1, k*k, L)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py", line 5530, in unfold
    return torch._C._nn.im2col(
           ^^^^^^^^^^^^^^^^^^^^
RuntimeError: [enforce fail at alloc_cpu.cpp:118] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 757873484280 bytes. Error code 12 (Cannot allocate memory)


## === cell 6
eval_dataset = SubvolumeDataset(image_stack, label, pixels_inside_rect)
eval_loader = data.DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=2 if NUM_WORKERS > 0 else None,
    collate_fn=make_collate_fn(image_stack),
)

output = torch.zeros_like(label).float()  # CPU output
model.eval()
with torch.no_grad():
    for i, (subvolumes, _) in enumerate(tqdm(eval_loader, desc="Eval on train rect")):
        subvolumes = subvolumes.to(DEVICE, non_blocking=True)
        preds = model(subvolumes).detach().squeeze(1).float().cpu()
        start = i * BATCH_SIZE
        end = min(start + len(preds), len(pixels_inside_rect))
        coords = pixels_inside_rect[start:end]
        output[coords[:, 0], coords[:, 1]] = preds[: end - start]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(output.numpy(), cmap="gray")
ax1.set_title("Model output (train rect)")
ax1.axis("off")
ax2.imshow(label.numpy(), cmap="gray")
ax2.set_title("Label")
ax2.axis("off")
plt.tight_layout()
plt.show()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3585765208.py in <cell line: 0>()
     14 model.eval()
     15 with torch.no_grad():
---> 16     for i, (subvolumes, _) in enumerate(tqdm(eval_loader, desc="Eval on train rect")):
     17         subvolumes = subvolumes.to(DEVICE, non_blocking=True)
     18         preds = model(subvolumes).detach().squeeze(1).float().cpu()

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/903999269.py", line 65, in _collate
    subvolumes = _batch_to_subvolumes(image_stack_cpu, ys, xs)  # (B,1,Z,k,k)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/903999269.py", line 49, in _batch_to_subvolumes
    uf = F.unfold(img, kernel_size=(k, k), padding=0, stride=1)  # (1, k*k, L)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py", line 5530, in unfold
    return torch._C._nn.im2col(
           ^^^^^^^^^^^^^^^^^^^^
RuntimeError: [enforce fail at alloc_cpu.cpp:118] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 757873484280 bytes. Error code 12 (Cannot allocate memory)


## === cell 7
THRESHOLD = 0.4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(output.gt(THRESHOLD).numpy(), cmap="gray")
ax1.set_title(f"Thresholded output (>{THRESHOLD})")
ax1.axis("off")
ax2.imshow(label.numpy(), cmap="gray")
ax2.set_title("Label")
ax2.axis("off")
plt.tight_layout()
plt.show()




## === cell 8
def rle_from_binary_mask(binary_mask_2d: np.ndarray) -> str:
    """
    binary_mask_2d: HxW array of {0,1} or bool.
    RLE uses 1-indexed pixels in row-major order.
    """
    pixels = binary_mask_2d.flatten(order="C").astype(np.uint8)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # 1-indexed due to pad
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return " ".join(str(int(x)) for x in runs)


def load_image_stack_for_fragment(fragment_dir: str) -> torch.Tensor:
    surface_dir = os.path.join(fragment_dir, "surface_volume")
    tif_files = sorted_tif_files(surface_dir)
    if len(tif_files) == 0:
        raise FileNotFoundError(f"No .tif files found in {surface_dir}")
    z_end = Z_START + Z_DIM
    if z_end > len(tif_files):
        raise ValueError(
            f"Requested Z slices [{Z_START}:{z_end}] but only {len(tif_files)} tif files available in {surface_dir}"
        )
    imgs = [
        np.array(Image.open(fn), dtype=np.float32) / 65535.0
        for fn in tif_files[Z_START:z_end]
    ]
    stack = torch.stack([torch.from_numpy(im) for im in imgs], dim=0)  # CPU
    return stack


def predict_fragment_binary_mask(fragment_dir: str) -> np.ndarray:
    frag_mask = np.array(
        Image.open(os.path.join(fragment_dir, "mask.png")).convert("1"), dtype=np.uint8
    ).astype(bool)

    not_border = np.zeros(frag_mask.shape, dtype=bool)
    not_border[
        BUFFER : frag_mask.shape[0] - BUFFER, BUFFER : frag_mask.shape[1] - BUFFER
    ] = True
    valid_pixels = frag_mask & not_border

    pixels = np.argwhere(valid_pixels)
    if len(pixels) == 0:
        return np.zeros(frag_mask.shape, dtype=np.uint8)

    stack = load_image_stack_for_fragment(fragment_dir)
    dummy_label = torch.zeros(
        frag_mask.shape, dtype=torch.float32
    )  # CPU; label not used in test

    ds = SubvolumeDataset(stack, dummy_label, pixels)

    dl = data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        collate_fn=make_collate_fn(stack),
    )

    prob = np.zeros(frag_mask.shape, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        offset = 0
        for subvolumes, _ in tqdm(
            dl, desc=f"Infer {os.path.basename(fragment_dir)}", leave=False
        ):
            subvolumes = subvolumes.to(DEVICE, non_blocking=True)
            preds = model(subvolumes).detach().squeeze(1).float().cpu().numpy()
            coords = pixels[offset : offset + len(preds)]
            prob[coords[:, 0], coords[:, 1]] = preds
            offset += len(preds)

    binary = (prob > THRESHOLD).astype(np.uint8)
    binary[~frag_mask] = 0
    return binary


test_fragment_ids = sorted(
    [d for d in os.listdir(TEST_ROOT) if os.path.isdir(os.path.join(TEST_ROOT, d))]
)
print("Test fragments:", test_fragment_ids)

rows = []
for fid in test_fragment_ids:
    fdir = os.path.join(TEST_ROOT, fid)
    pred_bin = predict_fragment_binary_mask(fdir)
    pred_rle = rle_from_binary_mask(pred_bin)
    rows.append({"Id": fid, "Predicted": pred_rle})

sub = pd.DataFrame(rows, columns=["Id", "Predicted"])
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
print("submission.csv size (bytes):", os.path.getsize("submission.csv"))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1267509618.py in <cell line: 0>()
     89 for fid in test_fragment_ids:
     90     fdir = os.path.join(TEST_ROOT, fid)
---> 91     pred_bin = predict_fragment_binary_mask(fdir)
     92     pred_rle = rle_from_binary_mask(pred_bin)
     93     rows.append({"Id": fid, "Predicted": pred_rle})

/tmp/ipykernel_11/1267509618.py in predict_fragment_binary_mask(fragment_dir)
     67     with torch.no_grad():
     68         offset = 0
---> 69         for subvolumes, _ in tqdm(
     70             dl, desc=f"Infer {os.path.basename(fragment_dir)}", leave=False
     71         ):

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/903999269.py", line 65, in _collate
    subvolumes = _batch_to_subvolumes(image_stack_cpu, ys, xs)  # (B,1,Z,k,k)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/903999269.py", line 49, in _batch_to_subvolumes
    uf = F.unfold(img, kernel_size=(k, k), padding=0, stride=1)  # (1, k*k, L)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py", line 5530, in unfold
    return torch._C._nn.im2col(
           ^^^^^^^^^^^^^^^^^^^^
RuntimeError: [enforce fail at alloc_cpu.cpp:118] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 582800791496 bytes. Error code 12 (Cannot allocate memory)
