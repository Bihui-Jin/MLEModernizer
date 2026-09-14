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

0.001372

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

DATA_ROOT = "/kaggle/input/vesuvius-challenge-ink-detection/"
TRAIN_ROOT = os.path.join(DATA_ROOT, "train")
TEST_ROOT = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

TRAIN_FRAGMENT_ID = "1"
PREFIX = os.path.join(TRAIN_ROOT, TRAIN_FRAGMENT_ID) + "/"

BUFFER = 30  # Buffer size in x and y direction
Z_START = 27  # First slice in the z direction to use
Z_DIM = 10  # Number of slices in the z direction
TRAINING_STEPS = 20000
LEARNING_RATE = 0.05
BATCH_SIZE = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True  # safe: shapes are fixed; faster convs

print("Using DEVICE:", DEVICE)
print("TRAIN PREFIX:", PREFIX)
print("Exists ir.png:", os.path.exists(os.path.join(PREFIX, "ir.png")))

ir_path = os.path.join(PREFIX, "ir.png")
plt.figure(figsize=(6, 6))
plt.imshow(Image.open(ir_path), cmap="gray")
plt.title(f"IR ({TRAIN_FRAGMENT_ID})")
plt.axis("off")
plt.show()



## === cell 1
mask_path = os.path.join(PREFIX, "mask.png")
label_path = os.path.join(PREFIX, "inklabels.png")

mask = np.array(Image.open(mask_path).convert("1"))
label = torch.from_numpy(np.array(Image.open(label_path))).gt(0).float()  # keep on CPU

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title("mask.png")
ax1.imshow(mask, cmap="gray")
ax1.axis("off")
ax2.set_title("inklabels.png")
ax2.imshow(label.detach().cpu(), cmap="gray")
ax2.axis("off")
plt.show()

print("mask shape:", mask.shape, "label shape:", tuple(label.shape))



## === cell 2
sv_glob = sorted(glob.glob(os.path.join(PREFIX, "surface_volume", "*.tif")))
if len(sv_glob) == 0:
    raise FileNotFoundError(
        f"No .tif slices found under: {os.path.join(PREFIX, 'surface_volume')}"
    )

slice_files = sv_glob[Z_START : Z_START + Z_DIM]
if len(slice_files) != Z_DIM:
    raise RuntimeError(
        f"Expected {Z_DIM} slices but got {len(slice_files)}. Available: {len(sv_glob)}; requested [{Z_START}:{Z_START+Z_DIM}]"
    )


def _read_tif_float01(path: str) -> np.ndarray:
    try:
        from torchvision.io import read_image

        t = read_image(path)  # uint16 for these TIFFs
        if t.ndim == 3:
            t = t[0]  # grayscale
        return (t.to(torch.float32).numpy() / 65535.0).astype(np.float32, copy=False)
    except Exception:
        return (np.array(Image.open(path), dtype=np.float32) / 65535.0).astype(
            np.float32, copy=False
        )


images = [
    _read_tif_float01(fn) for fn in tqdm(slice_files, desc="Loading train slices")
]
image_stack = torch.stack([torch.from_numpy(im) for im in images], dim=0)  # [Z, H, W]

fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
for image, ax in zip(images, axes):
    thumb = np.array(
        Image.fromarray(image).resize((image.shape[1] // 20, image.shape[0] // 20))
    )
    ax.imshow(thumb, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.show()

print(
    "image_stack:",
    tuple(image_stack.shape),
    image_stack.dtype,
    "device:",
    image_stack.device,
)



## === cell 3
rect = (1100, 3500, 700, 950)  # (x, y, w, h)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(label.detach().cpu(), cmap="gray")
patch = patches.Rectangle(
    (rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor="r", facecolor="none"
)
ax.add_patch(patch)
ax.set_title("Training label with eval rect")
ax.axis("off")
plt.show()




## === cell 4
class SubvolumeDataset(data.Dataset):
    def __init__(
        self, image_stack_cpu: torch.Tensor, label_cpu: torch.Tensor, pixels: np.ndarray
    ):
        self.image_stack = image_stack_cpu.contiguous()  # [Z,H,W] on CPU
        self.label = label_cpu.contiguous()  # [H,W] on CPU
        self.pixels = pixels.astype(np.int64, copy=False)

        self.H = int(self.image_stack.shape[1])
        self.W = int(self.image_stack.shape[2])
        self.K = BUFFER * 2 + 1

        x = self.image_stack.unsqueeze(0)  # [1,Z,H,W]
        u = F.unfold(x, kernel_size=self.K, padding=BUFFER, stride=1)  # [1, Z*K*K, H*W]
        u = u.squeeze(0).transpose(0, 1).contiguous()  # [H*W, Z*K*K]
        self._unfold = u.view(self.H * self.W, Z_DIM, self.K * self.K)  # [H*W, Z, K*K]

        self._lin = (self.pixels[:, 0] * self.W + self.pixels[:, 1]).astype(
            np.int64, copy=False
        )

    def __len__(self):
        return int(self._lin.shape[0])

    def __getitem__(self, index):
        lin = int(self._lin[index])
        patch = self._unfold[lin]  # [Z, K*K]
        subvolume = patch.view(Z_DIM, self.K, self.K).unsqueeze(0)  # [1,Z,K,K]
        y, x = self.pixels[index]
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

print(model)



## === cell 5
print("Generating pixel lists...")
not_border = np.zeros(mask.shape, dtype=bool)
not_border[BUFFER : mask.shape[0] - BUFFER, BUFFER : mask.shape[1] - BUFFER] = True
arr_mask = np.array(mask, dtype=bool) & not_border

inside_rect = np.zeros(mask.shape, dtype=bool)
inside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = True
inside_rect = inside_rect & arr_mask

outside_rect = arr_mask.copy()
outside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = False

pixels_inside_rect = np.argwhere(inside_rect)
pixels_outside_rect = np.argwhere(outside_rect)

print("pixels_inside_rect:", len(pixels_inside_rect))
print("pixels_outside_rect:", len(pixels_outside_rect))

print("Training...")
train_dataset = SubvolumeDataset(image_stack, label, pixels_outside_rect)

_num_workers = min(4, (os.cpu_count() or 2))
train_loader = data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
)

criterion = nn.BCELoss()
optimizer = optim.Rprop(model.parameters(), lr=LEARNING_RATE)
scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=2)
scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])

model.train()
for i, (subvolumes, inklabels) in tqdm(
    enumerate(train_loader), total=TRAINING_STEPS, desc="Train iters"
):
    if i >= TRAINING_STEPS:
        break
    optimizer.zero_grad(set_to_none=True)
    outputs = model(subvolumes.to(DEVICE, non_blocking=True))
    loss = criterion(outputs, inklabels.to(DEVICE, non_blocking=True))
    loss.backward()
    optimizer.step()
    scheduler.step()

print("Done training.")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2639141434.py in <cell line: 0>()
     18 
     19 print("Training...")
---> 20 train_dataset = SubvolumeDataset(image_stack, label, pixels_outside_rect)
     21 
     22 # Speed: use multi-worker loading + persistent workers to reduce Python overhead.

/tmp/ipykernel_11/1700360402.py in __init__(self, image_stack_cpu, label_cpu, pixels)
     18         x = self.image_stack.unsqueeze(0)  # [1,Z,H,W]
     19         # F.unfold expects N,C,H,W -> returns [N, C*K*K, L] where L=H*W (stride=1, padding=BUFFER)
---> 20         u = F.unfold(x, kernel_size=self.K, padding=BUFFER, stride=1)  # [1, Z*K*K, H*W]
     21         u = u.squeeze(0).transpose(0, 1).contiguous()  # [H*W, Z*K*K]
     22         self._unfold = u.view(self.H * self.W, Z_DIM, self.K * self.K)  # [H*W, Z, K*K]

/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py in unfold(input, kernel_size, dilation, padding, stride)
   5528             stride=stride,
   5529         )
-> 5530     return torch._C._nn.im2col(
   5531         input, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride)
   5532     )

RuntimeError: [enforce fail at alloc_cpu.cpp:118] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 7707788053200 bytes. Error code 12 (Cannot allocate memory)

## === cell 6
eval_dataset = SubvolumeDataset(image_stack, label, pixels_inside_rect)
_num_workers = min(4, (os.cpu_count() or 2))
eval_loader = data.DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
)

output = torch.zeros_like(label).float().cpu()

model.eval()
with torch.no_grad():
    for i, (subvolumes, _) in enumerate(tqdm(eval_loader, desc="Eval rect")):
        preds = model(subvolumes.to(DEVICE, non_blocking=True)).view(-1).detach().cpu()
        start = i * BATCH_SIZE
        end = start + len(preds)
        coords = pixels_inside_rect[start:end]
        output[coords[:, 0], coords[:, 1]] = preds

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(output.detach().cpu(), cmap="gray")
ax1.set_title("Pred prob (rect)")
ax1.axis("off")
ax2.imshow(label.detach().cpu(), cmap="gray")
ax2.set_title("GT label")
ax2.axis("off")
plt.show()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3890007954.py in <cell line: 0>()
----> 1 eval_dataset = SubvolumeDataset(image_stack, label, pixels_inside_rect)
      2 _num_workers = min(4, (os.cpu_count() or 2))
      3 eval_loader = data.DataLoader(
      4     eval_dataset,
      5     batch_size=BATCH_SIZE,

/tmp/ipykernel_11/1700360402.py in __init__(self, image_stack_cpu, label_cpu, pixels)
     18         x = self.image_stack.unsqueeze(0)  # [1,Z,H,W]
     19         # F.unfold expects N,C,H,W -> returns [N, C*K*K, L] where L=H*W (stride=1, padding=BUFFER)
---> 20         u = F.unfold(x, kernel_size=self.K, padding=BUFFER, stride=1)  # [1, Z*K*K, H*W]
     21         u = u.squeeze(0).transpose(0, 1).contiguous()  # [H*W, Z*K*K]
     22         self._unfold = u.view(self.H * self.W, Z_DIM, self.K * self.K)  # [H*W, Z, K*K]

/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py in unfold(input, kernel_size, dilation, padding, stride)
   5528             stride=stride,
   5529         )
-> 5530     return torch._C._nn.im2col(
   5531         input, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride)
   5532     )

RuntimeError: [enforce fail at alloc_cpu.cpp:118] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 7707788053200 bytes. Error code 12 (Cannot allocate memory)

## === cell 7
THRESHOLD = 0.4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(output.gt(THRESHOLD).detach().cpu(), cmap="gray")
ax1.set_title(f"Pred mask (thr={THRESHOLD})")
ax1.axis("off")
ax2.imshow(label.detach().cpu(), cmap="gray")
ax2.set_title("GT label")
ax2.axis("off")
plt.show()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/891330046.py in <cell line: 0>()
      1 THRESHOLD = 0.4
      2 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
----> 3 ax1.imshow(output.gt(THRESHOLD).detach().cpu(), cmap="gray")
      4 ax1.set_title(f"Pred mask (thr={THRESHOLD})")
      5 ax1.axis("off")

NameError: name 'output' is not defined

## === cell 8
def rle_from_binary_mask(mask_2d: np.ndarray) -> str:
    """
    Kaggle Vesuvius expects RLE over the flattened image in row-major order,
    using 1-indexed positions and (start, length) pairs.
    """
    pixels = mask_2d.flatten(order="C").astype(np.uint8)
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1  # 1-indexed in padded
    starts = changes[0::2]
    ends = changes[1::2]
    lengths = ends - starts
    if len(starts) == 0:
        return ""
    runs = np.column_stack([starts, lengths]).reshape(-1)
    return " ".join(str(int(x)) for x in runs)


def predict_fragment_rle(fragment_dir: str, threshold: float) -> str:
    """
    Run the trained pixel classifier over all valid (masked & not-border) pixels
    in the test fragment and return the RLE string.
    """
    mask_path = os.path.join(fragment_dir, "mask.png")
    mask = np.array(Image.open(mask_path).convert("1"), dtype=bool)

    sv_glob = sorted(glob.glob(os.path.join(fragment_dir, "surface_volume", "*.tif")))
    slice_files = sv_glob[Z_START : Z_START + Z_DIM]
    if len(slice_files) != Z_DIM:
        raise RuntimeError(
            f"{os.path.basename(fragment_dir)}: expected {Z_DIM} slices but got {len(slice_files)}"
        )

    imgs = [_read_tif_float01(fn) for fn in slice_files]
    stack = torch.stack([torch.from_numpy(im) for im in imgs], dim=0)  # CPU [Z,H,W]

    not_border = np.zeros(mask.shape, dtype=bool)
    not_border[BUFFER : mask.shape[0] - BUFFER, BUFFER : mask.shape[1] - BUFFER] = True
    valid = mask & not_border

    coords = np.argwhere(valid)  # (y,x)

    dummy_label = torch.zeros((1, 1), dtype=torch.float32)
    ds = SubvolumeDataset(
        stack, dummy_label.expand(mask.shape[0], mask.shape[1]).contiguous(), coords
    )

    _num_workers = min(4, (os.cpu_count() or 2))
    dl = data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=_num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(_num_workers > 0),
        prefetch_factor=2 if _num_workers > 0 else None,
    )

    prob_map = torch.zeros(
        (mask.shape[0], mask.shape[1]), device="cpu", dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        for i, (subvolumes, _) in enumerate(
            tqdm(dl, desc=f"Predict {os.path.basename(fragment_dir)}", leave=False)
        ):
            preds = (
                model(subvolumes.to(DEVICE, non_blocking=True)).view(-1).detach().cpu()
            )
            start = i * BATCH_SIZE
            end = start + len(preds)
            c = coords[start:end]
            prob_map[c[:, 0], c[:, 1]] = preds

    bin_mask = (prob_map.numpy() > threshold) & mask
    return rle_from_binary_mask(bin_mask)


sample_df = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = sample_df["Id"].tolist()

preds = []
for fid in test_ids:
    frag_dir = os.path.join(TEST_ROOT, fid)
    preds.append(predict_fragment_rle(frag_dir, THRESHOLD))

sub_df = pd.DataFrame({"Id": test_ids, "Predicted": preds})
sub_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(sub_df.head())
print("submission.csv size (bytes):", os.path.getsize("submission.csv"))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1106262528.py in <cell line: 0>()
     83 for fid in test_ids:
     84     frag_dir = os.path.join(TEST_ROOT, fid)
---> 85     preds.append(predict_fragment_rle(frag_dir, THRESHOLD))
     86 
     87 sub_df = pd.DataFrame({"Id": test_ids, "Predicted": preds})

/tmp/ipykernel_11/1106262528.py in predict_fragment_rle(fragment_dir, threshold)
     42     # Speed: avoid repeatedly allocating large dummy label maps; dataset only needs indexing.
     43     dummy_label = torch.zeros((1, 1), dtype=torch.float32)
---> 44     ds = SubvolumeDataset(
     45         stack, dummy_label.expand(mask.shape[0], mask.shape[1]).contiguous(), coords
     46     )

/tmp/ipykernel_11/1700360402.py in __init__(self, image_stack_cpu, label_cpu, pixels)
     18         x = self.image_stack.unsqueeze(0)  # [1,Z,H,W]
     19         # F.unfold expects N,C,H,W -> returns [N, C*K*K, L] where L=H*W (stride=1, padding=BUFFER)
---> 20         u = F.unfold(x, kernel_size=self.K, padding=BUFFER, stride=1)  # [1, Z*K*K, H*W]
     21         u = u.squeeze(0).transpose(0, 1).contiguous()  # [H*W, Z*K*K]
     22         self._unfold = u.view(self.H * self.W, Z_DIM, self.K * self.K)  # [H*W, Z, K*K]

/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py in unfold(input, kernel_size, dilation, padding, stride)
   5528             stride=stride,
   5529         )
-> 5530     return torch._C._nn.im2col(
   5531         input, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride)
   5532     )

RuntimeError: [enforce fail at alloc_cpu.cpp:118] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 5942272382960 bytes. Error code 12 (Cannot allocate memory)
