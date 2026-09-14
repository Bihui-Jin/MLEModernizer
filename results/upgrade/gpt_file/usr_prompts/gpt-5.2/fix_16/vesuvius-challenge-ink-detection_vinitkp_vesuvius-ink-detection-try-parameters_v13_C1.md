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
THRESHOLD = 0.4  # used for binarization in visualization + submission

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

torch.set_num_threads(min(8, os.cpu_count() or 1))

print("Using DEVICE:", DEVICE)
print("TRAIN PREFIX:", PREFIX)
print("Exists ir.png:", os.path.exists(os.path.join(PREFIX, "ir.png")))

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() != "batch":
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

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() != "batch":
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

from functools import lru_cache


@lru_cache(maxsize=256)
def _read_tif_float01(path: str) -> np.ndarray:
    import tifffile  # type: ignore

    arr = tifffile.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.asarray(arr, dtype=np.float32, order="C")
    arr *= 1.0 / 65535.0
    return arr


images = [
    _read_tif_float01(fn) for fn in tqdm(slice_files, desc="Loading train slices")
]
image_stack = torch.stack([torch.from_numpy(im) for im in images], dim=0)  # [Z, H, W]

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() != "batch":
    fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
    for image, ax in zip(images, axes):
        im_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        thumb = np.array(
            Image.fromarray(im_u8).resize((image.shape[1] // 20, image.shape[0] // 20))
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

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() != "batch":
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(label.detach().cpu(), cmap="gray")
    patch = patches.Rectangle(
        (rect[0], rect[1]),
        rect[2],
        rect[3],
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(patch)
    ax.set_title("Training label with eval rect")
    ax.axis("off")
    plt.show()



## === cell 4
_K = BUFFER * 2 + 1


class SubvolumeDatasetFast(data.Dataset):
    """
    Kept for compatibility with the original semantics.
    NOTE: The optimized pipeline below does NOT use __getitem__ during training
    to avoid Python overhead; it uses vectorized gather instead.
    """

    def __init__(
        self, image_stack_cpu: torch.Tensor, label_cpu: torch.Tensor, pixels: np.ndarray
    ):
        self.image_stack = image_stack_cpu.contiguous()  # [Z,H,W] on CPU
        self.label = label_cpu.contiguous()  # [H,W] on CPU
        self.pixels = pixels.astype(np.int64, copy=False)

        self.Z = int(self.image_stack.shape[0])
        self.H = int(self.image_stack.shape[1])
        self.W = int(self.image_stack.shape[2])
        self.K = _K

        self.padded_t = F.pad(
            self.image_stack.to(dtype=torch.float32),
            (BUFFER, BUFFER, BUFFER, BUFFER),  # W then H
            mode="constant",
            value=0.0,
        ).contiguous()  # [Z, H+2B, W+2B]

        self.label_t = self.label.to(dtype=torch.float32).contiguous()

    def __len__(self):
        return int(self.pixels.shape[0])

    def __getitem__(self, index):
        y, x = self.pixels[index]
        patch = self.padded_t[:, y : y + self.K, x : x + self.K]  # [Z,K,K]
        subvolume = patch.unsqueeze(0)  # [1,Z,K,K]
        inklabel = torch.tensor([float(self.label_t[y, x])], dtype=torch.float32)
        return subvolume, inklabel


def make_unfold_view(padded_zhw: torch.Tensor, K: int) -> torch.Tensor:
    return padded_zhw.unfold(1, K, 1).unfold(2, K, 1)


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

if DEVICE.type == "cuda":
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("torch.compile enabled")
    except Exception as e:
        print("torch.compile unavailable, continuing eager. Reason:", repr(e))

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

train_dataset = SubvolumeDatasetFast(image_stack, label, pixels_outside_rect)

criterion = nn.BCELoss()
optimizer = optim.Rprop(model.parameters(), lr=LEARNING_RATE)
scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=2)
scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])

padded_train_cpu = train_dataset.padded_t.contiguous()
label_train_cpu = train_dataset.label_t.contiguous()
coords_train_np = train_dataset.pixels  # numpy [N,2] (y,x)

rng = np.random.RandomState(0)  # deterministic, matches original seed usage
N = int(coords_train_np.shape[0])
K = train_dataset.K

if DEVICE.type == "cuda":
    padded_train_cpu = padded_train_cpu.pin_memory()
    label_train_cpu = label_train_cpu.pin_memory()

Hp = int(padded_train_cpu.shape[1])
Wp = int(padded_train_cpu.shape[2])
H = int(train_dataset.H)
W = int(train_dataset.W)

coords_train_lin_np = (coords_train_np[:, 0] * W + coords_train_np[:, 1]).astype(
    np.int64, copy=False
)

all_idx = rng.randint(0, N, size=(TRAINING_STEPS, BATCH_SIZE), dtype=np.int64)
all_lin = coords_train_lin_np[all_idx]  # [steps, B]
all_y = coords_train_np[:, 0]
all_x = coords_train_np[:, 1]

subvolumes_buf = torch.empty(
    (BATCH_SIZE, 1, Z_DIM, K, K), device=DEVICE, dtype=torch.float32
)

prefetch_stream = torch.cuda.Stream() if DEVICE.type == "cuda" else None


def _h2d_async(t: torch.Tensor) -> torch.Tensor:
    if DEVICE.type == "cuda":
        return t.to(device=DEVICE, non_blocking=True)
    return t.to(device=DEVICE)


def _make_unfold_on_device(padded_dev: torch.Tensor) -> torch.Tensor:
    return make_unfold_view(padded_dev, K)


model.train()
pbar = tqdm(range(TRAINING_STEPS), desc="Train iters", mininterval=0.5)

if DEVICE.type == "cuda":
    with torch.cuda.stream(prefetch_stream):
        padded_train_dev = _h2d_async(padded_train_cpu)
        label_train_dev = _h2d_async(label_train_cpu)
        unfold_train = _make_unfold_on_device(padded_train_dev)  # view
else:
    padded_train_dev = _h2d_async(padded_train_cpu)
    label_train_dev = _h2d_async(label_train_cpu)
    unfold_train = _make_unfold_on_device(padded_train_dev)

for it in pbar:
    if DEVICE.type == "cuda":
        torch.cuda.current_stream().wait_stream(prefetch_stream)

    lin_cpu = all_lin[it]  # numpy [B]
    lin_dev = torch.from_numpy(lin_cpu).to(
        device=DEVICE, dtype=torch.int64, non_blocking=True
    )

    unfold_flat = unfold_train.reshape(Z_DIM, H * W, K, K)  # view
    patches = torch.index_select(unfold_flat, 1, lin_dev)  # [Z, B, K, K]

    subvolumes_buf[:, 0].copy_(patches.permute(1, 0, 2, 3))

    label_flat = label_train_dev.reshape(-1)  # [H*W]
    inklabels = torch.index_select(label_flat, 0, lin_dev).view(-1, 1)  # [B,1]

    optimizer.zero_grad(set_to_none=True)
    outputs = model(subvolumes_buf)
    loss = criterion(outputs, inklabels)
    loss.backward()
    optimizer.step()
    scheduler.step()

print("Done training.")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_11/1981337352.py in <cell line: 0>()
    107 
    108     # unfold_train: [Z, H, W, K, K] -> flatten H*W
--> 109     unfold_flat = unfold_train.reshape(Z_DIM, H * W, K, K)  # view
    110     patches = torch.index_select(unfold_flat, 1, lin_dev)  # [Z, B, K, K]
    111 

OutOfMemoryError: CUDA out of memory. Tried to allocate 7178.44 GiB. GPU 0 has a total capacity of 47.53 GiB of which 45.02 GiB is free. Process 2479991 has 2.49 GiB memory in use. Of the allocated memory 2.16 GiB is allocated by PyTorch, and 19.51 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 6
eval_dataset = SubvolumeDatasetFast(image_stack, label, pixels_inside_rect)

padded_eval = eval_dataset.padded_t.to(DEVICE, non_blocking=True)
coords_eval_np = eval_dataset.pixels  # numpy [M,2]
K = eval_dataset.K
H = int(eval_dataset.H)
W = int(eval_dataset.W)

unfold_eval = make_unfold_view(padded_eval, K)  # [Z,H,W,K,K]
unfold_eval_flat = unfold_eval.reshape(Z_DIM, H * W, K, K)

coords_eval_lin_np = (coords_eval_np[:, 0] * W + coords_eval_np[:, 1]).astype(
    np.int64, copy=False
)

output = torch.zeros_like(label).float().cpu()

model.eval()
with torch.no_grad():
    ys_cpu = coords_eval_np[:, 0]
    xs_cpu = coords_eval_np[:, 1]

    for start in tqdm(
        range(0, len(coords_eval_np), BATCH_SIZE), desc="Eval rect", mininterval=0.5
    ):
        end = min(start + BATCH_SIZE, len(coords_eval_np))
        lin_dev = torch.from_numpy(coords_eval_lin_np[start:end]).to(
            device=DEVICE, dtype=torch.int64, non_blocking=True
        )
        patches = torch.index_select(unfold_eval_flat, 1, lin_dev)  # [Z,b,K,K]
        subvolumes = patches.permute(1, 0, 2, 3).unsqueeze(1)  # [b,1,Z,K,K]
        preds = model(subvolumes).view(-1).detach().cpu()
        output[ys_cpu[start:end], xs_cpu[start:end]] = preds

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() != "batch":
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
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_11/3220731491.py in <cell line: 0>()
      9 
     10 unfold_eval = make_unfold_view(padded_eval, K)  # [Z,H,W,K,K]
---> 11 unfold_eval_flat = unfold_eval.reshape(Z_DIM, H * W, K, K)
     12 
     13 coords_eval_lin_np = (coords_eval_np[:, 0] * W + coords_eval_np[:, 1]).astype(

OutOfMemoryError: CUDA out of memory. Tried to allocate 7178.44 GiB. GPU 0 has a total capacity of 47.53 GiB of which 43.06 GiB is free. Process 2479991 has 4.46 GiB memory in use. Of the allocated memory 4.12 GiB is allocated by PyTorch, and 20.69 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 7
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() != "batch":
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
/tmp/ipykernel_11/4026573561.py in <cell line: 0>()
      1 if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "").lower() != "batch":
      2     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
----> 3     ax1.imshow(output.gt(THRESHOLD).detach().cpu(), cmap="gray")
      4     ax1.set_title(f"Pred mask (thr={THRESHOLD})")
      5     ax1.axis("off")

NameError: name 'output' is not defined

## === cell 8
def rle_from_binary_mask(mask_2d: np.ndarray) -> str:
    """
    Kaggle Vesuvius expects RLE over the flattened image in row-major order,
    using 1-indexed positions and (start, length) pairs.
    """
    pixels = mask_2d.ravel(order="C").astype(np.uint8)
    if pixels.size == 0:
        return ""
    padded = np.empty(pixels.size + 2, dtype=np.uint8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = pixels
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    if changes.size == 0:
        return ""
    starts = changes[0::2]
    ends = changes[1::2]
    lengths = ends - starts
    runs = np.column_stack((starts, lengths)).ravel()
    return " ".join(map(str, runs.tolist()))


def predict_fragment_rle(fragment_dir: str, threshold: float) -> str:
    """
    Run the trained pixel classifier over all valid (masked & not-border) pixels
    in the test fragment and return the RLE string.
    """
    mask_path = os.path.join(fragment_dir, "mask.png")
    mask_local = np.array(Image.open(mask_path).convert("1"), dtype=bool)

    sv_glob_local = sorted(
        glob.glob(os.path.join(fragment_dir, "surface_volume", "*.tif"))
    )
    slice_files_local = sv_glob_local[Z_START : Z_START + Z_DIM]
    if len(slice_files_local) != Z_DIM:
        raise RuntimeError(
            f"{os.path.basename(fragment_dir)}: expected {Z_DIM} slices but got {len(slice_files_local)}"
        )

    imgs = [_read_tif_float01(fn) for fn in slice_files_local]
    stack_cpu = torch.stack([torch.from_numpy(im) for im in imgs], dim=0).to(
        dtype=torch.float32
    )  # CPU [Z,H,W]

    if DEVICE.type == "cuda":
        stack_cpu = stack_cpu.contiguous().pin_memory()

    padded = (
        F.pad(
            stack_cpu,
            (BUFFER, BUFFER, BUFFER, BUFFER),
            mode="constant",
            value=0.0,
        )
        .contiguous()
        .to(DEVICE, non_blocking=True)
    )  # [Z,Hp,Wp]

    unfold = make_unfold_view(padded, _K)  # [Z, H, W, K, K]
    H = int(stack_cpu.shape[1])
    W = int(stack_cpu.shape[2])
    unfold_flat = unfold.reshape(Z_DIM, H * W, _K, _K)

    not_border_local = np.zeros(mask_local.shape, dtype=bool)
    not_border_local[
        BUFFER : mask_local.shape[0] - BUFFER, BUFFER : mask_local.shape[1] - BUFFER
    ] = True
    valid = mask_local & not_border_local
    coords_np = np.argwhere(valid)  # (y,x)

    prob_map = torch.zeros(
        (mask_local.shape[0], mask_local.shape[1]), device="cpu", dtype=torch.float32
    )

    coords_lin_np = (coords_np[:, 0] * W + coords_np[:, 1]).astype(np.int64, copy=False)
    ys_cpu = coords_np[:, 0]
    xs_cpu = coords_np[:, 1]

    infer_bs = 1024 if DEVICE.type == "cuda" else BATCH_SIZE

    model.eval()
    with torch.no_grad():
        for start in tqdm(
            range(0, len(coords_np), infer_bs),
            desc=f"Predict {os.path.basename(fragment_dir)}",
            leave=False,
            mininterval=0.5,
        ):
            end = min(start + infer_bs, len(coords_np))
            lin_dev = torch.from_numpy(coords_lin_np[start:end]).to(
                device=DEVICE, dtype=torch.int64, non_blocking=True
            )
            patches = torch.index_select(unfold_flat, 1, lin_dev)  # [Z,b,K,K]
            subvolumes = patches.permute(1, 0, 2, 3).unsqueeze(1)  # [b,1,Z,K,K]
            preds = model(subvolumes).view(-1).detach().cpu()
            prob_map[ys_cpu[start:end], xs_cpu[start:end]] = preds

    bin_mask = (prob_map.numpy() > threshold) & mask_local
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
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_11/3931268539.py in <cell line: 0>()
    109 for fid in test_ids:
    110     frag_dir = os.path.join(TEST_ROOT, fid)
--> 111     preds.append(predict_fragment_rle(frag_dir, THRESHOLD))
    112 
    113 sub_df = pd.DataFrame({"Id": test_ids, "Predicted": preds})

/tmp/ipykernel_11/3931268539.py in predict_fragment_rle(fragment_dir, threshold)
     61     H = int(stack_cpu.shape[1])
     62     W = int(stack_cpu.shape[2])
---> 63     unfold_flat = unfold.reshape(Z_DIM, H * W, _K, _K)
     64 
     65     not_border_local = np.zeros(mask_local.shape, dtype=bool)

OutOfMemoryError: CUDA out of memory. Tried to allocate 5534.17 GiB. GPU 0 has a total capacity of 47.53 GiB of which 41.54 GiB is free. Process 2479991 has 5.97 GiB memory in use. Of the allocated memory 5.64 GiB is allocated by PyTorch, and 22.15 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
