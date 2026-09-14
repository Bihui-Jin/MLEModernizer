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
import os, glob, random
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_candidate_prefixes = [
    "/kaggle/input/vesuvius-challenge/train/1/",
    "/kaggle/input/vesuvius-challenge-ink-detection/train/1/",
    "/kaggle/data/vesuvius-challenge-ink-detection/train/1/",
    "/kaggle/data/train/1/",
]
PREFIX = next(
    (p for p in _candidate_prefixes if os.path.exists(p)), _candidate_prefixes[0]
)

_candidate_roots = [
    "/kaggle/input/vesuvius-challenge/",
    "/kaggle/input/vesuvius-challenge-ink-detection/",
    "/kaggle/data/vesuvius-challenge-ink-detection/",
    "/kaggle/data/",
]
DATA_ROOT = next(
    (r for r in _candidate_roots if os.path.exists(os.path.join(r, "test"))),
    _candidate_roots[0],
)

BUFFER = 30
Z_START = 27
Z_DIM = 10
TRAINING_STEPS = 20000
LEARNING_RATE = 0.05
BATCH_SIZE = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != "Batch":
    plt.imshow(Image.open(PREFIX + "ir.png"), cmap="gray")
    plt.axis("off")
    plt.show()



## === cell 1
mask = np.array(Image.open(PREFIX + "mask.png").convert("1"))
label = (
    torch.from_numpy(np.array(Image.open(PREFIX + "inklabels.png")))
    .gt(0)
    .float()
    .to(DEVICE)
)

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != "Batch":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title("mask.png")
    ax1.imshow(mask, cmap="gray")
    ax1.axis("off")
    ax2.set_title("inklabels.png")
    ax2.imshow(label.detach().cpu(), cmap="gray")
    ax2.axis("off")
    plt.show()



## === cell 2
slice_files_1 = sorted(glob.glob(PREFIX + "surface_volume/*.tif"))[
    Z_START : Z_START + Z_DIM
]

_img_open = Image.open
images = [
    np.array(_img_open(fn), dtype=np.float32) / 65535.0
    for fn in (
        tqdm(slice_files_1)
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != "Batch"
        else slice_files_1
    )
]
image_stack = torch.stack([torch.from_numpy(im) for im in images], dim=0).to(DEVICE)

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != "Batch":
    fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
    for image, ax in zip(images, axes):
        ax.imshow(
            np.array(
                Image.fromarray(image).resize(
                    (image.shape[1] // 20, image.shape[0] // 20)
                ),
                dtype=np.float32,
            ),
            cmap="gray",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()



## === cell 3
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != "Batch":
    fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
    for image, ax in zip(images, axes):
        ax.imshow(
            np.array(
                Image.fromarray(image).resize(
                    (image.shape[1] // 20, image.shape[0] // 20)
                ),
                dtype=np.float32,
            ),
            cmap="prism",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()



## === cell 4
rect = (1100, 3500, 700, 950)

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != "Batch":
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
    ax.axis("off")
    plt.show()




## === cell 5
class SubvolumeDataset(data.Dataset):
    def __init__(
        self, image_stack: torch.Tensor, label: torch.Tensor, pixels: np.ndarray
    ):
        """
        image_stack: (Z_DIM, H, W) on DEVICE
        label: (H, W) on DEVICE
        pixels: numpy array of shape (N, 2) with [y,x]
        """
        self.label = label
        self.pixels = pixels

        padded = torch.nn.functional.pad(image_stack, (BUFFER, BUFFER, BUFFER, BUFFER))
        self.windows = padded.unfold(1, BUFFER * 2 + 1, 1).unfold(2, BUFFER * 2 + 1, 1)

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        y, x = self.pixels[index]
        subvolume = self.windows[:, y, x].unsqueeze(0)
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
print("Generating pixel lists...")
not_border = np.zeros(mask.shape, dtype=bool)
not_border[BUFFER : mask.shape[0] - BUFFER, BUFFER : mask.shape[1] - BUFFER] = True
arr_mask = np.array(mask).astype(bool) & not_border

inside_rect = np.zeros(mask.shape, dtype=bool)
inside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = True
inside_rect = inside_rect & arr_mask

outside_rect = arr_mask.copy()
outside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = False

pixels_inside_rect = np.argwhere(inside_rect)
pixels_outside_rect = np.argwhere(outside_rect)

print("Training...")

criterion = nn.BCELoss()
optimizer = optim.ASGD(model.parameters(), lr=LEARNING_RATE)  # core logic preserved

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
INITIAL_LEARNING_RATE = 0.01
min_lr = 0.0001
lambda1 = lambda epoch: max(0.99**epoch, min_lr / INITIAL_LEARNING_RATE)
scheduler7 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
scheduler8 = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=4, power=1.0)
scheduler9 = optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=0.001, max_lr=0.1, mode="triangular2", cycle_momentum=False
)
scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        scheduler1,
        scheduler2,
        scheduler3,
        scheduler4,
        scheduler5,
        scheduler6,
        scheduler7,
        scheduler8,
        scheduler9,
    ]
)

padded = torch.nn.functional.pad(image_stack, (BUFFER, BUFFER, BUFFER, BUFFER))
windows = padded.unfold(1, BUFFER * 2 + 1, 1).unfold(2, BUFFER * 2 + 1, 1)

pixels_outside_t = torch.from_numpy(pixels_outside_rect).to(
    device=DEVICE, dtype=torch.long
)

model.train()
n_train = pixels_outside_t.shape[0]
steps_per_epoch = (n_train + BATCH_SIZE - 1) // BATCH_SIZE

gen = torch.Generator(device="cpu").manual_seed(SEED)


def _precompute_lr_sequence(train_steps: int):
    dummy_param = torch.nn.Parameter(torch.tensor(0.0))
    opt_d = optim.ASGD([dummy_param], lr=LEARNING_RATE)
    s1 = torch.optim.lr_scheduler.ConstantLR(opt_d, factor=0.1, total_iters=2)
    s2 = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=0.9)
    s3 = torch.optim.lr_scheduler.StepLR(opt_d, step_size=30, gamma=0.1)
    lm = lambda epoch: 0.95
    s4 = torch.optim.lr_scheduler.MultiplicativeLR(opt_d, lr_lambda=lm)
    s5 = torch.optim.lr_scheduler.LinearLR(opt_d, start_factor=0.5, total_iters=4)
    s6 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_d, T_0=30, T_mult=1)
    ILR = 0.01
    mlr = 0.0001
    lam1 = lambda epoch: max(0.99**epoch, mlr / ILR)
    s7 = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda=[lam1])
    s8 = torch.optim.lr_scheduler.PolynomialLR(opt_d, total_iters=4, power=1.0)
    s9 = optim.lr_scheduler.CyclicLR(
        opt_d, base_lr=0.001, max_lr=0.1, mode="triangular2", cycle_momentum=False
    )
    sch_d = torch.optim.lr_scheduler.ChainedScheduler(
        [s1, s2, s3, s4, s5, s6, s7, s8, s9]
    )
    lrs = np.empty(train_steps, dtype=np.float64)
    for i in range(train_steps):
        sch_d.step()
        lrs[i] = opt_d.param_groups[0]["lr"]
    return lrs


_lr_seq = _precompute_lr_sequence(TRAINING_STEPS)

pbar = tqdm(range(TRAINING_STEPS), total=TRAINING_STEPS)
for step in pbar:
    if step % steps_per_epoch == 0:
        perm = torch.randperm(n_train, generator=gen, device="cpu").to(DEVICE)

    start = (step % steps_per_epoch) * BATCH_SIZE
    end = min(start + BATCH_SIZE, n_train)
    idx = perm[start:end]

    coords = pixels_outside_t.index_select(0, idx)  # (bs,2)
    ys = coords[:, 0]
    xs = coords[:, 1]

    subvolumes = windows[:, ys, xs].permute(1, 0, 2, 3).unsqueeze(1).contiguous()
    inklabels = label[ys, xs].view(-1, 1)

    optimizer.param_groups[0]["lr"] = float(_lr_seq[step])

    optimizer.zero_grad(set_to_none=True)
    outputs = model(subvolumes)
    loss = criterion(outputs, inklabels)
    loss.backward()
    optimizer.step()




## === cell 7
pixels_inside_t = torch.from_numpy(pixels_inside_rect).to(
    device=DEVICE, dtype=torch.long
)

output = torch.zeros_like(label).float()
model.eval()
with torch.no_grad():
    n_eval = pixels_inside_t.shape[0]
    for start in tqdm(
        range(0, n_eval, BATCH_SIZE),
        disable=(os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") == "Batch"),
    ):
        end = min(start + BATCH_SIZE, n_eval)
        coords = pixels_inside_t[start:end]
        ys = coords[:, 0]
        xs = coords[:, 1]

        subvolumes = windows[:, ys, xs].permute(1, 0, 2, 3).unsqueeze(1).contiguous()
        preds = model(subvolumes).view(-1)
        output[ys, xs] = preds

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != "Batch":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(output.detach().cpu(), cmap="gray")
    ax1.set_title("model output (train/1 rect)")
    ax1.axis("off")
    ax2.imshow(label.detach().cpu(), cmap="gray")
    ax2.set_title("label (train/1)")
    ax2.axis("off")
    plt.show()



## === cell 8
THRESHOLD = 0.4

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != "Batch":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(output.gt(THRESHOLD).detach().cpu(), cmap="gray")
    ax1.set_title("thresholded output (train/1 rect)")
    ax1.axis("off")
    ax2.imshow(label.detach().cpu(), cmap="gray")
    ax2.set_title("label (train/1)")
    ax2.axis("off")
    plt.show()




## === cell 9
def rle_from_binary_mask(mask_2d_bool: np.ndarray) -> str:
    pixels = mask_2d_bool.astype(np.uint8).reshape(-1)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


_STACK_CACHE = {}


def load_image_stack(fragment_dir: str) -> torch.Tensor:
    if fragment_dir in _STACK_CACHE:
        return _STACK_CACHE[fragment_dir]
    slice_files = sorted(
        glob.glob(os.path.join(fragment_dir, "surface_volume", "*.tif"))
    )
    slice_files = slice_files[Z_START : Z_START + Z_DIM]
    _img_open_local = Image.open
    imgs = [
        np.array(_img_open_local(f), dtype=np.float32) / 65535.0 for f in slice_files
    ]
    stack = torch.stack([torch.from_numpy(im) for im in imgs], dim=0)
    stack = stack.to(DEVICE)
    _STACK_CACHE[fragment_dir] = stack
    return stack




## === cell 10
class PixelOnlyDataset(data.Dataset):
    def __init__(self, image_stack: torch.Tensor, pixels: np.ndarray):
        self.pixels = pixels
        padded = torch.nn.functional.pad(image_stack, (BUFFER, BUFFER, BUFFER, BUFFER))
        self.windows = padded.unfold(1, BUFFER * 2 + 1, 1).unfold(2, BUFFER * 2 + 1, 1)

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        y, x = self.pixels[index]
        subvolume = self.windows[:, y, x].unsqueeze(0)
        return subvolume, (y, x)


def predict_fragment(fragment_dir: str, threshold: float = THRESHOLD) -> str:
    frag_mask = np.array(
        Image.open(os.path.join(fragment_dir, "mask.png")).convert("1")
    ).astype(bool)

    not_border = np.zeros(frag_mask.shape, dtype=bool)
    not_border[
        BUFFER : frag_mask.shape[0] - BUFFER, BUFFER : frag_mask.shape[1] - BUFFER
    ] = True
    valid = frag_mask & not_border
    pixels = np.argwhere(valid)

    stack = load_image_stack(fragment_dir)

    padded = torch.nn.functional.pad(stack, (BUFFER, BUFFER, BUFFER, BUFFER))
    windows_local = padded.unfold(1, BUFFER * 2 + 1, 1).unfold(2, BUFFER * 2 + 1, 1)

    pixels_t = torch.from_numpy(pixels).to(device=DEVICE, dtype=torch.long)

    pred_mask = np.zeros(frag_mask.shape, dtype=bool)
    model.eval()
    with torch.no_grad():
        n = pixels_t.shape[0]
        for start in tqdm(
            range(0, n, BATCH_SIZE),
            desc=f"predict {os.path.basename(fragment_dir)}",
            leave=False,
            disable=True,  # tqdm is costly; disable for timed runs (no effect on predictions)
        ):
            end = min(start + BATCH_SIZE, n)
            coords = pixels_t[start:end]
            ys = coords[:, 0]
            xs = coords[:, 1]

            subvolumes = (
                windows_local[:, ys, xs].permute(1, 0, 2, 3).unsqueeze(1).contiguous()
            )
            probs = model(subvolumes).view(-1)

            keep = (probs > threshold).detach().cpu().numpy()
            yx = coords.detach().cpu().numpy()
            pred_mask[yx[:, 0], yx[:, 1]] = keep

    pred_mask &= frag_mask
    return rle_from_binary_mask(pred_mask)


test_dir = os.path.join(DATA_ROOT, "test")
test_ids = sorted(
    [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
)

rows = []
for fid in test_ids:
    frag_dir = os.path.join(test_dir, fid)
    pred_rle = predict_fragment(frag_dir, threshold=THRESHOLD)
    rows.append((fid, pred_rle))

sub = pd.DataFrame(rows, columns=["Id", "Predicted"])
sub.to_csv("submission.csv", index=False)
print(sub.head())
print(
    "Wrote submission.csv with", len(sub), "rows to", os.path.abspath("submission.csv")
)
