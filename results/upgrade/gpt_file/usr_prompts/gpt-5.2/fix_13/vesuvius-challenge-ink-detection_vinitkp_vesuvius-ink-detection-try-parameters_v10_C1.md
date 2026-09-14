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

0.038431

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

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BUFFER = 30  # Buffer size in x and y direction
Z_START = 27  # First slice in the z direction to use
Z_DIM = 10  # Number of slices in the z direction
TRAINING_STEPS = 20000
LEARNING_RATE = 0.05
BATCH_SIZE = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DL_GEN = torch.Generator()
DL_GEN.manual_seed(0)

NUM_WORKERS = min(4, os.cpu_count() or 1)
PREFETCH_FACTOR = 4 if NUM_WORKERS > 0 else 2


def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)


def find_comp_root():
    candidates = [
        "/kaggle/input/vesuvius-challenge-ink-detection",
        "/kaggle/input/vesuvius-challenge-ink-detection/vesuvius-challenge-ink-detection",
        "/kaggle/data/vesuvius-challenge-ink-detection",
        "/kaggle/data/vesuvius-challenge-ink-detection/vesuvius-challenge-ink-detection",
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "train")) and os.path.isdir(
            os.path.join(c, "test")
        ):
            return c
    for c in glob.glob("/kaggle/input/*"):
        if os.path.isdir(os.path.join(c, "train")) and os.path.isdir(
            os.path.join(c, "test")
        ):
            return c
        cc = os.path.join(c, "vesuvius-challenge-ink-detection")
        if os.path.isdir(os.path.join(cc, "train")) and os.path.isdir(
            os.path.join(cc, "test")
        ):
            return cc
    raise FileNotFoundError(
        "Could not find competition root containing train/ and test/ under /kaggle/input."
    )


COMP_ROOT = find_comp_root()
TRAIN_ROOT = os.path.join(COMP_ROOT, "train")
TEST_ROOT = os.path.join(COMP_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(COMP_ROOT, "sample_submission.csv")
if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = (
        "/kaggle/input/sample_submission.csv"
        if os.path.exists("/kaggle/input/sample_submission.csv")
        else SAMPLE_SUB_PATH
    )

print("COMP_ROOT:", COMP_ROOT)
print("DEVICE:", DEVICE)
print("NUM_WORKERS:", NUM_WORKERS)

PREFIX = os.path.join(TRAIN_ROOT, "1") + "/"

ir_path = os.path.join(PREFIX, "ir.png")
if os.path.exists(ir_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(Image.open(ir_path), cmap="gray")
    plt.title("train/1/ir.png")
    plt.axis("off")
    plt.show()
else:
    print("Warning: ir.png not found at", ir_path)



## === cell 1
mask_path = os.path.join(PREFIX, "mask.png")
label_path = os.path.join(PREFIX, "inklabels.png")

mask = np.array(Image.open(mask_path).convert("1"))
label = torch.from_numpy(np.array(Image.open(label_path))).gt(0).float()  # CPU

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title("mask.png")
ax1.imshow(mask, cmap="gray")
ax1.axis("off")
ax2.set_title("inklabels.png")
ax2.imshow(label, cmap="gray")
ax2.axis("off")
plt.tight_layout()
plt.show()



## === cell 2
sv_dir = os.path.join(PREFIX, "surface_volume")
tif_files = sorted(glob.glob(os.path.join(sv_dir, "*.tif")))
if len(tif_files) == 0:
    raise FileNotFoundError(f"No .tif files found under {sv_dir}")

use_files = tif_files[Z_START : Z_START + Z_DIM]
if len(use_files) < Z_DIM:
    raise RuntimeError(
        f"Not enough tif slices: requested {Z_DIM}, got {len(use_files)} from {sv_dir}"
    )

images = [
    np.array(Image.open(fn), dtype=np.float32) / 65535.0
    for fn in tqdm(use_files, desc="Loading slices")
]
image_stack = torch.from_numpy(np.stack(images, axis=0))  # (Z,H,W), CPU

fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
for image, ax in zip(images, axes):
    small = np.array(
        Image.fromarray((image * 255).astype(np.uint8)).resize(
            (image.shape[1] // 20, image.shape[0] // 20)
        )
    )
    ax.imshow(small, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.show()



## === cell 3
rect = (1100, 3500, 700, 950)
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(label, cmap="gray")
patch = patches.Rectangle(
    (rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor="r", facecolor="none"
)
ax.add_patch(patch)
ax.set_title("Training label with rect")
ax.axis("off")
plt.show()



## === cell 4
K = BUFFER * 2 + 1


def _coords_to_linear(coords_yx: np.ndarray, width: int) -> np.ndarray:
    return coords_yx[:, 0].astype(np.int64) * width + coords_yx[:, 1].astype(np.int64)


def _make_all_patches_view(image_stack_zhw: torch.Tensor, buffer: int) -> torch.Tensor:
    """
    Returns: (Z, Hc, Wc, K, K) where Hc=H-2*buffer, Wc=W-2*buffer.
    """
    assert image_stack_zhw.ndim == 3
    Z, H, W = image_stack_zhw.shape
    Kloc = buffer * 2 + 1
    Hc = H - 2 * buffer
    Wc = W - 2 * buffer
    if Hc <= 0 or Wc <= 0:
        raise ValueError("BUFFER too large for given image size")
    patches = image_stack_zhw.unfold(1, Kloc, 1).unfold(
        2, Kloc, 1
    )  # (Z,H-K+1,W-K+1,K,K)
    return patches


_PATCH_VIEW_CACHE = {}


def _get_patches_view_cached(
    image_stack_zhw: torch.Tensor, buffer: int
) -> torch.Tensor:
    key = (int(image_stack_zhw.data_ptr()), tuple(image_stack_zhw.shape), int(buffer))
    pv = _PATCH_VIEW_CACHE.get(key)
    if pv is None:
        pv = _make_all_patches_view(image_stack_zhw, buffer)
        _PATCH_VIEW_CACHE[key] = pv
    return pv


def _filter_valid_pixels(
    pixels_yx: np.ndarray, H: int, W: int, buffer: int
) -> np.ndarray:
    if pixels_yx.size == 0:
        return pixels_yx.astype(np.int64, copy=False).reshape(0, 2)
    p = pixels_yx.astype(np.int64, copy=False)
    y = p[:, 0]
    x = p[:, 1]
    ok = (y >= buffer) & (y < (H - buffer)) & (x >= buffer) & (x < (W - buffer))
    return p[ok]


class SubvolumeDatasetFast(data.Dataset):
    """
    Vectorized batch gather from the cached unfold-view to avoid per-sample __getitem__ overhead.
    """

    def __init__(
        self, image_stack: torch.Tensor, label: torch.Tensor, pixels: np.ndarray
    ):
        assert image_stack.ndim == 3, "image_stack must be (Z,H,W)"
        self.image_stack = image_stack.contiguous()  # CPU
        self.label = label  # CPU
        self.H = int(image_stack.shape[1])
        self.W = int(image_stack.shape[2])

        self.pixels = _filter_valid_pixels(pixels, self.H, self.W, BUFFER)

        self._patches = _get_patches_view_cached(
            self.image_stack, BUFFER
        )  # (Z,Hc,Wc,K,K)

        self._yc = torch.from_numpy((self.pixels[:, 0] - BUFFER).astype(np.int64))
        self._xc = torch.from_numpy((self.pixels[:, 1] - BUFFER).astype(np.int64))
        self._y = torch.from_numpy(self.pixels[:, 0].astype(np.int64))
        self._x = torch.from_numpy(self.pixels[:, 1].astype(np.int64))

        self._coords_lin = torch.from_numpy(_coords_to_linear(self.pixels, self.W))

        if NUM_WORKERS > 0:
            try:
                self._patches = self._patches.share_memory_()
                self.label = self.label.share_memory_()
                self._yc = self._yc.share_memory_()
                self._xc = self._xc.share_memory_()
                self._y = self._y.share_memory_()
                self._x = self._x.share_memory_()
                self._coords_lin = self._coords_lin.share_memory_()
            except Exception:
                pass

    def __len__(self):
        return int(self._yc.numel())

    def __getitem__(self, index):
        yc = int(self._yc[index])
        xc = int(self._xc[index])
        patch = self._patches[:, yc, xc].unsqueeze(0)  # (1,Z,K,K)
        y = int(self._y[index])
        x = int(self._x[index])
        inklabel = self.label[y, x].view(1)
        return patch, inklabel

    def get_batch(self, indices: torch.Tensor):
        yc = self._yc.index_select(0, indices)
        xc = self._xc.index_select(0, indices)
        patches = self._patches[:, yc, xc].permute(1, 0, 2, 3).contiguous()
        patches = patches.unsqueeze(1)  # (B,1,Z,K,K)
        y = self._y.index_select(0, indices)
        x = self._x.index_select(0, indices)
        inklabels = self.label[y, x].unsqueeze(1)  # (B,1)
        return patches, inklabels

    @property
    def coords_lin(self):
        return self._coords_lin


class _IndexDataset(data.Dataset):
    def __init__(self, n: int):
        self.n = int(n)

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        return i


def _make_fast_loader(ds: SubvolumeDatasetFast, shuffle: bool):
    index_ds = _IndexDataset(len(ds))
    sampler = (
        data.RandomSampler(index_ds, generator=DL_GEN)
        if shuffle
        else data.SequentialSampler(index_ds)
    )
    batch_sampler = data.BatchSampler(sampler, batch_size=BATCH_SIZE, drop_last=False)

    def collate_fn(batch_indices):
        idx = torch.as_tensor(batch_indices, dtype=torch.int64)
        return ds.get_batch(idx)

    return data.DataLoader(
        index_ds,
        batch_sampler=batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        collate_fn=collate_fn,
    )


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

H0, W0 = int(image_stack.shape[1]), int(image_stack.shape[2])
print("Train fragment stack shape:", tuple(image_stack.shape), "H,W:", (H0, W0))

if DEVICE.type == "cuda":
    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=False)
        print("torch.compile enabled")
    except Exception as e:
        print("torch.compile not enabled:", repr(e))



## === cell 5
print("Generating pixel lists...")

not_border = np.zeros(mask.shape, dtype=bool)
not_border[BUFFER : mask.shape[0] - BUFFER, BUFFER : mask.shape[1] - BUFFER] = True

arr_mask = mask.astype(bool) & not_border

inside_rect = np.zeros(mask.shape, dtype=bool)
y0, x0, w, h = rect[1], rect[0], rect[2], rect[3]
inside_rect[y0 : y0 + h, x0 : x0 + w] = True
inside_rect &= arr_mask

outside_rect = arr_mask.copy()
outside_rect[y0 : y0 + h, x0 : x0 + w] = False

pixels_inside_rect = np.argwhere(inside_rect)
pixels_outside_rect = np.argwhere(outside_rect)

H, W = mask.shape
pixels_inside_rect = _filter_valid_pixels(pixels_inside_rect, H, W, BUFFER)
pixels_outside_rect = _filter_valid_pixels(pixels_outside_rect, H, W, BUFFER)

print("pixels_inside_rect:", len(pixels_inside_rect))
print("pixels_outside_rect:", len(pixels_outside_rect))

if len(pixels_outside_rect) == 0:
    raise RuntimeError(
        "No training pixels found (outside_rect empty). Check mask/BUFFER/rect."
    )



## === cell 6
print("Training...")

train_dataset = SubvolumeDatasetFast(image_stack, label, pixels_outside_rect)
train_loader = _make_fast_loader(train_dataset, shuffle=True)

criterion = nn.BCELoss()
optimizer = optim.ASGD(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=4)


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


use_cuda_graphs = DEVICE.type == "cuda"
model.train()
it = infinite_loader(train_loader)

pbar = tqdm(range(TRAINING_STEPS), total=TRAINING_STEPS, miniters=200, mininterval=1.0)

if use_cuda_graphs:
    first_subvol, first_lbl = next(it)
    if first_subvol.shape[0] == BATCH_SIZE:
        static_x = first_subvol.to(DEVICE, non_blocking=True)
        static_y = first_lbl.to(DEVICE, non_blocking=True)
        static_out = None
        static_loss = None

        g = torch.cuda.CUDAGraph()
        optimizer.zero_grad(set_to_none=True)
        out = model(static_x)
        loss = criterion(out, static_y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            optimizer.zero_grad(set_to_none=True)
            static_out = model(static_x)
            static_loss = criterion(static_out, static_y)
            static_loss.backward()
            optimizer.step()
            scheduler.step()

        for i in pbar:
            subvolumes, inklabels = next(it)
            if subvolumes.shape[0] != BATCH_SIZE:
                optimizer.zero_grad(set_to_none=True)
                outputs = model(subvolumes.to(DEVICE, non_blocking=True))
                loss = criterion(outputs, inklabels.to(DEVICE, non_blocking=True))
                loss.backward()
                optimizer.step()
                scheduler.step()
            else:
                static_x.copy_(subvolumes, non_blocking=True)
                static_y.copy_(inklabels, non_blocking=True)
                g.replay()
                loss = static_loss  # for logging
            if (i + 1) % 200 == 0:
                pbar.set_postfix(loss=float(loss.detach().cpu()))
    else:
        print("CUDA Graphs disabled (first batch not full); using eager loop.")
        for i in pbar:
            subvolumes, inklabels = next(it)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(subvolumes.to(DEVICE, non_blocking=True))
            loss = criterion(outputs, inklabels.to(DEVICE, non_blocking=True))
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (i + 1) % 200 == 0:
                pbar.set_postfix(loss=float(loss.detach().cpu()))
else:
    for i in pbar:
        subvolumes, inklabels = next(it)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(subvolumes.to(DEVICE, non_blocking=True))
        loss = criterion(outputs, inklabels.to(DEVICE, non_blocking=True))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (i + 1) % 200 == 0:
            pbar.set_postfix(loss=float(loss.detach().cpu()))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/810219179.py in <cell line: 0>()
     50             optimizer.zero_grad(set_to_none=True)
---> 51             static_out = model(static_x)
     52             static_loss = criterion(static_out, static_y)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/external_utils.py in inner(*args, **kwargs)
     42 
---> 43     @functools.wraps(fn)
     44     def inner(*args: Any, **kwargs: Any) -> Any:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    744             try:
--> 745                 return fn(*args, **kwargs)
    746             finally:

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in forward(*runtime_args)
   1183         full_args.extend(runtime_args)
-> 1184         return compiled_fn(full_args)
   1185 

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in runtime_wrapper(args)
    309             ), torch.enable_grad():
--> 310                 all_outs = call_func_at_runtime_with_args(
    311                     compiled_fn, args_, disable_amp=disable_amp, steal_args=True

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in call_func_at_runtime_with_args(f, args, steal_args, disable_amp)
    125         if hasattr(f, "_boxed_call"):
--> 126             out = normalize_as_list(f(args))
    127         else:

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in g(args)
     99     def g(args):
--> 100         return f(*args)
    101 

/usr/local/lib/python3.11/dist-packages/torch/autograd/function.py in apply(cls, *args, **kwargs)
    574             args = _functorch.utils.unwrap_dead_wrappers(args)
--> 575             return super().apply(*args, **kwargs)  # type: ignore[misc]
    576 

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in forward(ctx, *deduped_flat_tensor_args)
   1584                 #   in the fw output order.
-> 1585                 fw_outs = call_func_at_runtime_with_args(
   1586                     CompiledFunction.compiled_fw,

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in call_func_at_runtime_with_args(f, args, steal_args, disable_amp)
    125         if hasattr(f, "_boxed_call"):
--> 126             out = normalize_as_list(f(args))
    127         else:

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in wrapper(runtime_args)
    489                 return out
--> 490             return compiled_fn(runtime_args)
    491 

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in inner_fn(args)
    671 
--> 672             outs = compiled_fn(args)
    673 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/output_code.py in __call__(self, inputs)
    465         try:
--> 466             return self.current_callable(inputs)
    467         finally:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in run(new_inputs)
   1207                 compiled_fn = cudagraphify_fn(model, new_inputs, static_input_idxs)
-> 1208         return compiled_fn(new_inputs)
   1209 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in deferred_cudagraphify(inputs)
    381         if fn is not None:
--> 382             return fn(inputs)
    383 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/utils.py in run(new_inputs)
   2127         copy_misaligned_inputs(new_inputs, inputs_to_check)
-> 2128         return model(new_inputs)
   2129 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in run(self, new_inputs, function_id)
   1946         self.mode = self.id_to_mode[function_id]
-> 1947         out = self._run(new_inputs, function_id)
   1948 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in _run(self, new_inputs, function_id)
   2126         ):
-> 2127             out = self.record_function(new_inputs, function_id)
   2128 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in record_function(self, new_inputs, function_id)
   2161         )
-> 2162         torch.cuda.synchronize()
   2163         node = CUDAGraphNode(

/usr/local/lib/python3.11/dist-packages/torch/cuda/__init__.py in synchronize(device)
    984     with torch.cuda.device(device):
--> 985         return torch._C._cuda_synchronize()
    986 

RuntimeError: CUDA error: operation not permitted when stream is capturing
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


During handling of the above exception, another exception occurred:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/810219179.py in <cell line: 0>()
     47         optimizer.zero_grad(set_to_none=True)
     48         torch.cuda.synchronize()
---> 49         with torch.cuda.graph(g):
     50             optimizer.zero_grad(set_to_none=True)
     51             static_out = model(static_x)

/usr/local/lib/python3.11/dist-packages/torch/cuda/graphs.py in __exit__(self, exc_type, exc_value, traceback)
    184 
    185     def __exit__(self, exc_type, exc_value, traceback):
--> 186         self.cuda_graph.capture_end()
    187         self.stream_ctx.__exit__(exc_type, exc_value, traceback)
    188         # returning None should propagate exceptions from either capture_end or stream_ctx.__exit__()

/usr/local/lib/python3.11/dist-packages/torch/cuda/graphs.py in capture_end(self)
     82         which call ``capture_end`` internally.
     83         """
---> 84         super().capture_end()
     85 
     86     def replay(self):

RuntimeError: CUDA error: operation failed due to a previous error during capture
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


## === cell 7
eval_dataset = SubvolumeDatasetFast(image_stack, label, pixels_inside_rect)
eval_loader = _make_fast_loader(eval_dataset, shuffle=False)

output = torch.zeros_like(label).float()  # CPU
model.eval()

H, W = output.shape
coords_lin_all_t = eval_dataset.coords_lin  # cached

with torch.inference_mode():
    offset = 0
    for subvolumes, _ in tqdm(eval_loader, desc="Eval rect"):
        preds = model(subvolumes.to(DEVICE, non_blocking=True)).detach().cpu().view(-1)
        n = preds.numel()
        output.view(-1)[coords_lin_all_t[offset : offset + n]] = preds.to(output.dtype)
        offset += n

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title("Model output (rect area only)")
ax1.imshow(output, cmap="gray")
ax1.axis("off")
ax2.set_title("Label")
ax2.imshow(label, cmap="gray")
ax2.axis("off")
plt.tight_layout()
plt.show()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
BackendCompilerFailed                     Traceback (most recent call last)
/tmp/ipykernel_11/136143896.py in <cell line: 0>()
     11     offset = 0
     12     for subvolumes, _ in tqdm(eval_loader, desc="Eval rect"):
---> 13         preds = model(subvolumes.to(DEVICE, non_blocking=True)).detach().cpu().view(-1)
     14         n = preds.numel()
     15         output.view(-1)[coords_lin_all_t[offset : offset + n]] = preds.to(output.dtype)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, frame_state)
   1378         with compile_lock, _disable_current_modes():
   1379             # skip=1: skip this frame
-> 1380             return self._torchdynamo_orig_callable(
   1381                 frame, cache_entry, self.hooks, frame_state, skip=1
   1382             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
   1162         counters["frames"]["total"] += 1
   1163         try:
-> 1164             result = self._inner_convert(
   1165                 frame, cache_entry, hooks, frame_state, skip=skip + 1
   1166             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
    545 
    546         with compile_context(CompileContext(compile_id)):
--> 547             return _compile(
    548                 frame.f_code,
    549                 frame.f_globals,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile(code, globals, locals, builtins, closure, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)
    984         guarded_code = None
    985         try:
--> 986             guarded_code = compile_inner(code, one_graph, hooks, transform)
    987 
    988             # NB: We only put_code_state in success case.  Success case here

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in compile_inner(code, one_graph, hooks, transform)
    713             stack.enter_context(torch._dynamo.callback_handler.install_callbacks())
    714             stack.enter_context(CompileTimeInstructionCounter.record())
--> 715             return _compile_inner(code, one_graph, hooks, transform)
    716 
    717         return None  # dead, but see https://github.com/python/mypy/issues/7577

/usr/local/lib/python3.11/dist-packages/torch/_utils_internal.py in wrapper_function(*args, **kwargs)
     93 
     94             if not StrobelightCompileTimeProfiler.enabled:
---> 95                 return function(*args, **kwargs)
     96 
     97             return StrobelightCompileTimeProfiler.profile_compile_time(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile_inner(code, one_graph, hooks, transform)
    748             CompileContext.get().attempt = attempt
    749             try:
--> 750                 out_code = transform_code_object(code, transform)
    751                 break
    752             except exc.RestartAnalysis as e:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/bytecode_transformation.py in transform_code_object(code, transformations, safe)
   1359     propagate_line_nums(instructions)
   1360 
-> 1361     transformations(instructions, code_options)
   1362     return clean_and_assemble_instructions(instructions, keys, code_options)[1]
   1363 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _fn(*args, **kwargs)
    229             exit_stack.enter_context(torch_function_mode_stack_state_mgr)
    230             try:
--> 231                 return fn(*args, **kwargs)
    232             finally:
    233                 cleanup.close()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in transform(instructions, code_options)
    660         try:
    661             with tracing(tracer.output.tracing_context), tracer.set_current_tx():
--> 662                 tracer.run()
    663         except exc.UnspecializeRestartAnalysis:
    664             speculation_log.clear()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   2866 
   2867     def run(self):
-> 2868         super().run()
   2869 
   2870     def should_compile_partial_graph(self):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in RETURN_VALUE(self, inst)
   3046 
   3047     def RETURN_VALUE(self, inst):
-> 3048         self._return(inst)
   3049 
   3050     def RETURN_CONST(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _return(self, inst)
   3031         )
   3032         log.debug("%s triggered compile", inst.opname)
-> 3033         self.output.compile_subgraph(
   3034             self,
   3035             reason=GraphCompileReason(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_subgraph(self, tx, partial_convert, reason)
   1099             # optimization to generate better code in a common case
   1100             self.add_output_instructions(
-> 1101                 self.compile_and_call_fx_graph(
   1102                     tx, list(reversed(stack_values)), root, output_replacements
   1103                 )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_and_call_fx_graph(self, tx, rv, root, replaced_outputs)
   1380 
   1381             with self.restore_global_state():
-> 1382                 compiled_fn = self.call_user_compiler(gm)
   1383 
   1384             from torch.fx._lazy_graph_module import _LazyGraphModule

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in call_user_compiler(self, gm)
   1430             dynamo_compile_column_us="aot_autograd_cumulative_compile_time_us",
   1431         ):
-> 1432             return self._call_user_compiler(gm)
   1433 
   1434     def _call_user_compiler(self, gm: fx.GraphModule) -> CompiledFn:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1481             raise e
   1482         except Exception as e:
-> 1483             raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
   1484                 e.__traceback__
   1485             ) from None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1460             if config.verify_correctness:
   1461                 compiler_fn = WrapperBackend(compiler_fn)
-> 1462             compiled_fn = compiler_fn(gm, self.example_inputs())
   1463             _step_logger()(logging.INFO, f"done compiler function {name}")
   1464             assert callable(compiled_fn), "compiler_fn did not return callable"

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/repro/after_dynamo.py in __call__(self, gm, example_inputs, **kwargs)
    128                     raise
    129         else:
--> 130             compiled_gm = compiler_fn(gm, example_inputs)
    131 
    132         return compiled_gm

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in __call__(self, model_, inputs_)
   2338         from torch._inductor.compile_fx import compile_fx
   2339 
-> 2340         return compile_fx(model_, inputs_, config_patches=self.config)
   2341 
   2342     def get_compiler_config(self):

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1550     if config_patches:
   1551         with config.patch(config_patches):
-> 1552             return compile_fx(
   1553                 model_,
   1554                 example_inputs_,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1861             unlift_effect_tokens=True
   1862         ):
-> 1863             return aot_autograd(
   1864                 fw_compiler=fw_compiler,
   1865                 bw_compiler=bw_compiler,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/backends/common.py in __call__(self, gm, example_inputs, **kwargs)
     81             # NB: NOT cloned!
     82             with enable_aot_logging(), patch_config:
---> 83                 cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
     84                 counters["aot_autograd"]["ok"] += 1
     85                 return disable(cg)

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in aot_module_simplified(mod, args, fw_compiler, bw_compiler, partition_fn, decompositions, keep_inference_input_mutations, inference_compiler, cudagraphs)
   1153         )
   1154     else:
-> 1155         compiled_fn = dispatch_and_compile()
   1156 
   1157     if isinstance(mod, torch._dynamo.utils.GmWrapper):

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in dispatch_and_compile()
   1129         functional_call = create_functional_call(mod, params_spec, params_len)
   1130         with compiled_autograd._disable():
-> 1131             compiled_fn, _ = create_aot_dispatcher_function(
   1132                 functional_call,
   1133                 fake_flat_args,

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    578 ) -> Tuple[Callable, ViewAndMutationMeta]:
    579     with dynamo_timed("create_aot_dispatcher_function", log_pt2_compile_event=True):
--> 580         return _create_aot_dispatcher_function(
    581             flat_fn, fake_flat_args, aot_config, fake_mode, shape_env
    582         )

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in _create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    828         compiler_fn = choose_dispatcher(needs_autograd, aot_config)
    829 
--> 830         compiled_fn, fw_metadata = compiler_fn(
    831             flat_fn,
    832             _dup_fake_script_obj(fake_flat_args),

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py in aot_dispatch_base(flat_fn, flat_args, aot_config, fw_metadata)
    201                 assert isinstance(fw_module, GraphModule)
    202                 tensorify_python_scalars(fw_module, fake_mode.shape_env, fake_mode)
--> 203             compiled_fw = compiler(fw_module, updated_flat_args)
    204 
    205         if fakified_out_wrapper.needs_post_compile:

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in __call__(self, gm, example_inputs)
    487         example_inputs: Sequence[InputType],
    488     ) -> OutputCode:
--> 489         return self.compiler_fn(gm, example_inputs)
    490 
    491 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in fw_compiler_base(gm, example_inputs, is_inference)
   1739                     model_outputs_node.meta["user_visible_output_idxs"] = []
   1740 
-> 1741                 return inner_compile(
   1742                     gm,
   1743                     example_inputs,

/usr/lib/python3.11/contextlib.py in inner(*args, **kwds)
     79         def inner(*args, **kwds):
     80             with self._recreate_cm():
---> 81                 return func(*args, **kwds)
     82         return inner
     83 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx_inner(gm, example_inputs, **kwargs)
    567         )
    568 
--> 569         return wrap_compiler_debug(_compile_fx_inner, compiler_name="inductor")(
    570             gm,
    571             example_inputs,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/repro/after_aot.py in debug_wrapper(gm, example_inputs, **kwargs)
    100             # Call the compiler_fn - which is either aot_autograd or inductor
    101             # with fake inputs
--> 102             inner_compiled_fn = compiler_fn(gm, example_inputs)
    103         except Exception as e:
    104             # TODO: Failures here are troublesome because no real inputs,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in _compile_fx_inner(gm, example_inputs, **graph_kwargs)
    683             TritonBundler.begin_compile()
    684             try:
--> 685                 mb_compiled_graph = fx_codegen_and_compile(
    686                     gm, example_inputs, inputs_to_check, **graph_kwargs
    687                 )

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in fx_codegen_and_compile(gm, example_inputs, inputs_to_check, **graph_kwargs)
   1127     scheme: FxCompile = _InProcessFxCompile()
   1128 
-> 1129     return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
   1130 
   1131 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in codegen_and_compile(self, gm, example_inputs, inputs_to_check, graph_kwargs)
   1042                                 )
   1043                         else:
-> 1044                             compiled_fn = graph.compile_to_module().call
   1045 
   1046                     num_bytes, nodes_num_elem, node_runtimes = graph.count_bytes()

/usr/local/lib/python3.11/dist-packages/torch/_inductor/graph.py in compile_to_module(self)
   2025             dynamo_compile_column_us="inductor_code_gen_cumulative_compile_time_us",
   2026         ):
-> 2027             return self._compile_to_module()
   2028 
   2029     def _compile_to_module(self) -> ModuleType:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/graph.py in _compile_to_module(self)
   2031 
   2032         code, linemap = (
-> 2033             self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
   2034         )
   2035         if config.triton.autotune_at_compile_time:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/graph.py in codegen(self)
   1962             self.init_wrapper_code()
   1963 
-> 1964             self.scheduler = Scheduler(self.operations)
   1965             V.debug.draw_orig_fx_graph(self.orig_gm, self.scheduler.nodes)
   1966 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in __init__(self, nodes)
   1796     def __init__(self, nodes: List[ir.Operation]) -> None:
   1797         with dynamo_timed("Scheduler.__init__"):
-> 1798             self._init(nodes)
   1799 
   1800     def _init(self, nodes: List[ir.Operation]) -> None:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in _init(self, nodes)
   1868         if config._pre_fusion_custom_pass is not None:
   1869             self.nodes = config._pre_fusion_custom_pass(self.nodes)
-> 1870         self.nodes = self.fuse_nodes(self.nodes)
   1871         if config.reorder_for_peak_memory:
   1872             from .memory import reorder_for_peak_memory

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in fuse_nodes(self, nodes)
   2375                     old_len,
   2376                 )
-> 2377                 nodes = self.fuse_nodes_once(nodes)
   2378                 new_len = len(nodes)
   2379                 fusion_log.debug(

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in fuse_nodes_once(self, nodes)
   2672                 node1, node2
   2673             ):
-> 2674                 if not self.speedup_by_fusion(node1, node2):
   2675                     continue
   2676                 fusion_log.debug(

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in speedup_by_fusion(self, node1, node2)
   2572 
   2573             _, ms1 = multi_node.get_min_choice()
-> 2574             ms2, path2 = self.benchmark_fused_nodes(node_list_2)
   2575 
   2576             min_ms_fused = float("inf")

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in benchmark_fused_nodes(self, nodes)
   2413         backend = self.get_backend(device)
   2414         with dynamo_timed("benchmark_fused_nodes"):
-> 2415             return backend.benchmark_fused_nodes(nodes)
   2416 
   2417     def finalize_multi_template_buffers(self) -> None:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/codegen/cuda_combined_scheduling.py in benchmark_fused_nodes(self, nodes)
     90 
     91     def benchmark_fused_nodes(self, nodes):
---> 92         return self._triton_scheduling.benchmark_fused_nodes(nodes)
     93 
     94     def generate_kernel_code_from_nodes(self, nodes, benchmark_kernel=False):

/usr/local/lib/python3.11/dist-packages/torch/_inductor/codegen/triton.py in benchmark_fused_nodes(self, nodes)
   3656                 return ms, mod.__file__
   3657 
-> 3658             args = mod.get_args()
   3659             call = mod.call
   3660             wrapped_jit_function = mod.triton_

/tmp/torchinductor_root/px/cpxacztjmflnlvwhvgyz7qq46eed5wuxcj2cfoacyeikucyknt2z.py in get_args()
     38 
     39 def get_args():
---> 40     arg_0 = rand_strided((24, 16, 10, 61, 61), (595360, 37210, 3721, 61, 1), device='cuda:0', dtype=torch.float32)
     41     arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
     42     arg_2 = rand_strided((24, 16, 10, 61, 61), (599040, 37440, 3744, 61, 1), device='cuda:0', dtype=torch.float32)

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/testing.py in rand_strided(size, stride, dtype, device, extra_size)
    390             )
    391         else:
--> 392             buffer = torch.randn(needed_size, dtype=dtype, device=device)
    393     else:
    394         buffer = torch.zeros(size=[needed_size], dtype=dtype, device=device)

BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Offset increment outside graph capture encountered unexpectedly.

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True


## === cell 8
THRESHOLD = 0.4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title(f"Output > {THRESHOLD}")
ax1.imshow(output.gt(THRESHOLD), cmap="gray")
ax1.axis("off")
ax2.set_title("Label")
ax2.imshow(label, cmap="gray")
ax2.axis("off")
plt.tight_layout()
plt.show()




## === cell 9
def rle_from_binary_mask(binary_mask_2d: np.ndarray) -> str:
    """
    binary_mask_2d: HxW, values {0,1} or bool.
    RLE is 1-indexed, row-major, and must be sorted with no duplicates.
    """
    if binary_mask_2d.size == 0:
        return ""
    pixels = binary_mask_2d.astype(np.uint8, copy=False).ravel(order="C")
    if pixels.max(initial=0) == 0:
        return ""
    pixels = np.concatenate(([0], pixels, [0])).astype(np.uint8, copy=False)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[0::2]
    return " ".join(map(str, runs.astype(np.int64).tolist()))


_FRAGMENT_STACK_CACHE = {}


def load_fragment_stack(fragment_dir: str) -> torch.Tensor:
    cached = _FRAGMENT_STACK_CACHE.get(fragment_dir)
    if cached is not None:
        return cached
    sv_dir = os.path.join(fragment_dir, "surface_volume")
    tifs = sorted(glob.glob(os.path.join(sv_dir, "*.tif")))
    if len(tifs) == 0:
        raise FileNotFoundError(f"No tif files found in {sv_dir}")
    use = tifs[Z_START : Z_START + Z_DIM]
    if len(use) < Z_DIM:
        raise RuntimeError(
            f"Not enough slices in {sv_dir}: need {Z_DIM}, got {len(use)}"
        )
    imgs = [np.array(Image.open(fn), dtype=np.float32) / 65535.0 for fn in use]
    out = torch.from_numpy(np.stack(imgs, axis=0)).contiguous()  # (Z,H,W), CPU
    _FRAGMENT_STACK_CACHE[fragment_dir] = out
    return out


def predict_fragment_rle(fragment_dir: str) -> str:
    m = np.array(
        Image.open(os.path.join(fragment_dir, "mask.png")).convert("1")
    ).astype(bool)
    H, W = m.shape

    not_border_local = np.zeros((H, W), dtype=bool)
    not_border_local[BUFFER : H - BUFFER, BUFFER : W - BUFFER] = True
    valid = m & not_border_local
    pixels = np.argwhere(valid)

    pixels = _filter_valid_pixels(pixels, H, W, BUFFER)
    if len(pixels) == 0:
        return ""

    stack = load_fragment_stack(fragment_dir)
    dummy_label = torch.zeros((H, W), dtype=torch.float32)  # CPU

    ds = SubvolumeDatasetFast(stack, dummy_label, pixels)
    loader = _make_fast_loader(ds, shuffle=False)

    probs_flat = torch.zeros((H * W,), dtype=torch.float32)  # CPU

    model.eval()
    coords_lin_all_t = ds.coords_lin  # cached

    with torch.inference_mode():
        offset = 0
        for subvolumes, _ in loader:
            pred = model(subvolumes.to(DEVICE, non_blocking=True)).cpu().view(-1)
            n = pred.numel()
            probs_flat[coords_lin_all_t[offset : offset + n]] = pred
            offset += n

    probs = probs_flat.view(H, W)
    binary = probs.numpy() > THRESHOLD
    binary &= m  # enforce original mask
    return rle_from_binary_mask(binary)


_prev_bench = torch.backends.cudnn.benchmark
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

sub = pd.read_csv(SAMPLE_SUB_PATH)
preds = []
for frag_id in sub["Id"].astype(str).tolist():
    frag_dir = os.path.join(TEST_ROOT, frag_id)
    if not os.path.isdir(frag_dir):
        raise FileNotFoundError(f"Test fragment directory not found: {frag_dir}")
    preds.append(predict_fragment_rle(frag_dir))

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = _prev_bench

submission = pd.DataFrame({"Id": sub["Id"].astype(str), "Predicted": preds})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")
print("submission.csv path:", os.path.abspath("submission.csv"))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
BackendCompilerFailed                     Traceback (most recent call last)
/tmp/ipykernel_11/1656069466.py in <cell line: 0>()
     89     if not os.path.isdir(frag_dir):
     90         raise FileNotFoundError(f"Test fragment directory not found: {frag_dir}")
---> 91     preds.append(predict_fragment_rle(frag_dir))
     92 
     93 if DEVICE.type == "cuda":

/tmp/ipykernel_11/1656069466.py in predict_fragment_rle(fragment_dir)
     68         offset = 0
     69         for subvolumes, _ in loader:
---> 70             pred = model(subvolumes.to(DEVICE, non_blocking=True)).cpu().view(-1)
     71             n = pred.numel()
     72             probs_flat[coords_lin_all_t[offset : offset + n]] = pred

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, frame_state)
   1378         with compile_lock, _disable_current_modes():
   1379             # skip=1: skip this frame
-> 1380             return self._torchdynamo_orig_callable(
   1381                 frame, cache_entry, self.hooks, frame_state, skip=1
   1382             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
   1162         counters["frames"]["total"] += 1
   1163         try:
-> 1164             result = self._inner_convert(
   1165                 frame, cache_entry, hooks, frame_state, skip=skip + 1
   1166             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
    545 
    546         with compile_context(CompileContext(compile_id)):
--> 547             return _compile(
    548                 frame.f_code,
    549                 frame.f_globals,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile(code, globals, locals, builtins, closure, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)
    984         guarded_code = None
    985         try:
--> 986             guarded_code = compile_inner(code, one_graph, hooks, transform)
    987 
    988             # NB: We only put_code_state in success case.  Success case here

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in compile_inner(code, one_graph, hooks, transform)
    713             stack.enter_context(torch._dynamo.callback_handler.install_callbacks())
    714             stack.enter_context(CompileTimeInstructionCounter.record())
--> 715             return _compile_inner(code, one_graph, hooks, transform)
    716 
    717         return None  # dead, but see https://github.com/python/mypy/issues/7577

/usr/local/lib/python3.11/dist-packages/torch/_utils_internal.py in wrapper_function(*args, **kwargs)
     93 
     94             if not StrobelightCompileTimeProfiler.enabled:
---> 95                 return function(*args, **kwargs)
     96 
     97             return StrobelightCompileTimeProfiler.profile_compile_time(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile_inner(code, one_graph, hooks, transform)
    748             CompileContext.get().attempt = attempt
    749             try:
--> 750                 out_code = transform_code_object(code, transform)
    751                 break
    752             except exc.RestartAnalysis as e:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/bytecode_transformation.py in transform_code_object(code, transformations, safe)
   1359     propagate_line_nums(instructions)
   1360 
-> 1361     transformations(instructions, code_options)
   1362     return clean_and_assemble_instructions(instructions, keys, code_options)[1]
   1363 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _fn(*args, **kwargs)
    229             exit_stack.enter_context(torch_function_mode_stack_state_mgr)
    230             try:
--> 231                 return fn(*args, **kwargs)
    232             finally:
    233                 cleanup.close()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in transform(instructions, code_options)
    660         try:
    661             with tracing(tracer.output.tracing_context), tracer.set_current_tx():
--> 662                 tracer.run()
    663         except exc.UnspecializeRestartAnalysis:
    664             speculation_log.clear()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   2866 
   2867     def run(self):
-> 2868         super().run()
   2869 
   2870     def should_compile_partial_graph(self):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in RETURN_VALUE(self, inst)
   3046 
   3047     def RETURN_VALUE(self, inst):
-> 3048         self._return(inst)
   3049 
   3050     def RETURN_CONST(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _return(self, inst)
   3031         )
   3032         log.debug("%s triggered compile", inst.opname)
-> 3033         self.output.compile_subgraph(
   3034             self,
   3035             reason=GraphCompileReason(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_subgraph(self, tx, partial_convert, reason)
   1099             # optimization to generate better code in a common case
   1100             self.add_output_instructions(
-> 1101                 self.compile_and_call_fx_graph(
   1102                     tx, list(reversed(stack_values)), root, output_replacements
   1103                 )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in compile_and_call_fx_graph(self, tx, rv, root, replaced_outputs)
   1380 
   1381             with self.restore_global_state():
-> 1382                 compiled_fn = self.call_user_compiler(gm)
   1383 
   1384             from torch.fx._lazy_graph_module import _LazyGraphModule

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in call_user_compiler(self, gm)
   1430             dynamo_compile_column_us="aot_autograd_cumulative_compile_time_us",
   1431         ):
-> 1432             return self._call_user_compiler(gm)
   1433 
   1434     def _call_user_compiler(self, gm: fx.GraphModule) -> CompiledFn:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1481             raise e
   1482         except Exception as e:
-> 1483             raise BackendCompilerFailed(self.compiler_fn, e).with_traceback(
   1484                 e.__traceback__
   1485             ) from None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/output_graph.py in _call_user_compiler(self, gm)
   1460             if config.verify_correctness:
   1461                 compiler_fn = WrapperBackend(compiler_fn)
-> 1462             compiled_fn = compiler_fn(gm, self.example_inputs())
   1463             _step_logger()(logging.INFO, f"done compiler function {name}")
   1464             assert callable(compiled_fn), "compiler_fn did not return callable"

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/repro/after_dynamo.py in __call__(self, gm, example_inputs, **kwargs)
    128                     raise
    129         else:
--> 130             compiled_gm = compiler_fn(gm, example_inputs)
    131 
    132         return compiled_gm

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in __call__(self, model_, inputs_)
   2338         from torch._inductor.compile_fx import compile_fx
   2339 
-> 2340         return compile_fx(model_, inputs_, config_patches=self.config)
   2341 
   2342     def get_compiler_config(self):

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1550     if config_patches:
   1551         with config.patch(config_patches):
-> 1552             return compile_fx(
   1553                 model_,
   1554                 example_inputs_,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx(model_, example_inputs_, inner_compile, config_patches, decompositions)
   1861             unlift_effect_tokens=True
   1862         ):
-> 1863             return aot_autograd(
   1864                 fw_compiler=fw_compiler,
   1865                 bw_compiler=bw_compiler,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/backends/common.py in __call__(self, gm, example_inputs, **kwargs)
     81             # NB: NOT cloned!
     82             with enable_aot_logging(), patch_config:
---> 83                 cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
     84                 counters["aot_autograd"]["ok"] += 1
     85                 return disable(cg)

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in aot_module_simplified(mod, args, fw_compiler, bw_compiler, partition_fn, decompositions, keep_inference_input_mutations, inference_compiler, cudagraphs)
   1153         )
   1154     else:
-> 1155         compiled_fn = dispatch_and_compile()
   1156 
   1157     if isinstance(mod, torch._dynamo.utils.GmWrapper):

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in dispatch_and_compile()
   1129         functional_call = create_functional_call(mod, params_spec, params_len)
   1130         with compiled_autograd._disable():
-> 1131             compiled_fn, _ = create_aot_dispatcher_function(
   1132                 functional_call,
   1133                 fake_flat_args,

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    578 ) -> Tuple[Callable, ViewAndMutationMeta]:
    579     with dynamo_timed("create_aot_dispatcher_function", log_pt2_compile_event=True):
--> 580         return _create_aot_dispatcher_function(
    581             flat_fn, fake_flat_args, aot_config, fake_mode, shape_env
    582         )

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in _create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    828         compiler_fn = choose_dispatcher(needs_autograd, aot_config)
    829 
--> 830         compiled_fn, fw_metadata = compiler_fn(
    831             flat_fn,
    832             _dup_fake_script_obj(fake_flat_args),

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py in aot_dispatch_base(flat_fn, flat_args, aot_config, fw_metadata)
    201                 assert isinstance(fw_module, GraphModule)
    202                 tensorify_python_scalars(fw_module, fake_mode.shape_env, fake_mode)
--> 203             compiled_fw = compiler(fw_module, updated_flat_args)
    204 
    205         if fakified_out_wrapper.needs_post_compile:

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in __call__(self, gm, example_inputs)
    487         example_inputs: Sequence[InputType],
    488     ) -> OutputCode:
--> 489         return self.compiler_fn(gm, example_inputs)
    490 
    491 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in fw_compiler_base(gm, example_inputs, is_inference)
   1739                     model_outputs_node.meta["user_visible_output_idxs"] = []
   1740 
-> 1741                 return inner_compile(
   1742                     gm,
   1743                     example_inputs,

/usr/lib/python3.11/contextlib.py in inner(*args, **kwds)
     79         def inner(*args, **kwds):
     80             with self._recreate_cm():
---> 81                 return func(*args, **kwds)
     82         return inner
     83 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in compile_fx_inner(gm, example_inputs, **kwargs)
    567         )
    568 
--> 569         return wrap_compiler_debug(_compile_fx_inner, compiler_name="inductor")(
    570             gm,
    571             example_inputs,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/repro/after_aot.py in debug_wrapper(gm, example_inputs, **kwargs)
    100             # Call the compiler_fn - which is either aot_autograd or inductor
    101             # with fake inputs
--> 102             inner_compiled_fn = compiler_fn(gm, example_inputs)
    103         except Exception as e:
    104             # TODO: Failures here are troublesome because no real inputs,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in _compile_fx_inner(gm, example_inputs, **graph_kwargs)
    683             TritonBundler.begin_compile()
    684             try:
--> 685                 mb_compiled_graph = fx_codegen_and_compile(
    686                     gm, example_inputs, inputs_to_check, **graph_kwargs
    687                 )

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in fx_codegen_and_compile(gm, example_inputs, inputs_to_check, **graph_kwargs)
   1127     scheme: FxCompile = _InProcessFxCompile()
   1128 
-> 1129     return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
   1130 
   1131 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in codegen_and_compile(self, gm, example_inputs, inputs_to_check, graph_kwargs)
   1042                                 )
   1043                         else:
-> 1044                             compiled_fn = graph.compile_to_module().call
   1045 
   1046                     num_bytes, nodes_num_elem, node_runtimes = graph.count_bytes()

/usr/local/lib/python3.11/dist-packages/torch/_inductor/graph.py in compile_to_module(self)
   2025             dynamo_compile_column_us="inductor_code_gen_cumulative_compile_time_us",
   2026         ):
-> 2027             return self._compile_to_module()
   2028 
   2029     def _compile_to_module(self) -> ModuleType:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/graph.py in _compile_to_module(self)
   2031 
   2032         code, linemap = (
-> 2033             self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
   2034         )
   2035         if config.triton.autotune_at_compile_time:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/graph.py in codegen(self)
   1962             self.init_wrapper_code()
   1963 
-> 1964             self.scheduler = Scheduler(self.operations)
   1965             V.debug.draw_orig_fx_graph(self.orig_gm, self.scheduler.nodes)
   1966 

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in __init__(self, nodes)
   1796     def __init__(self, nodes: List[ir.Operation]) -> None:
   1797         with dynamo_timed("Scheduler.__init__"):
-> 1798             self._init(nodes)
   1799 
   1800     def _init(self, nodes: List[ir.Operation]) -> None:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in _init(self, nodes)
   1868         if config._pre_fusion_custom_pass is not None:
   1869             self.nodes = config._pre_fusion_custom_pass(self.nodes)
-> 1870         self.nodes = self.fuse_nodes(self.nodes)
   1871         if config.reorder_for_peak_memory:
   1872             from .memory import reorder_for_peak_memory

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in fuse_nodes(self, nodes)
   2375                     old_len,
   2376                 )
-> 2377                 nodes = self.fuse_nodes_once(nodes)
   2378                 new_len = len(nodes)
   2379                 fusion_log.debug(

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in fuse_nodes_once(self, nodes)
   2672                 node1, node2
   2673             ):
-> 2674                 if not self.speedup_by_fusion(node1, node2):
   2675                     continue
   2676                 fusion_log.debug(

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in speedup_by_fusion(self, node1, node2)
   2572 
   2573             _, ms1 = multi_node.get_min_choice()
-> 2574             ms2, path2 = self.benchmark_fused_nodes(node_list_2)
   2575 
   2576             min_ms_fused = float("inf")

/usr/local/lib/python3.11/dist-packages/torch/_inductor/scheduler.py in benchmark_fused_nodes(self, nodes)
   2413         backend = self.get_backend(device)
   2414         with dynamo_timed("benchmark_fused_nodes"):
-> 2415             return backend.benchmark_fused_nodes(nodes)
   2416 
   2417     def finalize_multi_template_buffers(self) -> None:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/codegen/cuda_combined_scheduling.py in benchmark_fused_nodes(self, nodes)
     90 
     91     def benchmark_fused_nodes(self, nodes):
---> 92         return self._triton_scheduling.benchmark_fused_nodes(nodes)
     93 
     94     def generate_kernel_code_from_nodes(self, nodes, benchmark_kernel=False):

/usr/local/lib/python3.11/dist-packages/torch/_inductor/codegen/triton.py in benchmark_fused_nodes(self, nodes)
   3656                 return ms, mod.__file__
   3657 
-> 3658             args = mod.get_args()
   3659             call = mod.call
   3660             wrapped_jit_function = mod.triton_

/tmp/torchinductor_root/px/cpxacztjmflnlvwhvgyz7qq46eed5wuxcj2cfoacyeikucyknt2z.py in get_args()
     38 
     39 def get_args():
---> 40     arg_0 = rand_strided((24, 16, 10, 61, 61), (595360, 37210, 3721, 61, 1), device='cuda:0', dtype=torch.float32)
     41     arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
     42     arg_2 = rand_strided((24, 16, 10, 61, 61), (599040, 37440, 3744, 61, 1), device='cuda:0', dtype=torch.float32)

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/testing.py in rand_strided(size, stride, dtype, device, extra_size)
    390             )
    391         else:
--> 392             buffer = torch.randn(needed_size, dtype=dtype, device=device)
    393     else:
    394         buffer = torch.zeros(size=[needed_size], dtype=dtype, device=device)

BackendCompilerFailed: backend='inductor' raised:
RuntimeError: Offset increment outside graph capture encountered unexpectedly.

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
