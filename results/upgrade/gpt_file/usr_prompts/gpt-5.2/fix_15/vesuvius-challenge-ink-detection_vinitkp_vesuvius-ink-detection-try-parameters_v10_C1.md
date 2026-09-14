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

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

BUFFER = 30  # Buffer size in x and y direction
Z_START = 27  # First slice in the z direction to use
Z_DIM = 10  # Number of slices in the z direction
TRAINING_STEPS = 20000
LEARNING_RATE = 0.05
BATCH_SIZE = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DL_GEN = torch.Generator()
DL_GEN.manual_seed(0)

NUM_WORKERS = min(8, os.cpu_count() or 1)
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
        print("torch.compile not enabled (fallback to eager):", repr(e))

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

_AMP_ENABLED = DEVICE.type == "cuda"
_AMP_DTYPE = (
    torch.float16
)  # safe for inference/training speed; outputs are sigmoid+BCELoss



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


_prev_bench = torch.backends.cudnn.benchmark
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

model.train()
it = infinite_loader(train_loader)

scaler = torch.cuda.amp.GradScaler(enabled=_AMP_ENABLED)

pbar = tqdm(range(TRAINING_STEPS), total=TRAINING_STEPS, miniters=200, mininterval=1.0)

for i in pbar:
    subvolumes, inklabels = next(it)
    optimizer.zero_grad(set_to_none=True)

    x = subvolumes.to(DEVICE, non_blocking=True)
    y = inklabels.to(DEVICE, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=_AMP_ENABLED, dtype=_AMP_DTYPE):
        outputs = model(x)
        loss = criterion(outputs, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    if (i + 1) % 200 == 0:
        pbar.set_postfix(loss=float(loss.detach().cpu()))

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = _prev_bench



## === cell 7
eval_dataset = SubvolumeDatasetFast(image_stack, label, pixels_inside_rect)
eval_loader = _make_fast_loader(eval_dataset, shuffle=False)

output = torch.zeros_like(label).float()  # CPU
model.eval()

coords_lin_all_t = eval_dataset.coords_lin  # cached

with torch.inference_mode():
    offset = 0
    out_flat = output.view(-1)
    for subvolumes, _ in tqdm(eval_loader, desc="Eval rect"):
        x = subvolumes.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=_AMP_ENABLED, dtype=_AMP_DTYPE):
            preds = model(x)
        preds = preds.detach().cpu().view(-1)

        n = preds.numel()
        if offset + n > coords_lin_all_t.numel():
            n = max(0, int(coords_lin_all_t.numel() - offset))
            preds = preds[:n]
        out_flat[coords_lin_all_t[offset : offset + n]] = preds.to(out_flat.dtype)
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

    imgs = []
    for fn in use:
        imgs.append(np.array(Image.open(fn), dtype=np.float32) / 65535.0)
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
            x = subvolumes.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=_AMP_ENABLED, dtype=_AMP_DTYPE):
                pred = model(x)
            pred = pred.detach().cpu().view(-1)

            n = pred.numel()
            if offset + n > coords_lin_all_t.numel():
                n = max(0, int(coords_lin_all_t.numel() - offset))
                pred = pred[:n]
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
