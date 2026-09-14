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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

DATA_ROOT = "/kaggle/input/vesuvius-challenge-ink-detection"
TRAIN_FRAGMENT_ID = "1"
PREFIX = f"{DATA_ROOT}/train/{TRAIN_FRAGMENT_ID}/"

BUFFER = 30  # Buffer size in x and y direction
Z_START = 27  # First slice in the z direction to use
Z_DIM = 10  # Number of slices in the z direction
TRAINING_STEPS = 20000
LEARNING_RATE = 0.05
BATCH_SIZE = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INFER_BATCH_SIZE = 512 if torch.cuda.is_available() else 128

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.backends.cudnn.allow_tf32 = False  # preserve numerical behavior
except Exception:
    pass

NUM_WORKERS = min(4, (os.cpu_count() or 2))
PIN_MEMORY = torch.cuda.is_available()

try:
    import tifffile  # type: ignore

    _HAVE_TIFFFILE = True
except Exception:
    tifffile = None
    _HAVE_TIFFFILE = False

try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass

_CAN_COMPILE = False

SHOW_PLOTS = bool(int(os.environ.get("SHOW_PLOTS", "0")))

ir_path = os.path.join(PREFIX, "ir.png")
if SHOW_PLOTS and os.path.exists(ir_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(Image.open(ir_path), cmap="gray")
    plt.title(f"IR: train/{TRAIN_FRAGMENT_ID}")
    plt.axis("off")
    plt.show()



## === cell 1
mask_path = os.path.join(PREFIX, "mask.png")
label_path = os.path.join(PREFIX, "inklabels.png")

mask = np.array(Image.open(mask_path).convert("1"))
label = torch.from_numpy(np.array(Image.open(label_path))).gt(0).float().to(DEVICE)

if SHOW_PLOTS:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title("mask.png")
    ax1.imshow(mask, cmap="gray")
    ax1.axis("off")
    ax2.set_title("inklabels.png")
    ax2.imshow(label.detach().cpu().numpy(), cmap="gray")
    ax2.axis("off")
    plt.tight_layout()
    plt.show()



## === cell 2
tif_files = sorted(glob.glob(os.path.join(PREFIX, "surface_volume", "*.tif")))
assert len(tif_files) > 0, f"No tif files found under {PREFIX}/surface_volume"
tif_files = tif_files[Z_START : Z_START + Z_DIM]
assert (
    len(tif_files) == Z_DIM
), f"Expected {Z_DIM} slices, got {len(tif_files)}. Check Z_START/Z_DIM."


def _read_tif_to_float01(fn: str) -> np.ndarray:
    if _HAVE_TIFFFILE:
        arr = tifffile.imread(fn)
    else:
        arr = np.array(Image.open(fn))
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16, copy=False)
    out = arr.astype(np.float32, copy=False)
    out *= 1.0 / 65535.0
    return out


from concurrent.futures import ThreadPoolExecutor

_THREADPOOL = ThreadPoolExecutor(max_workers=NUM_WORKERS)


def _load_slices_parallel(file_list, desc: str):
    out = [None] * len(file_list)
    futures = {
        _THREADPOOL.submit(_read_tif_to_float01, fn): i
        for i, fn in enumerate(file_list)
    }
    for fut in tqdm(futures, total=len(futures), desc=desc):
        i = futures[fut]
        out[i] = fut.result()
    return out


images = _load_slices_parallel(tif_files, desc="Loading train slices")
cpu_stack = np.stack(images, axis=0)  # (Z,H,W) float32
image_stack = torch.from_numpy(cpu_stack).to(DEVICE)  # (Z,H,W)

if SHOW_PLOTS:
    fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
    for im, ax in zip(images, axes):
        thumb = np.array(
            Image.fromarray((im * 65535).astype(np.uint16)).resize(
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

if SHOW_PLOTS:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(label.detach().cpu().numpy(), cmap="gray")
    patch = patches.Rectangle(
        (rect[0], rect[1]),
        rect[2],
        rect[3],
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(patch)
    ax.set_title("Training fragment label with eval rect")
    ax.axis("off")
    plt.show()



## === cell 4
_K = BUFFER * 2 + 1
_AR_K_LONG = torch.arange(_K, device=DEVICE, dtype=torch.long)


def gather_subvolumes_from_stack(
    stack_zhw: torch.Tensor, pixels_yx_t: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """
    stack_zhw: (Z,H,W) float32 on DEVICE
    pixels_yx_t: (B,2) int64 on DEVICE, with y/x in [BUFFER..H-BUFFER-1], [BUFFER..W-BUFFER-1]
    returns: (B,1,Z,k,k)
    """
    y0 = pixels_yx_t[:, 0] - BUFFER
    x0 = pixels_yx_t[:, 1] - BUFFER

    yy = y0.unsqueeze(1) + _AR_K_LONG  # (B,k)
    xx = x0.unsqueeze(1) + _AR_K_LONG  # (B,k)

    sub = stack_zhw[:, yy[:, :, None], xx[:, None, :]]
    sub = sub.permute(1, 0, 2, 3).unsqueeze(1)  # (B,1,Z,k,k)

    if DEVICE.type == "cuda":
        sub = sub.contiguous(memory_format=torch.channels_last_3d)
    else:
        sub = sub.contiguous()

    if out is not None:
        out.copy_(sub)
        return out
    return sub


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

with torch.no_grad():
    dummy = torch.zeros((1, 1, Z_DIM, _K, _K), device=DEVICE, dtype=torch.float32)
    _ = model(dummy)

if DEVICE.type == "cuda":
    model = model.to(memory_format=torch.channels_last_3d)




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

print(
    f"pixels_outside_rect: {len(pixels_outside_rect):,} | pixels_inside_rect: {len(pixels_inside_rect):,}"
)

train_labels = (
    label[pixels_outside_rect[:, 0], pixels_outside_rect[:, 1]].contiguous().view(-1, 1)
)
eval_labels = (
    label[pixels_inside_rect[:, 0], pixels_inside_rect[:, 1]].contiguous().view(-1, 1)
)

print("Training (on-the-fly subvolume gather to avoid GPU OOM)...")

n_train = pixels_outside_rect.shape[0]
train_pixels_t = torch.as_tensor(pixels_outside_rect, device=DEVICE, dtype=torch.long)

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

model.train()

gen_dev = torch.Generator(device=DEVICE).manual_seed(SEED)

_subvol_buf = torch.empty(
    (BATCH_SIZE, 1, Z_DIM, _K, _K), device=DEVICE, dtype=torch.float32
)
if DEVICE.type == "cuda":
    _subvol_buf = _subvol_buf.contiguous(memory_format=torch.channels_last_3d)

for i in tqdm(range(TRAINING_STEPS), total=TRAINING_STEPS, desc="Train steps"):
    idx_t = torch.randint(
        0, n_train, (BATCH_SIZE,), generator=gen_dev, device=DEVICE, dtype=torch.long
    )

    batch_pixels_t = train_pixels_t.index_select(0, idx_t)  # (B,2)
    subvolumes = gather_subvolumes_from_stack(
        image_stack, batch_pixels_t, out=_subvol_buf
    )  # (B,1,Z,k,k)
    inklabels = train_labels.index_select(0, idx_t)  # (B,1)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(subvolumes)
    loss = criterion(outputs, inklabels)
    loss.backward()
    optimizer.step()
    scheduler.step()



## === cell 6
output = torch.zeros_like(label).float()
model.eval()

pixels_inside_t = torch.as_tensor(pixels_inside_rect, device=DEVICE, dtype=torch.long)

_inf_buf = torch.empty(
    (min(INFER_BATCH_SIZE, pixels_inside_t.shape[0]), 1, Z_DIM, _K, _K),
    device=DEVICE,
    dtype=torch.float32,
)
if DEVICE.type == "cuda":
    _inf_buf = _inf_buf.contiguous(memory_format=torch.channels_last_3d)

with torch.inference_mode():
    for base in tqdm(
        range(0, pixels_inside_t.shape[0], INFER_BATCH_SIZE), desc="Eval train-rect"
    ):
        batch_pixels_t = pixels_inside_t[base : base + INFER_BATCH_SIZE]
        buf = _inf_buf[: batch_pixels_t.shape[0]]
        subvolumes = gather_subvolumes_from_stack(image_stack, batch_pixels_t, out=buf)
        preds = model(subvolumes).squeeze(1)  # (B,)
        output[batch_pixels_t[:, 0], batch_pixels_t[:, 1]] = preds

if SHOW_PLOTS:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(output.detach().cpu().numpy(), cmap="gray")
    ax1.set_title("Model output (prob)")
    ax1.axis("off")
    ax2.imshow(label.detach().cpu().numpy(), cmap="gray")
    ax2.set_title("Label")
    ax2.axis("off")
    plt.tight_layout()
    plt.show()



## === cell 7
THRESHOLD = 0.4

if SHOW_PLOTS:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(output.gt(THRESHOLD).detach().cpu().numpy(), cmap="gray")
    ax1.set_title(f"Pred mask @ thr={THRESHOLD}")
    ax1.axis("off")
    ax2.imshow(label.detach().cpu().numpy(), cmap="gray")
    ax2.set_title("Label")
    ax2.axis("off")
    plt.tight_layout()
    plt.show()




## === cell 8
def rle_from_binary_mask(binary_mask: np.ndarray) -> str:
    """
    Kaggle Vesuvius expects 1-indexed RLE on flattened mask (row-major),
    with runs sorted and no duplicates.
    """
    pixels = binary_mask.flatten(order="C").astype(np.uint8)
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs = changes.copy()
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return " ".join(str(x) for x in runs)


def load_fragment_stack(
    fragment_dir: str, z_start: int, z_dim: int, device: torch.device
) -> torch.Tensor:
    tif_files = sorted(glob.glob(os.path.join(fragment_dir, "surface_volume", "*.tif")))
    if len(tif_files) < z_start + z_dim:
        raise RuntimeError(f"Not enough slices in {fragment_dir}: {len(tif_files)}")
    tif_files = tif_files[z_start : z_start + z_dim]

    imgs = _load_slices_parallel(
        tif_files, desc=f"Loading {os.path.basename(fragment_dir)} slices"
    )
    cpu_stack = np.stack(imgs, axis=0)
    stack = torch.from_numpy(cpu_stack).to(device)  # (Z,H,W)
    return stack


def predict_fragment(model: nn.Module, fragment_dir: str, threshold: float) -> str:
    """
    Predict full-size mask for a test fragment using the same per-pixel subvolume classifier.
    Only pixels within mask.png and away from border are predicted; others are 0.
    """
    mask_local = np.array(
        Image.open(os.path.join(fragment_dir, "mask.png")).convert("1")
    ).astype(bool)
    H, W = mask_local.shape

    not_border_local = np.zeros((H, W), dtype=bool)
    not_border_local[BUFFER : H - BUFFER, BUFFER : W - BUFFER] = True
    valid = mask_local & not_border_local
    pixels = np.argwhere(valid)

    if pixels.shape[0] == 0:
        return ""

    stack = load_fragment_stack(fragment_dir, Z_START, Z_DIM, DEVICE)

    out_t = torch.zeros((H, W), device=DEVICE, dtype=torch.float32)
    pixels_t = torch.as_tensor(pixels, device=DEVICE, dtype=torch.long)

    model.eval()
    local_buf = torch.empty(
        (min(INFER_BATCH_SIZE, pixels_t.shape[0]), 1, Z_DIM, _K, _K),
        device=DEVICE,
        dtype=torch.float32,
    )
    if DEVICE.type == "cuda":
        local_buf = local_buf.contiguous(memory_format=torch.channels_last_3d)

    with torch.inference_mode():
        for base in tqdm(
            range(0, pixels_t.shape[0], INFER_BATCH_SIZE),
            leave=False,
            desc=f"Infer {os.path.basename(fragment_dir)}",
        ):
            batch_pixels_t = pixels_t[base : base + INFER_BATCH_SIZE]
            buf = local_buf[: batch_pixels_t.shape[0]]
            subvols = gather_subvolumes_from_stack(stack, batch_pixels_t, out=buf)
            preds_t = model(subvols).squeeze(1)  # (B,)
            out_t[batch_pixels_t[:, 0], batch_pixels_t[:, 1]] = preds_t

    binary = (out_t > threshold).to(torch.uint8).cpu().numpy()
    return rle_from_binary_mask(binary)


sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
sub_df = pd.read_csv(sample_path)

preds = []
for frag_id in tqdm(sub_df["Id"].tolist(), desc="Predict test fragments"):
    frag_dir = os.path.join(DATA_ROOT, "test", str(frag_id))
    if not os.path.isdir(frag_dir):
        preds.append("")
        continue
    preds.append(predict_fragment(model, frag_dir, THRESHOLD))

submission = pd.DataFrame({"Id": sub_df["Id"].astype(str), "Predicted": preds})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv")
