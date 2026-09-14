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

0.012406

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

ir_path = os.path.join(PREFIX, "ir.png")
if os.path.exists(ir_path):
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
    arr = np.ascontiguousarray(arr, dtype=np.uint16)
    return arr.astype(np.float32) / 65535.0


images = [
    _read_tif_to_float01(fn) for fn in tqdm(tif_files, desc="Loading train slices")
]

cpu_stack = np.stack(images, axis=0)  # (Z,H,W) float32
image_stack = torch.from_numpy(cpu_stack).to(DEVICE, non_blocking=PIN_MEMORY)  # (Z,H,W)

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

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(label.detach().cpu().numpy(), cmap="gray")
patch = patches.Rectangle(
    (rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor="r", facecolor="none"
)
ax.add_patch(patch)
ax.set_title("Training fragment label with eval rect")
ax.axis("off")
plt.show()




## === cell 4
def build_windows_view(image_stack_zhw: torch.Tensor) -> torch.Tensor:
    """
    Returns a lazy windows view: (Z, H-k+1, W-k+1, k, k) on same device as input.
    This view is lightweight; it does not materialize all windows.
    """
    k = BUFFER * 2 + 1
    return image_stack_zhw.unfold(1, k, 1).unfold(2, k, 1)


def _precompute_windows_flat(windows_zhwkk: torch.Tensor):
    """
    Precompute a flattened view for fast batched gathers.

    windows_zhwkk: (Z, H2, W2, k, k)
    returns:
      windows_flat: (H2*W2, Z, k, k) view (no copy)
      H2, W2: spatial dims of top-left window positions
    """
    Z, H2, W2, k, _ = windows_zhwkk.shape
    windows_flat = windows_zhwkk.permute(1, 2, 0, 3, 4).reshape(H2 * W2, Z, k, k)
    return windows_flat, H2, W2


def _pixels_to_linear_indices(pixels_yx_t: torch.Tensor, W2: int) -> torch.Tensor:
    yy = pixels_yx_t[:, 0] - BUFFER
    xx = pixels_yx_t[:, 1] - BUFFER
    return yy * W2 + xx


def _gather_subvolumes_from_windows_flat_t(
    windows_flat_hzwkk: torch.Tensor, pixels_yx_t: torch.Tensor, W2: int
) -> torch.Tensor:
    """
    windows_flat_hzwkk: (H2*W2, Z, k, k)
    returns: (B,1,Z,k,k)
    """
    lin = _pixels_to_linear_indices(pixels_yx_t, W2)
    gathered = windows_flat_hzwkk.index_select(0, lin)  # (B,Z,k,k)
    out = gathered.unsqueeze(1)  # (B,1,Z,k,k)
    if DEVICE.type == "cuda":
        out = out.contiguous(memory_format=torch.channels_last_3d)
    else:
        out = out.contiguous()
    return out


class PixelSubvolumeDataset(data.Dataset):
    def __init__(
        self,
        windows_zhwkk: torch.Tensor,
        pixels_yx: np.ndarray,
        labels_1d: torch.Tensor,
    ):
        self.windows = windows_zhwkk
        self.pixels = pixels_yx.astype(np.int64, copy=False)
        self.labels_1d = labels_1d

    def __len__(self):
        return self.pixels.shape[0]

    def __getitem__(self, index):
        y, x = self.pixels[index]
        yy = int(y - BUFFER)
        xx = int(x - BUFFER)
        subvol = self.windows[:, yy, xx, :, :].unsqueeze(0).contiguous()
        return subvol, self.labels_1d[index]


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
    model = model.to(memory_format=torch.channels_last_3d)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1665166717.py in <cell line: 0>()
     87 
     88 if DEVICE.type == "cuda":
---> 89     model = model.to(memory_format=torch.channels_last_3d)
     90 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in to(self, *args, **kwargs)
   1341                     raise
   1342 
-> 1343         return self._apply(convert)
   1344 
   1345     def register_full_backward_pre_hook(

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _apply(self, fn, recurse)
    901         if recurse:
    902             for module in self.children():
--> 903                 module._apply(fn)
    904 
    905         def compute_should_use_set_data(tensor, tensor_applied):

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _apply(self, fn, recurse)
    928             # `with torch.no_grad():`
    929             with torch.no_grad():
--> 930                 param_applied = fn(param)
    931             p_should_use_set_data = compute_should_use_set_data(param, param_applied)
    932 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in convert(t)
   1320         def convert(t):
   1321             try:
-> 1322                 if convert_to_format is not None and t.dim() in (4, 5):
   1323                     return t.to(
   1324                         device,

/usr/local/lib/python3.11/dist-packages/torch/nn/parameter.py in __torch_function__(cls, func, types, args, kwargs)
    166                 kwargs = {}
    167             return super().__torch_function__(func, types, args, kwargs)
--> 168         raise ValueError(
    169             f"Attempted to use an uninitialized parameter in {func}. "
    170             "This error happens when you are using a `LazyModule` or "

ValueError: Attempted to use an uninitialized parameter in <method 'dim' of 'torch._C.TensorBase' objects>. This error happens when you are using a `LazyModule` or explicitly manipulating `torch.nn.parameter.UninitializedParameter` objects. When using LazyModules Call `forward` with a dummy batch to initialize the parameters before calling torch functions

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

windows_train = build_windows_view(image_stack)
windows_flat_train, _H2_train, _W2_train = _precompute_windows_flat(windows_train)

train_labels = label[pixels_outside_rect[:, 0], pixels_outside_rect[:, 1]].view(-1, 1)
eval_labels = label[pixels_inside_rect[:, 0], pixels_inside_rect[:, 1]].view(-1, 1)

print("Training...")

pixels_outside_t = torch.as_tensor(pixels_outside_rect, device=DEVICE, dtype=torch.long)
train_labels_t = train_labels  # already on DEVICE
n_train = pixels_outside_t.shape[0]

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
gen = torch.Generator(device="cpu").manual_seed(SEED)

for i in tqdm(range(TRAINING_STEPS), total=TRAINING_STEPS, desc="Train steps"):
    idx_t = torch.randint(
        0, n_train, (BATCH_SIZE,), generator=gen, device=DEVICE, dtype=torch.long
    )

    batch_pixels = pixels_outside_t.index_select(0, idx_t)  # (B,2)
    subvolumes = _gather_subvolumes_from_windows_flat_t(
        windows_flat_train, batch_pixels, _W2_train
    )  # (B,1,Z,k,k)
    inklabels = train_labels_t.index_select(0, idx_t)  # (B,1)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(subvolumes)
    loss = criterion(outputs, inklabels)
    loss.backward()
    optimizer.step()
    scheduler.step()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_11/1466264746.py in <cell line: 0>()
     21 windows_train = build_windows_view(image_stack)
     22 # Speed: precompute flattened window view once for the whole training fragment.
---> 23 windows_flat_train, _H2_train, _W2_train = _precompute_windows_flat(windows_train)
     24 
     25 train_labels = label[pixels_outside_rect[:, 0], pixels_outside_rect[:, 1]].view(-1, 1)

/tmp/ipykernel_11/1665166717.py in _precompute_windows_flat(windows_zhwkk)
     22     Z, H2, W2, k, _ = windows_zhwkk.shape
     23     # (Z,H2,W2,k,k) -> (H2*W2,Z,k,k)
---> 24     windows_flat = windows_zhwkk.permute(1, 2, 0, 3, 4).reshape(H2 * W2, Z, k, k)
     25     return windows_flat, H2, W2
     26 

OutOfMemoryError: CUDA out of memory. Tried to allocate 7058.25 GiB. GPU 0 has a total capacity of 47.53 GiB of which 45.10 GiB is free. Process 1610882 has 2.42 GiB memory in use. Of the allocated memory 2.12 GiB is allocated by PyTorch, and 1.73 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 6
pixels_inside_t = torch.as_tensor(pixels_inside_rect, device=DEVICE, dtype=torch.long)
eval_labels_t = eval_labels  # on DEVICE

output = torch.zeros_like(label).float()
model.eval()

with torch.inference_mode():
    for base in tqdm(
        range(0, pixels_inside_t.shape[0], INFER_BATCH_SIZE), desc="Eval train-rect"
    ):
        batch_pixels_t = pixels_inside_t[base : base + INFER_BATCH_SIZE]
        subvolumes = _gather_subvolumes_from_windows_flat_t(
            windows_flat_train, batch_pixels_t, _W2_train
        )
        preds = model(subvolumes).squeeze(1)  # (B,)
        output[batch_pixels_t[:, 0], batch_pixels_t[:, 1]] = preds

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(output.detach().cpu().numpy(), cmap="gray")
ax1.set_title("Model output (prob)")
ax1.axis("off")
ax2.imshow(label.detach().cpu().numpy(), cmap="gray")
ax2.set_title("Label")
ax2.axis("off")
plt.tight_layout()
plt.show()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1959447931.py in <cell line: 0>()
      1 pixels_inside_t = torch.as_tensor(pixels_inside_rect, device=DEVICE, dtype=torch.long)
----> 2 eval_labels_t = eval_labels  # on DEVICE
      3 
      4 output = torch.zeros_like(label).float()
      5 model.eval()

NameError: name 'eval_labels' is not defined

## === cell 7
THRESHOLD = 0.4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(output.gt(THRESHOLD).detach().cpu().numpy(), cmap="gray")
ax1.set_title(f"Pred mask @ thr={THRESHOLD}")
ax1.axis("off")
ax2.imshow(label.detach().cpu().numpy(), cmap="gray")
ax2.set_title("Label")
ax2.axis("off")
plt.tight_layout()
plt.show()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4198759701.py in <cell line: 0>()
      1 THRESHOLD = 0.4
      2 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
----> 3 ax1.imshow(output.gt(THRESHOLD).detach().cpu().numpy(), cmap="gray")
      4 ax1.set_title(f"Pred mask @ thr={THRESHOLD}")
      5 ax1.axis("off")

NameError: name 'output' is not defined

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

    imgs = [_read_tif_to_float01(fn) for fn in tif_files]
    cpu_stack = np.stack(imgs, axis=0)
    stack = torch.from_numpy(cpu_stack).to(device, non_blocking=PIN_MEMORY)  # (Z,H,W)
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

    stack = load_fragment_stack(fragment_dir, Z_START, Z_DIM, DEVICE)
    windows = build_windows_view(stack)
    windows_flat, _H2, _W2 = _precompute_windows_flat(windows)

    pixels_t = torch.as_tensor(pixels, device=DEVICE, dtype=torch.long)

    out_t = torch.zeros((H, W), device=DEVICE, dtype=torch.float32)

    model.eval()
    with torch.inference_mode():
        for base in tqdm(
            range(0, pixels_t.shape[0], INFER_BATCH_SIZE),
            leave=False,
            desc=f"Infer {os.path.basename(fragment_dir)}",
        ):
            batch_pixels_t = pixels_t[base : base + INFER_BATCH_SIZE]  # (B,2)
            subvols = _gather_subvolumes_from_windows_flat_t(
                windows_flat, batch_pixels_t, _W2
            )  # (B,1,Z,k,k)
            preds_t = model(subvols).squeeze(1)  # (B,)
            out_t[batch_pixels_t[:, 0], batch_pixels_t[:, 1]] = preds_t

    binary = (out_t > threshold).to(torch.uint8).cpu().numpy()
    return rle_from_binary_mask(binary)




## === cell 9
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

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_11/1642264696.py in <cell line: 0>()
      8         preds.append("")
      9         continue
---> 10     preds.append(predict_fragment(model, frag_dir, THRESHOLD))
     11 
     12 submission = pd.DataFrame({"Id": sub_df["Id"].astype(str), "Predicted": preds})

/tmp/ipykernel_11/3823097717.py in predict_fragment(model, fragment_dir, threshold)
     44     windows = build_windows_view(stack)
     45     # Speed: same flattened-gather approach for inference.
---> 46     windows_flat, _H2, _W2 = _precompute_windows_flat(windows)
     47 
     48     pixels_t = torch.as_tensor(pixels, device=DEVICE, dtype=torch.long)

/tmp/ipykernel_11/1665166717.py in _precompute_windows_flat(windows_zhwkk)
     22     Z, H2, W2, k, _ = windows_zhwkk.shape
     23     # (Z,H2,W2,k,k) -> (H2*W2,Z,k,k)
---> 24     windows_flat = windows_zhwkk.permute(1, 2, 0, 3, 4).reshape(H2 * W2, Z, k, k)
     25     return windows_flat, H2, W2
     26 

OutOfMemoryError: CUDA out of memory. Tried to allocate 5427.76 GiB. GPU 0 has a total capacity of 47.53 GiB of which 43.60 GiB is free. Process 1610882 has 3.92 GiB memory in use. Of the allocated memory 3.62 GiB is allocated by PyTorch, and 4.58 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
