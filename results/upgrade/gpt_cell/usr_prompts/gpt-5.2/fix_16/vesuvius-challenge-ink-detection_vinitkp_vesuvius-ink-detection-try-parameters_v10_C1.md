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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

TRAIN_PREFIX = "/kaggle/input/vesuvius-challenge-ink-detection/train/1/"
TEST_ROOT = "/kaggle/input/vesuvius-challenge-ink-detection/test/"
SAMPLE_SUB_PATH = "/kaggle/input/vesuvius-challenge-ink-detection/sample_submission.csv"

BUFFER = 30  # Buffer size in x and y direction
Z_START = 27  # First slice in the z direction to use
Z_DIM = 10  # Number of slices in the z direction
TRAINING_STEPS = 20000
LEARNING_RATE = 0.05
BATCH_SIZE = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

if os.environ.get("SHOW_PLOTS", "0") == "1":
    plt.imshow(Image.open(TRAIN_PREFIX + "ir.png"), cmap="gray")
    plt.axis("off")
    plt.show()



## === cell 1
mask = np.array(Image.open(TRAIN_PREFIX + "mask.png").convert("1"))
label = (
    torch.from_numpy(np.array(Image.open(TRAIN_PREFIX + "inklabels.png")))
    .gt(0)
    .float()
    .to(DEVICE)
)

if os.environ.get("SHOW_PLOTS", "0") == "1":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.set_title("mask.png")
    ax1.imshow(mask, cmap="gray")
    ax1.axis("off")
    ax2.set_title("inklabels.png")
    ax2.imshow(label.cpu(), cmap="gray")
    ax2.axis("off")
    plt.tight_layout()
    plt.show()



## === cell 2
_train_files = sorted(glob.glob(TRAIN_PREFIX + "surface_volume/*.tif"))[
    Z_START : Z_START + Z_DIM
]

images = [
    np.array(Image.open(filename), dtype=np.float32) / 65535.0
    for filename in tqdm(_train_files)
]
image_stack = torch.stack([torch.from_numpy(image) for image in images], dim=0).to(
    DEVICE
)

if os.environ.get("SHOW_PLOTS", "0") == "1":
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
rect = (1100, 3500, 700, 950)

if os.environ.get("SHOW_PLOTS", "0") == "1":
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(label.cpu(), cmap="gray")
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




## === cell 4
class SubvolumeDataset(data.Dataset):
    def __init__(self, label_cpu: torch.Tensor, pixels_yx):
        self.label = (
            label_cpu  # kept on CPU for DataLoader efficiency; moved to GPU per batch
        )
        if isinstance(pixels_yx, torch.Tensor):
            self.pixels = pixels_yx.to(dtype=torch.long, device="cpu")
        else:
            self.pixels = torch.as_tensor(pixels_yx, dtype=torch.long, device="cpu")

    def __len__(self):
        return int(self.pixels.shape[0])

    def __getitem__(self, index):
        yx = self.pixels[index]  # (2,) long CPU tensor
        y = yx[0]
        x = yx[1]
        inklabel = self.label[y, x].view(1)  # CPU float tensor
        return y, x, inklabel


_PATCH_OFFS_CACHE = {}  # key: (BUFFER, device) -> (offs_y, offs_x, k)
_PADDED_STACK_CACHE = {}  # key: (data_ptr, BUFFER, device) -> padded tensor


@torch.no_grad()
def _get_patch_offsets(buffer: int, device: torch.device):
    key = (buffer, device.type, device.index if device.type == "cuda" else None)
    if key in _PATCH_OFFS_CACHE:
        return _PATCH_OFFS_CACHE[key]
    k = buffer * 2 + 1
    oy, ox = torch.meshgrid(
        torch.arange(-buffer, buffer + 1, device=device, dtype=torch.long),
        torch.arange(-buffer, buffer + 1, device=device, dtype=torch.long),
        indexing="ij",
    )
    oy = oy.reshape(-1)  # (k*k,)
    ox = ox.reshape(-1)  # (k*k,)
    _PATCH_OFFS_CACHE[key] = (oy, ox, k)
    return oy, ox, k


@torch.no_grad()
def _get_padded_stack(image_stack_zhw: torch.Tensor, buffer: int) -> torch.Tensor:
    key = (int(image_stack_zhw.data_ptr()), buffer, image_stack_zhw.device.type)
    if key in _PADDED_STACK_CACHE:
        return _PADDED_STACK_CACHE[key]
    padded = torch.nn.functional.pad(
        image_stack_zhw, (buffer, buffer, buffer, buffer), mode="constant", value=0.0
    )
    _PADDED_STACK_CACHE[key] = padded
    return padded


@torch.no_grad()
def extract_subvolumes_grid_sample(
    image_stack_zhw: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor
) -> torch.Tensor:
    """
    Returns subvolumes of shape (N, 1, Z, k, k) centered at (ys, xs) for each sample.
    Fast equivalent of the original grid_sample(nearest, zeros) implementation.
    """
    oy, ox, k = _get_patch_offsets(BUFFER, ys.device)
    padded = _get_padded_stack(image_stack_zhw, BUFFER)  # (Z, H+2B, W+2B)

    y0 = ys.to(torch.long) + BUFFER
    x0 = xs.to(torch.long) + BUFFER

    y_idx = (y0.unsqueeze(1) + oy.unsqueeze(0)).clamp_(0, padded.shape[1] - 1)
    x_idx = (x0.unsqueeze(1) + ox.unsqueeze(0)).clamp_(0, padded.shape[2] - 1)

    patch = padded[:, y_idx, x_idx]
    patch = patch.permute(1, 0, 2).contiguous().view(-1, image_stack_zhw.shape[0], k, k)
    return patch.unsqueeze(1)


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
arr_mask = np.array(mask) * not_border

inside_rect = np.zeros(mask.shape, dtype=bool) * arr_mask
inside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = True

outside_rect = np.ones(mask.shape, dtype=bool) * arr_mask
outside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = False

pixels_inside_rect = np.argwhere(inside_rect)
pixels_outside_rect = np.argwhere(outside_rect)

print("Training...")

label_cpu = label.detach().to("cpu")
pixels_outside_t = torch.as_tensor(pixels_outside_rect, dtype=torch.long, device=DEVICE)

criterion = nn.BCELoss()
optimizer = optim.ASGD(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=4)

model.train()

_ = _get_padded_stack(image_stack, BUFFER)

gen = torch.Generator(device=DEVICE if DEVICE.type == "cuda" else "cpu")
gen.manual_seed(SEED)

n_train = int(pixels_outside_t.shape[0])
for i in tqdm(range(TRAINING_STEPS), total=TRAINING_STEPS):
    idx = torch.randint(n_train, (BATCH_SIZE,), generator=gen, device=DEVICE)
    yx = pixels_outside_t.index_select(0, idx)
    ys = yx[:, 0]
    xs = yx[:, 1]

    subvolumes = extract_subvolumes_grid_sample(image_stack, ys, xs)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(subvolumes)

    inklabels = label[ys, xs].view(-1, 1)
    loss = criterion(outputs, inklabels)
    loss.backward()
    optimizer.step()
    scheduler.step()



## === cell 6
pixels_inside_t = torch.as_tensor(pixels_inside_rect, dtype=torch.long, device=DEVICE)

output = torch.zeros_like(label).float()
model.eval()

_ = _get_padded_stack(image_stack, BUFFER)

with torch.no_grad():
    n_eval = int(pixels_inside_t.shape[0])
    for start in tqdm(range(0, n_eval, BATCH_SIZE)):
        yx = pixels_inside_t[start : start + BATCH_SIZE]
        ys_d = yx[:, 0]
        xs_d = yx[:, 1]
        subvolumes = extract_subvolumes_grid_sample(image_stack, ys_d, xs_d)
        preds = model(subvolumes).view(-1)
        output[ys_d, xs_d] = preds

if os.environ.get("SHOW_PLOTS", "0") == "1":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(output.cpu(), cmap="gray")
    ax1.set_title("train frag output (rect only)")
    ax1.axis("off")
    ax2.imshow(label.cpu(), cmap="gray")
    ax2.set_title("train frag label")
    ax2.axis("off")
    plt.tight_layout()
    plt.show()



## === cell 7
THRESHOLD = 0.4

if os.environ.get("SHOW_PLOTS", "0") == "1":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(output.gt(THRESHOLD).cpu(), cmap="gray")
    ax1.set_title(f"train frag binarized @ {THRESHOLD}")
    ax1.axis("off")
    ax2.imshow(label.cpu(), cmap="gray")
    ax2.set_title("train frag label")
    ax2.axis("off")
    plt.tight_layout()
    plt.show()




## === cell 8
def rle_from_binary_mask(binary_mask_2d: np.ndarray) -> str:
    """
    Kaggle RLE: pixels numbered left-to-right then top-to-bottom, starting at 1.
    binary_mask_2d must be 0/1 uint8 array.
    """
    pixels = binary_mask_2d.reshape(-1, order="C")
    pixels = np.concatenate(([0], pixels, [0]))
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs.tolist()))


_FRAGMENT_STACK_CACHE = {}
_FRAGMENT_FILES_CACHE = {}


def load_fragment_stack(prefix: str) -> torch.Tensor:
    cache_key = (prefix, Z_START, Z_DIM, DEVICE.type)
    if cache_key in _FRAGMENT_STACK_CACHE:
        return _FRAGMENT_STACK_CACHE[cache_key]

    if prefix in _FRAGMENT_FILES_CACHE:
        files = _FRAGMENT_FILES_CACHE[prefix]
    else:
        files = sorted(glob.glob(os.path.join(prefix, "surface_volume", "*.tif")))
        if len(files) == 0:
            raise FileNotFoundError(
                f"No .tif files found under: {prefix}/surface_volume/"
            )
        _FRAGMENT_FILES_CACHE[prefix] = files

    files = files[Z_START : Z_START + Z_DIM]
    imgs = [np.array(Image.open(f), dtype=np.float32) / 65535.0 for f in files]
    stack = torch.stack([torch.from_numpy(im) for im in imgs], dim=0).to(DEVICE)
    _FRAGMENT_STACK_CACHE[cache_key] = stack
    return stack


def predict_fragment(model: nn.Module, frag_dir: str, threshold: float) -> str:
    mask_np = np.array(
        Image.open(os.path.join(frag_dir, "mask.png")).convert("1")
    ).astype(bool)

    not_border_local = np.zeros(mask_np.shape, dtype=bool)
    not_border_local[
        BUFFER : mask_np.shape[0] - BUFFER, BUFFER : mask_np.shape[1] - BUFFER
    ] = True
    valid_pixels = mask_np & not_border_local
    pixels = np.argwhere(valid_pixels)
    if len(pixels) == 0:
        return ""

    stack = load_fragment_stack(frag_dir)
    _ = _get_padded_stack(stack, BUFFER)

    pixels_t = torch.as_tensor(pixels, dtype=torch.long, device=DEVICE)
    out = torch.zeros(mask_np.shape, dtype=torch.float32, device=DEVICE)

    model.eval()
    with torch.no_grad():
        n = int(pixels_t.shape[0])
        for start in tqdm(range(0, n, BATCH_SIZE), leave=False):
            yx = pixels_t[start : start + BATCH_SIZE]
            ys_d = yx[:, 0]
            xs_d = yx[:, 1]
            subvolumes = extract_subvolumes_grid_sample(stack, ys_d, xs_d)
            preds = model(subvolumes).view(-1)
            out[ys_d, xs_d] = preds

    bin_mask = (out > threshold).detach().cpu().numpy().astype(np.uint8)
    bin_mask[~mask_np] = 0
    return rle_from_binary_mask(bin_mask)




## === cell 9
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = sample_sub["Id"].tolist()

preds = []
for frag_id in tqdm(test_ids, desc="Predicting test fragments"):
    frag_dir = os.path.join(TEST_ROOT, str(frag_id))
    preds.append(predict_fragment(model, frag_dir, threshold=THRESHOLD))

sub = pd.DataFrame({"Id": test_ids, "Predicted": preds})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
print("submission.csv path:", os.path.abspath("submission.csv"))
