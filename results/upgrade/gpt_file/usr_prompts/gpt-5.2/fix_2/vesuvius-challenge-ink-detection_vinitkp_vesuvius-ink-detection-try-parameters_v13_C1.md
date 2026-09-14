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
label = torch.from_numpy(np.array(Image.open(label_path))).gt(0).float().to(DEVICE)

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

images = [
    np.array(Image.open(fn), dtype=np.float32) / 65535.0
    for fn in tqdm(slice_files, desc="Loading train slices")
]
image_stack = torch.stack([torch.from_numpy(im) for im in images], dim=0).to(
    DEVICE
)  # [Z, H, W]

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

print("image_stack:", tuple(image_stack.shape), image_stack.dtype)



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
    def __init__(self, image_stack, label, pixels):
        self.image_stack = image_stack
        self.label = label
        self.pixels = pixels

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        y, x = self.pixels[index]
        subvolume = self.image_stack[
            :, y - BUFFER : y + BUFFER + 1, x - BUFFER : x + BUFFER + 1
        ].view(1, Z_DIM, BUFFER * 2 + 1, BUFFER * 2 + 1)
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
train_loader = data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
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
/tmp/ipykernel_11/3442775275.py in <cell line: 0>()
     35 
     36 model.train()
---> 37 for i, (subvolumes, inklabels) in tqdm(
     38     enumerate(train_loader), total=TRAINING_STEPS, desc="Train iters"
     39 ):

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
    764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
--> 766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
    767         return data
    768 

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in pin_memory(data, device)
     96                 clone = copy.copy(data)  # type: ignore[arg-type]
     97                 for i, item in enumerate(data):
---> 98                     clone[i] = pin_memory(item, device)
     99                 return clone
    100             return type(data)([pin_memory(sample, device) for sample in data])  # type: ignore[call-arg]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in pin_memory(data, device)
     62 def pin_memory(data, device=None):
     63     if isinstance(data, torch.Tensor):
---> 64         return data.pin_memory(device)
     65     elif isinstance(data, (str, bytes)):
     66         return data

RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned

## === cell 6
eval_dataset = SubvolumeDataset(image_stack, label, pixels_inside_rect)
eval_loader = data.DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

output = torch.zeros_like(label).float()
model.eval()
with torch.no_grad():
    for i, (subvolumes, _) in enumerate(tqdm(eval_loader, desc="Eval rect")):
        preds = model(subvolumes.to(DEVICE, non_blocking=True)).view(-1).detach().cpu()
        start = i * BATCH_SIZE
        end = start + len(preds)
        coords = pixels_inside_rect[start:end]
        output[coords[:, 0], coords[:, 1]] = preds.to(output.device)

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
/tmp/ipykernel_11/3404311854.py in <cell line: 0>()
     12 model.eval()
     13 with torch.no_grad():
---> 14     for i, (subvolumes, _) in enumerate(tqdm(eval_loader, desc="Eval rect")):
     15         preds = model(subvolumes.to(DEVICE, non_blocking=True)).view(-1).detach().cpu()
     16         start = i * BATCH_SIZE

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
    764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
--> 766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
    767         return data
    768 

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in pin_memory(data, device)
     96                 clone = copy.copy(data)  # type: ignore[arg-type]
     97                 for i, item in enumerate(data):
---> 98                     clone[i] = pin_memory(item, device)
     99                 return clone
    100             return type(data)([pin_memory(sample, device) for sample in data])  # type: ignore[call-arg]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in pin_memory(data, device)
     62 def pin_memory(data, device=None):
     63     if isinstance(data, torch.Tensor):
---> 64         return data.pin_memory(device)
     65     elif isinstance(data, (str, bytes)):
     66         return data

RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned

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




## === cell 8
def rle_from_binary_mask(mask_2d: np.ndarray) -> str:
    """
    Kaggle Vesuvius expects RLE over the flattened image in row-major order,
    using 1-indexed positions and (start, length) pairs.
    """
    pixels = mask_2d.flatten(order="C").astype(np.uint8)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # positions in padded array
    runs[1::2] = runs[1::2] - runs[:-1:2]
    runs[::2] = runs[::2]
    return " ".join(str(x) for x in runs)


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
    imgs = [np.array(Image.open(fn), dtype=np.float32) / 65535.0 for fn in slice_files]
    stack = torch.stack([torch.from_numpy(im) for im in imgs], dim=0).to(
        DEVICE
    )  # [Z,H,W]

    not_border = np.zeros(mask.shape, dtype=bool)
    not_border[BUFFER : mask.shape[0] - BUFFER, BUFFER : mask.shape[1] - BUFFER] = True
    valid = mask & not_border

    coords = np.argwhere(valid)  # (y,x)
    ds = SubvolumeDataset(
        stack, torch.zeros((mask.shape[0], mask.shape[1]), device=DEVICE), coords
    )
    dl = data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
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
/tmp/ipykernel_11/3068427475.py in <cell line: 0>()
     78 for fid in test_ids:
     79     frag_dir = os.path.join(TEST_ROOT, fid)
---> 80     preds.append(predict_fragment_rle(frag_dir, THRESHOLD))
     81 
     82 sub_df = pd.DataFrame({"Id": test_ids, "Predicted": preds})

/tmp/ipykernel_11/3068427475.py in predict_fragment_rle(fragment_dir, threshold)
     56     model.eval()
     57     with torch.no_grad():
---> 58         for i, (subvolumes, _) in enumerate(
     59             tqdm(dl, desc=f"Predict {os.path.basename(fragment_dir)}", leave=False)
     60         ):

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
    764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
--> 766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
    767         return data
    768 

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in pin_memory(data, device)
     96                 clone = copy.copy(data)  # type: ignore[arg-type]
     97                 for i, item in enumerate(data):
---> 98                     clone[i] = pin_memory(item, device)
     99                 return clone
    100             return type(data)([pin_memory(sample, device) for sample in data])  # type: ignore[call-arg]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py in pin_memory(data, device)
     62 def pin_memory(data, device=None):
     63     if isinstance(data, torch.Tensor):
---> 64         return data.pin_memory(device)
     65     elif isinstance(data, (str, bytes)):
     66         return data

RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
