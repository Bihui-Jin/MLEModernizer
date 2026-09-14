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
label = torch.from_numpy(np.array(Image.open(label_path))).gt(0).float().to(DEVICE)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title("mask.png")
ax1.imshow(mask, cmap="gray")
ax1.axis("off")
ax2.set_title("inklabels.png")
ax2.imshow(label.detach().cpu(), cmap="gray")
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
image_stack = torch.stack([torch.from_numpy(im) for im in images], dim=0).to(
    DEVICE
)  # (Z, H, W)

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
ax.imshow(label.detach().cpu(), cmap="gray")
patch = patches.Rectangle(
    (rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor="r", facecolor="none"
)
ax.add_patch(patch)
ax.set_title("Training label with rect")
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

arr_mask = mask.astype(bool) & not_border

inside_rect = np.zeros(mask.shape, dtype=bool)
inside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = True
inside_rect &= arr_mask

outside_rect = arr_mask.copy()
outside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = False

pixels_inside_rect = np.argwhere(inside_rect)
pixels_outside_rect = np.argwhere(outside_rect)

print("pixels_inside_rect:", len(pixels_inside_rect))
print("pixels_outside_rect:", len(pixels_outside_rect))

if len(pixels_outside_rect) == 0:
    raise RuntimeError(
        "No training pixels found (outside_rect empty). Check mask/BUFFER/rect."
    )



## === cell 6
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
optimizer = optim.ASGD(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=4)

model.train()
pbar = tqdm(enumerate(train_loader), total=TRAINING_STEPS)
for i, (subvolumes, inklabels) in pbar:
    if i >= TRAINING_STEPS:
        break
    optimizer.zero_grad(set_to_none=True)
    outputs = model(subvolumes.to(DEVICE))
    loss = criterion(outputs, inklabels.to(DEVICE))
    loss.backward()
    optimizer.step()
    scheduler.step()
    if (i + 1) % 200 == 0:
        pbar.set_postfix(loss=float(loss.detach().cpu()))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/535454244.py in <cell line: 0>()
     18 model.train()
     19 pbar = tqdm(enumerate(train_loader), total=TRAINING_STEPS)
---> 20 for i, (subvolumes, inklabels) in pbar:
     21     if i >= TRAINING_STEPS:
     22         break

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
    for bi, (subvolumes, _) in enumerate(tqdm(eval_loader, desc="Eval rect")):
        preds = model(subvolumes.to(DEVICE)).detach().cpu().view(-1)
        start = bi * BATCH_SIZE
        end = start + len(preds)
        coords = pixels_inside_rect[start:end]
        output[coords[:, 0], coords[:, 1]] = preds.to(output.dtype)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title("Model output (rect area only)")
ax1.imshow(output.detach().cpu(), cmap="gray")
ax1.axis("off")
ax2.set_title("Label")
ax2.imshow(label.detach().cpu(), cmap="gray")
ax2.axis("off")
plt.tight_layout()
plt.show()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3025637071.py in <cell line: 0>()
     12 model.eval()
     13 with torch.no_grad():
---> 14     for bi, (subvolumes, _) in enumerate(tqdm(eval_loader, desc="Eval rect")):
     15         preds = model(subvolumes.to(DEVICE)).detach().cpu().view(-1)
     16         start = bi * BATCH_SIZE

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

## === cell 8
THRESHOLD = 0.4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title(f"Output > {THRESHOLD}")
ax1.imshow(output.gt(THRESHOLD).detach().cpu(), cmap="gray")
ax1.axis("off")
ax2.set_title("Label")
ax2.imshow(label.detach().cpu(), cmap="gray")
ax2.axis("off")
plt.tight_layout()
plt.show()




## === cell 9
def rle_from_binary_mask(binary_mask_2d: np.ndarray) -> str:
    """
    binary_mask_2d: HxW, values {0,1} or bool. RLE must be 1-indexed, row-major.
    """
    pixels = binary_mask_2d.astype(np.uint8).flatten(order="C")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = (
        np.where(pixels[1:] != pixels[:-1])[0] + 1
    )  # positions in this padded array (0-index)
    runs[1::2] = runs[1::2] - runs[0::2]
    runs[0::2] = runs[0::2]  # already accounts for padding shift via +1 above
    return " ".join(str(x) for x in runs)


def load_fragment_stack(fragment_dir: str) -> torch.Tensor:
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
    stack = torch.stack([torch.from_numpy(im) for im in imgs], dim=0).to(
        DEVICE
    )  # (Z,H,W)
    return stack


def predict_fragment_rle(fragment_dir: str) -> str:
    m = np.array(
        Image.open(os.path.join(fragment_dir, "mask.png")).convert("1")
    ).astype(bool)
    H, W = m.shape
    not_border_local = np.zeros((H, W), dtype=bool)
    not_border_local[BUFFER : H - BUFFER, BUFFER : W - BUFFER] = True
    valid = m & not_border_local
    pixels = np.argwhere(valid)

    if len(pixels) == 0:
        return ""

    stack = load_fragment_stack(fragment_dir)

    dummy_label = torch.zeros((H, W), device=DEVICE, dtype=torch.float32)
    ds = SubvolumeDataset(stack, dummy_label, pixels)
    loader = data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    probs = torch.zeros((H, W), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for bi, (subvolumes, _) in enumerate(
            tqdm(loader, desc=f"Predict {os.path.basename(fragment_dir)}", leave=False)
        ):
            pred = model(subvolumes.to(DEVICE)).detach().cpu().view(-1)
            start = bi * BATCH_SIZE
            end = start + len(pred)
            coords = pixels[start:end]
            probs[coords[:, 0], coords[:, 1]] = pred

    binary = probs.numpy() > THRESHOLD
    binary &= m
    return rle_from_binary_mask(binary)


sub = pd.read_csv(SAMPLE_SUB_PATH)
preds = []
for frag_id in sub["Id"].astype(str).tolist():
    frag_dir = os.path.join(TEST_ROOT, frag_id)
    if not os.path.isdir(frag_dir):
        raise FileNotFoundError(f"Test fragment directory not found: {frag_dir}")
    preds.append(predict_fragment_rle(frag_dir))

submission = pd.DataFrame({"Id": sub["Id"].astype(str), "Predicted": preds})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")
print("submission.csv path:", os.path.abspath("submission.csv"))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2796015980.py in <cell line: 0>()
     88         # Fallback: if test structure differs, try nested under COMP_ROOT/test/<id>
     89         raise FileNotFoundError(f"Test fragment directory not found: {frag_dir}")
---> 90     preds.append(predict_fragment_rle(frag_dir))
     91 
     92 submission = pd.DataFrame({"Id": sub["Id"].astype(str), "Predicted": preds})

/tmp/ipykernel_11/2796015980.py in predict_fragment_rle(fragment_dir)
     65     model.eval()
     66     with torch.no_grad():
---> 67         for bi, (subvolumes, _) in enumerate(
     68             tqdm(loader, desc=f"Predict {os.path.basename(fragment_dir)}", leave=False)
     69         ):

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
