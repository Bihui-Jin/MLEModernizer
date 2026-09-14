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
import random
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

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_base_candidates = [
    "/kaggle/input/vesuvius-challenge/",  # original (may exist on Kaggle)
    "/kaggle/input/vesuvius-challenge-ink-detection/",  # present in this environment
    "/kaggle/data/vesuvius-challenge-ink-detection/",  # alternate mount in this environment
]
BASE = next((p for p in _base_candidates if os.path.exists(p)), _base_candidates[1])

if os.path.exists(os.path.join(BASE, "train")) and os.path.exists(
    os.path.join(BASE, "test")
):
    DATA_ROOT = BASE
else:
    DATA_ROOT = os.path.join(BASE, "vesuvius-challenge-ink-detection")

PREFIX = os.path.join(DATA_ROOT, "train", "1") + "/"

BUFFER = 30  # Buffer size in x and y direction
Z_START = 27  # First slice in the z direction to use
Z_DIM = 10  # Number of slices in the z direction
TRAINING_STEPS = 20000
LEARNING_RATE = 0.05
BATCH_SIZE = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.figure(figsize=(6, 6))
plt.imshow(Image.open(PREFIX + "ir.png"), cmap="gray")
plt.axis("off")
plt.show()



## === cell 1
mask = np.array(Image.open(PREFIX + "mask.png").convert("1"), dtype=np.uint8)
label = (
    torch.from_numpy(np.array(Image.open(PREFIX + "inklabels.png")))
    .gt(0)
    .float()
    .to(DEVICE)
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title("mask.png")
ax1.imshow(mask, cmap="gray")
ax1.axis("off")
ax2.set_title("inklabels.png")
ax2.imshow(label.detach().cpu(), cmap="gray")
ax2.axis("off")
plt.show()



## === cell 2
tif_files = sorted(glob.glob(PREFIX + "surface_volume/*.tif"))[
    Z_START : Z_START + Z_DIM
]
images = [
    np.array(Image.open(fn), dtype=np.float32) / 65535.0 for fn in tqdm(tif_files)
]
image_stack = torch.stack([torch.from_numpy(im) for im in images], dim=0).to(DEVICE)

fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
for image, ax in zip(images, axes):
    ax.imshow(
        np.array(
            Image.fromarray(image).resize((image.shape[1] // 20, image.shape[0] // 20)),
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
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(label.detach().cpu())
patch = patches.Rectangle(
    (rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor="r", facecolor="none"
)
ax.add_patch(patch)
ax.axis("off")
plt.show()




## === cell 4
class SubvolumeDataset(data.Dataset):
    def __init__(self, image_stack, label, pixels):
        self.image_stack = image_stack  # [Z, H, W] on DEVICE
        self.label = label  # [H, W] on DEVICE
        self.pixels = np.ascontiguousarray(pixels, dtype=np.int64)  # [N, 2] (y, x), CPU

        self._pad = BUFFER
        self._padded = torch.nn.functional.pad(
            self.image_stack,
            (self._pad, self._pad, self._pad, self._pad),
            mode="constant",
            value=0.0,
        )  # [Z, H+2B, W+2B] on DEVICE

        k = 2 * BUFFER + 1
        dy, dx = torch.meshgrid(
            torch.arange(k, device=DEVICE, dtype=torch.long),
            torch.arange(k, device=DEVICE, dtype=torch.long),
            indexing="ij",
        )
        self._dy = dy  # [k, k]
        self._dx = dx  # [k, k]

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        y, x = self.pixels[index]
        return int(y), int(x)

    def collate_fn(self, batch):
        bx = torch.as_tensor(batch, device=DEVICE, dtype=torch.long)  # [B, 2]
        ys = bx[:, 0]
        xs = bx[:, 1]

        yy = ys[:, None, None] + self._dy[None, :, :]
        xx = xs[:, None, None] + self._dx[None, :, :]

        sub = self._padded[:, yy, xx].permute(1, 0, 2, 3).contiguous()
        sub = sub.view(sub.shape[0], 1, Z_DIM, 2 * BUFFER + 1, 2 * BUFFER + 1)

        ink = self.label[ys, xs].view(-1, 1)
        return sub, ink


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
arr_mask = (mask.astype(bool)) & not_border

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

print("Training...")
train_dataset = SubvolumeDataset(image_stack, label, pixels_outside_rect)

train_loader = data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    collate_fn=train_dataset.collate_fn,
    drop_last=False,
)

criterion = nn.BCELoss()
optimizer = optim.Rprop(model.parameters(), lr=LEARNING_RATE)
scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=2)
scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])

model.train()
for i, (subvolumes, inklabels) in tqdm(enumerate(train_loader), total=TRAINING_STEPS):
    if i >= TRAINING_STEPS:
        break
    optimizer.zero_grad(set_to_none=True)  # speed; gradients remain correct
    outputs = model(subvolumes)
    loss = criterion(outputs, inklabels)
    loss.backward()
    optimizer.step()
    scheduler.step()



## === cell 6
eval_dataset = SubvolumeDataset(image_stack, label, pixels_inside_rect)
eval_loader = data.DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    collate_fn=eval_dataset.collate_fn,
    drop_last=False,
)

output = torch.zeros_like(label).float()
model.eval()

pixels_inside_rect_t = torch.from_numpy(pixels_inside_rect).to(DEVICE, dtype=torch.long)

with torch.no_grad():
    idx = 0
    for subvolumes, _ in tqdm(eval_loader):
        preds = model(subvolumes).view(-1)
        b = preds.numel()
        ys = pixels_inside_rect_t[idx : idx + b, 0]
        xs = pixels_inside_rect_t[idx : idx + b, 1]
        output[ys, xs] = preds
        idx += b

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title("model output (train rect)")
ax1.imshow(output.detach().cpu(), cmap="gray")
ax1.axis("off")
ax2.set_title("label (train)")
ax2.imshow(label.detach().cpu(), cmap="gray")
ax2.axis("off")
plt.show()




## === cell 7
def fbeta_score_numpy(y_true, y_pred, beta=0.5, eps=1e-9):
    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r + eps)


calib_valid = arr_mask
y_true = label.detach().cpu().numpy()[calib_valid].astype(np.uint8)
y_prob = output.detach().cpu().numpy()[calib_valid].astype(np.float32)

thresholds = np.linspace(0.1, 0.9, 17, dtype=np.float32)
best_t, best_f = 0.4, -1.0
for t in thresholds:
    y_pred = (y_prob > float(t)).astype(np.uint8)
    f = fbeta_score_numpy(y_true, y_pred, beta=0.5)
    if f > best_f:
        best_f, best_t = f, float(t)

THRESHOLD = best_t
print(f"Calibrated THRESHOLD={THRESHOLD:.3f} (train fragment F0.5={best_f:.6f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title(f"binarized output @ t={THRESHOLD:.3f}")
ax1.imshow((output > THRESHOLD).detach().cpu(), cmap="gray")
ax1.axis("off")
ax2.set_title("label (train)")
ax2.imshow(label.detach().cpu(), cmap="gray")
ax2.axis("off")
plt.show()




## === cell 8
def rle_from_mask(mask2d_uint8):
    pixels = mask2d_uint8.flatten(order="C")
    pixels = np.concatenate([[0], pixels, [0]]).astype(np.uint8)
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs = changes.copy()
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def load_image_stack(fragment_dir):
    sv_dir = os.path.join(fragment_dir, "surface_volume")
    tif_files = sorted(glob.glob(os.path.join(sv_dir, "*.tif")))
    sel = tif_files[Z_START : Z_START + Z_DIM]
    imgs = [np.array(Image.open(fn), dtype=np.float32) / 65535.0 for fn in sel]
    stk = torch.stack([torch.from_numpy(im) for im in imgs], dim=0).to(DEVICE)
    return stk


def predict_fragment(fragment_dir, threshold):
    mask_path = os.path.join(fragment_dir, "mask.png")
    frag_mask = np.array(Image.open(mask_path).convert("1"), dtype=np.uint8)

    not_border = np.zeros(frag_mask.shape, dtype=bool)
    not_border[
        BUFFER : frag_mask.shape[0] - BUFFER, BUFFER : frag_mask.shape[1] - BUFFER
    ] = True
    valid = (frag_mask.astype(bool)) & not_border

    pixels = np.argwhere(valid)
    dummy_label = torch.zeros(frag_mask.shape, dtype=torch.float32, device=DEVICE)

    stk = load_image_stack(fragment_dir)
    ds = SubvolumeDataset(stk, dummy_label, pixels)
    dl = data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=ds.collate_fn,
        drop_last=False,
    )

    out = np.zeros(frag_mask.shape, dtype=np.uint8)
    model.eval()

    pixels_t = torch.from_numpy(pixels).to(DEVICE, dtype=torch.long)

    with torch.no_grad():
        idx = 0
        for subvolumes, _ in tqdm(dl, leave=False):
            preds = model(subvolumes).view(-1)
            b = preds.numel()
            ys = pixels_t[idx : idx + b, 0]
            xs = pixels_t[idx : idx + b, 1]
            pred_cpu = (preds > threshold).to(torch.uint8).detach().cpu().numpy()
            out[pixels[idx : idx + b, 0], pixels[idx : idx + b, 1]] = pred_cpu
            idx += b

    out[~valid] = 0
    return out


test_root = os.path.join(DATA_ROOT, "test")
test_ids = sorted(
    [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
)

print("Test fragments found:", test_ids)

rows = []
for fid in test_ids:
    frag_dir = os.path.join(test_root, fid)
    pred_mask = predict_fragment(frag_dir, THRESHOLD)
    pred_rle = rle_from_mask(pred_mask)
    rows.append((fid, pred_rle))

sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    pred_map = {k: v for k, v in rows}
    sub = sample.copy()
    sub["Predicted"] = sub["Id"].map(pred_map).fillna("")
else:
    sub = pd.DataFrame(rows, columns=["Id", "Predicted"])

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
