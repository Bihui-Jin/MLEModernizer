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
import random
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


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

_CANDIDATE_ROOTS = [
    "/kaggle/input/vesuvius-challenge-ink-detection/",
    "/kaggle/data/vesuvius-challenge-ink-detection/",
    "/kaggle/input/vesuvius-challenge/",
    "/kaggle/data/vesuvius-challenge/",
]
DATA_ROOT = next(
    (p for p in _CANDIDATE_ROOTS if os.path.exists(p)), _CANDIDATE_ROOTS[0]
)
if not DATA_ROOT.endswith("/"):
    DATA_ROOT += "/"

BUFFER = 30  # Buffer size in x and y direction
Z_START = 27  # First slice in the z direction to use
Z_DIM = 10  # Number of slices in the z direction
TRAINING_STEPS = 20000
LEARNING_RATE = 0.05
BATCH_SIZE = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rect = (1100, 3500, 700, 950)

print("DATA_ROOT =", DATA_ROOT)
print("DEVICE =", DEVICE)




## === cell 1
def load_image_stack(
    fragment_dir: str,
    z_start: int = Z_START,
    z_dim: int = Z_DIM,
    device: torch.device | None = None,
) -> torch.Tensor:
    tif_files = sorted(glob.glob(os.path.join(fragment_dir, "surface_volume", "*.tif")))
    if len(tif_files) == 0:
        raise FileNotFoundError(
            f"No .tif files found under {fragment_dir}/surface_volume/"
        )
    tif_files = tif_files[z_start : z_start + z_dim]
    if len(tif_files) != z_dim:
        raise ValueError(
            f"Expected {z_dim} slices but found {len(tif_files)} in {fragment_dir}"
        )
    images = [np.array(Image.open(fn), dtype=np.float32) / 65535.0 for fn in tif_files]
    stack = torch.stack([torch.from_numpy(im) for im in images], dim=0)  # (Z,H,W)
    if device is not None:
        stack = stack.to(device)
    return stack


def load_mask(fragment_dir: str) -> np.ndarray:
    mpath = os.path.join(fragment_dir, "mask.png")
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"Missing mask.png at {mpath}")
    return np.array(Image.open(mpath).convert("1"))


def load_label(fragment_dir: str, device: torch.device | None = None) -> torch.Tensor:
    lpath = os.path.join(fragment_dir, "inklabels.png")
    if not os.path.exists(lpath):
        raise FileNotFoundError(f"Missing inklabels.png at {lpath}")
    t = torch.from_numpy(np.array(Image.open(lpath))).gt(0).float()
    if device is not None:
        t = t.to(device)
    return t


PREFIX = os.path.join(DATA_ROOT, "train", "1") + "/"
_ir_path = os.path.join(PREFIX, "ir.png")
if os.path.exists(_ir_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(Image.open(_ir_path), cmap="gray")
    plt.title("train/1 ir.png")
    plt.axis("off")
    plt.show()



## === cell 2
mask = load_mask(PREFIX)
label = load_label(PREFIX, device=None)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title("mask.png")
ax1.imshow(mask, cmap="gray")
ax1.axis("off")
ax2.set_title("inklabels.png")
ax2.imshow(label, cmap="gray")
ax2.axis("off")
plt.tight_layout()
plt.show()



## === cell 3
image_stack = load_image_stack(PREFIX, Z_START, Z_DIM, device=None)

images_np = [image_stack[z].detach().cpu().numpy() for z in range(image_stack.shape[0])]
fig, axes = plt.subplots(1, len(images_np), figsize=(15, 3))
for image, ax in zip(images_np, axes):
    im_small = np.array(
        Image.fromarray(image).resize((image.shape[1] // 20, image.shape[0] // 20))
    )
    ax.imshow(im_small, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.show()



## === cell 4
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(label.cpu(), cmap="gray")
patch = patches.Rectangle(
    (rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor="r", facecolor="none"
)
ax.add_patch(patch)
ax.set_title("train/1 label with rect")
ax.axis("off")
plt.show()




## === cell 5
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




## === cell 6
def build_pixels_for_training(fragment_dir: str) -> np.ndarray:
    m = load_mask(fragment_dir)
    nb = np.zeros(m.shape, dtype=bool)
    nb[BUFFER : m.shape[0] - BUFFER, BUFFER : m.shape[1] - BUFFER] = True
    arr_mask = np.array(m) * nb
    return np.argwhere(arr_mask)


train_fragment_dirs = [
    os.path.join(DATA_ROOT, "train", "1"),
    os.path.join(DATA_ROOT, "train", "2"),
]

print("Generating pixel lists (train/1 + train/2)...")
train_datasets = []
for fdir in train_fragment_dirs:
    stack = load_image_stack(fdir, Z_START, Z_DIM, device=DEVICE)
    lbl = load_label(fdir, device=DEVICE)
    pixels = build_pixels_for_training(fdir)
    train_datasets.append(SubvolumeDataset(stack, lbl, pixels))

train_dataset = data.ConcatDataset(train_datasets)

train_loader = data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=False,
)

print("Training...")
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
scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [scheduler1, scheduler2, scheduler3, scheduler4, scheduler5, scheduler6]
)

model.train()
for i, (subvolumes, inklabels) in tqdm(enumerate(train_loader), total=TRAINING_STEPS):
    if i >= TRAINING_STEPS:
        break
    optimizer.zero_grad(set_to_none=True)
    outputs = model(subvolumes.to(DEVICE, non_blocking=True))
    loss = criterion(outputs, inklabels.to(DEVICE, non_blocking=True))
    loss.backward()
    optimizer.step()
    scheduler.step()


## === cell 7
print("Evaluating on train/1 inside rect (sanity-check)...")
m1 = load_mask(os.path.join(DATA_ROOT, "train", "1"))
lbl1_cpu = load_label(os.path.join(DATA_ROOT, "train", "1"), device=None)
stack1 = load_image_stack(
    os.path.join(DATA_ROOT, "train", "1"), Z_START, Z_DIM, device=DEVICE
)

not_border = np.zeros(m1.shape, dtype=bool)
not_border[BUFFER : m1.shape[0] - BUFFER, BUFFER : m1.shape[1] - BUFFER] = True
arr_mask = np.array(m1).astype(bool) & not_border

inside_rect = np.zeros(m1.shape, dtype=bool)
inside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = True
inside_rect = inside_rect & arr_mask

pixels_inside_rect = np.argwhere(inside_rect)

dummy_label = torch.zeros(
    m1.shape, dtype=torch.float32
)  # CPU labels are fine for visualization
eval_dataset = SubvolumeDataset(stack1, dummy_label, pixels_inside_rect)

eval_loader = data.DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=False,
)

output = torch.zeros_like(lbl1_cpu).float()
model.eval()
with torch.no_grad():
    for i, (subvolumes, _) in enumerate(tqdm(eval_loader)):
        preds = model(subvolumes.to(DEVICE, non_blocking=True)).view(-1).detach().cpu()
        base = i * BATCH_SIZE
        for j, value in enumerate(preds):
            if base + j >= len(pixels_inside_rect):
                break
            yx = tuple(pixels_inside_rect[base + j])
            output[yx] = value

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(output.cpu(), cmap="gray")
ax1.set_title("train/1 output (inside rect)")
ax1.axis("off")
ax2.imshow(lbl1_cpu.cpu(), cmap="gray")
ax2.set_title("train/1 label")
ax2.axis("off")
plt.tight_layout()
plt.show()


## === cell 8
def fbeta_score_from_masks(
    y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5
) -> float:
    y_true = (y_true.astype(np.uint8) > 0).reshape(-1)
    y_pred = (y_pred.astype(np.uint8) > 0).reshape(-1)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    if tp == 0:
        return 0.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2 = beta * beta
    denom = b2 * p + r
    return float(((1 + b2) * p * r / denom) if denom > 0 else 0.0)


def rle_from_binary_mask(binary_mask_2d: np.ndarray) -> str:
    if binary_mask_2d.size == 0:
        return ""
    pixels = binary_mask_2d.reshape(-1).astype(np.uint8)
    padded = np.concatenate([[0], pixels, [0]])
    changes = (
        np.where(padded[1:] != padded[:-1])[0] + 1
    )  # 1-indexed positions due to left padding
    runs = changes.copy()
    runs[1::2] = runs[1::2] - runs[::2]
    return " ".join(str(int(x)) for x in runs)


def predict_fragment_prob_map(fragment_dir: str) -> tuple[np.ndarray, np.ndarray]:
    m = load_mask(fragment_dir)
    stack = load_image_stack(fragment_dir, Z_START, Z_DIM, device=DEVICE)

    not_border = np.zeros(m.shape, dtype=bool)
    not_border[BUFFER : m.shape[0] - BUFFER, BUFFER : m.shape[1] - BUFFER] = True
    valid = np.array(m).astype(bool) & not_border

    pixels = np.argwhere(valid)
    prob = np.zeros((m.shape[0], m.shape[1]), dtype=np.float32)
    if len(pixels) == 0:
        return prob, valid

    dummy_label = torch.zeros(m.shape, dtype=torch.float32, device=DEVICE)
    ds = SubvolumeDataset(stack, dummy_label, pixels)

    dl = data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=False,
    )

    flat_prob = prob.reshape(-1)
    idx_flat = (pixels[:, 0] * m.shape[1] + pixels[:, 1]).astype(np.int64)

    model.eval()
    with torch.no_grad():
        for i, (subvolumes, _) in enumerate(tqdm(dl, leave=False)):
            preds = (
                model(subvolumes.to(DEVICE, non_blocking=True))
                .view(-1)
                .detach()
                .cpu()
                .numpy()
            )
            base = i * BATCH_SIZE
            end = min(base + preds.shape[0], idx_flat.shape[0])
            flat_prob[idx_flat[base:end]] = preds[: (end - base)]

    prob[~valid] = 0.0
    return prob, valid


def tune_threshold_on_train(candidates: np.ndarray | None = None) -> float:
    if candidates is None:
        candidates = np.array(
            [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55], dtype=np.float32
        )

    train_ids = ["1", "2"]
    cached = []
    for tid in train_ids:
        fdir = os.path.join(DATA_ROOT, "train", tid)
        print(f"Building prob map for threshold tuning on train/{tid} ...")
        prob, valid = predict_fragment_prob_map(fdir)
        y_true = load_label(fdir, device=None).numpy().astype(np.uint8)
        y_true = (y_true > 0).astype(np.uint8)
        cached.append((prob, valid, y_true))

    best_t = float(candidates[0])
    best_s = -1.0
    for t in candidates:
        scores = []
        for prob, valid, y_true in cached:
            y_pred = ((prob > float(t)) & valid).astype(np.uint8)
            y_true_v = (y_true & valid.astype(np.uint8)).astype(np.uint8)
            scores.append(fbeta_score_from_masks(y_true_v, y_pred, beta=0.5))
        s = float(np.mean(scores))
        if s > best_s:
            best_s = s
            best_t = float(t)

    print(
        f"Chosen THRESHOLD = {best_t:.3f} (mean F0.5 on train/1+2 valid pixels: {best_s:.6f})"
    )
    return best_t


THRESHOLD = tune_threshold_on_train()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(output.gt(THRESHOLD).cpu(), cmap="gray")
ax1.set_title(f"train/1 thresholded output (TH={THRESHOLD:.3f})")
ax1.axis("off")
ax2.imshow(lbl1_cpu.cpu(), cmap="gray")
ax2.set_title("train/1 label")
ax2.axis("off")
plt.tight_layout()
plt.show()


## === cell 9
def predict_fragment_rle(fragment_dir: str) -> str:
    prob, valid = predict_fragment_prob_map(fragment_dir)
    binary = ((prob > float(THRESHOLD)) & valid).astype(np.uint8)
    return rle_from_binary_mask(binary)


sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
sample = pd.read_csv(sample_path)

test_root = os.path.join(DATA_ROOT, "test")
if os.path.exists(os.path.join(test_root, "test")):
    test_root = os.path.join(test_root, "test")

preds = []
for fid in sample["Id"].tolist():
    fdir = os.path.join(test_root, str(fid))
    if not os.path.exists(fdir):
        preds.append("")
        continue
    print(f"Inferencing test fragment: {fid}")
    preds.append(predict_fragment_rle(fdir))

submission = pd.DataFrame({"Id": sample["Id"], "Predicted": preds})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
