# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.11

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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

try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

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
_TIF_CACHE = {}


def _read_tif_float01(path: str) -> np.ndarray:
    arr = _TIF_CACHE.get(path)
    if arr is None:
        arr = np.array(Image.open(path), dtype=np.float32) / 65535.0
        _TIF_CACHE[path] = arr
    return arr


tif_files = sorted(glob.glob(PREFIX + "surface_volume/*.tif"))[
    Z_START : Z_START + Z_DIM
]
images = [_read_tif_float01(fn) for fn in tqdm(tif_files)]
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
        self._k = 2 * BUFFER + 1
        self._HWp = None

        self._padded = torch.nn.functional.pad(
            self.image_stack,
            (self._pad, self._pad, self._pad, self._pad),
            mode="constant",
            value=0.0,
        )  # [Z, H+2B, W+2B] on DEVICE
        _, Hp, Wp = self._padded.shape
        self._HWp = Wp

        dy = torch.arange(self._k, device=DEVICE, dtype=torch.long)
        dx = torch.arange(self._k, device=DEVICE, dtype=torch.long)
        dy2d, dx2d = torch.meshgrid(dy, dx, indexing="ij")
        self._offsets = (dy2d * Wp + dx2d).reshape(-1)  # [k*k]

        self._padded_flat = self._padded.reshape(self._padded.shape[0], -1).contiguous()

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        y, x = self.pixels[index]
        return int(y), int(x)

    def collate_fn(self, batch):
        bx = torch.as_tensor(batch, device=DEVICE, dtype=torch.long)  # [B,2]
        ys = bx[:, 0]
        xs = bx[:, 1]

        base = ys * self._HWp + xs  # [B]

        idx2d = base[:, None] + self._offsets[None, :]

        gathered = self._padded_flat.gather(
            1, idx2d.unsqueeze(0).expand(self._padded_flat.shape[0], -1, -1)
        )
        sub = (
            gathered.permute(1, 0, 2)
            .contiguous()
            .view(-1, self._padded.shape[0], self._k, self._k)
        )
        sub = sub.view(sub.shape[0], 1, Z_DIM, self._k, self._k)

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


def _patched_collate_fn(self, batch):
    bx = torch.as_tensor(batch, device=DEVICE, dtype=torch.long)  # [B,2]
    ys = bx[:, 0]
    xs = bx[:, 1]

    base = ys * self._HWp + xs  # [B]
    idx2d = base[:, None] + self._offsets[None, :]  # [B, k*k]

    Z = self._padded_flat.shape[0]
    idx3d = idx2d.unsqueeze(0).expand(Z, -1, -1)  # [Z, B, k*k]
    inp3d = self._padded_flat.unsqueeze(1).expand(-1, idx2d.shape[0], -1)  # [Z, B, HW]

    gathered = inp3d.gather(2, idx3d)  # [Z, B, k*k]

    sub = (
        gathered.permute(1, 0, 2)
        .contiguous()
        .view(-1, self._padded.shape[0], self._k, self._k)
    )
    sub = sub.view(sub.shape[0], 1, Z_DIM, self._k, self._k)

    ink = self.label[ys, xs].view(-1, 1)
    return sub, ink


train_dataset = SubvolumeDataset(image_stack, label, pixels_outside_rect)
train_dataset.collate_fn = _patched_collate_fn.__get__(train_dataset, SubvolumeDataset)

num_workers = 0 if DEVICE.type == "cuda" else min(4, os.cpu_count() or 1)
train_loader = data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=False,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=train_dataset.collate_fn,
    drop_last=False,
)

criterion = nn.BCELoss()
optimizer = optim.Rprop(model.parameters(), lr=LEARNING_RATE)
scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=2)
scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])

model.train()

_prev_det = None
try:
    _prev_det = torch.are_deterministic_algorithms_enabled()
except Exception:
    _prev_det = None

if _prev_det:
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

try:
    for i, (subvolumes, inklabels) in tqdm(
        enumerate(train_loader), total=TRAINING_STEPS
    ):
        if i >= TRAINING_STEPS:
            break
        optimizer.zero_grad(set_to_none=True)  # speed; gradients remain correct
        outputs = model(subvolumes)
        loss = criterion(outputs, inklabels)
        loss.backward()
        optimizer.step()
        scheduler.step()
finally:
    if _prev_det:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


## === cell 6
eval_dataset = SubvolumeDataset(image_stack, label, pixels_inside_rect)

eval_dataset.collate_fn = _patched_collate_fn.__get__(eval_dataset, SubvolumeDataset)

num_workers = 0 if DEVICE.type == "cuda" else min(4, os.cpu_count() or 1)

eval_loader = data.DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=eval_dataset.collate_fn,
    drop_last=False,
)

output = torch.zeros_like(label).float()
model.eval()

pixels_inside_rect_t = torch.from_numpy(pixels_inside_rect).to(DEVICE, dtype=torch.long)

_prev_det = None
try:
    _prev_det = torch.are_deterministic_algorithms_enabled()
except Exception:
    _prev_det = None

if _prev_det:
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

try:
    with torch.inference_mode():
        idx = 0
        for subvolumes, _ in tqdm(eval_loader):
            preds = model(subvolumes).view(-1)
            b = preds.numel()
            ys = pixels_inside_rect_t[idx : idx + b, 0]
            xs = pixels_inside_rect_t[idx : idx + b, 1]
            output[ys, xs] = preds
            idx += b
finally:
    if _prev_det:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

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
    imgs = [_read_tif_float01(fn) for fn in sel]
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

    num_workers = min(4, os.cpu_count() or 1)
    dl = data.DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=ds.collate_fn,
        drop_last=False,
    )

    out = np.zeros(frag_mask.shape, dtype=np.uint8)
    model.eval()

    pixels_t = torch.from_numpy(pixels).to(DEVICE, dtype=torch.long)

    with torch.inference_mode():
        idx = 0
        for subvolumes, _ in tqdm(dl, leave=False):
            preds = model(subvolumes).view(-1)
            b = preds.numel()
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

## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1459138342.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     74[0m [0;32mfor[0m [0mfid[0m [0;32min[0m [0mtest_ids[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     75[0m     [0mfrag_dir[0m [0;34m=[0m [0mos[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mtest_root[0m[0;34m,[0m [0mfid[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 76[0;31m     [0mpred_mask[0m [0;34m=[0m [0mpredict_fragment[0m[0;34m([0m[0mfrag_dir[0m[0;34m,[0m [0mTHRESHOLD[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     77[0m     [0mpred_rle[0m [0;34m=[0m [0mrle_from_mask[0m[0;34m([0m[0mpred_mask[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     78[0m     [0mrows[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0;34m([0m[0mfid[0m[0;34m,[0m [0mpred_rle[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1459138342.py[0m in [0;36mpredict_fragment[0;34m(fragment_dir, threshold)[0m
[1;32m     53[0m     [0;32mwith[0m [0mtorch[0m[0;34m.[0m[0minference_mode[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     54[0m         [0midx[0m [0;34m=[0m [0;36m0[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 55[0;31m         [0;32mfor[0m [0msubvolumes[0m[0;34m,[0m [0m_[0m [0;32min[0m [0mtqdm[0m[0;34m([0m[0mdl[0m[0;34m,[0m [0mleave[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     56[0m             [0mpreds[0m [0;34m=[0m [0mmodel[0m[0;34m([0m[0msubvolumes[0m[0;34m)[0m[0;34m.[0m[0mview[0m[0;34m([0m[0;34m-[0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     57[0m             [0mb[0m [0;34m=[0m [0mpreds[0m[0;34m.[0m[0mnumel[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tqdm/std.py[0m in [0;36m__iter__[0;34m(self)[0m
[1;32m   1179[0m [0;34m[0m[0m
[1;32m   1180[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1181[0;31m             [0;32mfor[0m [0mobj[0m [0;32min[0m [0miterable[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1182[0m                 [0;32myield[0m [0mobj[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1183[0m                 [0;31m# Update and possibly print the progressbar.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m    706[0m                 [0;31m# TODO(https://github.com/pytorch/pytorch/issues/76750)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    707[0m                 [0mself[0m[0;34m.[0m[0m_reset[0m[0;34m([0m[0;34m)[0m  [0;31m# type: ignore[call-arg][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 708[0;31m             [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_next_data[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    709[0m             [0mself[0m[0;34m.[0m[0m_num_yielded[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m    710[0m             if (

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_next_data[0;34m(self)[0m
[1;32m   1478[0m                 [0;32mdel[0m [0mself[0m[0;34m.[0m[0m_task_info[0m[0;34m[[0m[0midx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1479[0m                 [0mself[0m[0;34m.[0m[0m_rcvd_idx[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1480[0;31m                 [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_process_data[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1481[0m [0;34m[0m[0m
[1;32m   1482[0m     [0;32mdef[0m [0m_try_put_index[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_process_data[0;34m(self, data)[0m
[1;32m   1503[0m         [0mself[0m[0;34m.[0m[0m_try_put_index[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1504[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mExceptionWrapper[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1505[0;31m             [0mdata[0m[0;34m.[0m[0mreraise[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1506[0m         [0;32mreturn[0m [0mdata[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1507[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_utils.py[0m in [0;36mreraise[0;34m(self)[0m
[1;32m    731[0m             [0;31m# instantiate since we don't know how to[0m[0;34m[0m[0;34m[0m[0m
[1;32m    732[0m             [0;32mraise[0m [0mRuntimeError[0m[0;34m([0m[0mmsg[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 733[0;31m         [0;32mraise[0m [0mexception[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    734[0m [0;34m[0m[0m
[1;32m    735[0m [0;34m[0m[0m

[0;31mRuntimeError[0m: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/2623515552.py", line 42, in collate_fn
    bx = torch.as_tensor(batch, device=DEVICE, dtype=torch.long)  # [B,2]
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/cuda/__init__.py", line 305, in _lazy_init
    raise RuntimeError(
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
