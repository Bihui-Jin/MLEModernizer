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
        pin_memory=torch.cuda.is_available(),
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




## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1688502001.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    110[0m [0;34m[0m[0m
[1;32m    111[0m [0;34m[0m[0m
[0;32m--> 112[0;31m [0mTHRESHOLD[0m [0;34m=[0m [0mtune_threshold_on_train[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    113[0m [0;34m[0m[0m
[1;32m    114[0m [0mfig[0m[0;34m,[0m [0;34m([0m[0max1[0m[0;34m,[0m [0max2[0m[0;34m)[0m [0;34m=[0m [0mplt[0m[0;34m.[0m[0msubplots[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0;36m2[0m[0;34m,[0m [0mfigsize[0m[0;34m=[0m[0;34m([0m[0;36m10[0m[0;34m,[0m [0;36m5[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1688502001.py[0m in [0;36mtune_threshold_on_train[0;34m(candidates)[0m
[1;32m     85[0m         [0mfdir[0m [0;34m=[0m [0mos[0m[0;34m.[0m[0mpath[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mDATA_ROOT[0m[0;34m,[0m [0;34m"train"[0m[0;34m,[0m [0mtid[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     86[0m         [0mprint[0m[0;34m([0m[0;34mf"Building prob map for threshold tuning on train/{tid} ..."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 87[0;31m         [0mprob[0m[0;34m,[0m [0mvalid[0m [0;34m=[0m [0mpredict_fragment_prob_map[0m[0;34m([0m[0mfdir[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     88[0m         [0my_true[0m [0;34m=[0m [0mload_label[0m[0;34m([0m[0mfdir[0m[0;34m,[0m [0mdevice[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m.[0m[0mnumpy[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0muint8[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     89[0m         [0;31m# Only score where valid pixels exist (mask & not-border), matching how we produce submission.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1688502001.py[0m in [0;36mpredict_fragment_prob_map[0;34m(fragment_dir)[0m
[1;32m     58[0m     [0mmodel[0m[0;34m.[0m[0meval[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     59[0m     [0;32mwith[0m [0mtorch[0m[0;34m.[0m[0mno_grad[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 60[0;31m         [0;32mfor[0m [0mi[0m[0;34m,[0m [0;34m([0m[0msubvolumes[0m[0;34m,[0m [0m_[0m[0;34m)[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mtqdm[0m[0;34m([0m[0mdl[0m[0;34m,[0m [0mleave[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     61[0m             preds = (
[1;32m     62[0m                 [0mmodel[0m[0;34m([0m[0msubvolumes[0m[0;34m.[0m[0mto[0m[0;34m([0m[0mDEVICE[0m[0;34m,[0m [0mnon_blocking[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m    764[0m         [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_dataset_fetcher[0m[0;34m.[0m[0mfetch[0m[0;34m([0m[0mindex[0m[0;34m)[0m  [0;31m# may raise StopIteration[0m[0;34m[0m[0;34m[0m[0m
[1;32m    765[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0m_pin_memory[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 766[0;31m             [0mdata[0m [0;34m=[0m [0m_utils[0m[0;34m.[0m[0mpin_memory[0m[0;34m.[0m[0mpin_memory[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_pin_memory_device[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    767[0m         [0;32mreturn[0m [0mdata[0m[0;34m[0m[0;34m[0m[0m
[1;32m    768[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py[0m in [0;36mpin_memory[0;34m(data, device)[0m
[1;32m     96[0m                 [0mclone[0m [0;34m=[0m [0mcopy[0m[0;34m.[0m[0mcopy[0m[0;34m([0m[0mdata[0m[0;34m)[0m  [0;31m# type: ignore[arg-type][0m[0;34m[0m[0;34m[0m[0m
[1;32m     97[0m                 [0;32mfor[0m [0mi[0m[0;34m,[0m [0mitem[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 98[0;31m                     [0mclone[0m[0;34m[[0m[0mi[0m[0;34m][0m [0;34m=[0m [0mpin_memory[0m[0;34m([0m[0mitem[0m[0;34m,[0m [0mdevice[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     99[0m                 [0;32mreturn[0m [0mclone[0m[0;34m[0m[0;34m[0m[0m
[1;32m    100[0m             [0;32mreturn[0m [0mtype[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m([0m[0;34m[[0m[0mpin_memory[0m[0;34m([0m[0msample[0m[0;34m,[0m [0mdevice[0m[0;34m)[0m [0;32mfor[0m [0msample[0m [0;32min[0m [0mdata[0m[0;34m][0m[0;34m)[0m  [0;31m# type: ignore[call-arg][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/pin_memory.py[0m in [0;36mpin_memory[0;34m(data, device)[0m
[1;32m     62[0m [0;32mdef[0m [0mpin_memory[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mdevice[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     63[0m     [0;32mif[0m [0misinstance[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mtorch[0m[0;34m.[0m[0mTensor[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 64[0;31m         [0;32mreturn[0m [0mdata[0m[0;34m.[0m[0mpin_memory[0m[0;34m([0m[0mdevice[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     65[0m     [0;32melif[0m [0misinstance[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0;34m([0m[0mstr[0m[0;34m,[0m [0mbytes[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     66[0m         [0;32mreturn[0m [0mdata[0m[0;34m[0m[0;34m[0m[0m

[0;31mRuntimeError[0m: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned

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
