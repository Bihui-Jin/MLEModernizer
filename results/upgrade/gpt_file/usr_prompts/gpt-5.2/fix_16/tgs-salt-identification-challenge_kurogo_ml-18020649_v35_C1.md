# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Segment regions of salt in seismic images.

## Metric
Mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:

$$\text{IoU}(A, B)=\frac{A \cap B}{A \cup B}$$

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: `(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)`. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value 𝑡t, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

$$\frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold:

$$\frac{1}{\mid \text { thresholds } \mid} \sum_t \frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

## Submission Format
Use run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed\
and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.

The file should contain a header and have the following format. Each row in your submission represents a single predicted salt segmentation for the given image.

```
id,rle_mask
3e06571ef3,1 1
a51b08d882,1 1
c32590b06f,1 1
etc.
```

## Dataset
The data is a set of images chosen at various locations chosen at random in the subsurface. The images are 101 x 101 pixels and each pixel is classified as either salt or sediment. In addition to the seismic images, the depth of the imaged location is provided for each image.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
        input/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
        working/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
```

-> data/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/tgs-salt-identification-challenge/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> data/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> input/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "1234")
np.random.seed(1234)
random.seed(1234)



## === cell 1
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
torch.set_num_interop_threads(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 2
CANDIDATE_ROOTS = [
    "/kaggle/input/tgs-salt-identification-challenge/",
    "/kaggle/input/tgs-salt-identification-challenge/tgs-salt-identification-challenge/",
    "/kaggle/data/tgs-salt-identification-challenge/",
    "/kaggle/data/",
    "/kaggle/input/",
]


def _find_dataset_root(candidates):
    for root in candidates:
        train_img = os.path.join(root, "train", "images")
        train_msk = os.path.join(root, "train", "masks")
        test_img = os.path.join(root, "test", "images")
        if (
            os.path.isdir(train_img)
            and os.path.isdir(train_msk)
            and os.path.isdir(test_img)
        ):
            return root
    for base in ["/kaggle/input", "/kaggle/data"]:
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            root = os.path.join(base, name)
            train_img = os.path.join(root, "train", "images")
            train_msk = os.path.join(root, "train", "masks")
            test_img = os.path.join(root, "test", "images")
            if (
                os.path.isdir(train_img)
                and os.path.isdir(train_msk)
                and os.path.isdir(test_img)
            ):
                return root
    raise FileNotFoundError(
        "Could not locate dataset root containing train/images, train/masks, test/images"
    )


ROOT_DATA_DIR = _find_dataset_root(CANDIDATE_ROOTS)
TRAIN_MASK_DIR = os.path.join(ROOT_DATA_DIR, "train", "masks") + os.sep
TRAIN_IMAGE_DIR = os.path.join(ROOT_DATA_DIR, "train", "images") + os.sep
TEST_IMAGE_DIR = os.path.join(ROOT_DATA_DIR, "test", "images") + os.sep

ROOT_DATA_DIR, TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, TEST_IMAGE_DIR




## === cell 3
def _find_file(possible_paths):
    for p in possible_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"File not found in any of: {possible_paths}")


train_csv = _find_file(
    [
        os.path.join(ROOT_DATA_DIR, "train.csv"),
        "/kaggle/input/tgs-salt-identification-challenge/train.csv",
        "/kaggle/data/train.csv",
        "/kaggle/input/train.csv",
    ]
)
sample_csv = _find_file(
    [
        os.path.join(ROOT_DATA_DIR, "sample_submission.csv"),
        "/kaggle/input/tgs-salt-identification-challenge/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
    ]
)
depths_csv = _find_file(
    [
        os.path.join(ROOT_DATA_DIR, "depths.csv"),
        "/kaggle/input/tgs-salt-identification-challenge/depths.csv",
        "/kaggle/data/depths.csv",
        "/kaggle/input/depths.csv",
    ]
)

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(sample_csv)
depths_df = pd.read_csv(depths_csv)

train_df.shape, test_df.shape, depths_df.shape




## === cell 4
def get_coverage(rle_mask):
    if pd.isna(rle_mask):
        return 0
    arr = rle_mask.split()
    coverage = sum(int(x) for x in arr[1::2]) / (101**2)
    return np.round(coverage, 1)


train_df["coverage"] = train_df["rle_mask"].map(get_coverage)
train_df.head()



## === cell 5
from concurrent.futures import ThreadPoolExecutor


def load_grayscale_101(path):
    with Image.open(path) as img:
        img = img.convert("L")
        if img.size != (101, 101):
            img = img.resize((101, 101), resample=Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def _load_pair(image_name):
    img = load_grayscale_101(os.path.join(TRAIN_IMAGE_DIR, image_name + ".png"))
    msk = load_grayscale_101(os.path.join(TRAIN_MASK_DIR, image_name + ".png"))
    return img, msk


train_ids = train_df["id"].values
n_train = len(train_ids)

train_imgs_u8 = np.empty((n_train, 101, 101), dtype=np.uint8)
train_masks_u8 = np.empty((n_train, 101, 101), dtype=np.uint8)

io_workers = max(1, min(8, (os.cpu_count() or 2)))
with ThreadPoolExecutor(max_workers=io_workers) as ex:
    for i, (img, msk) in enumerate(
        tqdm(
            ex.map(_load_pair, train_ids),
            total=n_train,
            desc="Loading train images/masks",
        )
    ):
        train_imgs_u8[i] = img
        train_masks_u8[i] = msk

train_imgs = (train_imgs_u8.astype(np.float32) / 255.0)[..., None]
train_masks = ((train_masks_u8 > 127).astype(np.uint8))[..., None]

train_imgs.shape, train_masks.shape, train_imgs.dtype, train_masks.dtype



## === cell 6
x_train, x_valid, y_train, y_valid = train_test_split(
    train_imgs,
    train_masks,
    test_size=0.2,
    stratify=train_df.coverage,
    random_state=1234,
)

x_train = np.concatenate([x_train, x_train[:, :, ::-1, :]], axis=0)
y_train = np.concatenate([y_train, y_train[:, :, ::-1, :]], axis=0)

x_train.shape, x_valid.shape, y_train.shape, y_valid.shape



## === cell 7
_THRESHOLDS_F32 = torch.arange(
    0.5, 1.0, 0.05, dtype=torch.float32, device=device
)  # (T,)
_THRESHOLDS_F32_ROW = _THRESHOLDS_F32.unsqueeze(0)  # (1,T)


def iou_vector(trues, preds):
    SMOOTH = 1e-10
    trues = trues.astype(bool)
    preds = preds.astype(bool)

    axes = tuple(range(1, trues.ndim))
    intersection = np.logical_and(trues, preds).sum(axis=axes).astype(np.float64)
    union = np.logical_or(trues, preds).sum(axis=axes).astype(np.float64)

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # (B,)

    hits = iou[:, None] > np.arange(0.5, 1.0, 0.05, dtype=np.float32)[None, :]
    metric = hits.mean(axis=1)
    return float(metric.mean())


def _batch_inter_union_from_logits(logits, targets):
    preds = logits > 0  # equivalent to sigmoid(logits) > 0.5
    trues = targets > 0.5
    inter = (preds & trues).sum(dim=(1, 2, 3), dtype=torch.int64)
    union = (preds | trues).sum(dim=(1, 2, 3), dtype=torch.int64)
    return inter, union


def _hits_sum_from_inter_union_torch(inter_i64, union_i64):
    union_is_zero = union_i64 == 0  # (B,)
    rhs = union_i64.to(torch.float32).unsqueeze(1) * _THRESHOLDS_F32_ROW  # (B,T)
    lhs = inter_i64.to(torch.float32).unsqueeze(1)  # (B,T)
    hits = lhs > rhs
    if union_is_zero.any():
        hits = torch.where(
            union_is_zero.unsqueeze(1), torch.ones_like(hits, dtype=torch.bool), hits
        )
    return hits.sum(dtype=torch.int64), int(inter_i64.numel())


def _epoch_iou_from_hits(hits_sum, n_images):
    return float(hits_sum / (max(1, n_images) * _THRESHOLDS_F32.numel()))




## === cell 8
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, padding=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = F.relu(x, inplace=True)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = ConvBlock(ch, ch, k=3, padding=1, activation=True)
        self.c2 = ConvBlock(ch, ch, k=3, padding=1, activation=False)

    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        out = out + x
        out = F.relu(out, inplace=True)
        return out


class UNetRes(nn.Module):
    def __init__(self, in_ch=1, start_feature=32, dropout=0.5):
        super().__init__()
        self.start_feature = start_feature
        self.dropout = dropout

        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for i in range(4):
            out_ch = start_feature * (2**i)
            block = nn.Sequential(
                ConvBlock(ch, out_ch, k=3, padding=1, activation=True),
                ResidualBlock(out_ch),
                ResidualBlock(out_ch),
            )
            self.enc_blocks.append(block)
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            ch = out_ch

        bott_ch = start_feature * (2**4)
        self.bottleneck = nn.Sequential(
            ConvBlock(ch, bott_ch, k=3, padding=1, activation=True),
            ResidualBlock(bott_ch),
            ResidualBlock(bott_ch),
        )

        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        ch = bott_ch
        for i in reversed(range(4)):
            out_ch = start_feature * (2**i)
            self.upconvs.append(
                nn.ConvTranspose2d(
                    ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
                )
            )
            self.dec_blocks.append(
                nn.Sequential(
                    nn.Dropout2d(p=dropout),
                    ConvBlock(out_ch + out_ch, out_ch, k=3, padding=1, activation=True),
                    ResidualBlock(out_ch),
                    ResidualBlock(out_ch),
                )
            )
            ch = out_ch

        self.final = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upconvs, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                diff_y = skip.shape[-2] - x.shape[-2]
                diff_x = skip.shape[-1] - x.shape[-1]
                pad_left = diff_x // 2
                pad_right = diff_x - pad_left
                pad_top = diff_y // 2
                pad_bottom = diff_y - pad_top
                x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
                if x.shape[-2] > skip.shape[-2] or x.shape[-1] > skip.shape[-1]:
                    x = x[:, :, : skip.shape[-2], : skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        logits = self.final(x)
        return logits


model = UNetRes(in_ch=1, start_feature=32, dropout=0.5).to(device)

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass

sum(p.numel() for p in model.parameters()) / 1e6




## === cell 9
def _to_chw_tensor(x_nhwc):
    x = np.ascontiguousarray(x_nhwc.transpose(0, 3, 1, 2))
    return torch.from_numpy(x).float()


x_train_t = _to_chw_tensor(x_train)
y_train_t = _to_chw_tensor(y_train)
x_valid_t = _to_chw_tensor(x_valid)
y_valid_t = _to_chw_tensor(y_valid)

from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train_t, y_train_t)
valid_ds = TensorDataset(x_valid_t, y_valid_t)

batch_size = 16

num_workers = 2 if (os.cpu_count() or 0) >= 4 else 0
loader_kwargs = dict(
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
)
if num_workers > 0:
    loader_kwargs["prefetch_factor"] = 4

shuffle_gen = torch.Generator().manual_seed(1234)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    generator=shuffle_gen,
    drop_last=False,
    **loader_kwargs,
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    **loader_kwargs,
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



## === cell 10
epochs = 100
patience_es = 15
patience_rlr = 5
factor = 0.5
min_lr = 1e-4

history = {"loss": [], "val_loss": [], "my_iou_metric": [], "val_my_iou_metric": []}
best_val = -1e9
best_state = None
epochs_no_improve = 0
epochs_no_improve_lr = 0

base_seed = 1234
for epoch in range(1, epochs + 1):
    model.train()

    tr_loss_sum = 0.0
    tr_n_batches = 0

    tr_hits_sum_t = torch.zeros((), dtype=torch.int64, device=device)
    tr_seen = 0

    train_loader.generator.manual_seed(base_seed + epoch)

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        tr_loss_sum += float(loss.detach().item())
        tr_n_batches += 1

        with torch.no_grad():
            inter, union = _batch_inter_union_from_logits(logits, yb)
            hs_t, nimg = _hits_sum_from_inter_union_torch(inter, union)
            tr_hits_sum_t += hs_t
            tr_seen += nimg

    model.eval()
    va_loss_sum = 0.0
    va_n_batches = 0

    va_hits_sum_t = torch.zeros((), dtype=torch.int64, device=device)
    va_seen = 0

    with torch.no_grad():
        for xb, yb in valid_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
                yb = yb.contiguous(memory_format=torch.channels_last)

            logits = model(xb)
            loss = criterion(logits, yb)

            va_loss_sum += float(loss.item())
            va_n_batches += 1

            inter, union = _batch_inter_union_from_logits(logits, yb)
            hs_t, nimg = _hits_sum_from_inter_union_torch(inter, union)
            va_hits_sum_t += hs_t
            va_seen += nimg

    tr_loss = tr_loss_sum / max(1, tr_n_batches)
    va_loss = va_loss_sum / max(1, va_n_batches)

    tr_hits_sum = int(tr_hits_sum_t.item())
    va_hits_sum = int(va_hits_sum_t.item())

    tr_iou = _epoch_iou_from_hits(tr_hits_sum, tr_seen)
    va_iou = _epoch_iou_from_hits(va_hits_sum, va_seen)

    history["loss"].append(float(tr_loss))
    history["val_loss"].append(float(va_loss))
    history["my_iou_metric"].append(tr_iou)
    history["val_my_iou_metric"].append(va_iou)

    print(
        f"Epoch {epoch:03d}/{epochs} - "
        f"loss: {tr_loss:.4f} - iou: {tr_iou:.4f} - "
        f"val_loss: {va_loss:.4f} - val_iou: {va_iou:.4f}"
    )

    if va_iou > best_val + 1e-8:
        best_val = va_iou
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        epochs_no_improve = 0
        epochs_no_improve_lr = 0
    else:
        epochs_no_improve += 1
        epochs_no_improve_lr += 1

    if epochs_no_improve_lr >= patience_rlr:
        epochs_no_improve_lr = 0
        for pg in optimizer.param_groups:
            old_lr = pg["lr"]
            new_lr = max(min_lr, old_lr * factor)
            pg["lr"] = new_lr
        print(f"ReduceLROnPlateau: lr set to {optimizer.param_groups[0]['lr']:.6f}")

    if epochs_no_improve >= patience_es:
        print("EarlyStopping: stopping training.")
        break

if best_state is not None:
    model.load_state_dict(best_state)



## === cell 11
if len(history["loss"]) == 0:
    print("No training history to plot (history is empty).")
else:
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15, 5))
    ax_loss.set_ylim(0, max(1.0, max(history["loss"] + history["val_loss"]) * 1.05))
    ax_loss.plot(history["loss"], label="Train loss")
    ax_loss.plot(history["val_loss"], label="Validation loss")
    ax_score.plot(history["my_iou_metric"], label="Train score")
    ax_score.plot(history["val_my_iou_metric"], label="Validation score")
    ax_loss.legend()
    ax_score.legend()
    plt.show()



## === cell 12
test_ids = test_df["id"].values
n_test = len(test_ids)
test_imgs_u8 = np.empty((n_test, 101, 101), dtype=np.uint8)


def _load_test(image_name):
    return load_grayscale_101(os.path.join(TEST_IMAGE_DIR, image_name + ".png"))


with ThreadPoolExecutor(max_workers=io_workers) as ex:
    for i, img in enumerate(
        tqdm(ex.map(_load_test, test_ids), total=n_test, desc="Loading test images")
    ):
        test_imgs_u8[i] = img

test_imgs = (test_imgs_u8.astype(np.float32) / 255.0)[..., None]
test_imgs_t = _to_chw_tensor(test_imgs)

test_ds = torch.utils.data.TensorDataset(test_imgs_t)
test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

model.eval()
pred_test = np.empty((n_test, 101, 101, 1), dtype=np.float32)
offset = 0
with torch.no_grad():
    for (xb,) in tqdm(test_loader, desc="Predicting"):
        bsz = xb.shape[0]
        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        logits = model(xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy()  # (B,1,H,W)
        pred_test[offset : offset + bsz] = probs.transpose(0, 2, 3, 1)
        offset += bsz

test_imgs_flip = test_imgs[:, :, ::-1, :]
test_imgs_flip_t = _to_chw_tensor(test_imgs_flip)

test_flip_ds = torch.utils.data.TensorDataset(test_imgs_flip_t)
test_flip_loader = DataLoader(
    test_flip_ds,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

pred_test_flip = np.empty((n_test, 101, 101, 1), dtype=np.float32)
offset = 0
with torch.no_grad():
    for (xb,) in tqdm(test_flip_loader, desc="Predicting flip TTA"):
        bsz = xb.shape[0]
        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        logits = model(xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        pred_test_flip[offset : offset + bsz] = probs.transpose(0, 2, 3, 1)
        offset += bsz

pred_test = (pred_test + pred_test_flip[:, :, ::-1, :]) / 2.0
pred_test.shape




## === cell 13
def mask2rle(mask):
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    m = (mask > 0.5).astype(np.uint8)

    pixels = m.reshape(-1, order="F")
    padded = np.concatenate([[0], pixels, [0]]).astype(np.uint8)
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    changes[1::2] -= changes[::2]
    if changes.size == 0:
        return ""
    return " ".join(map(str, changes))


rle_masks = [mask2rle(x) for x in pred_test]

test_df = test_df.copy()
test_df["rle_mask"] = rle_masks
test_df.to_csv("submission.csv", index=False)

test_df.head()



## === cell 14
assert os.path.exists("submission.csv")
sub = pd.read_csv("submission.csv")
sub.columns.tolist(), sub.shape, sub.head()
