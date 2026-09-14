# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

3.7

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

# 5. Target score

0.3897879705755084

# 6. Current score

0.4681

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.4681) has done: 'I remove the hard dependency on `fastai` (it isn’t installed, causing the first import crash) and instead implement a small, deterministic PyTorch U-Net training/inference pipeline that still follows the same core idea: train a segmentation model on the 101×101 masks and predict test masks, then RLE-encode them. I also fix all missing imports (`Path`, `ThreadPoolExecutor`, `cv2`, `plt`, etc.) and correct the test-set sizing bug (your code assumes 18,000 test images; the competition has 1,800). Finally, I ensure the submission strictly matches `sample_submission.csv` ids (1000 rows in your environment listing) and writes a valid `submission.csv` with the required `id,rle_mask` columns.'

# 9. Code solution

## === cell 0
import os, sys, json, random, math, time
import numpy as np
import pandas as pd



## === cell 1
print("Working dir:", os.getcwd())
print("Input root listing:", os.listdir("/kaggle/input")[:10])



## === cell 2
from pathlib import Path

PATH = Path("/kaggle/input/tgs-salt-identification-challenge")
print("Competition dir listing:", os.listdir(str(PATH))[:20])



## === cell 3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())



## === cell 4
TRAIN_IMG_DIR = PATH / "train/images"
TRAIN_MASK_DIR = PATH / "train/masks"
TEST_IMG_DIR = PATH / "test/images"

train_csv = pd.read_csv(PATH / "train.csv")
depths = pd.read_csv(PATH / "depths.csv")
sample_sub = pd.read_csv(PATH / "sample_submission.csv")

train_ids = train_csv["id"].tolist()
test_ids = sample_sub["id"].tolist()  # enforce exact required ids/rowcount

print("train_csv rows:", len(train_csv), "unique train ids:", len(set(train_ids)))
print(
    "sample_submission rows:", len(sample_sub), "unique test ids:", len(set(test_ids))
)

print("Example train image exists:", (TRAIN_IMG_DIR / f"{train_ids[0]}.png").exists())
print("Example train mask exists:", (TRAIN_MASK_DIR / f"{train_ids[0]}.png").exists())
print("Example test image exists:", (TEST_IMG_DIR / f"{test_ids[0]}.png").exists())



## === cell 5
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)




## === cell 6
def rle_encode(mask: np.ndarray) -> str:
    """
    mask: (H,W) binary {0,1} or bool
    Returns run-length encoding string, column-major (order='F'), 1-indexed.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def load_gray_png(path: Path) -> np.ndarray:
    img = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return img




## === cell 7
class TGSSaltDataset(Dataset):
    def __init__(self, ids, img_dir, mask_dir=None, augment=False):
        self.ids = list(ids)
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img = load_gray_png(self.img_dir / f"{id_}.png")  # (101,101)
        img = img.astype(np.float32) / 255.0
        img = img[None, :, :]  # (1,H,W)

        if self.mask_dir is not None:
            mask = load_gray_png(self.mask_dir / f"{id_}.png")
            mask = (mask > 127).astype(np.float32)  # (H,W)
            if self.augment:
                if random.random() < 0.5:
                    img = img[:, :, ::-1].copy()
                    mask = mask[:, ::-1].copy()
            return torch.from_numpy(img), torch.from_numpy(mask)
        else:
            return torch.from_numpy(img), id_




## === cell 8
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.outc = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)  # 101
        e2 = self.enc2(self.pool1(e1))  # 50
        e3 = self.enc3(self.pool2(e2))  # 25
        b = self.bottleneck(self.pool3(e3))  # 12

        d3 = self.up3(b)  # 24
        d3 = self._pad_or_crop(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)  # 48
        d2 = self._pad_or_crop(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)  # 96
        d1 = self._pad_or_crop(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.outc(d1)  # logits (B,1,H,W)
        return out

    @staticmethod
    def _pad_or_crop(x, ref):
        _, _, h, w = x.shape
        _, _, hr, wr = ref.shape
        dh, dw = hr - h, wr - w
        if dh == 0 and dw == 0:
            return x
        pad_top = max(dh // 2, 0)
        pad_bottom = max(dh - pad_top, 0)
        pad_left = max(dw // 2, 0)
        pad_right = max(dw - pad_left, 0)
        if pad_top or pad_bottom or pad_left or pad_right:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        _, _, h2, w2 = x.shape
        start_h = max((h2 - hr) // 2, 0)
        start_w = max((w2 - wr) // 2, 0)
        return x[:, :, start_h : start_h + hr, start_w : start_w + wr]




## === cell 9
def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.unsqueeze(1)  # (B,1,H,W)
    num = 2.0 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    dice = 1.0 - (num / den)
    return dice.mean()


bce = nn.BCEWithLogitsLoss()


def combined_loss(logits, targets):
    targets = targets.unsqueeze(1)
    return bce(logits, targets) + dice_loss_with_logits(logits, targets.squeeze(1))




## === cell 10
idxs = np.arange(len(train_ids))
np.random.shuffle(idxs)
split = int(0.9 * len(train_ids))
trn_ids = [train_ids[i] for i in idxs[:split]]
val_ids = [train_ids[i] for i in idxs[split:]]

train_ds = TGSSaltDataset(trn_ids, TRAIN_IMG_DIR, TRAIN_MASK_DIR, augment=True)
val_ds = TGSSaltDataset(val_ids, TRAIN_IMG_DIR, TRAIN_MASK_DIR, augment=False)

bs = 32 if torch.cuda.is_available() else 8
train_loader = DataLoader(
    train_ds,
    batch_size=bs,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

print("train/val:", len(train_ds), len(val_ds), "bs:", bs)



## === cell 11
model = UNetSmall(in_ch=1, base=32).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


def eval_iou_at_threshold(model, loader, thr=0.5):
    model.eval()
    ious = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs > thr).float()
            inter = (preds * y).sum(dim=(1, 2))
            union = (preds + y).sum(dim=(1, 2)) - inter + 1e-6
            iou = (inter / union).detach().cpu().numpy()
            ious.append(iou)
    return float(np.concatenate(ious).mean())


epochs = (
    6 if torch.cuda.is_available() else 2
)  # keep under time; still yields non-empty submission.
for ep in range(1, epochs + 1):
    model.train()
    tr_losses = []
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = combined_loss(logits, y)
        loss.backward()
        opt.step()
        tr_losses.append(loss.item())
    val_iou = eval_iou_at_threshold(model, val_loader, thr=0.5)
    print(
        f"epoch {ep}/{epochs} loss={np.mean(tr_losses):.4f} val_iou@0.5={val_iou:.4f}"
    )



## === cell 12
thr_candidates = np.linspace(0.3, 0.7, 9)
best_thr, best_iou = 0.5, -1
for thr in thr_candidates:
    miou = eval_iou_at_threshold(model, val_loader, thr=float(thr))
    if miou > best_iou:
        best_iou, best_thr = miou, float(thr)
print("Chosen threshold:", best_thr, "val mean IoU:", best_iou)



## === cell 13
test_ds = TGSSaltDataset(test_ids, TEST_IMG_DIR, mask_dir=None, augment=False)
test_loader = DataLoader(
    test_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

model.eval()
pred_masks = {}
with torch.no_grad():
    for x, batch_ids in test_loader:
        x = x.to(device)
        probs = torch.sigmoid(model(x)).squeeze(1).detach().cpu().numpy()  # (B,H,W)
        for p, id_ in zip(probs, batch_ids):
            pred_masks[id_] = (p > best_thr).astype(np.uint8)

print("Predicted masks:", len(pred_masks), "expected:", len(test_ids))



## === cell 14
rles = []
for id_ in test_ids:
    m = pred_masks[id_]
    rles.append(rle_encode(m))

sub = pd.DataFrame({"id": test_ids, "rle_mask": rles})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv; rows:", len(sub), "cols:", list(sub.columns))
print(
    "File exists:",
    Path("submission.csv").exists(),
    "size:",
    Path("submission.csv").stat().st_size,
)



## === cell 15
try:
    n_show = 6
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for ax, id_ in zip(axes.flat, test_ids[:n_show]):
        img = load_gray_png(TEST_IMG_DIR / f"{id_}.png")
        ax.imshow(img, cmap="gray")
        ax.imshow(pred_masks[id_] * 255, alpha=0.3, cmap="Reds")
        ax.set_title(id_)
        ax.axis("off")
    plt.tight_layout()
except Exception as e:
    print("Plotting skipped:", repr(e))
