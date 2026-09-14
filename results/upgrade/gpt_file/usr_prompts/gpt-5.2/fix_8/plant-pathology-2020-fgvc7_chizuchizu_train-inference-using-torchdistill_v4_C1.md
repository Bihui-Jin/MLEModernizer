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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.93438

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66459) has done: 'Your code currently won’t yield a Kaggle score because it depends on `timm`, which is not guaranteed to be installed in the provided environment (“No external packages required”). To make it run end-to-end and produce a valid `submission.csv`, I replace the `timm` model with a small built-in PyTorch CNN while keeping the same training loop, loss (BCEWithLogitsLoss), sigmoid post-processing, and submission schema unchanged. I also adjust the image normalization to standard ImageNet stats (more appropriate for pretrained-like pipelines, but here primarily for stability) while preserving the same transform structure. These changes are strictly to unblock execution and produce a valid submission, which is required before any score can move toward the 0.93438 target.'
- What this solution (achieved 0.66976) has done: 'To ensure you get a valid Kaggle score (and move toward the 0.93438 target), I’m keeping your exact core pipeline (SmallCNN + BCEWithLogitsLoss + sigmoid + same loop) but fixing the main reason scores often end up “Not yielded”: submission column mismatch with the competition’s required label names. Your current code uses `multiple_diseases` while the task description/submission example expects `combinations`, so I add a minimal label-column resolver that maps between the two safely based on `sample_submission.csv` / `train.csv`. I also make the train/val split use `args.seed` (not a hardcoded 42) to keep runs deterministic with your passed `--seed`, without changing the training approach. The script still write `submission.csv` and now guarantees the header matches what Kaggle expects in either naming variant.'

# 9. Code solution

## === cell 0
import pathlib

config_text = """\
model:
  timm_model_name: "tf_efficientnet_b0_ns"
  num_classes: 4
  pretrained: True
train:
  num_epochs: 3
  batch_size: 64
  lr: 0.005
data:
  root_dir: "../input/plant-pathology-2020-fgvc7"
  img_size: [224, 224]
output:
  save_path: "submission.csv"
"""
pathlib.Path("config.yaml").write_text(config_text)



## === cell 1
import pathlib

distill_py = r"""
import os
import time
import random
import argparse

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.backends import cudnn


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Change rationale (score stability): make training deterministic for the given seed to reduce
    # variance and help the score move upward consistently toward the target.
    cudnn.deterministic = True
    cudnn.benchmark = False


def resolve_root_dir():
    candidates = [
        "/kaggle/input/plant-pathology-2020-fgvc7",
        "/kaggle/data/plant-pathology-2020-fgvc7",
        "../input/plant-pathology-2020-fgvc7",
        "../data/plant-pathology-2020-fgvc7",
        "/kaggle/input",
        "/kaggle/data",
        "../input",
        "../data",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            if os.path.basename(c) in ("input", "data"):
                comp = os.path.join(c, "plant-pathology-2020-fgvc7")
                if os.path.exists(os.path.join(comp, "train.csv")):
                    return comp
            if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(os.path.join(c, "images")):
                return c
    return "../input/plant-pathology-2020-fgvc7"


def get_img_path(root_dir, image_id):
    p1 = os.path.join(root_dir, "images", f"{image_id}.jpg")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(root_dir, "images", "images", f"{image_id}.jpg")
    if os.path.exists(p2):
        return p2
    return p1


def _pil_to_chw_float(pil_img: Image.Image):
    arr = np.asarray(pil_img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def _normalize_chw(arr_chw, mean, std):
    mean = np.asarray(mean, dtype=np.float32)[:, None, None]
    std = np.asarray(std, dtype=np.float32)[:, None, None]
    arr = (arr_chw - mean) / std
    # Change rationale (improve AUC): remove clamping which can squash informative contrast/color
    # signals for disease patterns; keep standard normalization semantics.
    return arr


def build_transform_train(img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    def _transform(pil_img: Image.Image):
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        w, h = pil_img.size
        scale_min, scale_max = 0.75, 1.0
        target_area = random.uniform(scale_min, scale_max) * w * h
        aspect = random.uniform(0.9, 1.1)
        crop_w = int(round((target_area * aspect) ** 0.5))
        crop_h = int(round((target_area / aspect) ** 0.5))
        crop_w = max(1, min(crop_w, w))
        crop_h = max(1, min(crop_h, h))

        if crop_w < w:
            x0 = random.randint(0, w - crop_w)
        else:
            x0 = 0
        if crop_h < h:
            y0 = random.randint(0, h - crop_h)
        else:
            y0 = 0

        pil_img = pil_img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        pil_img = pil_img.resize((img_size, img_size), resample=Image.BICUBIC)

        if random.random() < 0.5:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

        arr = _pil_to_chw_float(pil_img)
        arr = _normalize_chw(arr, mean, std)
        return torch.from_numpy(arr)
    return _transform


def build_transform_eval(img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # Change rationale (improve score toward target): switch eval preprocessing to the common
    # resize-to-256 then center-crop-to-224 pattern (here derived from img_size). This often boosts
    # AUC by reducing distortion vs direct resize.
    def _transform(pil_img: Image.Image):
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        resize_size = int(round(img_size * 256 / 224))  # e.g., 224 -> 256
        w, h = pil_img.size
        if w < h:
            new_w = resize_size
            new_h = int(round(h * (resize_size / w)))
        else:
            new_h = resize_size
            new_w = int(round(w * (resize_size / h)))

        pil_img = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)

        left = max(0, (new_w - img_size) // 2)
        top = max(0, (new_h - img_size) // 2)
        pil_img = pil_img.crop((left, top, left + img_size, top + img_size))

        arr = _pil_to_chw_float(pil_img)
        arr = _normalize_chw(arr, mean, std)
        return torch.from_numpy(arr)
    return _transform


def resolve_label_cols(train_df: pd.DataFrame, sample_sub_df: pd.DataFrame | None = None):
    candidates = []
    if sample_sub_df is not None:
        candidates.append([c for c in sample_sub_df.columns if c != "image_id"])
    candidates.append([c for c in train_df.columns if c != "image_id"])

    for cols in candidates:
        if set(cols) == set(["healthy", "multiple_diseases", "rust", "scab"]):
            return ["healthy", "multiple_diseases", "rust", "scab"]
        if set(cols) == set(["healthy", "combinations", "rust", "scab"]):
            return ["healthy", "combinations", "rust", "scab"]

    cols = [c for c in train_df.columns if c != "image_id"]
    if len(cols) != 4:
        raise ValueError(f"Could not resolve 4 label columns. Found: {cols}")
    return cols


class Plant2020Dataset(Dataset):
    def __init__(self, csv_path, root_dir, label_cols, inf=False, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.root_dir = root_dir
        self.inf = inf
        self.transform = transform
        self.label_cols = label_cols
        if not inf:
            self.labels = self.df[self.label_cols].values.astype(np.float32)
        else:
            self.labels = np.zeros((len(self.df), 4), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        path = get_img_path(self.root_dir, image_id)
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, y


class SmallCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.head(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        running += loss.item() * bs
        n += bs
    return running / max(n, 1)


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    outs = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits)
        outs.append(probs.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def make_submission(root_dir, save_path, probs, label_cols):
    test_path = os.path.join(root_dir, "test.csv")
    sub = pd.read_csv(test_path)

    probs = np.clip(probs, 0.0, 1.0)
    if probs.shape != (len(sub), 4):
        raise ValueError(f"Expected probs shape {(len(sub), 4)}, got {probs.shape}")

    sub[label_cols] = probs
    sub.to_csv(save_path, index=False)
    return sub


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_path", default="submission.csv")
    ap.add_argument("--root_dir", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    root_dir = args.root_dir or resolve_root_dir()
    train_csv = os.path.join(root_dir, "train.csv")
    test_csv = os.path.join(root_dir, "test.csv")
    sample_sub_csv = os.path.join(root_dir, "sample_submission.csv")

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        raise FileNotFoundError(f"Could not find train/test CSVs under root_dir={root_dir}")

    tfm_train = build_transform_train(img_size=args.img_size)
    tfm_eval = build_transform_eval(img_size=args.img_size)

    train_df = pd.read_csv(train_csv)
    sample_sub_df = pd.read_csv(sample_sub_csv) if os.path.exists(sample_sub_csv) else None
    label_cols = resolve_label_cols(train_df, sample_sub_df)

    idx = np.arange(len(train_df))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    tmp_train = "tmp_train_split.csv"
    tmp_val = "tmp_val_split.csv"
    train_df.iloc[tr_idx].to_csv(tmp_train, index=False)
    train_df.iloc[va_idx].to_csv(tmp_val, index=False)

    ds_train = Plant2020Dataset(tmp_train, root_dir, label_cols=label_cols, inf=False, transform=tfm_train)
    ds_val = Plant2020Dataset(tmp_val, root_dir, label_cols=label_cols, inf=False, transform=tfm_eval)
    ds_test = Plant2020Dataset(test_csv, root_dir, label_cols=label_cols, inf=True, transform=tfm_eval)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=max(128, args.batch_size), shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=max(128, args.batch_size), shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = SmallCNN(num_classes=4).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs * 2, 1), eta_min=0)

    best_state = None
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(model, dl_train, optimizer, criterion, device)

        model.eval()
        val_running = 0.0
        n = 0
        with torch.no_grad():
            for x, y in dl_val:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                bs = x.size(0)
                val_running += loss.item() * bs
                n += bs
        va_loss = val_running / max(n, 1)
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()
        dt = time.time() - t0
        print(f"epoch={epoch+1}/{args.epochs} train_loss={tr_loss:.5f} val_loss={va_loss:.5f} time={dt:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    probs = predict_proba(model, dl_test, device)
    sub = make_submission(root_dir, args.save_path, probs, label_cols=label_cols)
    print("Resolved root_dir:", root_dir)
    print("Using label_cols:", label_cols)
    print("Saved:", args.save_path, "shape:", sub.shape, "cols:", list(sub.columns))


if __name__ == "__main__":
    main()
"""
pathlib.Path("distill.py").write_text(distill_py)



## === cell 2
import subprocess, sys, os, pandas as pd, pathlib

subprocess.run(
    [
        sys.executable,
        "distill.py",
        "--device",
        "cuda",
        "--seed",
        "10",
        "--epochs",
        "3",
        "--batch_size",
        "64",
        "--lr",
        "0.005",
        "--img_size",
        "224",
        "--num_workers",
        "2",
        "--save_path",
        "submission.csv",
    ],
    check=True,
)

print("submission exists:", os.path.exists("submission.csv"))
sub = pd.read_csv("submission.csv")
print(sub.head())
print("Row count:", len(sub), "Cols:", list(sub.columns))

root_dir_candidates = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "../input/plant-pathology-2020-fgvc7",
    "../data/plant-pathology-2020-fgvc7",
]
sample_path = None
for c in root_dir_candidates:
    p = os.path.join(c, "sample_submission.csv")
    if os.path.exists(p):
        sample_path = p
        break

if sample_path is not None:
    sample = pd.read_csv(sample_path)
    assert list(sub.columns) == list(sample.columns), (
        list(sub.columns),
        list(sample.columns),
    )
else:
    assert sub.columns[0] == "image_id" and len(sub.columns) == 5

assert len(sub) == 183
target_cols = [c for c in sub.columns if c != "image_id"]
assert sub[target_cols].min().min() >= -1e-6
assert sub[target_cols].max().max() <= 1.0 + 1e-6
print("OK")
print("Last lines of distill.py:")
print("\n".join(pathlib.Path("distill.py").read_text().splitlines()[-18:]))
