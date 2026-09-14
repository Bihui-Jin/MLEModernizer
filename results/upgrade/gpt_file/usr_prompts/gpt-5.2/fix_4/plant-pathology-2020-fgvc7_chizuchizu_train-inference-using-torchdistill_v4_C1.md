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

# 8. Previous improvement plan

N/A

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
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.backends import cudnn

import timm


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True


def resolve_root_dir():
    # Change rationale (submission validity): include /kaggle/data paths used in this environment,
    # and ensure we return the competition folder that actually contains train.csv/test.csv/images/.
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
            if os.path.exists(os.path.join(c, "train.csv")):
                return c
    return "../input/plant-pathology-2020-fgvc7"


def get_img_path(root_dir, image_id):
    # Change rationale (run end-to-end): handle both layouts:
    #   root/images/<id>.jpg  and root/images/images/<id>.jpg
    p1 = os.path.join(root_dir, "images", f"{image_id}.jpg")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(root_dir, "images", "images", f"{image_id}.jpg")
    if os.path.exists(p2):
        return p2
    # fallback for debugging
    return p1


def build_transform(img_size=224, mean=(0.49139968, 0.48215841, 0.44653091), std=(0.24703223, 0.24348513, 0.26158784)):
    def _transform(pil_img: Image.Image):
        img = pil_img.resize((img_size, img_size), resample=Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0  # HWC RGB
        arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return torch.from_numpy(arr)
    return _transform


class Plant2020Dataset(Dataset):
    def __init__(self, csv_path, root_dir, inf=False, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.root_dir = root_dir
        self.inf = inf
        self.transform = transform
        if not inf:
            self.labels = self.df[["healthy", "multiple_diseases", "rust", "scab"]].values.astype(np.float32)
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


def make_submission(root_dir, save_path, probs):
    # Change rationale (score + validity): build submission by joining with test.csv to guarantee
    # correct row order and image_id alignment, rather than trusting sample_submission ordering.
    test_path = os.path.join(root_dir, "test.csv")
    sub = pd.read_csv(test_path)
    cols = ["healthy", "multiple_diseases", "rust", "scab"]
    probs = np.clip(probs, 0.0, 1.0)
    if probs.shape != (len(sub), 4):
        raise ValueError(f"Expected probs shape {(len(sub), 4)}, got {probs.shape}")
    sub[cols] = probs
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

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        raise FileNotFoundError(f"Could not find train/test CSVs under root_dir={root_dir}")

    transform = build_transform(img_size=args.img_size)

    train_df = pd.read_csv(train_csv)
    idx = np.arange(len(train_df))
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    tmp_train = "tmp_train_split.csv"
    tmp_val = "tmp_val_split.csv"
    train_df.iloc[tr_idx].to_csv(tmp_train, index=False)
    train_df.iloc[va_idx].to_csv(tmp_val, index=False)

    ds_train = Plant2020Dataset(tmp_train, root_dir, inf=False, transform=transform)
    ds_val = Plant2020Dataset(tmp_val, root_dir, inf=False, transform=transform)
    ds_test = Plant2020Dataset(test_csv, root_dir, inf=True, transform=transform)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=max(128, args.batch_size), shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=max(128, args.batch_size), shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True, num_classes=4)
    model.to(device)

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
    sub = make_submission(root_dir, args.save_path, probs)
    print("Resolved root_dir:", root_dir)
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
assert list(sub.columns) == ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
assert len(sub) == 183
assert sub[["healthy", "multiple_diseases", "rust", "scab"]].min().min() >= -1e-6
assert sub[["healthy", "multiple_diseases", "rust", "scab"]].max().max() <= 1.0 + 1e-6
print("OK")
print("Last lines of distill.py:")
print("\n".join(pathlib.Path("distill.py").read_text().splitlines()[-15:]))
