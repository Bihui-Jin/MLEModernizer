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

0.91331

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, subprocess, textwrap, pathlib

print("Skipping external pip installs (offline Kaggle environment).")




## === cell 1
from pathlib import Path

distill_code = r"""
import argparse
import os
import random
import traceback

# FIX (stability only): If deterministic algorithms are enabled anywhere, CUDA backprop can error unless
# CUBLAS_WORKSPACE_CONFIG is set BEFORE importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Default paths; will be resolved to an existing layout at runtime.
ROOT_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
TEST_CSV = os.path.join(ROOT_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(ROOT_DIR, "sample_submission.csv")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def set_seed(seed: int = 42):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Keep cudnn deterministic flags but DO NOT force torch.use_deterministic_algorithms(True),
    # which can crash on Kaggle GPUs due to CuBLAS non-deterministic ops.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_data_paths():
    # FIX (score & validity): prefer the paths that actually exist in the user's environment tree.
    # This prevents training on missing/incorrect files (which would fail or produce invalid outputs).
    global ROOT_DIR, IMAGES_DIR, TRAIN_CSV, TEST_CSV, SAMPLE_SUB_CSV

    root_candidates = [
        # Most likely per provided tree
        "/kaggle/data/plant-pathology-2020-fgvc7",
        "/kaggle/input/plant-pathology-2020-fgvc7",
        # Some environments nest the dataset one level deeper
        "/kaggle/data/plant-pathology-2020-fgvc7/plant-pathology-2020-fgvc7",
        "/kaggle/input/plant-pathology-2020-fgvc7/plant-pathology-2020-fgvc7",
        # Fallback: search inside these bases
        "/kaggle/input",
        "/kaggle/data",
    ]

    chosen_root = None
    for r in root_candidates:
        if not r or not os.path.exists(r):
            continue

        # If r is a base folder like /kaggle/input or /kaggle/data, try to find competition folder inside it.
        if os.path.isdir(r) and os.path.basename(r) in ("input", "data"):
            for sub in (
                "plant-pathology-2020-fgvc7",
                "plant-pathology-2020-fgvc7/plant-pathology-2020-fgvc7",
            ):
                rr = os.path.join(r, sub)
                ok = (
                    os.path.exists(os.path.join(rr, "train.csv"))
                    and os.path.exists(os.path.join(rr, "test.csv"))
                    and os.path.exists(os.path.join(rr, "sample_submission.csv"))
                    and os.path.exists(os.path.join(rr, "images"))
                )
                if ok:
                    chosen_root = rr
                    break
            if chosen_root is not None:
                break

        ok = (
            os.path.exists(os.path.join(r, "train.csv"))
            and os.path.exists(os.path.join(r, "test.csv"))
            and os.path.exists(os.path.join(r, "sample_submission.csv"))
            and os.path.exists(os.path.join(r, "images"))
        )
        if ok:
            chosen_root = r
            break

    if chosen_root is None:
        chosen_root = ROOT_DIR

    ROOT_DIR = chosen_root
    IMAGES_DIR = os.path.join(ROOT_DIR, "images")
    TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
    TEST_CSV = os.path.join(ROOT_DIR, "test.csv")
    SAMPLE_SUB_CSV = os.path.join(ROOT_DIR, "sample_submission.csv")

    if not (
        os.path.exists(IMAGES_DIR)
        and os.path.exists(TRAIN_CSV)
        and os.path.exists(TEST_CSV)
        and os.path.exists(SAMPLE_SUB_CSV)
    ):
        raise FileNotFoundError(
            "Could not resolve required data paths. "
            f"ROOT_DIR={ROOT_DIR}, IMAGES_DIR={IMAGES_DIR}, TRAIN_CSV={TRAIN_CSV}, TEST_CSV={TEST_CSV}, SAMPLE_SUB_CSV={SAMPLE_SUB_CSV}"
        )


def get_image_path(image_id: str) -> str:
    image_id = str(image_id)
    candidates = []
    if image_id.lower().endswith(".jpg"):
        candidates.append(os.path.join(IMAGES_DIR, image_id))
    else:
        candidates.append(os.path.join(IMAGES_DIR, f"{image_id}.jpg"))
        candidates.append(os.path.join(IMAGES_DIR, image_id))

    for p in candidates:
        if os.path.exists(p):
            return p

    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")


class Plant2020Dataset(Dataset):
    def __init__(self, csv_path: str, train: bool, image_size: int = 224):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.train = train
        self.image_size = image_size

        if self.train:
            self.labels = self.df[TARGET_COLS].values.astype(np.float32)
        else:
            self.labels = np.zeros((len(self.df), 4), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def _transform(self, img: Image.Image) -> torch.Tensor:
        img = img.resize((self.image_size, self.image_size))
        x = np.asarray(img).astype(np.float32) / 255.0

        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        elif x.shape[2] == 4:
            x = x[:, :, :3]

        x = np.transpose(x, (2, 0, 1))

        # Uses ImageNet normalization (helps natural-image CNN training stability/score).
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        x = (x - mean) / std
        return torch.from_numpy(x)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        path = get_image_path(image_id)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path} (image_id={image_id}, IMAGES_DIR={IMAGES_DIR})")

        with Image.open(path) as im:
            img = im.convert("RGB")

        x = self._transform(img)
        y = torch.from_numpy(self.labels[idx]).float()
        return x, y


class SmallCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.head(x)


def mean_columnwise_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    aucs = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        yp = y_pred[:, c]
        if np.all(yt == 0) or np.all(yt == 1):
            continue
        order = np.argsort(yp)
        yt_sorted = yt[order]
        n_pos = yt_sorted.sum()
        n_neg = len(yt_sorted) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = np.arange(1, len(yt_sorted) + 1)
        sum_ranks_pos = ranks[yt_sorted == 1].sum()
        u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
        auc = u / (n_pos * n_neg)
        aucs.append(float(auc))
    return float(np.mean(aucs)) if aucs else 0.0


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for x, _ in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds.append(probs.cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * x.size(0)
    return running / max(1, len(loader.dataset))


def stratified_split_indices(df: pd.DataFrame, seed: int, train_frac: float = 0.8):
    # FIX (score): stratify train/val split by class to reduce imbalance and improve generalization.
    # This preserves the same overall training approach; only the split policy changes.
    labels = df[TARGET_COLS].values
    cls = labels.argmax(axis=1)

    rng = np.random.default_rng(seed)
    tr_idx = []
    va_idx = []
    for c in range(labels.shape[1]):
        idx_c = np.where(cls == c)[0]
        rng.shuffle(idx_c)
        n_tr = int(train_frac * len(idx_c))
        tr_idx.append(idx_c[:n_tr])
        va_idx.append(idx_c[n_tr:])

    tr_idx = np.concatenate(tr_idx) if tr_idx else np.array([], dtype=int)
    va_idx = np.concatenate(va_idx) if va_idx else np.array([], dtype=int)
    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)
    return tr_idx, va_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("-student_only", action="store_true")
    parser.add_argument("-test_only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    resolve_data_paths()

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    print(f"Using device={device} | ROOT_DIR={ROOT_DIR} | IMAGES_DIR={IMAGES_DIR}")

    full_train = pd.read_csv(TRAIN_CSV)

    tr_idx, va_idx = stratified_split_indices(full_train, seed=args.seed, train_frac=0.8)

    tr_df = full_train.iloc[tr_idx].reset_index(drop=True)
    va_df = full_train.iloc[va_idx].reset_index(drop=True)

    tr_csv = "train_split.csv"
    va_csv = "val_split.csv"
    tr_df.to_csv(tr_csv, index=False)
    va_df.to_csv(va_csv, index=False)

    train_ds = Plant2020Dataset(tr_csv, train=True, image_size=224)
    val_ds = Plant2020Dataset(va_csv, train=True, image_size=224)

    # FIX (validity & alignment): iterate test set in exact sample_submission order to avoid reindex/fill.
    sub = pd.read_csv(SAMPLE_SUB_CSV)
    test_order_csv = "test_ordered_like_sample_sub.csv"
    pd.DataFrame({"image_id": sub["image_id"].astype(str).tolist()}).to_csv(test_order_csv, index=False)
    test_ds = Plant2020Dataset(test_order_csv, train=False, image_size=224)

    num_workers = 2
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = SmallCNN(num_classes=4).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    num_epochs = 3
    for epoch in range(args.start_epoch, num_epochs):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_probs = predict(model, val_loader, device)
        val_true = val_ds.labels
        val_auc = mean_columnwise_roc_auc(val_true, val_probs)
        print(f"Epoch {epoch+1}/{num_epochs} | train_loss={tr_loss:.4f} | val_mean_auc={val_auc:.4f}")

    test_probs = predict(model, test_loader, device)

    if len(test_probs) != len(sub):
        raise RuntimeError(f"Prediction count mismatch: len(test_probs)={len(test_probs)} vs len(sample_submission)={len(sub)}")

    out = sub.copy()
    out[TARGET_COLS] = test_probs

    if out.shape[0] != sub.shape[0]:
        raise RuntimeError(f"Invalid submission row count: got {out.shape[0]} expected {sub.shape[0]}")
    if out.columns.tolist() != ["image_id"] + TARGET_COLS:
        raise RuntimeError(f"Invalid submission columns: {out.columns.tolist()}")

    save_path = "submission.csv"
    out.to_csv(save_path, index=False)
    print(f"Saved submission to {save_path} with shape {out.shape} and columns {list(out.columns)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FATAL ERROR:", repr(e))
        traceback.print_exc()
        raise
"""
Path("distill.py").write_text(distill_code)
print("Wrote distill.py")




## === cell 2
from pathlib import Path

config_text = r"""
# Kept for compatibility; simplified distill.py does not require torchdistill/timm configs.
"""
Path("config.yaml").write_text(config_text)
print("Wrote config.yaml")




## === cell 3
import sys, runpy

argv_backup = sys.argv[:]
sys.argv = [
    "distill.py",
    "--config",
    "config.yaml",
    "--device",
    "cuda" if __import__("torch").cuda.is_available() else "cpu",
    "--seed",
    "10",
    "-student_only",
    "--log",
    "log.log",
]
runpy.run_path("distill.py", run_name="__main__")
sys.argv = argv_backup




## === cell 4
from pathlib import Path
import pandas as pd

print("Submission exists:", Path("submission.csv").exists())
sub = pd.read_csv("submission.csv")
print(sub.head())
print(sub.shape)
print("Columns:", list(sub.columns))
assert sub.shape[0] == 183, f"Expected 183 rows but got {sub.shape[0]}"
assert sub.shape[1] == 5
assert sub.columns.tolist() == [
    "image_id",
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
]
assert sub["image_id"].isna().sum() == 0
print("submission.csv looks valid.")
