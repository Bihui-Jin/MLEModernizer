# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Higher is better

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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def resolve_data_paths():
    # Robustly find dataset root across both /kaggle/input and /kaggle/data,
    # and across both "flat" and "nested" folder layouts in the provided tree.
    global ROOT_DIR, IMAGES_DIR, TRAIN_CSV, TEST_CSV, SAMPLE_SUB_CSV

    root_candidates = [
        "/kaggle/input/plant-pathology-2020-fgvc7/plant-pathology-2020-fgvc7",
        "/kaggle/data/plant-pathology-2020-fgvc7/plant-pathology-2020-fgvc7",
        "/kaggle/input/plant-pathology-2020-fgvc7",
        "/kaggle/data/plant-pathology-2020-fgvc7",
        # also support the "flat" copies shown at /kaggle/input
        "/kaggle/input",
        "/kaggle/data",
    ]

    chosen_root = None
    for r in root_candidates:
        if not r or not os.path.exists(r):
            continue
        # if user points to /kaggle/input, try to locate the competition folder inside it
        if os.path.isdir(r) and os.path.basename(r) in ("input", "data"):
            for sub in ("plant-pathology-2020-fgvc7", "plant-pathology-2020-fgvc7/plant-pathology-2020-fgvc7"):
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
    # The CSV image_id values are like "Train_0"/"Test_0" without ".jpg" in this dataset.
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

        mean = np.array([0.49139968, 0.48215841, 0.44653091], dtype=np.float32)[:, None, None]
        std = np.array([0.24703223, 0.24348513, 0.26158784], dtype=np.float32)[:, None, None]
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

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(full_train))
    rng.shuffle(idx)
    n_tr = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:n_tr], idx[n_tr:]

    tr_df = full_train.iloc[tr_idx].reset_index(drop=True)
    va_df = full_train.iloc[va_idx].reset_index(drop=True)

    tr_csv = "train_split.csv"
    va_csv = "val_split.csv"
    tr_df.to_csv(tr_csv, index=False)
    va_df.to_csv(va_csv, index=False)

    train_ds = Plant2020Dataset(tr_csv, train=True, image_size=224)
    val_ds = Plant2020Dataset(va_csv, train=True, image_size=224)
    test_ds = Plant2020Dataset(TEST_CSV, train=False, image_size=224)

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

    # Ensure output row count and order match sample_submission exactly.
    sub = pd.read_csv(SAMPLE_SUB_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Use test.csv as the authoritative list of ids, and align to sample_submission order.
    test_ids = test_df["image_id"].astype(str).tolist()
    sub_ids = sub["image_id"].astype(str).tolist()

    if len(test_probs) != len(test_ids):
        raise RuntimeError(f"Prediction count mismatch: len(test_probs)={len(test_probs)} vs len(test_ids)={len(test_ids)}")

    pred_df = pd.DataFrame(test_probs, columns=TARGET_COLS)
    pred_df["image_id"] = test_ids

    # BUGFIX: if any duplicates exist, keep first to avoid reindex explosions and wrong row counts.
    pred_df = pred_df.drop_duplicates(subset=["image_id"], keep="first").set_index("image_id")

    aligned = pred_df.reindex(sub_ids)

    if aligned.isna().any().any():
        missing = aligned.index[aligned.isna().any(axis=1)].tolist()
        print(f"Warning: missing {len(missing)} predictions after aligning to sample_submission. Filling with 0.5. Example missing: {missing[:10]}")
        aligned = aligned.fillna(0.5)

    out = sub.copy()
    out[TARGET_COLS] = aligned[TARGET_COLS].to_numpy()

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
import subprocess, sys

subprocess.run(
    [
        sys.executable,
        "distill.py",
        "--config",
        "config.yaml",
        "--device",
        "cuda",
        "--seed",
        "10",
        "-student_only",
        "--log",
        "log.log",
    ],
    check=True,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
CalledProcessError                        Traceback (most recent call last)
/tmp/ipykernel_11/2903886045.py in <cell line: 0>()
      1 import subprocess, sys
      2 
----> 3 subprocess.run(
      4     [
      5         sys.executable,

/usr/lib/python3.11/subprocess.py in run(input, capture_output, timeout, check, *popenargs, **kwargs)
    569         retcode = process.poll()
    570         if check and retcode:
--> 571             raise CalledProcessError(retcode, process.args,
    572                                      output=stdout, stderr=stderr)
    573     return CompletedProcess(process.args, retcode, stdout, stderr)

CalledProcessError: Command '['/usr/bin/python3', 'distill.py', '--config', 'config.yaml', '--device', 'cuda', '--seed', '10', '-student_only', '--log', 'log.log']' returned non-zero exit status 1.

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

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1811335463.py in <cell line: 0>()
      3 
      4 print("Submission exists:", Path("submission.csv").exists())
----> 5 sub = pd.read_csv("submission.csv")
      6 print(sub.head())
      7 print(sub.shape)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'

## --- ERROR in outputing the csv:
Invalid submission: Expected submission to have 183 rows but got 328
