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

3.7

# 2. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os
import random
import time

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

BASE = "/kaggle/input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE, "train", "train")
TEST_DIR = os.path.join(BASE, "test", "test")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing: {TEST_DIR}"



## === cell 1
labels = pd.read_csv(TRAIN_CSV)
labels["has_cactus"] = labels["has_cactus"].astype(int)

train_df, valid_df = train_test_split(
    labels,
    stratify=labels["has_cactus"],
    test_size=0.2,
    random_state=SEED,
)

sample_sub = pd.read_csv(SAMPLE_SUB)
test_df = sample_sub[["id"]].copy()  # preserves expected order for submission

print("Train:", train_df.shape, "Valid:", valid_df.shape, "Test:", test_df.shape)



## === cell 2
IM_SIZE = 32

train_tfms = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((IM_SIZE, IM_SIZE)),
        T.RandomHorizontalFlip(p=0.5),  # mirrors your HorizontalFlip augmentation
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

val_tfms = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((IM_SIZE, IM_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


class CactusDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, has_labels=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "id"]
        path = os.path.join(self.img_dir, img_id)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(img)

        if self.has_labels:
            y = float(self.df.loc[idx, "has_cactus"])
            y = torch.tensor([y], dtype=torch.float32)
            return {"image": img, "label": y, "id": img_id}
        else:
            return {"image": img, "id": img_id}




## === cell 3
batch_size = 64
num_workers = 0  # keep as in your original (workers=0)

train_ds = CactusDataset(train_df, TRAIN_DIR, transforms=train_tfms, has_labels=True)
valid_ds = CactusDataset(valid_df, TRAIN_DIR, transforms=val_tfms, has_labels=True)
test_ds = CactusDataset(test_df, TEST_DIR, transforms=val_tfms, has_labels=False)

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)
valid_dl = DataLoader(
    valid_ds,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)
test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

print("Batches:", len(train_dl), len(valid_dl), len(test_dl))



## === cell 4


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=2):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(output_size=(size, size))
        self.mp = nn.AdaptiveMaxPool2d(output_size=(size, size))

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], dim=1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(nn.Module):
    def __init__(self, num_classes=1, p=0.2, pooling_size=2):
        super().__init__()
        backbone = torchvision.models.densenet169(
            weights=torchvision.models.DenseNet169_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features  # conv backbone

        in_features = (1664 * 2) * (pooling_size * pooling_size)

        self.head = nn.Sequential(
            AdaptiveConcatPool2d(size=pooling_size),
            Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(p),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = self.head(x)
        return x


model = Net(num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()



## === cell 5
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)


def run_epoch(train=True, scheduler=None):
    model.train(train)
    loader = train_dl if train else valid_dl

    total_loss = 0.0
    all_targets = []
    all_logits = []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * x.size(0)
        all_targets.append(y.detach().cpu().numpy().reshape(-1))
        all_logits.append(logits.detach().cpu().numpy().reshape(-1))

    all_targets = np.concatenate(all_targets)
    all_probs = 1 / (1 + np.exp(-np.concatenate(all_logits)))  # sigmoid

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = float("nan")

    return total_loss / len(loader.dataset), auc




## === cell 6
def train_one_cycle(epochs, max_lr, div_factor=25, pct_start=0.3):
    def _resolve_img_dir(img_dir):
        if os.path.isdir(img_dir):
            return img_dir
        alt = os.path.dirname(img_dir)
        if os.path.isdir(alt):
            return alt
        return img_dir

    global train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl
    resolved_train_dir = _resolve_img_dir(train_ds.img_dir)
    resolved_test_dir = _resolve_img_dir(test_ds.img_dir)

    train_ds.img_dir = resolved_train_dir
    valid_ds.img_dir = resolved_train_dir
    test_ds.img_dir = resolved_test_dir

    train_dl = DataLoader(
        train_ds,
        batch_size=train_dl.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=valid_dl.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=test_dl.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    steps_per_epoch = len(train_dl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        div_factor=div_factor,
        pct_start=pct_start,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
    )

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_auc = run_epoch(train=True, scheduler=scheduler)
        va_loss, va_auc = run_epoch(train=False, scheduler=None)
        dt = time.time() - t0
        print(
            f"[OneCycle max_lr={max_lr}] Epoch {ep}/{epochs} | "
            f"train loss {tr_loss:.4f} auc {tr_auc:.5f} | "
            f"valid loss {va_loss:.4f} auc {va_auc:.5f} | {dt:.1f}s"
        )


train_one_cycle(epochs=5, max_lr=1e-2, div_factor=25, pct_start=0.3)

train_one_cycle(epochs=3, max_lr=1e-3, div_factor=25, pct_start=0.3)


## --- ERROR in cell 6, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/331604613.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     70[0m [0;34m[0m[0m
[1;32m     71[0m [0;34m[0m[0m
[0;32m---> 72[0;31m [0mtrain_one_cycle[0m[0;34m([0m[0mepochs[0m[0;34m=[0m[0;36m5[0m[0;34m,[0m [0mmax_lr[0m[0;34m=[0m[0;36m1e-2[0m[0;34m,[0m [0mdiv_factor[0m[0;34m=[0m[0;36m25[0m[0;34m,[0m [0mpct_start[0m[0;34m=[0m[0;36m0.3[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     73[0m [0;34m[0m[0m
[1;32m     74[0m [0mtrain_one_cycle[0m[0;34m([0m[0mepochs[0m[0;34m=[0m[0;36m3[0m[0;34m,[0m [0mmax_lr[0m[0;34m=[0m[0;36m1e-3[0m[0;34m,[0m [0mdiv_factor[0m[0;34m=[0m[0;36m25[0m[0;34m,[0m [0mpct_start[0m[0;34m=[0m[0;36m0.3[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/331604613.py[0m in [0;36mtrain_one_cycle[0;34m(epochs, max_lr, div_factor, pct_start)[0m
[1;32m     60[0m     [0;32mfor[0m [0mep[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0mepochs[0m [0;34m+[0m [0;36m1[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     61[0m         [0mt0[0m [0;34m=[0m [0mtime[0m[0;34m.[0m[0mtime[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 62[0;31m         [0mtr_loss[0m[0;34m,[0m [0mtr_auc[0m [0;34m=[0m [0mrun_epoch[0m[0;34m([0m[0mtrain[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0mscheduler[0m[0;34m=[0m[0mscheduler[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     63[0m         [0mva_loss[0m[0;34m,[0m [0mva_auc[0m [0;34m=[0m [0mrun_epoch[0m[0;34m([0m[0mtrain[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0mscheduler[0m[0;34m=[0m[0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     64[0m         [0mdt[0m [0;34m=[0m [0mtime[0m[0;34m.[0m[0mtime[0m[0;34m([0m[0;34m)[0m [0;34m-[0m [0mt0[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1174355195.py[0m in [0;36mrun_epoch[0;34m(train, scheduler)[0m
[1;32m     12[0m     [0mall_logits[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     13[0m [0;34m[0m[0m
[0;32m---> 14[0;31m     [0;32mfor[0m [0mbatch[0m [0;32min[0m [0mloader[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     15[0m         [0mx[0m [0;34m=[0m [0mbatch[0m[0;34m[[0m[0;34m"image"[0m[0;34m][0m[0;34m.[0m[0mto[0m[0;34m([0m[0mdevice[0m[0;34m,[0m [0mnon_blocking[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m         [0my[0m [0;34m=[0m [0mbatch[0m[0;34m[[0m[0;34m"label"[0m[0;34m][0m[0;34m.[0m[0mto[0m[0;34m([0m[0mdevice[0m[0;34m,[0m [0mnon_blocking[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m    706[0m                 [0;31m# TODO(https://github.com/pytorch/pytorch/issues/76750)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    707[0m                 [0mself[0m[0;34m.[0m[0m_reset[0m[0;34m([0m[0;34m)[0m  [0;31m# type: ignore[call-arg][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 708[0;31m             [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_next_data[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    709[0m             [0mself[0m[0;34m.[0m[0m_num_yielded[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m    710[0m             if (

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_next_data[0;34m(self)[0m
[1;32m    762[0m     [0;32mdef[0m [0m_next_data[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    763[0m         [0mindex[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_next_index[0m[0;34m([0m[0;34m)[0m  [0;31m# may raise StopIteration[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 764[0;31m         [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_dataset_fetcher[0m[0;34m.[0m[0mfetch[0m[0;34m([0m[0mindex[0m[0;34m)[0m  [0;31m# may raise StopIteration[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    765[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0m_pin_memory[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    766[0m             [0mdata[0m [0;34m=[0m [0m_utils[0m[0;34m.[0m[0mpin_memory[0m[0;34m.[0m[0mpin_memory[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_pin_memory_device[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py[0m in [0;36mfetch[0;34m(self, possibly_batched_index)[0m
[1;32m     50[0m                 [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdataset[0m[0;34m.[0m[0m__getitems__[0m[0;34m([0m[0mpossibly_batched_index[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 52[0;31m                 [0mdata[0m [0;34m=[0m [0;34m[[0m[0mself[0m[0;34m.[0m[0mdataset[0m[0;34m[[0m[0midx[0m[0;34m][0m [0;32mfor[0m [0midx[0m [0;32min[0m [0mpossibly_batched_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     53[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     54[0m             [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdataset[0m[0;34m[[0m[0mpossibly_batched_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[1;32m     50[0m                 [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdataset[0m[0;34m.[0m[0m__getitems__[0m[0;34m([0m[0mpossibly_batched_index[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 52[0;31m                 [0mdata[0m [0;34m=[0m [0;34m[[0m[0mself[0m[0;34m.[0m[0mdataset[0m[0;34m[[0m[0midx[0m[0;34m][0m [0;32mfor[0m [0midx[0m [0;32min[0m [0mpossibly_batched_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     53[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     54[0m             [0mdata[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdataset[0m[0;34m[[0m[0mpossibly_batched_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/73089887.py[0m in [0;36m__getitem__[0;34m(self, idx)[0m
[1;32m     38[0m         [0mimg[0m [0;34m=[0m [0mcv2[0m[0;34m.[0m[0mimread[0m[0;34m([0m[0mpath[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     39[0m         [0;32mif[0m [0mimg[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 40[0;31m             [0;32mraise[0m [0mFileNotFoundError[0m[0;34m([0m[0mpath[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     41[0m         [0mimg[0m [0;34m=[0m [0mcv2[0m[0;34m.[0m[0mcvtColor[0m[0;34m([0m[0mimg[0m[0;34m,[0m [0mcv2[0m[0;34m.[0m[0mCOLOR_BGR2RGB[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     42[0m [0;34m[0m[0m

[0;31mFileNotFoundError[0m: /kaggle/input/aerial-cactus-identification/train/train/7dfdfbe16c392bd600291159b8dc81ce.jpg

## === cell 7
model.eval()
test_ids = []
test_probs = []

with torch.no_grad():
    for batch in test_dl:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x).view(-1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        test_probs.append(probs)
        test_ids.extend(batch["id"])

test_probs = np.concatenate(test_probs)

pred_map = dict(zip(test_ids, test_probs))
sub = sample_sub.copy()
sub["has_cactus"] = sub["id"].map(pred_map).astype(float)

assert sub.shape[0] == sample_sub.shape[0]
assert sub["has_cactus"].notnull().all()

sub.to_csv("sub.csv", index=False)
print(sub.head())
print("Wrote sub.csv with shape:", sub.shape)
