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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Target score

0.9998

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

import pretrainedmodels  # already available via pip in original notebook


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2139591584.py in <cell line: 0>()
     17 from sklearn.metrics import roc_auc_score, accuracy_score
     18 
---> 19 import pretrainedmodels  # already available via pip in original notebook
     20 
     21 

ModuleNotFoundError: No module named 'pretrainedmodels'

## === cell 1
DATA_DIR = Path("/kaggle/input/aerial-cactus-identification")
TRAIN_CSV = DATA_DIR / "train.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

assert TRAIN_CSV.exists(), f"Missing {TRAIN_CSV}"
assert SAMPLE_SUB.exists(), f"Missing {SAMPLE_SUB}"
assert TRAIN_DIR.exists(), f"Missing {TRAIN_DIR}"
assert TEST_DIR.exists(), f"Missing {TEST_DIR}"

labels = pd.read_csv(TRAIN_CSV)
labels["has_cactus"] = labels["has_cactus"].astype(int)
labels["data_type"] = "train"

sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub.head()



## === cell 2
test_img = sorted([p.name for p in TEST_DIR.glob("*.jpg")])
test_df = pd.DataFrame({"id": test_img})
test_df["has_cactus"] = -1
test_df["data_type"] = "test"

train_df, valid_df = train_test_split(
    labels, stratify=labels.has_cactus, test_size=0.2, random_state=42
)

train_df["data_type"] = "train"
valid_df["data_type"] = "train"

train_df.head(), valid_df.head(), test_df.head()




## === cell 3
class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(
        3, 1, 1
    )
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(
        3, 1, 1
    )
    return (x - mean) / std


def augs_hflip(image_rgb: np.ndarray, p: float = 0.5) -> np.ndarray:
    if np.random.rand() < p:
        return np.ascontiguousarray(image_rgb[:, ::-1, :])
    return image_rgb


class CactusDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, train: bool, size: int = 32, hflip_p: float = 0.5
    ):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.size = size
        self.hflip_p = hflip_p

    def __len__(self):
        return len(self.df)

    def _read_image(self, row):
        if row["data_type"] == "train":
            img_path = TRAIN_DIR / row["id"]
        else:
            img_path = TEST_DIR / row["id"]

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img_rgb = img_bgr[:, :, ::-1]  # BGR->RGB
        if self.size is not None:
            img_rgb = cv2.resize(
                img_rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR
            )
        if self.train:
            img_rgb = augs_hflip(img_rgb, p=self.hflip_p)
        return img_rgb

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self._read_image(row)

        x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        x = normalize_imagenet(x)

        y = torch.tensor([float(row["has_cactus"])], dtype=torch.float32)

        return {"image": x, "label": y}




## === cell 4
batch_size = 64
workers = 0

train_ds = CactusDataset(train_df, train=True, size=32, hflip_p=0.5)
val_ds = CactusDataset(valid_df, train=False, size=32, hflip_p=0.0)
test_ds = CactusDataset(test_df, train=False, size=32, hflip_p=0.0)

train_dl = DataLoader(
    train_ds, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True
)
val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=workers, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=workers, shuffle=False)

next(iter(train_dl))["image"].shape, next(iter(train_dl))["label"].shape




## === cell 5
class Net(nn.Module):
    def __init__(
        self,
        num_classes: int,
        p: float = 0.2,
        pooling_size: int = 2,
        last_conv_size: int = 1664,
        arch: str = "densenet169",
        pretrained: str = "imagenet",
    ) -> None:
        super().__init__()
        net = pretrainedmodels.__dict__[arch](pretrained=pretrained)
        modules = list(net.children())[:-1]  # delete last layer
        modules += [
            nn.Sequential(
                Flatten(),
                nn.BatchNorm1d(1664),
                nn.Dropout(p),
                nn.Linear(1664, num_classes),
            )
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        logits = self.net(x)
        return logits


model = Net(num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3453669057.py in <cell line: 0>()
     28 
     29 
---> 30 model = Net(num_classes=1).to(device)
     31 criterion = nn.BCEWithLogitsLoss()
     32 

/tmp/ipykernel_11/3453669057.py in __init__(self, num_classes, p, pooling_size, last_conv_size, arch, pretrained)
     11     ) -> None:
     12         super().__init__()
---> 13         net = pretrainedmodels.__dict__[arch](pretrained=pretrained)
     14         modules = list(net.children())[:-1]  # delete last layer
     15         modules += [

NameError: name 'pretrainedmodels' is not defined

## === cell 6
def bce_accuracy(
    target: torch.Tensor, preds: torch.Tensor, thresh: float = 0.5
) -> float:
    target_np = target.detach().cpu().numpy()
    pred_np = (torch.sigmoid(preds).detach().cpu().numpy() > thresh).astype(int)
    return accuracy_score(target_np, pred_np)


def roc_auc(target: torch.Tensor, preds: torch.Tensor) -> float:
    target_np = target.detach().cpu().numpy()
    pred_np = torch.sigmoid(preds).detach().cpu().numpy()
    try:
        return roc_auc_score(target_np, pred_np)
    except ValueError:
        return float("nan")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_t, all_p = [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
        all_t.append(y.detach().cpu())
        all_p.append(logits.detach().cpu())
    all_t = torch.cat(all_t, dim=0)
    all_p = torch.cat(all_p, dim=0)
    return {
        "loss": total_loss / max(n, 1),
        "acc": bce_accuracy(all_t, all_p),
        "auc": roc_auc(all_t, all_p),
    }


def one_cycle_train(
    model,
    train_loader,
    val_loader,
    max_lr,
    cycle_len,
    momentum_range,
    div_factor,
    increase_fraction,
):
    min_lr = max_lr / div_factor
    m_high, m_low = momentum_range

    for pg in optimizer.param_groups:
        pg["lr"] = min_lr
        if "momentum" in pg:
            pg["momentum"] = m_high

    steps_per_epoch = len(train_loader)
    total_steps = cycle_len * steps_per_epoch
    up_steps = int(total_steps * increase_fraction)
    down_steps = total_steps - up_steps

    global_step = 0
    for epoch in range(cycle_len):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            if global_step < up_steps:
                pct = global_step / max(up_steps, 1)
                lr = min_lr + pct * (max_lr - min_lr)
                mom = m_high + pct * (m_low - m_high)
            else:
                pct = (global_step - up_steps) / max(down_steps, 1)
                lr = max_lr - pct * (max_lr - min_lr)
                mom = m_low + pct * (m_high - m_low)

            for pg in optimizer.param_groups:
                pg["lr"] = lr
                if "momentum" in pg:
                    pg["momentum"] = mom

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            global_step += 1

        train_loss = running_loss / (
            len(train_loader.dataset) if hasattr(train_loader, "dataset") else 1
        )
        val_metrics = evaluate(model, val_loader)
        dt = time.time() - t0
        print(
            f"epoch {epoch+1}/{cycle_len} | "
            f"train_loss {train_loss:.4f} | val_loss {val_metrics['loss']:.4f} | "
            f"val_acc {val_metrics['acc']:.4f} | val_auc {val_metrics['auc']:.6f} | "
            f"time {dt:.1f}s"
        )




## === cell 7
one_cycle_train(
    model,
    train_dl,
    val_dl,
    max_lr=1e-2,
    cycle_len=5,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.3,
)

one_cycle_train(
    model,
    train_dl,
    val_dl,
    max_lr=1e-3,
    cycle_len=4,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.2,
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2154305431.py in <cell line: 0>()
      1 # Two-phase training preserved from original cells 19 & 20
      2 one_cycle_train(
----> 3     model,
      4     train_dl,
      5     val_dl,

NameError: name 'model' is not defined

## === cell 8
@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    preds = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        preds.append(p)
    return np.vstack(preds).reshape(-1)


preds = predict_loader(model, test_dl)
preds[:10], preds.shape



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3044554191.py in <cell line: 0>()
     11 
     12 
---> 13 preds = predict_loader(model, test_dl)
     14 preds[:10], preds.shape
     15 

NameError: name 'model' is not defined

## === cell 9
pred_df = pd.DataFrame(
    {"id": test_df["id"].values, "has_cactus": preds.astype(np.float64)}
)

sub = sample_sub[["id"]].merge(pred_df, on="id", how="left")
if sub["has_cactus"].isna().any():
    missing = sub[sub["has_cactus"].isna()]["id"].head(5).tolist()
    raise RuntimeError(f"Some test ids missing predictions, examples: {missing}")

sub_path = Path("sub.csv")
sub.to_csv(sub_path, index=False)

sub.head(), str(sub_path), sub.shape

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2711322098.py in <cell line: 0>()
      1 # Ensure submission order matches sample_submission ids (safest for Kaggle format)
      2 pred_df = pd.DataFrame(
----> 3     {"id": test_df["id"].values, "has_cactus": preds.astype(np.float64)}
      4 )
      5 

NameError: name 'preds' is not defined
