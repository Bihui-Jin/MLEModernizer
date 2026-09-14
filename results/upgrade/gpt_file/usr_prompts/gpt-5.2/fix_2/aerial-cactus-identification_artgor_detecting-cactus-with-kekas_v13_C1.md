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
import time
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as tvt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 1
import albumentations as A

BASE = "/kaggle/input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing: {TEST_DIR}"



## === cell 2
labels = pd.read_csv(TRAIN_CSV)

try:
    fig = plt.figure(figsize=(25, 8))
    train_imgs = os.listdir(TRAIN_DIR)
    for idx, img in enumerate(np.random.choice(train_imgs, 20, replace=False)):
        ax = fig.add_subplot(4, 20 // 4, idx + 1, xticks=[], yticks=[])
        im = Image.open(os.path.join(TRAIN_DIR, img))
        plt.imshow(im)
        lab = labels.loc[labels["id"] == img, "has_cactus"].values[0]
        ax.set_title(f"Label: {lab}")
    plt.show()
except Exception as e:
    print("Plotting skipped:", repr(e))



## === cell 3
test_img = sorted(os.listdir(TEST_DIR))
test_df = pd.DataFrame(test_img, columns=["id"])
test_df["has_cactus"] = -1
test_df["data_type"] = "test"

labels["has_cactus"] = labels["has_cactus"].astype(int)
labels["data_type"] = "train"

labels.head()



## === cell 4
labels.loc[labels["data_type"] == "train", "has_cactus"].value_counts()



## === cell 5
train, valid = train_test_split(
    labels, stratify=labels.has_cactus, test_size=0.2, random_state=SEED
)
train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)




## === cell 6
def reader_fn(i, row):
    if row["data_type"] == "train":
        p = os.path.join(TRAIN_DIR, row["id"])
    else:
        p = os.path.join(TEST_DIR, row["id"])
    image = cv2.imread(p)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {p}")
    image = image[:, :, ::-1]  # BGR -> RGB
    label = torch.tensor([float(row["has_cactus"])], dtype=torch.float32)
    return {"image": image, "label": label}




## === cell 7
def augs(p=0.5):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ],
        p=p,
    )




## === cell 8


class Transformer:
    def __init__(self, key, fn):
        self.key = key
        self.fn = fn

    def __call__(self, sample):
        sample[self.key] = self.fn(sample[self.key])
        return sample


def to_torch():
    def _fn(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.transpose(2, 0, 1)).float().div(255.0)
        elif torch.is_tensor(x):
            x = x.float()
        else:
            raise TypeError(f"Unsupported type for to_torch: {type(x)}")
        return x

    return _fn


def normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    def _fn(x):
        return (x - mean_t.to(x.device)) / std_t.to(x.device)

    return _fn


class Compose:
    def __init__(self, tfms):
        self.tfms = tfms

    def __call__(self, sample):
        for t in self.tfms:
            sample = t(sample)
        return sample


def get_transforms(dataset_key, size, p):
    PRE_TFMS = Transformer(
        dataset_key, lambda x: cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    )
    AUGS = Transformer(dataset_key, lambda x: augs(p=p)(image=x)["image"])
    NRM_TFMS = Compose(
        [
            Transformer(dataset_key, to_torch()),
            Transformer(dataset_key, normalize()),
        ]
    )

    train_tfms = Compose([PRE_TFMS, AUGS, NRM_TFMS])
    val_tfms = Compose([PRE_TFMS, NRM_TFMS])
    return train_tfms, val_tfms




## === cell 9
train_tfms, val_tfms = get_transforms("image", 32, 0.5)




## === cell 10
class DataKek(Dataset):
    def __init__(self, df, reader_fn, transforms=None):
        self.df = df.reset_index(drop=True)
        self.reader_fn = reader_fn
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i].to_dict()
        sample = self.reader_fn(i, row)
        if self.transforms is not None:
            sample = self.transforms(sample)
        if not torch.is_tensor(sample["label"]):
            sample["label"] = torch.tensor(sample["label"], dtype=torch.float32)
        return sample


def collate_dict(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {"image": images, "label": labels}


train_dk = DataKek(df=train, reader_fn=reader_fn, transforms=train_tfms)
val_dk = DataKek(df=valid, reader_fn=reader_fn, transforms=val_tfms)

batch_size = 64
workers = 0

train_dl = DataLoader(
    train_dk,
    batch_size=batch_size,
    num_workers=workers,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_dict,
)
val_dl = DataLoader(
    val_dk,
    batch_size=batch_size,
    num_workers=workers,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_dict,
)



## === cell 11
test_dk = DataKek(df=test_df, reader_fn=reader_fn, transforms=val_tfms)
test_dl = DataLoader(
    test_dk,
    batch_size=batch_size,
    num_workers=workers,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_dict,
)




## === cell 12
class Net(nn.Module):
    def __init__(self, num_classes: int, p: float = 0.2):
        super().__init__()
        backbone = torchvision.models.densenet169(
            weights=torchvision.models.DenseNet169_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(1664),
            nn.Dropout(p),
            nn.Linear(1664, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        x = self.head(x)
        return x




## === cell 13
model = Net(num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()




## === cell 14
def step_fn(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    inp = batch["image"].to(device)
    return model(inp)




## === cell 15
def bce_accuracy(
    target: torch.Tensor, preds: torch.Tensor, thresh: float = 0.5
) -> float:
    target = target.detach().cpu().numpy().reshape(-1)
    preds = (torch.sigmoid(preds).detach().cpu().numpy().reshape(-1) > thresh).astype(
        int
    )
    return accuracy_score(target, preds)


def roc_auc(target: torch.Tensor, preds: torch.Tensor) -> float:
    target = target.detach().cpu().numpy().reshape(-1)
    preds = torch.sigmoid(preds).detach().cpu().numpy().reshape(-1)
    try:
        return roc_auc_score(target, preds)
    except Exception:
        return float("nan")




## === cell 16
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)

epochs_stage1 = 5
steps_per_epoch = len(train_dl)
scheduler1 = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-2,
    epochs=epochs_stage1,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1e4,
)


def run_epoch(train_mode=True, scheduler=None):
    model.train(train_mode)
    total_loss = 0.0
    all_preds = []
    all_targs = []
    dl = train_dl if train_mode else val_dl

    for batch in dl:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if train_mode:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * x.size(0)
        all_preds.append(logits.detach().cpu())
        all_targs.append(y.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targs = torch.cat(all_targs, dim=0)
    avg_loss = total_loss / len(dl.dataset)
    acc = bce_accuracy(all_targs, all_preds)
    auc = roc_auc(all_targs, all_preds)
    return avg_loss, acc, auc




## === cell 17
t0 = time.time()
for ep in range(1, epochs_stage1 + 1):
    tr_loss, tr_acc, tr_auc = run_epoch(train_mode=True, scheduler=scheduler1)
    va_loss, va_acc, va_auc = run_epoch(train_mode=False, scheduler=None)
    print(
        f"[Stage1][{ep}/{epochs_stage1}] "
        f"train loss {tr_loss:.4f} acc {tr_acc:.4f} auc {tr_auc:.6f} | "
        f"val loss {va_loss:.4f} acc {va_acc:.4f} auc {va_auc:.6f}"
    )
print("Stage1 time:", round(time.time() - t0, 1), "s")



## === cell 18
epochs_stage2 = 5
scheduler2 = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=epochs_stage2,
    steps_per_epoch=len(train_dl),
    pct_start=0.2,
    div_factor=25.0,
    final_div_factor=1e4,
)

t0 = time.time()
for ep in range(1, epochs_stage2 + 1):
    tr_loss, tr_acc, tr_auc = run_epoch(train_mode=True, scheduler=scheduler2)
    va_loss, va_acc, va_auc = run_epoch(train_mode=False, scheduler=None)
    print(
        f"[Stage2][{ep}/{epochs_stage2}] "
        f"train loss {tr_loss:.4f} acc {tr_acc:.4f} auc {tr_auc:.6f} | "
        f"val loss {va_loss:.4f} acc {va_acc:.4f} auc {va_auc:.6f}"
    )
print("Stage2 time:", round(time.time() - t0, 1), "s")



## === cell 19
model.eval()
pred_list = []
with torch.no_grad():
    for batch in test_dl:
        x = batch["image"].to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        pred_list.append(probs)

preds = np.concatenate(pred_list, axis=0)
print("preds shape:", preds.shape, "min/max:", float(preds.min()), float(preds.max()))



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1945002401.py in <cell line: 0>()
      3 pred_list = []
      4 with torch.no_grad():
----> 5     for batch in test_dl:
      6         x = batch["image"].to(device)
      7         logits = model(x)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_11/4087837050.py in __getitem__(self, i)
     10     def __getitem__(self, i):
     11         row = self.df.iloc[i].to_dict()
---> 12         sample = self.reader_fn(i, row)
     13         if self.transforms is not None:
     14             sample = self.transforms(sample)

/tmp/ipykernel_11/2559441400.py in reader_fn(i, row)
      7     image = cv2.imread(p)
      8     if image is None:
----> 9         raise FileNotFoundError(f"Could not read image: {p}")
     10     image = image[:, :, ::-1]  # BGR -> RGB
     11     label = torch.tensor([float(row["has_cactus"])], dtype=torch.float32)

FileNotFoundError: Could not read image: /kaggle/input/aerial-cactus-identification/test/test

## === cell 20
sub = pd.read_csv(SAMPLE_SUB)
pred_map = dict(zip(test_df["id"].values, preds))
sub["has_cactus"] = sub["id"].map(pred_map).astype(float)

if sub["has_cactus"].isna().any():
    sub["has_cactus"] = sub["has_cactus"].fillna(float(np.mean(preds)))

sub.to_csv("sub.csv", index=False)
sub.head()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3767051576.py in <cell line: 0>()
      1 # Ensure submission ordering matches sample_submission.csv (safer than relying on directory order)
      2 sub = pd.read_csv(SAMPLE_SUB)
----> 3 pred_map = dict(zip(test_df["id"].values, preds))
      4 sub["has_cactus"] = sub["id"].map(pred_map).astype(float)
      5 

NameError: name 'preds' is not defined

## === cell 21
assert os.path.exists("sub.csv"), "sub.csv was not created"
check = pd.read_csv("sub.csv")
assert list(check.columns) == ["id", "has_cactus"], f"Wrong columns: {check.columns}"
assert len(check) == len(
    pd.read_csv(SAMPLE_SUB)
), "Row count mismatch vs sample_submission"
assert (
    check["has_cactus"].between(0, 1).all()
), "Predictions must be probabilities in [0,1]"
print("Submission ready:", check.shape)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/2746367054.py in <cell line: 0>()
      1 # Basic sanity checks for Kaggle submission validity
----> 2 assert os.path.exists("sub.csv"), "sub.csv was not created"
      3 check = pd.read_csv("sub.csv")
      4 assert list(check.columns) == ["id", "has_cactus"], f"Wrong columns: {check.columns}"
      5 assert len(check) == len(

AssertionError: sub.csv was not created
