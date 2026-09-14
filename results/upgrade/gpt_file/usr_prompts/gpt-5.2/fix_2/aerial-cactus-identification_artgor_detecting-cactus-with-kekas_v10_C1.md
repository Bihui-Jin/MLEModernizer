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

0.9997

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
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_PATH = "../input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train", "train")
TEST_DIR = os.path.join(BASE_PATH, "test", "test")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"



## === cell 1
import albumentations as A



def augs(p=0.5):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.10,
                rotate_limit=15,
                p=0.75,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
            A.HueSaturationValue(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ],
        p=p,
    )


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_to_tensor(img_rgb_uint8: np.ndarray, size: int = 32) -> torch.Tensor:
    img = cv2.resize(img_rgb_uint8, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()




## === cell 2
labels = pd.read_csv(TRAIN_CSV)
labels["has_cactus"] = labels["has_cactus"].astype(int)
labels["data_type"] = "train"

sample_sub = pd.read_csv(SAMPLE_SUB)
test_df = sample_sub[["id"]].copy()
test_df["has_cactus"] = -1
test_df["data_type"] = "test"

train, valid = train_test_split(
    labels, stratify=labels.has_cactus, test_size=0.2, random_state=SEED
)

train.head(), valid.head(), test_df.head()




## === cell 3
def reader_fn(i, row):
    if row["data_type"] == "train":
        pth = os.path.join(TRAIN_DIR, row["id"])
    else:
        pth = os.path.join(TEST_DIR, row["id"])
    image_bgr = cv2.imread(pth)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {pth}")
    image = image_bgr[:, :, ::-1]  # BGR -> RGB
    label = torch.tensor([float(row["has_cactus"])], dtype=torch.float32)
    return {"image": image, "label": label}


class CactusDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, is_train: bool, size: int = 32, aug_p: float = 0.5
    ):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.size = size
        self.aug = augs(p=aug_p) if is_train else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        item = reader_fn(idx, row)
        img = item["image"]
        if self.is_train and self.aug is not None:
            img = self.aug(image=img)["image"]
        x = preprocess_to_tensor(img, size=self.size)
        y = item["label"]
        return {"image": x, "label": y, "id": row["id"]}


batch_size = 64
workers = 2

train_ds = CactusDataset(train, is_train=True, size=32, aug_p=0.5)
val_ds = CactusDataset(valid, is_train=False, size=32, aug_p=0.0)
test_ds = CactusDataset(test_df, is_train=False, size=32, aug_p=0.0)

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
)
val_dl = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    pin_memory=torch.cuda.is_available(),
)
test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    pin_memory=torch.cuda.is_available(),
)



## === cell 4


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


class Net(nn.Module):
    def __init__(
        self,
        num_classes: int,
        p: float = 0.2,
        last_conv_size: int = 1664,
    ) -> None:
        super().__init__()
        backbone = torchvision.models.densenet169(
            weights=torchvision.models.DenseNet169_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            Flatten(),
            nn.BatchNorm1d(last_conv_size),
            nn.Dropout(p),
            nn.Linear(last_conv_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.head(x)
        return x


model = Net(num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()


def step_fn(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    inp = batch["image"].to(device, non_blocking=True)
    return model(inp)


def bce_accuracy(
    target: torch.Tensor, preds: torch.Tensor, thresh: float = 0.5
) -> float:
    target_np = target.detach().cpu().numpy()
    preds_np = (torch.sigmoid(preds).detach().cpu().numpy() > thresh).astype(int)
    return accuracy_score(target_np, preds_np)


def roc_auc(target: torch.Tensor, preds: torch.Tensor) -> float:
    target_np = target.detach().cpu().numpy()
    preds_np = torch.sigmoid(preds).detach().cpu().numpy()
    return roc_auc_score(target_np, preds_np)




## === cell 5

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)


def run_one_cycle(
    max_lr: float, cycle_len: int, div_factor: float, increase_fraction: float
):
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * cycle_len
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=increase_fraction,
        div_factor=div_factor,
        final_div_factor=div_factor,
        anneal_strategy="cos",
    )

    for epoch in range(cycle_len):
        model.train()
        tr_losses = []
        for batch in train_dl:
            optimizer.zero_grad(set_to_none=True)
            logits = step_fn(model, batch)
            target = batch["label"].to(device, non_blocking=True)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        all_t, all_p = [], []
        with torch.no_grad():
            for batch in val_dl:
                logits = step_fn(model, batch)
                target = batch["label"].to(device, non_blocking=True)
                loss = criterion(logits, target)
                va_losses.append(loss.item())
                all_t.append(target)
                all_p.append(logits)
        all_t = torch.cat(all_t, dim=0)
        all_p = torch.cat(all_p, dim=0)
        auc = roc_auc(all_t, all_p)
        acc = bce_accuracy(all_t, all_p)

        print(
            f"epoch {epoch+1}/{cycle_len} | train_loss {np.mean(tr_losses):.4f} | val_loss {np.mean(va_losses):.4f} | val_auc {auc:.5f} | val_acc {acc:.5f}"
        )


run_one_cycle(max_lr=1e-2, cycle_len=5, div_factor=25, increase_fraction=0.3)
run_one_cycle(max_lr=1e-3, cycle_len=3, div_factor=25, increase_fraction=0.2)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2013662794.py in <cell line: 0>()
     58 # Unfreeze/freeze semantics from kekas were used, but not essential for correctness here; keep all trainable.
     59 # Two one-cycle phases as in original notebook:
---> 60 run_one_cycle(max_lr=1e-2, cycle_len=5, div_factor=25, increase_fraction=0.3)
     61 run_one_cycle(max_lr=1e-3, cycle_len=3, div_factor=25, increase_fraction=0.2)
     62 

/tmp/ipykernel_11/2013662794.py in run_one_cycle(max_lr, cycle_len, div_factor, increase_fraction)
     24         model.train()
     25         tr_losses = []
---> 26         for batch in train_dl:
     27             optimizer.zero_grad(set_to_none=True)
     28             logits = step_fn(model, batch)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_11/197951066.py", line 29, in __getitem__
    item = reader_fn(idx, row)
           ^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/197951066.py", line 9, in reader_fn
    raise FileNotFoundError(f"Could not read image: {pth}")
FileNotFoundError: Could not read image: ../input/aerial-cactus-identification/train/train/cf8ff69595fcd40b29f26df3477b11d2.jpg


## === cell 6
model.eval()
pred_list = []
id_list = []
with torch.no_grad():
    for batch in test_dl:
        logits = step_fn(model, batch)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        pred_list.append(probs)
        id_list.extend(batch["id"])

preds = np.concatenate(pred_list, axis=0)

sub = sample_sub.copy()
pred_map = {i: p for i, p in zip(id_list, preds)}
sub["has_cactus"] = sub["id"].map(pred_map).astype(float)

assert sub.shape[0] == sample_sub.shape[0]
assert sub["has_cactus"].between(0.0, 1.0).all()

sub.to_csv("sub.csv", index=False)
sub.head()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2718929536.py in <cell line: 0>()
      4 id_list = []
      5 with torch.no_grad():
----> 6     for batch in test_dl:
      7         logits = step_fn(model, batch)
      8         probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_11/197951066.py", line 29, in __getitem__
    item = reader_fn(idx, row)
           ^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/197951066.py", line 9, in reader_fn
    raise FileNotFoundError(f"Could not read image: {pth}")
FileNotFoundError: Could not read image: ../input/aerial-cactus-identification/test/test/09034a34de0e2015a8a28dfe18f423f6.jpg
