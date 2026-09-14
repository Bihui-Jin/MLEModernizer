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

0.7758

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
import numpy as np
import pandas as pd

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "../input/aerial-cactus-identification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train", "train")
TEST_DIR = os.path.join(DATA_ROOT, "test", "test")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"



## === cell 1
labels = pd.read_csv(TRAIN_CSV)
labels["has_cactus"] = labels["has_cactus"].astype(np.float32)

sample_sub = pd.read_csv(SAMPLE_SUB)
test_df = sample_sub[["id"]].copy()
test_df["has_cactus"] = -1.0

print("train rows:", len(labels), "test rows:", len(test_df))
print(labels.head())
print(test_df.head())




## === cell 2
class CactusDataset(Dataset):
    def __init__(self, df, img_dir, train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.train = train

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        if train:
            self.tfms = T.Compose(
                [
                    T.ToPILImage(),
                    T.Resize((32, 32)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std),
                ]
            )
        else:
            self.tfms = T.Compose(
                [
                    T.ToPILImage(),
                    T.Resize((32, 32)),
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["id"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tfms(img)

        y = torch.tensor([float(row["has_cactus"])], dtype=torch.float32)
        return {"image": img, "label": y, "id": row["id"]}




## === cell 3
train_df, valid_df = train_test_split(
    labels, test_size=0.2, stratify=labels["has_cactus"], random_state=42
)

batch_size = 128
num_workers = 2

train_ds = CactusDataset(train_df, TRAIN_DIR, train=True)
valid_ds = CactusDataset(valid_df, TRAIN_DIR, train=False)
test_ds = CactusDataset(test_df, TEST_DIR, train=False)

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
valid_dl = DataLoader(
    valid_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)
test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

len(train_ds), len(valid_ds), len(test_ds)




## === cell 4
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=2):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d((size, size))
        self.mp = nn.AdaptiveMaxPool2d((size, size))

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
        self.features = backbone.features  # conv trunk
        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            AdaptiveConcatPool2d(size=pooling_size),
            Flatten(),
            nn.BatchNorm1d(2 * 1664 * pooling_size * pooling_size),
            nn.Dropout(p),
            nn.Linear(2 * 1664 * pooling_size * pooling_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


model = Net(num_classes=1, p=0.2, pooling_size=2).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)

print(model.__class__.__name__)




## === cell 5
def run_one_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    all_targets = []
    all_logits = []
    running_loss = 0.0
    n = 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        running_loss += loss.item() * bs
        n += bs

        all_targets.append(y.detach().cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())

    all_targets = np.concatenate(all_targets).reshape(-1)
    all_logits = np.concatenate(all_logits).reshape(-1)
    all_probs = 1 / (1 + np.exp(-all_logits))

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = float("nan")

    return running_loss / max(n, 1), auc




## === cell 6
epochs = 3
for epoch in range(1, epochs + 1):
    t0 = time.time()
    train_loss, train_auc = run_one_epoch(model, train_dl, optimizer=optimizer)
    valid_loss, valid_auc = run_one_epoch(model, valid_dl, optimizer=None)
    dt = time.time() - t0
    print(
        f"Epoch {epoch}/{epochs} | {dt:.1f}s | train loss {train_loss:.4f} auc {train_auc:.4f} | valid loss {valid_loss:.4f} auc {valid_auc:.4f}"
    )




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2405936422.py in <cell line: 0>()
      3 for epoch in range(1, epochs + 1):
      4     t0 = time.time()
----> 5     train_loss, train_auc = run_one_epoch(model, train_dl, optimizer=optimizer)
      6     valid_loss, valid_auc = run_one_epoch(model, valid_dl, optimizer=None)
      7     dt = time.time() - t0

/tmp/ipykernel_11/4118047424.py in run_one_epoch(model, loader, optimizer)
      9     n = 0
     10 
---> 11     for batch in loader:
     12         x = batch["image"].to(device, non_blocking=True)
     13         y = batch["label"].to(device, non_blocking=True)

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
  File "/tmp/ipykernel_11/1200557842.py", line 42, in __getitem__
    raise FileNotFoundError(f"Could not read image: {img_path}")
FileNotFoundError: Could not read image: ../input/aerial-cactus-identification/train/train/7dfdfbe16c392bd600291159b8dc81ce.jpg


## === cell 7
@torch.no_grad()
def predict_probs(model, loader, tta_flip=False):
    model.eval()
    probs = []
    ids = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x)

        if tta_flip:
            x_flip = torch.flip(x, dims=[3])  # horizontal flip (W dimension)
            logits_flip = model(x_flip)
            logits = 0.5 * (logits + logits_flip)

        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs.append(p)
        ids.extend(batch["id"])
    probs = np.concatenate(probs)
    return ids, probs


test_ids, test_probs = predict_probs(model, test_dl, tta_flip=True)
print(len(test_ids), len(test_probs), test_ids[:3], test_probs[:3])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2278518621.py in <cell line: 0>()
     21 
     22 
---> 23 test_ids, test_probs = predict_probs(model, test_dl, tta_flip=True)
     24 print(len(test_ids), len(test_probs), test_ids[:3], test_probs[:3])
     25 

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_11/2278518621.py in predict_probs(model, loader, tta_flip)
      5     probs = []
      6     ids = []
----> 7     for batch in loader:
      8         x = batch["image"].to(device, non_blocking=True)
      9         logits = model(x)

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
  File "/tmp/ipykernel_11/1200557842.py", line 42, in __getitem__
    raise FileNotFoundError(f"Could not read image: {img_path}")
FileNotFoundError: Could not read image: ../input/aerial-cactus-identification/test/test/09034a34de0e2015a8a28dfe18f423f6.jpg


## === cell 8
pred_map = dict(zip(test_ids, test_probs))
submission = sample_sub.copy()
submission["has_cactus"] = submission["id"].map(pred_map).astype(np.float32)

missing = submission["has_cactus"].isna().sum()
if missing:
    submission["has_cactus"] = submission["has_cactus"].fillna(0.5)

sub_path = "submission.csv"
submission.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", submission.shape, "missing filled:", int(missing))
print(submission.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4183405498.py in <cell line: 0>()
      1 # Create submission in exactly the required order/row count using sample_submission as template.
----> 2 pred_map = dict(zip(test_ids, test_probs))
      3 submission = sample_sub.copy()
      4 submission["has_cactus"] = submission["id"].map(pred_map).astype(np.float32)
      5 

NameError: name 'test_ids' is not defined
