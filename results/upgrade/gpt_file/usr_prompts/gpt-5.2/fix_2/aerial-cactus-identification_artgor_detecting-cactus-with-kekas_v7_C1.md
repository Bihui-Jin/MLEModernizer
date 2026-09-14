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

DATA_ROOT = "/kaggle/input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train", "train")
TEST_DIR = os.path.join(DATA_ROOT, "test", "test")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"



## === cell 1
labels = pd.read_csv(TRAIN_CSV)
labels["has_cactus"] = labels["has_cactus"].astype(int)

sample_sub = pd.read_csv(SAMPLE_SUB)
test_df = sample_sub[["id"]].copy()
test_df["has_cactus"] = -1
test_df["data_type"] = "test"

labels["data_type"] = "train"

train_df, valid_df = train_test_split(
    labels,
    stratify=labels["has_cactus"],
    test_size=0.2,
    random_state=SEED,
)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.head()




## === cell 2
def _read_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _preprocess_to_tensor(
    img_rgb: np.ndarray, size: int = 32, hflip: bool = False
) -> torch.Tensor:
    if img_rgb.shape[0] != size or img_rgb.shape[1] != size:
        img_rgb = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    if hflip:
        img_rgb = np.ascontiguousarray(img_rgb[:, ::-1, :])
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = (x - mean) / std
    return x


class CactusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = row["id"]
        if row["data_type"] == "train":
            img_path = os.path.join(TRAIN_DIR, img_id)
        else:
            img_path = os.path.join(TEST_DIR, img_id)

        img = _read_image_rgb(img_path)
        do_hflip = self.train and (random.random() < 0.5)
        x = _preprocess_to_tensor(img, size=32, hflip=do_hflip)

        y = torch.tensor([float(row["has_cactus"])], dtype=torch.float32)
        return {"image": x, "label": y, "id": img_id}




## === cell 3
batch_size = 64
workers = 2  # safe default on Kaggle, improves throughput without changing semantics

train_ds = CactusDataset(train_df, train=True)
val_ds = CactusDataset(valid_df, train=False)
test_ds = CactusDataset(test_df, train=False)

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    pin_memory=True,
    drop_last=True,
)
val_dl = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
)
test_dl = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
)

len(train_dl), len(val_dl), len(test_dl)




## === cell 4
class Net(nn.Module):
    def __init__(self, num_classes: int = 1, p: float = 0.2):
        super().__init__()
        backbone = torchvision.models.densenet169(
            weights=torchvision.models.DenseNet169_Weights.IMAGENET1K_V1
        )
        num_ftrs = backbone.classifier.in_features
        backbone.classifier = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(p),
            nn.Linear(num_ftrs, num_classes),
        )
        self.net = backbone

    def forward(self, x):
        return self.net(x)


model = Net(num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)




## === cell 5
def evaluate_auc(model: nn.Module, loader: DataLoader):
    model.eval()
    all_targets = []
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            logits = model(x)
            all_targets.append(y.detach().cpu().numpy().reshape(-1))
            all_logits.append(logits.detach().cpu().numpy().reshape(-1))
    all_targets = np.concatenate(all_targets)
    all_logits = np.concatenate(all_logits)
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    auc = roc_auc_score(all_targets, probs)
    acc = accuracy_score(all_targets.astype(int), (probs > 0.5).astype(int))
    return auc, acc




## === cell 6
def train_epochs(model, train_loader, val_loader, optimizer, epochs: int, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_auc, val_acc = evaluate_auc(model, val_loader)
        print(
            f"epoch={epoch}/{epochs} lr={lr:.0e} train_loss={running_loss/len(train_loader):.4f} val_auc={val_auc:.6f} val_acc={val_acc:.4f}"
        )


train_epochs(model, train_dl, val_dl, optimizer, epochs=5, lr=1e-2)

train_epochs(model, train_dl, val_dl, optimizer, epochs=3, lr=1e-3)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4184933562.py in <cell line: 0>()
     28 
     29 # Stage 1: lr=1e-2 for 5 epochs (matches original call)
---> 30 train_epochs(model, train_dl, val_dl, optimizer, epochs=5, lr=1e-2)
     31 
     32 # Stage 2: lr=1e-3 for 3 epochs (matches original call)

/tmp/ipykernel_11/4184933562.py in train_epochs(model, train_loader, val_loader, optimizer, epochs, lr)
      9         running_loss = 0.0
     10 
---> 11         for batch in train_loader:
     12             x = batch["image"].to(device, non_blocking=True)
     13             y = batch["label"].to(device, non_blocking=True)

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
  File "/tmp/ipykernel_11/1079092216.py", line 42, in __getitem__
    img = _read_image_rgb(img_path)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/1079092216.py", line 4, in _read_image_rgb
    raise FileNotFoundError(f"Failed to read image: {path}")
FileNotFoundError: Failed to read image: /kaggle/input/aerial-cactus-identification/train/train/cf8ff69595fcd40b29f26df3477b11d2.jpg


## === cell 7
model.eval()
test_probs = []
test_ids = []
with torch.no_grad():
    for batch in test_dl:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x).detach().cpu().numpy().reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        test_probs.append(probs)
        test_ids.extend(batch["id"])

test_probs = np.concatenate(test_probs)

sub = sample_sub.copy()
pred_map = dict(zip(test_ids, test_probs))
sub["has_cactus"] = sub["id"].map(pred_map).astype(float)

sub["has_cactus"] = sub["has_cactus"].fillna(0.5)

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
sub.head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1236977543.py in <cell line: 0>()
      4 test_ids = []
      5 with torch.no_grad():
----> 6     for batch in test_dl:
      7         x = batch["image"].to(device, non_blocking=True)
      8         logits = model(x).detach().cpu().numpy().reshape(-1)

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
  File "/tmp/ipykernel_11/1079092216.py", line 42, in __getitem__
    img = _read_image_rgb(img_path)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/1079092216.py", line 4, in _read_image_rgb
    raise FileNotFoundError(f"Failed to read image: {path}")
FileNotFoundError: Failed to read image: /kaggle/input/aerial-cactus-identification/test/test/09034a34de0e2015a8a28dfe18f423f6.jpg


## === cell 8
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["id", "has_cactus"]
assert len(chk) == len(sample_sub)
assert chk["has_cactus"].between(0, 1).all()
print("Wrote submission.csv with shape:", chk.shape)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/4223312736.py in <cell line: 0>()
      1 # Quick sanity checks for submission validity
----> 2 assert os.path.exists("submission.csv")
      3 chk = pd.read_csv("submission.csv")
      4 assert list(chk.columns) == ["id", "has_cactus"]
      5 assert len(chk) == len(sample_sub)

AssertionError:
