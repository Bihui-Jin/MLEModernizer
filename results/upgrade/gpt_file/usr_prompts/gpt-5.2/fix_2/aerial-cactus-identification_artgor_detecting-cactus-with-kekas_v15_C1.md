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
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF

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

BASE = "/kaggle/input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing: {TEST_DIR}"

print("Device:", device)
print("Train images:", len(os.listdir(TRAIN_DIR)))
print("Test images:", len(os.listdir(TEST_DIR)))



## === cell 1


def read_image_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = img[:, :, ::-1]  # BGR -> RGB
    return img


def resize32(img, size=32):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def augment_image(img, p=0.5):
    if random.random() < p:
        img = np.ascontiguousarray(img[:, ::-1, :])  # horizontal flip
    if random.random() < p:
        img = np.ascontiguousarray(img[::-1, :, :])  # vertical flip
    if random.random() < p:
        delta = random.uniform(-32, 32)
        img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    return img


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_torch_normalized(img):
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x




## === cell 2
labels = pd.read_csv(TRAIN_CSV)
labels["has_cactus"] = labels["has_cactus"].astype(int)
labels["data_type"] = "train"

test_img = sorted(os.listdir(TEST_DIR))
test_df = pd.DataFrame({"id": test_img})
test_df["has_cactus"] = -1
test_df["data_type"] = "test"

train, valid = train_test_split(
    labels, stratify=labels.has_cactus, test_size=0.2, random_state=SEED
)

print(labels.head())
print("Train split:", train.shape, "Valid split:", valid.shape, "Test:", test_df.shape)




## === cell 3
class CactusDataset(Dataset):
    def __init__(self, df, train_mode=True, img_size=32, aug_p=0.5):
        self.df = df.reset_index(drop=True)
        self.train_mode = train_mode
        self.img_size = img_size
        self.aug_p = aug_p

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]
        data_type = row["data_type"]
        if data_type == "train":
            path = os.path.join(TRAIN_DIR, img_id)
        else:
            path = os.path.join(TEST_DIR, img_id)

        img = read_image_rgb(path)
        img = resize32(img, self.img_size)
        if self.train_mode:
            img = augment_image(img, p=self.aug_p)
        x = to_torch_normalized(img)

        y = torch.tensor([float(row["has_cactus"])], dtype=torch.float32)
        return {"image": x, "label": y, "id": img_id}


batch_size = 64
workers = 2  # safe in Kaggle

train_ds = CactusDataset(train, train_mode=True, img_size=32, aug_p=0.5)
val_ds = CactusDataset(valid, train_mode=False, img_size=32, aug_p=0.0)
test_ds = CactusDataset(test_df, train_mode=False, img_size=32, aug_p=0.0)

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
class Net(nn.Module):
    def __init__(self, num_classes: int = 1, p: float = 0.2):
        super().__init__()
        backbone = torchvision.models.densenet169(
            weights=torchvision.models.DenseNet169_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features  # conv feature extractor
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(1664),
            nn.Dropout(p),
            nn.Linear(1664, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.head(x)
        return x


model = Net(num_classes=1, p=0.2).to(device)
criterion = nn.BCEWithLogitsLoss()




## === cell 5
def bce_accuracy(
    target: torch.Tensor, logits: torch.Tensor, thresh: float = 0.5
) -> float:
    target_np = target.detach().cpu().numpy().astype(int).reshape(-1)
    preds_np = (
        torch.sigmoid(logits).detach().cpu().numpy().reshape(-1) > thresh
    ).astype(int)
    return accuracy_score(target_np, preds_np)


def roc_auc(target: torch.Tensor, logits: torch.Tensor) -> float:
    target_np = target.detach().cpu().numpy().reshape(-1)
    preds_np = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    if len(np.unique(target_np)) < 2:
        return np.nan
    return roc_auc_score(target_np, preds_np)




## === cell 6
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)


def run_one_cycle(max_lr, cycle_len, div_factor=25.0):
    min_lr = max_lr / div_factor
    increase_fraction = 0.3 if max_lr >= 1e-2 else 0.2
    total_steps = cycle_len * len(train_dl)
    up_steps = max(1, int(total_steps * increase_fraction))
    down_steps = max(1, total_steps - up_steps)

    lrs = np.concatenate(
        [
            np.linspace(min_lr, max_lr, up_steps, endpoint=True),
            np.linspace(max_lr, min_lr, down_steps, endpoint=True),
        ]
    )

    global_step = 0
    for epoch in range(cycle_len):
        model.train()
        t0 = time.time()
        losses = []
        for batch in train_dl:
            lr = float(lrs[min(global_step, len(lrs) - 1)])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            global_step += 1

        model.eval()
        val_losses = []
        all_y = []
        all_logits = []
        with torch.no_grad():
            for batch in val_dl:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())
                all_y.append(y)
                all_logits.append(logits)

        all_y = torch.cat(all_y, dim=0)
        all_logits = torch.cat(all_logits, dim=0)
        auc = roc_auc(all_y, all_logits)
        acc = bce_accuracy(all_y, all_logits)

        dt = time.time() - t0
        print(
            f"epoch {epoch+1}/{cycle_len} - "
            f"train_loss {np.mean(losses):.4f} - val_loss {np.mean(val_losses):.4f} - "
            f"val_acc {acc:.4f} - val_auc {auc if not np.isnan(auc) else -1:.4f} - "
            f"time {dt:.1f}s"
        )




## === cell 7
run_one_cycle(max_lr=1e-2, cycle_len=5, div_factor=25.0)
run_one_cycle(max_lr=1e-3, cycle_len=5, div_factor=25.0)



## === cell 8
model.eval()
preds = []
ids = []
with torch.no_grad():
    for batch in test_dl:
        x = batch["image"].to(device, non_blocking=True)
        logits = model(x)
        prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        preds.append(prob)
        ids.extend(batch["id"])

preds = np.concatenate(preds, axis=0)
sub = pd.DataFrame({"id": ids, "has_cactus": preds.astype(np.float32)})

sample = pd.read_csv(SAMPLE_SUB)
sub = sample[["id"]].merge(sub, on="id", how="left")
assert sub["has_cactus"].isna().sum() == 0, "Missing predictions for some test ids."

sub.to_csv("sub.csv", index=False)
print(sub.head())
print("Wrote sub.csv with shape:", sub.shape)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/850031260.py in <cell line: 0>()
      4 ids = []
      5 with torch.no_grad():
----> 6     for batch in test_dl:
      7         x = batch["image"].to(device, non_blocking=True)
      8         logits = model(x)

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

FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 1.
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
  File "/tmp/ipykernel_11/1405631763.py", line 20, in __getitem__
    img = read_image_rgb(path)
          ^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/2071922652.py", line 15, in read_image_rgb
    raise FileNotFoundError(f"Cannot read image: {path}")
FileNotFoundError: Cannot read image: /kaggle/input/aerial-cactus-identification/test/test
