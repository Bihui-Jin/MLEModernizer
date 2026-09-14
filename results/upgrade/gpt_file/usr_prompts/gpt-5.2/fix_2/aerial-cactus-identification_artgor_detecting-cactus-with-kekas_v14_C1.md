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
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

import torchvision

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/kaggle/input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")




## === cell 1


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Transformer:
    """Applies a function to a specific key in a dict batch item."""

    def __init__(self, key, fn):
        self.key = key
        self.fn = fn

    def __call__(self, item):
        item[self.key] = self.fn(item[self.key])
        return item


def to_torch():
    def _to_torch(x):
        if isinstance(x, torch.Tensor):
            return x
        x = np.asarray(x)
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x)

    return _to_torch


def normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
    std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def _norm(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return (x - mean) / std

    return _norm


class ComposeDict:
    """Like torchvision.transforms.Compose but for dict items."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, item):
        for t in self.transforms:
            item = t(item)
        return item


class DataKek(Dataset):
    """Dataset that uses a dataframe and reader_fn returning dict(image,label)."""

    def __init__(self, df, reader_fn, transforms=None):
        self.df = df.reset_index(drop=True)
        self.reader_fn = reader_fn
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        item = self.reader_fn(i, row)
        if self.transforms is not None:
            item = self.transforms(item)
        return item


def collate_dict(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {"image": images, "label": labels}


class SimpleKeker:
    """A small stand-in for Keker: trains and predicts with the given step_fn/criterion."""

    def __init__(
        self,
        model,
        train_dl,
        val_dl,
        criterion,
        step_fn,
        target_key="label",
        metrics=None,
        opt=torch.optim.SGD,
        opt_params=None,
    ):
        self.model = model.to(device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.criterion = criterion
        self.step_fn = step_fn
        self.target_key = target_key
        self.metrics = metrics or {}
        self.opt_params = opt_params or {}
        self.optimizer = opt(self.model.parameters(), **self.opt_params)

    def unfreeze(self, model_attr=None):
        for p in self.model.parameters():
            p.requires_grad = True

    def freeze_to(self, layer_num=-1, model_attr=None):
        return

    def _run_epoch(self, train=True):
        dl = self.train_dl if train else self.val_dl
        self.model.train(mode=train)
        total_loss = 0.0

        all_targets = []
        all_logits = []

        for batch in dl:
            x = batch["image"].to(device)
            y = batch[self.target_key].to(device)  # shape [B,1]
            logits = self.step_fn(self.model, {"image": x, self.target_key: y})
            loss = self.criterion(logits, y)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            all_targets.append(y.detach().cpu())
            all_logits.append(logits.detach().cpu())

        all_targets = torch.cat(all_targets, dim=0)
        all_logits = torch.cat(all_logits, dim=0)

        out = {"loss": total_loss / len(dl.dataset)}
        for name, fn in self.metrics.items():
            out[name] = fn(all_targets, all_logits)
        return out

    def kek_one_cycle(
        self,
        max_lr=1e-2,
        cycle_len=5,
        momentum_range=(0.95, 0.85),
        div_factor=25,
        increase_fraction=0.3,
        logdir=None,
    ):
        min_lr = max_lr / div_factor
        steps_per_epoch = len(self.train_dl)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            epochs=cycle_len,
            steps_per_epoch=steps_per_epoch,
            pct_start=increase_fraction,
            anneal_strategy="cos",
            cycle_momentum=isinstance(self.optimizer, torch.optim.SGD),
            base_momentum=(
                momentum_range[0]
                if isinstance(self.optimizer, torch.optim.SGD)
                else 0.0
            ),
            max_momentum=(
                momentum_range[1]
                if isinstance(self.optimizer, torch.optim.SGD)
                else 0.0
            ),
            div_factor=div_factor,
            final_div_factor=div_factor,
        )

        history = []
        for epoch in range(cycle_len):
            t0 = time.time()
            train_stats = self._run_epoch(train=True)
            val_stats = self._run_epoch(train=False)
            history.append((train_stats, val_stats, time.time() - t0))

        history = []
        self.model.train(True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            epochs=cycle_len,
            steps_per_epoch=len(self.train_dl),
            pct_start=increase_fraction,
            anneal_strategy="cos",
            cycle_momentum=isinstance(self.optimizer, torch.optim.SGD),
            base_momentum=(
                momentum_range[0]
                if isinstance(self.optimizer, torch.optim.SGD)
                else 0.0
            ),
            max_momentum=(
                momentum_range[1]
                if isinstance(self.optimizer, torch.optim.SGD)
                else 0.0
            ),
            div_factor=div_factor,
            final_div_factor=div_factor,
        )

        for epoch in range(cycle_len):
            t0 = time.time()
            self.model.train(True)
            total_loss = 0.0
            all_targets = []
            all_logits = []
            for batch in self.train_dl:
                x = batch["image"].to(device)
                y = batch[self.target_key].to(device)
                logits = self.step_fn(self.model, {"image": x, self.target_key: y})
                loss = self.criterion(logits, y)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                total_loss += loss.item() * x.size(0)
                all_targets.append(y.detach().cpu())
                all_logits.append(logits.detach().cpu())

            all_targets = torch.cat(all_targets, dim=0)
            all_logits = torch.cat(all_logits, dim=0)
            train_stats = {"loss": total_loss / len(self.train_dl.dataset)}
            for name, fn in self.metrics.items():
                train_stats[name] = fn(all_targets, all_logits)

            val_stats = self._run_epoch(train=False)
            history.append((train_stats, val_stats, time.time() - t0))

        self.history_ = history
        return history

    @torch.no_grad()
    def predict_loader(self, loader):
        self.model.eval()
        preds = []
        for batch in loader:
            x = batch["image"].to(device)
            logits = self.model(x)
            preds.append(logits.detach().cpu().numpy())
        return np.concatenate(preds, axis=0)

    def plot_kek(self, logdir=None):
        return




## === cell 2
labels = pd.read_csv(TRAIN_CSV)
labels["has_cactus"] = labels["has_cactus"].astype(int)
labels["data_type"] = "train"

test_img = os.listdir(TEST_DIR)
test_df = pd.DataFrame(test_img, columns=["id"])
test_df["has_cactus"] = -1
test_df["data_type"] = "test"




## === cell 3
train, valid = train_test_split(
    labels, stratify=labels.has_cactus, test_size=0.2, random_state=SEED
)
train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)




## === cell 4
def reader_fn(i, row):
    img_path = os.path.join(BASE_DIR, row["data_type"], row["id"])
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    image = image[:, :, ::-1]  # BGR -> RGB
    label = torch.tensor([float(row["has_cactus"])], dtype=torch.float32)
    return {"image": image, "label": label}




## === cell 5
import albumentations as A


def augs(p=0.5):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ],
        p=p,
    )


def get_transforms(dataset_key, size, p):
    PRE_TFMS = Transformer(
        dataset_key,
        lambda x: cv2.resize(x, (size, size), interpolation=cv2.INTER_LINEAR),
    )
    AUGS = Transformer(dataset_key, lambda x: augs(p=p)(image=x)["image"])
    TO_T = Transformer(dataset_key, lambda x: to_torch()(x) / 255.0)
    NRM = Transformer(dataset_key, normalize())
    train_tfms = ComposeDict([PRE_TFMS, AUGS, TO_T, NRM])
    val_tfms = ComposeDict([PRE_TFMS, TO_T, NRM])
    return train_tfms, val_tfms


train_tfms, val_tfms = get_transforms("image", 32, 0.5)




## === cell 6
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

test_dk = DataKek(df=test_df, reader_fn=reader_fn, transforms=val_tfms)
test_dl = DataLoader(
    test_dk,
    batch_size=batch_size,
    num_workers=workers,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_dict,
)




## === cell 7


class Net(nn.Module):
    def __init__(self, num_classes: int, p: float = 0.2) -> None:
        super().__init__()
        weights = torchvision.models.DenseNet169_Weights.IMAGENET1K_V1
        backbone = torchvision.models.densenet169(weights=weights)
        num_ftrs = backbone.classifier.in_features  # 1664
        backbone.classifier = nn.Sequential(
            nn.BatchNorm1d(num_ftrs), nn.Dropout(p), nn.Linear(num_ftrs, num_classes)
        )
        self.net = backbone

    def forward(self, x):
        return self.net(x)


model = Net(num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()




## === cell 8
def step_fn(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    inp = batch["image"]
    return model(inp)


def bce_accuracy(
    target: torch.Tensor, preds: torch.Tensor, thresh: float = 0.5
) -> float:
    target_np = target.cpu().detach().numpy().reshape(-1)
    preds_np = (
        torch.sigmoid(preds).cpu().detach().numpy().reshape(-1) > thresh
    ).astype(int)
    return accuracy_score(target_np, preds_np)


def roc_auc(target: torch.Tensor, preds: torch.Tensor) -> float:
    target_np = target.cpu().detach().numpy().reshape(-1)
    preds_np = torch.sigmoid(preds).cpu().detach().numpy().reshape(-1)
    if len(np.unique(target_np)) < 2:
        return float("nan")
    return roc_auc_score(target_np, preds_np)




## === cell 9
keker = SimpleKeker(
    model=model,
    train_dl=train_dl,
    val_dl=val_dl,
    criterion=criterion,
    step_fn=step_fn,
    target_key="label",
    metrics={"acc": bce_accuracy, "auc": roc_auc},
    opt=torch.optim.SGD,
    opt_params={"lr": 1e-3, "momentum": 0.99, "weight_decay": 0.0},
)

keker.unfreeze(model_attr="net")
layer_num = -1
keker.freeze_to(layer_num, model_attr="net")




## === cell 10
_ = keker.kek_one_cycle(
    max_lr=1e-2,
    cycle_len=5,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.3,
    logdir="train_logs",
)
_ = keker.kek_one_cycle(
    max_lr=1e-3,
    cycle_len=5,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.2,
    logdir="train_logs1",
)




## === cell 11
preds_logits = keker.predict_loader(loader=test_dl)
preds = 1 / (1 + np.exp(-preds_logits.reshape(-1)))  # sigmoid to probability




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2745664586.py in <cell line: 0>()
----> 1 preds_logits = keker.predict_loader(loader=test_dl)
      2 preds = 1 / (1 + np.exp(-preds_logits.reshape(-1)))  # sigmoid to probability
      3 
      4 

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_11/1554577051.py in predict_loader(self, loader)
    260         self.model.eval()
    261         preds = []
--> 262         for batch in loader:
    263             x = batch["image"].to(device)
    264             logits = self.model(x)

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

/tmp/ipykernel_11/1554577051.py in __getitem__(self, i)
     76     def __getitem__(self, i):
     77         row = self.df.iloc[i]
---> 78         item = self.reader_fn(i, row)
     79         if self.transforms is not None:
     80             item = self.transforms(item)

/tmp/ipykernel_11/1899996337.py in reader_fn(i, row)
      4     image = cv2.imread(img_path)
      5     if image is None:
----> 6         raise FileNotFoundError(f"Could not read image: {img_path}")
      7     image = image[:, :, ::-1]  # BGR -> RGB
      8     label = torch.tensor([float(row["has_cactus"])], dtype=torch.float32)

FileNotFoundError: Could not read image: /kaggle/input/aerial-cactus-identification/test/test

## === cell 12
sub = pd.read_csv(SAMPLE_SUB)
pred_map = dict(zip(test_df["id"].values, preds))
sub["has_cactus"] = sub["id"].map(pred_map).astype(float)

if sub["has_cactus"].isna().any():
    sub["has_cactus"] = sub["has_cactus"].fillna(float(np.nanmean(preds)))

sub_path = "sub.csv"
sub.to_csv(sub_path, index=False)
print(sub.head())
print(f"Wrote submission to: {sub_path} (rows={len(sub)})")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3372496524.py in <cell line: 0>()
      1 # Create submission aligned to sample_submission ordering to avoid any id order mismatch.
      2 sub = pd.read_csv(SAMPLE_SUB)
----> 3 pred_map = dict(zip(test_df["id"].values, preds))
      4 sub["has_cactus"] = sub["id"].map(pred_map).astype(float)
      5 

NameError: name 'preds' is not defined
