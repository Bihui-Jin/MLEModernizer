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

0.9393

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
import torchvision.transforms as tv_transforms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "../input/aerial-cactus-identification"
if not os.path.exists(DATA_ROOT):
    if os.path.exists("../input/train.csv"):
        DATA_ROOT = "../input"
    else:
        for cand in [
            "/kaggle/input/aerial-cactus-identification",
            "/kaggle/input",
            "../input/aerial-cactus-identification",
            "../input",
        ]:
            if os.path.exists(cand):
                DATA_ROOT = cand
                break

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

print("Using DATA_ROOT:", DATA_ROOT)
print("Train CSV:", TRAIN_CSV, "exists:", os.path.exists(TRAIN_CSV))
print("Train dir:", TRAIN_DIR, "exists:", os.path.exists(TRAIN_DIR))
print("Test dir:", TEST_DIR, "exists:", os.path.exists(TEST_DIR))



## === cell 1
import albumentations as A



class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Transformer:
    def __init__(self, key, fn):
        self.key = key
        self.fn = fn

    def __call__(self, sample):
        sample[self.key] = self.fn(sample[self.key])
        return sample


def to_torch():
    def _fn(x):
        if isinstance(x, torch.Tensor):
            return x
        x = x.astype(np.float32)
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x)

    return _fn


def normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    def _fn(x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x / 255.0
        return (x - mean) / std

    return _fn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=1):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], dim=1)


class DataKek(torch.utils.data.Dataset):
    def __init__(self, df, reader_fn, transforms=None):
        self.df = df.reset_index(drop=True)
        self.reader_fn = reader_fn
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = self.reader_fn(idx, row)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


def _default_collate(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {"image": images, "label": labels}


class DataOwner:
    def __init__(self, train_dl, val_dl, test_dl=None):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl


class Keker:
    def __init__(
        self,
        model,
        dataowner,
        criterion,
        step_fn,
        target_key="label",
        metrics=None,
        opt=torch.optim.SGD,
        opt_params=None,
    ):
        self.model = model.to(device)
        self.dataowner = dataowner
        self.criterion = criterion
        self.step_fn = step_fn
        self.target_key = target_key
        self.metrics = metrics or {}
        self.opt_cls = opt
        self.opt_params = opt_params or {}
        self.optimizer = self.opt_cls(
            self.model.parameters(), lr=1e-3, **self.opt_params
        )

        self.history = []

    def unfreeze(self, model_attr=None):
        for p in self.model.parameters():
            p.requires_grad = True

    def freeze_to(self, layer_num, model_attr=None):
        return

    def _run_loader(self, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_logits = []
        n = 0
        for batch in loader:
            batch["image"] = batch["image"].to(device, non_blocking=True)
            batch["label"] = batch["label"].to(device, non_blocking=True)

            logits = self.step_fn(self.model, batch)
            target = batch[self.target_key]

            loss = self.criterion(logits, target)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            bs = target.size(0)
            total_loss += loss.detach().item() * bs
            n += bs
            all_targets.append(target.detach().cpu())
            all_logits.append(logits.detach().cpu())

        all_targets = torch.cat(all_targets, dim=0)
        all_logits = torch.cat(all_logits, dim=0)
        avg_loss = total_loss / max(n, 1)

        metric_vals = {}
        for name, fn in self.metrics.items():
            try:
                metric_vals[name] = float(fn(all_targets, all_logits))
            except Exception:
                metric_vals[name] = np.nan
        return avg_loss, metric_vals

    def kek_one_cycle(
        self,
        max_lr=1e-1,
        cycle_len=5,
        momentum_range=(0.95, 0.85),
        div_factor=25,
        increase_fraction=0.3,
        logdir=None,
    ):
        base_lr = max_lr / div_factor
        for g in self.optimizer.param_groups:
            g["lr"] = base_lr

        for epoch in range(cycle_len):
            t = (epoch + 1) / cycle_len
            if t <= increase_fraction:
                lr = base_lr + (max_lr - base_lr) * (t / increase_fraction)
            else:
                lr = max_lr - (max_lr - base_lr) * (
                    (t - increase_fraction) / max(1e-8, (1 - increase_fraction))
                )
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            tr_loss, tr_metrics = self._run_loader(self.dataowner.train_dl, train=True)
            va_loss, va_metrics = self._run_loader(self.dataowner.val_dl, train=False)
            rec = {
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": tr_loss,
                "val_loss": va_loss,
            }
            for k, v in tr_metrics.items():
                rec[f"train_{k}"] = v
            for k, v in va_metrics.items():
                rec[f"val_{k}"] = v
            self.history.append(rec)
            print(rec)

    def plot_kek(self, logdir=None):
        if len(self.history) == 0:
            return
        try:
            dfh = pd.DataFrame(self.history)
            ax = dfh.plot(
                x="epoch", y=["train_loss", "val_loss"], figsize=(8, 4), title="Loss"
            )
            plt.show()
        except Exception:
            pass

    @torch.no_grad()
    def predict_loader(self, loader):
        self.model.eval()
        preds = []
        for batch in loader:
            batch["image"] = batch["image"].to(device, non_blocking=True)
            logits = self.step_fn(self.model, batch)
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.vstack(preds)

    @torch.no_grad()
    def TTA(self, loader, tfms, savedir=None, prefix="preds"):
        os.makedirs(savedir, exist_ok=True) if savedir is not None else None
        out = {}
        for name, tfm in tfms.items():
            ds = loader.dataset
            old_t = ds.transforms
            ds.transforms = tfm
            pr = self.predict_loader(loader)
            ds.transforms = old_t
            out[name] = pr
            if savedir is not None:
                np.save(os.path.join(savedir, f"{prefix}_{name}.npy"), pr)
        return out




## === cell 2
labels = pd.read_csv(TRAIN_CSV)

try:
    fig = plt.figure(figsize=(16, 6))
    train_imgs = os.listdir(TRAIN_DIR)
    for idx, img in enumerate(np.random.choice(train_imgs, 12, replace=False)):
        ax = fig.add_subplot(3, 4, idx + 1, xticks=[], yticks=[])
        im = Image.open(os.path.join(TRAIN_DIR, img))
        plt.imshow(im)
        lab = labels.loc[labels["id"] == img, "has_cactus"].values[0]
        ax.set_title(f"Label: {lab}")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Skipping visualization:", repr(e))



## === cell 3
test_img = os.listdir(TEST_DIR)
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
    labels, stratify=labels.has_cactus, test_size=0.1, random_state=SEED
)




## === cell 6
def reader_fn(i, row):
    base_dir = TRAIN_DIR if row["data_type"] == "train" else TEST_DIR
    path = os.path.join(base_dir, row["id"])
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = image[:, :, ::-1]  # BGR -> RGB
    label = torch.tensor([float(row["has_cactus"])], dtype=torch.float32)
    return {"image": image, "label": label}




## === cell 7
def augs(p=0.5):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
        ],
        p=p,
    )




## === cell 8
def get_transforms(dataset_key, size, p):
    PRE_TFMS = Transformer(
        dataset_key, lambda x: cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    )
    AUGS = Transformer(dataset_key, lambda x: augs(p=p)(image=x)["image"])

    NRM_TFMS = tv_transforms.Compose(
        [
            Transformer(dataset_key, to_torch()),
            Transformer(dataset_key, normalize()),
        ]
    )

    train_tfms = tv_transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
    val_tfms = tv_transforms.Compose([PRE_TFMS, NRM_TFMS])

    return train_tfms, val_tfms




## === cell 9
train_tfms, val_tfms = get_transforms("image", 32, 0.5)



## === cell 10
train_dk = DataKek(df=train, reader_fn=reader_fn, transforms=train_tfms)
val_dk = DataKek(df=valid, reader_fn=reader_fn, transforms=val_tfms)

batch_size = 128
workers = 0

train_dl = torch.utils.data.DataLoader(
    train_dk,
    batch_size=batch_size,
    num_workers=workers,
    shuffle=True,
    drop_last=True,
    collate_fn=_default_collate,
)
val_dl = torch.utils.data.DataLoader(
    val_dk,
    batch_size=batch_size,
    num_workers=workers,
    shuffle=False,
    drop_last=False,
    collate_fn=_default_collate,
)



## === cell 11
test_dk = DataKek(df=test_df, reader_fn=reader_fn, transforms=val_tfms)
test_dl = torch.utils.data.DataLoader(
    test_dk,
    batch_size=batch_size,
    num_workers=workers,
    shuffle=False,
    drop_last=False,
    collate_fn=_default_collate,
)



## === cell 12
try:
    import pretrainedmodels
except Exception as e:
    raise ImportError(
        "pretrainedmodels is required by the original core logic but isn't available in this environment."
    ) from e


class Net(nn.Module):
    def __init__(
        self,
        num_classes: int,
        p: float = 0.2,
        pooling_size: int = 2,
        last_conv_size: int = 81536,
        arch: str = "densenet169",
        pretrained: str = "imagenet",
    ) -> None:
        super().__init__()
        net = pretrainedmodels.__dict__[arch](pretrained=pretrained)
        modules = list(net.children())[:-1]  # delete last layer
        modules += [
            nn.Sequential(
                AdaptiveConcatPool2d(size=pooling_size),
                Flatten(),
                nn.BatchNorm1d(13312),
                nn.Dropout(p),
                nn.Linear(13312, num_classes),
            )
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2710001513.py in <cell line: 0>()
      2 try:
----> 3     import pretrainedmodels
      4 except Exception as e:

ModuleNotFoundError: No module named 'pretrainedmodels'

The above exception was the direct cause of the following exception:

ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2710001513.py in <cell line: 0>()
      3     import pretrainedmodels
      4 except Exception as e:
----> 5     raise ImportError(
      6         "pretrainedmodels is required by the original core logic but isn't available in this environment."
      7     ) from e

ImportError: pretrainedmodels is required by the original core logic but isn't available in this environment.

## === cell 13
dataowner = DataOwner(train_dl, val_dl, None)
model = Net(num_classes=1)
criterion = nn.BCEWithLogitsLoss()




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3186603736.py in <cell line: 0>()
      1 dataowner = DataOwner(train_dl, val_dl, None)
----> 2 model = Net(num_classes=1)
      3 criterion = nn.BCEWithLogitsLoss()
      4 
      5 

NameError: name 'Net' is not defined

## === cell 14
def step_fn(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    inp = batch["image"]
    return model(inp)




## === cell 15
def bce_accuracy(
    target: torch.Tensor, preds: torch.Tensor, thresh: float = 0.5
) -> float:
    target = target.cpu().detach().numpy().reshape(-1)
    preds = (torch.sigmoid(preds).cpu().detach().numpy().reshape(-1) > thresh).astype(
        int
    )
    return accuracy_score(target, preds)


def roc_auc(target: torch.Tensor, preds: torch.Tensor) -> float:
    target = target.cpu().detach().numpy().reshape(-1)
    preds = torch.sigmoid(preds).cpu().detach().numpy().reshape(-1)
    if len(np.unique(target)) < 2:
        return float("nan")
    return roc_auc_score(target, preds)




## === cell 16
keker = Keker(
    model=model,
    dataowner=dataowner,
    criterion=criterion,
    step_fn=step_fn,
    target_key="label",
    metrics={"acc": bce_accuracy, "auc": roc_auc},
    opt=torch.optim.SGD,
    opt_params={"momentum": 0.99},
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4229198289.py in <cell line: 0>()
      1 keker = Keker(
----> 2     model=model,
      3     dataowner=dataowner,
      4     criterion=criterion,
      5     step_fn=step_fn,

NameError: name 'model' is not defined

## === cell 17
keker.unfreeze(model_attr="net")
layer_num = -1
keker.freeze_to(layer_num, model_attr="net")



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4222199681.py in <cell line: 0>()
----> 1 keker.unfreeze(model_attr="net")
      2 layer_num = -1
      3 keker.freeze_to(layer_num, model_attr="net")
      4 

NameError: name 'keker' is not defined

## === cell 18
keker.kek_one_cycle(
    max_lr=1e-1,
    cycle_len=5,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.3,
    logdir="train_logs11",
)
keker.plot_kek("train_logs11")



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/74099311.py in <cell line: 0>()
----> 1 keker.kek_one_cycle(
      2     max_lr=1e-1,
      3     cycle_len=5,
      4     momentum_range=(0.95, 0.85),
      5     div_factor=25,

NameError: name 'keker' is not defined

## === cell 19
preds = keker.predict_loader(loader=test_dl)
print("Raw preds shape:", preds.shape)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2783637778.py in <cell line: 0>()
----> 1 preds = keker.predict_loader(loader=test_dl)
      2 print("Raw preds shape:", preds.shape)
      3 

NameError: name 'keker' is not defined

## === cell 20
flip_ = A.HorizontalFlip(p=1.0)
transpose_ = A.Transpose(p=1.0)


def insert_aug(aug, dataset_key="image", size=32):
    PRE_TFMS = Transformer(
        dataset_key, lambda x: cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
    )
    AUGS = Transformer(dataset_key, lambda x: aug(image=x)["image"])
    NRM_TFMS = tv_transforms.Compose(
        [Transformer(dataset_key, to_torch()), Transformer(dataset_key, normalize())]
    )
    return tv_transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])


flip = insert_aug(flip_)
transpose = insert_aug(transpose_)

tta_tfms = {"flip": flip, "transpose": transpose}

tta_out = keker.TTA(loader=test_dl, tfms=tta_tfms, savedir="tta_preds1", prefix="preds")



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/299934885.py in <cell line: 0>()
     21 
     22 # Compute & (optionally) save like original; also return values so we don't depend on filesystem.
---> 23 tta_out = keker.TTA(loader=test_dl, tfms=tta_tfms, savedir="tta_preds1", prefix="preds")
     24 

NameError: name 'keker' is not defined

## === cell 21
prediction = preds.copy()
cnt = 1
for k, pr in tta_out.items():
    prediction += pr
    cnt += 1
prediction = prediction / cnt
print("Ensembled preds shape:", prediction.shape)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/409536862.py in <cell line: 0>()
      1 # Average base prediction + TTA predictions (same intent as original folder-averaging, but robust).
----> 2 prediction = preds.copy()
      3 cnt = 1
      4 for k, pr in tta_out.items():
      5     prediction += pr

NameError: name 'preds' is not defined

## === cell 22
sample = pd.read_csv(SAMPLE_SUB)
sub = sample.copy()

pred_map = dict(zip(test_df["id"].values, prediction.reshape(-1)))
sub["has_cactus"] = sub["id"].map(pred_map).astype(float)

if sub["has_cactus"].isna().any():
    sub["has_cactus"] = sub["has_cactus"].fillna(float(np.nanmean(prediction)))

sub.to_csv("sub.csv", index=False)
print(sub.head())
print("Saved sub.csv with shape:", sub.shape)
print("Any NA preds:", sub["has_cactus"].isna().any())

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1620804431.py in <cell line: 0>()
      3 sub = sample.copy()
      4 
----> 5 pred_map = dict(zip(test_df["id"].values, prediction.reshape(-1)))
      6 sub["has_cactus"] = sub["id"].map(pred_map).astype(float)
      7 

NameError: name 'prediction' is not defined
