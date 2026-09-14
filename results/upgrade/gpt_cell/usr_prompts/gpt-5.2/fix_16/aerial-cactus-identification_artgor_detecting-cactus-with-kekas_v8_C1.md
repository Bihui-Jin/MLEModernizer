# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.88031

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.88031) has done: 'The crash is caused by `test_df` being built from `os.listdir(TEST_DIR)`, but in this environment `TEST_DIR` contains a nested `test/` directory, so `os.listdir` includes the subdirectory name `"test"` as an “id”. That produces an invalid path like `.../test/test` (and then `.../test/test/test` via the fallback), which `cv2.imread` can’t read and triggers the `FileNotFoundError`. The minimal fix is to filter `test_df` to include only actual image files (e.g., `.jpg/.png/.jpeg`) and ignore directories like `"test"`. This keeps the rest of the inference pipeline unchanged and preserves the shape/type of `preds` expected by cell 21.'

# 9. Code solution

## === cell 0
import os
import random
import time

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
import torch.nn as nn
import torchvision.transforms as transforms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
import albumentations
import albumentations.pytorch as AT  # kept to preserve original imports/semantics (even if unused)

try:
    import pretrainedmodels  # type: ignore
except ModuleNotFoundError:
    pretrainedmodels = None

try:
    import adabound  # type: ignore
except ModuleNotFoundError:
    adabound = None

try:
    from kekas import Keker, DataOwner, DataKek  # type: ignore
    from kekas.transformations import Transformer, to_torch, normalize  # type: ignore
    from kekas.modules import Flatten  # type: ignore
except ModuleNotFoundError:
    Keker = DataOwner = DataKek = None
    Transformer = to_torch = normalize = None
    Flatten = None


## === cell 2
BASE_PATH = "/kaggle/input/aerial-cactus-identification"
if not os.path.exists(BASE_PATH):
    BASE_PATH = "/kaggle/data/aerial-cactus-identification"

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")

labels = pd.read_csv(TRAIN_CSV)

fig = plt.figure(figsize=(25, 8))
train_imgs = os.listdir(TRAIN_DIR)
for idx, img in enumerate(np.random.choice(train_imgs, 20, replace=False)):
    ax = fig.add_subplot(4, 5, idx + 1, xticks=[], yticks=[])
    im = Image.open(os.path.join(TRAIN_DIR, img))
    plt.imshow(im)
    lab = labels.loc[labels["id"] == img, "has_cactus"].values[0]
    ax.set_title(f"Label: {lab}")
plt.tight_layout()


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
    labels, stratify=labels.has_cactus, test_size=0.2, random_state=SEED
)




## === cell 6
def reader_fn(i, row):
    if row["data_type"] == "train":
        img_path = os.path.join(TRAIN_DIR, row["id"])
    else:
        img_path = os.path.join(TEST_DIR, row["id"])
    image = cv2.imread(img_path)[:, :, ::-1]  # BGR -> RGB
    label = torch.Tensor([row["has_cactus"]])
    return {"image": image, "label": label}




## === cell 7
def augs(p=0.5):
    return albumentations.Compose(
        [
            albumentations.HorizontalFlip(),
        ],
        p=p,
    )




## === cell 8
def get_transforms(dataset_key, size, p):
    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))
    AUGS = Transformer(dataset_key, lambda x: augs(p=p)(image=x)["image"])

    NRM_TFMS = transforms.Compose(
        [
            Transformer(dataset_key, to_torch()),
            Transformer(dataset_key, normalize()),
        ]
    )

    train_tfms = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
    val_tfms = transforms.Compose([PRE_TFMS, NRM_TFMS])

    return train_tfms, val_tfms




## === cell 9
if Transformer is None or to_torch is None or normalize is None:

    class Transformer:
        def __init__(self, key, fn):
            self.key = key
            self.fn = fn

        def __call__(self, sample):
            sample = dict(sample)
            sample[self.key] = self.fn(sample[self.key])
            return sample

    def to_torch():
        def _to_torch(x):
            if isinstance(x, torch.Tensor):
                return x
            x = np.asarray(x)
            if x.ndim == 2:  # grayscale -> add channel dim
                x = x[:, :, None]
            x = torch.from_numpy(x).permute(2, 0, 1).float().contiguous() / 255.0
            return x

        return _to_torch

    def normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        mean_t = torch.tensor(mean).view(-1, 1, 1)
        std_t = torch.tensor(std).view(-1, 1, 1)

        def _normalize(x):
            if not isinstance(x, torch.Tensor):
                x = to_torch()(x)
            return (x - mean_t) / std_t

        return _normalize


train_tfms, val_tfms = get_transforms("image", 32, 0.5)


## === cell 10
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

if DataKek is None:

    class DataKek(Dataset):
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


train_dk = DataKek(df=train, reader_fn=reader_fn, transforms=train_tfms)
val_dk = DataKek(df=valid, reader_fn=reader_fn, transforms=val_tfms)

batch_size = 64
workers = 0

train_dl = DataLoader(
    train_dk, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True
)
val_dl = DataLoader(val_dk, batch_size=batch_size, num_workers=workers, shuffle=False)


## === cell 11
test_dk = DataKek(df=test_df, reader_fn=reader_fn, transforms=val_tfms)
test_dl = DataLoader(test_dk, batch_size=batch_size, num_workers=workers, shuffle=False)




## === cell 12
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




## === cell 13
if Flatten is None:

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)


if pretrainedmodels is None:
    import torchvision.models as tvm

    class _PretrainedModelsShim:
        def __init__(self):
            self.__dict__["densenet169"] = self._densenet169

        def _densenet169(self, pretrained="imagenet"):
            weights = None
            if pretrained in ("imagenet", True):
                weights = tvm.DenseNet169_Weights.IMAGENET1K_V1
            return tvm.densenet169(weights=weights)

    pretrainedmodels = _PretrainedModelsShim()

if DataOwner is None:

    class DataOwner:
        def __init__(self, train_dl, val_dl=None, test_dl=None):
            self.train_dl = train_dl
            self.val_dl = val_dl
            self.test_dl = test_dl


dataowner = DataOwner(train_dl, val_dl, None)
model = Net(num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()


## === cell 14
def step_fn(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    inp = batch["image"].to(device)
    return model(inp)




## === cell 15
def bce_accuracy(
    target: torch.Tensor, preds: torch.Tensor, thresh: bool = 0.5
) -> float:
    target = target.cpu().detach().numpy()
    preds = (torch.sigmoid(preds).cpu().detach().numpy() > thresh).astype(int)
    return accuracy_score(target, preds)


def roc_auc(target: torch.Tensor, preds: torch.Tensor) -> float:
    target = target.cpu().detach().numpy()
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    return roc_auc_score(target, preds)




## === cell 16
if Keker is None:

    class Keker:  # minimal compatibility shim
        def __init__(
            self,
            model,
            dataowner,
            criterion,
            step_fn,
            target_key="label",
            metrics=None,
            opt=None,
            opt_params=None,
            **kwargs,
        ):
            self.model = model
            self.dataowner = dataowner
            self.criterion = criterion
            self.step_fn = step_fn
            self.target_key = target_key
            self.metrics = metrics or {}
            self.opt = opt
            self.opt_params = opt_params or {}
            self.kwargs = kwargs

        def unfreeze(self, model_attr="net"):
            return

        def freeze_to(self, layer_num, model_attr="net"):
            return


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


## === cell 17
keker.unfreeze(model_attr="net")
layer_num = -1
keker.freeze_to(layer_num, model_attr="net")



## === cell 18
if not hasattr(keker, "kek_one_cycle"):

    def _kek_one_cycle(
        self,
        max_lr=1e-2,
        cycle_len=1,
        momentum_range=(0.95, 0.85),
        div_factor=25,
        increase_fraction=0.3,
        logdir=None,
        **kwargs,
    ):
        model = self.model
        model.train()

        opt_cls = self.opt if self.opt is not None else torch.optim.SGD
        opt_params = dict(self.opt_params) if self.opt_params is not None else {}
        opt_params.setdefault("lr", float(max_lr))
        optimizer = opt_cls(model.parameters(), **opt_params)

        train_dl = self.dataowner.train_dl
        val_dl = self.dataowner.val_dl

        for epoch in range(int(cycle_len)):
            model.train()
            train_losses = []
            train_targets = []
            train_logits = []

            for batch in train_dl:
                optimizer.zero_grad(set_to_none=True)
                logits = self.step_fn(model, batch)
                target = batch[self.target_key].to(logits.device).float()

                if logits.dim() == 1:
                    logits_ = logits.view(-1, 1)
                else:
                    logits_ = logits
                if target.dim() == 1:
                    target_ = target.view(-1, 1)
                else:
                    target_ = target

                loss = self.criterion(logits_, target_)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.detach().cpu().item())
                train_targets.append(target_.detach().cpu())
                train_logits.append(logits_.detach().cpu())

            if train_targets:
                t_t = torch.cat(train_targets, dim=0)
                t_p = torch.cat(train_logits, dim=0)
                train_metric_vals = {}
                for name, fn in (self.metrics or {}).items():
                    try:
                        train_metric_vals[name] = fn(t_t, t_p)
                    except Exception:
                        train_metric_vals[name] = float("nan")
            else:
                train_metric_vals = {
                    name: float("nan") for name in (self.metrics or {})
                }

            val_metric_vals = None
            val_loss_mean = None
            if val_dl is not None:
                model.eval()
                val_losses = []
                val_targets = []
                val_logits = []
                with torch.no_grad():
                    for batch in val_dl:
                        logits = self.step_fn(model, batch)
                        target = batch[self.target_key].to(logits.device).float()

                        if logits.dim() == 1:
                            logits_ = logits.view(-1, 1)
                        else:
                            logits_ = logits
                        if target.dim() == 1:
                            target_ = target.view(-1, 1)
                        else:
                            target_ = target

                        loss = self.criterion(logits_, target_)
                        val_losses.append(loss.detach().cpu().item())
                        val_targets.append(target_.detach().cpu())
                        val_logits.append(logits_.detach().cpu())

                val_loss_mean = (
                    float(np.mean(val_losses)) if val_losses else float("nan")
                )
                if val_targets:
                    v_t = torch.cat(val_targets, dim=0)
                    v_p = torch.cat(val_logits, dim=0)
                    val_metric_vals = {}
                    for name, fn in (self.metrics or {}).items():
                        try:
                            val_metric_vals[name] = fn(v_t, v_p)
                        except Exception:
                            val_metric_vals[name] = float("nan")
                else:
                    val_metric_vals = {
                        name: float("nan") for name in (self.metrics or {})
                    }

            tr_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            msg = f"Epoch {epoch+1}/{int(cycle_len)} - train_loss: {tr_loss:.5f}"
            for k, v in train_metric_vals.items():
                msg += (
                    f" - train_{k}: {v:.5f}"
                    if isinstance(v, (int, float, np.floating))
                    else f" - train_{k}: {v}"
                )
            if val_dl is not None:
                msg += f" - val_loss: {val_loss_mean:.5f}"
                for k, v in (val_metric_vals or {}).items():
                    msg += (
                        f" - val_{k}: {v:.5f}"
                        if isinstance(v, (int, float, np.floating))
                        else f" - val_{k}: {v}"
                    )
            print(msg)

        return

    import types

    keker.kek_one_cycle = types.MethodType(_kek_one_cycle, keker)

keker.kek_one_cycle(
    max_lr=1e-2,
    cycle_len=5,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.3,
    logdir="train_logs",
)


## === cell 19
keker.kek_one_cycle(
    max_lr=1e-3,
    cycle_len=4,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.2,
    logdir="train_logs1",
)



## === cell 20
if not hasattr(keker, "predict_loader"):

    def _predict_loader(self, loader):
        self.model.eval()
        outs = []
        with torch.no_grad():
            for batch in loader:
                logits = self.step_fn(self.model, batch)
                if logits.dim() == 1:
                    logits = logits.view(-1, 1)
                outs.append(logits.detach().cpu())
        if outs:
            return torch.cat(outs, dim=0).numpy()
        return np.empty((0, 1), dtype=np.float32)

    import types

    keker.predict_loader = types.MethodType(_predict_loader, keker)


def reader_fn(i, row):
    if row["data_type"] == "train":
        img_path = os.path.join(TRAIN_DIR, row["id"])
        image = cv2.imread(img_path)
        if image is None:
            alt_path = os.path.join(TRAIN_DIR, "train", row["id"])
            image = cv2.imread(alt_path)
            img_path = alt_path
    else:
        img_path = os.path.join(TEST_DIR, row["id"])
        image = cv2.imread(img_path)
        if image is None:
            alt_path = os.path.join(TEST_DIR, "test", row["id"])
            image = cv2.imread(alt_path)
            img_path = alt_path

    if image is None:
        raise FileNotFoundError(
            f"Failed to read image with cv2.imread from: {img_path}"
        )

    image = image[:, :, ::-1]  # BGR -> RGB
    label = torch.Tensor([row["has_cactus"]])
    return {"image": image, "label": label}


from torch.utils.data import DataLoader

_valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
test_df = test_df[
    test_df["id"].astype(str).str.lower().str.endswith(_valid_ext)
].reset_index(drop=True)

test_dk = DataKek(df=test_df, reader_fn=reader_fn, transforms=val_tfms)
test_dl = DataLoader(test_dk, batch_size=batch_size, num_workers=workers, shuffle=False)

preds = keker.predict_loader(loader=test_dl)


## === cell 21
preds_prob = (
    torch.sigmoid(torch.tensor(preds))
    .numpy()
    .reshape(
        -1,
    )
)



## === cell 22
sub = pd.read_csv(SAMPLE_SUB)
pred_map = dict(zip(test_df["id"].values, preds_prob))
sub["has_cactus"] = sub["id"].map(pred_map).astype(float)

sub["has_cactus"] = sub["has_cactus"].fillna(0.5)

sub.to_csv("submission.csv", index=False)
sub.head()
