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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/kaggle/input/aerial-cactus-identification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")



## === cell 1
import albumentations

try:
    from albumentations import torch as AT  # legacy (albumentations<1.x)
except Exception:
    from albumentations.pytorch import ToTensorV2

    class _AT:
        @staticmethod
        def ToTensor(*args, **kwargs):
            return ToTensorV2(*args, **kwargs)

    AT = _AT

try:
    import pretrainedmodels  # type: ignore
except ModuleNotFoundError:
    import torchvision.models as _tv_models

    class _PretrainedModelsShim:
        def __getattr__(self, name):
            if not hasattr(_tv_models, name):
                raise AttributeError(
                    f"pretrainedmodels shim: torchvision.models has no attribute '{name}'."
                )
            fn = getattr(_tv_models, name)

            def _wrapper(*args, **kwargs):
                pretrained = kwargs.pop("pretrained", None)
                num_classes = kwargs.pop("num_classes", None)

                if pretrained is not None:
                    if "weights" in fn.__code__.co_varnames:
                        if pretrained:
                            weights_enum = getattr(
                                _tv_models, f"{name.upper()}_Weights", None
                            )
                            kwargs["weights"] = (
                                weights_enum.DEFAULT if weights_enum else "DEFAULT"
                            )
                        else:
                            kwargs["weights"] = None
                    else:
                        kwargs["pretrained"] = bool(pretrained)

                model = fn(*args, **kwargs)

                if num_classes is not None:
                    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Module):
                        in_features = getattr(model.fc, "in_features", None)
                        if in_features is not None:
                            model.fc = torch.nn.Linear(in_features, int(num_classes))
                    elif hasattr(model, "classifier"):
                        clf = model.classifier
                        if isinstance(clf, torch.nn.Linear):
                            model.classifier = torch.nn.Linear(
                                clf.in_features, int(num_classes)
                            )
                        elif isinstance(clf, torch.nn.Sequential) and len(clf) > 0:
                            for i in range(len(clf) - 1, -1, -1):
                                if isinstance(clf[i], torch.nn.Linear):
                                    model.classifier[i] = torch.nn.Linear(
                                        clf[i].in_features, int(num_classes)
                                    )
                                    break
                return model

            return _wrapper

    pretrainedmodels = _PretrainedModelsShim()

try:
    from kekas import Keker, DataOwner, DataKek
    from kekas.transformations import Transformer, to_torch, normalize
    from kekas.metrics import accuracy
    from kekas.modules import Flatten
except ModuleNotFoundError:

    class _KekasMissingError(ModuleNotFoundError):
        pass

    def _raise_kekas_missing(*args, **kwargs):
        raise _KekasMissingError(
            "Package 'kekas' is not installed in this environment, but the notebook "
            "attempted to use it. Install 'kekas' or replace its usage."
        )

    class Keker:
        def __init__(self, *args, **kwargs):
            _raise_kekas_missing()

    class DataOwner:
        def __init__(self, *args, **kwargs):
            _raise_kekas_missing()

    class DataKek:
        def __init__(self, *args, **kwargs):
            _raise_kekas_missing()

    class Transformer:
        def __init__(self, *args, **kwargs):
            _raise_kekas_missing()

    def to_torch(*args, **kwargs):
        _raise_kekas_missing()

    def normalize(*args, **kwargs):
        _raise_kekas_missing()

    def accuracy(*args, **kwargs):
        _raise_kekas_missing()

    class Flatten(nn.Module):
        def forward(self, x):
            _raise_kekas_missing()


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
    plt.close(fig)
except Exception:
    pass



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB)
test_df = sample_sub[["id"]].copy()
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

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    image = image[:, :, ::-1]  # BGR -> RGB

    label = torch.Tensor([row["has_cactus"]])
    return {"image": image, "label": label}




## === cell 7
def augs(p=0.5):
    return albumentations.Compose(
        [
            albumentations.HorizontalFlip(),
            albumentations.VerticalFlip(),
            albumentations.RandomBrightness(),
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
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(dataset_key, size, p):
    class _DictImageTransform:
        def __init__(self, train: bool):
            self.train = train
            self.resize_size = int(size)

        def __call__(self, sample):
            if isinstance(sample, dict):
                img = sample[dataset_key]
            else:
                img = sample

            img = cv2.resize(img, (self.resize_size, self.resize_size))

            if self.train:
                img = augs(p=p)(image=img)["image"]

            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img_t = torch.from_numpy(img)

            mean = torch.tensor(IMAGENET_MEAN, dtype=img_t.dtype).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD, dtype=img_t.dtype).view(3, 1, 1)
            img_t = (img_t - mean) / std

            if isinstance(sample, dict):
                sample = dict(sample)
                sample[dataset_key] = img_t
                return sample
            return img_t

    train_tfms = _DictImageTransform(train=True)
    val_tfms = _DictImageTransform(train=False)
    return train_tfms, val_tfms


train_tfms, val_tfms = get_transforms("image", 32, 0.5)


## === cell 10
from torch.utils.data import DataLoader, Dataset


class _DFDataset(Dataset):
    def __init__(self, df, reader_fn, transforms=None):
        self.df = df.reset_index(drop=True)
        self.reader_fn = reader_fn
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[int(idx)]
        sample = self.reader_fn(int(idx), row)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


train_dk = _DFDataset(df=train, reader_fn=reader_fn, transforms=train_tfms)
val_dk = _DFDataset(df=valid, reader_fn=reader_fn, transforms=val_tfms)

batch_size = 64
workers = 0

train_dl = DataLoader(
    train_dk, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True
)
val_dl = DataLoader(val_dk, batch_size=batch_size, num_workers=workers, shuffle=False)


## === cell 11
test_dk = _DFDataset(df=test_df, reader_fn=reader_fn, transforms=val_tfms)
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
dataowner = None


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

        try:
            ctor = pretrainedmodels.__dict__[arch]
        except Exception:
            ctor = getattr(pretrainedmodels, arch)

        net = ctor(pretrained=pretrained)
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


model = Net(num_classes=1).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()


## === cell 14
def step_fn(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    inp = batch["image"].to(DEVICE)
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
try:
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
except Exception:

    class _KekerFallback:
        def __init__(
            self,
            model,
            dataowner,
            criterion,
            step_fn,
            target_key,
            metrics,
            opt,
            opt_params,
        ):
            self.model = model
            self.dataowner = dataowner
            self.criterion = criterion
            self.step_fn = step_fn
            self.target_key = target_key
            self.metrics = metrics
            self.opt = opt
            self.opt_params = opt_params

        def _get_model_attr(self, model_attr=None):
            return getattr(self.model, model_attr) if model_attr else self.model

        def unfreeze(self, model_attr=None):
            m = self._get_model_attr(model_attr)
            for p in m.parameters():
                p.requires_grad = True

        def freeze_to(self, layer_num, model_attr=None):
            m = self._get_model_attr(model_attr)
            children = list(m.children())
            cutoff = int(layer_num)
            for idx, layer in enumerate(children):
                freeze = (
                    idx <= cutoff if cutoff >= 0 else idx <= (len(children) + cutoff)
                )
                if freeze:
                    for p in layer.parameters():
                        p.requires_grad = False

    keker = _KekerFallback(
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

import types


def _fallback_kek_one_cycle(
    self,
    max_lr,
    cycle_len,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.3,
    logdir=None,
):
    lr = float(max_lr)
    opt_kwargs = dict(self.opt_params) if isinstance(self.opt_params, dict) else {}
    optimizer = self.opt(self.model.parameters(), lr=lr, **opt_kwargs)

    history = {"train_loss": [], "val_loss": [], "train_metrics": [], "val_metrics": []}

    self.model.train()
    for epoch in range(int(cycle_len)):
        self.model.train()
        train_losses = []
        train_metric_sums = {k: 0.0 for k in (self.metrics or {}).keys()}
        train_batches = 0

        for batch in train_dl:
            optimizer.zero_grad(set_to_none=True)
            preds = self.step_fn(self.model, batch)
            target = batch[self.target_key].to(DEVICE)
            loss = self.criterion(preds, target)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.detach().cpu().item()))
            if self.metrics:
                for name, fn in self.metrics.items():
                    try:
                        train_metric_sums[name] += float(
                            fn(target.detach(), preds.detach())
                        )
                    except Exception:
                        train_metric_sums[name] += 0.0
            train_batches += 1

        self.model.eval()
        val_losses = []
        val_metric_sums = {k: 0.0 for k in (self.metrics or {}).keys()}
        val_batches = 0
        with torch.no_grad():
            for batch in val_dl:
                preds = self.step_fn(self.model, batch)
                target = batch[self.target_key].to(DEVICE)
                loss = self.criterion(preds, target)

                val_losses.append(float(loss.detach().cpu().item()))
                if self.metrics:
                    for name, fn in self.metrics.items():
                        try:
                            val_metric_sums[name] += float(
                                fn(target.detach(), preds.detach())
                            )
                        except Exception:
                            val_metric_sums[name] += 0.0
                val_batches += 1

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        train_metrics = {
            k: (v / max(1, train_batches)) for k, v in train_metric_sums.items()
        }
        val_metrics = {k: (v / max(1, val_batches)) for k, v in val_metric_sums.items()}

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metrics"].append(train_metrics)
        history["val_metrics"].append(val_metrics)

    self.history = history
    return history


def _fallback_plot_kek(self, logdir=None):
    return None


if not hasattr(keker, "kek_one_cycle"):
    keker.kek_one_cycle = types.MethodType(_fallback_kek_one_cycle, keker)
if not hasattr(keker, "plot_kek"):
    keker.plot_kek = types.MethodType(_fallback_plot_kek, keker)

keker.kek_one_cycle(
    max_lr=1e-2,
    cycle_len=5,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.3,
    logdir="train_logs",
)
try:
    keker.plot_kek("train_logs")
except Exception:
    pass


## --- ERROR in cell 18, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2843601002.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    100[0m     [0mkeker[0m[0;34m.[0m[0mplot_kek[0m [0;34m=[0m [0mtypes[0m[0;34m.[0m[0mMethodType[0m[0;34m([0m[0m_fallback_plot_kek[0m[0;34m,[0m [0mkeker[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    101[0m [0;34m[0m[0m
[0;32m--> 102[0;31m keker.kek_one_cycle(
[0m[1;32m    103[0m     [0mmax_lr[0m[0;34m=[0m[0;36m1e-2[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    104[0m     [0mcycle_len[0m[0;34m=[0m[0;36m5[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2843601002.py[0m in [0;36m_fallback_kek_one_cycle[0;34m(self, max_lr, cycle_len, momentum_range, div_factor, increase_fraction, logdir)[0m
[1;32m     31[0m         [0mtrain_batches[0m [0;34m=[0m [0;36m0[0m[0;34m[0m[0;34m[0m[0m
[1;32m     32[0m [0;34m[0m[0m
[0;32m---> 33[0;31m         [0;32mfor[0m [0mbatch[0m [0;32min[0m [0mtrain_dl[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     34[0m             [0moptimizer[0m[0;34m.[0m[0mzero_grad[0m[0;34m([0m[0mset_to_none[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     35[0m             [0mpreds[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mstep_fn[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mmodel[0m[0;34m,[0m [0mbatch[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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

[0;32m/tmp/ipykernel_11/1831241400.py[0m in [0;36m__getitem__[0;34m(self, idx)[0m
[1;32m     18[0m         [0msample[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mreader_fn[0m[0;34m([0m[0mint[0m[0;34m([0m[0midx[0m[0;34m)[0m[0;34m,[0m [0mrow[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mtransforms[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 20[0;31m             [0msample[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mtransforms[0m[0;34m([0m[0msample[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     21[0m         [0;32mreturn[0m [0msample[0m[0;34m[0m[0;34m[0m[0m
[1;32m     22[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/291920865.py[0m in [0;36m__call__[0;34m(self, sample)[0m
[1;32m     21[0m [0;34m[0m[0m
[1;32m     22[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0mtrain[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 23[0;31m                 [0mimg[0m [0;34m=[0m [0maugs[0m[0;34m([0m[0mp[0m[0;34m=[0m[0mp[0m[0;34m)[0m[0;34m([0m[0mimage[0m[0;34m=[0m[0mimg[0m[0;34m)[0m[0;34m[[0m[0;34m"image"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     24[0m [0;34m[0m[0m
[1;32m     25[0m             [0;31m# HWC uint8 -> CHW float32 in [0,1][0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1888095628.py[0m in [0;36maugs[0;34m(p)[0m
[1;32m      4[0m             [0malbumentations[0m[0;34m.[0m[0mHorizontalFlip[0m[0;34m([0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m             [0malbumentations[0m[0;34m.[0m[0mVerticalFlip[0m[0;34m([0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 6[0;31m             [0malbumentations[0m[0;34m.[0m[0mRandomBrightness[0m[0;34m([0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      7[0m         ],
[1;32m      8[0m         [0mp[0m[0;34m=[0m[0mp[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: module 'albumentations' has no attribute 'RandomBrightness'

## === cell 19
keker.kek_one_cycle(
    max_lr=1e-3,
    cycle_len=5,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.2,
    logdir="train_logs1",
)
try:
    keker.plot_kek("train_logs1")
except Exception:
    pass
