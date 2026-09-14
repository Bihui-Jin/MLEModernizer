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
import time
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from PIL import Image

import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from types import SimpleNamespace

AT = SimpleNamespace(ToTensorV2=ToTensorV2, ToTensor=ToTensorV2)

try:
    import pretrainedmodels  # type: ignore
except ModuleNotFoundError:

    class _PretrainedModelsShim:
        """
        Minimal subset of the `pretrainedmodels` API used in many Kaggle notebooks.
        It supports a few common model names via torchvision.models.
        """

        _NAME_MAP = {
            "resnet18": torchvision.models.resnet18,
            "resnet34": torchvision.models.resnet34,
            "resnet50": torchvision.models.resnet50,
            "resnet101": torchvision.models.resnet101,
            "resnet152": torchvision.models.resnet152,
            "densenet121": torchvision.models.densenet121,
            "densenet169": torchvision.models.densenet169,
            "densenet201": torchvision.models.densenet201,
            "densenet161": torchvision.models.densenet161,
        }

        def __getattr__(self, name):
            if name in self._NAME_MAP:
                ctor = self._NAME_MAP[name]

                def _wrapper(*args, **kwargs):
                    pretrained = kwargs.pop("pretrained", None)
                    if pretrained:
                        return ctor(weights="DEFAULT", *args, **kwargs)
                    return ctor(weights=None, *args, **kwargs)

                return _wrapper

            raise AttributeError(
                f"`pretrainedmodels` is not installed and model '{name}' is not available "
                f"in the shim. Available: {sorted(self._NAME_MAP.keys())}"
            )

    pretrainedmodels = _PretrainedModelsShim()

try:
    import adabound  # type: ignore
except ModuleNotFoundError:

    class _AdaBound(torch.optim.Adam):
        def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            final_lr=0.1,
            gamma=1e-3,
            eps=1e-8,
            weight_decay=0,
            amsbound=False,
        ):
            super().__init__(
                params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsbound,
            )
            self.final_lr = final_lr
            self.gamma = gamma

    adabound = SimpleNamespace(AdaBound=_AdaBound)

try:
    from kekas import Keker, DataOwner, DataKek  # type: ignore
    from kekas.transformations import Transformer, to_torch, normalize  # type: ignore
    from kekas.modules import Flatten, AdaptiveConcatPool2d  # type: ignore
except ModuleNotFoundError:

    class _KekasMissingDependency(RuntimeError):
        pass

    def _kekas_unavailable(*args, **kwargs):
        raise _KekasMissingDependency(
            "`kekas` is not installed in this environment, but the notebook attempted to use it."
        )

    class Keker:
        __init__ = _kekas_unavailable

    class DataOwner:
        __init__ = _kekas_unavailable

    class DataKek:
        __init__ = _kekas_unavailable

    class Transformer:
        __init__ = _kekas_unavailable

    def to_torch(*args, **kwargs):
        return _kekas_unavailable(*args, **kwargs)

    def normalize(*args, **kwargs):
        return _kekas_unavailable(*args, **kwargs)

    class Flatten(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            _kekas_unavailable(*args, **kwargs)

    class AdaptiveConcatPool2d(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            _kekas_unavailable(*args, **kwargs)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

DATA_ROOT = "/kaggle/input/aerial-cactus-identification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample_submission.csv at {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_DIR), f"Missing train dir at {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test dir at {TEST_DIR}"


## === cell 1
labels = pd.read_csv(TRAIN_CSV)

sample_sub = pd.read_csv(SAMPLE_SUB)
test_df = sample_sub[["id"]].copy()
test_df["has_cactus"] = -1
test_df["data_type"] = "test"

labels["has_cactus"] = labels["has_cactus"].astype(int)
labels["data_type"] = "train"

print(labels.head())
print("Train size:", labels.shape, "Test size:", test_df.shape)



## === cell 2
train, valid = train_test_split(
    labels, stratify=labels.has_cactus, test_size=0.2, random_state=42
)
print("Train split:", train.shape, "Valid split:", valid.shape)




## === cell 3
def reader_fn(i, row):
    img_path = os.path.join(DATA_ROOT, row["data_type"], row["id"])
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    image = image[:, :, ::-1]  # BGR -> RGB
    label = torch.Tensor([row["has_cactus"]])
    return {"image": image, "label": label}




## === cell 4
def augs(p=0.5):
    return albumentations.Compose(
        [
            albumentations.HorizontalFlip(),
        ],
        p=p,
    )


def get_transforms(dataset_key, size, p):
    aug = augs(p=p)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _apply(sample, do_aug: bool):
        img = sample[dataset_key]
        img = cv2.resize(img, (size, size))
        if do_aug:
            img = aug(image=img)["image"]
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        sample[dataset_key] = img
        return sample

    train_tfms = lambda sample: _apply(sample, do_aug=True)
    val_tfms = lambda sample: _apply(sample, do_aug=False)
    return train_tfms, val_tfms


IMG_SIZE = 32
train_tfms, val_tfms = get_transforms("image", IMG_SIZE, 0.5)


## === cell 5
class _SimpleDictDataset(torch.utils.data.Dataset):
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


train_dk = _SimpleDictDataset(df=train, reader_fn=reader_fn, transforms=train_tfms)
val_dk = _SimpleDictDataset(df=valid, reader_fn=reader_fn, transforms=val_tfms)

batch_size = 64
workers = 0

train_dl = torch.utils.data.DataLoader(
    train_dk, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True
)
val_dl = torch.utils.data.DataLoader(
    val_dk, batch_size=batch_size, num_workers=workers, shuffle=False
)

test_dk = _SimpleDictDataset(df=test_df, reader_fn=reader_fn, transforms=val_tfms)
test_dl = torch.utils.data.DataLoader(
    test_dk, batch_size=batch_size, num_workers=workers, shuffle=False
)


## === cell 6
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
        logits = self.net(x)
        return logits




## === cell 7
try:
    dataowner = DataOwner(train_dl, val_dl, None)
    _use_fallback_kekas = False
except Exception as e:
    if e.__class__.__name__ == "_KekasMissingDependency":
        _use_fallback_kekas = True
    else:
        raise

if _use_fallback_kekas:

    class DataOwner:  # minimal drop-in
        def __init__(self, train_dl, val_dl, test_dl=None):
            self.train_dl = train_dl
            self.val_dl = val_dl
            self.test_dl = test_dl

    class Keker:  # minimal drop-in
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
            self.model = model
            self.dataowner = dataowner
            self.criterion = criterion
            self.step_fn = step_fn
            self.target_key = target_key
            self.metrics = metrics or {}
            self.opt_cls = opt
            self.opt_params = opt_params or {}
            self.optimizer = self.opt_cls(self.model.parameters(), **self.opt_params)

        def unfreeze(self, model_attr="net"):
            module = getattr(self.model, model_attr, self.model)
            for p in module.parameters():
                p.requires_grad = True

        def freeze_to(self, layer_num, model_attr="net"):
            module = getattr(self.model, model_attr, self.model)
            children = list(module.children())
            if layer_num < 0:
                freeze_upto = len(children) + layer_num
            else:
                freeze_upto = layer_num
            for idx, ch in enumerate(children):
                req = idx > freeze_upto
                for p in ch.parameters():
                    p.requires_grad = req

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

    dataowner = DataOwner(train_dl, val_dl, None)

if hasattr(Net, "__init__") and hasattr(pretrainedmodels, "__class__"):
    _orig_net_init = Net.__init__

    def _patched_net_init(
        self,
        num_classes: int,
        p: float = 0.2,
        pooling_size: int = 2,
        last_conv_size: int = 81536,
        arch: str = "densenet169",
        pretrained: str = "imagenet",
    ) -> None:
        super(Net, self).__init__()
        try:
            ctor = getattr(pretrainedmodels, arch)
        except Exception:
            ctor = pretrainedmodels.__dict__[arch]
        net = ctor(pretrained=pretrained)
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

    Net.__init__ = _patched_net_init

model = Net(num_classes=1).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()


def step_fn(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    inp = batch["image"].to(DEVICE)
    return model(inp)


def bce_accuracy(
    target: torch.Tensor, preds: torch.Tensor, thresh: float = 0.5
) -> float:
    target = target.cpu().detach().numpy()
    preds = (torch.sigmoid(preds).cpu().detach().numpy() > thresh).astype(int)
    return accuracy_score(target, preds)


def roc_auc(target: torch.Tensor, preds: torch.Tensor) -> float:
    target = target.cpu().detach().numpy()
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    return roc_auc_score(target, preds)


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


## === cell 8
keker.unfreeze(model_attr="net")
layer_num = -1
keker.freeze_to(layer_num, model_attr="net")



## === cell 9
if not hasattr(keker, "kek_one_cycle"):

    def _kek_one_cycle(
        self,
        max_lr=1e-2,
        cycle_len=5,
        momentum_range=(0.95, 0.85),
        div_factor=25,
        increase_fraction=0.3,
        logdir=None,
    ):
        device = next(self.model.parameters()).device

        if not any(p.requires_grad for p in self.model.parameters()):
            self.unfreeze(model_attr="net")

        self.model.train()

        base_lr = float(max_lr) / float(div_factor)
        for pg in self.optimizer.param_groups:
            pg["lr"] = base_lr
            if "momentum" in pg:
                pg["momentum"] = float(momentum_range[0])

        def _lin(a, b, t):
            return a + (b - a) * t

        total_steps = max(1, int(cycle_len) * len(self.dataowner.train_dl))
        inc_steps = max(1, int(total_steps * float(increase_fraction)))
        dec_steps = max(1, total_steps - inc_steps)

        global_step = 0
        for epoch in range(int(cycle_len)):
            self.model.train()
            for batch in self.dataowner.train_dl:
                global_step += 1

                if global_step <= inc_steps:
                    t = (global_step - 1) / max(1, inc_steps - 1)
                    lr = _lin(base_lr, float(max_lr), t)
                    mom = _lin(float(momentum_range[0]), float(momentum_range[1]), t)
                else:
                    t = (global_step - inc_steps - 1) / max(1, dec_steps - 1)
                    lr = _lin(float(max_lr), base_lr, t)
                    mom = _lin(float(momentum_range[1]), float(momentum_range[0]), t)

                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr
                    if "momentum" in pg:
                        pg["momentum"] = mom

                self.optimizer.zero_grad(set_to_none=True)

                preds = self.step_fn(self.model, batch)
                target = batch[self.target_key].to(device)
                loss = self.criterion(preds, target)

                loss.backward()
                self.optimizer.step()

            self.model.eval()
            val_losses = []
            metric_sums = {k: 0.0 for k in self.metrics.keys()}
            metric_counts = 0

            with torch.no_grad():
                for batch in self.dataowner.val_dl:
                    preds = self.step_fn(self.model, batch)
                    target = batch[self.target_key].to(device)
                    vloss = self.criterion(preds, target)
                    val_losses.append(float(vloss.detach().cpu().item()))
                    for name, fn in self.metrics.items():
                        metric_sums[name] += float(fn(target, preds))
                    metric_counts += 1

            if metric_counts > 0:
                avg_metrics = {k: v / metric_counts for k, v in metric_sums.items()}
            else:
                avg_metrics = {k: float("nan") for k in metric_sums.keys()}
            avg_vloss = float(np.mean(val_losses)) if val_losses else float("nan")
            print(
                f"Epoch {epoch+1}/{cycle_len} - val_loss: {avg_vloss:.5f} "
                + " ".join([f"{k}: {v:.5f}" for k, v in avg_metrics.items()])
            )
            self.model.train()

        return self

    keker.kek_one_cycle = _kek_one_cycle.__get__(keker, keker.__class__)

keker.kek_one_cycle(
    max_lr=1e-2,
    cycle_len=5,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.3,
    logdir="train_logs",
)


## === cell 10
keker.kek_one_cycle(
    max_lr=1e-3,
    cycle_len=3,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.3,
    logdir="train_logs1",
)



## === cell 11
preds = keker.predict_loader(loader=test_dl)



## --- ERROR in cell 11, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3563403150.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# Base prediction (not used directly later, but kept to preserve original flow)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mpreds[0m [0;34m=[0m [0mkeker[0m[0;34m.[0m[0mpredict_loader[0m[0;34m([0m[0mloader[0m[0;34m=[0m[0mtest_dl[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0;34m[0m[0m

[0;31mAttributeError[0m: 'Keker' object has no attribute 'predict_loader'

## === cell 12
flip_ = albumentations.HorizontalFlip(always_apply=True)
transpose_ = albumentations.Transpose(always_apply=True)


def insert_aug(aug, dataset_key="image", size=32):
    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))
    AUGS = Transformer(dataset_key, lambda x: aug(image=x)["image"])
    NRM_TFMS = transforms.Compose(
        [Transformer(dataset_key, to_torch()), Transformer(dataset_key, normalize())]
    )
    tfm = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
    return tfm


flip = insert_aug(flip_, size=IMG_SIZE)
transpose = insert_aug(transpose_, size=IMG_SIZE)

tta_tfms = {"flip": flip, "transpose": transpose}

keker.TTA(loader=test_dl, tfms=tta_tfms, savedir="tta_preds1", prefix="preds")
