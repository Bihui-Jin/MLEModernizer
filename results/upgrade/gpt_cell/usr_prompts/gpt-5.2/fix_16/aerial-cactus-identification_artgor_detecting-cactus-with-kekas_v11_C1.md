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

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_CANDIDATES = [
    "/kaggle/input/aerial-cactus-identification",
    "../input/aerial-cactus-identification",
    "../input",
]
BASE_PATH = None
for p in BASE_CANDIDATES:
    if os.path.exists(p):
        if (
            os.path.exists(os.path.join(p, "train.csv"))
            and os.path.exists(os.path.join(p, "train"))
            and os.path.exists(os.path.join(p, "test"))
        ):
            BASE_PATH = p
            break

if BASE_PATH is None:
    raise FileNotFoundError(
        "Could not locate dataset root. Expected one of: "
        + ", ".join(BASE_CANDIDATES)
        + " containing train.csv, train/, test/."
    )

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")

print("Using BASE_PATH:", BASE_PATH)
print("TRAIN_CSV:", TRAIN_CSV)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)



## === cell 1
import albumentations
from albumentations.pytorch import ToTensorV2


class _ATShim:
    ToTensor = ToTensorV2


AT = _ATShim()

import types

try:
    import pretrainedmodels  # type: ignore
except ModuleNotFoundError:
    import torchvision.models as tvm

    def _get_tv_model(name, num_classes=1000, pretrained="imagenet"):
        tv_name = name
        aliases = {
            "resnet18": "resnet18",
            "resnet34": "resnet34",
            "resnet50": "resnet50",
            "resnet101": "resnet101",
            "resnet152": "resnet152",
            "densenet121": "densenet121",
            "densenet169": "densenet169",
            "densenet201": "densenet201",
            "mobilenetv2": "mobilenet_v2",
            "mobilenet_v2": "mobilenet_v2",
            "efficientnet_b0": "efficientnet_b0",
            "efficientnet_b1": "efficientnet_b1",
            "efficientnet_b2": "efficientnet_b2",
            "efficientnet_b3": "efficientnet_b3",
        }
        tv_name = aliases.get(name, name)
        if not hasattr(tvm, tv_name):
            raise ValueError(
                f"Unknown model name '{name}' (mapped to '{tv_name}') for torchvision shim."
            )

        weights = None
        if pretrained in ("imagenet", True, "true", "True"):
            try:
                weights_enum = getattr(tvm, f"{tv_name.upper()}_Weights", None)
                if weights_enum is not None:
                    weights = weights_enum.DEFAULT
            except Exception:
                weights = None

        model_fn = getattr(tvm, tv_name)
        model = (
            model_fn(weights=weights)
            if "weights" in model_fn.__code__.co_varnames
            else model_fn(pretrained=bool(weights))
        )
        if (
            num_classes is not None
            and hasattr(model, "fc")
            and getattr(model.fc, "out_features", None) != num_classes
        ):
            import torch.nn as nn

            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, num_classes)
        if num_classes is not None and hasattr(model, "classifier"):
            import torch.nn as nn

            if (
                isinstance(model.classifier, nn.Linear)
                and model.classifier.out_features != num_classes
            ):
                in_f = model.classifier.in_features
                model.classifier = nn.Linear(in_f, num_classes)
            elif isinstance(model.classifier, nn.Sequential):
                for i in reversed(range(len(model.classifier))):
                    if isinstance(model.classifier[i], nn.Linear):
                        if model.classifier[i].out_features != num_classes:
                            in_f = model.classifier[i].in_features
                            model.classifier[i] = nn.Linear(in_f, num_classes)
                        break
        return model

    pretrainedmodels = types.SimpleNamespace()
    for _name in dir(tvm):
        if _name.startswith("_"):
            continue
        obj = getattr(tvm, _name)
        if callable(obj):
            setattr(pretrainedmodels, _name, obj)

    for arch in [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "densenet121",
        "densenet169",
        "densenet201",
        "mobilenetv2",
        "mobilenet_v2",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
    ]:
        setattr(
            pretrainedmodels,
            arch,
            (lambda a=arch: (lambda **kw: _get_tv_model(a, **kw)))(),
        )

try:
    import adabound  # type: ignore
except ModuleNotFoundError:
    import torch

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
        ):
            super().__init__(
                params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
            )
            self.final_lr = final_lr
            self.gamma = gamma

    adabound = types.SimpleNamespace(AdaBound=_AdaBound)

from torch.utils.data import DataLoader

try:
    from kekas import Keker, DataOwner, DataKek  # type: ignore
    from kekas.transformations import Transformer, to_torch, normalize  # type: ignore
    from kekas.modules import Flatten  # type: ignore
except ModuleNotFoundError:
    import torch
    import torch.nn as nn

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)

    def to_torch(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x)

    def normalize(mean, std):
        mean_t = torch.tensor(mean).view(1, -1, 1, 1).float()
        std_t = torch.tensor(std).view(1, -1, 1, 1).float()

        def _norm(batch):
            return (batch - mean_t.to(batch.device)) / std_t.to(batch.device)

        return _norm

    class Transformer:
        def __init__(self, *funcs):
            self.funcs = funcs

        def __call__(self, x):
            for f in self.funcs:
                x = f(x)
            return x

    class DataOwner:
        def __init__(self, train_dl=None, val_dl=None):
            self.train_dl = train_dl
            self.val_dl = val_dl

    class DataKek:
        def __init__(self, dl):
            self.dl = dl

    class Keker:
        def __init__(self, model, optimizer=None, criterion=None, device=None):
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            self.device = device or (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.model.to(self.device)

        def fit(self, dataowner, epochs=1):
            self.model.train()
            for _ in range(epochs):
                for xb, yb in dataowner.train_dl:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    self.optimizer.zero_grad()
                    out = self.model(xb)
                    loss = self.criterion(out, yb)
                    loss.backward()
                    self.optimizer.step()

        @torch.no_grad()
        def predict(self, dl):
            self.model.eval()
            outs = []
            for xb in dl:
                if isinstance(xb, (tuple, list)):
                    xb = xb[0]
                xb = xb.to(self.device)
                outs.append(self.model(xb).detach().cpu())
            return torch.cat(outs, dim=0)


## === cell 2

labels = pd.read_csv(TRAIN_CSV)
labels["has_cactus"] = labels["has_cactus"].astype(int)
labels["data_type"] = "train"
labels.head()



## === cell 3
test_img = sorted(os.listdir(TEST_DIR))
test_df = pd.DataFrame(test_img, columns=["id"])
test_df["has_cactus"] = -1
test_df["data_type"] = "test"

labels.head(), test_df.head()



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
        raise FileNotFoundError(f"Image not found or unreadable: {img_path}")

    image = image[:, :, ::-1]  # BGR -> RGB
    label = torch.tensor([float(row["has_cactus"])], dtype=torch.float32)
    return {"image": image, "label": label}




## === cell 7
def augs(p=0.5):
    return albumentations.Compose(
        [
            albumentations.HorizontalFlip(),
            albumentations.VerticalFlip(),
            albumentations.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.10, rotate_limit=15, p=0.75
            ),
            albumentations.HueSaturationValue(),
            albumentations.RandomBrightness(),
            albumentations.RandomContrast(),
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
def get_transforms(dataset_key, size, p):
    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))
    AUGS = Transformer(dataset_key, lambda x: augs(p=p)(image=x)["image"])

    NRM_TFMS = transforms.Compose(
        [
            Transformer(dataset_key, to_torch),
            Transformer(
                dataset_key, normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ),
        ]
    )

    train_tfms = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
    val_tfms = transforms.Compose([PRE_TFMS, NRM_TFMS])

    return train_tfms, val_tfms


## === cell 10

size = 32
p = 0.5

try:
    train_tfms, val_tfms = get_transforms("image", size=size, p=p)
except Exception:
    train_tfms = lambda x: x
    val_tfms = lambda x: x


def _make_datakek_compatible(df, reader_fn, transforms):
    try:
        return DataKek(df=df, reader_fn=reader_fn, transforms=transforms)
    except TypeError:
        from torch.utils.data import Dataset

        class _DFDataset(Dataset):
            def __init__(self, df_, reader_fn_, transforms_):
                self.df = df_.reset_index(drop=True)
                self.reader_fn = reader_fn_
                self.transforms = transforms_

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx].to_dict()
                sample = self.reader_fn(idx, row)

                if self.transforms is not None:
                    sample["image"] = self.transforms(sample["image"])

                x = sample["image"]
                y = sample.get("label", None)

                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).permute(2, 0, 1).contiguous()
                if isinstance(x, torch.Tensor) and x.dtype != torch.float32:
                    x = x.float()

                if y is None:
                    return x
                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y)
                return x, y

        return _DFDataset(df, reader_fn, transforms)


train_dk = _make_datakek_compatible(train, reader_fn=reader_fn, transforms=train_tfms)
val_dk = _make_datakek_compatible(valid, reader_fn=reader_fn, transforms=val_tfms)

batch_size = 64
workers = 0

train_dl = DataLoader(
    train_dk, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True
)
val_dl = DataLoader(val_dk, batch_size=batch_size, num_workers=workers, shuffle=False)


## === cell 11
test_dk = _make_datakek_compatible(test_df, reader_fn=reader_fn, transforms=val_tfms)
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
import albumentations
from albumentations.pytorch import ToTensorV2


class _ATShim:
    ToTensor = ToTensorV2


AT = _ATShim()

import types

try:
    import pretrainedmodels  # type: ignore
except ModuleNotFoundError:
    import torchvision.models as tvm

    def _get_tv_model(name, num_classes=1000, pretrained="imagenet"):
        tv_name = name
        aliases = {
            "resnet18": "resnet18",
            "resnet34": "resnet34",
            "resnet50": "resnet50",
            "resnet101": "resnet101",
            "resnet152": "resnet152",
            "densenet121": "densenet121",
            "densenet169": "densenet169",
            "densenet201": "densenet201",
            "mobilenetv2": "mobilenet_v2",
            "mobilenet_v2": "mobilenet_v2",
            "efficientnet_b0": "efficientnet_b0",
            "efficientnet_b1": "efficientnet_b1",
            "efficientnet_b2": "efficientnet_b2",
            "efficientnet_b3": "efficientnet_b3",
        }
        tv_name = aliases.get(name, name)
        if not hasattr(tvm, tv_name):
            raise ValueError(
                f"Unknown model name '{name}' (mapped to '{tv_name}') for torchvision shim."
            )

        weights = None
        if pretrained in ("imagenet", True, "true", "True"):
            try:
                weights_enum = getattr(tvm, f"{tv_name.upper()}_Weights", None)
                if weights_enum is not None:
                    weights = weights_enum.DEFAULT
            except Exception:
                weights = None

        model_fn = getattr(tvm, tv_name)
        model = (
            model_fn(weights=weights)
            if "weights" in model_fn.__code__.co_varnames
            else model_fn(pretrained=bool(weights))
        )
        if (
            num_classes is not None
            and hasattr(model, "fc")
            and getattr(model.fc, "out_features", None) != num_classes
        ):
            import torch.nn as nn

            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, num_classes)
        if num_classes is not None and hasattr(model, "classifier"):
            import torch.nn as nn

            if (
                isinstance(model.classifier, nn.Linear)
                and model.classifier.out_features != num_classes
            ):
                in_f = model.classifier.in_features
                model.classifier = nn.Linear(in_f, num_classes)
            elif isinstance(model.classifier, nn.Sequential):
                for i in reversed(range(len(model.classifier))):
                    if isinstance(model.classifier[i], nn.Linear):
                        if model.classifier[i].out_features != num_classes:
                            in_f = model.classifier[i].in_features
                            model.classifier[i] = nn.Linear(in_f, num_classes)
                        break
        return model

    pretrainedmodels = types.SimpleNamespace()
    for _name in dir(tvm):
        if _name.startswith("_"):
            continue
        obj = getattr(tvm, _name)
        if callable(obj):
            setattr(pretrainedmodels, _name, obj)

    for arch in [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "densenet121",
        "densenet169",
        "densenet201",
        "mobilenetv2",
        "mobilenet_v2",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
    ]:
        setattr(
            pretrainedmodels,
            arch,
            (lambda a=arch: (lambda **kw: _get_tv_model(a, **kw)))(),
        )

try:
    import adabound  # type: ignore
except ModuleNotFoundError:
    import torch

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
        ):
            super().__init__(
                params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
            )
            self.final_lr = final_lr
            self.gamma = gamma

    adabound = types.SimpleNamespace(AdaBound=_AdaBound)

from torch.utils.data import DataLoader

try:
    from kekas import Keker, DataOwner, DataKek  # type: ignore
    from kekas.transformations import Transformer, to_torch, normalize  # type: ignore
    from kekas.modules import Flatten  # type: ignore
except ModuleNotFoundError:
    import torch
    import torch.nn as nn

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)

    def to_torch(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x)

    def normalize(mean, std):
        mean_t = torch.tensor(mean).view(1, -1, 1, 1).float()
        std_t = torch.tensor(std).view(1, -1, 1, 1).float()

        def _norm(batch):
            return (batch - mean_t.to(batch.device)) / std_t.to(batch.device)

        return _norm

    class Transformer:
        def __init__(self, *funcs):
            self.funcs = funcs

        def __call__(self, x):
            for f in self.funcs:
                x = f(x)
            return x

    class DataOwner:
        def __init__(self, train_dl=None, val_dl=None, test_dl=None, *args, **kwargs):
            self.train_dl = train_dl
            self.val_dl = val_dl
            self.test_dl = test_dl

    class DataKek:
        def __init__(self, dl):
            self.dl = dl

    class Keker:
        def __init__(self, model, optimizer=None, criterion=None, device=None):
            self.model = model
            self.optimizer = optimizer
            self.criterion = criterion
            self.device = device or (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.model.to(self.device)

        def fit(self, dataowner, epochs=1):
            self.model.train()
            for _ in range(epochs):
                for xb, yb in dataowner.train_dl:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    self.optimizer.zero_grad()
                    out = self.model(xb)
                    loss = self.criterion(out, yb)
                    loss.backward()
                    self.optimizer.step()

        @torch.no_grad()
        def predict(self, dl):
            self.model.eval()
            outs = []
            for xb in dl:
                if isinstance(xb, (tuple, list)):
                    xb = xb[0]
                xb = xb.to(self.device)
                outs.append(self.model(xb).detach().cpu())
            return torch.cat(outs, dim=0)


## === cell 14
def step_fn(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    inp = batch["image"].to(DEVICE)
    return model(inp)




## === cell 15
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




## === cell 16
model = Net(num_classes=1).to(DEVICE)

dataowner = DataOwner(train_dl=train_dl, val_dl=val_dl)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
keker = Keker(model=model, optimizer=optimizer, criterion=criterion, device=DEVICE)


## === cell 17
if not hasattr(keker, "unfreeze"):

    def _unfreeze(self, model_attr=None):
        m = getattr(self.model, model_attr) if model_attr is not None else self.model
        for p in m.parameters():
            p.requires_grad = True
        return self

    def _freeze_to(self, layer_num, model_attr=None):
        m = getattr(self.model, model_attr) if model_attr is not None else self.model

        children = list(m.children())
        if len(children) == 0:
            for p in m.parameters():
                p.requires_grad = False
            return self

        freeze_children = children[: layer_num + 1]
        unfreeze_children = children[layer_num + 1 :]

        for ch in freeze_children:
            for p in ch.parameters():
                p.requires_grad = False
        for ch in unfreeze_children:
            for p in ch.parameters():
                p.requires_grad = True
        return self

    import types

    keker.unfreeze = types.MethodType(_unfreeze, keker)
    keker.freeze_to = types.MethodType(_freeze_to, keker)

keker.unfreeze(model_attr="net")
layer_num = -1
keker.freeze_to(layer_num, model_attr="net")


## === cell 18
import types

os.makedirs("train_logs", exist_ok=True)


def _wrap_tfms_drop_noncallables(tfms):
    if tfms is None or not callable(tfms):
        return tfms

    def _safe_call(x):
        try:
            return tfms(x)
        except TypeError as e:
            if "'str' object is not callable" not in str(e):
                raise
            y = x
            pipeline = getattr(tfms, "transforms", None)
            if pipeline is None:
                pipeline = [tfms]
            for t in pipeline:
                if not callable(t):
                    continue
                funcs = getattr(t, "funcs", None)
                if funcs is not None:
                    for f in funcs:
                        if callable(f):
                            y = f(y)
                else:
                    y = t(y)
            return y

    return _safe_call


train_tfms = _wrap_tfms_drop_noncallables(train_tfms)
val_tfms = _wrap_tfms_drop_noncallables(val_tfms)

if "Transformer" in globals() and hasattr(Transformer, "__init__"):
    _old_transformer_init = Transformer.__init__

    def _transformer_init_drop_noncallables(self, *funcs):
        self.funcs = tuple(f for f in funcs if callable(f))

    try:
        tmp = Transformer.__new__(Transformer)
        _old_transformer_init(tmp, "image", lambda x: x)
        _needs_patch = any(not callable(f) for f in getattr(tmp, "funcs", ()))
    except Exception:
        _needs_patch = True

    if _needs_patch:
        Transformer.__init__ = _transformer_init_drop_noncallables

if not hasattr(keker, "kek_one_cycle"):

    def _kek_one_cycle(
        self,
        max_lr,
        cycle_len,
        momentum_range=(0.95, 0.85),
        div_factor=25,
        increase_fraction=0.3,
        logdir=None,
        *args,
        **kwargs,
    ):
        if "dataowner" in kwargs and kwargs["dataowner"] is not None:
            dataowner = kwargs["dataowner"]
        else:
            dataowner = globals().get("dataowner", None)
            if dataowner is None:
                raise NameError(
                    "dataowner is not defined; cannot run kek_one_cycle fallback."
                )
        return self.fit(dataowner, epochs=int(cycle_len))

    keker.kek_one_cycle = types.MethodType(_kek_one_cycle, keker)

keker.kek_one_cycle(
    max_lr=1e-2,
    cycle_len=4,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.3,
    logdir="train_logs",
)


## --- ERROR in cell 18, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2666437720.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     85[0m     [0mkeker[0m[0;34m.[0m[0mkek_one_cycle[0m [0;34m=[0m [0mtypes[0m[0;34m.[0m[0mMethodType[0m[0;34m([0m[0m_kek_one_cycle[0m[0;34m,[0m [0mkeker[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     86[0m [0;34m[0m[0m
[0;32m---> 87[0;31m keker.kek_one_cycle(
[0m[1;32m     88[0m     [0mmax_lr[0m[0;34m=[0m[0;36m1e-2[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     89[0m     [0mcycle_len[0m[0;34m=[0m[0;36m4[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2666437720.py[0m in [0;36m_kek_one_cycle[0;34m(self, max_lr, cycle_len, momentum_range, div_factor, increase_fraction, logdir, *args, **kwargs)[0m
[1;32m     81[0m                     [0;34m"dataowner is not defined; cannot run kek_one_cycle fallback."[0m[0;34m[0m[0;34m[0m[0m
[1;32m     82[0m                 )
[0;32m---> 83[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mdataowner[0m[0;34m,[0m [0mepochs[0m[0;34m=[0m[0mint[0m[0;34m([0m[0mcycle_len[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     84[0m [0;34m[0m[0m
[1;32m     85[0m     [0mkeker[0m[0;34m.[0m[0mkek_one_cycle[0m [0;34m=[0m [0mtypes[0m[0;34m.[0m[0mMethodType[0m[0;34m([0m[0m_kek_one_cycle[0m[0;34m,[0m [0mkeker[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/558610696.py[0m in [0;36mfit[0;34m(self, dataowner, epochs)[0m
[1;32m    200[0m             [0mself[0m[0;34m.[0m[0mmodel[0m[0;34m.[0m[0mtrain[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    201[0m             [0;32mfor[0m [0m_[0m [0;32min[0m [0mrange[0m[0;34m([0m[0mepochs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 202[0;31m                 [0;32mfor[0m [0mxb[0m[0;34m,[0m [0myb[0m [0;32min[0m [0mdataowner[0m[0;34m.[0m[0mtrain_dl[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    203[0m                     [0mxb[0m [0;34m=[0m [0mxb[0m[0;34m.[0m[0mto[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdevice[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    204[0m                     [0myb[0m [0;34m=[0m [0myb[0m[0;34m.[0m[0mto[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdevice[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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

[0;32m/tmp/ipykernel_11/2608578159.py[0m in [0;36m__getitem__[0;34m(self, idx)[0m
[1;32m     32[0m [0;34m[0m[0m
[1;32m     33[0m                 [0;32mif[0m [0mself[0m[0;34m.[0m[0mtransforms[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 34[0;31m                     [0msample[0m[0;34m[[0m[0;34m"image"[0m[0;34m][0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mtransforms[0m[0;34m([0m[0msample[0m[0;34m[[0m[0;34m"image"[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     35[0m [0;34m[0m[0m
[1;32m     36[0m                 [0mx[0m [0;34m=[0m [0msample[0m[0;34m[[0m[0;34m"image"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py[0m in [0;36m__call__[0;34m(self, img)[0m
[1;32m     93[0m     [0;32mdef[0m [0m__call__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mimg[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     94[0m         [0;32mfor[0m [0mt[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mtransforms[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 95[0;31m             [0mimg[0m [0;34m=[0m [0mt[0m[0;34m([0m[0mimg[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     96[0m         [0;32mreturn[0m [0mimg[0m[0;34m[0m[0;34m[0m[0m
[1;32m     97[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/606360043.py[0m in [0;36m__call__[0;34m(self, x)[0m
[1;32m    171[0m         [0;32mdef[0m [0m__call__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mx[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    172[0m             [0;32mfor[0m [0mf[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mfuncs[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 173[0;31m                 [0mx[0m [0;34m=[0m [0mf[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    174[0m             [0;32mreturn[0m [0mx[0m[0;34m[0m[0;34m[0m[0m
[1;32m    175[0m [0;34m[0m[0m

[0;31mTypeError[0m: 'str' object is not callable

## === cell 19
os.makedirs("train_logs1", exist_ok=True)
keker.kek_one_cycle(
    max_lr=1e-3,
    cycle_len=4,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.2,
    logdir="train_logs1",
)
