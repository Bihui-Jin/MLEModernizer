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
    pretrainedmodels.__dict__ = {}
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
        pretrainedmodels.__dict__[arch] = (
            lambda a=arch: (lambda **kw: _get_tv_model(a, **kw))
        )()

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


## --- ERROR in cell 1, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3794356300.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     16[0m [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 17[0;31m     [0;32mimport[0m [0mpretrainedmodels[0m  [0;31m# type: ignore[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     18[0m [0;32mexcept[0m [0mModuleNotFoundError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'pretrainedmodels'

During handling of the above exception, another exception occurred:

[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3794356300.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    100[0m             [0msetattr[0m[0;34m([0m[0mpretrainedmodels[0m[0;34m,[0m [0m_name[0m[0;34m,[0m [0mobj[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    101[0m     [0;31m# Add common names used in pretrainedmodels[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 102[0;31m     [0mpretrainedmodels[0m[0;34m.[0m[0m__dict__[0m [0;34m=[0m [0;34m{[0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    103[0m     for arch in [
[1;32m    104[0m         [0;34m"resnet18"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: readonly attribute

## === cell 2

labels = pd.read_csv(TRAIN_CSV)
labels["has_cactus"] = labels["has_cactus"].astype(int)
labels["data_type"] = "train"
labels.head()
