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
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score


SEED = 42
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
    from albumentations import torch as AT  # older albumentations
except Exception:
    try:
        from albumentations import pytorch as AT  # newer location in some versions
    except Exception:
        AT = None

try:
    import pretrainedmodels
except ModuleNotFoundError:
    pretrainedmodels = None

try:
    import adabound
except ModuleNotFoundError:
    adabound = None

try:
    from kekas import Keker, DataOwner, DataKek
    from kekas.transformations import Transformer, to_torch, normalize
    from kekas.modules import Flatten, AdaptiveConcatPool2d
except ModuleNotFoundError:
    Keker = DataOwner = DataKek = None
    Transformer = to_torch = normalize = None
    Flatten = AdaptiveConcatPool2d = None


## === cell 2
labels = pd.read_csv(TRAIN_CSV)
labels["has_cactus"] = labels["has_cactus"].astype(int)
labels["data_type"] = "train"




## === cell 3
test_img = sorted(os.listdir(TEST_DIR))
test_df = pd.DataFrame(test_img, columns=["id"])
test_df["has_cactus"] = -1
test_df["data_type"] = "test"



## === cell 4
_ = labels["has_cactus"].value_counts()



## === cell 5
train, valid = train_test_split(
    labels, stratify=labels.has_cactus, test_size=0.2, random_state=SEED
)




## === cell 6
def reader_fn(i, row):
    if row["data_type"] == "train":
        path = os.path.join(TRAIN_DIR, row["id"])
    else:
        path = os.path.join(TEST_DIR, row["id"])
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    image = image[:, :, ::-1]  # BGR -> RGB
    label = torch.tensor([float(row["has_cactus"])], dtype=torch.float32)
    return {"image": image, "label": label}




## === cell 7
def augs(p=0.5):
    return albumentations.Compose([albumentations.HorizontalFlip()], p=p)




## === cell 8
def get_transforms(dataset_key, size, p):
    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))
    AUGS = Transformer(dataset_key, lambda x: augs(p=p)(image=x)["image"])
    NRM_TFMS = transforms.Compose(
        [Transformer(dataset_key, to_torch()), Transformer(dataset_key, normalize())]
    )

    train_tfms = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
    val_tfms = transforms.Compose([PRE_TFMS, NRM_TFMS])
    return train_tfms, val_tfms




## === cell 9
def get_transforms(dataset_key, size, p):
    if Transformer is None or to_torch is None or normalize is None:

        class _ApplyToKey:
            def __init__(self, key, fn):
                self.key = key
                self.fn = fn

            def __call__(self, sample):
                sample[self.key] = self.fn(sample[self.key])
                return sample

        def _to_torch_fn(x):
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            x = x.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
            return torch.from_numpy(x)

        def _normalize_fn(x):
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device)[
                :, None, None
            ]
            std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device)[
                :, None, None
            ]
            return (x - mean) / std

        PRE_TFMS = _ApplyToKey(dataset_key, lambda x: cv2.resize(x, (size, size)))
        AUGS = _ApplyToKey(dataset_key, lambda x: augs(p=p)(image=x)["image"])
        NRM_TFMS = transforms.Compose(
            [
                _ApplyToKey(dataset_key, _to_torch_fn),
                _ApplyToKey(dataset_key, _normalize_fn),
            ]
        )

        train_tfms = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
        val_tfms = transforms.Compose([PRE_TFMS, NRM_TFMS])
        return train_tfms, val_tfms

    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))
    AUGS = Transformer(dataset_key, lambda x: augs(p=p)(image=x)["image"])
    NRM_TFMS = transforms.Compose(
        [Transformer(dataset_key, to_torch()), Transformer(dataset_key, normalize())]
    )

    train_tfms = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
    val_tfms = transforms.Compose([PRE_TFMS, NRM_TFMS])
    return train_tfms, val_tfms


## === cell 10
if DataKek is None:
    from torch.utils.data import Dataset

    class _FallbackDataKek(Dataset):
        def __init__(self, df, reader_fn, transforms=None):
            self.df = df.reset_index(drop=True)
            self.reader_fn = reader_fn
            self.transforms = transforms

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            row = self.df.iloc[i]
            sample = self.reader_fn(i, row)
            if self.transforms is not None:
                sample = self.transforms(sample)
            return sample

    DataKek = _FallbackDataKek  # keep the same name for compatibility with cell 11

train_tfms, val_tfms = get_transforms(dataset_key="image", size=32, p=0.5)

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




## === cell 13
if DataOwner is None:

    class DataOwner:
        def __init__(self, train_loader, val_loader=None, test_loader=None):
            self.train_dl = train_loader
            self.val_dl = val_loader
            self.test_dl = test_loader


if Flatten is None:

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)


if AdaptiveConcatPool2d is None:

    class AdaptiveConcatPool2d(nn.Module):
        def __init__(self, size=1):
            super().__init__()
            self.ap = nn.AdaptiveAvgPool2d(size)
            self.mp = nn.AdaptiveMaxPool2d(size)

        def forward(self, x):
            return torch.cat([self.mp(x), self.ap(x)], dim=1)


if pretrainedmodels is None:
    import torchvision.models as tvm

    class Net(nn.Module):
        def __init__(
            self,
            num_classes: int,
            p: float = 0.2,
            pooling_size: int = 2,
            last_conv_size: int = 81536,  # kept for signature compatibility; unused
            arch: str = "densenet169",
            pretrained: str = "imagenet",
        ) -> None:
            super().__init__()

            if arch != "densenet169":
                raise ValueError(
                    f"Fallback Net only supports arch='densenet169' when pretrainedmodels is unavailable; got: {arch}"
                )

            weights = None
            if pretrained in ("imagenet", "imagenet+5k"):
                try:
                    weights = tvm.DenseNet169_Weights.IMAGENET1K_V1
                except Exception:
                    weights = "IMAGENET1K_V1"

            net = tvm.densenet169(weights=weights)

            features = net.features

            out_ch = net.classifier.in_features
            head_in = (
                out_ch * 2 * pooling_size * pooling_size
            )  # concat pool doubles channels

            self.net = nn.Sequential(
                features,
                nn.ReLU(inplace=True),
                AdaptiveConcatPool2d(size=pooling_size),
                Flatten(),
                nn.BatchNorm1d(head_in),
                nn.Dropout(p),
                nn.Linear(head_in, num_classes),
            )

        def forward(self, x):
            logits = self.net(x)
            return logits


dataowner = DataOwner(train_dl, val_dl, None)
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
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4229198289.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m keker = Keker(
[0m[1;32m      2[0m     [0mmodel[0m[0;34m=[0m[0mmodel[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m     [0mdataowner[0m[0;34m=[0m[0mdataowner[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0mcriterion[0m[0;34m=[0m[0mcriterion[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m     [0mstep_fn[0m[0;34m=[0m[0mstep_fn[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: 'NoneType' object is not callable

## === cell 17
keker.unfreeze(model_attr="net")
layer_num = -1
keker.freeze_to(layer_num, model_attr="net")
