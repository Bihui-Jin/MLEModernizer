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

from torch.utils.data import DataLoader
from PIL import Image

plt.rcParams["figure.figsize"] = (10, 4)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
import albumentations

try:
    import pretrainedmodels  # noqa: F401
except ModuleNotFoundError:
    import torchvision.models as pretrainedmodels  # fallback alias

try:
    import adabound  # noqa: F401
except ModuleNotFoundError:
    adabound = None

try:
    from kekas import Keker, DataOwner, DataKek
    from kekas.transformations import Transformer, to_torch, normalize
    from kekas.metrics import accuracy
    from kekas.modules import Flatten
    from kekas.utils import DotDict
except ModuleNotFoundError:
    Keker = DataOwner = DataKek = None
    Transformer = to_torch = normalize = None
    accuracy = None
    Flatten = None
    DotDict = None


## === cell 2
BASE = "../input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")

labels = pd.read_csv(TRAIN_CSV)
labels["has_cactus"] = labels["has_cactus"].astype(int)
labels["data_type"] = "train"
labels.head()



## === cell 3
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
except Exception as e:
    print("Skipping visualization:", repr(e))



## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
test_df = sample_sub[["id"]].copy()
test_df["has_cactus"] = -1
test_df["data_type"] = "test"
test_df.head()



## === cell 5
labels.loc[labels["data_type"] == "train", "has_cactus"].value_counts()



## === cell 6
train, valid = train_test_split(
    labels, stratify=labels.has_cactus, test_size=0.2, random_state=42
)




## === cell 7
def reader_fn(i, row):
    if row["data_type"] == "train":
        path = os.path.join(TRAIN_DIR, row["id"])
    else:
        path = os.path.join(TEST_DIR, row["id"])
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = image[:, :, ::-1]  # BGR -> RGB
    label = torch.Tensor([row["has_cactus"]])
    return {"image": image, "label": label}




## === cell 8
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




## === cell 9


def get_transforms(dataset_key, size, p):
    global Transformer, to_torch, normalize

    if Transformer is None or to_torch is None or normalize is None:

        class _DictKeyTransform:
            def __init__(self, key, fn):
                self.key = key
                self.fn = fn

            def __call__(self, sample):
                sample[self.key] = self.fn(sample[self.key])
                return sample

        def _to_torch():
            def _fn(x):
                if isinstance(x, torch.Tensor):
                    t = x
                else:
                    t = torch.from_numpy(np.asarray(x))
                if t.ndim == 3 and t.shape[-1] in (1, 3):
                    t = t.permute(2, 0, 1)
                t = t.contiguous().float()
                if t.max() > 1.0:
                    t = t / 255.0
                return t

            return _fn

        def _normalize():
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

            def _fn(t):
                if not isinstance(t, torch.Tensor):
                    t = torch.as_tensor(t)
                return (t - mean.to(t.device, dtype=t.dtype)) / std.to(
                    t.device, dtype=t.dtype
                )

            return _fn

        Transformer = _DictKeyTransform
        to_torch = _to_torch
        normalize = _normalize

    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))
    AUGS = Transformer(dataset_key, lambda x: augs(p=p)(image=x)["image"])

    NRM_TFMS = transforms.Compose(
        [Transformer(dataset_key, to_torch()), Transformer(dataset_key, normalize())]
    )

    train_tfms = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
    val_tfms = transforms.Compose([PRE_TFMS, NRM_TFMS])

    return train_tfms, val_tfms


train_tfms, val_tfms = get_transforms("image", 32, 0.5)


## === cell 10
if DataKek is None:
    from torch.utils.data import Dataset

    class DataKek(Dataset):
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


train_dk = DataKek(
    df=train.reset_index(drop=True), reader_fn=reader_fn, transforms=train_tfms
)
val_dk = DataKek(
    df=valid.reset_index(drop=True), reader_fn=reader_fn, transforms=val_tfms
)

batch_size = 64
workers = 0

train_dl = DataLoader(
    train_dk, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True
)
val_dl = DataLoader(val_dk, batch_size=batch_size, num_workers=workers, shuffle=False)


## === cell 11
test_dk = DataKek(
    df=test_df.reset_index(drop=True), reader_fn=reader_fn, transforms=val_tfms
)
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
if Keker is None:

    class Keker:
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

        def unfreeze(self, model_attr=None):
            module = getattr(self.model, model_attr) if model_attr else self.model
            for p in module.parameters():
                p.requires_grad = True

        def freeze_to(self, layer_num, model_attr=None):
            module = getattr(self.model, model_attr) if model_attr else self.model
            children = list(module.children())
            if not children:
                return
            for i, child in enumerate(children):
                if i <= layer_num:
                    for p in child.parameters():
                        p.requires_grad = False


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
keker.kek_one_cycle(
    max_lr=1e-2,
    cycle_len=5,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.3,
    logdir="train_logs",
)



## --- ERROR in cell 18, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1912463346.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m keker.kek_one_cycle(
[0m[1;32m      2[0m     [0mmax_lr[0m[0;34m=[0m[0;36m1e-2[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m     [0mcycle_len[0m[0;34m=[0m[0;36m5[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0mmomentum_range[0m[0;34m=[0m[0;34m([0m[0;36m0.95[0m[0;34m,[0m [0;36m0.85[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m     [0mdiv_factor[0m[0;34m=[0m[0;36m25[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'Keker' object has no attribute 'kek_one_cycle'

## === cell 19
keker.kek_one_cycle(
    max_lr=1e-3,
    cycle_len=3,
    momentum_range=(0.95, 0.85),
    div_factor=25,
    increase_fraction=0.2,
    logdir="train_logs1",
)
