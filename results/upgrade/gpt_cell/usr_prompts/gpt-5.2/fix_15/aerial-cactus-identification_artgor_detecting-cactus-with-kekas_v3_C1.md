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
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score

DATA_ROOT = "../input/aerial-cactus-identification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train", "train")
TEST_DIR = os.path.join(DATA_ROOT, "test", "test")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

train_on_gpu = True



## === cell 1
import albumentations

try:
    import albumentations.pytorch as AT  # provides ToTensorV2
except Exception:
    from albumentations import torch as AT

try:
    import pretrainedmodels  # type: ignore
except ModuleNotFoundError:
    import types
    import torchvision.models as _tv_models

    def _wrap_tv_ctor(tv_ctor):
        def _ctor(*args, **kwargs):
            pretrained = kwargs.pop("pretrained", None)
            if pretrained in (None, False, "none"):
                return tv_ctor(*args, pretrained=False, **kwargs)
            return tv_ctor(*args, pretrained=True, **kwargs)

        return _ctor

    pretrainedmodels = types.SimpleNamespace()

    _name_map = {
        "resnet18": _tv_models.resnet18,
        "resnet34": _tv_models.resnet34,
        "resnet50": _tv_models.resnet50,
        "resnet101": _tv_models.resnet101,
        "resnet152": _tv_models.resnet152,
        "densenet121": _tv_models.densenet121,
        "densenet169": _tv_models.densenet169,
        "densenet201": _tv_models.densenet201,
        "densenet161": _tv_models.densenet161,
        "inceptionv3": _tv_models.inception_v3,
        "vgg16": _tv_models.vgg16,
        "vgg19": _tv_models.vgg19,
    }

    for _n, _ctor in _name_map.items():
        setattr(pretrainedmodels, _n, _wrap_tv_ctor(_ctor))

try:
    import adabound  # type: ignore
except ModuleNotFoundError:
    import types as _types
    import torch as _torch

    class _AdaBound(_torch.optim.Adam):
        def __init__(
            self,
            params,
            lr=1e-3,
            final_lr=0.1,
            gamma=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            amsbound=False,
            **kwargs,
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

    adabound = _types.SimpleNamespace(AdaBound=_AdaBound)

try:
    from kekas import Keker, DataOwner, DataKek  # type: ignore
    from kekas.transformations import Transformer, to_torch, normalize  # type: ignore
    from kekas.modules import Flatten, AdaptiveConcatPool2d  # type: ignore
except ModuleNotFoundError:
    import torch
    import torch.nn as nn

    class _MissingKekasError(ModuleNotFoundError):
        pass

    def _raise_missing_kekas(*args, **kwargs):
        raise _MissingKekasError(
            "The 'kekas' library is not installed in this environment. "
            "Install it or remove kekas-dependent training/inference code."
        )

    class Keker:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            _raise_missing_kekas()

    class DataOwner:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            _raise_missing_kekas()

    class DataKek:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            _raise_missing_kekas()

    class Transformer:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            _raise_missing_kekas()

    def to_torch(*args, **kwargs):  # pragma: no cover
        _raise_missing_kekas()

    def normalize(*args, **kwargs):  # pragma: no cover
        _raise_missing_kekas()

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)

    class AdaptiveConcatPool2d(nn.Module):
        def __init__(self, sz=1):
            super().__init__()
            self.ap = nn.AdaptiveAvgPool2d(sz)
            self.mp = nn.AdaptiveMaxPool2d(sz)

        def forward(self, x):
            return torch.cat([self.mp(x), self.ap(x)], 1)


## === cell 2
labels = pd.read_csv(TRAIN_CSV)

if (not os.path.isdir(TRAIN_DIR)) or (len(os.listdir(TRAIN_DIR)) == 0):
    candidate_roots = [
        DATA_ROOT,
        "/kaggle/input/aerial-cactus-identification",
        "/kaggle/data/aerial-cactus-identification",
        "../input/aerial-cactus-identification",
        "../data/aerial-cactus-identification",
        "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification",
        "/kaggle/data/aerial-cactus-identification/aerial-cactus-identification",
    ]
    candidate_dirs = []
    for root in candidate_roots:
        candidate_dirs.extend(
            [
                os.path.join(root, "train", "train"),
                os.path.join(root, "train"),
            ]
        )

    for d in candidate_dirs:
        if os.path.isdir(d):
            try:
                if any(
                    f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(d)
                ):
                    TRAIN_DIR = d
                    break
            except Exception:
                pass

fig = plt.figure(figsize=(25, 8))
train_imgs = os.listdir(TRAIN_DIR)

n_show = min(20, len(train_imgs))
for idx, img in enumerate(np.random.choice(train_imgs, n_show, replace=False)):
    ax = fig.add_subplot(4, 20 // 4, idx + 1, xticks=[], yticks=[])
    im = Image.open(os.path.join(TRAIN_DIR, img))
    plt.imshow(im)
    lab = labels.loc[labels["id"] == img, "has_cactus"].values[0]
    ax.set_title(f"Label: {lab}")


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
    labels, stratify=labels.has_cactus, test_size=0.2, random_state=42
)




## === cell 6
def reader_fn(i, row):
    image_path = os.path.join(DATA_ROOT, row["data_type"], row["data_type"], row["id"])
    image = cv2.imread(image_path)[:, :, ::-1]  # BGR -> RGB
    label = torch.Tensor([row["has_cactus"]])
    return {"image": image, "label": label}




## === cell 7
def augs(p=0.5):
    return albumentations.Compose(
        [
            albumentations.HorizontalFlip(),
            albumentations.Transpose(),
            albumentations.Flip(),
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

import torch
import numpy as np

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class _ResizeDict:
    def __init__(self, key, size):
        self.key = key
        self.size = size

    def __call__(self, sample):
        img = sample[self.key]
        sample[self.key] = cv2.resize(img, (self.size, self.size))
        return sample


class _AugsDict:
    def __init__(self, key, p):
        self.key = key
        self.p = p

    def __call__(self, sample):
        img = sample[self.key]
        sample[self.key] = augs(p=self.p)(image=img)["image"]
        return sample


class _ToTensorNormalizeDict:
    def __init__(self, key):
        self.key = key

    def __call__(self, sample):
        img = sample[self.key]
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float().div_(255.0)
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        sample[self.key] = t
        return sample


class _ComposeDict:
    def __init__(self, tfms):
        self.tfms = tfms

    def __call__(self, sample):
        for t in self.tfms:
            sample = t(sample)
        return sample


_size = 32
train_tfms = _ComposeDict(
    [
        _ResizeDict("image", _size),
        _AugsDict("image", p=0.5),
        _ToTensorNormalizeDict("image"),
    ]
)
val_tfms = _ComposeDict([_ResizeDict("image", _size), _ToTensorNormalizeDict("image")])


## === cell 10
from torch.utils.data import Dataset, DataLoader


class _SimpleDictDataset(Dataset):
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


try:
    train_dk = DataKek(df=train, reader_fn=reader_fn, transforms=train_tfms)
    val_dk = DataKek(df=valid, reader_fn=reader_fn, transforms=val_tfms)
except Exception as e:
    if e.__class__.__name__ == "_MissingKekasError":
        train_dk = _SimpleDictDataset(
            df=train, reader_fn=reader_fn, transforms=train_tfms
        )
        val_dk = _SimpleDictDataset(df=valid, reader_fn=reader_fn, transforms=val_tfms)
    else:
        raise

batch_size = 64
workers = 0

train_dl = DataLoader(
    train_dk, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True
)
val_dl = DataLoader(val_dk, batch_size=batch_size, num_workers=workers, shuffle=False)


## === cell 11
try:
    test_dk = DataKek(df=test_df, reader_fn=reader_fn, transforms=val_tfms)
except Exception as e:
    if e.__class__.__name__ == "_MissingKekasError":
        test_dk = _SimpleDictDataset(
            df=test_df, reader_fn=reader_fn, transforms=val_tfms
        )
    else:
        raise

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
try:
    dataowner = DataOwner(train_dl, val_dl, None)
except Exception as e:
    if e.__class__.__name__ == "_MissingKekasError":

        class _DataOwnerFallback:
            def __init__(self, train_dl, val_dl, test_dl=None):
                self.train_dl = train_dl
                self.val_dl = val_dl
                self.test_dl = test_dl

        dataowner = _DataOwnerFallback(train_dl, val_dl, None)
    else:
        raise

try:
    _acp_params = AdaptiveConcatPool2d.__init__.__code__.co_varnames
    if "size" not in _acp_params:

        class _AdaptiveConcatPool2dCompat(AdaptiveConcatPool2d):
            def __init__(self, size=1, sz=None):
                if sz is not None:
                    size = sz
                super().__init__(sz=size)

        AdaptiveConcatPool2d = _AdaptiveConcatPool2dCompat
except Exception:
    pass

model = Net(num_classes=1)
criterion = nn.BCEWithLogitsLoss()


## === cell 14
def step_fn(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    inp = batch["image"]
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
except Exception as e:
    if e.__class__.__name__ == "_MissingKekasError":

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

            def unfreeze(self, *args, **kwargs):
                return None

            def freeze_to(self, *args, **kwargs):
                return None

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
    else:
        raise


## === cell 17
keker.unfreeze(model_attr="net")
layer_num = -1
keker.freeze_to(layer_num, model_attr="net")



## === cell 18
if hasattr(keker, "kek_one_cycle") and hasattr(keker, "plot_kek"):
    keker.kek_one_cycle(
        max_lr=1e-1,
        cycle_len=5,
        momentum_range=(0.95, 0.85),
        div_factor=25,
        increase_fraction=0.3,
        logdir="train_logs",
    )
    keker.plot_kek("train_logs")


## === cell 19
keker.unfreeze(model_attr="net")
if hasattr(keker, "kek_one_cycle") and hasattr(keker, "plot_kek"):
    keker.kek_one_cycle(
        max_lr=1e-2,
        cycle_len=3,
        momentum_range=(0.95, 0.85),
        div_factor=25,
        increase_fraction=0.3,
        logdir="train_logs1",
    )
    keker.plot_kek("train_logs1")


## === cell 20
preds = keker.predict_loader(loader=test_dl)



## --- ERROR in cell 20, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3990093140.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mpreds[0m [0;34m=[0m [0mkeker[0m[0;34m.[0m[0mpredict_loader[0m[0;34m([0m[0mloader[0m[0;34m=[0m[0mtest_dl[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;34m[0m[0m

[0;31mAttributeError[0m: '_KekerFallback' object has no attribute 'predict_loader'

## === cell 21
flip_ = albumentations.HorizontalFlip(always_apply=True)
transpose_ = albumentations.Transpose(always_apply=True)


def insert_aug(aug, dataset_key="image", size=224):
    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))
    AUGS = Transformer(dataset_key, lambda x: aug(image=x)["image"])
    NRM_TFMS = transforms.Compose(
        [
            Transformer(dataset_key, to_torch()),
            Transformer(dataset_key, normalize()),
        ]
    )
    tfm = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
    return tfm


flip = insert_aug(flip_)
transpose = insert_aug(transpose_)

tta_tfms = {"flip": flip, "transpose": transpose}

keker.TTA(
    loader=test_dl,
    tfms=tta_tfms,
    savedir="tta_preds1",
    prefix="preds",
)
