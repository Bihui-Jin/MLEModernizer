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
    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))
    AUGS = Transformer(dataset_key, lambda x: augs(p=p)(image=x)["image"])
    NRM_TFMS = transforms.Compose(
        [Transformer(dataset_key, to_torch()), Transformer(dataset_key, normalize())]
    )
    train_tfms = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
    val_tfms = transforms.Compose([PRE_TFMS, NRM_TFMS])
    return train_tfms, val_tfms


IMG_SIZE = 32
train_tfms, val_tfms = get_transforms("image", IMG_SIZE, 0.5)



## --- ERROR in cell 4, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31m_KekasMissingDependency[0m                   Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4045482883.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     21[0m [0;31m# Keep original training size=32; ensure it is consistent everywhere (including TTA).[0m[0;34m[0m[0;34m[0m[0m
[1;32m     22[0m [0mIMG_SIZE[0m [0;34m=[0m [0;36m32[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 23[0;31m [0mtrain_tfms[0m[0;34m,[0m [0mval_tfms[0m [0;34m=[0m [0mget_transforms[0m[0;34m([0m[0;34m"image"[0m[0;34m,[0m [0mIMG_SIZE[0m[0;34m,[0m [0;36m0.5[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     24[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/4045482883.py[0m in [0;36mget_transforms[0;34m(dataset_key, size, p)[0m
[1;32m      9[0m [0;34m[0m[0m
[1;32m     10[0m [0;32mdef[0m [0mget_transforms[0m[0;34m([0m[0mdataset_key[0m[0;34m,[0m [0msize[0m[0;34m,[0m [0mp[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 11[0;31m     [0mPRE_TFMS[0m [0;34m=[0m [0mTransformer[0m[0;34m([0m[0mdataset_key[0m[0;34m,[0m [0;32mlambda[0m [0mx[0m[0;34m:[0m [0mcv2[0m[0;34m.[0m[0mresize[0m[0;34m([0m[0mx[0m[0;34m,[0m [0;34m([0m[0msize[0m[0;34m,[0m [0msize[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     12[0m     [0mAUGS[0m [0;34m=[0m [0mTransformer[0m[0;34m([0m[0mdataset_key[0m[0;34m,[0m [0;32mlambda[0m [0mx[0m[0;34m:[0m [0maugs[0m[0;34m([0m[0mp[0m[0;34m=[0m[0mp[0m[0;34m)[0m[0;34m([0m[0mimage[0m[0;34m=[0m[0mx[0m[0;34m)[0m[0;34m[[0m[0;34m"image"[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     13[0m     NRM_TFMS = transforms.Compose(

[0;32m/tmp/ipykernel_11/2361739388.py[0m in [0;36m_kekas_unavailable[0;34m(*args, **kwargs)[0m
[1;32m    103[0m [0;34m[0m[0m
[1;32m    104[0m     [0;32mdef[0m [0m_kekas_unavailable[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 105[0;31m         raise _KekasMissingDependency(
[0m[1;32m    106[0m             [0;34m"`kekas` is not installed in this environment, but the notebook attempted to use it."[0m[0;34m[0m[0;34m[0m[0m
[1;32m    107[0m         )

[0;31m_KekasMissingDependency[0m: `kekas` is not installed in this environment, but the notebook attempted to use it.

## === cell 5
train_dk = DataKek(df=train, reader_fn=reader_fn, transforms=train_tfms)
val_dk = DataKek(df=valid, reader_fn=reader_fn, transforms=val_tfms)

batch_size = 64
workers = 0

train_dl = torch.utils.data.DataLoader(
    train_dk, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True
)
val_dl = torch.utils.data.DataLoader(
    val_dk, batch_size=batch_size, num_workers=workers, shuffle=False
)

test_dk = DataKek(df=test_df, reader_fn=reader_fn, transforms=val_tfms)
test_dl = torch.utils.data.DataLoader(
    test_dk, batch_size=batch_size, num_workers=workers, shuffle=False
)
