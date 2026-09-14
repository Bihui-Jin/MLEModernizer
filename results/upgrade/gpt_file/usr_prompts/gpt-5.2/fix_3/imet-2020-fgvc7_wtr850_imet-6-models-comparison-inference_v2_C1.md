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
Label artwork images with significant attributes.

## Metric
Micro averaged F1 score.

## Submission Format
```
id,attribute_ids
00011f01965f141f5d1eea6592fa9862,0 1 2
00014abc91ed3e4bf1663fde8136fe80,0 1 2
0002e2054e303badc1a33463f6fb7973,0 1 2
```

## Dataset
Multiple modalities can be expected and the camera sources are unknown. The photographs are often centered for objects, and in the case where the museum artifact is an entire room, the images are scenic in nature.

Each object is annotated by a single annotator without a verification step. You should consider these annotations noisy.

The filename of each image is its `id`.

- **train.csv** gives the `attribute_ids` for the train images in **/train**
- **/test** contains the test images. You must predict the `attribute_ids` for these images.
- **sample_submission.csv** contains a submission in the correct format
- **labels.csv** provides descriptions of the attributes

# 2. Python version

3.8

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
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
scipy==1.15.3
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (81 lines)
            labels.csv (3475 lines)
            labels.csv.zip (28.4 kB)
            sample_submission.csv (21319 lines)
            sample_submission.csv.zip (426.2 kB)
            test.zip (3.8 GB)
            train.csv (120802 lines)
            train.csv.zip (3.2 MB)
            train.zip (21.4 GB)
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
            test/
                c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                ... and 21316 other files
                test/
            train/
                cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                ... and 120799 other files
                train/
        input/
            description.md (81 lines)
            labels.csv (3475 lines)
            labels.csv.zip (28.4 kB)
            sample_submission.csv (21319 lines)
            sample_submission.csv.zip (426.2 kB)
            test.zip (3.8 GB)
            train.csv (120802 lines)
            train.csv.zip (3.2 MB)
            train.zip (21.4 GB)
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
            test/
                c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                ... and 21316 other files
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
            train/
                cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                ... and 120799 other files
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
        working/
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
```

-> data/imet-2020-fgvc7/labels.csv has 3474 rows and 2 columns.
The columns are: attribute_id, attribute_name

-> data/imet-2020-fgvc7/sample_submission.csv has 21318 rows and 2 columns.
The columns are: id, attribute_ids

-> data/imet-2020-fgvc7/train.csv has 120801 rows and 2 columns.
The columns are: id, attribute_ids

-> data/labels.csv has 3474 rows and 2 columns.
The columns are: attribute_id, attribute_name

-> data/sample_submission.csv has 21318 rows and 2 columns.
The columns are: id, attribute_ids

-> data/train.csv has 120801 rows and 2 columns.
The columns are: id, attribute_ids

-> (stopped after 10 files for performance)

# 5. Target score

0.302447556218187

# 6. Current score

0.0033

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0033) has done: 'You’re failing because `efficientnet_pytorch` isn’t installed, so the model never gets created and inference/submission steps crash. I keep the same “EfficientNet-B0 logits → sigmoid → threshold” core inference logic, but switch to the built-in `torchvision.models.efficientnet_b0` (available in your environment) and load weights only if they exist (otherwise fall back to ImageNet weights) so the notebook runs end-to-end. I also fix the validation transforms (they currently do random crop/flip and can crash on smaller images) to deterministic `Resize` while preserving the same normalization. Finally, I harden image loading and ensure a valid `submission.csv` is always written with the required columns.'
- What this solution (achieved 0.0033) has done: 'Your current score is extremely low because the model is effectively untrained for this task when the custom checkpoint is missing, and the fixed global threshold (0.10) then produces almost-all-wrong label sets. To move the score toward the target with minimal semantic changes, I keep the same EfficientNet-B0 → sigmoid → threshold pipeline, but add a lightweight, deterministic threshold calibration step using a small validation split from `train.csv` (no architecture/loss/training-loop changes beyond adding a short fine-tune and choosing a better threshold). This aligns the post-processing with the micro-F1 metric and should yield a large improvement from 0.0033 without changing the overall approach. The code still writes a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import json
import sys
import gc
import random
import time
from typing import Dict
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict, Counter

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import scipy as sp

import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as M
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

from albumentations import (
    Compose,
    Normalize,
    Resize,
    RandomResizedCrop,
    RandomCrop,
    HorizontalFlip,
)
from albumentations.pytorch import ToTensorV2

INPUT_DIR = Path("/kaggle/input/imet-2020-fgvc7")
TRAIN_DIR = INPUT_DIR / "train"
TEST_DIR = INPUT_DIR / "test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 1
submission = pd.read_csv(str(INPUT_DIR / "sample_submission.csv"))
submission.head()




## === cell 2
@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f"[{name}] start")
    yield
    LOGGER.info(f"[{name}] done in {time.time() - t0:.0f} s.")


def init_logger(log_file="train.log"):
    from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler

    log_format = "%(asctime)s %(levelname)s %(message)s"

    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))

    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))

    logger = getLogger("Herbarium")
    logger.setLevel(DEBUG)
    if not logger.handlers:
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger


LOG_FILE = "train.log"
LOGGER = init_logger(LOG_FILE)


def seed_torch(seed=777):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 777
seed_torch(SEED)




## === cell 3
N_CLASSES = 3474


class TrainDataset(Dataset):
    def __init__(self, df, labels, transform=None):
        self.df = df.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["id"].values[idx]
        file_path = str(TRAIN_DIR / f"{file_name}.png")
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = self.labels.values[idx]
        target = torch.zeros(N_CLASSES, dtype=torch.float32)
        for cls in str(label).split():
            if cls != "":
                target[int(cls)] = 1.0

        return image, target


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["id"].values[idx]
        file_path = str(TEST_DIR / f"{file_name}.png")
        image = cv2.imread(file_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image




## === cell 4
HEIGHT = 128
WIDTH = 128


def get_transforms(*, data):
    """
    Keep same normalization; train uses RandomResizedCrop, valid uses deterministic Resize for stable
    calibration/inference (micro-F1 is sensitive to randomness at prediction time).
    """
    assert data in ("train", "valid")

    if data == "train":
        return Compose(
            [
                RandomResizedCrop(HEIGHT, WIDTH),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        return Compose(
            [
                Resize(HEIGHT, WIDTH),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )




## === cell 5
batch_size = 128

test_dataset = TestDataset(submission, transform=get_transforms(data="valid"))
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

len(test_dataset), len(test_loader)




## === cell 6
class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def create_net(net_cls, pretrained):
    if True and pretrained:
        net = net_cls()
        model_name = net_cls.__name__
        weights_path = f"../input/{model_name}/{model_name}.pth"
        net.load_state_dict(torch.load(weights_path))
    else:
        net = net_cls(pretrained=pretrained)
    return net


class ResNet(nn.Module):
    def __init__(
        self, num_classes, pretrained=False, net_cls=M.resnet50, dropout=False
    ):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Module):
    def __init__(self, num_classes, pretrained=False, net_cls=M.densenet121):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.avg_pool = AvgPool()
        self.net.classifier = nn.Linear(self.net.classifier.in_features, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out).view(out.size(0), -1)
        out = self.net.classifier(out)
        return out


resnet18 = partial(ResNet, net_cls=M.resnet18)
resnet34 = partial(ResNet, net_cls=M.resnet34)
resnet50 = partial(ResNet, net_cls=M.resnet50)
resnet101 = partial(ResNet, net_cls=M.resnet101)
resnet152 = partial(ResNet, net_cls=M.resnet152)

densenet121 = partial(DenseNet, net_cls=M.densenet121)
densenet169 = partial(DenseNet, net_cls=M.densenet169)
densenet201 = partial(DenseNet, net_cls=M.densenet201)
densenet161 = partial(DenseNet, net_cls=M.densenet161)




## === cell 7
criterion = nn.BCEWithLogitsLoss(reduction="none")




## === cell 8
DIR_WEIGHTS = Path("/kaggle/input/imet2020")
WEIGHTS_FILE = DIR_WEIGHTS / "EfficientNet_b0.pth"

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, N_CLASSES)

if WEIGHTS_FILE.exists():
    ckpt = torch.load(str(WEIGHTS_FILE), map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict) and any(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    LOGGER.info(
        f"Loaded weights from {WEIGHTS_FILE}. Missing={len(missing)}, Unexpected={len(unexpected)}"
    )
else:
    LOGGER.info(
        f"Weight file not found at {WEIGHTS_FILE}; using ImageNet pretrained backbone with fresh head."
    )

model.to(device)




## === cell 9
train_df = pd.read_csv(str(INPUT_DIR / "train.csv"))
train_df["n_labels"] = (
    train_df["attribute_ids"].fillna("").apply(lambda s: len(str(s).split()))
)
train_df["bucket"] = pd.cut(
    train_df["n_labels"], bins=[-1, 0, 1, 2, 3, 5, 10, 100], labels=False
)

rng = np.random.RandomState(SEED)
idx = np.arange(len(train_df))
rng.shuffle(idx)
train_df = train_df.iloc[idx].reset_index(drop=True)

valid_frac = 0.02  # small for speed; sufficient to calibrate threshold and improve a lot from near-zero
n_valid = max(512, int(len(train_df) * valid_frac))
valid_df = train_df.iloc[:n_valid].reset_index(drop=True)
tr_df = train_df.iloc[n_valid:].reset_index(drop=True)

valid_dataset = TrainDataset(
    valid_df, valid_df["attribute_ids"], transform=get_transforms(data="valid")
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

train_dataset = TrainDataset(
    tr_df, tr_df["attribute_ids"], transform=get_transforms(data="train")
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

do_finetune_head = not WEIGHTS_FILE.exists()

do_finetune_head, len(train_dataset), len(valid_dataset)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     66             schema_kwargs["strict"] = strict
---> 67             config = schema_cls(**schema_kwargs)
     68             validated_kwargs = config.model_dump()

/usr/local/lib/python3.11/dist-packages/pydantic/main.py in __init__(self, **data)
    249         __tracebackhide__ = True
--> 250         validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    251         if self is not validated_self:

ValidationError: 2 validation errors for InitSchema
scale
  Input should be a valid tuple [type=tuple_type, input_value=128, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=128, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4209057410.py in <cell line: 0>()
     35 # For fine-tuning we only train the classification head to preserve core model logic and keep runtime low.
     36 train_dataset = TrainDataset(
---> 37     tr_df, tr_df["attribute_ids"], transform=get_transforms(data="train")
     38 )
     39 train_loader = DataLoader(

/tmp/ipykernel_55/2222360043.py in get_transforms(data)
     13         return Compose(
     14             [
---> 15                 RandomResizedCrop(HEIGHT, WIDTH),
     16                 Normalize(
     17                     mean=[0.485, 0.456, 0.406],

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in custom_init(self, *args, **kwargs)
    103                 full_kwargs, param_names, strict = cls._process_init_parameters(original_init, args, kwargs)
    104 
--> 105                 validated_kwargs = cls._validate_parameters(
    106                     dct["InitSchema"],
    107                     full_kwargs,

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     69             validated_kwargs.pop("strict", None)
     70         except ValidationError as e:
---> 71             raise ValueError(str(e)) from e
     72         except Exception as e:
     73             if strict:

ValueError: 2 validation errors for InitSchema
scale
  Input should be a valid tuple [type=tuple_type, input_value=128, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=128, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 10
def micro_f1_from_logits(logits: np.ndarray, targets: np.ndarray, thr: float) -> float:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > thr).astype(np.uint8)
    t = targets.astype(np.uint8)
    tp = (preds & t).sum()
    fp = (preds & (1 - t)).sum()
    fn = ((1 - preds) & t).sum()
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom > 0 else 0.0


@torch.no_grad()
def predict_logits(loader, model, device):
    model.eval()
    outs = []
    tgts = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
            tgts.append(y.cpu().numpy())
        else:
            x = batch
        x = x.to(device, non_blocking=True)
        out = model(x)
        outs.append(out.detach().cpu().numpy())
    outs = np.concatenate(outs, axis=0)
    tgts = np.concatenate(tgts, axis=0) if tgts else None
    return outs, tgts




## === cell 11
if do_finetune_head:
    with timer("head_finetune"):
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True

        optimizer = Adam(model.classifier.parameters(), lr=1e-3)
        model.train()

        for images, targets in tqdm(train_loader, total=len(train_loader)):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets).mean()
            loss.backward()
            optimizer.step()

else:
    LOGGER.info("Skipping head fine-tune because task-specific weights were found.")




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/163745391.py in <cell line: 0>()
      1 # Change (score-relevant): head-only fine-tune with the same BCEWithLogitsLoss.
      2 # This keeps the same modeling approach, but avoids "random head" predictions when no checkpoint exists.
----> 3 if do_finetune_head:
      4     with timer("head_finetune"):
      5         # freeze all but classifier

NameError: name 'do_finetune_head' is not defined

## === cell 12
with timer("threshold_calibration"):
    valid_logits, valid_targets = predict_logits(valid_loader, model, device)

    thr_grid = np.linspace(0.02, 0.30, 15)
    best_thr, best_f1 = 0.10, -1.0
    for thr in thr_grid:
        f1 = micro_f1_from_logits(valid_logits, valid_targets, float(thr))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    LOGGER.info(f"Calibrated threshold={best_thr:.4f} on valid; micro-F1={best_f1:.4f}")




## === cell 13
with timer("inference"):
    model.eval()
    preds = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, images in tk0:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            y_preds = model(images)
        preds.append(torch.sigmoid(y_preds).detach().cpu().numpy())

preds = np.concatenate(preds, axis=0)
preds.shape




## === cell 14
threshold = float(locals().get("best_thr", 0.10))

predictions = preds > threshold

submission_out = submission.copy()
submission_out["attribute_ids"] = ""

for i, row in enumerate(predictions):
    ids = np.nonzero(row)[0]
    submission_out.iat[i, submission_out.columns.get_loc("attribute_ids")] = " ".join(
        map(str, ids.tolist())
    )

submission_out.to_csv("submission.csv", index=False)
submission_out.head()
