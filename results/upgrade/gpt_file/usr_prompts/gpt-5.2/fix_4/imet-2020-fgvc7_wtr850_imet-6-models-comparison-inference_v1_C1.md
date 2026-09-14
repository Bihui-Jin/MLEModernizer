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

0.4539213536862424

# 6. Current score

0.00346

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00346) has done: 'Your code doesn’t currently yield a Kaggle score because inference is run with the model still in training mode (BatchNorm/Dropout behavior is wrong) and the test-time augmentation pipeline is accidentally applying random crops/flips, making predictions unstable and generally worse. I make two minimal, score-relevant fixes: set `model.eval()` during inference and change the “valid” transforms to deterministic `Resize` + `Normalize` (no random augmentations) while keeping the rest of your logic identical. I also make the submission-writing step robust by using `.loc` assignment and ensuring `attribute_ids` is always a string (empty if nothing passes the threshold), which avoids occasional pandas pitfalls and guarantees a valid `submission.csv`.'
- What this solution (achieved 0.00346) has done: 'I fix the crash caused by a missing checkpoint by auto-discovering an available `.pt/.pth` model file under `../input` and loading it when found, otherwise falling back to running with the randomly initialized model (still producing a valid CSV). I also make `load_model` robust to both “raw state_dict” and “wrapped dict with 'model' key” formats, which avoids key errors across different saved checkpoints. Finally, I keep inference in `eval()` mode and write the submission with guaranteed string `attribute_ids`, preserving your model and thresholding logic while making the pipeline run end-to-end.'
- What this solution (achieved 0.00346) has done: 'Your current score is far below the target (0.00346 vs 0.4539), so we should safely improve the inference quality without changing the model architecture or training approach. The biggest score-relevant issue left is that you are effectively doing “top-anything-over-0.10”, which for this dataset typically produces too many/too few labels and tanks micro-F1; we keep the exact same model and logits but calibrate the decision rule using an internal train-split to pick a threshold that better matches micro-F1. Concretely, we (1) build a small validation split from `train.csv`, (2) run deterministic inference on that split to compute micro-F1 over a threshold grid, and (3) use the best threshold for test predictions. This is minimal (only affects post-processing) and should move the score substantially toward the target while still finishing within the time limit.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import json



## === cell 1
os.listdir("../input/imet-2020-fgvc7")



## === cell 2
submission = pd.read_csv("../input/imet-2020-fgvc7/sample_submission.csv")
submission.head()



## === cell 3
import sys

import gc
import os
import random
import time
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 4
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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


SEED = 777
seed_torch(SEED)



## === cell 5
N_CLASSES = 3474


class TrainDataset(Dataset):
    def __init__(self, df, labels, transform=None):
        self.df = df
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["id"].values[idx]
        file_path = f"../input/imet-2020-fgvc7/train/{file_name}.png"
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = self.labels.values[idx]
        target = torch.zeros(N_CLASSES)
        for cls in label.split():
            target[int(cls)] = 1

        return image, target


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["id"].values[idx]
        file_path = f"../input/imet-2020-fgvc7/test/{file_name}.png"
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image




## === cell 6
HEIGHT = 128
WIDTH = 128


def get_transforms(*, data):

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

    elif data == "valid":
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




## === cell 7
batch_size = 128

test_dataset = TestDataset(submission, transform=get_transforms(data="valid"))
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 8
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



## === cell 9
criterion = nn.BCEWithLogitsLoss(reduction="none")
model = resnet50(num_classes=N_CLASSES, pretrained=False)



## === cell 10
from typing import Dict




## === cell 11
def load_model(model, path):
    state = torch.load(str(path), map_location="cpu")
    if (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        model.load_state_dict(state["model"])
        print("Loaded model from epoch {epoch}, step {step:,}".format(**state))
        return state
    else:
        model.load_state_dict(state)
        print(f"Loaded raw state_dict from {path}")
        return {"epoch": None, "step": None}


def find_checkpoint(preferred_path: str):
    p = Path(preferred_path)
    if p.exists():
        return str(p)

    input_root = Path("../input")
    exts = (".pt", ".pth", ".bin")
    candidates = []
    for ext in exts:
        candidates.extend(list(input_root.rglob(f"*{ext}")))

    def score(path: Path):
        name = path.name.lower()
        s = 0
        if "best" in name:
            s += 10
        if "model" in name:
            s += 5
        if "checkpoint" in name or "ckpt" in name:
            s += 2
        return s

    candidates = sorted(
        candidates, key=lambda x: (score(x), x.stat().st_size), reverse=True
    )
    return str(candidates[0]) if candidates else None




## === cell 12
ckpt_path = find_checkpoint("../input/imet2020/best-model.pt")
if ckpt_path is None:
    print(
        "WARNING: No checkpoint found under ../input. Proceeding with randomly initialized model."
    )
else:
    print(f"Using checkpoint: {ckpt_path}")
    load_model(model, ckpt_path)




## === cell 13
def make_stratify_key(attr_str: str, max_labels: int = 3) -> str:
    parts = attr_str.split()
    parts = parts[:max_labels]
    return "_".join(parts) if parts else "none"


def infer_probs(model, loader):
    model.eval()
    all_probs = []
    tk0 = tqdm(loader, total=len(loader))
    for batch in tk0:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, _ = batch
        else:
            images = batch
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(images)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def micro_f1_from_probs(y_true_bin: np.ndarray, probs: np.ndarray, thr: float) -> float:
    y_pred_bin = (probs > thr).astype(np.uint8)
    return sklearn.metrics.f1_score(
        y_true_bin, y_pred_bin, average="micro", zero_division=0
    )


train_df = pd.read_csv("../input/imet-2020-fgvc7/train.csv")

calib_n = min(20000, len(train_df))
train_df = train_df.sample(n=calib_n, random_state=SEED).reset_index(drop=True)

train_df["stratify_key"] = train_df["attribute_ids"].fillna("").map(make_stratify_key)

from sklearn.model_selection import train_test_split

tr_df, va_df = train_test_split(
    train_df,
    test_size=0.2,
    random_state=SEED,
    stratify=train_df["stratify_key"],
)

va_labels = va_df["attribute_ids"].fillna("").astype(str).reset_index(drop=True)
va_dataset = TrainDataset(
    va_df.reset_index(drop=True), va_labels, transform=get_transforms(data="valid")
)
va_loader = DataLoader(
    va_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

model.to(device)

with timer("calibration_inference_valid"):
    va_probs = infer_probs(model, va_loader)

y_true = np.zeros((len(va_df), N_CLASSES), dtype=np.uint8)
for i, s in enumerate(va_labels.values):
    if s:
        for cls in s.split():
            y_true[i, int(cls)] = 1

threshold_grid = np.array(
    [
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
        0.12,
        0.14,
        0.16,
        0.18,
        0.20,
    ],
    dtype=np.float32,
)

best_thr = 0.10
best_f1 = -1.0
with timer("threshold_search"):
    for thr in threshold_grid:
        f1 = micro_f1_from_probs(y_true, va_probs, float(thr))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

print(f"Chosen threshold={best_thr:.3f} (valid micro-F1={best_f1:.5f})")

del va_probs, y_true, va_dataset, va_loader, tr_df, va_df
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2381928811.py in <cell line: 0>()
     43 from sklearn.model_selection import train_test_split
     44 
---> 45 tr_df, va_df = train_test_split(
     46     train_df,
     47     test_size=0.2,

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in train_test_split(test_size, train_size, random_state, shuffle, stratify, *arrays)
   2581         cv = CVClass(test_size=n_test, train_size=n_train, random_state=random_state)
   2582 
-> 2583         train, test = next(cv.split(X=arrays[0], y=stratify))
   2584 
   2585     return list(

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in split(self, X, y, groups)
   1687         """
   1688         X, y, groups = indexable(X, y, groups)
-> 1689         for train, test in self._iter_indices(X, y, groups):
   1690             yield train, test
   1691 

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in _iter_indices(self, X, y, groups)
   2076         class_counts = np.bincount(y_indices)
   2077         if np.min(class_counts) < 2:
-> 2078             raise ValueError(
   2079                 "The least populated class in y has only 1"
   2080                 " member, which is too few. The minimum"

ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.

## === cell 14
with timer("inference"):
    model.to(device)
    model.eval()

    preds = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, images in tk0:
        images = images.to(device, non_blocking=True)

        with torch.no_grad():
            y_preds = model(images)

        preds.append(torch.sigmoid(y_preds).to("cpu").numpy())



## === cell 15
threshold = best_thr if "best_thr" in globals() and best_thr is not None else 0.10
predictions = np.concatenate(preds, axis=0) > threshold

attr_col = submission.columns[submission.columns.str.lower() == "attribute_ids"]
attr_col = attr_col[0] if len(attr_col) else "attribute_ids"

for i, row in enumerate(predictions):
    ids = np.nonzero(row)[0]
    submission.loc[i, attr_col] = " ".join(map(str, ids.tolist())) if len(ids) else ""

submission[attr_col] = submission[attr_col].fillna("").astype(str)
submission.to_csv("submission.csv", index=False)
submission.head()
