# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.5395899547270685

# 6. Current score

0.00169

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0033) has done: 'The timeout is dominated by slow data loading/CPU preprocessing and Python-side postprocessing: per-sample `cv2.cvtColor` + Albumentations + low `num_workers` throttles GPU, and the submission loop builds strings row-by-row in Python. I keep the exact model, transforms, and thresholding logic, but speed up I/O by enabling persistent DataLoader workers, raising worker count safely, prefetching batches, and optimizing OpenCV reads. I also cut Python overhead by preallocating the prediction array and building `attribute_ids` strings with fast NumPy operations while preserving identical semantics. These changes are deterministic and do not alter the algorithm or outputs beyond negligible floating-point differences.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, and the biggest issue here is that you are very likely not actually loading the intended trained weights (the path `/kaggle/input/imet2020/EfficientNet_b0_epoch5.pth` doesn’t exist in the provided dataset tree), so the model is effectively random and yields near-zero F1. I keep the same EfficientNet-B0 architecture, transforms, and the same fixed thresholding logic, but make weight loading robust by auto-searching for the `.pth` inside `/kaggle/input` and failing loudly if none is found. I also preserve identical inference semantics while ensuring the submission row order matches `sample_submission.csv` and the output is always a valid `submission.csv`. These minimal changes should move the score substantially upward toward your target by ensuring you’re using the trained checkpoint you intended.'
- What this solution (achieved 0.0033) has done: 'The timeout is almost certainly from running full EfficientNet-B0 inference over 21k test images with an extremely wide 3474-logit head, where the bottleneck is GPU/CPU data transfer and per-batch Python overhead. I keep the exact same model and thresholding logic, but speed it up by (1) using channels-last + cuDNN autotuning for faster convolution inference, (2) using a tuned DataLoader with a safe worker count and pinned memory, (3) avoiding redundant conversions/copies by writing directly into a preallocated NumPy buffer via `torch.from_numpy`, and (4) reducing Python overhead in post-processing by vectorizing the “indices to string” conversion. These changes are provably equivalent in evaluation semantics (no sampling/early-exit/precision tricks) and should bring runtime under the 600s limit on Kaggle GPUs.'
- What this solution (achieved 0.0033) has done: 'Your notebook fails because it expects a pretrained `.pth` checkpoint under `/kaggle/input`, but none exists in the provided dataset tree, so `model` is never created and inference/submission crash. To run end-to-end reliably, I keep the same EfficientNet-B0 architecture, transforms, and thresholding, but change the weight-loading behavior to (1) try to find/load a checkpoint and use it if present, otherwise (2) fall back to ImageNet pretrained EfficientNet-B0 weights (still legitimate, and far better than random) and continue. I also fix the outdated `pretrained=` API usage in torchvision model wrappers to avoid potential runtime errors, and ensure the submission `.csv` is always written with the correct columns/order. These changes are minimal, unblock execution, and should move the score upward toward your target compared with a random/untrained model.'
- What this solution (achieved 0.0033) has done: 'I remove the hard failure when a trained `.pth` checkpoint is missing and instead fall back to ImageNet-pretrained EfficientNet-B0 (same architecture/head replacement) so the notebook runs end-to-end and always writes a valid `submission.csv`. This fixes the `model`/`pred_proba` `NameError`s by guaranteeing `model` is defined and inference executes. I keep the transforms, input size, thresholding, and overall inference semantics the same, only making weight loading robust and logging whether a real 3474-class checkpoint was found/loaded. This should substantially improve score versus random initialization while staying within the “minimal change” constraint.'
- What this solution (achieved 0.0033) has done: 'Your current score is far below the target, so the most likely issue is mismatched inference preprocessing versus how the checkpoint (or intended baseline) was trained. I keep the same EfficientNet-B0 model/head and the same fixed thresholding logic, but align the validation/test transform to the train-time crop by switching `valid` from `Resize` to `CenterCrop` after a resize, which is a minimal semantic change that typically boosts F1 for this competition. I also make weight-loading stricter: if a compatible 3474-head checkpoint is found, we ensure it’s actually used; if not, we still fall back to ImageNet weights as before. Finally, I keep your optimized DataLoader/inference path intact and only make a small, deterministic transform correction to move the score upward toward the target.'
- What this solution (achieved 0.00169) has done: 'The timeout is dominated by slow image decoding/augmentation in the DataLoader and by moving full 3474-dim probabilities back to CPU every batch. I keep the exact model and inference semantics, but speed up I/O by enabling OpenCV SIMD and increasing decoding throughput with more workers + larger prefetch, and I reduce CPU transfer overhead by copying logits directly into the preallocated NumPy buffer and applying `sigmoid` + thresholding in a single final vectorized pass. I also avoid redundant tensor→tensor copies and ensure contiguous/channels_last handling happens only when it helps. All changes are deterministic and preserve the same outputs up to negligible floating-point differences.'
- What this solution (achieved 0.00169) has done: 'Your current score (0.00169) is far below the target (0.5396), so the most likely cause is that you are still not loading an actual trained 3474-class checkpoint, meaning predictions are essentially untrained/random. I keep the same EfficientNet-B0 architecture and the same inference/thresholding semantics, but make checkpoint discovery/load stricter and more informative: we search for any `.pth` under `/kaggle/input`, try loading it, and only accept it if it truly contains a compatible 3474-dim classifier head; otherwise we clearly warn and fall back as before. Additionally, I fix a small but real numerical bug (`np.exp` doesn’t take `dtype=`) so sigmoid is computed correctly and consistently, which can materially affect F1. These are minimal changes that should move your score upward toward the target by ensuring the intended weights are actually used and probabilities are computed correctly.'
- What this solution (achieved 0.00169) has done: 'Your score is far below the target, so the smallest high-impact fix is to ensure we actually load a compatible 3474-class checkpoint (and not silently fall back to an untrained/random head). I make checkpoint loading stricter by detecting whether the classifier weights were truly loaded, and if not, we rebuild the model in the correct order to avoid overwriting loaded classifier weights. I also add safe handling for NaN/empty `attribute_ids` in `train.csv` and a numerically stable sigmoid (same semantics) so the thresholding behaves consistently. These are minimal changes that preserve the same EfficientNet-B0 architecture and inference pipeline, but should materially increase F1 toward your target by using real learned weights when available.'

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
import torchvision
import torchvision.models as M
from torchvision.models import (
    ResNet50_Weights,
    DenseNet121_Weights,
    ShuffleNet_V2_X1_0_Weights,
    SqueezeNet1_0_Weights,
    MobileNet_V2_Weights,
    EfficientNet_B0_Weights,
)
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from albumentations import (
    Compose,
    Normalize,
    Resize,
    RandomResizedCrop,
    RandomCrop,
    HorizontalFlip,
    CenterCrop,
)
from albumentations.pytorch import ToTensorV2

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(min(8, os.cpu_count() or 1))
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
submission = pd.read_csv("../input/imet-2020-fgvc7/sample_submission.csv")




## === cell 2
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


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f"[{name}] start")
    yield
    LOGGER.info(f"[{name}] done in {time.time() - t0:.0f} s.")


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
        self.df = df
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["id"].values[idx]
        file_path = f"../input/imet-2020-fgvc7/train/{file_name}.png"

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = self.labels.values[idx]
        if not isinstance(label, str):
            label = (
                ""
                if (label is None or (isinstance(label, float) and np.isnan(label)))
                else str(label)
            )

        target = torch.zeros(N_CLASSES, dtype=torch.float32)
        if label != "":
            for cls in label.split():
                target[int(cls)] = 1.0

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

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image




## === cell 4
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
                CenterCrop(HEIGHT, WIDTH),
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

_cpu = os.cpu_count() or 2
_num_workers = min(8, max(2, _cpu // 2))
_prefetch = 8 if _num_workers > 0 else None

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=_prefetch,
)




## === cell 6
class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


class ResNet(nn.Module):
    def __init__(
        self, num_classes, pretrained=False, net_cls=M.resnet50, dropout=False
    ):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.net = net_cls(weights=weights)
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
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.net = net_cls(weights=weights)
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


class ShuffleNet(nn.Module):
    def __init__(
        self, num_classes, pretrained=False, net_cls=M.shufflenet_v2_x1_0, dropout=False
    ):
        super().__init__()
        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        self.net = net_cls(weights=weights)
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        return self.net(x)


class SqueezeNet(nn.Module):
    def __init__(
        self, num_classes, pretrained=False, net_cls=M.squeezenet1_0, dropout=False
    ):
        super().__init__()
        weights = SqueezeNet1_0_Weights.DEFAULT if pretrained else None
        self.net = net_cls(weights=weights)

        fin_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            fin_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = self.net.classifier(out)
        out = out.view(-1, 3474)
        return out


class MobileNet(nn.Module):
    def __init__(
        self, num_classes, pretrained=False, net_cls=M.mobilenet_v2, dropout=False
    ):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.net = net_cls(weights=weights)

        self.net.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.net.last_channel, num_classes),
        )

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        return self.net(x)


resnet50 = partial(ResNet, net_cls=M.resnet50)
densenet121 = partial(DenseNet, net_cls=M.densenet121)
shufflenet = partial(ShuffleNet, net_cls=M.shufflenet_v2_x1_0)
squeezenet = partial(SqueezeNet, net_cls=M.squeezenet1_0)
mobilenet = partial(MobileNet, net_cls=M.mobilenet_v2)




## === cell 7
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _strip_prefix(k: str):
    for p in ("module.", "model.", "net."):
        if k.startswith(p):
            return k[len(p) :]
    return k


def find_weights_file(
    preferred_path: str, filename_hint: str = "EfficientNet_b0_epoch5.pth"
):
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    base = Path("/kaggle/input")
    if not base.exists():
        return None

    candidates = [str(p) for p in base.rglob(filename_hint)]
    if len(candidates) > 0:
        return sorted(candidates)[0]

    candidates = [str(p) for p in base.rglob("*efficientnet*b0*.pth")]
    if len(candidates) > 0:
        return sorted(candidates)[0]

    candidates = [str(p) for p in base.rglob("*.pth")]
    if len(candidates) == 0:
        return None
    return sorted(candidates)[0]


def load_pretrained_weights2(model, weights_path=None):
    if weights_path is None or not os.path.exists(weights_path):
        LOGGER.info(f"No weights loaded (file not found): {weights_path}")
        return False

    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)

    new_state = {}
    for k, v in state_dict.items():
        nk = _strip_prefix(k)
        new_state[nk] = v

    ret = model.load_state_dict(new_state, strict=False)
    LOGGER.info(
        f"Loaded weights from {weights_path}. Missing: {len(ret.missing_keys)} Unexpected: {len(ret.unexpected_keys)}"
    )

    classifier_keys = [("classifier.1.weight", "classifier.1.bias")]
    sd = model.state_dict()

    ok_shape = False
    for w_key, b_key in classifier_keys:
        if (
            w_key in sd
            and b_key in sd
            and sd[w_key].shape[0] == N_CLASSES
            and sd[b_key].shape[0] == N_CLASSES
        ):
            ok_shape = True
            w_missing = w_key in ret.missing_keys
            b_missing = b_key in ret.missing_keys
            if w_missing or b_missing:
                LOGGER.info(
                    "Classifier head weights exist but were not loaded from checkpoint (missing_keys)."
                )
                return False
            return True

    if not ok_shape:
        LOGGER.info(
            "Checkpoint did not result in a compatible 3474-output classifier head after load_state_dict."
        )
        return False

    return True




## === cell 8
DIR_WEIGHTS = "/kaggle/input/imet2020"
WEIGHTS_FILE = f"{DIR_WEIGHTS}/EfficientNet_b0_epoch5.pth"

resolved_weights = find_weights_file(
    WEIGHTS_FILE, filename_hint="EfficientNet_b0_epoch5.pth"
)
LOGGER.info(f"Resolved weights path: {resolved_weights}")

if resolved_weights is None:
    LOGGER.info(
        "No custom .pth checkpoint found under /kaggle/input. "
        "Falling back to EfficientNet_B0_Weights.DEFAULT (ImageNet-pretrained) for backbone initialization."
    )
    model = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, N_CLASSES)
    HAS_COMPAT_CKPT = False
else:
    LOGGER.info(
        "Building EfficientNet-B0 (weights=None), replacing head to 3474, then loading provided checkpoint."
    )
    model = torchvision.models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, N_CLASSES)

    loaded_ok = load_pretrained_weights2(model, weights_path=resolved_weights)
    if not loaded_ok:
        LOGGER.info(
            "Found a .pth checkpoint but it is incompatible with a 3474-class EfficientNet-B0 head. "
            "Continuing with ImageNet-pretrained backbone instead."
        )
        model = torchvision.models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, N_CLASSES)
        HAS_COMPAT_CKPT = False
    else:
        HAS_COMPAT_CKPT = True

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    model = model.to(memory_format=torch.channels_last)

model.to(device)



## === cell 9
criterion = nn.BCEWithLogitsLoss(reduction="none")



## === cell 10
train_df = pd.read_csv("../input/imet-2020-fgvc7/train.csv")
train_counts = (
    train_df["attribute_ids"]
    .fillna("")
    .astype(str)
    .apply(lambda s: 0 if s.strip() == "" else len(s.split()))
)
avg_pos_per_image = float(train_counts.mean())
LOGGER.info(f"Avg positive labels per train image: {avg_pos_per_image:.3f}")



## === cell 11
with timer("inference"):
    model.eval()

    n_test = len(test_dataset)

    pred_logits = np.empty((n_test, N_CLASSES), dtype=np.float32)
    pred_logits_t = torch.from_numpy(pred_logits)  # CPU tensor shares memory

    start = 0
    tk0 = tqdm(test_loader, total=len(test_loader))
    for images in tk0:
        bs = images.size(0)

        if torch.cuda.is_available():
            images = images.contiguous(memory_format=torch.channels_last)

        images = images.to(device, non_blocking=True)

        with torch.inference_mode():
            y_logits = model(images)

        pred_logits_t[start : start + bs].copy_(
            y_logits.detach().to(device="cpu", dtype=torch.float32)
        )
        start += bs

    if start != n_test:
        raise RuntimeError(f"Inference produced {start} rows, expected {n_test}.")



## === cell 12
pred_proba = 1.0 / (1.0 + np.exp(-np.clip(pred_logits, -50.0, 50.0)))

if HAS_COMPAT_CKPT:
    threshold = 0.10
else:
    target_k = int(max(1, round(avg_pos_per_image)))
    flat = pred_proba.reshape(-1)
    q = 1.0 - (target_k / float(N_CLASSES))
    q = float(np.clip(q, 0.0, 1.0))
    threshold = float(np.quantile(flat, q))
    LOGGER.info(
        f"Using calibrated threshold={threshold:.6f} (target_k={target_k}) due to missing compatible ckpt."
    )

predictions = pred_proba > threshold

attr_strings = [" ".join(map(str, row.nonzero()[0].tolist())) for row in predictions]

submission_out = submission.copy()
submission_out["attribute_ids"] = attr_strings

submission_out = submission_out[["id", "attribute_ids"]]
submission_out.to_csv("submission.csv", index=False)

print(submission_out.head())
print("Wrote submission.csv with rows:", len(submission_out))
print("HAS_COMPAT_CKPT:", HAS_COMPAT_CKPT, "threshold:", threshold)
