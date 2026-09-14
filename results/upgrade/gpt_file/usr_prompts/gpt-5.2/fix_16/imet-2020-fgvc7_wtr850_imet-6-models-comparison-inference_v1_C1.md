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

0.4539213536862424

# 6. Current score

0.0033

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00346) has done: 'Your code doesn’t currently yield a Kaggle score because inference is run with the model still in training mode (BatchNorm/Dropout behavior is wrong) and the test-time augmentation pipeline is accidentally applying random crops/flips, making predictions unstable and generally worse. I make two minimal, score-relevant fixes: set `model.eval()` during inference and change the “valid” transforms to deterministic `Resize` + `Normalize` (no random augmentations) while keeping the rest of your logic identical. I also make the submission-writing step robust by using `.loc` assignment and ensuring `attribute_ids` is always a string (empty if nothing passes the threshold), which avoids occasional pandas pitfalls and guarantees a valid `submission.csv`.'
- What this solution (achieved 0.00346) has done: 'I fix the crash caused by a missing checkpoint by auto-discovering an available `.pt/.pth` model file under `../input` and loading it when found, otherwise falling back to running with the randomly initialized model (still producing a valid CSV). I also make `load_model` robust to both “raw state_dict” and “wrapped dict with 'model' key” formats, which avoids key errors across different saved checkpoints. Finally, I keep inference in `eval()` mode and write the submission with guaranteed string `attribute_ids`, preserving your model and thresholding logic while making the pipeline run end-to-end.'
- What this solution (achieved 0.00346) has done: 'Your current score is far below the target (0.00346 vs 0.4539), so we should safely improve the inference quality without changing the model architecture or training approach. The biggest score-relevant issue left is that you are effectively doing “top-anything-over-0.10”, which for this dataset typically produces too many/too few labels and tanks micro-F1; we keep the exact same model and logits but calibrate the decision rule using an internal train-split to pick a threshold that better matches micro-F1. Concretely, we (1) build a small validation split from `train.csv`, (2) run deterministic inference on that split to compute micro-F1 over a threshold grid, and (3) use the best threshold for test predictions. This is minimal (only affects post-processing) and should move the score substantially toward the target while still finishing within the time limit.'
- What this solution (achieved 0.00364) has done: 'I fix the calibration split crash by making the stratification robust: we keep your exact “stratify_key” idea but collapse rare keys (count < 2) into a common bucket so `train_test_split(..., stratify=...)` always works. I also ensure the calibration inference actually uses the loaded weights by moving `model.to(device)` before running validation inference (score-relevant but not a logic change). Finally, I keep inference deterministic (`valid` transforms) and guarantee the submission is written with the exact required columns and `.csv` suffix.'
- What this solution (achieved 0.00364) has done: 'Your current score (0.00364) is far below the target (0.4539), so we should improve output quality with minimal, score-relevant changes while keeping the same model and thresholding semantics. The main issue is that if no checkpoint is found (or a wrong one is loaded), you end up predicting with a randomly initialized model, which produces near-random labels and extremely low micro-F1; we make checkpoint discovery prioritize iMet-2020-specific ResNet50 checkpoints under `../input/imet-2020-fgvc7/` first. Second, we harden checkpoint loading to handle common key-prefix formats (e.g., `module.`) so a good checkpoint actually loads instead of silently failing/mismatching. These changes keep your architecture/inference/threshold calibration logic intact but should move the score substantially toward the target by ensuring you’re using meaningful weights.'
- What this solution (achieved 0.00364) has done: 'Your score is far below the target, so the smallest high-impact fix is to ensure you actually load a compatible pretrained iMet checkpoint instead of silently falling back to random weights (which yields near-random predictions and ~0 micro-F1). I make checkpoint loading tolerant to common key mismatches (`module.` prefix, `net.` prefix, and `fc.` vs `net.fc.`) and allow `strict=False` as a fallback while reporting missing/unexpected keys so the run doesn’t “succeed” with effectively untrained heads. I also slightly widen the threshold search grid (still the same exact post-processing logic) because a too-coarse grid often picks a bad operating point for micro-F1. These are minimal changes that preserve your architecture, loss, training/inference flow, and submission semantics, but should move the score substantially toward the target by making the model weights + thresholding sane.'
- What this solution (achieved 0.0033) has done: 'Your score is far below the target, so the most likely reason is that you’re still effectively predicting with untrained/random weights because no compatible iMet checkpoint is being loaded. I make checkpoint discovery explicitly look for iMet-2020 ResNet50-style checkpoints (common Kaggle filenames) and, if none exist, fall back to loading torchvision ImageNet pretrained weights (same architecture, not a training-loop change) so predictions become meaningful. I also fix the `create_net()` pretrained path logic that currently points to a non-existent `../input/ResNet/...` folder, which can silently prevent good weights from ever being used. Finally, I keep your calibration + thresholding logic intact, but ensure `model.to(device)` happens before any forward pass and that submission writing remains identical and robust.'
- What this solution (achieved 0.0033) has done: 'We make two minimal, score-relevant fixes aimed at moving micro-F1 up toward the target by improving calibration fidelity without changing your model, loss, or training/inference loops. First, we ensure the calibration/validation split uses the same ImageNet normalization but a more faithful resize for ResNet50 (224x224) while keeping the exact same augmentation “type” (deterministic Resize+Normalize) and preserving the pipeline logic; this typically improves logits quality substantially versus 128x128. Second, we make the threshold selection robust by additionally tuning a `topk` fallback (predict at least K labels per image) on the same validation split, then apply the chosen (threshold, topk) rule at test time; this keeps semantics (sigmoid + decision rule) but avoids empty predictions that destroy micro-F1. Both changes are minimal, deterministic, and should run within the time limit.'
- What this solution (achieved 0.0033) has done: 'Your current score is far below the target, so the smallest high-impact move toward the target is to make sure inference uses meaningful (iMet-trained) weights instead of random/ImageNet-only weights. I keep your exact ResNet50 architecture and your calibration + (threshold, topk) decision rule, but (1) make checkpoint discovery prefer iMet-2020 ResNet50 classifier checkpoints and explicitly avoid picking optimizer/scheduler-only files, and (2) harden checkpoint loading so the final `fc` layer is correctly mapped even when saved under different key names (e.g., `classifier.*`, `head.*`, `net.fc.*`). This should materially increase micro-F1 without changing your training/inference loop semantics and still produce the same `submission.csv` format.'
- What this solution (achieved 0.0033) has done: 'Your current score (0.0033) is far below the target (0.4539), so the priority is to ensure you are not effectively predicting with random/untrained weights. I keep your exact ResNet50 architecture and inference/thresholding approach, but fix checkpoint discovery so it can actually find iMet-2020 weights (your current `find_checkpoint` is searching under `../input`, while your data is under `../kaggle/data/...`). I also make the fallback to ImageNet-pretrained weights explicit and guaranteed (so you never run with random init), and add a small safety check that the loaded checkpoint matches `N_CLASSES` to avoid silently loading an incompatible head. These are minimal, score-relevant changes that should move micro-F1 substantially upward toward the target.'
- What this solution (achieved 0.0033) has done: 'Your gap to target is very large (0.0033 vs 0.4539), and the most likely remaining root cause is that you’re still not loading a meaningful iMet-trained checkpoint (falling back to ImageNet or mis-loading the head), which yields near-random multi-label outputs. I make checkpoint discovery explicitly look under common Kaggle dataset roots for iMet ResNet50 weights (and avoid picking unrelated optimizer/scheduler files), and I harden loading to correctly map common head names (`fc`, `classifier`, `head`) into your `net.fc` while keeping the same ResNet50 architecture. I also add a lightweight sanity check after loading (verify `net.fc` weights are non-trivial and class dimension matches) so you don’t silently run with an uninitialized head. Core inference, transforms, and the (threshold, topk) calibration logic remain identical.'
- What this solution (achieved 0.0033) has done: 'Your score is extremely far below the target, so the smallest likely high-impact fix is to stop spending time on “calibration” with an effectively untrained iMet head and instead guarantee we use a meaningful backbone+head without changing your architecture or inference semantics. I keep your exact ResNet50 model class and sigmoid+threshold/topk post-processing, but I change the default initialization to ImageNet-pretrained (so you never run random weights) and then load a checkpoint on top if found. I also tighten checkpoint selection to prefer files that look like full model weights (not optimizer/scheduler) and add a hard check that the loaded `net.fc` matches `N_CLASSES`; otherwise we fall back to ImageNet-pretrained rather than proceeding with a broken head. These are minimal, score-relevant changes that should move micro-F1 substantially upward toward your target while still producing the same `submission.csv` format.'
- What this solution (achieved 0.0033) has done: 'Your current score is still near-random relative to the target, which strongly suggests the classifier head is not actually iMet-trained (either no real checkpoint is being found/loaded, or the loaded checkpoint doesn’t contain a compatible `fc` head). I make a minimal but high-impact change: if no compatible iMet checkpoint is found, we *not* proceed with an ImageNet-only 1000-class head; instead we (a) explicitly force the `fc` layer to remain a 3474-class head and (b) only accept checkpoints that contain a 3474-dim `fc.weight`, otherwise keep the randomly initialized 3474 head but avoid “false confidence” in calibration by falling back to a conservative fixed `(thr, topk)` chosen to avoid empty predictions. Additionally, I fix the current “revert to pretrained=True” fallback that can silently reintroduce the wrong head situation and add a strict shape check when loading so we never “successfully” load an incompatible state_dict. These changes preserve your exact architecture and inference logic (sigmoid + threshold/topk) while making sure the run uses valid iMet-compatible weights when available and avoids pathological all-empty/all-noisy outputs when not, moving micro-F1 materially toward the target. The submission writing stays identical and still produces a valid `submission.csv`.'
- What this solution (achieved 0.0033) has done: 'Your current score is near-random, and the biggest remaining score blocker is that inference is likely running with an iMet-untrained (random) 3474-class head because your checkpoint loader rejects most files and then silently falls back. To move the score toward the target with minimal semantic changes, I keep your exact model architecture and sigmoid+threshold/topk decision rule, but make checkpoint loading “head-tolerant”: load backbone weights even if the checkpoint head is missing/mismatched, while keeping your 3474-class `net.fc` intact. I also change checkpoint discovery to explicitly prefer the common iMet-2020 “baseline.pth” (which often stores a different key layout) and add a robust unwrapping of typical checkpoint dict formats (`state_dict`, `model_state_dict`). Finally, calibration is allowed whenever we have non-trivial weights loaded (not only when a perfect head exists), so threshold/topk selection becomes meaningful and should lift micro-F1 substantially toward your target.'
- What this solution (achieved 0.0033) has done: 'Your current score is near-random relative to the target, which strongly suggests the biggest remaining issue is not the model code but the post-processing: the threshold/topk rule is being calibrated on raw sigmoid probabilities that are poorly calibrated for micro-F1. To move the score upward with minimal changes and identical model/inference semantics, I keep your architecture and inference loop intact but change calibration to search over a much more appropriate decision rule for this competition: “predict top-K labels per image” (with an optional low threshold floor), which is a standard and score-relevant tweak for iMet-style noisy multi-label problems. This only changes how logits are turned into `attribute_ids` (not the model), and it stays deterministic and fast by using the existing 20k calibration subset. Finally, I keep your robust checkpoint discovery/loading and submission writing as-is.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import json



## === cell 1
(
    os.listdir("../input/imet-2020-fgvc7")
    if os.path.exists("../input/imet-2020-fgvc7")
    else os.listdir("../kaggle/data/imet-2020-fgvc7")
)



## === cell 2
DATA_ROOT = "../input/imet-2020-fgvc7"
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = "../kaggle/data/imet-2020-fgvc7"
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = "../kaggle/data/input/imet-2020-fgvc7"
if not os.path.exists(DATA_ROOT):
    raise FileNotFoundError(
        "Could not locate imet-2020-fgvc7 dataset folder under known Kaggle paths."
    )

submission = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")
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
        file_path = f"{DATA_ROOT}/train/{file_name}.png"
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
        file_path = f"{DATA_ROOT}/test/{file_name}.png"
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image




## === cell 6
HEIGHT = 224
WIDTH = 224


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


def create_net(net_cls, pretrained: bool):
    try:
        if pretrained:
            weights_enum = getattr(M, f"{net_cls.__name__}_Weights", None)
            if weights_enum is not None and hasattr(weights_enum, "DEFAULT"):
                return net_cls(weights=weights_enum.DEFAULT)
    except Exception:
        pass
    return net_cls(pretrained=pretrained)


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

model = resnet50(num_classes=N_CLASSES, pretrained=True)



## === cell 10
from typing import Dict




## === cell 11
def _strip_prefix_from_state_dict(
    state_dict: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _remap_state_dict_keys_for_model(
    sd: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    for pfx in ("module.", "model.", "net."):
        sd = _strip_prefix_from_state_dict(sd, pfx)

    if any(k.startswith("backbone.") for k in sd.keys()) and not any(
        k.startswith("conv1.") for k in sd.keys()
    ):
        sd = {k[len("backbone.") :]: v for k, v in sd.items()}

    if any(k.startswith("classifier.") for k in sd.keys()) and not any(
        k.startswith("fc.") for k in sd.keys()
    ):
        sd = {
            ("fc." + k[len("classifier.") :]) if k.startswith("classifier.") else k: v
            for k, v in sd.items()
        }

    if any(k.startswith("head.") for k in sd.keys()) and not any(
        k.startswith("fc.") for k in sd.keys()
    ):
        sd = {
            ("fc." + k[len("head.") :]) if k.startswith("head.") else k: v
            for k, v in sd.items()
        }

    if any(k.startswith("fc.") for k in sd.keys()) and not any(
        k.startswith("net.fc.") for k in sd.keys()
    ):
        sd = {
            (
                "net." + k
                if (
                    k.startswith("fc.")
                    or k.startswith("conv1.")
                    or k.startswith("bn1.")
                    or k.startswith("layer")
                    or k.startswith("avgpool.")
                )
                else k
            ): v
            for k, v in sd.items()
        }

    if any(k.startswith("net.classifier.") for k in sd.keys()) and not any(
        k.startswith("net.fc.") for k in sd.keys()
    ):
        sd = {
            (
                "net.fc." + k[len("net.classifier.") :]
                if k.startswith("net.classifier.")
                else k
            ): v
            for k, v in sd.items()
        }
    if any(k.startswith("net.head.") for k in sd.keys()) and not any(
        k.startswith("net.fc.") for k in sd.keys()
    ):
        sd = {
            ("net.fc." + k[len("net.head.") :] if k.startswith("net.head.") else k): v
            for k, v in sd.items()
        }

    return sd


def _unwrap_checkpoint_to_state_dict(state):
    if isinstance(state, dict):
        for k in ("state_dict", "model_state_dict", "model", "net"):
            if (
                k in state
                and isinstance(state[k], dict)
                and any(isinstance(v, torch.Tensor) for v in state[k].values())
            ):
                return state[k]
    if isinstance(state, dict) and any(
        isinstance(v, torch.Tensor) for v in state.values()
    ):
        return state
    return None


def _state_dict_has_imet_head(sd: Dict[str, torch.Tensor]) -> bool:
    w = sd.get("net.fc.weight", None)
    b = sd.get("net.fc.bias", None)
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        return False
    if w.shape[0] != N_CLASSES:
        return False
    if isinstance(b, torch.Tensor) and b.ndim == 1 and b.shape[0] != N_CLASSES:
        return False
    return True


def load_model(model, path):
    state = torch.load(str(path), map_location="cpu")
    raw_sd = _unwrap_checkpoint_to_state_dict(state)
    if raw_sd is None:
        raise ValueError(f"Unrecognized checkpoint format at {path}")

    sd = _remap_state_dict_keys_for_model(raw_sd)

    has_head = _state_dict_has_imet_head(sd)
    try:
        if has_head:
            model.load_state_dict(sd, strict=True)
            print(f"Loaded checkpoint (with iMet head) from {path}")
        else:
            sd2 = {k: v for k, v in sd.items() if not k.startswith("net.fc.")}
            missing, unexpected = model.load_state_dict(sd2, strict=False)
            print(
                f"Loaded checkpoint backbone only (head ignored/missing) from {path}. "
                f"Missing keys (first 10): {missing[:10]} | Unexpected keys (first 10): {unexpected[:10]}"
            )
    except RuntimeError as e:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded with strict=False due to: {e}")
        print(f"Missing keys (first 20): {missing[:20]}")
        print(f"Unexpected keys (first 20): {unexpected[:20]}")

    if isinstance(state, dict) and ("epoch" in state or "step" in state):
        return {"epoch": state.get("epoch", None), "step": state.get("step", None)}
    return {"epoch": None, "step": None}


def find_checkpoint(preferred_path: str):
    p = Path(preferred_path)
    if p.exists():
        return str(p)

    exts = (".pt", ".pth", ".bin", ".ckpt")

    prioritized_roots = [
        Path(DATA_ROOT),
        Path(DATA_ROOT) / "imet-2020-fgvc7",
        Path(DATA_ROOT).parent,
        Path("../input"),
        Path("../kaggle/data"),
        Path("../kaggle/data/input"),
        Path("../kaggle/working"),
    ]

    candidates = []
    for root in prioritized_roots:
        if root.exists():
            for ext in exts:
                candidates.extend(list(root.rglob(f"*{ext}")))

    if not candidates:
        return None

    def score(path: Path):
        name = path.name.lower()
        parent = str(path.parent).lower()
        full = str(path).lower()
        s = 0

        if name == "baseline.pth":
            s += 2000

        if "imet" in name or "imet" in parent or "imet" in full:
            s += 200
        if "fgvc" in name or "fgvc" in parent or "fgvc" in full:
            s += 50

        if "resnet50" in name or "r50" in name or "resnet_50" in name:
            s += 120
        if "resnet" in name:
            s += 15

        if "best" in name:
            s += 30
        if "final" in name:
            s += 10

        if "optimizer" in name or "optim" in name:
            s -= 1000
        if "sched" in name or "scheduler" in name:
            s -= 1000
        if "ema" in name:
            s -= 200

        if (
            "state_dict" in name
            or "model" in name
            or "weights" in name
            or "ckpt" in name
        ):
            s += 10

        try:
            sz = path.stat().st_size
        except Exception:
            sz = 0

        if sz < 5_000_000:
            s -= 800
        elif sz < 20_000_000:
            s -= 500
        elif sz > 70_000_000:
            s += 20

        return s

    candidates = sorted(
        candidates,
        key=lambda x: (score(x), x.stat().st_size if x.exists() else 0),
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def _fc_sanity_ok(m: nn.Module) -> bool:
    try:
        w = m.net.fc.weight.detach().cpu()
        if w.shape[0] != N_CLASSES:
            print(f"WARNING: net.fc out_features mismatch: {w.shape[0]} != {N_CLASSES}")
            return False
        wstd = float(w.std().item())
        print(f"net.fc weight std: {wstd:.6f}")
        return wstd > 1e-3
    except Exception as e:
        print(f"WARNING: failed fc sanity check: {e}")
        return False


def _backbone_sanity_ok(m: nn.Module) -> bool:
    try:
        w = m.net.conv1.weight.detach().cpu()
        wstd = float(w.std().item())
        print(f"net.conv1 weight std: {wstd:.6f}")
        return wstd > 1e-3
    except Exception as e:
        print(f"WARNING: failed backbone sanity check: {e}")
        return False




## === cell 12
ckpt_path = find_checkpoint("../input/imet2020/best-model.pt")

HAS_GOOD_WEIGHTS = False

if ckpt_path is None:
    print(
        "WARNING: No checkpoint found under known roots. "
        "Proceeding without iMet-trained weights (ImageNet backbone + random 3474 head)."
    )
    HAS_GOOD_WEIGHTS = True  # ImageNet pretrained backbone is still meaningful
else:
    print(f"Using checkpoint: {ckpt_path}")
    try:
        load_model(model, ckpt_path)
        HAS_GOOD_WEIGHTS = _backbone_sanity_ok(model)
        if not HAS_GOOD_WEIGHTS:
            print(
                "WARNING: Checkpoint loaded but backbone sanity check failed; "
                "proceeding with ImageNet-pretrained weights only."
            )
            model = resnet50(num_classes=N_CLASSES, pretrained=True)
            HAS_GOOD_WEIGHTS = True
    except Exception as e:
        print(
            f"WARNING: Failed to load checkpoint ({e}). Proceeding with ImageNet-pretrained weights."
        )
        model = resnet50(num_classes=N_CLASSES, pretrained=True)
        HAS_GOOD_WEIGHTS = True




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


def micro_f1_from_probs_topk_thr(
    y_true_bin: np.ndarray, probs: np.ndarray, topk: int, thr: float
) -> float:
    n, c = probs.shape
    pred = np.zeros((n, c), dtype=bool)

    if topk is not None and int(topk) > 0:
        k = int(topk)
        idx = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        pred[rows, idx] = True

    if thr is not None:
        pred |= probs > float(thr)

    return sklearn.metrics.f1_score(
        y_true_bin, pred.astype(np.uint8), average="micro", zero_division=0
    )


if HAS_GOOD_WEIGHTS:
    train_df = pd.read_csv(f"{DATA_ROOT}/train.csv")

    calib_n = min(20000, len(train_df))
    train_df = train_df.sample(n=calib_n, random_state=SEED).reset_index(drop=True)

    train_df["stratify_key"] = (
        train_df["attribute_ids"].fillna("").map(make_stratify_key)
    )

    key_counts = train_df["stratify_key"].value_counts()
    rare_keys = key_counts[key_counts < 2].index
    if len(rare_keys) > 0:
        train_df.loc[train_df["stratify_key"].isin(rare_keys), "stratify_key"] = "RARE"

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

    topk_grid = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    threshold_grid = np.array(
        [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10], dtype=np.float32
    )

    best_thr = 0.0
    best_topk = 5
    best_f1 = -1.0
    with timer("threshold_search_topk"):
        for topk in topk_grid:
            for thr in threshold_grid:
                f1 = micro_f1_from_probs_topk_thr(
                    y_true, va_probs, int(topk), float(thr)
                )
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = float(thr)
                    best_topk = int(topk)

    print(
        f"Chosen topk={best_topk}, threshold={best_thr:.3f} (valid micro-F1={best_f1:.5f})"
    )

    del va_probs, y_true, va_dataset, va_loader, tr_df, va_df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
else:
    best_thr = 0.0
    best_topk = 5
    print(
        "Calibration skipped (no usable weights). "
        f"Using fallback topk={best_topk}, threshold={best_thr:.3f}."
    )



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
topk = best_topk if "best_topk" in globals() and best_topk is not None else 0

probs_test = np.concatenate(preds, axis=0)

pred_mask = np.zeros_like(probs_test, dtype=bool)
if topk is not None and int(topk) > 0:
    topk = int(topk)
    idx = np.argpartition(-probs_test, kth=topk - 1, axis=1)[:, :topk]
    rows = np.arange(probs_test.shape[0])[:, None]
    pred_mask[rows, idx] = True

if threshold is not None:
    pred_mask |= probs_test > float(threshold)

if "id" not in submission.columns:
    raise ValueError("sample_submission is missing required 'id' column")

attr_col = submission.columns[submission.columns.str.lower() == "attribute_ids"]
attr_col = attr_col[0] if len(attr_col) else "attribute_ids"

for i, row in enumerate(pred_mask):
    ids = np.nonzero(row)[0]
    submission.loc[i, attr_col] = " ".join(map(str, ids.tolist())) if len(ids) else ""

submission[attr_col] = submission[attr_col].fillna("").astype(str)

submission = submission[["id", attr_col]].rename(columns={attr_col: "attribute_ids"})
submission.to_csv("submission.csv", index=False)
submission.head()
