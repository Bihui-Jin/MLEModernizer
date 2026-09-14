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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.7783388837413378

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on an external EfficientNet checkpoint (none exists in your `/input`), and instead train the same EfficientNet-B4-like architecture quickly on the provided training set so inference can run end-to-end. I also fix the mixed CPU/GPU dtype issue in TTA by ensuring the model weights and input tensors live on the same device. Finally, I keep your kappa-thresholding step but fit thresholds on a held-out validation split (instead of fixed constants) so the output is aligned to the quadratic-weighted-kappa metric and produces a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission/prediction issue rather than true model performance, because your checkpoint logic loads `efficientNet_best.pth` even when it was copied from a random `.pth` under `../input`, and you load it with `strict=True` which can silently make the notebook unusable if the weights don’t match (or you end up using an unrelated checkpoint that outputs garbage). I make the checkpoint selection require an actual state_dict match against your EfficientNet-B4-like model; otherwise we fall back to training exactly as you already do. I also ensure the threshold optimizer stays within a sorted, valid range to prevent Nelder-Mead from producing non-monotonic thresholds that collapse predictions into a single class (a common cause of near-zero kappa). These are minimal changes that preserve your core model/training/inference logic while making the pipeline reliably produce a meaningful submission.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most likely caused by invalid/degenerate predictions (e.g., almost all test images mapped into a single class) due to threshold fitting on a weak regressor trained for only 2 epochs. I keep your exact model/training/inference structure, but make one minimal metric-aligned improvement: fit thresholds on out-of-fold (OOF) predictions from the training set using the same model (no architecture change), which stabilizes the kappa-optimized discretization and prevents collapse. I also make the kappa optimizer’s predict step vectorized (same semantics, fewer numerical edge-cases) and ensure predictions are clipped to a sane range before thresholding (doesn’t change core logic, just prevents extreme outliers from breaking threshold optimization). This should move the score upward toward your target without changing the fundamental approach.'

# 9. Code solution

## === cell 0
import os, glob, sys, subprocess, textwrap, math
from pathlib import Path

print("Listing ../input (top-level only):")
for p in sorted(glob.glob("../input/*")):
    print(" -", p)



## === cell 1
import shutil
import torch

model_path = "efficientNet_best.pth"

patterns = [
    "../input/**/efficientNet_*.pth",
    "../input/**/efficientnet_*.pth",
    "../input/**/efficientnet*.pth",
    "../input/**/efficientnet*.pt",
    "../input/**/efficientnet*.bin",
    "../input/**/efficientnet*.ckpt",
    "../input/**/*.pth",
]
candidates = []
for pat in patterns:
    candidates.extend(glob.glob(pat, recursive=True))


def _rank(p):
    name = os.path.basename(p).lower()
    score = 0
    if "efficientnet" in name:
        score += 100
    if "best" in name:
        score += 10
    if name.endswith(".pth"):
        score += 5
    return -score, len(p)


candidates = sorted(set(candidates), key=_rank)


def _extract_state_dict(obj):
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        obj = obj["state_dict"]
    if isinstance(obj, dict):
        new_state = {}
        for k, v in obj.items():
            if isinstance(k, str):
                nk = k.replace("module.", "")
                new_state[nk] = v
        return new_state
    return None


chosen = None
if len(candidates) == 0:
    print(
        "WARNING: No checkpoint found under ../input. "
        "Will train EfficientNet from scratch and save to efficientNet_best.pth."
    )
else:
    print(
        f"Found {len(candidates)} candidate .pth files; will validate compatibility with model later."
    )



## === cell 2
import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class SqueezeExcitation(nn.Module):
    def __init__(self, inplanes, se_planes):
        super(SqueezeExcitation, self).__init__()
        self.reduce_expand = nn.Sequential(
            nn.Conv2d(
                inplanes, se_planes, kernel_size=1, stride=1, padding=0, bias=True
            ),
            Swish(),
            nn.Conv2d(
                se_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_se = torch.mean(x, dim=(-2, -1), keepdim=True)
        x_se = self.reduce_expand(x_se)
        return x_se * x


from torch.nn import functional as F


class MBConv(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size,
        stride,
        expand_rate=1.0,
        se_rate=0.25,
        drop_connect_rate=0.2,
    ):
        super(MBConv, self).__init__()

        expand_planes = int(inplanes * expand_rate)
        se_planes = max(1, int(inplanes * se_rate))

        self.expansion_conv = None
        if expand_rate > 1.0:
            self.expansion_conv = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    expand_planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
                Swish(),
            )
            inplanes = expand_planes

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                inplanes,
                expand_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=expand_planes,
                bias=False,
            ),
            nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
            Swish(),
        )

        self.squeeze_excitation = SqueezeExcitation(expand_planes, se_planes)

        self.project_conv = nn.Sequential(
            nn.Conv2d(
                expand_planes, planes, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(planes, momentum=0.01, eps=1e-3),
        )

        self.with_skip = stride == 1
        self.drop_connect_rate = float(drop_connect_rate)

    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate
        drop_mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) + keep_prob
        drop_mask = drop_mask.type_as(x)
        drop_mask.floor_()
        return drop_mask * x / keep_prob

    def forward(self, x):
        z = x
        if self.expansion_conv is not None:
            x = self.expansion_conv(x)

        x = self.depthwise_conv(x)
        x = self.squeeze_excitation(x)
        x = self.project_conv(x)

        if x.shape == z.shape and self.with_skip:
            if (
                self.training
                and self.drop_connect_rate is not None
                and self.drop_connect_rate > 0
            ):
                x = self._drop_connect(x)
            x += z
        return x


from collections import OrderedDict
import math as _math


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_out")
    elif isinstance(module, nn.Linear):
        init_range = 1.0 / _math.sqrt(module.weight.shape[1])
        nn.init.uniform_(module.weight, a=-init_range, b=init_range)


class EfficientNet(nn.Module):
    def _setup_repeats(self, num_repeats):
        return int(_math.ceil(self.depth_coefficient * num_repeats))

    def _setup_channels(self, num_channels):
        num_channels *= self.width_coefficient
        new_num_channels = _math.floor(num_channels / self.divisor + 0.5) * self.divisor
        new_num_channels = max(self.divisor, new_num_channels)
        if new_num_channels < 0.9 * num_channels:
            new_num_channels += self.divisor
        return new_num_channels

    def __init__(
        self,
        num_classes,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        se_rate=0.25,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
    ):
        super(EfficientNet, self).__init__()

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.divisor = 8

        list_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        list_channels = [self._setup_channels(c) for c in list_channels]

        list_num_repeats = [1, 2, 2, 3, 3, 4, 1]
        list_num_repeats = [self._setup_repeats(r) for r in list_num_repeats]

        expand_rates = [1, 6, 6, 6, 6, 6, 6]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, list_channels[0], kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(list_channels[0], momentum=0.01, eps=1e-3),
            Swish(),
        )

        blocks = []
        counter = 0
        num_blocks = sum(list_num_repeats)
        for idx in range(7):
            num_channels = list_channels[idx]
            next_num_channels = list_channels[idx + 1]
            num_repeats = list_num_repeats[idx]
            expand_rate = expand_rates[idx]
            kernel_size = kernel_sizes[idx]
            stride = strides[idx]
            drop_rate = drop_connect_rate * counter / num_blocks

            name = "MBConv{}_{}".format(expand_rate, counter)
            blocks.append(
                (
                    name,
                    MBConv(
                        num_channels,
                        next_num_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        expand_rate=expand_rate,
                        se_rate=se_rate,
                        drop_connect_rate=drop_rate,
                    ),
                )
            )
            counter += 1
            for i in range(1, num_repeats):
                name = "MBConv{}_{}".format(expand_rate, counter)
                drop_rate = drop_connect_rate * counter / num_blocks
                blocks.append(
                    (
                        name,
                        MBConv(
                            next_num_channels,
                            next_num_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            expand_rate=expand_rate,
                            se_rate=se_rate,
                            drop_connect_rate=drop_rate,
                        ),
                    )
                )
                counter += 1

        self.blocks = nn.Sequential(OrderedDict(blocks))

        self.head = nn.Sequential(
            nn.Conv2d(list_channels[-2], list_channels[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(list_channels[-1], momentum=0.01, eps=1e-3),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(list_channels[-1], num_classes),
        )

        self.apply(init_weights)

    def forward(self, x):
        f = self.stem(x)
        f = self.blocks(f)
        y = self.head(f)
        return y




## === cell 3
import random
import numpy as np

image_size = 380
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

best_model = EfficientNet(
    num_classes=1, width_coefficient=1.4, depth_coefficient=1.8
)  # B4-like

loaded_ok = False
if len(candidates) > 0:
    model_sd = best_model.state_dict()
    for cand in candidates[:50]:
        try:
            raw = torch.load(cand, map_location="cpu")
            sd = _extract_state_dict(raw)
            if sd is None:
                continue
            common = [
                k
                for k in sd.keys()
                if k in model_sd
                and hasattr(sd[k], "shape")
                and hasattr(model_sd[k], "shape")
            ]
            if len(common) < int(0.8 * len(model_sd)):  # require substantial overlap
                continue
            shape_ok = True
            for k in common[:200]:
                if tuple(sd[k].shape) != tuple(model_sd[k].shape):
                    shape_ok = False
                    break
            if not shape_ok:
                continue
            best_model.load_state_dict(sd, strict=False)
            shutil.copyfile(cand, "efficientNet_best.pth")
            print("Loaded compatible checkpoint:", cand)
            print("Copied to:", model_path, "size:", os.path.getsize(model_path))
            loaded_ok = True
            break
        except Exception:
            continue

if (not loaded_ok) and os.path.exists(model_path):
    try:
        state = torch.load(model_path, map_location="cpu")
        state = _extract_state_dict(state) or state
        if isinstance(state, dict):
            best_model.load_state_dict(state, strict=True)
            loaded_ok = True
            print("Loaded local checkpoint:", model_path)
    except Exception as e:
        print(
            "WARNING: Local checkpoint exists but failed to load strictly; will retrain. Error:",
            repr(e),
        )
        loaded_ok = False

if not loaded_ok:
    print("No compatible checkpoint found; will train and then save:", model_path)

best_model = best_model.to(device)

_CAN_COMPILE = hasattr(torch, "compile") and (device.type == "cuda")
if _CAN_COMPILE:
    try:
        best_model = torch.compile(best_model, mode="reduce-overhead", fullgraph=False)
        print("Using torch.compile for speed.")
    except Exception as e:
        print(
            "torch.compile unavailable/failed; continuing uncompiled. Error:", repr(e)
        )



## === cell 4
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor, Normalize
import pandas as pd
from PIL import Image
from PIL.Image import BICUBIC

Image.MAX_IMAGE_PIXELS = None


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, path_list, targets=None, transform=None, extension=".png"):
        super().__init__()
        self.root = root
        self.path_list = list(path_list)
        self.targets = targets
        self.transform = transform
        self.extension = extension
        if targets is not None:
            assert len(self.path_list) == len(self.targets)
            self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):
        path = self.path_list[index]
        fp = os.path.join(self.root, path + self.extension)
        with Image.open(fp) as im:
            sample = im.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)

        if self.targets is not None:
            return sample, self.targets[index]
        else:
            return sample, torch.LongTensor([])

    def __len__(self):
        return len(self.path_list)


test_transform = Compose(
    [
        Resize((image_size, image_size), BICUBIC),
        ToTensor(),
        Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
    ]
)

train_transform = test_transform  # preserve core preprocessing

df_train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
df_test = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")

train_dataset_full = ImageDataset(
    root="../input/aptos2019-blindness-detection/train_images",
    path_list=df_train.id_code.values,
    targets=df_train.diagnosis.values,
    transform=train_transform,
)

test_dataset = ImageDataset(
    root="../input/aptos2019-blindness-detection/test_images",
    path_list=df_test.id_code.values,
    transform=test_transform,
)

print("train size:", len(train_dataset_full), "test size:", len(test_dataset))



## === cell 5
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

batch_size = 16

_cpu = os.cpu_count() or 2
num_workers = min(8, _cpu)
print("num_workers:", num_workers)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
idx = np.arange(len(df_train))
train_idx, val_idx = next(splitter.split(idx, df_train["diagnosis"].values))

train_dataset = Subset(train_dataset_full, train_idx.tolist())
val_dataset = Subset(train_dataset_full, val_idx.tolist())

_loader_kwargs = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    drop_last=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    drop_last=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    drop_last=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

print(
    "train batches:",
    len(train_loader),
    "val batches:",
    len(val_loader),
    "test batches:",
    len(test_loader),
)



## === cell 6
from tqdm import tqdm
import torch.nn as nn



def _tqdm(loader, total=None, leave=False, desc=None):
    return tqdm(
        loader, total=total, leave=leave, desc=desc, mininterval=0.5, smoothing=0.0
    )


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, y in _tqdm(loader, total=len(loader), leave=False):
        x = x.to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(1, len(loader))


@torch.no_grad()
def predict_continuous(model, loader, n_items=None):
    model.eval()
    if n_items is None:
        try:
            n_items = len(loader.dataset)
        except Exception:
            n_items = None

    if n_items is not None:
        preds = np.empty(n_items, dtype=np.float32)
        ys = np.empty(n_items, dtype=np.int64)
        have_y = True
        pos = 0
        for x, y in _tqdm(loader, total=len(loader), leave=False):
            b = x.shape[0]
            x = x.to(device, non_blocking=True)
            p = model(x).view(-1).detach().cpu().numpy().astype(np.float32, copy=False)
            preds[pos : pos + b] = p
            if y.numel() > 0:
                ys[pos : pos + b] = (
                    y.view(-1).cpu().numpy().astype(np.int64, copy=False)
                )
            else:
                have_y = False
            pos += b
        preds = preds[:pos]
        ys = ys[:pos]
        return preds, (ys if have_y else None)
    else:
        preds_list = []
        ys_list = []
        for x, y in _tqdm(loader, total=len(loader), leave=False):
            x = x.to(device, non_blocking=True)
            p = model(x).view(-1).detach().cpu().numpy()
            preds_list.append(p.astype(np.float32, copy=False))
            if y.numel() > 0:
                ys_list.append(y.view(-1).cpu().numpy().astype(np.int64, copy=False))
        preds = (
            np.concatenate(preds_list, axis=0)
            if preds_list
            else np.zeros((0,), np.float32)
        )
        ys = np.concatenate(ys_list, axis=0) if ys_list else None
        return preds, ys


if not loaded_ok:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(best_model.parameters(), lr=2e-4)
    epochs = 2

    for ep in range(1, epochs + 1):
        loss = train_one_epoch(best_model, train_loader, optimizer, criterion)
        print(f"epoch {ep}/{epochs} - train loss: {loss:.5f}")

    torch.save(best_model.state_dict(), model_path)
    print("Saved trained model to:", model_path, "size:", os.path.getsize(model_path))

best_model.eval()




## === cell 7
def tta(x):
    """simple 8 fold TTA"""
    variants = []
    x0 = x
    for flip1 in range(2):  # flip dim=2 (H)
        for flip2 in range(2):  # flip dim=3 (W)
            for trans in range(2):  # transpose H/W
                xi = x0
                if flip1:
                    xi = xi.flip(2)
                if flip2:
                    xi = xi.flip(3)
                if trans:
                    xi = xi.transpose(2, 3)
                variants.append(xi)
    xb = torch.cat(variants, dim=0)  # (8B,C,H,W)
    pb = best_model(xb)[:, 0].view(8, x.shape[0])
    return pb.mean(dim=0)




## === cell 8
val_cont_preds, val_y = predict_continuous(
    best_model, val_loader, n_items=len(val_loader.dataset)
)
print("val preds shape:", val_cont_preds.shape, "val y shape:", val_y.shape)



## === cell 9
import scipy as sp
from sklearn.metrics import cohen_kappa_score
import numpy as np
import torch.nn as nn


class KappaOptimizer(nn.Module):
    def __init__(self, coef=[0.5, 1.5, 2.5, 3.5]):
        super().__init__()
        self.coef = coef
        self.func = self.quad_kappa

    def predict(self, preds):
        return self._predict(self.coef, preds)

    @classmethod
    def _predict(cls, coef, preds):
        coef = np.array(coef, dtype=np.float32)
        if type(preds).__name__ == "Tensor":
            y_hat = preds.detach().float().view(-1).cpu().numpy()
        else:
            y_hat = np.asarray(preds, dtype=np.float32).reshape(-1)

        y_hat = np.clip(y_hat, -1.0, 5.0)
        out = np.digitize(y_hat, bins=coef, right=False).astype(np.int32)
        return torch.from_numpy(out)

    def quad_kappa(self, preds, y):
        return self._quad_kappa(self.coef, preds, y)

    @classmethod
    def _quad_kappa(cls, coef, preds, y):
        y_hat = cls._predict(coef, preds).numpy()
        return cohen_kappa_score(y, y_hat, weights="quadratic")

    def fit(self, preds, y):
        print("Early score:", self.quad_kappa(preds, y))

        def _pack(t):
            t = np.array(t, dtype=np.float64)
            t = np.clip(t, -0.5, 4.5)
            t = np.sort(t)
            eps = 1e-3
            for i in range(1, len(t)):
                if t[i] <= t[i - 1] + eps:
                    t[i] = t[i - 1] + eps
            t = np.clip(t, -0.5, 4.5)
            return t

        def neg_kappa(coef):
            coef = _pack(coef)
            return -self._quad_kappa(coef, preds, y)

        opt_res = sp.optimize.minimize(
            neg_kappa,
            x0=_pack(self.coef),
            method="nelder-mead",
            options={"maxiter": 250, "fatol": 1e-7, "xatol": 1e-7},
        )
        print(opt_res)
        self.coef = _pack(opt_res.x).tolist()
        print("New score:", self.quad_kappa(preds, y))

    def forward(self, preds, y):
        return torch.tensor(self.quad_kappa(preds, y))


from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(df_train), dtype=np.float32)
oof_y = df_train["diagnosis"].values.astype(np.int64)

oof_epochs = 1
criterion = nn.MSELoss()

_oof_loader_kwargs = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

for fold, (tr_idx, va_idx) in enumerate(
    skf.split(np.zeros(len(df_train)), oof_y), start=1
):
    seed_everything(42 + fold)

    fold_model = EfficientNet(
        num_classes=1, width_coefficient=1.4, depth_coefficient=1.8
    ).to(device)

    fold_model.load_state_dict(best_model.state_dict(), strict=True)

    if _CAN_COMPILE:
        try:
            fold_model = torch.compile(
                fold_model, mode="reduce-overhead", fullgraph=False
            )
        except Exception:
            pass

    tr_ds = Subset(train_dataset_full, tr_idx.tolist())
    va_ds = Subset(train_dataset_full, va_idx.tolist())

    tr_ld = DataLoader(
        tr_ds,
        shuffle=True,
        drop_last=False,
        **{k: v for k, v in _oof_loader_kwargs.items() if v is not None},
    )
    va_ld = DataLoader(
        va_ds,
        shuffle=False,
        drop_last=False,
        **{k: v for k, v in _oof_loader_kwargs.items() if v is not None},
    )

    optimizer = torch.optim.Adam(fold_model.parameters(), lr=2e-4)

    for ep in range(1, oof_epochs + 1):
        loss = train_one_epoch(fold_model, tr_ld, optimizer, criterion)
        print(
            f"OOF fold {fold}/{n_splits} - epoch {ep}/{oof_epochs} - train loss: {loss:.5f}"
        )

    va_pred, va_y = predict_continuous(fold_model, va_ld, n_items=len(va_ld.dataset))
    oof_preds[va_idx] = va_pred.astype(np.float32)

kappa_opt = KappaOptimizer([0.5, 1.5, 2.5, 3.5])
kappa_opt.fit(oof_preds, oof_y)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3566481662.py in <cell line: 0>()
     96     ).to(device)
     97 
---> 98     fold_model.load_state_dict(best_model.state_dict(), strict=True)
     99 
    100     # Speed: compile each fold model too (same semantics). If fails, continue.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in load_state_dict(self, state_dict, strict, assign)
   2579 
   2580         if len(error_msgs) > 0:
-> 2581             raise RuntimeError(
   2582                 "Error(s) in loading state_dict for {}:\n\t{}".format(
   2583                     self.__class__.__name__, "\n\t".join(error_msgs)

RuntimeError: Error(s) in loading state_dict for EfficientNet:
	Missing key(s) in state_dict: "stem.0.weight", "stem.1.weight", "stem.1.bias", "stem.1.running_mean", "stem.1.running_var", "blocks.MBConv1_0.depthwise_conv.0.weight", "blocks.MBConv1_0.depthwise_conv.1.weight", "blocks.MBConv1_0.depthwise_conv.1.bias", "blocks.MBConv1_0.depthwise_conv.1.running_mean", "blocks.MBConv1_0.depthwise_conv.1.running_var", "blocks.MBConv1_0.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv1_0.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv1_0.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv1_0.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv1_0.project_conv.0.weight", "blocks.MBConv1_0.project_conv.1.weight", "blocks.MBConv1_0.project_conv.1.bias", "blocks.MBConv1_0.project_conv.1.running_mean", "blocks.MBConv1_0.project_conv.1.running_var", "blocks.MBConv1_1.depthwise_conv.0.weight", "blocks.MBConv1_1.depthwise_conv.1.weight", "blocks.MBConv1_1.depthwise_conv.1.bias", "blocks.MBConv1_1.depthwise_conv.1.running_mean", "blocks.MBConv1_1.depthwise_conv.1.running_var", "blocks.MBConv1_1.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv1_1.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv1_1.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv1_1.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv1_1.project_conv.0.weight", "blocks.MBConv1_1.project_conv.1.weight", "blocks.MBConv1_1.project_conv.1.bias", "blocks.MBConv1_1.project_conv.1.running_mean", "blocks.MBConv1_1.project_conv.1.running_var", "blocks.MBConv6_2.expansion_conv.0.weight", "blocks.MBConv6_2.expansion_conv.1.weight", "blocks.MBConv6_2.expansion_conv.1.bias", "blocks.MBConv6_2.expansion_conv.1.running_mean", "blocks.MBConv6_2.expansion_conv.1.running_var", "blocks.MBConv6_2.depthwise_conv.0.weight", "blocks.MBConv6_2.depthwise_conv.1.weight", "blocks.MBConv6_2.depthwise_conv.1.bias", "blocks.MBConv6_2.depthwise_conv.1.running_mean", "blocks.MBConv6_2.depthwise_conv.1.running_var", "blocks.MBConv6_2.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_2.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_2.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_2.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_2.project_conv.0.weight", "blocks.MBConv6_2.project_conv.1.weight", "blocks.MBConv6_2.project_conv.1.bias", "blocks.MBConv6_2.project_conv.1.running_mean", "blocks.MBConv6_2.project_conv.1.running_var", "blocks.MBConv6_3.expansion_conv.0.weight", "blocks.MBConv6_3.expansion_conv.1.weight", "blocks.MBConv6_3.expansion_conv.1.bias", "blocks.MBConv6_3.expansion_conv.1.running_mean", "blocks.MBConv6_3.expansion_conv.1.running_var", "blocks.MBConv6_3.depthwise_conv.0.weight", "blocks.MBConv6_3.depthwise_conv.1.weight", "blocks.MBConv6_3.depthwise_conv.1.bias", "blocks.MBConv6_3.depthwise_conv.1.running_mean", "blocks.MBConv6_3.depthwise_conv.1.running_var", "blocks.MBConv6_3.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_3.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_3.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_3.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_3.project_conv.0.weight", "blocks.MBConv6_3.project_conv.1.weight", "blocks.MBConv6_3.project_conv.1.bias", "blocks.MBConv6_3.project_conv.1.running_mean", "blocks.MBConv6_3.project_conv.1.running_var", "blocks.MBConv6_4.expansion_conv.0.weight", "blocks.MBConv6_4.expansion_conv.1.weight", "blocks.MBConv6_4.expansion_conv.1.bias", "blocks.MBConv6_4.expansion_conv.1.running_mean", "blocks.MBConv6_4.expansion_conv.1.running_var", "blocks.MBConv6_4.depthwise_conv.0.weight", "blocks.MBConv6_4.depthwise_conv.1.weight", "blocks.MBConv6_4.depthwise_conv.1.bias", "blocks.MBConv6_4.depthwise_conv.1.running_mean", "blocks.MBConv6_4.depthwise_conv.1.running_var", "blocks.MBConv6_4.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_4.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_4.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_4.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_4.project_conv.0.weight", "blocks.MBConv6_4.project_conv.1.weight", "blocks.MBConv6_4.project_conv.1.bias", "blocks.MBConv6_4.project_conv.1.running_mean", "blocks.MBConv6_4.project_conv.1.running_var", "blocks.MBConv6_5.expansion_conv.0.weight", "blocks.MBConv6_5.expansion_conv.1.weight", "blocks.MBConv6_5.expansion_conv.1.bias", "blocks.MBConv6_5.expansion_conv.1.running_mean", "blocks.MBConv6_5.expansion_conv.1.running_var", "blocks.MBConv6_5.depthwise_conv.0.weight", "blocks.MBConv6_5.depthwise_conv.1.weight", "blocks.MBConv6_5.depthwise_conv.1.bias", "blocks.MBConv6_5.depthwise_conv.1.running_mean", "blocks.MBConv6_5.depthwise_conv.1.running_var", "blocks.MBConv6_5.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_5.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_5.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_5.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_5.project_conv.0.weight", "blocks.MBConv6_5.project_conv.1.weight", "blocks.MBConv6_5.project_conv.1.bias", "blocks.MBConv6_5.project_conv.1.running_mean", "blocks.MBConv6_5.project_conv.1.running_var", "blocks.MBConv6_6.expansion_conv.0.weight", "blocks.MBConv6_6.expansion_conv.1.weight", "blocks.MBConv6_6.expansion_conv.1.bias", "blocks.MBConv6_6.expansion_conv.1.running_mean", "blocks.MBConv6_6.expansion_conv.1.running_var", "blocks.MBConv6_6.depthwise_conv.0.weight", "blocks.MBConv6_6.depthwise_conv.1.weight", "blocks.MBConv6_6.depthwise_conv.1.bias", "blocks.MBConv6_6.depthwise_conv.1.running_mean", "blocks.MBConv6_6.depthwise_conv.1.running_var", "blocks.MBConv6_6.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_6.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_6.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_6.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_6.project_conv.0.weight", "blocks.MBConv6_6.project_conv.1.weight", "blocks.MBConv6_6.project_conv.1.bias", "blocks.MBConv6_6.project_conv.1.running_mean", "blocks.MBConv6_6.project_conv.1.running_var", "blocks.MBConv6_7.expansion_conv.0.weight", "blocks.MBConv6_7.expansion_conv.1.weight", "blocks.MBConv6_7.expansion_conv.1.bias", "blocks.MBConv6_7.expansion_conv.1.running_mean", "blocks.MBConv6_7.expansion_conv.1.running_var", "blocks.MBConv6_7.depthwise_conv.0.weight", "blocks.MBConv6_7.depthwise_conv.1.weight", "blocks.MBConv6_7.depthwise_conv.1.bias", "blocks.MBConv6_7.depthwise_conv.1.running_mean", "blocks.MBConv6_7.depthwise_conv.1.running_var", "blocks.MBConv6_7.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_7.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_7.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_7.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_7.project_conv.0.weight", "blocks.MBConv6_7.project_conv.1.weight", "blocks.MBConv6_7.project_conv.1.bias", "blocks.MBConv6_7.project_conv.1.running_mean", "blocks.MBConv6_7.project_conv.1.running_var", "blocks.MBConv6_8.expansion_conv.0.weight", "blocks.MBConv6_8.expansion_conv.1.weight", "blocks.MBConv6_8.expansion_conv.1.bias", "blocks.MBConv6_8.expansion_conv.1.running_mean", "blocks.MBConv6_8.expansion_conv.1.running_var", "blocks.MBConv6_8.depthwise_conv.0.weight", "blocks.MBConv6_8.depthwise_conv.1.weight", "blocks.MBConv6_8.depthwise_conv.1.bias", "blocks.MBConv6_8.depthwise_conv.1.running_mean", "blocks.MBConv6_8.depthwise_conv.1.running_var", "blocks.MBConv6_8.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_8.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_8.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_8.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_8.project_conv.0.weight", "blocks.MBConv6_8.project_conv.1.weight", "blocks.MBConv6_8.project_conv.1.bias", "blocks.MBConv6_8.project_conv.1.running_mean", "blocks.MBConv6_8.project_conv.1.running_var", "blocks.MBConv6_9.expansion_conv.0.weight", "blocks.MBConv6_9.expansion_conv.1.weight", "blocks.MBConv6_9.expansion_conv.1.bias", "blocks.MBConv6_9.expansion_conv.1.running_mean", "blocks.MBConv6_9.expansion_conv.1.running_var", "blocks.MBConv6_9.depthwise_conv.0.weight", "blocks.MBConv6_9.depthwise_conv.1.weight", "blocks.MBConv6_9.depthwise_conv.1.bias", "blocks.MBConv6_9.depthwise_conv.1.running_mean", "blocks.MBConv6_9.depthwise_conv.1.running_var", "blocks.MBConv6_9.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_9.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_9.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_9.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_9.project_conv.0.weight", "blocks.MBConv6_9.project_conv.1.weight", "blocks.MBConv6_9.project_conv.1.bias", "blocks.MBConv6_9.project_conv.1.running_mean", "blocks.MBConv6_9.project_conv.1.running_var", "blocks.MBConv6_10.expansion_conv.0.weight", "blocks.MBConv6_10.expansion_conv.1.weight", "blocks.MBConv6_10.expansion_conv.1.bias", "blocks.MBConv6_10.expansion_conv.1.running_mean", "blocks.MBConv6_10.expansion_conv.1.running_var", "blocks.MBConv6_10.depthwise_conv.0.weight", "blocks.MBConv6_10.depthwise_conv.1.weight", "blocks.MBConv6_10.depthwise_conv.1.bias", "blocks.MBConv6_10.depthwise_conv.1.running_mean", "blocks.MBConv6_10.depthwise_conv.1.running_var", "blocks.MBConv6_10.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_10.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_10.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_10.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_10.project_conv.0.weight", "blocks.MBConv6_10.project_conv.1.weight", "blocks.MBConv6_10.project_conv.1.bias", "blocks.MBConv6_10.project_conv.1.running_mean", "blocks.MBConv6_10.project_conv.1.running_var", "blocks.MBConv6_11.expansion_conv.0.weight", "blocks.MBConv6_11.expansion_conv.1.weight", "blocks.MBConv6_11.expansion_conv.1.bias", "blocks.MBConv6_11.expansion_conv.1.running_mean", "blocks.MBConv6_11.expansion_conv.1.running_var", "blocks.MBConv6_11.depthwise_conv.0.weight", "blocks.MBConv6_11.depthwise_conv.1.weight", "blocks.MBConv6_11.depthwise_conv.1.bias", "blocks.MBConv6_11.depthwise_conv.1.running_mean", "blocks.MBConv6_11.depthwise_conv.1.running_var", "blocks.MBConv6_11.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_11.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_11.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_11.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_11.project_conv.0.weight", "blocks.MBConv6_11.project_conv.1.weight", "blocks.MBConv6_11.project_conv.1.bias", "blocks.MBConv6_11.project_conv.1.running_mean", "blocks.MBConv6_11.project_conv.1.running_var", "blocks.MBConv6_12.expansion_conv.0.weight", "blocks.MBConv6_12.expansion_conv.1.weight", "blocks.MBConv6_12.expansion_conv.1.bias", "blocks.MBConv6_12.expansion_conv.1.running_mean", "blocks.MBConv6_12.expansion_conv.1.running_var", "blocks.MBConv6_12.depthwise_conv.0.weight", "blocks.MBConv6_12.depthwise_conv.1.weight", "blocks.MBConv6_12.depthwise_conv.1.bias", "blocks.MBConv6_12.depthwise_conv.1.running_mean", "blocks.MBConv6_12.depthwise_conv.1.running_var", "blocks.MBConv6_12.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_12.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_12.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_12.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_12.project_conv.0.weight", "blocks.MBConv6_12.project_conv.1.weight", "blocks.MBConv6_12.project_conv.1.bias", "blocks.MBConv6_12.project_conv.1.running_mean", "blocks.MBConv6_12.project_conv.1.running_var", "blocks.MBConv6_13.expansion_conv.0.weight", "blocks.MBConv6_13.expansion_conv.1.weight", "blocks.MBConv6_13.expansion_conv.1.bias", "blocks.MBConv6_13.expansion_conv.1.running_mean", "blocks.MBConv6_13.expansion_conv.1.running_var", "blocks.MBConv6_13.depthwise_conv.0.weight", "blocks.MBConv6_13.depthwise_conv.1.weight", "blocks.MBConv6_13.depthwise_conv.1.bias", "blocks.MBConv6_13.depthwise_conv.1.running_mean", "blocks.MBConv6_13.depthwise_conv.1.running_var", "blocks.MBConv6_13.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_13.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_13.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_13.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_13.project_conv.0.weight", "blocks.MBConv6_13.project_conv.1.weight", "blocks.MBConv6_13.project_conv.1.bias", "blocks.MBConv6_13.project_conv.1.running_mean", "blocks.MBConv6_13.project_conv.1.running_var", "blocks.MBConv6_14.expansion_conv.0.weight", "blocks.MBConv6_14.expansion_conv.1.weight", "blocks.MBConv6_14.expansion_conv.1.bias", "blocks.MBConv6_14.expansion_conv.1.running_mean", "blocks.MBConv6_14.expansion_conv.1.running_var", "blocks.MBConv6_14.depthwise_conv.0.weight", "blocks.MBConv6_14.depthwise_conv.1.weight", "blocks.MBConv6_14.depthwise_conv.1.bias", "blocks.MBConv6_14.depthwise_conv.1.running_mean", "blocks.MBConv6_14.depthwise_conv.1.running_var", "blocks.MBConv6_14.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_14.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_14.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_14.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_14.project_conv.0.weight", "blocks.MBConv6_14.project_conv.1.weight", "blocks.MBConv6_14.project_conv.1.bias", "blocks.MBConv6_14.project_conv.1.running_mean", "blocks.MBConv6_14.project_conv.1.running_var", "blocks.MBConv6_15.expansion_conv.0.weight", "blocks.MBConv6_15.expansion_conv.1.weight", "blocks.MBConv6_15.expansion_conv.1.bias", "blocks.MBConv6_15.expansion_conv.1.running_mean", "blocks.MBConv6_15.expansion_conv.1.running_var", "blocks.MBConv6_15.depthwise_conv.0.weight", "blocks.MBConv6_15.depthwise_conv.1.weight", "blocks.MBConv6_15.depthwise_conv.1.bias", "blocks.MBConv6_15.depthwise_conv.1.running_mean", "blocks.MBConv6_15.depthwise_conv.1.running_var", "blocks.MBConv6_15.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_15.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_15.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_15.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_15.project_conv.0.weight", "blocks.MBConv6_15.project_conv.1.weight", "blocks.MBConv6_15.project_conv.1.bias", "blocks.MBConv6_15.project_conv.1.running_mean", "blocks.MBConv6_15.project_conv.1.running_var", "blocks.MBConv6_16.expansion_conv.0.weight", "blocks.MBConv6_16.expansion_conv.1.weight", "blocks.MBConv6_16.expansion_conv.1.bias", "blocks.MBConv6_16.expansion_conv.1.running_mean", "blocks.MBConv6_16.expansion_conv.1.running_var", "blocks.MBConv6_16.depthwise_conv.0.weight", "blocks.MBConv6_16.depthwise_conv.1.weight", "blocks.MBConv6_16.depthwise_conv.1.bias", "blocks.MBConv6_16.depthwise_conv.1.running_mean", "blocks.MBConv6_16.depthwise_conv.1.running_var", "blocks.MBConv6_16.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_16.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_16.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_16.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_16.project_conv.0.weight", "blocks.MBConv6_16.project_conv.1.weight", "blocks.MBConv6_16.project_conv.1.bias", "blocks.MBConv6_16.project_conv.1.running_mean", "blocks.MBConv6_16.project_conv.1.running_var", "blocks.MBConv6_17.expansion_conv.0.weight", "blocks.MBConv6_17.expansion_conv.1.weight", "blocks.MBConv6_17.expansion_conv.1.bias", "blocks.MBConv6_17.expansion_conv.1.running_mean", "blocks.MBConv6_17.expansion_conv.1.running_var", "blocks.MBConv6_17.depthwise_conv.0.weight", "blocks.MBConv6_17.depthwise_conv.1.weight", "blocks.MBConv6_17.depthwise_conv.1.bias", "blocks.MBConv6_17.depthwise_conv.1.running_mean", "blocks.MBConv6_17.depthwise_conv.1.running_var", "blocks.MBConv6_17.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_17.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_17.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_17.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_17.project_conv.0.weight", "blocks.MBConv6_17.project_conv.1.weight", "blocks.MBConv6_17.project_conv.1.bias", "blocks.MBConv6_17.project_conv.1.running_mean", "blocks.MBConv6_17.project_conv.1.running_var", "blocks.MBConv6_18.expansion_conv.0.weight", "blocks.MBConv6_18.expansion_conv.1.weight", "blocks.MBConv6_18.expansion_conv.1.bias", "blocks.MBConv6_18.expansion_conv.1.running_mean", "blocks.MBConv6_18.expansion_conv.1.running_var", "blocks.MBConv6_18.depthwise_conv.0.weight", "blocks.MBConv6_18.depthwise_conv.1.weight", "blocks.MBConv6_18.depthwise_conv.1.bias", "blocks.MBConv6_18.depthwise_conv.1.running_mean", "blocks.MBConv6_18.depthwise_conv.1.running_var", "blocks.MBConv6_18.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_18.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_18.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_18.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_18.project_conv.0.weight", "blocks.MBConv6_18.project_conv.1.weight", "blocks.MBConv6_18.project_conv.1.bias", "blocks.MBConv6_18.project_conv.1.running_mean", "blocks.MBConv6_18.project_conv.1.running_var", "blocks.MBConv6_19.expansion_conv.0.weight", "blocks.MBConv6_19.expansion_conv.1.weight", "blocks.MBConv6_19.expansion_conv.1.bias", "blocks.MBConv6_19.expansion_conv.1.running_mean", "blocks.MBConv6_19.expansion_conv.1.running_var", "blocks.MBConv6_19.depthwise_conv.0.weight", "blocks.MBConv6_19.depthwise_conv.1.weight", "blocks.MBConv6_19.depthwise_conv.1.bias", "blocks.MBConv6_19.depthwise_conv.1.running_mean", "blocks.MBConv6_19.depthwise_conv.1.running_var", "blocks.MBConv6_19.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_19.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_19.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_19.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_19.project_conv.0.weight", "blocks.MBConv6_19.project_conv.1.weight", "blocks.MBConv6_19.project_conv.1.bias", "blocks.MBConv6_19.project_conv.1.running_mean", "blocks.MBConv6_19.project_conv.1.running_var", "blocks.MBConv6_20.expansion_conv.0.weight", "blocks.MBConv6_20.expansion_conv.1.weight", "blocks.MBConv6_20.expansion_conv.1.bias", "blocks.MBConv6_20.expansion_conv.1.running_mean", "blocks.MBConv6_20.expansion_conv.1.running_var", "blocks.MBConv6_20.depthwise_conv.0.weight", "blocks.MBConv6_20.depthwise_conv.1.weight", "blocks.MBConv6_20.depthwise_conv.1.bias", "blocks.MBConv6_20.depthwise_conv.1.running_mean", "blocks.MBConv6_20.depthwise_conv.1.running_var", "blocks.MBConv6_20.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_20.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_20.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_20.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_20.project_conv.0.weight", "blocks.MBConv6_20.project_conv.1.weight", "blocks.MBConv6_20.project_conv.1.bias", "blocks.MBConv6_20.project_conv.1.running_mean", "blocks.MBConv6_20.project_conv.1.running_var", "blocks.MBConv6_21.expansion_conv.0.weight", "blocks.MBConv6_21.expansion_conv.1.weight", "blocks.MBConv6_21.expansion_conv.1.bias", "blocks.MBConv6_21.expansion_conv.1.running_mean", "blocks.MBConv6_21.expansion_conv.1.running_var", "blocks.MBConv6_21.depthwise_conv.0.weight", "blocks.MBConv6_21.depthwise_conv.1.weight", "blocks.MBConv6_21.depthwise_conv.1.bias", "blocks.MBConv6_21.depthwise_conv.1.running_mean", "blocks.MBConv6_21.depthwise_conv.1.running_var", "blocks.MBConv6_21.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_21.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_21.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_21.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_21.project_conv.0.weight", "blocks.MBConv6_21.project_conv.1.weight", "blocks.MBConv6_21.project_conv.1.bias", "blocks.MBConv6_21.project_conv.1.running_mean", "blocks.MBConv6_21.project_conv.1.running_var", "blocks.MBConv6_22.expansion_conv.0.weight", "blocks.MBConv6_22.expansion_conv.1.weight", "blocks.MBConv6_22.expansion_conv.1.bias", "blocks.MBConv6_22.expansion_conv.1.running_mean", "blocks.MBConv6_22.expansion_conv.1.running_var", "blocks.MBConv6_22.depthwise_conv.0.weight", "blocks.MBConv6_22.depthwise_conv.1.weight", "blocks.MBConv6_22.depthwise_conv.1.bias", "blocks.MBConv6_22.depthwise_conv.1.running_mean", "blocks.MBConv6_22.depthwise_conv.1.running_var", "blocks.MBConv6_22.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_22.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_22.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_22.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_22.project_conv.0.weight", "blocks.MBConv6_22.project_conv.1.weight", "blocks.MBConv6_22.project_conv.1.bias", "blocks.MBConv6_22.project_conv.1.running_mean", "blocks.MBConv6_22.project_conv.1.running_var", "blocks.MBConv6_23.expansion_conv.0.weight", "blocks.MBConv6_23.expansion_conv.1.weight", "blocks.MBConv6_23.expansion_conv.1.bias", "blocks.MBConv6_23.expansion_conv.1.running_mean", "blocks.MBConv6_23.expansion_conv.1.running_var", "blocks.MBConv6_23.depthwise_conv.0.weight", "blocks.MBConv6_23.depthwise_conv.1.weight", "blocks.MBConv6_23.depthwise_conv.1.bias", "blocks.MBConv6_23.depthwise_conv.1.running_mean", "blocks.MBConv6_23.depthwise_conv.1.running_var", "blocks.MBConv6_23.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_23.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_23.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_23.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_23.project_conv.0.weight", "blocks.MBConv6_23.project_conv.1.weight", "blocks.MBConv6_23.project_conv.1.bias", "blocks.MBConv6_23.project_conv.1.running_mean", "blocks.MBConv6_23.project_conv.1.running_var", "blocks.MBConv6_24.expansion_conv.0.weight", "blocks.MBConv6_24.expansion_conv.1.weight", "blocks.MBConv6_24.expansion_conv.1.bias", "blocks.MBConv6_24.expansion_conv.1.running_mean", "blocks.MBConv6_24.expansion_conv.1.running_var", "blocks.MBConv6_24.depthwise_conv.0.weight", "blocks.MBConv6_24.depthwise_conv.1.weight", "blocks.MBConv6_24.depthwise_conv.1.bias", "blocks.MBConv6_24.depthwise_conv.1.running_mean", "blocks.MBConv6_24.depthwise_conv.1.running_var", "blocks.MBConv6_24.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_24.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_24.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_24.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_24.project_conv.0.weight", "blocks.MBConv6_24.project_conv.1.weight", "blocks.MBConv6_24.project_conv.1.bias", "blocks.MBConv6_24.project_conv.1.running_mean", "blocks.MBConv6_24.project_conv.1.running_var", "blocks.MBConv6_25.expansion_conv.0.weight", "blocks.MBConv6_25.expansion_conv.1.weight", "blocks.MBConv6_25.expansion_conv.1.bias", "blocks.MBConv6_25.expansion_conv.1.running_mean", "blocks.MBConv6_25.expansion_conv.1.running_var", "blocks.MBConv6_25.depthwise_conv.0.weight", "blocks.MBConv6_25.depthwise_conv.1.weight", "blocks.MBConv6_25.depthwise_conv.1.bias", "blocks.MBConv6_25.depthwise_conv.1.running_mean", "blocks.MBConv6_25.depthwise_conv.1.running_var", "blocks.MBConv6_25.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_25.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_25.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_25.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_25.project_conv.0.weight", "blocks.MBConv6_25.project_conv.1.weight", "blocks.MBConv6_25.project_conv.1.bias", "blocks.MBConv6_25.project_conv.1.running_mean", "blocks.MBConv6_25.project_conv.1.running_var", "blocks.MBConv6_26.expansion_conv.0.weight", "blocks.MBConv6_26.expansion_conv.1.weight", "blocks.MBConv6_26.expansion_conv.1.bias", "blocks.MBConv6_26.expansion_conv.1.running_mean", "blocks.MBConv6_26.expansion_conv.1.running_var", "blocks.MBConv6_26.depthwise_conv.0.weight", "blocks.MBConv6_26.depthwise_conv.1.weight", "blocks.MBConv6_26.depthwise_conv.1.bias", "blocks.MBConv6_26.depthwise_conv.1.running_mean", "blocks.MBConv6_26.depthwise_conv.1.running_var", "blocks.MBConv6_26.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_26.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_26.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_26.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_26.project_conv.0.weight", "blocks.MBConv6_26.project_conv.1.weight", "blocks.MBConv6_26.project_conv.1.bias", "blocks.MBConv6_26.project_conv.1.running_mean", "blocks.MBConv6_26.project_conv.1.running_var", "blocks.MBConv6_27.expansion_conv.0.weight", "blocks.MBConv6_27.expansion_conv.1.weight", "blocks.MBConv6_27.expansion_conv.1.bias", "blocks.MBConv6_27.expansion_conv.1.running_mean", "blocks.MBConv6_27.expansion_conv.1.running_var", "blocks.MBConv6_27.depthwise_conv.0.weight", "blocks.MBConv6_27.depthwise_conv.1.weight", "blocks.MBConv6_27.depthwise_conv.1.bias", "blocks.MBConv6_27.depthwise_conv.1.running_mean", "blocks.MBConv6_27.depthwise_conv.1.running_var", "blocks.MBConv6_27.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_27.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_27.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_27.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_27.project_conv.0.weight", "blocks.MBConv6_27.project_conv.1.weight", "blocks.MBConv6_27.project_conv.1.bias", "blocks.MBConv6_27.project_conv.1.running_mean", "blocks.MBConv6_27.project_conv.1.running_var", "blocks.MBConv6_28.expansion_conv.0.weight", "blocks.MBConv6_28.expansion_conv.1.weight", "blocks.MBConv6_28.expansion_conv.1.bias", "blocks.MBConv6_28.expansion_conv.1.running_mean", "blocks.MBConv6_28.expansion_conv.1.running_var", "blocks.MBConv6_28.depthwise_conv.0.weight", "blocks.MBConv6_28.depthwise_conv.1.weight", "blocks.MBConv6_28.depthwise_conv.1.bias", "blocks.MBConv6_28.depthwise_conv.1.running_mean", "blocks.MBConv6_28.depthwise_conv.1.running_var", "blocks.MBConv6_28.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_28.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_28.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_28.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_28.project_conv.0.weight", "blocks.MBConv6_28.project_conv.1.weight", "blocks.MBConv6_28.project_conv.1.bias", "blocks.MBConv6_28.project_conv.1.running_mean", "blocks.MBConv6_28.project_conv.1.running_var", "blocks.MBConv6_29.expansion_conv.0.weight", "blocks.MBConv6_29.expansion_conv.1.weight", "blocks.MBConv6_29.expansion_conv.1.bias", "blocks.MBConv6_29.expansion_conv.1.running_mean", "blocks.MBConv6_29.expansion_conv.1.running_var", "blocks.MBConv6_29.depthwise_conv.0.weight", "blocks.MBConv6_29.depthwise_conv.1.weight", "blocks.MBConv6_29.depthwise_conv.1.bias", "blocks.MBConv6_29.depthwise_conv.1.running_mean", "blocks.MBConv6_29.depthwise_conv.1.running_var", "blocks.MBConv6_29.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_29.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_29.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_29.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_29.project_conv.0.weight", "blocks.MBConv6_29.project_conv.1.weight", "blocks.MBConv6_29.project_conv.1.bias", "blocks.MBConv6_29.project_conv.1.running_mean", "blocks.MBConv6_29.project_conv.1.running_var", "blocks.MBConv6_30.expansion_conv.0.weight", "blocks.MBConv6_30.expansion_conv.1.weight", "blocks.MBConv6_30.expansion_conv.1.bias", "blocks.MBConv6_30.expansion_conv.1.running_mean", "blocks.MBConv6_30.expansion_conv.1.running_var", "blocks.MBConv6_30.depthwise_conv.0.weight", "blocks.MBConv6_30.depthwise_conv.1.weight", "blocks.MBConv6_30.depthwise_conv.1.bias", "blocks.MBConv6_30.depthwise_conv.1.running_mean", "blocks.MBConv6_30.depthwise_conv.1.running_var", "blocks.MBConv6_30.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_30.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_30.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_30.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_30.project_conv.0.weight", "blocks.MBConv6_30.project_conv.1.weight", "blocks.MBConv6_30.project_conv.1.bias", "blocks.MBConv6_30.project_conv.1.running_mean", "blocks.MBConv6_30.project_conv.1.running_var", "blocks.MBConv6_31.expansion_conv.0.weight", "blocks.MBConv6_31.expansion_conv.1.weight", "blocks.MBConv6_31.expansion_conv.1.bias", "blocks.MBConv6_31.expansion_conv.1.running_mean", "blocks.MBConv6_31.expansion_conv.1.running_var", "blocks.MBConv6_31.depthwise_conv.0.weight", "blocks.MBConv6_31.depthwise_conv.1.weight", "blocks.MBConv6_31.depthwise_conv.1.bias", "blocks.MBConv6_31.depthwise_conv.1.running_mean", "blocks.MBConv6_31.depthwise_conv.1.running_var", "blocks.MBConv6_31.squeeze_excitation.reduce_expand.0.weight", "blocks.MBConv6_31.squeeze_excitation.reduce_expand.0.bias", "blocks.MBConv6_31.squeeze_excitation.reduce_expand.2.weight", "blocks.MBConv6_31.squeeze_excitation.reduce_expand.2.bias", "blocks.MBConv6_31.project_conv.0.weight", "blocks.MBConv6_31.project_conv.1.weight", "blocks.MBConv6_31.project_conv.1.bias", "blocks.MBConv6_31.project_conv.1.running_mean", "blocks.MBConv6_31.project_conv.1.running_var", "head.0.weight", "head.1.weight", "head.1.bias", "head.1.running_mean", "head.1.running_var", "head.6.weight", "head.6.bias". 
	Unexpected key(s) in state_dict: "_orig_mod.stem.0.weight", "_orig_mod.stem.1.weight", "_orig_mod.stem.1.bias", "_orig_mod.stem.1.running_mean", "_orig_mod.stem.1.running_var", "_orig_mod.stem.1.num_batches_tracked", "_orig_mod.blocks.MBConv1_0.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv1_0.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv1_0.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv1_0.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv1_0.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv1_0.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv1_0.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv1_0.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv1_0.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv1_0.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv1_0.project_conv.0.weight", "_orig_mod.blocks.MBConv1_0.project_conv.1.weight", "_orig_mod.blocks.MBConv1_0.project_conv.1.bias", "_orig_mod.blocks.MBConv1_0.project_conv.1.running_mean", "_orig_mod.blocks.MBConv1_0.project_conv.1.running_var", "_orig_mod.blocks.MBConv1_0.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv1_1.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv1_1.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv1_1.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv1_1.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv1_1.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv1_1.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv1_1.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv1_1.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv1_1.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv1_1.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv1_1.project_conv.0.weight", "_orig_mod.blocks.MBConv1_1.project_conv.1.weight", "_orig_mod.blocks.MBConv1_1.project_conv.1.bias", "_orig_mod.blocks.MBConv1_1.project_conv.1.running_mean", "_orig_mod.blocks.MBConv1_1.project_conv.1.running_var", "_orig_mod.blocks.MBConv1_1.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_2.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_2.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_2.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_2.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_2.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_2.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_2.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_2.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_2.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_2.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_2.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_2.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_2.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_2.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_2.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_2.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_2.project_conv.0.weight", "_orig_mod.blocks.MBConv6_2.project_conv.1.weight", "_orig_mod.blocks.MBConv6_2.project_conv.1.bias", "_orig_mod.blocks.MBConv6_2.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_2.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_2.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_3.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_3.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_3.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_3.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_3.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_3.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_3.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_3.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_3.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_3.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_3.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_3.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_3.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_3.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_3.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_3.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_3.project_conv.0.weight", "_orig_mod.blocks.MBConv6_3.project_conv.1.weight", "_orig_mod.blocks.MBConv6_3.project_conv.1.bias", "_orig_mod.blocks.MBConv6_3.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_3.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_3.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_4.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_4.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_4.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_4.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_4.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_4.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_4.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_4.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_4.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_4.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_4.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_4.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_4.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_4.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_4.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_4.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_4.project_conv.0.weight", "_orig_mod.blocks.MBConv6_4.project_conv.1.weight", "_orig_mod.blocks.MBConv6_4.project_conv.1.bias", "_orig_mod.blocks.MBConv6_4.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_4.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_4.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_5.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_5.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_5.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_5.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_5.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_5.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_5.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_5.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_5.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_5.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_5.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_5.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_5.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_5.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_5.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_5.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_5.project_conv.0.weight", "_orig_mod.blocks.MBConv6_5.project_conv.1.weight", "_orig_mod.blocks.MBConv6_5.project_conv.1.bias", "_orig_mod.blocks.MBConv6_5.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_5.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_5.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_6.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_6.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_6.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_6.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_6.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_6.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_6.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_6.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_6.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_6.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_6.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_6.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_6.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_6.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_6.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_6.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_6.project_conv.0.weight", "_orig_mod.blocks.MBConv6_6.project_conv.1.weight", "_orig_mod.blocks.MBConv6_6.project_conv.1.bias", "_orig_mod.blocks.MBConv6_6.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_6.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_6.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_7.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_7.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_7.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_7.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_7.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_7.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_7.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_7.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_7.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_7.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_7.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_7.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_7.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_7.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_7.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_7.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_7.project_conv.0.weight", "_orig_mod.blocks.MBConv6_7.project_conv.1.weight", "_orig_mod.blocks.MBConv6_7.project_conv.1.bias", "_orig_mod.blocks.MBConv6_7.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_7.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_7.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_8.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_8.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_8.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_8.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_8.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_8.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_8.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_8.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_8.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_8.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_8.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_8.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_8.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_8.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_8.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_8.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_8.project_conv.0.weight", "_orig_mod.blocks.MBConv6_8.project_conv.1.weight", "_orig_mod.blocks.MBConv6_8.project_conv.1.bias", "_orig_mod.blocks.MBConv6_8.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_8.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_8.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_9.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_9.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_9.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_9.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_9.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_9.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_9.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_9.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_9.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_9.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_9.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_9.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_9.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_9.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_9.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_9.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_9.project_conv.0.weight", "_orig_mod.blocks.MBConv6_9.project_conv.1.weight", "_orig_mod.blocks.MBConv6_9.project_conv.1.bias", "_orig_mod.blocks.MBConv6_9.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_9.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_9.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_10.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_10.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_10.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_10.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_10.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_10.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_10.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_10.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_10.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_10.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_10.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_10.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_10.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_10.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_10.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_10.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_10.project_conv.0.weight", "_orig_mod.blocks.MBConv6_10.project_conv.1.weight", "_orig_mod.blocks.MBConv6_10.project_conv.1.bias", "_orig_mod.blocks.MBConv6_10.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_10.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_10.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_11.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_11.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_11.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_11.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_11.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_11.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_11.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_11.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_11.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_11.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_11.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_11.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_11.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_11.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_11.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_11.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_11.project_conv.0.weight", "_orig_mod.blocks.MBConv6_11.project_conv.1.weight", "_orig_mod.blocks.MBConv6_11.project_conv.1.bias", "_orig_mod.blocks.MBConv6_11.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_11.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_11.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_12.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_12.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_12.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_12.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_12.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_12.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_12.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_12.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_12.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_12.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_12.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_12.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_12.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_12.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_12.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_12.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_12.project_conv.0.weight", "_orig_mod.blocks.MBConv6_12.project_conv.1.weight", "_orig_mod.blocks.MBConv6_12.project_conv.1.bias", "_orig_mod.blocks.MBConv6_12.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_12.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_12.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_13.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_13.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_13.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_13.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_13.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_13.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_13.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_13.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_13.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_13.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_13.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_13.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_13.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_13.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_13.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_13.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_13.project_conv.0.weight", "_orig_mod.blocks.MBConv6_13.project_conv.1.weight", "_orig_mod.blocks.MBConv6_13.project_conv.1.bias", "_orig_mod.blocks.MBConv6_13.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_13.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_13.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_14.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_14.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_14.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_14.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_14.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_14.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_14.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_14.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_14.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_14.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_14.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_14.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_14.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_14.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_14.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_14.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_14.project_conv.0.weight", "_orig_mod.blocks.MBConv6_14.project_conv.1.weight", "_orig_mod.blocks.MBConv6_14.project_conv.1.bias", "_orig_mod.blocks.MBConv6_14.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_14.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_14.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_15.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_15.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_15.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_15.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_15.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_15.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_15.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_15.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_15.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_15.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_15.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_15.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_15.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_15.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_15.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_15.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_15.project_conv.0.weight", "_orig_mod.blocks.MBConv6_15.project_conv.1.weight", "_orig_mod.blocks.MBConv6_15.project_conv.1.bias", "_orig_mod.blocks.MBConv6_15.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_15.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_15.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_16.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_16.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_16.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_16.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_16.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_16.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_16.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_16.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_16.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_16.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_16.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_16.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_16.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_16.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_16.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_16.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_16.project_conv.0.weight", "_orig_mod.blocks.MBConv6_16.project_conv.1.weight", "_orig_mod.blocks.MBConv6_16.project_conv.1.bias", "_orig_mod.blocks.MBConv6_16.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_16.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_16.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_17.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_17.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_17.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_17.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_17.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_17.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_17.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_17.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_17.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_17.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_17.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_17.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_17.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_17.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_17.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_17.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_17.project_conv.0.weight", "_orig_mod.blocks.MBConv6_17.project_conv.1.weight", "_orig_mod.blocks.MBConv6_17.project_conv.1.bias", "_orig_mod.blocks.MBConv6_17.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_17.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_17.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_18.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_18.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_18.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_18.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_18.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_18.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_18.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_18.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_18.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_18.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_18.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_18.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_18.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_18.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_18.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_18.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_18.project_conv.0.weight", "_orig_mod.blocks.MBConv6_18.project_conv.1.weight", "_orig_mod.blocks.MBConv6_18.project_conv.1.bias", "_orig_mod.blocks.MBConv6_18.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_18.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_18.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_19.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_19.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_19.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_19.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_19.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_19.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_19.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_19.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_19.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_19.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_19.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_19.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_19.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_19.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_19.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_19.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_19.project_conv.0.weight", "_orig_mod.blocks.MBConv6_19.project_conv.1.weight", "_orig_mod.blocks.MBConv6_19.project_conv.1.bias", "_orig_mod.blocks.MBConv6_19.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_19.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_19.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_20.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_20.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_20.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_20.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_20.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_20.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_20.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_20.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_20.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_20.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_20.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_20.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_20.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_20.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_20.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_20.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_20.project_conv.0.weight", "_orig_mod.blocks.MBConv6_20.project_conv.1.weight", "_orig_mod.blocks.MBConv6_20.project_conv.1.bias", "_orig_mod.blocks.MBConv6_20.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_20.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_20.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_21.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_21.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_21.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_21.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_21.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_21.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_21.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_21.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_21.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_21.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_21.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_21.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_21.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_21.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_21.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_21.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_21.project_conv.0.weight", "_orig_mod.blocks.MBConv6_21.project_conv.1.weight", "_orig_mod.blocks.MBConv6_21.project_conv.1.bias", "_orig_mod.blocks.MBConv6_21.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_21.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_21.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_22.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_22.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_22.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_22.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_22.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_22.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_22.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_22.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_22.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_22.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_22.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_22.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_22.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_22.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_22.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_22.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_22.project_conv.0.weight", "_orig_mod.blocks.MBConv6_22.project_conv.1.weight", "_orig_mod.blocks.MBConv6_22.project_conv.1.bias", "_orig_mod.blocks.MBConv6_22.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_22.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_22.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_23.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_23.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_23.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_23.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_23.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_23.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_23.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_23.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_23.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_23.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_23.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_23.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_23.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_23.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_23.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_23.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_23.project_conv.0.weight", "_orig_mod.blocks.MBConv6_23.project_conv.1.weight", "_orig_mod.blocks.MBConv6_23.project_conv.1.bias", "_orig_mod.blocks.MBConv6_23.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_23.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_23.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_24.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_24.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_24.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_24.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_24.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_24.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_24.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_24.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_24.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_24.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_24.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_24.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_24.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_24.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_24.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_24.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_24.project_conv.0.weight", "_orig_mod.blocks.MBConv6_24.project_conv.1.weight", "_orig_mod.blocks.MBConv6_24.project_conv.1.bias", "_orig_mod.blocks.MBConv6_24.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_24.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_24.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_25.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_25.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_25.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_25.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_25.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_25.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_25.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_25.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_25.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_25.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_25.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_25.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_25.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_25.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_25.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_25.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_25.project_conv.0.weight", "_orig_mod.blocks.MBConv6_25.project_conv.1.weight", "_orig_mod.blocks.MBConv6_25.project_conv.1.bias", "_orig_mod.blocks.MBConv6_25.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_25.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_25.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_26.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_26.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_26.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_26.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_26.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_26.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_26.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_26.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_26.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_26.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_26.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_26.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_26.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_26.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_26.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_26.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_26.project_conv.0.weight", "_orig_mod.blocks.MBConv6_26.project_conv.1.weight", "_orig_mod.blocks.MBConv6_26.project_conv.1.bias", "_orig_mod.blocks.MBConv6_26.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_26.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_26.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_27.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_27.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_27.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_27.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_27.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_27.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_27.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_27.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_27.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_27.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_27.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_27.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_27.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_27.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_27.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_27.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_27.project_conv.0.weight", "_orig_mod.blocks.MBConv6_27.project_conv.1.weight", "_orig_mod.blocks.MBConv6_27.project_conv.1.bias", "_orig_mod.blocks.MBConv6_27.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_27.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_27.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_28.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_28.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_28.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_28.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_28.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_28.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_28.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_28.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_28.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_28.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_28.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_28.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_28.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_28.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_28.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_28.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_28.project_conv.0.weight", "_orig_mod.blocks.MBConv6_28.project_conv.1.weight", "_orig_mod.blocks.MBConv6_28.project_conv.1.bias", "_orig_mod.blocks.MBConv6_28.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_28.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_28.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_29.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_29.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_29.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_29.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_29.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_29.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_29.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_29.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_29.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_29.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_29.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_29.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_29.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_29.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_29.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_29.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_29.project_conv.0.weight", "_orig_mod.blocks.MBConv6_29.project_conv.1.weight", "_orig_mod.blocks.MBConv6_29.project_conv.1.bias", "_orig_mod.blocks.MBConv6_29.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_29.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_29.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_30.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_30.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_30.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_30.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_30.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_30.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_30.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_30.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_30.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_30.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_30.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_30.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_30.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_30.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_30.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_30.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_30.project_conv.0.weight", "_orig_mod.blocks.MBConv6_30.project_conv.1.weight", "_orig_mod.blocks.MBConv6_30.project_conv.1.bias", "_orig_mod.blocks.MBConv6_30.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_30.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_30.project_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_31.expansion_conv.0.weight", "_orig_mod.blocks.MBConv6_31.expansion_conv.1.weight", "_orig_mod.blocks.MBConv6_31.expansion_conv.1.bias", "_orig_mod.blocks.MBConv6_31.expansion_conv.1.running_mean", "_orig_mod.blocks.MBConv6_31.expansion_conv.1.running_var", "_orig_mod.blocks.MBConv6_31.expansion_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_31.depthwise_conv.0.weight", "_orig_mod.blocks.MBConv6_31.depthwise_conv.1.weight", "_orig_mod.blocks.MBConv6_31.depthwise_conv.1.bias", "_orig_mod.blocks.MBConv6_31.depthwise_conv.1.running_mean", "_orig_mod.blocks.MBConv6_31.depthwise_conv.1.running_var", "_orig_mod.blocks.MBConv6_31.depthwise_conv.1.num_batches_tracked", "_orig_mod.blocks.MBConv6_31.squeeze_excitation.reduce_expand.0.weight", "_orig_mod.blocks.MBConv6_31.squeeze_excitation.reduce_expand.0.bias", "_orig_mod.blocks.MBConv6_31.squeeze_excitation.reduce_expand.2.weight", "_orig_mod.blocks.MBConv6_31.squeeze_excitation.reduce_expand.2.bias", "_orig_mod.blocks.MBConv6_31.project_conv.0.weight", "_orig_mod.blocks.MBConv6_31.project_conv.1.weight", "_orig_mod.blocks.MBConv6_31.project_conv.1.bias", "_orig_mod.blocks.MBConv6_31.project_conv.1.running_mean", "_orig_mod.blocks.MBConv6_31.project_conv.1.running_var", "_orig_mod.blocks.MBConv6_31.project_conv.1.num_batches_tracked", "_orig_mod.head.0.weight", "_orig_mod.head.1.weight", "_orig_mod.head.1.bias", "_orig_mod.head.1.running_mean", "_orig_mod.head.1.running_var", "_orig_mod.head.1.num_batches_tracked", "_orig_mod.head.6.weight", "_orig_mod.head.6.bias". 

## === cell 10
best_model.eval()
preds = np.empty(len(test_loader.dataset), dtype=np.float32)
pos = 0
with torch.no_grad():
    for x, _ in _tqdm(test_loader, total=len(test_loader), leave=False):
        x = x.to(device, non_blocking=True)
        p = tta(x).detach().cpu().numpy().astype(np.float32, copy=False)
        b = p.shape[0]
        preds[pos : pos + b] = p
        pos += b
preds = preds[:pos]

print("num preds:", len(preds), "num test:", len(df_test))

opt_preds = kappa_opt.predict(preds).cpu().numpy().tolist()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3423185297.py in <cell line: 0>()
     13 print("num preds:", len(preds), "num test:", len(df_test))
     14 
---> 15 opt_preds = kappa_opt.predict(preds).cpu().numpy().tolist()
     16 

NameError: name 'kappa_opt' is not defined

## === cell 11
import numpy as np
import pandas as pd

sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")

if len(opt_preds) != len(df_test):
    raise ValueError(
        f"Prediction length mismatch: {len(opt_preds)} preds vs {len(df_test)} test rows"
    )

out = df_test.copy()
out["diagnosis"] = np.array(opt_preds, dtype=np.int32)

sub = sub[["id_code"]].merge(out, on="id_code", how="left")
if sub["diagnosis"].isna().any():
    raise ValueError(
        "Some test ids missing predictions after merge; check id_code alignment."
    )
sub["diagnosis"] = sub["diagnosis"].astype(np.int32)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/799720561.py in <cell line: 0>()
      4 sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
      5 
----> 6 if len(opt_preds) != len(df_test):
      7     raise ValueError(
      8         f"Prediction length mismatch: {len(opt_preds)} preds vs {len(df_test)} test rows"

NameError: name 'opt_preds' is not defined

## === cell 12
pass



## === cell 13
pass
