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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on an external EfficientNet checkpoint (none exists in your `/input`), and instead train the same EfficientNet-B4-like architecture quickly on the provided training set so inference can run end-to-end. I also fix the mixed CPU/GPU dtype issue in TTA by ensuring the model weights and input tensors live on the same device. Finally, I keep your kappa-thresholding step but fit thresholds on a held-out validation split (instead of fixed constants) so the output is aligned to the quadratic-weighted-kappa metric and produces a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission/prediction issue rather than true model performance, because your checkpoint logic loads `efficientNet_best.pth` even when it was copied from a random `.pth` under `../input`, and you load it with `strict=True` which can silently make the notebook unusable if the weights don’t match (or you end up using an unrelated checkpoint that outputs garbage). I make the checkpoint selection require an actual state_dict match against your EfficientNet-B4-like model; otherwise we fall back to training exactly as you already do. I also ensure the threshold optimizer stays within a sorted, valid range to prevent Nelder-Mead from producing non-monotonic thresholds that collapse predictions into a single class (a common cause of near-zero kappa). These are minimal changes that preserve your core model/training/inference logic while making the pipeline reliably produce a meaningful submission.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most likely caused by invalid/degenerate predictions (e.g., almost all test images mapped into a single class) due to threshold fitting on a weak regressor trained for only 2 epochs. I keep your exact model/training/inference structure, but make one minimal metric-aligned improvement: fit thresholds on out-of-fold (OOF) predictions from the training set using the same model (no architecture change), which stabilizes the kappa-optimized discretization and prevents collapse. I also make the kappa optimizer’s predict step vectorized (same semantics, fewer numerical edge-cases) and ensure predictions are clipped to a sane range before thresholding (doesn’t change core logic, just prevents extreme outliers from breaking threshold optimization). This should move the score upward toward your target without changing the fundamental approach.'
- What this solution (achieved 0.0) has done: 'Most of the timeout comes from doing expensive work repeatedly: recursively globbing/trying to load many checkpoints, running 8x TTA inference over the whole test set, and (when no checkpoint is found) doing a full 5-fold OOF training loop just to fit thresholds. The optimizations below keep the same model, transforms, training loops, loss, and evaluation semantics, but remove redundant checkpoint scanning, make inference faster by batching the 8-way TTA into a single forward with a reshape/transpose view, and cut Python overhead in the data pipeline (faster PIL loading, fewer per-batch syncs). Where training happens, the code keeps exactly the same epochs/splits/optimizer, but speeds it up with safe CUDA settings and less overhead (no extra approximations). These changes are all equivalence-preserving (or only introduce negligible float noise) and are targeted strictly at runtime reduction.'

# 9. Code solution

## === cell 0
import os, glob, sys, subprocess, textwrap, math
from pathlib import Path

print("Listing ../input (top-level only):")
for p in sorted(glob.glob("../input/*")):
    print(" -", p)

import shutil
import torch

model_path = "efficientNet_best.pth"

LIKELY_CKPT_DIRS = [
    "../input",
    "../input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection/aptos2019-blindness-detection",
]

patterns = []
for d in LIKELY_CKPT_DIRS:
    patterns.extend(
        [
            f"{d}/**/efficientNet_*.pth",
            f"{d}/**/efficientnet_*.pth",
            f"{d}/**/efficientnet*.pth",
            f"{d}/**/efficientnet*.pt",
            f"{d}/**/efficientnet*.bin",
            f"{d}/**/efficientnet*.ckpt",
            f"{d}/**/*.pth",
        ]
    )

candidates = []
for pat in patterns:
    if len(candidates) >= 80:
        break
    found = glob.glob(pat, recursive=True)
    if found:
        candidates.extend(found[: max(0, 80 - len(candidates))])


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
                nk = nk.replace("_orig_mod.", "")
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



## === cell 1
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




## === cell 2
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

if device.type == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

best_model = EfficientNet(
    num_classes=1, width_coefficient=1.4, depth_coefficient=1.8
)  # B4-like

loaded_ok = False
if len(candidates) > 0:
    model_sd = best_model.state_dict()
    for cand in candidates[:25]:
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
            print(
                "Copied to:",
                "efficientNet_best.pth",
                "size:",
                os.path.getsize("efficientNet_best.pth"),
            )
            loaded_ok = True
            break
        except Exception:
            continue

if (not loaded_ok) and os.path.exists("efficientNet_best.pth"):
    try:
        state = torch.load("efficientNet_best.pth", map_location="cpu")
        state = _extract_state_dict(state) or state
        if isinstance(state, dict):
            best_model.load_state_dict(state, strict=True)
            loaded_ok = True
            print("Loaded local checkpoint:", "efficientNet_best.pth")
    except Exception as e:
        print(
            "WARNING: Local checkpoint exists but failed to load strictly; will retrain. Error:",
            repr(e),
        )
        loaded_ok = False

if not loaded_ok:
    print(
        "No compatible checkpoint found; will train and then save:",
        "efficientNet_best.pth",
    )

best_model = best_model.to(device)

if device.type == "cuda":
    best_model = best_model.to(memory_format=torch.channels_last)

_CAN_COMPILE = hasattr(torch, "compile") and (device.type == "cuda")

_base_state_dict = {
    k: v.detach().cpu().clone() for k, v in best_model.state_dict().items()
}

if _CAN_COMPILE:
    try:
        best_model = torch.compile(best_model, mode="reduce-overhead", fullgraph=False)
        print("Using torch.compile for speed.")
    except Exception as e:
        print(
            "torch.compile unavailable/failed; continuing uncompiled. Error:", repr(e)
        )



## === cell 3
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor, Normalize
import pandas as pd
from PIL import Image
from PIL.Image import BICUBIC

Image.MAX_IMAGE_PIXELS = None

try:
    Image.warnings.simplefilter("ignore")
except Exception:
    pass

try:
    from PIL import features as _pil_features

    if _pil_features.check("jpg_2000"):
        pass
except Exception:
    pass


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
        im = Image.open(fp)
        try:
            sample = im.convert("RGB")
        finally:
            im.close()
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



## === cell 4
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

batch_size = 16

_cpu = os.cpu_count() or 2
num_workers = min(4, _cpu)
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



## === cell 5
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
        if device.type == "cuda":
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        else:
            x = x.to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
    return total_loss / max(1, len(loader))


@torch.inference_mode()
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
            if device.type == "cuda":
                x = x.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
            else:
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
            if device.type == "cuda":
                x = x.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
            else:
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

    to_save = best_model.state_dict()
    to_save = _extract_state_dict(to_save) or to_save
    torch.save(to_save, model_path)
    print("Saved trained model to:", model_path, "size:", os.path.getsize(model_path))

best_model.eval()

try:
    _sd_now = best_model.state_dict()
    _sd_now = _extract_state_dict(_sd_now) or _sd_now
    _base_state_dict = {k: v.detach().cpu().clone() for k, v in _sd_now.items()}
except Exception:
    pass




## === cell 6
def tta(model, x):
    """simple 8 fold TTA (model passed explicitly to avoid global/compile edge cases)"""
    x0 = x
    x1 = x.flip(2)
    x2 = x.flip(3)
    x3 = x.flip(2).flip(3)
    xt = x.transpose(2, 3)
    x4 = xt
    x5 = xt.flip(2)
    x6 = xt.flip(3)
    x7 = xt.flip(2).flip(3)
    xb = torch.stack([x0, x1, x2, x3, x4, x5, x6, x7], dim=0)  # (8,B,C,H,W)
    xb = xb.view(-1, *x.shape[1:])  # (8B,C,H,W)
    pb = model(xb)[:, 0].view(8, x.shape[0])
    return pb.mean(dim=0)




## === cell 7
pass



## === cell 8
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
from torch.utils.data import DataLoader, Dataset

kappa_opt = KappaOptimizer([0.5, 1.5, 2.5, 3.5])

if not loaded_ok:
    try:
        val_pred, val_y = predict_continuous(
            best_model, val_loader, n_items=len(val_loader.dataset)
        )
        if val_y is None:
            raise RuntimeError("Validation labels missing; cannot fit thresholds.")
        kappa_opt.fit(val_pred, val_y.astype(np.int64, copy=False))
    except Exception as e:
        print(
            "WARNING: KappaOptimizer.fit failed; using default thresholds. Error:",
            repr(e),
        )
        kappa_opt = KappaOptimizer([0.5, 1.5, 2.5, 3.5])
else:
    print(
        "Skipping OOF threshold fitting (loaded checkpoint); using default thresholds:",
        kappa_opt.coef,
    )



## === cell 9
best_model.eval()
preds = np.empty(len(test_loader.dataset), dtype=np.float32)
pos = 0

with torch.inference_mode():
    for x, _ in _tqdm(test_loader, total=len(test_loader), leave=False):
        if device.type == "cuda":
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        else:
            x = x.to(device, non_blocking=True)
        p = tta(best_model, x).detach().cpu().numpy().astype(np.float32, copy=False)
        b = p.shape[0]
        preds[pos : pos + b] = p
        pos += b
preds = preds[:pos]

print("num preds:", len(preds), "num test:", len(df_test))

opt_preds = kappa_opt.predict(preds).cpu().numpy().tolist()



## === cell 10
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
