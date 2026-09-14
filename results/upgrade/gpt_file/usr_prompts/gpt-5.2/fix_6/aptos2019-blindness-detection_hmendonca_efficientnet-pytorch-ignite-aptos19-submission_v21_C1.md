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

0.7559410441391895

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on external `.pth` weights (none are attached), and instead train the same EfficientNet-B4-defined model on the provided `train.csv` images so the notebook runs end-to-end and yields a valid `submission.csv`. I also fix the CUDA/CPU dtype/device mismatch by ensuring both model and inputs are on the same device, and make the TTA indexing correct for a 1-output regression head. Finally, I add a small validation split to fit the existing `KappaOptimizer` thresholds on out-of-fold predictions (score-improving vs. fixed thresholds) while keeping the core model architecture and inference semantics the same.'
- What this solution (achieved 0.45446) has done: 'Your current 0.0 score strongly suggests the submission is structurally valid but predictions are effectively untrained/mis-calibrated for QWK (training only 2 epochs from scratch on 3k images is typically near-random). To move toward the 0.7559 target with minimal core-logic change, I (1) keep the exact EfficientNet-B4 regression setup and MSE loss, but (2) increase training epochs to a still-time-safe value and add a simple cosine LR schedule to make those same updates more effective, and (3) fit kappa thresholds on a better signal by collecting validation predictions after training with the same TTA used at test-time. I also clamp regression outputs to the valid label range [0,4] before threshold fitting/prediction (post-processing only) to stabilize the optimizer and reduce wild thresholds. These are minimal, metric-aligned adjustments that should raise score substantially from 0.0 without changing architecture or loss.'

# 9. Code solution

## === cell 0
import os, glob, subprocess, textwrap, sys, math, random
from pathlib import Path

print("Listing ../input (if present):")
if os.path.exists("../input"):
    for p in sorted(glob.glob("../input/*")):
        print(p)
else:
    print("WARNING: ../input not found; will rely on /kaggle/input if available.")

CANDIDATE_ROOTS = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
]
DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.exists(r):
        DATA_ROOT = r
        break
if DATA_ROOT is None:
    for r in ["../input", "/kaggle/input", "/kaggle/data"]:
        if os.path.exists(os.path.join(r, "train.csv")) and os.path.exists(
            os.path.join(r, "train_images")
        ):
            DATA_ROOT = r
            break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate dataset root containing train.csv/train_images."
    )

print("Using DATA_ROOT:", DATA_ROOT)

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_ROOT = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_ROOT = os.path.join(DATA_ROOT, "test_images")

for p in [TRAIN_CSV, TEST_CSV, SAMPLE_SUB, TRAIN_IMG_ROOT, TEST_IMG_ROOT]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing expected path: {p}")

model_path = "efficientNet_best.pth"
candidate_globs = [
    "../input/**/efficientNet_*.pth",
    "../input/**/efficientnet*.pth",
    "/kaggle/input/**/efficientNet_*.pth",
    "/kaggle/input/**/efficientnet*.pth",
    "/kaggle/data/**/efficientNet_*.pth",
    "/kaggle/data/**/efficientnet*.pth",
]
candidates = []
for g in candidate_globs:
    candidates.extend(glob.glob(g, recursive=True))
candidates = sorted(set(candidates))
preferred = [c for c in candidates if "best" in os.path.basename(c).lower()]
chosen = preferred[0] if preferred else (candidates[0] if candidates else None)

if chosen is not None:
    print("Found weight candidate:", chosen)
    with open(chosen, "rb") as fsrc, open(model_path, "wb") as fdst:
        fdst.write(fsrc.read())
    try:
        subprocess.run(["md5sum", chosen], check=False)
        subprocess.run(["md5sum", model_path], check=False)
    except Exception as e:
        print("md5sum not available:", repr(e))
        print("Copied to:", model_path, "size:", os.path.getsize(model_path))
else:
    print(
        "No external EfficientNet .pth weights found; will train weights from scratch and save to:",
        model_path,
    )



## === cell 1
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict


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
        self.drop_connect_rate = torch.tensor(drop_connect_rate, requires_grad=False)

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
            if self.training and self.drop_connect_rate is not None:
                x = self._drop_connect(x)
            x += z
        return x


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_out")
    elif isinstance(module, nn.Linear):
        init_range = 1.0 / math.sqrt(module.weight.shape[1])
        nn.init.uniform_(module.weight, a=-init_range, b=init_range)


class EfficientNet(nn.Module):
    def _setup_repeats(self, num_repeats):
        return int(math.ceil(self.depth_coefficient * num_repeats))

    def _setup_channels(self, num_channels):
        num_channels *= self.width_coefficient
        new_num_channels = math.floor(num_channels / self.divisor + 0.5) * self.divisor
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
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

image_size = 380
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

best_model = EfficientNet(
    num_classes=1, width_coefficient=1.4, depth_coefficient=1.8
)  # B4

weights_loaded_ok = False

if os.path.exists(model_path) and os.path.getsize(model_path) > 1024:
    state = torch.load(model_path, map_location="cpu")
    try:
        if (
            isinstance(state, dict)
            and all(isinstance(k, str) for k in state.keys())
            and any(
                k.startswith("stem.") or k.startswith("blocks.") for k in state.keys()
            )
        ):
            best_model.load_state_dict(state, strict=True)
            weights_loaded_ok = True
            print("Loaded state_dict from", model_path)
        elif (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            best_model.load_state_dict(state["state_dict"], strict=False)
            weights_loaded_ok = True
            print("Loaded checkpoint['state_dict'] from", model_path)
        else:
            best_model.load_state_dict(state, strict=False)
            weights_loaded_ok = True
            print("Loaded weights (non-strict) from", model_path)
    except Exception as e:
        print(
            "WARNING: Could not load weights from model_path; will train from scratch. Error:",
            repr(e),
        )
else:
    if os.path.exists(model_path):
        print(
            "Found model_path but it looks too small to be valid; will retrain:",
            model_path,
            "size:",
            os.path.getsize(model_path),
        )
    else:
        print("No model_path found; will train:", model_path)

best_model = best_model.to(device)



## === cell 3
from torchvision.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from torchvision.transforms import ToTensor, Normalize
import os
from PIL import Image
from PIL.Image import BICUBIC

Image.MAX_IMAGE_PIXELS = None
try:
    Image.draft  # attribute exists on PIL Image objects; just a sanity touch
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
            self.targets = torch.LongTensor(list(targets))

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path + self.extension)
        with Image.open(img_path) as im:
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

train_transform = Compose(
    [
        Resize((image_size, image_size), BICUBIC),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
    ]
)

df_test = pd.read_csv(TEST_CSV)
test_dataset = ImageDataset(
    root=TEST_IMG_ROOT, path_list=df_test.id_code.values, transform=test_transform
)

df_train = pd.read_csv(TRAIN_CSV)
y = df_train["diagnosis"].astype(int).values
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
tr_idx, va_idx = next(splitter.split(df_train["id_code"].values, y))
df_tr = df_train.iloc[tr_idx].reset_index(drop=True)
df_va = df_train.iloc[va_idx].reset_index(drop=True)

train_dataset = ImageDataset(
    root=TRAIN_IMG_ROOT,
    path_list=df_tr.id_code.values,
    targets=df_tr.diagnosis.values,
    transform=train_transform,
)
val_dataset = ImageDataset(
    root=TRAIN_IMG_ROOT,
    path_list=df_va.id_code.values,
    targets=df_va.diagnosis.values,
    transform=test_transform,  # eval transform
)

print(
    "train/val sizes:", len(train_dataset), len(val_dataset), "test:", len(test_dataset)
)



## === cell 4
from torch.utils.data import DataLoader

batch_size = 16
num_workers = min(4, os.cpu_count() or 0)
print("num_workers:", num_workers)

common_loader_kwargs = dict(
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    **{k: v for k, v in common_loader_kwargs.items() if v is not None},
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    **{k: v for k, v in common_loader_kwargs.items() if v is not None},
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    **{k: v for k, v in common_loader_kwargs.items() if v is not None},
)




## === cell 5
def tta(x):
    """simple 8-fold TTA"""
    xs = []
    for flip1 in range(2):  # flip dim 1 (height)
        for flip2 in range(2):  # flip dim 2 (width)
            for trans in range(2):  # transpose H/W
                xx = x
                if flip1:
                    xx = xx.flip(-2)
                if flip2:
                    xx = xx.flip(-1)
                if trans:
                    xx = xx.transpose(-1, -2)
                xs.append(xx)
    x8 = torch.cat(xs, dim=0)  # [8B,C,H,W]
    y8 = best_model(x8).view(-1)  # [8B]
    y8 = y8.view(8, x.shape[0])  # [8,B]
    return y8.mean(dim=0)  # [B]




## === cell 6
from tqdm import tqdm
import time



## === cell 7
need_train = not weights_loaded_ok
if need_train:
    print("Training because usable model weights are not loaded.")
    best_model.train()
    optimizer = torch.optim.Adam(best_model.parameters(), lr=2e-4, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()

    epochs = 10

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=5e-6
    )

    start = time.time()
    for epoch in range(1, epochs + 1):
        running = 0.0
        n = 0
        for xb, yb in tqdm(
            train_loader, desc=f"epoch {epoch}/{epochs} train", leave=False
        ):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            out = best_model(xb).view(-1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)
            n += xb.size(0)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch}: train_loss={running/max(n,1):.5f} lr={lr:.7f} elapsed={time.time()-start:.1f}s"
        )

    torch.save(best_model.state_dict(), model_path)
    print("Saved trained weights to:", model_path)

best_model.eval()

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True



## === cell 8
import scipy as sp
from sklearn.metrics import cohen_kappa_score


class KappaOptimizer(nn.Module):
    def __init__(self, coef=[0.5, 1.5, 2.5, 3.5]):
        super().__init__()
        self.coef = coef
        self.func = self.quad_kappa

    def predict(self, preds):
        return self._predict(self.coef, preds)

    @classmethod
    def _predict(cls, coef, preds):
        if type(preds).__name__ == "Tensor":
            y_hat = preds.detach().clone().view(-1)
            c = torch.tensor(coef, device=y_hat.device, dtype=y_hat.dtype)
            y = torch.bucketize(y_hat, c)  # returns 0..4
            return y.to(dtype=torch.int32)
        else:
            arr = np.asarray(preds, dtype=np.float32).reshape(-1)
            c = np.asarray(coef, dtype=np.float32)
            y = np.digitize(arr, c, right=False)  # returns 0..4
            return torch.from_numpy(y.astype(np.int32, copy=False))

    def quad_kappa(self, preds, y):
        return self._quad_kappa(self.coef, preds, y)

    @classmethod
    def _quad_kappa(cls, coef, preds, y):
        y_hat = cls._predict(coef, preds)
        return cohen_kappa_score(y, y_hat, weights="quadratic")

    def fit(self, preds, y):
        print("Early score:", self.quad_kappa(preds, y))
        neg_kappa = lambda coef: -self._quad_kappa(coef, preds, y)
        opt_res = sp.optimize.minimize(
            neg_kappa,
            x0=self.coef,
            method="nelder-mead",
            options={"maxiter": 250, "fatol": 1e-10, "xatol": 1e-10},
        )
        print(opt_res)

        c = np.array(opt_res.x, dtype=np.float64)
        c = np.sort(c)
        eps = 1e-3
        for i in range(1, len(c)):
            if c[i] <= c[i - 1] + eps:
                c[i] = c[i - 1] + eps

        self.coef = c.tolist()
        print("New coef:", self.coef)
        print("New score:", self.quad_kappa(preds, y))

    def forward(self, preds, y):
        return torch.tensor(self.quad_kappa(preds, y))




## === cell 9
val_preds = []
val_y = []
with torch.inference_mode():
    for xb, yb in tqdm(val_loader, desc="val inference", leave=False):
        xb = xb.to(device, non_blocking=True)
        pred = tta(xb).clamp(0.0, 4.0)
        val_preds.extend(pred.detach().cpu().view(-1).tolist())
        val_y.extend(yb.view(-1).cpu().tolist())

kappa_opt = KappaOptimizer([0.8, 1.5, 2.5, 3.5])
try:
    kappa_opt.fit(val_preds, val_y)
except Exception as e:
    print(
        "WARNING: KappaOptimizer.fit failed; will use default thresholds. Error:",
        repr(e),
    )

preds = []
with torch.inference_mode():
    for x, _ in tqdm(test_loader, total=int(len(test_loader)), desc="test inference"):
        x = x.to(device, non_blocking=True)
        pred = tta(x).clamp(0.0, 4.0)
        preds.extend(pred.detach().cpu().view(-1).tolist())

print("num preds:", len(preds), "expected:", len(df_test))

opt_preds = kappa_opt.predict(preds).tolist()



## === cell 10
sub = pd.read_csv(SAMPLE_SUB)

if len(opt_preds) != len(df_test):
    raise ValueError(
        f"Prediction length {len(opt_preds)} does not match test length {len(df_test)}"
    )

pred_df = pd.DataFrame(
    {
        "id_code": df_test["id_code"].values,
        "diagnosis": np.array(opt_preds, dtype=np.int32),
    }
)

sub = sub[["id_code"]].merge(pred_df, on="id_code", how="left")
if sub["diagnosis"].isna().any():
    missing = sub.loc[sub["diagnosis"].isna(), "id_code"].head().tolist()
    raise ValueError(f"Missing predictions for some id_code, e.g.: {missing}")

sub["diagnosis"] = sub["diagnosis"].astype(np.int32)
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)



## === cell 11
try:
    _ = sub.hist()
except Exception as e:
    print("Histogram skipped:", repr(e))



## === cell 12
tr = pd.read_csv(TRAIN_CSV)
try:
    _ = tr.hist()
except Exception as e:
    print("Histogram skipped:", repr(e))
