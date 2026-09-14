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

0.5991103578929535

# 6. Current score

0.72444

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.08312) has done: 'I remove the hard dependency on an external `efficientNet_*.pth` file (which is missing in your environment) by adding a minimal fallback that runs inference with the same EfficientNet-B4 architecture using torchvision’s pretrained EfficientNet weights. I also fix the CUDA/CPU dtype/device mismatch by explicitly moving the model to the selected device and ensuring inference tensors are on the same device. Finally, I keep the same TTA + argmax classification semantics, and ensure we always write a valid `submission.csv` with the required columns and correct row count/order.'
- What this solution (achieved 0.71626) has done: 'Your current negative kappa is very likely coming from using an ImageNet-pretrained EfficientNet-B4 with a randomly-initialized 5-class head (because no competition finetuned weights are found), which makes predictions close to random. To move the score toward the 0.599 target with minimal core-logic change, I add an in-notebook finetuning step that trains only the classifier head for a small, fixed number of epochs on the provided train set (same architecture, same argmax semantics, same TTA). I also switch normalization to the correct ImageNet stats only when using the torchvision fallback (so inputs match the pretrained backbone’s expectations), while keeping your original normalization when your custom `.pth` is available. Finally, I keep submission alignment checks and ensure `submission.csv` is written as before.'
- What this solution (achieved 0.71705) has done: 'Your current score (0.71626) is higher than the target (0.59911), so we should *reduce* performance slightly and safely to move closer to the target band without changing core modeling logic. The smallest, most controlled lever here is your prediction post-processing: keep the same model, same TTA, same argmax semantics, but apply a tiny, deterministic “temperature” to soften probabilities before argmax (this can modestly change class decisions and typically reduces score). I also make this behavior explicitly gated by a single constant so you can nudge it up/down if the score overshoots, while leaving everything else unchanged. The pipeline still runs end-to-end and writes a valid `submission.csv` with correct alignment and columns.'
- What this solution (achieved 0.08942) has done: 'Your current score (0.71705) is above the target (0.59911), so the goal is to *slightly reduce* performance in a controlled, deterministic way while preserving the same model, TTA structure, and argmax decision rule. The smallest lever is the post-processing temperature already present: we increase it a bit and keep everything else identical, which typically increases prediction uncertainty and shifts some class decisions downward toward the target band. To make this robust and deterministic, I also set seeds and force deterministic algorithms (this doesn’t change the core logic, just stabilizes the effect so the score change is repeatable). The script still run end-to-end and write a valid `submission.csv` with correct alignment and columns.'
- What this solution (achieved 0.72444) has done: 'I fix the crash in the finetuning cell by removing the strict deterministic-algorithms setting that triggers a CuBLAS determinism error on GPU, while keeping seeds and cuDNN settings for reasonable reproducibility. This unblocks training the classifier head (your core logic for improving from near-random torchvision fallback predictions), which should move the score back up toward the 0.599 target range from the current 0.089. I keep the model architecture, preprocessing, training loop, and TTA+argmax semantics unchanged, and still write a valid `submission.csv` with correct alignment. No other score-affecting changes are introduced beyond restoring the intended finetuning to run.'
- What this solution (achieved 0.72444) has done: 'Your current score (0.72444) is higher than the target (0.59911), so we should intentionally and deterministically *decrease* performance slightly to move closer to the target band with minimal risk and without changing the model/training core logic. The smallest controlled lever in your pipeline is the prediction post-processing: we keep the same model, same TTA structure, same argmax rule, but increase the TTA softmax temperature a bit to make decisions less confident and shift some class picks. To keep the change stable, I also explicitly set the model to `eval()` inside `tta()` (so BN/Dropout behavior cannot vary if `tta()` is called elsewhere) while leaving training exactly as-is. Everything else—including data paths, transforms, finetuning head training (when fallback), and submission alignment—remains unchanged.'
- What this solution (achieved 0.72444) has done: 'Your current score (0.72444) is above the target (0.59911), so we should intentionally and deterministically reduce performance slightly to move closer to the target tolerance band, while keeping the same model, TTA structure, and argmax decision rule. The smallest controlled lever is prediction post-processing: increase the softmax temperature used inside TTA so the model is less “peaky,” which typically shifts some borderline decisions and lowers QWK without changing architecture/training. I also add a tiny, deterministic logit “shrink” (a constant multiplier) before softmax; this is equivalent to additional temperature and keeps the change stable and easy to tune without touching training or data. Everything else (paths, transforms, finetuning head training when fallback, submission writing/alignment) remains unchanged and still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os, glob, sys, subprocess, textwrap, math
from pathlib import Path

print("Listing ../input:")
for p in sorted(glob.glob("../input/*")):
    print(" -", p)



## === cell 1
import shutil

target_pattern = "../input/**/efficientNet_*.pth"
model_path = "efficientNet_best.pth"

candidates = sorted(glob.glob(target_pattern, recursive=True))
if candidates:
    src = candidates[0]
    print("Found weights:", src)
    shutil.copyfile(src, model_path)
    print("Copied to:", model_path, "size(bytes)=", os.path.getsize(model_path))
else:
    model_path = None
    print(
        "WARNING: No external efficientNet_*.pth found. "
        "Will use torchvision pretrained EfficientNet-B4 weights instead."
    )



## === cell 2
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

            name = "MBConv{}_{}".format(expand_rate, counter)
            drop_rate = drop_connect_rate * counter / num_blocks
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
            for _ in range(1, num_repeats):
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
import torchvision

image_size = 380
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

use_torchvision_fallback = model_path is None

if not use_torchvision_fallback:
    best_model = EfficientNet(
        num_classes=5, width_coefficient=1.4, depth_coefficient=1.8
    )  # B4
    state = torch.load(model_path, map_location="cpu")
    best_model.load_state_dict(state)
else:
    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

    weights = EfficientNet_B4_Weights.DEFAULT
    best_model = efficientnet_b4(weights=weights)
    in_features = best_model.classifier[-1].in_features
    best_model.classifier[-1] = nn.Linear(in_features, 5)

best_model = best_model.to(device)



## === cell 4
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data import Subset
import torchvision.utils as vutils

import pandas as pd
from PIL import Image
from PIL.Image import BICUBIC


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, path_list, targets=None, transform=None, extension=".png"):
        super().__init__()
        self.root = root
        self.path_list = path_list
        self.targets = targets
        self.transform = transform
        self.extension = extension
        if targets is not None:
            assert len(self.path_list) == len(self.targets)
            self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):
        path = self.path_list[index]
        sample = Image.open(os.path.join(self.root, path + self.extension)).convert(
            "RGB"
        )
        if self.transform is not None:
            sample = self.transform(sample)

        if self.targets is not None:
            return sample, self.targets[index]
        else:
            return sample, torch.LongTensor([])

    def __len__(self):
        return len(self.path_list)


if use_torchvision_fallback:
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
else:
    norm_mean = [0.42, 0.22, 0.075]
    norm_std = [0.27, 0.15, 0.081]

test_transform = Compose(
    [
        Resize((image_size, image_size), BICUBIC),
        ToTensor(),
        Normalize(mean=norm_mean, std=norm_std),
    ]
)

df_test = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_dataset = ImageDataset(
    root="../input/aptos2019-blindness-detection/test_images",
    path_list=df_test.id_code.values,
    transform=test_transform,
)



## === cell 5
from torch.utils.data import DataLoader

batch_size = 16
cpu_cnt = os.cpu_count() or 2
num_workers = min(4, cpu_cnt)
print("num_workers:", num_workers)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)



## === cell 6
from sklearn.model_selection import train_test_split

if use_torchvision_fallback:
    from tqdm import tqdm

    df_train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")

    train_ids, val_ids, train_y, val_y = train_test_split(
        df_train["id_code"].values,
        df_train["diagnosis"].values,
        test_size=0.15,
        random_state=42,
        stratify=df_train["diagnosis"].values,
    )

    train_dataset = ImageDataset(
        root="../input/aptos2019-blindness-detection/train_images",
        path_list=train_ids,
        targets=train_y,
        transform=test_transform,  # keep same preprocessing for train/infer
    )
    val_dataset = ImageDataset(
        root="../input/aptos2019-blindness-detection/train_images",
        path_list=val_ids,
        targets=val_y,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    for p in best_model.parameters():
        p.requires_grad = False
    for p in best_model.classifier.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        best_model.classifier.parameters(), lr=3e-3, weight_decay=1e-4
    )

    epochs = 2
    best_model.train()
    for ep in range(epochs):
        running = 0.0
        for xb, yb in tqdm(
            train_loader,
            desc=f"Finetune head epoch {ep+1}/{epochs}",
            total=len(train_loader),
        ):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = best_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running += float(loss.detach().cpu())
        avg_loss = running / max(1, len(train_loader))
        print(f"epoch={ep+1} train_loss={avg_loss:.4f}")

    best_model.eval()



## === cell 7
TTA_TEMPERATURE = 3.20  # was 2.30
LOGIT_SHRINK = 0.92  # 1.0 would be no change; <1.0 slightly reduces separation


def tta(x):
    """simple 8 fold TTA"""
    best_model.eval()

    pred = []
    for flip1 in range(2):
        for flip2 in range(2):
            for trans in range(2):
                xi = x
                if flip1:
                    xi = xi.flip(2)  # H
                if flip2:
                    xi = xi.flip(3)  # W
                if trans:
                    xi = xi.transpose(-1, -2)
                pred.append(best_model(xi).unsqueeze(0))
    logits = torch.cat(pred, dim=0).mean(dim=0)
    logits = logits * LOGIT_SHRINK
    probs = F.softmax(logits / TTA_TEMPERATURE, dim=-1)
    return probs




## === cell 8
from tqdm import tqdm

preds = []
best_model.eval()
with torch.no_grad():
    for x, _ in tqdm(test_loader, total=len(test_loader)):
        x = x.to(device, non_blocking=True)
        pred = tta(x)
        batch_preds = torch.argmax(pred, dim=-1).detach().cpu().numpy().tolist()
        preds.extend(batch_preds)

print("Num test predictions:", len(preds), "expected:", len(df_test))



## === cell 9
sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")

if len(preds) != len(df_test):
    raise ValueError(f"Prediction length {len(preds)} != test length {len(df_test)}")

sub = sub.merge(df_test[["id_code"]], on="id_code", how="right")
if len(sub) != len(df_test):
    raise ValueError("Submission rows do not match test rows after alignment.")

sub["diagnosis"] = np.array(preds, dtype=int)
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)



## === cell 10
_ = sub.hist()



## === cell 11
tr = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
_ = tr.hist()
