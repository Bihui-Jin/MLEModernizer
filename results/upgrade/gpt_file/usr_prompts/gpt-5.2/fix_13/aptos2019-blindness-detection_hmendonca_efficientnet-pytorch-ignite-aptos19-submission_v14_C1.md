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

0.703759914575621

# 6. Current score

-0.05995

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.0446) has done: 'I remove the hard dependency on an external `.pth` checkpoint (none exists under `../input` here) by switching to a safe fallback that uses torchvision’s built-in EfficientNet-B4 weights when no local checkpoint is found, while keeping the “EfficientNet-B4 classifier” core logic intact. I also fix the CUDA dtype/device mismatch by ensuring the loaded model weights are moved onto the same device as the inputs and by making inference robust to both plain state_dict and checkpoint dict formats. Finally, I make prediction collection and submission writing run end-to-end, guaranteeing a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.09615) has done: 'I fix the failing ImageNet-weight fallback by replacing the brittle manual state_dict key mapping with torchvision’s native EfficientNet-B4 model, which preserves the “EfficientNet-B4 classifier” core intent but avoids KeyErrors across torchvision versions. I also fix the dtype/device mismatch by ensuring the model is moved to the same device as the input (and by loading checkpoints after the model is on device). Finally, I make inference/submission generation robust so `preds` is always produced and `submission.csv` is written with the required columns and row count.'
- What this solution (achieved 0.0) has done: 'Your current low score is consistent with inference using a randomly initialized 5-class head (because the “custom checkpoint” you copy is very likely not compatible with torchvision’s EfficientNet-B4 key structure), so the model effectively predicts near-random classes. I keep the same EfficientNet-B4 inference core, but make checkpoint discovery stricter (only accept files that actually load cleanly with very few missing/unexpected keys) and otherwise fall back to a deterministic, simple calibration that improves QWK: a class-prior-based constant prediction derived from the training label distribution. This is a minimal change that doesn’t alter the model architecture/training loop (there is none), but should move your score materially upward from ~0.096 toward the target band by avoiding random outputs. The submission writing and row alignment remain unchanged and we still always produce a valid `submission.csv`.'
- What this solution (achieved 0.04267) has done: 'Your current score of 0.0 is consistent with producing essentially constant predictions that don’t correlate with the hidden labels; the simplest legitimate way to move QWK upward (without changing model/training core) is to replace the “modal class” fallback with a small deterministic label-distribution-based fallback that better matches the dataset’s class imbalance. I keep your EfficientNet-B4 inference exactly as-is when a compatible checkpoint is loaded, and only change the fallback branch (when `loaded_custom` is False) to sample predictions from the training label prior using a fixed seed for reproducibility. This typically yields a noticeably higher QWK than always predicting the mode, while remaining minimal and fully within Kaggle constraints. Submission writing, row alignment, and file paths remain unchanged and still always produce a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current score is far below target, and the biggest issue is that when no compatible checkpoint loads you discard the model outputs and replace them with random label-prior samples, which are essentially uncorrelated with test labels and keep QWK near zero. I keep your EfficientNet-B4 inference core intact, but replace only the fallback branch to produce a deterministic constant prediction equal to the training-set median class (a minimal, legitimate baseline that typically yields a noticeably higher QWK than random/prior sampling). I also make the checkpoint “acceptance” stricter so we don’t accidentally use a mostly-mismatched head that behaves like random. Everything else (data loading, transforms, inference loop, submission writing) stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.05878) has done: 'Your 0.0 score is consistent with the current pipeline effectively producing non-informative predictions (either a randomly-initialized 5-class head when using ImageNet fallback, or the median-class constant fallback), which QWK heavily penalizes. I keep your EfficientNet-B4 inference core intact, but (1) make the ImageNet fallback produce meaningful predictions by converting it to a binary “has DR vs no DR” detector using the pretrained 1000-class head (so it’s no longer random), then (2) map that to the 5-class `diagnosis` space with a minimal, deterministic rule calibrated from the training label distribution (no training loop added). This should raise QWK materially from ~0 toward the target direction, while still always writing a valid `submission.csv`. All paths and the submission schema remain unchanged.'
- What this solution (achieved -0.16134) has done: 'Your current score is far below the target, and the biggest issue is that the “ImageNet-head confidence” proxy (max softmax) is essentially unrelated to DR severity, so the fallback predictions remain near-random for QWK. Keeping your same EfficientNet-B4 inference core and no training loop, I change only the fallback branch to use a simple, legitimate image-statistics proxy (brightness + contrast from the resized tensor) which correlates better with fundus image quality/pathology than ImageNet confidence, then keep your existing train-prior quantile mapping to 0–4. I also make the mapping robust by computing all 4 thresholds in one call and enforcing strict monotonicity, which prevents degenerate quantiles from collapsing classes. All I/O paths, submission schema, and the custom-checkpoint path remain unchanged, and we still write a valid `submission.csv`.'
- What this solution (achieved -0.1264) has done: 'Your score is very far below the target (need a large improvement), and the current “brightness/contrast abnormality” fallback is likely negatively correlated with true DR severity, which can drive QWK below 0. I keep your exact inference loop and quantile-to-classes mapping, but flip the fallback scalar so that “more abnormal” corresponds to darker + higher-contrast images in a way that is less likely to be anti-correlated (a minimal change that can move QWK upward without changing architecture or adding training). I also make the fallback statistic robust by computing it in the unnormalized [0,1] space you already reconstruct and using luminance (perceptual grayscale) instead of raw RGB mean, which is still an image-statistics proxy and keeps core semantics intact. Submission writing/format stays identical and still produces a valid `submission.csv`.'
- What this solution (achieved -0.09273) has done: 'Your score is far below the target and the main issue is the fallback branch: the current “luminance abnormality” proxy is likely weak/anti-correlated with true DR severity, which can drive QWK negative. I keep your model/inference loop unchanged, but make the fallback proxy more DR-relevant with a minimal, deterministic change: estimate “lesion-like signal” from the fraction of very bright pixels in the fundus (exudates/flash artifacts) combined with darkness, then keep your same train-prior quantile mapping to 0–4. This preserves the exact evaluation semantics (still outputs 0–4 labels, still no training loop) while making the scalar used for ranking more plausibly aligned with DR severity. Submission writing, row alignment, and all paths remain unchanged.'
- What this solution (achieved -0.05995) has done: 'Your current score is far below the target, and the main issue is still the no-checkpoint “fallback” branch producing a proxy that’s effectively uncorrelated (or anti-correlated) with DR severity. To move the score upward with minimal changes and without adding any training loop, I keep your exact inference pipeline and submission writing, but replace only the fallback scalar with a more DR-relevant, deterministic signal: a simple “red-channel lesion” proxy (high red relative to green/blue, plus edge density and darkness) computed from the same already-loaded tensors. I also make the train-prior quantile thresholding slightly more stable by using mid-quantiles between class CDF steps (reduces ties/degenerate thresholds) while keeping the same “match train class distribution” logic. If a compatible custom checkpoint loads, the model path remains unchanged and predictions still come from argmax logits as before.'

# 9. Code solution

## === cell 0
import os, glob, subprocess, textwrap, sys

subprocess.run("ls -la ../input | head -200", shell=True, check=False)



## === cell 1
import os, glob, shutil

model_path = "efficientNet_best.pth"

candidates = []
patterns = [
    "../input/**/*.pth",
    "../input/**/**/*.pth",
    "../input/**/**/**/*.pth",
    "../input/efficientnet*/**/*.pth",
]
for p in patterns:
    candidates.extend(glob.glob(p, recursive=True))


def score_path(p):
    name = os.path.basename(p).lower()
    s = 0
    if "efficient" in name:
        s += 5
    if "b4" in name:
        s += 2
    if "best" in name:
        s += 3
    if "fold" in name:
        s += 1
    if name.endswith(".pth"):
        s += 1
    return s


candidates = sorted(set(candidates), key=score_path, reverse=True)

ckpt_src = None
if candidates:
    ckpt_src = candidates[0]
    shutil.copyfile(ckpt_src, model_path)

print("Found checkpoint (first by name heuristic):", ckpt_src)
print("Local model_path:", model_path, "exists:", os.path.exists(model_path))



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
        self.drop_connect_rate = torch.tensor(drop_connect_rate, requires_grad=False)

    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate
        drop_mask = torch.rand(x.shape[0], 1, 1, 1) + keep_prob
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


from collections import OrderedDict
import math


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




## === cell 3
import torch
import torchvision
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

image_size = 380

best_model = torchvision.models.efficientnet_b4(
    weights=torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1
)
in_features = best_model.classifier[1].in_features
best_model.classifier[1] = nn.Linear(in_features, 5)
best_model = best_model.to(device)

imagenet_model = torchvision.models.efficientnet_b4(
    weights=torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1
).to(device)
imagenet_model.eval()

loaded_custom = False
ckpt_quality = None  # lower is better
if os.path.exists(model_path):
    try:
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(
            k.startswith("module.") for k in state.keys()
        ):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        if isinstance(state, dict):
            missing, unexpected = best_model.load_state_dict(state, strict=False)
            ckpt_quality = len(missing) + len(unexpected)
            print("Tried loading custom checkpoint:", model_path)
            print(
                "Missing keys:",
                len(missing),
                "Unexpected keys:",
                len(unexpected),
                "Quality:",
                ckpt_quality,
            )

            if ckpt_quality <= 2:
                loaded_custom = True
                print("Accepted custom checkpoint (high compatibility).")
            else:
                best_model = torchvision.models.efficientnet_b4(
                    weights=torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1
                )
                in_features = best_model.classifier[1].in_features
                best_model.classifier[1] = nn.Linear(in_features, 5)
                best_model = best_model.to(device)
                print(
                    "Rejected custom checkpoint due to poor compatibility; using fallback predictions."
                )
        else:
            print("Checkpoint format not understood; using fallback predictions.")
    except Exception as e:
        print("Failed to load checkpoint; using fallback predictions. Error:", repr(e))
else:
    print("No custom checkpoint found; using fallback predictions.")



## === cell 4
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor, Normalize

import os
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


test_transform = Compose(
    [
        Resize((image_size, image_size), BICUBIC),
        ToTensor(),
        Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
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
num_workers = min(4, (os.cpu_count() or 2))
print("num_workers:", num_workers)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=(device.type == "cuda"),
)



## === cell 6
from tqdm.auto import tqdm
import torch
import numpy as np
import pandas as pd

preds = []

best_model.eval()
with torch.no_grad():
    for x, _ in tqdm(test_loader, total=len(test_loader)):
        x = x.to(device, non_blocking=True)

        if loaded_custom:
            logits = best_model(x)
            batch_preds = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
        else:
            mean = torch.tensor([0.42, 0.22, 0.075], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.27, 0.15, 0.081], device=device).view(1, 3, 1, 1)
            x_u = (x * std + mean).clamp(0.0, 1.0)  # ~[0,1]

            r = x_u[:, 0]
            g = x_u[:, 1]
            b = x_u[:, 2]

            lum = 0.2989 * r + 0.5870 * g + 0.1140 * b
            mu = lum.mean(dim=(1, 2))

            red_excess = (r - 0.5 * (g + b)).clamp_min(0.0)
            red_excess_mean = red_excess.mean(dim=(1, 2))

            dx = torch.abs(lum[:, :, 1:] - lum[:, :, :-1])
            dy = torch.abs(lum[:, 1:, :] - lum[:, :-1, :])
            edge = 0.5 * (dx.mean(dim=(1, 2)) + dy.mean(dim=(1, 2)))

            abnormal = (1.0 - mu) + 1.75 * red_excess_mean + 0.75 * edge
            batch_preds = abnormal.detach().cpu().numpy().tolist()

        preds.extend(batch_preds)

print("num preds:", len(preds))

if not loaded_custom:
    tr = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")

    y = tr["diagnosis"].values
    prior = np.bincount(y, minlength=5).astype(np.float64)
    prior = prior / prior.sum()
    cum = np.cumsum(prior)  # cdf for classes 0..4

    abnormal = np.array(preds, dtype=np.float64)

    q_probs = [
        0.5 * (cum[0] + cum[1]),
        0.5 * (cum[1] + cum[2]),
        0.5 * (cum[2] + cum[3]),
        0.5 * (cum[3] + cum[4]),
    ]
    qs = np.quantile(abnormal, q_probs)
    qs = np.asarray(qs, dtype=np.float64)

    eps = 1e-6
    for i in range(1, 4):
        if qs[i] <= qs[i - 1]:
            qs[i] = qs[i - 1] + eps

    q1, q2, q3, q4 = qs.tolist()

    diag = np.zeros_like(abnormal, dtype=np.int64)
    diag[abnormal > q1] = 1
    diag[abnormal > q2] = 2
    diag[abnormal > q3] = 3
    diag[abnormal > q4] = 4
    preds = diag.tolist()

    counts = tr["diagnosis"].value_counts().sort_index()
    print(
        "No compatible custom checkpoint loaded -> using red-excess + edge + darkness proxy + train-prior quantile mapping.",
        "train counts:",
        counts.to_dict(),
        "prior:",
        {i: float(prior[i]) for i in range(5)},
        "quantile probs:",
        [float(p) for p in q_probs],
        "quantile thresholds (abnormal):",
        [float(q1), float(q2), float(q3), float(q4)],
    )



## === cell 7
import numpy as np
import pandas as pd

sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
if len(preds) != len(sub):
    raise ValueError(
        f"Pred length {len(preds)} does not match submission length {len(sub)}"
    )

sub["diagnosis"] = np.array(preds, dtype=int)
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)



## === cell 8
_ = sub.hist()



## === cell 9
tr = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
_ = tr.hist()
