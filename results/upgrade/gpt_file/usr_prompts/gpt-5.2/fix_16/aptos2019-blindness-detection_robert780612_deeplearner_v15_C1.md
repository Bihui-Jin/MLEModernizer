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

3.9

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

0.3225003966818477

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the SSL download crash by preventing any internet weight downloads: if the external checkpoint is missing, the model run with random initialization instead of attempting to fetch ImageNet weights via `model_zoo.load_url`. I also make the dataset path resolution more robust by checking the common Kaggle locations you listed, while keeping the same inference loop and post-processing thresholds. Finally, I ensure the submission is always written as `submission.csv` with the required `id_code,diagnosis` columns and correct row ordering to match `test.csv`. These changes are execution-stability focused (score may be lower without the checkpoint, but it produce a valid submission end-to-end).'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with doing inference using randomly initialized weights because the checkpoint path points to a dataset (`/kaggle/input/seresnet50norm/model30.pth`) that isn’t present in the provided file tree; that yields essentially random predictions and near-zero kappa. To move toward the target score with minimal change and without altering the model/inference logic, I (1) robustly locate an existing `.pth` checkpoint inside the available dataset roots (common in Kaggle notebooks) and load it if found, and (2) fix a small bug where `pretrain` is never enabled even if a checkpoint is present. If no checkpoint exists anywhere, the code still run end-to-end and write `submission.csv` exactly as required, but the score likely remain low.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from running inference with random weights because no checkpoint is actually being loaded (your `pretrain` flag is set to `False` even when a checkpoint exists, and the checkpoint search may pick an unrelated `.pth`). To move toward the target score with minimal changes and identical inference/post-processing semantics, I (1) correctly enable ImageNet pretraining when no DR checkpoint is found (this is the smallest legitimate lift from random), and (2) make checkpoint selection stricter so we only load a plausible DR model (avoid accidentally loading an incompatible/irrelevant `.pth` and getting garbage). Everything else (model architecture, TTA flip, resizing, thresholds, submission formatting) stays the same, and it still always write a valid `submission.csv`.'
- What this solution (achieved 0.01911) has done: 'Your 0.0 score is consistent with producing almost-constant predictions (very poor kappa), which can happen when the DR checkpoint isn’t found and the model runs with a random regression head; even with an ImageNet backbone this often collapses to near a single class after thresholding. To move toward the target with minimal core-logic changes, I (1) force deterministic, safe ImageNet initialization without any internet dependency by loading torchvision’s local `resnet50` ImageNet weights into your SE-ResNet50 backbone (only the matching conv/bn layers), and (2) if no DR checkpoint is available, shift the regression output by a tiny constant so the fixed thresholds don’t collapse everything to class 0 (this keeps the same model and thresholds but avoids the pathological all-zero submission). The inference loop, TTA flip, resize, and thresholding scheme remain the same, and the script still always writes a valid `submission.csv` with correct row order and columns. If a real DR checkpoint is found, none of the fallback calibration/shift is applied.'
- What this solution (achieved 0.099) has done: 'Your current score (0.01911) is far below the target (0.3225), so we should increase performance with the smallest changes that don’t alter the core model/inference/thresholding logic. The biggest remaining issue is that you’re effectively doing inference without a DR-trained checkpoint; the “local ImageNet init + output shift” fallback is still too weak and produces poor kappa. I keep your architecture, TTA-flip inference loop, resizing, and fixed thresholds identical, but (1) make checkpoint discovery smarter by also accepting smaller `.pth/.pt` files (many Kaggle DR checkpoints are <5MB) and preferring filenames that indicate APTOS/retinopathy, and (2) add robust handling for common checkpoint key prefixes (`module.`, `model.`) so a found checkpoint actually loads correctly instead of partially failing silently. If no checkpoint is found, behavior stays the same as your current fallback and still writes a valid `submission.csv`.'
- What this solution (achieved 0.01352) has done: 'Your current 0.099 is far below the 0.3225 target, so we should improve with the smallest possible, architecture-preserving change. The biggest lever is to actually use a DR-trained checkpoint if one exists anywhere in the filesystem; right now the search likely returns `None` most of the time, leaving a random head and weak predictions. I keep the exact model, resize/TTA, and thresholding semantics, but (1) broaden checkpoint discovery to include `.bin` and common names like `pytorch_model.bin`, (2) prefer checkpoints that contain `last_linear`/`avg_pool` keys that match your GeM head, and (3) handle a few more common nesting formats (`model_state_dict`, `net`, etc.) so loads don’t silently fail. If a good checkpoint is found and loads, the fallback output shift is automatically disabled (as it already is), improving kappa toward the target without changing inference logic.'
- What this solution (achieved 0.08366) has done: 'Your current score is far below the target, so the smallest likely-to-help change is to stop relying on random weights by finding and loading any compatible DR checkpoint that might already be present under the dataset directory (not just `/kaggle/input` and `/kaggle/working`). I expand the checkpoint search roots to include the detected `DATA_ROOT` and other common Kaggle-mounted locations, and I make the compatibility check a bit more permissive (accepting common head key names) so we don’t incorrectly discard a usable checkpoint. If a checkpoint is found, the existing “no-ckpt output shift” remains disabled exactly as in your current logic; if none is found, behavior is unchanged and still produces a valid `submission.csv`.'
- What this solution (achieved -0.0181) has done: 'Your current gap to target is large (0.08366 vs 0.3225), and the biggest limiter is still that you’re almost certainly not loading a DR-trained checkpoint, so predictions come from an ImageNet-ish backbone + random regression head (then thresholded), which yields weak kappa. With minimal changes and identical inference/thresholding semantics, I (1) make checkpoint discovery much more likely to find a real DR model by also scanning common Kaggle “input dataset” mount points under `/kaggle/data/input` and selecting checkpoints by inspecting their tensor shapes for a 2048→1 head (instead of filename heuristics only), and (2) make checkpoint loading more robust by handling a few more nested-key patterns so we don’t silently miss the true `state_dict`. If a compatible DR checkpoint is found, your existing “no-ckpt output shift” remains disabled exactly as before; if none is found, behavior is unchanged (still produces `submission.csv`). These are the smallest practical changes that can move score upward toward the target without changing the model architecture or inference procedure.'
- What this solution (achieved 0.0) has done: 'The crash happens because some predictions are non-finite (NaN/inf), so after thresholding you still have NaNs and `astype(int)` raises `IntCastingNaNError`. I add a minimal “prediction sanitization” step right after inference to replace non-finite values with a safe finite fallback (the median of finite predictions, or 0.0 if none), which is score-stable and unblocks submission writing. I also make the binning explicitly operate on a numeric numpy array to avoid any dtype edge-cases and ensure `diagnosis` is always a valid integer 0–4. Core model/inference logic, TTA, resizing, and thresholds remain unchanged.'
- What this solution (achieved 0.0) has done: 'Your score is far below the target, so we should improve it with the smallest change that increases the chance of producing meaningful (non-collapsed) ordinal predictions while keeping your model/inference/thresholding core intact. The main issue is that when no DR checkpoint is found, your “head-only fit” trains `last_linear` on raw labels with MSE, which tends to produce poorly calibrated outputs for QWK and can still collapse after fixed thresholds. I keep the same backbone, GeM head, TTA flip inference, fixed thresholds, and MSE head-fit loop, but make the head-fit consistent with your inference pipeline by fitting on the *regression targets implied by your thresholds* (class midpoints). I also ensure the model is in `eval()` mode during test-time inference (currently it may be left in `train()` after head-fit), which improves stability and should move QWK upward toward the target without changing architecture or post-processing.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we should improve it with the smallest change that increases the chance of meaningful ordinal predictions while keeping your exact model, resize/TTA, and fixed thresholding unchanged. The biggest practical issue is that when no DR checkpoint is found, your fallback “head-only fit” is training the regression head while the backbone is still in `train()` mode (BatchNorm updates), which can destabilize features and collapse predictions; we freeze the backbone in `eval()` and only train `last_linear`. Additionally, we ensure the head-fit always uses the exact same preprocessing as inference (already mostly true) and avoid accidental BN/Dropout effects by explicitly toggling modes before/after fitting. These are minimal, metric-aligned stability changes and should move QWK upward toward your target without changing architecture, loss, or threshold semantics.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division, absolute_import

import numpy as np
import pandas as pd



## === cell 1
"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.utils import model_zoo

__all__ = [
    "SENet",
    "senet154",
    "se_resnet50",
    "se_resnet101",
    "se_resnet152",
    "se_resnext50_32x4d",
    "se_resnext101_32x4d",
]

pretrained_settings = {
    "senet154": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "se_resnet50": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "se_resnet101": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "se_resnet152": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "se_resnext50_32x4d": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "se_resnext101_32x4d": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
}


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """

    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(
            planes * 2,
            planes * 4,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """

    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=False, stride=stride
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        groups,
        reduction,
        stride=1,
        downsample=None,
        base_width=4,
    ):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        groups,
        reduction,
        dropout_p=0.2,
        inplanes=128,
        input_3x3=True,
        downsample_kernel_size=3,
        downsample_padding=1,
        num_classes=1000,
    ):
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ("conv1", nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)),
                ("bn1", nn.BatchNorm2d(64)),
                ("relu1", nn.ReLU(inplace=True)),
                ("conv2", nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                ("bn2", nn.BatchNorm2d(64)),
                ("relu2", nn.ReLU(inplace=True)),
                ("conv3", nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
                ("bn3", nn.BatchNorm2d(inplanes)),
                ("relu3", nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                (
                    "conv1",
                    nn.Conv2d(
                        3, inplanes, kernel_size=7, stride=2, padding=3, bias=False
                    ),
                ),
                ("bn1", nn.BatchNorm2d(inplanes)),
                ("relu1", nn.ReLU(inplace=True)),
            ]
        layer0_modules.append(("pool", nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0,
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        groups,
        reduction,
        stride=1,
        downsample_kernel_size=1,
        downsample_padding=0,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=downsample_kernel_size,
                    stride=stride,
                    padding=downsample_padding,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, groups, reduction, stride, downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def initialize_pretrained_model(model, num_classes, settings):
    assert (
        num_classes == settings["num_classes"]
    ), "num_classes should be {}, but is {}".format(
        settings["num_classes"], num_classes
    )
    try:
        state = model_zoo.load_url(settings["url"])
        model.load_state_dict(state)
    except Exception as e:
        print("Warning: could not download pretrained weights:", settings["url"])
        print("Reason:", repr(e))
        print("Continuing without pretrained weights.")
    model.input_space = settings["input_space"]
    model.input_size = settings["input_size"]
    model.input_range = settings["input_range"]
    model.mean = settings["mean"]
    model.std = settings["std"]


def senet154(num_classes=1000, pretrained="imagenet"):
    model = SENet(
        SEBottleneck,
        [3, 8, 36, 3],
        groups=64,
        reduction=16,
        dropout_p=0.2,
        num_classes=num_classes,
    )
    if pretrained is not None:
        settings = pretrained_settings["senet154"][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet50(num_classes=1000, pretrained="imagenet"):
    model = SENet(
        SEResNetBottleneck,
        [3, 4, 6, 3],
        groups=1,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        num_classes=num_classes,
    )
    if pretrained is not None:
        settings = pretrained_settings["se_resnet50"][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet101(num_classes=1000, pretrained="imagenet"):
    model = SENet(
        SEResNetBottleneck,
        [3, 4, 23, 3],
        groups=1,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        num_classes=num_classes,
    )
    if pretrained is not None:
        settings = pretrained_settings["se_resnet101"][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet152(num_classes=1000, pretrained="imagenet"):
    model = SENet(
        SEResNetBottleneck,
        [3, 8, 36, 3],
        groups=1,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        num_classes=num_classes,
    )
    if pretrained is not None:
        settings = pretrained_settings["se_resnet152"][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext50_32x4d(num_classes=1000, pretrained="imagenet"):
    model = SENet(
        SEResNeXtBottleneck,
        [3, 4, 6, 3],
        groups=32,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        num_classes=num_classes,
    )
    if pretrained is not None:
        settings = pretrained_settings["se_resnext50_32x4d"][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext101_32x44d(num_classes=1000, pretrained="imagenet"):
    model = SENet(
        SEResNeXtBottleneck,
        [3, 4, 23, 3],
        groups=32,
        reduction=16,
        dropout_p=None,
        inplanes=64,
        input_3x3=False,
        downsample_kernel_size=1,
        downsample_padding=0,
        num_classes=num_classes,
    )
    if pretrained is not None:
        settings = pretrained_settings["se_resnext101_32x4d"][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model




## === cell 2
import sys

sys.path.append("/kaggle/working/")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


def get_se_resnet50_gem(pretrain):
    if pretrain == "imagenet":
        model = se_resnet50(num_classes=1000, pretrained="imagenet")
    else:
        model = se_resnet50(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(2048, 1)
    return model




## === cell 3
import os
import glob
import random

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_PATH = "/kaggle/input/seresnet50norm/model30.pth"

CANDIDATE_ROOTS = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/input",
    "/kaggle/data",
]

DATA_ROOT = None
for root in CANDIDATE_ROOTS:
    if os.path.exists(os.path.join(root, "test.csv")) and (
        os.path.exists(os.path.join(root, "test_images"))
        or os.path.exists(os.path.join(root, "test_images.zip"))
    ):
        DATA_ROOT = root
        break
if DATA_ROOT is None:
    root = "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection"
    if os.path.exists(os.path.join(root, "test.csv")):
        DATA_ROOT = root

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection dataset root with test.csv"
    )

TEST_IMAGE_PATH = os.path.join(DATA_ROOT, "test_images")
TEST_CSV_PATH = os.path.join(DATA_ROOT, "test.csv")
TRAIN_IMAGE_PATH = os.path.join(DATA_ROOT, "train_images")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")


def _strip_common_prefixes_from_state_dict(sd: dict) -> dict:
    """
    (score improvement) Strip common wrappers so a found checkpoint actually loads.
    """
    if not isinstance(sd, dict) or not sd:
        return sd

    keys = list(sd.keys())
    if all(isinstance(k, str) for k in keys):
        for pref in ("module.", "model.", "net.", "encoder."):
            if any(k.startswith(pref) for k in keys):
                sd = {
                    k[len(pref) :] if k.startswith(pref) else k: v
                    for k, v in sd.items()
                }
                keys = list(sd.keys())
    return sd


def _extract_state_dict(obj):
    """
    (score improvement) Handle more nesting patterns so we don't miss the actual weights.
    """
    if isinstance(obj, dict):
        for k in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
            "model_ema",
        ):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        for k in ("module", "student", "teacher"):
            if k in obj and isinstance(obj[k], dict):
                inner = obj[k]
                for kk in ("state_dict", "model_state_dict", "model", "net", "weights"):
                    if kk in inner and isinstance(inner[kk], dict):
                        return inner[kk]
    return obj


def _looks_compatible_with_gem_regression_head(sd: dict) -> bool:
    """
    (score improvement) More permissive head-key check.
    """
    if not isinstance(sd, dict) or not sd:
        return False
    keys = set(sd.keys())
    return any(
        k in keys
        for k in (
            "last_linear.weight",
            "last_linear.bias",
            "avg_pool.p",
            "fc.weight",
            "fc.bias",
            "linear.weight",
            "linear.bias",
        )
    )


def _looks_like_2048_to_1_head_by_shape(sd: dict) -> bool:
    """
    (score improvement) Prefer checkpoints whose tensors match our regression head
    (2048 -> 1). This avoids picking random unrelated .pth files and increases odds
    of loading a true DR-trained model.
    """
    if not isinstance(sd, dict) or not sd:
        return False
    for wkey in ("last_linear.weight", "fc.weight", "linear.weight"):
        if wkey in sd and torch.is_tensor(sd[wkey]):
            w = sd[wkey]
            if w.ndim == 2 and tuple(w.shape) == (1, 2048):
                return True
    for bkey in ("last_linear.bias", "fc.bias", "linear.bias"):
        if bkey in sd and torch.is_tensor(sd[bkey]):
            b = sd[bkey]
            if b.ndim == 1 and tuple(b.shape) == (1,):
                return True
    return False


def _find_any_checkpoint(data_root: str):
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    scan_roots = []
    for r in (
        "/kaggle/input",
        "/kaggle/working",
        "/kaggle/data",
        "/kaggle/data/input",
        data_root,
    ):
        if r and os.path.exists(r) and r not in scan_roots:
            scan_roots.append(r)

    candidates = []
    for r in scan_roots:
        candidates += glob.glob(os.path.join(r, "**", "*.pth"), recursive=True)
        candidates += glob.glob(os.path.join(r, "**", "*.pt"), recursive=True)
        candidates += glob.glob(os.path.join(r, "**", "*.bin"), recursive=True)

    min_size_bytes = 50_000  # 0.05MB
    candidates = [
        p
        for p in candidates
        if os.path.isfile(p) and os.path.getsize(p) >= min_size_bytes
    ]
    if not candidates:
        return None

    prefer_keys = [
        "aptos",
        "blindness",
        "retinopathy",
        "retina",
        "diabetic",
        "diabet",
        "dr",
        "kappa",
        "qwk",
        "seresnet",
        "se_resnet",
        "senet",
        "gem",
        "model",
        "best",
        "epoch",
        "checkpoint",
    ]
    avoid_keys = [
        "optimizer",
        "optim",
        "sched",
        "scheduler",
        "ema",
        "fold",
        "label",
    ]

    def _name_score(p):
        bn = os.path.basename(p).lower()
        return sum(k in bn for k in prefer_keys) - 2 * sum(k in bn for k in avoid_keys)

    candidates.sort(key=lambda p: (_name_score(p), os.path.getsize(p)), reverse=True)
    candidates = candidates[:80]  # keep bounded for runtime

    scored = []
    for p in candidates:
        try:
            obj = torch.load(p, map_location="cpu")
            sd = _extract_state_dict(obj)
            if isinstance(sd, dict):
                sd = _strip_common_prefixes_from_state_dict(sd)
                compat_keys = _looks_compatible_with_gem_regression_head(sd)
                compat_shape = _looks_like_2048_to_1_head_by_shape(sd)
            else:
                compat_keys, compat_shape = False, False

            scored.append(
                (
                    1 if compat_shape else 0,  # strongest signal
                    1 if compat_keys else 0,
                    _name_score(p),
                    os.path.getsize(p),
                    p,
                )
            )
        except Exception:
            continue

    if not scored:
        return None
    scored.sort(reverse=True)
    best = scored[0]

    if best[0] == 0 and best[1] == 0 and best[2] < 2:
        return None
    return best[-1]


def _safe_load_torchvision_resnet50_imagenet_into_se_resnet50(
    se_model: nn.Module,
) -> bool:
    """
    (score improvement) If no DR checkpoint is available, use local torchvision ImageNet weights
    to initialize matching backbone weights without any internet access.
    """
    try:
        tv = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        tv_sd = tv.state_dict()
        se_sd = se_model.state_dict()

        mapped = {}
        key_map = {
            "conv1.weight": "layer0.conv1.weight",
            "bn1.weight": "layer0.bn1.weight",
            "bn1.bias": "layer0.bn1.bias",
            "bn1.running_mean": "layer0.bn1.running_mean",
            "bn1.running_var": "layer0.bn1.running_var",
            "bn1.num_batches_tracked": "layer0.bn1.num_batches_tracked",
        }
        for k_tv, k_se in key_map.items():
            if (
                k_tv in tv_sd
                and k_se in se_sd
                and tv_sd[k_tv].shape == se_sd[k_se].shape
            ):
                mapped[k_se] = tv_sd[k_tv]

        for k_tv, v_tv in tv_sd.items():
            if not k_tv.startswith(("layer1.", "layer2.", "layer3.", "layer4.")):
                continue
            k_se = k_tv
            if k_se in se_sd and v_tv.shape == se_sd[k_se].shape:
                mapped[k_se] = v_tv

        if not mapped:
            return False

        se_sd.update(mapped)
        se_model.load_state_dict(se_sd, strict=False)
        return True
    except Exception as e:
        print("Warning: could not apply local torchvision ImageNet init:", repr(e))
        return False


def _label_to_regression_target_midpoint(y_int: int) -> float:
    """
    (score improvement) Keep the same fixed thresholds at inference time, but when fitting
    the regression head we train toward the midpoint of the bin that will be thresholded.
    This improves calibration toward QWK without changing the thresholding semantics.
    Bins implied by thresholds: (-inf,0.7)->0, [0.7,1.5)->1, [1.5,2.5)->2, [2.5,3.5)->3, [3.5,inf)->4
    """
    midpoints = {0: 0.35, 1: 1.10, 2: 2.00, 3: 3.00, 4: 4.00}
    return float(midpoints.get(int(y_int), 2.00))


def _set_bn_eval(m: nn.Module):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.eval()


def _fit_last_linear_on_train_features(
    model: nn.Module,
    train_csv_path: str,
    train_image_path: str,
    to_tensor_norm,
    device,
    max_samples: int = 800,
    batch_size: int = 16,
    epochs: int = 4,
    lr: float = 0.02,
):
    if not (os.path.exists(train_csv_path) and os.path.exists(train_image_path)):
        print("Head-fit skipped: train data not found at expected paths.")
        return False

    df = pd.read_csv(train_csv_path)
    if "id_code" not in df.columns or "diagnosis" not in df.columns:
        print("Head-fit skipped: train.csv schema unexpected.")
        return False

    df = df.sort_values("id_code").reset_index(drop=True)
    df = df.iloc[: min(max_samples, len(df))].copy()

    for p in model.parameters():
        p.requires_grad = False
    for p in model.last_linear.parameters():
        p.requires_grad = True

    model.eval()
    model.apply(_set_bn_eval)
    model.last_linear.train()

    opt = torch.optim.SGD(model.last_linear.parameters(), lr=lr, momentum=0.9)
    loss_fn = torch.nn.MSELoss()

    def _iter_batches():
        ids = df["id_code"].tolist()
        ys = (
            df["diagnosis"]
            .astype(int)
            .apply(_label_to_regression_target_midpoint)
            .astype(np.float32)
            .tolist()
        )
        for i in range(0, len(ids), batch_size):
            bid = ids[i : i + batch_size]
            by = ys[i : i + batch_size]
            imgs = []
            y_t = []
            for id_code, y in zip(bid, by):
                pth = os.path.join(train_image_path, f"{id_code}.png")
                if not os.path.exists(pth):
                    continue
                im = Image.open(pth).convert("RGB")
                im = im.resize((256, 256), resample=Image.BILINEAR)
                imgs.append(to_tensor_norm(im))
                y_t.append(y)
            if not imgs:
                continue
            x = torch.stack(imgs, dim=0).to(device)
            y = torch.tensor(y_t, dtype=torch.float32, device=device).view(-1, 1)
            yield x, y

    for ep in range(epochs):
        losses = []
        for x, y in _iter_batches():
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if losses:
            print(
                f"Head-fit epoch {ep+1}/{epochs}, mean MSE: {float(np.mean(losses)):.5f}"
            )
        else:
            print("Head-fit produced no batches; skipping.")
            model.eval()
            return False

    model.eval()
    return True


ckpt_path = _find_any_checkpoint(DATA_ROOT)
use_ckpt = ckpt_path is not None

model = get_se_resnet50_gem(pretrain=("imagenet" if not use_ckpt else False))
model.to(device)

if use_ckpt:
    state = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(state)
    if isinstance(state, dict):
        state = _strip_common_prefixes_from_state_dict(state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Loaded checkpoint:", ckpt_path)
    if missing or unexpected:
        print(
            "Warning: load_state_dict non-strict; missing:",
            len(missing),
            "unexpected:",
            len(unexpected),
        )
else:
    print(
        "Warning: DR checkpoint not found at hardcoded path or by search:",
        MODEL_PATH,
    )
    ok = _safe_load_torchvision_resnet50_imagenet_into_se_resnet50(model)
    print("Applied local torchvision ImageNet init to backbone:", ok)

model.eval()

test_df = pd.read_csv(TEST_CSV_PATH)
test_ids = test_df["id_code"].tolist()

to_tensor_norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if not use_ckpt:
    fitted = _fit_last_linear_on_train_features(
        model=model,
        train_csv_path=TRAIN_CSV_PATH,
        train_image_path=TRAIN_IMAGE_PATH,
        to_tensor_norm=to_tensor_norm,
        device=device,
        max_samples=800,
        batch_size=16,
        epochs=4,
        lr=0.02,
    )
    print("Head-only fit applied:", fitted)
    model.eval()

PRED_SHIFT_IF_NO_CKPT = 0.0

pred_map = {}
with torch.no_grad():
    for id_code in test_ids:
        im_path = os.path.join(TEST_IMAGE_PATH, f"{id_code}.png")
        image = Image.open(im_path).convert("RGB")
        image = image.resize((256, 256), resample=Image.BILINEAR)
        image = to_tensor_norm(image).to(device)

        output = model(image.unsqueeze(0))
        output_flip = model(torch.flip(image.unsqueeze(0), dims=(3,)))
        final_prediction = (output.item() + output_flip.item()) / 2.0
        final_prediction = final_prediction + PRED_SHIFT_IF_NO_CKPT
        pred_map[id_code] = final_prediction

raw_preds = np.array([pred_map[i] for i in test_ids], dtype=np.float32)
finite_mask = np.isfinite(raw_preds)
if not finite_mask.all():
    if finite_mask.any():
        fill_value = float(np.median(raw_preds[finite_mask]))
    else:
        fill_value = 0.0
    raw_preds = np.where(finite_mask, raw_preds, fill_value)

diagnosis = raw_preds.copy()
diagnosis[diagnosis < 0.7] = 0
diagnosis[(0.7 <= diagnosis) & (diagnosis < 1.5)] = 1
diagnosis[(1.5 <= diagnosis) & (diagnosis < 2.5)] = 2
diagnosis[(2.5 <= diagnosis) & (diagnosis < 3.5)] = 3
diagnosis[3.5 <= diagnosis] = 4
diagnosis = diagnosis.astype(np.int64)

submission = pd.DataFrame({"id_code": test_ids, "diagnosis": diagnosis})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("Diagnosis distribution:\n", submission["diagnosis"].value_counts().sort_index())
