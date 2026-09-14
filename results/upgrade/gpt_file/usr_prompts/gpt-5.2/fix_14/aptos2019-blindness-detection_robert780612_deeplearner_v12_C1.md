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

0.8982294236152204

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'You’re failing because the SENet implementation tries to download ImageNet weights from an external URL, which is blocked in Kaggle (SSL/cert error). I make pretrained loading “offline-safe” by catching download errors and continuing with random init, while still using your provided local checkpoint when available (score-critical). I also fix the `pretrain` flag bug (passing `False` instead of `None`) to avoid accidental weight-download attempts, and add a safe fallback to the sample submission for CSV alignment so `submission.csv` is always valid and in the expected order. No model architecture or thresholding logic is changed.'
- What this solution (achieved 0.0) has done: 'I fix the root cause of the 0.0 score by ensuring the model actually loads the provided local checkpoint (your current `MODEL_PATH` points to a dataset that isn’t present, so you likely ran with random weights). I make checkpoint discovery prioritize any `.pth` under `/kaggle/input` that matches the expected architecture (by testing load compatibility), while still never attempting an internet download. I also keep your exact inference and thresholding logic, but add a strict submission integrity check (row count/order matches `sample_submission.csv`) so the file is always accepted and aligned. These are minimal, score-relevant changes aimed at moving from 0.0 toward your ~0.898 target by restoring the intended pretrained model behavior.'
- What this solution (achieved 0.0) has done: 'I keep your SENet+GeM model and thresholding logic intact, but make the checkpoint loading reliable so you don’t accidentally run with random weights (which is the most likely reason you currently have no valid score / near-0.0 behavior). Specifically, I (1) force “offline-safe” behavior by never attempting URL downloads, and (2) improve checkpoint discovery by prioritizing any `.pth` already present under `/kaggle/input` and selecting the first one that actually changes the model parameter fingerprint after loading. Finally, I keep your submission alignment with `sample_submission.csv` but add one strict check that every test image exists (to avoid silent failures) and ensure `submission.csv` is always written.'
- What this solution (achieved 0.0) has done: 'I make the checkpoint loading deterministic and much more likely to actually load your intended weights (the biggest reason for “no score / 0.0”-like behavior), by prioritizing a direct existing `MODEL_PATH` and then selecting the *best-matching* `.pth/.pt` based on missing/unexpected key ratios without accidentally leaving the model in a partially-loaded state. I also ensure inference is consistent with the model’s expected input preprocessing by using the model’s own `mean/std` if present (falling back to ImageNet), which is a minimal, metric-aligned tweak that can materially move QWK toward your target. Finally, I keep your exact model architecture and thresholding logic intact, but add a strict check that predictions cover all `id_code`s and preserve ordering so the submission is always valid.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division, absolute_import

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from collections import OrderedDict
import math

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
        model.input_space = settings["input_space"]
        model.input_size = settings["input_size"]
        model.input_range = settings["input_range"]
        model.mean = settings["mean"]
        model.std = settings["std"]
    except Exception as e:
        print(
            f"[WARN] Could not load pretrained weights from URL (offline/SSL). "
            f"Continuing with randomly initialized backbone. Error: {type(e).__name__}: {e}"
        )
        model.input_space = settings.get("input_space", "RGB")
        model.input_size = settings.get("input_size", [3, 224, 224])
        model.input_range = settings.get("input_range", [0, 1])
        model.mean = settings.get("mean", [0.485, 0.456, 0.406])
        model.std = settings.get("std", [0.229, 0.224, 0.225])


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


def se_resnext101_32x4d(num_classes=1000, pretrained="imagenet"):
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
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_PATH = "/kaggle/input/aptos2019-blindness-detection/model30.pth"


def _find_existing_file(candidates):
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def _find_existing_dir(candidates):
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    if all(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _fingerprint_tensor(t: torch.Tensor, n: int = 256) -> float:
    t = t.detach().float().flatten()
    if t.numel() == 0:
        return 0.0
    t = t[: min(n, t.numel())].cpu()
    return float(t.abs().mean().item())


def _iter_ckpt_files(root_dirs, max_files=500):
    exts = (".pth", ".pt")
    seen = 0
    for root in root_dirs:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                lfn = fn.lower()
                if lfn.endswith(exts):
                    yield os.path.join(dirpath, fn)
                    seen += 1
                    if seen >= max_files:
                        return


def _compat_score_only(model, state_dict) -> float:
    """
    Change: compute a compatibility score WITHOUT mutating the final model.
    This reduces the chance of accidentally leaving a partially-loaded model,
    and improves odds we load the right checkpoint -> higher QWK toward target.
    """
    try:
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing, unexpected = incompatible
    except Exception:
        return -1.0

    overlap = len(set(model.state_dict().keys()).intersection(set(state_dict.keys())))
    if overlap == 0:
        return -1.0

    total_keys = max(1, len(model.state_dict()))
    missing_ratio = len(missing) / total_keys
    unexpected_ratio = len(unexpected) / max(1, len(state_dict))

    return 1.0 - 0.8 * missing_ratio - 0.2 * unexpected_ratio


def _find_best_compatible_checkpoint(device, root_dirs):
    best_path = None
    best_score = -1.0

    for p in _iter_ckpt_files(root_dirs):
        try:
            ckpt = torch.load(p, map_location="cpu")
            sd = _strip_module_prefix(_extract_state_dict(ckpt))
            if not isinstance(sd, dict) or len(sd) == 0:
                continue

            test_model = get_se_resnet50_gem(pretrain=None).to(device)
            test_model.eval()
            s = _compat_score_only(test_model, sd)
            if s > best_score:
                best_score = s
                best_path = p
        except Exception:
            continue

    return best_path, best_score


TEST_CSV_PATH = _find_existing_file(
    [
        "/kaggle/input/aptos2019-blindness-detection/test.csv",
        "/kaggle/data/aptos2019-blindness-detection/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
    ]
)
TEST_IMAGE_PATH = _find_existing_dir(
    [
        "/kaggle/input/aptos2019-blindness-detection/test_images",
        "/kaggle/data/aptos2019-blindness-detection/test_images",
        "/kaggle/input/test_images",
        "/kaggle/data/test_images",
    ]
)
SAMPLE_SUB_PATH = _find_existing_file(
    [
        "/kaggle/input/aptos2019-blindness-detection/sample_submission.csv",
        "/kaggle/data/aptos2019-blindness-detection/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

if TEST_CSV_PATH is None:
    raise FileNotFoundError("Could not locate test.csv in known Kaggle paths.")
if TEST_IMAGE_PATH is None:
    raise FileNotFoundError(
        "Could not locate test_images directory in known Kaggle paths."
    )
if SAMPLE_SUB_PATH is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in known Kaggle paths."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

use_external_weights = False
ckpt_path = None

if os.path.isfile(MODEL_PATH):
    ckpt_path = MODEL_PATH
    use_external_weights = True
else:
    compat_path, compat_score = _find_best_compatible_checkpoint(
        device,
        [
            "/kaggle/input/aptos2019-blindness-detection",
            "/kaggle/input",
            "/kaggle/data",
            "/kaggle/working",
        ],
    )
    if compat_path is not None and os.path.isfile(compat_path) and compat_score > 0:
        ckpt_path = compat_path
        use_external_weights = True
        print(
            f"[INFO] Auto-selected best compatible checkpoint score={compat_score:.4f}: {ckpt_path}"
        )

model = get_se_resnet50_gem(pretrain=None)
model.to(device)

if use_external_weights:
    print(f"[INFO] Using local checkpoint: {ckpt_path}")
    before_fp = _fingerprint_tensor(next(model.parameters()))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _strip_module_prefix(_extract_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    after_fp = _fingerprint_tensor(next(model.parameters()))
    print(
        f"[INFO] Loaded checkpoint with missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if abs(after_fp - before_fp) < 1e-8:
        print(
            "[WARN] Model parameters fingerprint did not change after loading checkpoint. "
            "Predictions may be near-random."
        )
else:
    print(
        "[WARN] No local .pth/.pt checkpoint found; running with random init (offline-safe)."
    )

model.eval()

test_df = pd.read_csv(TEST_CSV_PATH)

missing_images = []
for id_code in test_df["id_code"].tolist():
    im_path = os.path.join(TEST_IMAGE_PATH, f"{id_code}.png")
    if not os.path.isfile(im_path):
        missing_images.append(im_path)
if len(missing_images) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing_images)} test images. Example: {missing_images[0]}"
    )

mean = getattr(model, "mean", [0.485, 0.456, 0.406])
std = getattr(model, "std", [0.229, 0.224, 0.225])

to_tensor_norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

predictions = []
with torch.no_grad():
    for id_code in test_df["id_code"].tolist():
        im_path = os.path.join(TEST_IMAGE_PATH, f"{id_code}.png")
        image = Image.open(im_path).convert("RGB")
        image = image.resize((256, 256), resample=Image.BILINEAR)
        image = to_tensor_norm(image).to(device)
        output = model(image.unsqueeze(0))
        predictions.append((id_code, float(output.item())))

submission = pd.DataFrame(predictions, columns=["id_code", "diagnosis"])
submission.to_csv("submission_raw_value.csv", index=False)

submission.loc[submission.diagnosis < 0.7, "diagnosis"] = 0
submission.loc[
    (0.7 <= submission.diagnosis) & (submission.diagnosis < 1.5), "diagnosis"
] = 1
submission.loc[
    (1.5 <= submission.diagnosis) & (submission.diagnosis < 2.5), "diagnosis"
] = 2
submission.loc[
    (2.5 <= submission.diagnosis) & (submission.diagnosis < 3.5), "diagnosis"
] = 3
submission.loc[3.5 <= submission.diagnosis, "diagnosis"] = 4

submission["diagnosis"] = submission["diagnosis"].astype(int)
submission = submission[["id_code", "diagnosis"]]

sample = pd.read_csv(SAMPLE_SUB_PATH)

if submission["id_code"].duplicated().any():
    raise ValueError(
        "Duplicate id_code detected in predictions; cannot form valid submission."
    )
if set(submission["id_code"]) != set(sample["id_code"]):
    print(
        "[WARN] Predicted id_code set does not exactly match sample_submission id_code set."
    )

submission = sample[["id_code"]].merge(submission, on="id_code", how="left")
submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int)

assert list(submission.columns) == ["id_code", "diagnosis"]
assert submission.shape[0] == sample.shape[0]
assert submission["id_code"].tolist() == sample["id_code"].tolist()

submission.to_csv("submission.csv", index=False)
print(submission.head())
print(f"[INFO] Wrote submission.csv with shape={submission.shape}")
