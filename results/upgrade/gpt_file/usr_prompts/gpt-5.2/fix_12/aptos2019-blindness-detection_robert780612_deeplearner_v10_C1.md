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

0.1313481701203677

# 6. Current score

0.02639

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the submission generation robust and score-improving by (1) ensuring a valid checkpoint is actually found/loaded (your current `MODEL_PATH` likely doesn’t exist, leading to an untrained model and no meaningful score), (2) using the model’s intended ImageNet normalization so inference matches training distribution, and (3) keeping your exact regression-to-5-class thresholding logic but applying it after consistent preprocessing. These are minimal changes that preserve the core model and inference approach while making the pipeline produce a valid `submission.csv` with non-random predictions. This should move the score up toward your (low) target rather than staying near zero from an untrained/un-normalized run.'
- What this solution (achieved 0.02639) has done: 'I make the pipeline reliably load meaningful weights and run on Kaggle without internet by (1) switching the “ImageNet fallback” to use torchvision’s local pretrained ResNet50 weights (instead of the current SENet URL download), while keeping your GeM + 1-unit regression head and thresholding intact. I also make checkpoint loading slightly more tolerant by properly handling common checkpoint wrappers and ensuring `eval()` is called after loading. Finally, I keep your exact threshold-fitting and submission formatting, but add deterministic seeding and a small robustness guard for missing images so the run always finishes and writes `submission.csv`.'

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
        for i in range(1, blocks):
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
    model.load_state_dict(model_zoo.load_url(settings["url"]))
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

import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

MODEL_PATH = "/kaggle/input/seresnet50/model150.pth"
TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
TEST_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/test.csv"
TRAIN_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"
TRAIN_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/train_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_se_resnet50_gem(pretrain=False)
model.to(device)


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _sanitize_state_dict_keys(state):
    if not isinstance(state, dict):
        return state

    keys = list(state.keys())
    for prefix in ["module.", "model.", "net."]:
        if any(k.startswith(prefix) for k in keys):
            new_state = {}
            for k, v in state.items():
                if k.startswith(prefix):
                    new_state[k.replace(prefix, "", 1)] = v
                else:
                    new_state[k] = v
            state = new_state
            keys = list(state.keys())

    mapped = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("fc."):
            nk = nk.replace("fc.", "last_linear.", 1)
        if nk.startswith("classifier."):
            nk = nk.replace("classifier.", "last_linear.", 1)
        mapped[nk] = v
    return mapped


def find_checkpoint(preferred_path: str, model_ref=None):
    """
    Prefer a checkpoint that matches the intended regression head (last_linear out_features==1).
    This avoids loading unrelated .pth files and keeps predictions from being near-random.
    """
    if os.path.exists(preferred_path):
        return preferred_path

    search_roots = [
        "/kaggle/input/seresnet50",
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/input",
    ]

    candidates = []
    for root in search_roots:
        candidates.extend(glob.glob(os.path.join(root, "**", "*.pth"), recursive=True))
        candidates.extend(glob.glob(os.path.join(root, "**", "*.pt"), recursive=True))

    candidates = sorted(set(candidates))
    if not candidates:
        return None

    preferred_names = ("model150.pth", "model.pth", "best.pth", "checkpoint.pth")
    named = [p for p in candidates if os.path.basename(p) in preferred_names]
    scan_list = named + [p for p in candidates if p not in named]

    def looks_like_regression_head(pth_path: str) -> bool:
        if model_ref is None:
            return False
        try:
            raw = torch.load(pth_path, map_location="cpu")
            state = _extract_state_dict(raw)
            state = _sanitize_state_dict_keys(state)
            if not isinstance(state, dict):
                return False
            k = "last_linear.weight"
            if k in state:
                w = state[k]
                return hasattr(w, "shape") and len(w.shape) == 2 and w.shape[0] == 1
            return False
        except Exception:
            return False

    for p in scan_list:
        if looks_like_regression_head(p):
            return p

    if named:
        named = sorted(
            named, key=lambda x: (preferred_names.index(os.path.basename(x)), x)
        )
        return named[0]
    return scan_list[0]


ckpt_path = find_checkpoint(MODEL_PATH, model_ref=model)

loaded_any = False
if ckpt_path is not None:
    try:
        raw = torch.load(ckpt_path, map_location=device)
        state = _extract_state_dict(raw)
        state = _sanitize_state_dict_keys(state)

        if isinstance(state, dict):
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint (non-strict): {ckpt_path}")
            print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
            loaded_any = True
        else:
            print(
                f"WARNING: Checkpoint format not understood: {ckpt_path}. Using current weights."
            )
    except Exception as e:
        print(f"WARNING: Failed to load checkpoint {ckpt_path}: {e}")

if not loaded_any:
    try:
        import torchvision

        tv = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        tv_state = tv.state_dict()
        cur_state = model.state_dict()
        copied = 0

        for k, v in tv_state.items():
            if k in cur_state and v.shape == cur_state[k].shape:
                cur_state[k] = v
                copied += 1

        model.load_state_dict(cur_state, strict=False)
        print(
            f"Loaded torchvision ResNet50 ImageNet fallback weights for {copied} matching tensors."
        )
    except Exception as e:
        print(f"WARNING: Could not load torchvision ImageNet fallback weights: {e}")

model.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
infer_tfms = transforms.Compose(
    [
        transforms.Resize(
            (256, 256), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)


class RetinaDataset(Dataset):
    def __init__(self, id_codes, image_dir, tfms):
        self.id_codes = list(id_codes)
        self.image_dir = image_dir
        self.tfms = tfms

    def __len__(self):
        return len(self.id_codes)

    def __getitem__(self, idx):
        id_code = self.id_codes[idx]
        im_path = os.path.join(self.image_dir, f"{id_code}.png")

        try:
            image = Image.open(im_path).convert("RGB")
        except Exception:
            image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8), mode="RGB")

        image = self.tfms(image)
        return id_code, image


def predict_raw_for_ids(id_codes, image_dir, batch_size=16):
    ds = RetinaDataset(id_codes, image_dir, infer_tfms)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    preds = np.zeros((len(ds),), dtype=np.float32)
    offset = 0
    model.eval()
    with torch.no_grad():
        for _, images in dl:
            images = images.to(device, non_blocking=True)
            out = model(images).squeeze(1).detach().float().cpu().numpy()
            b = out.shape[0]
            preds[offset : offset + b] = out
            offset += b
    return preds


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_true = np.clip(y_true, 0, n_classes - 1)
    y_pred = np.clip(y_pred, 0, n_classes - 1)

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def apply_thresholds(raw_preds, thr):
    t0, t1, t2, t3 = thr
    raw_preds = np.asarray(raw_preds, dtype=np.float32)
    out = np.zeros_like(raw_preds, dtype=np.int64)
    out[raw_preds >= t0] = 1
    out[raw_preds >= t1] = 2
    out[raw_preds >= t2] = 3
    out[raw_preds >= t3] = 4
    return np.clip(out, 0, 4)


def fit_thresholds_via_nelder_mead(y_true, raw_preds, base_thr=(0.7, 1.5, 2.5, 3.5)):
    y_true = np.asarray(y_true, dtype=np.int64)
    raw_preds = np.asarray(raw_preds, dtype=np.float32)
    base = np.asarray(base_thr, dtype=np.float64)

    def _proj_monotone(thr_vec):
        thr_vec = np.asarray(thr_vec, dtype=np.float64)
        thr_sorted = np.sort(thr_vec)
        eps = 1e-4
        for i in range(1, 4):
            if thr_sorted[i] <= thr_sorted[i - 1] + eps:
                thr_sorted[i] = thr_sorted[i - 1] + eps
        return thr_sorted

    def objective(thr_vec):
        thr = _proj_monotone(thr_vec)
        pred_lbl = apply_thresholds(raw_preds, thr.astype(np.float32))
        k = quadratic_weighted_kappa(y_true, pred_lbl, n_classes=5)
        return -k  # minimize

    x0 = _proj_monotone(base)
    n = 4
    simplex = [x0]
    step = np.array([0.15, 0.15, 0.15, 0.15], dtype=np.float64)
    for i in range(n):
        xi = x0.copy()
        xi[i] += step[i]
        simplex.append(_proj_monotone(xi))
    simplex = np.array(simplex, dtype=np.float64)

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    max_iter = 60  # small, fast, deterministic

    fvals = np.array([objective(x) for x in simplex], dtype=np.float64)

    for _ in range(max_iter):
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]

        best = simplex[0]
        worst = simplex[-1]

        centroid = simplex[:-1].mean(axis=0)

        xr = _proj_monotone(centroid + alpha * (centroid - worst))
        fr = objective(xr)

        if fr < fvals[0]:
            xe = _proj_monotone(centroid + gamma * (xr - centroid))
            fe = objective(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            if fr < fvals[-1]:
                xc = _proj_monotone(centroid + rho * (xr - centroid))
            else:
                xc = _proj_monotone(centroid + rho * (worst - centroid))
            fc = objective(xc)

            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = _proj_monotone(best + sigma * (simplex[i] - best))
                    fvals[i] = objective(simplex[i])

    thr_best = _proj_monotone(simplex[np.argmin(fvals)]).astype(np.float32)
    pred_lbl = apply_thresholds(raw_preds, thr_best)
    best_k = quadratic_weighted_kappa(y_true, pred_lbl, n_classes=5)
    return tuple(thr_best.tolist()), float(best_k)


train_df = pd.read_csv(TRAIN_CSV_PATH)
train_ids = train_df["id_code"].astype(str).tolist()
train_y = train_df["diagnosis"].astype(np.int64).values

rng = np.random.RandomState(42)
val_size = min(512, len(train_ids) // 5)

val_idx = []
for c in range(5):
    cls_idx = np.where(train_y == c)[0]
    rng.shuffle(cls_idx)
    take = int(round(val_size * (len(cls_idx) / len(train_y))))
    take = min(take, len(cls_idx))
    val_idx.extend(cls_idx[:take].tolist())

val_idx = list(dict.fromkeys(val_idx))  # de-dup preserve order
if len(val_idx) < val_size:
    remaining = np.setdiff1d(
        np.arange(len(train_ids)), np.array(val_idx, dtype=np.int64)
    )
    rng.shuffle(remaining)
    need = val_size - len(val_idx)
    val_idx.extend(remaining[:need].tolist())
elif len(val_idx) > val_size:
    rng.shuffle(val_idx)
    val_idx = val_idx[:val_size]

val_ids = [train_ids[i] for i in val_idx]
val_y = train_y[np.array(val_idx, dtype=np.int64)]

val_raw = predict_raw_for_ids(val_ids, TRAIN_IMAGE_PATH, batch_size=16)
best_thr, best_k = fit_thresholds_via_nelder_mead(
    val_y, val_raw, base_thr=(0.7, 1.5, 2.5, 3.5)
)
print(f"Calibrated thresholds (val QWK={best_k:.5f}): {best_thr}")

test_df = pd.read_csv(TEST_CSV_PATH)
id_codes = test_df["id_code"].astype(str).tolist()

test_raw = predict_raw_for_ids(id_codes, TEST_IMAGE_PATH, batch_size=16)

raw_out = pd.DataFrame({"id_code": id_codes, "diagnosis": test_raw.astype(np.float32)})
raw_out.to_csv("submission_raw_value.csv", index=False)

test_labels = apply_thresholds(test_raw, np.asarray(best_thr, dtype=np.float32))

submission = pd.DataFrame(
    {"id_code": id_codes, "diagnosis": test_labels.astype(np.int64)}
)

submission = test_df[["id_code"]].merge(submission, on="id_code", how="left")
submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(np.int64).clip(0, 4)

submission.to_csv("submission.csv", index=False)

submission.head()
