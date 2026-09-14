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

-0.0823478533678401

# 6. Current score

-0.22569

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.2519) has done: 'I make the submission generation robust and metric-aligned while keeping your model and inference loop intact. First, I fix the cell numbering and harden path resolution so `test.csv` and `test_images/` are always found in this environment, ensuring a valid `submission.csv` is produced. Second, because QWK is computed on discrete classes 0–4, I add a minimal, data-driven calibration step that maps your continuous regression outputs to 5 ordered classes using the training label distribution (quantile-based binning), which typically improves agreement vs. fixed hand thresholds. If any image is missing/corrupt, the code safely fall back to a neutral prediction rather than crashing, so you always get a complete submission.'
- What this solution (achieved -0.24861) has done: 'You’re currently far below the target (−0.2519 vs −0.08235, higher-is-better), so we should make a small, metric-aligned improvement without changing your model or inference loop. The biggest issue is the quantile thresholds: they’re being taken from **test** predictions, which can easily scramble ordinal class boundaries and hurt QWK; we instead fit thresholds on **train** predictions using the same model, then apply them to test scores. This keeps your “regress then discretize” core logic identical, but makes discretization consistent with the label distribution (a standard QWK calibration step). We also keep the robust I/O and still write `submission.csv` exactly in the required format.'
- What this solution (achieved 0.0) has done: 'You currently don’t have a measurable Kaggle score because the run depends on a missing external `model.bin`; with only ImageNet weights, the regressor outputs are poorly calibrated and can collapse QWK. I keep your “ResNet50 regression → quantile thresholds → 0–4 labels” core logic, but add a minimal, metric-aligned calibration that directly optimizes the 4 thresholds on a small validation split of the training set (no model/loop changes). This replaces heuristic quantiles with data-driven thresholds that typically move QWK substantially upward from strongly negative values while remaining stable and fast (<600s). I also add a robust fallback to the original quantile thresholds if threshold optimization can’t run (e.g., missing train images), and keep submission formatting unchanged.'
- What this solution (achieved -0.00785) has done: 'Your current run reports `current_score=None`, so the first priority is to make sure this script reliably produces a valid `submission.csv` from the correct dataset location (so you can actually get a Kaggle score). Then, without changing your model/loop, I make a minimal metric-aligned improvement by preventing the “fit thresholds on test predictions” fallback (which can strongly hurt QWK); instead we always derive initial thresholds from *train* predictions when available, and otherwise from the *train label distribution* mapped onto the test-score scale (more stable than using test quantiles). Finally, I harden the image path resolution for both flat and nested directory layouts and ensure the output `diagnosis` is integer 0–4 with the exact required columns/order.'
- What this solution (achieved -0.10565) has done: 'Your current score (−0.00785) is higher (better) than the target (−0.08235), so we should *slightly reduce* performance toward the target band with the smallest, safest change. The most controllable, metric-relevant lever here is the threshold calibration step: we keep your model and inference identical, but stop the validation-based threshold optimization (which tends to boost QWK) and instead use only the stable quantile thresholds from train-score distribution. To nudge further toward the target without risking invalid submissions, we also apply a tiny, constant shift to the regression scores before thresholding (keeps semantics, just changes calibration) while keeping output formatting unchanged. All I/O paths and the submission schema remain the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved -0.22569) has done: 'Your current score is unknown because you haven’t produced a scored submission from this exact run yet, so the safest way to move toward the target (−0.0823) is to keep your model/inference identical and only make small, metric-relevant calibration changes that affect QWK. I (1) harden determinism and ensure the exact same image paths and CSV alignment always produce a valid `submission.csv`, and (2) slightly reduce the “performance-boosting” post-processing by disabling the train-histogram nudging (which can push QWK upward unpredictably) while keeping your thresholding logic intact. This should move results in a controlled way toward the target band without changing architecture, loss, or training (there is none here). The script still writes both `submission_raw_value.csv` and the required `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import math
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils import model_zoo

from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

NOTE: Core model definition is preserved; only runtime/IO and metric-aligned thresholding are adjusted elsewhere.
"""

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
            "input_size": [0, 0, 0],
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
    expansion = 4

    def __init__(
        self,
        self_inplanes,
        planes,
        groups,
        reduction,
        stride=1,
        downsample=None,
        base_width=4,
    ):
        super(SEResNeXtBottleneck, self).__init__()
        inplanes = self_inplanes
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
    model.load_state_dict(model_zoo.load_url(settings["url"]))
    model.input_space = settings["input_space"]
    model.input_size = settings["input_size"]
    model.input_range = settings["input_range"]
    model.mean = settings["mean"]
    model.std = settings["std"]


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




## === cell 2
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
from torchvision.models import resnet50, ResNet50_Weights


def resolve_base_dir():
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "test.csv")) and os.path.exists(
            os.path.join(c, "test_images")
        ):
            return c

    for c in candidates:
        nested = os.path.join(c, "aptos2019-blindness-detection")
        if os.path.exists(os.path.join(nested, "test.csv")) and os.path.exists(
            os.path.join(nested, "test_images")
        ):
            return nested

    return "/kaggle/input/aptos2019-blindness-detection"


def resolve_image_dir(base_dir: str, split: str) -> str:
    assert split in ("train", "test")
    d1 = os.path.join(base_dir, f"{split}_images")
    d2 = os.path.join(base_dir, f"{split}_images", f"{split}_images")
    if os.path.isdir(d1):
        return d1
    if os.path.isdir(d2):
        return d2
    return d1


BASE_DIR = resolve_base_dir()

MODEL_PATH = (
    "/kaggle/input/se-resnet50/model.bin"  # kept for compatibility; may not exist
)
TEST_IMAGE_PATH = resolve_image_dir(BASE_DIR, "test")
TRAIN_IMAGE_PATH = resolve_image_dir(BASE_DIR, "train")
TEST_CSV_PATH = os.path.join(BASE_DIR, "test.csv")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "train.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ResNet50_Weights.IMAGENET1K_V2
tv_model = resnet50(weights=weights)
tv_model.fc = nn.Linear(tv_model.fc.in_features, 1)
model = tv_model.to(device)

if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location=device)
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)

model.eval()

tfm = transforms.Compose(
    [
        transforms.Resize(
            (256, 256), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_df = pd.read_csv(TEST_CSV_PATH)
train_df = pd.read_csv(TRAIN_CSV_PATH) if os.path.exists(TRAIN_CSV_PATH) else None


def compute_quantile_targets_from_labels(train_labels: pd.Series):
    counts = (
        train_labels.value_counts()
        .sort_index()
        .reindex(range(5), fill_value=0)
        .astype(np.int64)
        .values
    )
    n = int(counts.sum())
    if n <= 0:
        return [0.70, 0.85, 0.95, 0.99]

    cum = np.cumsum(counts)
    qs = []
    for k in range(4):
        left = cum[k]
        right = cum[k + 1]
        qs.append((left + right) / (2.0 * n))

    eps = 1.0 / (2.0 * n)
    qs = np.clip(np.asarray(qs, dtype=np.float64), eps, 1.0 - eps)

    for i in range(1, len(qs)):
        if qs[i] <= qs[i - 1]:
            qs[i] = min(1.0 - eps, qs[i - 1] + eps)

    return [float(x) for x in qs]


if train_df is not None:
    quantile_targets = compute_quantile_targets_from_labels(train_df["diagnosis"])
else:
    quantile_targets = [0.70, 0.85, 0.95, 0.99]


def predict_scores(df: pd.DataFrame, image_dir: str):
    ids = df["id_code"].tolist()
    scores = []
    with torch.inference_mode():
        for img_id in ids:
            im_path = os.path.join(image_dir, f"{img_id}.png")
            try:
                image = Image.open(im_path).convert("RGB")
                image = tfm(image).to(device)
                output = model(image.unsqueeze(0))
                score = float(output.item())
            except Exception:
                score = 0.0
            scores.append(score)
    return np.asarray(scores, dtype=np.float64)


def apply_thresholds(scores, thresholds):
    thr = np.asarray(thresholds, dtype=np.float64)
    bins = [-np.inf] + thr.tolist() + [np.inf]
    labels = np.digitize(scores, bins=bins) - 1
    return np.clip(labels, 0, 4).astype(np.int64)


train_scores = None
if train_df is not None and os.path.exists(TRAIN_IMAGE_PATH):
    train_scores = predict_scores(train_df, TRAIN_IMAGE_PATH)

test_scores = predict_scores(test_df, TEST_IMAGE_PATH)

submission_raw = pd.DataFrame(
    {"id_code": test_df["id_code"].values, "diagnosis": test_scores}
)
submission_raw.to_csv("submission_raw_value.csv", index=False)

DO_THRESHOLD_OPTIM = False

if train_scores is not None and np.nanstd(train_scores) >= 1e-8:
    init_thresholds = np.quantile(train_scores, quantile_targets).tolist()
else:
    if np.nanstd(test_scores) >= 1e-8:
        mu = float(np.nanmean(test_scores))
        sd = float(np.nanstd(test_scores))
        try:
            from statistics import NormalDist

            nd = NormalDist()
            z = np.array([nd.inv_cdf(q) for q in quantile_targets], dtype=np.float64)
            init_thresholds = (mu + sd * z).tolist()
        except Exception:
            init_thresholds = (
                mu + sd * np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float64)
            ).tolist()
    else:
        init_thresholds = [0.7, 1.5, 2.5, 3.5]

thr = np.asarray(init_thresholds, dtype=np.float64)
for i in range(1, len(thr)):
    if not np.isfinite(thr[i]):
        thr[i] = thr[i - 1]
    if thr[i] <= thr[i - 1]:
        thr[i] = thr[i - 1] + 1e-6
thresholds = thr.tolist()

val_kappa = None  # kept for compatibility with prints below


def qwk(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    denom = float((n_classes - 1) ** 2)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / denom

    num = (W * O).sum()
    den = (W * E).sum()
    if den <= 0:
        return 0.0
    return 1.0 - (num / den)


if DO_THRESHOLD_OPTIM and (train_df is not None) and (train_scores is not None):
    y = train_df["diagnosis"].values.astype(np.int64)

    rng = np.random.RandomState(42)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_val = max(256, int(0.2 * len(idx)))
    val_idx = idx[:n_val]

    s_val = train_scores[val_idx]
    y_val = y[val_idx]

    t = np.array(thresholds, dtype=np.float64)

    def ensure_increasing(x):
        x = np.asarray(x, dtype=np.float64).copy()
        for i in range(1, len(x)):
            if not np.isfinite(x[i]):
                x[i] = x[i - 1]
            if x[i] <= x[i - 1]:
                x[i] = x[i - 1] + 1e-6
        return x

    t = ensure_increasing(t)

    base_scale = float(np.nanstd(s_val))
    if not np.isfinite(base_scale) or base_scale < 1e-8:
        base_scale = 1.0

    step_sizes = [0.50, 0.20, 0.08]  # relative to score std
    grid = [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]

    best_k = qwk(y_val, apply_thresholds(s_val, t.tolist()))
    best_t = t.copy()

    for step_rel in step_sizes:
        step = step_rel * base_scale
        improved = True
        for _ in range(3):
            if not improved:
                break
            improved = False
            for j in range(4):
                candidates = []
                for g in grid:
                    cand = best_t.copy()
                    cand[j] = cand[j] + g * step
                    cand = ensure_increasing(cand)
                    candidates.append(cand)
                local_best_k = best_k
                local_best_t = best_t
                for cand in candidates:
                    pred = apply_thresholds(s_val, cand.tolist())
                    k = qwk(y_val, pred)
                    if k > local_best_k + 1e-12:
                        local_best_k = k
                        local_best_t = cand
                if local_best_k > best_k + 1e-12:
                    best_k = local_best_k
                    best_t = local_best_t
                    improved = True

    thresholds = best_t.tolist()
    val_kappa = float(best_k)

SCORE_SHIFT = 0.08
test_scores_for_labels = test_scores + SCORE_SHIFT
labels = apply_thresholds(test_scores_for_labels, thresholds)


def nudge_to_match_train_hist(
    pred_labels: np.ndarray, train_labels: pd.Series
) -> np.ndarray:
    pred_labels = np.asarray(pred_labels, dtype=np.int64).copy()
    if train_labels is None or len(train_labels) == 0:
        return pred_labels

    n = len(pred_labels)
    if n == 0:
        return pred_labels

    target_counts = (
        train_labels.value_counts()
        .sort_index()
        .reindex(range(5), fill_value=0)
        .astype(np.int64)
        .values
    )
    target = np.floor(target_counts * (n / max(1, int(target_counts.sum())))).astype(
        np.int64
    )
    diff = int(n - target.sum())
    if diff != 0:
        j = int(np.argmax(target))
        target[j] = max(0, target[j] + diff)

    cur = np.bincount(pred_labels, minlength=5).astype(np.int64)
    if np.all(cur == target):
        return pred_labels

    order = np.argsort(test_scores_for_labels)  # low score -> low severity
    ranks = np.empty_like(order)
    ranks[order] = np.arange(n)

    def move(src, dst, k):
        if k <= 0:
            return 0
        idx_src = np.where(pred_labels == src)[0]
        if len(idx_src) == 0:
            return 0
        if dst > src:
            pick = idx_src[np.argsort(ranks[idx_src])[::-1]]
        else:
            pick = idx_src[np.argsort(ranks[idx_src])]
        take = pick[: min(k, len(pick))]
        pred_labels[take] = dst
        return len(take)

    for _ in range(20):  # hard cap for safety
        cur = np.bincount(pred_labels, minlength=5).astype(np.int64)
        delta = cur - target
        if np.all(delta == 0):
            break

        progressed = False
        for c in range(5):
            if delta[c] > 0:
                candidates = []
                if c - 1 >= 0 and delta[c - 1] < 0:
                    candidates.append(c - 1)
                if c + 1 <= 4 and delta[c + 1] < 0:
                    candidates.append(c + 1)
                if not candidates:
                    continue
                dst = min(candidates, key=lambda j: delta[j])  # most negative
                k = min(delta[c], -delta[dst])
                moved = move(c, dst, k)
                if moved > 0:
                    progressed = True

        if not progressed:
            break

    return np.clip(pred_labels, 0, 4).astype(np.int64)


DO_HIST_NUDGE = False
if DO_HIST_NUDGE:
    labels = nudge_to_match_train_hist(
        labels, train_df["diagnosis"] if train_df is not None else None
    )

FLATTEN_FRACTION = 0.0

submission = pd.DataFrame(
    {"id_code": test_df["id_code"].values, "diagnosis": labels.astype(np.int64)}
)
submission = submission[["id_code", "diagnosis"]]
submission.to_csv("submission.csv", index=False)

print("BASE_DIR:", BASE_DIR)
print("TRAIN_IMAGE_PATH:", TRAIN_IMAGE_PATH, "exists:", os.path.isdir(TRAIN_IMAGE_PATH))
print("TEST_IMAGE_PATH:", TEST_IMAGE_PATH, "exists:", os.path.isdir(TEST_IMAGE_PATH))
print("Quantile targets:", quantile_targets)
print("Using thresholds:", thresholds)
print("Threshold-optim enabled:", DO_THRESHOLD_OPTIM, "val_kappa:", val_kappa)
print("Applied SCORE_SHIFT:", SCORE_SHIFT)
print("Histogram nudge enabled:", DO_HIST_NUDGE)
print("Flatten fraction:", FLATTEN_FRACTION)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")
if train_scores is None:
    print(
        "Note: train-based threshold fitting skipped (train_images not found); used label-aligned scale fallback thresholds."
    )
else:
    print("Train scores std:", float(np.nanstd(train_scores)))
print("Test scores std:", float(np.nanstd(test_scores)))
print("Test scores std (shifted):", float(np.nanstd(test_scores_for_labels)))
print("Pred label counts:", np.bincount(labels, minlength=5).tolist())
if train_df is not None:
    tr_counts = (
        train_df["diagnosis"]
        .value_counts()
        .sort_index()
        .reindex(range(5), fill_value=0)
        .astype(int)
        .values
    )
    print("Train label counts:", tr_counts.tolist())
