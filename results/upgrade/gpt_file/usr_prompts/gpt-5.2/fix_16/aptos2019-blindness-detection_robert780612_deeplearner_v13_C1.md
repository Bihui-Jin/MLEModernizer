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

0.8884778343766443

# 6. Current score

None

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the SSL/runtime failure by preventing any network download for ImageNet weights (Kaggle has no guaranteed outbound SSL), and instead require a local checkpoint if available; if none is found, the code still run with randomly initialized weights and produce a valid `submission.csv`. I also make data/model paths robust to the provided directory layout by auto-detecting the correct `test.csv` and `test_images` directory. To keep inference stable and fast, I switch to batched DataLoader inference (same preprocessing and same thresholds), and ensure the submission has the exact required columns and `.csv` suffix. These changes are execution/stability fixes; they preserve the core model architecture and post-processing logic.'
- What this solution (achieved 0.0) has done: 'I fix the immediate runtime failure by removing the hard dependency on an external checkpoint: the script try to load a local `.pth/.pt` if present, otherwise it continue with randomly initialized weights and still produce a valid `submission.csv`. I also prevent any network download by ensuring the model is always created with `pretrained=None` (unchanged from your intent), keeping the core model architecture and the exact same preprocessing + thresholding logic. Finally, I make the test image loading a bit more robust by falling back across the known directory candidates if a file isn’t found in the chosen folder, so the run doesn’t crash due to path layout differences.'
- What this solution (achieved 0.74858) has done: 'Your current pipeline never trains on the provided `train.csv` labels, so if a local checkpoint isn’t found you effectively submit random predictions (score near 0). To move the score toward the target while preserving your model architecture and inference/thresholding semantics, I add a minimal fine-tuning step on the training images using the same regression head/loss style (MSE on the scalar output vs. diagnosis). I keep your threshold optimization on a held-out validation split, but ensure the split is reproducible and that training uses only the training split (no leakage). The script still loads a checkpoint if present; otherwise it performs a short, deterministic fine-tune and then produces a valid `submission.csv`.'
- What this solution (achieved 0.74427) has done: 'To move your score upward from “no score / likely invalid or very low,” the smallest meaningful improvement is to ensure the model always starts from reasonable ImageNet weights without any network access. I keep your exact model, loss (MSE regression), threshold optimization, and training/inference loops, but replace the fragile “try torchvision weights (may download)” logic with an explicit offline weight load from torchvision’s local weight file if present. I also add a small safety fix to always use global average pooling compatible with your 256×256 resize (otherwise the fixed 7×7 AvgPool can break), which preserves architecture intent while preventing shape-dependent failures. The script still produces `submission.csv` with the required columns.'

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

import os
import glob
import random
import time

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


def _offline_load_cadene_se_resnet50_imagenet_state_dict():
    candidates = []
    for root in [
        "/kaggle/input",
        "/kaggle/data",
        "/kaggle/working",
        os.path.expanduser("~"),
    ]:
        if os.path.isdir(root):
            candidates += glob.glob(
                os.path.join(root, "**", "se_resnet50*.pth"), recursive=True
            )
            candidates += glob.glob(
                os.path.join(root, "**", "se_resnet50*.pt"), recursive=True
            )
            candidates += glob.glob(
                os.path.join(root, "**", "*se_resnet50*ce0d4300*.pth"), recursive=True
            )
            candidates += glob.glob(
                os.path.join(root, "**", "*se_resnet50*ce0d4300*.pt"), recursive=True
            )
    candidates = sorted(
        set(candidates),
        key=lambda p: (0 if "ce0d4300" in os.path.basename(p) else 1, len(p), p),
    )
    for p in candidates:
        try:
            sd = torch.load(p, map_location="cpu")
            if (
                isinstance(sd, dict)
                and "state_dict" in sd
                and isinstance(sd["state_dict"], dict)
            ):
                sd = sd["state_dict"]
            if not isinstance(sd, dict) or len(sd) == 0:
                continue
            if any(isinstance(k, str) and k.startswith("layer0.") for k in sd.keys()):
                return sd
        except Exception:
            continue
    return None


def get_se_resnet50_gem(pretrain):
    if pretrain in ("imagenet", True):
        model = se_resnet50(num_classes=1000, pretrained=None)
        sd = _offline_load_cadene_se_resnet50_imagenet_state_dict()
        if sd is not None:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(
                f"Loaded offline Cadene se_resnet50 ImageNet weights (non-strict). missing={len(missing)} unexpected={len(unexpected)}"
            )
        else:
            print(
                "Offline Cadene se_resnet50 ImageNet weights not found; using random init."
            )
    else:
        model = se_resnet50(num_classes=1000, pretrained=None)

    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(2048, 1)
    return model


MODEL_PATH = "/kaggle/input/seresnet50-2/model90.pth"

TEST_CSV_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection/test.csv",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/test.csv",
    "/kaggle/data/aptos2019-blindness-detection/test.csv",
    "/kaggle/data/test.csv",
]
TEST_CSV_PATH = next((p for p in TEST_CSV_CANDIDATES if os.path.isfile(p)), None)

TEST_IMAGE_PATH_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection/test_images",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/test_images",
    "/kaggle/data/aptos2019-blindness-detection/test_images",
    "/kaggle/data/test_images",
]
TEST_IMAGE_PATH = next(
    (p for p in TEST_IMAGE_PATH_CANDIDATES if os.path.isdir(p)), None
)

TRAIN_CSV_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection/train.csv",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/train.csv",
    "/kaggle/data/aptos2019-blindness-detection/train.csv",
    "/kaggle/data/train.csv",
]
TRAIN_CSV_PATH = next((p for p in TRAIN_CSV_CANDIDATES if os.path.isfile(p)), None)

TRAIN_IMAGE_PATH_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection/train_images",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/train_images",
    "/kaggle/data/aptos2019-blindness-detection/train_images",
    "/kaggle/data/train_images",
]
TRAIN_IMAGE_PATH = next(
    (p for p in TRAIN_IMAGE_PATH_CANDIDATES if os.path.isdir(p)), None
)

if TEST_CSV_PATH is None or TRAIN_CSV_PATH is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv under expected Kaggle paths."
    )
if TEST_IMAGE_PATH is None or TRAIN_IMAGE_PATH is None:
    raise FileNotFoundError(
        "Could not locate train_images/test_images directories under expected Kaggle paths."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _find_checkpoint_path(preferred_path: str) -> str:
    if preferred_path and os.path.isfile(preferred_path):
        return preferred_path

    preferred_candidates = [
        "/kaggle/input/seresnet50-2/model90.pth",
        "/kaggle/input/seresnet50-2/model.pth",
        "/kaggle/input/seresnet50-2/best.pth",
        "/kaggle/input/seresnet50-2/checkpoint.pth",
        "/kaggle/input/seresnet50-2/model90.pt",
        "/kaggle/input/seresnet50-2/model.pt",
    ]
    for p in preferred_candidates:
        if os.path.isfile(p):
            return p

    roots = ["/kaggle/input", "/kaggle/data", "/kaggle/working"]
    candidates = []
    for root in roots:
        if os.path.isdir(root):
            candidates.extend(
                glob.glob(os.path.join(root, "**", "*.pth"), recursive=True)
            )
            candidates.extend(
                glob.glob(os.path.join(root, "**", "*.pt"), recursive=True)
            )

    ranked = []
    for p in candidates:
        bn = os.path.basename(p).lower()
        score = 0
        if "model90" in bn:
            score += 100
        if "best" in bn:
            score += 20
        if "checkpoint" in bn:
            score += 10
        if "seresnet50" in p.lower() or "se_resnet50" in p.lower():
            score += 30
        if "aptos" in p.lower():
            score += 5
        if bn.endswith(".pth"):
            score += 1
        ranked.append((score, len(p), p))
    ranked.sort(key=lambda x: (-x[0], x[1], x[2]))
    return ranked[0][2] if ranked else ""


def _extract_state_dict(state):
    if isinstance(state, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "ema_state_dict",
        ]:
            if key in state and isinstance(state[key], dict) and len(state[key]) > 0:
                state = state[key]
                break
    return state


def _sanitize_state_dict_keys(state):
    if not isinstance(state, dict) or len(state) == 0:
        return state
    keys = list(state.keys())
    if all(isinstance(k, str) and k.startswith("module.") for k in keys):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if all(isinstance(k, str) and k.startswith("model.") for k in state.keys()):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}
    return state


def _cohen_kappa_quadratic(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    N = n_classes

    O = np.zeros((N, N), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < N and 0 <= b < N:
            O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=N).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=N).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            W[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - (num / den)


def _apply_thresholds(preds_float, thresholds):
    t0, t1, t2, t3 = thresholds
    preds_float = np.asarray(preds_float, dtype=np.float64)
    out = np.zeros_like(preds_float, dtype=np.int64)
    out[preds_float >= t0] = 1
    out[preds_float >= t1] = 2
    out[preds_float >= t2] = 3
    out[preds_float >= t3] = 4
    return out


def _optimize_thresholds_greedy(y_true, preds_float, init=(0.7, 1.5, 2.5, 3.5)):
    y_true = np.asarray(y_true, dtype=np.int64)
    preds_float = np.asarray(preds_float, dtype=np.float64)

    thresholds = np.array(init, dtype=np.float64)
    thresholds.sort()

    best = thresholds.copy()
    best_score = _cohen_kappa_quadratic(y_true, _apply_thresholds(preds_float, best))

    lo = float(np.percentile(preds_float, 1))
    hi = float(np.percentile(preds_float, 99))
    lo = min(lo, best[0] - 1.0)
    hi = max(hi, best[-1] + 1.0)

    for _ in range(4):
        improved = False
        for i in range(4):
            left_bound = lo if i == 0 else best[i - 1] + 1e-4
            right_bound = hi if i == 3 else best[i + 1] - 1e-4
            if right_bound <= left_bound:
                continue

            grid1 = np.linspace(left_bound, right_bound, 31)
            for v in grid1:
                cand = best.copy()
                cand[i] = v
                score = _cohen_kappa_quadratic(
                    y_true, _apply_thresholds(preds_float, cand)
                )
                if score > best_score + 1e-12:
                    best_score = score
                    best = cand
                    improved = True

            fine_left = max(left_bound, best[i] - (right_bound - left_bound) / 30.0)
            fine_right = min(right_bound, best[i] + (right_bound - left_bound) / 30.0)
            grid2 = np.linspace(fine_left, fine_right, 21)
            for v in grid2:
                cand = best.copy()
                cand[i] = v
                score = _cohen_kappa_quadratic(
                    y_true, _apply_thresholds(preds_float, cand)
                )
                if score > best_score + 1e-12:
                    best_score = score
                    best = cand
                    improved = True
        if not improved:
            break

    return best.tolist(), float(best_score)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_transform = transforms.Compose(
    [
        transforms.Resize(
            (256, 256), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        normalize,
    ]
)

test_df = pd.read_csv(TEST_CSV_PATH)
test_ids = test_df["id_code"].astype(str).tolist()

train_df = pd.read_csv(TRAIN_CSV_PATH)
train_ids = train_df["id_code"].astype(str).tolist()
train_y = train_df["diagnosis"].astype(int).tolist()


class ImagesDS(Dataset):
    def __init__(self, ids, img_dir_candidates, transform):
        self.ids = ids
        self.img_dir_candidates = img_dir_candidates
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        last_path = None
        for d in self.img_dir_candidates:
            im_path = os.path.join(d, f"{id_code}.png")
            last_path = im_path
            if os.path.isfile(im_path):
                image = Image.open(im_path).convert("RGB")
                x = self.transform(image)
                return id_code, x
        raise FileNotFoundError(
            f"Could not find image for id_code={id_code}. Last tried: {last_path}"
        )


class ImagesTrainDS(Dataset):
    def __init__(self, ids, ys, img_dir_candidates, transform):
        self.ids = ids
        self.ys = ys
        self.img_dir_candidates = img_dir_candidates
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        y = float(self.ys[idx])
        last_path = None
        for d in self.img_dir_candidates:
            im_path = os.path.join(d, f"{id_code}.png")
            last_path = im_path
            if os.path.isfile(im_path):
                image = Image.open(im_path).convert("RGB")
                x = self.transform(image)
                return x, torch.tensor([y], dtype=torch.float32)
        raise FileNotFoundError(
            f"Could not find image for id_code={id_code}. Last tried: {last_path}"
        )


def _collate_test(batch):
    ids = [b[0] for b in batch]
    xs = torch.stack([b[1] for b in batch], dim=0)
    return ids, xs


def _seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _predict_regression(ids, img_dir_candidates, batch_size=16):
    ds = ImagesDS(ids, img_dir_candidates, img_transform)
    use_persistent = (os.name != "nt") and (device.type == "cuda")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_test,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(SEED),
        persistent_workers=use_persistent,
    )
    preds = []
    got_ids = []
    with torch.no_grad():
        for b_ids, x in loader:
            x = x.to(device, non_blocking=True)
            out = model(x).squeeze(1).detach().float().cpu().numpy()
            preds.append(out)
            got_ids.extend(b_ids)
    preds = np.concatenate(preds, axis=0)
    return got_ids, preds


def _train_finetune_regression(
    model,
    ids,
    ys,
    img_dir_candidates,
    epochs=1,
    batch_size=16,
    lr=1e-4,
    train_last_linear_only=False,
):
    if train_last_linear_only:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.last_linear.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True

    model.train()
    ds = ImagesTrainDS(ids, ys, img_dir_candidates, img_transform)
    use_persistent = (os.name != "nt") and (device.type == "cuda")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(SEED),
        persistent_workers=use_persistent,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = torch.nn.MSELoss()

    for ep in range(epochs):
        t0 = time.time()
        running = 0.0
        n_seen = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu().item()) * x.size(0)
            n_seen += x.size(0)
        print(
            f"Finetune epoch {ep+1}/{epochs} (last_linear_only={train_last_linear_only}): "
            f"loss={running/max(1,n_seen):.6f} time={time.time()-t0:.1f}s"
        )
    model.eval()
    return model




## === cell 3
ckpt_path = _find_checkpoint_path(MODEL_PATH)
use_checkpoint = bool(ckpt_path)

model = get_se_resnet50_gem(pretrain=("imagenet" if not use_checkpoint else False))
model.to(device)

if use_checkpoint:
    raw = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(raw)
    state = _sanitize_state_dict_keys(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint (non-strict): {ckpt_path}")
    if missing:
        print(f"Missing keys (count={len(missing)}), e.g.: {missing[:5]}")
    if unexpected:
        print(f"Unexpected keys (count={len(unexpected)}), e.g.: {unexpected[:5]}")
else:
    print(
        "WARNING: No external checkpoint found; using offline ImageNet init fallback (no network)."
    )

n = len(train_ids)
idx = np.arange(n)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_size = max(256, int(0.2 * n))
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

train_ids_tr = [train_ids[i] for i in tr_idx]
train_y_tr = [train_y[i] for i in tr_idx]
train_ids_val = [train_ids[i] for i in val_idx]
train_y_val = [train_y[i] for i in val_idx]

model = _train_finetune_regression(
    model,
    train_ids_tr,
    train_y_tr,
    TRAIN_IMAGE_PATH_CANDIDATES,
    epochs=2,
    batch_size=16,
    lr=3e-4,
    train_last_linear_only=True,
)

model = _train_finetune_regression(
    model,
    train_ids_tr,
    train_y_tr,
    TRAIN_IMAGE_PATH_CANDIDATES,
    epochs=1,
    batch_size=16,
    lr=1e-5,
    train_last_linear_only=False,
)

model.eval()

val_got_ids, val_preds = _predict_regression(
    train_ids_val, TRAIN_IMAGE_PATH_CANDIDATES, batch_size=16
)
if val_got_ids != train_ids_val:
    m = dict(zip(val_got_ids, val_preds.tolist()))
    val_preds = np.array([m[i] for i in train_ids_val], dtype=np.float64)

val_preds = np.clip(val_preds, 0.0, 4.0)

init_thr = (0.7, 1.5, 2.5, 3.5)
opt_thr, val_kappa = _optimize_thresholds_greedy(train_y_val, val_preds, init=init_thr)
print(f"Optimized thresholds: {opt_thr} (val QWK={val_kappa:.6f})")

test_got_ids, test_preds = _predict_regression(
    test_ids, TEST_IMAGE_PATH_CANDIDATES, batch_size=16
)
if test_got_ids != test_ids:
    m = dict(zip(test_got_ids, test_preds.tolist()))
    test_preds = np.array([m[i] for i in test_ids], dtype=np.float64)

test_preds = np.clip(test_preds, 0.0, 4.0)

submission_raw = pd.DataFrame(
    {"id_code": test_ids, "diagnosis": test_preds.astype(np.float64)}
)
submission_raw.to_csv("submission_raw_value.csv", index=False)

test_labels = _apply_thresholds(test_preds, opt_thr).astype(np.int64)
submission = pd.DataFrame({"id_code": test_ids, "diagnosis": test_labels})
submission = submission[["id_code", "diagnosis"]]
submission.to_csv("submission.csv", index=False)

print(
    f"Checkpoint used: {ckpt_path if use_checkpoint else 'NONE (offline ImageNet init fallback)'}"
)
print(f"Train CSV: {TRAIN_CSV_PATH}")
print(f"Train image dir: {TRAIN_IMAGE_PATH}")
print(f"Test CSV: {TEST_CSV_PATH}")
print(f"Test image dir (primary): {TEST_IMAGE_PATH}")
print(submission.head())
print("Wrote: submission.csv")
