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

0.8690917331934382

# 6. Current score

0.7452

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.6651) has done: 'The timeout is dominated by doing full 3-fold training over ~3.3k high-res PNGs plus repeated checkpoint reloads and slow PIL decode/transform in dataloader workers. I preserve the exact model, loss, and training loops, but eliminate redundant work: reuse a single loaded state_dict across folds, keep fold models in memory instead of re-loading from disk each time, and enable CUDA/CuDNN optimizations that are numerically equivalent (benchmark + channels_last) for faster convs. I also speed up data loading without changing the data by using persistent workers, prefetching, and faster PIL decoding settings, and vectorize QWK computation to remove Python loops. These changes reduce overhead and improve GPU utilization while keeping the algorithm and outputs consistent aside from negligible float differences.'
- What this solution (achieved 0.66999) has done: 'The timeout is dominated by repeatedly reading and transforming thousands of PNGs into memory multiple times (full-train cache + per-fold cache + test cache), plus using many DataLoader workers on already-cached tensors (extra overhead). I preserve the exact model, losses, epoch counts, thresholds fitting, and prediction semantics, but switch to a single shared in-memory image cache that is reused for fine-tune, OOF folds, and test inference so each image is decoded/transformed at most once. I also avoid spinning up multi-process DataLoader workers when the dataset is fully cached (use num_workers=0), and I prebuild lightweight index-based datasets for folds to remove per-fold caching overhead entirely. These changes are provably equivalent (same transformed tensors, just reused), but drastically cut I/O and redundant preprocessing.'
- What this solution (achieved 0.7452) has done: 'Main bottlenecks are (1) building two full in-memory image caches by decoding and transforming ~3662 PNGs in Python (slow and redundant with DataLoader), and (2) extremely expensive threshold fitting that repeatedly recomputes QWK from scratch inside nested loops. To finish under 600s without changing the model or training semantics, this refactor removes the eager image caching and instead uses a high-throughput DataLoader (multi-worker, pinned memory, persistent workers) with a cheap collate, while keeping the exact same transforms and train/predict loops. It also replaces the threshold search’s repeated full QWK recomputation with an equivalent-but-faster implementation that precomputes the weight matrix once and uses bincount on a single encoded confusion index (same metric, same thresholds/steps/search logic). Finally, it avoids unnecessary copies and uses `torch.inference_mode()` for prediction to reduce overhead while preserving outputs.'

# 9. Code solution

## === cell 0
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
        model = se_resnet50(num_classes=1000, pretrained=None)
    else:
        model = se_resnet50(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(2048, 1)
    return model




## === cell 3
import os
from glob import glob
import random

import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = (
        False  # keep as default; avoids forcing slow deterministic kernels
    )

Image.MAX_IMAGE_PIXELS = None

MODEL_PATH = "/kaggle/input/seresnet50-2/model15.pth"

TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
TEST_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/test.csv"

TRAIN_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/train_images"
TRAIN_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"


def find_checkpoint_path(preferred_path: str) -> str:
    if preferred_path and os.path.isfile(preferred_path):
        return preferred_path

    candidates = []
    for pattern in [
        "/kaggle/input/**/*.pth",
        "/kaggle/input/**/*.pt",
        "/kaggle/working/**/*.pth",
        "/kaggle/working/**/*.pt",
    ]:
        candidates.extend(glob(pattern, recursive=True))

    preferred_names = {"model15.pth", "model.pth", "model15.pt", "model.pt"}
    preferred = [
        p for p in candidates if os.path.basename(p).lower() in preferred_names
    ]
    if preferred:
        return sorted(preferred)[0]
    return ""


def normalize_state_dict_keys(state: dict) -> dict:
    if not isinstance(state, dict):
        return state

    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        nk = nk.replace(".se_fc1.", ".se_module.fc1.")
        nk = nk.replace(".se_fc2.", ".se_module.fc2.")
        new_state[nk] = v
    return new_state


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

img_tfms = transforms.Compose(
    [
        transforms.Resize(
            (256, 256), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class RetinoDataset(Dataset):
    def __init__(self, df, img_dir, tfms, with_label: bool):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfms = tfms
        self.with_label = with_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        with Image.open(im_path) as im:
            image = im.convert("RGB")
        image = self.tfms(image)
        if self.with_label:
            y = float(row["diagnosis"])
            return image, torch.tensor(y, dtype=torch.float32)
        return image, row["id_code"]


def train_one_epoch(model, dl, optimizer, criterion):
    model.train()
    running = 0.0
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        if xb.is_cuda:
            xb = xb.contiguous(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(xb).squeeze(1)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * xb.size(0)
    return running / len(dl.dataset)


@torch.inference_mode()
def predict_regression(model, dl):
    model.eval()
    preds = []
    ids = []
    for batch in dl:
        if (
            isinstance(batch, (list, tuple))
            and len(batch) == 2
            and torch.is_tensor(batch[0])
            and not torch.is_tensor(batch[1])
        ):
            xb, idb = batch
            xb = xb.to(device, non_blocking=True)
            if xb.is_cuda:
                xb = xb.contiguous(memory_format=torch.channels_last)
            out = model(xb).squeeze(1).detach().cpu().numpy()
            preds.append(out)
            ids.extend(list(idb))
        else:
            xb, yb = batch
            xb = xb.to(device, non_blocking=True)
            if xb.is_cuda:
                xb = xb.contiguous(memory_format=torch.channels_last)
            out = model(xb).squeeze(1).detach().cpu().numpy()
            preds.append(out)
    if preds:
        preds = np.concatenate(preds, axis=0)
    else:
        preds = np.array([], dtype=np.float32)
    return preds, ids


def _qwk_weight_matrix(n_classes=5):
    w_idx = np.arange(n_classes, dtype=np.float64)
    return (w_idx[:, None] - w_idx[None, :]) ** 2 / ((n_classes - 1) ** 2)


_W_QWK_5 = _qwk_weight_matrix(5)


def qwk_quadratic(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    valid = (y_true >= 0) & (y_true < n_classes) & (y_pred >= 0) & (y_pred < n_classes)
    yt = y_true[valid]
    yp = y_pred[valid]
    if yt.size == 0:
        return 0.0

    O = (
        np.bincount(yt * n_classes + yp, minlength=n_classes * n_classes)
        .astype(np.float64)
        .reshape(n_classes, n_classes)
    )

    act_hist = np.bincount(yt, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(yp, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = _W_QWK_5 if n_classes == 5 else _qwk_weight_matrix(n_classes)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


def apply_thresholds(x, thr):
    thr = np.asarray(thr, dtype=np.float64)
    return np.digitize(x, thr).astype(int)


def fit_thresholds_for_qwk(y_true, y_pred_cont):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float64)
    n_classes = 5
    W = _W_QWK_5

    valid = (y_true >= 0) & (y_true < n_classes)
    yt = y_true[valid]
    yp_cont = y_pred_cont[valid]

    act_hist = np.bincount(yt, minlength=n_classes).astype(np.float64)
    act_sum = float(act_hist.sum())

    def qwk_from_thresholds(thr_vec):
        yp = np.digitize(yp_cont, thr_vec).astype(np.int64)
        pred_hist = np.bincount(yp, minlength=n_classes).astype(np.float64)

        O = (
            np.bincount(yt * n_classes + yp, minlength=n_classes * n_classes)
            .astype(np.float64)
            .reshape(n_classes, n_classes)
        )
        E = np.outer(act_hist, pred_hist)
        e_sum = float(E.sum())
        if e_sum > 0:
            E *= act_sum / e_sum

        denom = float((W * E).sum())
        if denom == 0.0:
            return 0.0
        return 1.0 - float((W * O).sum()) / denom

    thr = np.array([0.7, 1.5, 2.5, 3.5], dtype=np.float64)

    best_thr = thr.copy()
    best_score = qwk_from_thresholds(best_thr)

    for step in [0.2, 0.1, 0.05]:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for direction in (-1.0, 1.0):
                    cand = best_thr.copy()
                    cand[i] = cand[i] + direction * step
                    if not (cand[0] < cand[1] < cand[2] < cand[3]):
                        continue
                    score = qwk_from_thresholds(cand)
                    if score > best_score + 1e-10:
                        best_score = score
                        best_thr = cand
                        improved = True
    return best_thr, best_score


class SubsetByIndexDataset(Dataset):
    def __init__(self, base_ds, indices):
        self.base_ds = base_ds
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base_ds[int(self.indices[i])]


def make_loader(ds, batch_size, shuffle, cached: bool):
    if cached:
        num_workers = 0
        persistent_workers = False
        prefetch_factor = None
    else:
        cpu = os.cpu_count() or 2
        num_workers = min(8, max(2, cpu // 2))
        persistent_workers = True
        prefetch_factor = 4

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

model = get_se_resnet50_gem(pretrain="none").to(device)
if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

MODEL_PATH = find_checkpoint_path(MODEL_PATH)
if MODEL_PATH:
    print("Using existing MODEL_PATH:", MODEL_PATH)
    base_state = torch.load(MODEL_PATH, map_location="cpu")
    base_state = normalize_state_dict_keys(base_state)
    model.load_state_dict(base_state, strict=True)
else:
    print(
        "No compatible checkpoint found; training a fallback checkpoint to /kaggle/working/model_fallback.pth"
    )

    ds_tmp = RetinoDataset(train_df, TRAIN_IMAGE_PATH, img_tfms, with_label=True)
    dl_tmp = make_loader(ds_tmp, batch_size=8, shuffle=True, cached=False)

    optimizer_tmp = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion_tmp = torch.nn.MSELoss()

    epochs = 1
    for ep in range(epochs):
        mse = train_one_epoch(model, dl_tmp, optimizer_tmp, criterion_tmp)
        print(f"epoch {ep+1}/{epochs} - train_mse: {mse:.5f}")

    MODEL_PATH = "/kaggle/working/model_fallback.pth"
    torch.save(model.state_dict(), MODEL_PATH)
    print("Saved fallback checkpoint:", MODEL_PATH)

    base_state = model.state_dict()

model.eval()

n_folds = 3
idx = np.arange(len(train_df))
rng = np.random.default_rng(42)
rng.shuffle(idx)
folds = np.array_split(idx, n_folds)

oof_pred = np.zeros(len(train_df), dtype=np.float32)

test_ds = RetinoDataset(test_df, TEST_IMAGE_PATH, img_tfms, with_label=False)
test_dl = make_loader(test_ds, batch_size=32, shuffle=False, cached=False)
test_pred_cont_accum = np.zeros(len(test_df), dtype=np.float64)

full_train_ds = RetinoDataset(train_df, TRAIN_IMAGE_PATH, img_tfms, with_label=True)

for fold_i in range(n_folds):
    val_idx = folds[fold_i]
    tr_idx = np.setdiff1d(idx, val_idx, assume_unique=False)

    fold_model = get_se_resnet50_gem(pretrain="none").to(device)
    if device.type == "cuda":
        fold_model = fold_model.to(memory_format=torch.channels_last)
    fold_model.load_state_dict(base_state, strict=True)

    tr_ds = SubsetByIndexDataset(full_train_ds, tr_idx)
    tr_dl = make_loader(tr_ds, batch_size=8, shuffle=True, cached=False)

    optimizer = torch.optim.Adam(fold_model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    mse = train_one_epoch(fold_model, tr_dl, optimizer, criterion)
    print(f"fold {fold_i+1}/{n_folds} - train_mse: {mse:.5f}")

    va_ds = SubsetByIndexDataset(full_train_ds, val_idx)
    va_dl = make_loader(va_ds, batch_size=32, shuffle=False, cached=False)
    preds_val, _ = predict_regression(fold_model, va_dl)
    oof_pred[val_idx] = preds_val.astype(np.float32)
    print(f"fold {fold_i+1}/{n_folds} - val_pred_done")

    fold_test_pred, test_ids_ref = predict_regression(fold_model, test_dl)
    test_pred_cont_accum += fold_test_pred.astype(np.float64)

test_pred_cont = (test_pred_cont_accum / float(n_folds)).astype(np.float32)

y_true = train_df["diagnosis"].astype(int).values
thr, oof_qwk = fit_thresholds_for_qwk(y_true, oof_pred)
print("Fitted thresholds:", thr.tolist())
print("OOF QWK (for sanity):", float(oof_qwk))

submission_raw = pd.DataFrame(
    {"id_code": test_ids_ref, "diagnosis": test_pred_cont.astype(float)}
)
submission_raw = test_df.merge(submission_raw, on="id_code", how="left")
submission_raw.to_csv("submission_raw_value.csv", index=False)

test_pred_cls = apply_thresholds(submission_raw["diagnosis"].values, thr)
submission = pd.DataFrame(
    {
        "id_code": submission_raw["id_code"].values,
        "diagnosis": test_pred_cls.astype(int),
    }
)

submission = test_df.merge(submission, on="id_code", how="left")
submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int)
submission = submission[["id_code", "diagnosis"]]

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Submission columns:", submission.columns.tolist())
