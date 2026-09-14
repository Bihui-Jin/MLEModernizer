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

0.910943132658642

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime failure by making the checkpoint path robust: the script search common Kaggle input locations for `fine_tune_256_model30.pth` and load it if present. To ensure the notebook always runs end-to-end and still produces a valid `submission.csv`, I add a safe fallback that runs inference with the (untrained) model weights if the checkpoint cannot be found, rather than crashing. I also harden checkpoint loading to handle common formats (`state_dict` wrappers, `module.` prefixes) without changing the model architecture or inference logic. Finally, I keep the existing test-time augmentation and thresholding semantics intact so the evaluation behavior remains the same.'
- What this solution (achieved 0.0) has done: 'I fix the runtime failure by preventing any internet download for ImageNet weights (the SSL error comes from `model_zoo.load_url`). Instead, if the fine-tuned checkpoint cannot be found, the script fall back to initializing from `torchvision`’s local ResNet-50 weights and then transplant those weights into the existing SE-ResNet-50 backbone in a best-effort way (conv/bn layers), keeping your model head/GeM/inference logic intact. This keeps the same pipeline semantics while ensuring the notebook always runs end-to-end and writes a valid `submission.csv`. I also make checkpoint discovery more robust by searching for the checkpoint filename anywhere under `/kaggle/input` and `/kaggle/data` and only using the fallback when it truly isn’t present.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with submitting essentially untrained/incorrectly initialized weights: the code’s “offline torchvision init” does not actually map ResNet-50 weights into your SE-ResNet backbone (key names don’t match), so predictions collapse and QWK tanks. I keep the same model/inference/TTA/thresholding logic, but change the fallback so it never tries to do partial weight transplant; instead it (1) robustly finds and loads the fine-tuned checkpoint by filename anywhere under Kaggle folders, and (2) if not found, uses a deterministic, label-distribution prior (from train.csv) to generate a sane constant prediction and thresholds (still producing a valid submission, but without crashing). This is the smallest change that should move the score upward toward your target when the checkpoint exists, and avoids the misleading “offline init” that produces near-random outputs. I also add a quick assertion that the submission rows align with test.csv to prevent silent formatting/alignment issues.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the fine-tuned checkpoint is not being loaded at all (so you’re submitting majority-class fallback or effectively random outputs). I make the checkpoint discovery actually find the file anywhere under Kaggle’s mounted data (including `/kaggle/data/input/...` and nested competition folders), and I fail “loudly” into a safer path by verifying the loaded state_dict matches the model (so we don’t silently run with mostly-missing weights). If the checkpoint still cannot be found, I keep your fallback submission behavior intact (still produces a valid `submission.csv`), but I also ensure thresholds aren’t “optimized” on garbage by only optimizing when a real checkpoint was loaded successfully. These are minimal changes that preserve your model/inference/TTA/thresholding logic while making it much more likely you actually use the intended trained weights—moving the score upward toward the 0.91 target.'
- What this solution (achieved 0.0) has done: 'I make the checkpoint loading actually succeed by fixing the `map_location` bug (passing a `torch.device` can break `torch.load` on some checkpoints) and by explicitly handling the common case where the checkpoint was saved from a `DataParallel` wrapper or saved as a full training dict. I also change the “loaded_ok” verification to be based on how many parameters were actually loaded (not just missing key count), so we don’t incorrectly fall back to majority-class predictions when the right checkpoint is present. Finally, I keep your model, TTA, and threshold optimization logic identical, but only run threshold optimization when we truly loaded the fine-tuned weights, which should move the score upward toward your 0.91 target instead of staying at 0.0.'

# 9. Code solution

## === cell 0
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
        model = se_resnet50(num_classes=1000, pretrained=None)
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
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "/kaggle/input/seresnet50pretrain/fine_tune_256_model30.pth"

TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
TEST_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/test.csv"
TRAIN_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/train_images"
TRAIN_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"


def resolve_checkpoint_path(primary_path: str) -> str:
    """
    Score fix: 0.0 strongly indicates we are not loading the intended fine-tuned checkpoint.
    Minimal change: broaden the search roots to include Kaggle's '/kaggle/data/input' and any nested folders.
    """
    if primary_path and os.path.exists(primary_path):
        return primary_path

    fname = (
        os.path.basename(primary_path) if primary_path else "fine_tune_256_model30.pth"
    )

    candidates = []
    for root in [
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/data",
        "/kaggle/working",
        "/kaggle",
    ]:
        if os.path.isdir(root):
            candidates.extend(
                glob.glob(os.path.join(root, "**", fname), recursive=True)
            )

    candidates = sorted(set(candidates))
    return candidates[0] if candidates else ""


def _coerce_map_location_for_torch_load(map_location):
    """
    Score fix: make checkpoint loading reliable.
    Minimal change: torch.load(map_location=...) is most robust with a string or callable;
    passing torch.device can fail depending on how the checkpoint was saved.
    """
    if isinstance(map_location, torch.device):
        return str(map_location)
    return map_location


def load_checkpoint_forgiving(model, ckpt_path: str, map_location):
    """
    Score fix: avoid silently "loading" a mismatched checkpoint (leading to near-untrained predictions).
    Minimal change: handle common checkpoint wrappers, strip 'module.' prefixes, and *verify*
    load quality by counting tensors that actually matched in name+shape.
    """
    map_location = _coerce_map_location_for_torch_load(map_location)
    ckpt = torch.load(ckpt_path, map_location=map_location)

    state_dict = ckpt
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break

    if not isinstance(state_dict, dict):
        return ["non_state_dict_checkpoint"], [], False

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model_sd = model.state_dict()
    matched = 0
    for k, v in state_dict.items():
        if k in model_sd and hasattr(v, "shape") and v.shape == model_sd[k].shape:
            matched += 1

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    total_keys = len(model_sd)
    matched_ratio = matched / max(total_keys, 1)

    loaded_ok = matched_ratio >= 0.80

    print(f"Checkpoint matched tensors: {matched}/{total_keys} ({matched_ratio:.3f})")
    return missing, unexpected, loaded_ok


def resolve_competition_paths():
    def first_existing(paths, kind="file"):
        for p in paths:
            if kind == "file" and os.path.isfile(p):
                return p
            if kind == "dir" and os.path.isdir(p):
                return p
        return ""

    test_csv = first_existing(
        [
            TEST_CSV_PATH,
            "/kaggle/input/test.csv",
            "/kaggle/data/test.csv",
            "/kaggle/input/aptos2019-blindness-detection/test.csv",
            "/kaggle/data/aptos2019-blindness-detection/test.csv",
        ],
        kind="file",
    )
    train_csv = first_existing(
        [
            TRAIN_CSV_PATH,
            "/kaggle/input/train.csv",
            "/kaggle/data/train.csv",
            "/kaggle/input/aptos2019-blindness-detection/train.csv",
            "/kaggle/data/aptos2019-blindness-detection/train.csv",
        ],
        kind="file",
    )
    test_img_dir = first_existing(
        [
            TEST_IMAGE_PATH,
            "/kaggle/input/test_images",
            "/kaggle/data/test_images",
            "/kaggle/input/aptos2019-blindness-detection/test_images",
            "/kaggle/data/aptos2019-blindness-detection/test_images",
        ],
        kind="dir",
    )
    train_img_dir = first_existing(
        [
            TRAIN_IMAGE_PATH,
            "/kaggle/input/train_images",
            "/kaggle/data/train_images",
            "/kaggle/input/aptos2019-blindness-detection/train_images",
            "/kaggle/data/aptos2019-blindness-detection/train_images",
        ],
        kind="dir",
    )

    if not test_csv or not test_img_dir:
        raise FileNotFoundError(
            f"Could not resolve test.csv/test_images. Got test_csv={test_csv!r}, test_img_dir={test_img_dir!r}"
        )

    return train_csv, train_img_dir, test_csv, test_img_dir


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape
    N = n_classes

    O = np.zeros((N, N), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < N and 0 <= b < N:
            O[a, b] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            W[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


def apply_thresholds(preds, ths):
    t0, t1, t2, t3 = ths
    preds = np.asarray(preds, dtype=np.float32)
    out = np.zeros_like(preds, dtype=np.int64)
    out[preds >= t0] = 1
    out[preds >= t1] = 2
    out[preds >= t2] = 3
    out[preds >= t3] = 4
    return out


def optimize_thresholds(preds, y_true, init_ths=(0.7, 1.5, 2.5, 3.5), iters=8):
    preds = np.asarray(preds, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int64)

    ths = np.array(init_ths, dtype=np.float32)
    best = quadratic_weighted_kappa(y_true, apply_thresholds(preds, ths))

    lo, hi = float(np.min(preds)), float(np.max(preds))
    span = max(hi - lo, 1e-3)
    step = span / 10.0

    for _ in range(iters):
        improved = False
        for k in range(4):
            for delta in (-step, step):
                cand = ths.copy()
                cand[k] = cand[k] + delta
                cand = np.sort(cand)
                cand[0] = max(cand[0], lo - 1.0)
                cand[3] = min(cand[3], hi + 1.0)
                score = quadratic_weighted_kappa(y_true, apply_thresholds(preds, cand))
                if score > best:
                    ths, best = cand, score
                    improved = True
        if not improved:
            step *= 0.5

    return ths, best


def majority_class_from_train(train_csv_path: str) -> int:
    """
    Score safety: if checkpoint is missing/mismatched, produce a stable submission.
    """
    try:
        df = pd.read_csv(train_csv_path)
        return int(df["diagnosis"].value_counts().idxmax())
    except Exception:
        return 0


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

train_csv, train_img_dir, test_csv, test_img_dir = resolve_competition_paths()

ckpt_path = resolve_checkpoint_path(MODEL_PATH)

model = get_se_resnet50_gem(pretrain=False)
model.to(device)

use_model_inference = True
checkpoint_loaded_ok = False

if ckpt_path:
    missing, unexpected, loaded_ok = load_checkpoint_forgiving(
        model, ckpt_path, map_location=device
    )
    checkpoint_loaded_ok = bool(loaded_ok)
    print("Found checkpoint:", ckpt_path)
    if missing:
        print("Warning: missing keys (showing up to 10):", missing[:10])
    if unexpected:
        print("Warning: unexpected keys (showing up to 10):", unexpected[:10])

    if not checkpoint_loaded_ok:
        use_model_inference = False
        print(
            "Checkpoint appears mismatched/insufficiently loaded; falling back to majority-class submission."
        )
else:
    use_model_inference = False
    print("Checkpoint not found; falling back to majority-class submission.")

model.eval()

norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_df = pd.read_csv(test_csv)
test_images = [
    os.path.join(test_img_dir, f"{i}.png") for i in test_df["id_code"].tolist()
]


def predict_image_paths(image_paths, batch_size=16):
    preds = []
    with torch.inference_mode():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            imgs = []
            imgs_flip = []
            for p in batch_paths:
                im = Image.open(p).convert("RGB")
                im = im.resize((256, 256), resample=Image.BILINEAR)
                t = norm(im)
                imgs.append(t)
                imgs_flip.append(torch.flip(t, dims=(2,)))  # flip width dim=2 (CHW)
            x = torch.stack(imgs, dim=0).to(device)
            x_f = torch.stack(imgs_flip, dim=0).to(device)
            out = model(x).squeeze(1)
            out_f = model(x_f).squeeze(1)
            final = (out + out_f) / 2.0
            preds.extend(final.detach().float().cpu().numpy().tolist())
    return np.array(preds, dtype=np.float32)


thresholds = np.array([0.7, 1.5, 2.5, 3.5], dtype=np.float32)

if use_model_inference:
    if (
        checkpoint_loaded_ok
        and train_csv
        and train_img_dir
        and os.path.isfile(train_csv)
        and os.path.isdir(train_img_dir)
    ):
        train_df = pd.read_csv(train_csv)
        idx = np.arange(len(train_df))
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
        n_val = max(300, int(0.15 * len(idx)))
        val_idx = idx[:n_val]
        val_df = train_df.iloc[val_idx].reset_index(drop=True)

        val_paths = [
            os.path.join(train_img_dir, f"{i}.png") for i in val_df["id_code"].tolist()
        ]
        val_preds = predict_image_paths(val_paths, batch_size=16)
        val_y = val_df["diagnosis"].astype(int).values

        thresholds, best_kappa = optimize_thresholds(
            val_preds, val_y, init_ths=tuple(thresholds), iters=10
        )
        print(
            "Optimized thresholds:",
            thresholds.tolist(),
            "holdout QWK:",
            float(best_kappa),
        )

    test_preds = predict_image_paths(test_images, batch_size=16)
    test_labels = apply_thresholds(test_preds, thresholds).astype(int)
else:
    maj = majority_class_from_train(train_csv)
    test_labels = np.full(shape=(len(test_df),), fill_value=maj, dtype=int)

submission = test_df.copy()
submission["diagnosis"] = test_labels

assert submission.shape[0] == test_df.shape[0]
assert (
    submission["id_code"].astype(str).tolist()
    == test_df["id_code"].astype(str).tolist()
)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("diagnosis value counts:\n", submission["diagnosis"].value_counts().sort_index())
