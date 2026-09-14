# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.9193205713721068

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import types
import re
from collections import OrderedDict
import math



## === cell 1
__all__ = [
    "alexnet",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "inceptionv3",
    "squeezenet1_0",
    "squeezenet1_1",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]

model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
    "densenet121": "http://data.lip6.fr/cadene/pretrainedmodels/densenet121-fbdb23505.pth",
    "densenet169": "http://data.lip6.fr/cadene/pretrainedmodels/densenet169-f470b90a4.pth",
    "densenet201": "http://data.lip6.fr/cadene/pretrainedmodels/densenet201-5750cbb1e.pth",
    "densenet161": "http://data.lip6.fr/cadene/pretrainedmodels/densenet161-347e6b360.pth",
    "inceptionv3": "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth",
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "squeezenet1_0": "https://download.pytorch.org/models/squeezenet1_0-a815701f.pth",
    "squeezenet1_1": "https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth",
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}

input_sizes = {}
means = {}
stds = {}

for model_name in __all__:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]

for model_name in ["inceptionv3"]:
    input_sizes[model_name] = [3, 299, 299]
    means[model_name] = [0.5, 0.5, 0.5]
    stds[model_name] = [0.5, 0.5, 0.5]

pretrained_settings = {}

for model_name in __all__:
    pretrained_settings[model_name] = {
        "imagenet": {
            "url": model_urls[model_name],
            "input_space": "RGB",
            "input_size": input_sizes[model_name],
            "input_range": [0, 1],
            "mean": means[model_name],
            "std": stds[model_name],
            "num_classes": 1000,
        }
    }


def update_state_dict(state_dict):
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict


def load_pretrained(model, num_classes, settings):
    assert (
        num_classes == settings["num_classes"]
    ), "num_classes should be {}, but is {}".format(
        settings["num_classes"], num_classes
    )
    state_dict = model_zoo.load_url(settings["url"])
    state_dict = update_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.input_space = settings["input_space"]
    model.input_size = settings["input_size"]
    model.input_range = settings["input_range"]
    model.mean = settings["mean"]
    model.std = settings["std"]
    return model


def modify_alexnet(model):
    model._features = model.features
    del model.features
    model.dropout0 = model.classifier[0]
    model.linear0 = model.classifier[1]
    model.relu0 = model.classifier[2]
    model.dropout1 = model.classifier[3]
    model.linear1 = model.classifier[4]
    model.relu1 = model.classifier[5]
    model.last_linear = model.classifier[6]
    del model.classifier

    def features(self, input):
        x = self._features(input)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.dropout0(x)
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.dropout1(x)
        x = self.linear1(x)
        return x

    def logits(self, features):
        x = self.relu1(features)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def alexnet(num_classes=1000, pretrained="imagenet"):
    model = models.alexnet(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["alexnet"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_alexnet(model)
    return model


def modify_densenets(model):
    model.last_linear = model.classifier
    del model.classifier

    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def densenet121(num_classes=1000, pretrained="imagenet"):
    model = models.densenet121(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["densenet121"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet169(num_classes=1000, pretrained="imagenet"):
    model = models.densenet169(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["densenet169"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet201(num_classes=1000, pretrained="imagenet"):
    model = models.densenet201(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["densenet201"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet161(num_classes=1000, pretrained="imagenet"):
    model = models.densenet161(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["densenet161"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def modify_resnets(model):
    model.last_linear = model.fc
    model.fc = None

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def resnet18(num_classes=1000, pretrained="imagenet"):
    model = models.resnet18(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["resnet18"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet34(num_classes=1000, pretrained="imagenet"):
    model = models.resnet34(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["resnet34"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet50(num_classes=1000, pretrained="imagenet"):
    model = models.resnet50(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["resnet50"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet101(num_classes=1000, pretrained="imagenet"):
    model = models.resnet101(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["resnet101"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet152(num_classes=1000, pretrained="imagenet"):
    model = models.resnet152(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["resnet152"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def modify_squeezenets(model):
    model.dropout = model.classifier[0]
    model.last_conv = model.classifier[1]
    model.relu = model.classifier[2]
    model.avgpool = model.classifier[3]
    del model.classifier

    def logits(self, features):
        x = self.dropout(features)
        x = self.last_conv(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def squeezenet1_0(num_classes=1000, pretrained="imagenet"):
    model = models.squeezenet1_0(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["squeezenet1_0"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_squeezenets(model)
    return model


def squeezenet1_1(num_classes=1000, pretrained="imagenet"):
    model = models.squeezenet1_1(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["squeezenet1_1"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_squeezenets(model)
    return model


def modify_vggs(model):
    model._features = model.features
    del model.features
    model.linear0 = model.classifier[0]
    model.relu0 = model.classifier[1]
    model.dropout0 = model.classifier[2]
    model.linear1 = model.classifier[3]
    model.relu1 = model.classifier[4]
    model.dropout1 = model.classifier[5]
    model.last_linear = model.classifier[6]
    del model.classifier

    def features(self, input):
        x = self._features(input)
        x = x.view(x.size(0), -1)
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.dropout0(x)
        x = self.linear1(x)
        return x

    def logits(self, features):
        x = self.relu1(features)
        x = self.dropout1(x)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def vgg11(num_classes=1000, pretrained="imagenet"):
    model = models.vgg11(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg11"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg11_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg11_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg11_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg13(num_classes=1000, pretrained="imagenet"):
    model = models.vgg13(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg13"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg13_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg13_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg13_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg16(num_classes=1000, pretrained="imagenet"):
    model = models.vgg16(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg16"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg16_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg16_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg16_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg19(num_classes=1000, pretrained="imagenet"):
    model = models.vgg19(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg19"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg19_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg19_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg19_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model




## === cell 2
"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from torch.utils import model_zoo as _model_zoo

__all__ += [
    "SENet",
    "senet154",
    "se_resnet50",
    "se_resnet101",
    "se_resnet152",
    "se_resnext50_32x4d",
    "se_resnext101_32x4d",
]

pretrained_settings_senet = {
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
        layers = [block(self.inplanes, planes, groups, reduction, stride, downsample)]
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
    assert num_classes == settings["num_classes"]
    model.load_state_dict(_model_zoo.load_url(settings["url"]))
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
        settings = pretrained_settings_senet["senet154"][pretrained]
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
        settings = pretrained_settings_senet["se_resnet50"][pretrained]
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
        settings = pretrained_settings_senet["se_resnet101"][pretrained]
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
        settings = pretrained_settings_senet["se_resnet152"][pretrained]
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
        settings = pretrained_settings_senet["se_resnext50_32x4d"][pretrained]
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
        settings = pretrained_settings_senet["se_resnext101_32x4d"][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model




## === cell 3
import os
from glob import glob
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
test_images = glob(os.path.join(TEST_IMAGE_PATH, "*.png"))
print(f"Found {len(test_images)} test images.")




## === cell 4
def make_predictions(model, test_images, transform, size=256, device=device):
    model.eval()
    predictions = []
    for im_path in test_images:
        with Image.open(im_path) as img:
            img = img.convert("RGB")
            img = img.resize((size, size), resample=Image.BILINEAR)
        img_tensor = transform(img).to(device)
        with torch.no_grad():
            out1 = model(img_tensor.unsqueeze(0))
            out2 = model(torch.flip(img_tensor.unsqueeze(0), dims=(3,)))
        pred = (out1.item() + out2.item()) / 2.0
        predictions.append((os.path.splitext(os.path.basename(im_path))[0], pred))
    return predictions




## === cell 5
def load_model_safe(builder_fn, weight_path):
    model = builder_fn(pretrain=False)
    model = model.to(device)
    if os.path.exists(weight_path):
        try:
            state = torch.load(weight_path, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded weights from {weight_path}")
        except Exception as e:
            print(f"Failed to load weights from {weight_path}: {e}")
    else:
        print(f"Weight file not found at {weight_path}; using random init.")
    model.eval()
    return model


densenet_weight_path = "/kaggle/input/densenet121/model_densenet121_bs64_30.pth"
model_densenet = load_model_safe(get_densenet121_gem, densenet_weight_path)
norm_dn = transforms.Compose([transforms.ToTensor()])
predictions_densenet = make_predictions(model_densenet, test_images, norm_dn, size=224)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/660908040.py in <cell line: 0>()
     18 # Densenet121 predictions
     19 densenet_weight_path = "/kaggle/input/densenet121/model_densenet121_bs64_30.pth"
---> 20 model_densenet = load_model_safe(get_densenet121_gem, densenet_weight_path)
     21 norm_dn = transforms.Compose([transforms.ToTensor()])
     22 predictions_densenet = make_predictions(model_densenet, test_images, norm_dn, size=224)

NameError: name 'get_densenet121_gem' is not defined

## === cell 6
seresnet_weight_path = "/kaggle/input/seresnet50testpseudo/model10.pth"
model_seresnet = load_model_safe(get_se_resnet50_gem, seresnet_weight_path)
norm_sr = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
predictions_seresnet = make_predictions(model_seresnet, test_images, norm_sr, size=256)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1323753157.py in <cell line: 0>()
      1 # SEResNet50 predictions (pseudo label)
      2 seresnet_weight_path = "/kaggle/input/seresnet50testpseudo/model10.pth"
----> 3 model_seresnet = load_model_safe(get_se_resnet50_gem, seresnet_weight_path)
      4 norm_sr = transforms.Compose(
      5     [

NameError: name 'get_se_resnet50_gem' is not defined

## === cell 7
seresnet512_weight_path = "/kaggle/input/seresnet50-512/model30_512.pth"
model_seresnet_512 = load_model_safe(get_se_resnet50_gem, seresnet512_weight_path)
norm_sr512 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
predictions_seresnet_512 = make_predictions(
    model_seresnet_512, test_images, norm_sr512, size=512
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3479767497.py in <cell line: 0>()
      1 # SEResNet50 predictions (512 size)
      2 seresnet512_weight_path = "/kaggle/input/seresnet50-512/model30_512.pth"
----> 3 model_seresnet_512 = load_model_safe(get_se_resnet50_gem, seresnet512_weight_path)
      4 norm_sr512 = transforms.Compose(
      5     [

NameError: name 'get_se_resnet50_gem' is not defined

## === cell 8
final_predictions = []
for idx in range(len(test_images)):
    id_code = os.path.splitext(os.path.basename(test_images[idx]))[0]
    vals = []
    if idx < len(predictions_densenet):
        vals.append(predictions_densenet[idx][1])
    if idx < len(predictions_seresnet):
        vals.append(predictions_seresnet[idx][1])
    if idx < len(predictions_seresnet_512):
        vals.append(predictions_seresnet_512[idx][1])
    avg_pred = sum(vals) / len(vals) if vals else 0.0
    final_predictions.append([id_code, avg_pred])



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2353983957.py in <cell line: 0>()
      4     id_code = os.path.splitext(os.path.basename(test_images[idx]))[0]
      5     vals = []
----> 6     if idx < len(predictions_densenet):
      7         vals.append(predictions_densenet[idx][1])
      8     if idx < len(predictions_seresnet):

NameError: name 'predictions_densenet' is not defined

## === cell 9
submission = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis"])
submission.loc[submission.diagnosis < 0.75, "diagnosis"] = 0
submission.loc[
    (0.75 <= submission.diagnosis) & (submission.diagnosis < 1.5), "diagnosis"
] = 1
submission.loc[
    (1.5 <= submission.diagnosis) & (submission.diagnosis < 2.5), "diagnosis"
] = 2
submission.loc[
    (2.5 <= submission.diagnosis) & (submission.diagnosis < 3.4), "diagnosis"
] = 3
submission.loc[submission.diagnosis >= 3.4, "diagnosis"] = 4
submission["diagnosis"] = submission["diagnosis"].astype(int)

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")

## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame should not be empty
