# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import types
import re

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
    raise RuntimeError(
        "External pretrained weight download disabled for offline Kaggle execution. "
        "Set pretrained=None (already enforced in this script)."
    )


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


def alexnet(num_classes=1000, pretrained=None):
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


def densenet121(num_classes=1000, pretrained=None):
    model = models.densenet121(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["densenet121"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet169(num_classes=1000, pretrained=None):
    model = models.densenet169(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["densenet169"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet201(num_classes=1000, pretrained=None):
    model = models.densenet201(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["densenet201"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet161(num_classes=1000, pretrained=None):
    model = models.densenet161(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["densenet161"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def inceptionv3(num_classes=1000, pretrained=None):
    model = models.inception_v3(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["inceptionv3"][pretrained]
        model = load_pretrained(model, num_classes, settings)

    model.last_linear = model.fc
    del model.fc

    def features(self, input):
        x = self.Conv2d_1a_3x3(input)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        if self.training and self.aux_logits:
            self._out_aux = self.AuxLogits(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        return x

    def logits(self, features):
        x = F.avg_pool2d(features, kernel_size=8)
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        if self.training and self.aux_logits:
            aux = self._out_aux
            self._out_aux = None
            return x, aux
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
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


def resnet18(num_classes=1000, pretrained=None):
    model = models.resnet18(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["resnet18"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet34(num_classes=1000, pretrained=None):
    model = models.resnet34(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["resnet34"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet50(num_classes=1000, pretrained=None):
    model = models.resnet50(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["resnet50"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet101(num_classes=1000, pretrained=None):
    model = models.resnet101(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["resnet101"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet152(num_classes=1000, pretrained=None):
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


def squeezenet1_0(num_classes=1000, pretrained=None):
    model = models.squeezenet1_0(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["squeezenet1_0"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_squeezenets(model)
    return model


def squeezenet1_1(num_classes=1000, pretrained=None):
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


def vgg11(num_classes=1000, pretrained=None):
    model = models.vgg11(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg11"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg11_bn(num_classes=1000, pretrained=None):
    model = models.vgg11_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg11_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg13(num_classes=1000, pretrained=None):
    model = models.vgg13(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg13"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg13_bn(num_classes=1000, pretrained=None):
    model = models.vgg13_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg13_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg16(num_classes=1000, pretrained=None):
    model = models.vgg16(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg16"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg16_bn(num_classes=1000, pretrained=None):
    model = models.vgg16_bn(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg16_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg19(num_classes=1000, pretrained=None):
    model = models.vgg19(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings["vgg19"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg19_bn(num_classes=1000, pretrained=None):
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
    raise RuntimeError(
        "External pretrained SENet weight download disabled for offline Kaggle execution."
    )


def senet154(num_classes=1000, pretrained=None):
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


def se_resnet50(num_classes=1000, pretrained=None):
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


def se_resnet101(num_classes=1000, pretrained=None):
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


def se_resnet152(num_classes=1000, pretrained=None):
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


def se_resnext50_32x4d(num_classes=1000, pretrained=None):
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


def se_resnext101_32x4d(num_classes=1000, pretrained=None):
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
import sys

sys.path.append("/kaggle/working/")

import os
import random
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


def _try_load_torchvision_imagenet_weights_resnet50(model):
    try:
        from torchvision.models import ResNet50_Weights

        sd = ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=False)
        return None
    except Exception:
        return None


def _try_load_torchvision_imagenet_weights_densenet121(model):
    try:
        from torchvision.models import DenseNet121_Weights

        sd = DenseNet121_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
        model.load_state_dict(sd, strict=False)
        return True
    except Exception:
        return False


def get_se_resnet50_gem(pretrain):
    model = se_resnet50(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(2048, 1)
    return model


def get_densenet121_gem(pretrain):
    model = densenet121(num_classes=1000, pretrained=None)

    model.avg_pool = GeM()
    model.last_linear = nn.Linear(1024, 1)

    def logits_gem(self, features):
        x = F.relu(features, inplace=True)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    model.logits = types.MethodType(logits_gem, model)
    return model




## === cell 4
from glob import glob
from PIL import Image, ImageFile
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _resolve_data_root():
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/data/input/aptos2019-blindness-detection",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            return c
    for base in ["/kaggle/input", "/kaggle/data", "/kaggle/data/input"]:
        hits = glob(
            os.path.join(base, "**", "aptos2019-blindness-detection", "train.csv"),
            recursive=True,
        )
        if hits:
            return os.path.dirname(hits[0])
    raise FileNotFoundError(
        "Could not resolve aptos2019-blindness-detection data root."
    )


DATA_ROOT = _resolve_data_root()
TEST_IMAGE_PATH = os.path.join(DATA_ROOT, "test_images")
TRAIN_IMAGE_PATH = os.path.join(DATA_ROOT, "train_images")
TEST_CSV_PATH = os.path.join(DATA_ROOT, "test.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

test_df = pd.read_csv(TEST_CSV_PATH)
train_df = pd.read_csv(TRAIN_CSV_PATH)

test_images = [
    os.path.join(TEST_IMAGE_PATH, f"{id_code}.png")
    for id_code in test_df["id_code"].tolist()
]



## === cell 5
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

import torchvision.transforms.functional as TF


class RetinopathyPathDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform, size=256, cache_images=True):
        self.image_paths = list(image_paths)
        self.transform = transform
        self.size = size
        self.cache_images = cache_images
        self._cache = {}  # idx -> PIL.Image (RGB, resized)

    def __len__(self):
        return len(self.image_paths)

    def _load_pil_resized(self, im_path):
        img = Image.open(im_path).convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), resample=Image.BILINEAR)
        return img

    def __getitem__(self, idx):
        im_path = self.image_paths[idx]
        if self.cache_images:
            img = self._cache.get(idx)
            if img is None:
                img = self._load_pil_resized(im_path)
                self._cache[idx] = img
        else:
            img = self._load_pil_resized(im_path)
        x = self.transform(img)
        return os.path.splitext(os.path.basename(im_path))[0], x


def _collate_id_x(batch):
    ids, xs = zip(*batch)
    return list(ids), torch.stack(xs, dim=0)


class FastToTensorNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img: Image.Image):
        x = TF.to_tensor(img)  # float32 [0,1], CHW
        x = TF.normalize(x, mean=self.mean, std=self.std)
        return x


def make_predictions(
    model, test_images, transforms_fn, size=256, device=device, batch_size=32
):
    ds = RetinopathyPathDataset(
        test_images, transforms_fn, size=size, cache_images=True
    )
    num_workers = min(4, (os.cpu_count() or 2))
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_collate_id_x,
    )

    predictions = []
    model.eval()
    with torch.inference_mode():
        for ids, xb in loader:
            xb = xb.to(device, non_blocking=True)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            out = model(xb)
            out_flip = model(torch.flip(xb, dims=(3,)))
            final = ((out.view(-1) + out_flip.view(-1)) * 0.5).detach().cpu().numpy()
            predictions.extend(list(zip(ids, final.astype(np.float32).tolist())))
    return predictions


def _clean_state_dict(sd):
    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and isinstance(sd["state_dict"], dict)
    ):
        sd = sd["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        new_sd[nk] = v
    return new_sd


def load_model_weights(model, model_path, device=device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    ckpt = torch.load(model_path, map_location=device)
    ckpt = _clean_state_dict(ckpt)
    model.load_state_dict(ckpt, strict=True)
    return model


def find_checkpoint(prefer_substrings, roots=("/kaggle/input", "/kaggle/data")):
    candidates = []
    for root in roots:
        if root and os.path.exists(root):
            candidates.extend(glob(os.path.join(root, "**", "*.pth"), recursive=True))
    if not candidates:
        return None
    lower = [(p, p.lower()) for p in candidates]
    for s in prefer_substrings:
        s = s.lower()
        hits = [p for (p, pl) in lower if s in pl]
        if hits:
            return sorted(hits)[0]
    return sorted(candidates)[0]


class RetinopathyTrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, images_root, transform, size=256, cache_images=True):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.transform = transform
        self.size = size
        self.cache_images = cache_images
        self._cache = {}  # idx -> PIL.Image

    def __len__(self):
        return len(self.df)

    def _load_pil_resized(self, im_path):
        img = Image.open(im_path).convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), resample=Image.BILINEAR)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im_path = os.path.join(self.images_root, f"{row['id_code']}.png")
        if self.cache_images:
            img = self._cache.get(idx)
            if img is None:
                img = self._load_pil_resized(im_path)
                self._cache[idx] = img
        else:
            img = self._load_pil_resized(im_path)

        x = self.transform(img)
        y = torch.tensor(float(row["diagnosis"]), dtype=torch.float32)
        return x, y


def train_regressor_mse(
    model, train_df, images_root, transform, size=256, device=device
):
    torch.manual_seed(0)
    np.random.seed(0)

    ds = RetinopathyTrainDataset(
        train_df, images_root, transform, size=size, cache_images=True
    )
    num_workers = min(4, (os.cpu_count() or 2))
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=16,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    model.train()
    model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    for _epoch in range(1):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True).view(-1, 1)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


def apply_thresholds(preds, thresholds):
    t0, t1, t2, t3 = thresholds
    preds = np.asarray(preds, dtype=np.float32)
    out = np.zeros_like(preds, dtype=np.int64)
    out[preds >= t0] = 1
    out[preds >= t1] = 2
    out[preds >= t2] = 3
    out[preds >= t3] = 4
    return out


def _zscore_fit(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd < 1e-6:
        sd = 1.0
    return mu, sd


def _zscore_apply(x: np.ndarray, mu: float, sd: float):
    x = np.asarray(x, dtype=np.float32)
    return (x - mu) / sd


def optimize_thresholds_qwk(y_true, y_pred_cont, seed=0, n_iter=6):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float32)

    init_t = np.array([0.75, 1.5, 2.5, 3.5], dtype=np.float32)

    def softplus(x):
        x = np.asarray(x, dtype=np.float64)
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    def unpack(params):
        t0 = float(params[0])
        d = softplus(params[1:])  # positive
        t1 = t0 + float(d[0])
        t2 = t1 + float(d[1])
        t3 = t2 + float(d[2])
        return np.array([t0, t1, t2, t3], dtype=np.float32)

    def objective(params):
        t = unpack(params)
        pred_cls = apply_thresholds(y_pred_cont, t)
        return -cohen_kappa_score(y_true, pred_cls, weights="quadratic")

    d0 = float(init_t[1] - init_t[0])
    d1 = float(init_t[2] - init_t[1])
    d2 = float(init_t[3] - init_t[2])

    def inv_softplus(y):
        y = float(max(y, 1e-6))
        return y if y > 20 else np.log(np.expm1(y))

    x0 = np.array(
        [init_t[0], inv_softplus(d0), inv_softplus(d1), inv_softplus(d2)],
        dtype=np.float64,
    )

    x0 = x0 + rng.normal(scale=0.05, size=x0.shape).astype(np.float64)

    def nelder_mead(f, x_start, step=0.15, max_iter=90, tol=1e-5):
        n = len(x_start)
        simplex = [x_start]
        for i in range(n):
            x = x_start.copy()
            x[i] += step
            simplex.append(x)
        simplex = np.array(simplex, dtype=np.float64)
        fvals = np.array([f(x) for x in simplex], dtype=np.float64)

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        for _ in range(max_iter):
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            if np.std(fvals) < tol:
                break

            x_best = simplex[0]
            x_worst = simplex[-1]
            x_centroid = np.mean(simplex[:-1], axis=0)

            x_r = x_centroid + alpha * (x_centroid - x_worst)
            f_r = f(x_r)

            if fvals[0] <= f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
                continue

            if f_r < fvals[0]:
                x_e = x_centroid + gamma * (x_r - x_centroid)
                f_e = f(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                continue

            x_c = x_centroid + rho * (x_worst - x_centroid)
            f_c = f(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
                continue

            for i in range(1, len(simplex)):
                simplex[i] = x_best + sigma * (simplex[i] - x_best)
                fvals[i] = f(simplex[i])

        best = simplex[np.argmin(fvals)]
        return best, float(np.min(fvals))

    best_params, best_obj = nelder_mead(objective, x0, step=0.2, max_iter=80, tol=1e-5)
    best_t = unpack(best_params)
    best_score = -best_obj
    return best_t.astype(np.float32), best_score


def train_regressor_mse_head_only(
    model, train_df, images_root, transform, size=256, device=device
):
    torch.manual_seed(0)
    np.random.seed(0)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.last_linear.parameters():
        p.requires_grad = True

    ds = RetinopathyTrainDataset(
        train_df, images_root, transform, size=size, cache_images=True
    )
    num_workers = min(4, (os.cpu_count() or 2))
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    model.train()
    model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.Adam(model.last_linear.parameters(), lr=3e-4)
    loss_fn = torch.nn.MSELoss()

    for _epoch in range(1):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True).view(-1, 1)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model




## === cell 6
import hashlib


def _ckpt_id(path_or_none: str):
    if path_or_none is None:
        return "none"
    try:
        st = os.stat(path_or_none)
        key = f"{os.path.basename(path_or_none)}_{st.st_size}_{int(st.st_mtime)}"
    except Exception:
        key = os.path.basename(path_or_none)
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:10]


def get_oof_predictions_cached(
    model_builder_fn,
    df,
    images_root,
    transform,
    model_tag,
    init_state_dict=None,
    init_ckpt_path=None,
    size=256,
    device=device,
    n_splits=3,
    seed=0,
):
    cache_path = (
        f"oof_{model_tag}_s{n_splits}_seed{seed}_ckpt{_ckpt_id(init_ckpt_path)}.csv"
    )
    if os.path.exists(cache_path):
        oof = pd.read_csv(cache_path)
        if set(["id_code", "pred"]).issubset(oof.columns) and len(oof) == len(df):
            return oof

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = df["diagnosis"].values.astype(int)
    oof_pred = np.zeros(len(df), dtype=np.float32)

    df_idx = df[["id_code"]].copy()
    df_idx["__idx__"] = np.arange(len(df_idx), dtype=np.int32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(df)), y), start=1):
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        va_df = df.iloc[va_idx].reset_index(drop=True)

        fold_model = model_builder_fn(pretrain=False).to(device)
        if init_state_dict is not None:
            fold_model.load_state_dict(init_state_dict, strict=True)

        fold_model = train_regressor_mse_head_only(
            fold_model, tr_df, images_root, transform, size=size, device=device
        )

        va_images = [
            os.path.join(images_root, f"{id_code}.png")
            for id_code in va_df["id_code"].tolist()
        ]
        va_preds = make_predictions(
            fold_model, va_images, transform, size=size, device=device, batch_size=32
        )
        va_pred_df = pd.DataFrame(va_preds, columns=["id_code", "pred"]).merge(
            va_df[["id_code"]], on="id_code", how="right"
        )

        if va_pred_df["pred"].isna().any():
            va_pred_df["pred"] = va_pred_df["pred"].astype(np.float32)
            va_pred_df["pred"] = va_pred_df["pred"].fillna(va_pred_df["pred"].mean())

        tmp = df_idx.merge(va_pred_df[["id_code", "pred"]], on="id_code", how="inner")
        oof_pred[tmp["__idx__"].values.astype(np.int64)] = tmp["pred"].values.astype(
            np.float32
        )

        print(f"[INFO] OOF fold {fold}/{n_splits} done for {model_tag}")

    oof = pd.DataFrame({"id_code": df["id_code"].values, "pred": oof_pred})
    oof.to_csv(cache_path, index=False)
    return oof




## === cell 7
MODEL_PATH_DENSENET = "/kaggle/input/densenet121/model_densenet121_bs64_30.pth"
if not os.path.exists(MODEL_PATH_DENSENET):
    alt = find_checkpoint(
        ["densenet121", "densenet"], roots=("/kaggle/input", "/kaggle/data")
    )
    if alt is not None:
        print(f"[INFO] Using discovered DenseNet checkpoint: {alt}")
        MODEL_PATH_DENSENET = alt

predictions_densenet = None
densenet_used_pretrained_ckpt = False
model_dn = None

norm_dn = FastToTensorNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

try:
    model_dn = get_densenet121_gem(pretrain=False)
    model_dn.to(device)
    if device.type == "cuda":
        model_dn = model_dn.to(memory_format=torch.channels_last)

    if os.path.exists(MODEL_PATH_DENSENET):
        load_model_weights(model_dn, MODEL_PATH_DENSENET, device=device)
        densenet_used_pretrained_ckpt = True
    else:
        ok = _try_load_torchvision_imagenet_weights_densenet121(model_dn)
        print(f"[INFO] DenseNet121 torchvision ImageNet init loaded: {ok}")

        print(
            "[INFO] DenseNet checkpoint not found; training DenseNet121+GeM HEAD-ONLY for 1 epoch to produce non-constant predictions."
        )
        model_dn = train_regressor_mse_head_only(
            model_dn, train_df, TRAIN_IMAGE_PATH, norm_dn, size=256, device=device
        )
        torch.save(model_dn.state_dict(), "trained_fallback_densenet121.pth")
        MODEL_PATH_DENSENET = "trained_fallback_densenet121.pth"
        densenet_used_pretrained_ckpt = False

    predictions_densenet = make_predictions(
        model_dn, test_images, norm_dn, size=256, device=device, batch_size=32
    )
except Exception as e:
    print(f"[WARN] Densenet model not used: {e}")



## === cell 8
MODEL_PATH_SERESNET_PSEUDO = "/kaggle/input/seresnet50testpseudo/model30.pth"
if not os.path.exists(MODEL_PATH_SERESNET_PSEUDO):
    alt = find_checkpoint(
        ["seresnet50", "se_resnet50", "seresnet"],
        roots=("/kaggle/input", "/kaggle/data"),
    )
    if alt is not None:
        print(f"[INFO] Using discovered SEResNet checkpoint: {alt}")
        MODEL_PATH_SERESNET_PSEUDO = alt

predictions_seresnet = None
seresnet_used_pretrained_ckpt = False
model_se = None

norm_se = FastToTensorNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

try:
    model_se = get_se_resnet50_gem(pretrain=False)
    model_se.to(device)
    if device.type == "cuda":
        model_se = model_se.to(memory_format=torch.channels_last)

    if os.path.exists(MODEL_PATH_SERESNET_PSEUDO):
        load_model_weights(model_se, MODEL_PATH_SERESNET_PSEUDO, device=device)
        seresnet_used_pretrained_ckpt = True
    else:
        print(
            "[INFO] SEResNet checkpoint not found; training SE-ResNet50+GeM HEAD-ONLY for 1 epoch to produce non-constant predictions."
        )
        model_se = train_regressor_mse_head_only(
            model_se, train_df, TRAIN_IMAGE_PATH, norm_se, size=256, device=device
        )
        torch.save(model_se.state_dict(), "trained_fallback_seresnet50.pth")
        MODEL_PATH_SERESNET_PSEUDO = "trained_fallback_seresnet50.pth"
        seresnet_used_pretrained_ckpt = False

    predictions_seresnet = make_predictions(
        model_se, test_images, norm_se, size=256, device=device, batch_size=32
    )
except Exception as e:
    print(f"[WARN] SEResNet pseudo model not used: {e}")



## === cell 9
if predictions_seresnet is not None and predictions_densenet is not None:
    df_se = pd.DataFrame(predictions_seresnet, columns=["id_code", "pred_se"])
    df_dn = pd.DataFrame(predictions_densenet, columns=["id_code", "pred_dn"])
    pred_df = test_df.merge(df_se, on="id_code", how="left").merge(
        df_dn, on="id_code", how="left"
    )
    if pred_df["pred_se"].isna().any():
        pred_df["pred_se"] = pred_df["pred_se"].astype(np.float32)
        pred_df["pred_se"] = pred_df["pred_se"].fillna(pred_df["pred_se"].mean())
    if pred_df["pred_dn"].isna().any():
        pred_df["pred_dn"] = pred_df["pred_dn"].astype(np.float32)
        pred_df["pred_dn"] = pred_df["pred_dn"].fillna(pred_df["pred_dn"].mean())

    pred_df["pred"] = 0.5 * pred_df["pred_se"].astype(np.float32) + 0.5 * pred_df[
        "pred_dn"
    ].astype(np.float32)
    chosen_name = "ensemble_se_dn"
elif predictions_seresnet is not None:
    pred_df = pd.DataFrame(predictions_seresnet, columns=["id_code", "pred"])
    pred_df = test_df.merge(pred_df, on="id_code", how="left")
    if pred_df["pred"].isna().any():
        pred_df["pred"] = pred_df["pred"].astype(np.float32)
        pred_df["pred"] = pred_df["pred"].fillna(pred_df["pred"].mean())
    chosen_name = "seresnet50_gem"
elif predictions_densenet is not None:
    pred_df = pd.DataFrame(predictions_densenet, columns=["id_code", "pred"])
    pred_df = test_df.merge(pred_df, on="id_code", how="left")
    if pred_df["pred"].isna().any():
        pred_df["pred"] = pred_df["pred"].astype(np.float32)
        pred_df["pred"] = pred_df["pred"].fillna(pred_df["pred"].mean())
    chosen_name = "densenet121_gem"
else:
    print(
        "[WARN] No model predictions available; writing constant predictions (all zeros)."
    )
    pred_df = test_df.copy()
    pred_df["pred"] = 0.0
    chosen_name = "constant_zero"

thresholds = np.array([0.75, 1.5, 2.5, 3.5], dtype=np.float32)

if chosen_name != "constant_zero":
    try:
        y_true = train_df["diagnosis"].values.astype(int)

        if chosen_name == "seresnet50_gem":
            base_model = get_se_resnet50_gem(pretrain=False).to(device)
            if device.type == "cuda":
                base_model = base_model.to(memory_format=torch.channels_last)

            if os.path.exists(MODEL_PATH_SERESNET_PSEUDO):
                load_model_weights(
                    base_model, MODEL_PATH_SERESNET_PSEUDO, device=device
                )
                init_sd = {
                    k: v.detach().cpu() for k, v in base_model.state_dict().items()
                }
            else:
                init_sd = (
                    {k: v.detach().cpu() for k, v in model_se.state_dict().items()}
                    if model_se is not None
                    else None
                )

            oof = get_oof_predictions_cached(
                get_se_resnet50_gem,
                train_df,
                TRAIN_IMAGE_PATH,
                norm_se,
                model_tag="seresnet50_gem_headonly",
                init_state_dict=init_sd,
                init_ckpt_path=(
                    MODEL_PATH_SERESNET_PSEUDO
                    if os.path.exists(MODEL_PATH_SERESNET_PSEUDO)
                    else None
                ),
                size=256,
                device=device,
                n_splits=3,
                seed=0,
            )

            mu, sd = _zscore_fit(oof["pred"].values.astype(np.float32))
            oof_z = _zscore_apply(oof["pred"].values.astype(np.float32), mu, sd)
            test_z = _zscore_apply(pred_df["pred"].values.astype(np.float32), mu, sd)

            best_t, best_qwk = optimize_thresholds_qwk(y_true, oof_z, seed=0, n_iter=6)
            thresholds = best_t
            pred_df["pred"] = test_z.astype(np.float32)
            print(
                f"[INFO] Learned thresholds (OOF/head-only, z-scored) from SEResNet QWK={best_qwk:.5f}: {thresholds.tolist()}"
            )

        elif chosen_name == "densenet121_gem":
            base_model = get_densenet121_gem(pretrain=False).to(device)
            if device.type == "cuda":
                base_model = base_model.to(memory_format=torch.channels_last)

            if os.path.exists(MODEL_PATH_DENSENET):
                load_model_weights(base_model, MODEL_PATH_DENSENET, device=device)
                init_sd = {
                    k: v.detach().cpu() for k, v in base_model.state_dict().items()
                }
            else:
                init_sd = (
                    {k: v.detach().cpu() for k, v in model_dn.state_dict().items()}
                    if model_dn is not None
                    else None
                )

            oof = get_oof_predictions_cached(
                get_densenet121_gem,
                train_df,
                TRAIN_IMAGE_PATH,
                norm_dn,
                model_tag="densenet121_gem_headonly",
                init_state_dict=init_sd,
                init_ckpt_path=(
                    MODEL_PATH_DENSENET if os.path.exists(MODEL_PATH_DENSENET) else None
                ),
                size=256,
                device=device,
                n_splits=3,
                seed=0,
            )

            mu, sd = _zscore_fit(oof["pred"].values.astype(np.float32))
            oof_z = _zscore_apply(oof["pred"].values.astype(np.float32), mu, sd)
            test_z = _zscore_apply(pred_df["pred"].values.astype(np.float32), mu, sd)

            best_t, best_qwk = optimize_thresholds_qwk(y_true, oof_z, seed=0, n_iter=6)
            thresholds = best_t
            pred_df["pred"] = test_z.astype(np.float32)
            print(
                f"[INFO] Learned thresholds (OOF/head-only, z-scored) from DenseNet QWK={best_qwk:.5f}: {thresholds.tolist()}"
            )

        else:
            base_model_se = get_se_resnet50_gem(pretrain=False).to(device)
            if device.type == "cuda":
                base_model_se = base_model_se.to(memory_format=torch.channels_last)

            if os.path.exists(MODEL_PATH_SERESNET_PSEUDO):
                load_model_weights(
                    base_model_se, MODEL_PATH_SERESNET_PSEUDO, device=device
                )
                init_sd_se = {
                    k: v.detach().cpu() for k, v in base_model_se.state_dict().items()
                }
            else:
                init_sd_se = (
                    {k: v.detach().cpu() for k, v in model_se.state_dict().items()}
                    if model_se is not None
                    else None
                )

            base_model_dn = get_densenet121_gem(pretrain=False).to(device)
            if device.type == "cuda":
                base_model_dn = base_model_dn.to(memory_format=torch.channels_last)

            if os.path.exists(MODEL_PATH_DENSENET):
                load_model_weights(base_model_dn, MODEL_PATH_DENSENET, device=device)
                init_sd_dn = {
                    k: v.detach().cpu() for k, v in base_model_dn.state_dict().items()
                }
            else:
                init_sd_dn = (
                    {k: v.detach().cpu() for k, v in model_dn.state_dict().items()}
                    if model_dn is not None
                    else None
                )

            oof_se = get_oof_predictions_cached(
                get_se_resnet50_gem,
                train_df,
                TRAIN_IMAGE_PATH,
                norm_se,
                model_tag="seresnet50_gem_headonly",
                init_state_dict=init_sd_se,
                init_ckpt_path=(
                    MODEL_PATH_SERESNET_PSEUDO
                    if os.path.exists(MODEL_PATH_SERESNET_PSEUDO)
                    else None
                ),
                size=256,
                device=device,
                n_splits=3,
                seed=0,
            )
            oof_dn = get_oof_predictions_cached(
                get_densenet121_gem,
                train_df,
                TRAIN_IMAGE_PATH,
                norm_dn,
                model_tag="densenet121_gem_headonly",
                init_state_dict=init_sd_dn,
                init_ckpt_path=(
                    MODEL_PATH_DENSENET if os.path.exists(MODEL_PATH_DENSENET) else None
                ),
                size=256,
                device=device,
                n_splits=3,
                seed=0,
            )
            oof_ens = oof_se.merge(oof_dn, on="id_code", suffixes=("_se", "_dn"))
            oof_ens["pred"] = 0.5 * oof_ens["pred_se"].astype(
                np.float32
            ) + 0.5 * oof_ens["pred_dn"].astype(np.float32)

            mu, sd = _zscore_fit(oof_ens["pred"].values.astype(np.float32))
            oof_z = _zscore_apply(oof_ens["pred"].values.astype(np.float32), mu, sd)
            test_z = _zscore_apply(pred_df["pred"].values.astype(np.float32), mu, sd)

            best_t, best_qwk = optimize_thresholds_qwk(y_true, oof_z, seed=0, n_iter=6)
            thresholds = best_t
            pred_df["pred"] = test_z.astype(np.float32)
            print(
                f"[INFO] Learned thresholds (OOF/head-only, z-scored) from ensemble QWK={best_qwk:.5f}: {thresholds.tolist()}"
            )
    except Exception as e:
        print(f"[WARN] Threshold learning failed; using default thresholds. Error: {e}")
else:
    print("[INFO] Using default thresholds because predictions are constant fallback.")

pred_df = test_df[["id_code"]].merge(
    pred_df[["id_code", "pred"]], on="id_code", how="left"
)
if pred_df["pred"].isna().any():
    pred_df["pred"] = pred_df["pred"].astype(np.float32)
    pred_df["pred"] = pred_df["pred"].fillna(pred_df["pred"].mean())

assert len(pred_df) == len(test_df), "Prediction row count mismatch with test.csv"
assert (
    pred_df["id_code"].values == test_df["id_code"].values
).all(), "id_code order mismatch vs test.csv"

submission = pd.DataFrame({"id_code": pred_df["id_code"].values})
submission["diagnosis"] = apply_thresholds(pred_df["pred"].values, thresholds).astype(
    np.int64
)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print(
    "[INFO] diagnosis value counts:\n",
    submission["diagnosis"].value_counts().sort_index(),
)

## --- ERROR in outputing the csv:
Invalid submission: Submission must contain the target column 'diagnosis'
