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
import os
import re
import sys
import math
import types
from glob import glob
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TORCH_HAS_CUDA = torch.cuda.is_available()
DEFAULT_NUM_WORKERS = min(8, os.cpu_count() or 2)
PIN_MEMORY = TORCH_HAS_CUDA

Image.MAX_IMAGE_PIXELS = None
_RESIZED_RGB_CACHE = {}  # key: (path, size) -> PIL.Image in RGB already resized


def _load_resize_rgb_cached(path: str, size: int) -> Image.Image:
    key = (path, int(size))
    im = _RESIZED_RGB_CACHE.get(key)
    if im is None:
        im = Image.open(path).convert("RGB")
        im = im.resize((size, size), resample=Image.BILINEAR)
        _RESIZED_RGB_CACHE[key] = im
    return im


def _make_loader(ds, batch_size, shuffle):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(DEFAULT_NUM_WORKERS > 0),
        prefetch_factor=2 if DEFAULT_NUM_WORKERS > 0 else None,
    )


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
    model = models.alexnet(weights=None)
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
    model = models.densenet121(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["densenet121"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet169(num_classes=1000, pretrained="imagenet"):
    model = models.densenet169(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["densenet169"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet201(num_classes=1000, pretrained="imagenet"):
    model = models.densenet201(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["densenet201"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet161(num_classes=1000, pretrained="imagenet"):
    model = models.densenet161(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["densenet161"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def inceptionv3(num_classes=1000, pretrained="imagenet"):
    model = models.inception_v3(weights=None, aux_logits=True)
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


def resnet18(num_classes=1000, pretrained="imagenet"):
    model = models.resnet18(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["resnet18"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet34(num_classes=1000, pretrained="imagenet"):
    model = models.resnet34(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["resnet34"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet50(num_classes=1000, pretrained="imagenet"):
    model = models.resnet50(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["resnet50"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet101(num_classes=1000, pretrained="imagenet"):
    model = models.resnet101(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["resnet101"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet152(num_classes=1000, pretrained="imagenet"):
    model = models.resnet152(weights=None)
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
    model = models.squeezenet1_0(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["squeezenet1_0"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_squeezenets(model)
    return model


def squeezenet1_1(num_classes=1000, pretrained="imagenet"):
    model = models.squeezenet1_1(weights=None)
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
    model = models.vgg11(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["vgg11"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg11_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg11_bn(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["vgg11_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg13(num_classes=1000, pretrained="imagenet"):
    model = models.vgg13(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["vgg13"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg13_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg13_bn(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["vgg13_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg16(num_classes=1000, pretrained="imagenet"):
    model = models.vgg16(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["vgg16"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg16_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg16_bn(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["vgg16_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg19(num_classes=1000, pretrained="imagenet"):
    model = models.vgg19(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["vgg19"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg19_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg19_bn(weights=None)
    if pretrained is not None:
        settings = pretrained_settings["vgg19_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model




## === cell 1
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
    if pretrain:
        model = se_resnet50(num_classes=1000, pretrained=None)
        try:
            from torchvision.models import ResNet50_Weights

            tv = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model.load_state_dict(tv.state_dict(), strict=False)
        except Exception as e:
            print(
                f"Warning: could not load torchvision ResNet50 ImageNet weights ({e}); using random init."
            )
    else:
        model = se_resnet50(num_classes=1000, pretrained=None)

    model.avg_pool = GeM()
    model.last_linear = nn.Linear(2048, 1)
    return model


def get_densenet121_gem(pretrain):
    if pretrain:
        model = densenet121(num_classes=1000, pretrained=None)
        try:
            from torchvision.models import DenseNet121_Weights

            tv = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            model.load_state_dict(tv.state_dict(), strict=False)
        except Exception as e:
            print(
                f"Warning: could not load torchvision DenseNet121 ImageNet weights ({e}); using random init."
            )
    else:
        model = densenet121(num_classes=1000, pretrained=None)

    model.avg_pool = GeM()
    model.last_linear = nn.Linear(1024, 1)
    return model




## === cell 3
DATA_ROOT = "/kaggle/input/aptos2019-blindness-detection"
TEST_IMAGE_PATH = os.path.join(DATA_ROOT, "test_images")
TEST_CSV_PATH = os.path.join(DATA_ROOT, "test.csv")
TRAIN_IMAGE_PATH = os.path.join(DATA_ROOT, "train_images")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

test_df = pd.read_csv(TEST_CSV_PATH)
test_images = [
    os.path.join(TEST_IMAGE_PATH, f"{i}.png") for i in test_df["id_code"].tolist()
]

missing = [p for p in test_images if not os.path.exists(p)]
if len(missing) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images. Example: {missing[0]}"
    )

train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df["path"] = train_df["id_code"].apply(
    lambda x: os.path.join(TRAIN_IMAGE_PATH, f"{x}.png")
)
missing_train = train_df.loc[~train_df["path"].apply(os.path.exists)]
if len(missing_train) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing_train)} train images. Example id: {missing_train.iloc[0]['id_code']}"
    )




## === cell 4
class DRTestDataset(Dataset):
    def __init__(self, image_paths, size, tfm):
        self.image_paths = list(image_paths)
        self.size = int(size)
        self.tfm = tfm

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        im_path = self.image_paths[idx]
        image_id = os.path.splitext(os.path.basename(im_path))[0]
        img = _load_resize_rgb_cached(im_path, self.size)
        x = self.tfm(img)
        return image_id, x


def make_predictions(
    model, test_images, transforms, size=256, device=device, batch_size=32
):
    ds = DRTestDataset(test_images, size=size, tfm=transforms)
    dl = _make_loader(ds, batch_size=batch_size, shuffle=False)

    predictions = []
    model.eval()
    with torch.no_grad():
        for ids, xb in dl:
            xb = xb.to(device, non_blocking=True)
            if TORCH_HAS_CUDA:
                xb = xb.contiguous(memory_format=torch.channels_last)

            xb_flip = torch.flip(xb, dims=(3,))
            xcat = torch.cat([xb, xb_flip], dim=0)
            out = model(xcat).view(-1)
            out0 = out[: xb.size(0)]
            out1 = out[xb.size(0) :]
            final = (out0 + out1) * 0.5
            final_np = final.detach().cpu().numpy()
            predictions.extend(list(zip(list(ids), final_np.tolist())))
    return predictions


class DRTrainPredDataset(Dataset):
    def __init__(self, df, size, tfm):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.loc[idx, "path"]
        image_id = self.df.loc[idx, "id_code"]
        y = int(self.df.loc[idx, "diagnosis"])
        img = _load_resize_rgb_cached(p, self.size)
        x = self.tfm(img)
        return image_id, x, y


def predict_on_df(model, df, tfm, size, device=device, batch_size=32):
    ds = DRTrainPredDataset(df, size=size, tfm=tfm)
    dl = _make_loader(ds, batch_size=batch_size, shuffle=False)
    ids, preds, ys = [], [], []
    model.eval()
    with torch.no_grad():
        for bid, xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            if TORCH_HAS_CUDA:
                xb = xb.contiguous(memory_format=torch.channels_last)

            xb_flip = torch.flip(xb, dims=(3,))
            xcat = torch.cat([xb, xb_flip], dim=0)
            out = model(xcat).view(-1)
            out0 = out[: xb.size(0)]
            out1 = out[xb.size(0) :]
            final = (out0 + out1) * 0.5
            ids.extend(list(bid))
            preds.extend(final.detach().cpu().numpy().tolist())
            ys.extend(list(map(int, yb.numpy().tolist())))
    return (
        np.array(ids),
        np.array(preds, dtype=np.float32),
        np.array(ys, dtype=np.int64),
    )




## === cell 5
def _find_local_checkpoint(prefer_patterns):
    search_roots = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/input",
        "/kaggle/working",
    ]
    candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for pat in prefer_patterns:
            candidates.extend(glob(os.path.join(root, pat)))
            candidates.extend(glob(os.path.join(root, "**", pat), recursive=True))
    candidates = [p for p in candidates if os.path.isfile(p)]
    candidates = sorted(set(candidates), key=lambda p: (len(p), p))
    return candidates[0] if candidates else None


def _extract_state_dict(obj):
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        return obj["state_dict"]
    if (
        isinstance(obj, dict)
        and "model_state_dict" in obj
        and isinstance(obj["model_state_dict"], dict)
    ):
        return obj["model_state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Unsupported checkpoint format")


def _clean_state_keys(state):
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        new_state[nk] = v
    return new_state


def _load_checkpoint_into_model(model, ckpt_path, device=device):
    obj = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(obj)
    state = _clean_state_keys(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


def _load_if_compatible(model, ckpt_path, max_missing_ratio=0.25, device=device):
    missing, unexpected = _load_checkpoint_into_model(model, ckpt_path, device=device)
    total = len(model.state_dict())
    missing_ratio = len(missing) / max(total, 1)
    ok = missing_ratio <= max_missing_ratio
    return ok, missing, unexpected, missing_ratio


class DRTrainDataset(Dataset):
    def __init__(self, df, size, tfm):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.loc[idx, "path"]
        y = float(self.df.loc[idx, "diagnosis"])
        img = _load_resize_rgb_cached(p, self.size)
        x = self.tfm(img)
        return x, torch.tensor([y], dtype=torch.float32)


def _set_requires_grad(model, trainable_names):
    for n, p in model.named_parameters():
        p.requires_grad = any(tn in n for tn in trainable_names)


def train_regression_head(
    model, train_df, size, tfm, device=device, epochs=2, batch_size=16, lr=2e-3
):
    model.train()

    _set_requires_grad(model, trainable_names=["last_linear", "avg_pool.p"])

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable parameters selected for head training.")

    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.MSELoss()

    ds = DRTrainDataset(train_df, size=size, tfm=tfm)
    dl = _make_loader(ds, batch_size=batch_size, shuffle=True)

    for ep in range(epochs):
        running = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if TORCH_HAS_CUDA:
                xb = xb.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)
            n += xb.size(0)
        print(
            f"Head-train epoch {ep+1}/{epochs} @size={size}: mse={running/max(n,1):.5f}"
        )

    model.eval()
    return model


norm_imagenet = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model_dn = get_densenet121_gem(pretrain=True).to(device).eval()
model_rn = get_se_resnet50_gem(pretrain=True).to(device).eval()
if TORCH_HAS_CUDA:
    model_dn = model_dn.to(memory_format=torch.channels_last)
    model_rn = model_rn.to(memory_format=torch.channels_last)

ckpt_dn = _find_local_checkpoint(
    [
        "*densenet121*gem*.pth",
        "*densenet121*gem*.pt",
        "*densenet121*fold*.pth",
        "*densenet121*fold*.pt",
        "*densenet121*.pth",
        "*densenet121*.pt",
        "*dense121*.pth",
        "*dense121*.pt",
        "*dense*gem*.pth",
        "*dense*gem*.pt",
    ]
)
dn_loaded = False
if ckpt_dn is not None:
    ok, missing, unexpected, mr = _load_if_compatible(model_dn, ckpt_dn, device=device)
    if ok:
        dn_loaded = True
        print(
            f"Loaded DenseNet121+GeM checkpoint: {ckpt_dn} (missing={len(missing)}/{len(model_dn.state_dict())}={mr:.2%}, unexpected={len(unexpected)})"
        )
    else:
        print(
            f"Found DenseNet checkpoint but incompatible (missing={len(missing)}/{len(model_dn.state_dict())}={mr:.2%}); will train head locally."
        )
else:
    print("No local DenseNet checkpoint found; will train head locally.")

ckpt_rn = _find_local_checkpoint(
    [
        "*se_resnet50*gem*.pth",
        "*se_resnet50*gem*.pt",
        "*seresnet50*gem*.pth",
        "*seresnet50*gem*.pt",
        "*se_resnet50*fold*.pth",
        "*se_resnet50*fold*.pt",
        "*seresnet*fold*.pth",
        "*seresnet*fold*.pt",
        "*se_resnet50*.pth",
        "*se_resnet50*.pt",
        "*seresnet*.pth",
        "*seresnet*.pt",
        "*resnet50*gem*.pth",
        "*resnet50*gem*.pt",
    ]
)
rn_loaded = False
if ckpt_rn is not None:
    ok, missing, unexpected, mr = _load_if_compatible(model_rn, ckpt_rn, device=device)
    if ok:
        rn_loaded = True
        print(
            f"Loaded SE-ResNet50+GeM checkpoint: {ckpt_rn} (missing={len(missing)}/{len(model_rn.state_dict())}={mr:.2%}, unexpected={len(unexpected)})"
        )
    else:
        print(
            f"Found SE-ResNet checkpoint but incompatible (missing={len(missing)}/{len(model_rn.state_dict())}={mr:.2%}); will train head locally."
        )
else:
    print("No local SE-ResNet checkpoint found; will train head locally.")

if not dn_loaded:
    model_dn = train_regression_head(
        model_dn,
        train_df,
        size=224,
        tfm=norm_imagenet,
        device=device,
        epochs=2,
        batch_size=16,
        lr=2e-3,
    )
if not rn_loaded:
    model_rn = train_regression_head(
        model_rn,
        train_df,
        size=256,
        tfm=norm_imagenet,
        device=device,
        epochs=2,
        batch_size=12,
        lr=2e-3,
    )

predictions_densenet = make_predictions(
    model_dn,
    test_images,
    norm_imagenet,
    size=224,
    device=device,
    batch_size=48 if TORCH_HAS_CUDA else 16,
)
predictions_seresnet = make_predictions(
    model_rn,
    test_images,
    norm_imagenet,
    size=256,
    device=device,
    batch_size=32 if TORCH_HAS_CUDA else 12,
)
predictions_seresnet_512 = make_predictions(
    model_rn,
    test_images,
    norm_imagenet,
    size=512,
    device=device,
    batch_size=12 if TORCH_HAS_CUDA else 4,
)

print(
    "Pred lengths:",
    len(predictions_densenet),
    len(predictions_seresnet),
    len(predictions_seresnet_512),
)




## === cell 6
def qwk(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    idx = y_true * n_classes + y_pred
    O = (
        np.bincount(idx, minlength=n_classes * n_classes)
        .reshape(n_classes, n_classes)
        .astype(np.float64)
    )

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E / E.sum() * O.sum()

    i = np.arange(n_classes, dtype=np.float64)
    W = (i[:, None] - i[None, :]) ** 2 / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den > 0 else 0.0


def apply_thresholds(preds, thr):
    t0, t1, t2, t3 = thr
    out = np.zeros_like(preds, dtype=np.int64)
    out[preds >= t0] = 1
    out[preds >= t1] = 2
    out[preds >= t2] = 3
    out[preds >= t3] = 4
    return out


def optimize_thresholds(y_true, preds, init_thr=(0.75, 1.5, 2.5, 3.4), n_iter=20):
    y_true = np.asarray(y_true, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.float64)

    thr = np.array(init_thr, dtype=np.float64)
    thr = np.sort(thr)

    best = qwk(y_true, apply_thresholds(preds, thr))
    step = 0.5
    for _ in range(n_iter):
        improved = False
        for k in range(4):
            for direction in (-1.0, 1.0):
                cand = thr.copy()
                cand[k] += direction * step
                cand = np.sort(cand)
                cand = np.clip(cand, -1.0, 5.0)
                score = qwk(y_true, apply_thresholds(preds, cand))
                if score > best + 1e-8:
                    thr, best = cand, score
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-3:
                break
    return tuple(thr.tolist()), best


def stratified_folds(df, n_splits=3, seed=42):
    rng = np.random.RandomState(seed)
    y = df["diagnosis"].values.astype(int)
    idx_by_class = {c: np.where(y == c)[0].tolist() for c in range(5)}
    for c in idx_by_class:
        rng.shuffle(idx_by_class[c])
    folds = [[] for _ in range(n_splits)]
    for c, idxs in idx_by_class.items():
        for i, ix in enumerate(idxs):
            folds[i % n_splits].append(ix)
    return [np.array(sorted(f), dtype=np.int64) for f in folds]


folds = stratified_folds(train_df, n_splits=3, seed=42)
oof_pred = np.zeros(len(train_df), dtype=np.float32)
oof_true = train_df["diagnosis"].values.astype(int)

for fi, val_idx in enumerate(folds):
    tr_idx = np.setdiff1d(np.arange(len(train_df)), val_idx)
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[val_idx].reset_index(drop=True)

    m = get_se_resnet50_gem(pretrain=True).to(device).eval()
    if TORCH_HAS_CUDA:
        m = m.to(memory_format=torch.channels_last)

    if ckpt_rn is not None:
        ok, missing, unexpected, mr = _load_if_compatible(m, ckpt_rn, device=device)
        if not ok:
            m = train_regression_head(
                m,
                tr_df,
                size=256,
                tfm=norm_imagenet,
                device=device,
                epochs=2,
                batch_size=12,
                lr=2e-3,
            )
    else:
        m = train_regression_head(
            m,
            tr_df,
            size=256,
            tfm=norm_imagenet,
            device=device,
            epochs=2,
            batch_size=12,
            lr=2e-3,
        )

    _, preds_va, y_va = predict_on_df(
        m,
        va_df,
        tfm=norm_imagenet,
        size=256,
        device=device,
        batch_size=32 if TORCH_HAS_CUDA else 12,
    )
    oof_pred[val_idx] = preds_va.astype(np.float32)

    fold_qwk = qwk(y_va, apply_thresholds(preds_va, (0.75, 1.5, 2.5, 3.4)))
    print(f"Fold {fi+1}/{len(folds)}: baseline-thr QWK={fold_qwk:.5f}")

opt_thr, opt_qwk = optimize_thresholds(
    oof_true, oof_pred, init_thr=(0.75, 1.5, 2.5, 3.4), n_iter=25
)
print(f"Optimized thresholds from OOF: {opt_thr} with OOF QWK={opt_qwk:.5f}")



## === cell 7
final_predictions = []
for x, y, z in zip(
    predictions_densenet, predictions_seresnet, predictions_seresnet_512
):
    if not (x[0] == y[0] == z[0]):
        raise ValueError(f"ID mismatch: {x[0]}, {y[0]}, {z[0]}")
    final_predictions.append([x[0], (x[1] + y[1] + z[1]) / 3.0])

submission = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis"])

preds = submission["diagnosis"].values.astype(np.float64)
submission["diagnosis"] = apply_thresholds(preds, opt_thr).astype(int)

pred_map = dict(zip(submission["id_code"].tolist(), submission["diagnosis"].tolist()))
missing_ids = [i for i in test_df["id_code"].tolist() if i not in pred_map]
if missing_ids:
    raise KeyError(
        f"Missing predictions for {len(missing_ids)} ids. Example: {missing_ids[0]}"
    )

submission = pd.DataFrame(
    {
        "id_code": test_df["id_code"].values,
        "diagnosis": [pred_map[i] for i in test_df["id_code"].values],
    }
)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print(f"Wrote {out_path} with shape {submission.shape}")
print(submission.head())
