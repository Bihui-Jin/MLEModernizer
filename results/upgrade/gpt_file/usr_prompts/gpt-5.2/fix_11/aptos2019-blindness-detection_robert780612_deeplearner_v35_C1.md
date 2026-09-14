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

0.9202652974471286

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the non-code prose and stray markdown backticks that are currently causing `SyntaxError` so the notebook runs as pure Python. I also fix `from __future__` imports (they must be at the very top of the file) by removing them since they’re unnecessary in Python 3.9, which unblocks execution without changing the modeling logic. Then I make the test image list deterministically aligned to `test.csv` `id_code` ordering (instead of relying on filesystem glob order), preventing silent ID/prediction misalignment that would tank kappa. Finally, I keep your checkpoint-missing fallback, but ensure it always writes a valid `submission.csv` with the exact required columns and correct row count.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a “works but predicts the same label for almost everything” situation: your checkpoints likely aren’t being found, so you fall back to `sample_submission` (all zeros), and/or the model outputs are being discretized with thresholds that aren’t calibrated to your model’s regression scale. I make two minimal, score-relevant fixes: (1) robustly locate the three checkpoint files under `/kaggle/input/**` (without changing which checkpoints you use), so inference actually runs; and (2) replace the hard-coded cutpoints with an automatic, label-distribution-matching discretization (rank-based) when producing the required integer classes, which is a common safe improvement for QWK with regression outputs. Core model definitions, inference loop, and ensembling stay the same; we just ensure weights load and discretization better matches the metric.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with your script falling back to `sample_submission` (all zeros) because none of the three checkpoints are actually found/loaded, and/or using the wrong `/kaggle/input/...` base path. I make the smallest score-relevant change: robustly search for the checkpoint files by filename across `/kaggle/input/**` and also accept `.pt/.bin` variants, so your existing inference logic actually runs. I also ensure test image paths always follow `test.csv` order (never silently switching to glob order), because misalignment can destroy kappa even if predictions are good. Core model definitions, inference loop, and ensembling are unchanged; we just ensure weights load and IDs stay aligned so the score moves toward your 0.920 target.'
- What this solution (achieved 0.0) has done: 'I fix the `KeyError: 'densenet121'` by separating the two conflicting `pretrained_settings` dictionaries (one for the generic models and one for SENet) so DenseNet looks up its own settings again. Then I make the checkpoint-missing logic score-safe and execution-safe by always running inference (ImageNet weights if checkpoints aren’t found), and by guarding the ensembling step so it never crashes with “No objects to concatenate”. Finally, I keep your distribution-matching discretization and ensure a valid `submission.csv` is always written with correct `test.csv` ordering and required columns.'
- What this solution (achieved 0.0) has done: 'I fix the SSL/download crash by preventing any internet weight downloads and instead relying on locally available checkpoints; if no checkpoints are found, the code still run by initializing the same architectures with random weights (score be lower, but it produce a valid submission instead of crashing). I also only add models to the ensemble if their checkpoint actually loaded successfully, so we avoid ensembling random/untrained nets that would hurt QWK and move the score away from your target. Finally, I fix the `NameError` by defining `loaded_flags/loaded_success` robustly even if a previous cell fails, and ensure the submission is always aligned to `test.csv` ordering and written as `submission.csv` with required columns.'
- What this solution (achieved 0.11308) has done: 'Your 0.0 score is most consistent with “valid CSV but predictions effectively random/all-same” because inference is running with random weights: the `pretrain` flag is inverted (you disable ImageNet pretraining when checkpoints are missing). I make the smallest score-relevant fix by correctly using ImageNet weights as the fallback only when a checkpoint is not found/loaded, without changing your architectures or inference loop. I also ensure we don’t try to download weights (Kaggle internet-off) by using torchvision’s built-in weights when available and otherwise keeping the existing random fallback; this keeps execution stable while improving predictions toward the 0.920 target. Everything else (test ordering, TTA, ensembling, distribution-matching discretization, submission writing) is kept intact.'
- What this solution (achieved 0.0) has done: 'Your score (0.113) is far below the target (0.920), so we need a real inference-quality fix with minimal semantic change. The biggest score-killer in your current pipeline is that your DenseNet branch is missing ImageNet normalization (it only uses `ToTensor()`), which makes its outputs poorly calibrated and drags down the ensemble heavily. I change DenseNet preprocessing to use the correct ImageNet mean/std (matching your other models) and add a tiny “only ensemble models that actually loaded a checkpoint” guard to avoid mixing in random/untrusted weights if a checkpoint isn’t present (keeping your same architectures and inference loop). Everything else (test ordering, TTA, ensembling-by-mean, distribution-matching discretization, submission format) stays the same.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score strongly suggests you’re still falling into the “no checkpoints loaded → fallback submission (all zeros)” path, even though inference code exists. I make the checkpoint discovery more robust by also searching under `/kaggle/input/aptos2019-blindness-detection/**` and by accepting common checkpoint naming variants, so your existing ensemble actually runs. I also make `_safe_load_state_dict` handle common wrapper formats (e.g., `{"model": state_dict}`, `{"model_state_dict": ...}`) so a found file actually loads, without changing any architecture or inference logic. These are minimal, score-relevant changes intended to move the score upward toward your 0.920 target.'

# 9. Code solution

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

import torchvision.models as models
from torchvision import transforms

from PIL import Image, ImageFile



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

pretrained_settings_generic = {}

for model_name in __all__:
    pretrained_settings_generic[model_name] = {
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
    try:
        state_dict = model_zoo.load_url(settings["url"])
        state_dict = update_state_dict(state_dict)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(
            f"[WARN] Could not download pretrained weights from {settings['url']}: {type(e).__name__}: {e}"
        )
        print("[WARN] Continuing with randomly initialized weights for this backbone.")
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
        settings = pretrained_settings_generic["alexnet"][pretrained]
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
        settings = pretrained_settings_generic["densenet121"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet169(num_classes=1000, pretrained="imagenet"):
    model = models.densenet169(weights=None)
    if pretrained is not None:
        settings = pretrained_settings_generic["densenet169"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet201(num_classes=1000, pretrained="imagenet"):
    model = models.densenet201(weights=None)
    if pretrained is not None:
        settings = pretrained_settings_generic["densenet201"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def densenet161(num_classes=1000, pretrained="imagenet"):
    model = models.densenet161(weights=None)
    if pretrained is not None:
        settings = pretrained_settings_generic["densenet161"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_densenets(model)
    return model


def inceptionv3(num_classes=1000, pretrained="imagenet"):
    model = models.inception_v3(weights=None, aux_logits=True)
    if pretrained is not None:
        settings = pretrained_settings_generic["inceptionv3"][pretrained]
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
        settings = pretrained_settings_generic["resnet18"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet34(num_classes=1000, pretrained="imagenet"):
    model = models.resnet34(weights=None)
    if pretrained is not None:
        settings = pretrained_settings_generic["resnet34"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet50(num_classes=1000, pretrained="imagenet"):
    model = models.resnet50(weights=None)
    if pretrained is not None:
        settings = pretrained_settings_generic["resnet50"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet101(num_classes=1000, pretrained="imagenet"):
    model = models.resnet101(weights=None)
    if pretrained is not None:
        settings = pretrained_settings_generic["resnet101"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_resnets(model)
    return model


def resnet152(num_classes=1000, pretrained="imagenet"):
    model = models.resnet152(weights=None)
    if pretrained is not None:
        settings = pretrained_settings_generic["resnet152"][pretrained]
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
        settings = pretrained_settings_generic["squeezenet1_0"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_squeezenets(model)
    return model


def squeezenet1_1(num_classes=1000, pretrained="imagenet"):
    model = models.squeezenet1_1(weights=None)
    if pretrained is not None:
        settings = pretrained_settings_generic["squeezenet1_1"][pretrained]
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
        settings = pretrained_settings_generic["vgg11"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg11_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg11_bn(weights=None)
    if pretrained is not None:
        settings = pretrained_settings_generic["vgg11_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg13(num_classes=1000, pretrained="imagenet"):
    model = models.vgg13(weights=None)
    if pretrained is not None:
        settings = pretrained_settings_generic["vgg13"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg13_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg13_bn(weights=None)
    if pretrained is not None:
        settings = pretrained_settings_generic["vgg13_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg16(num_classes=1000, pretrained="imagenet"):
    model = models.vgg16(weights=None)
    if pretrained is not None:
        settings = pretrained_settings_generic["vgg16"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg16_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg16_bn(weights=None)
    if pretrained is not None:
        settings = pretrained_settings_generic["vgg16_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg19(num_classes=1000, pretrained="imagenet"):
    model = models.vgg19(weights=None)
    if pretrained is not None:
        settings = pretrained_settings_generic["vgg19"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model


def vgg19_bn(num_classes=1000, pretrained="imagenet"):
    model = models.vgg19_bn(weights=None)
    if pretrained is not None:
        settings = pretrained_settings_generic["vgg19_bn"][pretrained]
        model = load_pretrained(model, num_classes, settings)
    model = modify_vggs(model)
    return model




## === cell 2
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
        base_width=4,
        inplanes=None,
        planes=None,
        groups=None,
        reduction=None,
        stride=1,
        downsample=None,
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
        model.load_state_dict(model_zoo.load_url(settings["url"]))
    except Exception as e:
        print(
            f"[WARN] Could not download pretrained SENet weights from {settings['url']}: {type(e).__name__}: {e}"
        )
        print("[WARN] Continuing with randomly initialized weights for this backbone.")
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
sys.path.append("/kaggle/working/")


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


def get_densenet121_gem(pretrain):
    if pretrain == "imagenet":
        model = densenet121(num_classes=1000, pretrained="imagenet")
    else:
        model = densenet121(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(1024, 1)
    return model




## === cell 4
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
test_df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/test.csv")
test_ids = test_df["id_code"].tolist()
test_images = [os.path.join(TEST_IMAGE_PATH, f"{i}.png") for i in test_ids]

missing_imgs = [p for p in test_images if not os.path.exists(p)]
if len(missing_imgs) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing_imgs)} test images under {TEST_IMAGE_PATH}. Example: {missing_imgs[0]}"
    )




## === cell 5
def make_predictions(
    model, test_images, transforms, size=256, device=torch.device("cpu")
):
    predictions = []
    model.eval()
    with torch.no_grad():
        for im_path in test_images:
            image = Image.open(im_path).convert("RGB")
            image = image.resize((size, size), resample=Image.BILINEAR)

            image_t = transforms(image).to(device)
            out = model(image_t.unsqueeze(0))
            out_flip = model(torch.flip(image_t.unsqueeze(0), dims=(3,)))

            final_prediction = (
                float(out.squeeze().item()) + float(out_flip.squeeze().item())
            ) / 2.0
            predictions.append(
                (os.path.splitext(os.path.basename(im_path))[0], final_prediction)
            )
    return predictions


def _safe_load_state_dict(model, path, map_location):
    if not os.path.exists(path):
        return False
    ckpt = torch.load(path, map_location=map_location)

    state = None
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        if state is None:
            if len(ckpt) > 0 and all(isinstance(k, str) for k in ckpt.keys()):
                state = ckpt
    elif isinstance(ckpt, OrderedDict):
        state = ckpt

    if state is None or not isinstance(state, dict) or len(state) == 0:
        return False

    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_state[nk] = v

    model.load_state_dict(new_state, strict=False)
    return True


def _preds_to_series(preds, index_ids):
    d = dict(preds)
    return pd.Series([d[i] for i in index_ids], index=index_ids, dtype=float)


def _resolve_ckpt_path(preferred_path: str, filename_hint: str):
    if os.path.exists(preferred_path):
        return preferred_path

    base = os.path.basename(filename_hint)
    stem, ext = os.path.splitext(base)
    hints = {base}
    if ext.lower() == ".pth":
        hints.update({stem + ".pt", stem + ".bin"})
    elif ext.lower() == ".pt":
        hints.update({stem + ".pth", stem + ".bin"})
    else:
        hints.update({stem + ".pth", stem + ".pt"})

    candidates = []
    search_roots = [
        "/kaggle/input",
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/working",
    ]
    for root in search_roots:
        for h in sorted(hints):
            candidates.extend(glob(f"{root}/**/{h}", recursive=True))

    if len(candidates) == 0:
        return preferred_path
    candidates = sorted(set(candidates), key=lambda p: (len(p), p))
    return candidates[0]


def _apply_distribution_matching_bins(
    pred_values: pd.Series, train_labels: pd.Series, n_classes: int = 5
) -> pd.Series:
    train_counts = (
        train_labels.value_counts()
        .reindex(range(n_classes), fill_value=0)
        .astype(int)
        .values
    )
    total = int(train_counts.sum())
    if total <= 0:
        return pred_values.fillna(0).round().clip(0, n_classes - 1).astype(int)

    proportions = train_counts / total
    desired = np.floor(proportions * len(pred_values)).astype(int)
    remainder = len(pred_values) - int(desired.sum())
    if remainder > 0:
        fracs = (proportions * len(pred_values)) - np.floor(
            proportions * len(pred_values)
        )
        order = np.argsort(-fracs)
        for i in range(remainder):
            desired[order[i % n_classes]] += 1

    s = pred_values.copy()
    s = s.fillna(s.median() if np.isfinite(s.median()) else 0.0)
    sorted_idx = s.sort_values(kind="mergesort").index
    out = pd.Series(index=s.index, dtype=int)

    start = 0
    for cls in range(n_classes):
        cnt = int(desired[cls])
        if cnt <= 0:
            continue
        cls_idx = sorted_idx[start : start + cnt]
        out.loc[cls_idx] = cls
        start += cnt

    out = out.fillna(n_classes - 1).astype(int)
    return out




## === cell 6
available_models = []

MODEL_PATH_DENSENET = "../input/densenet121/model_densenet121_bs64_30.pth"
MODEL_PATH_SERESNET_PSEUDO = "../input/seresnet50testpseudo/model10.pth"
MODEL_PATH_SERESNET_512 = "../input/seresnet50-512/model30_512.pth"

MODEL_PATH_DENSENET = _resolve_ckpt_path(
    MODEL_PATH_DENSENET, "model_densenet121_bs64_30.pth"
)
MODEL_PATH_SERESNET_PSEUDO = _resolve_ckpt_path(
    MODEL_PATH_SERESNET_PSEUDO, "model10.pth"
)
MODEL_PATH_SERESNET_512 = _resolve_ckpt_path(MODEL_PATH_SERESNET_512, "model30_512.pth")

loaded_flags = {
    "densenet": os.path.exists(MODEL_PATH_DENSENET),
    "seresnet_pseudo": os.path.exists(MODEL_PATH_SERESNET_PSEUDO),
    "seresnet_512": os.path.exists(MODEL_PATH_SERESNET_512),
}
loaded_success = {"densenet": False, "seresnet_pseudo": False, "seresnet_512": False}

model = get_densenet121_gem(
    pretrain=("imagenet" if not loaded_flags["densenet"] else None)
)
model.to(device)
loaded = _safe_load_state_dict(model, MODEL_PATH_DENSENET, map_location=device)
loaded_success["densenet"] = bool(loaded)
norm_densenet = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
predictions_densenet = make_predictions(
    model, test_images, norm_densenet, size=224, device=device
)

if loaded_success["densenet"]:
    available_models.append(("densenet", predictions_densenet))

model = get_se_resnet50_gem(
    pretrain=("imagenet" if not loaded_flags["seresnet_pseudo"] else None)
)
model.to(device)
loaded_pseudo = _safe_load_state_dict(
    model, MODEL_PATH_SERESNET_PSEUDO, map_location=device
)
loaded_success["seresnet_pseudo"] = bool(loaded_pseudo)
norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
predictions_seresnet = make_predictions(
    model, test_images, norm, size=256, device=device
)
if loaded_success["seresnet_pseudo"]:
    available_models.append(("seresnet256", predictions_seresnet))

model = get_se_resnet50_gem(
    pretrain=("imagenet" if not loaded_flags["seresnet_512"] else None)
)
model.to(device)
loaded_512 = _safe_load_state_dict(model, MODEL_PATH_SERESNET_512, map_location=device)
loaded_success["seresnet_512"] = bool(loaded_512)
predictions_seresnet_512 = make_predictions(
    model, test_images, norm, size=512, device=device
)
if loaded_success["seresnet_512"]:
    available_models.append(("seresnet512", predictions_seresnet_512))



## === cell 7
if len(available_models) == 0:
    sample = pd.read_csv(
        "/kaggle/input/aptos2019-blindness-detection/sample_submission.csv"
    )
    submission = sample.set_index("id_code").reindex(test_ids).reset_index()
    submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int).clip(0, 4)
else:
    pred_series = []
    for _, preds in available_models:
        pred_series.append(_preds_to_series(preds, test_ids))
    avg_pred = pd.concat(pred_series, axis=1).mean(axis=1)
    submission = pd.DataFrame({"id_code": avg_pred.index, "diagnosis": avg_pred.values})
    submission = submission.set_index("id_code").reindex(test_ids).reset_index()

    train_df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
    train_labels = train_df["diagnosis"].astype(int)

    if not pd.api.types.is_integer_dtype(submission["diagnosis"]):
        submission["diagnosis"] = _apply_distribution_matching_bins(
            submission["diagnosis"].astype(float),
            train_labels=train_labels,
            n_classes=5,
        )
    else:
        submission["diagnosis"] = submission["diagnosis"].astype(int)

    submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int).clip(0, 4)

submission = submission[["id_code", "diagnosis"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print(
    "Resolved checkpoint paths:",
    {
        "densenet": MODEL_PATH_DENSENET,
        "seresnet_pseudo": MODEL_PATH_SERESNET_PSEUDO,
        "seresnet_512": MODEL_PATH_SERESNET_512,
    },
)
print("Checkpoint exists:", loaded_flags)
print("Checkpoint loaded successfully:", loaded_success)
print("Loaded models for ensemble:", [name for name, _ in available_models])
print("Wrote submission.csv with shape:", submission.shape)
