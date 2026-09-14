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

0.906439070074833

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the non-Python prose that was accidentally included as executable code (and the stray Markdown fence) since those are the direct causes of the SyntaxErrors. Then I fix the `from __future__` imports by moving them to the very top of the script (Python requires this), which unblocks the rest of the code from running. Finally, I keep the model/prediction/thresholding logic identical, but make the weight-loading tolerant to common checkpoint formats and map `pretrain=False` to `pretrain=None` so it doesn’t unintentionally try to download ImageNet weights (no internet), ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the code falling back to all-zero predictions because it never finds the pretrained checkpoint files (the `candidate_paths` don’t exist in your provided `/kaggle/input/aptos2019-blindness-detection/...` tree). The smallest score-improving change (without changing model/thresholding core logic) is to automatically discover `.pth` checkpoints available under `/kaggle/input` and load the first compatible one, instead of relying on three hardcoded missing paths. I also make the weight loader tolerant to common checkpoint wrappers (e.g., `model`, `model_state_dict`) while keeping strict loading when keys match, so you actually use the intended pretrained weights. With a real checkpoint loaded, predictions no longer be constant zeros, moving QWK upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the script not actually loading any usable checkpoint and therefore falling back to constant predictions (which yields ~0 QWK). To move the score upward toward your target with minimal disruption, I keep your model definitions and thresholding exactly the same, but make checkpoint loading more tolerant (try `strict=False` as a fallback after `strict=True`) and smarter about picking likely-relevant files (prioritize filenames containing `seresnet50`/`se_resnet50`/`densenet121` and ignore optimizer-only shards). I also ensure we don’t accidentally try any internet download by keeping `pretrain=None` as you already do. These changes are narrowly targeted to make “loaded_any=True” much more likely, so predictions become non-constant and QWK should increase.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with the code submitting near-constant predictions (because it never finds/loads a compatible checkpoint, so it falls back to all zeros). To move the score upward toward your target while keeping your model and thresholding logic intact, I only make checkpoint discovery/loading more likely to succeed: I expand search to include common `.bin` files, prioritize checkpoints under the competition dataset folder, and add a lightweight compatibility filter to skip non-model artifacts. I also add a final “state_dict key cleaning” step (handle `model.` prefixes) and ensure we always write a correctly aligned `submission.csv`. No architecture, transforms, or thresholds are changed.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the code not actually using any meaningful weights (falling back to constant-ish predictions), because the checkpoint search is likely picking incompatible files and/or failing strict loading silently. I keep your model definitions, image preprocessing, TTA, and threshold-to-classes logic identical, but make checkpoint discovery strongly prioritize the classic APTOS “public” pretrained weights that are commonly bundled in Kaggle datasets (and de-prioritize random `.pth/.pt/.bin` artifacts). I also add a tiny diagnostic: if no compatible checkpoint loads, we still write a valid `submission.csv`, but we print the top few candidate paths and the reason weights were rejected (without changing predictions). These minimal changes should make it much more likely that a real compatible checkpoint is loaded, moving QWK upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from the “no checkpoint loaded → all predictions = 0.0” fallback, which yields near-random agreement on QWK. To move the score upward toward your target with minimal change, I keep your models, preprocessing, TTA, and thresholding exactly the same, but I make checkpoint selection/loading more likely to succeed by (1) only scanning within the competition dataset folder first, (2) preferring reasonably-sized weight files, and (3) adding a last-mile key remap for common `last_linear.*` vs `fc.*` naming mismatches before attempting strict loading. This should make it much more likely that at least one real, compatible checkpoint is used, producing non-constant predictions and a much higher QWK. The script still always writes a valid `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the script never finding any compatible checkpoints (or not having any in the attached dataset), so it falls back to predicting 0 for every test image, which yields ~0 QWK. To move the score upward toward your target while preserving your model definitions, preprocessing, TTA, and thresholding, I add a minimal “train-time fallback” that fits only the existing final regression head (`last_linear`) on top of frozen pretrained ImageNet backbones using the provided `train.csv` and images. This keeps the architecture/loss semantics (still a 1D regression output thresholded to 0–4) but ensures non-constant predictions even when no external `.pth` is available. I also keep your checkpoint discovery/loading unchanged and only trigger the fallback if no checkpoint successfully loads, and I still write an aligned `submission.csv` in the required format.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from the “offline ImageNet preload attempt fails → head-only fallback crashes → all-zero submission” path, because your backbone constructors still call `model_zoo.load_url(...)` when `pretrain="imagenet"` and Kaggle has no internet. To move the score up toward the target with minimal disruption, I keep your model architecture, TTA, regression output, and thresholding identical, but make pretrained loading *offline-safe* by using torchvision’s built-in ImageNet weights (which are shipped with the environment) instead of URLs. I also make the head-only fallback reuse the same normalization as inference (so it won’t silently degrade) and ensure the fallback actually runs end-to-end on CPU/GPU within the time limit. These changes are narrowly targeted to avoid the all-zero submission and produce meaningful predictions, which should lift QWK from ~0 toward your 0.906 target.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division, absolute_import

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
    if pretrained is not None:
        try:
            w = models.AlexNet_Weights.IMAGENET1K_V1
            model = models.alexnet(weights=w)
        except Exception:
            model = models.alexnet(weights=None)
    else:
        model = models.alexnet(weights=None)
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
    if pretrained is not None:
        try:
            w = models.DenseNet121_Weights.IMAGENET1K_V1
            model = models.densenet121(weights=w)
        except Exception:
            model = models.densenet121(weights=None)
    else:
        model = models.densenet121(weights=None)
    model = modify_densenets(model)
    return model


def densenet169(num_classes=1000, pretrained="imagenet"):
    if pretrained is not None:
        try:
            w = models.DenseNet169_Weights.IMAGENET1K_V1
            model = models.densenet169(weights=w)
        except Exception:
            model = models.densenet169(weights=None)
    else:
        model = models.densenet169(weights=None)
    model = modify_densenets(model)
    return model


def densenet201(num_classes=1000, pretrained="imagenet"):
    if pretrained is not None:
        try:
            w = models.DenseNet201_Weights.IMAGENET1K_V1
            model = models.densenet201(weights=w)
        except Exception:
            model = models.densenet201(weights=None)
    else:
        model = models.densenet201(weights=None)
    model = modify_densenets(model)
    return model


def densenet161(num_classes=1000, pretrained="imagenet"):
    if pretrained is not None:
        try:
            w = models.DenseNet161_Weights.IMAGENET1K_V1
            model = models.densenet161(weights=w)
        except Exception:
            model = models.densenet161(weights=None)
    else:
        model = models.densenet161(weights=None)
    model = modify_densenets(model)
    return model


def inceptionv3(num_classes=1000, pretrained="imagenet"):
    if pretrained is not None:
        try:
            w = models.Inception_V3_Weights.IMAGENET1K_V1
            model = models.inception_v3(weights=w, aux_logits=True)
        except Exception:
            model = models.inception_v3(weights=None, aux_logits=True)
    else:
        model = models.inception_v3(weights=None, aux_logits=True)

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
    if pretrained is not None:
        try:
            w = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=w)
        except Exception:
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)
    model = modify_resnets(model)
    return model


def resnet34(num_classes=1000, pretrained="imagenet"):
    if pretrained is not None:
        try:
            w = models.ResNet34_Weights.IMAGENET1K_V1
            model = models.resnet34(weights=w)
        except Exception:
            model = models.resnet34(weights=None)
    else:
        model = models.resnet34(weights=None)
    model = modify_resnets(model)
    return model


def resnet50(num_classes=1000, pretrained="imagenet"):
    if pretrained is not None:
        try:
            w = models.ResNet50_Weights.IMAGENET1K_V2
            model = models.resnet50(weights=w)
        except Exception:
            model = models.resnet50(weights=None)
    else:
        model = models.resnet50(weights=None)
    model = modify_resnets(model)
    return model


def resnet101(num_classes=1000, pretrained="imagenet"):
    if pretrained is not None:
        try:
            w = models.ResNet101_Weights.IMAGENET1K_V2
            model = models.resnet101(weights=w)
        except Exception:
            model = models.resnet101(weights=None)
    else:
        model = models.resnet101(weights=None)
    model = modify_resnets(model)
    return model


def resnet152(num_classes=1000, pretrained="imagenet"):
    if pretrained is not None:
        try:
            w = models.ResNet152_Weights.IMAGENET1K_V2
            model = models.resnet152(weights=w)
        except Exception:
            model = models.resnet152(weights=None)
    else:
        model = models.resnet152(weights=None)
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
    if pretrained is not None:
        try:
            w = models.SqueezeNet1_0_Weights.IMAGENET1K_V1
            model = models.squeezenet1_0(weights=w)
        except Exception:
            model = models.squeezenet1_0(weights=None)
    else:
        model = models.squeezenet1_0(weights=None)
    model = modify_squeezenets(model)
    return model


def squeezenet1_1(num_classes=1000, pretrained="imagenet"):
    if pretrained is not None:
        try:
            w = models.SqueezeNet1_1_Weights.IMAGENET1K_V1
            model = models.squeezenet1_1(weights=w)
        except Exception:
            model = models.squeezenet1_1(weights=None)
    else:
        model = models.squeezenet1_1(weights=None)
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
    if pretrained is not None:
        try:
            w = models.VGG11_Weights.IMAGENET1K_V1
            model = models.vgg11(weights=w)
        except Exception:
            model = models.vgg11(weights=None)
    else:
        model = models.vgg11(weights=None)
    model = modify_vggs(model)
    return model


def vgg11_bn(num_classes=1000, pretrained="imagenet"):
    if pretrained is not None:
        try:
            w = models.VGG11_BN_Weights.IMAGENET1K_V1
            model = models.vgg11_bn(weights=w)
        except Exception:
            model = models.vgg11_bn(weights=None)
    else:
        model = models.vgg11_bn(weights=None)
    model = modify_vggs(model)
    return model


def vgg13(num_classes=1000, pretrained="imagenet"):
    if pretrained is not None:
        try:
            w = models.VGG13_Weights.IMAGENET1K_V1
            model = models.vgg13(weights=w)
        except Exception:
            model = models.vgg13(weights=None)
    else:
        model = models.vgg13(weights=None)
    model = modify_vggs(model)
    return model


def vgg13_bn(num_classes=1000, pretrained="imagenet"):
    if pretrained is not None:
        try:
            w = models.VGG13_BN_Weights.IMAGENET1K_V1
            model = models.vgg13_bn(weights=w)
        except Exception:
            model = models.vgg13_bn(weights=None)
    else:
        model = models.vgg13_bn(weights=None)
    model = modify_vggs(model)
    return model


def vgg16(num_classes=1000, pretrained="imagenet"):
    if pretrained is not None:
        try:
            w = models.VGG16_Weights.IMAGENET1K_V1
            model = models.vgg16(weights=w)
        except Exception:
            model = models.vgg16(weights=None)
    else:
        model = models.vgg16(weights=None)
    model = modify_vggs(model)
    return model


def vgg16_bn(num_classes=1000, pretrained="imagenet"):
    if pretrained is not None:
        try:
            w = models.VGG16_BN_Weights.IMAGENET1K_V1
            model = models.vgg16_bn(weights=w)
        except Exception:
            model = models.vgg16_bn(weights=None)
    else:
        model = models.vgg16_bn(weights=None)
    model = modify_vggs(model)
    return model


def vgg19(num_classes=1000, pretrained="imagenet"):
    if pretrained is not None:
        try:
            w = models.VGG19_Weights.IMAGENET1K_V1
            model = models.vgg19(weights=w)
        except Exception:
            model = models.vgg19(weights=None)
    else:
        model = models.vgg19(weights=None)
    model = modify_vggs(model)
    return model


def vgg19_bn(num_classes=1000, pretrained="imagenet"):
    if pretrained is not None:
        try:
            w = models.VGG19_BN_Weights.IMAGENET1K_V1
            model = models.vgg19_bn(weights=w)
        except Exception:
            model = models.vgg19_bn(weights=None)
    else:
        model = models.vgg19_bn(weights=None)
    model = modify_vggs(model)
    return model




## === cell 2
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
    assert (
        num_classes == settings["num_classes"]
    ), "num_classes should be {}, but is {}".format(
        settings["num_classes"], num_classes
    )
    raise RuntimeError("Offline environment: pretrained URL weights are unavailable.")


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
        pass
    return model




## === cell 3
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


def get_densenet121_gem(pretrain):
    if pretrain == "imagenet":
        model = densenet121(num_classes=1000, pretrained="imagenet")
    else:
        model = densenet121(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(1024, 1)
    return model




## === cell 4
import os
from glob import glob

import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ImageFile.LOAD_TRUNCATED_IMAGES = True

TEST_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/test_images"
TEST_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/test.csv"

test_df = pd.read_csv(TEST_CSV_PATH)

test_images = [
    os.path.join(TEST_IMAGE_PATH, f"{idc}.png") for idc in test_df["id_code"].tolist()
]




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

            image = transforms(image).to(device)
            output = model(image.unsqueeze(0))
            output_flip = model(torch.flip(image.unsqueeze(0), dims=(3,)))

            final_prediction = (output.item() + output_flip.item()) / 2.0
            predictions.append(
                (os.path.splitext(os.path.basename(im_path))[0], final_prediction)
            )
    return predictions


def _looks_like_state_dict(d):
    if not isinstance(d, dict) or len(d) == 0:
        return False
    k0 = next(iter(d.keys()))
    return isinstance(k0, str) and (
        k0.endswith(".weight")
        or k0.endswith(".bias")
        or "running_mean" in k0
        or "running_var" in k0
    )


def _remap_head_keys_for_compat(state, model):
    if not isinstance(state, dict) or not isinstance(model, torch.nn.Module):
        return state

    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())

    if any(k.startswith("fc.") for k in state_keys) and any(
        k.startswith("last_linear.") for k in model_keys
    ):
        remapped = {}
        for k, v in state.items():
            if k.startswith("fc."):
                remapped["last_linear." + k[len("fc.") :]] = v
            else:
                remapped[k] = v
        state = remapped
        state_keys = set(state.keys())

    if any(k.startswith("last_linear.") for k in state_keys) and any(
        k.startswith("fc.") for k in model_keys
    ):
        remapped = {}
        for k, v in state.items():
            if k.startswith("last_linear."):
                remapped["fc." + k[len("last_linear.") :]] = v
            else:
                remapped[k] = v
        state = remapped

    return state


def _safe_load_weights(model, model_path, device):
    if model_path is None or (not os.path.exists(model_path)):
        return False, "missing"

    try:
        state = torch.load(model_path, map_location=device)
    except Exception as e:
        return False, f"torch.load failed: {type(e).__name__}"

    if isinstance(state, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if not _looks_like_state_dict(state):
        return False, "not a state_dict-like object"

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if any(k.startswith("model.") for k in state.keys()):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}

    state = _remap_head_keys_for_compat(state, model)

    try:
        model.load_state_dict(state, strict=True)
        return True, "loaded strict=True"
    except Exception:
        try:
            model.load_state_dict(state, strict=False)
            return True, "loaded strict=False"
        except Exception as e:
            return False, f"load_state_dict failed: {type(e).__name__}"


def _discover_checkpoint_paths():
    roots = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/input",
    ]
    found = []
    for r in roots:
        found.extend(glob(os.path.join(r, "**", "*.pth"), recursive=True))
        found.extend(glob(os.path.join(r, "**", "*.pt"), recursive=True))
        found.extend(glob(os.path.join(r, "**", "*.bin"), recursive=True))

    def _rank(p):
        path_l = p.lower()
        name = os.path.basename(path_l)
        score = 0

        if "aptos2019-blindness-detection" in path_l:
            score -= 500

        if any(tok in name for tok in ["aptos", "blind", "retina", "dr", "kappa"]):
            score -= 80

        if "seresnet50" in name or "se_resnet50" in name or "senet" in name:
            score -= 60
        if "densenet121" in name:
            score -= 50

        if "best" in name:
            score -= 10
        if "fold" in name:
            score -= 5
        if "model" in name or "ckpt" in name or "checkpoint" in name:
            score -= 3

        if any(
            tok in name for tok in ["optim", "optimizer", "sched", "scheduler", "ema"]
        ):
            score += 500

        try:
            sz = os.path.getsize(p)
        except Exception:
            sz = 0

        if sz < 200_000:
            score += 300
        if sz < 5_000_000:
            score += 50
        if sz > 1_500_000_000:
            score += 50

        return (score, len(p))

    found = sorted(list(set(found)), key=_rank)
    return found


class _TrainDS(torch.utils.data.Dataset):
    def __init__(self, df, image_dir, tfm, size=256):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.tfm = tfm
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im_path = os.path.join(self.image_dir, f"{row['id_code']}.png")
        img = Image.open(im_path).convert("RGB")
        img = img.resize((self.size, self.size), resample=Image.BILINEAR)
        x = self.tfm(img)
        y = torch.tensor(float(row["diagnosis"]), dtype=torch.float32)
        return x, y


def _fit_head_only_and_predict(
    arch="se_resnet50_gem",
    size=256,
    batch_size=16,
    epochs=2,
    lr=2e-3,
    device=torch.device("cpu"),
):
    TRAIN_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    TRAIN_IMAGE_PATH = "/kaggle/input/aptos2019-blindness-detection/train_images"
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    if arch == "se_resnet50_gem":
        model = get_se_resnet50_gem(pretrain=None).to(device)
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        model = get_densenet121_gem(pretrain="imagenet").to(device)
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    for p in model.parameters():
        p.requires_grad = False
    for p in model.last_linear.parameters():
        p.requires_grad = True

    ds = _TrainDS(train_df, TRAIN_IMAGE_PATH, tfm, size=size)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    opt = torch.optim.Adam(model.last_linear.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).view(-1, 1)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

    return model, tfm




## === cell 6
candidate_paths = [
    "../input/densenet121/model_densenet121_bs64_30.pth",
    "../input/seresnet50testpseudo/model15.pth",
    "../input/seresnet50-512/model30_512.pth",
]

discovered = _discover_checkpoint_paths()
candidate_paths = candidate_paths + discovered

loaded_any = False
final_predictions = None
loaded_from = None
loaded_msg = None
reject_log = []

for p in candidate_paths:
    if not os.path.exists(p):
        continue

    try:
        model = get_se_resnet50_gem(pretrain=None).to(device)
        ok, msg = _safe_load_weights(model, p, device)
        if ok:
            norm = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            final_predictions = make_predictions(
                model, test_images, norm, size=256, device=device
            )
            loaded_any = True
            loaded_from = p
            loaded_msg = msg
            break
        else:
            reject_log.append((p, "se_resnet50_gem", msg))
    except Exception as e:
        reject_log.append((p, "se_resnet50_gem", f"exception: {type(e).__name__}"))

    try:
        model = get_densenet121_gem(pretrain=None).to(device)
        ok, msg = _safe_load_weights(model, p, device)
        if ok:
            norm = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            final_predictions = make_predictions(
                model, test_images, norm, size=256, device=device
            )
            loaded_any = True
            loaded_from = p
            loaded_msg = msg
            break
        else:
            reject_log.append((p, "densenet121_gem", msg))
    except Exception as e:
        reject_log.append((p, "densenet121_gem", f"exception: {type(e).__name__}"))

if not loaded_any:
    try:
        model, norm = _fit_head_only_and_predict(
            arch="densenet121_gem",
            size=256,
            batch_size=16,
            epochs=2,
            lr=2e-3,
            device=device,
        )
        final_predictions = make_predictions(
            model, test_images, norm, size=256, device=device
        )
        loaded_any = True
        loaded_from = "head_only_finetune(densenet121_gem, imagenet_offline)"
        loaded_msg = "trained last_linear only"
    except Exception as e:
        final_predictions = [(idc, 0.0) for idc in test_df["id_code"].tolist()]
        loaded_from = None
        loaded_msg = f"fallback_zeros_due_to: {type(e).__name__}"

print("Loaded weights:", loaded_any, "| from:", loaded_from, "|", loaded_msg)
print(
    "Num candidate checkpoint files checked (existing only):",
    sum(int(os.path.exists(p)) for p in candidate_paths),
)
if not loaded_any:
    print("Top 10 discovered checkpoint candidates (ranked):")
    for p in discovered[:10]:
        try:
            print(" ", p, "| size:", os.path.getsize(p))
        except Exception:
            print(" ", p)
    print("Last 10 rejections (path, model, reason):")
    for row in reject_log[-10:]:
        print(" ", row)



## === cell 7
submission = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis"])

submission.loc[submission.diagnosis < 0.75, "diagnosis"] = 0
submission.loc[
    (0.75 <= submission.diagnosis) & (submission.diagnosis < 1.5), "diagnosis"
] = 1
submission.loc[
    (1.5 <= submission.diagnosis) & (submission.diagnosis < 2.5), "diagnosis"
] = 2
submission.loc[
    (2.5 <= submission.diagnosis) & (submission.diagnosis < 3.5), "diagnosis"
] = 3
submission.loc[3.5 <= submission.diagnosis, "diagnosis"] = 4
submission["diagnosis"] = submission["diagnosis"].astype(int)

submission = submission.set_index("id_code").loc[test_df["id_code"]].reset_index()

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
