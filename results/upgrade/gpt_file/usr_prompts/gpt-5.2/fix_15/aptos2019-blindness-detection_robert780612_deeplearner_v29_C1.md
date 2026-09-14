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

0.9194098121606712

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the runtime failure by making the script run on CPU when CUDA isn’t available and by loading checkpoints with a device-agnostic `map_location`. I also make the test image ordering deterministic (sorted by `id_code`) and wrap inference in `torch.no_grad()` to avoid unnecessary overhead while keeping identical model logic. Finally, I guard against missing model files (common when Kaggle datasets aren’t attached) by falling back to a valid default prediction using `sample_submission.csv`, ensuring a proper `submission.csv` is always produced. These changes are correctness/stability focused and should allow you to actually obtain a Kaggle score again.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from the “fallback” path that writes `sample_submission.csv` unchanged when checkpoint files aren’t found (it predicts all zeros), so the smallest improvement is to ensure the notebook can actually see and load the 3 `.pth` files when they are present. I add a tiny auto-discovery step that searches `/kaggle/input/**` for the expected checkpoint filenames and uses the first match, while keeping the exact same models, TTA, and rounding logic. I also align the DenseNet normalization to ImageNet stats (the model definition can be loaded either way, but the input normalization must match training; this is a minimal inference-only fix that usually moves QWK up a lot). Finally, I make the “no checkpoint found” fallback still produce a valid submission, but emit a clear warning so it’s obvious why scores would be near 0.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the “no checkpoints loaded → write sample_submission (all zeros)” fallback path being taken, so the smallest score-moving change is to reliably find and load the provided `.pth` files when they exist under `/kaggle/input`. I keep your exact models, TTA (horizontal flip), resizing, and rounding bins, but make checkpoint discovery more robust by searching for `.pth` filenames *and* common alternative locations, then logging which ones were loaded. I also ensure inference uses the same normalization consistently (already ImageNet) and enforce deterministic `test_images` alignment strictly by `test.csv` order (your current code already does this) while guarding against missing image files causing silent crashes. These changes should move you off 0.0 toward your target by actually using the trained ensemble when present, without changing core modeling logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from the “no checkpoints loaded → write sample_submission (all zeros)” path, so the smallest score-moving fix is to make checkpoint discovery and loading actually work in Kaggle’s input layout. I add a robust search for the three expected `.pth` filenames under `/kaggle/input/**` (including nested dataset folders), and make loading tolerant to common checkpoint formats (`state_dict`, `model_state_dict`, DataParallel `module.` prefix) without changing the model architectures. I also enforce that we always attempt to load by basename first (so slight folder differences don’t break it), while keeping your exact inference/TTA and binning logic intact. This should move you off 0.0 toward your target by ensuring the ensemble is used when the model files are present.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from the “no checkpoints loaded → sample_submission (all zeros)” fallback path, so the smallest score-moving fix is to reliably locate and load your three `.pth` files under Kaggle’s `/kaggle/input/**` layout. I keep the exact same model architectures, TTA (horizontal flip), image resizing, and binning thresholds, but make checkpoint discovery smarter by (1) searching by basename and (2) also looking for any `.pth` that contains a distinctive substring (e.g., `densenet121`, `fine_tune_256`, `512`) when the exact filename isn’t found. I also make loading tolerant to common checkpoint wrappers (`state_dict`, `model_state_dict`) and `DataParallel` prefixes (already present), and ensure we always load to CPU/GPU correctly. This should move you off 0.0 toward your target by actually using the trained ensemble when the model files exist, without changing inference semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the code taking the “no checkpoints loaded → write sample_submission (all zeros)” path, so the smallest score-improving change is to make checkpoint discovery/load actually succeed in Kaggle’s folder layout. I keep your exact models, TTA (horizontal flip), resizing, and binning, but (1) broaden the `.pth` search to include `.pt`/`.bin` and common substrings, (2) make state_dict extraction handle more checkpoint wrappers (including nested `state_dict` and `ema`), and (3) fail loudly with a clear report of what files were found so you can confirm the ensemble is being used. These changes should move you off 0.0 toward your target by ensuring real model predictions are produced rather than the all-zero fallback, without changing evaluation semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because no checkpoints are actually being loaded, so you’re submitting the all-zeros `sample_submission.csv` fallback. I make the checkpoint discovery/loading more robust (without changing any model architecture or inference logic) by (1) searching `/kaggle/input/**` for any `.pth/.pt/.bin` that matches each model’s expected “signature”, and (2) improving state_dict extraction so common wrappers (including nested keys and DataParallel prefixes) load correctly. I also make the script fail over more intelligently only if a specific model weight truly can’t be found, while still always producing a valid `submission.csv`. These changes should move you off 0.0 and toward your target by ensuring the ensemble actually runs when weights exist in the environment.'
- What this solution (achieved 0.0) has done: 'I fix the runtime failure by making the “no checkpoints loaded” path always produce a valid `submission.csv` instead of raising, since your environment clearly doesn’t include the external weight datasets referenced by those hardcoded paths. To still move score upward (toward your target) without changing the model architecture or inference semantics, I add a minimal fallback model that uses a standard pretrained torchvision backbone (available offline via torchvision weights) to generate non-degenerate predictions, and then apply your same flip-TTA and threshold binning. I also add a small, safe path auto-detection for `DATA_ROOT` so the script works whether the competition files are in `/kaggle/input/...` or `/kaggle/data/...`. All changes are strictly to unblock end-to-end execution and avoid the all-zero submission.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly caused by the script taking the fallback path (no competition checkpoints found/loaded) and then using a randomly initialized regression head on top of a pretrained ResNet50, which produces near-constant/poor ordinal predictions after binning. To move the score toward your target with minimal logic change, I keep your exact ensemble/TTA/bins, but (1) make checkpoint discovery load *any* compatible `.pth/.pt/.bin` if the “signature” substrings don’t match, and (2) make the fallback head non-random by setting the ResNet50 FC bias to the training-set mean diagnosis (a simple calibration that typically beats random and avoids degenerate all-zeros). I also align the test image root detection to handle the common nested `test_images/test_images` folder layout so images are actually found. These changes are strictly to ensure real weights are used when present, and to make the fallback produce sensible ordinal outputs when not.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from the fallback path (no competition checkpoints actually loaded), and the current fallback model produces near-constant predictions after binning, which yields poor QWK. To move the score upward toward your target with minimal disruption, I keep your exact ensemble/TTA and binning logic, but make checkpoint discovery/load more robust (handle more checkpoint wrappers and key prefixes) and—when no checkpoints are found—fit only the final linear layer of the existing torchvision ResNet50 regressor on the provided `train.csv` using frozen ImageNet features (very small training loop, same inference semantics). This keeps the core modeling approach intact (regression + fixed thresholds) while ensuring the fallback produces meaningful ordinal predictions instead of near-constant outputs. The script still always writes a valid `submission.csv` and stays within Kaggle’s offline constraints.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the ensemble checkpoints still aren’t being found/loaded in your Kaggle run, so you’re falling back to the weak “train just the final FC” model. I make the smallest change that increases score toward your target: (1) broaden checkpoint discovery to search both `/kaggle/input/**` and `/kaggle/data/**`, (2) require “signature substring” matches only as a preference (not a hard gate), and (3) add a tiny second-pass load that pick the best matching weight file per model when the exact path isn’t present. This keeps your exact architectures, TTA, resizing, and binning intact, but should move you off the fallback path and toward your target by actually using your trained ensemble when the weight files exist.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is almost certainly because no real competition checkpoints are being loaded, so you’re falling back to the weak ResNet50 regressor path; the smallest move toward your 0.919 target is to reliably load the intended 3 trained checkpoints. I (1) make checkpoint discovery choose the *best* match per model (instead of “first .pth anywhere”), (2) load checkpoints more robustly by filtering candidate files by substring “signature” and preferring shallower paths, and (3) stop the fallback from accidentally picking unrelated weights. This keeps your exact model architectures, TTA, resizing, and threshold binning unchanged, but should move you off the degenerate fallback and toward the target by actually using the ensemble when weights exist.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the script still not loading the intended three competition checkpoints, so it falls back to a weak model and/or mismatched weights that collapse predictions after binning. I make the smallest change that increases the chance of loading the *right* `.pth` files by (1) preferring candidates by “signature” match (must-contain substrings) and (2) additionally preferring files whose state_dict actually matches the model (by trying a lightweight `load_state_dict(strict=False)` check before committing). I also ensure we don’t accidentally pick unrelated weights by only falling back to “any .pth/.pt/.bin” when signature-based search yields nothing. This keeps your architectures, TTA, resizing, and binning unchanged, but should move the score upward toward the target by using the real ensemble when present.'

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

import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_CANDIDATE_ROOTS = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_ROOT = next(
    (p for p in _CANDIDATE_ROOTS if os.path.exists(p)), _CANDIDATE_ROOTS[0]
)


def _resolve_images_dir(root: str, split: str) -> str:
    cands = [
        os.path.join(root, "aptos2019-blindness-detection", f"{split}_images"),
        os.path.join(
            root, "aptos2019-blindness-detection", f"{split}_images", f"{split}_images"
        ),
        os.path.join(root, f"{split}_images"),
        os.path.join(root, f"{split}_images", f"{split}_images"),
    ]
    for p in cands:
        if os.path.isdir(p):
            return p
    return cands[0]


def _resolve_csv(root: str, name: str) -> str:
    cands = [
        os.path.join(root, "aptos2019-blindness-detection", name),
        os.path.join(root, name),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return cands[0]


TEST_IMAGE_PATH = _resolve_images_dir(DATA_ROOT, "test")
TRAIN_IMAGE_PATH = _resolve_images_dir(DATA_ROOT, "train")
SAMPLE_SUB_PATH = _resolve_csv(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = _resolve_csv(DATA_ROOT, "train.csv")
TEST_CSV_PATH = _resolve_csv(DATA_ROOT, "test.csv")

ALLOW_ALL_ZERO_FALLBACK = True



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

senet_pretrained_settings = {
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
        settings = senet_pretrained_settings["se_resnet50"][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model




## === cell 3
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


def get_densenet121_gem(pretrain):
    if pretrain == "imagenet":
        model = densenet121(num_classes=1000, pretrained="imagenet")
    else:
        model = densenet121(num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(1024, 1)
    return model




## === cell 4
test_df = pd.read_csv(TEST_CSV_PATH)

if not os.path.isdir(TEST_IMAGE_PATH):
    raise FileNotFoundError(f"TEST_IMAGE_PATH not found: {TEST_IMAGE_PATH}")

test_images = [
    os.path.join(TEST_IMAGE_PATH, f"{i}.png") for i in test_df["id_code"].tolist()
]


def make_predictions(model, test_images, tfms, size=256, device=device):
    predictions = []
    model.eval()
    with torch.no_grad():
        for im_path in test_images:
            try:
                image = Image.open(im_path).convert("RGB")
            except Exception:
                predictions.append(
                    (os.path.splitext(os.path.basename(im_path))[0], 0.0)
                )
                continue

            image = image.resize((size, size), resample=Image.BILINEAR)
            image = tfms(image).to(device)

            out = model(image.unsqueeze(0))
            out_flip = model(torch.flip(image.unsqueeze(0), dims=(3,)))
            final_prediction = (out.item() + out_flip.item()) / 2.0

            predictions.append(
                (os.path.splitext(os.path.basename(im_path))[0], final_prediction)
            )
    return predictions


def resolve_ckpt_path(preferred_path: str, must_contain_substr=None) -> str:
    if os.path.exists(preferred_path):
        return preferred_path

    fname = os.path.basename(preferred_path)
    base, ext = os.path.splitext(fname)

    search_roots = ["/kaggle/input", "/kaggle/data"]
    candidates = []
    for sr in search_roots:
        candidates.extend(glob(os.path.join(sr, "**", fname), recursive=True))

    if len(candidates) == 0 and ext.lower() == ".pth":
        for alt_ext in [".pt", ".bin"]:
            for sr in search_roots:
                candidates.extend(
                    glob(os.path.join(sr, "**", base + alt_ext), recursive=True)
                )

    if len(candidates) == 0 and must_contain_substr is None:
        for e in ["*.pth", "*.pt", "*.bin"]:
            for sr in search_roots:
                candidates.extend(glob(os.path.join(sr, "**", e), recursive=True))

    candidates = sorted(set(candidates), key=lambda p: (len(p), p))
    if len(candidates) == 0:
        return preferred_path

    if must_contain_substr is not None:
        if isinstance(must_contain_substr, str):
            must_contain_substr = [must_contain_substr]
        req = [str(s).lower() for s in must_contain_substr if s is not None and str(s)]

        def _score(p: str):
            pl = p.lower()
            hit = sum(1 for s in req if s in pl)
            return (-hit, len(p), p)

        ranked = sorted(candidates, key=_score)
        best = ranked[0]
        if any(s in best.lower() for s in req):
            return best
        return preferred_path

    return candidates[0]


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "net",
            "model",
            "weights",
            "ema_state_dict",
            "ema",
            "student",
            "teacher",
        ]:
            if k in obj:
                candidate = obj[k]
                if isinstance(candidate, dict):
                    return candidate
        for k in list(obj.keys()):
            v = obj.get(k)
            if isinstance(v, dict) and any(
                sk in v for sk in ["state_dict", "model_state_dict"]
            ):
                return _extract_state_dict(v)
        if len(obj) > 0 and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    return obj


def _strip_prefixes(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    out = dict(state_dict)

    def strip_one(d, pref):
        keys = list(d.keys())
        if all(isinstance(k, str) and k.startswith(pref) for k in keys):
            return {k[len(pref) :]: v for k, v in d.items()}
        return d

    for pref in ["module.", "model.", "net.", "backbone."]:
        out = strip_one(out, pref)
    for _ in range(2):
        if not isinstance(out, dict) or len(out) == 0:
            break
        for pref in ["module.", "model.", "net.", "backbone."]:
            out = strip_one(out, pref)
    return out


def _candidate_ckpts_for_signature(preferred_path: str, must_contain_substr):
    fname = os.path.basename(preferred_path)
    base, ext = os.path.splitext(fname)

    search_roots = ["/kaggle/input", "/kaggle/data"]
    cands = []
    for sr in search_roots:
        cands.extend(glob(os.path.join(sr, "**", fname), recursive=True))

    if len(cands) == 0 and ext.lower() == ".pth":
        for alt_ext in [".pth", ".pt", ".bin"]:
            for sr in search_roots:
                cands.extend(
                    glob(os.path.join(sr, "**", base + alt_ext), recursive=True)
                )

    cands = sorted(set(cands), key=lambda p: (len(p), p))

    if must_contain_substr is None:
        return cands

    if isinstance(must_contain_substr, str):
        must_contain_substr = [must_contain_substr]
    req = [str(s).lower() for s in must_contain_substr if s is not None and str(s)]

    def _hitcount(p: str):
        pl = p.lower()
        return sum(1 for s in req if s in pl)

    cands = sorted(cands, key=lambda p: (-_hitcount(p), len(p), p))
    return cands


def safe_load_state_dict(model, ckpt_path, device=device, must_contain_substr=None):
    candidates = _candidate_ckpts_for_signature(
        ckpt_path, must_contain_substr=must_contain_substr
    )

    if os.path.exists(ckpt_path):
        candidates = [ckpt_path] + [c for c in candidates if c != ckpt_path]

    tried = 0
    for cand in candidates[:30]:  # cap tries for runtime safety
        if not os.path.exists(cand):
            continue
        tried += 1
        try:
            raw = torch.load(cand, map_location=device)
            state = _strip_prefixes(_extract_state_dict(raw))
            missing, unexpected = model.load_state_dict(state, strict=False)
            if len(missing) >= max(1, len(model.state_dict()) - 2):
                continue
            print(
                f"Loaded checkpoint: {cand} | missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
            )
            return True
        except Exception:
            continue

    resolved = resolve_ckpt_path(ckpt_path, must_contain_substr=must_contain_substr)
    if not os.path.exists(resolved):
        print(
            f"Checkpoint not found: {resolved} (from preferred {ckpt_path}) | tried={tried}"
        )
        return False

    try:
        raw = torch.load(resolved, map_location=device)
        state = _strip_prefixes(_extract_state_dict(raw))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"Loaded checkpoint: {resolved} | missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
        )
        return True
    except Exception as e:
        print(f"Failed to load checkpoint: {resolved} | error: {e}")
        return False


IMAGENET_NORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 5
predictions_densenet = None
predictions_seresnet = None
predictions_seresnet_512 = None

MODEL_PATH = "/kaggle/input/densenet121/model_densenet121_bs64_30.pth"
model = get_densenet121_gem(pretrain=False).to(device)
if safe_load_state_dict(
    model, MODEL_PATH, device=device, must_contain_substr=["densenet121", "bs64", "30"]
):
    predictions_densenet = make_predictions(
        model, test_images, IMAGENET_NORM, size=224, device=device
    )



## === cell 6
MODEL_PATH = "/kaggle/input/seresnet50pretrain/fine_tune_256_model30.pth"
model = get_se_resnet50_gem(pretrain=False).to(device)
if safe_load_state_dict(
    model,
    MODEL_PATH,
    device=device,
    must_contain_substr=["fine", "tune", "256", "model30"],
):
    predictions_seresnet = make_predictions(
        model, test_images, IMAGENET_NORM, size=256, device=device
    )



## === cell 7
MODEL_PATH = "/kaggle/input/seresnet50-512/model30_512.pth"
model = get_se_resnet50_gem(pretrain=False).to(device)
if safe_load_state_dict(
    model, MODEL_PATH, device=device, must_contain_substr=["512", "model30"]
):
    predictions_seresnet_512 = make_predictions(
        model, test_images, IMAGENET_NORM, size=512, device=device
    )




## === cell 8
def _train_label_mean_default(train_csv_path: str) -> float:
    try:
        if os.path.exists(train_csv_path):
            df = pd.read_csv(train_csv_path)
            if "diagnosis" in df.columns:
                return float(df["diagnosis"].mean())
    except Exception:
        pass
    return 0.0


class TVResNet50Regressor(nn.Module):
    def __init__(self, init_bias: float = 0.0):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)
        with torch.no_grad():
            nn.init.zeros_(self.backbone.fc.weight)
            self.backbone.fc.bias.fill_(float(init_bias))

    def forward(self, x):
        return self.backbone(x)


def _fit_fallback_fc_on_train(
    model: TVResNet50Regressor,
    train_csv_path: str,
    train_img_dir: str,
    tfms,
    size: int = 256,
    max_samples: int = 800,
    batch_size: int = 32,
    epochs: int = 2,
    lr: float = 5e-2,
    device=device,
):
    if not (os.path.exists(train_csv_path) and os.path.isdir(train_img_dir)):
        print("Fallback FC fit skipped: train CSV or train image dir missing.")
        return

    df = pd.read_csv(train_csv_path)
    if "id_code" not in df.columns or "diagnosis" not in df.columns:
        print("Fallback FC fit skipped: train CSV missing required columns.")
        return

    df = df[["id_code", "diagnosis"]].copy()
    df["path"] = df["id_code"].apply(lambda x: os.path.join(train_img_dir, f"{x}.png"))
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    if len(df) == 0:
        print("Fallback FC fit skipped: no train images found.")
        return

    df = df.iloc[: min(len(df), max_samples)].reset_index(drop=True)

    class _DS(torch.utils.data.Dataset):
        def __init__(self, df_):
            self.df = df_

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            p = self.df.loc[idx, "path"]
            y = float(self.df.loc[idx, "diagnosis"])
            img = (
                Image.open(p)
                .convert("RGB")
                .resize((size, size), resample=Image.BILINEAR)
            )
            x = tfms(img)
            return x, torch.tensor([y], dtype=torch.float32)

    ds = _DS(df)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )

    for param in model.parameters():
        param.requires_grad = False
    for param in model.backbone.fc.parameters():
        param.requires_grad = True

    model.train()
    optim = torch.optim.SGD(model.backbone.fc.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        running = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        print(f"Fallback FC fit epoch {ep+1}/{epochs} | mse={running/max(n,1):.4f}")

    model.eval()


available = [
    p
    for p in [predictions_densenet, predictions_seresnet, predictions_seresnet_512]
    if p is not None
]

final_predictions = []
if len(available) == 0:
    found_any = []
    for e in ["*.pth", "*.pt", "*.bin"]:
        found_any.extend(glob(os.path.join("/kaggle/input", "**", e), recursive=True))
        found_any.extend(glob(os.path.join("/kaggle/data", "**", e), recursive=True))
    found_any = sorted(set(found_any), key=lambda p: (len(p), p))

    print(
        "WARNING: No competition checkpoints found/loaded.\n"
        "Discovered weight-like files under /kaggle/input and /kaggle/data (first 50):"
    )
    for p in found_any[:50]:
        print("  ", p)

    mu = _train_label_mean_default(TRAIN_CSV_PATH)
    print(
        f"Falling back to torchvision pretrained ResNet50 regressor (init bias=train mean={mu:.4f})."
    )

    fb_model = TVResNet50Regressor(init_bias=mu).to(device)

    _fit_fallback_fc_on_train(
        fb_model,
        train_csv_path=TRAIN_CSV_PATH,
        train_img_dir=TRAIN_IMAGE_PATH,
        tfms=IMAGENET_NORM,
        size=256,
        max_samples=800,
        batch_size=32,
        epochs=2,
        lr=5e-2,
        device=device,
    )

    final_predictions = make_predictions(
        fb_model, test_images, IMAGENET_NORM, size=256, device=device
    )
else:
    pred_maps = [{k: v for k, v in preds} for preds in available]
    for id_code in test_df["id_code"].tolist():
        vals = [m[id_code] for m in pred_maps]
        final_predictions.append([id_code, float(np.mean(vals))])

submission = pd.DataFrame(final_predictions, columns=["id_code", "diagnosis"])

if (
    submission["diagnosis"].dtype != np.int64
    and submission["diagnosis"].dtype != np.int32
):
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

sample = pd.read_csv(SAMPLE_SUB_PATH)
submission = sample[["id_code"]].merge(submission, on="id_code", how="left")
submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print(
    "Wrote submission.csv with shape:",
    submission.shape,
    "on device:",
    device,
    "| models_loaded:",
    len(available),
)
