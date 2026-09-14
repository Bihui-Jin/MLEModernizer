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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
timm==1.0.19
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

0.9184567493500784

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the failing `pip install` cell and rely on the already-installed `timm` to avoid missing-wheel errors, (2) make the code robust to CPU-only Kaggle runtimes by selecting `cuda` only if available and eliminating hardcoded `.cuda()` calls, and (3) fix the missing weights path by automatically locating a `.pkl`/`.pt` file under the available input directories (or falling back to untrained weights so the pipeline still produces a non-empty `submission.csv`). I also fix a couple of runtime issues that would stop inference (missing `warnings` import and a bad `is` string comparison) while keeping the model and prediction logic unchanged. The end result run end-to-end and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'I (1) fix the checkpoint discovery so it finds weights that are shipped inside the competition dataset folders (not just `/kaggle/input`) and avoid hard-failing before producing a submission, (2) fix the CUDA/CPU dtype/device mismatch by ensuring the model is moved to `device` before any forward pass and by loading state dicts after `.to(device)` safely, and (3) fix the submission assembly bug by keeping `submission` as a list and checking its length before converting to a DataFrame. These changes are minimal, keep the same model and inference logic, and ensure an end-to-end run that writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with running inference from (effectively) untrained weights because the checkpoint isn’t being found/loaded; the smallest meaningful improvement is to reliably locate and load the intended `B4_3stage_12epoch_320.pkl` from the actual dataset folders that exist in this environment. I expand the checkpoint search to include the known competition directory locations under `/kaggle/data/...` (including nested duplicates) and make loading slightly more permissive (still safe) by accepting a lower key-match ratio when the head shapes match, so we don’t unnecessarily reject the correct file. This keeps the exact same model, transforms, and prediction logic; it only increases the chance we use the shipped trained weights, which should move the score sharply upward toward your target. The submission writing stays identical and still always produces a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the predictions are effectively random/constant, which in this pipeline most likely comes from either (a) failing to load the intended checkpoint or (b) using the wrong inference head/output for the checkpoint you actually have. I make the smallest changes that (1) more reliably locate and load `B4_3stage_12epoch_320.pkl` specifically (and avoid accidentally picking an unrelated `.pkl/.pth`), and (2) use the model’s built-in 3-head fusion (`combine3output`) that matches the provided architecture (still the same model and same outputs, just combining them as intended rather than using only the regression head). These two tweaks should move your score sharply upward toward the target without changing architecture, training, or data preprocessing. The script still run end-to-end and always write a valid `submission.csv` with the correct columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the checkpoint not actually being loaded (so predictions come from essentially random/untrained weights). I make the smallest change that increases the chance of loading the *exact* intended `B4_3stage_12epoch_320.pkl` by expanding the search to include the dataset’s real directory layout (including the `train_images.zip`/`test_images.zip` sibling locations) and also accepting common checkpoint key names (`model_state_dict`, `net`, `ema`, etc.) without changing the model or inference logic. I also add a lightweight sanity print of the prediction label distribution so we can detect the “all-one-class” failure mode that often yields near-zero kappa. The submission format and inference loop stay the same and still always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the submission rows being misaligned with Kaggle’s expected `test.csv` order (even if the CSV is valid), which can drive kappa to ~0. I make the smallest change to guarantee `submission.csv` is merged back onto the official `test.csv` `id_code` order (and to detect any missing images) without changing your model, transforms, or prediction logic. I also clamp and type-cast predictions exactly as before, but now after the merge so the file is guaranteed aligned and complete. This should move the score sharply upward toward your target if the weights are loading correctly.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with the model effectively producing near-random/constant outputs because the intended checkpoint is still not being found/loaded in this environment. I make the smallest change that increases the probability of loading the exact shipped weights by (1) searching specifically under the known competition dataset directories (including nested duplicates) and (2) adding a safe fallback to load the raw `state_dict` even when the key-match ratio is low, as long as the critical head layers match shapes. This keeps your model, transforms, and inference logic the same; it only changes how robustly the correct weights are discovered/loaded, which should move QWK sharply upward toward your target. The submission writing and test.csv alignment remain unchanged to ensure a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model effectively using untrained/random weights (checkpoint not found/loaded) or loading the wrong file; the smallest change that should move QWK sharply upward toward your target is to reliably locate the shipped `B4_3stage_12epoch_320.pkl` inside the dataset tree you actually have and make the loader accept the common “wrapped checkpoint” formats. I keep your model, transforms, and inference logic identical, but expand the weight search into `/kaggle/data/**` and `/kaggle/input/**` recursively with a strict preference for the exact filename, then robustly extract the correct `state_dict` key (including nested `state_dict`/`model`/`ema`). Finally, I add a hard check that we don’t silently proceed with random weights when a compatible checkpoint exists (still producing `submission.csv` either way), which prevents the common “valid CSV but 0.0 score” failure mode.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most plausibly coming from predictions made with untrained weights because the checkpoint still isn’t being found/loaded (or is loaded under a slightly different filename), which makes kappa collapse toward 0. The smallest change that should move you sharply upward toward the target is to broaden the checkpoint discovery to accept common filename variants and extensions (while still preferring the exact intended file) and to make the loader accept either raw `state_dict` or nested checkpoints without rejecting a good match unnecessarily. I also add a hard “if a matching file exists, load it” behavior (still producing a submission either way) to avoid silently running random weights. The model, transforms, inference loop, and prediction combination logic remain unchanged.'
- What this solution (achieved 0.0) has done: 'I fix the `AttributeError` caused by `timm` EfficientNet API differences by making `backboneNet_efficient` pull activation and pooling modules robustly (handling `act1` vs `act_fn` and `global_pool` vs `pool`), without changing the model’s heads or forward semantics. Then I ensure the model instantiation succeeds so `net1` exists and inference actually runs, producing non-empty predictions. Finally, I keep the same submission assembly logic but add a small safety fallback for any missing images so a valid `submission.csv` is always written.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with using untrained/random weights or loading a mismatched checkpoint, so the smallest improvement toward your 0.918 target is to (1) ensure we actually find the shipped `B4_3stage_12epoch_320.*` file if it exists anywhere under the provided dataset tree, and (2) correctly map checkpoint key prefixes (e.g., `net.`, `model.`, `module.`) so the state dict loads instead of being mostly rejected. I keep your exact model, transforms, and `combine3output` prediction logic unchanged; only checkpoint discovery/loading robustness is adjusted. I also add a hard preference for the exact intended basename to avoid accidentally loading an unrelated `.pth/.pkl`. This should move the score sharply upward if the weights are present, while still always writing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely because the loaded checkpoint (if any) doesn’t actually match `backboneNet_efficient`’s parameter names, so almost nothing meaningful gets loaded and inference runs with effectively random weights. To move the score toward your 0.918 target with minimal change, I keep your model/inference logic intact but (1) make weight discovery also consider common “3stage” checkpoints that would match this architecture, and (2) fix the key-normalization so we don’t incorrectly strip `backbone.` (which would break matching for this model) and also accept `state_dict` entries under common keys. This increases the chance that we genuinely load trained weights for `backboneNet_efficient`, which should sharply improve QWK while preserving the rest of your pipeline and still producing a valid `submission.csv` aligned to `test.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import time
import math
import json
import copy
import csv
import pickle
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT

from PIL import Image, ImageChops, ImageEnhance, ImageOps
import cv2

from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
from torch.optim import lr_scheduler
from timm.models import *
from torchvision import transforms as tv_transforms, utils, models, datasets
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork,
    LastLevelMaxPool,
)
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d as TVFrozenBatchNorm2d
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import misc as misc_nn_ops
from torch import Tensor
from torch.jit.annotations import List, Optional, Tuple


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

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


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed
    """

    def __init__(self, num_features: int, eps: float = 1e-5, n: Optional[int] = None):
        if n is not None:
            warnings.warn(
                "`n` argument is deprecated and has been renamed `num_features`",
                DeprecationWarning,
            )
            num_features = n
        super(FrozenBatchNorm2d, self).__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.eps})"


class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
            return F.linear(input, self.weight.to(dtype=input.dtype), bias=bias)
        else:
            return F.linear(input, self.weight, self.bias)


def _pick_attr(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


class backboneNet_efficient(nn.Module):
    def __init__(self):
        super(backboneNet_efficient, self).__init__()
        net = timm.create_model("tf_efficientnet_b4_ns", pretrained=False)

        layers_to_train = ["blocks"]
        for name, parameter in net.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        self.num_features = 1792
        self.conv_stem = net.conv_stem
        self.bn1 = net.bn1

        self.act1 = _pick_attr(net, ["act1", "act_fn"], default=nn.Identity())

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]
        self.conv_head = net.conv_head
        self.bn2 = net.bn2

        self.act2 = _pick_attr(
            net, ["act2", "act_fn2", "act_fn"], default=nn.Identity()
        )

        self.global_pool = _pick_attr(
            net, ["global_pool", "pool"], default=nn.Identity()
        )

        self.drop_rate = 0.4

        self.rg_cls = Linear(self.num_features, 1, bias=True)
        self.cls_cls = Linear(self.num_features, 5, bias=True)
        self.ord_cls = Linear(self.num_features, 4, bias=True)

    def forward(self, x):
        x1 = self.conv_stem(x)
        x2 = self.bn1(x1)
        x3 = self.act1(x2)

        x4 = self.block0(x1)
        x5 = self.block1(x4)
        x6 = self.block2(x5)
        x7 = self.block3(x6)
        x8 = self.block4(x7)
        x9 = self.block5(x8)
        x10 = self.block6(x9)
        x11 = self.conv_head(x10)
        x12 = self.bn2(x11)
        x13 = self.act2(x12)
        x14 = self.global_pool(x13)

        if self.drop_rate > 0.0:
            x14 = F.dropout(x14, p=self.drop_rate, training=self.training)

        x15 = self.rg_cls(x14)
        x16 = self.cls_cls(x14)
        x17 = self.ord_cls(x14)

        return x15, x16, x17




## === cell 2
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = 0
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu().item()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(out[i].item()))
            l2 = int(math.ceil(out[i].item()))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob


def combine3output(r_out, c_out, o_out):
    R = regress2class(r_out.data)
    _, C = torch.max(c_out.data, 1)
    C = C.squeeze().item()
    _, O = torch.max(o_out.data, 1)
    O = O.squeeze().item()

    P = (R + C + O) / 3.0
    P = int(round(P))
    return P




## === cell 3
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

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


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=False)
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=False)
        self.backbone.global_pool = GeM(flatten=True)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 4),
        )

        self.final_regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x, final=False):
        x = self.backbone(x)

        c_out = self.classifier(x)
        r_out = self.regressor(x)
        o_out = self.ordinal(x)

        if final:
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            r_out = torch.sigmoid(r_out) * 4.5
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out




## === cell 4
class photometric_distort(object):
    def __call__(self, image):
        distortions = [
            FT.adjust_brightness,
            FT.adjust_contrast,
            FT.adjust_saturation,
            FT.adjust_hue,
        ]
        random.shuffle(distortions)

        for d in distortions:
            if random.random() < 0.5:
                if d.__name__ == "adjust_hue":
                    adjust_factor = random.uniform(-16 / 255.0, 16 / 255.0)
                else:
                    adjust_factor = random.uniform(0.7, 1.3)
                image = d(image, adjust_factor)
        return image


class cropTo4_3(object):
    def __call__(self, image):
        w, h = image.size
        if (w / h) >= (4 / 3):
            new_h = h
            new_w = int(h * 4 / 3)
        else:
            new_h = int(w * 3 / 4)
            new_w = w

        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = left + new_w
        bottom = top + new_h
        return image.crop((left, top, right, bottom))


class trim(object):
    def __call__(self, image):
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        return image


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img




## === cell 5
BASE_DIR_CANDIDATES = [
    "/kaggle/input",
    "/kaggle/data",  # important for this environment
    "/kaggle/data/input",
    "../input",
    "../data",
    "../data/input",
]
COMP_SLUG = "aptos2019-blindness-detection"


def first_existing_path(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        nk = k
        for pref in ("module.", "model.", "net.", "network."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        out[nk] = v
    return out


def _extract_state_dict(maybe_ckpt):
    if isinstance(maybe_ckpt, dict):
        obj = maybe_ckpt
        for _ in range(6):
            if not isinstance(obj, dict):
                break
            found = None
            for key in [
                "state_dict",
                "model_state_dict",
                "model",
                "net",
                "network",
                "ema_state_dict",
                "ema",
                "teacher",
                "student",
            ]:
                if key in obj and isinstance(obj[key], dict):
                    found = obj[key]
                    break
            if found is None:
                break
            obj = found
        return obj
    return maybe_ckpt


def load_weights_safely(model: nn.Module, weight_path: str) -> bool:
    state = torch.load(weight_path, map_location="cpu")
    state = _extract_state_dict(state)
    if not isinstance(state, dict):
        return False

    state = _normalize_state_dict_keys(state)
    model_sd = model.state_dict()

    filtered = {}
    for k, v in state.items():
        if k in model_sd and hasattr(v, "shape") and v.shape == model_sd[k].shape:
            filtered[k] = v

    critical_keys = [
        "rg_cls.weight",
        "rg_cls.bias",
        "cls_cls.weight",
        "cls_cls.bias",
        "ord_cls.weight",
        "ord_cls.bias",
    ]
    critical_ok = all(
        (k in filtered) and (k in model_sd) and (filtered[k].shape == model_sd[k].shape)
        for k in critical_keys
    )

    load_ratio = len(filtered) / max(1, len(model_sd))

    if (load_ratio < 0.55) and (not critical_ok):
        print(
            f"Weight candidate rejected (load_ratio={load_ratio:.3f}, critical_ok={critical_ok})."
        )
        return False

    model.load_state_dict(filtered, strict=False)
    print(
        f"Loaded weights: {weight_path} (load_ratio={load_ratio:.3f}, critical_ok={critical_ok})"
    )
    return True


def find_weight_file(preferred_names, search_roots):
    candidates = []
    for r in search_roots:
        if not (r and os.path.exists(r)):
            continue
        for nm in preferred_names:
            for cand in [
                os.path.join(r, nm),
                os.path.join(r, "weights", nm),
                os.path.join(r, "checkpoints", nm),
                os.path.join(r, "models", nm),
            ]:
                if os.path.isfile(cand):
                    candidates.append(cand)

    recursive_roots = ["/kaggle/data", "/kaggle/input", "../data", "../input"]
    for rr in recursive_roots:
        if rr and os.path.exists(rr):
            for nm in preferred_names:
                candidates.extend(glob.glob(os.path.join(rr, "**", nm), recursive=True))

    if candidates:
        candidates = sorted(set(candidates), key=lambda p: (len(p), p))
        return candidates[0]

    basenames = {os.path.splitext(nm)[0] for nm in preferred_names}
    hits = []
    for rr in recursive_roots:
        if rr and os.path.exists(rr):
            for ext in [".pkl", ".pth", ".pt", ".ckpt"]:
                for b in basenames:
                    hits.extend(
                        glob.glob(os.path.join(rr, "**", b + ext), recursive=True)
                    )
    if hits:
        hits = sorted(set(hits), key=lambda p: (len(p), p))
        return hits[0]
    return None


input_root = first_existing_path(BASE_DIR_CANDIDATES) or "/kaggle/input"
comp_root = os.path.join(input_root, COMP_SLUG)

test_csv_path = first_existing_path(
    [
        os.path.join(comp_root, "test.csv"),
        os.path.join(input_root, "test.csv"),
        os.path.join("/kaggle/data", COMP_SLUG, "test.csv"),
        os.path.join("/kaggle/data", "aptos2019-blindness-detection", "test.csv"),
        "../input/aptos2019-blindness-detection/test.csv",
        "../data/aptos2019-blindness-detection/test.csv",
        "../input/test.csv",
    ]
)

test_images_dir = first_existing_path(
    [
        os.path.join(comp_root, "test_images"),
        os.path.join(input_root, "test_images"),
        os.path.join("/kaggle/data", COMP_SLUG, "test_images"),
        os.path.join("/kaggle/data", "aptos2019-blindness-detection", "test_images"),
        "../input/aptos2019-blindness-detection/test_images",
        "../data/aptos2019-blindness-detection/test_images",
        "../input/test_images",
    ]
)

if test_csv_path is None or test_images_dir is None:
    raise FileNotFoundError(
        f"Could not locate test.csv or test_images directory. "
        f"test_csv_path={test_csv_path}, test_images_dir={test_images_dir}, input_root={input_root}"
    )

test_df = pd.read_csv(test_csv_path)
test_df["id_code"] = test_df["id_code"].astype(str)
test_ids = test_df["id_code"].values

transform1 = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

transform2 = transforms.Compose(
    [
        transforms.Resize((280, 280)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

net1 = backboneNet_efficient().to(device)

PREFERRED_NAMES = [
    "B4_3stage_12epoch_320.pkl",
    "B4_3stage_12epoch_320.pth",
    "B4_3stage_12epoch_320.pt",
    "B4_3stage_12epoch_320.ckpt",
    "b4_3stage_12epoch_320.pkl",
    "b4_3stage_12epoch_320.pth",
    "b4_3stage_12epoch_320.pt",
    "b4_3stage_12epoch_320.ckpt",
    "B4_3stage.pkl",
    "B4_3stage.pth",
    "B4_3stage.pt",
    "B4_3stage.ckpt",
]

search_roots = [
    os.path.join("/kaggle/data", COMP_SLUG),
    os.path.join("/kaggle/data", "aptos2019-blindness-detection"),
    os.path.join("/kaggle/data", "input", COMP_SLUG),
    os.path.join("/kaggle/data", "input", "aptos2019-blindness-detection"),
    os.path.join("/kaggle/working", COMP_SLUG),
    os.path.join("/kaggle/working", "aptos2019-blindness-detection"),
    input_root,
    comp_root,
]

weight_path = find_weight_file(PREFERRED_NAMES, search_roots)

loaded_ok = False
if weight_path is not None and os.path.isfile(weight_path):
    loaded_ok = load_weights_safely(net1, weight_path)

if not loaded_ok:
    warnings.warn(
        f"Could not load a compatible checkpoint (weight_path={weight_path}). "
        f"Continuing with untrained weights to produce a valid submission (expected very low score)."
    )

net1.eval()

print(f"device={device}")
print(f"test_csv_path={test_csv_path}")
print(f"test_images_dir={test_images_dir}")
print(f"weight_path={weight_path}")
print(f"loaded_ok={loaded_ok}")



## === cell 6
submission = []
missing_images = 0

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print(f"{i}/{len(test_ids)}")

        image_name = os.path.join(test_images_dir, f"{idx}.png")

        if not os.path.exists(image_name):
            missing_images += 1
            continue

        img1 = Image.open(image_name).convert("RGB")
        img1 = transform1(img1).unsqueeze(0).to(device)

        r_out, c_out, o_out = net1(img1)
        P = combine3output(r_out.view(-1), c_out, o_out)

        submission.append([idx, P])

print("missing_images:", missing_images)
print("raw_predictions:", len(submission), "of", len(test_ids))



## === cell 7
if len(submission) == 0:
    raise RuntimeError(
        "Submission is empty; inference did not run or test_ids was empty."
    )

pred_df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
pred_df["id_code"] = pred_df["id_code"].astype(str)

df = test_df[["id_code"]].merge(pred_df, on="id_code", how="left")

if df["diagnosis"].isna().any():
    n_missing = int(df["diagnosis"].isna().sum())
    warnings.warn(
        f"{n_missing} test ids have no prediction (likely missing images). "
        f"Filling with 0 to keep a valid submission (may reduce score if widespread)."
    )
    df["diagnosis"] = df["diagnosis"].fillna(0)

df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

print(
    "Pred label distribution:",
    df["diagnosis"].value_counts(dropna=False).sort_index().to_dict(),
)

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
