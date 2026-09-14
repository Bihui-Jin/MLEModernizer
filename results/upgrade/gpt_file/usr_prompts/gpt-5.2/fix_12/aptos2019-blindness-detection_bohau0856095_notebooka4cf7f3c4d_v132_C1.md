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

0.9105275162245642

# 6. Current score

0.02882

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing weights issue by loading from the Kaggle dataset paths you actually have, with a safe fallback to CPU-friendly inference if the weights aren’t present (so a non-empty valid submission is always produced). I also fix the “no NVIDIA driver” crash by selecting `cuda` only when available and by removing hardcoded `.cuda()` usage in helper functions. Finally, I ensure test image paths resolve correctly in this environment and that `submission.csv` is written with the exact required columns and row count.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with running inference with random (untrained) weights or with the “weights not found” fallback that predicts all zeros, so the smallest meaningful step toward the 0.91 target is to reliably load a real checkpoint if present and use the model’s full 3-head output fusion (already defined) instead of only the regression head. I keep your architecture and inference loop intact, but (1) broaden the checkpoint search to include common Kaggle working/input locations, (2) make state-dict loading more robust (handle nested keys like `model`, `net`, etc.), and (3) when weights are loaded, use `combine3output(r_out, c_out, o_out)` so predictions match the intended “three-stage” semantics that typically improves quadratic kappa vs regression-only rounding. If no weights exist, it still produce a valid submission.csv (all zeros), but with weights it should move the score substantially upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from the `weights_loaded == False` fallback that predicts all zeros, so the smallest meaningful move toward the 0.91 target is to make checkpoint discovery/loading succeed reliably (without changing your model or inference semantics). I expand the checkpoint search to include the actual dataset/working directories you have, and make the state-dict extraction more robust (handle common wrapper keys and “model.” prefixes) so `load_state_dict` actually populates weights. I also make `combine3output` work in true batch mode (your current version effectively assumes batch size 1 and can silently mis-handle tensors), while keeping the same 3-head fusion logic. If no checkpoint exists, it still produce a valid `submission.csv` (all zeros), but with a found checkpoint it should increase your score substantially toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the model is still running with the `weights_loaded == False` fallback (all-zero predictions), so the smallest meaningful step toward your 0.9105 target is to make checkpoint discovery/loading succeed reliably without changing your model or inference loop. I (1) expand the weight search to include any `.pth/.pt/.pkl` under the actual dataset root you detected and `/kaggle/input` (common for attached datasets), (2) make state-dict key sanitization robust without incorrectly stripping `backbone.` (which can silently prevent loading the right keys), and (3) ensure we only set `weights_loaded=True` when at least some tensors match, so you don’t “think” weights loaded when they didn’t. With real weights loaded, your existing `combine3output` path should move QWK substantially upward toward the target while keeping core logic intact.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still almost certainly coming from the `weights_loaded == False` branch (predicting all zeros), so the smallest change that should move you toward 0.91 is to make weight loading succeed more often and to verify it actually loaded meaningful tensors. I expand checkpoint discovery to include common EfficientNet/3-stage filenames (not just “B4_3stage”), and I make the state-dict extraction handle common nesting (`model_state_dict`, `state_dict`) plus key prefixes (`module.`, `backbone.`) safely so more checkpoints match your model. Finally, I only set `weights_loaded=True` if a non-trivial fraction of parameters were actually loaded, avoiding a silent partial-load that still behaves like random weights; the inference and `combine3output` fusion stay unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score indicates you’re still effectively submitting the all-zero fallback, so the smallest move toward the 0.9105 target is to (1) make weight discovery/loading succeed more reliably and (2) ensure the “weights loaded” flag only turns on when the load is truly meaningful. I keep your model, transforms, and inference loop intact, but I improve checkpoint parsing to handle more real-world saved formats (including `DataParallel`/Lightning prefixes and nested dicts) and I attempt a second pass load after sanitizing keys to maximize matches. Finally, I harden `combine3output` to always work batch-wise and clamp predictions to `[0,4]` (legitimate for this metric) to avoid any out-of-range artifacts that can hurt QWK.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score indicates you’re still hitting the all-zero fallback because no real checkpoint is being loaded, so the smallest step toward the 0.9105 target is to (1) reliably find and load a compatible checkpoint (if present) and (2) avoid “false positive” loads where almost nothing matches your model’s parameters. I keep your model, transforms, and inference logic the same, but improve checkpoint selection by prioritizing filenames that match your exact architecture (“ThreeStage/B4/3stage”) and ensure we only accept a checkpoint when a meaningful fraction of tensors actually load. I also make the state-dict sanitization safer for this exact model (handle common wrappers/prefixes without stripping `backbone.` incorrectly) and print the top candidate paths to make debugging on Kaggle straightforward. If no checkpoint exists in your environment, the script still finish and write a valid `submission.csv`, but when a checkpoint is present it should move the score up substantially from 0.0 toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with your code selecting a non-model `.pth/.pt/.pkl` (or an incompatible checkpoint) as the “top candidate”, causing `weights_loaded=False` and triggering the all-zero fallback. I make the smallest change that increases the chance of loading the correct weights: only consider checkpoint files that actually look like PyTorch model checkpoints (state-dict-like) and that can load a meaningful fraction of tensors into your existing `ThreeStage_Model`. This preserves your architecture and inference semantics, but prevents “false” candidates (e.g., optimizer states, pickles, other competition artifacts) from blocking real weight loading. If no compatible checkpoint exists in your environment, it still produce a valid `submission.csv` (all zeros), but if one exists, this should move your score upward toward the target.'
- What this solution (achieved -0.05081) has done: 'Your 0.0 score is coming from the `weights_loaded == False` branch, which still happens when no compatible checkpoint is found; the smallest change that can move you toward the 0.9105 target is to stop depending on external weights and instead use the pretrained EfficientNet backbone weights that are available via `timm` in this environment. This preserves your exact model architecture, heads, transforms, and inference loop, but makes `weights_loaded` effectively true by initializing the backbone with pretrained weights so predictions are no longer all zeros. I also add a safe fallback for missing/corrupt images to avoid breaking submission generation, and keep the submission schema unchanged. This should substantially increase QWK from 0.0 (all zeros) toward your target without changing evaluation semantics.'
- What this solution (achieved -0.06462) has done: 'Your current negative QWK strongly suggests the random (untrained) heads are dominating despite the pretrained backbone, so we should keep your exact model/inference logic but make the prediction more stable and less noisy. I add a minimal test-time augmentation ensemble (original + horizontal flip) and average the *three heads’* outputs before your existing `combine3output`, which often improves kappa without changing architecture or training. I also switch inference to a DataLoader batch loop for consistent tensor shapes and faster, more stable execution (same transforms, same model). Finally, I ensure the submission order matches `test.csv` exactly and keep the output schema unchanged.'
- What this solution (achieved 0.02882) has done: 'Your negative QWK suggests inference is effectively uncalibrated (random heads) and the current fusion is mixing *argmax of ordinal probabilities* incorrectly, which can actively hurt kappa. I keep your exact model and inference flow, but fix `combine3output` so the ordinal head is converted to a class using its intended semantics (counting sigmoid outputs > 0.5), and I make the classifier vote use `softmax` for numerical stability. I also switch the fusion to a simple majority vote among the three discrete predictions (with regression used as a tie-breaker), which is a minimal post-processing change aligned to QWK’s discrete labels and typically improves stability without changing training/architecture. Everything else (paths, transforms, TTA, submission format) stays the same and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import time
import math
import glob
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT

from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
import torch
import torch.nn as nn
import torchvision
import csv
import timm
import copy
import json
import pickle

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, models, datasets
from PIL import ImageEnhance, ImageOps
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
from torch import Tensor, Size
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
        self.act1 = net.act1
        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]
        self.conv_head = net.conv_head
        self.bn2 = net.bn2
        self.act2 = net.act2
        self.global_pool = net.global_pool
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
    if isinstance(out, torch.Tensor):
        t = out
        if t.ndim == 0:
            t = t.view(1)
        if t.ndim == 2 and t.size(1) == 1:
            t = t.squeeze(1)
        preds = torch.zeros(t.size(0), device=t.device, dtype=torch.long)
        for i in range(4):
            preds += (t >= threshold[i]).long()
        return preds
    else:
        prediction = 0
        for i in range(4):
            prediction += int(out >= threshold[i])
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
    R = regress2class(r_out.detach())  # (B,)

    C = torch.softmax(c_out.detach(), dim=1).argmax(dim=1).long()  # (B,)

    O = (o_out.detach() > 0.5).sum(dim=1).long()  # (B,)

    P = torch.empty_like(R)
    for i in range(R.size(0)):
        a, b, c = int(R[i].item()), int(C[i].item()), int(O[i].item())
        if a == b or a == c:
            P[i] = a
        elif b == c:
            P[i] = b
        else:
            P[i] = a  # tie among three different labels -> fall back to regression
    P = torch.clamp(P, 0, 4)
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

        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
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
        return image  # ensure image is always returned


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
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection",
    "/kaggle/data",
    "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(os.path.join(p, "test.csv")) and os.path.exists(
        os.path.join(p, "test_images")
    ):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find aptos2019-blindness-detection dataset directory with test.csv and test_images."
    )

test_csv_path = os.path.join(DATA_ROOT, "test.csv")
test_img_dir = os.path.join(DATA_ROOT, "test_images")

test_ids_df = pd.read_csv(test_csv_path)
test_ids = np.squeeze(test_ids_df["id_code"].values)

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

net1 = ThreeStage_Model().to(device)
net1.eval()

SEARCH_DIRS = [
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/working",
    "../input",
    DATA_ROOT,
]
SEARCH_PATTERNS = ["**/*.pth", "**/*.pt", "**/*.pkl"]

WEIGHTS_CANDIDATES = []
for sd in SEARCH_DIRS:
    if os.path.exists(sd):
        for pat in SEARCH_PATTERNS:
            WEIGHTS_CANDIDATES.extend(glob.glob(os.path.join(sd, pat), recursive=True))

_seen = set()
WEIGHTS_CANDIDATES = [p for p in WEIGHTS_CANDIDATES if not (p in _seen or _seen.add(p))]


def _weight_priority(path: str) -> tuple:
    bn = os.path.basename(path).lower()
    score = 0
    if any(s in bn for s in ["3stage", "three"]):
        score -= 10
    if "b4" in bn:
        score -= 6
    if "efficientnet" in bn or "effnet" in bn:
        score -= 2
    if any(s in bn for s in ["aptos", "blind", "retina", "dr"]):
        score -= 2
    if any(s in bn for s in ["best", "epoch", "fold"]):
        score -= 1
    ext_pref = 0 if bn.endswith((".pth", ".pt")) else 1
    return (ext_pref, score, len(path))


WEIGHTS_CANDIDATES_SORTED = sorted(
    [p for p in WEIGHTS_CANDIDATES if os.path.isfile(p)], key=_weight_priority
)

print(
    "Found", len(WEIGHTS_CANDIDATES_SORTED), "checkpoint-like files. Top 10 candidates:"
)
for p in WEIGHTS_CANDIDATES_SORTED[:10]:
    print(" -", p)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "ema_state_dict",
        ]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _sanitize_state_dict_keys(state, model):
    if not isinstance(state, dict):
        return state
    model_keys = set(model.state_dict().keys())
    new_state = {}
    for k, v in state.items():
        nk = k
        for pref in ("module.", "model.", "net.", "encoder."):
            if nk.startswith(pref):
                nk2 = nk[len(pref) :]
                if nk2 in model_keys:
                    nk = nk2
        if nk.startswith("model.") and nk[len("model.") :] in model_keys:
            nk = nk[len("model.") :]
        new_state[nk] = v
    return new_state


def _try_load_state(model, state):
    if not isinstance(state, dict):
        return False, 0.0, ([], [])
    missing, unexpected = model.load_state_dict(state, strict=False)
    total = len(model.state_dict())
    loaded = total - len(missing)
    frac = loaded / max(1, total)
    return True, frac, (missing, unexpected)


def _is_plausible_checkpoint(path: str, model: nn.Module, min_frac: float = 0.60):
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        return (False, 0.0, None)
    state = _extract_state_dict(ckpt)
    if not isinstance(state, dict):
        return (False, 0.0, None)
    ok, frac, _ = _try_load_state(model, state)
    if (not ok) or (frac < min_frac):
        state2 = _sanitize_state_dict_keys(state, model)
        ok2, frac2, _ = _try_load_state(model, state2)
        if ok2 and frac2 > frac:
            state = state2
            frac = frac2
    return (frac >= min_frac, float(frac), state)


weights_path = None
best_frac = -1.0

MAX_VALIDATE = 40
for p in WEIGHTS_CANDIDATES_SORTED[:MAX_VALIDATE]:
    plausible, frac, _ = _is_plausible_checkpoint(p, net1, min_frac=0.60)
    if plausible and frac > best_frac:
        best_frac = frac
        weights_path = p

print("Selected weights_path:", weights_path, "| best_loaded_frac:", best_frac)

weights_loaded = False
loaded_frac = 0.0
if weights_path is not None:
    try:
        ckpt = torch.load(weights_path, map_location="cpu")
        state = _extract_state_dict(ckpt)

        ok, frac, (missing, unexpected) = _try_load_state(net1, state)

        if (not ok) or (frac < 0.60):
            state2 = _sanitize_state_dict_keys(state, net1)
            ok2, frac2, (missing2, unexpected2) = _try_load_state(net1, state2)
            if ok2 and frac2 >= frac:
                frac, missing, unexpected = frac2, missing2, unexpected2

        loaded_frac = frac
        weights_loaded = loaded_frac >= 0.60

        print(
            f"Checkpoint load report: loaded_frac={loaded_frac:.3f}, missing={len(missing)}, unexpected={len(unexpected)}"
        )
    except Exception as e:
        print("Failed to load weights from:", weights_path)
        print("Exception:", repr(e))
        weights_loaded = False

net1 = net1.to(device)
net1.eval()

if not weights_loaded:
    print(
        "No compatible external weights found; using timm pretrained backbone + random heads."
    )

print("DATA_ROOT:", DATA_ROOT)
print("Using device:", device)
print(
    "Weights loaded:",
    weights_loaded,
    "| loaded_frac:",
    loaded_frac,
    "| path:",
    weights_path,
)




## === cell 6
class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        path = os.path.join(self.img_dir, f"{idx}.png")
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = None
        if img is None:
            return idx, None
        x = self.transform(img)  # (C,H,W)
        return idx, x


def _collate(batch):
    ids = [b[0] for b in batch]
    xs = [b[1] for b in batch]
    ok_mask = [x is not None for x in xs]
    if any(ok_mask):
        x_tensor = torch.stack([x for x in xs if x is not None], dim=0)
        ok_ids = [ids[j] for j, ok in enumerate(ok_mask) if ok]
    else:
        x_tensor = None
        ok_ids = []
    bad_ids = [ids[j] for j, ok in enumerate(ok_mask) if not ok]
    return ids, ok_ids, bad_ids, x_tensor


test_ds = TestDataset(test_ids, test_img_dir, transform1)
test_loader = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    collate_fn=_collate,
)

pred_map = {}

net1.eval()
with torch.no_grad():
    seen = 0
    for batch_i, (all_ids, ok_ids, bad_ids, x) in enumerate(test_loader):
        seen += len(all_ids)
        if seen % 50 == 0 or batch_i == 0:
            print("Predicting", min(seen, len(test_ds)), "/", len(test_ds))

        for bid in bad_ids:
            pred_map[str(bid)] = 0

        if x is None:
            continue

        x = x.to(device, non_blocking=True)

        c1, r1, o1 = net1(x)
        x_flip = torch.flip(x, dims=[3])
        c2, r2, o2 = net1(x_flip)

        c_out = (c1 + c2) / 2.0
        r_out = (r1 + r2) / 2.0
        o_out = (o1 + o2) / 2.0

        P = (
            combine3output(r_out, c_out, o_out)
            .detach()
            .cpu()
            .numpy()
            .astype(int)
            .tolist()
        )
        for iid, pp in zip(ok_ids, P):
            pred_map[str(iid)] = int(pp)

submission = []
for idx in test_ids:
    submission.append([str(idx), int(pred_map.get(str(idx), 0))])

submission = np.array(submission, dtype=object)



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

assert (
    len(df) == len(test_ids) and len(df) > 0
), "Submission DataFrame is empty or wrong length."

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", df.shape)
print(df.head())
