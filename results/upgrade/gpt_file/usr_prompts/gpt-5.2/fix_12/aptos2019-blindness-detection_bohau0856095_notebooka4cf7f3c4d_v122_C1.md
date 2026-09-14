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

0.9296087650720126

# 6. Current score

0.67802

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the model construction crash by making the EfficientNet stem activation lookup compatible with current `timm` (it no longer exposes `act1/act` for this model) and also correct a clear forward-pass bug where blocks were fed the wrong tensor (`x1` instead of the activated stem output). Then I make weight loading robust to mismatched prefixes/keys so the notebook doesn’t stop before inference (and so `net3` is always defined). Finally, I keep the inference and submission logic the same, but add minimal safeguards so the loop actually produces rows and writes a non-empty `submission.csv` in the required format.'
- What this solution (achieved 0.0) has done: 'I make the weights loading robust so the notebook doesn’t crash when `../input/weights/2.pth` is missing, and instead falls back to using the model’s default (random) weights so a valid submission is always produced. I also fix the CUDA/CPU dtype/device mismatch that currently crashes inference by ensuring the model parameters and input tensors are on the same device and dtype. Finally, I keep your existing inference logic and submission formatting intact, only adding minimal guards to ensure all `test.csv` ids are predicted (or safely defaulted) and `submission.csv` is always written.'
- What this solution (achieved -0.13305) has done: 'Your current 0.0 score is consistent with running inference using randomly initialized weights (because `../input/weights/2.pth` doesn’t exist in the provided data paths), which yields near-random predictions. To move the score toward the 0.9296 target with minimal disruption, I keep your exact model and inference logic but load ImageNet-pretrained weights for the same EfficientNet backbone (`pretrained=True`) as a strong, legitimate fallback when the competition checkpoint is missing. I also ensure the model is put on the right device before loading the optional checkpoint (safer with some state dicts) and keep submission alignment unchanged. This should substantially improve QWK versus random without changing architecture or post-processing.'
- What this solution (achieved 0.0) has done: 'Your current score (-0.133) indicates the head layers (regressor/classifier/ordinal) are effectively random even though the backbone is ImageNet-pretrained, so we need a stronger *legitimate* signal without changing the architecture or training loops. The smallest change that typically moves QWK up a lot for this competition is to use test-time augmentation (TTA) and ensemble the three heads using their *probability outputs* (instead of hard class votes) so the final class is more stable and better calibrated. I keep your model and transforms intact, but run a few deterministic TTAs per image (flip + mild scale/crop), average the resulting class probabilities, and then take argmax to get the final `diagnosis`. This preserves evaluation semantics (still 0–4 integer predictions) while usually improving agreement toward your 0.9296 target versus near-random heads.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with using the `backboneNet_efficient` head layers at random initialization (the checkpoint is missing), even though the EfficientNet backbone is ImageNet-pretrained—so predictions are essentially untrained. To move the score strongly upward toward 0.9296 without changing your overall approach, I keep the same model and inference pipeline but load the best available APTOS-trained weights that already exist in your dataset: `../input/aptos2019-blindness-detection/efficientnet-b4.bin` (commonly used as a finetuned checkpoint for this competition). I also keep your existing `../input/weights/2.pth` as the first priority if it exists, and only fall back to ImageNet if neither checkpoint exists. This should materially improve QWK while keeping architecture, transforms, and output semantics unchanged and still producing `submission.csv` end-to-end.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is consistent with inference running with essentially random heads (even if the backbone is ImageNet-pretrained), so the smallest legitimate move toward 0.9296 is to actually use the available APTOS-trained checkpoint more reliably and in the right format. I add a robust loader that handles common checkpoint key patterns (e.g., `state_dict`, `model_state_dict`, nested keys, and mismatched prefixes) and ensures tensors are mapped correctly before `load_state_dict`, without changing the model or inference logic. I also make sure we always select the most likely finetuned weight file first (the competition-provided `efficientnet-b4.bin`) and only fall back to the missing `../input/weights/2.pth` or ImageNet. This keeps architecture, forward pass, TTA, and submission semantics unchanged, but should move the score substantially upward toward the target by avoiding the “random head” situation.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with running a model whose *head weights are not actually loaded*, so predictions collapse to something close to random/majority even though the backbone is ImageNet-pretrained. To move the score upward toward 0.9296 with minimal change and without altering the architecture or inference semantics, I (1) correctly prioritize and load the commonly-included finetuned checkpoint `efficientnet-b4.bin` by handling the frequent “raw state_dict with mismatched head names” pattern, and (2) add a very small, metric-aligned post-processing step: fit 4 thresholds on a tiny deterministic validation split of `train.csv` to convert the regression output into classes (then combine as you already do via probabilities). This keeps your model, forward pass, and TTA intact, but replaces hardcoded thresholds with data-driven ones, which typically improves QWK substantially while remaining legitimate and fast.'
- What this solution (achieved -0.02392) has done: 'Your score is 0.0 because your inference is almost certainly running with essentially random head weights (the listed finetuned checkpoint path likely doesn’t exist, and `efficientnet-b4.bin` is usually for a different wrapper), so predictions are near-random/majority. To move toward the 0.9296 target with minimal change and without changing your model/training approach, I (1) robustly load a finetuned APTOS checkpoint if present by handling common checkpoint formats and key prefixes, and (2) only if finetuned weights still can’t be loaded, fit thresholds on a small deterministic validation split and use those thresholds to discretize the regression output (metric-aligned for QWK) while keeping the same 0–4 submission semantics. I also ensure the regression-to-class probability uses the fitted thresholds (currently it ignores them), which is a small but directly score-relevant calibration fix. Everything else (architecture, TTA, averaging of three heads, submission formatting/paths) stays the same.'
- What this solution (achieved 0.59582) has done: 'Your current negative QWK strongly suggests the EfficientNet heads (`rg_cls/cls_cls/ord_cls`) are still effectively random because none of the candidate finetuned checkpoints actually exist in your input paths, so we need a minimal, legitimate source of DR-specific signal. The smallest change that preserves your overall inference approach is to add a quick “calibration” step: train only the three head layers for a single short epoch on resized images (backbone frozen) using simple, metric-aligned losses (CE for classifier, BCE for ordinal, MSE for regressor). Then keep your existing threshold fitting + TTA + probability averaging exactly as-is, but now the heads are no longer random, which should move QWK substantially upward toward the target. This stays within Kaggle constraints, finishes within the time limit by using a small image size and few batches, and still produces the same `submission.csv` format.'
- What this solution (achieved 0.67802) has done: 'Your current gap to the target (0.59582 vs 0.92961, higher-is-better) is large, so we need a meaningful but still “core-logic-preserving” improvement that stays within runtime. The main issue is that when no finetuned checkpoint loads, the heads are only lightly calibrated and your threshold fitting uses just 256 samples and only the regressor, which limits QWK. I keep your exact model, losses, and inference/TTA/averaging semantics, but (1) make head-only calibration cover a bit more data deterministically (still one epoch, backbone frozen) and (2) fit thresholds using more validation predictions and apply a tiny ordered constraint to avoid degenerate thresholds. These are minimal, metric-aligned changes that should move QWK upward toward your 0.9296 target without changing architecture or adding early stopping/sampling approximations.'

# 9. Code solution

## === cell 0
import random
import time
import math
import os
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

import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
import glob
import copy
import json
import pickle

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import utils, models, datasets
from PIL import ImageEnhance, ImageOps


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super().__init__()
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

    def __init__(self, num_features: int, eps: float = 1e-5, n: int = None):
        if n is not None:
            warnings.warn(
                "`n` argument is deprecated and has been renamed `num_features`",
                DeprecationWarning,
            )
            num_features = n
        super().__init__()
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
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        super().__init__()
        net = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)

        layers_to_train = ["blocks"]
        for name, parameter in net.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        self.num_features = 1792

        self.conv_stem = net.conv_stem
        self.bn1 = net.bn1

        act = getattr(net, "act1", None)
        if act is None:
            act = getattr(net, "act", None)
        if act is None:
            act = nn.SiLU(inplace=True)
        self.act1 = act

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]
        self.conv_head = net.conv_head
        self.bn2 = net.bn2

        act2 = getattr(net, "act2", None)
        if act2 is None:
            act2 = getattr(net, "act", None)
        if act2 is None:
            act2 = nn.SiLU(inplace=True)
        self.act2 = act2

        self.global_pool = net.global_pool
        self.drop_rate = 0.4

        self.rg_cls = Linear(self.num_features, 1, bias=True)
        self.cls_cls = Linear(self.num_features, 5, bias=True)
        self.ord_cls = Linear(self.num_features, 4, bias=True)

    def forward(self, x):
        x1 = self.conv_stem(x)
        x2 = self.bn1(x1)
        x3 = self.act1(x2)

        x4 = self.block0(x3)
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
    """
    Metric-relevant: use the (possibly fitted) global `threshold` to map regression to class probabilities.
    """
    if out.dim() == 2 and out.size(1) == 1:
        out = out.view(-1)
    out = out.clamp(0.0, 4.5)

    thr = torch.tensor(threshold, device=out.device, dtype=out.dtype)  # (4,)
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)

    edges = torch.cat(
        [
            torch.tensor([-1e9], device=out.device, dtype=out.dtype),
            thr,
            torch.tensor([1e9], device=out.device, dtype=out.dtype),
        ]
    )

    for i in range(out.size(0)):
        v = out[i]
        cls = int(torch.sum(v >= thr).item())
        cls = max(0, min(4, cls))
        left = edges[cls]
        right = edges[cls + 1]
        if cls == 0:
            dist = (right - v).clamp(min=0.0)
            w1 = torch.exp(-dist) * 0.15
            w0 = 1.0 - w1
            pred_prob[i, 0] = w0
            pred_prob[i, 1] = w1
        elif cls == 4:
            dist = (v - left).clamp(min=0.0)
            w3 = torch.exp(-dist) * 0.15
            w4 = 1.0 - w3
            pred_prob[i, 4] = w4
            pred_prob[i, 3] = w3
        else:
            dl = (v - left).clamp(min=0.0)
            dr = (right - v).clamp(min=0.0)
            leak = 0.20
            if dl < dr:
                w_neighbor = torch.exp(-dl) * leak
                w_main = 1.0 - w_neighbor
                pred_prob[i, cls] = w_main
                pred_prob[i, cls - 1] = w_neighbor
            else:
                w_neighbor = torch.exp(-dr) * leak
                w_main = 1.0 - w_neighbor
                pred_prob[i, cls] = w_main
                pred_prob[i, cls + 1] = w_neighbor

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
    if img is None:
        return img
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img




## === cell 5
BASE = "../input/aptos2019-blindness-detection"
test_df = pd.read_csv(f"{BASE}/test.csv")
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

net3 = backboneNet_efficient()
net3 = net3.to(device)


def _strip_prefix_if_present(state_dict, prefix):
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        nk = k[len(prefix) :] if k.startswith(prefix) else k
        out[nk] = v
    return out


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def _prepare_state_dict_for_net(sd):
    if not isinstance(sd, dict):
        return sd
    sd = _strip_prefix_if_present(sd, "module.")
    sd = _strip_prefix_if_present(sd, "model.")
    sd = _strip_prefix_if_present(sd, "net.")
    sd = _strip_prefix_if_present(sd, "backbone.")
    return sd


def _coerce_tensor_device(sd):
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        out[k] = v.to("cpu") if torch.is_tensor(v) else v
    return out


def _remap_common_wrapper_keys(sd, model):
    """
    Score-relevant: map common checkpoint head names to this wrapper so heads don't stay random.
    """
    if not isinstance(sd, dict):
        return sd

    mapped = dict(sd)

    for cand_w, cand_b in [
        ("_fc.weight", "_fc.bias"),
        ("fc.weight", "fc.bias"),
        ("head.weight", "head.bias"),
        ("classifier.weight", "classifier.bias"),
    ]:
        if cand_w in mapped and "cls_cls.weight" not in mapped:
            if tuple(mapped[cand_w].shape) == tuple(model.cls_cls.weight.shape):
                mapped["cls_cls.weight"] = mapped[cand_w]
        if cand_b in mapped and "cls_cls.bias" not in mapped:
            if tuple(mapped[cand_b].shape) == tuple(model.cls_cls.bias.shape):
                mapped["cls_cls.bias"] = mapped[cand_b]

    for cand_w, cand_b in [
        ("regressor.weight", "regressor.bias"),
        ("reg_head.weight", "reg_head.bias"),
        ("out.weight", "out.bias"),
    ]:
        if cand_w in mapped and "rg_cls.weight" not in mapped:
            if tuple(mapped[cand_w].shape) == tuple(model.rg_cls.weight.shape):
                mapped["rg_cls.weight"] = mapped[cand_w]
        if cand_b in mapped and "rg_cls.bias" not in mapped:
            if tuple(mapped[cand_b].shape) == tuple(model.rg_cls.bias.shape):
                mapped["rg_cls.bias"] = mapped[cand_b]

    for cand_w, cand_b in [
        ("ordinal.weight", "ordinal.bias"),
        ("ord_head.weight", "ord_head.bias"),
    ]:
        if cand_w in mapped and "ord_cls.weight" not in mapped:
            if tuple(mapped[cand_w].shape) == tuple(model.ord_cls.weight.shape):
                mapped["ord_cls.weight"] = mapped[cand_w]
        if cand_b in mapped and "ord_cls.bias" not in mapped:
            if tuple(mapped[cand_b].shape) == tuple(model.ord_cls.bias.shape):
                mapped["ord_cls.bias"] = mapped[cand_b]

    return mapped


weights_candidates = [
    f"{BASE}/efficientnet-b4.bin",
    f"{BASE}/efficientnet_b4.bin",
    f"{BASE}/model.bin",
    "../input/weights/2.pth",
]

loaded_from = None
for wp in weights_candidates:
    if os.path.exists(wp):
        try:
            raw = torch.load(wp, map_location="cpu")
            state = _extract_state_dict(raw)
            state = _prepare_state_dict_for_net(state)
            state = _remap_common_wrapper_keys(state, net3)
            state = _coerce_tensor_device(state)

            missing, unexpected = net3.load_state_dict(state, strict=False)
            loaded_from = wp
            print(f"[INFO] Loaded weights from: {wp}")
            if len(missing) > 0 or len(unexpected) > 0:
                print(
                    f"[load_state_dict] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
                )
            break
        except Exception as e:
            print(f"[WARN] Failed to load weights from {wp}: {repr(e)}")

if loaded_from is None:
    print(
        "[WARN] No external finetuned weights found/loaded; using ImageNet-pretrained backbone only (heads may remain random)."
    )

net3.eval()




## === cell 6
train_df = pd.read_csv(f"{BASE}/train.csv")

calib_transform = transforms.Compose(
    [
        transforms.Resize((280, 280)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class AptosTrainDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = row["id_code"]
        y = int(row["diagnosis"])
        image_name = f"{self.img_dir}/{idx}.png"

        img2 = cv2.imread(image_name)
        img2 = crop_image_from_gray(img2)
        if img2 is None:
            img_pil = Image.open(image_name).convert("RGB")
        else:
            img_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

        x = self.transform(img_pil)
        return x, y


def _make_ordinal_targets(y, n_ordinal=4):
    y = y.view(-1, 1)
    ks = torch.arange(1, n_ordinal + 1, device=y.device).view(1, -1)
    return (y >= ks).float()


if loaded_from is None:
    for name, p in net3.named_parameters():
        p.requires_grad_(False)
    for p in net3.rg_cls.parameters():
        p.requires_grad_(True)
    for p in net3.cls_cls.parameters():
        p.requires_grad_(True)
    for p in net3.ord_cls.parameters():
        p.requires_grad_(True)

    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.65, random_state=42)
    keep_idx, _ = next(splitter.split(train_df["id_code"], train_df["diagnosis"]))
    calib_df = train_df.iloc[keep_idx].reset_index(drop=True)

    calib_ds = AptosTrainDataset(calib_df, f"{BASE}/train_images", calib_transform)
    calib_loader = DataLoader(
        calib_ds,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    net3.train()
    opt = torch.optim.AdamW(
        [p for p in net3.parameters() if p.requires_grad],
        lr=2e-3,
        weight_decay=1e-4,
    )

    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    t0 = time.time()
    max_batches = 220  # Score-relevant: a bit more head updates, still bounded for runtime (<600s).
    for bi, (x, y) in enumerate(calib_loader):
        if bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        rg_out, cls_out, ord_out = net3(x)

        rg_pred = torch.sigmoid(rg_out).view(-1) * 4.5
        rg_tgt = y.float()
        loss_r = mse(rg_pred, rg_tgt)

        loss_c = ce(cls_out, y)

        ord_tgt = _make_ordinal_targets(y, n_ordinal=4)
        loss_o = bce(ord_out, ord_tgt)

        loss = 0.45 * loss_c + 0.35 * loss_o + 0.20 * loss_r
        loss.backward()
        opt.step()

    net3.eval()
    print(
        f"[INFO] Head-only calibration finished in {time.time()-t0:.1f}s on {min((bi+1), max_batches)} batches."
    )
else:
    print("[INFO] External weights loaded; skipping head calibration.")




## === cell 7
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score


def _apply_thresholds(y_reg, thr):
    thr = list(thr)
    y = np.zeros_like(y_reg, dtype=np.int64)
    for t in thr:
        y += (y_reg >= t).astype(np.int64)
    return np.clip(y, 0, 4)


def _fit_thresholds_bruteforce(y_true, y_reg):
    base = np.array([0.75, 1.5, 2.5, 3.5], dtype=np.float32)
    deltas = np.array([-0.3, -0.15, 0.0, 0.15, 0.3], dtype=np.float32)

    best_thr = base.copy()
    best_k = -1.0

    for d0 in deltas:
        for d1 in deltas:
            for d2 in deltas:
                for d3 in deltas:
                    thr = base + np.array([d0, d1, d2, d3], dtype=np.float32)
                    if not (thr[0] < thr[1] < thr[2] < thr[3]):
                        continue
                    y_pred = _apply_thresholds(y_reg, thr)
                    k = cohen_kappa_score(y_true, y_pred, weights="quadratic")
                    if k > best_k:
                        best_k = k
                        best_thr = thr.copy()
    return best_thr.tolist(), best_k


splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=42)
tr_idx, va_idx = next(splitter.split(train_df["id_code"], train_df["diagnosis"]))
val_df = train_df.iloc[va_idx].reset_index(drop=True)

val_ids = val_df["id_code"].tolist()

preds = []
ids_ok = []

for idx in val_ids[:900]:
    image_name = f"{BASE}/train_images/{idx}.png"
    if not os.path.exists(image_name):
        continue
    img2 = cv2.imread(image_name)
    img2 = crop_image_from_gray(img2)
    if img2 is None:
        img_pil = Image.open(image_name).convert("RGB")
    else:
        img_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    img_t = transform2(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        rg_out, _, _ = net3(img_t)
        rg_out = (torch.sigmoid(rg_out) * 4.5).view(-1).detach().cpu().numpy()[0]
    preds.append(rg_out)
    ids_ok.append(idx)

if len(preds) >= 100:
    y_reg = np.array(preds, dtype=np.float32)
    y_true = val_df.set_index("id_code").loc[ids_ok, "diagnosis"].values.astype(int)

    best_thr, best_k = _fit_thresholds_bruteforce(y_true, y_reg)

    best_thr = np.array(best_thr, dtype=np.float32)
    for i in range(1, 4):
        if best_thr[i] <= best_thr[i - 1] + 1e-3:
            best_thr[i] = best_thr[i - 1] + 1e-3

    threshold = best_thr.tolist()
    print(f"[INFO] Fitted thresholds (val QWK={best_k:.4f}): {threshold}")
else:
    print(
        "[WARN] Not enough validation predictions to fit thresholds; keeping defaults."
    )




## === cell 8
submission = []

cls_softmax = torch.nn.Softmax(dim=1)


def make_tta_views(pil_img):
    views = []
    views.append(pil_img)
    views.append(FT.hflip(pil_img))
    views.append(FT.center_crop(FT.resize(pil_img, [300, 300]), [256, 256]))
    views.append(FT.hflip(FT.center_crop(FT.resize(pil_img, [300, 300]), [256, 256])))
    return views


with torch.no_grad():
    for i, idx in enumerate(test_ids):
        image_name = f"{BASE}/test_images/{idx}.png"
        if not os.path.exists(image_name):
            continue

        img2 = cv2.imread(image_name)
        img2 = crop_image_from_gray(img2)
        if img2 is None:
            img2_pil = Image.open(image_name).convert("RGB")
        else:
            img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

        tta_imgs = make_tta_views(img2_pil)

        prob_sum = torch.zeros(5, device=device, dtype=torch.float32)

        for view in tta_imgs:
            if view.size == (256, 256):
                x = FT.to_tensor(view)
                x = FT.normalize(
                    x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
                img_t = x.unsqueeze(0).to(device)
            else:
                img_t = transform2(view).unsqueeze(0).to(device)

            rg_out, cls_out, ord_out = net3(img_t)

            rg_out = torch.sigmoid(rg_out) * 4.5  # (1,1) in [0,4.5]
            cls_prob = cls_softmax(cls_out)  # (1,5)

            ord_out = torch.sigmoid(ord_out)  # (1,4) in [0,1]
            ord_prob = ordinal2class_prob(ord_out)  # (1,5)

            rg_prob = regress2class_prob(rg_out.view(-1))
            if rg_prob.dim() == 1:
                rg_prob = rg_prob.unsqueeze(0)

            comb_prob = (rg_prob + cls_prob + ord_prob) / 3.0  # (1,5)
            prob_sum += comb_prob.squeeze(0)

        prob_avg = prob_sum / float(len(tta_imgs))
        P = int(torch.argmax(prob_avg).item())
        P = int(np.clip(P, 0, 4))

        submission.append([idx, P])

if len(submission) == 0:
    submission = [[idx, 0] for idx in test_ids]

submission = np.array(submission, dtype=object)




## === cell 9
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["diagnosis"] = df["diagnosis"].astype(int)

df = test_df.merge(df, on="id_code", how="left")
if df["diagnosis"].isna().any():
    df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("diagnosis value counts:\n", df["diagnosis"].value_counts().sort_index())
