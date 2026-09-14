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

0.919280522017301

# 6. Current score

0.24758

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the two execution blockers: (1) make the code run on CPU when no GPU driver is available, and (2) remove the missing external weights dependency by loading weights only if they exist and otherwise running with randomly initialized weights so a valid submission CSV is always produced. I also correct a couple of small but real runtime bugs in the transforms/helpers (`is` vs `==`, `trim()` returning `None`, and a device mixup in `ordinal2class_prob`) that can crash or silently misbehave. Finally, I speed up inference within the same core logic by using a DataLoader instead of per-image loops, ensuring it finishes under the time limit and writes `submission.csv` with the required columns.'
- What this solution (achieved 0.03158) has done: 'Your current 0.0 score is consistent with running the model with random weights (because the external weight file is missing), so the smallest legitimate way to move toward the 0.919 target is to load meaningful pretrained weights that are available inside the Kaggle image environment. I keep your exact model architecture and inference logic, but change `pretrained=False` to `pretrained=True` for the EfficientNet backbones so timm loads built-in ImageNet weights (no external files). I also keep the existing “load competition weights if present” branch, so if you later add the original `.pkl` it still take precedence. This should substantially improve predictions versus random initialization while preserving the same thresholds and submission format.'
- What this solution (achieved 0.13356) has done: 'Your current score is far below the target, so the smallest safe way to move it upward (without changing your model architecture or training loops) is to (1) use the model’s intended 3-head fusion at inference instead of only the regressor head, and (2) align the input preprocessing to EfficientNet’s pretrained normalization rather than a custom mean/std that likely mismatches the backbone. These two changes preserve your core model and thresholds, but typically improve calibration and class agreement for quadratic weighted kappa. I also keep your “load competition weights if present” behavior unchanged, and only touch inference-time details plus a small DataLoader stability tweak. The script still runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.09835) has done: 'Your current score (0.13356) is far below the target (0.91928), so we should cautiously improve performance without changing the model or training logic. The smallest high-impact fix is to correct the ordinal-head postprocessing: your model outputs 4 sigmoid logits intended for ordinal probabilities, but the code currently uses `argmax` directly on those 4 values and then clamps, which is not a valid 5-class mapping and severely harms kappa. I keep the same 3-head averaging fusion, but convert the ordinal outputs into a proper 5-class prediction using the existing `ordinal2class_prob()` (already defined) and then argmax over 5 classes. Everything else (architecture, thresholds, weights-loading behavior, transforms, and submission format) stays the same, and it still writes `submission.csv`.'
- What this solution (achieved 0.09835) has done: 'Your score is far below the target, so we should improve inference correctness without changing the model architecture or training approach. The biggest remaining issue is that `ordinal2class_prob()` currently applies a softmax to values that are already a valid 5-class probability decomposition, which distorts the ordinal head signal and can hurt kappa; removing that softmax is a minimal, semantics-preserving fix. Additionally, the ordinal conversion should be numerically safe by clamping to [0,1] and renormalizing, ensuring stable probabilities. Everything else (model, weights-loading behavior, transforms, thresholds, fusion, and submission format) stays the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.10687) has done: 'Your current score is far below the target, so we should improve prediction correctness without changing your model architecture or training approach. The biggest issue left is that you never use `transform2` (256 crop) even though the model definition strongly suggests it was trained/inferred at 256 (and you even defined a second pipeline for it); using the mismatched 288x384 preprocessing can heavily hurt kappa. I switch the test dataset to use `transform2` (keeping the same EfficientNet/ImageNet normalization) and keep everything else (model, weights-loading behavior, 3-head fusion, thresholds, ordinal conversion, submission writing) identical. I also set deterministic cuDNN flags for stability (no metric change intended, just reproducibility).'
- What this solution (achieved 0.10687) has done: 'Your score is far below the target, so we should improve prediction correctness without changing the model, its heads, thresholds, or the inference fusion strategy. The most likely high-impact bug is in `backboneNet_efficient.forward`: it incorrectly feeds `x1` (pre-BN/activation) into `block0` instead of `x3`, which breaks the EfficientNet feature flow and harms outputs. Even though your current inference uses `ThreeStage_Model`, this broken backbone can still be used elsewhere and it’s a legitimate core-correctness fix with negligible risk. I apply that one-line fix and keep everything else (preprocessing, fusion, ordinal conversion, submission writing) identical to preserve evaluation semantics while nudging performance upward.'
- What this solution (achieved 0.15889) has done: 'Your current score (0.10687) is far below the target (0.91928), so we should improve prediction correctness with the smallest change that preserves your model and inference fusion logic. The main remaining mismatch is that `transform2` does not include the same border-trimming and 4:3 center-cropping you used in `transform1`, which can shift retina placement and strongly degrade a pretrained EfficientNet’s predictions. I minimally modify `transform2` to include `trim()` and `cropTo4_3()` before the 256 crop, keeping the same final input size (256) and the same ImageNet normalization. Everything else (model, weights-loading behavior, 3-head fusion, thresholds, ordinal conversion, submission writing) stays unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 0.08006) has done: 'Your current score (0.15889) is far below the target (0.91928), so we should improve inference correctness while keeping your exact model and heads unchanged. The smallest high-impact fix is to load the model’s own learned head weights (not just the timm backbone) by relaxing `load_state_dict(strict=True)` to `strict=False` when a competition checkpoint is found, because many Kaggle checkpoints include extra/mismatched keys and strict loading can silently prevent using them (or force you to skip loading entirely). Additionally, if no competition weights exist, we should explicitly run the model in “backbone-only pretrained” mode but avoid a common calibration error by averaging *probabilities* (regression-derived class-prob + softmax class-prob + ordinal class-prob) instead of averaging hard class indices; this preserves your 3-head fusion idea while aligning better with QWK. These changes are inference-only, keep the architecture/loss/training untouched, and still write a valid `submission.csv`.'
- What this solution (achieved 0.24734) has done: 'Your current score is far below the target, so we should increase performance with the smallest changes that keep your model/heads and inference fusion intact. The biggest remaining issue is that when the competition checkpoint is missing, you are effectively using an ImageNet backbone with randomly initialized heads, which produces near-random kappa. I keep your exact architecture and prediction fusion, but add a lightweight “test-time calibration” step: fit 4 optimal thresholds on a small held-out split of the training set to map the fused continuous expected class to final integers, which is directly aligned with QWK. This doesn’t change training loops/loss/model; it only tunes post-processing thresholds and should move the score materially toward the target while still writing a valid `submission.csv`.'
- What this solution (achieved 0.24758) has done: 'Your current score (0.247) is far below the target (0.919), so we should improve toward the target with minimal, inference-only changes that keep your model/heads and fusion intact. The biggest correctness gap is that threshold-tuning is done on a random validation split but then applied to test without any distribution alignment; we can make it substantially more robust by (1) fitting thresholds using out-of-fold predictions (same model, no training) and (2) using a slightly more thorough but still fast coordinate-ascent search. I also fix one small inefficiency/bug risk in `regress2class_prob` by vectorizing it (same semantics) so threshold fitting can evaluate many candidates quickly within the time limit. Everything else (architecture, preprocessing, 3-head probability fusion, and submission format) stays the same, and the script still writes `submission.csv`.'
- What this solution (achieved 0.24758) has done: 'Your current score (0.2476) is far below the target (0.9193), so we should improve it with minimal, inference-only changes that preserve your model and fusion logic. The biggest remaining calibration gap is that your threshold tuning uses OOF predictions from a model in `eval()` mode but still with active dropout inside the backbone, which makes OOF predictions noisy/inconsistent and hurts QWK; we disable dropout deterministically at inference by setting `net1.backbone.drop_rate = 0.0` and `net1.backbone.drop_path_rate = 0.0`. Then, because thresholds depend on the scale of the “expected class” signal, we also adjust the initial threshold guess from `[0.75,1.5,2.5,3.5]` to `[0.5,1.5,2.5,3.5]` (same semantics, just a better starting point) to help the coordinate search converge to a better solution without changing the search method. Everything else (architecture, transforms, 3-head probability fusion, OOF fitting, and submission writing) stays the same and still produces `submission.csv`.'
- What this solution (achieved 0.24758) has done: 'Your score is far below the target, so the smallest safe way to move it upward is to make the threshold-fitting step more aligned with the final test-time prediction distribution without changing the model or fusion logic. I keep your exact model, transforms, and 3-head probability averaging, but change threshold tuning to use a simple “blend” of (a) out-of-fold train expected-class predictions and (b) test expected-class predictions (unlabeled) so the thresholds are calibrated to the test distribution. This is inference-only post-processing (no label leakage from test) and commonly improves QWK in this competition. I also keep the original OOF-only thresholds as a fallback and choose whichever gives better OOF QWK to avoid hurting score.'

# 9. Code solution

## === cell 0
import random
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
import timm

import os
import warnings

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
import torch
import torch.nn as nn
import torchvision
import csv
import timm
import glob
import copy
import json
import pickle
from torch.optim import lr_scheduler
from timm.models import *
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, models, datasets
from PIL import Image, ImageEnhance, ImageOps
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
from torch.nn.parameter import Parameter


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
    (kept as in original code).
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
        net = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
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
threshold = [0.5, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = 0
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu().item()
    return prediction


def ordinal2class_prob(out):
    out = out.clamp(0.0, 1.0)
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    pred_prob = pred_prob.clamp_min(0.0)
    pred_prob = pred_prob / (pred_prob.sum(dim=1, keepdim=True).clamp_min(1e-12))
    return pred_prob


def regress2class_prob(out):
    out = out.clamp(0.0, 4.0)
    b = out.size(0)
    pred_prob = torch.zeros((b, 5), device=out.device, dtype=out.dtype)

    l1 = torch.floor(out).to(torch.long)  # (B,)
    l2 = torch.ceil(out).to(torch.long)  # (B,)

    frac = (out - l1.to(out.dtype)).clamp(0.0, 1.0)  # (B,)
    same = l1 == l2

    pred_prob[torch.arange(b, device=out.device), l1] = 1.0 - frac
    pred_prob[torch.arange(b, device=out.device), l2] += frac

    if same.any():
        idx = torch.arange(b, device=out.device)[same]
        pred_prob[idx] = 0.0
        pred_prob[idx, l1[same]] = 1.0

    pred_prob = pred_prob / (pred_prob.sum(dim=1, keepdim=True).clamp_min(1e-12))
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
        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=True)
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
DATA_DIR = "/kaggle/input/aptos2019-blindness-detection"
train_csv_path = os.path.join(DATA_DIR, "train.csv")
train_img_dir = os.path.join(DATA_DIR, "train_images")
test_csv_path = os.path.join(DATA_DIR, "test.csv")
test_img_dir = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

test_ids = test_df["id_code"].values.tolist()

transform1 = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform2 = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((280, 280)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

net1 = ThreeStage_Model()

possible_weight_paths = [
    "/kaggle/input/weights/B4_3stage_24epoch_384finetune.pkl",
    "../input/weights/B4_3stage_24epoch_384finetune.pkl",
]
weight_path = next((p for p in possible_weight_paths if os.path.exists(p)), None)

if weight_path is not None:
    state = torch.load(weight_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        state = new_state
    missing, unexpected = net1.load_state_dict(state, strict=False)
    print("Loaded competition weights (strict=False):", weight_path)
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
else:
    print(
        "Competition weights not found; using timm pretrained ImageNet weights for backbone (heads remain random)."
    )

net1 = net1.to(device)

if hasattr(net1, "backbone"):
    if hasattr(net1.backbone, "drop_rate"):
        net1.backbone.drop_rate = 0.0
    if hasattr(net1.backbone, "drop_path_rate"):
        net1.backbone.drop_path_rate = 0.0

net1.eval()




## === cell 6
class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        image_name = os.path.join(self.img_dir, f"{id_code}.png")
        img = Image.open(image_name).convert("RGB")
        img = self.transform(img)
        return id_code, img


class TrainDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_code = row["id_code"]
        y = int(row["diagnosis"])
        image_name = os.path.join(self.img_dir, f"{id_code}.png")
        img = Image.open(image_name).convert("RGB")
        img = self.transform(img)
        return img, y




## === cell 7
def _predict_expected_class_from_loader(model, loader):
    """
    Inference-only helper: preserves your 3-head fusion and returns expected class in [0,4].
    """
    model.eval()
    ys = []
    exps = []
    with torch.no_grad():
        for batch in loader:
            imgs, y = batch
            imgs = imgs.to(device, non_blocking=True)

            c_out, r_out, o_out = model(imgs)

            r_out = r_out.squeeze(1)
            R_prob = regress2class_prob(r_out)  # (B,5)

            C_prob = torch.softmax(c_out, dim=1).clamp_min(0.0)
            C_prob = C_prob / (C_prob.sum(dim=1, keepdim=True).clamp_min(1e-12))

            O_prob = ordinal2class_prob(o_out)  # (B,5)

            P_prob = (R_prob + C_prob + O_prob) / 3.0
            P_prob = P_prob / (P_prob.sum(dim=1, keepdim=True).clamp_min(1e-12))

            cls = torch.arange(5, device=P_prob.device, dtype=P_prob.dtype).view(1, -1)
            exp = (P_prob * cls).sum(dim=1)

            ys.append(torch.as_tensor(y, dtype=torch.int64))
            exps.append(exp.detach().cpu())

    ys = torch.cat(ys).cpu().numpy().astype(int)
    exps = torch.cat(exps).cpu().numpy()
    return exps, ys


def _predict_expected_class_test_from_loader(model, loader):
    """
    Change (score-related, inference-only): compute test expected-class distribution to
    calibrate thresholds to test-time prediction scale (no labels used).
    """
    model.eval()
    exps = []
    with torch.no_grad():
        for batch_ids, batch_imgs in loader:
            batch_imgs = batch_imgs.to(device, non_blocking=True)

            c_out, r_out, o_out = model(batch_imgs)

            r_out = r_out.squeeze(1)
            R_prob = regress2class_prob(r_out)  # (B,5)

            C_prob = torch.softmax(c_out, dim=1).clamp_min(0.0)
            C_prob = C_prob / (C_prob.sum(dim=1, keepdim=True).clamp_min(1e-12))

            O_prob = ordinal2class_prob(o_out)  # (B,5)

            P_prob = (R_prob + C_prob + O_prob) / 3.0
            P_prob = P_prob / (P_prob.sum(dim=1, keepdim=True).clamp_min(1e-12))

            cls = torch.arange(5, device=P_prob.device, dtype=P_prob.dtype).view(1, -1)
            exp = (P_prob * cls).sum(dim=1).detach().cpu().numpy()
            exps.append(exp)
    return np.concatenate(exps, axis=0)


def apply_thresholds(x, thr):
    thr = list(thr)
    out = np.zeros_like(x, dtype=np.int64)
    out += (x >= thr[0]).astype(np.int64)
    out += (x >= thr[1]).astype(np.int64)
    out += (x >= thr[2]).astype(np.int64)
    out += (x >= thr[3]).astype(np.int64)
    return out.clip(0, 4)


def fit_qwk_thresholds(
    x, y, init_thr=(0.75, 1.5, 2.5, 3.5), grid_step=0.02, radius=0.8, n_rounds=4
):
    thr = np.array(init_thr, dtype=np.float64)

    def score(thr_vec):
        pred = apply_thresholds(x, thr_vec)
        return cohen_kappa_score(y, pred, weights="quadratic")

    best = score(thr)
    for _ in range(n_rounds):
        for i in range(4):
            candidates = np.arange(thr[i] - radius, thr[i] + radius + 1e-9, grid_step)
            best_local_thr = thr[i]
            best_local = best
            for t in candidates:
                tmp = thr.copy()
                tmp[i] = t
                if not (tmp[0] < tmp[1] < tmp[2] < tmp[3]):
                    continue
                s = score(tmp)
                if s > best_local:
                    best_local = s
                    best_local_thr = t
            thr[i] = best_local_thr
            best = best_local
        radius = max(radius * 0.5, grid_step * 2.0)
    return thr, best


def fit_thresholds_oof(train_df, img_dir, transform, model, n_splits=5, batch_size=16):
    n = len(train_df)
    idx = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(idx)

    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    starts = np.concatenate([[0], np.cumsum(fold_sizes)[:-1]])
    folds = [idx[s : s + fs] for s, fs in zip(starts, fold_sizes)]

    oof_x = np.zeros(n, dtype=np.float32)
    oof_y = train_df["diagnosis"].values.astype(int)

    t0 = time.time()
    for fi, val_idx in enumerate(folds):
        val_df = train_df.iloc[val_idx].copy()
        val_ds = TrainDataset(val_df, img_dir, transform)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )
        vx, vy = _predict_expected_class_from_loader(model, val_loader)
        oof_x[val_idx] = vx.astype(np.float32)
        print(
            f"OOF fold {fi+1}/{n_splits}: n={len(val_idx)} done in {time.time()-t0:.1f}s"
        )

    thr_opt, qwk = fit_qwk_thresholds(oof_x, oof_y, init_thr=threshold)
    return thr_opt, qwk, oof_x, oof_y




## === cell 8
t0 = time.time()
thr_oof, qwk_oof, oof_x, oof_y = fit_thresholds_oof(
    train_df, train_img_dir, transform2, net1, n_splits=5, batch_size=16
)
print("OOF threshold tuning done in %.1fs" % (time.time() - t0))
print("Optimized thresholds (OOF):", thr_oof.tolist(), "OOF QWK:", float(qwk_oof))

test_ds_for_cal = TestDataset(test_ids, test_img_dir, transform2)
test_loader_for_cal = DataLoader(
    test_ds_for_cal,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
)
t1 = time.time()
test_x = _predict_expected_class_test_from_loader(net1, test_loader_for_cal).astype(
    np.float32
)
print("Computed test expected-class distribution in %.1fs" % (time.time() - t1))

alpha = 0.35  # small blend weight; minimal change but nudges calibration toward test
x_blend = (1.0 - alpha) * oof_x + alpha * np.random.RandomState(42).choice(
    test_x, size=oof_x.shape[0], replace=True
)

thr_blend, qwk_blend = fit_qwk_thresholds(x_blend, oof_y, init_thr=thr_oof)
print(
    "Optimized thresholds (blend):",
    thr_blend.tolist(),
    "Blend QWK proxy:",
    float(qwk_blend),
)

qwk_oof_with_oof = cohen_kappa_score(
    oof_y, apply_thresholds(oof_x, thr_oof), weights="quadratic"
)
qwk_oof_with_blend = cohen_kappa_score(
    oof_y, apply_thresholds(oof_x, thr_blend), weights="quadratic"
)
print(
    "OOF QWK using OOF thr:",
    float(qwk_oof_with_oof),
    "OOF QWK using blend thr:",
    float(qwk_oof_with_blend),
)

thr_opt = thr_blend if qwk_oof_with_blend >= qwk_oof_with_oof else thr_oof
print("Selected thresholds:", thr_opt.tolist())



## === cell 9
test_ds = TestDataset(test_ids, test_img_dir, transform2)
test_loader = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
)

submission = []
with torch.no_grad():
    for batch_ids, batch_imgs in test_loader:
        batch_imgs = batch_imgs.to(device, non_blocking=True)

        c_out, r_out, o_out = net1(batch_imgs)

        r_out = r_out.squeeze(1)
        R_prob = regress2class_prob(r_out)  # (B,5)

        C_prob = torch.softmax(c_out, dim=1).clamp_min(0.0)
        C_prob = C_prob / (C_prob.sum(dim=1, keepdim=True).clamp_min(1e-12))

        O_prob = ordinal2class_prob(o_out)  # (B,5)

        P_prob = (R_prob + C_prob + O_prob) / 3.0
        P_prob = P_prob / (P_prob.sum(dim=1, keepdim=True).clamp_min(1e-12))

        cls = torch.arange(5, device=P_prob.device, dtype=P_prob.dtype).view(1, -1)
        exp = (P_prob * cls).sum(dim=1).detach().cpu().numpy()

        P = apply_thresholds(exp, thr_opt).astype(int)

        for _id, p in zip(batch_ids, P):
            submission.append([_id, int(p)])

submission = np.array(submission, dtype=object)
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["diagnosis"] = df["diagnosis"].astype(int)
assert len(df) == len(
    test_ids
), f"Submission rows ({len(df)}) != test rows ({len(test_ids)})"
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
