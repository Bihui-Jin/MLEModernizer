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

# 5. Code solution

## === cell 0
import os
import random
import time
import math
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.backends.cudnn.benchmark = True

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
import csv
import glob
import copy
import json
import pickle

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import utils, models, datasets
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


import warnings


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

        if hasattr(net, "act1"):
            self.act1 = net.act1
        elif hasattr(net, "act_fn"):
            self.act1 = net.act_fn
        else:
            self.act1 = nn.Identity()

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]
        self.conv_head = net.conv_head
        self.bn2 = net.bn2
        self.act2 = (
            net.act2
            if hasattr(net, "act2")
            else (net.act_fn if hasattr(net, "act_fn") else nn.Identity())
        )
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
    out = out.view(-1)
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)
    out_clipped = torch.clamp(out, 0.0, 4.0)
    l1 = torch.floor(out_clipped).long()
    l2 = torch.ceil(out_clipped).long()
    w1 = 1.0 - (out_clipped - l1.to(out_clipped.dtype))
    w2 = 1.0 - (l2.to(out_clipped.dtype) - out_clipped)
    idx = torch.arange(out.size(0), device=out.device)
    pred_prob[idx, l1] = w1
    pred_prob[idx, l2] = torch.maximum(pred_prob[idx, l2], w2)
    mask4 = out >= 4.0
    if mask4.any():
        pred_prob[mask4] = 0
        pred_prob[mask4, 4] = 1.0
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


def combine3output_batch(r_out, c_out, o_out, thr_list):
    thr = torch.tensor(thr_list, device=r_out.device, dtype=r_out.dtype).view(1, 4)
    r = r_out.view(-1, 1)
    R = (r >= thr).sum(dim=1).to(torch.float32)
    C = torch.argmax(c_out, dim=1).to(torch.float32)
    O = torch.argmax(o_out, dim=1).to(torch.float32)
    P = (R + C + O) / 3.0
    P = torch.round(P).to(torch.int64)
    return P


def apply_thresholds_to_regression(r_pred: np.ndarray, thr):
    thr0, thr1, thr2, thr3 = thr
    y = np.zeros_like(r_pred, dtype=np.int64)
    y += (r_pred >= thr0).astype(np.int64)
    y += (r_pred >= thr1).astype(np.int64)
    y += (r_pred >= thr2).astype(np.int64)
    y += (r_pred >= thr3).astype(np.int64)
    return y


def tune_thresholds_for_qwk(r_pred: np.ndarray, y_true: np.ndarray, init_thr=None):
    if init_thr is None:
        init_thr = [0.75, 1.5, 2.5, 3.5]
    thr = np.array(init_thr, dtype=np.float64)

    def score(thr_arr):
        y_hat = apply_thresholds_to_regression(r_pred, thr_arr)
        return cohen_kappa_score(y_true, y_hat, weights="quadratic")

    best = score(thr)
    steps = [0.05, 0.1]
    for step in steps:
        improved = True
        it = 0
        while improved and it < 10:
            improved = False
            it += 1
            for k in range(4):
                candidates = thr[k] + step * np.array(
                    [-4, -3, -2, -1, 1, 2, 3, 4], dtype=np.float64
                )
                for cand in candidates:
                    cand_thr = thr.copy()
                    cand_thr[k] = float(cand)
                    if not (cand_thr[0] < cand_thr[1] < cand_thr[2] < cand_thr[3]):
                        continue
                    cand_thr = np.clip(cand_thr, 0.0, 4.5)
                    s = score(cand_thr)
                    if s > best + 1e-6:
                        best = s
                        thr = cand_thr
                        improved = True
    return thr.tolist(), float(best)


def tune_thresholds_for_qwk_combine(
    r_out_cpu: torch.Tensor,
    c_out_cpu: torch.Tensor,
    o_out_cpu: torch.Tensor,
    y_true: np.ndarray,
    init_thr=None,
):
    if init_thr is None:
        init_thr = [0.75, 1.5, 2.5, 3.5]
    thr = np.array(init_thr, dtype=np.float64)

    r_out = r_out_cpu.to(dtype=torch.float32, device="cpu")
    c_out = c_out_cpu.to(dtype=torch.float32, device="cpu")
    o_out = o_out_cpu.to(dtype=torch.float32, device="cpu")

    def score(thr_arr):
        pred = combine3output_batch(r_out, c_out, o_out, thr_arr).numpy()
        pred = np.clip(pred, 0, 4).astype(np.int64)
        return cohen_kappa_score(y_true, pred, weights="quadratic")

    best = score(thr)
    steps = [0.05, 0.1]
    for step in steps:
        improved = True
        it = 0
        while improved and it < 10:
            improved = False
            it += 1
            for k in range(4):
                candidates = thr[k] + step * np.array(
                    [-4, -3, -2, -1, 1, 2, 3, 4], dtype=np.float64
                )
                for cand in candidates:
                    cand_thr = thr.copy()
                    cand_thr[k] = float(cand)
                    if not (cand_thr[0] < cand_thr[1] < cand_thr[2] < cand_thr[3]):
                        continue
                    cand_thr = np.clip(cand_thr, 0.0, 4.5)
                    s = score(cand_thr)
                    if s > best + 1e-6:
                        best = s
                        thr = cand_thr
                        improved = True
    return thr.tolist(), float(best)




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
        arr = np.asarray(image)
        if arr.size == 0:
            return image
        bg = arr[0, 0].astype(np.int16)
        diff = np.abs(arr.astype(np.int16) - bg[None, None, :]).astype(np.uint8)
        diff2 = diff.astype(np.int16) * 2 - 10
        diff2 = np.clip(diff2, 0, 255).astype(np.uint8)
        mask = diff2.max(axis=2) > 0
        if not mask.any():
            return image
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        cropped = arr[y0:y1, x0:x1, :]
        return Image.fromarray(cropped, mode="RGB")




## === cell 5
CANDIDATE_BASES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
]
BASE_INPUT = None
for p in CANDIDATE_BASES:
    if os.path.exists(p):
        BASE_INPUT = p
        break
if BASE_INPUT is None:
    raise FileNotFoundError(
        f"Could not find dataset folder in any of: {CANDIDATE_BASES}"
    )
print("BASE_INPUT:", BASE_INPUT)

test_df = pd.read_csv(os.path.join(BASE_INPUT, "test.csv"))
test_ids = test_df["id_code"].astype(str).values


class CachedTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform
        self._cache = {}

    def __call__(self, img, cache_key=None):
        if cache_key is None:
            return self.base_transform(img)
        v = self._cache.get(cache_key, None)
        if v is not None:
            return v
        v = self.base_transform(img)
        self._cache[cache_key] = v
        return v


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
        trim(),
        transforms.Resize((280, 280)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

transform1_cached = CachedTransform(transform1)

net4 = backboneNet_efficient()


def find_weight_file():
    candidates = [
        "trained_net4.pth",  # local working dir from previous run
        "../input/weights/3.pth",
        "/kaggle/input/weights/3.pth",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for root in ["/kaggle/input", "/kaggle/data", "../input"]:
        if os.path.exists(root):
            hits = glob.glob(os.path.join(root, "**", "*.pth"), recursive=True)
            for h in hits:
                if os.path.basename(h) == "3.pth":
                    return h
            if len(hits) > 0:
                return hits[0]
    return None


WEIGHTS_PATH = find_weight_file()
if WEIGHTS_PATH is not None:
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        net4.load_state_dict(state["state_dict"], strict=True)
        if "thresholds" in state:
            threshold = list(state["thresholds"])
    else:
        net4.load_state_dict(state, strict=True)
    print("Loaded weights from:", WEIGHTS_PATH)
else:
    print(
        "WARNING: No weights .pth found. Will train lightweight head on train.csv and save to ./trained_net4.pth"
    )

net4 = net4.to(device)

print("Test samples:", len(test_ids))




## === cell 6
class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform, use_cache_key=False):
        self.df = df.reset_index(drop=True)
        self.ids = self.df["id_code"].astype(str).values
        self.labels = self.df["diagnosis"].astype(np.int64).values
        self.img_dir = img_dir
        self.transform = transform
        self.use_cache_key = use_cache_key

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, i):
        img_path = os.path.join(self.img_dir, f"{self.ids[i]}.png")
        img = Image.open(img_path).convert("RGB")
        if self.use_cache_key and isinstance(self.transform, CachedTransform):
            x = self.transform(img, cache_key=img_path)
        else:
            x = self.transform(img)
        y = int(self.labels[i])
        return x, y


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def make_ordinal_targets(y: torch.Tensor) -> torch.Tensor:
    y = y.view(-1, 1)
    return (y > torch.tensor([0, 1, 2, 3], device=y.device).view(1, -1)).float()


trained_path = "trained_net4.pth"

if WEIGHTS_PATH is None:
    train_csv = os.path.join(BASE_INPUT, "train.csv")
    train_df = pd.read_csv(train_csv)

    train_img_dir = os.path.join(BASE_INPUT, "train_images")
    if not os.path.exists(train_img_dir):
        alt = os.path.join(BASE_INPUT, "aptos2019-blindness-detection", "train_images")
        if os.path.exists(alt):
            train_img_dir = alt
    print("train_img_dir:", train_img_dir)

    full_ds = APTOSDataset(train_df, train_img_dir, transform2)

    g = torch.Generator().manual_seed(42)
    n = len(full_ds)
    n_val = max(300, int(0.15 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    num_workers = 4 if os.cpu_count() and os.cpu_count() >= 4 else 2
    common_dl_kwargs = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        common_dl_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        **common_dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        **common_dl_kwargs,
    )

    y_train_split = full_ds.labels[np.asarray(train_ds.indices, dtype=np.int64)]
    label_counts = (
        pd.Series(y_train_split)
        .value_counts()
        .reindex([0, 1, 2, 3, 4])
        .fillna(0)
        .values
    )
    class_weights = label_counts.sum() / (label_counts + 1e-6)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(
        [p for p in net4.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4
    )
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights_t)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()

    best_kappa = -1e9
    best_state = None
    best_thresholds = threshold[:]

    epochs = 3
    start = time.time()
    for epoch in range(1, epochs + 1):
        net4.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
            yb = yb.to(device=device, dtype=torch.long, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            r_out, c_out, o_out = net4(xb)

            r_pred = torch.sigmoid(r_out).squeeze(1) * 4.5
            y_float = yb.float()

            o_tgt = make_ordinal_targets(yb)  # (B,4)
            loss_ce = criterion_ce(c_out, yb)
            loss_ord = criterion_bce(o_out, o_tgt)
            loss_reg = criterion_mse(r_pred, y_float)

            loss = loss_ce + 0.3 * loss_ord + 0.2 * loss_reg

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        net4.eval()
        y_true = []
        y_pred_before = []
        all_r_out = []
        all_c_out = []
        all_o_out = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
                r_out, c_out, o_out = net4(xb)

                all_r_out.append(r_out.detach().cpu())
                all_c_out.append(c_out.detach().cpu())
                all_o_out.append(o_out.detach().cpu())

                preds_b = (
                    combine3output_batch(r_out, c_out, o_out, threshold)
                    .detach()
                    .cpu()
                    .numpy()
                )
                preds_b = np.clip(preds_b, 0, 4).astype(np.int64)
                y_pred_before.extend(preds_b.tolist())

                y_true.extend(yb.numpy().tolist())

        kappa_before = qwk(y_true, y_pred_before)

        y_true_np = np.asarray(y_true, dtype=np.int64)
        r_out_cat = torch.cat(all_r_out, dim=0)
        c_out_cat = torch.cat(all_c_out, dim=0)
        o_out_cat = torch.cat(all_o_out, dim=0)

        tuned_thr, kappa_after = tune_thresholds_for_qwk_combine(
            r_out_cat, c_out_cat, o_out_cat, y_true_np, init_thr=threshold
        )
        old_thr = threshold[:]
        threshold = tuned_thr

        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} "
            f"val_qwk_before={kappa_before:.4f} val_qwk_after={kappa_after:.4f} "
            f"thr {old_thr} -> {threshold} elapsed={elapsed:.1f}s"
        )

        if kappa_after > best_kappa:
            best_kappa = kappa_after
            best_state = copy.deepcopy(net4.state_dict())
            best_thresholds = threshold[:]

    if best_state is not None:
        torch.save(
            {"state_dict": best_state, "thresholds": best_thresholds},
            trained_path,
        )
        net4.load_state_dict(best_state, strict=True)
        threshold = best_thresholds[:]
        print(
            "Saved best trained weights to:",
            trained_path,
            "best_val_qwk:",
            best_kappa,
            "best_thresholds:",
            threshold,
        )
    else:
        print(
            "WARNING: Training did not produce a best_state; proceeding with current weights."
        )

else:
    pass

net4.eval()



## === cell 7
submission = []
test_img_dir = os.path.join(BASE_INPUT, "test_images")
if not os.path.exists(test_img_dir):
    alt = os.path.join(BASE_INPUT, "aptos2019-blindness-detection", "test_images")
    if os.path.exists(alt):
        test_img_dir = alt
print("test_img_dir:", test_img_dir)


class APTOSTestDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = np.asarray(list(map(str, ids)))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        x = self.transform(img)
        return idx, x


test_ds = APTOSTestDataset(test_ids, test_img_dir, transform2)
num_workers = 4 if os.cpu_count() and os.cpu_count() >= 4 else 2
test_loader_kwargs = dict(
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
)
if num_workers > 0:
    test_loader_kwargs["prefetch_factor"] = 4

test_loader = DataLoader(test_ds, **test_loader_kwargs)

net4.eval()
with torch.inference_mode():
    seen = 0
    for id_batch, xb in test_loader:
        if seen % 200 == 0:
            print("Predicting", seen, "/", len(test_ids))
        xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
        r_out, c_out, o_out = net4(xb)
        pred = combine3output_batch(r_out, c_out, o_out, threshold)
        pred = torch.clamp(pred, 0, 4).to(torch.int64).cpu().numpy().tolist()
        submission.extend(list(zip(list(id_batch), pred)))
        seen += len(id_batch)

submission = np.array(submission, dtype=object)
print("Submission rows:", submission.shape[0], "Expected:", len(test_ids))
print("Using thresholds:", threshold)



## === cell 8
sub_df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
sub_df["id_code"] = sub_df["id_code"].astype(str)
sub_df["diagnosis"] = sub_df["diagnosis"].astype(int)

sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
if os.path.exists(sample_path):
    sample_df = pd.read_csv(sample_path)
    sub_df = sample_df[["id_code"]].merge(sub_df, on="id_code", how="left")
    sub_df["diagnosis"] = sub_df["diagnosis"].fillna(0).astype(int)

assert len(sub_df) > 0, "Submission DataFrame should not be empty"
assert list(sub_df.columns) == [
    "id_code",
    "diagnosis",
], "Submission must have columns: id_code, diagnosis"
assert len(sub_df) == len(
    test_ids
), "Submission must have same number of rows as test.csv"

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub_df.head())
