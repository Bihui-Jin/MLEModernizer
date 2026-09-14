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

0.9254702889569488

# 6. Current score

0.04284

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the EfficientNet activation lookup that currently raises an AttributeError by making it robust to timm’s newer module naming (so `backboneNet_efficient()` can be constructed). Then I ensure `net4` is always defined and inference runs even if the external weights file is missing, so the submission array is populated to 367 rows. Finally, I keep the model/inference logic unchanged but make the path handling and image loading more defensive, guaranteeing `submission.csv` is written with the required columns and correct row order.'
- What this solution (achieved 0.0) has done: 'I remove the hard failure when the external `3.pth` checkpoint is missing and instead run with randomly initialized weights (this is necessary to always produce a valid `submission.csv` in this environment). I also fix the CUDA/CPU dtype mismatch by moving the model to `device` before calling `load_state_dict`, ensuring model weights and inputs are on the same device during inference. Finally, I add a small safeguard so inference always produces exactly 367 predictions (one per `id_code`) and the submission is written in the required order and format.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with running inference from randomly initialized weights (the code falls back when `3.pth` is missing), so the smallest meaningful improvement is to ensure we actually use valid pretrained weights available in the environment. I keep your model and thresholding logic unchanged, but (1) make the weight search prefer EfficientNet-B4 checkpoints (and avoid accidentally loading unrelated `.pth` files), (2) add a safe “pretrained backbone fallback” only when no checkpoint is found (so predictions are not random), and (3) fix OpenCV color conversion in `crop_image_from_gray` by reading images as RGB before converting to grayscale (avoids masking artifacts). These changes keep evaluation semantics intact (still produces 0–4 via the same regression thresholds) while making predictions non-random and typically much closer to the target band.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with inference using mostly-random heads (when `3.pth` isn’t available or doesn’t match), so the smallest legitimate move toward the 0.925 target is to ensure we actually use a compatible pretrained checkpoint if one exists and to make the fallback non-random. I keep your architecture and inference mapping (sigmoid * 4.5 + thresholds) unchanged, but (1) improve checkpoint discovery to also include `.pt/.bin` and prioritize files whose tensors match your model’s key shapes, and (2) if no compatible finetuned checkpoint is found, I initialize the EfficientNet-B4 backbone from ImageNet **and** set the regression head bias to the dataset mean (from `train.csv`) so predictions aren’t essentially random. I also fix a subtle color-space issue in `crop_image_from_gray` by ensuring it receives BGR consistently (as read by OpenCV), while still converting to RGB only for PIL/torchvision transforms. These are minimal, execution-safe changes that should lift the score substantially above 0.0 without changing your core logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission-format/alignment issue rather than pure model quality (random weights usually still score nonzero), so the smallest high-impact fix is to guarantee the submission exactly matches `sample_submission.csv` ordering and dtypes. I keep your model, thresholds, transforms, and inference logic unchanged, but (1) build the submission by starting from `sample_submission.csv` and filling predictions by `id_code` mapping, (2) hard-validate that every `id_code` is present and no NaNs remain, and (3) clamp predictions to {0,1,2,3,4} to avoid any accidental out-of-range values. This is minimal, execution-safe, and directly targets common causes of an otherwise inexplicably-low kappa score.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most plausibly caused by a submission alignment/format problem (e.g., predicting in one order but writing in another, or accidentally writing strings/objects), so I make the submission generation strictly start from `sample_submission.csv` and fill predictions by `id_code` (guaranteed correct order). To avoid any accidental `object` dtype issues, I keep predictions as a plain Python `dict`/`list` and enforce `int` dtype before writing. I also add a final hard validation that the written `submission.csv` exactly matches the sample’s row order/count and contains only labels in `{0,1,2,3,4}`. These are minimal changes that keep your model/inference logic intact while addressing the most common cause of a flat 0.0 kappa.'
- What this solution (achieved 0.03848) has done: 'Your 0.0 score is most likely coming from predicting with a mismatched head/threshold pipeline: you’re using a B4 backbone with a 1-d regressor head and fixed thresholds, but you’re applying `sigmoid()*4.5` and then thresholds tuned for a different model, which can collapse predictions to a near-constant class and yield ~0 kappa. I keep the exact model and inference structure (same net, same transforms, same `sigmoid()*4.5`, same thresholding), but I calibrate the 4 thresholds *using train.csv distribution* so the predicted class proportions roughly match the training label proportions (a minimal, metric-aligned post-processing step for kappa). I also make sure the model is put into `eval()` and uses `inference_mode()` for deterministic inference, and I keep the submission creation strictly based on `sample_submission.csv` to avoid any ordering/alignment pitfalls. These changes are small, don’t alter architecture or training loops (there are none), and should move the score upward from 0.0 toward your target band.'
- What this solution (achieved 0.03848) has done: 'Your current score (0.03848) is far below the target (0.92547), so we should cautiously increase performance without changing the core model/inference semantics. The biggest likely issue is that you’re calibrating thresholds using *test* predictions’ quantiles, which can easily collapse classes and hurt QWK; instead, we compute thresholds from the **train label distribution** (fixed, leakage-free) and apply them to the same scalar regressor outputs. We also ensure `net4` runs in true eval mode (including disabling dropout) and keep the submission ordering strictly identical to `sample_submission.csv` (already correct). These are minimal changes that keep your architecture, `sigmoid()*4.5` regression head, and overall pipeline intact but should move the score materially upward toward the target band.'
- What this solution (achieved 0.04284) has done: 'Your score is extremely far below the target (0.03848 vs 0.92547), so we need a small but high-impact fix that improves predictive signal without changing your architecture or adding training. The biggest issue is that you compute thresholds from **test** regressor outputs, which can badly distort class boundaries under domain shift; instead we should compute thresholds once from **train** regressor outputs (same model) to match the metric’s ordinal nature without leakage. This keeps your exact model forward/inference (`sigmoid()*4.5`) and only changes how the 4 cutpoints are derived. To keep runtime under 600s, we compute train thresholds on a deterministic subset of train images (fixed seed) and then apply them to all test predictions, preserving submission ordering via `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import math
import random
import time
import json
import glob
import copy
import pickle
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT

from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

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
import time
import glob
import copy
import os
import json
import cv2
import numpy as np
import pickle
import torch.optim as optim
import torch.nn.functional as F
from torch import nn as nn
from torch.optim import lr_scheduler
from timm.models import *
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, models, datasets
from PIL import Image
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
from torch import Tensor, Size
from torch.jit.annotations import List, Optional, Tuple
from torch.nn.parameter import Parameter
import warnings


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


def _get_effnet_activation(net):
    for name in [
        "act1",
        "act2",
        "act_fn",
        "act_layer",
        "activation",
        "swish",
        "silu",
        "relu",
        "gelu",
        "act",
    ]:
        act = getattr(net, name, None)
        if act is not None:
            return act
    for m in net.modules():
        if isinstance(m, (nn.SiLU,)):
            return m
    return None


class backboneNet_efficient(nn.Module):
    def __init__(self, pretrained_backbone=False):
        """
        Keep core logic unchanged; only allow ImageNet-pretrained backbone as fallback when no
        finetuned checkpoint exists to avoid fully-random predictions.
        """
        super(backboneNet_efficient, self).__init__()
        net = timm.create_model("tf_efficientnet_b4_ns", pretrained=pretrained_backbone)

        layers_to_train = ["blocks"]
        for name, parameter in net.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        self.num_features = 1792
        self.conv_stem = net.conv_stem
        self.bn1 = net.bn1

        self.act1 = _get_effnet_activation(net)
        if self.act1 is None:
            self.act1 = nn.SiLU()

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]
        self.conv_head = net.conv_head
        self.bn2 = net.bn2

        self.act2 = getattr(net, "act2", None)
        if self.act2 is None:
            self.act2 = _get_effnet_activation(net)

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
        x13 = self.act2(x12) if self.act2 is not None else x12
        x14 = self.global_pool(x13)

        if x14.ndim > 2:
            x14 = x14.flatten(1)

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
            l1 = int(math.floor(out[i]))
            l2 = int(math.ceil(out[i]))
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
    if img is None:
        return None
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
BASE = "/kaggle/input/aptos2019-blindness-detection"

test_df = pd.read_csv(f"{BASE}/test.csv")
test_ids = test_df["id_code"].astype(str).tolist()

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


def _iter_weight_files():
    search_roots = ["/kaggle/input", "/kaggle/data", "/kaggle/working"]
    exts = (".pth", ".pt", ".bin")
    for root in search_roots:
        for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            if p.lower().endswith(exts) and os.path.isfile(p):
                yield p


def _load_state_dict_any(path):
    obj = torch.load(path, map_location="cpu")
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        obj = obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    return obj if isinstance(obj, dict) else None


def _compat_score(state_dict, model_state):
    if state_dict is None:
        return -1
    overlap = 0
    exact = 0
    for k, v in state_dict.items():
        if k in model_state and isinstance(v, torch.Tensor):
            overlap += 1
            if tuple(v.shape) == tuple(model_state[k].shape):
                exact += 1
    if overlap == 0:
        return -1
    return exact * 10 + overlap


def _find_best_weights(model):
    model_state = model.state_dict()
    best = (None, -1)
    for p in _iter_weight_files():
        bn = os.path.basename(p).lower()
        if not any(
            tok in bn
            for tok in [
                "eff",
                "efficient",
                "b4",
                "aptos",
                "blind",
                "dr",
                "retina",
                "stage",
                "net",
            ]
        ):
            continue
        try:
            sd = _load_state_dict_any(p)
        except Exception:
            continue
        sc = _compat_score(sd, model_state)
        if sc > best[1]:
            best = (p, sc)
    return best[0]


WEIGHTS_PATH = "/kaggle/input/weights/3.pth"
net4_probe = backboneNet_efficient(pretrained_backbone=False)

if not os.path.exists(WEIGHTS_PATH):
    WEIGHTS_PATH = None

if WEIGHTS_PATH is not None:
    try:
        _ = _compat_score(_load_state_dict_any(WEIGHTS_PATH), net4_probe.state_dict())
        if _ < 0:
            WEIGHTS_PATH = None
    except Exception:
        WEIGHTS_PATH = None

if WEIGHTS_PATH is None:
    WEIGHTS_PATH = _find_best_weights(net4_probe)

use_pretrained_backbone = not (
    WEIGHTS_PATH is not None and os.path.exists(WEIGHTS_PATH)
)
net4 = backboneNet_efficient(pretrained_backbone=use_pretrained_backbone).to(device)
net4.eval()

loaded_ok = False
if WEIGHTS_PATH is not None and os.path.exists(WEIGHTS_PATH):
    state = _load_state_dict_any(WEIGHTS_PATH)
    if state is not None:
        missing, unexpected = net4.load_state_dict(state, strict=False)
        loaded_ok = True
        print("Loaded weights from:", WEIGHTS_PATH)
        if missing:
            print(f"Warning: missing keys when loading weights: {len(missing)}")
        if unexpected:
            print(f"Warning: unexpected keys when loading weights: {len(unexpected)}")

if not loaded_ok:
    print(
        "Warning: no compatible finetuned checkpoint found. "
        "Using ImageNet-pretrained EfficientNet-B4 backbone."
    )
    train_df = pd.read_csv(f"{BASE}/train.csv")
    mean_diag = float(train_df["diagnosis"].mean())
    mean_diag = max(0.0, min(4.5, mean_diag))
    p = np.clip(mean_diag / 4.5, 1e-4, 1 - 1e-4)
    bias = float(np.log(p / (1 - p)))
    with torch.no_grad():
        net4.rg_cls.bias.fill_(bias)

train_df = pd.read_csv(f"{BASE}/train.csv")
label_counts = (
    train_df["diagnosis"].value_counts().reindex([0, 1, 2, 3, 4], fill_value=0).values
)
cum = np.cumsum(label_counts) / float(label_counts.sum())
target_cuts = [float(cum[0]), float(cum[1]), float(cum[2]), float(cum[3])]




## === cell 6
def _quantile_from_sorted(sorted_vals, q):
    q = float(np.clip(q, 0.0, 1.0))
    if sorted_vals.size == 0:
        return 0.0
    k = int(round(q * (sorted_vals.size - 1)))
    k = max(0, min(sorted_vals.size - 1, k))
    return float(sorted_vals[k])


def _infer_rg_for_ids(id_list, img_dir, max_n=None):
    """
    Change is directly score-relevant: derive thresholds from TRAIN outputs (not TEST outputs)
    to avoid unstable/shifted cutpoints that can collapse QWK.

    Runtime guard: compute on a fixed-size deterministic subset if max_n is set.
    """
    ids = [str(x) for x in id_list]
    if max_n is not None and len(ids) > max_n:
        rng = np.random.RandomState(42)
        sel = rng.choice(len(ids), size=int(max_n), replace=False)
        ids = [ids[i] for i in sel.tolist()]

    raw = []
    with torch.inference_mode():
        for idx in ids:
            image_name = os.path.join(img_dir, f"{idx}.png")
            img2 = cv2.imread(image_name)  # BGR
            if img2 is None:
                raw.append((idx, 0.0))
                continue
            img2 = crop_image_from_gray(img2)  # expects BGR; returns BGR
            if img2 is None:
                raw.append((idx, 0.0))
                continue
            img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            x = transform2(img2).unsqueeze(0).to(device)

            rg_outputs, cls_outputs, ord_outputs = net4(x)
            rg_outputs = torch.sigmoid(rg_outputs) * 4.5
            rg = float(rg_outputs.view(-1)[0].item())
            raw.append((idx, rg))
    return raw


train_ids = train_df["id_code"].astype(str).tolist()

TRAIN_CALIB_N = 900  # small-but-signal-bearing; deterministic via seed above
train_raw_rg = _infer_rg_for_ids(train_ids, f"{BASE}/train_images", max_n=TRAIN_CALIB_N)

train_rg_vals = np.array([v for _, v in train_raw_rg], dtype=np.float32)
train_rg_sorted = np.sort(train_rg_vals)
thrs = [_quantile_from_sorted(train_rg_sorted, q) for q in target_cuts]
for i in range(1, 4):
    if thrs[i] <= thrs[i - 1]:
        thrs[i] = thrs[i - 1] + 1e-6

print("Calibrated thresholds (from TRAIN outputs) used:", thrs)
print("Target cut proportions from train labels:", target_cuts)
print("Using train subset for calibration:", len(train_raw_rg), "images")

pred_map = {}
test_raw_rg = _infer_rg_for_ids(test_ids, f"{BASE}/test_images", max_n=None)
for idx, rg in test_raw_rg:
    if rg < thrs[0]:
        P = 0
    elif rg < thrs[1]:
        P = 1
    elif rg < thrs[2]:
        P = 2
    elif rg < thrs[3]:
        P = 3
    else:
        P = 4
    pred_map[idx] = int(max(0, min(4, P)))

if len(pred_map) != len(test_ids):
    raise RuntimeError(
        f"Internal error: got {len(pred_map)} preds for {len(test_ids)} ids"
    )

print(
    "Predicted class distribution:",
    pd.Series(list(pred_map.values())).value_counts().sort_index().to_dict(),
)



## === cell 7
sample_path = f"{BASE}/sample_submission.csv"
sample = pd.read_csv(sample_path)
sample["id_code"] = sample["id_code"].astype(str)

missing_ids = [x for x in sample["id_code"].tolist() if x not in pred_map]
if missing_ids:
    raise RuntimeError(
        f"Missing predictions for {len(missing_ids)} ids, first few: {missing_ids[:5]}"
    )

sample["diagnosis"] = sample["id_code"].map(pred_map).astype(int)

if sample.shape[0] != len(sample["id_code"].unique()):
    raise RuntimeError(
        "Duplicate id_code rows detected in sample_submission (unexpected)."
    )

if sample["diagnosis"].isna().any():
    bad = sample.loc[sample["diagnosis"].isna(), "id_code"].tolist()[:5]
    raise RuntimeError(f"NaN diagnoses after mapping; first few bad ids: {bad}")

bad_vals = (
    sample.loc[~sample["diagnosis"].isin([0, 1, 2, 3, 4]), "diagnosis"]
    .unique()
    .tolist()
)
if bad_vals:
    raise RuntimeError(f"Out-of-range diagnosis values found: {bad_vals}")

sample.to_csv("submission.csv", index=False)

print(sample.head())
print("Wrote submission.csv with shape:", sample.shape)
print("Device used:", device)
print("Weights path used:", WEIGHTS_PATH)
print("Pretrained backbone used:", use_pretrained_backbone)
