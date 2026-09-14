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

0.9260258431995622

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the two blocking runtime issues: (1) the notebook hardcodes `cuda:0` even when no GPU is available, and (2) the model weights path `../input/weights/...` does not exist in your dataset, so inference never starts and the submission stays empty. To keep core logic intact, I make device selection automatic (CPU fallback) and load weights only if present; otherwise the model run with random weights but still produce a valid submission CSV. I also fix a couple of small bugs that can crash transforms (`trim` returning `None`, and `is` vs `==` string comparison), and ensure the script always writes `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.0) has done: 'I fix the two blockers that prevent any submission from being produced: (1) the code currently hard-fails if pretrained weights aren’t found, and (2) the model/inputs end up on different devices causing a dtype/device mismatch at inference. To keep the core model/inference logic intact, I make weight loading optional (warn + proceed) and ensure the model is moved to the selected device after loading. I also make `regress2class` robust to both scalar and 1-element tensors so the prediction loop doesn’t crash. Finally, I ensure a `submission.csv` is always written with the required columns and exactly the test row count.'
- What this solution (achieved -0.03935) has done: 'Your current 0.0 score is consistent with running inference from randomly initialized weights (no usable pretrained checkpoint in the environment), so the smallest legitimate way to move toward the 0.926 target is to ensure the backbone uses ImageNet-pretrained weights when the competition finetuned weights file isn’t found. This keeps your architecture and inference semantics identical, but greatly improves predictions versus random. I also make weight loading tolerant of minor key mismatches (`strict=False`) so a near-match checkpoint can still load instead of silently falling back to random, which again moves score upward without changing the core approach. Finally, I keep the same submission writing logic but add a deterministic clamp to [0,4] to avoid any out-of-range labels.'
- What this solution (achieved -0.05081) has done: 'Your current score is far below the 0.926 target, so we should increase performance with the smallest changes that preserve your model/inference semantics. The biggest issue is that you’re only using the `final=True` regressor head which is random if finetuned weights aren’t found, even though you *do* have decent ImageNet features from the pretrained backbone; we can instead use the already-defined `combine3output(c_out,r_out,o_out)` path (still your same model) so predictions are not purely from a random head. I keep your architecture and transforms unchanged, keep optional finetuned weight loading as-is, but switch inference to use the 3-head combination and clamp to [0,4]. This is a minimal change that should move QWK substantially upward toward your target without changing training loops (you have none) or the model definition.'
- What this solution (achieved -0.13158) has done: 'Your score is far below the 0.926 target, and the biggest remaining issue is that `combine3output` currently mixes one meaningful head (classifier) with two essentially uncalibrated heads (regressor/ordinal) when finetuned weights are missing, which drags predictions toward noise and can produce negative QWK. To move toward the target with minimal change and identical inference semantics, I make the fallback behavior explicit: if finetuned weights are not loaded, use only the classifier argmax (the only head with a sensible ImageNet-initialized backbone signal) instead of averaging in random heads. If finetuned weights are loaded, keep your current 3-head combination exactly as-is. This keeps your model and transforms intact, but should substantially improve agreement versus the current noisy averaging.'
- What this solution (achieved 0.0) has done: 'Your score is far below the 0.926 target, and the dominant issue is that you’re effectively doing “no-training” inference with mostly random heads, which can easily yield negative QWK. To move the score upward with minimal changes and identical overall semantics, I keep your same `ThreeStage_Model` and transforms, but (1) switch the EfficientNet backbone in `ThreeStage_Model` to `pretrained=False` so timm won’t attempt any internet download and we can deterministically load local ImageNet weights if available, and (2) add a robust local checkpoint search for an ImageNet-pretrained `tf_efficientnet_b4_ns` (common in Kaggle images) and load it into `net1.backbone` (strict=False). This keeps the architecture and inference path intact, but should turn the classifier head’s inputs from random features into meaningful features, improving QWK toward the target without changing training or adding approximations. If no local backbone weights are found, the code falls back to your current behavior and still writes a valid `submission.csv`.'

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
import glob
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        n: Optional[int] = None,
    ):
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
    out_t = torch.as_tensor(out)
    out_val = out_t.detach().flatten()[0].cpu().item()
    prediction = 0
    for i in range(4):
        prediction += int(out_val >= threshold[i])
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




## === cell 5
BASE_INPUT_CANDIDATES = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/input",
    "/kaggle/data",
]
base_input = None
for p in BASE_INPUT_CANDIDATES:
    if os.path.exists(os.path.join(p, "test.csv")):
        base_input = p
        break
if base_input is None:
    for p in BASE_INPUT_CANDIDATES:
        if os.path.exists(os.path.join(p, "aptos2019-blindness-detection", "test.csv")):
            base_input = os.path.join(p, "aptos2019-blindness-detection")
            break
if base_input is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection directory (test.csv) in expected locations."
    )

test_csv_path = os.path.join(base_input, "test.csv")

TEST_IMG_DIR_CANDIDATES = [
    os.path.join(base_input, "test_images"),
    os.path.join(base_input, "aptos2019-blindness-detection", "test_images"),
]
test_img_dir = None
for d in TEST_IMG_DIR_CANDIDATES:
    if os.path.exists(d):
        test_img_dir = d
        break
if test_img_dir is None:
    raise FileNotFoundError(
        "Could not locate test_images directory in expected locations."
    )

test_df = pd.read_csv(test_csv_path)
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

net1 = ThreeStage_Model()

WEIGHTS_CANDIDATES = [
    "../input/weights/B4_3stage_3epoch_320finetune.pkl",
    "/kaggle/input/weights/B4_3stage_3epoch_320finetune.pkl",
    os.path.join(base_input, "weights", "B4_3stage_3epoch_320finetune.pkl"),
    os.path.join(base_input, "B4_3stage_3epoch_320finetune.pkl"),
    os.path.join(base_input, "weights", "B4_3stage_3epoch_320finetune.pth"),
    os.path.join(base_input, "B4_3stage_3epoch_320finetune.pth"),
]
weights_path = None
for wp in WEIGHTS_CANDIDATES:
    if os.path.exists(wp):
        weights_path = wp
        break

if weights_path is None:
    patterns = [
        os.path.join(base_input, "**", "*B4*3stage*finetune*.pkl"),
        os.path.join(base_input, "**", "*B4*3stage*finetune*.pth"),
        os.path.join(base_input, "**", "*B4*3stage*finetune*.pt"),
        os.path.join(base_input, "**", "*.pth"),
        os.path.join(base_input, "**", "*.pkl"),
        os.path.join(base_input, "**", "*.pt"),
    ]
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            weights_path = sorted(hits)[0]
            break

loaded_weights = False
if weights_path is not None and os.path.exists(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = net1.load_state_dict(state, strict=False)
    loaded_weights = True
    print("Loaded finetuned weights (strict=False):", weights_path)
    if missing:
        print("Missing keys (showing up to 20):", missing[:20])
    if unexpected:
        print("Unexpected keys (showing up to 20):", unexpected[:20])
else:
    warnings.warn(
        "No finetuned model weights found in the environment. Proceeding with randomly initialized heads."
    )

if not loaded_weights:
    backbone_weights_path = None
    BACKBONE_PATTERNS = [
        "/kaggle/input/**/tf_efficientnet_b4_ns*.pth",
        "/kaggle/input/**/efficientnet_b4*.pth",
        "/kaggle/input/**/efficientnetb4*.pth",
        "/kaggle/data/**/tf_efficientnet_b4_ns*.pth",
        "/kaggle/data/**/efficientnet_b4*.pth",
        "/kaggle/data/**/efficientnetb4*.pth",
        "/kaggle/working/**/tf_efficientnet_b4_ns*.pth",
        "/kaggle/working/**/efficientnet_b4*.pth",
        "/kaggle/working/**/efficientnetb4*.pth",
    ]
    for pat in BACKBONE_PATTERNS:
        hits = glob.glob(pat, recursive=True)
        if hits:
            backbone_weights_path = sorted(hits)[0]
            break

    if backbone_weights_path is not None and os.path.exists(backbone_weights_path):
        bstate = torch.load(backbone_weights_path, map_location="cpu")
        if (
            isinstance(bstate, dict)
            and "state_dict" in bstate
            and isinstance(bstate["state_dict"], dict)
        ):
            bstate = bstate["state_dict"]
        if isinstance(bstate, dict) and any(
            k.startswith("module.") for k in bstate.keys()
        ):
            bstate = {k.replace("module.", "", 1): v for k, v in bstate.items()}

        if isinstance(bstate, dict):
            candidates = [bstate]
            for key in ["model", "net", "backbone"]:
                if key in bstate and isinstance(bstate[key], dict):
                    candidates.append(bstate[key])
            loaded_backbone = False
            for cand in candidates:
                try:
                    miss_b, unexp_b = net1.backbone.load_state_dict(cand, strict=False)
                    print(
                        "Loaded local ImageNet backbone weights (strict=False):",
                        backbone_weights_path,
                    )
                    if miss_b:
                        print("Backbone missing keys (showing up to 20):", miss_b[:20])
                    if unexp_b:
                        print(
                            "Backbone unexpected keys (showing up to 20):", unexp_b[:20]
                        )
                    loaded_backbone = True
                    break
                except Exception:
                    continue
            if not loaded_backbone:
                warnings.warn(
                    "Found a backbone checkpoint but could not load it into net1.backbone; continuing."
                )
    else:
        warnings.warn(
            "No local ImageNet EfficientNet-B4 backbone checkpoint found; continuing with random backbone."
        )

net1 = net1.to(device)
net1.eval()



## === cell 6
submission = []
missing_images = 0

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print("Predicting", i, "/", len(test_ids))

        image_name = os.path.join(test_img_dir, f"{idx}.png")
        if not os.path.exists(image_name):
            missing_images += 1
            submission.append([idx, 0])
            continue

        img = Image.open(image_name).convert("RGB")
        img1 = transform1(img).unsqueeze(0).to(device)

        c_out, r_out, o_out = net1(img1, final=False)

        if loaded_weights:
            pred1 = int(combine3output(r_out, c_out, o_out))
        else:
            pred1 = int(torch.argmax(c_out, dim=1).item())

        pred1 = int(np.clip(pred1, 0, 4))
        submission.append([idx, pred1])

if missing_images:
    print(f"WARNING: {missing_images} test images were missing; predicted 0 for them.")

submission = np.array(submission, dtype=object)
print("Collected predictions:", submission.shape)



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["diagnosis"] = df["diagnosis"].astype(int)

df = df.set_index("id_code").reindex(test_ids).reset_index()
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

assert len(df) == len(
    test_ids
), f"Submission row count {len(df)} != test row count {len(test_ids)}"

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())



## === cell 8
df.head()
