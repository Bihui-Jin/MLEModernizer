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
    r = r_out.detach()
    if r.ndim == 2 and r.size(1) == 1:
        r = r.squeeze(1)

    p_reg = regress2class_prob(r)  # (B,5)
    p_cls = torch.softmax(c_out.detach(), dim=1)  # (B,5)
    p_ord = ordinal2class_prob(o_out.detach())  # (B,5)

    p = (p_reg + p_cls + p_ord) / 3.0

    classes = torch.arange(5, device=p.device, dtype=p.dtype).view(1, 5)  # (1,5)
    expected = (p * classes).sum(dim=1)  # (B,)
    P = regress2class(expected)  # (B,)

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
train_csv_path = os.path.join(DATA_ROOT, "train.csv")
train_img_dir = os.path.join(DATA_ROOT, "train_images")

test_ids_df = pd.read_csv(test_csv_path)
test_ids = np.squeeze(test_ids_df["id_code"].values)

train_df = pd.read_csv(train_csv_path)
train_ids = train_df["id_code"].astype(str).values
train_y = train_df["diagnosis"].astype(int).values

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
class TrainDataset(Dataset):
    def __init__(self, ids, y, img_dir, transform):
        self.ids = list(ids)
        self.y = np.asarray(y).astype(int)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = str(self.ids[i])
        label = int(self.y[i])
        path = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, label


def _make_ordinal_targets(y: torch.Tensor) -> torch.Tensor:
    B = y.size(0)
    t = torch.zeros((B, 4), device=y.device, dtype=torch.float32)
    for k in range(4):
        t[:, k] = (y > k).float()
    return t


def _qwk_np(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


if not weights_loaded:
    for p in net1.backbone.parameters():
        p.requires_grad_(False)
    net1.train()

    rng = np.random.RandomState(42)
    idxs = np.arange(len(train_ids))
    rng.shuffle(idxs)
    val_size = max(1, int(0.15 * len(idxs)))
    val_idxs = idxs[:val_size]
    trn_idxs = idxs[val_size:]

    trn_ds = TrainDataset(
        train_ids[trn_idxs], train_y[trn_idxs], train_img_dir, transform2
    )
    val_ds = TrainDataset(
        train_ids[val_idxs], train_y[val_idxs], train_img_dir, transform2
    )

    trn_loader = DataLoader(
        trn_ds,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    head_params = (
        list(net1.classifier.parameters())
        + list(net1.regressor.parameters())
        + list(net1.ordinal.parameters())
    )
    opt = torch.optim.AdamW(head_params, lr=3e-4, weight_decay=1e-4)

    ce_loss = nn.CrossEntropyLoss()
    l1_loss = nn.SmoothL1Loss(beta=1.0)
    bce_loss = nn.BCEWithLogitsLoss()

    EPOCHS = 3
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        net1.train()
        tr_losses = []
        for xb, yb in trn_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).long()

            c_out, r_out, o_out = net1(xb)  # r_out already sigmoid*4.5, o_out sigmoid

            loss_c = ce_loss(c_out, yb)

            y_reg = yb.float()
            r = r_out.squeeze(1)
            loss_r = l1_loss(r, y_reg)

            o_prob = torch.clamp(o_out, 1e-4, 1 - 1e-4)
            o_logit = torch.log(o_prob / (1 - o_prob))
            t_ord = _make_ordinal_targets(yb)
            loss_o = bce_loss(o_logit, t_ord)

            loss = loss_c + loss_r + 0.5 * loss_o

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_losses.append(loss.item())

        net1.eval()
        y_true_all = []
        y_pred_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                c_out, r_out, o_out = net1(xb)
                pred = (
                    combine3output(r_out, c_out, o_out)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                    .tolist()
                )
                y_pred_all.extend(pred)
                y_true_all.extend(yb.numpy().astype(int).tolist())
        qwk = _qwk_np(np.array(y_true_all), np.array(y_pred_all))
        print(
            f"Epoch {epoch}/{EPOCHS} | train_loss={np.mean(tr_losses):.4f} | val_qwk={qwk:.4f} | time={time.time()-t0:.1f}s"
        )

    net1.eval()
else:
    net1.eval()




## === cell 7
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


test_ds = TestDataset(test_ids, test_img_dir, transform2)
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



## === cell 8
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
