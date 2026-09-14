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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



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
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    (Kept for backward compatibility with the original notebook code.)
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
    """
    EfficientNet backbone + 3 heads (regression/class/ordinal).
    """

    def __init__(self, pretrained_backbone=False):
        super(backboneNet_efficient, self).__init__()
        self.net = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=pretrained_backbone
        )
        self.num_features = getattr(self.net, "num_features", 1792)
        self.drop_rate = 0.4

        self.rg_cls = Linear(self.num_features, 1, bias=True)
        self.cls_cls = Linear(self.num_features, 5, bias=True)
        self.ord_cls = Linear(self.num_features, 4, bias=True)

    def forward(self, x):
        feats = self.net.forward_features(x)
        x14 = self.net.forward_head(feats, pre_logits=True)

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
    pred_prob = pred_prob / torch.clamp(pred_prob.sum(dim=1, keepdim=True), min=1e-12)
    return pred_prob


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


def _regress_to_prob_thresholded(reg_vals_0_4p5, th):
    v = np.asarray(reg_vals_0_4p5, dtype=np.float32)
    th = np.asarray(th, dtype=np.float32)
    bins = np.digitize(v, th, right=False)  # 0..4
    prob = np.zeros((len(v), 5), dtype=np.float32)
    prob[np.arange(len(v)), bins] = 1.0
    return prob


def _regress_to_prob_soft(reg_vals_0_4p5, th, softness=0.18):
    v = np.asarray(reg_vals_0_4p5, dtype=np.float32).reshape(-1)
    th = np.asarray(th, dtype=np.float32).reshape(-1)
    th = np.sort(th)
    n = len(v)
    p = np.zeros((n, 5), dtype=np.float32)

    def sstep(x):
        return np.clip(0.5 + 0.5 * (x / softness), 0.0, 1.0)

    t0, t1, t2, t3 = th.tolist()
    p[:, 0] = 1.0 - sstep(v - t0)
    p[:, 4] = sstep(v - t3)

    left1 = sstep(v - t0)
    right1 = 1.0 - sstep(v - t1)
    p[:, 1] = left1 * right1

    left2 = sstep(v - t1)
    right2 = 1.0 - sstep(v - t2)
    p[:, 2] = left2 * right2

    left3 = sstep(v - t2)
    right3 = 1.0 - sstep(v - t3)
    p[:, 3] = left3 * right3

    ps = p.sum(axis=1, keepdims=True)
    hard = _regress_to_prob_thresholded(v, th)
    p = np.where(ps > 1e-6, p / np.clip(ps, 1e-12, None), hard)
    return p.astype(np.float32)


def _predict_from_fused_probs(p_reg, p_cls, p_ord, w):
    w = np.asarray(w, dtype=np.float32)
    w = np.maximum(w, 0)
    s = float(w.sum())
    if s <= 0:
        w = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        s = 3.0
    w = w / s
    p = w[0] * p_reg + w[1] * p_cls + w[2] * p_ord
    return np.argmax(p, axis=1).astype(np.int64)




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
        return image  # ensure transform always returns an image


def crop_image_from_gray(img, tol=7):
    if img is None:
        return None
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
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data",
    "/kaggle/input",
]
BASE_DIR = None
for cand in BASE_DIR_CANDIDATES:
    if os.path.exists(os.path.join(cand, "test.csv")) and os.path.exists(
        os.path.join(cand, "train.csv")
    ):
        BASE_DIR = cand
        break
if BASE_DIR is None:
    BASE_DIR = "/kaggle/input/aptos2019-blindness-detection"
print("BASE_DIR:", BASE_DIR)

train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
test_ids = test_df["id_code"].astype(str).values

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

WEIGHTS_PATHS = [
    "/kaggle/input/weights/2.pth",
    "../input/weights/2.pth",
    "/kaggle/working/weights/2.pth",
]
found_weights = None
for wp in WEIGHTS_PATHS:
    if os.path.exists(wp):
        found_weights = wp
        break

train_img_dir_candidates = [
    os.path.join(BASE_DIR, "train_images"),
    os.path.join(BASE_DIR, "aptos2019-blindness-detection", "train_images"),
]
train_img_dir = None
for cand in train_img_dir_candidates:
    if os.path.isdir(cand):
        train_img_dir = cand
        break
if train_img_dir is None:
    train_img_dir = os.path.join(BASE_DIR, "train_images")
print("train_img_dir:", train_img_dir)

test_img_dir_candidates = [
    os.path.join(BASE_DIR, "test_images"),
    os.path.join(BASE_DIR, "aptos2019-blindness-detection", "test_images"),
]
test_img_dir = None
for cand in test_img_dir_candidates:
    if os.path.isdir(cand):
        test_img_dir = cand
        break
if test_img_dir is None:
    test_img_dir = os.path.join(BASE_DIR, "test_images")
print("test_img_dir:", test_img_dir)


def _extract_pooled_features(
    net3_model: backboneNet_efficient, df, img_dir, max_images, seed, batch_size=16
):
    rng = np.random.RandomState(seed)
    if len(df) > max_images:
        df_use = df.iloc[
            rng.choice(len(df), size=max_images, replace=False)
        ].reset_index(drop=True)
    else:
        df_use = df.reset_index(drop=True)

    feats_list = []
    ys = []
    net3_model.eval()

    batch_imgs = []
    batch_y = []

    with torch.no_grad():
        for i, row in enumerate(df_use.itertuples(index=False)):
            idx = str(getattr(row, "id_code"))
            image_name = os.path.join(img_dir, f"{idx}.png")
            img = cv2.imread(image_name)
            if img is None:
                continue
            img = crop_image_from_gray(img)
            if img is None:
                continue
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            batch_imgs.append(transform2(img))
            if "diagnosis" in df_use.columns:
                batch_y.append(int(getattr(row, "diagnosis")))

            if len(batch_imgs) >= batch_size:
                x = torch.stack(batch_imgs, dim=0).to(device)
                feats = net3_model.net.forward_features(x)
                pooled = net3_model.net.forward_head(feats, pre_logits=True)  # (B,F)
                feats_list.append(pooled.detach().float().cpu().numpy())
                if "diagnosis" in df_use.columns:
                    ys.extend(batch_y)
                batch_imgs = []
                batch_y = []

        if len(batch_imgs) > 0:
            x = torch.stack(batch_imgs, dim=0).to(device)
            feats = net3_model.net.forward_features(x)
            pooled = net3_model.net.forward_head(feats, pre_logits=True)
            feats_list.append(pooled.detach().float().cpu().numpy())
            if "diagnosis" in df_use.columns:
                ys.extend(batch_y)

    X = (
        np.concatenate(feats_list, axis=0)
        if len(feats_list)
        else np.zeros((0, net3_model.num_features), np.float32)
    )
    y = np.array(ys, dtype=np.int64) if len(ys) else np.zeros((0,), np.int64)
    return X.astype(np.float32), y


def _fit_ridge_closed_form(X, Y, lam=3.0):
    """
    Closed-form ridge initialization for heads using pretrained pooled features -> targets.
    No training loop/optimizer; just a deterministic linear solve.
    """
    n, d = X.shape
    X1 = np.concatenate([X, np.ones((n, 1), np.float32)], axis=1)  # add bias
    d1 = d + 1
    A = X1.T @ X1
    A.flat[:: d1 + 1] += lam  # ridge
    B = X1.T @ Y
    W = np.linalg.solve(A.astype(np.float64), B.astype(np.float64)).astype(np.float32)
    return W  # (d1, k)


def _logit(p):
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p)).astype(np.float32)


def _bake_feature_norm_into_linear(W, b, mean, std):
    """
    Change-of-variables so that Linear(x_norm) == Linear_baked(x_raw),
    where x_norm = (x_raw - mean) / std.
    This keeps model forward identical (still uses raw pooled features),
    but makes closed-form fit consistent with inference.
    """
    std = np.asarray(std, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    W_baked = W / std[None, :]
    b_baked = b - (W * (mean / std)[None, :]).sum(axis=1)
    return W_baked.astype(np.float32), b_baked.astype(np.float32)


def _init_heads_from_closed_form_label_fit(
    net3_model: backboneNet_efficient, df_train, img_dir
):
    max_images = min(3000, len(df_train))
    X, y = _extract_pooled_features(
        net3_model, df_train, img_dir, max_images=max_images, seed=SEED, batch_size=16
    )
    if len(y) < 80:
        print(
            "WARNING: too few samples for closed-form head init; keeping default init."
        )
        return

    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-3
    Xn = (X - mu) / sd

    Yc = np.eye(5, dtype=np.float32)[np.clip(y, 0, 4)]
    Wc = _fit_ridge_closed_form(Xn, Yc, lam=5.0)  # (d+1,5)

    Yr_prob = y.astype(np.float32) / 4.0  # in [0,1]
    Wr = _fit_ridge_closed_form(Xn, _logit(Yr_prob).reshape(-1, 1), lam=5.0)  # (d+1,1)

    Yo_prob = np.zeros((len(y), 4), dtype=np.float32)
    for k in range(4):
        Yo_prob[:, k] = (y > k).astype(np.float32)
    Wo = _fit_ridge_closed_form(Xn, _logit(Yo_prob), lam=5.0)  # (d+1,4)

    mu1 = mu.reshape(-1)
    sd1 = sd.reshape(-1)

    Wc_b, bc_b = _bake_feature_norm_into_linear(
        W=Wc[:-1].T, b=Wc[-1], mean=mu1, std=sd1
    )
    Wr_b, br_b = _bake_feature_norm_into_linear(
        W=Wr[:-1].T, b=Wr[-1].reshape(-1), mean=mu1, std=sd1
    )
    Wo_b, bo_b = _bake_feature_norm_into_linear(
        W=Wo[:-1].T, b=Wo[-1], mean=mu1, std=sd1
    )

    with torch.no_grad():
        net3_model.cls_cls.weight.copy_(torch.from_numpy(Wc_b).to(device))
        net3_model.cls_cls.bias.copy_(torch.from_numpy(bc_b).to(device))

        net3_model.rg_cls.weight.copy_(torch.from_numpy(Wr_b).to(device))
        net3_model.rg_cls.bias.copy_(torch.from_numpy(br_b).to(device))

        net3_model.ord_cls.weight.copy_(torch.from_numpy(Wo_b).to(device))
        net3_model.ord_cls.bias.copy_(torch.from_numpy(bo_b).to(device))


net3 = backboneNet_efficient(pretrained_backbone=True).to(device)

if found_weights is not None:
    print("Loading custom weights:", found_weights)
    state = torch.load(found_weights, map_location="cpu")
    net3.load_state_dict(state, strict=True)
else:
    print(
        "WARNING: custom weights not found; using pretrained backbone + closed-form head init (no training loop)."
    )
    _init_heads_from_closed_form_label_fit(net3, train_df, train_img_dir)

net3.eval()


def _infer_outputs_on_train(net, df, img_dir, max_images=3200, seed=0, batch_size=16):
    rng = np.random.RandomState(seed)
    if len(df) > max_images:
        df_use = df.iloc[
            rng.choice(len(df), size=max_images, replace=False)
        ].reset_index(drop=True)
    else:
        df_use = df.reset_index(drop=True)

    reg_vals = []
    cls_logits = []
    ord_logits = []
    ys = []

    batch_imgs = []
    batch_y = []

    with torch.no_grad():
        for i, row in enumerate(df_use.itertuples(index=False)):
            if i % 250 == 0:
                print("calib infer", i, "/", len(df_use))
            idx = str(getattr(row, "id_code"))
            y = int(getattr(row, "diagnosis"))
            image_name = os.path.join(img_dir, f"{idx}.png")
            img = cv2.imread(image_name)
            if img is None:
                continue
            img = crop_image_from_gray(img)
            if img is None:
                continue
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            batch_imgs.append(transform2(img))
            batch_y.append(y)

            if len(batch_imgs) >= batch_size:
                x = torch.stack(batch_imgs, dim=0).to(device)
                rg, cls, ord_ = net(x)
                rg = torch.sigmoid(rg).squeeze(1) * 4.5  # (B,)
                reg_vals.extend(rg.detach().cpu().numpy().astype(np.float32).tolist())
                cls_logits.append(cls.detach().cpu().numpy().astype(np.float32))
                ord_logits.append(ord_.detach().cpu().numpy().astype(np.float32))
                ys.extend(batch_y)
                batch_imgs = []
                batch_y = []

        if len(batch_imgs) > 0:
            x = torch.stack(batch_imgs, dim=0).to(device)
            rg, cls, ord_ = net(x)
            rg = torch.sigmoid(rg).squeeze(1) * 4.5
            reg_vals.extend(rg.detach().cpu().numpy().astype(np.float32).tolist())
            cls_logits.append(cls.detach().cpu().numpy().astype(np.float32))
            ord_logits.append(ord_.detach().cpu().numpy().astype(np.float32))
            ys.extend(batch_y)

    return (
        np.array(reg_vals, dtype=np.float32),
        (
            np.concatenate(cls_logits, axis=0)
            if len(cls_logits)
            else np.zeros((0, 5), np.float32)
        ),
        (
            np.concatenate(ord_logits, axis=0)
            if len(ord_logits)
            else np.zeros((0, 4), np.float32)
        ),
        np.array(ys, dtype=np.int64),
    )


def _calibrate_fusion_and_thresholds(reg_vals, cls_logits, ord_logits, y_true):
    if len(y_true) < 50:
        return [0.75, 1.5, 2.5, 3.5], np.array([1.0, 1.0, 1.0], dtype=np.float32)

    p_cls = np.exp(cls_logits - cls_logits.max(axis=1, keepdims=True))
    p_cls = p_cls / np.clip(p_cls.sum(axis=1, keepdims=True), 1e-12, None)

    o_sig = 1.0 / (1.0 + np.exp(-ord_logits))  # sigmoid
    p_ord = np.zeros((len(o_sig), 5), dtype=np.float32)
    p_ord[:, 0] = 1 - o_sig[:, 0]
    p_ord[:, 1] = o_sig[:, 0] * (1 - o_sig[:, 1])
    p_ord[:, 2] = o_sig[:, 1] * (1 - o_sig[:, 2])
    p_ord[:, 3] = o_sig[:, 2] * (1 - o_sig[:, 3])
    p_ord[:, 4] = o_sig[:, 3]
    p_ord = p_ord / np.clip(p_ord.sum(axis=1, keepdims=True), 1e-12, None)

    def score_for(th_try, w_try, use_soft=True):
        if use_soft:
            p_reg = _regress_to_prob_soft(reg_vals, th_try, softness=0.18)
        else:
            p_reg = _regress_to_prob_thresholded(reg_vals, th_try)
        pred = _predict_from_fused_probs(p_reg, p_cls, p_ord, w_try)
        return cohen_kappa_score(y_true, pred, weights="quadratic")

    w_grid = [
        np.array([1.0, 1.0, 1.0], np.float32),
        np.array([2.0, 1.0, 1.0], np.float32),
        np.array([1.0, 2.0, 1.0], np.float32),
        np.array([1.0, 1.0, 2.0], np.float32),
        np.array([3.0, 1.0, 1.0], np.float32),
        np.array([1.0, 3.0, 1.0], np.float32),
        np.array([1.0, 1.0, 3.0], np.float32),
        np.array([2.0, 2.0, 1.0], np.float32),
        np.array([2.0, 1.0, 2.0], np.float32),
        np.array([1.0, 2.0, 2.0], np.float32),
        np.array([4.0, 1.0, 1.0], np.float32),
        np.array([1.0, 4.0, 1.0], np.float32),
        np.array([1.0, 1.0, 4.0], np.float32),
        np.array([3.0, 2.0, 1.0], np.float32),
        np.array([3.0, 1.0, 2.0], np.float32),
        np.array([2.0, 3.0, 1.0], np.float32),
        np.array([1.0, 3.0, 2.0], np.float32),
        np.array([2.0, 1.0, 3.0], np.float32),
        np.array([1.0, 2.0, 3.0], np.float32),
        np.array([5.0, 1.0, 1.0], np.float32),
        np.array([1.0, 5.0, 1.0], np.float32),
        np.array([1.0, 1.0, 5.0], np.float32),
    ]

    th_best = np.array([0.75, 1.5, 2.5, 3.5], dtype=np.float32)
    best_w = w_grid[0]
    best_s = score_for(th_best, best_w, use_soft=True)

    for w in w_grid[1:]:
        s = score_for(th_best, w, use_soft=True)
        if s > best_s:
            best_s, best_w = s, w

    for pass_id in range(10):
        improved = False
        steps = [0.35, 0.2, 0.1] if pass_id < 3 else [0.12, 0.06, 0.03]
        for k in range(4):
            base = float(th_best[k])
            candidates = [base]
            for st in steps:
                candidates.extend([base - st, base + st])
            candidates = np.array(candidates, dtype=np.float32)

            best_local_th = th_best.copy()
            best_local_w = best_w
            best_local_s = best_s

            for cand in candidates:
                th_try = th_best.copy()
                th_try[k] = float(cand)
                th_try = np.clip(th_try, 0.05, 4.45)
                th_try = np.sort(th_try)
                for j in range(1, 4):
                    if th_try[j] <= th_try[j - 1] + 1e-3:
                        th_try[j] = th_try[j - 1] + 1e-3

                for w_try in w_grid:
                    s = score_for(th_try, w_try, use_soft=True)
                    if s > best_local_s:
                        best_local_s = s
                        best_local_th = th_try
                        best_local_w = w_try

            if best_local_s > best_s + 1e-6:
                best_s = best_local_s
                th_best = best_local_th
                best_w = best_local_w
                improved = True

        if not improved:
            break

    best_hard_s = score_for(th_best, best_w, use_soft=False)
    best_hard_th = th_best.copy()
    best_hard_w = best_w

    for w in w_grid:
        s_h = score_for(th_best, w, use_soft=False)
        if s_h > best_hard_s + 1e-12:
            best_hard_s = s_h
            best_hard_w = w

    for pass_id in range(2):
        steps = [0.06, 0.03]
        for k in range(4):
            base = float(best_hard_th[k])
            candidates = [base]
            for st in steps:
                candidates.extend([base - st, base + st])
            for cand in candidates:
                th_try = best_hard_th.copy()
                th_try[k] = float(cand)
                th_try = np.clip(th_try, 0.05, 4.45)
                th_try = np.sort(th_try)
                for j in range(1, 4):
                    if th_try[j] <= th_try[j - 1] + 1e-3:
                        th_try[j] = th_try[j - 1] + 1e-3
                s_h = score_for(th_try, best_hard_w, use_soft=False)
                if s_h > best_hard_s + 1e-12:
                    best_hard_s = s_h
                    best_hard_th = th_try

    return [float(x) for x in best_hard_th.tolist()], best_hard_w.astype(np.float32)


reg_vals, cls_logits, ord_logits, y_true = _infer_outputs_on_train(
    net3, train_df, train_img_dir, max_images=3200, seed=SEED, batch_size=16
)

calib_thresholds, calib_w = _calibrate_fusion_and_thresholds(
    reg_vals, cls_logits, ord_logits, y_true
)
print("Calibrated thresholds:", calib_thresholds)
print("Calibrated fusion weights:", calib_w.tolist())

threshold = calib_thresholds  # keeps regress2class consistent if used elsewhere



## === cell 6
submission = []

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 25 == 0:
            print("infer", i, "/", len(test_ids))

        image_name = os.path.join(test_img_dir, f"{idx}.png")

        img2 = cv2.imread(image_name)
        if img2 is None:
            submission.append([idx, 0])
            continue

        img2 = crop_image_from_gray(img2)
        if img2 is None:
            submission.append([idx, 0])
            continue

        img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        img2_t = transform2(img2).unsqueeze(0).to(device)

        rg_outputs, cls_outputs, ord_outputs = net3(img2_t)
        rg_outputs = torch.sigmoid(rg_outputs) * 4.5  # (1,1)
        ord_outputs = torch.sigmoid(ord_outputs)  # (1,4)

        p_reg = _regress_to_prob_thresholded(
            np.array(
                [float(rg_outputs.squeeze().detach().cpu().item())], dtype=np.float32
            ),
            threshold,
        )
        p_cls = F.softmax(cls_outputs, dim=1).detach().cpu().numpy().astype(np.float32)
        p_ord = (
            ordinal2class_prob(ord_outputs).detach().cpu().numpy().astype(np.float32)
        )

        P = int(_predict_from_fused_probs(p_reg, p_cls, p_ord, calib_w)[0])
        submission.append([idx, P])

submission = np.array(submission, dtype=object)
print("submission rows:", len(submission))



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df = test_df.merge(df, on="id_code", how="left")
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)
df = df[["id_code", "diagnosis"]]

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(df))
print(df.head())
