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

0.9011750960803924

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix three execution blockers while keeping your model and inference logic intact: (1) make the code run on CPU when no GPU is available (your current `cuda:0` hard-crashes), (2) robustly locate the pretrained weight file and load it with `map_location` so it works in Kaggle’s filesystem, and (3) remove hardcoded `.cuda()` tensor creation inside helper functions so inference doesn’t fail on CPU. Finally, I ensure the loop produces a non-empty `submission.csv` in the required format by keeping ids as strings and adding a safe fallback if weights are missing (so you still get a valid CSV even if the weight file truly isn’t present). These changes are score-neutral when weights load successfully, and they unblock end-to-end execution.'
- What this solution (achieved 0.00014) has done: 'I (1) make weight loading robust to different checkpoint formats and ensure the model weights are moved onto the same device as the inputs to fix the CUDA/CPU dtype mismatch, (2) stop hard-failing when the external weights file is missing and instead fall back to the model’s own pretrained backbone weights (so you still get a valid, non-empty submission CSV in this Kaggle environment), and (3) make inference resilient to missing/corrupt images so the loop always completes. These changes preserve your model/inference logic (same architecture and same `regress2class` mapping) and primarily fix execution blockers so a valid `submission.csv` is produced. With the pretrained-backbone fallback, the score should be substantially better than the previous “not yielded/0.0” situation while remaining faithful to your existing approach.'
- What this solution (achieved 0.67186) has done: 'Your current score (0.00014) is far below the target (0.9012), and the biggest reason is that when the external checkpoint is missing you fall back to an untrained head, producing near-random ordinal outputs. I keep your exact model, transforms, and `regress2class` mapping, but make the fallback path legitimately stronger by fitting only the final regressor layer on the provided training set using frozen pretrained EfficientNet features (same forward semantics; no architecture changes). This is a minimal, metric-aligned improvement because it makes the regression output meaningful, which directly improves quadratic weighted kappa after thresholding. I also ensure robust image loading (PIL fallback for OpenCV issues) and keep the output CSV format/ordering identical.'
- What this solution (achieved -0.03689) has done: 'Your current score (0.67186) is well below the target (0.9012), so we should cautiously improve performance while keeping your model, transforms, and `regress2class` mapping intact. The biggest low-risk gain is to make the fallback “head-fit” path actually train the exact regressor head you use at inference (`net1.regressor`) by feeding it the same feature representation it sees at inference, which your current code does not do (it calls `net1(x)` and unintentionally uses the backbone’s original `forward`, not the GeM-pooled feature vector the regressor expects). I minimally patch the fallback training to compute backbone features via `forward_features` + GeM pool (no architecture change) and then run `net1.regressor(features)` so gradients correctly update the regressor. I also keep everything deterministic and ensure the submission writing remains identical.'
- What this solution (achieved -0.00264) has done: 'I fix the fallback head-fitting path so it trains the exact regressor head with the correct feature dimensionality (your EfficientNet-B4 backbone produces 1792-d pooled features, but the regressor expects 1000-d inputs, causing the matmul shape crash). To keep the core model and inference semantics intact, I extract features using the same backbone path but then add a minimal linear projection (1792→1000) used only in the fallback training loop, so `net1.regressor(...)` receives the expected shape without changing the model definition. I also ensure the projection is trained together with the regressor head (backbone stays frozen), and keep everything device-safe/deterministic. This unblocks end-to-end execution and should improve score versus the current broken/degenerate fallback behavior, while leaving the external-weight path unchanged.'
- What this solution (achieved 0.0) has done: 'Main bottlenecks are (1) per-image prediction loop with single-image GPU calls and repeated PIL decoding/transform overhead, and (2) an expensive fallback path that trains on the full training set if weights aren’t found. To fit in 600s without changing the model or semantics, I batch test-time inference with a DataLoader (same transforms, same thresholding) and enable pinned-memory/non-blocking transfers plus persistent workers to reduce CPU↔GPU stalls. I also speed up the fallback training/calibration path by caching extracted backbone features once (exactly equivalent to re-extracting them every epoch since backbone is frozen) and by vectorizing threshold search evaluation (same grid/logic, just fewer Python loops). These changes preserve the architecture, losses, thresholds, and prediction mapping, only removing redundant work and Python overhead.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with a “bad fallback” path being used (external weights missing) plus a mismatch between what the regressor is trained on (projected features) and what inference sees, so I make the fallback training/inference use exactly the same feature tensor path to reduce that gap without changing your model definition. I also stop `shuffle=True` in the cached-feature regressor fitting so the learned mapping is deterministic given your fixed seeds (this tends to stabilize QWK and avoids occasional regressions). Finally, I keep your thresholding semantics but calibrate thresholds using out-of-fold predictions (single split) instead of in-sample predictions, which is a minimal, metric-aligned change that typically improves public QWK and should move you toward the 0.90 target. All changes are only active when external weights are not found; if the checkpoint exists, your original inference path remains unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score indicates the submission is likely being scored as essentially constant/random, which most commonly happens here when the fallback path is used and the regressor-only fit is too weak/miscalibrated for QWK. To move toward the 0.901 target without changing your core model/inference semantics, I keep the exact architecture and thresholding approach but (only when external weights are missing) train the regressor+projection a bit longer and calibrate thresholds on out-of-fold predictions from a 5-fold split (instead of a single holdout), which is a minimal, metric-aligned improvement for QWK. I also make the fallback training actually use the same regression head output tensor shape consistently and ensure the calibrated thresholds are strictly increasing and stable. The external-weights path remains unchanged.'

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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm
import os
import glob
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

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



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
    prediction = 0
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu().item()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)
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
DATA_ROOT = "../input/aptos2019-blindness-detection"
train_df = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
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

net1 = ThreeStage_Model()

weight_filename = "B4_3stage_24epoch_320.pkl"
search_roots = ["../input", "/kaggle/input"]
candidate_paths = []
for root in search_roots:
    candidate_paths.append(os.path.join(root, "weights", weight_filename))
    candidate_paths.extend(
        glob.glob(os.path.join(root, "**", weight_filename), recursive=True)
    )

weights_path = None
for p in candidate_paths:
    if os.path.isfile(p):
        weights_path = p
        break

loaded_external_weights = False
if weights_path is not None:
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    missing, unexpected = net1.load_state_dict(cleaned, strict=False)
    print("Loaded external weights from:", weights_path)
    if len(missing) > 0:
        print("Missing keys (non-fatal, strict=False):", missing[:10], "...")
    if len(unexpected) > 0:
        print("Unexpected keys (non-fatal, strict=False):", unexpected[:10], "...")
    loaded_external_weights = True
else:
    print(
        f"WARNING: External weights '{weight_filename}' not found. "
        "Falling back to pretrained backbone + lightweight fitting of the regressor head."
    )
    net1.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
    net1.backbone.global_pool = GeM(flatten=True)

net1.fallback_proj = None

net1 = net1.to(device)
net1.eval()
print(
    "Model on device:",
    next(net1.parameters()).device,
    "| external_weights_loaded:",
    loaded_external_weights,
)




## === cell 6
class APTOSRegressionDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            return img
        except Exception:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_id = str(row["id_code"])
        y = float(row["diagnosis"])
        path = os.path.join(self.img_dir, f"{img_id}.png")
        img = self._load_image(path)
        x = self.transform(img)
        return x, torch.tensor([y], dtype=torch.float32)


if not loaded_external_weights:
    train_img_dir = os.path.join(DATA_ROOT, "train_images")
    ds = APTOSRegressionDataset(train_df, train_img_dir, transform1)

    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,  # deterministic cache order
        num_workers=min(4, os.cpu_count() or 2),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if (os.cpu_count() or 0) > 1 else False,
    )

    for p in net1.parameters():
        p.requires_grad = False
    for p in net1.regressor.parameters():
        p.requires_grad = True

    net1.train()

    def _extract_features_b4ns(m, xb):
        feats = m.backbone.forward_features(xb)  # (B, C, H, W)
        feats = m.backbone.global_pool(feats)  # (B, C)
        return feats

    with torch.no_grad():
        xb0, _ = next(iter(loader))
        xb0 = xb0.to(device, non_blocking=True)
        f0 = _extract_features_b4ns(net1, xb0)
        feat_dim = int(f0.shape[1])

    proj = nn.Linear(feat_dim, 1000, bias=True).to(device)
    for p in proj.parameters():
        p.requires_grad = True

    t_cache = time.time()
    all_feats = []
    all_y = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            feats = _extract_features_b4ns(net1, xb).detach().cpu()
            all_feats.append(feats)
            all_y.append(yb.detach().cpu())
    X_cache = torch.cat(all_feats, dim=0)  # (N, feat_dim) on CPU
    Y_cache = torch.cat(all_y, dim=0)  # (N, 1) on CPU
    print(
        f"Cached frozen-backbone features: X={tuple(X_cache.shape)} in {time.time()-t_cache:.1f}s"
    )

    class _FeatureTensorDataset(Dataset):
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, i):
            return self.X[i], self.Y[i]

    feat_ds_all = _FeatureTensorDataset(X_cache, Y_cache)
    feat_loader_all = DataLoader(
        feat_ds_all,
        batch_size=256,
        shuffle=False,  # deterministic order for stability
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    opt = optim.Adam(
        list(net1.regressor.parameters()) + list(proj.parameters()), lr=1e-3
    )
    loss_fn = nn.MSELoss()

    epochs = 6
    for ep in range(epochs):
        t0 = time.time()
        running = 0.0
        n = 0
        for xfeat_cpu, yb_cpu in feat_loader_all:
            xfeat = xfeat_cpu.to(device, non_blocking=True)
            yb = yb_cpu.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            xfeat_1000 = proj(xfeat)  # (B, 1000)
            r_out = net1.regressor(xfeat_1000)
            r_out = torch.sigmoid(r_out) * 4.5  # keep identical regression scaling

            loss = loss_fn(r_out, yb)
            loss.backward()
            opt.step()

            running += loss.item() * xfeat.size(0)
            n += xfeat.size(0)

        print(
            f"Fallback head-fit epoch {ep+1}/{epochs} | loss={running/max(n,1):.5f} | time={time.time()-t0:.1f}s"
        )

    net1.fallback_proj = proj.eval()
    net1.eval()

    def _predict_reg_from_cached(X_cpu, batch=1024):
        preds_reg = []
        with torch.no_grad():
            for start in range(0, X_cpu.shape[0], batch):
                xfeat = X_cpu[start : start + batch].to(device, non_blocking=True)
                xfeat_1000 = net1.fallback_proj(xfeat)
                r_out = net1.regressor(xfeat_1000)
                r_out = torch.sigmoid(r_out) * 4.5
                preds_reg.append(r_out.squeeze(1).cpu().numpy())
        return np.concatenate(preds_reg, axis=0)

    def _apply_thresholds_vec(reg, thr):
        reg = reg.reshape(-1)
        thr = np.asarray(thr, dtype=np.float32).reshape(1, 4)
        return (reg.reshape(-1, 1) >= thr).sum(axis=1).astype(np.int64)

    y_all = Y_cache.squeeze(1).numpy().astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_reg = np.zeros(len(y_all), dtype=np.float32)
    for fold, (_, va_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all), 1):
        X_va = X_cache[va_idx]
        oof_reg[va_idx] = _predict_reg_from_cached(X_va).astype(np.float32)
        print(f"OOF fold {fold}/5 done | n_val={len(va_idx)}")

    base_thr = np.array(threshold, dtype=np.float32)
    best_thr = base_thr.copy()
    best_kappa = cohen_kappa_score(
        y_all, _apply_thresholds_vec(oof_reg, best_thr), weights="quadratic"
    )

    grids = [
        np.linspace(0.2, 1.6, 15),  # for t0
        np.linspace(0.8, 2.4, 17),  # for t1
        np.linspace(1.6, 3.2, 17),  # for t2
        np.linspace(2.4, 4.2, 19),  # for t3
    ]

    for _ in range(3):  # a few coordinate-ascent passes (still fast)
        for j in range(4):
            cand = best_thr.copy()
            for v in grids[j]:
                cand[j] = float(v)
                if not (cand[0] < cand[1] < cand[2] < cand[3]):
                    continue
                k = cohen_kappa_score(
                    y_all,
                    _apply_thresholds_vec(oof_reg, cand),
                    weights="quadratic",
                )
                if k > best_kappa:
                    best_kappa = k
                    best_thr = cand.copy()

    best_thr = np.sort(best_thr)
    for j in range(1, 4):
        if best_thr[j] <= best_thr[j - 1]:
            best_thr[j] = best_thr[j - 1] + 1e-3

    threshold = [float(x) for x in best_thr.tolist()]
    print(
        "Calibrated fallback thresholds (OOF):",
        threshold,
        "| OOF QWK:",
        float(best_kappa),
    )



## === cell 7
submission = []
test_img_dir = os.path.join(DATA_ROOT, "test_images")


def _predict_regression_fallback(net, xb):
    feats = net.backbone.forward_features(xb)
    feats = net.backbone.global_pool(feats)
    feats_1000 = net.fallback_proj(feats)
    r = net.regressor(feats_1000)
    r = torch.sigmoid(r) * 4.5
    return r


class APTOSTestDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = [str(x) for x in ids]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def _load_image(self, path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)

    def __getitem__(self, i):
        idx = self.ids[i]
        path = os.path.join(self.img_dir, f"{idx}.png")
        img = self._load_image(path)
        x = self.transform(img)
        return idx, x


test_ds = APTOSTestDataset(test_ids, test_img_dir, transform1)
test_loader = DataLoader(
    test_ds,
    batch_size=16 if torch.cuda.is_available() else 4,
    shuffle=False,
    num_workers=min(4, os.cpu_count() or 2),
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True if (os.cpu_count() or 0) > 1 else False,
)

net1.eval()
t_pred = time.time()
with torch.no_grad():
    done = 0
    for ids_batch, xb in test_loader:
        xb = xb.to(device, non_blocking=True)
        if loaded_external_weights:
            _, r_out, _ = net1(xb)  # (B,1)
        else:
            r_out = _predict_regression_fallback(net1, xb)  # (B,1)

        reg = r_out.squeeze(1).detach().cpu().numpy()
        thr = np.asarray(threshold, dtype=np.float32).reshape(1, 4)
        preds = (reg.reshape(-1, 1) >= thr).sum(axis=1).astype(int)

        for idx, p in zip(ids_batch, preds):
            submission.append([idx, int(p)])

        done += len(ids_batch)
        if done % 100 <= len(ids_batch):
            print("Predicting", done, "/", len(test_ids))

print(f"Prediction time: {time.time()-t_pred:.1f}s")
submission = np.array(submission, dtype=object)



## === cell 8
if len(submission) == 0:
    raise RuntimeError("Submission is empty; prediction loop did not run.")

df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df = df.set_index("id_code").reindex(test_df["id_code"].astype(str)).reset_index()
if df["diagnosis"].isna().any():
    missing = df[df["diagnosis"].isna()]["id_code"].head(10).tolist()
    raise RuntimeError(
        f"Found missing predictions after reindexing (sample): {missing}"
    )

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("submission.csv path:", os.path.abspath("submission.csv"))
