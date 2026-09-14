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

0.9299673551684156

# 6. Current score

0.11085

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00184) has done: 'The main runtime error comes from `rg_outputs.tolist()` producing a nested list; after your extra `unsqueeze(1)` the predicted class becomes a list instead of a scalar, which then breaks arithmetic with floats. I remove that unnecessary `unsqueeze(1)` and extract `rg_predict` as a proper Python scalar so the ensemble averaging works. I also ensure the prediction loop always appends to `submission` and that the submission is written as a valid `submission.csv` with the required columns. These changes are score-neutral (they fix type/shape bugs) and keep your model/inference logic intact.'
- What this solution (achieved 0.01335) has done: 'Your current score is extremely far below the target (0.00184 vs 0.9299), which strongly suggests a mismatch between the loaded weights and the model definition used for inference, so predictions are essentially random. I make a minimal, inference-only fix to load the checkpoint in a robust way (handling common “module.” prefixes and nested dict keys like `state_dict`), and I verify the loaded tensors match expected shapes so we don’t silently run with wrong weights. I also align inference preprocessing to the transform you already defined for this exact network (`transform1`), because using the wrong normalization/crop can heavily collapse performance while keeping the same core model. These changes keep your architecture and prediction logic intact, but should move the kappa sharply upward toward your target by restoring the intended trained model + intended preprocessing.'
- What this solution (achieved 0.1634) has done: 'Your current score (0.01335) is far below the target, so we should improve performance without changing the model itself. The biggest low-risk gain is to make inference preprocessing match what EfficientNet-B4 NoisyStudent expects: use bicubic resize to 380×380 and ImageNet mean/std, and remove the extra custom mean/std that likely mismatches the checkpoint’s training. I keep your exact model/heads and the same ensemble logic (regression/classification/ordinal averaging), but I add a tiny test-time augmentation (horizontal flip) and average logits/probabilities, which typically boosts kappa for this task without altering architecture or training. Finally, I make the checkpoint loading slightly more robust for nested keys like `net.`/`backbone.` while still loading strictly when it matches, so we don’t silently run with random weights.'
- What this solution (achieved 0.18654) has done: 'The timeout is dominated by per-image overhead: you run the model twice per image (TTA flip) but also do GPU transfers and model calls one image at a time for 3,295 train + 367 test, plus Python loops inside threshold fitting. I keep the same model, transforms, TTA, and threshold search logic, but batch inference with a proper Dataset/DataLoader to amortize GPU/kernel launch overhead and enable pinned-memory + async H2D copies. I also vectorize the kappa scoring inside the threshold fitting (same grid/steps, same clamp/sort) so it stops spending most time in Python loops. These changes preserve evaluation semantics (same per-image preprocessing, same logits/activations, same averaging and rounding), but significantly reduce runtime.'
- What this solution (achieved 0.18654) has done: 'Your current score (0.18654) is far below the target (0.92997), so we need a real performance gain while keeping your model and overall inference/ensemble logic intact. The most likely cause is a preprocessing mismatch: you are feeding EfficientNet-B4 NoisyStudent with extra trimming/cropping that can shift the optic disc/lesion distribution versus how the checkpoint was trained, which can heavily degrade QWK. I keep your exact network, checkpoint loading, TTA, and threshold-fitting approach, but add an inference-time “fast center crop” transform (no trim/gray-crop) and then *choose between transform1 vs the new transform* by comparing train QWK during the existing threshold calibration step; we use whichever yields higher train QWK (a legitimate selection because it doesn’t use test labels). This is a minimal, score-relevant change and should move the submission closer to your target by restoring the checkpoint’s intended input distribution.'
- What this solution (achieved 0.11085) has done: 'Your current score is far below the target, so we need a real-but-minimal accuracy lift without changing your model or training loop. The biggest low-risk issue is that your ordinal branch is being converted to a hard class by thresholding at 0.5 per-logit, which throws away confidence information; for QWK, using the expected class value from the full ordinal probability mass typically improves calibration and kappa while preserving the same heads and outputs. I keep your same network, checkpoint loading, transforms, TTA, and threshold-fitting procedure, but change only how `ord_pred_float` is computed during threshold calibration and test prediction (from hard count to expected value). I also keep the existing selection between `transform1` and `transform_center` unchanged, just feeding it the improved ordinal aggregation so it can pick the better preprocessing more reliably.'
- What this solution (achieved 0.11085) has done: 'Your current score is far below the target, so the smallest high-impact fix is to make your ordinal branch mathematically consistent: `ordinal2class_prob` should build a valid 5-class probability mass from the 4 “P(y>k)” sigmoid outputs, not softmax an already-normalized construction (which distorts it and hurts QWK). I remove the softmax and clamp/renormalize the ordinal mass to keep it a proper distribution, then use the same expected-value aggregation you already implemented. This preserves your model, weights, transforms, TTA, and threshold fitting logic, but should improve calibration and move QWK upward. I also make the regression threshold default match your earlier threshold list (`[0.75,1.5,2.5,3.5]`) to reduce mismatch during calibration.'

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
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2
import os
import glob

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
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
    Kept for weight compatibility; not used in inference script directly.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, n: Optional[int] = None):
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
    def __init__(self, pretrained=False):
        super(backboneNet_efficient, self).__init__()
        net = timm.create_model("tf_efficientnet_b4_ns", pretrained=pretrained)

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

        if hasattr(net, "act2"):
            self.act2 = net.act2
        elif hasattr(net, "act_head"):
            self.act2 = net.act_head
        else:
            self.act2 = nn.Identity()

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
    eps = 1e-7
    out = out.clamp(min=eps, max=1 - eps)

    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = 1 - out[:, 0]
    pred_prob[:, 1] = out[:, 0] * (1 - out[:, 1])
    pred_prob[:, 2] = out[:, 1] * (1 - out[:, 2])
    pred_prob[:, 3] = out[:, 2] * (1 - out[:, 3])
    pred_prob[:, 4] = out[:, 3]

    pred_prob = pred_prob.clamp(min=0.0)
    s = pred_prob.sum(dim=1, keepdim=True).clamp(min=eps)
    pred_prob = pred_prob / s
    return pred_prob


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
INPUT_ROOT = "../input/aptos2019-blindness-detection"

test_df = pd.read_csv(f"{INPUT_ROOT}/test.csv")
if "id_code" not in test_df.columns:
    raise ValueError(
        f"Expected 'id_code' column in test.csv, got columns={list(test_df.columns)}"
    )
test_ids = test_df["id_code"].astype(str).tolist()
if len(test_ids) == 0:
    raise RuntimeError("No test IDs found; cannot create a non-empty submission.")

train_df = pd.read_csv(f"{INPUT_ROOT}/train.csv")
if not set(["id_code", "diagnosis"]).issubset(train_df.columns):
    raise ValueError(
        f"Expected columns id_code, diagnosis in train.csv, got columns={list(train_df.columns)}"
    )

transform1 = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize(
            (380, 380), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_center = transforms.Compose(
    [
        transforms.Resize(
            (400, 400), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    return obj


def _strip_prefix(sd, prefix):
    if not isinstance(sd, dict):
        return sd
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[len(prefix) :]: v for k, v in sd.items()}


def _try_load_strict_with_prefix_fixes(model, state):
    candidates = []
    s0 = state
    candidates.append(s0)
    for p in ["module.", "model.", "net.", "backbone."]:
        candidates.append(_strip_prefix(s0, p))
    for s in candidates:
        try:
            model.load_state_dict(s, strict=True)
            return True, None
        except Exception as e:
            last_err = e
    return False, last_err


weight_candidates = [
    "../input/weights/1.pth",
    "../input/aptos2019-blindness-detection/weights/1.pth",
]
weight_path = None
for p in weight_candidates:
    if os.path.exists(p):
        weight_path = p
        break
if weight_path is None:
    found = glob.glob("../input/**/1.pth", recursive=True)
    if len(found) > 0:
        weight_path = found[0]

net2 = backboneNet_efficient(pretrained=False)

if weight_path is not None:
    raw = torch.load(weight_path, map_location="cpu")
    state = _extract_state_dict(raw)
    ok, err = _try_load_strict_with_prefix_fixes(net2, state)
    if not ok:
        raise RuntimeError(
            f"Failed to strictly load checkpoint weights from {weight_path}. Error: {err}"
        )
    print("Loaded weights from:", weight_path)
else:
    net2 = backboneNet_efficient(pretrained=True)
    print(
        "WARNING: Could not find '1.pth' under ../input; using ImageNet-pretrained backbone weights instead."
    )

net2 = net2.to(device)
net2.eval()



## === cell 6
submission = []


class RetinaDataset(Dataset):
    def __init__(self, ids, labels, img_dir, transform, use_gray_crop=True):
        self.ids = list(ids)
        self.labels = None if labels is None else list(labels)
        self.img_dir = img_dir
        self.transform = transform
        self.use_gray_crop = use_gray_crop

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = f"{self.img_dir}/{idx}.png"
        img = cv2.imread(image_name)
        if img is None:
            return idx, None, None if self.labels is None else self.labels[i]

        if self.use_gray_crop:
            img = crop_image_from_gray(img)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        x = self.transform(img)  # CHW float tensor
        y = None if self.labels is None else int(self.labels[i])
        return idx, x, y


def _collate_skip_missing(batch):
    ids, xs, ys = [], [], []
    for b in batch:
        idx, x, y = b
        if x is None:
            continue
        ids.append(idx)
        xs.append(x)
        ys.append(y)
    if len(xs) == 0:
        return [], None, None
    xs = torch.stack(xs, dim=0)
    if all(v is None for v in ys):
        ys = None
    else:
        ys = torch.tensor(ys, dtype=torch.long)
    return ids, xs, ys


def _seed_worker(worker_id):
    base = 42 + worker_id
    np.random.seed(base)
    random.seed(base)
    torch.manual_seed(base)


def _predict_batch(x_bchw):
    rg_outputs, cls_outputs, ord_outputs = net2(x_bchw)
    rg_outputs = torch.sigmoid(rg_outputs) * 4.5
    cls_probs = torch.softmax(cls_outputs, dim=1)
    ord_probs = torch.sigmoid(ord_outputs)
    return rg_outputs, cls_probs, ord_probs


def _apply_regression_thresholds_vec(rg_vals, thrs):
    thrs = np.asarray(thrs, dtype=np.float64)
    rg_vals = np.asarray(rg_vals, dtype=np.float64)
    return (rg_vals[:, None] >= thrs[None, :]).sum(axis=1).astype(np.int64)


def _score_kappa_from_preds_vec(
    y_true, rg_pred_float, cls_pred_int, ord_pred_float, thrs
):
    y_true = np.asarray(y_true, dtype=np.int64)
    rg_pred_float = np.asarray(rg_pred_float, dtype=np.float64)
    cls_pred_int = np.asarray(cls_pred_int, dtype=np.float64)
    ord_pred_float = np.asarray(ord_pred_float, dtype=np.float64)

    rg_cls = _apply_regression_thresholds_vec(rg_pred_float, thrs).astype(np.float64)
    p = np.rint((rg_cls + cls_pred_int + ord_pred_float) / 3.0)
    p = np.clip(p, 0, 4).astype(np.int64)
    return cohen_kappa_score(y_true, p, weights="quadratic")


def _fit_thresholds_qwk(y_true, rg_pred_float, cls_pred_int, ord_pred_float, init_thrs):
    thrs = np.array(init_thrs, dtype=np.float64)

    def clamp_and_sort(t):
        t = np.array(t, dtype=np.float64)
        t = np.clip(t, 0.0, 4.5)
        eps = 1e-6
        t = np.maximum.accumulate(t + eps * np.arange(4))
        t[0] = min(t[0], 4.5 - 3 * eps)
        t[1] = min(t[1], 4.5 - 2 * eps)
        t[2] = min(t[2], 4.5 - 1 * eps)
        t[3] = min(t[3], 4.5)
        t = np.maximum.accumulate(t)
        return t

    thrs = clamp_and_sort(thrs)
    best = _score_kappa_from_preds_vec(
        y_true, rg_pred_float, cls_pred_int, ord_pred_float, thrs
    )

    for step in [0.10, 0.02]:
        improved = True
        while improved:
            improved = False
            for j in range(4):
                for delta in (-step, step):
                    t2 = thrs.copy()
                    t2[j] += delta
                    t2 = clamp_and_sort(t2)
                    sc = _score_kappa_from_preds_vec(
                        y_true, rg_pred_float, cls_pred_int, ord_pred_float, t2
                    )
                    if sc > best + 1e-8:
                        best = sc
                        thrs = t2
                        improved = True
    return thrs, best


BATCH_SIZE_TRAIN = 16 if device.type == "cuda" else 4
BATCH_SIZE_TEST = 16 if device.type == "cuda" else 4
NUM_WORKERS = min(4, os.cpu_count() or 1)
PIN_MEMORY = device.type == "cuda"

train_ids = train_df["id_code"].astype(str).tolist()
train_labels = train_df["diagnosis"].astype(int).tolist()


def _calibrate_for_transform(transform, use_gray_crop, tag):
    train_ds = RetinaDataset(
        train_ids,
        train_labels,
        f"{INPUT_ROOT}/train_images",
        transform,
        use_gray_crop=use_gray_crop,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=_seed_worker,
        collate_fn=_collate_skip_missing,
    )

    rg_preds = []
    cls_preds = []
    ord_preds = []
    y_used = []

    t0 = time.time()
    with torch.no_grad():
        seen = 0
        for ids_b, x_b, y_b in train_loader:
            if x_b is None:
                continue
            seen += len(ids_b)
            if seen % 400 == 0:
                print(
                    f"[{tag}] Calibrating thresholds - infer train {seen}/{len(train_ids)}"
                )

            x_b = x_b.to(device, non_blocking=True)
            x_flip = torch.flip(x_b, dims=[3])

            rg1, cls1, ord1 = _predict_batch(x_b)
            rg2, cls2, ord2 = _predict_batch(x_flip)

            rg_out = (rg1 + rg2) / 2.0
            cls_out = (cls1 + cls2) / 2.0
            ord_out = (ord1 + ord2) / 2.0  # shape [B,4] in [0,1]

            rg_preds.extend(
                rg_out.squeeze(1).detach().cpu().numpy().astype(np.float64).tolist()
            )
            cls_preds.extend(
                torch.argmax(cls_out, dim=1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
                .tolist()
            )

            ord_class_prob = ordinal2class_prob(ord_out)  # [B,5]
            ord_expect = (
                ord_class_prob
                * torch.arange(5, device=ord_out.device, dtype=ord_out.dtype)
            ).sum(dim=1)
            ord_preds.extend(
                ord_expect.detach().cpu().numpy().astype(np.float64).tolist()
            )

            y_used.extend(y_b.numpy().astype(np.int64).tolist())

    used_n = len(rg_preds)
    if used_n == 0:
        raise RuntimeError(
            f"[{tag}] No training images were successfully read; cannot calibrate thresholds."
        )

    default_thrs = [0.75, 1.5, 2.5, 3.5]
    fit_thrs, fit_kappa = _fit_thresholds_qwk(
        y_used, rg_preds, cls_preds, ord_preds, default_thrs
    )
    print(f"[{tag}] Threshold calibration done in %.1fs" % (time.time() - t0))
    print(f"[{tag}] Fitted regression thresholds:", fit_thrs.tolist())
    print(
        f"[{tag}] Train QWK using fitted thresholds (for selection):", float(fit_kappa)
    )
    return fit_thrs, float(fit_kappa)


fit_thrs_1, k1 = _calibrate_for_transform(
    transform1, use_gray_crop=True, tag="transform1(trim+4:3+graycrop)"
)
fit_thrs_c, kc = _calibrate_for_transform(
    transform_center, use_gray_crop=False, tag="transform_center(simple center-crop)"
)

if kc > k1:
    best_transform = transform_center
    best_use_gray_crop = False
    fit_thrs = fit_thrs_c
    print("Selected transform_center based on higher train QWK.")
else:
    best_transform = transform1
    best_use_gray_crop = True
    fit_thrs = fit_thrs_1
    print("Selected transform1 based on higher (or equal) train QWK.")

test_ds = RetinaDataset(
    test_ids,
    labels=None,
    img_dir=f"{INPUT_ROOT}/test_images",
    transform=best_transform,
    use_gray_crop=best_use_gray_crop,
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE_TEST,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    worker_init_fn=_seed_worker,
    collate_fn=_collate_skip_missing,
)

with torch.no_grad():
    done = 0
    for ids_b, x_b, _ in test_loader:
        if x_b is None:
            continue
        if done % 100 == 0:
            print("Predicting", done, "/", len(test_ids))
        done += len(ids_b)

        x_b = x_b.to(device, non_blocking=True)
        x_flip = torch.flip(x_b, dims=[3])

        rg1, cls1, ord1 = _predict_batch(x_b)
        rg2, cls2, ord2 = _predict_batch(x_flip)

        rg_out = (rg1 + rg2) / 2.0
        cls_out = (cls1 + cls2) / 2.0
        ord_out = (ord1 + ord2) / 2.0

        cls_predict = (
            torch.argmax(cls_out, dim=1).detach().cpu().numpy().astype(np.int64)
        )
        rg_value = rg_out.squeeze(1).detach().cpu().numpy().astype(np.float64)
        rg_predict = _apply_regression_thresholds_vec(rg_value, fit_thrs).astype(
            np.int64
        )

        ord_class_prob = ordinal2class_prob(ord_out)  # [B,5]
        ord_expect = (
            ord_class_prob * torch.arange(5, device=ord_out.device, dtype=ord_out.dtype)
        ).sum(dim=1)
        ord_predict = ord_expect.detach().cpu().numpy().astype(np.float64)

        P = np.rint(
            (
                rg_predict.astype(np.float64)
                + cls_predict.astype(np.float64)
                + ord_predict
            )
            / 3.0
        )
        P = np.clip(P, 0, 4).astype(np.int64)

        submission.extend([[idx, int(p)] for idx, p in zip(ids_b, P.tolist())])

if len(submission) != len(test_ids):
    have = set([r[0] for r in submission])
    missing = [i for i in test_ids if i not in have]
    if len(missing) > 0:
        raise FileNotFoundError(
            f"Could not read {len(missing)} test images. Example missing: {missing[0]}"
        )

submission = np.array(submission, dtype=object)
print("Built submission rows:", submission.shape[0])



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

if df.shape[0] == 0:
    raise RuntimeError(
        "Submission DataFrame is empty; cannot write a valid submission."
    )

df = df.set_index("id_code").loc[test_ids].reset_index()

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("Saved to:", os.path.abspath("submission.csv"))
