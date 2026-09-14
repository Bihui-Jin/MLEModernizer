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

0.9258867812669724

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the hard failures preventing an end-to-end run: (1) remove the broken `pip install` of a non-existent wheel and instead use the already-installed `timm`, (2) make the code CPU-safe by selecting `cuda` only if available and removing `.cuda()` assumptions, and (3) fix the missing weights path by searching common Kaggle input locations and falling back to an untrained model only if weights truly aren’t present (still producing a valid CSV). I also fix a couple of runtime bugs in transforms (`trim()` returning `None`) and a Python logic bug (`is` vs `==` for string compare) that can break preprocessing. Finally, I ensure the submission is always populated and saved as `submission.csv` with the correct columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with running an untrained model because the expected weight file is not present in your environment, so the smallest legitimate improvement is to (1) reliably locate any `.pkl/.pth` weights shipped inside the dataset folders and (2) load them in a tolerant way (handle common “state_dict” wrappers and `module.` prefixes) without changing the model itself. I also ensure inference uses the exact `final=True` head that the ThreeStage model provides (still the same architecture/outputs, just using the intended combined regressor instead of only the intermediate regression branch), which typically improves QWK for this style of model without changing training. Finally, I keep the submission ordering aligned to `test.csv` and always write `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with “all/most predictions collapse to the same class”, which commonly happens here because `regress2class()` is designed for scalar thresholds but your inference passes a tensor and uses a `.cpu().item()` pattern that effectively uses a single boolean to set all thresholds. I keep your exact model and weights-loading logic, but fix `regress2class()` to be tensor-safe and return the intended per-sample class mapping (for batch size 1, it becomes correct and deterministic). I also clamp the final regressor output into the valid [0, 4] range before binning to avoid edge effects from the model’s 0..4.5 scaling. These are minimal, evaluation-relevant changes that should move the score upward toward your target without changing architecture or training.'
- What this solution (achieved 0.0) has done: 'Your 0.0 QWK strongly suggests the current submission predictions are effectively broken/collapsed (e.g., almost all the same class), and the most likely cause is that `regress2class()` still isn’t actually being applied per-sample in a tensor-safe way during inference. I make the smallest evaluation-relevant fix: update `regress2class()` to accept tensors/arrays of any shape and return per-element classes, then use it in cell 6 without the fragile scalar-only assumptions. I also keep your exact model/weights logic and inference path (`final=True`) unchanged, but ensure the final predictions are clipped to [0, 4] and converted correctly for every test image. This should legitimately move the score upward toward your target while preserving your core modeling approach and producing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 QWK is most consistent with predictions collapsing due to a broken weights load (e.g., loading the wrong `.pth/.pkl` from a different model) rather than the inference loop itself. I make the smallest score-relevant change: restrict the automatic weight search to only files that look like the intended ThreeStage EfficientNet-B4 checkpoint (and refuse clearly incompatible checkpoints), so you don’t silently run with random/unrelated weights. I also harden the checkpoint extraction to handle common wrappers (`{"model": ...}`, `{"state_dict": ...}`) and ensure we only accept a state_dict that contains this model’s keys (e.g., backbone blocks). Everything else (model, transforms, inference, submission format) stays the same to preserve core logic and semantics while moving score upward toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the model effectively running with random/unloaded weights (or silently loading an incompatible checkpoint), which collapses predictions and yields near-random agreement. I make the smallest score-relevant change by (1) expanding the weight search to include common EfficientNet-B4 ThreeStage checkpoint naming patterns and (2) tightening checkpoint validation by checking that key tensor shapes match this exact model (not just key names), so we don’t “accept” wrong weights. I also ensure BatchNorm/Dropout behavior is fully in eval mode (already mostly done) and keep your inference path (`final=True`, same transforms, same thresholding) unchanged to preserve core logic. This should legitimately move QWK upward toward your target when the correct weights exist in the environment, while still producing a valid `submission.csv` either way.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the model effectively running without the intended pretrained checkpoint (or silently rejecting it), which makes predictions near-random/collapsed. The smallest score-relevant improvement is to (1) broaden the weight search to include common checkpoint extensions like `.bin` and to (2) accept a matching checkpoint even if it’s a full model object (not just a raw `state_dict` dict) by extracting `state_dict()` when present. To avoid a second common “0.0” failure mode, I also make `combine3output()` tensor-safe (it currently `.item()`s class outputs and breaks if ever used on batched tensors), while keeping your inference path unchanged (`final=True` and `regress2class()` binning). Everything else (model, transforms, prediction logic, submission format/path) stays the same.'

# 9. Code solution

## === cell 0
import os
import glob
import math
import random
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT

from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
import time
import copy
import json
import pickle

from torch.optim import lr_scheduler
from torch.utils.data import random_split

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


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    (Defined here because original notebook included it; keep for compatibility.)
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
        x4 = self.block0(x1)  # preserve original (even if odd)
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
    out_t = torch.as_tensor(out).detach().float().reshape(-1)
    preds = torch.zeros_like(out_t, dtype=torch.long)
    for t in threshold:
        preds += (out_t >= t).long()
    if preds.numel() == 1:
        return int(preds.item())
    return preds.cpu().numpy().astype(int)


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
    R = torch.as_tensor(regress2class(r_out.data))
    C = torch.argmax(c_out.data, dim=1).long().view(-1)
    O = torch.argmax(o_out.data, dim=1).long().view(-1)
    P = (R.to(C.device).view(-1) + C + O).float() / 3.0
    P = torch.round(P).long().clamp(0, 4)
    return int(P.item()) if P.numel() == 1 else P.cpu().numpy().astype(int)




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
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _extract_state_dict(obj):
    if (
        hasattr(obj, "state_dict")
        and callable(getattr(obj, "state_dict"))
        and not isinstance(obj, dict)
    ):
        try:
            return obj.state_dict()
        except Exception:
            pass

    if isinstance(obj, dict):
        for k in ("state_dict", "model", "net", "model_state", "model_state_dict"):
            if k in obj:
                v = obj[k]
                if isinstance(v, dict):
                    return v
                if hasattr(v, "state_dict") and callable(getattr(v, "state_dict")):
                    try:
                        return v.state_dict()
                    except Exception:
                        pass
        return obj
    return obj


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _is_plausible_threestage_b4_state_dict(state_dict: dict) -> bool:
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return False

    must_have_any = [
        "backbone.conv_stem.weight",
        "backbone.blocks.0.0.conv_dw.weight",
        "classifier.1.weight",
        "regressor.1.weight",
        "ordinal.1.weight",
        "final_regressor.1.weight",
    ]
    if not any(k in state_dict for k in must_have_any):
        return False

    def _shape(k):
        v = state_dict.get(k, None)
        return tuple(v.shape) if hasattr(v, "shape") else None

    shape_checks = [
        ("classifier.1.weight", (500, 1000)),
        ("classifier.3.weight", (5, 500)),
        ("regressor.1.weight", (500, 1000)),
        ("regressor.3.weight", (1, 500)),
        ("ordinal.1.weight", (500, 1000)),
        ("ordinal.3.weight", (4, 500)),
        ("final_regressor.1.weight", (1, 10)),
    ]
    ok = 0
    for k, s in shape_checks:
        sh = _shape(k)
        if sh is None:
            continue
        if sh == s:
            ok += 1
    return ok >= 4  # tolerate partial/missing keys but require strong evidence


def _find_weight_file(search_roots, name_hints=None):
    if name_hints is None:
        name_hints = [
            "b4_3stage",
            "3stage",
            "three_stage",
            "threestage",
            "stage3",
            "efficientnet_b4",
            "tf_efficientnet_b4",
            "finetune",
            "aptos",
            "blindness",
        ]

    exts = ("*.pkl", "*.pth", "*.pt", "*.bin")
    candidates = []
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for ext in exts:
            candidates.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))

    hinted = []
    for p in candidates:
        base = os.path.basename(p).lower()
        if any(h.lower() in base for h in name_hints):
            hinted.append(p)
    ordered = (
        sorted(hinted, key=lambda x: (len(x), x))
        if hinted
        else sorted(candidates, key=lambda x: (len(x), x))
    )

    for p in ordered:
        try:
            ckpt = torch.load(p, map_location="cpu")
            state = _strip_module_prefix(_extract_state_dict(ckpt))
            if _is_plausible_threestage_b4_state_dict(state):
                return p
        except Exception:
            continue
    return None


DATA_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
]
DATA_DIR = _first_existing(DATA_CANDIDATES)
if DATA_DIR is None:
    raise FileNotFoundError(f"Could not find dataset dir in: {DATA_CANDIDATES}")

TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

test_df = pd.read_csv(TEST_CSV)
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

WEIGHTS_CANDIDATES = [
    "/kaggle/input/weights/B4_3stage_13epoch_320finetune.pkl",
    "../input/weights/B4_3stage_13epoch_320finetune.pkl",
    "/kaggle/input/aptos2019-blindness-detection/B4_3stage_13epoch_320finetune.pkl",
    "../input/aptos2019-blindness-detection/B4_3stage_13epoch_320finetune.pkl",
]
weights_path = _first_existing(WEIGHTS_CANDIDATES)

if weights_path is None:
    weights_path = _find_weight_file(
        search_roots=[
            "/kaggle/input",
            "../input",
            "/kaggle/data",
            DATA_DIR,
        ]
    )

net1 = ThreeStage_Model()

if weights_path is not None:
    ckpt = torch.load(weights_path, map_location="cpu")
    state = _strip_module_prefix(_extract_state_dict(ckpt))

    if not _is_plausible_threestage_b4_state_dict(state):
        warnings.warn(
            f"Found weights at {weights_path} but state_dict doesn't match ThreeStage_Model; ignoring."
        )
        weights_path = None
    else:
        missing, unexpected = net1.load_state_dict(state, strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            warnings.warn(
                f"Loaded weights from {weights_path} with strict=False; missing={len(missing)}, unexpected={len(unexpected)}"
            )
else:
    warnings.warn(
        "Pretrained weights not found; predictions will come from an untrained model (likely very low score)."
    )

net1 = net1.to(device)
net1.eval()
for m in net1.modules():
    if isinstance(m, (nn.Dropout, nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
        m.eval()

print(f"DATA_DIR={DATA_DIR}")
print(f"TEST_IMG_DIR={TEST_IMG_DIR}")
print(f"weights_path={weights_path}")



## === cell 6
submission = []
missing_images = 0

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
        if not os.path.exists(image_name):
            missing_images += 1
            submission.append([idx, 0])
            continue

        img = Image.open(image_name).convert("RGB")
        img1 = transform1(img).unsqueeze(0).to(device)

        out = net1(img1, final=True)  # [1,1] in ~[0,4.5]
        out = out.clamp(0.0, 4.0)

        pred1 = regress2class(out)
        submission.append([idx, int(pred1)])

submission = np.array(submission, dtype=object)

if missing_images > 0:
    warnings.warn(
        f"{missing_images} test images were missing; filled with diagnosis=0."
    )



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["diagnosis"] = df["diagnosis"].astype(int)

df = df.set_index("id_code").loc[test_ids].reset_index()

out_path = "submission.csv"
df.to_csv(out_path, index=False)

print(df.head())
print(f"Saved submission to: {out_path} with shape {df.shape}")
