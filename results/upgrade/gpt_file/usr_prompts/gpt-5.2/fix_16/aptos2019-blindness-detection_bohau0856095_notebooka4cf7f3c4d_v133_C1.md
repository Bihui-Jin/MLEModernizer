# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.9013950906465862

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the two blockers preventing an end-to-end run: (1) make the script CPU-safe by selecting `cuda` only if available, and (2) resolve the missing weights path by searching the expected Kaggle input folders and loading with `map_location`. Because a missing weights file would otherwise make inference impossible, I add a minimal fallback to run the exact same model with random weights (still producing a valid CSV) rather than crashing. I also fix small logic/runtime issues that can crash transforms (`is` vs `==`, `trim()` returning `None`) and ensure the submission is built from `test.csv` ids with correct dtypes and non-empty output. These changes are correctness/stability-focused; if the weights are present, predictions come from the provided checkpoint and score should move toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the model effectively running untrained (missing checkpoint), and also with a prediction post-processing mismatch: you’re taking only the regressor head and discretizing with fixed thresholds that may not match the checkpoint’s intended ensembling. I make the smallest changes that (1) reliably find and load the checkpoint from the provided dataset folders, and (2) use the model’s existing three-head design by combining regression/classification/ordinal outputs via your already-defined `combine3output` (preserving architecture and inference semantics). I also add a safe image-path fallback to the duplicate `/kaggle/data/...` tree so inference actually runs on real images in this environment, ensuring a non-degenerate submission. These are minimal, directly score-relevant changes and keep the overall approach unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with (a) not actually loading the trained checkpoint (so predictions are effectively random/degenerate) and/or (b) a subtle inference bug in your EfficientNet forward (feeding `block0` the wrong tensor), which would make even a correct checkpoint perform badly. I make the smallest score-relevant fixes: correct the backbone forward to use `x3` (post-stem activations) into `block0`, and make the checkpoint search include the dataset base folder so it can be found in this environment. I also ensure ordinal/classification heads are interpreted consistently (apply `sigmoid` to ordinal logits before argmax) without changing your model or post-processing logic. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the checkpoint not being found/loaded, leaving the model essentially random/degenerate; so the smallest score-relevant improvement is to make checkpoint discovery robust in this environment and to correctly handle common checkpoint formats. I expand the weight search to include the provided `../input/aptos2019-blindness-detection` and `/kaggle/data/...` trees (where Kaggle datasets actually live here), and I add safe unwrapping for checkpoints saved as full dicts (e.g., `{'model': ...}`, `{'model_state_dict': ...}`, `{'net': ...}`), while keeping `strict=False` so architecture stays identical. I also ensure we don’t accidentally apply sigmoid twice to the ordinal head (it’s already sigmoid’d inside `ThreeStage_Model`), because double-sigmoid compresses outputs and can hurt kappa. These are minimal changes focused purely on making inference use the intended trained weights and correct head semantics, producing a valid `submission.csv` as before.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the checkpoint still isn’t being loaded (random weights → near-random predictions) or that inference is subtly misinterpreting the three heads. I make the smallest score-relevant changes to (1) robustly discover the weights file by also searching the provided dataset trees for any `.pkl/.pth/.pt` checkpoint that contains “B4_3stage_19epoch_320”, (2) correctly unwrap more common checkpoint formats (including nested `{'model': {'state_dict': ...}}`), and (3) compute predictions using probability fusion (your already-defined `*_prob` helpers) rather than averaging hard classes, which is more stable for kappa without changing the model. The rest of the pipeline (model, transforms, per-image loop, and CSV format) stays the same and it still always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with inference still not using the intended trained checkpoint (so outputs are effectively random/degenerate), and/or with a head-order mixup when fusing the three outputs. I make the smallest score-relevant changes: (1) make checkpoint discovery robust by scanning for any plausible `.pkl/.pth/.pt` in the dataset tree (not just a specific filename), and unwrap common checkpoint formats more safely; (2) fix the argument order when calling `combine3output_prob` so each head is interpreted correctly (classification logits, regression scalar, ordinal probabilities). These do not change the model architecture or training approach; they only ensure we actually use the provided weights (when available) and correctly post-process outputs for kappa. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the checkpoint not actually being loaded (so predictions are effectively random/degenerate) and/or with a broken interpretation of the ordinal head inside `combine3output_prob` (it expects probabilities but can be passed logits depending on model). I make two minimal, score-relevant changes: (1) make checkpoint loading robust to the common “`state_dict` keys are prefixed (e.g., `model.` / `net.` / `backbone.`)” case by auto-stripping prefixes until keys match, and (2) make `ordinal2class_prob` accept either probabilities or logits by applying a sigmoid only when needed (preventing accidental misuse). These preserve the model architecture and inference loop, but ensure we actually use the intended trained weights and correctly fuse the three heads for a kappa-consistent submission. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with the checkpoint not being loaded (random weights) and/or a head-fusion mismatch during inference. I make two minimal, score-relevant fixes: (1) force a strict filename-prioritized checkpoint search (so we reliably load the intended trained weights when present), and (2) fix the argument order bug when calling `combine3output_prob` (it expects `(r_out, c_out, o_out)` but the code currently passes `(r_out, c_out, o_out)` *to a function defined as* `(r_out, c_out, o_out)`?—the actual bug is that we pass `c_out`/`o_out` into the wrong slots because the function signature is `(r_out, c_out, o_out)` but earlier plans indicate it was being called inconsistently; we enforce the correct order explicitly and validate tensor shapes). I also ensure we use `torch.inference_mode()` (same semantics as `no_grad()` but safer) and robustly handle missing images without breaking the submission, without changing the model or transforms. These changes are minimal and directly aimed at moving the score upward toward your target by ensuring inference uses the trained checkpoint and correct post-processing.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the trained checkpoint not being found/loaded, so the smallest score-relevant change is to make checkpoint discovery deterministic and exhaustive within the provided dataset tree and to support more checkpoint file types (including `.pth`/`.pt` and pickled dicts). I also add a hard safety check that confirms we actually loaded non-random weights (by counting matched keys) and only fall back to random weights if we truly cannot load anything usable. Finally, I ensure the inference uses the same three-head probability fusion you already implemented, but fix a subtle bug where `ordinal2class_prob` applies `softmax` to an already-valid probability construction (it should remain a proper distribution without extra softmax), which can flatten confidence and hurt kappa. These are minimal changes that preserve your model and inference semantics while making it much more likely you’re evaluating the intended trained model.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with inference running but producing near-constant predictions (e.g., missing/incorrect checkpoint load or too-symmetric fusion), which yields near-random agreement under quadratic weighted kappa. To move the score up toward your target with minimal semantic change, I keep your exact model and transforms, but (1) make checkpoint loading more reliable for PyTorch-Lightning/timm-style saved dicts by stripping `state_dict` prefixes like `model.`/`backbone.` and also handling `OrderedDict` cleanly, and (2) switch the three-head fusion from equal weights to a slightly regression-favored probability fusion (still using your existing helpers) because regression outputs are already calibrated to the 0–4 range and typically stabilize kappa. I also add a small safety print of prediction distribution to quickly detect degenerate outputs; it does not affect predictions. The script still runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with the checkpoint still not being loaded correctly (so predictions are effectively random/degenerate), which makes quadratic weighted kappa collapse. I keep your exact model and inference flow, but make checkpoint loading stricter and more deterministic: we (1) prioritize the intended filename, (2) correctly handle PyTorch-Lightning style `state_dict` with `model.` prefixes and `_metadata`, and (3) actually verify that a meaningful fraction of keys were loaded; otherwise we fail loudly instead of silently producing a bad submission. This is a minimal, directly score-relevant change because it doesn’t alter architecture or post-processing—only ensures inference uses the trained weights. The submission writing and image loading paths remain unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I make the checkpoint loading non-fatal so the notebook always runs end-to-end and writes a valid `submission.csv` even when the expected weights file is not present in this environment. I also fix the CUDA/CPU dtype/device mismatch by ensuring the model is moved to the selected device before inference (and confirming inputs/weights are on the same device). These changes are minimal and preserve your model/inference logic; they primarily unblock execution and, when weights are available, allow the intended predictions (improving score toward the target from 0.0). If weights truly don’t exist, the code still produce a non-empty submission instead of crashing.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 strongly suggests the model is still not actually using the intended trained weights (or is loading an incompatible checkpoint silently), which makes predictions near-random/degenerate and collapses kappa. I make checkpoint loading deterministic and “fail-fast”: prioritize loading the exact filename, broaden search to any `.pkl/.pth/.pt` under the dataset tree, and require a high key-match ratio (otherwise stop instead of writing a bad submission). I also add one minimal, score-relevant inference fix: ensure we pass the heads into `combine3output_prob` in the expected order and with the expected shapes (regression `(B,1)`, classification `(B,5)`, ordinal `(B,4)`), since any accidental swap yields nonsense classes. These changes preserve your architecture and inference semantics; they only ensure the correct checkpoint is used and correctly interpreted, which is the smallest plausible move upward toward the target.'
- What this solution (achieved 0.0) has done: 'I make two minimal, directly score-relevant stability fixes so inference runs end-to-end: (1) stop hard-failing when the external checkpoint isn’t present (it isn’t in this environment), and instead run with default-initialized weights so a valid submission is always produced; and (2) fix the device/dtype mismatch that currently happens because the model’s EfficientNet backbone stays on CPU after `timm` replaces `global_pool`. I also ensure the model is moved to the selected device *after* all module replacements and checkpoint loading, which is score-neutral when weights are missing but required for correctness. The rest of the model architecture, transforms, and post-processing (three-head probability fusion) is preserved.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the inference still running with default/random weights because no compatible checkpoint is actually found/loaded in this environment. To move the score up toward the 0.901 target with minimal change, I (1) make checkpoint discovery also search the *entire* `/kaggle/data` and `/kaggle/input` trees (where the weights might live even if not under the competition folder), and (2) make loading more robust to common formats (including Lightning `state_dict` with nested keys/prefixes). I also add a fail-fast guard: if no usable checkpoint is loaded, the script stop instead of silently writing a low-scoring submission (this directly prevents the 0.0 outcome). Core model, transforms, and prediction fusion are unchanged.'

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
import os
import glob
import warnings
import pickle
from collections import OrderedDict

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)



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
from torchvision.ops.misc import FrozenBatchNorm2d
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
    out = out.float()
    if out.min().item() < -1e-6 or out.max().item() > 1 + 1e-6:
        out = torch.sigmoid(out)

    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()

    pred_prob = pred_prob.clamp_min(0.0)
    pred_prob = pred_prob / (pred_prob.sum(dim=1, keepdim=True).clamp_min(1e-12))
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


def combine3output_prob(r_out, c_out, o_out):
    c_prob = F.softmax(c_out, dim=1)
    r_prob = regress2class_prob(r_out.squeeze(1) if r_out.ndim == 2 else r_out)
    o_prob = ordinal2class_prob(o_out)

    w_r, w_c, w_o = 0.50, 0.25, 0.25
    prob = (w_c * c_prob + w_r * r_prob + w_o * o_prob) / (w_r + w_c + w_o)
    pred = int(torch.argmax(prob, dim=1).item())
    return pred




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
BASE_CANDIDATES = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data/input/aptos2019-blindness-detection",
    "/kaggle/data",
    "/kaggle/input",
]
BASE = next(
    (p for p in BASE_CANDIDATES if os.path.exists(p) and os.path.isdir(p)), None
)
if BASE is None:
    raise FileNotFoundError(
        f"Could not find dataset base folder. Tried: {BASE_CANDIDATES}"
    )

if os.path.basename(BASE) in ("data", "input"):
    hits = []
    for root in [BASE]:
        hits += glob.glob(
            os.path.join(root, "**", "aptos2019-blindness-detection"), recursive=True
        )
    hits = [h for h in hits if os.path.isdir(h)]
    if len(hits) > 0:
        BASE = sorted(hits, key=lambda x: (len(x.split(os.sep)), x))[0]

TEST_CSV = os.path.join(BASE, "test.csv")
if not os.path.exists(TEST_CSV):
    raise FileNotFoundError(f"test.csv not found at expected path: {TEST_CSV}")

TEST_IMG_DIR_CANDIDATES = [
    os.path.join(BASE, "test_images"),
    os.path.join(BASE, "aptos2019-blindness-detection", "test_images"),
]
TEST_IMG_DIR = next((p for p in TEST_IMG_DIR_CANDIDATES if os.path.exists(p)), None)
if TEST_IMG_DIR is None:
    img_hits = glob.glob(os.path.join(BASE, "**", "test_images"), recursive=True)
    img_hits = [h for h in img_hits if os.path.isdir(h)]
    TEST_IMG_DIR = img_hits[0] if len(img_hits) else None
if TEST_IMG_DIR is None:
    raise FileNotFoundError(
        f"Could not find test_images directory. Tried: {TEST_IMG_DIR_CANDIDATES}"
    )

test_df = pd.read_csv(TEST_CSV)
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

net1 = ThreeStage_Model()

ckpt_name = "B4_3stage_19epoch_320.pkl"

ckpt_patterns = [
    ckpt_name,
    ckpt_name.replace(".pkl", ".pth"),
    ckpt_name.replace(".pkl", ".pt"),
    "B4_3stage_19epoch_320*.pkl",
    "B4_3stage_19epoch_320*.pth",
    "B4_3stage_19epoch_320*.pt",
    "*3stage*B4*.pkl",
    "*3stage*B4*.pth",
    "*3stage*B4*.pt",
    "*B4*3stage*.pkl",
    "*B4*3stage*.pth",
    "*B4*3stage*.pt",
]

search_roots = [
    BASE,
    os.path.dirname(BASE),
    "../input",
    "/kaggle/input",
    "/kaggle/data",
]


def _unwrap_state_dict(obj):
    if isinstance(obj, nn.Module):
        return obj.state_dict()

    if isinstance(obj, dict) and "_metadata" in obj:
        obj = {k: v for k, v in obj.items() if k != "_metadata"}

    if isinstance(obj, (OrderedDict, dict)) and len(obj) > 0:
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return dict(obj)

    if not isinstance(obj, dict):
        return obj

    for k in [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "network",
        "weights",
        "ema",
        "student",
        "teacher",
    ]:
        if k in obj:
            inner = obj[k]
            if isinstance(inner, (OrderedDict, dict)):
                return _unwrap_state_dict(inner)
    return obj


def _looks_like_state_dict(d):
    if not isinstance(d, dict) or len(d) == 0:
        return False
    keys = list(d.keys())
    if not isinstance(keys[0], str):
        return False
    return ("." in keys[0]) or ("weight" in keys[0]) or ("bias" in keys[0])


def _try_strip_prefixes_to_match(model: nn.Module, sd: dict):
    if not isinstance(sd, dict) or len(sd) == 0:
        return sd

    model_keys = set(model.state_dict().keys())
    if len(model_keys) == 0:
        return sd

    def score(d):
        if not isinstance(d, dict) or len(d) == 0:
            return 0
        return sum(1 for k in d.keys() if k in model_keys)

    best_sd = sd
    best_score = score(sd)

    prefixes = [
        "module.",
        "state_dict.",
        "model.",
        "net.",
        "network.",
        "backbone.",
        "backbone.model.",
    ]
    for _ in range(10):
        improved = False
        for pref in prefixes:
            if any(isinstance(k, str) and k.startswith(pref) for k in best_sd.keys()):
                stripped = {
                    (
                        k[len(pref) :]
                        if isinstance(k, str) and k.startswith(pref)
                        else k
                    ): v
                    for k, v in best_sd.items()
                }
                s = score(stripped)
                if s > best_score:
                    best_sd, best_score = stripped, s
                    improved = True
        if not improved:
            break
    return best_sd


def _match_count(model: nn.Module, sd: dict) -> int:
    if not isinstance(sd, dict):
        return 0
    mk = set(model.state_dict().keys())
    return sum(1 for k in sd.keys() if k in mk)


def _rank_ckpt(p: str) -> tuple:
    base = os.path.basename(p)
    exact = int(base == ckpt_name)
    contains = int("B4_3stage_19epoch_320" in base)
    three_stage = int(("3stage" in base.lower()) or ("three" in base.lower()))
    b4 = int("b4" in base.lower())
    return (-exact, -contains, -three_stage, -b4, len(p), p)


candidate_weights = []
for root in search_roots:
    if not (os.path.exists(root) and os.path.isdir(root)):
        continue
    for pat in ckpt_patterns:
        candidate_weights += glob.glob(os.path.join(root, "**", pat), recursive=True)

if len(candidate_weights) == 0:
    broad_patterns = ["*.pth", "*.pt", "*.pkl"]
    for root in ["/kaggle/input", "/kaggle/data", "../input"]:
        if not (os.path.exists(root) and os.path.isdir(root)):
            continue
        for pat in broad_patterns:
            candidate_weights += glob.glob(
                os.path.join(root, "**", pat), recursive=True
            )
    candidate_weights = [
        p
        for p in candidate_weights
        if (
            "3stage" in os.path.basename(p).lower()
            or "three" in os.path.basename(p).lower()
        )
        and (
            "b4" in os.path.basename(p).lower()
            or "efficientnet" in os.path.basename(p).lower()
        )
    ]

seen = set()
candidate_weights = [p for p in candidate_weights if not (p in seen or seen.add(p))]
candidate_weights = [
    p for p in candidate_weights if os.path.exists(p) and os.path.isfile(p)
]
candidate_weights = sorted(candidate_weights, key=_rank_ckpt)

weights_path = None
best_match = -1
best_sd = None

for p in candidate_weights[:400]:
    try:
        obj = torch.load(p, map_location="cpu")
    except Exception:
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
        except Exception:
            continue

    sd = _unwrap_state_dict(obj)
    if not _looks_like_state_dict(sd):
        continue
    sd = _try_strip_prefixes_to_match(net1, sd)
    mc = _match_count(net1, sd)
    if mc > best_match:
        best_match = mc
        best_sd = sd
        weights_path = p
    if best_match >= int(0.995 * len(net1.state_dict())):
        break

loaded_ok = False
if weights_path is not None and best_sd is not None and best_match > 0:
    missing, unexpected = net1.load_state_dict(best_sd, strict=False)
    print("Loaded weights from:", weights_path)
    print(
        f"Matched keys: {best_match}/{len(net1.state_dict())} | missing: {len(missing)} | unexpected: {len(unexpected)}"
    )
    if len(unexpected) > 0:
        warnings.warn(f"Unexpected keys in state_dict (ignored): {unexpected[:10]}")
    if len(missing) > 0:
        warnings.warn(f"Missing keys in state_dict (left default init): {missing[:10]}")
    loaded_ok = best_match >= int(0.80 * len(net1.state_dict()))

if not loaded_ok:
    raise FileNotFoundError(
        "No usable checkpoint was loaded (matched keys too low or none found). "
        "To improve Kaggle score above 0.0, you must provide the trained weights file in the environment."
    )

net1 = net1.to(device)
net1.eval()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3112597392.py in <cell line: 0>()
    279 
    280 if not loaded_ok:
--> 281     raise FileNotFoundError(
    282         "No usable checkpoint was loaded (matched keys too low or none found). "
    283         "To improve Kaggle score above 0.0, you must provide the trained weights file in the environment."

FileNotFoundError: No usable checkpoint was loaded (matched keys too low or none found). To improve Kaggle score above 0.0, you must provide the trained weights file in the environment.

## === cell 6
submission = []
preds_debug = []

with torch.inference_mode():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print(f"{i}/{len(test_ids)}")

        image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
        if not os.path.exists(image_name):
            submission.append([idx, 0])
            preds_debug.append(0)
            continue

        img1 = Image.open(image_name).convert("RGB")
        img1 = transform1(img1).unsqueeze(0).to(device)

        c_out, r_out, o_out = net1(img1)

        if r_out.ndim == 1:
            r_out = r_out.unsqueeze(1)
        if c_out.ndim != 2 or c_out.size(1) != 5:
            raise RuntimeError(f"Unexpected c_out shape: {tuple(c_out.shape)}")
        if o_out.ndim != 2 or o_out.size(1) != 4:
            raise RuntimeError(f"Unexpected o_out shape: {tuple(o_out.shape)}")

        P = int(combine3output_prob(r_out=r_out, c_out=c_out, o_out=o_out))

        if P < 0:
            P = 0
        elif P > 4:
            P = 4

        submission.append([idx, P])
        preds_debug.append(P)

submission = np.array(submission, dtype=object)

vc = pd.Series(preds_debug).value_counts().sort_index()
print("Prediction distribution:", vc.to_dict())



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3360310912.py in <cell line: 0>()
     16         img1 = transform1(img1).unsqueeze(0).to(device)
     17 
---> 18         c_out, r_out, o_out = net1(img1)
     19 
     20         if r_out.ndim == 1:

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/1773810286.py in forward(self, x, final)
     77 
     78     def forward(self, x, final=False):
---> 79         x = self.backbone(x)
     80 
     81         c_out = self.classifier(x)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/timm/models/efficientnet.py in forward(self, x)
    337     def forward(self, x: torch.Tensor) -> torch.Tensor:
    338         """Forward pass."""
--> 339         x = self.forward_features(x)
    340         x = self.forward_head(x)
    341         return x

/usr/local/lib/python3.11/dist-packages/timm/models/efficientnet.py in forward_features(self, x)
    310     def forward_features(self, x: torch.Tensor) -> torch.Tensor:
    311         """Forward pass through feature extraction layers."""
--> 312         x = self.conv_stem(x)
    313         x = self.bn1(x)
    314         if self.grad_checkpointing and not torch.jit.is_scripting():

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/timm/layers/conv2d_same.py in forward(self, x)
     51 
     52     def forward(self, x):
---> 53         return conv2d_same(
     54             x, self.weight, self.bias,
     55             self.stride, self.padding, self.dilation, self.groups,

/usr/local/lib/python3.11/dist-packages/timm/layers/conv2d_same.py in conv2d_same(x, weight, bias, stride, padding, dilation, groups)
     26 ):
     27     x = pad_same(x, weight.shape[-2:], stride, dilation)
---> 28     return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
     29 
     30 

RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same

## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df = test_df[["id_code"]].merge(df, on="id_code", how="left")
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", df.shape)
print(df.head())
