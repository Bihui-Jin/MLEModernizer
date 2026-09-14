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

0.6144767469649086

# 6. Current score

0.74059

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the model construction to be compatible with the installed `timm==1.0.19` (the EfficientNet attribute names changed, causing the `act1` error) while keeping the exact same architecture/forward logic. I also make device selection safe by falling back to CPU when CUDA isn’t available (your environment has no NVIDIA driver), which fixes the runtime crash during `.to(device)`. Finally, I ensure the inference loop actually runs end-to-end and always writes a non-empty `submission.csv` with the required `id_code,diagnosis` columns. These changes are execution/stability fixes and should not alter intended modeling semantics beyond negligible numeric differences.'
- What this solution (achieved 0.0) has done: 'I first fix the hard failure caused by trying to load a non-existent weights file by adding a safe fallback that uses the model’s randomly initialized weights when the checkpoint is missing, so the notebook always produces a submission.csv. Next, I fix the device/dtype mismatch that caused the CUDA tensor vs CPU weights runtime error by ensuring both model and inputs are on the same device and by forcing CPU execution in Kaggle environments without a usable GPU driver. Finally, I keep the existing model/forward logic and post-processing unchanged, and ensure the submission is aligned to `sample_submission.csv` and written with the required `id_code,diagnosis` columns.'
- What this solution (achieved 0.01112) has done: 'Your 0.0 score is coming from using randomly initialized weights (the `../input/weights/1.pth` checkpoint doesn’t exist), so predictions are effectively random. To move the score upward toward the 0.614 target while preserving your model and inference logic, I switch to using the built-in pretrained EfficientNet weights (`pretrained=True`) as a safe, minimal fallback when the external checkpoint is missing. I also fix a small but important forward bug where `block0` is fed `x1` instead of the activated stem output, which keeps the intended EfficientNet feature flow and improves inference quality without changing your architecture. Everything else (thresholding, `combine3output`, transforms, submission formatting) stays the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.04135) has done: 'Your current score is far below the target, so the smallest safe way to move it upward is to improve the inference-time mapping from your model’s three heads to the final 0–4 label, without changing the model architecture or any training logic. Right now `combine3output` mistakenly turns the ordinal head into a 5-class argmax over 4 logits, which makes that head essentially uninformative and hurts kappa. I fix this by converting the ordinal head’s 4 sigmoid outputs into a proper 5-class distribution and then taking its expected class (or argmax), while keeping the same averaging-with-regression-and-classifier idea. Everything else (pretrained fallback, transforms, file paths, and submission formatting) stays the same and the script still write a valid `submission.csv`.'
- What this solution (achieved -0.13501) has done: 'Your current score (0.04135) is far below the target (0.61448), so we should move it upward with the smallest changes that preserve your model and inference approach. The biggest remaining issue is that your `ordinal2class_prob()` incorrectly applies a `softmax` to values that are already meant to be a probability distribution, which distorts the ordinal head and harms the final combined prediction. I remove that `softmax` and add a tiny numeric normalization (clamp + renorm) so the expected-class computation is well-behaved, while keeping the exact same three-head averaging logic. Everything else (model, transforms, checkpoint fallback, submission formatting) stays the same.'
- What this solution (achieved 0.12099) has done: 'Your score is far below the target (higher is better), so the smallest safe move upward is to fix a key inference bug: the three-head outputs are unpacked in the wrong order in the test loop, so regression/classification/ordinal heads are mixed up and predictions become effectively broken. I only change the inference unpacking to match the model’s `forward()` return order, and I add a tiny safety clamp in `combine3output` to keep outputs within valid label range (no change to core model/logic). Everything else (model definition, transforms, thresholds, pretrained fallback, and submission formatting) stays the same, and it still write a valid `submission.csv`.'
- What this solution (achieved -0.01698) has done: 'Your current score (0.12099) is far below the target (0.61448), so we should cautiously improve inference quality without changing the model architecture or training approach. The biggest low-risk gain is to use the same preprocessing you already defined for fundus images (trim + 4:3 crop) during test inference; right now you define it as `transform1` but never use it, which creates a train/test-style preprocessing mismatch and hurts kappa. I switch the inference pipeline to apply `transform1` before the EfficientNet normalization/resize (`transform2`), keeping your model, heads, and `combine3output` logic unchanged. I also add a tiny safety fallback to handle rare image read errors by predicting 0, so the submission is always complete and valid.'
- What this solution (achieved 0.00717) has done: 'Your score is far below the target (0.614), so we should make the smallest inference-only fixes that legitimately improve kappa without changing your model architecture or any training logic. The biggest issue is a head-order mismatch: `backboneNet_efficient.forward()` returns `(regression, classification, ordinal)`, but your loop unpacks it as `(regression, classification, ordinal)` while `combine3output()` expects `(regression, classification, ordinal)`—however the actual variable names and usage currently mix this up because `combine3output()` treats `c_out` as 5-class logits and `o_out` as 4-class logits, while your model’s `cls_cls` and `ord_cls` align, but the *unpacking order* must be made explicit and consistent to avoid silent mistakes. Additionally, `combine3output()` currently uses `torch.max` directly on raw logits; applying `softmax` before argmax won’t change argmax, but it make the head’s scale consistent if later extended—so we keep argmax but ensure tensors are the right shapes (batch=1) and remove `.data` usage to avoid edge-case behavior. Finally, we ensure the test IDs are read as strings (some `id_code`s can lose leading zeros if treated as numeric), preventing filename mismatches that silently trigger the “diagnosis=0 fallback” and tank kappa.'
- What this solution (achieved 0.00852) has done: 'Your current score (0.00717) is far below the target (0.61448), so we should make the smallest inference-only changes that can legitimately improve quadratic weighted kappa without changing your model architecture or any training. The biggest low-risk win is to fix the preprocessing: you currently normalize twice by converting `transform1(img)` (already normalized tensor) back to a PIL image and then applying `transform2` (another normalize), which destroys input scale and tanks predictions. I change the inference pipeline to apply only your fundus-specific crop/trim (no tensor/normalize) and then apply the EfficientNet ImageNet `transform2` once, keeping everything else (model, heads, `combine3output`, thresholds, submission formatting) unchanged. This should move the score upward toward the target while staying within the same core logic and runtime constraints.'
- What this solution (achieved -0.14844) has done: 'Your current score (0.00852) is far below the target (0.61448), so we should make a small inference-only fix that improves predictive signal without changing the model architecture or training. Right now `combine3output()` mixes a proper class prediction (C) with a regression-to-class (R) and an *expected value* from the ordinal head (O), and then rounds—this tends to over-smooth and can hurt quadratic weighted kappa. I keep the same three-head combination idea but change only the ordinal contribution to produce a discrete class (argmax of the ordinal-derived 5-class distribution) instead of an expected value, which typically improves agreement on an ordinal metric while remaining within your existing logic. Everything else (paths, preprocessing, pretrained fallback, submission alignment) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved -0.10472) has done: 'Your current score (-0.148) is far below the target (0.614, higher-is-better), so we should make a small inference-only change that adds real predictive signal without changing your model architecture or training. The biggest issue is a scale mismatch: your regression head outputs raw logits (unbounded), but `regress2class()` compares them to thresholds calibrated for a 0–4.5 sigmoid-scaled regression output; this makes the regression vote essentially garbage and can drag kappa negative. I fix this by applying the same `sigmoid()*4.5` scaling to `r_out` inside `combine3output()` (keeping everything else identical), and I also ensure `combine3output()` can safely handle batch>1 tensors (though you use batch=1). This should move the score upward toward the target with minimal, metric-relevant impact.'
- What this solution (achieved 0.02736) has done: 'Your current score (-0.10472) is far below the target (0.61448, higher-is-better), so we should move it upward with the smallest inference-only change that adds real signal while preserving your model and overall approach. Right now you use the EfficientNet classifier head (`cls_cls`) and ordinal head (`ord_cls`) as raw ImageNet-pretrained logits, which are essentially misaligned with DR labels and can actively hurt kappa. The minimal fix is to make `combine3output()` rely primarily on the (scaled) regression head—which is the only head that has any plausible ordinal structure even under ImageNet pretraining—while still keeping the same 3-head combine function and pipeline intact. This should substantially reduce “random” label noise and move the score closer to the target without changing architecture, training, or preprocessing.'
- What this solution (achieved 0.0259) has done: 'Your score (0.02736) is far below the target (0.61448), so we need a legitimate accuracy lift with minimal disruption. The biggest blocker is that you’re using ImageNet-pretrained EfficientNet with randomly initialized DR heads, so the classifier/ordinal heads are noise and the regression head is only weakly correlated; the smallest fix that preserves your exact model/3-head combine logic is to calibrate only the final label mapping. I add an (optional) quick calibration step on the training set to learn 4 optimal thresholds that convert the regression output into 0–4 labels maximizing quadratic weighted kappa, and then use those calibrated thresholds at test time (same model, same inference). If calibration can’t run for any reason, it safely falls back to your current fixed thresholds and still writes a valid `submission.csv`.'
- What this solution (achieved 0.03093) has done: 'Your current score (0.0259) is far below the target (0.6145), so we should increase it with minimal, inference-only changes that preserve your model and overall approach. The biggest low-risk issue is that your calibration computes thresholds using only the regression head, but your final prediction uses a weighted mix of regression/classification/ordinal; this makes the learned thresholds misaligned with what you actually submit. I change calibration to optimize thresholds directly on the combined prediction *before* thresholding (i.e., treat the combined output as a continuous 0–4.5 score and learn 4 cutpoints for it), while keeping `backboneNet_efficient`, transforms, and `combine3output` semantics intact. I also remove duplicated calibration inference code and reuse the existing helper to keep runtime within limits.'
- What this solution (achieved 0.74059) has done: 'Your current score (0.03093) is far below the target (0.61448), so we should add real predictive signal with the smallest possible change while preserving your model and inference semantics. The main issue is that you never load DR-trained weights, so your three heads are essentially random; the most direct fix is to add a minimal training step that fine-tunes only your existing heads (rg/cls/ord) on `train.csv` for a few epochs, keeping the same backbone, heads, and combine logic. To keep within the time limit, we train on a small, stratified subset with light augmentation and then keep your existing calibration + thresholding pipeline unchanged. This should move QWK materially upward toward the target without changing architecture, loss definition, or output format.'

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


def _select_device():
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0], device="cuda")
            return "cuda:0"
        except Exception:
            return "cpu"
    return "cpu"


device = _select_device()
print("Using device:", device)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.set_num_threads(min(4, os.cpu_count() or 4))
warnings.filterwarnings("once")



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
import torch
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

        if hasattr(net, "act1"):
            self.act1 = net.act1
        elif hasattr(net, "activation"):
            self.act1 = net.activation
        elif hasattr(net, "act"):
            self.act1 = net.act
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
        elif hasattr(net, "activation"):
            self.act2 = net.activation
        elif hasattr(net, "act"):
            self.act2 = net.act
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


def regress2class(out, thr=None):
    thr = threshold if thr is None else thr
    prediction = 0
    for i in range(4):
        prediction += (out >= thr[i]).squeeze().cpu().item()
    return int(prediction)


def ordinal2class_prob(out):
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


def combine3output(r_out, c_out, o_out, thr=None):
    r_scaled = torch.sigmoid(r_out) * 4.5  # shape [B,1]
    R = regress2class(r_scaled[0], thr=thr)

    _, C = torch.max(c_out, 1)
    C = int(C[0].item())

    o_sig = torch.sigmoid(o_out)
    o_prob = ordinal2class_prob(o_sig)
    _, O = torch.max(o_prob, 1)
    O = int(O[0].item())

    P = (2.0 * R + 0.5 * C + 0.5 * O) / 3.0
    P = int(round(P))
    P = max(0, min(4, P))
    return P


def combine3score_continuous(r_out, c_out, o_out):
    r_scaled = torch.sigmoid(r_out) * 4.5  # [B,1]
    r_val = float(r_scaled.view(-1)[0].detach().cpu().item())

    _, C = torch.max(c_out, 1)
    c_val = float(C[0].detach().cpu().item())

    o_sig = torch.sigmoid(o_out)
    o_prob = ordinal2class_prob(o_sig)
    _, O = torch.max(o_prob, 1)
    o_val = float(O[0].detach().cpu().item())

    p_val = (2.0 * r_val + 0.5 * c_val + 0.5 * o_val) / 3.0
    return float(np.clip(p_val, 0.0, 4.5))




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
test_df = pd.read_csv(
    "../input/aptos2019-blindness-detection/test.csv", dtype={"id_code": str}
)
test_ids = test_df["id_code"].tolist()

train_df = pd.read_csv(
    "../input/aptos2019-blindness-detection/train.csv", dtype={"id_code": str}
)

fundus_pil_pre = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
    ]
)

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


class AptosTrainDataset(Dataset):
    def __init__(self, df, img_dir, tfm):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        idx = str(self.df.loc[i, "id_code"])
        y = int(self.df.loc[i, "diagnosis"])
        p = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(p).convert("RGB")
        img = fundus_pil_pre(img)
        x = self.tfm(img)
        return x, y


net2 = backboneNet_efficient()

ckpt_path = "../input/weights/1.pth"
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    net2.load_state_dict(state, strict=True)
    print("Loaded weights:", ckpt_path)
else:
    warnings.warn(
        f"Checkpoint not found at {ckpt_path}. Using timm pretrained EfficientNet weights as fallback."
    )

net2 = net2.to(device)

train_img_dir = "../input/aptos2019-blindness-detection/train_images"
train_tfm = transforms.Compose(
    [
        photometric_distort(),  # mild augmentation, consistent with fundus pipeline
        transforms.Resize((280, 280)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

sub_n = min(1400, len(train_df))
train_sub = (
    train_df.groupby("diagnosis", group_keys=False)
    .apply(
        lambda x: x.sample(
            n=max(1, int(round(len(x) * (sub_n / len(train_df))))), random_state=0
        )
    )
    .reset_index(drop=True)
)
if len(train_sub) > sub_n:
    train_sub = train_sub.sample(n=sub_n, random_state=0).reset_index(drop=True)

ds = AptosTrainDataset(train_sub, train_img_dir, train_tfm)
dl = DataLoader(
    ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=(device != "cpu")
)

ce = nn.CrossEntropyLoss()
bce = nn.BCEWithLogitsLoss()
l1 = nn.SmoothL1Loss(beta=1.0)

params = (
    list(net2.rg_cls.parameters())
    + list(net2.cls_cls.parameters())
    + list(net2.ord_cls.parameters())
)
opt = optim.Adam(params, lr=2e-3, weight_decay=1e-4)


def _labels_to_ordinal_targets(y, n_classes=5):
    y = y.view(-1, 1)
    ks = torch.arange(n_classes - 1, device=y.device).view(1, -1)
    return (y > ks).float()


net2.train()
t0 = time.time()
epochs = 2  # minimal change; enough to move off-random while keeping runtime bounded
for ep in range(epochs):
    running = 0.0
    n_seen = 0
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        r_out, c_out, o_out = net2(xb)

        y_reg = yb.float().view(-1, 1)
        r_scaled = torch.sigmoid(r_out) * 4.5
        loss_r = l1(r_scaled, y_reg)

        loss_c = ce(c_out, yb)

        y_ord = _labels_to_ordinal_targets(yb, n_classes=5)
        loss_o = bce(o_out, y_ord)

        loss = loss_r + loss_c + loss_o
        loss.backward()
        opt.step()

        running += float(loss.detach().cpu().item()) * xb.size(0)
        n_seen += xb.size(0)

        if time.time() - t0 > 260:
            break

    print(f"Epoch {ep+1}/{epochs} loss:", running / max(1, n_seen))
    if time.time() - t0 > 260:
        print(
            "Stopping training early due to time budget (does not change inference logic)."
        )
        break

net2.eval()




## === cell 6
def _predict_combined_score_for_ids(ids, img_dir, max_n=600):
    xs = []
    n = min(len(ids), max_n)
    for i in range(n):
        idx = ids[i]
        image_name = os.path.join(img_dir, f"{idx}.png")
        try:
            img = Image.open(image_name).convert("RGB")
        except Exception:
            continue
        img_pil = fundus_pil_pre(img)
        x = transform2(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            r_out, c_out, o_out = net2(x)
            p_val = combine3score_continuous(r_out, c_out, o_out)
        xs.append(float(p_val))
    return np.array(xs, dtype=np.float32)


def _optimize_thresholds_4(x, y, init_thr):
    thr = np.array(init_thr, dtype=np.float32)
    thr.sort()

    def apply_thr(xv, t):
        return np.digitize(xv, t, right=False).astype(np.int64)

    best_k = cohen_kappa_score(y, apply_thr(x, thr), weights="quadratic")

    for _ in range(3):
        improved = False
        for j in range(4):
            lo = (thr[j - 1] + 1e-3) if j > 0 else 0.0
            hi = (thr[j + 1] - 1e-3) if j < 3 else 4.5
            center = float(thr[j])
            cand = np.linspace(center - 0.6, center + 0.6, 25, dtype=np.float32)
            cand = cand[(cand > lo) & (cand < hi)]
            if cand.size == 0:
                continue
            local_best_k = best_k
            local_best_t = center
            for tval in cand:
                t_try = thr.copy()
                t_try[j] = float(tval)
                t_try.sort()
                k = cohen_kappa_score(y, apply_thr(x, t_try), weights="quadratic")
                if k > local_best_k + 1e-6:
                    local_best_k = k
                    local_best_t = float(tval)
            if local_best_k > best_k + 1e-6:
                thr[j] = local_best_t
                thr.sort()
                best_k = local_best_k
                improved = True
        if not improved:
            break
    return thr.tolist(), float(best_k)


calibrated_thresholds = None
try:
    calib_n = 700
    calib_df = train_df.sample(
        n=min(calib_n, len(train_df)), random_state=0
    ).reset_index(drop=True)
    calib_ids = calib_df["id_code"].tolist()
    calib_y_all = calib_df["diagnosis"].astype(int).values

    train_img_dir = "../input/aptos2019-blindness-detection/train_images"

    calib_x = []
    calib_y = []
    n = min(len(calib_ids), calib_n)
    for i in range(n):
        idx = calib_ids[i]
        yv = int(calib_y_all[i])
        image_name = os.path.join(train_img_dir, f"{idx}.png")
        try:
            img = Image.open(image_name).convert("RGB")
        except Exception:
            continue
        img_pil = fundus_pil_pre(img)
        x = transform2(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            r_out, c_out, o_out = net2(x)
            p_val = combine3score_continuous(r_out, c_out, o_out)
        calib_x.append(float(p_val))
        calib_y.append(yv)

    calib_x = np.array(calib_x, dtype=np.float32)
    calib_y = np.array(calib_y, dtype=np.int64)

    if len(calib_x) >= 200 and len(np.unique(calib_y)) >= 3:
        calibrated_thresholds, calib_kappa = _optimize_thresholds_4(
            calib_x, calib_y, threshold
        )
        print("Calibrated thresholds (on combined score):", calibrated_thresholds)
        print("Calibration QWK (on sampled train):", calib_kappa)
    else:
        warnings.warn(
            "Calibration skipped (insufficient readable samples or label variety). Using default thresholds."
        )
except Exception as e:
    warnings.warn(f"Calibration failed due to {e}. Using default thresholds.")



## === cell 7
submission = []
test_img_dir = "../input/aptos2019-blindness-detection/test_images"

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 25 == 0:
            print("Predicting", i, "/", len(test_ids))
        image_name = os.path.join(test_img_dir, f"{idx}.png")

        try:
            img = Image.open(image_name).convert("RGB")
        except Exception as e:
            warnings.warn(
                f"Failed to read {image_name} due to {e}. Using diagnosis=0 fallback."
            )
            submission.append([idx, 0])
            continue

        img_pil = fundus_pil_pre(img)
        img2 = transform2(img_pil).unsqueeze(0).to(device)

        r_out, c_out, o_out = net2(img2)
        pred2 = combine3output(r_out, c_out, o_out, thr=calibrated_thresholds)

        submission.append([idx, pred2])

submission = np.array(submission, dtype=object)



## === cell 8
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

sample_sub = pd.read_csv(
    "../input/aptos2019-blindness-detection/sample_submission.csv",
    dtype={"id_code": str},
)
df = sample_sub[["id_code"]].merge(df, on="id_code", how="left")
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("Saved to:", os.path.abspath("submission.csv"))
