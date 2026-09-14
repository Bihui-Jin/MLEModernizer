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

0.916054033611703

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the failing `pip install` cell and make the script use the already-installed `timm`, (2) make device selection robust so it runs on CPU when no GPU is available, and (3) fix the missing weight-file path by searching common Kaggle input locations and falling back safely if the weights aren’t present. I also fix a couple of runtime issues in transforms/utilities (`is` vs `==`, `trim()` returning `None`) that can silently break inference. Finally, I ensure we always create a non-empty `submission.csv` with the exact required columns and order aligned to `test.csv`.'
- What this solution (achieved 0.00014) has done: 'Your 0.0 score is almost certainly coming from using randomly initialized weights because the expected `.pkl` weight file is not actually present in your Kaggle inputs, so the model outputs near-constant/garbage predictions. I keep your exact model and inference logic, but make the weight loading robust to common checkpoint formats (`state_dict` wrappers, `module.` prefixes) and point the search to the actual competition dataset directory you already have. If the intended weight still isn’t found, I deterministically fall back to using `timm` pretrained backbone weights (without changing architecture) so predictions become meaningful and the kappa should move upward toward your target band. The submission writing/alignment be kept identical, just made slightly safer against missing/corrupt images.'
- What this solution (achieved -0.00991) has done: 'Your current score is extremely low because inference is effectively using untrained/random heads (and possibly a missing checkpoint), plus the prediction post-processing is not well aligned to the kappa metric (fixed thresholds without any calibration). I keep your exact model and inference flow, but (1) make checkpoint discovery also look in `/kaggle/data/...` (where your dataset actually is) and support common key-prefix mismatches, and (2) add a tiny in-notebook threshold calibration step using a train/val split to choose the 4 regression thresholds that maximize quadratic weighted kappa on a held-out fold. This does not change the model architecture, loss, or training loop (there is no training), it only improves the mapping from your regressor output to the 0–4 labels, which is directly what the metric evaluates. The rest of the submission writing/alignment stays the same, ensuring a valid `submission.csv`.'
- What this solution (achieved -0.05613) has done: 'Your score is far below the target, so we should improve the prediction quality without changing your model architecture or adding training. The biggest issue is that you only use the regression head (`r_out`) and ignore the classifier/ordinal heads that are already produced by the same forward pass; combining them should meaningfully improve QWK with minimal logic change. I keep your existing threshold calibration idea, but calibrate thresholds on the *combined* per-image score (average of regression class, classifier argmax, ordinal argmax) instead of just regression. Finally, I make inference slightly more robust by running in `eval()` with `inference_mode()` and ensuring the same combined-score logic is used for both calibration and test predictions.'
- What this solution (achieved -0.07781) has done: 'Your current score is far below the target, so the safest way to move it upward (without changing model/training core logic) is to fix two inference mismatches that currently make predictions effectively inconsistent: (1) you apply `sigmoid` twice to the ordinal head (once in the model forward already, again in `_predict_combined_score`/test loop), and (2) your backbone forward has a bug (`block0` uses `x1` instead of the activated tensor), which makes any loaded weights behave incorrectly. I minimally correct those two issues while keeping the same architecture/heads and the same “combined score + threshold calibration” approach. I also vectorize the calibration/test inference in small batches to stay under the 600s timeout without changing semantics. The output submission format and paths stay identical and still produce `submission.csv`.'
- What this solution (achieved 0.11434) has done: 'I fix the root runtime error by making `_get_timm_act` robust to the current `timm==1.0.19` EfficientNet API (where `act1/act2` may not exist) and falling back to safe identity activations when needed, so `backboneNet_efficient()` instantiates. Then I fix the inference mismatch bug where `r_raw` and `o_raw` are incorrectly passed through `sigmoid` again even though the model already outputs calibrated `r_out` and `o_out`; this keeps the model logic the same but restores correct semantics and should substantially improve QWK. Finally, I ensure the pipeline always defines `net1`, runs calibration/inference end-to-end, and writes a valid `submission.csv` with the required columns aligned to `test.csv`.'
- What this solution (achieved 0.19871) has done: 'Your current score (0.11434) is far below the target (0.916...), so we should increase performance with the smallest changes that fix clear inference mistakes without altering the model/training approach. The biggest issue is in `_infer_combined_scores_batch`: it incorrectly applies `sigmoid` again to `r_raw` and `o_raw` even though `backboneNet_efficient.forward()` already returns `r_out` and `o_out` in the right ranges; this double-squashing destroys calibration and hurts QWK. I change `_infer_combined_scores_batch` to use the model outputs directly (no extra sigmoid), keeping the same combined-score + threshold calibration pipeline. I also make the regressor-to-class conversion inside batching consistent with your existing `regress2class` logic by using the calibrated thresholds rather than hardcoding the old ones during score construction.'
- What this solution (achieved 0.14477) has done: 'Your score gap to the target is large, so we should improve QWK with the smallest changes that fix clear metric-alignment issues without changing your model or adding training. The main issue is that your “combined score” is mixing a continuous regressor value with discrete class labels, which hurts threshold calibration; we instead convert each head into a comparable 0–4 scale and then average (still the same “combine 3 heads then threshold” core logic). Next, we calibrate thresholds on an out-of-fold prediction set using 3-fold CV over a small fixed subset (still no training), which makes threshold selection more stable and typically improves public QWK versus a single split. Finally, we ensure the global `threshold` used by `regress2class()` is synchronized to the calibrated thresholds so the regressor head contributes consistently during inference.'
- What this solution (achieved 0.14477) has done: 'Your current score (0.14477) is far below the target (0.916...), so we should increase QWK with minimal, metric-aligned fixes that don’t change your architecture or add training. The biggest correctness issue is that `backboneNet_efficient.forward()` returns raw logits for regressor/ordinal, but your batch inference assumes it receives those logits and then applies `sigmoid`; meanwhile other parts of your code define “final outputs” differently—this mismatch harms calibration and thresholds. I make the model forward return the same semantic outputs as your `ThreeStage_Model` (regressor in 0..4.5, ordinal in 0..1, classifier logits unchanged), and then simplify `_infer_combined_scores_batch` to use these outputs directly (no extra sigmoid), keeping the exact “combine 3 heads then threshold” logic. Finally, I make OOF threshold calibration consistent by computing the combined continuous score on the same scale for train/val and test.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we should improve prediction quality without changing the model architecture or adding training. The biggest likely issue is that you are using `transform1` (384x288 + custom normalization) with a model that is almost certainly trained with a different preprocessing; we keep your core “combined 3 heads + threshold calibration” logic but calibrate and infer using a consistent transform that better matches EfficientNet expectations. To stay minimal, we (1) add a small transform selection that uses the same transform for both OOF calibration and test inference, (2) slightly increase the OOF subset size (still safe under 600s) to stabilize thresholds, and (3) ensure missing/corrupt images don’t silently become a score of 0 by imputing a neutral score (mean) during calibration/inference.'
- What this solution (achieved 0.0) has done: 'Your score is far below the target, so the smallest likely win (without changing the model/training approach) is to fix the evaluation-metric alignment: QWK is computed on discrete labels, and your current threshold search is a coarse local step search that can easily get stuck and yield poor thresholds. I keep your exact model and “combined continuous score → 4 thresholds → 0–4 labels” pipeline, but replace the threshold fitting with a deterministic, stronger 1D coordinate descent (monotonic constraints preserved) that directly maximizes QWK on the same OOF scores you already compute. I also make the OOF split deterministic and properly cover the whole subset (still no training), so the calibration signal is less noisy, and then use those thresholds unchanged for test inference and submission.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with the model running with effectively random weights (the expected `.pkl` is very likely not present), and the current fallback only initializes the backbone while leaving the three heads random, yielding near-random labels after thresholding. I keep your exact inference/core logic (“three heads → continuous combined score → 4 thresholds → 0–4 labels”) but make the weight-loading fallback fully compatible by instantiating the *same* timm EfficientNet-B4 module whose weights we can actually load, and then mapping those weights into your wrapper. I also add a small, deterministic “sanity check” that if no real checkpoint is found, we skip threshold fitting on the random-head outputs and instead use a safe default threshold set (so we don’t overfit noise into extreme thresholds), which should move the score upward from 0.0. Finally, the submission writing remains identical and aligned to `test.csv`.'

# 9. Code solution

## === cell 0
import os
import math
import random
import glob
import time
import copy
import json
import pickle
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT

from PIL import Image, ImageChops, ImageEnhance, ImageOps
import cv2

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
from torch import Tensor
from torch.jit.annotations import List, Optional, Tuple


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super().__init__()
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
        super().__init__()
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

        super()._load_from_state_dict(
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
        return F.linear(input, self.weight, self.bias)


def _get_timm_act(net, prefer: str):
    """
    Bugfix: timm EfficientNet activation attributes vary across versions/models.
    For timm==1.0.19 some EfficientNet variants may not expose act1/act2 on the root module.
    We try common names; if none exist we fall back to Identity to keep execution stable.
    """
    candidates = []
    if prefer == "act1":
        candidates = ["act1", "swish", "act", "activation", "relu"]
    else:
        candidates = ["act2", "swish", "act", "activation", "relu"]

    for name in candidates:
        if hasattr(net, name):
            mod = getattr(net, name)
            if isinstance(mod, nn.Module):
                return mod

    if hasattr(net, "act_layer") and callable(getattr(net, "act_layer")):
        try:
            m = net.act_layer(inplace=True)
            if isinstance(m, nn.Module):
                return m
        except TypeError:
            m = net.act_layer()
            if isinstance(m, nn.Module):
                return m

    warnings.warn(
        f"Could not locate activation module for {prefer} on timm model {type(net)}; using Identity()."
    )
    return nn.Identity()


class backboneNet_efficient(nn.Module):
    def __init__(self):
        super().__init__()
        net = timm.create_model("tf_efficientnet_b4_ns", pretrained=False)

        layers_to_train = ["blocks"]
        for name, parameter in net.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        self.num_features = 1792
        self.conv_stem = net.conv_stem
        self.bn1 = net.bn1
        self.act1 = _get_timm_act(net, "act1")

        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]
        self.conv_head = net.conv_head
        self.bn2 = net.bn2
        self.act2 = _get_timm_act(net, "act2")

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

        x15 = self.rg_cls(x14)  # regressor logit
        x16 = self.cls_cls(x14)  # classifier logits
        x17 = self.ord_cls(x14)  # ordinal logits

        r_out = torch.sigmoid(x15) * 4.5
        o_out = torch.sigmoid(x17)
        return r_out, x16, o_out




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
        super().__init__()
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
        super().__init__()
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
        super().__init__()

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
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img




## === cell 5
BASE_INPUT = "/kaggle/input/aptos2019-blindness-detection"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../input/aptos2019-blindness-detection"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/data/aptos2019-blindness-detection"

train_csv_path = os.path.join(BASE_INPUT, "train.csv")
train_img_dir = os.path.join(BASE_INPUT, "train_images")

test_csv_path = os.path.join(BASE_INPUT, "test.csv")
test_img_dir = os.path.join(BASE_INPUT, "test_images")

train_df = pd.read_csv(train_csv_path)
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

ACTIVE_TRANSFORM = transform2


def find_weight_file(filename: str) -> str:
    candidates = [
        os.path.join("/kaggle/input", "weights", filename),
        os.path.join("../input", "weights", filename),
        os.path.join("/kaggle/input", filename),
        os.path.join("../input", filename),
        os.path.join("/kaggle/input/aptos2019-blindness-detection", filename),
        os.path.join("../input/aptos2019-blindness-detection", filename),
        os.path.join("/kaggle/data/aptos2019-blindness-detection", filename),
        os.path.join("/kaggle/data", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join("/kaggle/input", "**", filename), recursive=True)
    if hits:
        return hits[0]
    hits = glob.glob(os.path.join("../input", "**", filename), recursive=True)
    if hits:
        return hits[0]
    hits = glob.glob(os.path.join("/kaggle/data", "**", filename), recursive=True)
    if hits:
        return hits[0]
    return ""


def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "network"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _strip_common_prefixes(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    return state_dict


weight_name = "B4_3stage_4epoch_384finetune.pkl"
weight_path = find_weight_file(weight_name)

loaded_ckpt = False
used_pretrained_backbone = False
net1 = backboneNet_efficient().to(device)

if weight_path:
    ckpt = torch.load(weight_path, map_location=device)
    sd = _strip_common_prefixes(_unwrap_state_dict(ckpt))
    try:
        net1.load_state_dict(sd, strict=True)
        loaded_ckpt = True
        print(
            f"Loaded checkpoint strictly into backboneNet_efficient from: {weight_path}"
        )
    except Exception:
        incompatible = net1.load_state_dict(sd, strict=False)
        loaded_ckpt = True
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        print(
            f"Loaded checkpoint non-strictly into backboneNet_efficient from: {weight_path}"
        )
        print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
else:
    warnings.warn(
        f"Weight file '{weight_name}' not found under /kaggle/input, ../input, or /kaggle/data. "
        "Will fall back to ImageNet-pretrained EfficientNet-B4 weights."
    )

if not loaded_ckpt:
    try:
        tmp = (
            timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
            .to(device)
            .eval()
        )
        net1.conv_stem.load_state_dict(tmp.conv_stem.state_dict())
        net1.bn1.load_state_dict(tmp.bn1.state_dict())
        net1.act1 = _get_timm_act(tmp, "act1")

        for i, blk in enumerate(
            [
                net1.block0,
                net1.block1,
                net1.block2,
                net1.block3,
                net1.block4,
                net1.block5,
                net1.block6,
            ]
        ):
            blk.load_state_dict(tmp.blocks[i].state_dict())

        net1.conv_head.load_state_dict(tmp.conv_head.state_dict())
        net1.bn2.load_state_dict(tmp.bn2.state_dict())
        net1.act2 = _get_timm_act(tmp, "act2")
        net1.global_pool = tmp.global_pool

        net1 = net1.to(device)
        used_pretrained_backbone = True
        print(
            "Initialized backboneNet_efficient backbone with pretrained EfficientNet-B4 weights as fallback."
        )
    except Exception as e:
        warnings.warn(
            f"Pretrained backbone fallback failed; proceeding with random weights. Error: {e}"
        )

net1.eval()




## === cell 6
def _apply_thresholds(vals: np.ndarray, thr: np.ndarray) -> np.ndarray:
    thr = np.asarray(thr, dtype=np.float32)
    preds = np.zeros(vals.shape[0], dtype=np.int64)
    for t in thr:
        preds += (vals >= t).astype(np.int64)
    return preds


def _qwk_for_thr(y_true: np.ndarray, y_cont: np.ndarray, thr: np.ndarray) -> float:
    return cohen_kappa_score(
        y_true, _apply_thresholds(y_cont, thr), weights="quadratic"
    )


def _fit_thresholds_for_qwk(y_true: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """
    Metric-aligned coordinate descent for 4 ordered thresholds.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.float32)

    q = np.quantile(y_val, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    thr = np.clip(q, 0.05, 3.95)
    thr = np.maximum.accumulate(thr)  # monotone

    best_thr = thr.copy()
    best_score = _qwk_for_thr(y_true, y_val, best_thr)

    schedules = [
        dict(span=0.60, n=41),
        dict(span=0.30, n=41),
        dict(span=0.15, n=41),
        dict(span=0.08, n=33),
    ]

    for sch in schedules:
        span = sch["span"]
        n = sch["n"]
        improved_any = True
        it = 0
        while improved_any and it < 20:
            improved_any = False
            it += 1
            for i in range(4):
                lo = 0.0 if i == 0 else best_thr[i - 1] + 1e-4
                hi = 4.0 if i == 3 else best_thr[i + 1] - 1e-4
                if hi <= lo:
                    continue

                center = float(best_thr[i])
                a = max(lo, center - span)
                b = min(hi, center + span)
                if b <= a:
                    continue

                grid = np.linspace(a, b, n, dtype=np.float32)
                if not np.any(np.isclose(grid, best_thr[i], atol=1e-6)):
                    grid = np.sort(
                        np.unique(np.concatenate([grid, best_thr[i : i + 1]]))
                    ).astype(np.float32)

                local_best = best_thr[i]
                local_best_score = best_score
                for cand in grid:
                    trial = best_thr.copy()
                    trial[i] = float(cand)
                    s = _qwk_for_thr(y_true, y_val, trial)
                    if s > local_best_score + 1e-8:
                        local_best_score = s
                        local_best = float(cand)

                if local_best_score > best_score + 1e-8:
                    best_score = local_best_score
                    best_thr[i] = local_best
                    improved_any = True

    best_thr = np.clip(best_thr, 0.0, 4.0).astype(np.float32)
    best_thr = np.maximum.accumulate(best_thr)
    return best_thr


def _infer_combined_scores_batch(model, img_paths, batch_size=16) -> np.ndarray:
    """
    Core logic unchanged: convert each head to 0..4 continuous scale and average.
    Kept consistent ACTIVE_TRANSFORM for calibration + test.
    """
    scores = np.full(len(img_paths), np.nan, dtype=np.float32)
    model.eval()

    class_vals = torch.arange(5, device=device, dtype=torch.float32)

    t0 = time.time()
    for start in range(0, len(img_paths), batch_size):
        end = min(len(img_paths), start + batch_size)
        xs = []
        pos = []
        for j, p in enumerate(img_paths[start:end]):
            try:
                img = Image.open(p).convert("RGB")
                xs.append(ACTIVE_TRANSFORM(img))
                pos.append(start + j)
            except Exception:
                continue

        if len(xs) == 0:
            continue

        x = torch.stack(xs, dim=0).to(device)
        with torch.inference_mode():
            r_out, c_out, o_out = model(x)

            r_cont = r_out.squeeze(1).clamp(0.0, 4.0)  # (B,)
            c_prob = F.softmax(c_out, dim=1)  # (B,5)
            c_cont = (c_prob * class_vals.unsqueeze(0)).sum(dim=1).clamp(0.0, 4.0)
            o_cont = o_out.sum(dim=1).clamp(0.0, 4.0)  # ordinal expected class approx

            P = (
                ((r_cont + c_cont + o_cont) / 3.0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        for k, idx in enumerate(pos):
            scores[idx] = P[k]

        if (end % 128) == 0 or end == len(img_paths):
            print(
                f"Batched inference {end}/{len(img_paths)} done. Elapsed {time.time()-t0:.1f}s"
            )

    if np.isnan(scores).any():
        mean_val = np.nanmean(scores)
        if not np.isfinite(mean_val):
            mean_val = 0.0
        scores = np.where(np.isnan(scores), mean_val, scores).astype(np.float32)

    return scores


rng = np.random.RandomState(42)

subset_n = min(2560, len(train_df))
all_idxs = np.arange(len(train_df))
rng.shuffle(all_idxs)
subset = all_idxs[:subset_n]

y_subset = train_df.loc[subset, "diagnosis"].astype(int).values
order = np.lexsort((rng.rand(subset_n), y_subset))  # by label then random within label
subset = subset[order]
y_subset = y_subset[order]

folds = 3
oof_score = np.zeros(subset_n, dtype=np.float32)
oof_true = y_subset.copy()

for f in range(folds):
    val_mask = (np.arange(subset_n) % folds) == f
    val_idxs = subset[val_mask]
    val_paths = [
        os.path.join(train_img_dir, f"{train_df.loc[i, 'id_code']}.png")
        for i in val_idxs
    ]
    oof_score[val_mask] = _infer_combined_scores_batch(net1, val_paths, batch_size=16)

DEFAULT_THR = np.array([0.75, 1.5, 2.5, 3.5], dtype=np.float32)

if loaded_ckpt:
    best_thr = _fit_thresholds_for_qwk(oof_true, oof_score)
    print("Calibrated thresholds (combined score):", best_thr.tolist())
    print(
        "OOF QWK with calibrated thresholds:",
        cohen_kappa_score(
            oof_true, _apply_thresholds(oof_score, best_thr), weights="quadratic"
        ),
    )
else:
    best_thr = DEFAULT_THR
    print(
        "No competition checkpoint loaded; using default thresholds:", best_thr.tolist()
    )

threshold = [float(x) for x in best_thr.tolist()]



## === cell 7
test_paths = [os.path.join(test_img_dir, f"{idx}.png") for idx in test_ids]
test_scores = _infer_combined_scores_batch(net1, test_paths, batch_size=16)
test_preds = _apply_thresholds(test_scores, best_thr)
test_preds = np.clip(test_preds, 0, 4).astype(int)

submission = pd.DataFrame({"id_code": test_ids, "diagnosis": test_preds})



## === cell 8
assert submission.shape[0] == len(test_df), "Submission row count mismatch vs test.csv"
submission = submission.merge(test_df[["id_code"]], on="id_code", how="right")
submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int)
submission = submission[["id_code", "diagnosis"]]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
