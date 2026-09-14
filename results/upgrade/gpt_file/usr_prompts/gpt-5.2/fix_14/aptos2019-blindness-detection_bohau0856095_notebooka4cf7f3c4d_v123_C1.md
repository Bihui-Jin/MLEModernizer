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

0.9259551359794772

# 6. Current score

0.04562

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.11192) has done: 'I fix the missing checkpoint path by loading pretrained EfficientNet weights (available in the Kaggle environment) and running inference without relying on a non-existent `../input/weights/3.pth`. I also fix the device/dtype mismatch by ensuring the model and inputs are on the same device, and make submission creation robust by keeping `submission` as a list and using `len(submission)` (not `.shape`). These changes keep the same model architecture/heads and the same prediction aggregation logic, but make the notebook run end-to-end and actually write a valid `submission.csv`. Since no valid submission was yielded before, the priority is correctness and producing a valid file; any score now be meaningful and should be reasonable due to using pretrained backbone weights instead of random weights.'
- What this solution (achieved -0.11192) has done: 'Your score is far below the target, so the smallest reliable way to move it upward is to fix the inference-time mismatch between your model head and the pretrained backbone features. Right now you’re taking `forward_features()` output but never applying EfficientNet’s `bn2`/`classifier`-equivalent final normalization, so the heads see poorly calibrated features and predictions collapse (very low QWK). I keep the same EfficientNet backbone and the same three-head architecture and aggregation logic, but change only the feature extraction line to use `forward_head(..., pre_logits=True)` when available so the pooled embedding matches what the pretrained weights were trained to produce. This should materially improve agreement without changing the overall approach, and it keeps the script end-to-end and submission format identical.'
- What this solution (achieved -0.12612) has done: 'Your current score is far below the target, so we should only make small inference-time fixes that legitimately improve prediction quality without changing the model or training approach. The biggest issue is that you are using a timm-pretrained backbone but leaving your three heads randomly initialized, which makes predictions close to random (very low QWK). I keep the same EfficientNet backbone and the same three-head/aggregation logic, but make the backbone output class logits via its own pretrained classifier for the “classification head” stream, and derive regression/ordinal streams from that same pretrained probability vector (so all three streams are at least consistent and non-random). This is a minimal change (no training, no architecture rewrite) and should move QWK strongly upward toward the target band while still writing a valid `submission.csv`.'
- What this solution (achieved 0.12659) has done: 'Your current QWK is far below the target, so we should make the smallest inference-only change that legitimately improves prediction quality without changing the overall approach. Right now the “cls_logits_5” are taken from 5 arbitrary ImageNet classes, which are not meaningful for DR and makes the 3-stream aggregation close to random. I keep the same EfficientNet backbone and the same three-output interface/aggregation, but instead map the pretrained 1000-class probability vector into 5 DR classes using a fixed, monotonic binning of ImageNet class indices (a deterministic, training-free projection). This preserves the core logic (pretrained backbone -> 3 outputs -> average of 3 discrete predictions) while making the classification stream non-degenerate and typically much closer to ordinal structure, which should move QWK strongly upward toward the target band and still writes a valid submission.csv.'
- What this solution (achieved 0.12659) has done: 'Your current score (0.12659) is far below the target (0.92596), so we should make the smallest legitimate inference-time change that improves alignment with the QWK metric without changing the training approach or introducing new modeling. The biggest lever available in your current pipeline is the fixed post-processing thresholds: QWK is very sensitive to the 4 cut points that map a continuous severity signal to {0..4}. I keep your exact model, transforms, and 3-stream aggregation, but I (1) compute thresholds once from the training label distribution (quantile-based cut points), and (2) apply those thresholds to the regression stream instead of hardcoded values; this typically moves predictions closer to the dataset’s ordinal prevalence and improves QWK. Everything still runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.05728) has done: 'Your current gap to the target is large (0.1266 vs 0.926, higher-is-better), so the smallest legitimate improvement is to make the 3-stream ensemble less noisy without changing the model or training. I keep your backbone, 5-bin projection, and 3-output interface exactly the same, but (1) compute class-dependent weights from the training label distribution and use a weighted average instead of a plain mean when combining the three predictions (this often improves QWK because it reduces variance from weaker streams). I also (2) switch test-time preprocessing from a plain center-crop to a deterministic 5-crop TTA (center + 4 corners) and average the model outputs across crops; this is still inference-only, deterministic, and keeps evaluation semantics identical while typically raising QWK. Everything still runs end-to-end and writes a valid `submission.csv` in the required format.'
- What this solution (achieved 0.08472) has done: 'Your current score (0.05728) is far below the target (0.92596), so we need a real predictive lift while keeping your inference-only pipeline and 3-stream aggregation intact. The biggest issue is that your three “heads” are effectively synthetic projections of ImageNet logits and are not calibrated to the DR label distribution, so the final rounding is close to random. With minimal change and no new training loop, we can fit only the 4 regression cutpoints (thresholds) on a small validation split using QWK directly (a standard post-processing step for this metric), then use those optimized thresholds at inference; this preserves your model, forward pass, and ensemble logic. I also keep your existing quantile thresholds as a safe fallback if optimization fails, and I keep the same submission writing/format.'
- What this solution (achieved 0.45368) has done: 'I make a minimal, inference-only improvement that directly targets the low QWK by replacing the current synthetic 5-bin projection from ImageNet classes with a dataset-calibrated projection learned from a small validation split (no training loop, no new model, just a deterministic post-processing mapping). This keeps your EfficientNet backbone inference and the same 3-stream aggregation structure, but makes the classification stream non-random by mapping the 1000-d ImageNet probability vector into 5 DR classes using non-negative least squares fitted to match the training labels on a small held-out set. I keep your existing threshold optimization for the regression stream, and I also calibrate the stream weights on the same validation split via a tiny grid search to improve agreement without changing the model. The result remains deterministic, runs end-to-end under 600s, and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.33879) has done: 'Your current score (0.45368) is far below the target (0.92596), so we should make small, inference-only calibration fixes that can legitimately lift QWK without changing your model/backbone or introducing training loops. The biggest bottleneck is that your calibrated 1000→5 projection is currently fit with an ad-hoc gradient procedure that can be unstable and underfit; replacing it with a deterministic, regularized ridge regression on the *log-probabilities* (then row-normalizing) typically yields a stronger, smoother class projection while keeping the same “1000→5 mapping” core logic. Second, your regression stream currently derives `rg_val` from the regression logit but your optimized thresholds are fit on a combined ensemble; we keep that unchanged, but make the per-image regression value less noisy by averaging the *pre-sigmoid logits* across crops (you already do) and ensuring the same transform path is used in calibration and test (it is). These changes are minimal, deterministic, and should move the score upward toward the target band while still producing a valid `submission.csv`.'
- What this solution (achieved 0.46926) has done: 'Your current score is far below the target (0.3388 vs 0.926, higher-is-better), so we should make small inference-only calibration improvements without changing the backbone/model wrapper or introducing any training loops. The main issue is that the current 1000→5 ridge projection is fit as a plain classifier but then used as a *probability mixing* matrix, which benefits from being fit directly on a probability simplex; switching to a stabilized multiplicative-IPF fit for a non-negative, row-stochastic matrix usually improves the classification stream a lot while keeping the exact same “ImageNet probs → 5-class probs → log → aggregate” logic. Second, the regression stream currently uses a derived sigmoid-logit; adding a tiny monotonic calibration of the continuous rg_val (a 1D affine fit on the validation subset) reduces bias before thresholding and tends to lift QWK without changing thresholds/aggregation semantics. Both changes are deterministic, keep runtime reasonable, and still write a valid `submission.csv`.'
- What this solution (achieved 0.04562) has done: 'Your current score (0.46926) is far below the target (0.92596), so we should make a small, legitimate calibration change that improves QWK without altering the backbone/model wrapper or adding any training loop. The biggest remaining mismatch is that you currently mix *hard* class decisions from the cls/ord streams with the reg stream, which throws away useful uncertainty information and typically hurts QWK; we can keep the same three-stream aggregation but combine them at the probability level and only discretize once at the end. Concretely, we (1) compute a 5-class probability vector from each stream (reg via a smooth triangular distribution around rg_val, cls via softmax, ord via the standard ordinal→class conversion), (2) average those probabilities with the same weights, and (3) take argmax to get the final class. This preserves your inference-only approach, your three outputs, your existing calibration (IPF projection + affine rg + threshold optimization), and still writes a valid `submission.csv`.'
- What this solution (achieved 0.04562) has done: 'Your current score is far below the target, so the smallest reliable lift (without changing the backbone/model or adding training) is to fix a bug in the ordinal-to-probability conversion: right now it incorrectly applies a softmax to already-valid ordinal class probabilities, which destroys the intended distribution and hurts QWK. I keep your exact three-stream probability-level ensemble, calibration steps, and thresholds/weights logic, but return the ordinal probabilities normalized (not softmaxed) so that the ordinal stream meaningfully contributes. This is a minimal, deterministic change that should move the score upward toward the target while keeping runtime and submission format unchanged.'

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
import csv
import glob
import copy
import os
import json
import pickle
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as tv_transforms, utils, models, datasets
from PIL import ImageEnhance, ImageOps

import warnings


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
    BatchNorm2d where the batch statistics and affine parameters are fixed.
    (Kept to preserve original script structure; not used in final inference.)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, n: int = None):
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
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
            return F.linear(input, self.weight.to(dtype=input.dtype), bias=bias)
        else:
            return F.linear(input, self.weight, self.bias)


class backboneNet_efficient(nn.Module):
    """
    Inference-only model wrapper (core logic preserved):
    pretrained ImageNet EfficientNet forward() logits -> (project to 5 DR logits) -> derive regression/ordinal.
    """

    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        self.net = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=pretrained_backbone
        )

        layers_to_train = ["blocks"]
        for name, parameter in self.net.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        self.num_features = getattr(self.net, "num_features", 1792)
        self.drop_rate = 0.4

        self.rg_cls = Linear(self.num_features, 1, bias=True)
        self.cls_cls = Linear(self.num_features, 5, bias=True)
        self.ord_cls = Linear(self.num_features, 4, bias=True)

        self.register_buffer("proj_W_5x1000", torch.empty(0), persistent=False)

    def set_projection(self, W_5x1000: np.ndarray):
        W = torch.tensor(W_5x1000, dtype=torch.float32)
        self.proj_W_5x1000 = W

    def forward(self, x):
        logits_1000 = self.net(x)  # [B, 1000] (pretrained ImageNet logits)
        probs_1000 = torch.softmax(logits_1000, dim=1)

        if self.proj_W_5x1000.numel() == 5 * 1000:
            W = self.proj_W_5x1000.to(
                device=probs_1000.device, dtype=probs_1000.dtype
            )  # [5,1000]
            probs5 = probs_1000 @ W.t()  # [B,5]
            probs5 = probs5.clamp(1e-8, 1.0)
            probs5 = probs5 / probs5.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            b0 = probs_1000[:, 0:200].sum(dim=1, keepdim=True)
            b1 = probs_1000[:, 200:400].sum(dim=1, keepdim=True)
            b2 = probs_1000[:, 400:600].sum(dim=1, keepdim=True)
            b3 = probs_1000[:, 600:800].sum(dim=1, keepdim=True)
            b4 = probs_1000[:, 800:1000].sum(dim=1, keepdim=True)
            probs5 = torch.cat([b0, b1, b2, b3, b4], dim=1).clamp(1e-8, 1.0)

        cls_logits_5 = torch.log(probs5)

        exp_class = (
            probs5 * torch.arange(5, device=probs5.device, dtype=probs5.dtype)
        ).sum(
            dim=1
        )  # [B]
        exp_scaled = (exp_class / 4.5).clamp(1e-5, 1 - 1e-5)  # target for sigmoid
        rg_logit = torch.log(exp_scaled / (1 - exp_scaled)).unsqueeze(1)  # [B,1]

        p_gt = torch.stack(
            [
                1.0 - probs5[:, 0],
                probs5[:, 2] + probs5[:, 3] + probs5[:, 4],
                probs5[:, 3] + probs5[:, 4],
                probs5[:, 4],
            ],
            dim=1,
        ).clamp(1e-5, 1 - 1e-5)
        ord_logits = torch.log(p_gt / (1 - p_gt))  # [B,4]

        return rg_logit, cls_logits_5, ord_logits




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
    pred_prob = pred_prob.clamp_min(1e-8)
    pred_prob = pred_prob / pred_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return pred_prob


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(float(out[i].item())))
            l2 = int(math.ceil(float(out[i].item())))
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
    if img is None:
        return None
    if img.ndim == 2:
        mask = img > tol
        if mask.any():
            return img[np.ix_(mask.any(1), mask.any(0))]
        return img
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)
    return img




## === cell 5
DATA_DIR = "/kaggle/input/aptos2019-blindness-detection"
WEIGHTS_PATH = (
    "../input/weights/3.pth"  # kept unchanged; we will gracefully fallback if missing.
)

test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
test_ids = test_df["id_code"].values

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
_train_labels = train_df["diagnosis"].astype(int).values

_props = np.cumsum(np.bincount(_train_labels, minlength=5)[:4]) / len(_train_labels)
thrs_quant = (
    np.quantile(_train_labels.astype(np.float32), _props) * (4.5 / 4.0)
).tolist()
thrs_quant = [float(np.clip(t, 0.01, 4.49)) for t in thrs_quant]
thrs_quant = sorted(thrs_quant)
print("Baseline (quantile) regression thresholds:", thrs_quant)

_label_counts = np.bincount(_train_labels, minlength=5).astype(np.float64)
_label_probs = _label_counts / _label_counts.sum()
imbalance = float(
    (_label_probs.max() - _label_probs.min()) / (_label_probs.max() + 1e-12)
)
w_reg = 0.30
w_cls = 0.40
w_ord = 0.30 + 0.20 * min(1.0, imbalance)
wsum = w_reg + w_cls + w_ord
w_reg, w_cls, w_ord = w_reg / wsum, w_cls / wsum, w_ord / wsum
print("Initial stream weights (reg, cls, ord):", (w_reg, w_cls, w_ord))

_base_transform = transforms.Compose(
    [
        transforms.Resize((280, 280)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def five_crop_tensors(img_pil, crop_size=256):
    crops = FT.five_crop(img_pil, size=(crop_size, crop_size))  # tuple of 5 PIL images
    xs = [(_base_transform(c)).unsqueeze(0) for c in crops]  # list of [1,3,H,W]
    return torch.cat(xs, dim=0)  # [5,3,H,W]


use_external_ckpt = os.path.exists(WEIGHTS_PATH)
net4 = backboneNet_efficient(pretrained_backbone=(not use_external_ckpt))

if use_external_ckpt:
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
    state_dict = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )
    if isinstance(state_dict, dict):
        new_sd = {}
        for k, v in state_dict.items():
            nk = k.replace("module.", "")
            new_sd[nk] = v
        state_dict = new_sd

    missing, unexpected = net4.load_state_dict(state_dict, strict=False)
    print(
        "Loaded external weights. Missing keys:",
        len(missing),
        "Unexpected keys:",
        len(unexpected),
    )
else:
    print(
        f"WARNING: Checkpoint not found at {WEIGHTS_PATH}. Using timm pretrained backbone weights."
    )

net4 = net4.to(device)
net4.eval()


def _predict_one_image_outputs_and_probs1000(img_pil):
    """Returns (rg_val_continuous, cls_predict, ord_predict, probs_1000_avg) for ONE image, using 5-crop TTA."""
    x5 = five_crop_tensors(img_pil, crop_size=256).to(device)

    logits1000_5 = net4.net(x5)  # [5,1000]
    probs1000_5 = torch.softmax(logits1000_5, dim=1)
    probs1000_avg = probs1000_5.mean(dim=0, keepdim=True)  # [1,1000]

    rg_outputs, cls_outputs, ord_outputs = net4(x5)
    rg_logit = rg_outputs.mean(dim=0, keepdim=True)
    cls_logit = cls_outputs.mean(dim=0, keepdim=True)
    ord_logit = ord_outputs.mean(dim=0, keepdim=True)

    rg_val = float((torch.sigmoid(rg_logit) * 4.5).squeeze().item())
    cls_probs = torch.softmax(cls_logit, dim=1)
    cls_predict = int(torch.argmax(cls_probs, dim=1).item())
    ord_probs = torch.sigmoid(ord_logit)
    ord_predict = int((ord_probs >= 0.5).sum(dim=1).item())

    return (
        rg_val,
        cls_predict,
        ord_predict,
        probs1000_avg.squeeze(0).detach().cpu().numpy(),
    )


def _apply_thresholds_to_rg(rg_val, thrs4):
    if rg_val < thrs4[0]:
        return 0
    elif rg_val < thrs4[1]:
        return 1
    elif rg_val < thrs4[2]:
        return 2
    elif rg_val < thrs4[3]:
        return 3
    else:
        return 4


def _optimize_thresholds_qwk(
    rg_vals, cls_preds, ord_preds, y_true, w_reg, w_cls, w_ord, init_thrs
):
    """Coordinate-descent threshold optimization for QWK. Keeps core logic; changes only cutpoints."""
    rg_vals = np.asarray(rg_vals, dtype=np.float32)
    cls_preds = np.asarray(cls_preds, dtype=np.int64)
    ord_preds = np.asarray(ord_preds, dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.int64)

    def score(thrs):
        rg_pred = np.zeros_like(y_true)
        rg_pred[rg_vals >= thrs[0]] += 1
        rg_pred[rg_vals >= thrs[1]] += 1
        rg_pred[rg_vals >= thrs[2]] += 1
        rg_pred[rg_vals >= thrs[3]] += 1
        P = w_reg * rg_pred + w_cls * cls_preds + w_ord * ord_preds
        P = np.rint(P).astype(int)
        P = np.clip(P, 0, 4)
        return cohen_kappa_score(y_true, P, weights="quadratic")

    best = list(map(float, init_thrs))
    best_score = score(best)

    for _ in range(6):
        improved_any = False
        for j in range(4):
            cur = best[j]
            lo = 0.01 if j == 0 else best[j - 1] + 1e-3
            hi = 4.49 if j == 3 else best[j + 1] - 1e-3
            lo = float(max(lo, cur - 0.6))
            hi = float(min(hi, cur + 0.6))
            if hi <= lo + 1e-4:
                continue
            candidates = np.linspace(lo, hi, 31, dtype=np.float32)
            local_best_t = cur
            local_best_s = best_score
            for t in candidates:
                th = best.copy()
                th[j] = float(t)
                s = score(th)
                if s > local_best_s + 1e-12:
                    local_best_s = s
                    local_best_t = float(t)
            if local_best_t != cur:
                best[j] = local_best_t
                best_score = local_best_s
                improved_any = True
        if not improved_any:
            break

    return best, float(best_score)


def _fit_projection_ipf(X_probs1000, y, alpha=1e-3, iters=80):
    """
    Fit W as a probability-mixing matrix using an IPF-like multiplicative update.
    """
    X = np.asarray(X_probs1000, dtype=np.float64)  # [N,1000], rows sum to 1
    y = np.asarray(y, dtype=np.int64)
    N, D = X.shape
    K = 5

    A = np.zeros((K, D), dtype=np.float64)
    for k in range(K):
        m = y == k
        if m.any():
            A[k] = X[m].sum(axis=0)
        else:
            A[k] = 0.0

    A = A + float(alpha)
    W = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)

    target_col = X.sum(axis=0) + float(alpha)  # [D]
    for _ in range(int(iters)):
        col_sum = W.sum(axis=0) + 1e-12
        W *= (target_col / col_sum)[None, :]
        W = np.maximum(W, 0.0)
        W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-12)

    return W.astype(np.float32)


def _fit_rg_affine_calibration(rg_vals, y_true):
    """
    Tiny monotonic calibration for regression stream before thresholding.
    """
    rg = np.asarray(rg_vals, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)

    x = rg
    X = np.stack([x, np.ones_like(x)], axis=1)  # [N,2]
    XtX = X.T @ X
    XtX.flat[::3] += 1e-6  # tiny ridge for stability
    w = np.linalg.solve(XtX, X.T @ y)  # [2]
    a, b = float(w[0]), float(w[1])

    a = float(np.clip(a, 0.25, 4.0))
    b = float(np.clip(b, -2.0, 2.0))
    return a, b


def _optimize_stream_weights_qwk(rg_vals, cls_preds, ord_preds, y_true, thrs, w_init):
    rg_vals = np.asarray(rg_vals, dtype=np.float32)
    cls_preds = np.asarray(cls_preds, dtype=np.int64)
    ord_preds = np.asarray(ord_preds, dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.int64)
    thrs = list(map(float, thrs))

    rg_pred = np.zeros_like(y_true)
    rg_pred[rg_vals >= thrs[0]] += 1
    rg_pred[rg_vals >= thrs[1]] += 1
    rg_pred[rg_vals >= thrs[2]] += 1
    rg_pred[rg_vals >= thrs[3]] += 1

    def score(wr, wc, wo):
        P = wr * rg_pred + wc * cls_preds + wo * ord_preds
        P = np.rint(P).astype(int)
        P = np.clip(P, 0, 4)
        return cohen_kappa_score(y_true, P, weights="quadratic")

    best_w = tuple(map(float, w_init))
    best_s = score(*best_w)

    grid = np.linspace(0.05, 0.90, 18)
    for wr in grid:
        for wc in grid:
            wo = 1.0 - wr - wc
            if wo < 0.05 or wo > 0.90:
                continue
            s = score(wr, wc, wo)
            if s > best_s + 1e-12:
                best_s = float(s)
                best_w = (float(wr), float(wc), float(wo))

    return best_w, float(best_s)


def _regress_value_to_prob5(rg_val_cont):
    rg = float(np.clip(rg_val_cont, 0.0, 4.5))
    prob = np.zeros(5, dtype=np.float32)
    if rg >= 4.0:
        prob[4] = 1.0
        return prob
    l1 = int(math.floor(rg))
    l2 = int(math.ceil(rg))
    if l1 == l2:
        prob[l1] = 1.0
    else:
        prob[l1] = float(1.0 - (rg - l1))
        prob[l2] = float(1.0 - (l2 - rg))
    s = float(prob.sum())
    if s <= 0:
        prob[0] = 1.0
    else:
        prob /= s
    return prob


def _ordinal_logits_to_prob5(ord_logits_1x4):
    p = torch.sigmoid(ord_logits_1x4)  # [1,4]
    prob = torch.zeros((1, 5), device=p.device, dtype=p.dtype)
    prob[:, 0] = 1 - p[:, 0]
    prob[:, 1] = p[:, 0] * (1 - p[:, 1])
    prob[:, 2] = p[:, 1] * (1 - p[:, 2])
    prob[:, 3] = p[:, 2] * (1 - p[:, 3])
    prob[:, 4] = p[:, 3]
    prob = prob.clamp_min(1e-8)
    prob = prob / prob.sum(dim=1, keepdim=True)
    return prob  # [1,5]


val_size = 260  # kept small for speed; deterministic subset
rng = np.random.RandomState(42)
perm = rng.permutation(len(train_df))
val_idx = perm[:val_size]

val_rg_vals, val_cls_preds, val_ord_preds, val_y = [], [], [], []
val_probs1000 = []
train_img_dir = f"{DATA_DIR}/train_images"

t0 = time.time()
with torch.no_grad():
    for k, ridx in enumerate(val_idx):
        if k % 50 == 0:
            print("Calibrating on val:", k, "/", len(val_idx))
        img_id = train_df.iloc[ridx]["id_code"]
        y = int(train_df.iloc[ridx]["diagnosis"])
        image_name = f"{train_img_dir}/{img_id}.png"

        img_bgr = cv2.imread(image_name)
        if img_bgr is None:
            img_pil = Image.open(image_name).convert("RGB")
        else:
            img_bgr = crop_image_from_gray(img_bgr)
            if img_bgr is None:
                img_pil = Image.open(image_name).convert("RGB")
            else:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

        rg_val, cls_p, ord_p, p1000 = _predict_one_image_outputs_and_probs1000(img_pil)
        val_rg_vals.append(rg_val)
        val_cls_preds.append(cls_p)
        val_ord_preds.append(ord_p)
        val_probs1000.append(p1000)
        val_y.append(y)

print("Calibration inference seconds:", round(time.time() - t0, 2))

val_probs1000 = np.asarray(val_probs1000, dtype=np.float32)

try:
    W_cal = _fit_projection_ipf(val_probs1000, np.asarray(val_y), alpha=1e-3, iters=80)
    net4.set_projection(W_cal)
    print(
        "Installed calibrated 1000->5 projection W (IPF / row-stochastic) into model."
    )
except Exception as e:
    print(
        "WARNING: projection fitting failed; keeping default binning. Error:", repr(e)
    )

try:
    rg_a, rg_b = _fit_rg_affine_calibration(val_rg_vals, val_y)
    print(
        "Installed regression affine calibration: rg' = a*rg + b with a,b =",
        (rg_a, rg_b),
    )
except Exception as e:
    rg_a, rg_b = 1.0, 0.0
    print(
        "WARNING: regression affine calibration failed; using identity. Error:", repr(e)
    )

val_rg_vals2, val_cls_preds2, val_ord_preds2 = [], [], []
t1 = time.time()
with torch.no_grad():
    for k, ridx in enumerate(val_idx):
        img_id = train_df.iloc[ridx]["id_code"]
        image_name = f"{train_img_dir}/{img_id}.png"

        img_bgr = cv2.imread(image_name)
        if img_bgr is None:
            img_pil = Image.open(image_name).convert("RGB")
        else:
            img_bgr = crop_image_from_gray(img_bgr)
            if img_bgr is None:
                img_pil = Image.open(image_name).convert("RGB")
            else:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

        x5 = five_crop_tensors(img_pil, crop_size=256).to(device)
        rg_outputs, cls_outputs, ord_outputs = net4(x5)

        rg_logit = rg_outputs.mean(dim=0, keepdim=True)
        cls_logit = cls_outputs.mean(dim=0, keepdim=True)
        ord_logit = ord_outputs.mean(dim=0, keepdim=True)

        rg_val = float((torch.sigmoid(rg_logit) * 4.5).squeeze().item())
        rg_val = float(np.clip(rg_a * rg_val + rg_b, 0.0, 4.5))

        cls_probs = torch.softmax(cls_logit, dim=1)
        cls_predict = int(torch.argmax(cls_probs, dim=1).item())
        ord_probs = torch.sigmoid(ord_logit)
        ord_predict = int((ord_probs >= 0.5).sum(dim=1).item())

        val_rg_vals2.append(rg_val)
        val_cls_preds2.append(cls_predict)
        val_ord_preds2.append(ord_predict)

print("Recalibration pass seconds:", round(time.time() - t1, 2))

try:
    thrs_opt, qwk_opt = _optimize_thresholds_qwk(
        val_rg_vals2,
        val_cls_preds2,
        val_ord_preds2,
        val_y,
        w_reg,
        w_cls,
        w_ord,
        thrs_quant,
    )
    qwk_base = cohen_kappa_score(
        np.asarray(val_y),
        np.clip(
            np.rint(
                w_reg
                * (
                    np.sum(
                        np.asarray(val_rg_vals2)[:, None]
                        >= np.asarray(thrs_quant)[None, :],
                        axis=1,
                    )
                )
                + w_cls * np.asarray(val_cls_preds2)
                + w_ord * np.asarray(val_ord_preds2)
            ).astype(int),
            0,
            4,
        ),
        weights="quadratic",
    )
    print("Val QWK baseline (quantile thrs):", float(qwk_base))
    print("Val QWK optimized (opt thrs):    ", float(qwk_opt))
    thrs = thrs_opt
except Exception as e:
    print(
        "WARNING: threshold optimization failed, falling back to quantile thresholds. Error:",
        repr(e),
    )
    thrs = thrs_quant

try:
    (w_reg2, w_cls2, w_ord2), qwk_w = _optimize_stream_weights_qwk(
        val_rg_vals2, val_cls_preds2, val_ord_preds2, val_y, thrs, (w_reg, w_cls, w_ord)
    )
    print("Val QWK with optimized weights:", float(qwk_w))
    w_reg, w_cls, w_ord = w_reg2, w_cls2, w_ord2
except Exception as e:
    print(
        "WARNING: weight optimization failed; keeping initial weights. Error:", repr(e)
    )

print("Using regression thresholds:", thrs)
print("Using stream weights (reg, cls, ord):", (w_reg, w_cls, w_ord))



## === cell 6
submission = []

test_img_dir = f"{DATA_DIR}/test_images"

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print("Predicting", i, "/", len(test_ids))

        image_name = f"{test_img_dir}/{idx}.png"

        img_bgr = cv2.imread(image_name)
        if img_bgr is None:
            img_pil = Image.open(image_name).convert("RGB")
        else:
            img_bgr = crop_image_from_gray(img_bgr)
            if img_bgr is None:
                img_pil = Image.open(image_name).convert("RGB")
            else:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

        x5 = five_crop_tensors(img_pil, crop_size=256).to(device)
        rg_outputs, cls_outputs, ord_outputs = net4(x5)

        rg_logit = rg_outputs.mean(dim=0, keepdim=True)
        cls_logit = cls_outputs.mean(dim=0, keepdim=True)
        ord_logit = ord_outputs.mean(dim=0, keepdim=True)

        rg_val = float((torch.sigmoid(rg_logit) * 4.5).squeeze().item())
        rg_val = float(np.clip(rg_a * rg_val + rg_b, 0.0, 4.5))

        p_reg = _regress_value_to_prob5(rg_val)  # [5] numpy
        p_cls = (
            torch.softmax(cls_logit, dim=1)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )  # [5]
        p_ord = (
            _ordinal_logits_to_prob5(ord_logit)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )  # [5]

        p_ens = (w_reg * p_reg + w_cls * p_cls + w_ord * p_ord).astype(np.float32)
        p_ens = np.clip(p_ens, 1e-8, 1.0)
        p_ens = p_ens / float(p_ens.sum())
        P = int(np.argmax(p_ens))

        submission.append([idx, P])

print("Submission rows:", len(submission))



## === cell 7
if len(submission) == 0:
    raise RuntimeError("No predictions were generated; submission would be empty.")

df_pred = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df_pred["diagnosis"] = df_pred["diagnosis"].astype(int)

df = test_df.merge(df_pred, on="id_code", how="left")
if df["diagnosis"].isna().any():
    missing_ids = df.loc[df["diagnosis"].isna(), "id_code"].tolist()[:5]
    raise RuntimeError(
        f"Some test ids are missing predictions (examples: {missing_ids})."
    )

df[["id_code", "diagnosis"]].to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with", len(df), "rows")
print("submission.csv columns:", list(df[["id_code", "diagnosis"]].columns))
