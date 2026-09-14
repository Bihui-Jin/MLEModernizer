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

3.7

# 3. Installed packages

No external packages required in the script and installed.

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

0.911371255572809

# 6. Current score

0.03321

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the notebook-only magics and fix missing imports so the script can run as a normal Kaggle Python run. Because `fastai` is not available (and `fastai.callbacks` import fails), I keep your EfficientNet core model code but replace the fastai data pipeline/inference with a minimal pure-PyTorch inference path, loading weights if present and otherwise falling back to a safe baseline prediction so a valid `submission.csv` is always produced. I also fix the EfficientNet block decoder stride bug (`strides` vs `stride`) and prevent any internet weight डाउनलोड attempts (Kaggle has no internet), which were another likely runtime failure. Finally, I ensure the submission matches the required columns (`id_code,diagnosis`) and writes with a `.csv` suffix.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with an invalid prediction pipeline for this metric: the model outputs a single unconstrained regression logit with random weights (no local checkpoint found), then fixed thresholds map almost everything to the same class, yielding near-random kappa. To move the score upward toward the target without changing the model architecture or training loop, I (1) ensure we actually use compatible local pretrained weights if present (search common Kaggle input locations), and (2) if no weights exist, fit only the four rounding thresholds on a small internal validation split using your existing `OptimizedRounder` logic (no model training) so predictions are better calibrated for quadratic kappa. I also switch `IMG_SIZE` to EfficientNet-B5’s native resolution (456) only when weights are loaded (otherwise keep 224 for speed), because mismatched resolution hurts when using real B5 weights. These are minimal, metric-aligned changes that preserve your core model and inference semantics while making the submission meaningfully better than random.'
- What this solution (achieved 0.02917) has done: 'Your 0.0 score is most consistent with the inference producing essentially constant/poorly-calibrated classes because the model is almost certainly running with random weights (no real checkpoint loaded) and the fallback “calibration” currently has a label/path mismatch bug. I (1) fix the calibration split so it uses real `id_code` and correct image paths (so validation kappa and thresholds are meaningful), and (2) if no weights are found, replace the regression-head random model output with a minimal, legitimate baseline that predicts the training label prior distribution (this usually yields a non-trivial kappa vs near-random). If weights are successfully loaded, behavior stays essentially the same (only negligible differences), and we keep default thresholds. These changes preserve your EfficientNet architecture and inference loop and should move your score upward toward the target band from 0.0.'
- What this solution (achieved 0.0) has done: 'Your current low kappa is mainly because when no weights are found you fall back to *randomly sampling* from the label prior, which is near-random agreement; switching that fallback to a deterministic “most probable class” (mode) is a minimal, legitimate change that typically yields a noticeably higher quadratic kappa. Additionally, when weights are loaded you currently ignore the `coefficients` argument and always use `DEFAULT_COEFS`; I fix that so calibrated thresholds can be applied consistently (without changing the model). Finally, when weights are *not* loaded, we can still use your existing threshold calibration code path to choose thresholds for the raw model outputs; but since the model is random, we instead keep things stable by using the deterministic prior baseline only (no extra approximations/training). These are minimal changes aimed at improving score toward the target while preserving your core architecture and inference loop.'
- What this solution (achieved 0.03321) has done: 'Your 0.0 score strongly suggests the model is being evaluated with random/untrained weights (no checkpoint loaded), and the current fallback (constant mode class) typically yields very low quadratic kappa. To move upward toward the target with minimal change, I keep your EfficientNet and inference pipeline intact but replace the no-weights fallback with a deterministic, metric-aligned baseline: assign labels by matching the training label distribution (quantile/binning over the ordered test ids) rather than predicting a single class. I also make the weight-loading check more reliable by accepting checkpoints that don’t include `_fc.` (common when saved from different wrappers) as long as the backbone loads meaningfully, which increases the chance real weights are used if present. Finally, I ensure the chosen coefficients are actually used (and optionally calibrated only when weights are loaded and you explicitly enable it), keeping runtime within limits and producing a valid `submission.csv`.'
- What this solution (achieved 0.03321) has done: 'Your current score (0.03321) is far below the target (0.91137), and the main blocker is that you are almost certainly scoring with the no-weights fallback, which cannot reach the target. The most minimal, directly score-relevant change is to make weight loading actually succeed when a common checkpoint format is found: (1) handle nested keys like `model.*`, `net.*`, `state_dict.*`, and strip common prefixes, and (2) accept checkpoints where the regression head is named `fc.*` (not just `_fc.*`). If weights still can’t be found, your fallback remains unchanged to ensure a valid submission is produced. These changes preserve your model architecture and inference logic; they only improve the probability that the intended trained weights are used, which is the only plausible path toward the target score.'
- What this solution (achieved 0.03321) has done: 'Your score is extremely far below the target, and with this exact code the only plausible reason is that `WEIGHTS_LOADED` is still `False` on Kaggle and you’re submitting the label-prior baseline (which can’t approach ~0.91 kappa). I make the smallest changes that directly increase the chance the intended trained weights actually load: broaden checkpoint discovery to include `.ckpt`, handle deeper nesting (`ema_state_dict`, `model_ema`, `student/teacher`), and strip additional common prefixes so keys match your `EfficientNet` module names. I also tighten the “acceptance” rule to require a high shape-matched load fraction (and not rely on `fc` presence, since your head is `_fc`), preventing false positives that silently keep you on bad weights. If weights still can’t be found, the fallback remains unchanged to guarantee a valid `submission.csv`.'
- What this solution (achieved 0.03321) has done: 'I make the smallest change that can materially move your score toward the target: fix the calibration gating bug so that when weights are loaded, you actually fit kappa-optimal rounding thresholds on a small internal validation split (instead of always using DEFAULT_COEFS). This preserves your core EfficientNet model and inference loop, but aligns post-processing with the QWK metric, which typically gives a large boost for this competition. I also make the validation split deterministic-but-representative by stratifying on `diagnosis`, which improves threshold stability without changing the model/training. If no weights are loaded, behavior remains the same prior-distribution fallback to ensure a valid submission is always produced.'
- What this solution (achieved 0.03321) has done: 'Your current score strongly indicates you are still not loading any real trained weights and are submitting the deterministic label-prior baseline, which cannot get near the 0.91 target. The smallest change that can move the score meaningfully toward the target is to (1) make checkpoint loading succeed for common EfficientNet wrappers by also remapping `fc.*` → `_fc.*` and accepting high-but-not-perfect matches (since heads often differ), and (2) when weights are loaded, calibrate thresholds on a deterministic stratified validation split as you already intend (keeping your model/inference loop unchanged). If weights still can’t be found, the fallback remains exactly as-is to guarantee a valid submission. These changes are directly score-relevant and do not alter your architecture, training approach, or loss.'
- What this solution (achieved 0.03321) has done: 'Your score is far below the target, so the main objective is to actually get you onto a real trained checkpoint (your current baseline/fallback can’t reach ~0.91 QWK). I make minimal, directly score-relevant changes to broaden checkpoint discovery and improve state-dict key normalization so EfficientNet-B5 weights saved under common wrappers (timm/lightning/fastai-style prefixes, `features.*`, `classifier.*`, etc.) successfully load into your unchanged model. I also slightly relax the acceptance threshold and report the match ratio so we don’t incorrectly reject a good checkpoint due to a head mismatch, and keep your existing threshold calibration (only when weights are loaded) to better align with QWK. Everything else (model definition, inference loop, submission format/paths) stays the same.'
- What this solution (achieved 0.03321) has done: 'Your current score is far below target, and with this code the only realistic way to move toward ~0.91 is to ensure you actually load a real trained checkpoint and then apply your existing threshold calibration. I make minimal, directly score-relevant changes to (1) fix a critical EfficientNet stride bug in the block decoder (it currently decodes `s22` as stride `[2]` instead of `[2,2]`, which prevents compatibility with most saved checkpoints), and (2) slightly improve checkpoint key normalization for common wrappers so more real weights load successfully. Everything else (model definition, inference loop, calibration method, submission format/paths) stays the same so semantics are preserved and runtime remains reasonable.'
- What this solution (achieved 0.03321) has done: 'Your current score (0.03321) indicates you’re still almost certainly not loading a real trained checkpoint, so the biggest lever (without changing the model/training core logic) is to make checkpoint loading succeed when a compatible file exists in `../input`. I make two minimal, score-relevant fixes: (1) correct a critical bug in `Conv2dStaticSamePadding` where `pad_w` mistakenly uses `kh` instead of `kw` (this can break compatibility/performance vs the original EfficientNet implementation), and (2) slightly relax and improve the checkpoint loader so it can accept “backbone-only” matches (common when the head differs) by loading the best shape-matched subset directly into the model state dict. If no weights are present, behavior remains unchanged (your deterministic label-prior fallback still writes a valid submission). These changes preserve your architecture, inference loop, and metric semantics, but increase the probability you actually run with meaningful weights and thus move toward the target.'

# 9. Code solution

## === cell 0
import os, sys, math, re, json, collections, glob
from functools import partial

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from sklearn import metrics
from sklearn.metrics import cohen_kappa_score

torch.manual_seed(42)
np.random.seed(42)

BASE_DIR = "../input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)



## === cell 1
"""
This file contains helper functions for building the model and for loading model parameters.
These helper functions are built to mirror those in the official TensorFlow implementation.
"""

GlobalParams = collections.namedtuple(
    "GlobalParams",
    [
        "batch_norm_momentum",
        "batch_norm_epsilon",
        "dropout_rate",
        "num_classes",
        "width_coefficient",
        "depth_coefficient",
        "depth_divisor",
        "min_depth",
        "drop_connect_rate",
        "image_size",
    ],
)

BlockArgs = collections.namedtuple(
    "BlockArgs",
    [
        "kernel_size",
        "num_repeat",
        "input_filters",
        "output_filters",
        "expand_ratio",
        "id_skip",
        "stride",
        "se_ratio",
    ],
)

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def relu_fn(x):
    """Swish activation function"""
    return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    """Calculate and round number of filters based on depth multiplier."""
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """Drop connect."""
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand(
        [batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device
    )
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise."""
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(
        self, in_channels, out_channels, kernel_size, image_size=None, **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            )
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x


def efficientnet_params(model_name):
    """Map EfficientNet model name to parameter coefficients."""
    params_dict = {
        "efficientnet-b0": (1.0, 1.0, 224, 0.2),
        "efficientnet-b1": (1.0, 1.1, 240, 0.2),
        "efficientnet-b2": (1.1, 1.2, 260, 0.3),
        "efficientnet-b3": (1.2, 1.4, 300, 0.3),
        "efficientnet-b4": (1.4, 1.8, 380, 0.4),
        "efficientnet-b5": (1.6, 2.2, 456, 0.4),
        "efficientnet-b6": (1.8, 2.6, 528, 0.5),
        "efficientnet-b7": (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """Block Decoder for readability, straight from the official TensorFlow repository"""

    @staticmethod
    def _decode_block_string(block_string):
        assert isinstance(block_string, str)

        ops = block_string.split("_")
        options = {}
        for op in ops:
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        assert ("s" in options and len(options["s"]) == 1) or (
            len(options["s"]) == 2 and options["s"][0] == options["s"][1]
        )
        if len(options["s"]) == 1:
            stride = [int(options["s"][0]), int(options["s"][0])]
        else:
            stride = [int(options["s"][0]), int(options["s"][1])]

        return BlockArgs(
            kernel_size=int(options["k"]),
            num_repeat=int(options["r"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            expand_ratio=int(options["e"]),
            id_skip=("noskip" not in block_string),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=stride,
        )

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        s = block.stride
        s0 = s[0] if isinstance(s, (list, tuple)) else s
        s1 = s[1] if isinstance(s, (list, tuple)) and len(s) > 1 else s0
        args = [
            "r%d" % block.num_repeat,
            "k%d" % block.kernel_size,
            "s%d%d" % (s0, s1),
            "e%s" % block.expand_ratio,
            "i%d" % block.input_filters,
            "o%d" % block.output_filters,
        ]
        if 0 < (block.se_ratio or 0) <= 1:
            args.append("se%s" % block.se_ratio)
        if block.id_skip is False:
            args.append("noskip")
        return "_".join(args)

    @staticmethod
    def decode(string_list):
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet(
    width_coefficient=None,
    depth_coefficient=None,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    image_size=None,
    num_classes=1000,
):
    """Creates a efficientnet model."""
    blocks_args = [
        "r1_k3_s11_e1_i32_o16_se0.25",
        "r2_k3_s22_e6_i16_o24_se0.25",
        "r2_k5_s22_e6_i24_o40_se0.25",
        "r3_k3_s22_e6_i40_o80_se0.25",
        "r3_k5_s11_e6_i80_o112_se0.25",
        "r4_k5_s22_e6_i112_o192_se0.25",
        "r1_k3_s11_e6_i192_o320_se0.25",
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )
    return blocks_args, global_params


def get_model_params(model_name, override_params):
    if model_name.startswith("efficientnet"):
        w, d, s, p = efficientnet_params(model_name)
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s
        )
    else:
        raise NotImplementedError("model name is not pre-defined: %s" % model_name)
    if override_params:
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


class MBConvBlock(nn.Module):
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
            0 < self._block_args.se_ratio <= 1
        )
        self.id_skip = block_args.id_skip

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False
            )
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
            )

        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,
            kernel_size=k,
            stride=s,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
        )

        if self.has_se:
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            self._se_reduce = Conv2d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1
            )
            self._se_expand = Conv2d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1
            )

        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps
        )

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        input_filters, output_filters = (
            self._block_args.input_filters,
            self._block_args.output_filters,
        )
        stride = self._block_args.stride
        stride_val = stride[0] if isinstance(stride, (list, tuple)) else stride
        if self.id_skip and stride_val == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x


class EfficientNet(nn.Module):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), "blocks_args should be a list"
        assert len(blocks_args) > 0, "block args must be greater than 0"
        self._global_params = global_params
        self._blocks_args = blocks_args

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        in_channels = 3
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=False
        )
        self._bn0 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )

        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters, self._global_params
                ),
                output_filters=round_filters(
                    block_args.output_filters, self._global_params
                ),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params),
            )

            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )

        self._dropout = self._global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def extract_features(self, inputs):
        x = relu_fn(self._bn0(self._conv_stem(inputs)))
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        x = relu_fn(self._bn1(self._conv_head(x)))
        return x

    def forward(self, inputs):
        x = self.extract_features(inputs)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ["efficientnet_b" + str(i) for i in range(num_models)]
        if model_name.replace("-", "_") not in valid_models:
            raise ValueError("model_name should be one of: " + ", ".join(valid_models))




## === cell 2
md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
md_ef = md_ef.to(DEVICE)
md_ef.eval()


def _discover_weight_candidates():
    patterns = [
        "../input/**/*.pth",
        "../input/**/*.pt",
        "../input/**/*.bin",
        "../input/**/*.ckpt",
        "../input/**/*.pth.tar",
        "../input/**/*.pkl",
        "../working/**/*.pth",
        "../working/**/*.pt",
        "../working/**/*.ckpt",
        "../working/**/*.pth.tar",
        "../working/**/*.pkl",
    ]
    found = []
    for pat in patterns:
        found.extend(glob.glob(pat, recursive=True))

    preferred = []
    for p in found:
        lp = os.path.basename(p).lower()
        if any(
            k in lp
            for k in [
                "eff",
                "efficient",
                "b5",
                "aptos",
                "blind",
                "retina",
                "model",
                "ckpt",
                "checkpoint",
                "best",
                "fold",
                "ema",
                "final",
            ]
        ):
            preferred.append(p)

    allc = preferred + found
    dedup = []
    seen = set()
    for p in allc:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup[:120]


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
            "ema_state_dict",
            "model_ema",
            "ema",
            "student",
            "teacher",
            "model_state",
            "model_dict",
        ]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                inner = ckpt_obj[key]
                for key2 in [
                    "state_dict",
                    "model_state_dict",
                    "model",
                    "net",
                    "weights",
                    "params",
                ]:
                    if key2 in inner and isinstance(inner[key2], dict):
                        return inner[key2]
                return inner
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return ckpt_obj
    return None


def _normalize_state_keys(state):
    out = {}
    for k, v in state.items():
        nk = k

        for pref in [
            "module.",
            "model.",
            "net.",
            "encoder.",
            "backbone.",
            "student.",
            "teacher.",
            "ema.",
            "model_ema.",
            "learner.model.",
            "model.module.",
        ]:
            if nk.startswith(pref):
                nk = nk[len(pref) :]

        if nk.startswith("classifier."):
            nk = nk[len("classifier.") :]
        if nk.startswith("head."):
            nk = nk[len("head.") :]
        if nk.startswith("features."):
            nk = nk[len("features.") :]

        if nk.startswith("fc."):
            nk = "_fc." + nk[len("fc.") :]
        if nk.startswith("classifier."):
            nk = "_fc." + nk[len("classifier.") :]
        if nk.startswith("_classifier."):
            nk = "_fc." + nk[len("_classifier.") :]
        if nk.startswith("last_linear."):
            nk = "_fc." + nk[len("last_linear.") :]

        out[nk] = v
    return out


def _try_load_weights(model, candidates):
    model_sd = model.state_dict()
    best = (
        0.0,
        None,
        None,
        0,
        0,
    )  # loaded_frac, path, new_state, loaded_cnt, total_cnt
    for p in candidates:
        if p and os.path.isfile(p):
            try:
                ckpt = torch.load(p, map_location="cpu")
            except Exception:
                continue

            state = _extract_state_dict(ckpt)
            if not isinstance(state, dict) or len(state) == 0:
                continue
            new_state = _normalize_state_keys(state)

            total = len(model_sd)
            loaded = 0
            for k, v in new_state.items():
                if (
                    (k in model_sd)
                    and hasattr(v, "shape")
                    and (tuple(model_sd[k].shape) == tuple(v.shape))
                ):
                    loaded += 1
            loaded_frac = loaded / max(1, total)

            if loaded_frac > best[0]:
                best = (loaded_frac, p, new_state, loaded, total)

    if best[1] is not None and best[0] >= 0.50:
        p = best[1]
        new_state = best[2]
        filtered = {}
        for k, v in new_state.items():
            if (
                (k in model_sd)
                and hasattr(v, "shape")
                and (tuple(model_sd[k].shape) == tuple(v.shape))
            ):
                filtered[k] = v
        model_sd.update(filtered)
        model.load_state_dict(model_sd, strict=False)
        print(f"Loaded (filtered) weights from: {p}")
        print(
            f"Shape-matched keys loaded: {best[3]}/{best[4]} ({best[0]:.2%}); "
            f"Filtered keys applied: {len(filtered)}"
        )
        return True

    if best[1] is not None:
        print(
            f"Found candidate weights but match too low ({best[0]:.2%}) at: {best[1]} "
            f"=> not using to avoid poor/invalid inference."
        )

    print(
        "No compatible local weights found; will use a distribution-matching baseline (deterministic) to avoid near-zero kappa."
    )
    return False


weight_candidates = [
    "models/abcdef.pth",
    "models/abcdef",
    "../input/kaggle-public/abcdef.pth",
    "../input/kaggle-public/abcdef",
    "../input/abcdef.pth",
    "../input/abcdef",
] + _discover_weight_candidates()

WEIGHTS_LOADED = _try_load_weights(md_ef, weight_candidates)




## === cell 3
def get_df():
    df = pd.read_csv(TRAIN_CSV)
    df["path"] = df["id_code"].map(lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.png"))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    test_ids = pd.read_csv(TEST_CSV)
    test_ids["path"] = test_ids["id_code"].map(
        lambda x: os.path.join(TEST_IMG_DIR, f"{x}.png")
    )
    return df, test_ids


df, test_df = get_df()
print(df.shape, test_df.shape)
print(test_df.head())



## === cell 4
IMG_SIZE = 456 if WEIGHTS_LOADED else 224

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_rgb(path, img_size):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(arr)


class TestDataset(Dataset):
    def __init__(self, df, img_size):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "path"]
        x = load_image_rgb(path, self.img_size)
        return x, self.df.loc[idx, "id_code"]




## === cell 5
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        ll = metrics.cohen_kappa_score(y, X_p.astype(int), weights="quadratic")
        return -ll

    def predict(self, X, coef):
        X = np.asarray(X).reshape(-1)
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p.astype(int)


DEFAULT_COEFS = [0.5, 1.5, 2.5, 3.5]


def fit_thresholds_coord_descent(raw_preds, y_true, init=None, n_iter=25):
    coef = np.array(DEFAULT_COEFS if init is None else init, dtype=np.float32)
    coef.sort()
    best = -cohen_kappa_score(
        y_true, OptimizedRounder().predict(raw_preds, coef), weights="quadratic"
    )

    rp = np.asarray(raw_preds).reshape(-1)
    lo, hi = np.percentile(rp, 1), np.percentile(rp, 99)
    span = max(1e-3, float(hi - lo))

    steps = [0.25, 0.1, 0.05, 0.02]
    for _ in range(n_iter):
        improved = False
        for j in range(4):
            base = coef[j]
            for step_frac in steps:
                step = step_frac * span
                for direction in (-1.0, 1.0):
                    trial = coef.copy()
                    trial[j] = base + direction * step
                    trial.sort()
                    if not np.all(np.diff(trial) > 1e-4):
                        continue
                    kappa = cohen_kappa_score(
                        y_true,
                        OptimizedRounder().predict(rp, trial),
                        weights="quadratic",
                    )
                    score = -kappa
                    if score < best:
                        best = score
                        coef = trial
                        improved = True
        if not improved:
            break
    return coef.tolist()




## === cell 6
@torch.no_grad()
def predict_test(model, test_df, batch_size=16, img_size=224):
    ds = TestDataset(test_df[["id_code", "path"]], img_size=img_size)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    preds = []
    ids = []
    model.eval()
    for xb, idb in dl:
        xb = xb.to(DEVICE, non_blocking=True)
        out = model(xb).float().detach().cpu().numpy().reshape(-1)
        preds.append(out)
        ids.extend(list(idb))
    preds = np.concatenate(preds, axis=0)
    return np.array(ids), preds


def _stratified_val_split(df_train, n_val, seed=42):
    parts = []
    for cls, grp in df_train.groupby("diagnosis"):
        g = grp.sample(frac=1, random_state=seed).reset_index(drop=True)
        k = int(round(n_val * (len(g) / len(df_train))))
        k = max(1, min(len(g), k))
        parts.append(g.iloc[:k])
    val = (
        pd.concat(parts, axis=0)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    if len(val) > n_val:
        val = val.iloc[:n_val].copy()
    return val


def _get_calibration_coefs_if_needed(model, df_train, img_size, batch_size=16):
    if not WEIGHTS_LOADED:
        return DEFAULT_COEFS

    n = len(df_train)
    n_val = max(400, int(0.2 * n))
    val = _stratified_val_split(df_train, n_val=n_val, seed=42)

    val_df = val[["id_code", "path"]].copy()
    y_val = val["diagnosis"].values.astype(int)

    _, raw_val = predict_test(model, val_df, batch_size=batch_size, img_size=img_size)
    raw_val = np.clip(raw_val, -1.0, 5.0)

    coefs = fit_thresholds_coord_descent(raw_val, y_val, init=DEFAULT_COEFS, n_iter=30)
    pred_val = OptimizedRounder().predict(raw_val, coefs)
    kappa = cohen_kappa_score(y_val, pred_val, weights="quadratic")
    print("Calibrated coefs (weights loaded):", coefs, "val kappa:", float(kappa))
    return coefs


def _predict_from_label_prior(test_df, train_df):
    counts = train_df["diagnosis"].value_counts().sort_index()
    n = len(test_df)

    probs = (counts / counts.sum()).reindex([0, 1, 2, 3, 4]).fillna(0.0).values
    raw = probs * n
    alloc = np.floor(raw).astype(int)
    remainder = n - alloc.sum()
    if remainder > 0:
        frac = raw - np.floor(raw)
        for idx in np.argsort(-frac)[:remainder]:
            alloc[idx] += 1
    elif remainder < 0:
        for idx in np.argsort(raw)[:(-remainder)]:
            if alloc[idx] > 0:
                alloc[idx] -= 1

    labels = []
    for cls, k in enumerate(alloc.tolist()):
        labels.extend([cls] * k)
    labels = np.array(labels[:n], dtype=int)

    ids = test_df["id_code"].values.astype(str)
    return ids, labels




## === cell 7
def run_subm(model, test_df, coefficients=DEFAULT_COEFS, out_path="submission.csv"):
    if WEIGHTS_LOADED:
        ids, raw_preds = predict_test(model, test_df, batch_size=16, img_size=IMG_SIZE)
        raw_preds = np.clip(raw_preds, -1.0, 5.0)
        opt = OptimizedRounder()
        diag = opt.predict(raw_preds, coefficients)
        sub = pd.DataFrame({"id_code": ids, "diagnosis": diag.astype(int)})
    else:
        ids, diag = _predict_from_label_prior(test_df, df)
        sub = pd.DataFrame({"id_code": ids, "diagnosis": diag.astype(int)})

    test_ids = pd.read_csv(TEST_CSV)["id_code"].tolist()
    sub["id_code"] = pd.Categorical(sub["id_code"], categories=test_ids, ordered=True)
    sub = sub.sort_values("id_code").reset_index(drop=True)
    sub["id_code"] = sub["id_code"].astype(str)

    sub.to_csv(out_path, index=False)
    print(
        "Wrote:",
        out_path,
        "rows:",
        len(sub),
        "unique diagnoses:",
        sub["diagnosis"].value_counts().to_dict(),
        "WEIGHTS_LOADED:",
        WEIGHTS_LOADED,
        "IMG_SIZE:",
        IMG_SIZE,
    )
    return sub


coeffs_to_use = _get_calibration_coefs_if_needed(
    md_ef, df, img_size=IMG_SIZE, batch_size=16
)
submission = run_subm(
    md_ef, test_df, coefficients=coeffs_to_use, out_path="submission.csv"
)
submission.head()



## === cell 8
assert os.path.isfile("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["id_code", "diagnosis"]
assert len(chk) == len(pd.read_csv(TEST_CSV))
assert chk["diagnosis"].between(0, 4).all()
print("Submission looks valid.")
