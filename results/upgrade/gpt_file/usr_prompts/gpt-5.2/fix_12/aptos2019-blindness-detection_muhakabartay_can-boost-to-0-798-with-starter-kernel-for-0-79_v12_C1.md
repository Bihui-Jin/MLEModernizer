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

0.9080228716816332

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `fastai` (it’s not available in your environment) while keeping the core approach: EfficientNet-B5 + pretrained weights + regression output + fixed threshold rounding into 0–4 classes. I also fix missing imports (`os`, `collections`) and eliminate notebook magics/`!cp` commands that break in a plain Kaggle Python run. Because internet is unavailable, I load weights from a local path if present; otherwise the script still run end-to-end with random weights and produce a valid `submission.csv`. Finally, I implement a minimal PyTorch inference pipeline that reads images, applies ImageNet normalization, runs the model, rounds via the provided coefficients, and writes the submission with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from using randomly initialized weights (no `abcdef.pth` found) and/or a preprocessing mismatch (EfficientNet-B5 expects 456px inputs while your loader uses 224px). To move the score upward toward the 0.908 target without changing the core model or inference semantics, I (1) load ImageNet-pretrained weights from the model’s local cache if available (no internet required) and (2) set inference image size to the native EfficientNet-B5 resolution (456) with the same normalization you already use. I also add a small, safe fallback that searches `/kaggle/input/**` for any `.pth` checkpoint to load (still the same architecture), because using a real competition checkpoint is the only realistic way to approach ~0.9 QWK. These are minimal changes focused solely on getting non-random predictions and matching expected preprocessing while still producing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the model running with random weights because `from_pretrained()` currently does not actually load ImageNet weights (it only builds the architecture), and your `model_zoo.load_url()` path cannot work without internet. I make the smallest change that preserves your architecture and inference semantics: explicitly load a local EfficientNet-B5 ImageNet checkpoint if it exists in the torch cache (and also search common cache locations), and only then fall back to any `.pth` under `/kaggle/input` as you already do. This should move the score upward toward the target by ensuring non-random feature extraction while keeping the same EfficientNet-B5 + regression + fixed thresholds approach. I also keep the submission ordering safeguard and output exactly `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with running EfficientNet-B5 with random weights (no internet download and no local checkpoint found/loaded), which produces near-random class predictions and thus near-zero QWK. To move the score upward toward the 0.908 target while preserving the exact core architecture and inference semantics, I make the weight-loading robust: (1) first try to load a valid local ImageNet EfficientNet-B5 checkpoint from common torch cache locations, (2) then scan `/kaggle/input` for plausible checkpoints and prefer ones that actually match the model keys, and (3) if none are found, fail fast with a clear error instead of silently submitting random predictions. I also ensure the model head shape handling is correct when loading ImageNet weights (drop `_fc.*` safely) and keep preprocessing at 456px + ImageNet normalization as you already do.'
- What this solution (achieved 0.0) has done: 'I remove the hard failure that stops the notebook when no `.pth` checkpoint is found, because that currently prevents producing any valid `submission.csv`. To move the score up from 0.0 toward the target while preserving the same EfficientNet-B5 + 1D regressor + fixed-threshold rounding core logic, I instead (a) try to load any available checkpoint as before, (b) otherwise fall back to loading ImageNet weights via `torch.hub.load_state_dict_from_url` (which uses Kaggle’s persistent cache when available), and only if that also fails (no cache) proceed with random weights but still generate a valid submission. I also fix a small logic bug in `MBConvBlock` where the skip-connection stride check compares a list to an int, which can break correctness and weight compatibility. All paths and submission format remain unchanged, and the script always write `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly due to producing near-constant/near-random predictions from missing/incorrectly-loaded weights, which makes QWK collapse. To move the score upward toward the 0.908 target while preserving your exact core approach (EfficientNet-B5 regressor + fixed thresholds), I make weight loading stricter and safer: (1) correctly adapt ImageNet B5 weights by filtering out the classifier head (so the backbone loads cleanly), and (2) prefer checkpoints that actually match backbone keys and shapes. I also add a hard guard that prevents silently submitting random-weight predictions (the main cause of 0.0), so you only generate a submission when meaningful weights were loaded. All preprocessing, architecture, thresholds, and submission format remain unchanged.'
- What this solution (achieved 0.0) has done: 'I remove the hard failure when no weights are found so the script always runs end-to-end and writes a valid `submission.csv` (your current run stops in cell 2, so you can’t even submit). To still move score upward toward the target when possible without changing the model/inference semantics, I keep the existing checkpoint search and loading logic, but make it more robust by (a) trying to load torchvision’s built-in ImageNet EfficientNet-B5 weights when available offline and (b) otherwise proceeding with the current model weights (with a clear warning). I also fix the EfficientNet block decoding stride to be two-dimensional (`[s, s]`), which improves compatibility with standard checkpoints and correctness while preserving architecture intent. All preprocessing, regression head, and fixed threshold rounding remain unchanged.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with producing effectively random predictions because no meaningful weights were loaded; to move toward the 0.908 target with minimal disruption, the most direct fix is to *stop submitting random-weight outputs* and instead require that at least ImageNet-pretrained weights (cached locally) or a compatible `.pth` checkpoint from `/kaggle/input` is actually loaded. I keep your exact EfficientNet-B5 regressor + fixed-threshold rounding logic, but make weight loading stricter: if no usable weights are found, the script raise with a clear message (so you don’t unknowingly submit a 0.0). I also make the checkpoint scan prefer filenames commonly used in this competition (e.g., containing `b5`, `efficientnet`, `aptos`, `blind`, `kappa`) to increase the chance we pick a good checkpoint when multiple exist, without changing model logic. Finally, I keep preprocessing and submission formatting the same so evaluation semantics are unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from submitting with either random weights or mismatched weights/head, so the smallest change that actually moves toward 0.908 is to (1) keep your EfficientNet-B5 regressor + fixed thresholds exactly as-is, but (2) make checkpoint loading work reliably in this offline Kaggle environment by correctly extracting the backbone from common checkpoint formats and (3) correctly mapping torchvision’s EfficientNet-B5 ImageNet weights (offline) into your custom EfficientNet implementation. This preserves your core logic and inference semantics, but ensures you get meaningful predictions instead of near-random outputs. I also keep the 456px preprocessing and submission alignment, and ensure `submission.csv` is always written when weights load successfully.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from not being able to load a *competition-trained* checkpoint and then refusing to write a submission (or previously submitting random weights). To move the score upward toward the 0.908 target while keeping your exact EfficientNet-B5 regressor + fixed-threshold rounding logic, I (1) make checkpoint discovery/load more robust for common Kaggle checkpoint formats (nested keys like `model_state`, `ema`, `student`, etc.), and (2) make the “best checkpoint” selection prefer *max key/shape matches* and penalize obvious non-model files. I also ensure the script always produces `submission.csv`: if no meaningful checkpoint is found, it fall back to torchvision ImageNet weights (still same architecture/inference), rather than erroring out. These are minimal, directly score-relevant changes focused on getting non-random, correctly-loaded weights and correct inference preprocessing.'
- What this solution (achieved 0.0) has done: 'Your 0.0 QWK strongly indicates the model is still effectively untrained (random or poorly loaded weights), so the smallest score-relevant improvement is to make checkpoint loading actually succeed for common APTOS-trained formats and to ensure the model head matches what most competition checkpoints contain. I keep the same EfficientNet-B5 backbone and the same fixed-threshold rounding into 0–4, but (1) replace the 1-unit regression head with a 5-class classification head (many APTOS checkpoints are 5-way logits), (2) map logits to a scalar severity via expected value (still produces a single continuous score that gets thresholded exactly like before), and (3) improve checkpoint extraction to handle nested keys (e.g., `model`, `model_state_dict`) and prefixes while scoring candidate checkpoints by key/shape match. This should move the score upward toward the 0.908 target when a usable checkpoint exists in `/kaggle/input`, while still running end-to-end and always writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import re
import math
import json
import collections
from functools import partial

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F

from sklearn import metrics
from PIL import Image

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




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
        super(Identity, self).__init__()

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
        pad_w = max((ow - 1) * self.stride[1] + (kh - 1) * self.dilation[1] + 1 - iw, 0)
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
    """Block Decoder for readability"""

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

        s0 = int(options["s"][0])
        return BlockArgs(
            kernel_size=int(options["k"]),
            num_repeat=int(options["r"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            expand_ratio=int(options["e"]),
            id_skip=("noskip" not in block_string),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=[s0, s0],
        )

    @staticmethod
    def decode(string_list):
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args


def efficientnet(
    width_coefficient=None,
    depth_coefficient=None,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    image_size=None,
    num_classes=1000,
):
    """Creates an efficientnet model."""
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


url_map = {
    "efficientnet-b0": "http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth",
    "efficientnet-b1": "http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth",
    "efficientnet-b2": "http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth",
    "efficientnet-b3": "http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth",
    "efficientnet-b4": "http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth",
    "efficientnet-b5": "http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth",
}


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

        stride_is_one = (
            (self._block_args.stride == 1)
            if isinstance(self._block_args.stride, int)
            else (len(self._block_args.stride) > 0 and self._block_args.stride[0] == 1)
        )

        input_filters, output_filters = (
            self._block_args.input_filters,
            self._block_args.output_filters,
        )
        if self.id_skip and stride_is_one and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x


class EfficientNet(nn.Module):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list)
        assert len(blocks_args) > 0
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
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        return model




## === cell 2
md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=5)
md_ef = md_ef.to(device)
md_ef.eval()


def _find_any_pth_under_input(max_hits=400):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        for fn in files:
            lfn = fn.lower()
            if lfn.endswith((".pth", ".pt", ".bin")):
                hits.append(os.path.join(root, fn))
                if len(hits) >= max_hits:
                    return hits
    return hits


def _normalize_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return None
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_state[nk] = v
    return new_state


def _extract_state_dict_container(sd):
    if not isinstance(sd, dict):
        return sd

    for key in [
        "state_dict",
        "model_state_dict",
        "model",
        "model_ema",
        "ema",
        "student",
        "teacher",
        "net",
        "network",
        "weights",
        "params",
        "checkpoint",
    ]:
        if key in sd and isinstance(sd[key], dict):
            return sd[key]

    tensor_like = 0
    for k, v in sd.items():
        if isinstance(k, str) and (hasattr(v, "shape") or torch.is_tensor(v)):
            tensor_like += 1
        if tensor_like >= 10:
            return sd
    return sd


def _filter_backbone_only_for_imagenet(sd):
    if sd is None:
        return None
    sd = dict(sd)
    sd.pop("_fc.weight", None)
    sd.pop("_fc.bias", None)
    return sd


def _score_state_dict_fit(model, state_dict):
    if state_dict is None or not isinstance(state_dict, dict):
        return -1
    model_sd = model.state_dict()
    matched = 0
    for k, v in state_dict.items():
        if k in model_sd and hasattr(v, "shape") and model_sd[k].shape == v.shape:
            matched += 1
    return matched


def _filename_prior(ckpt_path):
    name = os.path.basename(ckpt_path).lower()
    pri = 0
    for token, w in [
        ("efficientnet", 5),
        ("b5", 5),
        ("aptos", 4),
        ("blind", 3),
        ("kappa", 3),
        ("best", 2),
        ("fold", 1),
        ("checkpoint", 1),
    ]:
        if token in name:
            pri += w
    for token, w in [
        ("optimizer", -5),
        ("sched", -3),
        ("scheduler", -3),
        ("adam", -2),
        ("sgd", -2),
    ]:
        if token in name:
            pri += w
    return pri


def _try_load_state_dict_into_b5_classifier(model, state_dict, is_imagenet=False):
    state_dict = _extract_state_dict_container(state_dict)
    state_dict = _normalize_state_dict_keys(state_dict)
    if state_dict is None:
        return False, "state_dict not loadable"

    if is_imagenet:
        state_dict = _filter_backbone_only_for_imagenet(state_dict)

    if (
        "_fc.weight" in state_dict
        and state_dict["_fc.weight"].shape != model._fc.weight.shape
    ):
        state_dict.pop("_fc.weight", None)
        state_dict.pop("_fc.bias", None)

    try:
        res = model.load_state_dict(state_dict, strict=False)
        return (
            True,
            f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}",
        )
    except Exception as e:
        return False, str(e)


def _candidate_imagenet_b5_paths():
    cached_name = os.path.basename(url_map["efficientnet-b5"])
    candidates = [
        os.path.expanduser(
            os.path.join("~", ".cache", "torch", "checkpoints", cached_name)
        ),
        os.path.expanduser(
            os.path.join("~", ".cache", "torch", "hub", "checkpoints", cached_name)
        ),
        os.path.join("/root", ".cache", "torch", "checkpoints", cached_name),
        os.path.join("/root", ".cache", "torch", "hub", "checkpoints", cached_name),
        os.path.join("/kaggle", "working", cached_name),
        os.path.join("/kaggle", "temp", cached_name),
    ]
    return candidates


loaded_any = False
load_messages = []

for p in _candidate_imagenet_b5_paths():
    if os.path.exists(p):
        try:
            sd = torch.load(p, map_location="cpu")
            ok, msg = _try_load_state_dict_into_b5_classifier(
                md_ef, sd, is_imagenet=True
            )
            if ok:
                load_messages.append(
                    f"Loaded cached ImageNet weights from: {p} ({msg})"
                )
                loaded_any = True
                break
        except Exception as e:
            load_messages.append(
                f"WARNING: Failed loading cached weights from {p}: {e}"
            )

if not loaded_any:
    all_ckpts = [
        "/kaggle/input/abcdef.pth",
        "/kaggle/input/kaggle-public/abcdef.pth",
        "/kaggle/input/aptos2019-blindness-detection/abcdef.pth",
        "/kaggle/working/models/abcdef.pth",
        "/kaggle/working/abcdef.pth",
    ]
    all_ckpts.extend(_find_any_pth_under_input(max_hits=400))

    best_path, best_score, best_prior = None, -1, -(10**9)
    for ckpt_path in all_ckpts:
        if not os.path.exists(ckpt_path):
            continue
        try:
            sd_raw = torch.load(ckpt_path, map_location="cpu")
            sd = _extract_state_dict_container(sd_raw)
            nsd = _normalize_state_dict_keys(sd)
            sc = _score_state_dict_fit(md_ef, nsd if nsd is not None else {})
            pri = _filename_prior(ckpt_path)
            if (sc > best_score) or (sc == best_score and pri > best_prior):
                best_score = sc
                best_path = ckpt_path
                best_prior = pri
        except Exception:
            continue

    if best_path is not None and best_score > 0:
        try:
            sd = torch.load(best_path, map_location="cpu")
            ok, msg = _try_load_state_dict_into_b5_classifier(
                md_ef, sd, is_imagenet=False
            )
            if ok:
                load_messages.append(
                    f"Loaded best-matching checkpoint: {best_path} (match_keys={best_score}, name_prior={best_prior}; {msg})"
                )
                loaded_any = True
        except Exception as e:
            load_messages.append(
                f"WARNING: Found checkpoint but failed to load ({best_path}): {e}"
            )


def _try_load_torchvision_efficientnet_b5_into_custom(model):
    try:
        from torchvision.models import efficientnet_b5

        try:
            from torchvision.models import EfficientNet_B5_Weights

            tv = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        except Exception:
            tv = efficientnet_b5(pretrained=True)
        tv_sd = tv.state_dict()
    except Exception as e:
        return (
            False,
            f"torchvision not available or cannot create pretrained efficientnet_b5: {e}",
        )

    custom_sd = model.state_dict()
    mapped = {}
    matched = 0

    def _maybe(dst, src):
        nonlocal matched
        if (
            src in tv_sd
            and dst in custom_sd
            and tv_sd[src].shape == custom_sd[dst].shape
        ):
            mapped[dst] = tv_sd[src]
            matched += 1

    _maybe("_conv_stem.weight", "features.0.0.weight")
    for bn_field, tv_key in [
        ("weight", "features.0.1.weight"),
        ("bias", "features.0.1.bias"),
        ("running_mean", "features.0.1.running_mean"),
        ("running_var", "features.0.1.running_var"),
    ]:
        _maybe(f"_bn0.{bn_field}", tv_key)

    block_idx = 0
    for stage in range(1, 8):
        stage_prefix = f"features.{stage}."
        block_ids = set()
        for k in tv_sd.keys():
            if k.startswith(stage_prefix):
                rest = k[len(stage_prefix) :]
                b = rest.split(".", 1)[0]
                if b.isdigit():
                    block_ids.add(int(b))
        for b in sorted(block_ids):
            base = f"{stage_prefix}{b}."
            exp_w_key = base + "block.0.0.weight"
            has_expand = (
                exp_w_key in tv_sd
                and f"_blocks.{block_idx}._expand_conv.weight" in custom_sd
            )

            if has_expand:
                _maybe(f"_blocks.{block_idx}._expand_conv.weight", exp_w_key)
                for bn_field in ["weight", "bias", "running_mean", "running_var"]:
                    _maybe(
                        f"_blocks.{block_idx}._bn0.{bn_field}",
                        base + f"block.0.1.{bn_field}",
                    )
                dw_w = base + "block.1.0.weight"
                dw_bn = base + "block.1.1."
                se_base = base + "block.2."
                proj_w = base + "block.3.0.weight"
                proj_bn = base + "block.3.1."
            else:
                dw_w = base + "block.0.0.weight"
                dw_bn = base + "block.0.1."
                se_base = base + "block.1."
                proj_w = base + "block.2.0.weight"
                proj_bn = base + "block.2.1."

            _maybe(f"_blocks.{block_idx}._depthwise_conv.weight", dw_w)
            for bn_field in ["weight", "bias", "running_mean", "running_var"]:
                _maybe(f"_blocks.{block_idx}._bn1.{bn_field}", dw_bn + bn_field)

            _maybe(f"_blocks.{block_idx}._se_reduce.weight", se_base + "fc1.weight")
            _maybe(f"_blocks.{block_idx}._se_reduce.bias", se_base + "fc1.bias")
            _maybe(f"_blocks.{block_idx}._se_expand.weight", se_base + "fc2.weight")
            _maybe(f"_blocks.{block_idx}._se_expand.bias", se_base + "fc2.bias")

            _maybe(f"_blocks.{block_idx}._project_conv.weight", proj_w)
            for bn_field in ["weight", "bias", "running_mean", "running_var"]:
                _maybe(f"_blocks.{block_idx}._bn2.{bn_field}", proj_bn + bn_field)

            block_idx += 1
            if block_idx >= len(model._blocks):
                break
        if block_idx >= len(model._blocks):
            break

    _maybe("_conv_head.weight", "features.8.0.weight")
    for bn_field, tv_key in [
        ("weight", "features.8.1.weight"),
        ("bias", "features.8.1.bias"),
        ("running_mean", "features.8.1.running_mean"),
        ("running_var", "features.8.1.running_var"),
    ]:
        _maybe(f"_bn1.{bn_field}", tv_key)

    if matched == 0:
        return False, "no matching keys found between torchvision and custom model"
    model.load_state_dict(mapped, strict=False)
    return True, f"loaded {matched} tensors from torchvision ImageNet weights"


if not loaded_any:
    ok, msg = _try_load_torchvision_efficientnet_b5_into_custom(md_ef)
    if ok:
        load_messages.append(
            f"Loaded torchvision ImageNet EfficientNet-B5 weights ({msg})"
        )
        loaded_any = True
    else:
        load_messages.append(
            f"WARNING: torchvision ImageNet weight load not available: {msg}"
        )

for m in load_messages:
    print(m)

if not loaded_any:
    print(
        "WARNING: No usable weights found (cached ImageNet / checkpoints under /kaggle/input). "
        "Proceeding with randomly initialized weights will likely yield ~0.0 QWK."
    )




## === cell 3
BASE = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

test_df = pd.read_csv(TEST_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)
sub_df = sub_df[["id_code", "diagnosis"]].copy()
test_df = test_df[["id_code"]].copy()




## === cell 4
IM_SIZE = 456
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_tensor(png_path, size=IM_SIZE):
    img = Image.open(png_path).convert("RGB")
    img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - IM_MEAN) / IM_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(arr)




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

        ll = metrics.cohen_kappa_score(y, X_p, weights="quadratic")
        return -ll

    def predict(self, X, coef):
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
        return X_p




## === cell 6
def run_subm(
    model,
    test_ids,
    coefficients=(0.6, 1.6, 2.6, 3.6),
    batch_size=8,
    out_path="submission.csv",
):
    model.eval()
    preds = []

    sev_levels = torch.arange(5, device=device, dtype=torch.float32).view(1, 5)

    with torch.no_grad():
        for i in range(0, len(test_ids), batch_size):
            batch_ids = test_ids[i : i + batch_size]
            batch_tensors = []
            for id_code in batch_ids:
                img_path = os.path.join(TEST_IMG_DIR, f"{id_code}.png")
                batch_tensors.append(load_image_tensor(img_path))
            x = torch.stack(batch_tensors, dim=0).to(device)

            logits = model(x)
            prob = torch.softmax(logits.float(), dim=1)
            y = (prob * sev_levels).sum(dim=1).detach().cpu().numpy()
            preds.append(y)

    preds = np.concatenate(preds, axis=0)

    opt = OptimizedRounder()
    diag = opt.predict(preds, np.array(coefficients, dtype=np.float32)).astype(int)

    submission = pd.DataFrame({"id_code": test_ids, "diagnosis": diag})
    if "id_code" in sub_df.columns and len(sub_df) == len(submission):
        submission = (
            submission.set_index("id_code").loc[sub_df["id_code"]].reset_index()
        )

    submission.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} with shape {submission.shape} and columns {list(submission.columns)}"
    )
    return submission




## === cell 7
test_ids = test_df["id_code"].tolist()
_ = run_subm(
    md_ef,
    test_ids,
    coefficients=(0.6, 1.6, 2.6, 3.6),
    batch_size=8,
    out_path="submission.csv",
)
