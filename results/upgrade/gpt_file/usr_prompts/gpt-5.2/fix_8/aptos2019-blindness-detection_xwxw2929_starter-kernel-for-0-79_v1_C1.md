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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `fastai` (which is not available in your environment) and replace the data pipeline + `Learner` usage with a minimal PyTorch-only inference pipeline while preserving the core EfficientNet model definition you already have. I also fix missing imports (`os`, `collections`) and remove the internet weight download by loading weights from a local `.pth` if present (otherwise it still run with random weights and produce a valid submission). Finally, I fix the submission creation logic: use `test.csv` ids, align predictions to `id_code`, and always write `submission.csv` with the required columns and integer classes 0–4.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with running EfficientNet with random weights (no local checkpoint found), so the smallest legitimate move toward the 0.911 target is to actually load a real pretrained EfficientNet-B5 backbone without changing your architecture or inference pipeline. I keep your exact model definition and regression head, but add a local, offline weight loader that searches common Kaggle locations for an EfficientNet-B5 ImageNet `.pth` and loads only matching keys (ignoring the classifier head). I also add a safe fallback path search for the competition dataset root so image paths resolve reliably, preventing silent misalignment or file-not-found issues. Submission writing stays identical (same columns, same ids, integer 0–4).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the model is effectively untrained (random weights) and/or because the input resolution is mismatched for EfficientNet-B5, so predictions are near-random even though the submission is valid. To move toward the 0.911 target with minimal changes, I keep your exact EfficientNet-B5 definition and regression head, but (1) make the weight loader actually find common local pretrained EfficientNet-B5 checkpoints by searching `/kaggle/input/**` recursively, and (2) set inference resize to the model’s native 456 resolution (matching the `image_size` embedded in your EfficientNet-B5 global params) to avoid a large distribution shift. I also add a lightweight sanity check that all test image paths exist to prevent silent failures/misalignment. Submission formatting remains identical (`id_code`, `diagnosis`, integer 0–4) and writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly indicates the model is producing near-random predictions because it never loads a useful checkpoint (only a backbone ImageNet checkpoint is searched, and even that likely isn’t found in your environment). To move toward the 0.911 target with minimal, semantics-preserving changes, I (1) broaden the offline checkpoint search to also find *APTOS fine-tuned* `.pth` files (not just “efficientnet b5” filenames), then load them strictly (so the regression head weights are included), and only fall back to the current “ignore head” ImageNet loading if needed. I also fix a subtle bug in the skip-connection condition (`stride` is stored as a list, so the current `== 1` check is wrong), which can materially damage inference even with good weights while keeping the architecture logically the same. Submission writing stays identical and still always produce `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with predictions being effectively random and/or badly calibrated for QWK because the model is never trained and likely isn’t finding a useful fine-tuned checkpoint. To move toward the 0.911 target with minimal, semantics-preserving changes, I keep your EfficientNet-B5 definition and pure-inference pipeline, but (1) improve the offline checkpoint search to reliably pick up common APTOS fine-tuned files and load them strictly when possible, and (2) if only an ImageNet backbone is found, I calibrate the 4 rounding thresholds on a small held-out split of `train.csv` (no training; just threshold fitting) to improve QWK mapping without changing the model. I also make the dataloaders explicit (remove reliance on the global `test_dl` inside `run_subm`) so the validation calibration and test inference use consistent transforms and batching. Submission format and file path remain the same, and it always writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score indicates the model is still effectively untrained or mis-loaded, so the smallest legitimate step toward the 0.911 target is to reliably load a real fine-tuned checkpoint (including the regression head) when it exists, and otherwise at least guarantee strong ImageNet backbone loading. I keep your EfficientNet-B5 architecture and inference pipeline identical, but make the checkpoint loader more robust by (1) handling common checkpoint formats (`state_dict`, `model_state_dict`, nested dicts) and (2) selecting the best candidate by preferring “best/fold/qwk/kappa/aptos” and avoiding optimizer-only files. Finally, I always run threshold calibration on a held-out split (even for full checkpoints) to better align regression outputs to the QWK ordinal labels without changing the model/training approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the model is running with random weights or only partial ImageNet backbone weights, which yields near-random ordinal predictions and QWK≈0. To move the score toward the 0.911 target with minimal changes and without changing the architecture/training logic, I (1) make the checkpoint loader reliably find and prefer *APTOS fine-tuned* checkpoints by searching `/kaggle/data` as well as `/kaggle/input`, and (2) make state-dict cleaning robust to common key prefixes (including `md_ef.`) so strict loading succeeds when the correct checkpoint exists. I keep your threshold calibration (no training) but ensure the validation split is stratified so the fitted thresholds are stable and not biased by shuffle order. Submission writing stays identical (`submission.csv` with `id_code,diagnosis`) and the script still runs end-to-end within Kaggle constraints.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import math
import re
import json
import collections
from functools import partial

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn import metrics

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




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
        ih, iw = (
            image_size if isinstance(image_size, list) else [image_size, image_size]
        )
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
        """Gets a block through a string notation of arguments."""
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

        return BlockArgs(
            kernel_size=int(options["k"]),
            num_repeat=int(options["r"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            expand_ratio=int(options["e"]),
            id_skip=("noskip" not in block_string),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=[int(options["s"][0])],
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
    """Get the block args and global params for a given model"""
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
        stride_val = (
            self._block_args.stride[0]
            if isinstance(self._block_args.stride, (list, tuple))
            else self._block_args.stride
        )
        if self.id_skip and stride_val == 1 and input_filters == output_filters:
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
def _find_existing_file(candidates):
    for p in candidates:
        if p and os.path.exists(p) and os.path.isfile(p):
            return p
    return None


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            v = obj.get(k, None)
            if isinstance(v, dict) and len(v) > 0:
                return v
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    return obj


def _clean_state_dict_keys(state):
    state = _extract_state_dict(state)
    cleaned = {}
    if not isinstance(state, dict):
        return cleaned
    for k, v in state.items():
        nk = k
        for pref in (
            "module.",
            "model.",
            "net.",
            "learner.",
            "backbone.",
            "md_ef.",
            "encoder.",
        ):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        cleaned[nk] = v
    return cleaned


def load_backbone_weights_ignore_head(model, state_dict):
    model_sd = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return missing, unexpected, len(filtered)


def _glob_weight_candidates(root):
    hits = []
    if not os.path.exists(root):
        return hits
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            lfn = fn.lower()
            if not (
                lfn.endswith(".pth") or lfn.endswith(".pt") or lfn.endswith(".bin")
            ):
                continue
            full = os.path.join(dirpath, fn)
            kw = (
                "aptos",
                "blind",
                "retina",
                "diabetic",
                "kappa",
                "qwk",
                "fold",
                "stage",
                "best",
                "finetune",
                "fine-tune",
                "efficientnet",
                "efnet",
                "b5",
            )
            if any(k in lfn for k in kw):
                hits.append(full)
    return sorted(set(hits))


def _rank_ckpt_path(p):
    b = os.path.basename(p).lower()
    score = 0.0
    pos = [
        ("best", 5.0),
        ("qwk", 4.0),
        ("kappa", 4.0),
        ("aptos", 3.0),
        ("blind", 2.0),
        ("retina", 2.0),
        ("fold", 1.0),
        ("efficientnet", 1.0),
        ("b5", 1.0),
    ]
    neg = [
        ("optim", -6.0),
        ("optimizer", -6.0),
        ("sched", -4.0),
        ("scheduler", -4.0),
    ]
    for k, w in pos:
        if k in b:
            score += w
    for k, w in neg:
        if k in b:
            score += w
    try:
        score += min(os.path.getsize(p) / 1e8, 3.0)
    except Exception:
        pass
    return -score  # for ascending sort


md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1).to(DEVICE)
md_ef.eval()

loaded_any = False
loaded_full = False
loaded_path = None

search_roots = ["/kaggle/input", "/kaggle/data", "../input", "../data"]
globbed = []
for r in search_roots:
    globbed += _glob_weight_candidates(r)

fine_tuned_candidates = [
    os.path.join("models", "model.pth"),
    os.path.join("models", "best.pth"),
    os.path.join("models", "checkpoint.pth"),
] + globbed

fine_tuned_candidates = sorted(set(fine_tuned_candidates), key=_rank_ckpt_path)
fine_path = _find_existing_file(fine_tuned_candidates)

if fine_path is not None:
    state = torch.load(fine_path, map_location=DEVICE)
    cleaned = _clean_state_dict_keys(state)
    if len(cleaned) > 0:
        try:
            md_ef.load_state_dict(cleaned, strict=True)
            print(f"Loaded fine-tuned checkpoint (strict): {fine_path}")
            loaded_any = True
            loaded_full = True
            loaded_path = fine_path
        except Exception as e:
            print(
                f"Found checkpoint but strict load failed, will fall back. Path: {fine_path}"
            )
            print(f"Strict load error: {repr(e)}")

if not loaded_any:
    imagenet_candidates = [
        os.path.join("..", "input", "efficientnet-pytorch", "efficientnet-b5.pth"),
        os.path.join(
            "..", "input", "efficientnet-pytorch", "efficientnet-b5-355c32eb.pth"
        ),
        os.path.join("..", "input", "efficientnet-pytorch", "efficientnet_b5.pth"),
    ] + globbed

    imagenet_candidates = sorted(set(imagenet_candidates), key=_rank_ckpt_path)
    imgnet_path = _find_existing_file(imagenet_candidates)

    if imgnet_path is not None:
        state = torch.load(imgnet_path, map_location=DEVICE)
        cleaned = _clean_state_dict_keys(state)
        missing, unexpected, nloaded = load_backbone_weights_ignore_head(md_ef, cleaned)
        print(
            f"Loaded ImageNet EfficientNet-B5 weights (filtered, head ignored): {imgnet_path}"
        )
        print(
            f"Loaded keys: {nloaded}, Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
        )
        loaded_any = nloaded > 0
        loaded_path = imgnet_path
    else:
        print(
            "WARNING: No local fine-tuned checkpoint or ImageNet EfficientNet-B5 weights found; "
            "running with random weights (submission will be valid but likely low score)."
        )

print(f"Any weights loaded: {loaded_any} (full fine-tuned: {loaded_full})")
if loaded_path is not None:
    print(f"Checkpoint used: {loaded_path}")




## === cell 3
def _find_dataset_root():
    candidates = [
        os.path.join("..", "input", "aptos2019-blindness-detection"),
        os.path.join("..", "input", "kaggle", "data", "aptos2019-blindness-detection"),
        os.path.join("/kaggle", "input", "aptos2019-blindness-detection"),
        os.path.join("/kaggle", "data", "aptos2019-blindness-detection"),
        os.path.join(
            "/kaggle",
            "data",
            "aptos2019-blindness-detection",
            "aptos2019-blindness-detection",
        ),
        os.path.join(
            "/kaggle",
            "input",
            "aptos2019-blindness-detection",
            "aptos2019-blindness-detection",
        ),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            return c
    fallback = os.path.join("..", "input")
    return os.path.join(fallback, "aptos2019-blindness-detection")


def get_df():
    base_image_dir = _find_dataset_root()
    train_dir = os.path.join(base_image_dir, "train_images")
    test_dir = os.path.join(base_image_dir, "test_images")

    train_csv = os.path.join(base_image_dir, "train.csv")
    test_csv = os.path.join(base_image_dir, "test.csv")

    df = pd.read_csv(train_csv)
    df["path"] = df["id_code"].map(lambda x: os.path.join(train_dir, f"{x}.png"))
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    test_df = pd.read_csv(test_csv)
    test_df["path"] = test_df["id_code"].map(
        lambda x: os.path.join(test_dir, f"{x}.png")
    )
    return df, test_df


df, test_df = get_df()

missing_paths = test_df.loc[~test_df["path"].map(os.path.exists), "path"].tolist()
if len(missing_paths) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing_paths)} test images, e.g.: {missing_paths[:3]}"
    )

df.head(), test_df.head()




## === cell 4
sz = int(getattr(md_ef._global_params, "image_size", 456) or 456)
bs = 8

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

tfm = transforms.Compose(
    [
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)


class RetinaDataset(Dataset):
    def __init__(self, df, transform, label_col=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.loc[idx, "path"]
        with Image.open(p) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        if self.label_col is None:
            return x
        y = int(self.df.loc[idx, self.label_col])
        return x, y


def make_loader(ds, batch_size, shuffle):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


test_ds = RetinaDataset(test_df, tfm, label_col=None)
test_dl = make_loader(test_ds, batch_size=bs, shuffle=False)




## === cell 5
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def predict(self, X, coef):
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
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
        return X_p.astype(np.int64)


def _quadratic_weighted_kappa(y_true, y_pred):
    return metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")


def _fit_thresholds_on_val(
    model, df_with_labels, n_val=384, init_coef=(0.5, 1.5, 2.5, 3.5)
):
    n_val = int(min(max(n_val, 128), len(df_with_labels) // 2))
    df_in = df_with_labels.copy().reset_index(drop=True)

    rng = np.random.RandomState(SEED)
    parts = []
    per_class = max(1, n_val // 5)
    for c in [0, 1, 2, 3, 4]:
        dfi = df_in[df_in["diagnosis"] == c]
        take = min(per_class, len(dfi))
        if take > 0:
            parts.append(dfi.sample(n=take, random_state=int(rng.randint(0, 10**9))))
    val_df = (
        pd.concat(parts, axis=0)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    if len(val_df) < n_val:
        rest = df_in.drop(val_df.index, errors="ignore")
        need = n_val - len(val_df)
        if need > 0 and len(rest) > 0:
            val_df = pd.concat(
                [val_df, rest.sample(n=min(need, len(rest)), random_state=SEED)],
                axis=0,
            ).reset_index(drop=True)
    val_df = val_df.iloc[:n_val].copy().reset_index(drop=True)

    val_ds = RetinaDataset(val_df, tfm, label_col="diagnosis")
    val_dl = make_loader(val_ds, batch_size=bs, shuffle=False)

    model.eval()
    preds_all, y_all = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(DEVICE, non_blocking=True)
            logits = model(xb).view(-1)
            preds_all.append(logits.detach().cpu())
            y_all.append(yb.detach().cpu())
    preds = torch.cat(preds_all).numpy().reshape(-1)
    y = torch.cat(y_all).numpy().astype(int).reshape(-1)

    coef = np.array(init_coef, dtype=np.float64)
    opt = OptimizedRounder()

    def score(c):
        c = np.sort(np.clip(np.asarray(c, dtype=np.float64), -3.0, 7.0))
        yp = opt.predict(preds, c)
        return _quadratic_weighted_kappa(y, yp)

    best = score(coef)

    for step in [0.5, 0.2, 0.1]:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for delta in (-step, step):
                    trial = coef.copy()
                    trial[i] += delta
                    trial = np.sort(trial)
                    s = score(trial)
                    if s > best + 1e-6:
                        coef = trial
                        best = s
                        improved = True

    print(f"Calibrated thresholds (val QWK={best:.4f}): {coef.tolist()}")
    return coef.tolist()




## === cell 6
def run_subm(
    model,
    test_df,
    test_dl,
    coefficients=(0.5, 1.5, 2.5, 3.5),
    out_path="submission.csv",
):
    model.eval()
    preds_all = []
    with torch.no_grad():
        for xb in test_dl:
            xb = xb.to(DEVICE, non_blocking=True)
            logits = model(xb).view(-1)
            preds_all.append(logits.detach().cpu())
    preds = torch.cat(preds_all, dim=0)

    opt = OptimizedRounder()
    diag = opt.predict(preds, coefficients)

    sub = pd.DataFrame({"id_code": test_df["id_code"].values, "diagnosis": diag})
    sub["diagnosis"] = sub["diagnosis"].clip(0, 4).astype(int)
    sub.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {sub.shape}")
    return sub


best_coef = _fit_thresholds_on_val(md_ef, df, n_val=384, init_coef=(0.5, 1.5, 2.5, 3.5))

submission = run_subm(
    md_ef,
    test_df=test_df,
    test_dl=test_dl,
    coefficients=best_coef,
    out_path="submission.csv",
)
submission.head()
