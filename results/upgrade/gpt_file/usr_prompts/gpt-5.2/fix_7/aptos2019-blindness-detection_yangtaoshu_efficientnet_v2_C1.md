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

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on fastai (not installed in this environment) and replace it with a minimal pure-PyTorch inference pipeline while preserving the core model definition (EfficientNet-B5) and the same rounding/thresholding semantics used for the Kaggle submission. I also fix missing imports (`os`, `collections`, etc.), stop attempts to download pretrained weights (no internet), and instead load a local checkpoint if present; if not, the script still run end-to-end with a deterministic fallback that produces a valid `submission.csv`. Finally, I fix path handling to use the provided `/kaggle/input/aptos2019-blindness-detection/...` structure and ensure the CSV has exactly `id_code,diagnosis` and a `.csv` suffix.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with running inference from random weights because no checkpoint is found, so the smallest meaningful improvement is to actually load a real model file that exists in the dataset rather than looking for a hardcoded `abcdef.pth`. I keep your EfficientNet-B5 definition and the same regression→threshold rounding semantics, but broaden checkpoint discovery to scan `/kaggle/input` and `/kaggle/working` for plausible `.pth/.pt/.bin` files and then select the best match based on name hints (efficientnet/b5/aptos/blindness) and state_dict compatibility. I also make the state_dict extraction more robust (common keys like `model`, `net`, `state_dict`) and enforce deterministic ordering/alignment so the submission rows match `test.csv` exactly. These changes are directly aimed at moving the score up toward your target by ensuring you’re not submitting random predictions while preserving your core approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the submission is being generated from random weights (no compatible checkpoint is actually being loaded), so the smallest meaningful change is to make checkpoint discovery/loading both stricter (so we don’t accept a partially-overlapping wrong model) and broader (so we more reliably find a real APTOS EfficientNet-B5 regression checkpoint if it exists under `/kaggle/input`). I keep the exact EfficientNet-B5 architecture and the same regression→fixed-threshold mapping, but improve `try_load_checkpoint()` to (1) prefer checkpoints inside the APTOS dataset folder, (2) require very high key overlap for a load to be considered valid, and (3) handle common checkpoint formats more robustly (including `ema_state_dict` / nested `state_dict`). If no suitable checkpoint is found, the script still completes and writes `submission.csv` exactly as before, but now it’s much more likely to load the intended weights and move the score up toward your target.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with submitting essentially random predictions because no compatible checkpoint is being loaded; the smallest change that can move you toward the target is to actually use a meaningful pretrained backbone when a task-specific checkpoint isn’t found. I keep your EfficientNet-B5 model and the same inference + fixed-threshold rounding semantics, but add an offline, deterministic ImageNet-weight loader by searching `/kaggle/input` for common EfficientNet-B5 weight files and loading them only into the backbone (skipping the 1000-class FC). I also make checkpoint loading slightly more permissive for the head (accepting either `_fc.*` or `fc.*`) while still requiring high overlap for safety, so real competition checkpoints are more likely to load if present. If neither competition nor ImageNet weights are found, the script still produces a valid `submission.csv` exactly as before.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from running EfficientNet-B5 with random weights (no compatible checkpoint exists in this dataset), so the smallest legitimate way to move toward your 0.911 target is to use an actually-available pretrained backbone without changing your model definition or inference semantics. I keep your EfficientNet-B5(num_classes=1) and your fixed thresholds, but (1) add an offline loader that pulls ImageNet EfficientNet-B5 weights from `torchvision` if available, mapping `features.*` → your `_conv/_bn/_blocks` keys and skipping the classification head, and (2) fix a small bug in `MBConvBlock` where `stride` is stored as a list (so the residual condition is never true), which can hurt any loaded weights and is consistent with preserving intended EfficientNet core logic. If neither a competition checkpoint nor torchvision ImageNet weights can be loaded, the script still deterministically write a valid `submission.csv` exactly as before.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the model still effectively behaving like random/untrained weights; the most direct minimal fix is to ensure we always load meaningful ImageNet EfficientNet-B5 weights offline (without internet) when no competition checkpoint exists. I keep your exact EfficientNet-B5 architecture and the same regression→fixed-threshold mapping, but (1) add a robust, version-tolerant torchvision weight loader that works across older/newer torchvision APIs and (2) fix a small padding bug in `Conv2dStaticSamePadding` (pad_w used `kh` instead of `kw`) that can otherwise prevent correct weight transfer and harm predictions. These changes keep semantics intact while materially increasing the chance of non-random predictions and moving your score upward toward the target. The script still run end-to-end and write a valid `submission.csv`.'

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
from torch.utils.data import Dataset, DataLoader

from PIL import Image

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
"""
EfficientNet (core model definition preserved from original code).

Score-toward-target fixes (minimal + semantics-preserving):
1) Keep stride as int (already done) so skip connections behave correctly.
2) Fix Conv2dStaticSamePadding pad_w computation: it incorrectly used kh instead of kw,
   which breaks "same" padding for non-square kernels and can hurt mapped pretrained weights.
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
    return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
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
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
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
        return BlockArgs(
            kernel_size=int(options["k"]),
            num_repeat=int(options["r"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            expand_ratio=int(options["e"]),
            id_skip=("noskip" not in block_string),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=int(options["s"][0]),
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
        if (
            self.id_skip
            and self._block_args.stride == 1
            and input_filters == output_filters
        ):
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1).to(device)
md_ef.eval()



## === cell 3
BASE_DIR = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")

assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

test_df = pd.read_csv(TEST_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)




## === cell 4
def _iter_checkpoint_candidates(search_roots, exts=(".pth", ".pt", ".bin")):
    out = []
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                lfn = fn.lower()
                if lfn.endswith(exts):
                    out.append(os.path.join(dirpath, fn))
    return out


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in [
            "ema_state_dict",
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
        ]:
            if key in ckpt_obj and isinstance(
                ckpt_obj[key], (dict, collections.OrderedDict)
            ):
                return ckpt_obj[key]
        looks_like_sd = any(isinstance(v, torch.Tensor) for v in ckpt_obj.values())
        if looks_like_sd:
            return ckpt_obj
    return None


def _strip_prefixes(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for pref in ["module.", "model.", "net.", "backbone."]:
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        new_sd[nk] = v
    return new_sd


def _score_candidate_path(p):
    lp = p.lower()
    name = os.path.basename(p).lower()
    score = 0

    if "aptos2019-blindness-detection" in lp:
        score += 25
    if "/kaggle/input/" in lp:
        score += 5
    if "/kaggle/working/" in lp:
        score += 1

    for tok, w in [
        ("aptos", 10),
        ("blind", 8),
        ("retina", 6),
        ("diabet", 6),
        ("efficientnet", 10),
        ("b5", 10),
        ("effb5", 10),
        ("reg", 3),
        ("model", 1),
        ("best", 2),
        ("fold", 1),
    ]:
        if tok in name:
            score += w

    score -= min(5, lp.count(os.sep) // 15)
    return score


def _remap_fc_keys_if_needed(sd: dict) -> dict:
    if ("fc.weight" in sd or "fc.bias" in sd) and (
        "_fc.weight" not in sd and "_fc.bias" not in sd
    ):
        nsd = dict(sd)
        if "fc.weight" in nsd:
            nsd["_fc.weight"] = nsd.pop("fc.weight")
        if "fc.bias" in nsd:
            nsd["_fc.bias"] = nsd.pop("fc.bias")
        return nsd
    return sd


def try_load_checkpoint(model: nn.Module):
    search_roots = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/working",
        "/kaggle/input",
    ]
    candidates = _iter_checkpoint_candidates(search_roots)
    if not candidates:
        print(
            "Warning: no .pth/.pt/.bin found under search roots. Will try torchvision ImageNet weights; else random."
        )
        return False

    candidates = sorted(candidates, key=_score_candidate_path, reverse=True)
    topk = candidates[:120]

    best_path = None
    best_loaded = None

    model_sd = model.state_dict()
    model_keys = list(model_sd.keys())
    model_keyset = set(model_keys)

    for p in topk:
        try:
            ckpt = torch.load(p, map_location="cpu")
            sd = _extract_state_dict(ckpt)
            if sd is None:
                continue
            sd = _strip_prefixes(sd)
            sd = _remap_fc_keys_if_needed(sd)

            shared_keys = [
                k
                for k in sd.keys()
                if k in model_keyset
                and hasattr(sd[k], "shape")
                and sd[k].shape == model_sd[k].shape
            ]
            overlap = len(shared_keys)

            if overlap < int(0.85 * len(model_keys)):
                continue

            missing, unexpected = model.load_state_dict(sd, strict=False)

            if ("_fc.weight" in model_sd) and ("_fc.weight" in sd):
                if sd["_fc.weight"].shape != model_sd["_fc.weight"].shape:
                    continue

            best_path = p
            best_loaded = (missing, unexpected, overlap, len(model_keys))
            break
        except Exception:
            continue

    if best_path is None:
        print(
            "Warning: found checkpoint files, but none were sufficiently compatible with EfficientNet-B5(num_classes=1)."
        )
        return False

    missing, unexpected, overlap, total = best_loaded
    print(
        f"Loaded competition checkpoint: {best_path} (matched_keys={overlap}/{total})"
    )
    if missing:
        print(f"Note: missing keys (showing up to 10): {missing[:10]}")
    if unexpected:
        print(f"Note: unexpected keys (showing up to 10): {unexpected[:10]}")
    return True


def try_load_imagenet_backbone_weights(model: nn.Module):
    search_roots = ["/kaggle/input", "/kaggle/working"]
    candidates = _iter_checkpoint_candidates(search_roots)
    if not candidates:
        return False

    def score_imagenet(p):
        lp = p.lower()
        name = os.path.basename(p).lower()
        s = 0
        for tok, w in [
            ("efficientnet", 10),
            ("b5", 10),
            ("imagenet", 8),
            ("noisy", 2),
            ("advprop", 2),
            ("weights", 2),
            ("pretrain", 2),
        ]:
            if tok in name:
                s += w
        if "aptos" in lp or "blind" in lp:
            s -= 10
        return s

    candidates = sorted(candidates, key=score_imagenet, reverse=True)
    topk = candidates[:200]

    model_sd = model.state_dict()
    model_keyset = set(model_sd.keys())

    for p in topk:
        try:
            ckpt = torch.load(p, map_location="cpu")
            sd = _extract_state_dict(ckpt)
            if sd is None:
                continue
            sd = _strip_prefixes(sd)

            filtered = {}
            for k, v in sd.items():
                if k.startswith("_fc.") or k.startswith("fc."):
                    continue
                if (
                    k in model_keyset
                    and hasattr(v, "shape")
                    and v.shape == model_sd[k].shape
                ):
                    filtered[k] = v

            if len(filtered) < int(0.75 * (len(model_sd) - 2)):
                continue

            model.load_state_dict(filtered, strict=False)
            print(
                f"Loaded ImageNet backbone weights (no head) from: {p} (loaded_tensors={len(filtered)})"
            )
            return True
        except Exception:
            continue

    return False


def try_load_torchvision_efficientnet_b5_backbone(model: nn.Module) -> bool:
    """
    Score-toward-target change: robustly load offline torchvision ImageNet EfficientNet-B5 weights
    across torchvision versions (old/new APIs), then map backbone tensors into this implementation.
    This avoids random predictions when no competition checkpoint is available.
    """
    try:
        import torchvision
        from torchvision.models import efficientnet_b5
    except Exception:
        return False

    tv = None
    tv_sd = None
    try:
        from torchvision.models import EfficientNet_B5_Weights  # type: ignore

        try:
            tv = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        except Exception:
            tv = efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        tv_sd = tv.state_dict()
    except Exception:
        try:
            tv = efficientnet_b5(pretrained=True)
            tv_sd = tv.state_dict()
        except Exception:
            return False

    my_sd = model.state_dict()
    mapped = {}

    if "features.0.0.weight" in tv_sd and "_conv_stem.weight" in my_sd:
        if tv_sd["features.0.0.weight"].shape == my_sd["_conv_stem.weight"].shape:
            mapped["_conv_stem.weight"] = tv_sd["features.0.0.weight"]
    for suf in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
        tk = f"features.0.1.{suf}"
        mk = f"_bn0.{suf}"
        if (
            tk in tv_sd
            and mk in my_sd
            and hasattr(tv_sd[tk], "shape")
            and tv_sd[tk].shape == my_sd[mk].shape
        ):
            mapped[mk] = tv_sd[tk]

    tv_block_keys = [(k, v) for k, v in tv_sd.items() if k.startswith("features.1.")]
    if not tv_block_keys:
        return False

    def _convert_block_key(k: str) -> str:
        return k.replace("features.1.", "_blocks.", 1)

    for k, v in tv_block_keys:
        mk = _convert_block_key(k)
        if mk in my_sd and hasattr(v, "shape") and v.shape == my_sd[mk].shape:
            mapped[mk] = v

    if "features.2.0.weight" in tv_sd and "_conv_head.weight" in my_sd:
        if tv_sd["features.2.0.weight"].shape == my_sd["_conv_head.weight"].shape:
            mapped["_conv_head.weight"] = tv_sd["features.2.0.weight"]
    for suf in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
        tk = f"features.2.1.{suf}"
        mk = f"_bn1.{suf}"
        if (
            tk in tv_sd
            and mk in my_sd
            and hasattr(tv_sd[tk], "shape")
            and tv_sd[tk].shape == my_sd[mk].shape
        ):
            mapped[mk] = tv_sd[tk]

    if len(mapped) < int(0.70 * (len(my_sd) - 2)):  # ignore _fc.{weight,bias}
        return False

    model.load_state_dict(mapped, strict=False)
    print(
        f"Loaded torchvision EfficientNet-B5 ImageNet backbone weights (mapped_tensors={len(mapped)})"
    )
    return True


loaded_comp = try_load_checkpoint(md_ef)
if not loaded_comp:
    loaded_imnet = try_load_imagenet_backbone_weights(md_ef)
    if not loaded_imnet:
        loaded_tv = try_load_torchvision_efficientnet_b5_backbone(md_ef)
        if not loaded_tv:
            print(
                "Warning: no compatible competition checkpoint, no local ImageNet weights, and no torchvision weights; using random weights -> low score."
            )
md_ef.to(device).eval()



## === cell 5
IMG_SIZE = 456
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TestImageDataset(Dataset):
    def __init__(self, df, img_dir, img_size=456):
        self.ids = df["id_code"].values
        self.img_dir = img_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.ids)

    def _load_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        x = self._load_image(img_path)
        return x, img_id


test_ds = TestImageDataset(test_df, TEST_IMG_DIR, img_size=IMG_SIZE)
test_dl = DataLoader(
    test_ds,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 6
def apply_thresholds(preds: np.ndarray, coef=(0.5, 1.5, 2.5, 3.5)) -> np.ndarray:
    preds = preds.astype(np.float32).reshape(-1)
    out = np.zeros_like(preds, dtype=np.int64)
    out[preds >= coef[0]] = 1
    out[preds >= coef[1]] = 2
    out[preds >= coef[2]] = 3
    out[preds >= coef[3]] = 4
    return out


@torch.no_grad()
def predict_test(model, loader):
    model.eval()
    all_ids = []
    all_preds = []
    for xb, ids in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).squeeze(1)
        all_preds.append(logits.detach().float().cpu().numpy())
        all_ids.extend(list(ids))
    return np.concatenate(all_preds, axis=0), np.array(all_ids)


raw_preds, ids = predict_test(md_ef, test_dl)



## === cell 7
diagnosis = apply_thresholds(raw_preds, coef=(0.5, 1.5, 2.5, 3.5))

pred_df = pd.DataFrame({"id_code": ids, "diagnosis": diagnosis.astype(int)})
sub = test_df[["id_code"]].merge(pred_df, on="id_code", how="left")
sub["diagnosis"] = sub["diagnosis"].fillna(0).astype(int)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape={sub.shape} and columns={list(sub.columns)}")
print(sub.head())
print(
    "diagnosis value counts:\n",
    sub["diagnosis"].value_counts(dropna=False).sort_index(),
)
