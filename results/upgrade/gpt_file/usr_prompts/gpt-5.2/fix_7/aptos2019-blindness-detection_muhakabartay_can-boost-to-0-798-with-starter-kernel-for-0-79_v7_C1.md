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

0.9088854482695004

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the import errors caused by an unexpected fastai version by switching to fastai v2 equivalents (DataBlock, vision_learner, Resize, etc.) while keeping the same core model (EfficientNet-B5 regression head) and the same quadratic-kappa-oriented rounding at submission time. I also remove the broken fastai-v1-only APIs (`ImageList`, `Learner`, `DatasetType`, `to_fp16`) and replace them with a minimal, deterministic training/inference pipeline that still trains (if no checkpoint exists) and produces `submission.csv` with the required columns. I keep the existing “load local EfficientNet weights if present, otherwise train briefly” behavior, but make it actually run end-to-end within the Kaggle filesystem provided. Finally, I ensure test predictions are aligned with `test.csv`/`sample_submission.csv` and that the output is integer classes 0–4.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission/data alignment issue rather than model quality, so I make the smallest fixes that ensure `test_df` is built from `test.csv` (not `sample_submission.csv`), and that the model outputs a sensible 0–4 prediction even when no checkpoint is found. I also remove the dependency on `scipy` during inference by fitting kappa-optimized thresholds with a lightweight grid search on the validation set (same rounding/thresholding semantics, just actually calibrated to this run). Finally, I ensure the submission rows exactly match `test.csv` order and `id_code`, producing a valid `submission.csv` every time.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly due to an `id_code`/row alignment bug: `test_df` coming from `get_df()` has no `path`, while `test_dl` is built from a different dataframe (`test_csv`), so predictions are written onto a mismatched/possibly shuffled `test_df`. I make the smallest change to ensure the exact same `test_df` (built from `test.csv` in the same order) is used for both `test_dl` creation and submission writing, preserving your model and rounding logic. I also fix a minor EfficientNet skip-connection condition bug (`stride` is a list in this implementation) that can destabilize training/inference, without changing the architecture. These two fixes should move the score up from 0.0 toward your target by producing a correctly aligned, valid submission.'

# 9. Code solution

## === cell 0
import os
import math
import re
import json
import collections
from functools import partial
from collections import Counter

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from torch.utils.data import Dataset

from sklearn import metrics
from sklearn.metrics import cohen_kappa_score

from fastai.vision.all import (
    DataBlock,
    ImageBlock,
    RegressionBlock,
    Resize,
    aug_transforms,
    Normalize,
    imagenet_stats,
    RandomSplitter,
    Learner,
    set_seed,
)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
set_seed(42, reproducible=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)



## === cell 1
"""
EfficientNet implementation (as provided), plus helpers.
Only minimal bug fixes: keep core logic intact.
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
        pad_w = max((oh - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
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


def load_pretrained_weights(model, model_name, load_fc=True):
    state_dict = model_zoo.load_url(url_map[model_name])
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop("_fc.weight")
        state_dict.pop("_fc.bias")
        res = model.load_state_dict(state_dict, strict=False)
        assert str(res.missing_keys) == str(
            ["_fc.weight", "_fc.bias"]
        ), "issue loading pretrained weights"
    print("Loaded pretrained weights for {}".format(model_name))


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

        stride_is_1 = (self._block_args.stride == 1) or (self._block_args.stride == [1])

        if self.id_skip and stride_is_1 and input_filters == output_filters:
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
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ["efficientnet_b" + str(i) for i in range(num_models)]
        if model_name.replace("-", "_") not in valid_models:
            raise ValueError("model_name should be one of: " + ", ".join(valid_models))




## === cell 2
md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)


def _unwrap_state_dict(sd):
    if not isinstance(sd, dict):
        return sd
    for k in ("state_dict", "model", "net"):
        if k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    if any(key.startswith("module.") for key in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    if any(key.startswith("model.") for key in sd.keys()):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    return sd


def try_load_efficientnet_local(model, model_name="efficientnet-b5"):
    common_filenames = [
        "efficientnet-b5-586e6cc6.pth",
        "efficientnet_b5.pth",
        "efficientnet-b5.pth",
    ]
    search_roots = [
        "/kaggle/input",
        "/kaggle/working",
    ]
    for root in search_roots:
        for dirpath, _, filenames in os.walk(root):
            for fn in common_filenames:
                if fn in filenames:
                    p = os.path.join(dirpath, fn)
                    sd = torch.load(p, map_location="cpu")
                    sd = _unwrap_state_dict(sd)
                    if isinstance(sd, dict) and "_fc.weight" in sd:
                        sd.pop("_fc.weight", None)
                    if isinstance(sd, dict) and "_fc.bias" in sd:
                        sd.pop("_fc.bias", None)
                    res = model.load_state_dict(sd, strict=False)
                    print(f"Loaded local EfficientNet weights from: {p}")
                    print(
                        f"Missing keys: {res.missing_keys}; Unexpected keys: {res.unexpected_keys}"
                    )
                    return True
    print("No local EfficientNet pretrained weights found; will continue without them.")
    return False


_ = try_load_efficientnet_local(md_ef, "efficientnet-b5")
md_ef = md_ef.to(DEVICE)




## === cell 3
def get_df():
    base_image_dir = "/kaggle/input/aptos2019-blindness-detection"
    train_dir = os.path.join(base_image_dir, "train_images")
    test_dir = os.path.join(base_image_dir, "test_images")

    df = pd.read_csv(os.path.join(base_image_dir, "train.csv"))
    df["path"] = df["id_code"].map(lambda x: os.path.join(train_dir, f"{x}.png"))
    df = df.drop(columns=["id_code"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    test_df = pd.read_csv(os.path.join(base_image_dir, "test.csv"))
    test_df["path"] = test_df["id_code"].map(
        lambda x: os.path.join(test_dir, f"{x}.png")
    )
    return df, test_df


df, test_df = get_df()
print(df.head())
print(test_df.head())

missing_test = (~test_df["path"].map(os.path.isfile)).sum()
missing_train = (~df["path"].map(os.path.isfile)).sum()
print(
    "Missing train images:",
    int(missing_train),
    "Missing test images:",
    int(missing_test),
)
assert (
    missing_test == 0
), "Some test image paths do not exist; cannot produce valid predictions."



## === cell 4
bs = 64

sz = 456

item_tfms = Resize(sz, method="squish")
batch_tfms = [
    *aug_transforms(do_flip=True, flip_vert=True),
    Normalize.from_stats(*imagenet_stats),
]


def _get_x(r):
    return r["path"]


def _get_y(r):
    return float(r["diagnosis"])


dblock = DataBlock(
    blocks=(ImageBlock, RegressionBlock),
    get_x=_get_x,
    get_y=_get_y,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=item_tfms,
    batch_tfms=batch_tfms,
)

dls = dblock.dataloaders(df, bs=bs, num_workers=2)




## === cell 5
def qk_metric(inp, targ):
    pred = torch.round(inp).clamp(0, 4).detach().cpu().numpy().reshape(-1)
    true = targ.detach().cpu().numpy().reshape(-1)
    return cohen_kappa_score(true, pred, weights="quadratic")


learn = Learner(dls, md_ef, metrics=[qk_metric], model_dir="/kaggle/working/models")
if torch.cuda.is_available() and hasattr(learn, "to_fp16"):
    learn = learn.to_fp16()
print("Learner ready. fp16:", torch.cuda.is_available() and hasattr(learn, "to_fp16"))

test_dl = learn.dls.test_dl(test_df, with_labels=False)
print("Test items:", len(test_df))



## === cell 6
model_name = "abcdef"
candidate_paths = [
    os.path.join("/kaggle/input", "kaggle-public", f"{model_name}.pth"),
    os.path.join("/kaggle/input", f"{model_name}.pth"),
    os.path.join("/kaggle/working", "models", f"{model_name}.pth"),
    os.path.join("/kaggle/working", f"{model_name}.pth"),
]

os.makedirs("/kaggle/working/models", exist_ok=True)
loaded = False
for p in candidate_paths:
    if os.path.isfile(p):
        try:
            if os.path.dirname(p) != "/kaggle/working/models":
                import shutil

                dst = os.path.join("/kaggle/working/models", f"{model_name}.pth")
                shutil.copy2(p, dst)
            learn.load(model_name)
            loaded = True
            print(f"Loaded fastai checkpoint: {p}")
            break
        except Exception as e:
            print(
                f"Found {p} but learn.load failed ({e}); trying torch.load state_dict."
            )
            sd = torch.load(p, map_location="cpu")
            sd = _unwrap_state_dict(sd)
            res = learn.model.load_state_dict(sd, strict=False)
            loaded = True
            print(f"Loaded state_dict from: {p}")
            print(
                f"Missing keys: {res.missing_keys}; Unexpected keys: {res.unexpected_keys}"
            )
            break

if not loaded:
    print(
        "WARNING: pretrained weights file 'abcdef.pth' not found; "
        "training a small baseline model to produce a valid submission."
    )
    learn.model.train()
    learn.fit_one_cycle(1, lr_max=1e-3)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_56/2211991504.py in <cell line: 0>()
     41     )
     42     learn.model.train()
---> 43     learn.fit_one_cycle(1, lr_max=1e-3)
     44 
     45 

/usr/local/lib/python3.11/dist-packages/fastai/callback/schedule.py in fit_one_cycle(self, n_epoch, lr_max, div, div_final, pct_start, wd, moms, cbs, reset_opt, start_epoch)
    119     scheds = {'lr': combined_cos(pct_start, lr_max/div, lr_max, lr_max/div_final),
    120               'mom': combined_cos(pct_start, *(self.moms if moms is None else moms))}
--> 121     self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd, start_epoch=start_epoch)
    122 
    123 # %% ../../nbs/14_callback.schedule.ipynb 50

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in fit(self, n_epoch, lr, wd, cbs, reset_opt, start_epoch)
    270             self.opt.set_hypers(lr=self.lr if lr is None else lr)
    271             self.n_epoch = n_epoch
--> 272             self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)
    273 
    274     def _end_cleanup(self): self.dl,self.xb,self.yb,self.pred,self.loss = None,(None,),(None,),None,None

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_fit(self)
    259         for epoch in range(self.n_epoch):
    260             self.epoch=epoch
--> 261             self._with_events(self._do_epoch, 'epoch', CancelEpochException)
    262 
    263     def fit(self, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False, start_epoch=0):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch(self)
    253 
    254     def _do_epoch(self):
--> 255         self._do_epoch_train()
    256         self._do_epoch_validate()
    257 

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch_train(self)
    245     def _do_epoch_train(self):
    246         self.dl = self.dls.train
--> 247         self._with_events(self.all_batches, 'train', CancelTrainException)
    248 
    249     def _do_epoch_validate(self, ds_idx=1, dl=None):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in all_batches(self)
    211     def all_batches(self):
    212         self.n_iter = len(self.dl)
--> 213         for o in enumerate(self.dl): self.one_batch(*o)
    214 
    215     def _backward(self): self.loss_grad.backward()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in one_batch(self, i, b)
    241         b = self._set_device(b)
    242         self._split(b)
--> 243         self._with_events(self._do_one_batch, 'batch', CancelBatchException)
    244 
    245     def _do_epoch_train(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_one_batch(self)
    222 
    223     def _do_one_batch(self):
--> 224         self.pred = self.model(*self.xb)
    225         self('after_pred')
    226         if len(self.yb):

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

/tmp/ipykernel_56/350219730.py in forward(self, inputs)
    435 
    436     def forward(self, inputs):
--> 437         x = self.extract_features(inputs)
    438         x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
    439         if self._dropout:

/tmp/ipykernel_56/350219730.py in extract_features(self, inputs)
    430             if drop_connect_rate:
    431                 drop_connect_rate *= float(idx) / len(self._blocks)
--> 432             x = block(x, drop_connect_rate=drop_connect_rate)
    433         x = relu_fn(self._bn1(self._conv_head(x)))
    434         return x

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

/tmp/ipykernel_56/350219730.py in forward(self, inputs, drop_connect_rate)
    348         x = inputs
    349         if self._block_args.expand_ratio != 1:
--> 350             x = relu_fn(self._bn0(self._expand_conv(inputs)))
    351         x = relu_fn(self._bn1(self._depthwise_conv(x)))
    352 

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/batchnorm.py in forward(self, input)
    191         used for normalization (i.e. in eval mode when buffers are not None).
    192         """
--> 193         return F.batch_norm(
    194             input,
    195             # If buffers are not to be tracked, ensure that they won't be updated

/usr/local/lib/python3.11/dist-packages/torch/nn/functional.py in batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)
   2820         _verify_batch_size(input.size())
   2821 
-> 2822     return torch.batch_norm(
   2823         input,
   2824         weight,

OutOfMemoryError: CUDA out of memory. Tried to allocate 44.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 20.88 MiB is free. Process 3047950 has 47.50 GiB memory in use. Of the allocated memory 46.87 GiB is allocated by PyTorch, and 291.83 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 7
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = None

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred < coef[1]:
                X_p[i] = 1
            elif pred < coef[2]:
                X_p[i] = 2
            elif pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def fit_grid(self, X, y, grid=None):
        if grid is None:
            grid = np.arange(0.2, 3.9, 0.1, dtype=np.float32)

        best_kappa = -1.0
        best_coef = (0.5, 1.5, 2.5, 3.5)

        X = np.asarray(X, dtype=np.float32).reshape(-1)
        y = np.asarray(y).reshape(-1)

        for c0 in grid:
            for c1 in grid:
                if c1 <= c0:
                    continue
                for c2 in grid:
                    if c2 <= c1:
                        continue
                    for c3 in grid:
                        if c3 <= c2:
                            continue
                        coef = (float(c0), float(c1), float(c2), float(c3))
                        pred = self.predict(X, coef).astype(np.int64)
                        k = cohen_kappa_score(y, pred, weights="quadratic")
                        if k > best_kappa:
                            best_kappa = k
                            best_coef = coef
        self.coef_ = best_coef
        print("Best val QWK (grid):", best_kappa, "coef:", best_coef)
        return best_coef




## === cell 8
def run_subm(
    learn,
    test_df,
    test_dl,
    out_path="submission.csv",
):
    opt = OptimizedRounder()

    val_preds, val_targs = learn.get_preds(dl=learn.dls.valid)
    val_preds_np = val_preds.detach().cpu().numpy().reshape(-1).astype(np.float32)
    val_targs_np = val_targs.detach().cpu().numpy().reshape(-1).astype(np.int64)

    coef = opt.fit_grid(val_preds_np, val_targs_np)

    preds, _ = learn.get_preds(dl=test_dl)
    preds_np = preds.detach().cpu().numpy().reshape(-1).astype(np.float32)

    tst_pred = opt.predict(preds_np, coef).astype(np.int64)
    tst_pred = np.clip(tst_pred, 0, 4)

    assert len(test_df) == len(
        tst_pred
    ), "Prediction length does not match test_df length."

    subm = test_df[["id_code"]].copy()
    subm["diagnosis"] = tst_pred
    subm.to_csv(out_path, index=False)
    print(f"Saved {out_path} with shape={subm.shape} and columns={list(subm.columns)}")
    print(subm.head())
    return subm


_ = run_subm(learn, test_df, test_dl, out_path="submission.csv")
print("Submission file exists:", os.path.isfile("submission.csv"))
print(
    "Submission file size (bytes):",
    os.path.getsize("submission.csv") if os.path.isfile("submission.csv") else None,
)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_56/1785259685.py in <cell line: 0>()
     31 
     32 
---> 33 _ = run_subm(learn, test_df, test_dl, out_path="submission.csv")
     34 print("Submission file exists:", os.path.isfile("submission.csv"))
     35 print(

/tmp/ipykernel_56/1785259685.py in run_subm(learn, test_df, test_dl, out_path)
      7     opt = OptimizedRounder()
      8 
----> 9     val_preds, val_targs = learn.get_preds(dl=learn.dls.valid)
     10     val_preds_np = val_preds.detach().cpu().numpy().reshape(-1).astype(np.float32)
     11     val_targs_np = val_targs.detach().cpu().numpy().reshape(-1).astype(np.int64)

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in get_preds(self, ds_idx, dl, with_input, with_decoded, with_loss, act, inner, reorder, cbs, **kwargs)
    314         if with_loss: ctx_mgrs.append(self.loss_not_reduced())
    315         with ContextManagers(ctx_mgrs):
--> 316             self._do_epoch_validate(dl=dl)
    317             if act is None: act = getcallable(self.loss_func, 'activation')
    318             res = cb.all_tensors()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch_validate(self, ds_idx, dl)
    250         if dl is None: dl = self.dls[ds_idx]
    251         self.dl = dl
--> 252         with torch.no_grad(): self._with_events(self.all_batches, 'validate', CancelValidException)
    253 
    254     def _do_epoch(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in all_batches(self)
    211     def all_batches(self):
    212         self.n_iter = len(self.dl)
--> 213         for o in enumerate(self.dl): self.one_batch(*o)
    214 
    215     def _backward(self): self.loss_grad.backward()

/usr/local/lib/python3.11/dist-packages/fastai/data/load.py in __iter__(self)
    131             if self.pin_memory and type(b) == list: b = tuple(b)
    132             if self.device is not None: b = to_device(b, self.device)
--> 133             yield self.after_batch(b)
    134         self.after_iter()
    135         if hasattr(self, 'it'): del(self.it)

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in __call__(self, o)
    246         self.fs = self.fs.sorted(key='order')
    247 
--> 248     def __call__(self, o): return compose_tfms(o, tfms=self.fs, split_idx=self.split_idx)
    249     def __repr__(self): return f"Pipeline: {' -> '.join([f.name for f in self.fs if f.name != 'noop'])}"
    250     def __getitem__(self,i): return self.fs[i]

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in compose_tfms(x, tfms, is_enc, reverse, **kwargs)
    195     for f in tfms:
    196         if not is_enc: f = f.decode
--> 197         x = f(x, **kwargs)
    198     return x
    199 

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in __call__(self, split_idx, *args, **kwargs)
    112         dec = len(self.decodes.methods) if hasattr(self, 'decodes') else 0
    113         return f'{self.name}(enc:{enc},dec:{dec})'
--> 114     def __call__(self,*args,split_idx=None, **kwargs): return self._call('encodes', *args, split_idx=split_idx, **kwargs)
    115     def decode(self, *args,split_idx=None, **kwargs): return self._call('decodes', *args, split_idx=split_idx, **kwargs)
    116     def setup(self, items=None, train_setup=False):

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in _call(self, nm, split_idx, *args, **kwargs)
    123         if split_idx!=self.split_idx and self.split_idx is not None: return args[0]
    124         if not hasattr(self, nm): return args[0]
--> 125         return self._do_call(nm, *args, **kwargs)
    126 
    127     def _do_call(self, nm, *args, **kwargs):

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in _do_call(self, nm, *args, **kwargs)
    127     def _do_call(self, nm, *args, **kwargs):
    128         if _is_tuple(x:=args[0]):
--> 129             res = tuple(self._do_call(nm, x_, *args[1:], **kwargs) for x_ in x)
    130             return retain_type(res, x, Any)
    131         f = getattr(self,nm)

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in <genexpr>(.0)
    127     def _do_call(self, nm, *args, **kwargs):
    128         if _is_tuple(x:=args[0]):
--> 129             res = tuple(self._do_call(nm, x_, *args[1:], **kwargs) for x_ in x)
    130             return retain_type(res, x, Any)
    131         f = getattr(self,nm)

/usr/local/lib/python3.11/dist-packages/fasttransform/transform.py in _do_call(self, nm, *args, **kwargs)
    134         try: method, ret_type = f._resolve_method_with_cache(f_args)
    135         except NotFoundLookupError: return x
--> 136         return retain_type(method(*f_args,**kwargs), x, ret_type)
    137 
    138 add_docs(Transform, decode="Delegate to decodes to undo transform", setup="Delegate to setups to set up transform")

/usr/local/lib/python3.11/dist-packages/fastai/data/transforms.py in encodes(self, o)
    350     order = 10 #Need to run after PIL transforms on the GPU
    351     def __init__(self, div=255., div_mask=1): store_attr()
--> 352     def encodes(self, o:TensorImage): return o.float().div_(self.div)
    353     def encodes(self, o:TensorMask ): return (o.long() / self.div_mask).long()
    354     def decodes(self, o:TensorImage): return ((o.clamp(0., 1.) * self.div).long()) if self.div else o

/usr/local/lib/python3.11/dist-packages/fastai/torch_core.py in __torch_function__(cls, func, types, args, kwargs)
    382         if cls.debug and func.__name__ not in ('__str__','__repr__'): print(func, types, args, kwargs)
    383         if _torch_handled(args, cls._opt, func): types = (torch.Tensor,)
--> 384         res = super().__torch_function__(func, types, args, ifnone(kwargs, {}))
    385         dict_objs = _find_args(args) if args else _find_args(list(kwargs.values()))
    386         if issubclass(type(res),TensorBase) and dict_objs: res.set_meta(dict_objs[0],as_copy=True)

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in __torch_function__(cls, func, types, args, kwargs)
   1646 
   1647         with _C.DisableTorchFunctionSubclass():
-> 1648             ret = func(*args, **kwargs)
   1649             if func in get_default_nowrap_functions():
   1650                 return ret

OutOfMemoryError: CUDA out of memory. Tried to allocate 154.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 20.88 MiB is free. Process 3047950 has 47.50 GiB memory in use. Of the allocated memory 46.87 GiB is allocated by PyTorch, and 291.83 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
