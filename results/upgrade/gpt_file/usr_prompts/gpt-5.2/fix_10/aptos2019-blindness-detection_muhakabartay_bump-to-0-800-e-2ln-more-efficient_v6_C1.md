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

0.9039714717836792

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the unavailable `fastai` dependency (it’s the root runtime blocker) and replace the minimal training/inference parts with a pure-PyTorch pipeline that keeps the same core idea: EfficientNet-B5 backbone, single-regression output, then optimized rounding to 0–4. I also fix missing imports (`os`, `collections`) and remove the internet weight download (Kaggle kernels are offline) by using `torchvision`’s built-in `efficientnet_b5` weights when available, falling back safely if not. To move score upward toward the target, I fit the rounding thresholds on an out-of-fold validation split using quadratic kappa (same semantics as your OptimizedRounder), then apply them to test predictions. Finally, the script always write `submission.csv` with columns `id_code,diagnosis` in the working directory.'
- What this solution (achieved 0.66808) has done: 'I fix the CUDA out-of-memory crash by reducing batch size and enabling automatic mixed precision (AMP) during training and inference, which preserves the same model architecture and training objective while drastically lowering activation memory. I also add safe CUDA memory cleanup between phases and make DataLoader settings more robust to avoid excessive pinned-memory pressure. These changes are purely runtime/stability fixes so the pipeline can complete end-to-end and write a valid `submission.csv`. Finally, I ensure the submission is always generated in the correct order/format even if CUDA isn’t available.'

# 9. Code solution

## === cell 0
import os
import math
import re
import json
import random
import collections
from functools import partial
from collections import Counter
import copy
import hashlib
import time

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics

import scipy as sp

from PIL import Image

import torchvision
from torchvision import transforms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
"""
This cell preserves your EfficientNet implementation (core model logic).
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
def build_model():
    md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)

    def try_load_torchvision_efficientnet_b5_weights_into_custom(
        model_custom: nn.Module,
    ):
        """
        Best-effort init: if torchvision's efficientnet_b5 pretrained weights are available locally,
        we copy matching weights into the custom EfficientNet where tensor shapes match.
        """
        try:
            if hasattr(torchvision.models, "EfficientNet_B5_Weights"):
                weights = torchvision.models.EfficientNet_B5_Weights.IMAGENET1K_V1
                tv = torchvision.models.efficientnet_b5(weights=weights)
            else:
                tv = torchvision.models.efficientnet_b5(pretrained=True)
        except Exception as e:
            print(
                f"Could not load torchvision pretrained EfficientNet-B5 weights ({e}). Using random init."
            )
            return model_custom

        tv_sd = tv.state_dict()
        cust_sd = model_custom.state_dict()

        copied, skipped = 0, 0
        new_sd = {}
        for k, v in cust_sd.items():
            if k in tv_sd and tv_sd[k].shape == v.shape:
                new_sd[k] = tv_sd[k]
                copied += 1
            else:
                new_sd[k] = v
                skipped += 1

        model_custom.load_state_dict(new_sd, strict=False)
        print(f"Pretrained init: copied {copied} tensors, skipped {skipped}.")
        return model_custom

    md_ef = try_load_torchvision_efficientnet_b5_weights_into_custom(md_ef)
    md_ef = md_ef.to(device)
    return md_ef


_md = build_model()
_md.eval()
BASE_MODEL_STATE = copy.deepcopy(_md.state_dict())
del _md
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 3
os.makedirs("models", exist_ok=True)
os.makedirs("cache", exist_ok=True)




## === cell 4
def get_df():
    base_image_dir = os.path.join("/kaggle", "data", "aptos2019-blindness-detection")
    if not os.path.exists(base_image_dir):
        base_image_dir = os.path.join(
            "/kaggle", "input", "aptos2019-blindness-detection"
        )

    train_dir = os.path.join(base_image_dir, "train_images")
    test_dir = os.path.join(base_image_dir, "test_images")

    df = pd.read_csv(os.path.join(base_image_dir, "train.csv"))
    df["path"] = train_dir + os.sep + df["id_code"].astype(str) + ".png"

    test_df = pd.read_csv(os.path.join(base_image_dir, "test.csv"))
    test_df["path"] = test_dir + os.sep + test_df["id_code"].astype(str) + ".png"

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df, test_df


df, test_df = get_df()
df.head(), test_df.head()



## === cell 5
bs = 8
sz = 456

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose(
    [
        transforms.Resize((sz, sz)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)

test_tfms = transforms.Compose(
    [
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)




## === cell 6
class _BaseTensorCacheDataset(Dataset):
    def __init__(self, df, tfms_det, cache_dir, cache_key, has_labels: bool):
        df = df.reset_index(drop=True)
        self.tfms_det = tfms_det
        self.cache_dir = cache_dir
        self.cache_key = cache_key
        self.has_labels = has_labels

        os.makedirs(self.cache_dir, exist_ok=True)
        self._paths = df["path"].astype(str).values
        self._ids = (
            df["id_code"].astype(str).values if "id_code" in df.columns else None
        )
        self._y = df["diagnosis"].astype(np.float32).values if self.has_labels else None

    def __len__(self):
        return len(self._paths)

    def _cache_path(self, id_code):
        return os.path.join(self.cache_dir, f"{id_code}_{self.cache_key}.pt")

    def _load_or_build_det_tensor(self, idx):
        id_code = self._ids[idx] if self._ids is not None else str(idx)
        cp = self._cache_path(id_code)
        if os.path.exists(cp):
            x = torch.load(cp, map_location="cpu")
        else:
            img_pil = Image.open(self._paths[idx]).convert("RGB")
            x = self.tfms_det(img_pil) if self.tfms_det is not None else img_pil
            torch.save(x, cp)
        return x, id_code


class RetinopathyTestDatasetCached(_BaseTensorCacheDataset):
    def __init__(self, df, tfms_det, cache_dir, cache_key):
        super().__init__(
            df,
            tfms_det=tfms_det,
            cache_dir=cache_dir,
            cache_key=cache_key,
            has_labels=False,
        )

    def __getitem__(self, idx):
        x, id_code = self._load_or_build_det_tensor(idx)
        return x, id_code


class RetinopathyValidDatasetCached(_BaseTensorCacheDataset):
    def __init__(self, df, tfms_det, cache_dir, cache_key):
        super().__init__(
            df,
            tfms_det=tfms_det,
            cache_dir=cache_dir,
            cache_key=cache_key,
            has_labels=True,
        )

    def __getitem__(self, idx):
        x, _id_code = self._load_or_build_det_tensor(idx)
        y = float(self._y[idx])
        return x, torch.tensor([y], dtype=torch.float32)


class RetinopathyTrainDatasetFlipFromCached(_BaseTensorCacheDataset):
    def __init__(self, df, tfms_det, cache_dir, cache_key, fold, epoch, base_seed=SEED):
        super().__init__(
            df,
            tfms_det=tfms_det,
            cache_dir=cache_dir,
            cache_key=cache_key,
            has_labels=True,
        )
        self.fold = int(fold)
        self.epoch = int(epoch)
        self.base_seed = int(base_seed)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _sample_seed(self, idx):
        return (
            self.base_seed * 1000003 + self.fold * 1009 + self.epoch * 9176 + idx
        ) & 0xFFFFFFFF

    def __getitem__(self, idx):
        x, _id_code = self._load_or_build_det_tensor(idx)

        rng = random.Random(self._sample_seed(idx))
        if rng.random() < 0.5:
            x = torch.flip(x, dims=[2])  # width
        if rng.random() < 0.5:
            x = torch.flip(x, dims=[1])  # height

        y = float(self._y[idx])
        return x, torch.tensor([y], dtype=torch.float32)


_det_tfms = test_tfms


def make_loaders(df_train, df_valid, test_df, bs, fold=0, epoch=0):
    cpu_cnt = os.cpu_count() or 2
    num_workers = min(8, max(2, cpu_cnt - 1))
    pin = torch.cuda.is_available()

    cache_key = f"sz{sz}_mean{','.join(map(str,imagenet_mean))}_std{','.join(map(str,imagenet_std))}_det_v3"

    ds_train = RetinopathyTrainDatasetFlipFromCached(
        df_train,
        tfms_det=_det_tfms,
        cache_dir="cache",
        cache_key="train_" + cache_key,
        fold=fold,
        epoch=epoch,
        base_seed=SEED,
    )
    ds_valid = RetinopathyValidDatasetCached(
        df_valid, tfms_det=_det_tfms, cache_dir="cache", cache_key="valid_" + cache_key
    )
    ds_test = RetinopathyTestDatasetCached(
        test_df, tfms_det=_det_tfms, cache_dir="cache", cache_key="test_" + cache_key
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return dl_train, dl_valid, dl_test




## === cell 7
def qk_np(y_pred_cont, y_true_int):
    y_pred_round = np.clip(np.rint(y_pred_cont), 0, 4).astype(int)
    return cohen_kappa_score(y_true_int.astype(int), y_pred_round, weights="quadratic")




## === cell 8
use_amp = torch.cuda.is_available()


def make_scaler():
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def train_one_epoch(model, loader, optimizer, loss_fn, scaler):
    model.train()
    total_loss = 0.0
    non_block = torch.cuda.is_available()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=non_block)
        yb = yb.to(device, non_blocking=non_block)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(xb)
            loss = loss_fn(out, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().cpu().item()) * xb.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_continuous(model, loader, return_ids=False):
    model.eval()
    preds = []
    ids = []
    non_block = torch.cuda.is_available()
    for batch in loader:
        if return_ids:
            xb, idb = batch
        else:
            xb, _yb = batch
        xb = xb.to(device, non_blocking=non_block)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(xb)
        out = out.float().detach().cpu().numpy().reshape(-1)
        preds.append(out)
        if return_ids:
            ids.extend(list(idb))
    preds = np.concatenate(preds, axis=0)
    if return_ids:
        return preds, np.array(ids)
    return preds




## === cell 9
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    @staticmethod
    def _apply_coef(X, coef):
        X = X.astype(float)
        return np.digitize(X, bins=coef, right=False).astype(int)

    def _kappa_loss(self, coef, X, y):
        X_p = self._apply_coef(X, coef)
        ll = metrics.cohen_kappa_score(
            y.astype(int), X_p.astype(int), weights="quadratic"
        )
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.57, 1.57, 2.57, 3.57]
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method="nelder-mead"
        )
        print("Optimized kappa:", -loss_partial(self.coef_["x"]))

    def predict(self, X, coef):
        return self._apply_coef(np.copy(X), coef)

    def coefficients(self):
        return self.coef_["x"]




## === cell 10
def run_cv_and_submit(n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof_pred = np.zeros(len(df), dtype=np.float32)
    tst_pred_sum = np.zeros(len(test_df), dtype=np.float32)

    epochs = 2
    lr = 1e-4
    loss_fn = nn.MSELoss()

    tst_ids = None

    for fold, (train_idx, valid_idx) in enumerate(skf.split(df, df["diagnosis"])):
        print(f"\n=== fold {fold+1}/{n_splits} ===")
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_valid = df.iloc[valid_idx].reset_index(drop=True)

        model = build_model()
        model.load_state_dict(BASE_MODEL_STATE, strict=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scaler = make_scaler()

        dl_train, dl_valid, dl_test = make_loaders(
            df_train, df_valid, test_df, bs, fold=fold, epoch=0
        )

        last_val_pred_fold_epoch = None
        for epoch in range(epochs):
            if hasattr(dl_train.dataset, "set_epoch"):
                dl_train.dataset.set_epoch(epoch)

            tr_loss = train_one_epoch(model, dl_train, optimizer, loss_fn, scaler)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            last_val_pred_fold_epoch = predict_continuous(
                model, dl_valid, return_ids=False
            )
            val_y_fold = df_valid["diagnosis"].values
            print(
                f"fold={fold} epoch={epoch} train_loss={tr_loss:.4f} val_qk_round={qk_np(last_val_pred_fold_epoch, val_y_fold):.4f}"
            )

        val_pred_fold = last_val_pred_fold_epoch
        oof_pred[valid_idx] = val_pred_fold.astype(np.float32)

        tst_pred_fold, tst_ids_fold = predict_continuous(
            model, dl_test, return_ids=True
        )
        if tst_ids is None:
            tst_ids = tst_ids_fold
        tst_pred_sum += tst_pred_fold.astype(np.float32)

        del model, optimizer, scaler, dl_train, dl_valid, dl_test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tst_pred = tst_pred_sum / float(n_splits)

    opt = OptimizedRounder()
    opt.fit(oof_pred, df["diagnosis"].values.astype(int))
    coefficients = opt.coefficients()

    tst_cls = opt.predict(tst_pred, coefficients)

    sub = pd.DataFrame({"id_code": tst_ids, "diagnosis": tst_cls.astype(int)})
    sub = sub.set_index("id_code").loc[test_df["id_code"].values].reset_index()
    sub.to_csv("submission.csv", index=False)
    print("\ndone, wrote submission.csv")
    print("coefficients:", coefficients)
    return sub, coefficients


_ = run_cv_and_submit(n_splits=5)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_56/2463792090.py in <cell line: 0>()
     76 
     77 
---> 78 _ = run_cv_and_submit(n_splits=5)

/tmp/ipykernel_56/2463792090.py in run_cv_and_submit(n_splits)
     63 
     64     opt = OptimizedRounder()
---> 65     opt.fit(oof_pred, df["diagnosis"].values.astype(int))
     66     coefficients = opt.coefficients()
     67 

/tmp/ipykernel_56/1957358044.py in fit(self, X, y)
     18         loss_partial = partial(self._kappa_loss, X=X, y=y)
     19         initial_coef = [0.57, 1.57, 2.57, 3.57]
---> 20         self.coef_ = sp.optimize.minimize(
     21             loss_partial, initial_coef, method="nelder-mead"
     22         )

/usr/local/lib/python3.11/dist-packages/scipy/optimize/_minimize.py in minimize(fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
    724 
    725     if meth == 'nelder-mead':
--> 726         res = _minimize_neldermead(fun, x0, args, callback, bounds=bounds,
    727                                    **options)
    728     elif meth == 'powell':

/usr/local/lib/python3.11/dist-packages/scipy/optimize/_optimize.py in _minimize_neldermead(func, x0, args, callback, maxiter, maxfev, disp, return_all, initial_simplex, xatol, fatol, adaptive, bounds, **unknown_options)
    863                 if bounds is not None:
    864                     xe = np.clip(xe, lower_bound, upper_bound)
--> 865                 fxe = func(xe)
    866 
    867                 if fxe < fxr:

/usr/local/lib/python3.11/dist-packages/scipy/optimize/_optimize.py in function_wrapper(x, *wrapper_args)
    540         ncalls[0] += 1
    541         # A copy of x is sent to the user function (gh13740)
--> 542         fx = function(np.copy(x), *(wrapper_args + args))
    543         # Ideally, we'd like to a have a true scalar returned from f(x). For
    544         # backwards-compatibility, also allow np.array([1.3]),

/tmp/ipykernel_56/1957358044.py in _kappa_loss(self, coef, X, y)
      9 
     10     def _kappa_loss(self, coef, X, y):
---> 11         X_p = self._apply_coef(X, coef)
     12         ll = metrics.cohen_kappa_score(
     13             y.astype(int), X_p.astype(int), weights="quadratic"

/tmp/ipykernel_56/1957358044.py in _apply_coef(X, coef)
      6     def _apply_coef(X, coef):
      7         X = X.astype(float)
----> 8         return np.digitize(X, bins=coef, right=False).astype(int)
      9 
     10     def _kappa_loss(self, coef, X, y):

/usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py in digitize(x, bins, right)
   5723     mono = _monotonicity(bins)
   5724     if mono == 0:
-> 5725         raise ValueError("bins must be monotonically increasing or decreasing")
   5726 
   5727     # this is backwards because the arguments below are swapped

ValueError: bins must be monotonically increasing or decreasing
