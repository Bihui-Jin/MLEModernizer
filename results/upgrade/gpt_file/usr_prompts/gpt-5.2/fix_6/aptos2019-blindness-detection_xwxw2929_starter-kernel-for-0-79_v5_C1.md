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

0.901645703725582

# 6. Current score

0.63186

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the unavailable fastai dependency and replace it with a minimal pure-PyTorch inference pipeline that preserves your core idea: EfficientNet-B5 with a single regression output plus optimized rounding into classes 0–4. I also fix the missing imports (`os`, `collections`) and remove the broken pretrained-weight download/fastai load calls, instead loading a local `.pth` if present (and otherwise still producing a valid submission via an untrained model so you always get a CSV). Finally, I ensure test image ordering matches `sample_submission.csv` and that the output file is written as `submission.csv` with the required `id_code,diagnosis` columns.'
- What this solution (achieved 0.515) has done: 'The timeout is dominated by repeatedly decoding/resizing PNGs in Python and by per-iteration overhead in the training loop. I keep the exact same EfficientNet model, loss, and two-stage training logic, but speed up input handling by (1) adding an in-memory image tensor cache for the training/validation sets (safe because they are only ~3.3k images) and (2) optimizing the DataLoader settings (more workers, persistent workers, prefetch, pinned memory). I also enable CuDNN benchmarking (safe here because input size is fixed at 456×456) and use `optimizer.zero_grad(set_to_none=True)` to reduce per-step overhead without changing math. Test images are not fully cached to avoid memory blowups; they still benefit from the faster loader settings.'
- What this solution (achieved 0.62699) has done: 'I keep your EfficientNet-B5 regression + optimized-rounding pipeline unchanged, but fix one key training/metric mismatch: right now the rounder thresholds are optimized on a single random split that may not reflect the class distribution well, which can cap kappa. I switch the split to a stratified split by `diagnosis` (same fraction/seed) so the validation set is representative and the optimized thresholds generalize better, typically improving QWK without changing model architecture or training loops. I also add simple clipping of regression outputs to the valid label range [0, 4] before threshold optimization and inference; this doesn’t change the semantics (still regression + rounding) but stabilizes the optimizer and reduces extreme outliers harming kappa. Everything else (model, loss, two-stage training, data paths, submission format) stays the same and still produces `submission.csv`.'
- What this solution (achieved 0.63186) has done: 'Your current 0.62699 score is well below the 0.9016 target, so we should improve performance with minimal, low-risk changes that preserve your EfficientNet-B5 regression + optimized rounding approach. The biggest likely limiter is that the model is effectively training from scratch (no pretrained ImageNet weights are ever loaded), which severely caps achievable QWK; we add a safe in-notebook loader for locally available EfficientNet-B5 pretrained weights (common Kaggle inputs) and only fall back to your current behavior if none are found. We also fix a small skip-connection condition bug (`stride` is stored as a list) that slightly alters the intended EfficientNet behavior and can hurt accuracy, without changing the architecture. Finally, we keep your stratified split and clipping, and still write a valid `submission.csv` with the same required columns.'

# 9. Code solution

## === cell 0
import os
import re
import math
import json
import warnings
import collections
from functools import partial

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True



## === cell 1
"""
EfficientNet implementation (as provided) + small bugfixes:
- Keep core architecture identical.
- Avoid any external dependency on fastai.
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

        stride_is_1 = (self._block_args.stride == 1) or (
            isinstance(self._block_args.stride, (list, tuple))
            and self._block_args.stride[0] == 1
        )

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
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        return model




## === cell 2
BASE_DIR_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/input",
    "/kaggle/data",
]
BASE_DIR = None
for c in BASE_DIR_CANDIDATES:
    if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
        os.path.join(c, "train_images")
    ):
        BASE_DIR = c
        break
if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection dataset directory."
    )

print("Using BASE_DIR:", BASE_DIR)

train_csv = os.path.join(BASE_DIR, "train.csv")
test_csv = os.path.join(BASE_DIR, "test.csv")
sample_sub_csv = os.path.join(BASE_DIR, "sample_submission.csv")
train_img_dir = os.path.join(BASE_DIR, "train_images")
test_img_dir = os.path.join(BASE_DIR, "test_images")

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
sample_sub = pd.read_csv(sample_sub_csv)

test_df = test_df.merge(sample_sub[["id_code"]], on="id_code", how="right")
assert test_df.shape[0] == sample_sub.shape[0]

print(train_df.head())
print(test_df.head())



## === cell 3
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE = efficientnet_params("efficientnet-b5")[2]  # 456


def load_image_as_tensor(path, img_size=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(arr)


class ImageDataset(Dataset):
    def __init__(self, df, img_dir, with_labels, cache_images: bool = False):
        self.ids = df["id_code"].tolist()
        self.paths = [os.path.join(img_dir, f"{i}.png") for i in self.ids]
        self.with_labels = with_labels
        if with_labels:
            self.y = df["diagnosis"].astype(np.float32).values
        else:
            self.y = None

        self.cache_images = bool(cache_images)
        self._cache = {} if self.cache_images else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self._cache is not None:
            x = self._cache.get(idx)
            if x is None:
                x = load_image_as_tensor(self.paths[idx])
                self._cache[idx] = x
        else:
            x = load_image_as_tensor(self.paths[idx])

        if self.with_labels:
            return x, self.y[idx], self.ids[idx]
        return x, self.ids[idx]




## === cell 4
def make_train_val_split(df, val_frac=0.15, seed=42):
    rng = np.random.RandomState(seed)
    y = df["diagnosis"].astype(int).values
    trn_idx = []
    val_idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_val = int(round(len(cls_idx) * val_frac))
        val_idx.append(cls_idx[:n_val])
        trn_idx.append(cls_idx[n_val:])
    trn_idx = np.concatenate(trn_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(trn_idx)
    rng.shuffle(val_idx)
    return df.iloc[trn_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(
        drop=True
    )


train_trn_df, train_val_df = make_train_val_split(train_df, val_frac=0.15, seed=42)
print("Train/Val sizes:", train_trn_df.shape, train_val_df.shape)
print("Train class counts:\n", train_trn_df["diagnosis"].value_counts().sort_index())
print("Val class counts:\n", train_val_df["diagnosis"].value_counts().sort_index())

train_trn_ds = ImageDataset(
    train_trn_df, train_img_dir, with_labels=True, cache_images=True
)
train_val_ds = ImageDataset(
    train_val_df, train_img_dir, with_labels=True, cache_images=True
)

TRAIN_BS = 4 if torch.cuda.is_available() else 2
VAL_BS = 4 if torch.cuda.is_available() else 2

num_workers = min(4, (os.cpu_count() or 2))
pin = torch.cuda.is_available()

train_trn_loader = DataLoader(
    train_trn_ds,
    batch_size=TRAIN_BS,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin,
    drop_last=False,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)
train_val_loader = DataLoader(
    train_val_ds,
    batch_size=VAL_BS,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    drop_last=False,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)



## === cell 5
md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1).to(DEVICE)


def _try_load_pretrained_backbone_weights(model: EfficientNet):
    """
    Change (score improvement): previously no ImageNet-pretrained weights were ever loaded,
    so training starts from scratch and QWK is capped. This attempts to load a locally
    available EfficientNet-B5 pretrained checkpoint (common in Kaggle inputs). If not
    found, we keep prior behavior.
    """
    candidates = []
    roots = ["/kaggle/input", "/kaggle/data", "/kaggle/working"]
    exts = (".pth", ".pt")
    patterns = ("efficientnet", "b5")

    for r in roots:
        if not os.path.exists(r):
            continue
        for dirpath, dirnames, filenames in os.walk(r):
            rel_depth = dirpath[len(r) :].count(os.sep)
            if rel_depth > 3:
                dirnames[:] = []
                continue
            for fn in filenames:
                lfn = fn.lower()
                if lfn.endswith(exts) and all(p in lfn for p in patterns):
                    candidates.append(os.path.join(dirpath, fn))

    candidates.extend(
        [
            "/kaggle/input/kaggle-public/abcdef.pth",
            "/kaggle/input/abcdef.pth",
            "/kaggle/working/models/abcdef.pth",
            "/kaggle/working/abcdef.pth",
        ]
    )

    seen = set()
    candidates = [
        p
        for p in candidates
        if (p not in seen and not seen.add(p)) and os.path.exists(p)
    ]
    if len(candidates) == 0:
        return None

    candidates = sorted(candidates, key=lambda p: (len(p), p))

    def _extract_state_dict(ckpt):
        if isinstance(ckpt, dict):
            for key in ["state_dict", "model", "model_state_dict", "net", "weights"]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    return ckpt[key]
            if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
                return ckpt
        return ckpt if isinstance(ckpt, dict) else None

    current = model.state_dict()
    for pth in candidates[:15]:
        try:
            ckpt = torch.load(pth, map_location="cpu")
            sd = _extract_state_dict(ckpt)
            if sd is None:
                continue
            if any(k.startswith("module.") for k in sd.keys()):
                sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

            filtered = {}
            for k, v in sd.items():
                if k in current and current[k].shape == v.shape:
                    filtered[k] = v

            if len(filtered) < 50:
                continue

            missing, unexpected = model.load_state_dict(filtered, strict=False)
            print("Loaded pretrained backbone weights from:", pth)
            print(
                "Matched keys:",
                len(filtered),
                "Missing keys:",
                len(missing),
                "Unexpected keys:",
                len(unexpected),
            )
            return pth
        except Exception as e:
            continue
    return None


pretrained_backbone_path = _try_load_pretrained_backbone_weights(md_ef)

candidate_weight_paths = [
    "/kaggle/input/kaggle-public/abcdef.pth",
    "/kaggle/input/abcdef.pth",
    "/kaggle/working/models/abcdef.pth",
    "/kaggle/working/abcdef.pth",
]
weight_path = next((p for p in candidate_weight_paths if os.path.exists(p)), None)

if weight_path is not None:
    ckpt = torch.load(weight_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if isinstance(state_dict, dict) and any(
        k.startswith("module.") for k in state_dict.keys()
    ):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = md_ef.load_state_dict(state_dict, strict=False)
    print("Loaded fine-tuned weights:", weight_path)
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
else:
    print(
        "No fine-tuned weights file found; will train within this notebook. "
        "If pretrained backbone weights were found above, training will start from that initialization."
    )




## === cell 6
def set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad = flag


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb, _ in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True).view(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def predict_regression(model, loader):
    model.eval()
    preds = []
    ys = []
    for xb, yb, _ in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        out = model(xb).view(-1).detach().cpu().numpy()
        preds.append(out)
        ys.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(ys)


if weight_path is None:
    set_requires_grad(md_ef, False)
    md_ef._fc.weight.requires_grad = True
    md_ef._fc.bias.requires_grad = True

    crit = nn.MSELoss()
    opt1 = torch.optim.Adam([p for p in md_ef.parameters() if p.requires_grad], lr=1e-2)

    for epoch in range(1, 2 + 1):
        tr_loss = train_one_epoch(md_ef, train_trn_loader, opt1, crit)
        val_pred, val_y = predict_regression(md_ef, train_val_loader)
        val_rmse = float(np.sqrt(np.mean((val_pred - val_y) ** 2)))
        print(
            f"[Stage1][Epoch {epoch}] train_loss={tr_loss:.4f} val_rmse={val_rmse:.4f}"
        )

    set_requires_grad(md_ef, True)
    opt2 = torch.optim.Adam(md_ef.parameters(), lr=1e-4)

    for epoch in range(1, 2 + 1):
        tr_loss = train_one_epoch(md_ef, train_trn_loader, opt2, crit)
        val_pred, val_y = predict_regression(md_ef, train_val_loader)
        val_rmse = float(np.sqrt(np.mean((val_pred - val_y) ** 2)))
        print(
            f"[Stage2][Epoch {epoch}] train_loss={tr_loss:.4f} val_rmse={val_rmse:.4f}"
        )

md_ef.eval()




## === cell 7
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = None

    def _kappa_loss(self, coef, X, y):
        X_p = self.predict(X, coef)
        ll = metrics.cohen_kappa_score(y, X_p, weights="quadratic")
        return -ll

    def fit(self, X, y):
        import scipy as sp

        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method="nelder-mead"
        )
        print("Best kappa (val):", -loss_partial(self.coef_["x"]))

    def predict(self, X, coef):
        X = np.asarray(X, dtype=np.float32).reshape(-1)
        X_p = np.zeros_like(X, dtype=np.int64)
        X_p[X >= coef[0]] = 1
        X_p[X >= coef[1]] = 2
        X_p[X >= coef[2]] = 3
        X_p[X >= coef[3]] = 4
        return X_p

    def coefficients(self):
        return self.coef_["x"]




## === cell 8
val_pred, val_y = predict_regression(md_ef, train_val_loader)
val_pred = np.clip(val_pred, 0.0, 4.0)

opt = OptimizedRounder()
opt.fit(val_pred, val_y.astype(int))
best_coef = opt.coefficients()
print("Optimized coefficients:", best_coef)



## === cell 9
test_ds = ImageDataset(test_df, test_img_dir, with_labels=False, cache_images=False)

num_workers = min(4, (os.cpu_count() or 2))
pin = torch.cuda.is_available()

test_loader = DataLoader(
    test_ds,
    batch_size=8,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

all_ids = []
all_preds = []

with torch.no_grad():
    for xb, ids in test_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        out = md_ef(xb).view(-1)  # regression values
        all_preds.append(out.detach().cpu().numpy())
        all_ids.extend(ids)

all_preds = np.concatenate(all_preds, axis=0)
all_preds = np.clip(all_preds, 0.0, 4.0)

pred_df = pd.DataFrame({"id_code": all_ids, "pred": all_preds})

pred_df = sample_sub[["id_code"]].merge(pred_df, on="id_code", how="left")
assert pred_df["pred"].isna().sum() == 0

diagnosis = opt.predict(pred_df["pred"].values, best_coef).astype(int)

submission = pd.DataFrame(
    {"id_code": pred_df["id_code"].values, "diagnosis": diagnosis}
)
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
