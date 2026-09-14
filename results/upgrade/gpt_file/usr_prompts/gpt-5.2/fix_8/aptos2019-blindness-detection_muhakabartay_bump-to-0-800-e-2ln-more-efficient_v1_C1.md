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

0.9039714717836792

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the import/runtime failures by removing the broken fastai dependency (your environment has fastai v2, while the code is written for fastai v1) and replacing only the data/learner plumbing with a tiny PyTorch inference wrapper that preserves your EfficientNet model code and the exact rounding/threshold submission logic. I also prevent any internet downloads for pretrained weights (Kaggle offline) and ensure the script finds and loads `abcdef.pth` if it exists, otherwise still produces a valid `submission.csv`. Finally, I make sure test image paths are resolved correctly and predictions are generated deterministically and written with the required columns and `.csv` suffix.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with the model running inference from random/untrained weights (or failing to load the real checkpoint), which produce near-random labels and very low kappa. I make two minimal, score-relevant fixes: (1) prevent any online pretrained-weight download (Kaggle offline) and instead initialize EfficientNet without downloading, then (2) make the checkpoint loader robust to common Kaggle formats (`state_dict`, `model_state_dict`, `net`, etc.) and to a mismatched head by safely loading with `strict=False` while still keeping your exact architecture and rounding thresholds. These changes preserve your model definition and submission logic, but greatly increase the chance that `abcdef.pth` actually loads and thus moves the score up toward your target. The script still always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the model is effectively untrained at inference time (checkpoint not found/loaded correctly), so the minimal way to move toward your 0.904 target is to make checkpoint discovery and loading more reliable without changing the model, preprocessing, or rounding logic. I (1) expand the weight search to include common filenames and locations while still preferring `abcdef.pth`, and (2) make the state_dict key-matching slightly more robust (handling `fc.` vs `_fc.` naming) so the head weights load when present. If no usable checkpoint is found, the script still produce a valid `submission.csv` exactly as before.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with “the checkpoint isn’t actually being loaded into the right parameter names”, so predictions come from (effectively) random weights. I keep your model, preprocessing, and fixed rounding thresholds unchanged, and only make the weight loading more robust by (1) trying additional common head key mappings (`_fc` vs `fc`), and (2) auto-detecting whether the saved checkpoint was for 5-class logits and, if so, converting it to your 1-output regression head by loading a kappa-neutral expected-value regressor derived from the 5 logits’ weights/bias (no change to inference semantics after the head). If no checkpoint is found or no compatible weights exist, it still write a valid `submission.csv` exactly as before. This should move the score upward toward your target without altering the core pipeline.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the submission is being produced but the model is effectively untrained at inference (no real weights loaded), so the smallest score-relevant improvement is to (1) reliably locate a real checkpoint under `/kaggle/input` (including common Kaggle dataset/workdir locations) and (2) load it more robustly by matching keys to your exact EfficientNet definition (especially handling `._fc` vs `.fc` and DataParallel prefixes). I keep your model, preprocessing, and fixed rounding coefficients unchanged, and I won’t add any new training or change thresholds. I also add a defensive fallback to search `/kaggle/working` and `/kaggle/input/**/models` so that if you attached a dataset with `abcdef.pth` it actually gets used. These changes should move the kappa up toward your target by ensuring you’re using the intended trained weights rather than random initialization.'

# 9. Code solution

## === cell 0
import os
import re
import math
import json
import collections
from functools import partial
from collections import Counter

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics
from sklearn.metrics import cohen_kappa_score

from PIL import Image

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



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
    """Block Decoder"""

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
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        return model




## === cell 2
md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1).to(device)



## === cell 3
os.makedirs("models", exist_ok=True)


def _find_weight_file():
    """
    Change (score-relevant): expand search locations (still prefers abcdef.pth) so the trained
    checkpoint is actually discovered in common Kaggle layouts; this directly impacts kappa.
    """
    preferred = "abcdef.pth"
    common_names = [
        preferred,
        "model.pth",
        "best.pth",
        "weights.pth",
        "checkpoint.pth",
        "final.pth",
        "model.pt",
        "weights.pt",
        "checkpoint.pt",
    ]

    search_roots = [
        "/kaggle/input",
        "/kaggle/working",
    ]

    candidates = []
    for base in search_roots:
        if not os.path.exists(base):
            continue
        for root, dirs, files in os.walk(base):
            for fn in files:
                lfn = fn.lower()
                if fn in common_names or lfn.endswith((".pth", ".pt")):
                    full = os.path.join(root, fn)
                    try:
                        if os.path.getsize(full) < 50_000:
                            continue
                    except OSError:
                        continue
                    candidates.append(full)

    for p in candidates:
        if os.path.basename(p) == preferred:
            return p

    models_candidates = [p for p in candidates if (os.sep + "models" + os.sep) in p]
    if models_candidates:
        return max(models_candidates, key=lambda p: os.path.getsize(p))

    if candidates:
        return max(candidates, key=lambda p: os.path.getsize(p))

    return None


found_weight = _find_weight_file()
if found_weight is not None:
    import shutil

    shutil.copy(found_weight, os.path.join("models", "abcdef.pth"))
    print(f"Found and copied weights: {found_weight} -> models/abcdef.pth")
else:
    print(
        "Warning: no usable .pth/.pt weights found under /kaggle/input or /kaggle/working. "
        "Will skip weight loading if missing."
    )




## === cell 4
def get_df():
    base_image_dir = "/kaggle/input/aptos2019-blindness-detection"
    train_dir = os.path.join(base_image_dir, "train_images")

    df = pd.read_csv(os.path.join(base_image_dir, "train.csv"))
    df["path"] = df["id_code"].map(lambda x: os.path.join(train_dir, f"{x}.png"))
    df = df.drop(columns=["id_code"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    test_df = pd.read_csv(os.path.join(base_image_dir, "test.csv"))
    return df, test_df


df, test_df = get_df()
print(df.head())
print(test_df.head())



## === cell 5
bs = 128
sz = 256  # kept from original; we'll use it for resizing images for model input

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_pil(img: Image.Image, size: int = 256) -> torch.Tensor:
    img = img.convert("RGB")
    img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(arr)


class AptosTestDataset(Dataset):
    def __init__(self, df, base_dir, folder="test_images", suffix=".png", size=256):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.folder = folder
        self.suffix = suffix
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_code = self.df.loc[idx, "id_code"]
        path = os.path.join(self.base_dir, self.folder, f"{id_code}{self.suffix}")
        img = Image.open(path)
        x = preprocess_pil(img, self.size)
        return x, id_code




## === cell 6
def qk(y_pred, y):
    yp = torch.round(y_pred).detach().cpu().numpy().reshape(-1)
    yt = y.detach().cpu().numpy().reshape(-1)
    return torch.tensor(
        cohen_kappa_score(yp, yt, weights="quadratic"), device=y_pred.device
    )




## === cell 7
def _extract_state_dict(maybe_ckpt):
    """
    Change (score-relevant): Robustly extract the actual state_dict from common checkpoint formats.
    """
    if isinstance(maybe_ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in maybe_ckpt and isinstance(maybe_ckpt[key], dict):
                return maybe_ckpt[key]
    return maybe_ckpt


def _strip_prefix(state_dict, prefix):
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        nk = k[len(prefix) :] if k.startswith(prefix) else k
        out[nk] = v
    return out


def _rename_key_prefix(state_dict, old_prefix, new_prefix):
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith(old_prefix):
            out[new_prefix + k[len(old_prefix) :]] = v
        else:
            out[k] = v
    return out


def _maybe_convert_fc5_to_fc1(state_dict):
    """
    Change (score-relevant, minimal): if checkpoint head is 5 logits (common),
    convert to 1-output regression head by taking expected value weights:
      y = sum_c c * logit_c
    """
    if not isinstance(state_dict, dict):
        return state_dict, False

    w_key = None
    b_key = None
    for wk, bk in [("_fc.weight", "_fc.bias"), ("fc.weight", "fc.bias")]:
        if wk in state_dict and bk in state_dict:
            w_key, b_key = wk, bk
            break
    if w_key is None:
        return state_dict, False

    W = state_dict[w_key]
    b = state_dict[b_key]
    if not (isinstance(W, torch.Tensor) and W.ndim == 2 and W.shape[0] == 5):
        return state_dict, False

    coeff = torch.tensor([0, 1, 2, 3, 4], dtype=W.dtype, device=W.device).view(1, 5)
    new_W = coeff @ W  # (1, in_features)
    new_b = (coeff.view(-1) * b).sum().view(1)

    state_dict = dict(state_dict)
    state_dict[w_key] = new_W
    state_dict[b_key] = new_b
    return state_dict, True


weights_path = os.path.join("models", "abcdef.pth")
if os.path.exists(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    state = _extract_state_dict(state)

    for pref in ["module.", "model.", "net.", "backbone.", "encoder."]:
        state = _strip_prefix(state, pref)

    tried = []
    loaded = False
    for variant in ["as_is", "fc_to__fc", "_fc_to_fc"]:
        sd = state
        if variant == "fc_to__fc":
            sd = _rename_key_prefix(sd, "fc.", "_fc.")
        elif variant == "_fc_to_fc":
            sd = _rename_key_prefix(sd, "_fc.", "fc.")

        sd, converted = _maybe_convert_fc5_to_fc1(sd)

        res = md_ef.load_state_dict(sd, strict=False)
        tried.append(
            (variant, converted, len(res.missing_keys), len(res.unexpected_keys))
        )

        if len(res.unexpected_keys) == 0 and len(res.missing_keys) < 50:
            loaded = True
            print(
                f"Loaded model weights (strict=False) using variant={variant}, converted_fc5_to_fc1={converted}"
            )
            print(
                "Missing keys:",
                len(res.missing_keys),
                "Unexpected keys:",
                len(res.unexpected_keys),
            )
            break

    if not loaded:
        best = min(tried, key=lambda t: (t[2], t[3]))
        variant, converted, mk, uk = best
        sd = state
        if variant == "fc_to__fc":
            sd = _rename_key_prefix(sd, "fc.", "_fc.")
        elif variant == "_fc_to_fc":
            sd = _rename_key_prefix(sd, "_fc.", "fc.")
        sd, converted2 = _maybe_convert_fc5_to_fc1(sd)
        res = md_ef.load_state_dict(sd, strict=False)
        print(
            f"Loaded model weights (best-effort) variant={variant}, converted_fc5_to_fc1={converted2} from {weights_path}"
        )
        print(
            "Missing keys:",
            len(res.missing_keys),
            "Unexpected keys:",
            len(res.unexpected_keys),
        )
else:
    print(
        "Warning: models/abcdef.pth not found; predictions will be from untrained model (valid CSV will still be created)."
    )

md_ef.to(device)
md_ef.eval()




## === cell 8
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




## === cell 9
def run_subm(
    model, test_df, coefficients=[0.57, 1.57, 2.57, 3.57], out_path="submission.csv"
):
    base_image_dir = "/kaggle/input/aptos2019-blindness-detection"

    ds = AptosTestDataset(
        test_df, base_dir=base_image_dir, folder="test_images", suffix=".png", size=sz
    )
    dl = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for xb, idb in dl:
            xb = xb.to(device, non_blocking=True).float()
            pred = model(xb).view(-1)  # regression output
            all_preds.append(pred.detach().cpu().numpy())
            all_ids.extend(list(idb))

    preds_np = np.concatenate(all_preds, axis=0).astype(np.float32)
    opt = OptimizedRounder()
    tst_pred = opt.predict(preds_np, coefficients).astype(np.int64)

    sub = pd.DataFrame({"id_code": all_ids, "diagnosis": tst_pred})
    sub = test_df[["id_code"]].merge(sub, on="id_code", how="left")
    sub["diagnosis"] = sub["diagnosis"].fillna(0).astype(int)

    sub.to_csv(out_path, index=False)
    print(f"done -> {out_path}, shape={sub.shape}, head:\n{sub.head()}")




## === cell 10
run_subm(
    model=md_ef,
    test_df=test_df,
    coefficients=[0.57, 1.57, 2.57, 3.57],
    out_path="submission.csv",
)
