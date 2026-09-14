# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
from torch.utils import model_zoo
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn import metrics

import scipy as sp

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True


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
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[0] + 1 - iw, 0)
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
        pad_w = max((oh - 1) * self.stride[1] + (kw - 1) * self.dilation[0] + 1 - iw, 0)
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


class Identity(nn.Module):
    def __init__(
        self,
    ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


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


url_map = {
    "efficientnet-b0": "http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth",
    "efficientnet-b1": "http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth",
    "efficientnet-b2": "http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth",
    "efficientnet-b3": "http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth",
    "efficientnet-b4": "http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth",
    "efficientnet-b5": "http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth",
}


def load_pretrained_weights(model, model_name, load_fc=True):
    """Loads pretrained weights, and downloads if loading for the first time."""
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
        if (
            self.id_skip
            and self._block_args.stride == [1]
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
                    input_filters=block_args.output_filters, stride=[1]
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
        num_models = 8
        valid_models = ["efficientnet-b" + str(i) for i in range(num_models)]
        model_name2 = model_name.replace("_", "-")
        if model_name not in valid_models and model_name2 not in valid_models:
            raise ValueError("model_name should be one of: " + ", ".join(valid_models))




## === cell 1
sz = 456
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

md_ef = EfficientNet.from_name(
    "efficientnet-b5", override_params={"num_classes": 5, "image_size": sz}
)
md_ef = md_ef.to(device)



## === cell 2
os.makedirs("models", exist_ok=True)
os.makedirs("cache", exist_ok=True)




## === cell 3
def _find_base_dir():
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "../input/aptos2019-blindness-detection",
        "../data/aptos2019-blindness-detection",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for root in ["/kaggle/input", "/kaggle/data"]:
        if os.path.exists(root):
            for name in os.listdir(root):
                p = os.path.join(root, name)
                if os.path.isdir(p) and "aptos2019-blindness-detection" in name:
                    return p
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection directory in expected locations."
    )


def get_df():
    base_image_dir = _find_base_dir()
    train_dir = os.path.join(base_image_dir, "train_images")
    df = pd.read_csv(os.path.join(base_image_dir, "train.csv"))
    df["path"] = df["id_code"].map(lambda x: os.path.join(train_dir, f"{x}.png"))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    test_df = pd.read_csv(os.path.join(base_image_dir, "test.csv"))
    return df, test_df, base_image_dir


df, test_df, base_image_dir = get_df()
print("base_image_dir:", base_image_dir)
print(df.head())
print(test_df.head())



## === cell 4
bs = 16  # keep identical batch size

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_IS_CUDA = torch.cuda.is_available()
_NUM_WORKERS = min(8, max(2, (os.cpu_count() or 2)))
_PREFETCH_FACTOR = 2 if _NUM_WORKERS > 0 else None



## === cell 5
from PIL import Image


def _load_and_preprocess_image_to_chw_float32(path, size):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return arr  # numpy float32 CHW


class TestImageDataset(Dataset):
    def __init__(
        self, df, base_dir, folder="test_images", suffix=".png", size=224, preload=True
    ):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.folder = folder
        self.suffix = suffix
        self.size = size

        self.ids = self.df["id_code"].tolist()
        self.paths = [
            os.path.join(self.base_dir, self.folder, f"{id_code}{self.suffix}")
            for id_code in self.ids
        ]

        self.preload = bool(preload)
        self._x = None
        if self.preload:
            xs = np.empty((len(self.paths), 3, self.size, self.size), dtype=np.float32)
            for i, p in enumerate(self.paths):
                xs[i] = _load_and_preprocess_image_to_chw_float32(p, self.size)
            self._x = torch.from_numpy(xs)  # CPU float32 tensor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self._x is None:
            x = torch.from_numpy(
                _load_and_preprocess_image_to_chw_float32(self.paths[idx], self.size)
            )
        else:
            x = self._x[idx]
        return x, self.ids[idx]


class TrainImageDataset(Dataset):
    def __init__(self, df, size=224, cache_dir="cache/train_tensors", use_cache=True):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.paths = self.df["path"].tolist()
        self.labels = self.df["diagnosis"].astype(np.int64).tolist()
        self.use_cache = bool(use_cache)
        self.cache_dir = cache_dir
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.paths)

    def _cache_path(self, img_path):
        base = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.cache_dir, f"{base}_sz{self.size}.pt")

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        if self.use_cache:
            cp = self._cache_path(img_path)
            if os.path.exists(cp):
                x = torch.load(cp, map_location="cpu")
            else:
                x = torch.from_numpy(
                    _load_and_preprocess_image_to_chw_float32(img_path, self.size)
                )
                torch.save(x, cp)
        else:
            x = torch.from_numpy(
                _load_and_preprocess_image_to_chw_float32(img_path, self.size)
            )
        y = int(self.labels[idx])
        return x, y


test_ds = TestImageDataset(
    test_df[["id_code"]],
    base_image_dir,
    folder="test_images",
    suffix=".png",
    size=sz,
    preload=True,
)
test_dl = DataLoader(
    test_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=0,  # preloaded -> no workers needed (avoids IPC overhead)
    pin_memory=_IS_CUDA,
)

print("test samples:", len(test_ds), "num_workers:", 0, "preloaded:", True)




## === cell 6
def qk(y_pred, y):
    y_hat = torch.argmax(y_pred, dim=1)
    k = cohen_kappa_score(
        y.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), weights="quadratic"
    )
    return torch.tensor(k)




## === cell 7
class _SimpleLearner:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    @torch.no_grad()
    def get_preds(self):
        self.model.eval()
        all_logits = []
        all_ids = []
        for xb, ids in self.test_loader:
            xb = xb.to(self.device, non_blocking=True)
            logits = self.model(xb)
            all_logits.append(logits.detach().cpu())
            all_ids.extend(list(ids))
        return torch.cat(all_logits, dim=0), all_ids


learn = _SimpleLearner(md_ef, test_dl, device)




## === cell 8
def _try_load_local_efficientnet_weights(
    model, model_name="efficientnet-b5", load_fc=False
):
    fn = os.path.basename(url_map[model_name])
    candidates = []
    for root in [
        os.path.expanduser("~/.cache/torch/hub/checkpoints"),
        os.path.expanduser("~/.cache/torch/checkpoints"),
        "/root/.cache/torch/hub/checkpoints",
        "/kaggle/working",
        "/kaggle/input",
        "/kaggle/data",
    ]:
        candidates.append(os.path.join(root, fn))
    for p in candidates:
        if os.path.exists(p) and os.path.isfile(p):
            try:
                state_dict = torch.load(p, map_location="cpu")
                if not load_fc:
                    state_dict.pop("_fc.weight", None)
                    state_dict.pop("_fc.bias", None)
                    res = model.load_state_dict(state_dict, strict=False)
                    print("Loaded local pretrained weights:", p)
                    print(
                        "Missing keys:",
                        len(res.missing_keys),
                        "Unexpected keys:",
                        len(res.unexpected_keys),
                    )
                else:
                    model.load_state_dict(state_dict, strict=True)
                    print("Loaded local pretrained weights (with fc):", p)
                return True
            except Exception as e:
                print("Failed local weight load from", p, "error:", repr(e))
    return False


weights_candidates = [
    os.path.join("models", "abcdef.pth"),
    os.path.join(base_image_dir, "abcdef.pth"),
    os.path.join("/kaggle/input", "abcdef.pth"),
]
loaded = False
for w in weights_candidates:
    if os.path.exists(w) and os.path.isfile(w):
        try:
            state = torch.load(w, map_location="cpu")
            if (
                isinstance(state, dict)
                and "state_dict" in state
                and isinstance(state["state_dict"], dict)
            ):
                state = state["state_dict"]
            if isinstance(state, dict):
                new_state = {}
                for k, v in state.items():
                    nk = k.replace("module.", "")
                    new_state[nk] = v
                missing, unexpected = md_ef.load_state_dict(new_state, strict=False)
                print("Loaded weights from:", w)
                print(
                    "Missing keys:", len(missing), "Unexpected keys:", len(unexpected)
                )
                loaded = True
                break
        except Exception as e:
            print("Failed to load weights from", w, "error:", repr(e))

if not loaded:
    print(
        "No external weights found; will train a small fixed schedule to avoid random predictions."
    )
    ok_local = _try_load_local_efficientnet_weights(
        md_ef, "efficientnet-b5", load_fc=False
    )
    if not ok_local:
        try:
            load_pretrained_weights(md_ef, "efficientnet-b5", load_fc=False)
            print(
                "Backbone initialized from ImageNet weights (fc remains 5-class head init)."
            )
        except Exception as e:
            print("Could not load ImageNet weights (offline/no cache). Error:", repr(e))




## === cell 9
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    @staticmethod
    def _apply_coef(X, coef):
        coef = np.sort(np.asarray(coef, dtype=np.float64))
        return np.digitize(X, coef, right=False).astype(np.int64)

    def _kappa_loss(self, coef, X, y):
        X_p = self._apply_coef(X, coef)
        ll = metrics.cohen_kappa_score(y, X_p, weights="quadratic")
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method="nelder-mead"
        )
        print("Optimized QWK:", -loss_partial(self.coef_["x"]))

    def predict(self, X, coef):
        return self._apply_coef(X, coef)

    def coefficients(self):
        return np.sort(np.asarray(self.coef_["x"], dtype=np.float64))




## === cell 10
def _stratified_split(df, valid_frac=0.2, seed=42, label_col="diagnosis"):
    rng = np.random.RandomState(seed)
    valid_idx = []
    for c, g in df.groupby(label_col):
        idxs = g.index.values.copy()
        rng.shuffle(idxs)
        n_valid = max(1, int(round(len(idxs) * valid_frac)))
        valid_idx.append(idxs[:n_valid])
    valid_idx = np.concatenate(valid_idx)
    train_idx = np.setdiff1d(df.index.values, valid_idx, assume_unique=False)
    return train_idx, valid_idx


def _make_class_weights(train_df, num_classes=5):
    counts = train_df["diagnosis"].value_counts().to_dict()
    total = float(len(train_df))
    w = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        n = float(counts.get(c, 0.0))
        w[c] = total / max(1.0, n)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def _predict_expected_scores(model, data_loader, device, num_classes=5):
    model.eval()
    class_idx = torch.arange(num_classes, device=device).float()
    all_exp = []
    all_y = []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            prob = torch.softmax(logits.float(), dim=1)
            exp_score = (prob * class_idx).sum(dim=1)
            all_exp.append(exp_score.detach().cpu().numpy())
            all_y.append(yb.detach().cpu().numpy())
    return np.concatenate(all_exp), np.concatenate(all_y)


def _set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad = bool(flag)


def _train_one_split(
    model, train_df, valid_df, device, size, batch_size, epochs=6, lr=3e-4
):
    train_ds = TrainImageDataset(
        train_df, size=size, cache_dir="cache/train_tensors", use_cache=True
    )
    valid_ds = TrainImageDataset(
        valid_df, size=size, cache_dir="cache/valid_tensors", use_cache=True
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=_NUM_WORKERS,
        pin_memory=_IS_CUDA,
        persistent_workers=(_NUM_WORKERS > 0),
        prefetch_factor=_PREFETCH_FACTOR if _NUM_WORKERS > 0 else None,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=_IS_CUDA,
        persistent_workers=(_NUM_WORKERS > 0),
        prefetch_factor=_PREFETCH_FACTOR if _NUM_WORKERS > 0 else None,
    )

    warmup_epochs = 1
    finetune_epochs = max(0, int(epochs) - warmup_epochs)

    class_w = _make_class_weights(train_df, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)

    _set_requires_grad(model, False)
    for p in model._fc.parameters():
        p.requires_grad = True

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, warmup_epochs // 2), gamma=0.3
    )

    for ep in range(warmup_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())

        scheduler.step()
        exp_sc, y_true = _predict_expected_scores(
            model, valid_dl, device, num_classes=5
        )
        base_coef = [0.5, 1.5, 2.5, 3.5]
        y_hat = OptimizedRounder().predict(exp_sc, base_coef).astype(int)
        qwk_val = metrics.cohen_kappa_score(y_true, y_hat, weights="quadratic")
        print(
            f"warmup epoch {ep+1}/{warmup_epochs} - lr={scheduler.get_last_lr()[0]:.2e} - train_loss={total_loss/max(1,len(train_dl)):.4f} - val_qwk(base_thr)={qwk_val:.4f}"
        )

    _set_requires_grad(model, True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, finetune_epochs // 2), gamma=0.3
    )

    for ep in range(finetune_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())

        scheduler.step()

        exp_sc, y_true = _predict_expected_scores(
            model, valid_dl, device, num_classes=5
        )
        base_coef = [0.5, 1.5, 2.5, 3.5]
        y_hat = OptimizedRounder().predict(exp_sc, base_coef).astype(int)
        qwk_val = metrics.cohen_kappa_score(y_true, y_hat, weights="quadratic")
        print(
            f"finetune epoch {ep+1}/{finetune_epochs} - lr={scheduler.get_last_lr()[0]:.2e} - train_loss={total_loss/max(1,len(train_dl)):.4f} - val_qwk(base_thr)={qwk_val:.4f}"
        )

    return train_dl, valid_dl


def _make_stratified_folds(df, n_splits=3, seed=42, label_col="diagnosis"):
    rng = np.random.RandomState(seed)
    fold_ids = -np.ones(len(df), dtype=np.int64)
    for c, g in df.groupby(label_col):
        idxs = g.index.values.copy()
        rng.shuffle(idxs)
        for i, idx in enumerate(idxs):
            fold_ids[idx] = i % n_splits
    assert (fold_ids >= 0).all()
    return fold_ids


final_coef = [0.5, 1.5, 2.5, 3.5]
if not loaded:
    n_folds = 3
    folds = _make_stratified_folds(df, n_splits=n_folds, seed=42, label_col="diagnosis")

    oof_exp = np.zeros(len(df), dtype=np.float32)
    oof_y = df["diagnosis"].astype(np.int64).values.copy()

    for fold in range(n_folds):
        tr_df = df.loc[folds != fold].reset_index(drop=True)
        va_df = df.loc[folds == fold].reset_index(drop=True)

        _train_one_split(
            md_ef,
            tr_df,
            va_df,
            device=device,
            size=sz,
            batch_size=bs,
            epochs=6,
            lr=3e-4,
        )

        va_ds = TrainImageDataset(
            va_df, size=sz, cache_dir="cache/valid_tensors", use_cache=True
        )
        va_dl = DataLoader(
            va_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=_NUM_WORKERS,
            pin_memory=_IS_CUDA,
            persistent_workers=(_NUM_WORKERS > 0),
            prefetch_factor=_PREFETCH_FACTOR if _NUM_WORKERS > 0 else None,
        )
        exp_va, y_va = _predict_expected_scores(md_ef, va_dl, device, num_classes=5)

        va_orig_idx = df.index.values[folds == fold]
        oof_exp[va_orig_idx] = exp_va.astype(np.float32)

        base_y_hat = (
            OptimizedRounder().predict(exp_va, [0.5, 1.5, 2.5, 3.5]).astype(int)
        )
        qwk_fold = metrics.cohen_kappa_score(y_va, base_y_hat, weights="quadratic")
        print(f"fold {fold+1}/{n_folds} - val_qwk(base_thr)={qwk_fold:.4f}")

    optR = OptimizedRounder()
    optR.fit(oof_exp, oof_y)
    final_coef = list(optR.coefficients())
    print("Using optimized coefficients (OOF):", final_coef)
else:
    print("External weights loaded; using default coefficients:", final_coef)




## === cell 11
def run_subm(learn, test_df, coefficients=[0.5, 1.5, 2.5, 3.5]):
    opt = OptimizedRounder()

    preds, ids = learn.get_preds()

    prob = torch.softmax(preds.float(), dim=1)
    class_idx_cpu = torch.arange(5, device=prob.device).float()
    exp_score = (prob * class_idx_cpu).sum(dim=1)

    coefficients = list(np.sort(np.asarray(coefficients, dtype=np.float64)))
    tst_pred = opt.predict(exp_score.detach().cpu().numpy(), coefficients).astype(int)

    sub = test_df.copy()
    sub = sub[["id_code"]].copy()
    sub["diagnosis"] = tst_pred
    sub.to_csv("submission.csv", index=False)
    print("done: wrote submission.csv with", len(sub), "rows")
    return sub


sub = run_subm(learn=learn, test_df=test_df, coefficients=final_coef)
print(sub.head())
print(
    "submission exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv"),
)
print(pd.read_csv("submission.csv").head())
