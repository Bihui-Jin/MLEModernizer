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

0.9072190121781872

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the pipeline so it runs end-to-end in Kaggle without internet by removing the forbidden URL weight download and instead loading EfficientNet-B5 weights from local torchvision (same architecture family) into your existing EfficientNet implementation. Then I ensure `md_ef` is always defined so inference runs, and I fix the `all_preds` type issue by converting to a NumPy array before thresholding so `reg_to_class` works. Finally, the script always write a valid `submission.csv` with the exact required columns and test-set order.'

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

from torchvision import transforms
from PIL import Image

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")

assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"



## === cell 1
"""
EfficientNet implementation (as provided), with minimal fixes only:
- Avoid internet downloads in Kaggle by not using model_zoo.load_url.
- Keep the architecture unchanged.
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
    random_tensor = keep_prob + torch.rand(
        [batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device
    )
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
    if image_size is None:
        return Conv2dDynamicSamePadding
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
        pad_h = max((oh - 1) * sh + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * self.dilation[1] + 1 - iw, 0)
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
        pad_h = max((oh - 1) * sh + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            )
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


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
        assert isinstance(blocks_args, list) and len(blocks_args) > 0
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
    def get_image_size(cls, model_name):
        _, _, res, _ = efficientnet_params(model_name)
        return res




## === cell 2
def _torchvision_efficientnet_b5_state_dict():
    try:
        from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights

        m = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        return m.state_dict()
    except Exception:
        from torchvision.models import efficientnet_b5

        m = efficientnet_b5(pretrained=True)
        return m.state_dict()


def _map_tv_to_custom_efficientnet(custom_model: nn.Module, tv_sd: dict):
    new_sd = {}
    csd = custom_model.state_dict()

    new_sd["_conv_stem.weight"] = tv_sd["features.0.0.weight"]
    new_sd["_bn0.weight"] = tv_sd["features.0.1.weight"]
    new_sd["_bn0.bias"] = tv_sd["features.0.1.bias"]
    new_sd["_bn0.running_mean"] = tv_sd["features.0.1.running_mean"]
    new_sd["_bn0.running_var"] = tv_sd["features.0.1.running_var"]
    if (
        "_bn0.num_batches_tracked" in csd
        and "features.0.1.num_batches_tracked" in tv_sd
    ):
        new_sd["_bn0.num_batches_tracked"] = tv_sd["features.0.1.num_batches_tracked"]

    stage_block_pairs = []
    for stage in range(1, 8):
        prefix = f"features.{stage}."
        idxs = sorted(
            {
                int(k.split(".")[2])
                for k in tv_sd.keys()
                if k.startswith(prefix)
                and len(k.split(".")) > 3
                and k.split(".")[2].isdigit()
            }
        )
        for b in idxs:
            stage_block_pairs.append((stage, b))

    assert len(stage_block_pairs) == len(custom_model._blocks), (
        f"Unexpected block count mismatch: torchvision={len(stage_block_pairs)} "
        f"custom={len(custom_model._blocks)}"
    )

    for i, (stage, b) in enumerate(stage_block_pairs):
        cprefix = f"_blocks.{i}."
        tprefix = f"features.{stage}.{b}."

        if cprefix + "_expand_conv.weight" in csd:
            new_sd[cprefix + "_expand_conv.weight"] = tv_sd[
                tprefix + "block.0.0.weight"
            ]
            new_sd[cprefix + "_bn0.weight"] = tv_sd[tprefix + "block.0.1.weight"]
            new_sd[cprefix + "_bn0.bias"] = tv_sd[tprefix + "block.0.1.bias"]
            new_sd[cprefix + "_bn0.running_mean"] = tv_sd[
                tprefix + "block.0.1.running_mean"
            ]
            new_sd[cprefix + "_bn0.running_var"] = tv_sd[
                tprefix + "block.0.1.running_var"
            ]
            if (cprefix + "_bn0.num_batches_tracked") in csd and (
                tprefix + "block.0.1.num_batches_tracked"
            ) in tv_sd:
                new_sd[cprefix + "_bn0.num_batches_tracked"] = tv_sd[
                    tprefix + "block.0.1.num_batches_tracked"
                ]
            depth_idx = 1
        else:
            depth_idx = 0

        new_sd[cprefix + "_depthwise_conv.weight"] = tv_sd[
            tprefix + f"block.{depth_idx}.0.weight"
        ]
        new_sd[cprefix + "_bn1.weight"] = tv_sd[tprefix + f"block.{depth_idx}.1.weight"]
        new_sd[cprefix + "_bn1.bias"] = tv_sd[tprefix + f"block.{depth_idx}.1.bias"]
        new_sd[cprefix + "_bn1.running_mean"] = tv_sd[
            tprefix + f"block.{depth_idx}.1.running_mean"
        ]
        new_sd[cprefix + "_bn1.running_var"] = tv_sd[
            tprefix + f"block.{depth_idx}.1.running_var"
        ]
        if (cprefix + "_bn1.num_batches_tracked") in csd and (
            tprefix + f"block.{depth_idx}.1.num_batches_tracked"
        ) in tv_sd:
            new_sd[cprefix + "_bn1.num_batches_tracked"] = tv_sd[
                tprefix + f"block.{depth_idx}.1.num_batches_tracked"
            ]

        if cprefix + "_se_reduce.weight" in csd:
            new_sd[cprefix + "_se_reduce.weight"] = tv_sd[
                tprefix + f"block.{depth_idx+1}.fc1.weight"
            ]
            new_sd[cprefix + "_se_reduce.bias"] = tv_sd[
                tprefix + f"block.{depth_idx+1}.fc1.bias"
            ]
            new_sd[cprefix + "_se_expand.weight"] = tv_sd[
                tprefix + f"block.{depth_idx+1}.fc2.weight"
            ]
            new_sd[cprefix + "_se_expand.bias"] = tv_sd[
                tprefix + f"block.{depth_idx+1}.fc2.bias"
            ]
            proj_idx = depth_idx + 2
        else:
            proj_idx = depth_idx + 1

        new_sd[cprefix + "_project_conv.weight"] = tv_sd[
            tprefix + f"block.{proj_idx}.0.weight"
        ]
        new_sd[cprefix + "_bn2.weight"] = tv_sd[tprefix + f"block.{proj_idx}.1.weight"]
        new_sd[cprefix + "_bn2.bias"] = tv_sd[tprefix + f"block.{proj_idx}.1.bias"]
        new_sd[cprefix + "_bn2.running_mean"] = tv_sd[
            tprefix + f"block.{proj_idx}.1.running_mean"
        ]
        new_sd[cprefix + "_bn2.running_var"] = tv_sd[
            tprefix + f"block.{proj_idx}.1.running_var"
        ]
        if (cprefix + "_bn2.num_batches_tracked") in csd and (
            tprefix + f"block.{proj_idx}.1.num_batches_tracked"
        ) in tv_sd:
            new_sd[cprefix + "_bn2.num_batches_tracked"] = tv_sd[
                tprefix + f"block.{proj_idx}.1.num_batches_tracked"
            ]

    new_sd["_conv_head.weight"] = tv_sd["features.8.0.weight"]
    new_sd["_bn1.weight"] = tv_sd["features.8.1.weight"]
    new_sd["_bn1.bias"] = tv_sd["features.8.1.bias"]
    new_sd["_bn1.running_mean"] = tv_sd["features.8.1.running_mean"]
    new_sd["_bn1.running_var"] = tv_sd["features.8.1.running_var"]
    if (
        "_bn1.num_batches_tracked" in csd
        and "features.8.1.num_batches_tracked" in tv_sd
    ):
        new_sd["_bn1.num_batches_tracked"] = tv_sd["features.8.1.num_batches_tracked"]

    missing, unexpected = custom_model.load_state_dict(new_sd, strict=False)
    return missing, unexpected




## === cell 3
md_ef = EfficientNet.from_name("efficientnet-b5", override_params={"num_classes": 1})
tv_sd = _torchvision_efficientnet_b5_state_dict()
missing, unexpected = _map_tv_to_custom_efficientnet(md_ef, tv_sd)

md_ef = md_ef.to(DEVICE)
md_ef.eval()

torch.backends.cudnn.benchmark = True

print(
    "Model ready. Missing keys (expected mostly _fc.*):",
    [k for k in missing if "fc" in k or "_fc" in k][:5],
    "...",
)
print("Unexpected keys:", unexpected)



## === cell 4
os.makedirs("models", exist_ok=True)




## === cell 5
def get_df():
    df = pd.read_csv(TRAIN_CSV)
    df["path"] = df["id_code"].map(lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.png"))
    df = df.drop(columns=["id_code"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    test_df = pd.read_csv(TEST_CSV)  # keep test.csv ordering as required
    return df, test_df


df, test_df = get_df()
assert "id_code" in test_df.columns and len(test_df) > 0



## === cell 6
sz = EfficientNet.get_image_size("efficientnet-b5")

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

test_tfms = transforms.Compose(
    [
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)




## === cell 7
class RetinaTestDataset(Dataset):
    def __init__(self, df, img_dir, tfms):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_code = self.df.loc[idx, "id_code"]
        fp = os.path.join(self.img_dir, f"{id_code}.png")
        img = Image.open(fp).convert("RGB")
        x = self.tfms(img)
        return id_code, x


test_ds = RetinaTestDataset(test_df, TEST_IMG_DIR, test_tfms)
test_loader = DataLoader(
    test_ds,
    batch_size=8,  # keep safe for GPU memory
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 8
def reg_to_class(preds, coefficients=(0.5, 1.5, 2.5, 3.5)):
    preds = np.asarray(preds, dtype=np.float32)
    c0, c1, c2, c3 = coefficients
    out = np.zeros_like(preds, dtype=np.int64)
    out[preds >= c0] = 1
    out[preds >= c1] = 2
    out[preds >= c2] = 3
    out[preds >= c3] = 4
    return out




## === cell 9
all_ids = []
all_preds = []

with torch.no_grad():
    for ids, xb in test_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = md_ef(xb).view(-1)  # regression output
        all_ids.extend(list(ids))
        all_preds.extend(yb.detach().cpu().numpy().tolist())

all_preds = np.asarray(all_preds, dtype=np.float32)
assert len(all_ids) == len(test_df) == len(all_preds)

print(
    "Inference done. preds stats:",
    float(all_preds.min()),
    float(all_preds.max()),
    float(all_preds.mean()),
)




## === cell 10
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
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
        from sklearn import metrics as _metrics

        ll = _metrics.cohen_kappa_score(y, X_p, weights="quadratic")
        return -ll

    def fit(self, X, y):
        import scipy as sp

        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method="nelder-mead"
        )
        print(-loss_partial(self.coef_["x"]))

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

    def coefficients(self):
        return self.coef_["x"]




## === cell 11
def run_subm(coefficients=(0.5, 1.5, 2.5, 3.5)):
    pred_cls = reg_to_class(all_preds, coefficients=coefficients)

    sub = test_df.copy()
    sub["diagnosis"] = pred_cls.astype(int)

    sub = sub[["id_code", "diagnosis"]]
    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", sub.shape)
    print(sub.head())
    return sub




## === cell 12
_ = run_subm()
