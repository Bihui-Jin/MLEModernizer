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

0.9020970980040276

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the environment/import issues by removing notebook magics and making the code work without `fastai`, which is not installed here (that’s the root cause of the first failure and all downstream NameErrors). I keep the core intent identical: load an EfficientNet model, load provided weights, run inference on the test images, then apply fixed optimized rounding thresholds to produce integer diagnoses 0–4. I also fix broken paths and missing imports (`os`, `collections`, `scipy` usage), and replace the nonexistent `../input/kaggle-public/abcdef.pth` with a safe local search for the weights file; if no weights exist, the script still produce a valid submission (but score likely be low). Finally, I ensure `submission.csv` is written with exactly the required columns `id_code,diagnosis` and correct row alignment with `test.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the submission is effectively constant/wrongly calibrated because the EfficientNet-B5 is being used without the correct pretrained competition weights (your code “searches” for a `.pth` but the dataset may not contain one), so logits are near-random and thresholding collapses to a single class. To move the score upward with minimal changes and without changing the model/training, I (1) ensure we load valid weights if present by searching more robustly and requiring a reasonable key match, and (2) if no valid weights are found, fit only the 4 rounding thresholds on a small validation split using the training CSV (no model training) to better align continuous outputs to QWK. This keeps the same core inference pipeline and just fixes the two biggest causes of a 0.0: missing weights and mis-calibrated thresholds. The output submission format and paths remain unchanged, and the script still always writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model running with essentially-random weights (or mismatched weights), then being thresholded into a near-constant class. To move the score upward with minimal risk and without changing the model architecture or any training loop, I (1) make weight-file discovery prefer APTOS/EfficientNet checkpoints and verify shape-compatible loading, and (2) when weights are not found/usable, fall back to a deterministic ImageNet-pretrained EfficientNet-B0 from `torchvision` (still EfficientNet, no training) so predictions are at least meaningful rather than random. I also fit the 4 rounding thresholds on a larger validation subset (still no training) even when weights are loaded, because QWK is highly sensitive to calibration and this is a minimal post-processing change aligned to the metric. The script still run end-to-end and always write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import re
import math
import json
import glob
import collections
from functools import partial

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

from sklearn import metrics

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)



## === cell 1
"""
EfficientNet implementation (as provided), with minimal fixes required to run.
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

        assert "s" in options and (len(options["s"]) == 2)
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
            and self._block_args.stride == [1, 1]
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
                    input_filters=block_args.output_filters, stride=[1, 1]
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
md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
md_ef = md_ef.to(DEVICE)
md_ef.eval()
print("Model created:", type(md_ef).__name__)



## === cell 3
os.makedirs("models", exist_ok=True)


def find_weight_file():
    candidates = []
    candidates += glob.glob("/kaggle/input/**/**.pth", recursive=True)
    candidates += glob.glob("/kaggle/input/**/**.pt", recursive=True)
    candidates += glob.glob("../input/**/**.pth", recursive=True)
    candidates += glob.glob("../input/**/**.pt", recursive=True)
    candidates += glob.glob("/kaggle/working/**/**.pth", recursive=True)
    candidates += glob.glob("/kaggle/working/**/**.pt", recursive=True)
    candidates += glob.glob("./models/*.pth")
    candidates += glob.glob("./models/*.pt")
    candidates += glob.glob("./*.pth")
    candidates += glob.glob("./*.pt")

    prefer = []
    for p in candidates:
        bn = os.path.basename(p).lower()
        if any(
            k in bn for k in ["aptos", "blind", "retina", "efficientnet", "b5", "dr"]
        ):
            prefer.append(p)
    ordered = prefer + [p for p in candidates if p not in prefer]

    ordered = sorted(
        ordered,
        key=lambda x: os.path.getsize(x) if os.path.exists(x) else 0,
        reverse=True,
    )
    return ordered[0] if ordered else None


WEIGHTS_PATH = find_weight_file()
print("Found weights candidate:", WEIGHTS_PATH)


def _extract_state_dict(obj):
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        return obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    return None


def try_load_weights(model, weights_path, min_key_match_ratio=0.2):
    if not weights_path:
        return False

    try:
        raw = torch.load(weights_path, map_location="cpu")
    except Exception as e:
        print("Failed reading weights:", repr(e))
        return False

    sd = _extract_state_dict(raw)
    if sd is None or not isinstance(sd, dict):
        print("Weights file did not contain a state_dict-like dict.")
        return False

    cleaned = {}
    for k, v in sd.items():
        nk = k.replace("module.", "")
        nk = nk.replace("model.", "")
        cleaned[nk] = v
    sd = cleaned

    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    sd_keys = set(sd.keys())
    match = len(model_keys & sd_keys)
    ratio = match / max(1, len(model_keys))
    print(f"State_dict key match: {match}/{len(model_keys)} = {ratio:.3f}")

    if ratio < min_key_match_ratio:
        print("Key match ratio too low; treating as not loaded.")
        return False

    if "_fc.weight" in sd and "_fc.weight" in model_sd:
        if tuple(sd["_fc.weight"].shape) != tuple(model_sd["_fc.weight"].shape):
            print(
                "Head shape mismatch:",
                tuple(sd["_fc.weight"].shape),
                "vs",
                tuple(model_sd["_fc.weight"].shape),
            )
    try:
        res = model.load_state_dict(sd, strict=False)
        print("Loaded weights with strict=False.")
        if hasattr(res, "missing_keys"):
            print(
                "Missing keys:",
                len(res.missing_keys),
                "Unexpected keys:",
                len(res.unexpected_keys),
            )
        return True
    except Exception as e:
        print("Failed loading weights:", repr(e))
        return False


WEIGHTS_LOADED = try_load_weights(md_ef, WEIGHTS_PATH)
print("WEIGHTS_LOADED:", WEIGHTS_LOADED)




## === cell 4
def resolve_base_dir():
    p1 = "/kaggle/input/aptos2019-blindness-detection"
    p2 = "../input/aptos2019-blindness-detection"
    p3 = "/kaggle/data/aptos2019-blindness-detection"
    for p in (p1, p2, p3):
        if os.path.exists(p):
            return p
    for p in glob.glob("/kaggle/**/aptos2019-blindness-detection", recursive=True):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not locate aptos2019-blindness-detection directory")


BASE_DIR = resolve_base_dir()
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")


def get_df():
    train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    train_df["path"] = train_df["id_code"].map(
        lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.png")
    )
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
    return train_df, test_df


train_df, test_df = get_df()
print(train_df.shape, test_df.shape)
print(test_df.head())



## === cell 5
IMG_SIZE = 456
BATCH_SIZE = 8  # inference only

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_tensor(path, img_size=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(arr)




## === cell 6
def batch_iter(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


test_paths = [
    os.path.join(TEST_IMG_DIR, f"{idc}.png") for idc in test_df["id_code"].values
]
missing = [p for p in test_paths if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images; first missing: {missing[0]}"
    )
print("Test images:", len(test_paths))




## === cell 7
def qwk_score(y_true, y_pred_int):
    y_pred_int = np.clip(np.asarray(y_pred_int).astype(int), 0, 4)
    return metrics.cohen_kappa_score(y_true, y_pred_int, weights="quadratic")




## === cell 8
@torch.no_grad()
def predict_paths(model, paths, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    model.eval()
    preds = np.zeros((len(paths),), dtype=np.float32)
    for batch_idx, batch_paths in enumerate(batch_iter(paths, batch_size)):
        xb = torch.stack(
            [load_image_tensor(p, img_size=img_size) for p in batch_paths]
        ).to(DEVICE)
        out = model(xb).view(-1)
        preds[batch_idx * batch_size : batch_idx * batch_size + len(batch_paths)] = (
            out.detach().float().cpu().numpy()
        )
    return preds




## === cell 9
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

    def predict(self, X, coef):
        X = np.asarray(X).reshape(-1)
        coef = np.asarray(coef).reshape(-1)
        X_p = np.zeros_like(X, dtype=np.int64)
        X_p[X < coef[0]] = 0
        X_p[(X >= coef[0]) & (X < coef[1])] = 1
        X_p[(X >= coef[1]) & (X < coef[2])] = 2
        X_p[(X >= coef[2]) & (X < coef[3])] = 3
        X_p[X >= coef[3]] = 4
        return X_p

    def coefficients(self):
        return self.coef_




## === cell 10
def fit_rounding_thresholds(model, train_df, val_size=800, seed=42, img_size=IMG_SIZE):
    df = train_df.copy()
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_val = df.iloc[:val_size].copy()

    df_val = df_val[df_val["path"].map(os.path.exists)].reset_index(drop=True)
    if len(df_val) < 64:
        print("Not enough validation images found to fit thresholds; using defaults.")
        return np.array([0.60, 1.60, 2.60, 3.60], dtype=np.float32)

    y_true = df_val["diagnosis"].values.astype(int)
    val_paths = df_val["path"].values.tolist()
    y_cont = predict_paths(model, val_paths, batch_size=BATCH_SIZE, img_size=img_size)

    opt = OptimizedRounder()
    coef = np.array([0.60, 1.60, 2.60, 3.60], dtype=np.float32)

    def eval_coef(c):
        c = np.sort(c)
        pred = opt.predict(y_cont, c)
        return qwk_score(y_true, pred)

    best = eval_coef(coef)
    print("Initial val QWK:", float(best), "coef:", coef.tolist())

    steps = [0.25, 0.10, 0.05]
    for step in steps:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for delta in (-step, step):
                    cand = coef.copy()
                    cand[i] = cand[i] + delta
                    cand = np.sort(cand)
                    if not (cand[0] < cand[1] < cand[2] < cand[3]):
                        continue
                    score = eval_coef(cand)
                    if score > best:
                        best = score
                        coef = cand
                        improved = True
        print("Step", step, "best val QWK:", float(best), "coef:", coef.tolist())

    return coef.astype(np.float32)


FALLBACK_MODEL = None
FALLBACK_IMG_SIZE = 224

if not WEIGHTS_LOADED:
    try:
        import torchvision
        from torchvision import models as tvm

        try:
            weights_enum = tvm.EfficientNet_B0_Weights.DEFAULT
            b0 = tvm.efficientnet_b0(weights=weights_enum)
        except Exception:
            b0 = tvm.efficientnet_b0(pretrained=True)

        in_features = b0.classifier[-1].in_features
        b0.classifier[-1] = nn.Linear(in_features, 1)
        FALLBACK_MODEL = b0.to(DEVICE).eval()
        print(
            "Using fallback torchvision efficientnet_b0 (ImageNet) with linear head (random)."
        )
    except Exception as e:
        print("Fallback model not available:", repr(e))
        FALLBACK_MODEL = None

if WEIGHTS_LOADED:
    INFER_MODEL = md_ef
    INFER_IMG_SIZE = IMG_SIZE
else:
    INFER_MODEL = FALLBACK_MODEL if FALLBACK_MODEL is not None else md_ef
    INFER_IMG_SIZE = FALLBACK_IMG_SIZE if FALLBACK_MODEL is not None else IMG_SIZE

COEFS = fit_rounding_thresholds(
    INFER_MODEL, train_df, val_size=800, seed=42, img_size=INFER_IMG_SIZE
)
print("Using coefficients:", COEFS.tolist())



## === cell 11
test_pred_cont = predict_paths(
    INFER_MODEL, test_paths, batch_size=BATCH_SIZE, img_size=INFER_IMG_SIZE
)
print(
    "Test pred stats:",
    float(test_pred_cont.min()),
    float(test_pred_cont.max()),
    float(test_pred_cont.mean()),
)




## === cell 12
def run_subm(test_df, test_pred_cont, coefficients, out_path="submission.csv"):
    opt = OptimizedRounder()
    tst_pred = opt.predict(test_pred_cont, coefficients)
    subm = pd.DataFrame(
        {"id_code": test_df["id_code"].values, "diagnosis": tst_pred.astype(int)}
    )
    subm.to_csv(out_path, index=False)
    print("Wrote", out_path, "shape:", subm.shape)
    print(subm.head())
    return subm


run_subm(
    test_df=test_df,
    test_pred_cont=test_pred_cont,
    coefficients=COEFS,
    out_path="submission.csv",
)
assert os.path.exists("submission.csv") and os.path.getsize("submission.csv") > 0
print("Done.")
