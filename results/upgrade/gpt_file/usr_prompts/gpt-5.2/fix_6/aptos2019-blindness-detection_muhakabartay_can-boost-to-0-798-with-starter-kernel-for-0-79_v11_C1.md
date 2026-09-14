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

0.7858841696273159

# 6. Current score

0.88877

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the notebook-only magics and fix the import failures by making fastai optional (your environment doesn’t have it) and running inference directly in pure PyTorch instead. I also fix missing imports (`os`, `collections`) and remove the broken external weight copy step that references a non-existent path, replacing it with a safe local weight loader that only loads if the file exists. To keep the core logic intact, I preserve the EfficientNet-B5 architecture definition and the original “optimized rounding” post-processing, but use fixed coefficients (as in your code) since there is no validation-training loop here. Finally, I generate `submission.csv` with exactly the required `id_code,diagnosis` columns and correct row alignment to `test.csv`.'
- What this solution (achieved 0.02916) has done: 'Your 0.0 score is coming from using randomly initialized EfficientNet-B5 weights (no trained checkpoint is found), so the smallest legitimate way to move toward the target is to (1) load ImageNet-pretrained EfficientNet-B5 weights from the official URL using the already-imported `model_zoo`, (2) adapt the classifier head to your `num_classes=1` regression head by loading all matching backbone weights with `strict=False`, and (3) use a calibration of the regression output to the 0–4 label space before applying your existing fixed rounding thresholds. These changes keep the same architecture, loss/eval semantics, and inference flow, but replace random weights with meaningful pretrained features and align prediction scale with the kappa-style ordinal labels, which should substantially increase the score from 0.0 toward your target band. The rest of your pipeline (dataset, loader, submission alignment/format) is kept intact.'
- What this solution (achieved -0.01833) has done: 'Your score is far below the target, so the most likely issue is that the model is still effectively untrained for this task: ImageNet backbone features + a random regression head produces poorly scaled outputs and near-random ordinal predictions. To move toward the target with minimal core-logic changes, I (1) raise inference resolution to EfficientNet-B5’s native 456 and use higher-quality resizing, (2) add simple test-time augmentation (horizontal flip) averaged at logit level (no training change), and (3) replace the percentile-based scaling with a more stable monotonic mapping (sigmoid + fixed affine) that keeps outputs in a reasonable 0–4 range before applying your same fixed thresholds. These changes keep the architecture, inference loop structure, and rounding semantics intact, but should materially improve the kappa versus the current near-random behavior.'
- What this solution (achieved 0.2193) has done: 'Your score is far below the target, so the most likely cause is that the model is still effectively “not trained for APTOS”: you’re loading ImageNet weights but leaving the final regression head randomly initialized, which makes outputs poorly calibrated and near-random after thresholding. To move the score upward with minimal changes and without altering the model architecture/training loop, I (1) replace the fixed thresholds with thresholds fit on a small validation split using your existing `OptimizedRounder` objective (quadratic kappa), and (2) slightly stabilize prediction scaling by standardizing the raw logits using train-time statistics from the validation fold before applying the learned thresholds. This keeps the same EfficientNet-B5 regression + rounding core logic, but makes the rounding step task-calibrated, which is typically the biggest gain for kappa on this competition. The submission format/paths remain unchanged and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.88877) has done: 'Your current gap to the target is large (0.2193 vs 0.7859), and the biggest limiter is that you never actually train the regression head (and likely much of the network) on APTOS labels—so the validation-fitted thresholds can’t recover meaningful ordinal predictions. Keeping the EfficientNet-B5 regression+rounding core logic intact, I add a minimal fine-tuning step on the training set (MSE on labels 0–4), then keep your existing validation-based scaling + optimized rounding and the same test-time augmentation. To preserve evaluation semantics and avoid leakage, I fit thresholds only on a held-out validation split and train only on the remaining fold. I also keep runtime under the 600s budget by using a small number of epochs with frozen backbone then a brief unfreeze, without changing architecture or inference flow.'

# 9. Code solution

## === cell 0
import os
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
from torch.utils import model_zoo
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from sklearn import metrics

try:
    from fastai import *  # noqa: F401,F403
    from fastai.vision import *  # noqa: F401,F403

    FASTAI_AVAILABLE = True
except Exception:
    FASTAI_AVAILABLE = False

print("FASTAI_AVAILABLE =", FASTAI_AVAILABLE)
print("torch =", torch.__version__)
print("cuda available =", torch.cuda.is_available())

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



## === cell 1
"""
EfficientNet (PyTorch) implementation from the provided code.
Core logic preserved.
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
    """Swish activation function."""
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
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ["efficientnet_b" + str(i) for i in range(num_models)]
        if model_name.replace("-", "_") not in valid_models:
            raise ValueError("model_name should be one of: " + ", ".join(valid_models))




## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1).to(device)
md_ef.eval()

os.makedirs("models", exist_ok=True)

EFFICIENTNET_IMAGENET_URLS = {
    "efficientnet-b5": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth",
}


def _load_state_dict_safely(model, state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"Loaded with strict=False. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
    )


def try_load_weights(model, base_name="abcdef"):
    candidates = [
        os.path.join("models", f"{base_name}.pth"),
        os.path.join("models", base_name),
        f"{base_name}.pth",
        base_name,
    ]
    for path in candidates:
        if os.path.isfile(path):
            ckpt = torch.load(path, map_location=device)
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            _load_state_dict_safely(model, state)
            print("Loaded weights from:", path)
            return True
    return False


loaded_local = try_load_weights(md_ef, "abcdef")

if not loaded_local:
    try:
        url = EFFICIENTNET_IMAGENET_URLS["efficientnet-b5"]
        sd = model_zoo.load_url(url, map_location=device)
        _load_state_dict_safely(md_ef, sd)
        print("Loaded ImageNet pretrained weights from URL:", url)
    except Exception as e:
        print(
            "Failed to load ImageNet weights; falling back to random init. Error:",
            repr(e),
        )

md_ef.eval()



## === cell 3
BASE = "../input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

assert "id_code" in test_df.columns and len(test_df.columns) == 1
print("train:", train_df.shape, "test:", test_df.shape)
print(
    "train diagnosis distribution:\n", train_df["diagnosis"].value_counts().sort_index()
)



## === cell 4
IM_SIZE = 456

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class RetinopathyTestDataset(Dataset):
    def __init__(self, df, img_dir, img_size=456):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_code = self.df.loc[idx, "id_code"]
        path = os.path.join(self.img_dir, f"{id_code}.png")
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), resample=Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        x = torch.from_numpy(arr)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, id_code


class RetinopathyTrainDataset(Dataset):
    def __init__(self, df, img_dir, img_size=456):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_code = self.df.loc[idx, "id_code"]
        y = int(self.df.loc[idx, "diagnosis"])
        path = os.path.join(self.img_dir, f"{id_code}.png")
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), resample=Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        x = torch.from_numpy(arr)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, y


test_ds = RetinopathyTestDataset(test_df, TEST_IMG_DIR, img_size=IM_SIZE)
test_loader = DataLoader(
    test_ds,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




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

    def fit_cd(self, X, y, init_coef=None, n_iter=6, step=0.05):
        X = np.asarray(X, dtype=np.float32).reshape(-1)
        y = np.asarray(y, dtype=np.int64).reshape(-1)

        if init_coef is None:
            coef = np.array([0.8, 1.6, 2.4, 3.2], dtype=np.float32)
        else:
            coef = np.array(init_coef, dtype=np.float32)

        coef = np.sort(coef)
        coef = np.clip(coef, 0.01, 3.99)

        best_coef = coef.copy()
        best_loss = self._kappa_loss(best_coef, X, y)

        for _ in range(n_iter):
            for j in range(4):
                grid = best_coef[j] + np.arange(-10, 11, dtype=np.float32) * step
                for cand in grid:
                    trial = best_coef.copy()
                    trial[j] = float(cand)
                    trial = np.sort(trial)
                    trial = np.clip(trial, 0.01, 3.99)
                    loss = self._kappa_loss(trial, X, y)
                    if loss < best_loss:
                        best_loss = loss
                        best_coef = trial
        self.coef_ = best_coef.tolist()
        return self.coef_




## === cell 6
@torch.inference_mode()
def predict_test_tta(model, loader):
    model.eval()
    all_ids = []
    all_preds = []
    for xb, ids in loader:
        xb = xb.to(device, non_blocking=True)

        out1 = model(xb).view(-1)
        out2 = model(torch.flip(xb, dims=[3])).view(-1)  # horizontal flip (W dim)
        out = 0.5 * (out1 + out2)

        all_ids.extend(list(ids))
        all_preds.append(out.detach().float().cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    return np.array(all_ids), all_preds


@torch.inference_mode()
def predict_val_tta(model, loader):
    model.eval()
    all_y = []
    all_preds = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)

        out1 = model(xb).view(-1)
        out2 = model(torch.flip(xb, dims=[3])).view(-1)
        out = 0.5 * (out1 + out2)

        all_y.append(np.asarray(yb, dtype=np.int64))
        all_preds.append(out.detach().float().cpu().numpy())
    return np.concatenate(all_y, axis=0), np.concatenate(all_preds, axis=0)


def set_trainable_backbone(model, trainable: bool):
    for name, p in model.named_parameters():
        if name.startswith("_fc."):
            p.requires_grad = True
        else:
            p.requires_grad = bool(trainable)


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    running = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).float()

        pred = model(xb).view(-1)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        running += float(loss.detach().cpu()) * bs
        n += bs
    return running / max(1, n)


def eval_regression_mse(model, loader):
    model.eval()
    loss_fn = nn.MSELoss(reduction="mean")
    tot = 0.0
    n = 0
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()
            pred = model(xb).view(-1)
            loss = loss_fn(pred, yb)
            bs = xb.size(0)
            tot += float(loss.detach().cpu()) * bs
            n += bs
    return tot / max(1, n)


rng = np.random.RandomState(SEED)
idx = np.arange(len(train_df))
rng.shuffle(idx)
val_size = max(256, int(0.2 * len(train_df)))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

val_df = train_df.iloc[val_idx].reset_index(drop=True)
trn_df = train_df.iloc[trn_idx].reset_index(drop=True)

train_ds = RetinopathyTrainDataset(trn_df, TRAIN_IMG_DIR, img_size=IM_SIZE)
val_ds = RetinopathyTrainDataset(val_df, TRAIN_IMG_DIR, img_size=IM_SIZE)

train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

loss_fn = nn.MSELoss()

set_trainable_backbone(md_ef, trainable=False)
opt1 = torch.optim.Adam([p for p in md_ef.parameters() if p.requires_grad], lr=1e-3)

for epoch in range(
    1
):  # minimal epochs to keep runtime bounded while boosting above 0.2193
    tr_loss = train_one_epoch(md_ef, train_loader, opt1, loss_fn)
    va_mse = eval_regression_mse(md_ef, val_loader)
    print(f"[phase1 epoch {epoch+1}/1] train_mse={tr_loss:.4f} val_mse={va_mse:.4f}")

set_trainable_backbone(md_ef, trainable=True)
opt2 = torch.optim.Adam([p for p in md_ef.parameters() if p.requires_grad], lr=1e-4)

for epoch in range(1):  # minimal unfreeze to adapt features slightly
    tr_loss = train_one_epoch(md_ef, train_loader, opt2, loss_fn)
    va_mse = eval_regression_mse(md_ef, val_loader)
    print(f"[phase2 epoch {epoch+1}/1] train_mse={tr_loss:.4f} val_mse={va_mse:.4f}")

y_val, pred_val_raw = predict_val_tta(md_ef, val_loader)

mu = float(np.mean(pred_val_raw))
sigma = float(np.std(pred_val_raw) + 1e-6)


def to_score_space(raw):
    z = (raw - mu) / sigma
    s = (1.0 / (1.0 + np.exp(-z))) * 4.0
    return np.clip(s, 0.0, 4.0).astype(np.float32)


pred_val_scaled = to_score_space(pred_val_raw)

opt = OptimizedRounder()
coefficients = opt.fit_cd(
    pred_val_scaled, y_val, init_coef=[0.99, 1.99, 2.99, 3.99], n_iter=6, step=0.05
)
print("Fitted coefficients:", coefficients)

val_kappa = metrics.cohen_kappa_score(
    y_val, opt.predict(pred_val_scaled, coefficients).astype(int), weights="quadratic"
)
print("Validation QWK (for sanity):", val_kappa)

ids, preds_raw = predict_test_tta(md_ef, test_loader)

pred_map = {i: p for i, p in zip(ids.tolist(), preds_raw.tolist())}
ordered_preds_raw = np.array(
    [pred_map[i] for i in test_df["id_code"].values], dtype=np.float32
)

ordered_preds_scaled = to_score_space(ordered_preds_raw)
diag = opt.predict(ordered_preds_scaled, coefficients).astype(int)

sub = pd.DataFrame({"id_code": test_df["id_code"].values, "diagnosis": diag})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
assert (
    os.path.isfile("submission.csv")
    and sub.shape[0] == test_df.shape[0]
    and list(sub.columns) == ["id_code", "diagnosis"]
)
