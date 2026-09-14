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

0.9077260590880342

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the import/runtime breakage caused by using fastai v1 APIs (`ImageList`, `Learner`, etc.) in an environment that has fastai v2, by replacing the data pipeline/training/inference glue with a small pure-PyTorch loop while keeping your EfficientNet model, regression output (num_classes=1), MSE loss, and threshold-based rounding logic intact. I also fix the `SEED` scoping issue by moving all seeds/constants into the first executable cell so later cells can rely on them. To ensure the notebook always produces a valid `submission.csv`, I implement a deterministic train/valid split, a minimal image dataset/loader, and a test prediction path that reads `test.csv` and writes the required columns. Finally, I keep your provided coefficients (score-calibration) unchanged to avoid unintended score shifts while making the pipeline run end-to-end.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with the model starting from random weights because `from_pretrained()` does not actually load ImageNet weights in this implementation, and the checkpoint path is a dummy (`models/abcdef.pth`) so the fallback 1-epoch training produces near-constant predictions. To move the score upward toward the target with minimal change, I keep your EfficientNet-B5 regression + MSE + threshold-rounding core logic intact, but (1) add real ImageNet pretrained weight loading via `torchvision`’s `efficientnet_b5` and copy its feature weights into your EfficientNet, and (2) set `global_params.image_size=None` to fix a shape mismatch caused by hardcoded static padding (300 vs your 224 input). This preserves the same training loop and inference semantics while making predictions meaningfully calibrated out-of-the-box, which should substantially increase QWK from 0.0. The script still writes a valid `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission alignment/format issue (e.g., NaNs in `diagnosis` due to ID mismatches) rather than pure modeling, so I first make the submission build directly from `test.csv` order (not via `sample_submission` mapping) and add strict checks that no NaNs exist and row counts match. Then, to improve score toward the target with minimal risk and without changing core model/training logic, I (a) clamp regression outputs to the valid label range before thresholding, and (b) (if SciPy is available) fit rounding thresholds on the existing validation split and use those for test, otherwise fall back to your fixed coefficients. These changes keep the EfficientNet-B5 regression + MSE + 1-epoch fallback training intact, but should eliminate the “all-missing/all-wrong” submission failure mode and yield a non-zero QWK. Finally, I keep paths and the overall pipeline structure unchanged and still write `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely coming from predictions collapsing to mostly a single class and/or miscalibrated regression outputs after the 1-epoch fallback, rather than a CSV format issue (your submission writing logic is already correct and has NaN/shape guards). To move the score upward toward the target with minimal change and without altering the core model/training approach, I (1) switch the input resolution to EfficientNet-B5’s native 456 and (2) match torchvision’s EfficientNet preprocessing (center-crop style resize and correct interpolation) so the loaded ImageNet weights are actually useful. I also (3) ensure we use the exact `test.csv` order (already done) and keep the same regression+MSE+thresholding logic, only improving the data pipeline consistency that directly affects QWK. These changes are small, keep your architecture/training loop intact, and should lift you far above 0.0 toward the target band.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission alignment/type issue causing Kaggle to effectively treat predictions as invalid or constant, rather than only “model quality.” I make two minimal, score-relevant fixes: (1) ensure the submission `diagnosis` column is the exact integer dtype Kaggle expects (not object/float), and (2) remove any possibility of ID/order mismatch by building the submission strictly from `test.csv` and asserting exact row/ID equality against `sample_submission.csv`. I also keep your existing EfficientNet-B5 regression + MSE + threshold-rounding core logic intact, but make coefficient fitting deterministic and robust by falling back cleanly when SciPy is absent (no partial fit that could yield bad coefs). These changes should move you upward from 0.0 toward the target without altering the model/training semantics.'

# 9. Code solution

## === cell 0
import os
import re
import math
import json
import random
import collections
from functools import partial

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F

from sklearn import metrics
from sklearn.metrics import cohen_kappa_score

try:
    import scipy as sp
except Exception:
    sp = None

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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


class Identity(nn.Module):
    def __init__(self):
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
        return [BlockDecoder._decode_block_string(s) for s in string_list]


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
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        return model




## === cell 2
md_ef = EfficientNet.from_pretrained(
    "efficientnet-b5",
    num_classes=1,
).to(device)

md_ef._global_params = md_ef._global_params._replace(image_size=None)

print("model ready")




## === cell 3
def _load_torchvision_efficientnet_b5_imagenet_into_custom(custom_model: EfficientNet):
    try:
        import torchvision
        from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
    except Exception as e:
        print(
            "torchvision not available; skipping ImageNet weight init. Error:", repr(e)
        )
        return False

    tv = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
    tv_sd = tv.state_dict()

    custom_sd = custom_model.state_dict()
    mapped = {}
    missing = []
    copied = 0

    stem_map = {
        "_conv_stem.weight": "features.0.0.weight",
        "_bn0.weight": "features.0.1.weight",
        "_bn0.bias": "features.0.1.bias",
        "_bn0.running_mean": "features.0.1.running_mean",
        "_bn0.running_var": "features.0.1.running_var",
        "_bn0.num_batches_tracked": "features.0.1.num_batches_tracked",
    }

    head_map = {
        "_conv_head.weight": "features.8.0.weight",
        "_bn1.weight": "features.8.1.weight",
        "_bn1.bias": "features.8.1.bias",
        "_bn1.running_mean": "features.8.1.running_mean",
        "_bn1.running_var": "features.8.1.running_var",
        "_bn1.num_batches_tracked": "features.8.1.num_batches_tracked",
    }

    for k_custom, k_tv in {**stem_map, **head_map}.items():
        if (
            k_custom in custom_sd
            and k_tv in tv_sd
            and custom_sd[k_custom].shape == tv_sd[k_tv].shape
        ):
            mapped[k_custom] = tv_sd[k_tv]
            copied += 1
        else:
            missing.append((k_custom, k_tv))

    for k_custom, v_custom in custom_sd.items():
        if k_custom.startswith("_fc."):
            continue
        if k_custom in mapped:
            continue
        candidates = []
        for k_tv, v_tv in tv_sd.items():
            if "classifier" in k_tv:
                continue
            if v_tv.shape == v_custom.shape:
                if ("bn" in k_custom and ("bn" in k_tv or "norm" in k_tv)) or (
                    "conv" in k_custom and "weight" in k_tv
                ):
                    candidates.append(k_tv)
        if len(candidates) == 1:
            mapped[k_custom] = tv_sd[candidates[0]]
            copied += 1

    if copied == 0:
        print("Could not map any torchvision weights; keeping random init.")
        return False

    custom_sd.update(mapped)
    custom_model.load_state_dict(custom_sd, strict=False)
    print(f"Loaded ImageNet init from torchvision for {copied} tensors (strict=False).")
    if len(missing) > 0:
        print("Some expected stem/head keys were not mapped (ok):", missing[:3], "...")
    return True


_ = _load_torchvision_efficientnet_b5_imagenet_into_custom(md_ef)




## === cell 4
os.makedirs("models", exist_ok=True)




## === cell 5
def get_df():
    base_image_dir = os.path.join(
        "/", "kaggle", "input", "aptos2019-blindness-detection"
    )
    train_dir = os.path.join(base_image_dir, "train_images")
    test_dir = os.path.join(base_image_dir, "test_images")

    df = pd.read_csv(os.path.join(base_image_dir, "train.csv"))
    df["path"] = df["id_code"].map(lambda x: os.path.join(train_dir, f"{x}.png"))
    df = df.drop(columns=["id_code"])
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle

    test_df = pd.read_csv(os.path.join(base_image_dir, "test.csv"))
    test_df["path"] = test_df["id_code"].map(
        lambda x: os.path.join(test_dir, f"{x}.png")
    )

    sample_sub = pd.read_csv(os.path.join(base_image_dir, "sample_submission.csv"))
    return df, test_df, sample_sub, base_image_dir


df, test_df, sample_sub, base_image_dir = get_df()
print(df.shape, test_df.shape, sample_sub.shape)




## === cell 6
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _pil_resize_shorter_side(img, shorter_side, resample=Image.BICUBIC):
    w, h = img.size
    if w == 0 or h == 0:
        return img
    if w < h:
        new_w = shorter_side
        new_h = int(round(h * (shorter_side / w)))
    else:
        new_h = shorter_side
        new_w = int(round(w * (shorter_side / h)))
    return img.resize((new_w, new_h), resample=resample)


def _pil_center_crop(img, size):
    w, h = img.size
    th, tw = size, size
    i = max(0, int(round((h - th) / 2.0)))
    j = max(0, int(round((w - tw) / 2.0)))
    return img.crop((j, i, j + tw, i + th))


def _load_image_rgb(path):
    img = Image.open(path).convert("RGB")
    return img


def _to_tensor_normalized(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(arr)


class RetinoDataset(Dataset):
    def __init__(self, df, size=456, train=True):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = _load_image_rgb(row["path"])

        img = _pil_resize_shorter_side(img, int(round(self.size * 256 / 224)))
        img = _pil_center_crop(img, self.size)

        x = _to_tensor_normalized(img)

        if self.train:
            y = np.float32(row["diagnosis"])
            return x, torch.tensor([y], dtype=torch.float32)
        else:
            return x


n = len(df)
valid_sz = int(round(0.2 * n))
valid_df = df.iloc[:valid_sz].reset_index(drop=True)
train_df = df.iloc[valid_sz:].reset_index(drop=True)
print("train/valid:", train_df.shape, valid_df.shape)

bs = 16
sz = 456

train_loader = DataLoader(
    RetinoDataset(train_df, size=sz, train=True),
    batch_size=bs,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
valid_loader = DataLoader(
    RetinoDataset(valid_df, size=sz, train=True),
    batch_size=bs,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
test_loader = DataLoader(
    RetinoDataset(test_df, size=sz, train=False),
    batch_size=bs,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 7
def qk_np(y_pred_cont, y_true_int):
    y_pred_int = np.round(y_pred_cont).astype(np.int64)
    y_true_int = y_true_int.astype(np.int64)
    return cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic")




## === cell 8
ckpt_name = "abcdef"
ckpt_path = os.path.join("models", f"{ckpt_name}.pth")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(md_ef.parameters(), lr=1e-3)

if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location=device)
    md_ef.load_state_dict(state)
    print("Loaded checkpoint:", ckpt_path)
else:
    print(
        f"Warning: checkpoint not found at {ckpt_path}. Training a small fallback model so submission is meaningful."
    )

    md_ef.train()
    for epoch in range(1):  # keep 1 epoch as in original fallback
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds = md_ef(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)

        md_ef.eval()
        with torch.no_grad():
            vp, vt = [], []
            for xb, yb in valid_loader:
                xb = xb.to(device, non_blocking=True)
                preds = md_ef(xb).detach().cpu().numpy().reshape(-1)
                vp.append(preds)
                vt.append(yb.numpy().reshape(-1))
            vp = np.concatenate(vp)
            vt = np.concatenate(vt)
            print(
                f"epoch {epoch+1} train_loss={running/len(train_df):.5f} valid_qwk(round)={qk_np(vp, vt):.5f}"
            )

    torch.save(md_ef.state_dict(), ckpt_path)
    print("Saved fallback checkpoint:", ckpt_path)




## === cell 9
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
        ll = metrics.cohen_kappa_score(y, X_p.astype(int), weights="quadratic")
        return -ll

    def fit(self, X, y):
        if sp is None:
            raise RuntimeError(
                "scipy is required for OptimizedRounder.fit but is not available."
            )
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
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_["x"]




## === cell 10
def _predict_continuous(model, loader):
    model.eval()
    preds_all = []
    with torch.no_grad():
        for xb in loader:
            if isinstance(xb, (list, tuple)):
                xb = xb[0]
            xb = xb.to(device, non_blocking=True)
            preds = model(xb).detach().cpu().numpy().reshape(-1)
            preds_all.append(preds)
    preds_np = np.concatenate(preds_all, axis=0)
    preds_np = np.clip(preds_np, 0.0, 4.0)
    return preds_np


def _fit_coefficients_on_valid(
    model, valid_loader, default_coef=(0.57, 1.37, 2.57, 3.57)
):
    if sp is None:
        print("SciPy not available; using default coefficients:", default_coef)
        return np.array(default_coef, dtype=np.float32)

    model.eval()
    vp, vt = [], []
    with torch.no_grad():
        for xb, yb in valid_loader:
            xb = xb.to(device, non_blocking=True)
            preds = model(xb).detach().cpu().numpy().reshape(-1)
            vp.append(preds)
            vt.append(yb.numpy().reshape(-1))
    vp = np.clip(np.concatenate(vp), 0.0, 4.0)
    vt = np.concatenate(vt).astype(np.int64)

    opt = OptimizedRounder()
    opt.fit(vp, vt)
    coef = opt.coefficients().astype(np.float32)
    coef = np.sort(coef)
    print("Using fitted coefficients:", coef.tolist())
    print(
        "valid_qwk(optimized_round):",
        cohen_kappa_score(vt, opt.predict(vp, coef).astype(int), weights="quadratic"),
    )
    return coef


def run_subm(model, test_df, sample_sub, coefficients=(0.57, 1.37, 2.57, 3.57)):
    coef = _fit_coefficients_on_valid(model, valid_loader, default_coef=coefficients)

    preds_np = _predict_continuous(model, test_loader)

    opt = OptimizedRounder()
    tst_pred = opt.predict(preds_np, coef)
    tst_pred = np.clip(tst_pred, 0, 4).astype(np.int64)

    sub = pd.DataFrame(
        {
            "id_code": test_df["id_code"].astype(str).values,
            "diagnosis": pd.Series(tst_pred, dtype="int64").values,
        }
    )

    assert sub.shape[0] == test_df.shape[0], "Row count mismatch vs test.csv"
    assert sub["diagnosis"].isna().sum() == 0, "NaNs in diagnosis would cause 0.0 score"
    assert sub["id_code"].isna().sum() == 0, "NaNs in id_code"
    assert set(sub.columns.tolist()) == {"id_code", "diagnosis"}
    assert np.issubdtype(
        sub["diagnosis"].dtype, np.integer
    ), "diagnosis must be integer dtype"

    ss_ids = sample_sub["id_code"].astype(str).values
    assert sub.shape[0] == sample_sub.shape[0], "Mismatch vs sample_submission rows"
    assert np.array_equal(
        sub["id_code"].values, ss_ids
    ), "ID/order mismatch vs sample_submission"

    sub.to_csv("submission.csv", index=False)
    print("done, wrote submission.csv with shape:", sub.shape)
    print(sub.head())
    assert os.path.exists("submission.csv")


run_subm(md_ef, test_df, sample_sub)
