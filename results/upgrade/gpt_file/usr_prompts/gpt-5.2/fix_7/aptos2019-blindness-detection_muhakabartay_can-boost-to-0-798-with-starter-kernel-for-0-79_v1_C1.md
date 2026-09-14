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

0.9114099378652444

# 6. Current score

0.73087

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the incompatible fastai v1 imports/calls that are causing immediate runtime errors in this environment, while preserving your core EfficientNet model definition and prediction-to-ordinal rounding logic. Since no trained weights file is actually available (the code looks for a placeholder `abcdef.pth`), I switch to a stable, lightweight PyTorch-only inference path that still produces a valid `submission.csv` with the required columns. To keep the semantics as close as possible to your existing post-processing, I continue to compute an expected class score from softmax logits and apply the same fixed thresholds `[0.5, 1.5, 2.5, 3.5]`. This run end-to-end within the time limit and generate a properly formatted submission file.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with producing predictions from an untrained random model (because `abcdef.pth` is missing), which makes quadratic kappa near zero. The smallest legitimate improvement toward the 0.911 target—without changing your EfficientNet architecture or inference semantics—is to actually train the existing model on `train.csv` for a short, fixed number of epochs, then run the same expected-score + fixed-threshold rounding to generate `submission.csv`. I add a minimal train/valid split, a simple training loop using standard cross-entropy, and (optionally) fit the OptimizedRounder on the validation expected-scores to calibrate thresholds (still the same rounding idea, just data-driven). All paths remain the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with a label/index alignment issue rather than “just a weak model”: in `qk()` you currently pass arguments to `cohen_kappa_score` in the wrong order, and in validation thresholding you accidentally create `all_y` from GPU tensors via `yb.numpy()` (which can silently break/produce wrong arrays). I make minimal fixes to compute QWK correctly (true labels vs predicted labels) and to always move validation labels/predictions to CPU safely before NumPy conversion. I also make the train/valid split stratified (same data, same model, same loss/loop) to stabilize QWK-based threshold optimization so the learned thresholds generalize better to test. These changes keep your core EfficientNet-B5 + cross-entropy training + expected-score + threshold rounding logic intact, but should move your score meaningfully upward toward the 0.911 target.'
- What this solution (achieved 0.73087) has done: 'Main bottlenecks are (1) expensive training fallback (EfficientNet-B5 over 3295 images for 4 epochs) when `abcdef.pth` is missing and (2) slow image I/O/resize/normalize done in Python/PIL in every epoch/prediction. To keep core logic identical while avoiding the 10-minute timeout, the script below adds an on-disk preprocessed image cache (same resized+normalized CHW float32 tensor saved once) and uses persistent DataLoader workers with tuned worker count/prefetching to reduce per-batch overhead. It also avoids repeatedly recomputing constant tensors (class index vector) and uses a vectorized thresholding implementation in `OptimizedRounder` that is mathematically equivalent but much faster. No model/loss/training loop semantics are changed; results should match up to negligible floating-point differences.'

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
from torch.utils import model_zoo
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn import metrics

import scipy as sp


torch.manual_seed(42)
np.random.seed(42)

torch.backends.cudnn.benchmark = True




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
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[0] + 1 - iw, 0)
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
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ["efficientnet_b" + str(i) for i in range(num_models)]
        if model_name.replace("-", "_") not in valid_models:
            raise ValueError("model_name should be one of: " + ", ".join(valid_models))




## === cell 2
md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
md_ef = md_ef.to(device)




## === cell 3
os.makedirs("models", exist_ok=True)




## === cell 4
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
    test_df = pd.read_csv(os.path.join(base_image_dir, "sample_submission.csv"))
    return df, test_df, base_image_dir


df, test_df, base_image_dir = get_df()
print("base_image_dir:", base_image_dir)
print(df.head())
print(test_df.head())




## === cell 5
bs = 16  # keep identical batch size
sz = 224

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_IS_CUDA = torch.cuda.is_available()
_NUM_WORKERS = min(4, (os.cpu_count() or 2))
_PREFETCH_FACTOR = 4 if _NUM_WORKERS > 0 else None




## === cell 6
from PIL import Image

_CACHE_DIR = os.path.join("models", "img_cache_sz224")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_path_for_id(id_code, size, split_tag):
    return os.path.join(_CACHE_DIR, f"{split_tag}_{id_code}_{size}.pt")


def _load_and_preprocess_image_to_tensor(path, size):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(arr)  # float32


class TestImageDataset(Dataset):
    def __init__(self, df, base_dir, folder="test_images", suffix=".png", size=224):
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

        cpath = _cache_path_for_id(id_code, self.size, split_tag="test")
        if os.path.exists(cpath):
            x = torch.load(cpath, map_location="cpu")
        else:
            x = _load_and_preprocess_image_to_tensor(path, self.size)
            torch.save(x, cpath)

        return x, id_code


class TrainImageDataset(Dataset):
    def __init__(self, df, size=224):
        self.df = df.reset_index(drop=True)
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "path"]
        y = int(self.df.loc[idx, "diagnosis"])
        id_code = self.df.loc[idx, "id_code"]

        cpath = _cache_path_for_id(id_code, self.size, split_tag="train")
        if os.path.exists(cpath):
            x = torch.load(cpath, map_location="cpu")
        else:
            x = _load_and_preprocess_image_to_tensor(path, self.size)
            torch.save(x, cpath)

        return x, y


test_ds = TestImageDataset(
    test_df[["id_code"]], base_image_dir, folder="test_images", suffix=".png", size=sz
)
test_dl = DataLoader(
    test_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=_NUM_WORKERS,
    pin_memory=_IS_CUDA,
    persistent_workers=(_NUM_WORKERS > 0),
    prefetch_factor=_PREFETCH_FACTOR if _NUM_WORKERS > 0 else None,
)

print("test samples:", len(test_ds), "num_workers:", _NUM_WORKERS)




## === cell 7
def qk(y_pred, y):
    y_hat = torch.argmax(y_pred, dim=1)
    k = cohen_kappa_score(
        y.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), weights="quadratic"
    )
    return torch.tensor(k)




## === cell 8
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




## === cell 9
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
        "Warning: pretrained weights 'abcdef.pth' not found/loaded; will train a few epochs to avoid near-random predictions."
    )




## === cell 10
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    @staticmethod
    def _apply_coef(X, coef):
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
        return self.coef_["x"]




## === cell 11
def _stratified_split_df(df, valid_frac=0.15, seed=42, label_col="diagnosis"):
    rng = np.random.RandomState(seed)
    parts = []
    for cls, g in df.groupby(label_col):
        idx = g.index.values.copy()
        rng.shuffle(idx)
        n_valid = max(1, int(round(valid_frac * len(idx))))
        parts.append((idx[:n_valid], idx[n_valid:]))
    valid_idx = np.concatenate([p[0] for p in parts])
    train_idx = np.concatenate([p[1] for p in parts])
    rng.shuffle(valid_idx)
    rng.shuffle(train_idx)
    return df.loc[train_idx].reset_index(drop=True), df.loc[valid_idx].reset_index(
        drop=True
    )


def _try_load_imagenet_backbone(model, model_name="efficientnet-b5"):
    """
    Change is score-relevant: starting from ImageNet weights (when available locally)
    drastically improves QWK vs random init, without changing architecture or evaluation semantics.
    Offline-safe: only loads if weights are already cached; otherwise it skips.
    """
    try:
        load_pretrained_weights(model, model_name, load_fc=False)
        return True
    except Exception as e:
        print(
            "Could not load ImageNet EfficientNet weights (will train from scratch). Error:",
            repr(e),
        )
        return False


def _train_if_needed(model, df, device, size, batch_size, epochs=4, lr=3e-4):
    train_df, valid_df = _stratified_split_df(
        df, valid_frac=0.15, seed=42, label_col="diagnosis"
    )

    train_ds = TrainImageDataset(train_df, size=size)
    valid_ds = TrainImageDataset(valid_df, size=size)

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

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, epochs // 2), gamma=0.3
    )
    criterion = nn.CrossEntropyLoss()

    class_idx = torch.arange(5, device=device).float()

    model.train()
    for ep in range(epochs):
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

        model.eval()
        all_exp = []
        all_y = []
        with torch.no_grad():
            for xb, yb in valid_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                prob = torch.softmax(logits.float(), dim=1)
                exp_score = (prob * class_idx).sum(dim=1)
                all_exp.append(exp_score.detach().cpu().numpy())
                all_y.append(yb.detach().cpu().numpy())
        all_exp = np.concatenate(all_exp)
        all_y = np.concatenate(all_y)
        base_coef = [0.5, 1.5, 2.5, 3.5]
        y_hat = OptimizedRounder().predict(all_exp, base_coef).astype(int)
        qwk_val = metrics.cohen_kappa_score(all_y, y_hat, weights="quadratic")
        print(
            f"epoch {ep+1}/{epochs} - lr={scheduler.get_last_lr()[0]:.2e} - train_loss={total_loss/max(1,len(train_dl)):.4f} - val_qwk(base_thr)={qwk_val:.4f}"
        )

    return train_df, valid_df


train_df_split, valid_df_split = None, None
if not loaded:
    _try_load_imagenet_backbone(md_ef, model_name="efficientnet-b5")
    train_df_split, valid_df_split = _train_if_needed(
        md_ef, df, device=device, size=sz, batch_size=bs, epochs=4, lr=3e-4
    )




## === cell 12
def run_subm(learn, test_df, coefficients=[0.5, 1.5, 2.5, 3.5]):
    opt = OptimizedRounder()

    preds, ids = learn.get_preds()

    prob = torch.softmax(preds.float(), dim=1)
    class_idx_cpu = torch.arange(5, device=prob.device).float()
    exp_score = (prob * class_idx_cpu).sum(dim=1)

    tst_pred = opt.predict(exp_score.detach().cpu().numpy(), coefficients).astype(int)

    sub = test_df.copy()
    sub = sub[["id_code"]].copy()
    sub["diagnosis"] = tst_pred
    sub.to_csv("submission.csv", index=False)
    print("done: wrote submission.csv with", len(sub), "rows")
    return sub




## === cell 13
final_coef = [0.5, 1.5, 2.5, 3.5]
if valid_df_split is not None:
    valid_ds = TrainImageDataset(valid_df_split, size=sz)
    valid_dl = DataLoader(
        valid_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=_IS_CUDA,
        persistent_workers=(_NUM_WORKERS > 0),
        prefetch_factor=_PREFETCH_FACTOR if _NUM_WORKERS > 0 else None,
    )
    md_ef.eval()

    class_idx = torch.arange(5, device=device).float()
    all_exp = []
    all_y = []
    with torch.no_grad():
        for xb, yb in valid_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = md_ef(xb)
            prob = torch.softmax(logits.float(), dim=1)
            exp_score = (prob * class_idx).sum(dim=1)
            all_exp.append(exp_score.detach().cpu().numpy())
            all_y.append(yb.detach().cpu().numpy())
    all_exp = np.concatenate(all_exp)
    all_y = np.concatenate(all_y)

    optR = OptimizedRounder()
    optR.fit(all_exp, all_y)
    final_coef = list(optR.coefficients())
    print("Using optimized coefficients:", final_coef)

sub = run_subm(learn=learn, test_df=test_df, coefficients=final_coef)
print(sub.head())
print(
    "submission exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv"),
)
print(pd.read_csv("submission.csv").head())
