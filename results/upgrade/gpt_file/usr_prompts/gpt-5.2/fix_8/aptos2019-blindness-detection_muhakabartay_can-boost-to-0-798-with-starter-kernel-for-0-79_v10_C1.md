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

0.7857652083460929

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the fastai version mismatch that causes `get_transforms` (and thus `data`/`learn`) to be undefined by adding a small compatibility layer: use the existing fastai v1 pipeline when available, otherwise fall back to a minimal fastai v2 `DataBlock` that preserves the same training semantics (regression head + rounding to classes for QWK). I also ensure the EfficientNet model outputs shape `(N,1)` consistently so both metric computation and thresholding work. Finally, I make dataset path resolution robust to your provided folder layout and guarantee `submission.csv` is written with the required columns and row alignment.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an invalid/degenerate submission caused by (a) never training the model at all (so predictions are essentially random/near-constant) and (b) using fixed, untuned rounding thresholds that may collapse most outputs to a single class. To move the score toward your target with minimal change to core logic, I add a short, deterministic fine-tuning step using the existing Learner setup (same model, same loss/semantics) and then fit the rounding thresholds on the validation predictions to better match the QWK metric. I also clamp predictions to a reasonable range before thresholding to avoid extreme outputs dominating class assignment. The script still writes a valid `submission.csv` with correct row alignment and columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely coming from a near-constant or badly calibrated prediction stream (random init because pretrained weights couldn’t download + only 1 epoch), which collapses QWK on the leaderboard. To move the score upward toward your target with minimal disruption, I keep your exact model/training approach but (1) make label handling explicitly consistent (float regression targets in both fastai v1/v2), (2) ensure EfficientNet’s stride handling is correct (avoid subtle shape/behavior issues), (3) run the same `fit_one_cycle` loop for a few epochs only when no checkpoint is found (still the same training semantics), and (4) replace the coarse threshold grid with a deterministic coordinate-descent refinement (still “rounding to classes”, just better calibrated for QWK). These changes should materially increase score from 0.0 while keeping the core pipeline intact and still writing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with predictions collapsing to a single class because the EfficientNet head is outputting a 1×1 tensor per item while fastai’s regression pipeline expects a flat `(N,)` (or `(N,1)` consistently) and because the validation/test prediction APIs differ between fastai v1/v2. I make a minimal, score-relevant fix by wrapping the model so it always returns a flattened `(N,)` regression output (no change to architecture internals), and I make `qk()` compute kappa from properly flattened arrays. I also ensure test predictions are pulled from the correct dataloader in both fastai versions and that the submission rows align exactly to `test.csv` order. These changes preserve your exact training approach and rounding-to-classes semantics, but should move the score upward toward your target by preventing degenerate predictions.'

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

from sklearn.metrics import cohen_kappa_score
from sklearn import metrics

FASTAI_V2 = False
try:
    from fastai.vision.all import *  # fastai v2

    FASTAI_V2 = True
except Exception:
    from fastai import *  # fastai v1
    from fastai.vision import *

try:
    from fastai.callbacks import *  # noqa: F401,F403 (fastai v1 optional)
except Exception:
    pass

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
        if isinstance(self.stride, tuple):
            self.stride = (int(self.stride[0]), int(self.stride[1]))
        else:
            self.stride = (int(self.stride), int(self.stride))

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
            (sh, sw),
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
        if isinstance(self.stride, tuple):
            self.stride = (int(self.stride[0]), int(self.stride[1]))
        else:
            self.stride = (int(self.stride), int(self.stride))

        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
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
        sh, sw = self.stride
        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            (sh, sw),
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
    """Block Decoder for readability"""

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

        s = int(options["s"][0])
        return BlockArgs(
            kernel_size=int(options["k"]),
            num_repeat=int(options["r"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            expand_ratio=int(options["e"]),
            id_skip=("noskip" not in block_string),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=[s, s],
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


def load_pretrained_weights(model, model_name, load_fc=True, weights_path=None):
    """
    Loads pretrained weights from a local path if provided/found; otherwise attempts URL.
    In Kaggle offline environment, URL may fail; we handle exceptions and proceed.
    """
    state_dict = None
    if weights_path is not None and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
    else:
        try:
            state_dict = model_zoo.load_url(url_map[model_name], progress=False)
        except Exception as e:
            print(
                f"WARNING: Could not download pretrained weights ({e}). Using random init."
            )
            return model

    if load_fc:
        model.load_state_dict(state_dict, strict=True)
    else:
        state_dict.pop("_fc.weight", None)
        state_dict.pop("_fc.bias", None)
        _ = model.load_state_dict(state_dict, strict=False)
    return model


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
        stride1 = (self._block_args.stride == 1) or (self._block_args.stride == [1, 1])

        if self.id_skip and stride1 and input_filters == output_filters:
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
        model = load_pretrained_weights(
            model, model_name, load_fc=False, weights_path=None
        )
        return model




## === cell 2
class FlattenRegressor(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        y = self.base_model(x)
        return y.view(y.size(0))


md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
md_ef = FlattenRegressor(md_ef)

os.makedirs("models", exist_ok=True)
local_ckpt_candidates = [
    "../input/kaggle-public/abcdef.pth",  # original attempted path
    "../input/abcdef.pth",
    "/kaggle/input/kaggle-public/abcdef.pth",
]
ckpt_path = next((p for p in local_ckpt_candidates if os.path.exists(p)), None)

if ckpt_path is not None:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        md_ef.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint weights from: {ckpt_path}")
    except Exception as e:
        print(
            f"Found checkpoint at {ckpt_path} but failed to load ({e}). Proceeding without it."
        )
else:
    print(
        "No local 'abcdef.pth' found; proceeding without it (ImageNet pretrain will be used if available)."
    )




## === cell 3
def _resolve_base_dir():
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/data/input/aptos2019-blindness-detection",
        "../input/aptos2019-blindness-detection",
        "/kaggle/input",
        "/kaggle/data",
        "/kaggle/data/input",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "train_images")
        ):
            return c
        nested = os.path.join(c, "aptos2019-blindness-detection")
        if os.path.exists(os.path.join(nested, "train.csv")) and os.path.exists(
            os.path.join(nested, "train_images")
        ):
            return nested
    raise FileNotFoundError(
        "Could not resolve dataset base directory containing train.csv and train_images/"
    )


BASE_DIR = _resolve_base_dir()
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")


def get_df():
    df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    df["path"] = df["id_code"].map(lambda x: os.path.join(TRAIN_DIR, f"{x}.png"))
    df["diagnosis"] = df["diagnosis"].astype(np.float32)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    test_ids = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
    test_df = test_ids.copy()
    test_df["diagnosis"] = 0.0
    return df, test_df


df, test_df = get_df()
print(df.shape, test_df.shape, BASE_DIR)



## === cell 4
bs = 64
sz = 112

if not FASTAI_V2:
    tfms = get_transforms(do_flip=True, flip_vert=True)

    data = (
        ImageList.from_df(df=df, path="./", cols="path")
        .split_by_rand_pct(0.2, seed=42)
        .label_from_df(cols="diagnosis", label_cls=FloatList)
        .transform(
            tfms, size=sz, resize_method=ResizeMethod.SQUISH, padding_mode="zeros"
        )
        .databunch(bs=bs, num_workers=2)
        .normalize(imagenet_stats)
    )
else:
    item_tfms = Resize(sz, method=ResizeMethod.Squish)
    batch_tfms = [
        *aug_transforms(
            do_flip=True,
            flip_vert=True,
            max_rotate=0.0,
            max_zoom=1.0,
            max_lighting=0.0,
            max_warp=0.0,
        ),
        Normalize.from_stats(*imagenet_stats),
    ]

    def _label_func(row):
        return float(row["diagnosis"])

    dblock = DataBlock(
        blocks=(ImageBlock, RegressionBlock),
        get_x=ColReader("path"),
        get_y=_label_func,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )
    data = dblock.dataloaders(df, bs=bs, num_workers=2)




## === cell 5
def qk(y_pred, y):
    yp = torch.round(y_pred.view(-1)).detach().cpu().numpy().astype(int)
    yt = y.view(-1).detach().cpu().numpy().astype(int)
    return torch.tensor(cohen_kappa_score(yp, yt, weights="quadratic"))


if not FASTAI_V2:
    learn = Learner(data, md_ef, metrics=[qk], model_dir="models")
    try:
        learn = learn.to_fp16()
    except Exception:
        pass

    test_df2 = test_df.copy()
    test_df2["path"] = test_df2["id_code"].map(
        lambda x: os.path.join(TEST_DIR, f"{x}.png")
    )
    learn.data.add_test(ImageList.from_df(test_df2, path=".", cols="path"))
else:
    learn = Learner(data, md_ef, metrics=[qk], model_dir="models")
    try:
        learn = learn.to_fp16()
    except Exception:
        pass

    test_df2 = test_df.copy()
    test_df2["path"] = test_df2["id_code"].map(
        lambda x: os.path.join(TEST_DIR, f"{x}.png")
    )
    test_dl = data.test_dl(test_df2)
    learn.dls.test = test_dl



## === cell 6
try:
    if os.path.exists(os.path.join("models", "abcdef.pth")):
        if not FASTAI_V2:
            learn.load("abcdef")
        else:
            learn.load("abcdef", with_opt=False)
        print("Loaded learner weights: models/abcdef.pth")
    else:
        print(
            "No models/abcdef.pth to load via learn.load; proceeding without loading."
        )
except Exception as e:
    print(f"learn.load failed ({e}); proceeding without loading.")

if not os.path.exists(os.path.join("models", "abcdef.pth")):
    try:
        n_epochs = 3
        if not FASTAI_V2:
            learn.fit_one_cycle(n_epochs, max_lr=1e-3)
        else:
            learn.fit_one_cycle(n_epochs, lr_max=1e-3)
        print(f"Completed {n_epochs} epoch fine-tune (no checkpoint was provided).")
    except Exception as e:
        print(f"WARNING: training step failed ({e}); proceeding without training.")




## === cell 7
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




## === cell 8
def _kappa_from_coef(p, t, coef):
    pred_cls = OptimizedRounder().predict(p, coef).astype(int)
    return metrics.cohen_kappa_score(t, pred_cls, weights="quadratic")


def _fit_thresholds_from_valid(learn, default_coef=(0.5, 1.5, 2.5, 3.5)):
    """
    Change (score improvement): keep the exact same rounding-to-classes idea, but fit better cutpoints.
    Use deterministic coordinate descent around quantile init (fast, no external deps), improving QWK calibration.
    """
    if not FASTAI_V2:
        preds, y = learn.get_preds(ds_type=DatasetType.Valid)
    else:
        preds, y = learn.get_preds(dl=learn.dls.valid)

    p = preds.detach().cpu().numpy().reshape(-1).astype(np.float32)
    t = y.detach().cpu().numpy().reshape(-1).astype(int)

    p = np.clip(p, -0.5, 4.5)

    init = np.quantile(p, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    coef = np.sort(init).tolist()

    def _project(c):
        c = [float(x) for x in c]
        c = [max(-0.5, min(4.5, x)) for x in c]
        c = sorted(c)
        eps = 1e-3
        for i in range(1, 4):
            if c[i] <= c[i - 1] + eps:
                c[i] = c[i - 1] + eps
        c[-1] = min(c[-1], 4.5)
        return c

    coef = _project(coef)
    best_k = _kappa_from_coef(p, t, coef)

    steps = [0.5, 0.2, 0.1, 0.05]
    for st in steps:
        improved = True
        while improved:
            improved = False
            for j in range(4):
                candidates = []
                for d in (-st, 0.0, st):
                    c2 = coef.copy()
                    c2[j] = c2[j] + d
                    c2 = _project(c2)
                    candidates.append(c2)
                ks = [(_kappa_from_coef(p, t, c), c) for c in candidates]
                ks.sort(key=lambda x: x[0], reverse=True)
                if ks[0][0] > best_k + 1e-8:
                    best_k, coef = ks[0]
                    improved = True

    print("Fitted thresholds (valid QWK={:.5f}): {}".format(best_k, coef))
    return coef


def run_subm(learn, test_df, coefficients=None):
    """
    Fixes:
    - Use the correct fastai API for fetching test predictions (v1 vs v2).
    - Convert preds tensor to 1D numpy float array before thresholding.
    - Ensure output aligns with test_df row order and has correct columns.
    """
    opt = OptimizedRounder()

    if coefficients is None:
        try:
            coefficients = _fit_thresholds_from_valid(learn)
        except Exception as e:
            print(f"WARNING: threshold fitting failed ({e}); using defaults.")
            coefficients = [0.99, 1.99, 2.99, 3.99]

    if not FASTAI_V2:
        preds, _ = learn.get_preds(DatasetType.Test)
    else:
        preds, _ = learn.get_preds(dl=learn.dls.test)

    preds_np = preds.detach().cpu().numpy().reshape(-1).astype(np.float32)
    preds_np = np.clip(preds_np, -0.5, 4.5)
    tst_pred = opt.predict(preds_np, coefficients).astype(int)

    test_ids = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
    sub = test_ids[["id_code"]].copy()
    sub["diagnosis"] = tst_pred
    sub.to_csv("submission.csv", index=False)
    print("done; wrote submission.csv with shape:", sub.shape)
    return sub


sub = run_subm(learn=learn, test_df=test_df, coefficients=None)
print(sub.head())
print("submission.csv exists:", os.path.exists("submission.csv"))
print("submission.csv rows:", len(sub))
