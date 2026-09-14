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

0.911371255572809

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `fastai` (not available in this environment) and replace the data pipeline + learner with a minimal PyTorch inference-only path that preserves the core idea: EfficientNet-B5 producing a single regression output which is then discretized into classes 0–4 by fixed thresholds. I also fix missing imports (`os`, `collections`, `torch`, etc.) and remove the invalid model weight download/copy steps, instead using pretrained EfficientNet weights already bundled with the competition dataset if present; otherwise it fall back to an untrained model but still produce a valid submission. Finally, I ensure paths point to `/kaggle/input/aptos2019-blindness-detection/` and that a `submission.csv` with columns `id_code,diagnosis` is always written.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with effectively random/constant predictions because the EfficientNet-B5 regression head is never loaded with meaningful APTOS-trained weights (the code looks for a non-existent `abcdef.pth`), so the submission ends up mostly/all zeros. To move the score up toward the target while keeping the same core model and inference logic, I (1) add a robust search for common `.pth/.pt/.bin` weight files inside `/kaggle/input/` and the dataset folder and load the first compatible checkpoint into the same EfficientNet-B5(num_classes=1). I also (2) ensure deterministic evaluation and (3) keep the same fixed threshold discretization (0.5,1.5,2.5,3.5) so evaluation semantics remain unchanged; this should substantially improve over 0.0 if any valid pretrained checkpoint exists. If no weights are present, it still produce a valid `submission.csv` exactly as before.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the model producing essentially constant/random outputs because no meaningful APTOS-trained weights are actually being loaded. I keep your exact EfficientNet-B5 regression + fixed-threshold discretization core logic, but make checkpoint loading more compatible with common Kaggle formats (e.g., `model`, `ema`, `net`, nested `state_dict`, and `DataParallel` prefixes) and also ensure the loaded checkpoint really contains the correct `_fc` head for `num_classes=1` before accepting it. Additionally, I fall back to a second pass that can remap `fc.*`/`classifier.*` keys to your `_fc.*` only when shapes match, which often fixes “loads but predicts nonsense” situations. These minimal changes should move the score upward toward your target if any suitable checkpoint exists in `/kaggle/input` (otherwise behavior remains the same and still produces a valid `submission.csv`).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the inference is running with essentially random weights (no APTOS-trained checkpoint found), so the smallest score-improving change is to reliably load a real pretrained EfficientNet-B5 backbone from the local Kaggle environment. I keep your exact model (EfficientNet-B5 with a 1-unit regression head) and the same fixed thresholds, but add a *torchvision* EfficientNet-B5 fallback that copies matching backbone weights into your implementation when no compatible `.pth` is found. This preserves the core inference semantics while turning “random features” into meaningful ImageNet features, which should move the score upward from 0.0 without changing training loops (none) or the discretization. The script still always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from using a randomly-initialized regression head (and possibly random backbone parts), which makes the discretized classes essentially useless for quadratic kappa. To move the score upward with minimal core-logic change, I (1) ensure we load *real* ImageNet pretrained weights in a fully compatible way by switching to `torchvision.models.efficientnet_b5` and copying its backbone weights deterministically into your custom EfficientNet (instead of the current shape-based “guessing” mapping), and (2) add a small, valid-time-only calibration step that estimates the 4 thresholds on a held-out slice of `train.csv` using the same regression outputs, keeping your “regression + thresholding” evaluation semantics intact. This keeps architecture and inference path the same (EfficientNet-B5 -> 1 regression output -> thresholds -> 0–4), but replaces fixed thresholds with data-fitted ones, which is directly aligned to maximizing kappa and should move you substantially above 0.0 toward the target. The script still always write a correct `submission.csv` with `id_code,diagnosis`.'

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

from PIL import Image

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
"""
EfficientNet implementation (as provided), with only execution fixes (imports already present).
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1).to(device)
md_ef.eval()



## === cell 3
BASE_DIR = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
assert "id_code" in test_df.columns
test_df["diagnosis"] = 0




## === cell 4
def _iter_weight_candidates(search_roots):
    exts = (".pth", ".pt", ".bin")
    for root in search_roots:
        if not os.path.exists(root):
            continue
        if os.path.isfile(root) and root.lower().endswith(exts):
            yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                fpl = fn.lower()
                if fpl.endswith(exts):
                    yield os.path.join(dirpath, fn)


def _extract_state_dict(obj):
    if not isinstance(obj, dict):
        return None
    for k in [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "network",
        "ema",
        "teacher",
        "student",
    ]:
        if k in obj and isinstance(obj[k], dict):
            inner = obj[k]
            if "state_dict" in inner and isinstance(inner["state_dict"], dict):
                return inner["state_dict"]
            return inner
    tensor_like = 0
    for k, v in obj.items():
        if isinstance(k, str) and torch.is_tensor(v):
            tensor_like += 1
        if tensor_like >= 5:
            return obj
    return obj


def _sanitize_keys(sd):
    new_sd = {}
    for k, v in sd.items():
        if not isinstance(k, str):
            continue
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module.") :]
        if kk.startswith("model."):
            kk = kk[len("model.") :]
        new_sd[kk] = v
    return new_sd


def _remap_classifier_keys(sd):
    remapped = dict(sd)
    for src_prefix in ["fc.", "classifier.", "head.", "linear."]:
        w_key = src_prefix + "weight"
        b_key = src_prefix + "bias"
        if w_key in sd:
            remapped["_fc.weight"] = sd[w_key]
        if b_key in sd:
            remapped["_fc.bias"] = sd[b_key]
    return remapped


def _try_load_checkpoint_into_model(model, ckpt_path, device):
    try:
        obj = torch.load(ckpt_path, map_location=device)
    except Exception:
        return False, None, None, "torch.load failed"

    sd = _extract_state_dict(obj)
    if not isinstance(sd, dict):
        return False, None, None, "no state_dict found"

    sd = _sanitize_keys(sd)

    need_w_shape = tuple(model._fc.weight.shape)
    need_b_shape = tuple(model._fc.bias.shape)

    def head_ok(state_dict):
        ok = False
        if "_fc.weight" in state_dict and torch.is_tensor(state_dict["_fc.weight"]):
            ok = ok or (tuple(state_dict["_fc.weight"].shape) == need_w_shape)
        if "_fc.bias" in state_dict and torch.is_tensor(state_dict["_fc.bias"]):
            ok = ok or (tuple(state_dict["_fc.bias"].shape) == need_b_shape)
        return ok

    sd_try = sd
    if not head_ok(sd_try):
        sd_try = _remap_classifier_keys(sd_try)

    if not head_ok(sd_try):
        return (
            False,
            None,
            None,
            "checkpoint does not contain compatible regression head",
        )

    try:
        missing, unexpected = model.load_state_dict(sd_try, strict=False)
    except Exception:
        return False, None, None, "load_state_dict failed"

    return True, missing, unexpected, None


def _load_torchvision_imagenet_backbone_into_custom_efficientnet(custom_model, device):
    try:
        from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
    except Exception as e:
        return False, f"torchvision import failed: {e}"

    try:
        tv_model = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1).to(
            device
        )
        tv_model.eval()
    except Exception as e:
        return False, f"torchvision efficientnet_b5 load failed: {e}"

    tv_sd = tv_model.state_dict()
    csd = custom_model.state_dict()
    new_sd = dict(csd)

    name_map = {
        "_conv_stem.weight": "features.0.0.weight",
        "_bn0.weight": "features.0.1.weight",
        "_bn0.bias": "features.0.1.bias",
        "_bn0.running_mean": "features.0.1.running_mean",
        "_bn0.running_var": "features.0.1.running_var",
        "_bn0.num_batches_tracked": "features.0.1.num_batches_tracked",
        "_conv_head.weight": "features.8.0.weight",
        "_bn1.weight": "features.8.1.weight",
        "_bn1.bias": "features.8.1.bias",
        "_bn1.running_mean": "features.8.1.running_mean",
        "_bn1.running_var": "features.8.1.running_var",
        "_bn1.num_batches_tracked": "features.8.1.num_batches_tracked",
    }

    copied = 0
    for ck, tk in name_map.items():
        if ck in csd and tk in tv_sd and csd[ck].shape == tv_sd[tk].shape:
            new_sd[ck] = tv_sd[tk].detach().clone()
            copied += 1

    tv_items = [(k, v) for k, v in tv_sd.items() if torch.is_tensor(v)]

    def find_unique_tv_key_for_custom(custom_key, custom_tensor):
        suffix = custom_key.split(".", 1)[-1]  # drop "_blocks.<idx>."
        candidates = [
            k
            for (k, v) in tv_items
            if k.endswith(suffix) and v.shape == custom_tensor.shape
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    for ck, cv in csd.items():
        if not (isinstance(ck, str) and ck.startswith("_blocks.")):
            continue
        if not torch.is_tensor(cv):
            continue
        tvk = find_unique_tv_key_for_custom(ck, cv)
        if tvk is None:
            continue
        new_sd[ck] = tv_sd[tvk].detach().clone()
        copied += 1

    custom_model.load_state_dict(new_sd, strict=False)
    custom_model.eval()
    return (
        True,
        f"copied {copied} tensors from torchvision EfficientNet-B5 ImageNet weights",
    )


search_roots = [
    BASE_DIR,
    "/kaggle/input",
    "/kaggle/input/aptos2019-blindness-detection",
]
loaded = False
loaded_path = None

cands = list(dict.fromkeys(_iter_weight_candidates(search_roots)))
preferred = []
other = []
for p in cands:
    name = os.path.basename(p).lower()
    if any(
        tok in name
        for tok in ["aptos", "blind", "dr", "retina", "efficientnet", "b5", "kappa"]
    ):
        preferred.append(p)
    else:
        other.append(p)
ordered = preferred + other

last_reasons = []
for p in ordered:
    ok, missing, unexpected, reason = _try_load_checkpoint_into_model(md_ef, p, device)
    if ok:
        md_ef.eval()
        loaded = True
        loaded_path = p
        print("Loaded APTOS/compatible weights from:", loaded_path)
        if missing:
            print("Missing keys (first 10):", list(missing)[:10])
        if unexpected:
            print("Unexpected keys (first 10):", list(unexpected)[:10])
        break
    else:
        if reason is not None:
            last_reasons.append((p, reason))

if not loaded:
    ok, msg = _load_torchvision_imagenet_backbone_into_custom_efficientnet(
        md_ef, device
    )
    if ok:
        loaded = True
        loaded_path = "torchvision://efficientnet_b5_imagenet_backbone_copied"
        print(
            "No compatible APTOS checkpoint found; using torchvision ImageNet backbone fallback:",
            msg,
        )
    else:
        print(
            "No compatible pretrained checkpoint found and torchvision fallback failed; will run fully random -> likely low score."
        )
        print("Fallback error:", msg)
        if last_reasons:
            print("First 5 reject reasons:")
            for p, r in last_reasons[:5]:
                print(" -", os.path.basename(p), ":", r)



## === cell 5
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 456
BATCH_SIZE = 8


def load_image_tensor(path, img_size=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(arr)


@torch.no_grad()
def predict_regression(model, ids, img_dir):
    preds = []
    for i in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[i : i + BATCH_SIZE]
        xs = []
        for id_code in batch_ids:
            p = os.path.join(img_dir, f"{id_code}.png")
            xs.append(load_image_tensor(p))
        x = torch.stack(xs, dim=0).to(device)
        y = model(x).view(-1)  # (bs,)
        preds.append(y.detach().float().cpu().numpy())
    return np.concatenate(preds, axis=0)




## === cell 6
def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)

    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E / E.sum() * O.sum()

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = None

    def predict(self, X, coef):
        X = np.asarray(X, dtype=np.float32).reshape(-1)
        coef = np.asarray(coef, dtype=np.float32).reshape(-1)
        X_p = np.zeros_like(X, dtype=np.int64)
        X_p[X >= coef[0]] = 1
        X_p[X >= coef[1]] = 2
        X_p[X >= coef[2]] = 3
        X_p[X >= coef[3]] = 4
        return X_p


def _fit_thresholds_by_grid_search(reg_preds, y_true):
    reg_preds = np.asarray(reg_preds, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)

    coef = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    opt = OptimizedRounder()

    lo = float(np.percentile(reg_preds, 1))
    hi = float(np.percentile(reg_preds, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return tuple(coef.tolist())

    steps = [0.25, 0.1]
    for step in steps:
        for t in range(4):
            best = coef[t]
            best_score = -1e9
            base = coef.copy()

            grid = np.arange(lo, hi + 1e-6, step, dtype=np.float32)
            for v in grid:
                cand = base.copy()
                cand[t] = v
                cand = np.sort(cand)  # enforce monotonic thresholds
                pred_cls = opt.predict(reg_preds, cand)
                score = quadratic_weighted_kappa(y_true, pred_cls, n_classes=5)
                if score > best_score:
                    best_score = score
                    best = v

            coef[t] = best
            coef = np.sort(coef)

    return tuple(float(x) for x in coef)


val_frac = 0.2
rng = np.random.RandomState(42)
idx = np.arange(len(train_df))
rng.shuffle(idx)
val_n = int(len(idx) * val_frac)
val_idx = idx[:val_n]

val_ids = train_df.loc[val_idx, "id_code"].tolist()
val_y = train_df.loc[val_idx, "diagnosis"].values

md_ef.eval()
val_reg = predict_regression(md_ef, val_ids, TRAIN_IMG_DIR)

fitted_coef = _fit_thresholds_by_grid_search(val_reg, val_y)
print("Using coefficients:", fitted_coef)




## === cell 7
def run_subm(model, test_df, coefficients, out_path="submission.csv"):
    ids = test_df["id_code"].tolist()
    reg_preds = predict_regression(model, ids, TEST_IMG_DIR)
    opt = OptimizedRounder()
    cls_preds = opt.predict(reg_preds, coefficients)
    sub = pd.DataFrame({"id_code": ids, "diagnosis": cls_preds.astype(int)})
    sub.to_csv(out_path, index=False)
    return sub


sub = run_subm(md_ef, test_df, coefficients=fitted_coef, out_path="submission.csv")
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("diagnosis value counts:\n", sub["diagnosis"].value_counts().sort_index())
