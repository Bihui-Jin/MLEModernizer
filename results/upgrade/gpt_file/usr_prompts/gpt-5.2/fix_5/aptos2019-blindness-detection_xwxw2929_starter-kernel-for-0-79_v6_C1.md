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

- What this solution (achieved 0.0) has done: 'The failures come from importing and using `fastai` APIs that aren’t available in this environment (`fastai.callbacks` and then `get_transforms`, `Learner`, etc.). To keep the core idea intact (EfficientNet + regression head + optimized rounding thresholds), I replace the fastai `DataBunch/Learner` pipeline with a minimal PyTorch `Dataset/DataLoader` and inference loop, and I make sure we still optionally load `abcdef.pth` if it exists. I also fix a logic bug in `MBConvBlock` where stride is stored as a list (so the residual path condition never triggers), and I make `from_pretrained` actually load weights if a local pretrained file exists (but still works without any weights). Finally, the script always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with producing essentially random labels because the EfficientNet is running with random weights when `abcdef.pth` is not found, and the fixed thresholds then map noise to classes. To move the score toward the 0.902 target with minimal semantic change, I (1) make the model load a local EfficientNet-B5 pretrained backbone if available in the dataset tree (no internet), and (2) calibrate the rounding thresholds on a small held-out validation split using the already-present `OptimizedRounder` (same regression→rounding approach, just using better coefficients). If no pretrained weights are found, the script still run and emit a valid submission, but with the best available coefficients from the split. All I/O paths stay the same and a valid `submission.csv` is always written.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with the model producing essentially untrained/random predictions (because no real pretrained weights are found) and/or a mismatch between training-time image preprocessing and what the weights expect. To move the score up toward your 0.902 target while keeping core logic intact (EfficientNet-B5 regression + optimized rounding), I make two minimal, directly relevant changes: (1) use the correct EfficientNet-B5 input resolution (456) instead of 256, and (2) add a robust local weight-file resolver that can load the common Kaggle “efficientnet-b5” ImageNet weights and also properly handle checkpoints that store weights under `state_dict` with prefixes. Everything else (model, loss semantics, threshold optimization, submission format/paths) stays the same and the script still always writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import re
import math
import json
import warnings
import collections
from functools import partial
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F

from sklearn import metrics
from sklearn.metrics import cohen_kappa_score

from PIL import Image


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



## === cell 1
"""
EfficientNet implementation (as provided), with minimal fixes:
- Fix BlockDecoder stride to be int (was list), so residual/skip condition works correctly.
- Fix MBConvBlock skip condition to compare against int stride.
- Add optional local weight loading in from_pretrained (no internet).
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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


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

        stride_int = int(options["s"][0])

        return BlockArgs(
            kernel_size=int(options["k"]),
            num_repeat=int(options["r"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            expand_ratio=int(options["e"]),
            id_skip=("noskip" not in block_string),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=stride_int,
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
            and (self._block_args.stride == 1)
            and (input_filters == output_filters)
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
    def from_pretrained(
        cls,
        model_name,
        num_classes=1000,
        weights_path=None,
        strict=False,
        map_location="cpu",
    ):
        model = EfficientNet.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        if weights_path is not None and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=map_location)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            new_state = {}
            for k, v in state.items():
                nk = k
                for prefix in ("module.", "model."):
                    if nk.startswith(prefix):
                        nk = nk[len(prefix) :]
                new_state[nk] = v
            missing, unexpected = model.load_state_dict(new_state, strict=strict)
            print(
                f"Loaded pretrained weights from {weights_path}. Missing: {len(missing)}, unexpected: {len(unexpected)}"
            )
        return model




## === cell 2
def find_any_weight_file(patterns, search_roots=("/kaggle/input", "/kaggle/data")):
    for root in search_roots:
        rootp = Path(root)
        if not rootp.exists():
            continue
        for pat in patterns:
            hits = list(rootp.rglob(pat))
            if hits:
                return str(hits[0])
    return None


imagenet_b5_path = find_any_weight_file(
    patterns=(
        "*efficientnet_b5*imagenet*.pth",
        "*efficientnet-b5*imagenet*.pth",
        "*efficientnet*b5*imagenet*.pth",
        "*efficientnet_b5*.pth",
        "*efficientnet-b5*.pth",
        "*efficientnet*b5*.pth",
        "*efficientnet_b5*.pt",
        "*efficientnet-b5*.pt",
        "*efficientnet*b5*.pt",
        "*efficientnet_b5*.bin",
        "*efficientnet-b5*.bin",
        "*efficientnet*b5*.bin",
    )
)
print("Found candidate ImageNet weights:", imagenet_b5_path)

md_ef = EfficientNet.from_pretrained(
    "efficientnet-b5",
    num_classes=1,
    weights_path=imagenet_b5_path,
    strict=False,
    map_location="cpu",
).to(device)
md_ef.eval()



## === cell 3
os.makedirs("models", exist_ok=True)


def find_weight_file(filename="abcdef.pth", search_roots=("/kaggle",)):
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        hits = list(root.rglob(filename))
        if hits:
            return str(hits[0])
    return None


weights_path = find_weight_file("abcdef.pth")
print("Found finetuned weights:", weights_path)




## === cell 4
def get_df():
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/input",  # fallback root
        "/kaggle/data",
    ]
    base_image_dir = None
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and (
            os.path.exists(os.path.join(c, "train_images"))
            or os.path.exists(os.path.join(c, "train_images.zip"))
        ):
            base_image_dir = c
            break

    if base_image_dir is None:
        possible = list(
            Path("/kaggle").rglob("aptos2019-blindness-detection/train.csv")
        )
        if not possible:
            possible = list(Path("/kaggle").rglob("train.csv"))
        if not possible:
            raise FileNotFoundError(
                "Could not locate aptos2019-blindness-detection dataset under /kaggle."
            )
        base_image_dir = str(possible[0].parent)

    train_dir = os.path.join(base_image_dir, "train_images")
    test_dir = os.path.join(base_image_dir, "test_images")

    df = pd.read_csv(os.path.join(base_image_dir, "train.csv"))
    df["path"] = df["id_code"].map(lambda x: os.path.join(train_dir, f"{x}.png"))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    test_df = pd.read_csv(os.path.join(base_image_dir, "sample_submission.csv"))
    test_df["path"] = test_df["id_code"].map(
        lambda x: os.path.join(test_dir, f"{x}.png")
    )
    return df, test_df, base_image_dir


df, test_df, base_image_dir = get_df()
print("base_image_dir:", base_image_dir)
print(df.head())
print(test_df.head())




## === cell 5
class RetinopathyDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels=None, image_size=456):
        self.paths = list(paths)
        self.labels = None if labels is None else np.asarray(labels).astype(np.float32)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.paths)

    def _load_image(self, p):
        img = Image.open(p).convert("RGB")
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0  # HWC in [0,1]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        x = self._load_image(self.paths[idx])
        if self.labels is None:
            return x
        y = torch.tensor(self.labels[idx])
        return x, y


def predict_regression(model, paths, image_size=456, batch_size=8, num_workers=2):
    ds = RetinopathyDataset(paths, labels=None, image_size=image_size)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device, non_blocking=True)
            out = model(xb).view(-1)
            preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)




## === cell 6
if weights_path is not None and os.path.exists(weights_path):
    dst = os.path.join("models", "abcdef.pth")
    if os.path.abspath(weights_path) != os.path.abspath(dst):
        import shutil

        shutil.copy2(weights_path, dst)

    state = torch.load(dst, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_state = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        new_state[nk] = v
    missing, unexpected = md_ef.load_state_dict(new_state, strict=False)
    md_ef.to(device).eval()
    print(
        "Loaded finetuned weights into model. Missing:",
        len(missing),
        "Unexpected:",
        len(unexpected),
    )
else:
    print(
        "No finetuned weights file found; proceeding with current (possibly ImageNet-pretrained) weights."
    )




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
def fit_rounding_coefficients(
    model,
    df,
    image_size=456,
    batch_size=8,
    num_workers=2,
    val_frac=0.15,
    seed=42,
):
    n = len(df)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_frac))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    val_paths = df.loc[val_idx, "path"].values
    val_y = df.loc[val_idx, "diagnosis"].values.astype(int)

    val_preds = predict_regression(
        model,
        val_paths,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    ).reshape(-1)

    init = np.array([0.6, 1.6, 2.6, 3.6], dtype=np.float32)
    best = init.copy()
    opt = OptimizedRounder()

    def score(coef):
        pred = opt.predict(val_preds, coef)
        pred = np.clip(pred, 0, 4).astype(int)
        return cohen_kappa_score(val_y, pred, weights="quadratic")

    best_score = score(best)
    print("Initial coeffs:", best.tolist(), "val_kappa:", float(best_score))

    steps = [0.25, 0.10, 0.05]
    for st in steps:
        improved = True
        while improved:
            improved = False
            for j in range(4):
                for delta in (-st, st):
                    cand = best.copy()
                    cand[j] = cand[j] + delta
                    if not (cand[0] < cand[1] < cand[2] < cand[3]):
                        continue
                    sc = score(cand)
                    if sc > best_score + 1e-8:
                        best_score = sc
                        best = cand
                        improved = True
        print(
            f"After step {st}: coeffs:", best.tolist(), "val_kappa:", float(best_score)
        )

    return tuple(float(x) for x in best), float(best_score)


best_coef, best_val_kappa = fit_rounding_coefficients(
    md_ef, df, image_size=456, batch_size=8, num_workers=2, val_frac=0.15, seed=42
)
print("Using coefficients for submission:", best_coef, "val_kappa:", best_val_kappa)




## === cell 9
def run_subm(
    model,
    test_df,
    coefficients=(0.6, 1.6, 2.6, 3.6),
    out_path="submission.csv",
    image_size=456,
):
    opt = OptimizedRounder()
    preds_np = predict_regression(
        model,
        test_df["path"].values,
        image_size=image_size,
        batch_size=8,
        num_workers=2,
    ).reshape(-1)
    tst_pred = opt.predict(preds_np, np.array(coefficients, dtype=np.float32))
    tst_pred = np.clip(tst_pred, 0, 4).astype(int)

    sub = test_df[["id_code"]].copy()
    sub["diagnosis"] = tst_pred
    sub.to_csv(out_path, index=False)
    print("Wrote:", out_path, "shape:", sub.shape)
    return sub


submission = run_subm(
    model=md_ef,
    test_df=test_df,
    coefficients=best_coef,  # calibrated coefficients (same semantics: regression -> thresholds)
    out_path="submission.csv",
    image_size=456,
)
print(submission.head())



## === cell 10
assert os.path.exists("submission.csv")
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["id_code", "diagnosis"]
assert len(sub) == len(test_df)
assert sub["diagnosis"].between(0, 4).all()
print("Submission file looks valid:", sub.shape)
