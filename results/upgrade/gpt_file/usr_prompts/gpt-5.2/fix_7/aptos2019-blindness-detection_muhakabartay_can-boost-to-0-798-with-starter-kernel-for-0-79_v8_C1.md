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
import glob
import collections
from functools import partial

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms

from sklearn import metrics

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "../data/aptos2019-blindness-detection",
]
BASE_DIR = next((p for p in BASE_CANDIDATES if os.path.exists(p)), None)
if BASE_DIR is None:
    raise FileNotFoundError(
        f"Could not find aptos dataset folder in candidates: {BASE_CANDIDATES}"
    )

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing: {TEST_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)
assert "id_code" in train_df.columns and "diagnosis" in train_df.columns
assert "id_code" in test_df.columns

print("BASE_DIR:", BASE_DIR)
print(
    "train rows:",
    len(train_df),
    "test rows:",
    len(test_df),
    "sample rows:",
    len(sample_df),
)



## === cell 2
"""
EfficientNet implementation (as provided). Core logic preserved.
Only change: we will NOT download weights from the internet in this environment.
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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

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
        pad_w = max((ow - 1) * self.stride[1] + (kh - 1) * self.dilation[0] + 1 - iw, 0)
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


url_map = {
    "efficientnet-b0": "http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth",
    "efficientnet-b1": "http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth",
    "efficientnet-b2": "http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth",
    "efficientnet-b3": "http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth",
    "efficientnet-b4": "http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth",
    "efficientnet-b5": "http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth",
}


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




## === cell 3
try:
    from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights

    _tv_ok = True
except Exception as e:
    print("torchvision efficientnet_b5 not available:", repr(e))
    _tv_ok = False

if _tv_ok:
    md_ef = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
    md_ef.classifier[1] = nn.Linear(md_ef.classifier[1].in_features, 1)
    md_ef = md_ef.to(device)
    md_ef.eval()
    print("Using torchvision EfficientNet-B5 backbone with ImageNet weights.")
else:
    md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1).to(device)
    md_ef.eval()
    print("Falling back to custom EfficientNet-B5 (no guaranteed pretrained backbone).")



## === cell 4
os.makedirs("models", exist_ok=True)


def _unwrap_state_dict(sd):
    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and isinstance(sd["state_dict"], dict)
    ):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _load_state_dict_flexible(model: nn.Module, sd: dict):
    sd = _unwrap_state_dict(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected, sd


def _try_load_weights(model: nn.Module):
    candidates = []
    candidates += glob.glob("/kaggle/input/**/*.pth", recursive=True)
    candidates += glob.glob("../input/**/*.pth", recursive=True)
    candidates += glob.glob("models/*.pth")
    candidates += glob.glob("/kaggle/working/models/*.pth")
    candidates = list(dict.fromkeys(candidates))

    preferred = []
    for p in candidates:
        bn = os.path.basename(p).lower()
        if any(tok in bn for tok in ["aptos", "blind", "retina", "dr", "kappa", "b5"]):
            preferred.append(p)
    candidates = preferred + [p for p in candidates if p not in preferred]

    for p in candidates:
        try:
            sd0 = torch.load(p, map_location="cpu")
            missing, unexpected, sd = _load_state_dict_flexible(model, sd0)

            ok = False
            if hasattr(model, "_fc"):
                if "_fc.weight" in sd and "_fc.bias" in sd:
                    w = sd["_fc.weight"]
                    b = sd["_fc.bias"]
                    ok = (
                        isinstance(w, torch.Tensor)
                        and isinstance(b, torch.Tensor)
                        and tuple(w.shape) == tuple(model._fc.weight.shape)
                        and tuple(b.shape) == tuple(model._fc.bias.shape)
                    )
            else:
                if "classifier.1.weight" in sd and "classifier.1.bias" in sd:
                    w = sd["classifier.1.weight"]
                    b = sd["classifier.1.bias"]
                    ok = (
                        isinstance(w, torch.Tensor)
                        and isinstance(b, torch.Tensor)
                        and tuple(w.shape) == tuple(model.classifier[1].weight.shape)
                        and tuple(b.shape) == tuple(model.classifier[1].bias.shape)
                    )

            if ok:
                print(f"Loaded DR regressor weights from: {p}")
                print(
                    "Missing keys:", len(missing), "Unexpected keys:", len(unexpected)
                )
                return True, p

        except Exception:
            continue
    return False, None


loaded_dr, dr_path = _try_load_weights(md_ef)
if not loaded_dr:
    print(
        "No compatible DR checkpoint found; will rely on ImageNet init + head-only training."
    )



## === cell 5
IMG_SIZE = 456
infer_tfms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_image_tensor(img_path: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    return infer_tfms(img)


class RetinaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, train: bool):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        x = load_image_tensor(img_path)
        if self.train:
            y = torch.tensor(float(row["diagnosis"]), dtype=torch.float32)
            return x, y
        return x




## === cell 6
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = None

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


def _kappa(y_true, y_pred):
    return metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")


def fit_rounder_simple(y_true: np.ndarray, y_pred_cont: np.ndarray):
    y_true = y_true.astype(int)
    y_pred_cont = y_pred_cont.astype(np.float32)
    y_pred_cont = np.clip(y_pred_cont, 0.0, 4.0)

    best = [0.5, 1.5, 2.5, 3.5]
    opt = OptimizedRounder()
    best_score = _kappa(y_true, opt.predict(y_pred_cont, best).astype(int))

    grids = [
        np.linspace(0.1, 1.4, 27),
        np.linspace(0.8, 2.4, 33),
        np.linspace(1.8, 3.6, 37),
        np.linspace(2.6, 4.6, 41),
    ]

    for it in range(2):
        for j in range(4):
            cand_best = best[j]
            for v in grids[j]:
                coef = best.copy()
                coef[j] = float(v)
                if not (coef[0] < coef[1] < coef[2] < coef[3]):
                    continue
                sc = _kappa(y_true, opt.predict(y_pred_cont, coef).astype(int))
                if sc > best_score:
                    best_score = sc
                    cand_best = float(v)
            best[j] = cand_best
    print("Fitted coefficients:", best, "val kappa:", best_score)
    return best




## === cell 7
def _stratified_split_indices(y: np.ndarray, n_splits: int = 4, seed: int = 42):
    rng = np.random.RandomState(seed)
    y = np.asarray(y).astype(int)
    folds = [[] for _ in range(n_splits)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        parts = np.array_split(idx, n_splits)
        for k in range(n_splits):
            folds[k].extend(parts[k].tolist())
    folds = [np.array(sorted(f), dtype=int) for f in folds]
    return folds


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    preds = []
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            xb, _ = batch
        else:
            xb = batch
        xb = xb.to(device, non_blocking=True)
        out = model(xb).view(-1).detach().float().cpu().numpy()
        preds.append(out)
    return np.concatenate(preds, axis=0)


def _set_head_trainable_only(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "_fc"):
        for p in model._fc.parameters():
            p.requires_grad = True
        return model._fc.parameters()
    else:
        for p in model.classifier[1].parameters():
            p.requires_grad = True
        return model.classifier[1].parameters()


def head_only_fit(
    model: nn.Module, df: pd.DataFrame, epochs: int, batch_size: int, lr: float
):
    ds = RetinaDataset(df, TRAIN_IMG_DIR, train=True)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    head_params = _set_head_trainable_only(model)
    model.train()
    optim = torch.optim.Adam(head_params, lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            out = model(xb).view(-1)
            loss = loss_fn(out, yb)
            loss.backward()
            optim.step()
            running += float(loss.detach().cpu().item()) * xb.size(0)
        running /= len(ds)
        print(f"  epoch {ep+1}/{epochs} train_mse={running:.4f}")
    model.eval()


def train_and_get_oof_coefficients(
    model: nn.Module,
    epochs: int = 3,
    batch_size: int = 12,
    lr: float = 3e-4,
    n_splits: int = 4,
):
    if loaded_dr:
        print("DR checkpoint loaded; skipping training and OOF threshold fit.")
        return [0.99, 1.99, 2.99, 3.99]

    df = train_df.copy().reset_index(drop=True)
    y = df["diagnosis"].values.astype(int)
    fold_val_indices = _stratified_split_indices(y, n_splits=n_splits, seed=SEED)

    oof_pred = np.zeros(len(df), dtype=np.float32)

    for k in range(n_splits):
        val_idx = fold_val_indices[k]
        tr_idx = np.setdiff1d(np.arange(len(df)), val_idx)

        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        va_df = df.iloc[val_idx].reset_index(drop=True)

        if _tv_ok:
            m = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
        else:
            m = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
        m = m.to(device)

        print(
            f"Fold {k+1}/{n_splits}: fit head on {len(tr_df)} train, validate on {len(va_df)}"
        )
        head_only_fit(m, tr_df, epochs=epochs, batch_size=batch_size, lr=lr)

        va_ds = RetinaDataset(va_df, TRAIN_IMG_DIR, train=True)
        va_loader = DataLoader(
            va_ds,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        pred_va = predict_loader(m, va_loader)
        oof_pred[val_idx] = pred_va.astype(np.float32)

        pred_clip = np.clip(pred_va.astype(np.float32), 0.0, 4.0)
        kappa_fold = _kappa(
            va_df["diagnosis"].values.astype(int),
            OptimizedRounder().predict(pred_clip, [0.5, 1.5, 2.5, 3.5]).astype(int),
        )
        print(f"  fold default-round kappa={kappa_fold:.4f}")

        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_clip = np.clip(oof_pred, 0.0, 4.0)
    coefficients = fit_rounder_simple(y, oof_clip)

    print("Final head-only fit on full training set for test inference...")
    head_only_fit(model, df, epochs=epochs, batch_size=batch_size, lr=lr)

    return coefficients


coefficients = train_and_get_oof_coefficients(
    md_ef, epochs=3, batch_size=12, lr=3e-4, n_splits=4
)




## === cell 8
@torch.no_grad()
def predict_df(
    model: nn.Module, df: pd.DataFrame, img_dir: str, batch_size: int = 8
) -> np.ndarray:
    model.eval()
    paths = [os.path.join(img_dir, f"{i}.png") for i in df["id_code"].values]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing image: {p}")

    preds = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        batch = torch.stack([load_image_tensor(p) for p in batch_paths], dim=0).to(
            device
        )
        out = model(batch).view(-1)
        preds.append(out.detach().float().cpu().numpy())
    return np.concatenate(preds, axis=0)


raw_preds = predict_df(md_ef, test_df, TEST_IMG_DIR, batch_size=8)
raw_preds = np.clip(raw_preds.astype(np.float32), 0.0, 4.0)
print(
    "raw_preds:",
    raw_preds.shape,
    "min/max:",
    float(raw_preds.min()),
    float(raw_preds.max()),
)




## === cell 9
def run_subm(coefficients, out_path="submission.csv"):
    opt = OptimizedRounder()
    tst_pred = opt.predict(raw_preds.astype(np.float32), coefficients)
    subm = pd.DataFrame(
        {"id_code": test_df["id_code"].values, "diagnosis": tst_pred.astype(int)}
    )
    subm["diagnosis"] = subm["diagnosis"].clip(0, 4).astype(int)
    subm.to_csv(out_path, index=False)
    print("Wrote", out_path, "rows:", len(subm))
    return subm


submission = run_subm(coefficients, out_path="submission.csv")
print(submission.head())



## === cell 10
assert list(submission.columns) == ["id_code", "diagnosis"]
assert len(submission) == len(sample_df) == len(test_df)
print("Submission format OK.")
