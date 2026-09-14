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

-0.00029

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.05247) has done: 'I fixed the file‑path look‑ups so the script finds the CSV and image folders both in `input/…` and in the absolute Kaggle `/kaggle/input` location. I also wrapped the pretrained model loading in a try/except; if the download fails (no internet), the code falls back to a randomly‑initialised EfficientNet‑B0 so that inference can still run. Finally, I set `num_workers=0` for the DataLoader to avoid multiprocessing issues. These changes eliminate the FileNotFoundError and the undefined‑variable errors, letting the notebook finish and write a valid `submission.csv` file.'
- What this solution (achieved 0.0) has done: 'I replace the random‑model inference with a simple majority‑class baseline: after loading the training data I compute the most common diagnosis label and assign that label to every test image. This eliminates the ineffective random predictions and moves the quadratic weighted kappa score far closer to the target while keeping the overall pipeline structure intact.'
- What this solution (achieved 0.03275) has done: 'I replace the majority‑class baseline with predictions from the pretrained EfficientNet model, performing inference on the test images and writing those class indices to the submission file. This keeps the original model architecture and loading logic unchanged while providing more informative predictions, which should raise the quadratic weighted kappa score toward the target.'
- What this solution (achieved 0.14259) has done: 'I replace the random model‑based inference with a very simple intensity‑based heuristic: compute the average pixel brightness for each training image, derive a mean brightness per diagnosis class, and then predict each test image by assigning the class whose mean brightness is closest to the test image’s brightness. This change keeps the overall pipeline structure (loading data, creating a submission) but replaces the untrained model predictions with a lightweight rule that is expected to raise the quadratic weighted kappa score toward the target.'
- What this solution (achieved 0.00789) has done: 'I replace the simple intensity‑based rule with a lightweight EfficientNet inference that uses the pretrained ImageNet weights already loaded in the script. By resizing each test image to the model’s required size, normalizing it, and taking the arg‑max of the logits, we keep the original architecture and loading logic untouched while providing more informative predictions that should raise the quadratic weighted kappa toward the target. The rest of the pipeline (CSV handling, submission writing) remains unchanged.'
- What this solution (achieved 0.14259) has done: 'I replace the random EfficientNet inference with a lightweight intensity‑based heuristic that uses the training images to compute average brightness per diagnosis class and then assigns each test image to the class whose mean intensity is closest. This simple calibration keeps the overall pipeline unchanged while providing much more informative predictions, moving the quadratic weighted kappa score markedly closer to the target.'
- What this solution (achieved -0.03352) has done: 'I replace the simplistic intensity‑based rule with a proper EfficientNet inference pipeline: a lightweight Dataset that loads test images, applies standard ImageNet preprocessing, and uses the pretrained EfficientNet (or the randomly‑initialized fallback) to produce class predictions. This keeps the original model architecture while providing much more informative predictions, moving the quadratic weighted kappa score substantially closer to the target.'
- What this solution (achieved 0.14259) has done: 'I add a lightweight intensity‑based fallback that replaces low‑confidence EfficientNet predictions with the class whose average training brightness is closest to the test image brightness. This introduces a `predict_by_intensity` helper and updates the inference loop to use the fallback when the model’s max soft‑max probability is below a modest threshold, which should raise the quadratic weighted kappa toward the target while keeping the original EfficientNet architecture untouched.'
- What this solution (achieved -0.00029) has done: 'Implemented fixes to resolve runtime errors and ensure a valid submission:
- Corrected intensity tensor handling by concatenating tensors before conversion.
- Added a `rgb_mean_features` helper to compute RGB‑mean vectors for fallback predictions.
- Fixed the fallback prediction function to use the new helper and handle missing images safely.
- Adjusted DataLoader worker count to avoid multiprocessing issues in the Kaggle environment.'
- What this solution (achieved 0.14259) has done: 'The update forces the model to rely on the lightweight intensity‑based fallback for every test image, which is far more informative than the untrained EfficientNet predictions. By using the average pixel brightness to pick the nearest diagnosis class, we expect a noticeable rise in the quadratic weighted kappa score, moving it closer to the target while keeping the overall pipeline unchanged. The changes are confined to the fallback prediction function and the confidence threshold call.'
- What this solution (achieved -0.00029) has done: 'I keep the overall pipeline unchanged but add a lightweight LogisticRegression fallback that uses the RGB‑mean features we already compute, and lower the confidence threshold so that the pretrained (or random) EfficientNet predictions are used only when the model is reasonably sure. This should give more informative predictions than the pure intensity fallback and move the quadratic weighted kappa score closer to the target.'

# 9. Code solution

## === cell 0
import os
import re
import math
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision import models as md
from torch.utils import model_zoo
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from PIL import Image



## === cell 1
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
    return inputs / keep_prob * binary_tensor


def get_same_padding_conv2d(image_size=None):
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return lambda *args, **kwargs: Conv2dStaticSamePadding(
            *args, image_size=image_size, **kwargs
        )


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


class Conv2dStaticSamePadding(nn.Conv2d):
    def __init__(
        self, in_channels, out_channels, kernel_size, image_size=None, **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        assert image_size is not None
        ih, iw = (
            image_size
            if isinstance(image_size, (list, tuple))
            else [image_size, image_size]
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
            self.static_padding = nn.Identity()

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


class Identity(nn.Module):
    def forward(self, x):
        return x


class BlockDecoder:
    @staticmethod
    def _decode_block_string(block_string):
        ops = block_string.split("_")
        options = {}
        for op in ops:
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value
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
        return [BlockDecoder._decode_block_string(s) for s in string_list]


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
        oup = inp * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(inp, oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(oup, momentum=self._bn_mom, eps=self._bn_eps)

        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            oup, oup, groups=oup, kernel_size=k, stride=s, bias=False
        )
        self._bn1 = nn.BatchNorm2d(oup, momentum=self._bn_mom, eps=self._bn_eps)

        if self.has_se:
            num_squeezed = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            self._se_reduce = Conv2d(oup, num_squeezed, kernel_size=1)
            self._se_expand = Conv2d(num_squeezed, oup, kernel_size=1)

        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(oup, final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))
        if self.has_se:
            x_s = F.adaptive_avg_pool2d(x, 1)
            x_s = self._se_expand(relu_fn(self._se_reduce(x_s)))
            x = torch.sigmoid(x_s) * x
        x = self._bn2(self._project_conv(x))
        if (
            self.id_skip
            and self._block_args.stride == 1
            and self._block_args.input_filters == self._block_args.output_filters
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
        bn_mom = 1 - global_params.batch_norm_momentum
        bn_eps = global_params.batch_norm_epsilon

        in_c = 3
        out_c = round_filters(32, global_params)
        self._conv_stem = Conv2d(in_c, out_c, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(out_c, momentum=bn_mom, eps=bn_eps)

        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, global_params),
                output_filters=round_filters(block_args.output_filters, global_params),
                num_repeat=round_repeats(block_args.num_repeat, global_params),
            )
            self._blocks.append(MBConvBlock(block_args, global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, global_params))

        in_c = block_args.output_filters
        out_c = round_filters(1280, global_params)
        self._conv_head = Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(out_c, momentum=bn_mom, eps=bn_eps)

        self._dropout = global_params.dropout_rate
        self._fc = nn.Linear(out_c, global_params.num_classes)

    def extract_features(self, inputs):
        x = relu_fn(self._bn0(self._conv_stem(inputs)))
        for idx, block in enumerate(self._blocks):
            drop_rate = self._global_params.drop_connect_rate
            if drop_rate:
                drop_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_rate)
        x = relu_fn(self._bn1(self._conv_head(x)))
        return x

    def forward(self, inputs):
        x = self.extract_features(inputs)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)
        return self._fc(x)

    @classmethod
    def from_name(cls, model_name, override_params=None):
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=5):
        base_model = cls.from_name(model_name, override_params={"num_classes": 1000})
        state_dict = model_zoo.load_url(url_map[model_name])
        filtered_sd = {k: v for k, v in state_dict.items() if not k.startswith("_fc.")}
        base_model.load_state_dict(filtered_sd, strict=False)
        in_features = base_model._fc.in_features
        base_model._fc = nn.Linear(in_features, num_classes)
        return base_model

    @staticmethod
    def _check_model_name_is_valid(model_name):
        valid = [f"efficientnet-b{i}" for i in range(8)]
        if model_name not in valid:
            raise ValueError(f"Invalid model name: {model_name}")


def get_model_params(model_name, override_params):
    if model_name.startswith("efficientnet"):
        w, d, s, p = efficientnet_params(model_name)
        blocks_args = BlockDecoder.decode(
            [
                "r1_k3_s11_e1_i32_o16_se0.25",
                "r2_k3_s22_e6_i16_o24_se0.25",
                "r2_k5_s22_e6_i24_o40_se0.25",
                "r3_k3_s22_e6_i40_o80_se0.25",
                "r3_k5_s11_e6_i80_o112_se0.25",
                "r4_k5_s22_e6_i112_o192_se0.25",
                "r1_k3_s11_e6_i192_o320_se0.25",
            ]
        )
        gp = GlobalParams(
            batch_norm_momentum=0.99,
            batch_norm_epsilon=1e-3,
            dropout_rate=p,
            drop_connect_rate=0.2,
            num_classes=5,
            width_coefficient=w,
            depth_coefficient=d,
            depth_divisor=8,
            min_depth=None,
            image_size=s,
        )
        if override_params:
            gp = gp._replace(**override_params)
        return blocks_args, gp
    raise NotImplementedError


url_map = {
    "efficientnet-b0": "http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth",
    "efficientnet-b1": "http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth",
    "efficientnet-b2": "http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth",
    "efficientnet-b3": "http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth",
    "efficientnet-b4": "http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth",
    "efficientnet-b5": "http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth",
    "efficientnet-b6": "http://storage.googleapis.com/public-models/efficientnet-b6-ec0b3e88.pth",
    "efficientnet-b7": "http://storage.googleapis.com/public-models/efficientnet-b7-6dee8b9b.pth",
}



## === cell 2
try:
    model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=5)
except Exception as e:
    print(f"Pretrained load failed ({e}), using randomly initialized model.")
    model = EfficientNet.from_name(
        "efficientnet-b0", override_params={"num_classes": 5}
    )
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



## === cell 3
os.makedirs("models", exist_ok=True)




## === cell 4
def get_df():
    possible_bases = [
        os.path.join("input", "aptos2019-blindness-detection"),
        "/kaggle/input/aptos2019-blindness-detection",
    ]
    for base_dir in possible_bases:
        train_csv = os.path.join(base_dir, "train.csv")
        test_csv = os.path.join(base_dir, "sample_submission.csv")
        train_dir = os.path.join(base_dir, "train_images")
        if os.path.exists(train_csv) and os.path.isdir(train_dir):
            df = pd.read_csv(train_csv)
            df["path"] = df["id_code"].apply(
                lambda x: os.path.join(train_dir, f"{x}.png")
            )
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            test_df = pd.read_csv(test_csv)
            return df, test_df
    raise FileNotFoundError("Could not locate training CSV or image folder.")


df, test_df = get_df()




## === cell 5
class TrainFeatureDataset(Dataset):
    def __init__(self, df):
        self.paths = df["path"].values
        self.transform = T.ToTensor()  # yields 3 x H x W in [0,1]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        tensor = self.transform(img)  # (3,H,W) float tensor
        rgb_mean = tensor.mean(dim=(1, 2))  # (3,)
        gray = img.convert("L")
        intensity = np.array(gray).mean()  # scalar
        return rgb_mean.float(), float(intensity)


num_workers = max(1, os.cpu_count() - 1)
loader = DataLoader(
    TrainFeatureDataset(df),
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False,
)

rgb_means = []
intensity_tensors = []
for rgb_batch, int_batch in loader:
    rgb_means.append(rgb_batch)  # (B,3)
    intensity_tensors.append(int_batch)  # (B,)

train_features = torch.cat(rgb_means, dim=0).numpy()  # (N,3)
train_intensities = torch.cat(intensity_tensors, dim=0).numpy()  # (N,)

train_labels = df["diagnosis"].values

fallback_lr = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", max_iter=200, n_jobs=1
)
fallback_lr.fit(train_features, train_labels)

class_intensity_means = {}
for cls in np.unique(train_labels):
    mask = train_labels == cls
    class_intensity_means[int(cls)] = float(train_intensities[mask].mean())


def rgb_mean_features(image_path):
    """Return a 3‑element array with the per‑channel mean of the image."""
    img = Image.open(image_path).convert("RGB")
    tensor = T.ToTensor()(img)  # (3,H,W) in [0,1]
    return tensor.mean(dim=(1, 2)).cpu().numpy()


def fallback_predict_by_intensity(id_codes):
    """
    Predict diagnoses for given id_codes using average pixel intensity.
    The class whose mean intensity is closest to the image intensity is chosen.
    """
    base_dirs = [
        os.path.join("input", "aptos2019-blindness-detection", "test_images"),
        "/kaggle/input/aptos2019-blindness-detection/test_images",
    ]
    preds = []
    for cid in id_codes:
        img_path = None
        for base in base_dirs:
            cand = os.path.join(base, f"{cid}.png")
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            pred = max(class_intensity_means, key=class_intensity_means.get)
        else:
            gray = Image.open(img_path).convert("L")
            intensity = float(np.array(gray).mean())
            diffs = {
                cls: abs(intensity - mean)
                for cls, mean in class_intensity_means.items()
            }
            pred = min(diffs, key=diffs.get)
        preds.append(int(pred))
    return id_codes, preds


def fallback_predict_by_lr(id_codes):
    """
    Predict diagnoses using the LogisticRegression model trained on RGB‑mean features.
    """
    base_dirs = [
        os.path.join("input", "aptos2019-blindness-detection", "test_images"),
        "/kaggle/input/aptos2019-blindness-detection/test_images",
    ]
    feats = []
    valid_ids = []
    for cid in id_codes:
        img_path = None
        for base in base_dirs:
            cand = os.path.join(base, f"{cid}.png")
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is not None:
            feats.append(rgb_mean_features(img_path))
            valid_ids.append(cid)
        else:
            feats.append(None)
            valid_ids.append(cid)
    preds = []
    if any(f is not None for f in feats):
        arr = np.array([f for f in feats if f is not None])
        lr_preds = fallback_lr.predict(arr)
        idx = 0
        for f in feats:
            if f is None:
                preds.append(None)
            else:
                preds.append(int(lr_preds[idx]))
                idx += 1
    else:
        preds = [None] * len(feats)
    return valid_ids, preds




## === cell 6
class TestDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.base_dirs = [
            os.path.join("input", "aptos2019-blindness-detection", "test_images"),
            "/kaggle/input/aptos2019-blindness-detection/test_images",
        ]
        self.transform = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.463], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def _find_path(self, id_code):
        for base in self.base_dirs:
            cand = os.path.join(base, f"{id_code}.png")
            if os.path.exists(cand):
                return cand
        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._find_path(row["id_code"])
        if img_path is None:
            img_tensor = torch.zeros(3, 224, 224)
        else:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
        return img_tensor, row["id_code"]


def predict_with_model(test_df, model, device, batch_size=32, confidence_thr=0.6):
    """
    Perform EfficientNet inference. When the model confidence is below the
    threshold we fall back to the LogisticRegression predictor (trained on RGB
    means). This provides more signal than the pure intensity fallback and
    nudges the score toward the target.
    """
    model.eval()
    dataset = TestDataset(test_df)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    all_ids = []
    all_preds = []
    low_conf_ids = []

    with torch.no_grad():
        for imgs, ids in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            max_probs = max_probs.cpu().numpy()
            preds = preds.cpu().numpy()
            for i, pid, prob in zip(range(len(ids)), ids, max_probs):
                all_ids.append(pid)
                if prob < confidence_thr:
                    low_conf_ids.append(pid)
                    all_preds.append(None)
                else:
                    all_preds.append(int(preds[i]))

    if low_conf_ids:
        _, lr_preds = fallback_predict_by_lr(low_conf_ids)
        lr_map = dict(zip(low_conf_ids, lr_preds))
        all_preds = [
            lr_map.get(i, p) if p is None else p for i, p in zip(all_ids, all_preds)
        ]

    return all_ids, all_preds


pred_ids, pred_labels = predict_with_model(test_df, model, device)

submission = pd.DataFrame({"id_code": pred_ids, "diagnosis": pred_labels})
submission.to_csv("submission.csv", index=False)
print(
    "submission.csv written with",
    len(submission),
    "rows using EfficientNet when confident, otherwise LogisticRegression fallback.",
)
