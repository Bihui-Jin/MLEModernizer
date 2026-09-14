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

0.9088854482695004

# 6. Current score

-0.06994

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the import errors caused by an unexpected fastai version by switching to fastai v2 equivalents (DataBlock, vision_learner, Resize, etc.) while keeping the same core model (EfficientNet-B5 regression head) and the same quadratic-kappa-oriented rounding at submission time. I also remove the broken fastai-v1-only APIs (`ImageList`, `Learner`, `DatasetType`, `to_fp16`) and replace them with a minimal, deterministic training/inference pipeline that still trains (if no checkpoint exists) and produces `submission.csv` with the required columns. I keep the existing “load local EfficientNet weights if present, otherwise train briefly” behavior, but make it actually run end-to-end within the Kaggle filesystem provided. Finally, I ensure test predictions are aligned with `test.csv`/`sample_submission.csv` and that the output is integer classes 0–4.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission/data alignment issue rather than model quality, so I make the smallest fixes that ensure `test_df` is built from `test.csv` (not `sample_submission.csv`), and that the model outputs a sensible 0–4 prediction even when no checkpoint is found. I also remove the dependency on `scipy` during inference by fitting kappa-optimized thresholds with a lightweight grid search on the validation set (same rounding/thresholding semantics, just actually calibrated to this run). Finally, I ensure the submission rows exactly match `test.csv` order and `id_code`, producing a valid `submission.csv` every time.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly due to an `id_code`/row alignment bug: `test_df` coming from `get_df()` has no `path`, while `test_dl` is built from a different dataframe (`test_csv`), so predictions are written onto a mismatched/possibly shuffled `test_df`. I make the smallest change to ensure the exact same `test_df` (built from `test.csv` in the same order) is used for both `test_dl` creation and submission writing, preserving your model and rounding logic. I also fix a minor EfficientNet skip-connection condition bug (`stride` is a list in this implementation) that can destabilize training/inference, without changing the architecture. These two fixes should move the score up from 0.0 toward your target by producing a correctly aligned, valid submission.'
- What this solution (achieved 0.0) has done: 'I fix the CUDA OOM by enabling mixed precision correctly for fastai v2 and by using a safer default batch size that still preserves the same model and training loop semantics. I also prevent accidental huge-memory usage by ensuring inference runs in `no_grad`/eval via fastai’s `get_preds` while freeing cached GPU memory after training or checkpoint loading. Finally, I keep your data alignment intact (the same `test_df` used to build `test_dl` is used to write the submission) and ensure `submission.csv` is always produced with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from an invalid QWK computation during threshold search: your validation targets are floats from a regression block, but `cohen_kappa_score` expects integer class labels 0–4; this can lead to bad/degenerate thresholds and effectively random submissions. I keep your exact model, regression training, and overall pipeline, and only (1) make validation targets consistently rounded/clipped to int 0–4 for both the metric and threshold optimization, and (2) ensure the metric receives integer labels (not floats). This is a minimal semantic fix aligned with the competition’s discrete labels and should move the score up substantially toward your target without changing architecture or training approach. The submission writing and test order/alignment remain unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is overwhelmingly likely to come from a submission/order mismatch (or a degenerate prediction pipeline), not from model quality. I make the smallest changes that (1) force the submission rows to match `sample_submission.csv`/`test.csv` order exactly, and (2) ensure inference uses deterministic, non-augmented transforms (no flips) so the same image doesn’t get randomly altered at test time. I also make `OptimizedRounder` use integer targets explicitly during grid search (it’s currently safe, but this removes any dtype ambiguity) and add a sanity check that the produced `id_code` list exactly matches the sample submission before writing the CSV. These changes preserve your model, training approach, and rounding/thresholding semantics, but should move the score up from 0.0 toward your target by guaranteeing a valid, correctly-aligned submission.'
- What this solution (achieved 0.72949) has done: 'The timeout is dominated by two hotspots: (1) the 4-level nested Python-loop grid search in `OptimizedRounder.fit_grid` (millions of `cohen_kappa_score` calls), and (2) slow image loading/augmentation due to low dataloader parallelism and non-persistent workers. I keep the model/training/prediction logic identical, but replace the grid search with an equivalent dynamic-programming solution that finds the exact best thresholds on the same grid while computing the quadratic weighted kappa from a confusion matrix (no sklearn calls inside loops). I also increase dataloader throughput (more workers + persistent workers + pinned memory + prefetching) without changing transforms or batch semantics. These changes preserve evaluation semantics (same discrete grid, same QWK definition) while cutting runtime dramatically so the notebook can finish within 600 seconds.'
- What this solution (achieved 0.0) has done: 'I fix the fastai v2 dataloader transform assignment that currently crashes because `after_batch` no longer has a `.new` method; instead I build the `DataLoaders` with valid train/valid `batch_tfms` from the start (same transforms/semantics, just wired correctly). I also correct `prefetch_factor` usage (it must be an int when `num_workers>0`, and not passed at all when `num_workers==0`) to avoid intermittent dataloader errors across environments. These changes are runtime/stability fixes and do not change the model or training logic, and they should allow the pipeline to run end-to-end and produce `submission.csv`. Everything else (EfficientNet-B5 regression, training loop, and kappa-optimized rounding) is kept intact.'
- What this solution (achieved -0.05648) has done: 'I fix the fastai v2 DataLoader transform wiring that currently crashes training/validation with `TypeError: 'list' object is not callable` by ensuring the valid dataloader uses a proper `Pipeline`/`Transform` for `after_batch` rather than assigning a raw Python list. I keep your model, regression setup, augmentations, and metric unchanged, and only adjust how `valid_batch_tfms` is attached so the pipeline runs end-to-end. This should also move the score up from 0.0 by actually completing training/inference and producing non-degenerate, correctly-aligned predictions in `submission.csv`. No changes are made to the architecture, loss, or rounding/thresholding semantics.'
- What this solution (achieved -0.06994) has done: 'Your current score (-0.05648) is far below the target (0.9089), which is most consistent with the model not learning anything useful (random-ish predictions) rather than a submission-format issue (your alignment checks look correct). The smallest high-impact fix that preserves your core logic (EfficientNet-B5 with a 1-unit regression head and rounding/thresholding at the end) is to load real ImageNet pretrained weights instead of attempting an internet download that silently fail on Kaggle; without pretrained weights, 3 epochs from scratch on 456px typically perform terribly. I add an offline weight loader that searches common Kaggle-installed torchvision EfficientNet-B5 checkpoints and uses them if found (still the same architecture, just better initialization), and I also make cuDNN deterministic (turn off benchmark) to stabilize results without changing semantics. Everything else (DataBlock, augmentations, training loop, OptimizedRounder grid semantics, submission writing) stays the same.'

# 9. Code solution

## === cell 0
import os
import math
import re
import json
import collections
from functools import partial
from collections import Counter

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F

from sklearn.metrics import cohen_kappa_score

from fastai.vision.all import (
    DataBlock,
    ImageBlock,
    RegressionBlock,
    Resize,
    aug_transforms,
    Normalize,
    imagenet_stats,
    RandomSplitter,
    Learner,
    set_seed,
    untar_data,
    Pipeline,
)
from fastai.callback.fp16 import MixedPrecision

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
set_seed(42, reproducible=True)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)



## === cell 1
"""
EfficientNet implementation (as provided), plus helpers.
Only minimal bug fixes: keep core logic intact.
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
        ih, iw = (
            image_size if isinstance(image_size, list) else [image_size, image_size]
        )
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

        stride_is_1 = (self._block_args.stride == 1) or (self._block_args.stride == [1])

        if self.id_skip and stride_is_1 and input_filters == output_filters:
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
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ["efficientnet_b" + str(i) for i in range(num_models)]
        if model_name.replace("-", "_") not in valid_models:
            raise ValueError("model_name should be one of: " + ", ".join(valid_models))




## === cell 2
md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)


def _unwrap_state_dict(sd):
    if not isinstance(sd, dict):
        return sd
    for k in ("state_dict", "model", "net"):
        if k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    if any(key.startswith("module.") for key in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    if any(key.startswith("model.") for key in sd.keys()):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    return sd


def load_efficientnet_imagenet_weights_via_fastai(model, model_name="efficientnet-b5"):
    url_map = {
        "efficientnet-b0": "http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth",
        "efficientnet-b1": "http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth",
        "efficientnet-b2": "http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth",
        "efficientnet-b3": "http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth",
        "efficientnet-b4": "http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth",
        "efficientnet-b5": "http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth",
    }
    if model_name not in url_map:
        return False
    try:
        p = untar_data(url_map[model_name], c_key="model")  # fastai cache
        sd = torch.load(p, map_location="cpu")
        sd = _unwrap_state_dict(sd)
        sd.pop("_fc.weight", None)
        sd.pop("_fc.bias", None)
        res = model.load_state_dict(sd, strict=False)
        print(f"Loaded EfficientNet ImageNet weights via fastai cache: {p}")
        print(
            f"Missing keys: {res.missing_keys}; Unexpected keys: {res.unexpected_keys}"
        )
        return True
    except Exception as e:
        print("Fastai cached download load failed:", repr(e))
        return False


def try_load_efficientnet_local(model):
    common_filenames = [
        "efficientnet-b5-586e6cc6.pth",
        "efficientnet_b5.pth",
        "efficientnet-b5.pth",
    ]
    candidate_dirs = [
        "/kaggle/input",
        "/kaggle/working",
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection",
    ]
    for d in candidate_dirs:
        for fn in common_filenames:
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                sd = torch.load(p, map_location="cpu")
                sd = _unwrap_state_dict(sd)
                if isinstance(sd, dict):
                    sd.pop("_fc.weight", None)
                    sd.pop("_fc.bias", None)
                res = model.load_state_dict(sd, strict=False)
                print(f"Loaded local EfficientNet weights from: {p}")
                print(
                    f"Missing keys: {res.missing_keys}; Unexpected keys: {res.unexpected_keys}"
                )
                return True
    return False


def try_load_torchvision_efficientnet_b5_weights(model):
    candidate_paths = [
        "/kaggle/input/torchvision-models/efficientnet_b5_rwightman.pth",
        "/kaggle/input/torchvision-models/efficientnet_b5_lukemelas.pth",
        "/kaggle/input/torchvision-models/efficientnet_b5.pth",
        "/kaggle/input/models/efficientnet_b5.pth",
        "/kaggle/input/models/efficientnet_b5-*.pth",
        os.path.expanduser(
            "~/.cache/torch/hub/checkpoints/efficientnet_b5_rwightman-*.pth"
        ),
        os.path.expanduser("~/.cache/torch/hub/checkpoints/efficientnet_b5-*.pth"),
    ]

    import glob

    expanded = []
    for p in candidate_paths:
        if "*" in p or "?" in p or "[" in p:
            expanded.extend(glob.glob(p))
        else:
            expanded.append(p)

    def _load_sd(path):
        sd = torch.load(path, map_location="cpu")
        sd = _unwrap_state_dict(sd)
        return sd if isinstance(sd, dict) else None

    loaded_any = False
    for p in expanded:
        if not os.path.isfile(p):
            continue
        sd = _load_sd(p)
        if sd is None:
            continue

        if any(
            k.startswith("_conv_stem") or k.startswith("_blocks") for k in sd.keys()
        ):
            sd.pop("_fc.weight", None)
            sd.pop("_fc.bias", None)
            res = model.load_state_dict(sd, strict=False)
            print(f"Loaded EfficientNet-format ImageNet weights from: {p}")
            print(
                f"Missing keys: {res.missing_keys}; Unexpected keys: {res.unexpected_keys}"
            )
            return True

        print(
            f"Found candidate torchvision-like weights at {p} but key format is incompatible; skipping."
        )
        loaded_any = loaded_any or False

    return loaded_any


_loaded_local = try_load_efficientnet_local(md_ef)
if not _loaded_local:
    _loaded_tv = try_load_torchvision_efficientnet_b5_weights(md_ef)
    if not _loaded_tv:
        _ = load_efficientnet_imagenet_weights_via_fastai(md_ef, "efficientnet-b5")

md_ef = md_ef.to(DEVICE)




## === cell 3
def get_df():
    base_image_dir = "/kaggle/input/aptos2019-blindness-detection"
    train_dir = os.path.join(base_image_dir, "train_images")
    test_dir = os.path.join(base_image_dir, "test_images")

    df = pd.read_csv(os.path.join(base_image_dir, "train.csv"))
    df["path"] = df["id_code"].map(lambda x: os.path.join(train_dir, f"{x}.png"))
    df = df.drop(columns=["id_code"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    test_df = pd.read_csv(os.path.join(base_image_dir, "test.csv"))
    test_df["path"] = test_df["id_code"].map(
        lambda x: os.path.join(test_dir, f"{x}.png")
    )
    return df, test_df


df, test_df = get_df()
print(df.head())
print(test_df.head())

missing_test = (~test_df["path"].map(os.path.isfile)).sum()
missing_train = (~df["path"].map(os.path.isfile)).sum()
print(
    "Missing train images:",
    int(missing_train),
    "Missing test images:",
    int(missing_test),
)
assert (
    missing_test == 0
), "Some test image paths do not exist; cannot produce valid predictions."



## === cell 4
bs = 8
sz = 456

item_tfms = Resize(sz, method="squish")

train_batch_tfms = [
    *aug_transforms(do_flip=True, flip_vert=True),
    Normalize.from_stats(*imagenet_stats),
]
valid_batch_tfms = [Normalize.from_stats(*imagenet_stats)]


def _get_x(r):
    return r["path"]


def _get_y(r):
    return float(r["diagnosis"])


_num_workers = min(8, (os.cpu_count() or 2))

dl_kwargs = dict(
    bs=bs,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
)
if _num_workers > 0:
    dl_kwargs["prefetch_factor"] = 4

dblock = DataBlock(
    blocks=(ImageBlock, RegressionBlock),
    get_x=_get_x,
    get_y=_get_y,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=item_tfms,
    batch_tfms=train_batch_tfms,
)

dls = dblock.dataloaders(df, **dl_kwargs)

dls.valid.after_batch = Pipeline(valid_batch_tfms)




## === cell 5
def qk_metric(inp, targ):
    pred = (
        torch.round(inp).clamp(0, 4).detach().cpu().numpy().reshape(-1).astype(np.int64)
    )
    true = (
        torch.round(targ)
        .clamp(0, 4)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(np.int64)
    )
    return cohen_kappa_score(true, pred, weights="quadratic")


cbs = []
if torch.cuda.is_available():
    cbs.append(MixedPrecision())

learn = Learner(
    dls, md_ef, metrics=[qk_metric], model_dir="/kaggle/working/models", cbs=cbs
)
print("Learner ready. mixed_precision:", torch.cuda.is_available())

test_dl_kwargs = dict(
    with_labels=False,
    item_tfms=item_tfms,
    batch_tfms=valid_batch_tfms,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
)
if _num_workers > 0:
    test_dl_kwargs["prefetch_factor"] = 4

test_dl = learn.dls.test_dl(test_df, **test_dl_kwargs)
print("Test items:", len(test_df))



## === cell 6
model_name = "abcdef"
candidate_paths = [
    os.path.join("/kaggle/input", "kaggle-public", f"{model_name}.pth"),
    os.path.join("/kaggle/input", f"{model_name}.pth"),
    os.path.join("/kaggle/working", "models", f"{model_name}.pth"),
    os.path.join("/kaggle/working", f"{model_name}.pth"),
]

os.makedirs("/kaggle/working/models", exist_ok=True)
loaded = False
for p in candidate_paths:
    if os.path.isfile(p):
        try:
            if os.path.dirname(p) != "/kaggle/working/models":
                import shutil

                dst = os.path.join("/kaggle/working/models", f"{model_name}.pth")
                shutil.copy2(p, dst)
            learn.load(model_name)
            loaded = True
            print(f"Loaded fastai checkpoint: {p}")
            break
        except Exception as e:
            print(
                f"Found {p} but learn.load failed ({e}); trying torch.load state_dict."
            )
            sd = torch.load(p, map_location="cpu")
            sd = _unwrap_state_dict(sd)
            res = learn.model.load_state_dict(sd, strict=False)
            loaded = True
            print(f"Loaded state_dict from: {p}")
            print(
                f"Missing keys: {res.missing_keys}; Unexpected keys: {res.unexpected_keys}"
            )
            break

if not loaded:
    print(
        "WARNING: pretrained weights file 'abcdef.pth' not found; "
        "training a baseline model to produce a valid submission."
    )
    learn.model.train()
    learn.fit_one_cycle(3, lr_max=1e-3)

if torch.cuda.is_available():
    torch.cuda.empty_cache()




## === cell 7
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = None

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

    @staticmethod
    def _qwk_from_confusion(O):
        O = O.astype(np.float64, copy=False)
        n = O.sum()
        if n <= 0:
            return -1.0
        act_hist = O.sum(axis=1)
        pred_hist = O.sum(axis=0)
        E = np.outer(act_hist, pred_hist) / n

        W = np.zeros((5, 5), dtype=np.float64)
        for i in range(5):
            for j in range(5):
                W[i, j] = ((i - j) ** 2) / 16.0

        num = (W * O).sum()
        den = (W * E).sum()
        if den == 0:
            return -1.0
        return 1.0 - num / den

    def fit_grid(self, X, y, grid=None):
        if grid is None:
            grid = np.arange(0.2, 3.9, 0.1, dtype=np.float32)

        best_kappa = -1.0
        best_coef = (0.5, 1.5, 2.5, 3.5)

        X = np.asarray(X, dtype=np.float32).reshape(-1)
        y = np.asarray(y, dtype=np.int64).reshape(-1)

        g = np.asarray(grid, dtype=np.float32)
        M = g.shape[0]

        bidx = np.searchsorted(g, X, side="right").astype(np.int32)  # 0..M

        counts_bin = np.zeros((M + 1, 5), dtype=np.int64)
        for t in range(5):
            mask = y == t
            if mask.any():
                bc = np.bincount(bidx[mask], minlength=M + 1)
                counts_bin[:, t] = bc.astype(np.int64)

        pref = np.zeros((M + 2, 5), dtype=np.int64)
        pref[1 : M + 2] = np.cumsum(counts_bin, axis=0)

        def seg_counts(l_bin_inclusive, r_bin_inclusive):
            return pref[r_bin_inclusive + 1] - pref[l_bin_inclusive]

        for i in range(M):
            s0 = seg_counts(0, i)
            for j in range(i + 1, M):
                s1 = seg_counts(i + 1, j)
                for k in range(j + 1, M):
                    s2 = seg_counts(j + 1, k)
                    for l in range(k + 1, M):
                        s3 = seg_counts(k + 1, l)
                        s4 = seg_counts(l + 1, M)

                        O = np.zeros((5, 5), dtype=np.int64)
                        O[:, 0] = s0
                        O[:, 1] = s1
                        O[:, 2] = s2
                        O[:, 3] = s3
                        O[:, 4] = s4

                        kappa = self._qwk_from_confusion(O)
                        if kappa > best_kappa:
                            best_kappa = kappa
                            best_coef = (
                                float(g[i]),
                                float(g[j]),
                                float(g[k]),
                                float(g[l]),
                            )

        self.coef_ = best_coef
        print("Best val QWK (grid):", best_kappa, "coef:", best_coef)
        return best_coef




## === cell 8
def run_subm(
    learn,
    test_df,
    test_dl,
    out_path="submission.csv",
):
    opt = OptimizedRounder()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    val_preds, val_targs = learn.get_preds(dl=learn.dls.valid)

    val_preds_np = val_preds.detach().cpu().numpy().reshape(-1).astype(np.float32)

    val_targs_np = (
        torch.round(val_targs)
        .clamp(0, 4)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(np.int64)
    )

    coef = opt.fit_grid(val_preds_np, val_targs_np)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    preds, _ = learn.get_preds(dl=test_dl)
    preds_np = preds.detach().cpu().numpy().reshape(-1).astype(np.float32)

    tst_pred = opt.predict(preds_np, coef).astype(np.int64)
    tst_pred = np.clip(tst_pred, 0, 4)

    assert len(test_df) == len(
        tst_pred
    ), "Prediction length does not match test_df length."

    base_image_dir = "/kaggle/input/aptos2019-blindness-detection"
    sample_sub = pd.read_csv(os.path.join(base_image_dir, "sample_submission.csv"))
    assert "id_code" in sample_sub.columns and "diagnosis" in sample_sub.columns

    test_ids = set(test_df["id_code"].astype(str).tolist())
    sample_ids = sample_sub["id_code"].astype(str).tolist()
    assert (
        set(sample_ids) == test_ids
    ), "Mismatch between test_df ids and sample_submission ids."

    id_to_pred = dict(zip(test_df["id_code"].astype(str).tolist(), tst_pred.tolist()))
    subm = sample_sub[["id_code"]].copy()
    subm["diagnosis"] = subm["id_code"].astype(str).map(id_to_pred).astype(np.int64)

    assert (
        subm["diagnosis"].isna().sum() == 0
    ), "Some predictions are missing after alignment."
    assert (
        subm["id_code"].astype(str).tolist() == sample_ids
    ), "Submission id_code order mismatch."

    subm.to_csv(out_path, index=False)
    print(f"Saved {out_path} with shape={subm.shape} and columns={list(subm.columns)}")
    print(subm.head())
    return subm


_ = run_subm(learn, test_df, test_dl, out_path="submission.csv")
print("Submission file exists:", os.path.isfile("submission.csv"))
print(
    "Submission file size (bytes):",
    os.path.getsize("submission.csv") if os.path.isfile("submission.csv") else None,
)
