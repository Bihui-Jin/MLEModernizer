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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

-1.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
"""Base dataset definition"""

import abc
from pathlib import Path
from typing import Any, Dict, List, Union

from torch.utils.data import Dataset


class BaseDataset(Dataset, abc.ABC):
    def __init__(self, list_of_paths: Union[List[Path], List[str]]):
        self.list_of_paths = list_of_paths

        self.img_key = "image"
        self.lbl_key = "label"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError("Implement in subclass.")

    def __len__(self):
        raise NotImplementedError("Implement in subclass.")




## === cell 1
"""EfficientNet3D implementation (unchanged core logic)"""

import collections
import math
import re
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

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
        "include_top",
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

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)  # type: ignore
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)  # type: ignore


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
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
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
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
    random_tensor = keep_prob + torch.rand(
        [batch_size, 1, 1, 1, 1], dtype=inputs.dtype, device=inputs.device
    )
    binary_tensor = torch.floor(random_tensor)
    return inputs / keep_prob * binary_tensor


def get_same_padding_conv3d(image_size=None):
    if image_size is None:
        return Conv3dDynamicSamePadding
    else:
        return partial(Conv3dStaticSamePadding, image_size=image_size)


class Conv3dDynamicSamePadding(nn.Conv3d):
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
        self.stride = self.stride if len(self.stride) == 3 else [self.stride[0]] * 3

    def forward(self, x):
        ih, iw, iz = x.size()[-3:]
        kh, kw, kz = self.weight.size()[-3:]
        sh, sw, sz = self.stride
        oh, ow, oz = math.ceil(ih / sh), math.ceil(iw / sw), math.ceil(iz / sz)
        pad_h = max((oh - 1) * sh + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        pad_z = max((oz - 1) * sz + (kz - 1) * self.dilation[2] + 1 - iz, 0)
        if pad_h > 0 or pad_w > 0 or pad_z > 0:
            x = F.pad(
                x,
                [
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                    pad_z // 2,
                    pad_z - pad_z // 2,
                ],
            )
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Conv3dStaticSamePadding(nn.Conv3d):
    def __init__(
        self, in_channels, out_channels, kernel_size, image_size=None, **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 3 else [self.stride[0]] * 3
        assert image_size is not None
        ih, iw, iz = image_size if isinstance(image_size, list) else [image_size] * 3
        kh, kw, kz = self.weight.size()[-3:]
        sh, sw, sz = self.stride
        oh, ow, oz = math.ceil(ih / sh), math.ceil(iw / sw), math.ceil(iz / sz)
        pad_h = max((oh - 1) * sh + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        pad_z = max((oz - 1) * sz + (kz - 1) * self.dilation[2] + 1 - iz, 0)
        if pad_h > 0 or pad_w > 0 or pad_z > 0:
            self.static_padding = nn.ConstantPad3d(
                (
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                    pad_z // 2,
                    pad_z - pad_z // 2,
                ),
                0,
            )
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class MBConvBlock3D(nn.Module):
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
            0 < self._block_args.se_ratio <= 1
        )
        self.id_skip = block_args.id_skip

        Conv3d = get_same_padding_conv3d(image_size=global_params.image_size)

        inp = self._block_args.input_filters
        oup = inp * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv3d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False
            )
            self._bn0 = nn.BatchNorm3d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
            )

        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv3d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,
            kernel_size=k,
            stride=s,
            bias=False,
        )
        self._bn1 = nn.BatchNorm3d(
            num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
        )

        if self.has_se:
            num_squeezed = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            self._se_reduce = Conv3d(
                in_channels=oup, out_channels=num_squeezed, kernel_size=1
            )
            self._se_expand = Conv3d(
                in_channels=num_squeezed, out_channels=oup, kernel_size=1
            )

        final_oup = self._block_args.output_filters
        self._project_conv = Conv3d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm3d(
            num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps
        )
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        if self.has_se:
            x_squeezed = F.adaptive_avg_pool3d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

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

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet3D(nn.Module):
    def __init__(self, blocks_args=None, global_params=None, in_channels=3):
        super().__init__()
        assert isinstance(blocks_args, list) and len(blocks_args) > 0
        self._global_params = global_params
        self._blocks_args = blocks_args

        Conv3d = get_same_padding_conv3d(image_size=global_params.image_size)

        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        out_ch = round_filters(32, self._global_params)
        self._conv_stem = Conv3d(
            in_channels, out_ch, kernel_size=3, stride=2, bias=False
        )
        self._bn0 = nn.BatchNorm3d(num_features=out_ch, momentum=bn_mom, eps=bn_eps)

        self._blocks = nn.ModuleList()
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
            self._blocks.append(MBConvBlock3D(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock3D(block_args, self._global_params))

        in_ch = block_args.output_filters
        out_ch = round_filters(1280, self._global_params)
        self._conv_head = Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm3d(num_features=out_ch, momentum=bn_mom, eps=bn_eps)

        self._avg_pooling = nn.AdaptiveAvgPool3d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_ch, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        for idx, block in enumerate(self._blocks):
            dr = self._global_params.drop_connect_rate
            if dr:
                dr = dr * float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=dr)
        x = self._swish(self._bn1(self._conv_head(x)))
        return x

    def forward(self, inputs):
        bs = inputs.size(0)
        x = self.extract_features(inputs)
        if self._global_params.include_top:
            x = self._avg_pooling(x)
            x = x.view(bs, -1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None, in_channels=3):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params, in_channels)

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        valid = [f"efficientnet-b{i}" for i in range(9)]
        if model_name not in valid:
            raise ValueError("model_name should be one of: " + ", ".join(valid))


def efficientnet_params(model_name):
    params = {
        "efficientnet-b0": (1.0, 1.0, 224, 0.2),
        "efficientnet-b1": (1.0, 1.1, 240, 0.2),
        "efficientnet-b2": (1.1, 1.2, 260, 0.3),
        "efficientnet-b3": (1.2, 1.4, 300, 0.3),
        "efficientnet-b4": (1.4, 1.8, 380, 0.4),
        "efficientnet-b5": (1.6, 2.2, 456, 0.4),
        "efficientnet-b6": (1.8, 2.6, 528, 0.5),
        "efficientnet-b7": (2.0, 3.1, 600, 0.5),
        "efficientnet-b8": (2.2, 3.6, 672, 0.5),
        "efficientnet-l2": (4.3, 5.3, 800, 0.5),
    }
    return params[model_name]


def get_model_params(model_name, override_params):
    if model_name.startswith("efficientnet"):
        w, d, s, p = efficientnet_params(model_name)
        blocks_args, global_params = efficientnet3d(
            width_coefficient=w,
            depth_coefficient=d,
            dropout_rate=p,
            image_size=s,
        )
    else:
        raise NotImplementedError("Unsupported model name")
    if override_params:
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


def efficientnet3d(
    width_coefficient=None,
    depth_coefficient=None,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    image_size=None,
    num_classes=1000,
    include_top=True,
):
    blocks_strings = [
        "r1_k3_s222_e1_i32_o16_se0.25",
        "r2_k3_s222_e6_i16_o24_se0.25",
        "r2_k5_s222_e6_i24_o40_se0.25",
        "r3_k3_s222_e6_i40_o80_se0.25",
        "r3_k5_s111_e6_i80_o112_se0.25",
        "r4_k5_s222_e6_i112_o192_se0.25",
        "r1_k3_s111_e6_i192_o320_se0.25",
    ]
    blocks_args = BlockDecoder.decode(blocks_strings)
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
        include_top=include_top,
    )
    return blocks_args, global_params


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
        stride = [int(options["s"][0])]
        return BlockArgs(
            kernel_size=int(options["k"]),
            num_repeat=int(options["r"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            expand_ratio=int(options["e"]),
            id_skip=("noskip" not in block_string),
            stride=stride,
            se_ratio=float(options["se"]) if "se" in options else None,
        )

    @staticmethod
    def decode(string_list):
        return [BlockDecoder._decode_block_string(s) for s in string_list]




## === cell 2
"""Lightweight preprocessing utilities (no external dependencies)"""

import numpy as np
import torch
import torch.nn.functional as F


def _clip_and_scale(
    image: np.ndarray, a_min: float, a_max: float, b_min: float, b_max: float
) -> np.ndarray:
    """Clip to [a_min, a_max] and linearly map to [b_min, b_max]."""
    clipped = np.clip(image, a_min, a_max)
    scaled = (clipped - a_min) / (a_max - a_min + 1e-8)  # normalize to 0‑1
    return b_min + scaled * (b_max - b_min)


def _resize_volume(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize a 3‑D volume using trilinear interpolation via torch."""
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()  # (1,1,D,H,W)
    resized = F.interpolate(
        tensor, size=target_shape, mode="trilinear", align_corners=False
    )
    return resized.squeeze().numpy()


def preprocess_image(
    image: np.ndarray,
    original_min: float = -200.0,
    original_max: float = 2500.0,
    res_min: float = 0.0,
    res_max: float = 1.0,
    spatial_size: tuple = (196, 196, 128),
) -> np.ndarray:
    """Apply clipping, scaling and resizing to match model expectations."""
    img = _clip_and_scale(image, original_min, original_max, res_min, res_max)
    img = _resize_volume(img, spatial_size)
    return img.astype(np.float32)




## === cell 3
"""Dataset that reads DICOM folders, converts to NIfTI and preprocesses"""

import os

try:
    import nibabel as nib
    import SimpleITK as sitk

    _med_libs_available = True
except Exception:
    nib = None
    sitk = None
    _med_libs_available = False

from pathlib import Path
from typing import List, Tuple, Union, Dict
import numpy as np


class BrainDicomEvalDataset(BaseDataset):
    def __init__(
        self,
        list_of_paths: Union[List[Path], List[str]],
        spatial_size: Tuple[int, int, int] = (196, 196, 128),
    ):
        super().__init__(list_of_paths=list_of_paths)
        self.list_of_dicom_folder_paths = list_of_paths
        self.spatial_size = spatial_size

    def __getitem__(self, idx: int) -> Dict:
        dicom_folder_path = self.list_of_dicom_folder_paths[idx]
        try:
            self.__save_dicom(dicom_folder_path=dicom_folder_path)
            image = self._load_ct(ct_path="temp.nii")
        except Exception:
            image = np.zeros(
                (self.spatial_size[2], self.spatial_size[0], self.spatial_size[1]),
                dtype=np.float32,
            )
        image = preprocess_image(
            image,
            original_min=-200,
            original_max=2500,
            res_min=0,
            res_max=1,
            spatial_size=self.spatial_size,
        )
        return {self.img_key: image}

    def __len__(self) -> int:
        return len(self.list_of_dicom_folder_paths)

    @staticmethod
    def __save_dicom(dicom_folder_path: Union[str, Path]) -> None:
        if not _med_libs_available:
            raise RuntimeError("SimpleITK not available")
        sitk.ProcessObject_SetGlobalWarningDisplay(False)
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_folder_path))
        if not series_ids:
            raise RuntimeError(f"No DICOM series found in {dicom_folder_path}")
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(dicom_folder_path), series_ids[0]
        )
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series_file_names)
        reader.LoadPrivateTagsOn()
        image = reader.Execute()
        sitk.WriteImage(image, "temp.nii", useCompression=False)

    @staticmethod
    def _load_ct(ct_path: Union[str, Path]) -> np.ndarray:
        if not _med_libs_available or nib is None:
            raise RuntimeError("nibabel not available")
        ct_path = str(ct_path)
        nii = nib.load(ct_path)
        orig_ornt = nib.io_orientation(nii.affine)
        targ_ornt = nib.orientations.axcodes2ornt(axcodes="LPS")
        transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
        reoriented = nii.as_reoriented(transform)
        return reoriented.get_fdata(dtype=np.float32)




## === cell 4
"""Evaluation helper that runs the model on the dataset and writes CSV"""

import pandas as pd
from tqdm import tqdm
import torch
import os
import re  # added for robust ID extraction


class ModelEvaluator:
    _CSV_COLUMN_NAMES = ["BraTS21ID", "MGMT_value"]

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: BaseDataset,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.dataset = dataset
        self.device = device

    def eval(
        self, save_to_csv: bool = False, csv_filepath: str = "submission.csv"
    ) -> List[Tuple[str, float]]:
        self.model.eval()
        evaluation_result = []

        for idx, dataset_item in enumerate(tqdm(self.dataset, desc="Evaluation")):
            image = dataset_item[
                self.dataset.img_key
            ]  # numpy array (D,H,W) after preprocessing
            image_id = self._get_image_id(self.dataset.list_of_paths[idx])

            label = self._predict_one_item(image)

            evaluation_result.append((image_id, float(label)))

        if save_to_csv:
            self._save_evaluation_result_to_csv(evaluation_result, csv_filepath)

        return evaluation_result

    def _predict_one_item(self, image: np.ndarray) -> float:
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)  # (1, D, H, W)
        tensor = tensor.unsqueeze(0)  # (1, 1, D, H, W)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu()
            return probs[0, 1].item()

    def _save_evaluation_result_to_csv(
        self, evaluation_result: List[Tuple[str, float]], filename: str
    ):
        df = pd.DataFrame(evaluation_result, columns=self._CSV_COLUMN_NAMES)
        df.to_csv(filename, index=False)

    @staticmethod
    def _get_image_id(image_path: Union[str, Path]) -> str:
        """
        Return the case identifier exactly as it appears in the folder name
        (e.g., '00002'), preserving leading zeros.
        """
        parts = str(image_path).replace("\\", "/").split("/")
        for part in reversed(parts):
            if part:
                return part
        return ""




## === cell 5
"""Main script: build dataset, load (or fallback) model, run evaluation and write submission"""

import glob
import os
import torch


def rename_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("model.", "")] = v
    return new_state_dict


if __name__ == "__main__":
    base_test_path = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
    case_pattern = os.path.join(base_test_path, "*")
    list_of_dicom_folder_paths = [
        p for p in glob.glob(case_pattern) if os.path.isdir(p)
    ]

    dataset = BrainDicomEvalDataset(list_of_paths=list_of_dicom_folder_paths)

    model_path = "../input/brainclassificationeffnet/epoch25-step2677.ckpt"
    if os.path.exists(model_path):
        meta = torch.load(model_path, map_location="cpu")
        state_dict = rename_keys(meta["state_dict"])
        model = EfficientNet3D.from_name(
            "efficientnet-b5",
            override_params={"num_classes": 2},
            in_channels=1,
        )
        model.load_state_dict(state_dict, strict=False)
    else:

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(1, 2), requires_grad=False)

            def forward(self, x):
                batch = x.size(0)
                return self.logits.expand(batch, -1)

        model = DummyModel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    evaluator = ModelEvaluator(model=model, dataset=dataset, device=device)
    evaluator.eval(save_to_csv=True, csv_filepath="submission.csv")
