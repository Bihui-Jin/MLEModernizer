# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import math
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



## === cell 1
is_submission = True
data_path = "/kaggle/input/cassava-leaf-disease-classification"
csv_file_name = "train.csv" if not is_submission else "sample_submission.csv"
batch_size = 8
num_workers = 2

mean, std = 0.5, 0.5
efficient_net_version = 3  # (1.4, 1.2)

use_pre_trained_weight = True

pre_trained_weight_path = os.path.join(
    "/kaggle/input", "cassava-leaf-disease-classification-weight", "weight.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

do_head_calibration_if_imagenet = True

head_calibration_epochs = 5
head_calibration_lr = 3e-3
head_calibration_batch_size = 64
head_calibration_weight_decay = 1e-4

use_weighted_sampler_for_head_calibration = True

use_val_selection_for_head_calibration = True
head_calibration_val_frac = 0.1
head_calibration_seed = 123




## === cell 2
class CassavaLeafDiseaseDataset(Dataset):
    "Cassava Leaf Disease"

    def __init__(self, csv_file, root_dir, transform=None, train=True):
        """
        Args:
            csv_file (string): csv file path
            root_dir (string): directory path of exist all image
            transform (callable, optional): Optional transform for sample
        """
        self.cassava_leaf_disease = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.folder = "train_images" if train else "test_images"

    def __len__(self):
        return len(self.cassava_leaf_disease)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = "/".join(
            [self.root_dir, self.folder, self.cassava_leaf_disease.image_id[idx]]
        )
        img = io.imread(img_name)

        if self.transform:
            img = self.transform(img)

        label = int(self.cassava_leaf_disease.label.iloc[idx])
        sample = (img, label)
        return sample




## === cell 3
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((mean, mean, mean), (std, std, std)),
    ]
)

dataset = CassavaLeafDiseaseDataset(
    csv_file="/".join([data_path, csv_file_name]),
    root_dir=data_path,
    transform=transform,
    train=not is_submission,
)

data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)




## === cell 4
class ConvUnit(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        groups: int = 1,
        batch_norm=True,
        activation=True,
    ):
        super().__init__()
        modules = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                groups=groups,
                bias=not batch_norm,
            )
        ]
        if batch_norm:
            modules.append(torch.nn.BatchNorm2d(out_channels))
        if activation:
            modules.append(torch.nn.LeakyReLU())
        self.sequence = torch.nn.Sequential(*modules)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.sequence(x)
        return x


def DConvUnit(in_channels: int, kernel_size: int, stride: int = 1) -> torch.nn.Module:
    padding = kernel_size // 2
    return ConvUnit(in_channels, in_channels, kernel_size, padding, stride, in_channels)


def EConvUnit(in_channels: int, factor: int = 6) -> torch.nn.Module:
    return ConvUnit(in_channels, factor * in_channels, 1)


def PConvUnit(in_channels, out_channels, activation=False) -> torch.nn.Module:
    return ConvUnit(in_channels, out_channels, 1, activation=activation)


class DSConvUnit(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation=True,
    ):
        super().__init__()
        modules = [
            DConvUnit(in_channels, kernel_size, stride),
            PConvUnit(in_channels, out_channels, activation),
        ]
        self.sequence = torch.nn.Sequential(*modules)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.sequence(x)
        return x


class SEUnit(torch.nn.Module):
    def __init__(self, in_channels: int, se_ratio: float):
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.sequence = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            ConvUnit(in_channels, se_channels, 1, batch_norm=False),
            ConvUnit(se_channels, in_channels, 1, batch_norm=False, activation=False),
            torch.nn.Sigmoid(),
        )
        self.out_channels = in_channels

    def forward(self, x):
        y = self.sequence(x)
        z = x * y
        return z


class BottleneckUnit(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 6,
        kernel_size: int = 3,
        stride: int = 1,
        has_se: bool = True,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        modules = []
        self.residual = stride == 1 and in_channels == out_channels
        hid_channels = factor * in_channels
        if factor > 1:
            modules.append(EConvUnit(in_channels, factor))
        modules.append(DConvUnit(hid_channels, kernel_size, stride))
        if has_se:
            calced_se_ratio = se_ratio / factor
            modules.append(SEUnit(hid_channels, calced_se_ratio))
        modules.append(PConvUnit(hid_channels, out_channels, False))
        self.sequence = torch.nn.Sequential(*modules)
        self.activation = torch.nn.LeakyReLU()

        self.out_channels = out_channels

    def forward(self, x):
        y = self.sequence(x)
        z = self.activation(x + y if self.residual else y)
        return z




## === cell 5
def round_filters(filters, coef: float, divisor: int) -> int:
    filters *= coef
    filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    return filters


def round_repeats(repeats, coef):
    repeats = int(math.ceil(repeats * coef)) if repeats > 0 else -repeats
    return repeats


class BlockArgs:
    def __init__(
        self, module: torch.nn.Module, filters: int, num_repeats: int = 1, **kwargs
    ):
        self.module = module
        self.filters = filters
        self.num_repeats = num_repeats
        self.kwargs = kwargs

    def __call__(
        self,
        in_channels: int,
        width_coef: float = 1.0,
        depth_coef: float = 1.0,
        divisor: int = 8,
    ) -> torch.nn.Module:
        out_channels = round_filters(self.filters, width_coef, divisor)
        num_repeats = round_repeats(self.num_repeats, depth_coef)
        modules = []
        kwargs_local = dict(self.kwargs)
        for rep_i in range(num_repeats):
            modules.append(self.module(in_channels, out_channels, **kwargs_local))
            if "stride" in kwargs_local:
                del kwargs_local["stride"]
            in_channels = out_channels
        return torch.nn.Sequential(*modules), in_channels


EFFICIENT_NET_BLOCK_ARGS = [
    BlockArgs(ConvUnit, 32, -1, kernel_size=3, stride=2, padding=1),
    BlockArgs(BottleneckUnit, 16, 1, factor=1),
    BlockArgs(BottleneckUnit, 24, 2, stride=2),
    BlockArgs(BottleneckUnit, 40, 2, kernel_size=5, stride=2),
    BlockArgs(BottleneckUnit, 80, 3, stride=2),
    BlockArgs(BottleneckUnit, 112, 3, kernel_size=5),
    BlockArgs(BottleneckUnit, 192, 4, kernel_size=5, stride=2),
    BlockArgs(BottleneckUnit, 320, 1),
]

EFFICIENT_NET_COMPOUND_COEF = [
    (1.0, 1.0),
    (1.1, 1.0),
    (1.2, 1.1),
    (1.4, 1.2),
    (1.8, 1.4),
    (2.2, 1.6),
    (2.6, 1.8),
    (3.1, 2.0),
]


class EfficientNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_of_class: int = 1000,
        depth_coef: float = 1.0,
        width_coef: float = 1.0,
        hidden_channels: int = 1280,
        block_args=EFFICIENT_NET_BLOCK_ARGS,
    ):
        super().__init__()
        modules = []
        self.hidden_channels = round_filters(hidden_channels, width_coef, 8)
        for block_arg in block_args:
            module, in_channels = block_arg(in_channels, width_coef, depth_coef)
            modules.append(module)
        modules.append(PConvUnit(in_channels, self.hidden_channels, True))
        modules.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        self.sequence = torch.nn.Sequential(*modules)
        self.linear = torch.nn.Linear(self.hidden_channels, num_of_class)

    def forward(self, x):
        x = self.sequence(x)
        x = x.view(-1, self.hidden_channels)
        x = self.linear(x)
        return x




## === cell 6
def _try_load_local_checkpoint(model, path: str) -> bool:
    if not (path and os.path.exists(path)):
        return False
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    if isinstance(ckpt, dict):
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    return True


def _load_torchvision_efficientnet_b3_strict_map(model: torch.nn.Module) -> bool:
    """
    NOTE(score): Map torchvision EfficientNet-B3 ImageNet weights into this implementation
    so backbone features aren't random when no local cassava checkpoint exists.
    """
    try:
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
    except Exception:
        return False

    try:
        tv = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    except Exception:
        return False

    tv_sd = tv.state_dict()
    my_sd = model.state_dict()
    new_sd = dict(my_sd)

    def copy_param(my_key, tv_key):
        if (
            my_key in my_sd
            and tv_key in tv_sd
            and my_sd[my_key].shape == tv_sd[tv_key].shape
        ):
            new_sd[my_key] = tv_sd[tv_key].detach().clone()
            return 1
        return 0

    copied = 0

    copied += copy_param("sequence.0.sequence.0.weight", "features.0.0.weight")
    copied += copy_param("sequence.0.sequence.1.weight", "features.0.1.weight")
    copied += copy_param("sequence.0.sequence.1.bias", "features.0.1.bias")
    copied += copy_param(
        "sequence.0.sequence.1.running_mean", "features.0.1.running_mean"
    )
    copied += copy_param(
        "sequence.0.sequence.1.running_var", "features.0.1.running_var"
    )
    copied += copy_param(
        "sequence.0.sequence.1.num_batches_tracked", "features.0.1.num_batches_tracked"
    )

    stage_tv_to_my = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}

    for tv_stage, my_stage in stage_tv_to_my.items():
        tv_stage_mod = getattr(tv.features, str(tv_stage))
        for bi, _ in enumerate(tv_stage_mod):
            my_prefix = f"sequence.{my_stage}.{bi}."
            tv_prefix = f"features.{tv_stage}.{bi}."

            has_expand = any(
                k.startswith(tv_prefix + "block.0.0.weight") for k in tv_sd.keys()
            )
            if has_expand:
                copied += copy_param(
                    my_prefix + "sequence.0.sequence.0.weight",
                    tv_prefix + "block.0.0.weight",
                )
                copied += copy_param(
                    my_prefix + "sequence.0.sequence.1.weight",
                    tv_prefix + "block.0.1.weight",
                )
                copied += copy_param(
                    my_prefix + "sequence.0.sequence.1.bias",
                    tv_prefix + "block.0.1.bias",
                )
                copied += copy_param(
                    my_prefix + "sequence.0.sequence.1.running_mean",
                    tv_prefix + "block.0.1.running_mean",
                )
                copied += copy_param(
                    my_prefix + "sequence.0.sequence.1.running_var",
                    tv_prefix + "block.0.1.running_var",
                )
                copied += copy_param(
                    my_prefix + "sequence.0.sequence.1.num_batches_tracked",
                    tv_prefix + "block.0.1.num_batches_tracked",
                )

                my_dw = my_prefix + "sequence.1.sequence.0.weight"
                tv_dw = tv_prefix + "block.1.0.weight"
                my_dw_bn = my_prefix + "sequence.1.sequence.1."
                tv_dw_bn = tv_prefix + "block.1.1."
                copied += copy_param(my_dw, tv_dw)
                copied += copy_param(my_dw_bn + "weight", tv_dw_bn + "weight")
                copied += copy_param(my_dw_bn + "bias", tv_dw_bn + "bias")
                copied += copy_param(
                    my_dw_bn + "running_mean", tv_dw_bn + "running_mean"
                )
                copied += copy_param(my_dw_bn + "running_var", tv_dw_bn + "running_var")
                copied += copy_param(
                    my_dw_bn + "num_batches_tracked", tv_dw_bn + "num_batches_tracked"
                )

                se_idx = 2
                proj_idx = 3
                my_se = my_prefix + "sequence.2."
                my_proj = my_prefix + "sequence.3."
            else:
                my_dw = my_prefix + "sequence.0.sequence.0.weight"
                tv_dw = tv_prefix + "block.0.0.weight"
                my_dw_bn = my_prefix + "sequence.0.sequence.1."
                tv_dw_bn = tv_prefix + "block.0.1."
                copied += copy_param(my_dw, tv_dw)
                copied += copy_param(my_dw_bn + "weight", tv_dw_bn + "weight")
                copied += copy_param(my_dw_bn + "bias", tv_dw_bn + "bias")
                copied += copy_param(
                    my_dw_bn + "running_mean", tv_dw_bn + "running_mean"
                )
                copied += copy_param(my_dw_bn + "running_var", tv_dw_bn + "running_var")
                copied += copy_param(
                    my_dw_bn + "num_batches_tracked", tv_dw_bn + "num_batches_tracked"
                )

                se_idx = 1
                proj_idx = 2
                my_se = my_prefix + "sequence.1."
                my_proj = my_prefix + "sequence.2."

            if any(
                k.startswith(tv_prefix + f"block.{se_idx}.fc1.weight")
                for k in tv_sd.keys()
            ):
                copied += copy_param(
                    my_se + "sequence.1.sequence.0.weight",
                    tv_prefix + f"block.{se_idx}.fc1.weight",
                )
                copied += copy_param(
                    my_se + "sequence.1.sequence.0.bias",
                    tv_prefix + f"block.{se_idx}.fc1.bias",
                )
                copied += copy_param(
                    my_se + "sequence.2.sequence.0.weight",
                    tv_prefix + f"block.{se_idx}.fc2.weight",
                )
                copied += copy_param(
                    my_se + "sequence.2.sequence.0.bias",
                    tv_prefix + f"block.{se_idx}.fc2.bias",
                )

            copied += copy_param(
                my_proj + "sequence.0.weight", tv_prefix + f"block.{proj_idx}.0.weight"
            )
            copied += copy_param(
                my_proj + "sequence.1.weight", tv_prefix + f"block.{proj_idx}.1.weight"
            )
            copied += copy_param(
                my_proj + "sequence.1.bias", tv_prefix + f"block.{proj_idx}.1.bias"
            )
            copied += copy_param(
                my_proj + "sequence.1.running_mean",
                tv_prefix + f"block.{proj_idx}.1.running_mean",
            )
            copied += copy_param(
                my_proj + "sequence.1.running_var",
                tv_prefix + f"block.{proj_idx}.1.running_var",
            )
            copied += copy_param(
                my_proj + "sequence.1.num_batches_tracked",
                tv_prefix + f"block.{proj_idx}.1.num_batches_tracked",
            )

    copied += copy_param("sequence.9.sequence.0.weight", "features.8.0.weight")
    copied += copy_param("sequence.9.sequence.1.weight", "features.8.1.weight")
    copied += copy_param("sequence.9.sequence.1.bias", "features.8.1.bias")
    copied += copy_param(
        "sequence.9.sequence.1.running_mean", "features.8.1.running_mean"
    )
    copied += copy_param(
        "sequence.9.sequence.1.running_var", "features.8.1.running_var"
    )
    copied += copy_param(
        "sequence.9.sequence.1.num_batches_tracked", "features.8.1.num_batches_tracked"
    )

    model.load_state_dict(new_sd, strict=False)
    return copied > 150


model = EfficientNet(3, 5, *EFFICIENT_NET_COMPOUND_COEF[efficient_net_version])

loaded = False
used_torchvision_imagenet = False
if use_pre_trained_weight:
    loaded = _try_load_local_checkpoint(model, pre_trained_weight_path)
    if not loaded:
        loaded = _load_torchvision_efficientnet_b3_strict_map(model)
        used_torchvision_imagenet = loaded

if used_torchvision_imagenet:
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    dataset.transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

model = model.to(device)



## === cell 7
if is_submission and used_torchvision_imagenet and do_head_calibration_if_imagenet:
    from torch.utils.data import WeightedRandomSampler, Subset

    train_ds_full = CassavaLeafDiseaseDataset(
        csv_file=os.path.join(data_path, "train.csv"),
        root_dir=data_path,
        transform=dataset.transform,  # same normalization as inference
        train=True,
    )

    if use_val_selection_for_head_calibration:
        rng = np.random.default_rng(head_calibration_seed)
        n = len(train_ds_full)
        idxs = np.arange(n)
        rng.shuffle(idxs)
        n_val = max(1, int(n * head_calibration_val_frac))
        val_idxs = idxs[:n_val].tolist()
        tr_idxs = idxs[n_val:].tolist()
        train_ds = Subset(train_ds_full, tr_idxs)
        val_ds = Subset(train_ds_full, val_idxs)
    else:
        train_ds = train_ds_full
        val_ds = None

    if use_weighted_sampler_for_head_calibration:
        if isinstance(train_ds, Subset):
            base_labels = train_ds.dataset.cassava_leaf_disease["label"].to_numpy()
            labels_np = base_labels[np.asarray(train_ds.indices)]
        else:
            labels_np = train_ds.cassava_leaf_disease["label"].to_numpy()

        class_counts = np.bincount(labels_np, minlength=5).astype(np.float64)
        class_weights = 1.0 / np.maximum(class_counts, 1.0)
        sample_weights = class_weights[labels_np]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=head_calibration_batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=head_calibration_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=head_calibration_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        val_loader = None

    for p in model.sequence.parameters():
        p.requires_grad = False
    for p in model.linear.parameters():
        p.requires_grad = True

    model.sequence.eval()
    model.linear.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.linear.parameters(),
        lr=head_calibration_lr,
        weight_decay=head_calibration_weight_decay,
    )

    best_sd = None
    best_val_acc = -1.0

    for _ in range(head_calibration_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if val_loader is not None:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
                    outputs = model(inputs)
                    pred = torch.argmax(outputs, dim=1)
                    correct += int((pred == labels).sum().item())
                    total += int(labels.numel())
            val_acc = correct / max(1, total)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_sd = {
                    k: v.detach().clone() for k, v in model.linear.state_dict().items()
                }
            model.sequence.eval()
            model.linear.train()

    if best_sd is not None:
        model.linear.load_state_dict(best_sd)

    model.eval()



## === cell 8
if is_submission:
    model.eval()

    submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
    submission_ids = submission["image_id"].tolist()

    pred_by_id = {}

    with torch.no_grad():
        offset = 0
        for data in data_loader:
            inputs, _labels = data
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            batch_preds = predicted.detach().cpu().numpy().tolist()

            batch_ids = (
                dataset.cassava_leaf_disease["image_id"]
                .iloc[offset : offset + len(batch_preds)]
                .tolist()
            )
            for img_id, pred in zip(batch_ids, batch_preds):
                pred_by_id[img_id] = int(pred)
            offset += len(batch_preds)

    submission["label"] = [pred_by_id.get(img_id, 0) for img_id in submission_ids]
    submission.to_csv("submission.csv", index=False)




## === cell 9
def progress_print(comment, current, total, num_of_print=50, slow=5):
    end = "\r" if current < total else "\n"
    symbol = ["\\", "/", "-"]
    done = int((current - 1) / total * num_of_print)
    doing = symbol[(current % (len(symbol) * slow) // slow)] if current < total else ""
    yet = max(0, num_of_print - 1 - done)
    print(comment + ": " + "#" * done + doing + "*" * yet, end=end)


def get_confusion_matrix(model, data_loader, num_classes):
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            predicts = predicted.squeeze()
            for p, l in zip(predicts, labels):
                confusion_matrix[p.item()][int(l)] += 1

            progress_print("confusion_matrix", i, len(data_loader), slow=3)
    return confusion_matrix




## === cell 10
def show_confusion_matrix(confusion_matrix):
    diagonal_confusion_matrix = deepcopy(confusion_matrix)
    np.fill_diagonal(diagonal_confusion_matrix, 0)
    plt.figure(figsize=(2, 1))
    plt.matshow(confusion_matrix, cmap="gray")
    plt.matshow(diagonal_confusion_matrix, cmap="gray")
    plt.show()
    print(confusion_matrix)




## === cell 11
if not is_submission:
    confusion_matrix = get_confusion_matrix(model, data_loader, 5)



## === cell 12
if not is_submission:
    row_sums = confusion_matrix.sum(axis=1)
    normalized_confusion_matrix = confusion_matrix / row_sums[:, np.newaxis]
    show_confusion_matrix(normalized_confusion_matrix)
