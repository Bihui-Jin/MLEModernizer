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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tqdm==4.67.1

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

# 5. Target score

0.8830462375339981

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the inference-time crash by ensuring the ensemble loop actually finds checkpoint files (your glob path is misspelled) and by making the code robust when zero checkpoints are found. I also prevent a separate hidden bug in CutMix (`rand_bbox` using an undefined `lam`) by passing `lam` explicitly—this is score-neutral since `TRAINING=False`, but it makes the script runnable if you enable training later. Finally, I switch inference to `torch.no_grad()` to avoid unnecessary memory use and ensure a valid `submission.csv` is always written (using a safe fallback prediction if no weights are available).'
- What this solution (achieved 0.05531) has done: 'Your score is extremely far below the target (0.055 → 0.883), which strongly suggests you are effectively predicting a near-constant label because you are not loading the intended trained weights. I make the smallest change that moves score upward: restrict checkpoint discovery to the specific `WEIGHT` path you provided (and its `.pkl/.pth/.pt` variants) plus any `.pkl` in that directory, instead of ensembling arbitrary `.pkl` files from other datasets/competitions. I also fix a silent but critical bug in test preprocessing (you currently apply training-time random horizontal flip to test), switching test to the deterministic `val_transform` so predictions are stable and match typical evaluation semantics. Finally, I make loading robust to common checkpoint formats (raw `state_dict`, `{"state_dict":...}`, `{"model":...}`) without changing the model architecture or training logic, and still always write a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your score (0.055) is far below the target (0.883), which strongly indicates the model is not actually loading the intended trained weights and is effectively predicting near-random/constant outputs. I make the smallest changes that directly increase correctness: (1) fix checkpoint discovery so it finds your intended file(s) under `/kaggle/input/cutmix/` (your current `../input/...` path is wrong in Kaggle), (2) load checkpoints via `model.load_state_dict(..., strict=False)` to avoid silently “loading” almost nothing, and (3) use `softmax` and average probabilities across checkpoints (same ensemble idea, but properly calibrated for argmax). Core model architecture and training code remain unchanged; only inference-time robustness/weight-loading is corrected to move accuracy upward toward the target.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, which most likely means you are not loading the intended trained weights (or you are accidentally ensembling many irrelevant checkpoints from the directory), producing near-random predictions. I make the smallest inference-only changes to (1) discover only the specific WEIGHT file (and its common extensions) plus an optional small set of “matching” fold checkpoints, (2) ensure checkpoints are actually applied by filtering to those with a plausible number of matching keys (skip bad/incompatible ones), and (3) harden output alignment by using `sample_submission.csv` order so `image_id` ordering always matches Kaggle’s expected test set listing. This preserves your model architecture and training loop, and only adjusts inference robustness to move accuracy upward toward the target band.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.055) is so far below the target (0.883) that the most likely cause is still “wrong/no weights actually being applied,” which makes predictions near-constant or random. I make the smallest inference-only changes to reliably locate and load the intended checkpoint(s): (1) fix the WEIGHT path to the correct Kaggle input location and allow the common case where the directory contains the checkpoint files, (2) improve checkpoint discovery to include “*.pkl/*.pth/*.pt” inside the WEIGHT directory (not just prefix-matching a filename that may not exist), and (3) make the key-match gate less brittle and explicitly drop the classifier head if the checkpoint is a 1000-class ImageNet state_dict (so the backbone still loads). These changes preserve your architecture and training code, but should move accuracy sharply upward by ensuring you’re using the trained cassava weights rather than falling back.'
- What this solution (achieved 0.05531) has done: 'Your score (0.055) is so far below the target (0.883) that the dominant issue is almost certainly “no real cassava-trained weights are being loaded,” causing near-random/constant predictions. I make a minimal inference-only change to guarantee we actually load a valid checkpoint from the Kaggle dataset directory by expanding checkpoint discovery to search the whole `/kaggle/input/cutmix/` folder (your current `WEIGHT` points to a likely non-existent file/prefix). I also remove the overly-brittle “match ratio” gate (keep `strict=False` + drop incompatible `fc.*`), because it can wrongly skip good checkpoints and leave you with the fallback. Core model, transforms, and training loop remain unchanged; only checkpoint discovery/loading robustness is adjusted to move accuracy sharply upward toward the target.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.055) is so far below the target (0.883) that the dominant issue is still almost certainly “no valid cassava-trained checkpoint is being loaded,” so the model effectively outputs garbage/near-constant predictions. I make the smallest inference-only changes to reliably (1) locate real checkpoint files under the actual Kaggle input directory, and (2) load them correctly when they are full saved models vs. plain `state_dict`s (a very common mismatch that silently breaks loading). I also fix one score-critical preprocessing mismatch: your ResNeXt backbone expects ImageNet-style 224-ish inputs; forcing 448 can degrade accuracy a lot if the checkpoint was trained at 224, so I switch inference resizing to 224 while keeping your normalization and model intact. The training code/architecture/loss remain unchanged; only checkpoint discovery/loading and deterministic inference preprocessing are adjusted to move accuracy sharply upward toward the target.'
- What this solution (achieved 0.05531) has done: 'Your gap to target is huge (0.055 → 0.883), so the dominant failure is still that inference is effectively untrained: the code never loads real cassava-trained weights because `WEIGHT` points to a non-existent dataset (`/kaggle/input/cutmix/...`) and the model falls back to near-constant predictions. I make the smallest score-relevant inference-only changes to (1) automatically discover checkpoints inside the competition dataset’s input folders (`/kaggle/input/cassava-leaf-disease-classification/` and `/kaggle/input/`), not just `/kaggle/input/cutmix/`, (2) handle common checkpoint key prefixes like `model.` / `net.` in addition to `module.`, and (3) enforce deterministic test preprocessing (already using `val_transform`) and correct submission alignment (already using `sample_submission.csv`). This preserves your model architecture/training code and only fixes weight discovery/loading so the model can actually use trained parameters and move accuracy sharply upward toward the target band.'

# 9. Code solution

## === cell 0
BATCH_SIZE = 16
EPOCH = 5
WD = 1e-4
LR = 0.0001
VAL_RATIO = 0.2
PHASE = ["train", "val"]
BETA = 1.0
CUTMIX_PROB = 1.0
TRAINING = False

WEIGHT = "/kaggle/input/cutmix/resnext_kfold0_17_0.839"
K_FOLD = 5



## === cell 1
import os
import glob
import numpy as np



## === cell 2
from torch.utils.data.dataset import Dataset
import pandas as pd
from PIL import Image


class CLD_Dataset(Dataset):
    def __init__(self, image_root, label_path=None, transform=None, return_name=False):
        super(CLD_Dataset, self).__init__()
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(image_root, "*.jpg"))
        self.image_paths.sort()
        if not return_name:
            self.label = pd.read_csv(label_path, index_col="image_id")
        self.return_name = return_name

    def __getitem__(self, x):
        img = Image.open(self.image_paths[x]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.return_name:
            return img, self.image_paths[x].split("/")[-1]
        else:
            label = self.label.loc[self.image_paths[x].split("/")[-1]].label
            return img, label

    def __len__(self):
        return len(self.image_paths)




## === cell 3
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import KFold

train_transform = transform.Compose(
    [
        transform.Resize((448, 448)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

all_train_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    train_transform,
)
dataset_size = len(all_train_dataset)

fold_dataloader = []
if K_FOLD != 1:
    kf = KFold(K_FOLD, shuffle=True, random_state=42)

    index = 0
    for train_idx, val_idx in kf.split(range(len(all_train_dataset))):
        train_dataset = torch.utils.data.Subset(all_train_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(all_train_dataset, val_idx)
        fold_dataloader.append(
            {
                "train": DataLoader(
                    train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
                ),
                "val": DataLoader(
                    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
                ),
            }
        )
        index += 1
    print(f"Prepared {len(fold_dataloader)} fold dataloaders")
else:
    fold_dataloader.append(
        {
            "train": DataLoader(
                all_train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
            ),
            "val": None,
        }
    )



## === cell 4
val_transform = transform.Compose(
    [
        transform.Resize((224, 224)),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 5
import torch
import torch.nn as nn



## === cell 6
__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    load = []
    not_load = []
    if pretrained:
        state_dict = torch.load(
            "../input/resnext50-32x4d/resnext50_32x4d.pth".format(arch),
            map_location="cpu",
        )
        for name, param in state_dict.items():
            if name in model.state_dict():
                try:
                    load.append(name)
                    model.state_dict()[name].copy_(param)
                except Exception:
                    not_load.append(name)

    print("Load : {} layers".format(len(load)))
    print("Miss : {} layers".format(len(not_load)))
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext_50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )




## === cell 7
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(device)


def create_new_model():
    return resnext50_32x4d(num_classes=5, pretrained=True).to(device)




## === cell 8
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)




## === cell 9
def create_loss_opti():
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
    )
    return criterion, optimizer, lr_scheduler




## === cell 10
pass




## === cell 11
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2




## === cell 12
pass




## === cell 13
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, acc):
        self.reset()
        self.acc = acc

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, batch):
        self.value = value
        if self.acc:
            self.sum += value
        else:
            self.sum += value * batch
        self.count += batch
        self.avg = self.sum / self.count




## === cell 14
pass




## === cell 15
def train_step(model, criterion, optimizer, image, label, phase):
    b_image = image.to(device)
    b_label = label.to(device)

    r = np.random.rand(1)
    if BETA > 0 and r < CUTMIX_PROB:
        lam = np.random.beta(BETA, BETA)
        rand_index = torch.randperm(b_image.size()[0]).to(device)
        target_a = b_label
        target_b = b_label[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(b_image.size(), lam)
        b_image[:, :, bbx1:bbx2, bby1:bby2] = b_image[
            rand_index, :, bbx1:bbx2, bby1:bby2
        ]
        lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (b_image.size()[-1] * b_image.size()[-2])
        )

        output = model(b_image)
        loss = criterion(output, target_a) * lam + criterion(output, target_b) * (
            1.0 - lam
        )
    else:
        output = model(b_image)
        loss = criterion(output, b_label)

    _, predicted = torch.max(output.data, dim=1)
    correct = (predicted.cpu() == label).sum().item()
    if phase == "train":
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct, loss.item()




## === cell 16
from tqdm import tqdm

max_acc = 0.0
ACCMeter = []
LOSSMeter = []
for i in range(K_FOLD):
    ACCMeter.append(AverageMeter(True))
    LOSSMeter.append(AverageMeter(False))

if TRAINING:
    for index, dataloader in enumerate(fold_dataloader):
        model = create_new_model()
        criterion, optimizer, lr_scheduler = create_loss_opti()
        Best_ACC = 0.0
        tmp_ACCMeter = AverageMeter(True)
        tmp_LOSSMeter = AverageMeter(False)
        for epoch in range(1, EPOCH + 1):
            correct_t = 0
            total = 0
            loss_t = 0.0
            for phase in PHASE:
                if phase == "train":
                    model.train(True)
                else:
                    model.train(False)

                for image, label in tqdm(
                    dataloader[phase],
                    total=len(dataloader[phase]),
                    position=0,
                    leave=True,
                ):
                    correct, loss = train_step(
                        model, criterion, optimizer, image, label, phase
                    )

                    if phase == "val":
                        tmp_ACCMeter.update(correct, label.size(0))
                        tmp_LOSSMeter.update(loss, label.size(0))
                        total += label.size(0)
                        loss_t += loss * label.size(0)
                        correct_t += correct

                if phase == "val" and Best_ACC < tmp_ACCMeter.avg:
                    Best_ACC = tmp_ACCMeter.avg
                    ACCMeter[index] = tmp_ACCMeter
                    LOSSMeter[index] = tmp_LOSSMeter
                    torch.save(
                        model.state_dict(),
                        "./resnext50_32x4d_kfold_{}_{}_{:.2f}.pkl".format(
                            index + 1, epoch, tmp_ACCMeter.avg
                        ),
                    )

            lr_scheduler.step()
            print(
                "Fold : {}/ {} Epoch : {} / {} loss : {:.6f} ACC : {:.6f}".format(
                    index + 1, K_FOLD, epoch, EPOCH, loss_t / total, correct_t / total
                )
            )



## === cell 17
pass



## === cell 18
acc_sum = 0
loss_sum = 0
for i in range(K_FOLD):
    acc_sum += ACCMeter[i].avg
    loss_sum += LOSSMeter[i].avg

print("K-fold {} ACC : {:.6f}".format(K_FOLD, acc_sum / K_FOLD))
print("K-fold {} ACC : {:.6f}".format(K_FOLD, loss_sum / K_FOLD))



## === cell 19
pass



## === cell 20
import pandas as pd

test_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    transform=val_transform,
    return_name=True,
)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    return ckpt_obj


def _strip_known_prefixes(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    prefixes = ("module.", "model.", "net.")
    keys = list(state_dict.keys())
    for p in prefixes:
        if any(k.startswith(p) for k in keys):
            state_dict = {k[len(p) :]: v for k, v in state_dict.items()}
            keys = list(state_dict.keys())
    return state_dict


def _normalize_weight_path(p: str) -> str:
    if p.startswith("../input/"):
        return "/kaggle/input/" + p[len("../input/") :]
    if p.startswith("../kaggle/input/"):
        return "/kaggle/input/" + p[len("../kaggle/input/") :]
    return p


def _drop_incompatible_classifier_head(params: dict):
    if not isinstance(params, dict):
        return params
    to_drop = []
    for k, v in params.items():
        if k in ("fc.weight", "fc.bias") and hasattr(v, "shape"):
            if (k == "fc.weight" and v.shape[0] != 5) or (
                k == "fc.bias" and v.shape[0] != 5
            ):
                to_drop.append(k)
    for k in to_drop:
        params.pop(k, None)
    return params


def _candidate_checkpoints(weight_path: str):
    """
    Change (score-relevant): your WEIGHT points to /kaggle/input/cutmix/... which likely doesn't exist
    in this environment, so you end up with fallback predictions. We therefore broaden discovery to
    also search the competition dataset directory and general /kaggle/input, preferring "cassava"/"cutmix"/"resnext".
    """
    wp = _normalize_weight_path(weight_path)
    exts = (".pkl", ".pth", ".pt")

    if os.path.isfile(wp):
        return [wp]
    for ext in exts:
        if os.path.isfile(wp + ext):
            return [wp + ext]

    search_dirs = []

    if os.path.isdir(wp):
        search_dirs.append(wp)

    parent = os.path.dirname(wp) if os.path.dirname(wp) else ""
    if parent and os.path.isdir(parent):
        search_dirs.append(parent)

    if os.path.isdir("/kaggle/input/cutmix"):
        search_dirs.append("/kaggle/input/cutmix")

    comp_dir = "/kaggle/input/cassava-leaf-disease-classification"
    if os.path.isdir(comp_dir):
        search_dirs.append(comp_dir)

    if os.path.isdir("/kaggle/input"):
        search_dirs.append("/kaggle/input")

    candidates = []
    for d in search_dirs:
        for ext in exts:
            candidates += glob.glob(os.path.join(d, f"**/*{ext}"), recursive=True)

    preferred_tokens = ("cassava", "cutmix", "resnext", "kfold", "leaf", "disease")
    filtered = []
    for c in candidates:
        bn = os.path.basename(c).lower()
        if any(t in bn for t in preferred_tokens):
            filtered.append(c)
    if filtered:
        candidates = filtered

    base = os.path.basename(wp).lower()
    if base:
        base_pref = [c for c in candidates if base in os.path.basename(c).lower()]
        if base_pref:
            candidates = base_pref

    candidates = sorted(set(candidates))
    return candidates[:10]


def _load_ckpt_into_model(model, ckpt_path: str):
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")

    if hasattr(ckpt_obj, "state_dict") and not isinstance(ckpt_obj, dict):
        ckpt_obj = ckpt_obj.state_dict()

    params = _strip_known_prefixes(_extract_state_dict(ckpt_obj))
    if not isinstance(params, dict):
        return None
    params = _drop_incompatible_classifier_head(params)
    incompatible = model.load_state_dict(params, strict=False)
    return incompatible


ckpt_candidates = _candidate_checkpoints(WEIGHT)
print(
    f"Found {len(ckpt_candidates)} checkpoint candidate(s) from WEIGHT='{WEIGHT}' -> '{_normalize_weight_path(WEIGHT)}'."
)
if len(ckpt_candidates) > 0:
    print("Candidates:", ckpt_candidates[:10])

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
expected_order = sample_sub["image_id"].tolist()
expected_set = set(expected_order)
name_to_idx = {n: i for i, n in enumerate(expected_order)}

probs_sum = None
probs_count = 0

if len(ckpt_candidates) == 0:
    df = pd.DataFrame(
        {"image_id": expected_order, "label": np.zeros(len(expected_order), dtype=int)}
    )
    df.to_csv("/kaggle/working/submission.csv", index=False)
    print(df.head())
    print("Wrote /kaggle/working/submission.csv (fallback, no checkpoints found)")
else:
    for params_path in ckpt_candidates:
        model = create_new_model()
        incompatible = _load_ckpt_into_model(model, params_path)
        if incompatible is None:
            print(f"Skipped checkpoint (unreadable/invalid): {params_path}")
            continue

        model.eval()
        try:
            missing = len(incompatible.missing_keys)
            unexpected = len(incompatible.unexpected_keys)
        except Exception:
            missing, unexpected = -1, -1
        print(
            f"Using checkpoint: {params_path} (missing_keys={missing}, unexpected_keys={unexpected})"
        )

        probs_epoch = np.zeros((len(expected_order), 5), dtype=np.float64)
        seen = 0

        with torch.no_grad():
            for img, img_name in test_dataloader:
                b_img = img.to(device)
                logits = model(b_img)
                p = torch.softmax(logits, dim=1).detach().cpu().numpy()

                for j, n in enumerate(list(img_name)):
                    if n in expected_set:
                        probs_epoch[name_to_idx[n]] = p[j]
                        seen += 1

        if seen != len(expected_order):
            print(
                f"Warning: checkpoint {params_path} produced probs for {seen}/{len(expected_order)} images; missing filled with zeros."
            )

        probs_sum = probs_epoch if probs_sum is None else (probs_sum + probs_epoch)
        probs_count += 1

    if probs_count == 0:
        df = pd.DataFrame(
            {
                "image_id": expected_order,
                "label": np.zeros(len(expected_order), dtype=int),
            }
        )
        df.to_csv("/kaggle/working/submission.csv", index=False)
        print(df.head())
        print(
            "Wrote /kaggle/working/submission.csv (fallback, all checkpoints invalid/unloadable)"
        )
    else:
        image_labels = probs_sum.argmax(axis=1).astype(int)
        df = pd.DataFrame({"image_id": expected_order, "label": image_labels})
        df.to_csv("/kaggle/working/submission.csv", index=False)
        print(df.head())
        print(
            f"Wrote /kaggle/working/submission.csv (ensembled {probs_count} checkpoint(s))"
        )
