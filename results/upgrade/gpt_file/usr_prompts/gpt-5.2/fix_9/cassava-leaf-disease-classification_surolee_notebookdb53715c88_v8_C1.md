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

albumentations==2.0.8
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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

# 5. Target score

0.8582653369598066

# 6. Current score

0.55643

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.56988) has done: 'I remove the TensorBoard import that crashes in this environment, since it isn’t used for inference and blocks execution. I also fix test image listing so it filters out the nested `test_images/` directory that currently causes `IsADirectoryError`. Next, I make the checkpoint loading robust: if `/kaggle/input/pretrained1/best_net.pth` is missing, the code fall back to a standard torchvision MobileNetV2 classifier adjusted to 5 classes so it still produces a valid submission. Finally, I correct the model output classes to 5 (the competition has 5 labels) to avoid invalid argmax predictions and improve accuracy versus the current 10-class heads.'
- What this solution (achieved 0.32885) has done: 'Your current score is far below the target, and the main reason is that the fallback model (ImageNet-pretrained MobileNetV2 with a randomly initialized 5-class head) produces near-random predictions without any training. To move toward the target with minimal semantic change, I keep your exact inference pipeline but (1) switch the fallback to a stronger ImageNet-pretrained backbone (ResNet18) while still only replacing the final 5-class layer, and (2) fix the “aug=False” torchvision transform branch which currently applies `ToTensor()` before `RandomCrop`, which is invalid and can hurt/interrupt proper preprocessing. These changes preserve your overall approach (single-model inference, argmax labels, same submission writing) while improving the quality of predictions when the custom checkpoint is missing. The code still writes a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.57063) has done: 'Your current score is far below the target, and the main cause is that when the competition checkpoint is missing the code falls back to an ImageNet-pretrained model with a randomly initialized 5-class head, which yields near-random accuracy. To move toward the target with minimal semantic change, I keep your exact single-model argmax inference pipeline, but make the fallback head a deterministic “nearest class prototype” classifier computed once from the provided `train_images/train.csv` using the frozen ImageNet backbone features (no training loop, no loss change). This uses the dataset’s actual label structure to produce meaningful predictions while preserving the same architecture backbone and overall approach. I also fix the non-aug torchvision transform order (crop should happen before `ToTensor`) and keep submission ordering aligned to `sample_submission.csv`.'
- What this solution (achieved 0.48729) has done: 'To move your score upward toward the 0.858 target without changing the overall inference approach, I fix a key mismatch in the fallback “prototype head” path: it currently builds prototypes using *test* preprocessing (center crop) rather than the *train* preprocessing, which makes the class prototypes less representative and hurts accuracy. I also make the prototype feature computation align better with ResNet18’s own expected normalization (ImageNet mean/std) while keeping your existing cassava pipeline intact for inference, by using a dedicated ImageNet-normalized transform only for building the prototypes (not for test predictions). Finally, I correct the feature dimension handling so the prototype head matches whatever backbone feature size is used (avoids silent shape assumptions), improving stability and likely accuracy.'
- What this solution (achieved 0.5852) has done: 'Your score gap to the target is large (0.487 → 0.858), and the biggest limiter is the fallback path: it builds class prototypes with a random crop + different normalization than the test pipeline, so the nearest-prototype head is poorly aligned to test-time features. I keep your exact single-model argmax inference approach, but make the fallback prototype-building transform match the test-time spatial preprocessing (center crop/resize) and use ImageNet normalization consistently for both prototype building and test inference when using the ImageNet backbone. This is a minimal change that directly improves the quality/consistency of the fallback predictions without adding training loops or changing the model family. I also ensure test image ordering matches `sample_submission.csv` first (when available) to avoid any accidental mismatch.'
- What this solution (achieved 0.58296) has done: 'Your current score (0.5852) is far below the target (0.8583), and the biggest limiter is the fallback “prototype head” quality: it uses raw (unnormalized) backbone features and an L2-style bias without feature normalization, which tends to be poorly calibrated for cosine-like similarity. I keep the exact fallback approach (frozen ImageNet backbone + prototypes from train.csv, no training loop) but make the prototype classifier use L2-normalized features/prototypes (cosine similarity) which is a minimal, standard fix that typically boosts nearest-prototype accuracy. I also keep preprocessing consistent and increase `max_per_class` modestly (still bounded) to reduce prototype noise without changing the core method. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.55194) has done: 'Your current score (0.583) is far below the target (0.858), so we should carefully improve the fallback path (used when the checkpoint is missing) without changing the overall approach (single-pass inference, argmax labels, no training loop). The biggest low-risk gain here is improving the prototype quality: instead of one mean prototype per class, compute multiple prototypes per class via a tiny k-means on frozen features (still no training, no loss changes), then classify by maximum cosine similarity to any prototype. This keeps the same backbone/features and “prototype head” idea, but better captures intra-class variation and typically boosts accuracy substantially. I also make the feature extractor explicitly use `tv_r18.fc = nn.Identity()` to avoid any ambiguity and keep transforms consistent (center-crop + ImageNet normalization) for both prototype building and test inference.'
- What this solution (achieved 0.55643) has done: 'Your current score (0.55194) is far below the target (0.8583), so we should improve the fallback (checkpoint-missing) path while keeping your exact overall approach: frozen ImageNet backbone → prototype head → argmax labels → submission. The biggest low-risk issue is that prototypes are currently built from a globally `head(max_per_class)`-truncated dataframe, which can unintentionally bias which samples get selected per class (and makes results sensitive to CSV ordering), weakening prototypes. I change prototype sampling to be deterministic but more representative by taking a stratified evenly-spaced selection across each class (still capped by `max_per_class`, no extra training), and I slightly strengthen the multi-prototype resolution by increasing `k_per_class` to 8 (same k-means logic, just a small parameter tweak). These changes directly improve prototype quality/coverage and should move accuracy upward toward the target without changing model family, inference semantics, or adding any training loop.'

# 9. Code solution

## === cell 0
import sys

root = "/kaggle/"
sys.path.append(root)

"""
Import Libraries
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

import matplotlib.pyplot as plt

import time
import copy



## === cell 1
""" 
Dataset Class
"""


class CSVDataset(Dataset):
    def __init__(
        self, annotations_df, img_dir, transform=None, target_transform=None, aug=True
    ):
        self.img_labels = annotations_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.aug = aug

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            if self.aug:
                image = np.array(image)
                image = self.transform(image=image)
                image = image["image"]
            else:
                image = self.transform(image)

        sample = {"image": image}
        return sample




## === cell 2
"""
Resnet / MobileNetV2 / VIT
"""


class ResNet(nn.Module):
    def __init__(self, layers, dropout=0.0):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, padding=3, stride=2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(64, layers[0])
        self.layer2 = self.make_layer(128, layers[1], stride=2)
        self.layer3 = self.make_layer(256, layers[2], stride=2)
        self.layer4 = self.make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 5)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(ResBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

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


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0, dropout=0.0):
        super(MobileNetV2, self).__init__()
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = 32
        last_channel = 1280

        input_channel = _make_divisible(input_channel * width_mult, 8)
        last_channel = _make_divisible(
            last_channel * max(1.0, width_mult) * width_mult, 8
        )
        features = [ConvBNReLU(3, input_channel, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(
                        input_channel, output_channel, stride, expand_ratio=t
                    )
                )
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_channel, 5),
        )

    def forward(self, x):
        out = self.features(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(round(in_planes * expand_ratio))
        self.use_res_connect = stride == 1 and in_planes == out_planes

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_planes, hidden_dim, kernel_size=1))
        layers.extend(
            [
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                nn.Conv2d(hidden_dim, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v




## === cell 3
"""
Auxiliary Functions
"""


def get_model(model, width_mult=1.0, dropout=0.2):
    if model == "base":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 5)
        return m
    elif model == "resnet":
        return ResNet([2, 2, 2, 2], dropout)
    elif model == "mobilenet":
        return MobileNetV2(width_mult=width_mult, dropout=dropout)
    elif model == "VIT":
        return VIT(
            image_size=(384, 384),
            patch_size=16,
            num_classes=5,
            dim=512,
            depth=6,
            heads=12,
            mlp_dim=1024,
        )
    else:
        raise NotImplementedError(f"Model [{model}] not implemented")


def get_transforms(aug=True, p=0.3):
    if aug:
        train_transforms = A.Compose(
            [
                A.RandomCrop(288, 288),
                A.Resize(384, 384),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=p
                ),
                A.RandomBrightnessContrast(p=p),
                A.HorizontalFlip(p=p),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )
        val_transforms = A.Compose(
            [
                A.CenterCrop(288, 288),
                A.Resize(384, 384),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )
    else:
        train_transforms = transforms.Compose(
            [
                transforms.RandomCrop((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        val_transforms = transforms.Compose(
            [
                transforms.CenterCrop((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    return train_transforms, val_transforms


def save_model(net, name, epoch, save_dir):
    path = os.path.join(save_dir, f"{epoch}_net.pth")
    torch.save(net.state_dict(), path)


def load_model(net, name, epoch, save_dir, device):
    path = os.path.join(save_dir, f"{epoch}_net.pth")
    state = torch.load(path, map_location=device)
    net.load_state_dict(state, strict=False)
    return net


def print_and_save_args(args, path):
    message = ""
    for k, v in args.items():
        message += f"{str(k):>15}: {str(v):<10}\n"
    print(" " * 20 + "[OPTIONS]" + " " * 20)
    print(message)
    with open(path, "w") as f:
        f.write(message)


class CosinePrototypeHead(nn.Module):
    """
    Keep the same "prototype from train" fallback, but improve it (still no training loop):
    - store L2-normalized prototypes
    - classify by max cosine similarity across multiple prototypes per class
    """

    def __init__(
        self, prototypes: torch.Tensor, proto_labels: torch.Tensor, scale: float = 10.0
    ):
        super().__init__()
        protos = F.normalize(prototypes, p=2, dim=1)
        self.register_buffer("protos", protos)  # [K, D]
        self.register_buffer("proto_labels", proto_labels.long())  # [K]
        self.scale = float(scale)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)  # [B, D]
        sims = self.scale * (x @ self.protos.t())  # [B, K]
        B = sims.shape[0]
        out = torch.full((B, 5), -1e9, device=sims.device, dtype=sims.dtype)
        for c in range(5):
            m = self.proto_labels == c
            if m.any():
                out[:, c] = sims[:, m].max(dim=1).values
        return out


@torch.no_grad()
def _kmeans_torch(
    x: torch.Tensor, k: int, iters: int = 15, seed: int = 0
) -> torch.Tensor:
    """
    Tiny k-means used only to refine prototypes from frozen features (no backprop).
    x: [N, D] float tensor (on GPU/CPU)
    returns centroids [k, D]
    """
    n = x.shape[0]
    if n <= k:
        reps = (k + n - 1) // n
        x2 = x.repeat(reps, 1)[:k]
        return x2

    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g, device=x.device)[:k]
    c = x[idx].clone()  # [k, D]

    for _ in range(iters):
        xn = F.normalize(x, p=2, dim=1)
        cn = F.normalize(c, p=2, dim=1)
        sim = xn @ cn.t()  # [N, k]
        labels = sim.argmax(dim=1)  # [N]
        for j in range(k):
            m = labels == j
            if m.any():
                c[j] = x[m].mean(dim=0)
    return c


@torch.no_grad()
def build_prototypes_from_train(
    backbone,
    device,
    data_dir,
    transform,
    batch_size=64,
    max_per_class=600,
    k_per_class=5,
):
    train_csv = os.path.join(data_dir, "train.csv")
    train_img_dir = os.path.join(data_dir, "train_images")
    df = pd.read_csv(train_csv)

    dfs = []
    for c in range(5):
        d = df[df["label"] == c].sort_values("image_id").reset_index(drop=True)
        n = len(d)
        take = min(max_per_class, n)
        if take <= 0:
            continue
        if take == n:
            d_sel = d
        else:
            idx = np.linspace(0, n - 1, num=take, dtype=int)
            d_sel = d.iloc[idx]
        dfs.append(d_sel)

    df_small = pd.concat(dfs, axis=0).reset_index(drop=True)

    ds = CSVDataset(
        df_small[["image_id"]], train_img_dir, transform=transform, aug=False
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    backbone = backbone.to(device).eval()
    feats_all = []
    for batch in dl:
        x = batch["image"].float().to(device, non_blocking=True)
        feats = backbone(x)  # [B, D] because fc is Identity
        feats_all.append(feats.detach().cpu())
    feats_all = torch.cat(feats_all, dim=0)  # [N, D] on CPU

    y_all = torch.as_tensor(df_small["label"].values, dtype=torch.long)

    protos = []
    proto_labels = []

    for c in range(5):
        xc = feats_all[y_all == c]  # [Nc, D]
        xc = xc.to(device)
        centers = (
            _kmeans_torch(xc, k=k_per_class, iters=15, seed=123 + c).detach().cpu()
        )
        protos.append(centers)
        proto_labels.append(torch.full((centers.shape[0],), c, dtype=torch.long))

    protos = torch.cat(protos, dim=0)  # [K, D]
    proto_labels = torch.cat(proto_labels, dim=0)  # [K]
    head = CosinePrototypeHead(
        protos.to(device), proto_labels.to(device), scale=10.0
    ).to(device)
    return head




## === cell 4
args = {}
args["name"] = "mobilenet_384_randomcrop_width_mult_1.8"
args["batch_size"] = 32
args["width_mult"] = 1.8
args["dropout"] = 0.0
args["aug"] = False
args["model"] = "mobilenet"
args["gpu_id"] = 0

assert args["name"] is not None, "Must set experiment name before training"

data_dir = os.path.join(root, "input/cassava-leaf-disease-classification/")
save_dir = os.path.join(root, "input/pretrained1")

img_dir = os.path.join(data_dir, "test_images")

sample_path = os.path.join(data_dir, "sample_submission.csv")
if os.path.exists(sample_path):
    sample_sub = pd.read_csv(sample_path)
    files = sample_sub["image_id"].tolist()
else:
    files = []
    for f in os.listdir(img_dir):
        fp = os.path.join(img_dir, f)
        if os.path.isfile(fp) and f.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(f)
    files = sorted(files)

test_pd = pd.DataFrame({"image_id": files})
num_test = len(test_pd)

_, test_transforms = get_transforms(args["aug"])

test_dataset = CSVDataset(test_pd, img_dir, transform=test_transforms, aug=args["aug"])
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args["batch_size"],
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

device = "cuda:" + str(args["gpu_id"]) if torch.cuda.is_available() else "cpu"
print(f"test images: {num_test} \t device: {device}")

net = get_model(
    args["model"], width_mult=args["width_mult"], dropout=args["dropout"]
).to(device)

ckpt_path = os.path.join(save_dir, "best_net.pth")
using_fallback_imagenet = False

if os.path.exists(ckpt_path):
    net = load_model(net, args["name"], "best", save_dir, device)
    print(f"Loaded checkpoint: {ckpt_path}")
else:
    using_fallback_imagenet = True
    print(
        f"Checkpoint not found at {ckpt_path}. Falling back to torchvision resnet18 pretrained on ImageNet "
        f"with a multi-prototype cosine head built from train.csv (frozen features; no training loop)."
    )

    tv_r18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    tv_r18.fc = nn.Identity()
    tv_r18.eval()

    proto_transform = transforms.Compose(
        [
            transforms.CenterCrop((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    proto_head = build_prototypes_from_train(
        backbone=tv_r18,
        device=device,
        data_dir=data_dir,
        transform=proto_transform,
        batch_size=64,
        max_per_class=600,
        k_per_class=8,
    )

    net = nn.Sequential(tv_r18, proto_head).to(device)

net.eval()



## === cell 5
"""
Test / Submission
"""
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)


def human_format(num):
    magnitude = 0
    num = float(num)
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "%.2f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


print(f"Number of total parameters: {human_format(num_params)}")

if using_fallback_imagenet:
    imagenet_test_transform = transforms.Compose(
        [
            transforms.CenterCrop((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    test_dataset = CSVDataset(
        test_pd, img_dir, transform=imagenet_test_transform, aug=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

pred_list = []
with torch.no_grad():
    for data in test_dataloader:
        imgs = data["image"].float().to(device, non_blocking=True)
        outputs = net(imgs)
        pred_list.extend(outputs.argmax(dim=1).detach().cpu().tolist())

test_pd["label"] = np.array(pred_list, dtype=np.int64)

if os.path.exists(sample_path):
    sample_sub = pd.read_csv(sample_path)
    sub = sample_sub[["image_id"]].merge(
        test_pd[["image_id", "label"]], on="image_id", how="left"
    )
    sub["label"] = sub["label"].fillna(0).astype(int)
else:
    sub = test_pd[["image_id", "label"]].copy()

print(sub.head())
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
