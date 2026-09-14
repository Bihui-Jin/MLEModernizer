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

0.8360531882744031

# 6. Current score

0.75

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I remove the TensorBoard import that crashes in this environment (it isn’t used for inference), fix test image discovery so directories like `test_images/test_images` don’t get treated as images, and make the model checkpoint loading robust by searching for an available `.pth` in the given folder instead of assuming `best_net.pth` exists. I also fix two logic bugs that prevent correct inference: the custom `ResNet`/`MobileNetV2` heads output 10 classes instead of 5, and `Ensemble.forward()` forgets to return the combined logits. Finally, I ensure a valid `submission.csv` is always written with the exact required columns and deterministic ordering matching `sample_submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11024) strongly suggests the model is effectively untrained/random at inference time because no valid pretrained checkpoint is being loaded (your loader falls back to random weights if it can’t find a `.pth`). To move the score toward the target, I make the checkpoint discovery deterministic and comprehensive (search recursively under `save_dir`, prioritize filenames like `best*.pth`, otherwise take the newest), and I fail loudly if nothing is found so you don’t accidentally submit random predictions again. I also ensure the test set ordering exactly matches `sample_submission.csv` (no filtering/reindex drift) while still avoiding directory artifacts, so predictions align to the required rows. These are minimal inference-only changes that preserve your model and transforms, but should drastically increase accuracy if a real checkpoint exists in the provided folder.'
- What this solution (achieved 0.07623) has done: 'I fix the crash by making checkpoint loading robust to the Kaggle environment where `/kaggle/input/pretrained3` contains no `.pth`, by switching to torchvision’s built-in pretrained weights when no local checkpoint is available (this keeps your ResNet inference core logic intact but avoids random/untrained predictions). I keep the existing model choice and transforms, but ensure the classifier head remains 5 classes and that the state-dict load is strict when a compatible checkpoint exists, falling back safely otherwise. This should substantially raise accuracy from ~0.11 toward your target since it eliminates random weights. The script still always write a valid `submission.csv` with the correct columns and ordering matching `sample_submission.csv`.'
- What this solution (achieved 0.56315) has done: 'Your current score is far below the target, which strongly indicates the inference pipeline is using mismatched preprocessing and/or a weak/random checkpoint fallback. I make two minimal inference-only fixes that preserve your model code: (1) use ImageNet normalization for the torchvision-pretrained ResNet fallback (so the fallback is actually meaningful), and (2) improve the fallback by using the full torchvision pretrained ResNet18 when no local `.pth` exists (instead of partially copying weights into a custom ResNet with a different stem), while keeping the 5-class head unchanged. I also add a tiny “train-distribution prior” bias (computed from train.csv) applied only in the no-checkpoint fallback case to nudge predictions toward realistic class frequencies; this is small, deterministic, and should raise accuracy versus near-uniform/random outputs. The script still run end-to-end and always write a valid `submission.csv` matching `sample_submission.csv` order.'
- What this solution (achieved 0.11061) has done: 'Your current score (0.56315) is far below the target (0.83605), and the biggest likely cause is that you are running the torchvision-pretrained fallback with an untrained 5-class head, which cap accuracy. I make the smallest changes that legitimately improve accuracy without changing your core model/training logic: (1) use the correct torchvision inference resize/crop pipeline for ResNet18 (Resize→CenterCrop to 224) when the fallback is used, and (2) replace the untrained 5-class head in fallback mode with a deterministic “nearest ImageNet class-prototype” head computed from the pretrained 1000-way weights (so predictions become meaningfully image-dependent rather than nearly arbitrary). This preserves your inference-only approach and still writes the same `submission.csv` format and ordering, but should move accuracy substantially upward toward the target band. If a real local `.pth` exists, nothing changes (your checkpoint is still preferred).'
- What this solution (achieved 0.11061) has done: 'Your current score is far below the target, and the biggest remaining “minimal-change” likely issue is that you’re still not using a cassava-trained checkpoint at inference (your `save_dir` points to `/kaggle/input/pretrained3`, which often doesn’t exist in this dataset-only environment), so the fallback can’t reach ~0.83. I keep your model/inference core logic intact, but (1) expand checkpoint discovery to also search common Kaggle locations (especially under the competition dataset folder) and (2) ensure we load the checkpoint strictly when architectures match, so we don’t silently run with partially-mismatched weights. If no checkpoint is found anywhere, we still produce a valid submission exactly as before (torchvision fallback), but this change should move you sharply upward toward the target whenever the checkpoint is actually present in the input tree.'
- What this solution (achieved 0.11061) has done: 'Your current score (0.11061) is so far below the target (0.83605) that the most likely cause is still “no real cassava-trained checkpoint is being used at inference”, so you’re effectively submitting a weak fallback. I make checkpoint discovery more precise by (1) requiring the checkpoint filename to match your experiment name when possible and (2) preferring checkpoints that actually contain 5-class classifier weights, which prevents accidentally loading unrelated `.pth` files. If no suitable 5-class cassava checkpoint is found, the code still fall back exactly as before (torchvision pretrained + prototype head) and still write a valid `submission.csv`. These changes preserve your model/inference core logic; they only improve the odds that you load the intended trained weights, which is the minimal path to move accuracy toward the target.'
- What this solution (achieved 0.11061) has done: 'Your current score is far below the target, and the most plausible cause is still that you are not loading a real cassava-trained checkpoint and are instead using the weak fallback; the single most effective minimal change is to actually use the official torchvision ImageNet preprocessing (including correct interpolation and input scaling) via `ResNet18_Weights.DEFAULT.transforms()` when in fallback mode. I keep your model code and inference loop intact, but make the fallback transforms exactly match the pretrained backbone’s expected pipeline, and I also fix a subtle ordering/transform issue by ensuring PIL images are always passed through torchvision transforms (and albumentations only used when requested). These changes should increase accuracy materially (toward your target) without changing architecture, training, or loss. The script still runs end-to-end and writes a valid `submission.csv` in the required format/order.'
- What this solution (achieved 0.11061) has done: 'Your score (0.11061) is so far below the target (0.83605) that the most likely cause is still incorrect preprocessing for the model you’re actually using at inference time (especially when a real cassava checkpoint is found and loaded). I make a minimal, inference-only change: automatically choose the validation transforms based on the loaded model type (custom ResNet expects your 384 CenterCrop pipeline; torchvision-resnet fallback expects the official 224 pipeline), instead of tying the choice to `used_tv_fallback` alone. I also fix a subtle ordering bug in the non-aug torchvision transform branch where `ToTensor()` was applied before `RandomCrop/CenterCrop` (those expect PIL), which can quietly degrade predictions. These keep your architecture and inference loop intact, but should move accuracy upward toward the target by aligning inputs with what the checkpoint/backbone expects, while still always writing a valid `submission.csv`.'
- What this solution (achieved 0.11061) has done: 'Your current score (~0.11) is consistent with inference running in the torchvision-fallback path with a 5-class “prototype head”, which cannot reach the target; the only minimal, legitimate way toward ~0.83 is to actually load a cassava-trained checkpoint. I keep your model/inference loop intact, but (1) fix a critical transform bug in the non-aug torchvision branch (crop must happen on PIL before `ToTensor()`), and (2) make checkpoint discovery explicitly look for common Kaggle notebook output locations (e.g., `/kaggle/working/**`) so a real trained `.pth` can be found and loaded if it exists in this session. I also add a small, safe compatibility shim to accept checkpoints saved as full models (`torch.save(model)`) as well as state_dicts, without changing architecture or outputs when a proper state_dict exists. This should move the score sharply upward if a real cassava checkpoint is present; otherwise, it at least prevent the transform bug from further degrading the fallback.'
- What this solution (achieved 0.11061) has done: 'Your current score (~0.11) indicates the inference path is still effectively “non-cassava-trained,” so the most direct minimal improvement is to correctly locate and load a real 5-class cassava checkpoint if it exists in the Kaggle filesystem. I (1) expand checkpoint discovery to include `.pt`/`.ckpt` files (common in Lightning) and not just `.pth`, (2) make the “5-class head” compatibility check robust to different key names (e.g., `head.weight`, `classifier.weight`, `fc.*`), and (3) actually use that compatibility check to choose the best checkpoint instead of often returning the first “best.pth” it sees. These are inference-only changes (no architecture/training loop changes) and should move accuracy sharply upward toward the target whenever a proper checkpoint is present; otherwise behavior stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.75) has done: 'I fix the validation/test transform bug by ensuring albumentations transforms are always called with named arguments (`image=image`) and that the correct `aug` flag is used for each dataset split, which removes the DataLoader KeyError. I also make checkpoint usage robust by falling back to the in-memory trained model if the on-disk best checkpoint was not written (e.g., if training crashed before saving), which prevents the FileNotFoundError and guarantees end-to-end execution. These changes preserve your model/training loop/core logic while ensuring a valid `submission.csv` is always produced in the required order/columns. Finally, I keep determinism and paths intact and avoid any architecture/training refactors.'

# 9. Code solution

## === cell 0
import sys

root = "/kaggle/"
sys.path.append(root)

import os
import glob
import re
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

import time
import copy
import random



## === cell 1
""" 
Dataset Class
"""


class CSVDataset(Dataset):
    def __init__(
        self,
        annotations_df,
        img_dir,
        transform=None,
        target_transform=None,
        aug=True,
        return_label=False,
    ):
        self.img_labels = annotations_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.aug = aug
        self.return_label = return_label

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            if isinstance(self.transform, A.core.composition.BaseCompose):
                image = np.array(image)
                image = self.transform(image=image)["image"]
            else:
                image = self.transform(image)

        sample = {"image": image}
        if self.return_label:
            y = int(self.img_labels.iloc[idx, 1])
            sample["label"] = y
        return sample




## === cell 2
"""
Resnet / MobileNet / ViT definitions
"""


class ResNet(nn.Module):
    def __init__(self, layers, dropout=0.0, num_classes=5):
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
        self.fc = nn.Linear(512, num_classes)
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
    def __init__(self, width_mult=1.0, dropout=0.0, num_classes=5):
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
            nn.Dropout(dropout), nn.Linear(last_channel, num_classes)
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


def _ensure_einops():
    try:
        from einops import rearrange, repeat
        from einops.layers.torch import Rearrange

        return rearrange, repeat, Rearrange
    except Exception as e:
        raise ImportError(
            "ViT requires 'einops' which is not available in this environment."
        ) from e


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.rearrange, _, _ = _ensure_einops()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        rearrange = self.rearrange
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = self.attend(dots)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class VIT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        _, repeat, Rearrange = _ensure_einops()
        self.repeat = repeat

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        repeat = self.repeat
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


class Ensemble(nn.Module):
    def __init__(self, modelA, modelB):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        outA = self.modelA(x)
        outB = self.modelB(x)
        out = outA + outB
        return out




## === cell 3
"""
Auxiliary Functions
"""


def get_model(model, width_mult=1.0, dropout=0.2):
    if model == "base":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 5)
        return m
    elif model == "resnet":
        return ResNet([2, 2, 2, 2], dropout, num_classes=5)
    elif model == "mobilenet":
        return MobileNetV2(width_mult=width_mult, dropout=dropout, num_classes=5)
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


def get_transforms(aug=True, p=0.3, use_imagenet_norm=False, tv_resnet_eval_224=False):
    if use_imagenet_norm:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

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
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
        val_transforms = A.Compose(
            [
                A.CenterCrop(288, 288),
                A.Resize(384, 384),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
        return train_transforms, val_transforms

    if tv_resnet_eval_224:
        preset = models.ResNet18_Weights.DEFAULT.transforms()
        return preset, preset

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.CenterCrop((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transforms, val_transforms


def save_model_state_dict(net, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(net.state_dict(), path)


def _extract_state_dict(maybe_state):
    if isinstance(maybe_state, nn.Module):
        return maybe_state.state_dict()
    if (
        isinstance(maybe_state, dict)
        and "state_dict" in maybe_state
        and isinstance(maybe_state["state_dict"], dict)
    ):
        return maybe_state["state_dict"]
    if isinstance(maybe_state, dict):
        return maybe_state
    return None


def _normalize_state_keys(state):
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        if nk.startswith("net."):
            nk = nk[len("net.") :]
        new_state[nk] = v
    return new_state


def _try_load_state_dict(net, state, device, strict_prefer=True):
    state = _extract_state_dict(state)
    if state is None:
        raise ValueError("Checkpoint does not contain a state_dict-like mapping.")

    state = _normalize_state_keys(state)

    if strict_prefer:
        try:
            net.load_state_dict(state, strict=True)
            print("[INFO] Loaded checkpoint with strict=True")
            return net
        except Exception as e:
            print(
                f"[WARN] strict=True load failed ({type(e).__name__}); retrying strict=False"
            )

    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[WARN] Loaded with strict=False. missing={len(missing)} unexpected={len(unexpected)}"
        )
    return net


def _list_images_only(img_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = []
    for fn in os.listdir(img_dir):
        p = os.path.join(img_dir, fn)
        if os.path.isfile(p) and os.path.splitext(fn.lower())[1] in exts:
            files.append(fn)
    return sorted(files)




## === cell 4
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def accuracy_from_logits(logits, y):
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()


seed_everything(42)

args = {}
args["name"] = "mobilenet_384_randomcrop_width_mult_1.8"  # experiment name (kept)
args["batch_size"] = 32
args["width_mult"] = 1.0
args["dropout"] = 0.0
args["aug"] = True
args["model"] = "resnet"
args["gpu_id"] = 0
args["epochs"] = 4
args["lr"] = 2e-4

data_dir = os.path.join(root, "input/cassava-leaf-disease-classification/")
save_dir = os.path.join(root, "working", "pretrained3")
os.makedirs(save_dir, exist_ok=True)

train_csv_path = os.path.join(data_dir, "train.csv")
train_img_dir = os.path.join(data_dir, "train_images")
test_img_dir = os.path.join(data_dir, "test_images")

sample_path = os.path.join(data_dir, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)

device = "cuda:" + str(args["gpu_id"]) if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

train_df = pd.read_csv(train_csv_path)
perm = np.random.RandomState(42).permutation(len(train_df))
val_size = max(1, int(0.05 * len(train_df)))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]
trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

train_tfms, val_tfms = get_transforms(
    aug=args["aug"], use_imagenet_norm=False, tv_resnet_eval_224=False
)

train_ds = CSVDataset(
    trn_df, train_img_dir, transform=train_tfms, aug=True, return_label=True
)
val_ds = CSVDataset(
    val_df, train_img_dir, transform=val_tfms, aug=False, return_label=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=args["batch_size"],
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_ds,
    batch_size=args["batch_size"],
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

net = get_model(
    args["model"], width_mult=args["width_mult"], dropout=args["dropout"]
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=args["lr"])

best_val_acc = -1.0
best_path = os.path.join(save_dir, "best_net.pth")
best_state = (
    None  # Bugfix: ensure we can still run inference even if file wasn't written.
)

t0 = time.time()
for epoch in range(1, args["epochs"] + 1):
    net.train()
    tr_loss = 0.0
    tr_acc = 0.0
    ntr = 0

    for batch in train_loader:
        x = batch["image"].float().to(device, non_blocking=True)
        y = torch.tensor(batch["label"], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = net(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        tr_loss += loss.item() * bs
        tr_acc += accuracy_from_logits(logits.detach(), y) * bs
        ntr += bs

    net.eval()
    va_loss = 0.0
    va_acc = 0.0
    nva = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].float().to(device, non_blocking=True)
            y = torch.tensor(batch["label"], dtype=torch.long, device=device)
            logits = net(x)
            loss = criterion(logits, y)

            bs = x.size(0)
            va_loss += loss.item() * bs
            va_acc += accuracy_from_logits(logits, y) * bs
            nva += bs

    tr_loss /= max(1, ntr)
    tr_acc /= max(1, ntr)
    va_loss /= max(1, nva)
    va_acc /= max(1, nva)

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_state = copy.deepcopy(net.state_dict())
        save_model_state_dict(net, best_path)

    print(
        f"epoch {epoch:02d}/{args['epochs']}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  val_loss={va_loss:.4f} val_acc={va_acc:.4f}  best_val_acc={best_val_acc:.4f}"
    )

print(f"[INFO] Training done in {time.time()-t0:.1f}s. Best checkpoint: {best_path}")



## === cell 5
available = set(_list_images_only(test_img_dir))
missing = [x for x in sample_sub["image_id"].tolist() if x not in available]
if missing:
    raise FileNotFoundError(
        f"{len(missing)} image_ids from sample_submission.csv are missing in {test_img_dir}. "
        f"First missing: {missing[0]}"
    )

test_pd = sample_sub[["image_id"]].copy()
num_test = len(test_pd)
print(f"test images: {num_test}")

net = get_model(
    args["model"], width_mult=args["width_mult"], dropout=args["dropout"]
).to(device)

used_checkpoint = None
if os.path.isfile(best_path):
    state = torch.load(best_path, map_location=device)
    net = _try_load_state_dict(net, state, device, strict_prefer=True)
    used_checkpoint = best_path
elif best_state is not None:
    net.load_state_dict(best_state, strict=True)
    used_checkpoint = "(in-memory best_state)"
else:
    used_checkpoint = "(random init)"

net.eval()

_, test_transforms = get_transforms(
    aug=False, use_imagenet_norm=False, tv_resnet_eval_224=False
)

test_dataset = CSVDataset(
    test_pd, test_img_dir, transform=test_transforms, aug=False, return_label=False
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
        pred_list.extend(outputs.argmax(dim=1).tolist())

if len(pred_list) != len(test_pd):
    raise RuntimeError(
        f"Pred length mismatch: got {len(pred_list)} preds for {len(test_pd)} rows"
    )

sub = test_pd.copy()
sub["label"] = np.array(pred_list, dtype=np.int64)
sub = sub[["image_id", "label"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print(f"[INFO] Wrote submission.csv with {len(sub)} rows")
print(f"[INFO] Used checkpoint: {used_checkpoint}")
