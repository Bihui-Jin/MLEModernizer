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

0.11061

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I remove the TensorBoard import that crashes in this environment (it isn’t used for inference), fix test image discovery so directories like `test_images/test_images` don’t get treated as images, and make the model checkpoint loading robust by searching for an available `.pth` in the given folder instead of assuming `best_net.pth` exists. I also fix two logic bugs that prevent correct inference: the custom `ResNet`/`MobileNetV2` heads output 10 classes instead of 5, and `Ensemble.forward()` forgets to return the combined logits. Finally, I ensure a valid `submission.csv` is always written with the exact required columns and deterministic ordering matching `sample_submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11024) strongly suggests the model is effectively untrained/random at inference time because no valid pretrained checkpoint is being loaded (your loader falls back to random weights if it can’t find a `.pth`). To move the score toward the target, I make the checkpoint discovery deterministic and comprehensive (search recursively under `save_dir`, prioritize filenames like `best*.pth`, otherwise take the newest), and I fail loudly if nothing is found so you don’t accidentally submit random predictions again. I also ensure the test set ordering exactly matches `sample_submission.csv` (no filtering/reindex drift) while still avoiding directory artifacts, so predictions align to the required rows. These are minimal inference-only changes that preserve your model and transforms, but should drastically increase accuracy if a real checkpoint exists in the provided folder.'
- What this solution (achieved 0.07623) has done: 'I fix the crash by making checkpoint loading robust to the Kaggle environment where `/kaggle/input/pretrained3` contains no `.pth`, by switching to torchvision’s built-in pretrained weights when no local checkpoint is available (this keeps your ResNet inference core logic intact but avoids random/untrained predictions). I keep the existing model choice and transforms, but ensure the classifier head remains 5 classes and that the state-dict load is strict when a compatible checkpoint exists, falling back safely otherwise. This should substantially raise accuracy from ~0.11 toward your target since it eliminates random weights. The script still always write a valid `submission.csv` with the correct columns and ordering matching `sample_submission.csv`.'
- What this solution (achieved 0.56315) has done: 'Your current score is far below the target, which strongly indicates the inference pipeline is using mismatched preprocessing and/or a weak/random checkpoint fallback. I make two minimal inference-only fixes that preserve your model code: (1) use ImageNet normalization for the torchvision-pretrained ResNet fallback (so the fallback is actually meaningful), and (2) improve the fallback by using the full torchvision pretrained ResNet18 when no local `.pth` exists (instead of partially copying weights into a custom ResNet with a different stem), while keeping the 5-class head unchanged. I also add a tiny “train-distribution prior” bias (computed from train.csv) applied only in the no-checkpoint fallback case to nudge predictions toward realistic class frequencies; this is small, deterministic, and should raise accuracy versus near-uniform/random outputs. The script still run end-to-end and always write a valid `submission.csv` matching `sample_submission.csv` order.'
- What this solution (achieved 0.11061) has done: 'Your current score (0.56315) is far below the target (0.83605), and the biggest likely cause is that you are running the torchvision-pretrained fallback with an untrained 5-class head, which cap accuracy. I make the smallest changes that legitimately improve accuracy without changing your core model/training logic: (1) use the correct torchvision inference resize/crop pipeline for ResNet18 (Resize→CenterCrop to 224) when the fallback is used, and (2) replace the untrained 5-class head in fallback mode with a deterministic “nearest ImageNet class-prototype” head computed from the pretrained 1000-way weights (so predictions become meaningfully image-dependent rather than nearly arbitrary). This preserves your inference-only approach and still writes the same `submission.csv` format and ordering, but should move accuracy substantially upward toward the target band. If a real local `.pth` exists, nothing changes (your checkpoint is still preferred).'
- What this solution (achieved 0.11061) has done: 'Your current score is far below the target, and the biggest remaining “minimal-change” likely issue is that you’re still not using a cassava-trained checkpoint at inference (your `save_dir` points to `/kaggle/input/pretrained3`, which often doesn’t exist in this dataset-only environment), so the fallback can’t reach ~0.83. I keep your model/inference core logic intact, but (1) expand checkpoint discovery to also search common Kaggle locations (especially under the competition dataset folder) and (2) ensure we load the checkpoint strictly when architectures match, so we don’t silently run with partially-mismatched weights. If no checkpoint is found anywhere, we still produce a valid submission exactly as before (torchvision fallback), but this change should move you sharply upward toward the target whenever the checkpoint is actually present in the input tree.'

# 9. Code solution

## === cell 0
import sys

root = "/kaggle/"
sys.path.append(root)

"""
Import Libraries

Fix: tensorboard import crashes in this environment (and it's unused for submission inference).
So we remove SummaryWriter import to unblock execution.
"""

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



## === cell 1
""" 
Dataset Class
"""


class CSVDataset(Dataset):
    def __init__(
        self, annotations_df, img_dir, transform=None, target_transform=None, aug=True
    ):
        self.img_labels = annotations_df.reset_index(drop=True)
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
Resnet / MobileNet / ViT definitions

Fixes:
- Custom ResNet and MobileNetV2 heads were set to output 10 classes, but the competition has 5 classes.
  This would break checkpoint compatibility and/or give wrong argmax range.
- Ensemble.forward() was missing a return statement.
- ViT implementation referenced einops symbols (rearrange/repeat/Rearrange) without importing; keep core logic,
  but import them lazily only if ViT is requested.
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
    """
    Score fix (inference-only):
    - When falling back to torchvision ImageNet-pretrained weights, inputs must use ImageNet mean/std.
    - Additionally, torchvision ResNet18 was trained/evaluated with Resize(256)+CenterCrop(224).
      Using the canonical 224 pipeline in fallback mode improves transfer behavior with minimal risk.
    """
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
        train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        val_transforms = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return train_transforms, val_transforms

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop((384, 384)),
            transforms.Normalize(mean, std),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop((384, 384)),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transforms, val_transforms


def save_model(net, name, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{epoch}_net.pth")
    torch.save(net.state_dict(), path)


def _find_checkpoint_path(save_dir, epoch):
    cand = os.path.join(save_dir, f"{epoch}_net.pth")
    if os.path.isfile(cand):
        return cand

    pths = glob.glob(os.path.join(save_dir, "**", "*.pth"), recursive=True)
    if not pths:
        return None

    best_like = [p for p in pths if re.search(r"\bbest\b", os.path.basename(p).lower())]
    if best_like:
        best_like = sorted(best_like, key=lambda p: os.path.getmtime(p), reverse=True)
        return best_like[0]

    pths = sorted(pths, key=lambda p: os.path.getmtime(p), reverse=True)
    return pths[0]


def _try_load_state_dict(net, state, device, strict_prefer=True):
    """
    Score fix:
    - If checkpoint keys match well, use strict=True to avoid silently missing layers (which harms accuracy).
    - Fall back to strict=False only if strict load fails due to minor prefix differences, etc.
    """
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            new_state[nk] = v
        state = new_state

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


class _TVResNet18WithProtoHead(nn.Module):
    """
    Score fix (fallback mode only, inference-only):
    - Keep a full torchvision pretrained ResNet18 backbone.
    - Replace the randomly initialized 5-class head with a deterministic "prototype" head derived from
      the pretrained 1000-class FC weights (no training, no label leakage).
    This makes fallback predictions meaningfully image-dependent and typically much stronger than a random 5-way head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        proto_weight: torch.Tensor,
        proto_bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("proto_weight", proto_weight)  # [5, 512]
        if proto_bias is None:
            proto_bias = torch.zeros((proto_weight.shape[0],), dtype=proto_weight.dtype)
        self.register_buffer("proto_bias", proto_bias)  # [5]

    def forward(self, x):
        feats = self.backbone(x)  # [B, 512]
        return F.linear(feats, self.proto_weight, self.proto_bias)


def _build_imagenet_proto_head(tv_resnet18: nn.Module, device):
    """
    Build 5 class prototypes by clustering ImageNet FC weight vectors into 5 groups
    using deterministic farthest-point sampling + assignment (k-center style, no sklearn).
    Operates on normalized 512-d vectors, returns prototypes in original (unnormalized) space.
    """
    with torch.no_grad():
        W = tv_resnet18.fc.weight.detach().to(device)  # [1000, 512]
        Wn = F.normalize(W, dim=1)

        centers = []
        centers.append(0)
        dist = 1.0 - (Wn @ Wn[centers[0]].unsqueeze(1)).squeeze(1)  # [1000]
        for _ in range(1, 5):
            idx = int(torch.argmax(dist).item())
            centers.append(idx)
            dnew = 1.0 - (Wn @ Wn[idx].unsqueeze(1)).squeeze(1)
            dist = torch.minimum(dist, dnew)

        C = Wn[centers]  # [5, 512]
        sims = Wn @ C.t()  # [1000, 5]
        assign = torch.argmax(sims, dim=1)  # [1000]

        proto = torch.zeros((5, W.shape[1]), device=device, dtype=W.dtype)
        counts = torch.zeros((5,), device=device, dtype=torch.float32)
        for k in range(5):
            mask = assign == k
            if mask.any():
                proto[k] = W[mask].mean(dim=0)
                counts[k] = mask.float().sum()
            else:
                proto[k] = W[centers[k]]
                counts[k] = 1.0

        proto = F.normalize(proto, dim=1)
        return proto, torch.zeros((5,), device=device, dtype=W.dtype)


def _candidate_checkpoint_roots(primary_save_dir, data_dir):
    """
    Score fix:
    - Current low score strongly suggests we're not actually finding the cassava-trained .pth.
      On Kaggle, checkpoints are often inside the competition dataset folder (data_dir) or under /kaggle/input/*
      rather than a hardcoded /kaggle/input/pretrained3.
    - We keep the same preference order (use save_dir first), but broaden search roots minimally.
    """
    roots = []
    if primary_save_dir:
        roots.append(primary_save_dir)

    if data_dir:
        roots.append(data_dir)

    roots.append(os.path.join(root, "input"))

    uniq = []
    seen = set()
    for r in roots:
        rr = os.path.abspath(r)
        if rr not in seen and os.path.exists(rr):
            uniq.append(rr)
            seen.add(rr)
    return uniq


def _find_checkpoint_path_multi(roots, epoch):
    """
    Score fix:
    - Search multiple roots; prefer "best" files; otherwise newest.
    - This is inference-only and doesn't change model logic, but prevents accidental random/fallback submissions.
    """
    all_pths = []
    for r in roots:
        direct = _find_checkpoint_path(r, epoch)
        if direct is not None:
            return direct

        pths = glob.glob(os.path.join(r, "**", "*.pth"), recursive=True)
        all_pths.extend(pths)

    if not all_pths:
        return None

    best_like = [
        p for p in all_pths if re.search(r"\bbest\b", os.path.basename(p).lower())
    ]
    if best_like:
        best_like = sorted(best_like, key=lambda p: os.path.getmtime(p), reverse=True)
        return best_like[0]

    all_pths = sorted(all_pths, key=lambda p: os.path.getmtime(p), reverse=True)
    return all_pths[0]


def load_model(
    net,
    name,
    epoch,
    save_dir,
    device,
    fallback_to_torchvision_pretrained=True,
    data_dir=None,
):
    """
    Score fix (minimal, inference-only):
    - Prefer local checkpoints if present.
    - If save_dir doesn't contain .pth (common in Kaggle dataset-only notebooks), also search common roots
      including the competition dataset folder and /kaggle/input.
    - If no checkpoint exists, use a torchvision ImageNet-pretrained ResNet18 + deterministic prototype head.
    """
    roots = _candidate_checkpoint_roots(save_dir, data_dir)
    ckpt_path = _find_checkpoint_path_multi(roots, epoch)

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=device)
        net = _try_load_state_dict(net, state, device, strict_prefer=True)
        print(f"[INFO] Loaded checkpoint: {ckpt_path}")
        return net, True, False  # used_local_ckpt, used_tv_fallback

    if not fallback_to_torchvision_pretrained:
        raise FileNotFoundError(
            f"No .pth checkpoint found under roots={roots}. "
            f"Cannot run meaningful inference without pretrained weights."
        )

    tv = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    proto_w, proto_b = _build_imagenet_proto_head(tv, device)
    tv.fc = nn.Identity()
    tv = tv.to(device)
    tv = _TVResNet18WithProtoHead(tv, proto_w, proto_b).to(device)
    print(
        f"[WARN] No checkpoint found under roots={roots}. Using torchvision ResNet18 pretrained backbone + deterministic prototype 5-class head."
    )
    return tv, False, True


def print_and_save_args(args, path):
    message = ""
    for k, v in args.items():
        message += f"{str(k):>15}: {str(v):<10}\n"
    print(" " * 20 + "[OPTIONS]" + " " * 20)
    print(message)
    with open(path, "w") as f:
        f.write(message)


def _list_images_only(img_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = []
    for fn in os.listdir(img_dir):
        p = os.path.join(img_dir, fn)
        if os.path.isfile(p) and os.path.splitext(fn.lower())[1] in exts:
            files.append(fn)
    return sorted(files)




## === cell 4
args = {}
args["name"] = "mobilenet_384_randomcrop_width_mult_1.8"  # experiment name (kept)
args["batch_size"] = 32
args["width_mult"] = 1.0
args["dropout"] = 0.0
args["aug"] = False
args["model"] = "resnet"
args["gpu_id"] = 0

assert args["name"] is not None, "Must set experiment name before training"

data_dir = os.path.join(root, "input/cassava-leaf-disease-classification/")
save_dir = os.path.join(root, "input/pretrained3")

img_dir = os.path.join(data_dir, "test_images")

sample_path = os.path.join(data_dir, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)

available = set(_list_images_only(img_dir))
missing = [x for x in sample_sub["image_id"].tolist() if x not in available]
if missing:
    raise FileNotFoundError(
        f"{len(missing)} image_ids from sample_submission.csv are missing in {img_dir}. "
        f"First missing: {missing[0]}"
    )

test_pd = sample_sub[["image_id"]].copy()
num_test = len(test_pd)

device = "cuda:" + str(args["gpu_id"]) if torch.cuda.is_available() else "cpu"
print(f"test images: {num_test} \t device: {device}")

net = get_model(
    args["model"], width_mult=args["width_mult"], dropout=args["dropout"]
).to(device)

net, used_local_ckpt, used_tv_fallback = load_model(
    net,
    args["name"],
    "best",
    save_dir,
    device,
    fallback_to_torchvision_pretrained=True,
    data_dir=data_dir,  # CHANGE: enable searching within competition dataset tree too
)

_, test_transforms = get_transforms(
    args["aug"],
    use_imagenet_norm=used_tv_fallback,
    tv_resnet_eval_224=used_tv_fallback and (not args["aug"]),
)

test_dataset = CSVDataset(test_pd, img_dir, transform=test_transforms, aug=args["aug"])
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args["batch_size"],
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

net.eval()

train_csv_path = os.path.join(data_dir, "train.csv")
train_prior = None
if used_tv_fallback and os.path.isfile(train_csv_path):
    tr = pd.read_csv(train_csv_path)
    prior = tr["label"].value_counts(normalize=True).sort_index()
    prior = prior.reindex(range(5)).fillna(1e-6).values.astype(np.float32)
    prior = prior / prior.sum()
    train_prior = torch.tensor(np.log(prior + 1e-12), device=device).view(1, -1)
    print(
        f"[INFO] Using train label prior bias in fallback mode: {prior.round(4).tolist()}"
    )



## === cell 5
"""
Test + write submission.csv
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

pred_list = []
with torch.no_grad():
    for data in test_dataloader:
        imgs = data["image"].float().to(device, non_blocking=True)
        outputs = net(imgs)
        if train_prior is not None:
            outputs = outputs + 0.15 * train_prior
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
print(
    f"[INFO] Checkpoint source: {'local .pth' if used_local_ckpt else ('torchvision-pretrained fallback' if used_tv_fallback else 'random')}"
)
