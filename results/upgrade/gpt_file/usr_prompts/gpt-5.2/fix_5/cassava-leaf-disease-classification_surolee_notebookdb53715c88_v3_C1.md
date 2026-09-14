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

0.6128739800543971

# 6. Current score

0.33931

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50523) has done: 'I remove the TensorBoard `SummaryWriter` import because it’s crashing due to an incompatible tensorboard/protobuf stack and it isn’t used for inference. I fix test image listing to include only actual `.jpg` files (the folder contains a nested `test_images/` directory that currently gets treated as an image). I also make model loading robust: if `/kaggle/input/pretrained/best_net.pth` is missing, the code fall back to an ImageNet-pretrained backbone with the final classifier resized to 5 classes so inference can still run end-to-end and produce `submission.csv`. Finally, I ensure the submission uses the exact `sample_submission.csv` ordering/rows so the output format is valid for Kaggle.'
- What this solution (achieved 0.32025) has done: 'Your current score is below target (0.50523 vs 0.61287), so we should improve accuracy with minimal, low-risk changes while keeping the same model and inference flow. The biggest issue is that when the pretrained checkpoint is missing, the fallback currently resets the classifier head randomly, which yields weak predictions; instead we load the ImageNet backbone and keep its head, then map its 1000-class probabilities down to the 5 cassava classes via a fixed, data-driven mapping computed from the training set (no test leakage). This keeps the “mobilenet inference” core logic intact (single forward pass + argmax) but makes predictions substantially more meaningful. We also fix the val/test transforms to match ImageNet normalization when using the ImageNet fallback so the backbone sees the input distribution it expects, which typically improves accuracy.'
- What this solution (achieved 0.33931) has done: 'We keep your model/inference logic intact and make two minimal changes that should legitimately increase accuracy toward the target: (1) ensure the ImageNet→Cassava mapping is computed using the *same input preprocessing the torchvision MobileNetV2 weights expect* (resize/crop + correct interpolation), and (2) make test-time preprocessing match that same expected preprocessing when the fallback wrapper is used. Right now the mapping step uses a different preprocessing order (ToTensor before Resize) and lacks the standard center-crop pipeline, which can make the 1000→5 probability mapping noisy and hurt accuracy. These changes don’t alter the architecture, training loop, or prediction semantics (still argmax over 5 logits), but typically improve the fallback’s calibration/accuracy. The script still runs end-to-end and writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.33931) has done: 'We’re currently far below the target (0.33931 vs 0.61287, higher-is-better), so we should make a small, low-risk change that improves accuracy without changing your model/training/inference semantics. The biggest accuracy drag in this script is that `args["aug"]=False` makes the non-aug torchvision pipeline apply `ToTensor()` before `CenterCrop/RandomCrop`, which is not the standard/expected order and can hurt performance; fixing the transform order keeps the same operations but makes them correct. We also make `DataLoader` use a non-zero `num_workers` and `pin_memory` when on CUDA for more stable throughput (no metric change), and keep submission alignment exactly as before. No architecture, loss, or inference logic is changed: it’s still single-pass logits → argmax → submission.'

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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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
Resnet
"""


class ResNet(nn.Module):
    def __init__(self, layers, dropout=0.0):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, padding=3, stride=2, bias=False
        )  ## this is stride2 in the original implementation
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(64, layers[0])
        self.layer2 = self.make_layer(128, layers[1], stride=2)
        self.layer3 = self.make_layer(256, layers[2], stride=2)
        self.layer4 = self.make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)
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
            nn.Dropout(dropout), nn.Linear(last_channel, 10)
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

        layers = []  # depthwise separable convolution with bottleneck
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
            nn.ReLU6(inplace=True),  # necessary? consider switching to relu
        )

    def forward(self, x):
        out = self.layers(x)
        return out


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor

    new_v = max(
        min_value, int(v + divisor / 2) // divisor * divisor
    )  # rounds numbers in range [num-divisor/2, num+divisor/2-1] to num, where num is a multiple of divisor

    if new_v < 0.9 * v:
        new_v += divisor  # ensures that round down does not decrease v by more than 10%

    return new_v


"""
VIT
NOTE: This block is kept as-is (core logic preservation). It is not used in the current args.
It references einops symbols that are not imported; that's fine as long as VIT isn't instantiated.
"""


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
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

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
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
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
        return models.resnet18()
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


def get_transforms(aug=True, p=0.3, imagenet_norm=False):
    if imagenet_norm:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    train_transforms = None
    val_transforms = None
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
    else:
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


def save_model(net, name, epoch, save_dir):
    path = os.path.join(save_dir, f"{epoch}_net.pth")
    torch.save(net.state_dict(), path)


def load_model(net, name, epoch, save_dir, device):
    path = os.path.join(save_dir, f"{epoch}_net.pth")
    net.load_state_dict(torch.load(path, map_location=device), strict=False)
    return net


class ImageNetToCassavaWrapper(nn.Module):
    def __init__(self, imagenet_model, idx_map_5xK: torch.Tensor):
        super().__init__()
        self.imagenet_model = imagenet_model
        self.register_buffer("idx_map_5xK", idx_map_5xK.long())

    def forward(self, x):
        logits_1000 = self.imagenet_model(x)
        probs_1000 = torch.softmax(logits_1000, dim=1)
        gathered = probs_1000[:, self.idx_map_5xK]  # [B, 5, K]
        probs_5 = gathered.sum(dim=2)  # [B, 5]
        logits_5 = torch.log(probs_5.clamp_min(1e-12))
        return logits_5


def build_imagenet_to_cassava_mapping(
    train_csv_path, train_img_dir, device, topk=25, batch_size=32, num_workers=2
):
    train_df = pd.read_csv(train_csv_path)
    train_df = train_df[["image_id", "label"]].copy()
    train_df["label"] = train_df["label"].astype(int)

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    tfm = weights.transforms()

    ds = CSVDataset(train_df, train_img_dir, transform=tfm, aug=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = models.mobilenet_v2(weights=weights).to(device)
    model.eval()

    sums = torch.zeros(5, 1000, device=device)
    counts = torch.zeros(5, device=device)

    with torch.no_grad():
        base_idx = 0
        for batch in dl:
            b = batch["image"].to(device).float()
            logits = model(b)
            probs = torch.softmax(logits, dim=1)  # [B,1000]
            labels = torch.tensor(
                train_df.iloc[base_idx : base_idx + b.size(0)]["label"].values,
                device=device,
                dtype=torch.long,
            )
            base_idx += b.size(0)
            for c in range(5):
                m = labels == c
                if m.any():
                    sums[c] += probs[m].sum(dim=0)
                    counts[c] += m.sum()

    means = sums / counts.clamp_min(1.0).unsqueeze(1)
    idx = torch.topk(means, k=topk, dim=1).indices  # [5, topk]
    return idx.detach().cpu()


def load_model_or_fallback(net, name, epoch, save_dir, device, args, data_dir):
    path = os.path.join(save_dir, f"{epoch}_net.pth")
    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location=device), strict=False)
        return net, f"loaded checkpoint: {path}", False

    if args["model"] == "mobilenet":
        train_csv_path = os.path.join(data_dir, "train.csv")
        train_img_dir = os.path.join(data_dir, "train_images")
        idx_map = build_imagenet_to_cassava_mapping(
            train_csv_path=train_csv_path,
            train_img_dir=train_img_dir,
            device=device,
            topk=25,
            batch_size=32,
            num_workers=2,
        )
        idx_map = idx_map.to(device)

        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        tv = models.mobilenet_v2(weights=weights).to(device)

        wrapped = ImageNetToCassavaWrapper(tv, idx_map_5xK=idx_map).to(device)
        return (
            wrapped,
            "fallback to ImageNet mobilenet_v2 + train-derived 1000->5 class probability mapping",
            True,
        )
    elif args["model"] == "base":
        tv = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        tv.fc = nn.Linear(tv.fc.in_features, 5)
        return (
            tv.to(device),
            "fallback to torchvision resnet18 ImageNet weights (head reset to 5 classes)",
            True,
        )
    else:
        return (
            net.to(device),
            "fallback to randomly initialized model (no checkpoint found)",
            False,
        )


def print_and_save_args(args, path):
    message = ""
    for k, v in args.items():
        message += f"{str(k):>15}: {str(v):<10}\n"
    print(" " * 20 + "[OPTIONS]" + " " * 20)
    print(message)
    with open(path, "w") as f:
        f.write(message)




## === cell 4
args = {}
args["name"] = (
    "mobilenet_384_randomcrop_width_mult_1.8"  # MUST SET EXPERIMENT NAME BEFORE TRAINING (for saving model)
)
args["batch_size"] = 1
args["width_mult"] = 1.8
args["dropout"] = 0.0
args["aug"] = False
args["model"] = "mobilenet"
args["gpu_id"] = 0

assert args["name"] is not None, "Must set experiment name before training"

data_dir = os.path.join(root, "input/cassava-leaf-disease-classification/")
save_dir = os.path.join(root, "input/pretrained")

img_dir = os.path.join(data_dir, "test_images")

test_image_ids = sorted(
    [
        f
        for f in os.listdir(img_dir)
        if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(img_dir, f))
    ]
)
test_pd = pd.DataFrame({"image_id": test_image_ids})
num_test = len(test_pd)

device = "cuda:" + str(args["gpu_id"]) if torch.cuda.is_available() else "cpu"
print(f"test images: {num_test} \t device: {device}")

net = get_model(
    args["model"], width_mult=args["width_mult"], dropout=args["dropout"]
).to(device)
net, load_msg, used_imagenet_fallback = load_model_or_fallback(
    net, args["name"], "best", save_dir, device, args, data_dir
)
print(load_msg)

if used_imagenet_fallback:
    test_transforms = models.MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
else:
    _, test_transforms = get_transforms(args["aug"], imagenet_norm=False)

test_dataset = CSVDataset(test_pd, img_dir, transform=test_transforms, aug=args["aug"])

num_workers = 2
pin_memory = torch.cuda.is_available()
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args["batch_size"],
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

if isinstance(net, MobileNetV2):
    if (
        isinstance(net.classifier, nn.Sequential)
        and isinstance(net.classifier[-1], nn.Linear)
        and net.classifier[-1].out_features != 5
    ):
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 5).to(device)
if isinstance(net, ResNet):
    if isinstance(net.fc, nn.Linear) and net.fc.out_features != 5:
        net.fc = nn.Linear(net.fc.in_features, 5).to(device)



## === cell 5
"""
Test + Submission writing
"""
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "%.2f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


print(f"Number of total parameters: {human_format(num_params)}")

net.eval()
pred_list = []
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        imgs = data["image"].float().to(device, non_blocking=True)
        outputs = net(imgs)
        pred_list += [outputs.argmax(dim=1).item()]

test_pd["label"] = np.array(pred_list, dtype=int)

sample_path = os.path.join(data_dir, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)
sub = sample_sub[["image_id"]].merge(
    test_pd[["image_id", "label"]], on="image_id", how="left"
)
sub["label"] = sub["label"].fillna(0).astype(int)

print(sub.head())
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
