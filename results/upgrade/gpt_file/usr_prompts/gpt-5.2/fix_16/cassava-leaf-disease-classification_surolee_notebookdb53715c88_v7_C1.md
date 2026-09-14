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

0.8513145965548504

# 6. Current score

0.3651

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11659) has done: 'I remove the TensorBoard `SummaryWriter` import that is crashing due to a broken/incompatible tensorboard installation, since it isn’t used for inference. I also fix test image discovery so it only includes actual image files (and not the nested `test_images/` directory), which currently causes `IsADirectoryError`. Next, I make model weight loading robust: if the expected `/kaggle/input/pretrained1/best_net.pth` is missing, the code fall back to torchvision pretrained MobileNetV2 weights (same model family) and adapt the classifier to 5 classes so inference can still run end-to-end and produce a valid `submission.csv`. Finally, I fix the custom `Ensemble` forward to return its output to avoid silent `None` propagation if ever used.'
- What this solution (achieved 0.14499) has done: 'Your current score (0.11659) is far below the target (0.8513), so we should improve accuracy without changing the core model/inference logic. The biggest likely issue is a preprocessing mismatch: the fallback torchvision MobileNetV2 expects ImageNet normalization, but your pipeline uses mean/std=(0.5,0.5,0.5), which can tank accuracy. I add a minimal “inference-normalization” switch: when using the torchvision pretrained fallback, use its recommended resize/center-crop + ImageNet normalization; when using your checkpoint, keep your existing albumentations transforms unchanged. I also make the weight-loading path robust to either a plain state_dict or a checkpoint dict, so if `best_net.pth` exists but is saved differently, it still load correctly (and avoid silently bad weights).'
- What this solution (achieved 0.071) has done: 'Your low score is consistent with running an untrained 5-class head in the torchvision fallback path (you replace the classifier with a new Linear but never train it), so accuracy collapses. To move toward the 0.8513 target with minimal change and without altering your model/training logic, I keep your existing pipeline but change the fallback to use a MobileNetV2 that is fully pretrained on ImageNet and then map its 1000-class predictions into 5 cassava labels via a simple, deterministic grouping heuristic. I also make checkpoint loading stricter for shape mismatches so that if `best_net.pth` exists but has incompatible classifier dimensions, we reliably fall back instead of silently producing bad logits. This should substantially improve the submission compared to the current “random head” behavior while preserving your inference-only workflow and producing a valid `submission.csv`.'
- What this solution (achieved 0.15957) has done: 'Your current 0.071 score is far below the 0.8513 target because the torchvision fallback path is effectively guessing: mapping 1000 ImageNet logits into 5 cassava classes by index ranges has no semantic link to the cassava labels. With minimal change and preserving your existing model/training/inference structure, I switch the fallback to a fully pretrained ImageNet MobileNetV2 but use its own 1000-way argmax as a stable “visual descriptor” and then map those predicted ImageNet class IDs into 5 labels via a deterministic hash (still inference-only, no extra training). I also ensure the fallback transform exactly matches the official torchvision weights preprocessing (resize short side -> center crop), which avoids additional distribution shift. This should move accuracy upward from near-random while still keeping your checkpoint-first behavior unchanged when `best_net.pth` is available.'
- What this solution (achieved 0.54111) has done: 'Your score is far below the target, and the current fallback path is effectively random because it uses an ImageNet-1000 model with a hash-to-5 mapping. The smallest score-moving change that preserves your overall inference pipeline is to use the actual cassava training set to learn a tiny “adapter” from the pretrained MobileNetV2’s 1000-way logits to the 5 cassava labels (a simple linear layer trained for 1 epoch), without changing your core model architecture or loss for the main model. If your original `best_net.pth` loads successfully, nothing changes; the adapter is only trained/used when the checkpoint is missing/incompatible. This should move accuracy substantially upward toward the target while staying within the 600s budget and still writing a valid `submission.csv`.'
- What this solution (achieved 0.39275) has done: 'Your current score (0.54111) is far below the target (0.8513), so we should improve accuracy with minimal changes while keeping your inference-first structure. The biggest issue is the adapter training labels are misaligned with images because the DataLoader is shuffled but labels are sliced from `train_df` by index, effectively training on wrong targets. I fix this by including `label` in the training dataset samples and returning it from `CSVDataset` when available, so adapter training uses the correct per-image label while keeping the same adapter model and 1-epoch-style loop. I also make the adapter train on the full training set (instead of a fixed 350 batches) to move score upward toward the target, still staying within time by using the same efficient “frozen net + small linear head” approach.'
- What this solution (achieved 0.43274) has done: 'Your current score (0.39275) is far below the target (0.8513), so we should cautiously increase accuracy without changing the main model/loop design. The most likely remaining issue in your fallback+adapter path is that the MobileNetV2 backbone is left in `eval()` during adapter training, so its BatchNorm layers never adapt to the cassava domain; enabling BN updates (while still freezing weights) often provides a large accuracy jump with minimal code change. I also make the adapter train a bit more stable by using a deterministic train/val split and selecting the best adapter by validation accuracy (still the same linear adapter and CrossEntropy), and I keep inference semantics unchanged. These are minimal, directly score-relevant adjustments that should move you closer to the target without altering your core architecture/training approach.'
- What this solution (achieved 0.44432) has done: 'Your current score (0.43274) is far below the target (0.8513), so we should improve the fallback+adapter path with minimal, score-relevant changes while keeping your model/inference structure intact. The biggest remaining issue is that adapter training uses `with torch.no_grad(): logits1000 = net(imgs)`, which prevents gradients from flowing through BatchNorm, so your `net.train()` “BN updates enabled” comment isn’t actually true; removing `no_grad()` (while keeping all net params frozen) lets BN stats adapt to cassava without changing the core approach. To make the adapter fit cleaner without changing its architecture or loss, we also normalize the 1000-d logits before the linear layer (a simple, deterministic calibration) and use a small LR scheduler; both are lightweight and directly improve accuracy. Finally, we keep your submission format/alignment logic unchanged and ensure everything still runs within the time budget.'
- What this solution (achieved 0.43087) has done: 'Your current score (0.44432) is far below the target (0.8513), so we should increase accuracy with minimal, score-relevant changes while preserving your overall fallback+adapter approach. The biggest likely correctness issue is that your adapter training currently runs MobileNetV2 in `train()` mode, which updates BatchNorm and applies Dropout—this can make the 1000-logit features unstable and hurt generalization; we keep BN adaptation but disable Dropout by setting Dropout layers to `eval()` while keeping BN layers in `train()`. Next, we remove the LR scheduler mismatch (you step once per epoch but use a `T_max=2` schedule) by switching to a stable, minimal `StepLR` per epoch. Finally, we add a very small amount of label-smoothing in `CrossEntropyLoss` for the adapter to improve calibration without changing the core model or loss type (still CE), which typically nudges accuracy upward.'
- What this solution (achieved 0.42265) has done: 'Your current gap to the target is large (0.43087 vs 0.8513), and the biggest remaining bottleneck is the fallback+adapter path: you’re learning from MobileNetV2’s final 1000-way logits, which are less transferable than the penultimate pooled features. With minimal changes and the same overall “frozen backbone + small linear adapter trained briefly on train.csv” core logic, we switch the adapter input from 1000-logit outputs to MobileNetV2’s 1280-d pooled features. We also ensure BatchNorm adaptation is actually happening during adapter training by running only the backbone features in train-mode (BN train, dropout eval), while keeping the classifier unused. Finally, inference use the same feature extractor + adapter, and we keep the submission formatting/alignment logic unchanged.'
- What this solution (achieved 0.43161) has done: 'Your current score (0.42265) is far below the target (0.8513), so we should improve accuracy in the only place that can realistically move the needle: the fallback+adapter path. The adapter is currently trained with MobileNetV2 in `train()` (to adapt BatchNorm), but you immediately switch the whole net to `eval()` for validation each epoch, which resets BN behavior and makes the learned adapter less consistent; we keep the backbone in “BN-train/dropout-eval” mode throughout adapter training+validation and only use full `eval()` at final test inference. We also make adapter training use the true pooled features by registering a forward hook on `net.features` (before any classifier), which is more stable than recomputing in multiple places and avoids accidental mode mismatches. Finally, we fix a small but impactful inference bug: you currently compute `outputs = net(imgs)` even in the adapter path and then recompute features again—wasting time and potentially using different BN behavior; we directly compute features->adapter once for the fallback path.'
- What this solution (achieved 0.35389) has done: 'Your current score (0.43161) is far below the target (0.8513), so we should increase accuracy with minimal, score-relevant changes in the only path that can affect this run: the torchvision-fallback + adapter training. The largest likely bottleneck is that the adapter trains on a randomly shuffled split without stratification, and trains for only 2 epochs with a relatively aggressive LR; this can underfit/overfit unevenly and hurt generalization. I keep the exact same backbone (torchvision MobileNetV2), same “frozen backbone + BN adaptation + linear adapter + CrossEntropy” core logic, but make the split stratified, add class-balanced weights to CE (still CE), and slightly stabilize optimization (lower LR and a few more epochs) to move accuracy upward toward the target. Submission writing/format and checkpoint-first behavior remain unchanged.'
- What this solution (achieved 0.39574) has done: 'The timeout is dominated by the fallback path: it reads and decodes ~18k training JPGs and runs 8 epochs of feature extraction+adapter training before doing test inference. To keep identical core logic and accuracy while fitting under 600s, the key speedups are: (1) avoid the fallback training entirely when the intended checkpoint exists by also looking in the dataset’s `pretrained1/` folder (same filename, same load semantics), (2) make dataloading/inference faster via more workers, persistent workers, prefetching, and pinned-memory, and (3) eliminate per-batch Python overhead in prediction by preallocating the prediction array and filling slices. These changes do not alter the model architecture, loss, augmentation, or evaluation logic; they only remove unnecessary work and improve I/O throughput.'
- What this solution (achieved 0.3651) has done: 'Your current score is far below the target, so we should improve accuracy (not just speed) in the only path that affects this run: the torchvision-fallback + adapter training. The biggest accuracy issue is that you’re using ImageNet MobileNetV2’s preprocessing (224 center-crop) and then adapting on those features, which is a poor match to cassava and wastes the higher-res 384 pipeline your original model expects. With minimal changes and identical “frozen backbone + BN adaptation + linear adapter + CrossEntropy” core logic, I switch the fallback preprocessing to your existing 384 albumentations validation transform for both adapter train/val and test inference, and I add a small warmup+cosine schedule (still full epochs, no early stopping) to stabilize training toward higher accuracy. I also make the backbone BN adaptation more correct by forcing only BN layers to train while everything else stays eval, which usually improves transfer without changing architecture.'

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
        self.img_labels = annotations_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.aug = aug

        self.has_label = "label" in self.img_labels.columns

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]["image_id"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            if self.aug:
                image = np.array(image)
                image = self.transform(image=image)
                image = image["image"]
            else:
                image = self.transform(image)

        sample = {"image": image}

        if self.has_label:
            label = int(self.img_labels.iloc[idx]["label"])
            if self.target_transform:
                label = self.target_transform(label)
            sample["label"] = label

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
        )
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
        out = self.layers(x)
        return out


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor

    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


"""
VIT
NOTE: kept as-is; not used in current args.
"""


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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


def get_transforms(aug=True, p=0.3):
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
                transforms.ToTensor(),
                transforms.RandomCrop((384, 384)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        val_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop((384, 384)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    return train_transforms, val_transforms


def save_model(net, name, epoch, save_dir):
    path = os.path.join(save_dir, f"{epoch}_net.pth")
    torch.save(net.state_dict(), path)


def load_model(net, name, epoch, save_dir, device):
    path = os.path.join(save_dir, f"{epoch}_net.pth")
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            state = obj["model_state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            state = obj["model"]
        else:
            state = obj
    else:
        state = obj

    model_state = net.state_dict()
    for k, v in list(state.items()):
        if k in model_state and hasattr(v, "shape") and model_state[k].shape != v.shape:
            raise RuntimeError(
                f"Checkpoint incompatible at {k}: ckpt{tuple(v.shape)} != model{tuple(model_state[k].shape)}"
            )

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




## === cell 4
args = {}
args["name"] = "mobilenet_384_randomcrop_width_mult_1.8"
args["batch_size"] = 32
args["width_mult"] = 1.8
args["dropout"] = 0.0
args["aug"] = True
args["model"] = "mobilenet"
args["gpu_id"] = 0

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

assert args["name"] is not None, "Must set experiment name before training"

data_dir = os.path.join(root, "input/cassava-leaf-disease-classification/")
save_dir = os.path.join(root, "input/pretrained1")

img_dir = os.path.join(data_dir, "test_images")

valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")
image_files = []
for fn in os.listdir(img_dir):
    fp = os.path.join(img_dir, fn)
    if os.path.isfile(fp) and fn.lower().endswith(valid_ext):
        image_files.append(fn)
image_files = sorted(image_files)

test_pd = pd.DataFrame({"image_id": image_files})
num_test = len(test_pd)

device = "cuda:" + str(args["gpu_id"]) if torch.cuda.is_available() else "cpu"
print(f"test images: {num_test} \t device: {device}")

net = get_model(
    args["model"], width_mult=args["width_mult"], dropout=args["dropout"]
).to(device)

ckpt_candidates = [
    os.path.join(save_dir, "best_net.pth"),
    os.path.join(data_dir, "pretrained1", "best_net.pth"),
]
ckpt_path = next((p for p in ckpt_candidates if os.path.exists(p)), ckpt_candidates[0])

using_torchvision_fallback = False
fallback_uses_imagenet1000 = False

if os.path.exists(ckpt_path):
    try:
        net = load_model(net, args["name"], "best", os.path.dirname(ckpt_path), device)
        print(f"Loaded checkpoint: {ckpt_path}")
    except Exception as e:
        using_torchvision_fallback = True
        print(f"WARNING: checkpoint load failed ({e}). Falling back to torchvision.")
else:
    using_torchvision_fallback = True
    print(f"WARNING: checkpoint not found at {ckpt_path}. Falling back to torchvision.")

if using_torchvision_fallback:
    tv_weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
    net = models.mobilenet_v2(weights=tv_weights).to(device)
    fallback_uses_imagenet1000 = True

adapter = None
tv_weights_for_transforms = None

_, cassava_val_tf = get_transforms(args["aug"])

if using_torchvision_fallback:
    test_transforms = cassava_val_tf
    test_dataset = CSVDataset(test_pd, img_dir, transform=test_transforms, aug=True)
else:
    _, test_transforms = get_transforms(args["aug"])
    test_dataset = CSVDataset(
        test_pd, img_dir, transform=test_transforms, aug=args["aug"]
    )

nw = min(8, (os.cpu_count() or 2))
test_dataloader = DataLoader(
    test_dataset,
    batch_size=args["batch_size"],
    shuffle=False,
    num_workers=nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(nw > 0),
    prefetch_factor=4 if nw > 0 else None,
)

net.eval()



## === cell 5
"""
Test
"""
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "%.2f%s" % (num, ["", "K", "M", "G", "T", "P"][magnitude])


print(f"Number of total parameters: {human_format(num_params)}")


class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self, mnet: nn.Module):
        super().__init__()
        self.mnet = mnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mnet.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)  # [B, 1280]
        return x


if fallback_uses_imagenet1000:
    train_csv_path = os.path.join(data_dir, "train.csv")
    train_img_dir = os.path.join(data_dir, "train_images")
    train_df = pd.read_csv(train_csv_path)

    train_transforms_fallback = cassava_val_tf

    rng = np.random.default_rng(SEED)
    labels_np = train_df["label"].to_numpy()
    all_idx = np.arange(len(train_df))
    val_frac = 0.1
    val_idx_list = []
    tr_idx_list = []
    for c in np.unique(labels_np):
        c_idx = all_idx[labels_np == c]
        rng.shuffle(c_idx)
        c_val_n = max(1, int(round(val_frac * len(c_idx))))
        val_idx_list.append(c_idx[:c_val_n])
        tr_idx_list.append(c_idx[c_val_n:])
    val_idx = np.concatenate(val_idx_list)
    tr_idx = np.concatenate(tr_idx_list)
    rng.shuffle(val_idx)
    rng.shuffle(tr_idx)

    train_dataset = CSVDataset(
        train_df.iloc[tr_idx][["image_id", "label"]],
        train_img_dir,
        transform=train_transforms_fallback,
        aug=True,
    )
    val_dataset = CSVDataset(
        train_df.iloc[val_idx][["image_id", "label"]],
        train_img_dir,
        transform=train_transforms_fallback,
        aug=True,
    )

    nw = min(8, (os.cpu_count() or 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )

    class FeaturesAdapter(nn.Module):
        def __init__(self, in_dim=1280):
            super().__init__()
            self.norm = nn.LayerNorm(in_dim)
            self.fc = nn.Linear(in_dim, 5)

        def forward(self, x):
            x = self.norm(x)
            return self.fc(x)

    adapter = FeaturesAdapter(in_dim=1280).to(device)

    def set_only_bn_train(m: nn.Module):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()
        else:
            m.eval()

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=8e-4, weight_decay=1e-4)

    EPOCHS = 10
    steps_per_epoch = max(1, len(train_loader))
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    counts = train_df["label"].value_counts().reindex(range(5), fill_value=1).to_numpy()
    class_w = (counts.sum() / counts).astype(np.float32)
    class_w = class_w / class_w.mean()
    class_w_t = torch.tensor(class_w, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w_t, label_smoothing=0.05)

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    net.apply(set_only_bn_train)
    adapter.train()

    feature_extractor = MobileNetV2FeatureExtractor(net).to(device)

    t0 = time.time()
    best_state = None
    best_val_acc = -1.0
    global_step = 0

    for ep in range(EPOCHS):
        for bidx, batch in enumerate(train_loader):
            imgs = batch["image"].float().to(device, non_blocking=True)
            labels = torch.as_tensor(batch["label"], device=device, dtype=torch.long)

            feats1280 = feature_extractor(imgs)
            logits5 = adapter(feats1280)
            loss = criterion(logits5, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

        adapter.eval()
        net.eval()
        net.apply(set_only_bn_train)

        correct = 0
        total = 0
        with torch.no_grad():
            for vbatch in val_loader:
                vimgs = vbatch["image"].float().to(device, non_blocking=True)
                vlabels = torch.as_tensor(
                    vbatch["label"], device=device, dtype=torch.long
                )
                vfeats1280 = feature_extractor(vimgs)
                vlogits5 = adapter(vfeats1280)
                pred = vlogits5.argmax(dim=1)
                correct += (pred == vlabels).sum().item()
                total += vlabels.numel()

        val_acc = correct / max(1, total)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(adapter.state_dict())

        adapter.train()

    if best_state is not None:
        adapter.load_state_dict(best_state)

    print(
        f"Trained adapter on fallback path in {time.time()-t0:.1f}s (epochs={EPOCHS}, best_val_acc={best_val_acc:.4f})"
    )

pred_arr = np.empty((len(test_dataset),), dtype=np.int64)

net.eval()
if adapter is not None:
    adapter.eval()

use_tta = bool(fallback_uses_imagenet1000 and adapter is not None)

if fallback_uses_imagenet1000:
    feature_extractor = MobileNetV2FeatureExtractor(net).to(device)
    feature_extractor.eval()

offset = 0
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        imgs = data["image"].float().to(device, non_blocking=True)
        bs = imgs.size(0)

        if fallback_uses_imagenet1000:
            if adapter is None:
                outputs = net(imgs)
                preds = (
                    (outputs.argmax(dim=1) % 5)
                    .to("cpu", non_blocking=True)
                    .long()
                    .numpy()
                )
            else:
                if use_tta:
                    feats1 = feature_extractor(imgs)
                    logits1 = adapter(feats1)

                    imgs_flip = torch.flip(imgs, dims=[3])
                    feats2 = feature_extractor(imgs_flip)
                    logits2 = adapter(feats2)

                    logits5 = 0.5 * (logits1 + logits2)
                else:
                    feats1280 = feature_extractor(imgs)
                    logits5 = adapter(feats1280)

                preds = (
                    logits5.argmax(dim=1).to("cpu", non_blocking=True).long().numpy()
                )
        else:
            outputs = net(imgs)
            if outputs.ndim == 2 and outputs.size(1) != 5:
                outputs = outputs[:, :5]
            preds = outputs.argmax(dim=1).to("cpu", non_blocking=True).long().numpy()

        pred_arr[offset : offset + bs] = preds
        offset += bs

test_pd["label"] = pred_arr

sample_path = os.path.join(data_dir, "sample_submission.csv")
if os.path.exists(sample_path):
    sample_sub = pd.read_csv(sample_path)
    test_pd = sample_sub[["image_id"]].merge(test_pd, on="image_id", how="left")
    test_pd["label"] = test_pd["label"].fillna(0).astype(int)

print(test_pd.head())
test_pd.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
