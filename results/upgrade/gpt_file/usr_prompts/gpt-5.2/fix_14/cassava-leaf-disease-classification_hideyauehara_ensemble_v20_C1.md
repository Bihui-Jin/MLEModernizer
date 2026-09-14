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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
seaborn==0.12.2
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

0.8919613176186159

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the missing `efficientnet_pytorch` dependency by switching only that branch to use `torchvision`’s EfficientNet-B7 while keeping the same ensemble/forward API so the rest of the pipeline is unchanged. I also make the input-path detection robust for this Kaggle environment (your filesystem uses `/kaggle/input/...` rather than `../input/...`) so `test_files` and `df_test` are created correctly. Albumentations v2 changed some transform signatures, so I update `RandomResizedCrop` and `CenterCrop`/`Rotate` usage to the current API to remove the validation error without changing the intended TTA logic. Finally, I guard against the “no models found” case and ensure `probability` always stacks to a 3D array so the mean/argmax logic can’t crash and a valid `submission.csv` is always written.'
- What this solution (achieved 0.11584) has done: 'Your very low score strongly suggests the submission rows are not aligned to Kaggle’s provided `sample_submission.csv` (ordering/contents), and/or the model weights are being loaded into a slightly different architecture than they were trained with (especially EfficientNet-B7), yielding near-random predictions. I make two minimal, directly score-relevant fixes: (1) always build `df_test` from `sample_submission.csv` so row order and image_ids match the official test set exactly, and (2) load EfficientNet-B7 with the correct `num_classes=5` head so checkpoints load cleanly and logits map to the right labels. These changes preserve your ensemble/TTA logic and inference semantics while addressing the two most common causes of ~0.11 accuracy in this competition. The script still write a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.11584) has done: 'Your 0.11584 accuracy is consistent with a checkpoint load mismatch: the saved `.pth` likely contains a wrapped module (e.g., `model.` prefix) and/or stores a dict with `state_dict`, so `net.load_state_dict(state)` silently fails to load the intended weights (or loads wrong keys), yielding near-random predictions. I add a minimal, robust checkpoint loader that extracts the real `state_dict`, strips common prefixes, and loads with strict checking (falling back to non-strict only when necessary while reporting missing/unexpected keys). I also fix a small but impactful bug in the EfficientNet wrapper: `FinalLayerMixupModelEN` currently ignores the `alpha` argument and overwrites `self.alpha`, which can change the module structure relative to how it was saved. These changes keep the same model families, TTA, and inference flow, but make sure the ensemble is actually using the learned weights, which should move accuracy sharply toward your target.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, which is most consistent with the pretrained EfficientNet-B7 checkpoints not being loaded into an architecture that matches how they were saved, making predictions effectively random. I keep your ensemble/TTA/inference flow unchanged, but make the EfficientNet branch build the exact same “backbone + replaced head” structure as your wrapper expects, and I make the checkpoint loader handle the common EfficientNet key renames (`_fc.` ↔ `classifier.*`) so the final classification layer weights load correctly instead of being silently dropped. I also ensure the test image path is resolved robustly (some datasets are nested) without changing the ordering (still driven by `sample_submission.csv`). These are minimal, directly score-relevant fixes that should move accuracy sharply upward toward your target.'
- What this solution (achieved 0.11584) has done: 'Your accuracy is near random, which most often happens here when the EfficientNet-B7 checkpoints were trained with a different implementation (`efficientnet_pytorch`) than the current `torchvision` model, so the weights don’t load into the right layers/head even if `load_state_dict` “succeeds.” To move your score sharply upward toward the target while keeping your ensemble/TTA/inference logic unchanged, I switch only the EfficientNet-B7 branch to instantiate an `EfficientNet` from `torchvision.models` and replace the classifier head in the standard way so the key names (`features.*` and `classifier.1.*`) match typical saved checkpoints. I also make the checkpoint remapping strictly one-way (from `_fc.*` to `classifier.1.*`) to avoid accidentally remapping in the wrong direction, and I ensure `predict_model` calls the model correctly for EfficientNet (since your wrapper expects three args). These are minimal, directly score-relevant changes aimed at making the loaded weights actually take effect.'
- What this solution (achieved 0.11584) has done: 'Your score (0.11584) is near-random for 5 classes, which strongly suggests the test-time forward path is not actually using the learned heads/weights from the checkpoints (i.e., a wrapper/head mismatch), even if `load_state_dict` doesn’t crash. I make two minimal, directly score-relevant fixes: (1) in `FinalLayerMixupModelEN`, stop replacing the EfficientNet classifier twice (once in the base model and again inside the wrapper), and (2) make `predict_model` call the model correctly whether it is a wrapper expecting `(inputs, labels, phase)` or a plain torchvision model expecting only `inputs`. These changes preserve your ensemble/TTA and overall inference semantics, but ensure the loaded checkpoint weights map to the exact parameters used at inference, which should move accuracy sharply upward toward your target.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random for 5 classes, so the most likely cause is that the loaded checkpoints are not being applied to the actual parameters used at inference (especially for the EfficientNet branch), even if `load_state_dict` doesn’t crash. I keep your ensemble/TTA/inference flow unchanged, but (1) make the EfficientNet wrapper always use the model’s own classifier head (so checkpoint head weights map to the same module you call at inference), and (2) strengthen the checkpoint key-remapping to cover both common EfficientNet head conventions (`_fc.*`, `classifier.*`, `classifier.1.*`) so the 5-class head actually loads. I also make the final prediction assignment explicitly cast to `int` and ensure submission order stays exactly the `sample_submission.csv` order (already mostly correct) to avoid any silent formatting/alignment issues. These are minimal, directly score-relevant changes intended to move accuracy sharply upward toward your target band.'
- What this solution (achieved 0.11584) has done: 'Your current score (~0.116) is near-random for 5 classes, so the most likely issue is that your checkpoints are not actually being loaded into the exact parameters used at inference (especially the EfficientNet-B7 head), even if the script runs and writes a CSV. I keep your ensemble + TTA + inference loop intact, but make the EfficientNet-B7 wrapper use the model’s own `classifier` (not a separate head) and strengthen checkpoint key-remapping so common EfficientNet head keys load into `classifier.1.*` reliably. I also harden loading so it prefers strict loading but automatically retry with a couple of known EfficientNet key patterns before falling back to non-strict, which should move accuracy sharply upward toward your target. Finally, I keep submission order driven strictly by `sample_submission.csv` (already correct) and ensure the label column is always written as integer.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random for 5 classes, which strongly suggests the pretrained EfficientNet-B7 checkpoints are not actually being loaded into the same parameter names you use at inference (so the ensemble is effectively untrained). I keep your ensemble/TTA and model wrappers intact, but make the EfficientNet-B7 branch instantiate the backbone in a way that matches the most common checkpoint key convention (`_fc.*` from `efficientnet_pytorch`) by adding a lightweight `_fc` alias to `torchvision`’s classifier. I also make the checkpoint loader remap EfficientNet feature keys (`_conv_stem`, `_blocks`, etc.) to `torchvision`’s (`features.*`) when needed, while still preferring strict loading. These minimal, directly score-relevant changes should move accuracy sharply upward toward your target without changing the overall inference semantics or submission format.'
- What this solution (achieved 0.11584) has done: 'Your current score is near-random, which strongly suggests the EfficientNet-B7 checkpoints aren’t actually being loaded into matching parameter names/structure (so inference runs with mostly-random weights). I keep your ensemble/TTA and wrappers unchanged, but make the checkpoint loader explicitly support the common `efficientnet_pytorch`→`torchvision` key mapping for EfficientNet-B7 (including `_blocks.*` to the nested `features.1.*.block.*` structure) and validate that the classifier head weights are truly loaded. I also make submission writing unconditional from `sample_submission.csv` order (still the same semantics) and add a small safety check that warns if too many keys are missing, because that correlates with ~0.11 accuracy. These are minimal, directly score-relevant changes intended to move accuracy sharply upward toward your target band without changing the modeling/inference approach.'
- What this solution (achieved 0.11584) has done: 'Your current accuracy (~0.116) is close to random for 5 classes, which strongly suggests the EfficientNet-B7 checkpoints are not actually being loaded into a matching architecture/key-space (so inference uses mostly untrained weights). I keep your ensemble + TTA + inference loop intact, but switch only the EfficientNet-B7 backbone to the exact `efficientnet_b7` definition and parameter naming used by `efficientnet_pytorch` (which these Kaggle EB7 checkpoints were typically trained with), so `load_state_dict(strict=True)` can succeed and predictions become meaningful. I do this by adding a tiny in-notebook EfficientNet implementation (no extra packages/files) that matches the common checkpoint keys (`_conv_stem`, `_bn0`, `_blocks.*`, `_conv_head`, `_bn1`, `_fc`) and wiring it into your existing `FinalLayerMixupModelEN` wrapper unchanged. I also make the checkpoint loader prefer strict loading for EfficientNet and print a clear warning if strict loading fails (because that correlates with near-random score), without changing submission ordering (still driven by `sample_submission.csv`).'
- What this solution (achieved 0.11584) has done: 'Your current accuracy (~0.116) is near-random for 5 classes, so the most likely cause is that the EfficientNet-B7 checkpoints still aren’t being loaded into a truly matching key-space/architecture (even if the script runs). To move sharply upward toward your target while keeping your ensemble/TTA/inference flow intact, I (1) add a minimal key-remapping path specifically for `efficientnet_pytorch`-style checkpoints into your `EfficientNetPyTorchLike` model (including common head keys like `_fc.*` and `classifier.*`), and (2) enforce that EfficientNet loads strictly after remapping (otherwise we warn and skip that model rather than averaging in random predictions). Finally, I make the final prediction aggregation average over models first and then over TTA per-model (same semantics as averaging all proba arrays, but avoids overweighting a model that failed to load cleanly). These changes are directly score-relevant and keep the core architecture, TTA set, and submission formatting unchanged.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random for 5 classes, so the highest-probability issue is that the EfficientNet checkpoints (the main signal in this ensemble) still aren’t loading into a *matching* EfficientNet implementation, meaning you’re effectively averaging untrained/random models. I make a minimal, score-critical change: use `torchvision.models.efficientnet_b7(weights=None)` and wrap it with a tiny `_fc` alias so `efficientnet_pytorch`-style checkpoints (`_fc.*`) load into `classifier.1.*` without changing the ensemble/TTA logic. I also make EfficientNet loading stricter and skip only truly-mismatched checkpoints (as you already do) to avoid poisoning the ensemble with random predictions. Finally, I keep submission ordering driven strictly by `sample_submission.csv` and ensure `label` is always written as `int`.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
import os

CANDIDATE_MODEL_GLOBS = [
    "../input/eb7slseed70/efficientnet-b7sl_SEED70.orig/*.pth",
    "/kaggle/input/eb7slseed70/efficientnet-b7sl_SEED70.orig/*.pth",
]
pretrained_models = []
for pat in CANDIDATE_MODEL_GLOBS:
    pretrained_models.extend(glob.glob(pat))
pretrained_models = sorted(set(pretrained_models))

print(f"{len(pretrained_models)} models found.")
print("\n".join(np.sort(pretrained_models)[:50]))
if len(pretrained_models) > 50:
    print(f"... ({len(pretrained_models)-50} more)")



## === cell 2
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

import torchvision
from torchvision import models, transforms
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

from pathlib import Path
import random
import json
import time
import pickle
import sys

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import cv2


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)



## === cell 3
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple


def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dDynamicSamePadding(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
    ):
        if isinstance(kernel_size, int):
            k = (kernel_size, kernel_size)
        else:
            k = kernel_size
        if isinstance(stride, int):
            s = (stride, stride)
        else:
            s = stride
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k,
            stride=s,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        oh = (ih + sh - 1) // sh
        ow = (iw + sw - 1) // sw
        pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return super().forward(x)


class BatchNorm2d(nn.BatchNorm2d):
    pass


class SqueezeExcite(nn.Module):
    def __init__(self, in_ch, se_ratio=0.25):
        super().__init__()
        reduced = max(1, int(in_ch * se_ratio))
        self.fc1 = nn.Conv2d(in_ch, reduced, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced, in_ch, kernel_size=1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.fc1(s)
        s = F.silu(s) if hasattr(F, "silu") else s * torch.sigmoid(s)
        s = self.fc2(s)
        s = torch.sigmoid(s)
        return x * s


@dataclass
class BlockArgs:
    kernel_size: int
    num_repeat: int
    input_filters: int
    output_filters: int
    expand_ratio: int
    id_skip: bool
    stride: Tuple[int, int]
    se_ratio: float


class MBConvBlock(nn.Module):
    def __init__(self, block_args: BlockArgs, drop_connect_rate: float = 0.0):
        super().__init__()
        self._block_args = block_args
        self.drop_connect_rate = drop_connect_rate

        inp = block_args.input_filters
        oup = block_args.output_filters
        exp = block_args.expand_ratio
        k = block_args.kernel_size
        s = block_args.stride
        se_ratio = block_args.se_ratio

        mid = inp * exp

        self.has_expansion = exp != 1
        if self.has_expansion:
            self._expand_conv = Conv2dDynamicSamePadding(
                inp, mid, kernel_size=1, stride=1, bias=False
            )
            self._bn0 = BatchNorm2d(mid)

        self._depthwise_conv = Conv2dDynamicSamePadding(
            mid, mid, kernel_size=k, stride=s, groups=mid, bias=False
        )
        self._bn1 = BatchNorm2d(mid)

        self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        if self.has_se:
            self._se = SqueezeExcite(mid, se_ratio=se_ratio)

        self._project_conv = Conv2dDynamicSamePadding(
            mid, oup, kernel_size=1, stride=1, bias=False
        )
        self._bn2 = BatchNorm2d(oup)

        self._swish = Swish()

    def _drop_connect(self, x, p, training):
        if not training or p == 0.0:
            return x
        keep_prob = 1.0 - p
        batch_size = x.shape[0]
        random_tensor = keep_prob + torch.rand(
            [batch_size, 1, 1, 1], dtype=x.dtype, device=x.device
        )
        binary = torch.floor(random_tensor)
        return x / keep_prob * binary

    def forward(self, inputs):
        x = inputs
        if self.has_expansion:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        if self.has_se:
            x = self._se(x)

        x = self._project_conv(x)
        x = self._bn2(x)

        if self._block_args.id_skip:
            if (
                self._block_args.stride == (1, 1)
                and self._block_args.input_filters == self._block_args.output_filters
            ):
                if self.drop_connect_rate:
                    x = self._drop_connect(
                        x, p=self.drop_connect_rate, training=self.training
                    )
                x = x + inputs
        return x


class EfficientNetPyTorchLike(nn.Module):
    def __init__(
        self,
        width_coefficient,
        depth_coefficient,
        dropout_rate,
        num_classes=5,
        drop_connect_rate=0.2,
    ):
        super().__init__()
        self._drop_connect_rate = drop_connect_rate

        out_ch = _make_divisible(32 * width_coefficient, 8)
        self._conv_stem = Conv2dDynamicSamePadding(
            3, out_ch, kernel_size=3, stride=2, bias=False
        )
        self._bn0 = BatchNorm2d(out_ch)
        self._swish = Swish()

        base_blocks = [
            BlockArgs(3, 1, 32, 16, 1, True, (1, 1), 0.25),
            BlockArgs(3, 2, 16, 24, 6, True, (2, 2), 0.25),
            BlockArgs(5, 2, 24, 40, 6, True, (2, 2), 0.25),
            BlockArgs(3, 3, 40, 80, 6, True, (2, 2), 0.25),
            BlockArgs(5, 3, 80, 112, 6, True, (1, 1), 0.25),
            BlockArgs(5, 4, 112, 192, 6, True, (2, 2), 0.25),
            BlockArgs(3, 1, 192, 320, 6, True, (1, 1), 0.25),
        ]

        def round_filters(filters):
            return _make_divisible(filters * width_coefficient, 8)

        def round_repeats(repeats):
            return int(np.ceil(depth_coefficient * repeats))

        blocks = []
        for b in base_blocks:
            inp = round_filters(b.input_filters)
            oup = round_filters(b.output_filters)
            reps = round_repeats(b.num_repeat)
            blocks.append(
                BlockArgs(
                    b.kernel_size,
                    reps,
                    inp,
                    oup,
                    b.expand_ratio,
                    b.id_skip,
                    b.stride,
                    b.se_ratio,
                )
            )

        self._blocks = nn.ModuleList([])
        block_idx = 0
        total_blocks = sum(b.num_repeat for b in blocks)
        in_ch = out_ch
        for b in blocks:
            for i in range(b.num_repeat):
                stride = b.stride if i == 0 else (1, 1)
                ba = BlockArgs(
                    b.kernel_size,
                    1,
                    in_ch,
                    b.output_filters,
                    b.expand_ratio,
                    b.id_skip,
                    stride,
                    b.se_ratio,
                )
                dc = (
                    self._drop_connect_rate
                    * float(block_idx)
                    / float(max(total_blocks - 1, 1))
                )
                self._blocks.append(MBConvBlock(ba, drop_connect_rate=dc))
                in_ch = b.output_filters
                block_idx += 1

        head_in = in_ch
        head_out = _make_divisible(1280 * width_coefficient, 8)
        self._conv_head = Conv2dDynamicSamePadding(
            head_in, head_out, kernel_size=1, stride=1, bias=False
        )
        self._bn1 = BatchNorm2d(head_out)

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(dropout_rate)
        self._fc = nn.Linear(head_out, num_classes)

    def extract_features(self, inputs):
        x = self._conv_stem(inputs)
        x = self._bn0(x)
        x = self._swish(x)
        for block in self._blocks:
            x = block(x)
        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._swish(x)
        return x

    def forward(self, inputs):
        x = self.extract_features(inputs)
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        return x


def efficientnet_b7_pytorch_compatible(num_classes=5):
    return EfficientNetPyTorchLike(
        width_coefficient=2.0,
        depth_coefficient=3.1,
        dropout_rate=0.5,
        num_classes=num_classes,
    )




## === cell 4
SIZE = 512
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 6
def _first_existing_dir(candidates):
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return None


BASE_DIR = _first_existing_dir(
    [
        "../input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "data/cassava-leaf-disease-classification",
        "data",
    ]
)

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate cassava dataset directory in expected locations."
    )

run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE", "")
is_kaggle = os.path.isdir("/kaggle") and os.path.isdir("/kaggle/input")

if run_type == "Interactive":
    print("Test run in Kaggle environment.")
    TEST_PATH = f"{BASE_DIR}/train_images"
elif run_type == "Batch" or is_kaggle:
    print("In Kaggle environment.")
    TEST_PATH = f"{BASE_DIR}/test_images"
else:
    print("In the local environment.")
    TEST_PATH = f"{BASE_DIR}/train_images"

TEST_PATH = (
    _first_existing_dir(
        [
            TEST_PATH,
            os.path.join(
                BASE_DIR, "cassava-leaf-disease-classification", "test_images"
            ),
            os.path.join(
                BASE_DIR, "cassava-leaf-disease-classification", "train_images"
            ),
            os.path.join(
                "/kaggle/input/cassava-leaf-disease-classification", "test_images"
            ),
            os.path.join(
                "/kaggle/input/cassava-leaf-disease-classification",
                "cassava-leaf-disease-classification",
                "test_images",
            ),
        ]
    )
    or TEST_PATH
)

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEST_PATH: {TEST_PATH}")



## === cell 7
df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")[["image_id", "label"]].copy()
df_test["label"] = 1  # placeholder; will be overwritten by predictions
print(df_test.head())
print(f"Number of test images (from sample_submission): {len(df_test)}")



## === cell 8
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)



## === cell 9
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333333),
                    p=1.0,
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333333),
                    p=1.0,
                ),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(
                    size=(SIZE, SIZE),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333333),
                    p=1.0,
                ),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(limit=30, p=1.0),
                A.RandomResizedCrop(
                    size=(SIZE, SIZE),
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333333333),
                    p=1.0,
                ),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}



## === cell 10
pass




## === cell 11
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = x.squeeze()
            outputs = self.fc(x)
            return outputs

        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels))

        x1 = inputs
        x2 = inputs[index]

        x1 = self.convlayer(x1)
        x2 = self.convlayer(x2)

        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 12
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.convlayer = model.features
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        num_ftrs = model.classifier.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze()
            outputs = self.fc(x)
            return outputs

        alpha = self.alpha
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(len(labels))

        x1 = inputs
        x2 = inputs[index]

        x1 = self.convlayer(x1)
        x2 = self.convlayer(x2)

        x1 = self.AdaptiveAvgPool2d(x1)
        x2 = self.AdaptiveAvgPool2d(x2)

        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 13
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelEN, self).__init__()
        self.model = model
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            outputs = self.model(inputs)
            return outputs

        print("ここにきてはいけない")
        sys.exit()




## === cell 14
pass




## === cell 15
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img = cv2.imread(f"{TEST_PATH}/{image_id}")
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {TEST_PATH}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 16
def predict_model(basename, net, dataloader):
    model_start_time = time.time()

    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    probability = []

    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")
        for inputs, image_ids in progress:
            inputs = inputs.to(device)
            try:
                outputs = net(inputs, False, "test")
            except TypeError:
                outputs = net(inputs)
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 17
def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model", "net", "model_state_dict"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    return ckpt_obj


def _strip_prefix_from_state_dict(state_dict, prefixes=("module.", "model.", "net.")):
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    for p in prefixes:
        if len(keys) > 0 and all((k.startswith(p) for k in keys)):
            return {k[len(p) :]: v for k, v in state_dict.items()}
    return state_dict


def _remap_efficientnet_head_keys_to_torchvision(sd: dict) -> dict:
    """
    NOTE (score-critical): Many EB7 checkpoints from this competition are saved from
    efficientnet_pytorch with head keys like '_fc.weight/_fc.bias'. Torchvision EB7 uses
    'classifier.1.weight/.bias'. Remapping these prevents loading a random head.
    """
    if not isinstance(sd, dict):
        return sd
    out = OrderedDict()
    for k, v in sd.items():
        nk = k
        if nk.startswith("_fc."):
            nk = nk.replace("_fc.", "classifier.1.")
        out[nk] = v
    return out


def load_checkpoint_robust(
    net,
    pretrained_model_path,
    prefer_strict=True,
    efficientnet_remap_to_torchvision=False,
):
    ckpt = torch.load(pretrained_model_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)
    sd = _strip_prefix_from_state_dict(sd)

    if efficientnet_remap_to_torchvision:
        sd = _remap_efficientnet_head_keys_to_torchvision(sd)

    if prefer_strict:
        try:
            net.load_state_dict(sd, strict=True)
            print("Loaded checkpoint with strict=True")
            return True
        except Exception as e:
            print(f"STRICT load failed: {type(e).__name__}: {e}")

    try:
        net.load_state_dict(sd, strict=True)
        print("Loaded checkpoint with strict=True")
        return True
    except Exception:
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print("Loaded checkpoint with strict=False (fallback)")
        print(f"Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}")
        if len(missing) > 0:
            print("Sample missing keys:", missing[:10])
        if len(unexpected) > 0:
            print("Sample unexpected keys:", unexpected[:10])
        if len(missing) > 200:
            print(
                "WARNING: Many missing keys; checkpoint/model mismatch likely. Score may be near-random."
            )
        return False




## === cell 18
def build_efficientnet_b7_torchvision_with_fc_alias(num_classes: int = 5) -> nn.Module:
    m = models.efficientnet_b7(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    m._fc = m.classifier[1]
    return m


if len(pretrained_models) == 0:
    print(
        "WARNING: No pretrained *.pth models found. Falling back to sample_submission.csv."
    )
    df_sub = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
    df_sub.to_csv("submission.csv", index=False)
else:
    probability = []
    start_time = time.time()

    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]
        criterion = nn.CrossEntropyLoss()

        if "resnet18" in basename:
            MODEL_NAME = "resnet18"
            net = models.resnet18(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 64
            strict_ok = load_checkpoint_robust(
                net, pretrained_model, prefer_strict=False
            )

        elif "resnet50" in basename:
            MODEL_NAME = "resnet50"
            net = models.resnet50(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 32
            strict_ok = load_checkpoint_robust(
                net, pretrained_model, prefer_strict=False
            )

        elif "resnet152" in basename:
            MODEL_NAME = "resnet152"
            net = models.resnet152(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 16
            strict_ok = load_checkpoint_robust(
                net, pretrained_model, prefer_strict=False
            )

        elif "resnext101" in basename:
            MODEL_NAME = "resnext101"
            net = models.resnext101_32x8d(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 12
            strict_ok = load_checkpoint_robust(
                net, pretrained_model, prefer_strict=False
            )

        elif "densenet201" in basename:
            MODEL_NAME = "densenet201"
            net = models.densenet201(weights=None)
            net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
            BATCH_SIZE = 12
            strict_ok = load_checkpoint_robust(
                net, pretrained_model, prefer_strict=False
            )

        elif "efficientnet-b7" in basename:
            MODEL_NAME = "efficientnet-b7"
            base = build_efficientnet_b7_torchvision_with_fc_alias(
                num_classes=num_classes
            )
            net = FinalLayerMixupModelEN(base, criterion, num_classes, False)
            BATCH_SIZE = 10
            strict_ok = load_checkpoint_robust(
                net,
                pretrained_model,
                prefer_strict=True,
                efficientnet_remap_to_torchvision=True,
            )
            if not strict_ok:
                print(
                    "WARNING: EfficientNet checkpoint load mismatch; skipping this model from ensemble."
                )
                del net
                torch.cuda.empty_cache()
                continue

        else:
            print(f"{basename} is not supported.")
            sys.exit()

        print(f"{basename}: {MODEL_NAME}")

        for param in net.parameters():
            param.requires_grad = False

        tta_probas = []
        for tid, transform_ in enumerate(transform["test"]):
            print(f"transform loop={tid}")
            dataset = {
                "test": TestDataset(df_test, transform=transform_),
            }
            dataloader = {
                "test": torch.utils.data.DataLoader(
                    dataset["test"],
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=min(8, os.cpu_count() or 1),
                    pin_memory=True,
                ),
            }

            proba = predict_model(basename, net, dataloader)
            tta_probas.append(proba)

        probability.append(np.mean(np.stack(tta_probas, axis=0), axis=0))

        del net
        torch.cuda.empty_cache()

    if len(probability) == 0:
        print(
            "WARNING: No models successfully loaded; falling back to sample_submission.csv."
        )
        df_sub = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
        df_sub.to_csv("submission.csv", index=False)
    else:
        prob_arr = np.stack(probability, axis=0)  # [n_models, n_test, n_classes]
        df_test["mean"] = prob_arr.mean(axis=0).argmax(axis=1).astype(int)
        print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 19
if "submission.csv" not in os.listdir("."):
    if len(df_test) == 2 and df_test.loc[0, "image_id"] == df_test.loc[1, "image_id"]:
        df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
    else:
        df_test["label"] = df_test["mean"].astype(int)



## === cell 20
if "submission.csv" not in os.listdir("."):
    display_cols = ["image_id", "label"] if "label" in df_test.columns else ["image_id"]
    print(df_test[display_cols].head())



## === cell 21
if "submission.csv" not in os.listdir("."):
    sub = df_test[["image_id", "label"]].copy()
    sub = sub.merge(
        pd.read_csv(f"{BASE_DIR}/sample_submission.csv")[["image_id"]],
        on="image_id",
        how="right",
        sort=False,
    )
    sub["label"] = sub["label"].fillna(0).astype(int)
    sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(pd.read_csv("submission.csv").head())
print(f"submission.csv rows: {pd.read_csv('submission.csv').shape[0]}")
