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

0.8934723481414325

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the missing `efficientnet_pytorch` dependency by switching to torchvision’s built-in EfficientNet-B7 with a compatible final layer, keeping the ensemble logic the same. I also make the dataset path detection robust for this environment (using `/kaggle/input/...`) so `test_files` and `df_test` are always defined and the pipeline can run end-to-end. Albumentations v2 changed the `RandomResizedCrop` API, so I update those calls to use `size=(H,W)` to stop the validation error without changing the intent of the augmentations. Finally, I make the probability aggregation robust (avoid empty list/shape issues) and ensure we always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.05531) has done: 'Your current 0.11584 score is consistent with a submission/label alignment problem rather than a weak model: in Interactive mode you are predicting on `train_images` (first 32 files) and then writing that as the submission, which be catastrophically wrong for the real test set. I make the path/file-list logic always build `df_test` directly from `sample_submission.csv` (the authoritative test image_id order), and force `TEST_PATH` to `test_images` whenever it exists, so predictions align row-by-row with Kaggle’s evaluation. I also remove the “duplicate row when len(df_test)==1” workaround (it can silently break alignment) and ensure outputs preserve the same ordering as `sample_submission.csv`. These are minimal changes that preserve your ensemble/model logic but should move the score sharply upward toward the target.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is far below the target (0.89347), and the most likely cause is that the loaded checkpoints are not being applied correctly for the wrapped models (ResNet/DenseNet), so predictions are effectively near-random. I minimally fix checkpoint loading to handle common formats (`state_dict`, `model_state_dict`, `module.` prefixes) and, crucially, load weights into the correct submodule (`net.model` / `net.convlayer`+`net.fc`) depending on wrapper type, without changing the model architectures or inference logic. I also keep the test ordering strictly aligned to `sample_submission.csv` (already mostly correct) and make sure the submission is always written with the right columns and row count. These changes should move accuracy sharply upward toward your target by ensuring the ensemble is actually using the trained weights.'
- What this solution (achieved 0.05531) has done: 'Your very low score is most consistent with “weights not actually being applied” to the inference graph, even though the script loads a checkpoint file without crashing. I make the checkpoint loader map common training-time key patterns into your wrapper modules (`convlayer.*`, `fc.*`, and the EfficientNet `model.*` nesting), and I fail-fast if a checkpoint loads essentially nothing (to avoid silently producing near-random predictions). I also fix one architecture mismatch for ResNet wrappers (your wrapper uses `model.fc` but you never replaced `net.fc` before wrapping), and make the feature squeeze safe for batch size 1 to prevent shape-dependent wrong logits. These are minimal changes that keep the ensemble/TTA logic and architectures intact, but should move accuracy sharply upward toward your target.'
- What this solution (achieved 0.05531) has done: 'Your score is far below target, so we should make the smallest changes that plausibly recover the intended pretrained performance without altering the overall ensemble/TTA logic. The main likely issue is that the wrapper networks’ feature heads don’t match what was trained: your ResNet wrapper ignores `model.fc` entirely and instead uses a new `self.fc`, so checkpoints that contain `fc.*` won’t map unless we explicitly copy weights from the backbone or load into the wrapper correctly. I (1) fix the ResNet wrapper to use the backbone’s own `fc` (same computation, but now checkpoint keys align), (2) strengthen checkpoint key-mapping for DenseNet/EfficientNet vs wrapper prefixes while still using `strict=False`, and (3) make preprocessing match standard ImageNet inference by using a resize+center-crop (your current pure CenterCrop can heavily mis-crop images and crater accuracy). These are minimal, score-relevant changes and still write a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'The current score (0.05531) is so far below the target that the most likely remaining issue is still “weights not actually being used” for some of the wrappers, especially DenseNet (your wrapper creates a new `self.fc` but the checkpoint likely contains `classifier.*` from the original DenseNet, so the trained head isn’t loaded). I minimally adjust the DenseNet wrapper to reuse `model.classifier` (resized to 5 classes) instead of creating a new Linear, so checkpoints load cleanly without changing the backbone/forward logic. I also harden the checkpoint loader to map `classifier.* -> fc.*` and to explicitly accept `net.fc.*` and `net.model.*` prefix variants, avoiding silent partial loads. These are small, score-critical fixes that preserve your ensemble/TTA/inference pipeline and should move accuracy sharply upward toward the target.'
- What this solution (achieved 0.05531) has done: 'Your current score is far below the target, so the most likely remaining issue is still that the test-time preprocessing doesn’t match what these public checkpoints were trained with, causing near-random predictions even when weights load. I make the smallest score-relevant change by switching the “test” transforms to the standard Cassava baseline preprocessing (Resize to SIZE then Normalize), and keep only light deterministic TTA (original + horizontal flip) to avoid stochastic `RandomResizedCrop` at inference that can severely destabilize accuracy. I also ensure the probability aggregation groups correctly per-model across TTAs (so the ensemble mean is computed over models, not mixing dimensions), without changing the ensemble logic itself. Everything else (model wrappers, checkpoint loading, dataloader order, and submission alignment to `sample_submission.csv`) stays the same.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, so the most likely remaining issue is still that inference-time preprocessing doesn’t match what these public checkpoints were trained on (causing near-random predictions even when weights load). I make a minimal, score-relevant adjustment by switching the test pipeline from `Resize(512x512)` to the common Cassava baseline `Resize(SIZE,SIZE) + CenterCrop(SIZE,SIZE) + Normalize`, preserving the rest of your ensemble/TTA logic and model wrappers. I also fix the EfficientNet-B7 classifier replacement to match torchvision’s current structure (it uses `classifier[1]`), which can otherwise prevent correct head sizing/weight loading. Finally, I keep strict submission alignment to `sample_submission.csv` and ensure the output `submission.csv` is always valid.'
- What this solution (achieved 0.05531) has done: 'I make the smallest set of inference-time fixes that plausibly explain a 0.055 score (near-random) without changing your overall ensemble/TTA design. The key issue is that your wrappers currently bypass important backbone behavior: DenseNet should apply `relu` before pooling, and EfficientNet-B7 in torchvision needs its own forward (features + avgpool + classifier) rather than `self.model(inputs)` after classifier surgery; both can crater accuracy even if checkpoints load. I also slightly harden the checkpoint-load “fail-fast” so it doesn’t falsely pass when the backbone loads but the head doesn’t (or vice versa), while still using `strict=False` and the same model list and aggregation. Everything else (paths, ordering via `sample_submission.csv`, TTA=2, ensemble mean, submission writing) stays the same.'
- What this solution (achieved 0.05531) has done: 'I make two minimal, score-relevant fixes that commonly cause “near-random” submissions in this exact Cassava setup: (1) ensure we actually find the pretrained checkpoints by searching both `../input/...` and `/kaggle/input/...` (your current glob likely finds 0 models in this environment, forcing label=0 and yielding ~0.055), and (2) ensure EfficientNet-B7 checkpoint loading targets the wrapper itself (so the replaced classifier weights load), not only `net.model` (which can silently skip the head and tank accuracy). These changes preserve your ensemble/TTA/inference logic and keep the submission strictly aligned to `sample_submission.csv`. Everything still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'I make the smallest changes that address the most likely reason for a near-random 0.055 score: inference preprocessing mismatch with how the public checkpoints were trained. Specifically, I switch the *test-time* preprocessing from `Resize(576)->CenterCrop(512)` to the common Cassava baseline `Resize(512)->CenterCrop(512)` (keeping normalization identical), because the larger resize can shift scale/zoom and degrade accuracy badly. I also ensure EfficientNet runs in true inference mode by calling `self.model.eval()` inside the wrapper (no architecture change, just guarantees dropout is disabled even if something upstream toggles modes). Everything else—model wrappers, ensemble averaging, TTA=2, checkpoint loading, and submission alignment to `sample_submission.csv`—stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is near-random, and the most likely minimal root-cause left is that `pretrained_models` includes non-weight files (e.g., training logs/optim states) or mismatched checkpoints that “load” but don’t correspond to the selected architecture, producing garbage predictions. I (1) filter the checkpoint list to only plausible model-weight `.pth` files and (2) add a tight architecture-name guard so each checkpoint is only loaded into the matching model wrapper, preventing accidental cross-loading. I also make the ensemble averaging more robust by skipping any model whose predicted probability array shape doesn’t exactly match `(len(df_test), num_classes)` (alignment safety), while keeping the same models, transforms, and mean-ensemble semantics. These are minimal, score-relevant safety checks that should move accuracy sharply upward toward your target by ensuring only valid, correctly-loaded models contribute.'
- What this solution (achieved 0.05531) has done: 'We make the smallest changes that plausibly explain a near-random 0.055 accuracy: your inference wrapper is calling `net(inputs, False, "test")`, which passes `labels=False` into code that expects a tensor (and can silently break behavior), and your ResNet wrapper currently rebuilds the backbone via `children()[:-1]`, which is brittle and can mismatch checkpoint keys and forward semantics. I (1) make inference call `net(inputs, None, "test")` and adjust wrappers to ignore labels in test mode, and (2) modify the ResNet wrapper to use the backbone’s own `forward` path up to `avgpool` and then `fc` (same architecture, more faithful to torchvision ResNet behavior and checkpoint compatibility). Everything else—models used, ensemble averaging, TTA, transforms, paths, and submission alignment—stays the same, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import glob

ckpt_patterns = [
    "../input/densenet201-seed60/*.pth",
    "../input/eb7m-seed70/*.pth",
    "/kaggle/input/densenet201-seed60/*.pth",
    "/kaggle/input/eb7m-seed70/*.pth",
]

pretrained_models = []
for pat in ckpt_patterns:
    pretrained_models += glob.glob(pat)


def _is_plausible_ckpt(p):
    bn = p.lower()
    supported = ("resnet", "resnext", "densenet201", "efficientnet-b7", "eb7", "b7")
    bad = ("optimizer", "sched", "scheduler", "history", "log", "fold", "metrics")
    return (
        bn.endswith(".pth")
        and any(s in bn for s in supported)
        and not any(b in bn for b in bad)
    )


pretrained_models = [p for p in pretrained_models if _is_plausible_ckpt(p)]

print(f"{len(pretrained_models)} models found after filtering.")
print("\n".join(np.sort(pretrained_models)))



## === cell 1
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

import torchvision
from torchvision import models

import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

import os
from pathlib import Path
import random
import json
import time
import pickle

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



## === cell 2
from torchvision.models import efficientnet_b7



## === cell 3
SIZE = 512  # image size
num_classes = 5



## === cell 4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 5
kaggle_input_base = "/kaggle/input/cassava-leaf-disease-classification"
if Path(kaggle_input_base).exists():
    BASE_DIR = kaggle_input_base
else:
    BASE_DIR = (
        "../input/cassava-leaf-disease-classification"
        if Path("../input/cassava-leaf-disease-classification").exists()
        else "data/cassava-leaf-disease-classification"
    )

sample_sub_path = f"{BASE_DIR}/sample_submission.csv"
if not Path(sample_sub_path).exists():
    raise FileNotFoundError(f"sample_submission.csv not found at: {sample_sub_path}")
df_test = pd.read_csv(sample_sub_path)[["image_id", "label"]]

candidate_test = f"{BASE_DIR}/test_images"
candidate_train = f"{BASE_DIR}/train_images"
if Path(candidate_test).exists():
    TEST_PATH = candidate_test
else:
    TEST_PATH = candidate_train  # fallback for local debugging only

test_files = df_test["image_id"].tolist()

print(f"BASE_DIR={BASE_DIR}")
print(f"TEST_PATH={TEST_PATH}")
print(f"Number of test images (from sample_submission): {len(test_files)}")



## === cell 6
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

RESIZE_TO = SIZE  # keep deterministic resize scale

transform = {
    "test": [
        Compose(
            [
                A.Resize(RESIZE_TO, RESIZE_TO, interpolation=cv2.INTER_LINEAR),
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.Resize(RESIZE_TO, RESIZE_TO, interpolation=cv2.INTER_LINEAR),
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}




## === cell 7
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        """
        Minimal accuracy fix:
        Use the backbone's native ResNet forward path (conv1..layer4->avgpool->fc) rather than
        rebuilding a convlayer via children() slicing, which can subtly mismatch behavior and checkpoint keys.
        This keeps the same architecture and head, but makes inference faithful to torchvision ResNet.
        """
        super(FinalLayerMixupModel, self).__init__()
        self.model = model  # model.fc already sized to num_classes by caller
        self.fc = model.fc
        self.criterion = criterion
        self.alpha = alpha

    def _forward_features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self._forward_features(inputs)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self._forward_features(inputs)
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

        x1 = self._forward_features(x1).view(x1.size(0), -1, 1, 1)
        x2 = self._forward_features(x2).view(x2.size(0), -1, 1, 1)

        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.flatten(1)
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 8
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        """
        DenseNet forward applies relu before global pooling; omitting it can seriously harm predictions.
        Keep the same "features -> pool -> classifier" logic, but match torchvision DenseNet behavior.
        """
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.convlayer = model.features
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        self.fc = model.classifier

        self.criterion = criterion
        self.alpha = alpha

    def _forward_features(self, x):
        x = self.convlayer(x)
        x = F.relu(x, inplace=True)
        x = self.AdaptiveAvgPool2d(x)
        x = x.flatten(1)
        return x

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self._forward_features(inputs)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self._forward_features(inputs)
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

        x1 = self._forward_features(x1).view(x1.size(0), -1, 1, 1)
        x2 = self._forward_features(x2).view(x2.size(0), -1, 1, 1)

        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.flatten(1)
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 9
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        """
        Call EfficientNet's correct inference path (features->avgpool->classifier) explicitly.
        """
        super(FinalLayerMixupModelEN, self).__init__()

        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            if len(model.classifier) >= 2 and isinstance(
                model.classifier[1], nn.Linear
            ):
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(in_features, num_classes)
            elif isinstance(model.classifier[-1], nn.Linear):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                raise ValueError(
                    "EfficientNet classifier does not contain a Linear layer."
                )
        else:
            raise ValueError(
                "Unexpected EfficientNet model structure; cannot locate classifier layer."
            )

        self.model = model
        self.criterion = criterion
        self.model.eval()

    def _forward_logits(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x

    def forward(self, inputs, labels, phase):
        if phase == "val":
            outputs = self._forward_logits(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            outputs = self._forward_logits(inputs)
            return outputs

        print("ここにきてはいけない")
        raise SystemExit(1)




## === cell 10
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




## === cell 11
def _extract_state_dict(ckpt):
    """
    - Accept raw state_dict or dict wrappers (state_dict/model_state_dict).
    - Strip common 'module.' prefix from DataParallel checkpoints.
    """
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    if not isinstance(sd, dict):
        raise ValueError("Checkpoint format not understood (no state dict found).")

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _maybe_strip_prefix(state_dict, prefix):
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return None
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _load_checkpoint_into_net(net, model_name, state_dict):
    sd = state_dict

    for pref in ("net.", "model."):
        stripped = _maybe_strip_prefix(sd, pref)
        if stripped is not None:
            sd = stripped

    if model_name == "efficientnet-b7":
        stripped2 = _maybe_strip_prefix(sd, "model.")
        if stripped2 is not None:
            sd = stripped2
        missing, unexpected = net.load_state_dict(sd, strict=False)
        return missing, unexpected

    if any(k.startswith("convlayer.") or k.startswith("fc.") for k in sd.keys()):
        missing, unexpected = net.load_state_dict(sd, strict=False)
        return missing, unexpected

    if model_name == "densenet201":
        if any(k.startswith("classifier.") for k in sd.keys()) and not any(
            k.startswith("fc.") for k in sd.keys()
        ):
            remap = {}
            for k, v in sd.items():
                if k.startswith("classifier."):
                    remap["fc." + k[len("classifier.") :]] = v
                else:
                    remap[k] = v
            sd = remap

    if hasattr(net, "convlayer") and hasattr(net, "fc"):
        sd_backbone = {}
        sd_fc = {}
        for k, v in sd.items():
            if k in ["fc.weight", "fc.bias"]:
                sd_fc[k.replace("fc.", "")] = v
            else:
                sd_backbone[k] = v
        try:
            missing_b, unexpected_b = net.convlayer.load_state_dict(
                sd_backbone, strict=False
            )
            missing_f, unexpected_f = (
                net.fc.load_state_dict(sd_fc, strict=False) if len(sd_fc) else ([], [])
            )
            return (missing_b + missing_f), (unexpected_b + unexpected_f)
        except Exception:
            missing, unexpected = net.load_state_dict(sd, strict=False)
            return missing, unexpected

    missing, unexpected = net.load_state_dict(sd, strict=False)
    return missing, unexpected


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
            outputs = net(inputs, None, "test")
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    if len(probability) == 0:
        return np.zeros((0, num_classes), dtype=np.float32)
    return np.concatenate(probability, axis=0)




## === cell 12
probability = []
start_time = time.time()


def _infer_model_name_from_basename(basename: str):
    b = basename.lower()
    if "resnet18" in b:
        return "resnet18"
    if "resnet50" in b:
        return "resnet50"
    if "resnet152" in b:
        return "resnet152"
    if "resnext101" in b:
        return "resnext101"
    if "densenet201" in b:
        return "densenet201"
    if "efficientnet-b7" in b or "eb7" in b or "b7" in b:
        return "efficientnet-b7"
    return None


if len(pretrained_models) == 0:
    print(
        "WARNING: No pretrained models found. Falling back to sample_submission labels=0."
    )
    df_test["label"] = 0
    df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
else:
    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]
        inferred = _infer_model_name_from_basename(basename)
        if inferred is None:
            print(f"Skipping unsupported checkpoint name: {basename}")
            continue

        criterion = nn.CrossEntropyLoss()

        if inferred == "resnet18":
            MODEL_NAME = "resnet18"
            net_backbone = models.resnet18(weights=None)
            net_backbone.fc = nn.Linear(net_backbone.fc.in_features, num_classes)
            net = FinalLayerMixupModel(net_backbone, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif inferred == "resnet50":
            MODEL_NAME = "resnet50"
            net_backbone = models.resnet50(weights=None)
            net_backbone.fc = nn.Linear(net_backbone.fc.in_features, num_classes)
            net = FinalLayerMixupModel(net_backbone, criterion, num_classes, False)
            BATCH_SIZE = 32
        elif inferred == "resnet152":
            MODEL_NAME = "resnet152"
            net_backbone = models.resnet152(weights=None)
            net_backbone.fc = nn.Linear(net_backbone.fc.in_features, num_classes)
            net = FinalLayerMixupModel(net_backbone, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif inferred == "resnext101":
            MODEL_NAME = "resnext101"
            net_backbone = models.resnext101_32x8d(weights=None)
            net_backbone.fc = nn.Linear(net_backbone.fc.in_features, num_classes)
            net = FinalLayerMixupModel(net_backbone, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif inferred == "densenet201":
            MODEL_NAME = "densenet201"
            net_backbone = models.densenet201(weights=None)
            net = FinalLayerMixupModelDenseNet(
                net_backbone, criterion, num_classes, False
            )
            BATCH_SIZE = 12
        elif inferred == "efficientnet-b7":
            MODEL_NAME = "efficientnet-b7"
            net_backbone = efficientnet_b7(weights=None)
            net = FinalLayerMixupModelEN(net_backbone, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(f"{basename} is not supported.")
            raise SystemExit(1)

        print(f"{basename}: {MODEL_NAME}")

        ckpt = torch.load(pretrained_model, map_location="cpu")
        state_dict = _extract_state_dict(ckpt)
        missing, unexpected = _load_checkpoint_into_net(net, MODEL_NAME, state_dict)
        print(f"Loaded ckpt. missing={len(missing)} unexpected={len(unexpected)}")

        model_keys = len(net.state_dict().keys())
        loaded_keys_est = model_keys - len(missing)
        if model_keys > 0 and loaded_keys_est <= max(1, int(0.05 * model_keys)):
            raise RuntimeError(
                f"Checkpoint appears not to load into model (loaded too low). "
                f"basename={basename} model_keys={model_keys} loaded~={loaded_keys_est} "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

        for param in net.parameters():
            param.requires_grad = False

        tta_probas = []
        for tid, transform_ in enumerate(transform["test"]):
            print(f"transform loop={tid}")
            dataset = {"test": TestDataset(df_test, transform=transform_)}
            dataloader = {
                "test": torch.utils.data.DataLoader(
                    dataset["test"],
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=torch.cuda.is_available(),
                ),
            }

            proba = predict_model(basename, net, dataloader)
            tta_probas.append(proba)

        if len(tta_probas) == 0:
            model_proba = np.zeros((len(df_test), num_classes), dtype=np.float32)
        else:
            model_proba = np.mean(np.stack(tta_probas, axis=0), axis=0)

        if model_proba.shape != (len(df_test), num_classes):
            print(
                f"WARNING: Skipping {basename} due to bad proba shape {model_proba.shape} "
                f"(expected {(len(df_test), num_classes)})."
            )
        else:
            probability.append(model_proba)

        del net
        torch.cuda.empty_cache()

    if len(probability) == 0:
        print(
            "WARNING: No valid model probabilities collected. Falling back to label=0."
        )
        df_test["label"] = 0
    else:
        prob_arr = np.stack(probability, axis=0)
        df_test["mean"] = prob_arr.mean(axis=0).argmax(axis=1)

    print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 13
if "mean" in df_test.columns:
    df_test["label"] = df_test["mean"]
else:
    if "label" not in df_test.columns:
        df_test["label"] = 0



## === cell 14
df_test.head()



## === cell 15
sub = pd.read_csv(sample_sub_path)[["image_id"]].merge(
    df_test[["image_id", "label"]], on="image_id", how="left", validate="one_to_one"
)
if sub["label"].isna().any():
    sub["label"] = sub["label"].fillna(0).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
