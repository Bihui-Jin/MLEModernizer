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

0.8765488062858869

# 6. Current score

0.77691

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I remove the offline `pip install efficientnet_pytorch` dependency and switch to the equivalent `torchvision` EfficientNet backbone so imports work in this environment. I also fix inference-time albumentations errors by replacing deprecated/changed transforms (e.g., `RandomResizedCrop` now requires `size`, and `Cutout` is not available in albumentations v2) while keeping the same inference-TTA intent. Finally, I make the code run on both CPU and GPU (no hard CUDA assert) and ensure the submission rows exactly match `sample_submission.csv` ordering/length so Kaggle accepts the file.'
- What this solution (achieved 0.08483) has done: 'I fix the immediate runtime failure by making checkpoint loading robust to missing files and by pointing `checkpoint_path` to the actual dataset directory structure (and searching common locations) instead of a non-existent `../input/focalloss`. To move accuracy toward your target (your current 0.11584 is far below), I also load ImageNet pretrained EfficientNet weights when no custom checkpoint is found, which is a minimal change that keeps the same model architecture and inference flow but yields sane predictions. I additionally make the inference transforms deterministic/valid for test-time inference by replacing the stochastic `RandomResizedCrop` with a standard `Resize` (still keeping flips/transpose TTA as intended) to avoid harmful random crops at inference. Finally, I ensure the submission is written with the correct columns/order and `.csv` suffix.'
- What this solution (achieved 0.09006) has done: 'Your score is far below the target, so the smallest safe way to move accuracy up (without changing the model/loop) is to fix the normalization mismatch: torchvision EfficientNet pretrained weights expect inputs normalized with its own mean/std (not generic ImageNet values). I switch the `Normalize` parameters to `EfficientNet_B4_Weights`’ built-in transforms (mean/std), and keep your existing TTA/epoch-averaging inference exactly the same. I also set `cudnn.benchmark=False` when `deterministic=True` to avoid nondeterministic kernels that can slightly destabilize predictions run-to-run (this shouldn’t hurt accuracy, but improves reproducibility). All paths and submission formatting stay unchanged.'
- What this solution (achieved 0.08483) has done: 'Your current score (0.09006) is far below the target (0.87655), so we should safely increase accuracy with minimal changes that preserve your inference-only pipeline. The biggest issue is that your “TTA across epochs” is actually re-running *random* flips/transpose each epoch and then averaging logits, which can severely degrade predictions when there’s no trained cassava checkpoint; I make inference-time transforms deterministic by setting flip/transpose probabilities to 0 while keeping the same transform list/flow. I also align preprocessing with the torchvision EfficientNet weights end-to-end by using the weights’ recommended resize/crop size (380 for B4) instead of 384, and ensure the model uses the weights’ default inference transform statistics (you already fixed mean/std). These are small, semantics-preserving edits intended to move the score upward toward the target without changing the model architecture or inference loop structure.'
- What this solution (achieved 0.07997) has done: 'Your current score is far below the target, so the most likely issue is not the inference loop but that the model you submit is effectively “untrained for cassava” (ImageNet head / missing cassava checkpoint), producing near-random labels. With minimal changes and preserving your EfficientNet-B4 + averaging loop, I (1) make checkpoint discovery also look for common Kaggle Cassava pretrained filenames and (2) load checkpoints more robustly (including Lightning-style keys) so you actually use a cassava-trained head when present. I also align preprocessing to the EfficientNet-B4 weights’ *actual* recommended resize/crop pipeline by switching from a plain square resize to `SmallestMaxSize + CenterCrop` (deterministic, inference-safe), which typically gives a noticeable accuracy lift without changing model semantics. The submission writing/order stays exactly as required.'
- What this solution (achieved 0.07997) has done: 'Your score is far below the target, and the current code is almost certainly submitting predictions from an ImageNet-pretrained backbone with a randomly initialized 5-class head (because no Cassava checkpoint is found/loaded), which yields near-random accuracy. The smallest change that materially improves accuracy (without changing the model/loop/augment intent) is to correctly load the official Cassava competition checkpoint if it exists in the dataset folder, including handling common Kaggle notebook `.pth` files and Lightning/EMA key prefixes. I extend checkpoint discovery to include known Cassava pretrained filenames and also search for any `.pth/.pt` under `../input/cassava-leaf-disease-classification` that contains “cassava/effnet/b4/best” in the name, then load it robustly. Everything else (EfficientNet-B4, deterministic center-crop preprocessing, averaging across epochs, submission formatting) stays the same.'
- What this solution (achieved 0.07063) has done: 'Your current score (0.07997) is far below the target (0.87655), which strongly suggests you are not actually using a Cassava-trained checkpoint and are effectively predicting with an ImageNet model + randomly initialized 5-class head. With minimal changes and preserving your EfficientNet-B4 + inference-averaging core logic, I (1) extend checkpoint discovery to search the full `../input` tree for plausible `.pth/.pt` files, (2) load checkpoints more robustly by handling common nesting (`state_dict`, `model`, etc.) and key prefixes, and (3) add a safe fallback that adapts 1000-class ImageNet classifier weights into a 5-class head when no Cassava checkpoint exists (better than random). These changes are directly aimed at moving accuracy upward toward the target while keeping the same model/inference semantics and still writing a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current accuracy is far below the target, and the most likely cause (given this is inference-only) is that you’re still not loading a Cassava-trained checkpoint and are effectively predicting with an ImageNet model + a weakly initialized 5-class head. To move the score upward with minimal changes and identical inference semantics, I add a robust “train.csv label-prior bias calibration” fallback that only activates when no Cassava checkpoint is found: it adds a constant logit bias based on Cassava class frequencies, which typically boosts accuracy versus near-random heads without changing the model architecture or loop. I also fix the preprocessing mismatch by switching `Normalize` to the official EfficientNet-B4 weights’ mean/std (already computed) and ensure the input scaling matches what torchvision weights expect by explicitly dividing by 255 before normalization (albumentations Normalize does this, but this guards against version differences). Finally, I keep the same deterministic center-crop pipeline and submission alignment to `sample_submission.csv`.'
- What this solution (achieved 0.70852) has done: 'Your current 0.61099 suggests you’re still effectively using an ImageNet model with a weak 5-class head (no real cassava checkpoint), and the class-prior bias helped but not enough. With minimal changes that preserve your model/inference loop, I (1) align preprocessing exactly to torchvision EfficientNet-B4 weights by using the weights’ recommended resize+center-crop and normalization in one place, and (2) strengthen the “no-checkpoint fallback” by doing a tiny, deterministic linear-probe training of the existing 5-class head on `train.csv` (backbone frozen) for a couple epochs—this keeps the same architecture/loss (CrossEntropy) and avoids changing your overall approach, but should move accuracy substantially toward your 0.8765 target. I keep your test inference averaging loop intact and still write `submission.csv` in the sample submission order. All changes are constrained to the fallback path (only when no cassava checkpoint is found), so if a real checkpoint exists it be used as before.'
- What this solution (achieved 0.76046) has done: 'Your current accuracy (0.7085) is well below the target (0.8765), so we should increase performance with minimal, semantics-preserving fixes. The biggest issue in your fallback head-training is that it trains with inference-only transforms (center-crop + no augmentation), which can underfit and limit gains; I add a *separate* deterministic/standard train augmentation pipeline used only in the fallback head-training path, while keeping your test/inference transforms unchanged. I also make the fallback head training class-balanced via a `WeightedRandomSampler` (still CrossEntropy, same head-only training loop) to better match the metric (accuracy) on an imbalanced dataset. Finally, I keep runtime under 600s by limiting the fallback head training to a capped number of batches per epoch (deterministic cap), without changing your main inference averaging logic or submission formatting.'
- What this solution (achieved 0.76682) has done: 'We keep your EfficientNet-B4 + head-only fallback training + averaging inference exactly as-is, and only make small changes that are directly tied to improving accuracy toward 0.8765. The main gain (with minimal risk) is to match the official torchvision EfficientNet-B4 inference preprocessing: `Resize(384) + CenterCrop(380)` and weights’ mean/std; right now your center-crop after square-resize is redundant and slightly mismatched. In the fallback head-training path, we also align the *training* preprocessing to the same geometry (resize to 384 then random-crop to 380), which tends to improve transfer without changing the training loop/optimizer/loss. Finally, we add a deterministic train/val split to optionally calibrate the class-prior logit bias strength (only in the no-checkpoint fallback), which is a tiny, metric-aligned calibration step that usually boosts accuracy a bit without touching the core model logic.'
- What this solution (achieved 0.77691) has done: 'Your current score (0.76682) is below the target (0.87655), so we should make a small, low-risk improvement that keeps your EfficientNet-B4 + head-only fallback training + inference-averaging intact. The biggest gap in the fallback training is that the backbone remains in `train()` mode, so BatchNorm stats update on cassava images even though the backbone is frozen; this often hurts transfer and reduces accuracy. I keep the same training loop/optimizer/loss, but switch the backbone to `eval()` and only the classifier head to `train()` during fallback head training, plus enable `torch.inference_mode()` for inference to reduce numerical/overhead variance without changing semantics. No paths, model architecture, or submission formatting are changed.'

# 9. Code solution

## === cell 0
import os
import gc
import time
import random
import numpy as np
import pandas as pd
import cv2

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from tqdm import tqdm
from torchvision import models



## === cell 1
image_size = 380



## === cell 2
config = dict(
    seed=22,
    experiment_name="focal_loss",
    test_location="../input/cassava-leaf-disease-classification/test_images",
    train_location="../input/cassava-leaf-disease-classification/train_images",
    checkpoint_path="../input/cassava-leaf-disease-classification",
    checkpoint="focal_loss.pt",
    model="efficientnet-b4",  # keep semantic name; we will map to torchvision efficientnet_b4
    epochs=5,  # inference averaging epochs (kept identical)
    batch_size=32,
    workers=2,
    inference_augmentations=[
        dict(name="SmallestMaxSize", params=dict(max_size=384, p=1.0)),
        dict(
            name="CenterCrop", params=dict(height=image_size, width=image_size, p=1.0)
        ),
        dict(name="HorizontalFlip", params=dict(always_apply=False, p=0.0)),
        dict(name="VerticalFlip", params=dict(always_apply=False, p=0.0)),
        dict(name="Transpose", params=dict(p=0.0)),
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
        ),
    ],
    fallback_train_head=True,
    fallback_head_epochs=2,
    fallback_head_lr=5e-3,
    fallback_head_batch_size=64,
    fallback_train_augmentations=[
        dict(name="SmallestMaxSize", params=dict(max_size=384, p=1.0)),
        dict(
            name="RandomCrop", params=dict(height=image_size, width=image_size, p=1.0)
        ),
        dict(name="HorizontalFlip", params=dict(p=0.5)),
        dict(name="VerticalFlip", params=dict(p=0.2)),
        dict(
            name="ShiftScaleRotate",
            params=dict(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
        ),
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
        ),
    ],
    fallback_use_weighted_sampler=True,
    fallback_max_batches_per_epoch=220,
    fallback_bias_calibrate=True,
    fallback_bias_val_frac=0.10,
    fallback_bias_alpha_grid=[0.0, 0.5, 1.0, 1.5, 2.0],
)



## === cell 3
sample_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test = pd.read_csv(sample_path)
test.head()




## === cell 4
def seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




## === cell 5
seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 6
_EFFNET_B4_DEFAULT_WEIGHTS = models.EfficientNet_B4_Weights.IMAGENET1K_V1
_EFFNET_B4_MEAN = list(_EFFNET_B4_DEFAULT_WEIGHTS.transforms().mean)
_EFFNET_B4_STD = list(_EFFNET_B4_DEFAULT_WEIGHTS.transforms().std)

for aug in config["inference_augmentations"]:
    if aug.get("name") == "Normalize":
        aug["params"]["mean"] = _EFFNET_B4_MEAN
        aug["params"]["std"] = _EFFNET_B4_STD
        aug["params"]["max_pixel_value"] = 255.0

for aug in config["fallback_train_augmentations"]:
    if aug.get("name") == "Normalize":
        aug["params"]["mean"] = _EFFNET_B4_MEAN
        aug["params"]["std"] = _EFFNET_B4_STD
        aug["params"]["max_pixel_value"] = 255.0


def _build_backbone_from_config(pretrained: bool = False):
    name = config["model"].lower()
    if name in ["efficientnet-b4", "efficientnet_b4"]:
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b4(weights=weights)
    elif name in ["efficientnet-b0", "efficientnet_b0"]:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    else:
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b4(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)
    return model


def _iter_checkpoint_candidates():
    common_names = [
        config["checkpoint"],
        "best.pth",
        "best.pt",
        "best_model.pth",
        "best_model.pt",
        "model_best.pth",
        "model_best.pt",
        "final.pth",
        "final.pt",
        "checkpoint.pth",
        "checkpoint.pt",
        "efficientnet_b4.pth",
        "efficientnet-b4.pth",
        "effnet_b4.pth",
        "effb4.pth",
        "cassava_effnet_b4.pth",
        "cassava_effnetb4.pth",
        "cassava_b4.pth",
        "effnetb4_cassava.pth",
        "effnet_b4_cassava.pth",
    ]

    roots = [
        config["checkpoint_path"],
        "../input",
        "../input/cassava-leaf-disease-classification",
    ]
    for r in roots:
        for n in common_names:
            yield os.path.join(r, n)

    for search_root in [config["checkpoint_path"], "../input"]:
        if os.path.isdir(search_root):
            for root, _, files in os.walk(search_root):
                for n in common_names:
                    if n in files:
                        yield os.path.join(root, n)

            for root, _, files in os.walk(search_root):
                for fn in files:
                    lfn = fn.lower()
                    if not (lfn.endswith(".pth") or lfn.endswith(".pt")):
                        continue
                    if any(
                        k in lfn
                        for k in [
                            "cassava",
                            "effnet",
                            "efficientnet",
                            "b4",
                            "best",
                            "fold",
                        ]
                    ):
                        yield os.path.join(root, fn)


def _find_checkpoint_path():
    seen = set()
    for p in _iter_checkpoint_candidates():
        if not p or p in seen:
            continue
        seen.add(p)
        if os.path.isfile(p):
            return p
    return None


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in [
            "model",
            "state_dict",
            "model_state_dict",
            "net",
            "weights",
            "ema",
            "teacher",
        ]:
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return checkpoint_obj[key]
        if all(isinstance(k, str) for k in checkpoint_obj.keys()) and any(
            "." in k for k in checkpoint_obj.keys()
        ):
            return checkpoint_obj
    return None


def _strip_known_prefixes(k: str) -> str:
    nk = k
    for pref in ["module.", "model.", "net.", "backbone.", "encoder."]:
        if nk.startswith(pref):
            nk = nk[len(pref) :]
    while nk.startswith("model."):
        nk = nk[len("model.") :]
    return nk


def _try_load_state_dict_flex(model: nn.Module, state_dict: dict):
    cleaned = {_strip_known_prefixes(k): v for k, v in state_dict.items()}

    remapped = dict(cleaned)
    if "classifier.0.weight" in cleaned and "classifier.1.weight" not in cleaned:
        remapped["classifier.1.weight"] = remapped.pop("classifier.0.weight")
    if "classifier.0.bias" in cleaned and "classifier.1.bias" not in cleaned:
        remapped["classifier.1.bias"] = remapped.pop("classifier.0.bias")

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    return missing, unexpected


def _init_head_from_imagenet_logits(model: nn.Module):
    with torch.no_grad():
        src = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        W = src.classifier[1].weight.detach().cpu()  # [1000, in_features]
        b = src.classifier[1].bias.detach().cpu()  # [1000]
        in_features = W.shape[1]

        if (
            model.classifier[1].in_features != in_features
            or model.classifier[1].out_features != 5
        ):
            return

        W5 = torch.zeros((5, in_features), dtype=W.dtype)
        b5 = torch.zeros((5,), dtype=b.dtype)
        counts = torch.zeros((5,), dtype=torch.float32)
        for i in range(1000):
            g = i % 5
            W5[g] += W[i]
            b5[g] += b[i]
            counts[g] += 1.0
        W5 = W5 / counts[:, None]
        b5 = b5 / counts

        model.classifier[1].weight.copy_(W5.to(model.classifier[1].weight.device))
        model.classifier[1].bias.copy_(b5.to(model.classifier[1].bias.device))


def _cassava_class_prior_logit_bias():
    train_csv = "../input/cassava-leaf-disease-classification/train.csv"
    if not os.path.isfile(train_csv):
        return None
    df = pd.read_csv(train_csv)
    if "label" not in df.columns:
        return None
    counts = df["label"].value_counts().sort_index()
    p = np.ones((5,), dtype=np.float64)
    for k in range(5):
        if k in counts.index:
            p[k] = float(counts.loc[k])
    p = p / p.sum()
    p = np.clip(p, 1e-6, 1.0)
    bias = np.log(p).astype(np.float32)
    return torch.tensor(bias, dtype=torch.float32, device=device)


class LogitBiasWrapper(nn.Module):
    def __init__(self, model: nn.Module, bias: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("bias", bias)

    def forward(self, x):
        return self.model(x) + self.bias


def _calibrate_bias_alpha_if_enabled(model: nn.Module, base_bias: torch.Tensor):
    """
    Change rationale (score-up toward target, minimal semantic change, fallback-only):
    The class-prior bias can help when the head is weak, but strength matters.
    We tune a single scalar alpha on a deterministic holdout split to maximize
    accuracy. This only changes an additive constant in logits (calibration),
    and only in the no-checkpoint path.
    """
    if not config.get("fallback_bias_calibrate", True):
        return 1.0

    train_csv = "../input/cassava-leaf-disease-classification/train.csv"
    if not os.path.isfile(train_csv):
        return 1.0

    df = pd.read_csv(train_csv)
    if not {"image_id", "label"}.issubset(df.columns):
        return 1.0

    rng = np.random.RandomState(config["seed"])
    idx = np.arange(len(df))
    rng.shuffle(idx)
    val_n = int(
        max(1, round(len(df) * float(config.get("fallback_bias_val_frac", 0.10))))
    )
    val_idx = idx[:val_n]
    val_df = df.iloc[val_idx].reset_index(drop=True)

    val_tfms = get_transforms()
    val_ds = CassavaTrainDataset(val_df, val_tfms, root_dir=config["train_location"])
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model.eval()
    best_alpha = 1.0
    best_acc = -1.0
    alphas = list(config.get("fallback_bias_alpha_grid", [0.0, 0.5, 1.0, 1.5, 2.0]))

    with torch.no_grad():
        for a in alphas:
            correct = 0
            total = 0
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb) + (float(a) * base_bias)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
            acc = correct / max(total, 1)
            if acc > best_acc:
                best_acc = acc
                best_alpha = float(a)

    print(f"Calibrated bias alpha={best_alpha} on holdout acc={best_acc:.4f}")
    return best_alpha




## === cell 7
def get_transforms():
    transforms = [
        getattr(A, item["name"])(**item["params"])
        for item in config["inference_augmentations"]
    ]
    return A.Compose(transforms)


def get_fallback_train_transforms():
    transforms = [
        getattr(A, item["name"])(**item["params"])
        for item in config["fallback_train_augmentations"]
    ]
    return A.Compose(transforms)




## === cell 8
class CassavaDataset(Dataset):
    def __init__(self, images, transforms, root_dir):
        self.images = images
        self.transforms = transforms
        self.root_dir = root_dir

    def __getitem__(self, n):
        img_name = self.images[n]
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

        image = self.transforms(image=image)["image"]
        image = np.moveaxis(image, -1, 0)
        image = torch.tensor(image, dtype=torch.float32)
        return image

    def __len__(self):
        return len(self.images)




## === cell 9
def get_dataloader():
    transforms = get_transforms()
    test_data = np.array(test["image_id"].values)

    data = CassavaDataset(test_data, transforms, root_dir=config["test_location"])
    dataloader = DataLoader(
        data,
        shuffle=False,
        batch_size=config["batch_size"],
        pin_memory=torch.cuda.is_available(),
        num_workers=config["workers"],
    )
    return dataloader




## === cell 10
def infer(model, dataloader):
    print("Running inference...")
    model.eval()
    predictions = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch = batch.to(device, non_blocking=True)
            batch_hat = model(batch)
            predictions.append(batch_hat.detach().cpu())

    return torch.cat(predictions, dim=0)




## === cell 11
class CassavaTrainDataset(Dataset):
    def __init__(self, df, transforms, root_dir):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.root_dir = root_dir

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"]
        y = int(self.df.loc[idx, "label"])
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = self.transforms(image=image)["image"]
        image = np.moveaxis(image, -1, 0)
        x = torch.tensor(image, dtype=torch.float32)
        return x, torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.df)


def _train_head_only_on_train_csv(model: nn.Module):
    """
    Change rationale (score-up toward target, minimal core logic impact):
    If we have no cassava checkpoint, the 5-class head is not cassava-trained.
    We keep the same EfficientNet-B4 architecture but freeze the backbone and
    train only classifier[1] on train.csv for a couple deterministic epochs.

    Additional small fix to move accuracy upward:
    - Keep frozen backbone in eval() so BatchNorm running stats do NOT update
      on cassava images while only training the linear head (common transfer best-practice).
    """
    train_csv = "../input/cassava-leaf-disease-classification/train.csv"
    if (not config.get("fallback_train_head", True)) or (not os.path.isfile(train_csv)):
        return model

    df = pd.read_csv(train_csv)
    if not {"image_id", "label"}.issubset(df.columns):
        return model

    for p in model.parameters():
        p.requires_grad = False
    base_model = model.model if isinstance(model, LogitBiasWrapper) else model
    for p in base_model.classifier[1].parameters():
        p.requires_grad = True

    transforms = get_fallback_train_transforms()
    train_ds = CassavaTrainDataset(df, transforms, root_dir=config["train_location"])

    sampler = None
    shuffle = True
    if config.get("fallback_use_weighted_sampler", True):
        counts = df["label"].value_counts().to_dict()
        w = (
            df["label"]
            .map(lambda x: 1.0 / max(float(counts.get(int(x), 1)), 1.0))
            .values
        )
        w = torch.tensor(w, dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=w,
            num_samples=len(w),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        shuffle=shuffle,
        sampler=sampler,
        batch_size=int(config.get("fallback_head_batch_size", 64)),
        num_workers=config["workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    base_model.to(device)

    base_model.eval()
    base_model.classifier[1].train()

    optim = torch.optim.AdamW(
        base_model.classifier[1].parameters(),
        lr=float(config.get("fallback_head_lr", 5e-3)),
        weight_decay=0.0,
    )
    criterion = nn.CrossEntropyLoss()

    max_batches = int(config.get("fallback_max_batches_per_epoch", 220))
    for ep in range(int(config.get("fallback_head_epochs", 2))):
        total_loss = 0.0
        total_n = 0
        for bi, (xb, yb) in enumerate(
            tqdm(
                train_loader,
                desc=f"Fallback head-train epoch {ep+1}",
                total=min(len(train_loader), max_batches),
            )
        ):
            if bi >= max_batches:
                break

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            logits = base_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

            bs = xb.size(0)
            total_loss += float(loss.detach().cpu()) * bs
            total_n += bs

        print(f"Fallback head-train epoch {ep+1}: loss={total_loss/max(total_n,1):.4f}")

    base_model.eval()
    if isinstance(model, LogitBiasWrapper):
        model.model = base_model
        return model
    return base_model


def load_model():
    ckpt_path = _find_checkpoint_path()

    if ckpt_path is None:
        print(
            f"WARNING: no checkpoint found (looked for '{config['checkpoint']}' and common names, searched ../input). "
            "Falling back to ImageNet-pretrained EfficientNet weights + minimal head calibration."
        )
        model = _build_backbone_from_config(pretrained=True).to(device)
        _init_head_from_imagenet_logits(model)

        base_bias = _cassava_class_prior_logit_bias()
        if base_bias is not None:
            alpha = _calibrate_bias_alpha_if_enabled(model, base_bias)
            model = LogitBiasWrapper(model, float(alpha) * base_bias).to(device)
            print("Applied Cassava class-prior logit bias fallback.")

        model = _train_head_only_on_train_csv(model)
        return model

    model = _build_backbone_from_config(pretrained=False)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    if state_dict is None:
        raise ValueError(f"Unrecognized checkpoint format at: {ckpt_path}")

    missing, unexpected = _try_load_state_dict_flex(model, state_dict)
    print("Loading model from checkpoint:", ckpt_path)
    if missing:
        print(
            "Missing keys (non-fatal):",
            missing[:12],
            "..." if len(missing) > 12 else "",
        )
    if unexpected:
        print(
            "Unexpected keys (non-fatal):",
            unexpected[:12],
            "..." if len(unexpected) > 12 else "",
        )
    return model.to(device)




## === cell 12
if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataloader = get_dataloader()
    model = load_model()
    predictions = None
    print("Inferring experiment", config["experiment_name"])

    for epoch in range(config["epochs"]):
        print("Epoch:", epoch)
        start_time = time.time()

        if epoch == 0:
            predictions = infer(model, dataloader)
        else:
            predictions += infer(model, dataloader)

        print("Time:", time.time() - start_time)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    predictions /= config["epochs"]
    results = predictions.numpy()
    test["label"] = np.argmax(results, axis=-1).astype(int)



## === cell 13
sub = test[["image_id", "label"]].copy()
sample = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
sub = sample[["image_id"]].merge(sub, on="image_id", how="left")
assert len(sub) == len(sample), "Submission length mismatch"
assert sub["label"].isna().sum() == 0, "Some test images missing predictions"
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
