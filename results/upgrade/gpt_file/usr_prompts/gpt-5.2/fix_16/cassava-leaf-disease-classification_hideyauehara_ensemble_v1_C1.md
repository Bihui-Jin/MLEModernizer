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

0.8945300695074041

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I remove the hard dependency on `efficientnet_pytorch` (it isn’t installed) by skipping EfficientNet weights if present, while keeping the same ensemble/inference logic for the supported torchvision models. I also fix dataset path detection to always point at the provided Cassava dataset directory, so `test_images` is found both in interactive and batch runs. Albumentations v2 changed `RandomResizedCrop`’s API, so I update those calls to use `size=(H,W)` and a valid `scale` tuple to prevent the current validation error. Finally, I fix the ensembling aggregation so it averages probabilities over all model×TTA predictions and writes a valid `submission.csv` with exactly `image_id,label`.'
- What this solution (achieved 0.11584) has done: 'Your very low score (0.11584) is consistent with the code not finding any checkpoints under `../input/...` and then falling back to `sample_submission.csv` (which effectively guesses a constant label), so the main fix is to correctly discover checkpoints from the actual dataset location(s) available in this environment. I minimally change the checkpoint search to look under `/kaggle/input/**` and `/kaggle/data/input/**` (and keep the old `../input/**` patterns too), while keeping the same model wrapping/ensemble/TTA logic. I also ensure `df_test` always comes from `sample_submission.csv` for ordering (so predictions align to the required test row order) and only fill its `label` column from the ensemble output. These changes keep the same architecture/inference semantics but should move accuracy sharply upward toward your target by actually using the provided pretrained weights (when present).'
- What this solution (achieved 0.11584) has done: 'The score (0.11584) strongly suggests your inference is effectively using no real trained weights (either because no checkpoints are actually found/loaded, or because the loaded checkpoint keys don’t match your wrapped model so the classifier stays random). I make two minimal, score-relevant fixes: (1) make checkpoint discovery robust but restricted to cassava-related input folders (so you don’t accidentally load unrelated `.pth` files), and (2) load checkpoints in a tolerant way that can handle common key-prefix patterns (`module.`, `model.`, `state_dict`) and can also map “trained head” weights onto your wrapper’s `fc` (or `model._fc` for EfficientNet), without changing architecture. Finally, I enforce the test order to exactly match `sample_submission.csv` (stable alignment) and keep the same ensemble/TTA averaging logic so the only effect is “actually use the intended weights,” which should move accuracy sharply toward your target.'
- What this solution (achieved 0.11584) has done: 'Your score (0.11584) is consistent with the submission being effectively “random/constant-ish,” which usually happens when no real cassava-trained checkpoints are being loaded. I make the smallest score-relevant changes to (1) reliably discover cassava `.pth` checkpoints only under the current dataset folders you actually have (`/kaggle/input` and `/kaggle/data/input`), and (2) load them more correctly by mapping common checkpoint key patterns into your wrapper modules (especially mapping `fc/classifier` weights into `FinalLayerMixupModel.fc`). I also ensure inference uses a safe `.reshape` instead of `.squeeze()` to avoid accidental shape bugs at batch size 1 (which can silently wreck predictions). Core logic (same models, same wrappers, same TTA averaging, same argmax) is preserved; we’re just making “use the intended weights correctly” more likely, which should move accuracy sharply upward toward your target.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target, and the most likely cause is that you’re still not actually loading any cassava-trained checkpoints (or you’re loading the wrong files / wrong head keys), so inference is effectively random/constant. I make minimal, score-relevant changes to (1) broaden-but-safe checkpoint discovery to include common Kaggle naming (not only folders containing “cassava*”), and (2) improve checkpoint head-key remapping so `FinalLayerMixupModel.fc` (and DenseNet’s `fc`) reliably receives the trained classifier weights even when checkpoints were saved from an unwrapped base model (`model.fc`, `classifier`, etc.). I also harden image path resolution so it never silently points to the wrong folder (which can also destroy accuracy via missing/failed reads). Core model wrappers, TTA loop, probability averaging, and argmax submission logic are preserved.'
- What this solution (achieved 0.11584) has done: 'Your low score is most consistent with “wrong weights loaded or no meaningful head loaded”: many cassava checkpoints are saved from a plain torchvision model (with `fc`/`classifier`) while your wrapper expects its own `fc`, and the current remap only triggers in a narrow case. I make the smallest possible change to robustly map common head keys (including `convlayer.*` vs `layer*` / `features.*` and multiple `fc/classifier` naming variants) into your wrapper before loading, without changing the model architectures, TTA, or averaging logic. I also add a safety filter to avoid accidentally loading unrelated `.pth` files by requiring the checkpoint to have at least one plausible backbone key, which prevents “random head” behavior that tanks accuracy. These changes should move accuracy sharply upward toward your target by making the ensemble actually use the intended trained weights.'
- What this solution (achieved 0.11584) has done: 'Your score is extremely low for Cassava, which is most consistent with either (a) no real cassava-trained checkpoints being loaded, or (b) checkpoints being loaded but the backbone/head weights not mapping into your wrapper modules—so inference behaves like a random model. I keep your exact ensemble + TTA inference logic, but make the smallest score-relevant fixes to checkpoint discovery and weight loading: restrict to cassava-relevant `.pth` files, and robustly remap both backbone keys and classifier-head keys into `FinalLayerMixupModel*` (including ResNet `convlayer.0.*` mapping and DenseNet `features.*` mapping). I also ensure the test image directory resolves correctly in all provided folder layouts and that the submission is always aligned to `sample_submission.csv` order. These changes should move accuracy sharply upward toward your target by making it much more likely you’re actually using trained weights without changing your modeling/inference semantics.'
- What this solution (achieved 0.11584) has done: 'Your current score strongly suggests that most/all checkpoints are still being skipped or not being matched to the right architecture branch, so the ensemble ends up close to a constant/random predictor. I make two minimal, score-relevant fixes: (1) make checkpoint→model selection more robust by matching on the full checkpoint path (not just basename) and adding a safe fallback that infers the architecture from the checkpoint’s tensor shapes/keys when the filename doesn’t contain the model name, and (2) make head-weight remapping actually take effect by loading the remapped head into the wrapper with the correct key prefix (`fc.*` vs `model._fc.*`) and doing it in the right order. This preserves your exact inference/TTA/averaging logic while making it much more likely that real trained weights are used, which should move accuracy sharply upward toward your target.'
- What this solution (achieved 0.11584) has done: 'Your score (0.11584) is far below the target, and the most likely reason is that even when checkpoints are found they are not actually being applied correctly to your wrapper models (especially for ResNet: your wrapper’s `convlayer` is a `Sequential`, but the remap currently writes keys like `convlayer.0.conv1.weight`, which won’t match and leaves the backbone largely random). I make the smallest weight-loading fix: remap ResNet backbone keys directly to `convlayer.0.*` through `convlayer.7.*` by matching each original child module name (`conv1`, `bn1`, `layer1`...) to the correct index inside `convlayer`. I also tighten the “looks-like-backbone” check to avoid skipping good checkpoints due to key-prefix differences, without changing the ensemble/TTA/argmax logic. These changes preserve your architecture and inference loop but should move accuracy sharply upward toward the target by making the loaded weights actually take effect.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is so far below the target that it’s almost certainly not using meaningful cassava-trained weights; the biggest low-risk improvement is to stop loading irrelevant `.pth` files and to correctly map ResNet backbone keys into your `convlayer` wrapper so the backbone isn’t left randomly initialized. I make two minimal, score-relevant fixes: (1) tighten checkpoint discovery to only `.pth` that are very likely cassava models (and verify they contain plausible backbone keys before running inference), and (2) fix the ResNet remapping bug so `conv1/bn1/layer1..4` keys map to `convlayer.{idx}.*` (not `convlayer.{idx}.conv1.weight`-style which won’t match). Everything else (models, TTA list, averaging, argmax, submission format/path) stays the same. These changes should substantially increase accuracy toward your target by ensuring the ensemble actually uses trained weights.'
- What this solution (achieved 0.11584) has done: 'Your score is extremely low relative to the target, which strongly indicates the ensemble is still effectively producing near-random predictions because the cassava checkpoints aren’t being loaded into your wrapper correctly (especially the ResNet backbone keys). I make a minimal, score-relevant fix in the checkpoint remapping so ResNet keys like `conv1.weight`, `bn1.*`, `layer1.*` map into your wrapper’s `convlayer.{idx}.*` exactly, and I remove an early-return condition that currently prevents remapping from ever happening once prefixes are stripped. I also slightly broaden the “looks-like-backbone” check to avoid skipping valid checkpoints that store weights already under `convlayer.*` (or with other common prefixes) while keeping the same ensemble/TTA/argmax logic. These changes preserve your model definitions and inference approach, but make it much more likely you’re actually using trained weights, which should move accuracy sharply upward toward your target.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so we should make the smallest changes that are most likely to make the loaded checkpoints actually take effect (instead of silently leaving large parts of the wrapper randomly initialized). The main fix is in ResNet backbone key remapping: your wrapper’s `convlayer` is a `Sequential` of whole child modules, so we must map `conv1.*`, `bn1.*`, `layer1.*` etc. onto `convlayer.{idx}.*` (not onto nested non-existent subkeys), otherwise most weights remain unloaded and predictions become near-random. I also make model-name inference slightly safer for ResNet by distinguishing `resnet18` vs `resnet50` using the `fc.weight` shape when present, which helps avoid loading a checkpoint into the wrong architecture. These are minimal, score-relevant changes that preserve your ensemble/TTA/argmax submission semantics and should increase accuracy toward your target by ensuring checkpoints genuinely load.'
- What this solution (achieved 0.11584) has done: 'Your score (0.11584) is far below the target, so the smallest likely cause is that the ensemble is still effectively using near-random weights because the ResNet backbone weights aren’t being mapped into your `FinalLayerMixupModel.convlayer` correctly. I minimally fix the ResNet key remapping so keys like `conv1.weight`, `bn1.*`, `layer1.*` map onto `convlayer.{idx}.*` (without creating non-existent nested keys), and broaden the “looks-like-backbone” check to accept common saved formats (already-wrapped `convlayer.*`, or `backbone.*`). I also add a small safety print to confirm at least one checkpoint actually loads substantial backbone tensors (to avoid silently submitting constant-ish predictions), while preserving your model choices, TTA list, probability averaging, and argmax submission semantics. No training logic or architecture is changed; this only makes the intended weights actually take effect.'
- What this solution (achieved 0.11584) has done: 'Your current score is so far below target that the smallest likely issue is still “weights not actually being applied,” especially for ResNet: your wrapper stores whole child modules inside `convlayer`, so checkpoint keys like `conv1.weight` must map to `convlayer.0.weight` (not `convlayer.0.conv1.weight`). I minimally fix the ResNet backbone key remapping to correctly translate `conv1/bn1/relu/maxpool/layer1..4/avgpool` into the correct `convlayer.{idx}.*` keys so the backbone loads instead of staying random. I also make the backbone-check a bit more accepting for already-wrapped checkpoints (e.g., `convlayer.*`) without changing the ensemble/TTA/argmax semantics. Everything else (models, TTA list, probability averaging, submission format/path) stays the same, but this should move accuracy sharply upward toward your target by ensuring checkpoints truly load.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so the smallest likely fix is to ensure checkpoints actually load into your wrapper correctly (right now ResNet weights can silently miss because keys like `conv1.weight` must map to `convlayer.0.*`, not `convlayer.0.conv1.*`). I minimally fix the ResNet backbone key remapping to translate `conv1/bn1/relu/maxpool/layer1..4/avgpool` into the exact `FinalLayerMixupModel.convlayer.{idx}.*` keys, and I also accept already-wrapped formats like `model.convlayer.*`. I keep your ensemble + TTA + probability averaging + argmax submission logic unchanged, but add a tiny “loaded params” sanity print so you can see whether weights are being applied (without altering predictions). These changes should move accuracy sharply upward toward your target by making the intended trained weights actually take effect.'

# 9. Code solution

## === cell 0
import numpy as np
import glob


def _glob_many(patterns):
    out = []
    for p in patterns:
        out.extend(glob.glob(p, recursive=True))
    return list(np.unique(out))


pretrained_models = _glob_many(
    [
        "../input/**/cassava*/*.pth",
        "../input/**/cassava*/*/*.pth",
        "/kaggle/input/**/cassava*/*.pth",
        "/kaggle/input/**/cassava*/*/*.pth",
        "/kaggle/data/input/**/cassava*/*.pth",
        "/kaggle/data/input/**/cassava*/*/*.pth",
        "/kaggle/data/**/cassava*/*.pth",
        "/kaggle/data/**/cassava*/*/*.pth",
        "/kaggle/input/**/*.pth",
        "/kaggle/data/input/**/*.pth",
        "../input/**/*.pth",
    ]
)


def _is_likely_cassava_ckpt(path: str) -> bool:
    name = path.lower()
    strong_keywords = ["cassava", "leaf", "disease", "cldc"]
    arch_keywords = [
        "resnet",
        "resnext",
        "densenet",
        "efficientnet",
        "b7",
        "fold",
        "epoch",
        "best",
        "finetune",
        "fine-tune",
    ]
    return (any(k in name for k in strong_keywords)) and (
        any(k in name for k in arch_keywords)
    )


pretrained_models = [p for p in pretrained_models if _is_likely_cassava_ckpt(p)]

print(f"{len(pretrained_models)} models found after filename filtering.")
if len(pretrained_models) > 0:
    print("\n".join(np.sort(pretrained_models)))



## === cell 1
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
from torchvision import models, transforms  # pretrained models, transforms

import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

import os
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



## === cell 2
try:
    from efficientnet_pytorch import EfficientNet  # type: ignore

    _HAS_EFFICIENTNET = True
except Exception:
    EfficientNet = None
    _HAS_EFFICIENTNET = False
    print(
        "efficientnet_pytorch is not available; EfficientNet checkpoints (if any) will be skipped."
    )



## === cell 3
SIZE = 512  # image size
num_classes = 5



## === cell 4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 5
def _pick_existing_base_dir(candidates):
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


BASE_DIR = _pick_existing_base_dir(
    [
        "../input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "/kaggle/data/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
)

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset directory in expected locations."
    )


def _resolve_test_images_dir(base_dir: str) -> str:
    candidates = [
        os.path.join(base_dir, "test_images"),
        os.path.join(base_dir, "cassava-leaf-disease-classification", "test_images"),
        os.path.join(base_dir, "test_images", "test_images"),
        os.path.join(
            base_dir,
            "cassava-leaf-disease-classification",
            "test_images",
            "test_images",
        ),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"test_images not found under BASE_DIR={base_dir}")


TEST_PATH = _resolve_test_images_dir(BASE_DIR)
test_files = os.listdir(TEST_PATH)

print(f"BASE_DIR: {BASE_DIR}")
print(f"TEST_PATH: {TEST_PATH}")
print(f"Number of test images: {len(test_files)}")



## === cell 6
df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")



## === cell 7
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)



## === cell 8
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



## === cell 9
pass




## === cell 10
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        """
        model: 学習済みモデルを指定
        """
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = x.reshape(x.size(0), -1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)

            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = x.reshape(x.size(0), -1)
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
        mixed_x = mixed_x.reshape(mixed_x.size(0), -1)
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 11
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        """
        model: 学習済みモデルを指定
        """
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
            x = x.reshape(x.size(0), -1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)

            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.reshape(x.size(0), -1)
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
        mixed_x = mixed_x.reshape(mixed_x.size(0), -1)
        outputs = self.fc(mixed_x)

        labels_a = labels
        labels_b = labels[index]

        pred = outputs
        loss = lam * self.criterion(pred, labels_a) + (1 - lam) * self.criterion(
            pred, labels_b
        )

        return outputs, loss, labels_a, labels_b, lam




## === cell 12
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelEN, self).__init__()

        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_classes)

        self.model = model
        self.criterion = criterion

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




## === cell 13
pass




## === cell 14
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()

        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img_path = f"{TEST_PATH}/{image_id}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        img = self.load_image(image_id)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, image_id




## === cell 15
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            sd = obj["model_state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        else:
            sd = obj
    else:
        sd = obj

    if not isinstance(sd, dict):
        raise ValueError(
            "Unsupported checkpoint format; expected dict-like state_dict."
        )

    new_sd = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "model.", "net.", "backbone."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        new_sd[nk] = v
    return new_sd


def _remap_backbone_keys_for_wrapper(net, sd, model_name: str):
    if model_name == "efficientnet-b7":
        return sd  # handled by net.model load directly

    sd2 = dict(sd)

    if any(k.startswith("convlayer.") for k in sd2.keys()):
        return sd2

    if isinstance(net, FinalLayerMixupModelDenseNet) or model_name == "densenet201":
        for k, v in sd.items():
            if k.startswith("features."):
                sd2["convlayer." + k] = v
        return sd2

    name_to_idx = {
        "conv1": 0,
        "bn1": 1,
        "relu": 2,
        "maxpool": 3,
        "layer1": 4,
        "layer2": 5,
        "layer3": 6,
        "layer4": 7,
        "avgpool": 8,
    }
    for k, v in sd.items():
        prefix = k.split(".", 1)[0]
        if prefix in name_to_idx:
            idx = name_to_idx[prefix]
            if "." in k:
                rest = k.split(".", 1)[1]
                sd2[f"convlayer.{idx}.{rest}"] = v
            else:
                sd2[f"convlayer.{idx}"] = v

    return sd2


def _remap_head_weights_for_wrapper(net, sd, model_name: str):
    remap = {}

    candidates = [
        ("fc.weight", "fc.bias"),
        ("classifier.weight", "classifier.bias"),
        ("head.weight", "head.bias"),
        ("linear.weight", "linear.bias"),
        ("_fc.weight", "_fc.bias"),
    ]

    if model_name == "efficientnet-b7":
        if (
            hasattr(net, "model")
            and hasattr(net.model, "_fc")
            and isinstance(net.model._fc, nn.Linear)
        ):
            for sw, sb in candidates:
                if sw in sd and sb in sd:
                    if tuple(sd[sw].shape) == tuple(net.model._fc.weight.shape):
                        remap["model._fc.weight"] = sd[sw]
                        remap["model._fc.bias"] = sd[sb]
                        break
        return remap

    if hasattr(net, "fc") and isinstance(net.fc, nn.Linear):
        for sw, sb in candidates:
            if sw in sd and sb in sd:
                if tuple(sd[sw].shape) == tuple(net.fc.weight.shape):
                    remap["fc.weight"] = sd[sw]
                    remap["fc.bias"] = sd[sb]
                    break

    return remap


def _looks_like_backbone_checkpoint(sd: dict, model_name: str) -> bool:
    keys = sd.keys()
    if model_name in ("resnet18", "resnet50", "resnet152", "resnext101"):
        return (
            ("conv1.weight" in keys)
            or any(k.startswith("layer") for k in keys)
            or any(k.startswith("bn1.") for k in keys)
            or any(k.startswith("convlayer.") for k in keys)
            or any(k.startswith("convlayer.0.") for k in keys)
        )
    if model_name == "densenet201":
        return (
            any(k.startswith("features.") for k in keys)
            or any(k.startswith("convlayer.features.") for k in keys)
            or any(k.startswith("convlayer.") for k in keys)
        )
    if model_name == "efficientnet-b7":
        return any(k.startswith("_conv_stem") for k in keys) or any(
            "blocks" in k for k in keys
        )
    return True


def _infer_model_name_from_state_dict(sd: dict):
    keys = list(sd.keys())

    if any(k.startswith("features.") for k in keys) or any(
        k.startswith("convlayer.features.") for k in keys
    ):
        return "densenet201"

    if any(k.startswith("_conv_stem") for k in keys) or any(
        "blocks." in k for k in keys
    ):
        return "efficientnet-b7"

    if any(k == "conv1.weight" for k in keys) or any(
        k.startswith("convlayer.0.") for k in keys
    ):
        if "fc.weight" in sd:
            in_features = int(sd["fc.weight"].shape[1])
            if in_features == 512:
                return "resnet18"
            if in_features == 2048:
                return "resnet50"
        return "resnet50"

    return None


def _load_checkpoint_into_wrapper(net, state_dict, model_name):
    sd = state_dict

    sd = _remap_backbone_keys_for_wrapper(net, sd, model_name)

    if model_name == "efficientnet-b7":
        target = net.model
        missing, unexpected = target.load_state_dict(sd, strict=False)
        head_remap = _remap_head_weights_for_wrapper(net, sd, model_name)
        if len(head_remap) > 0:
            net.load_state_dict(head_remap, strict=False)
        return missing, unexpected

    missing, unexpected = net.load_state_dict(sd, strict=False)

    if ("fc.weight" in missing) or ("fc.bias" in missing):
        head_remap = _remap_head_weights_for_wrapper(net, sd, model_name)
        if len(head_remap) > 0:
            net.load_state_dict(head_remap, strict=False)

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
            outputs = net(inputs, False, "test")
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")

    return np.concatenate(probability, axis=0)




## === cell 16
df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")[["image_id", "label"]].copy()

probability = []
start_time = time.time()

loaded_any = (
    False  # score-relevant sanity: avoid silently submitting fallback-like predictions
)

if len(pretrained_models) == 0:
    print("No pretrained models found; falling back to sample_submission.csv.")
else:
    for pretrained_model in pretrained_models:
        path_lower = pretrained_model.lower()
        basename = os.path.splitext(os.path.basename(pretrained_model))[0].lower()
        tag = f"{path_lower} {basename}"

        criterion = nn.CrossEntropyLoss()

        try:
            raw = torch.load(pretrained_model, map_location="cpu")
            sd = _extract_state_dict(raw)
        except Exception as e:
            print(f"{basename}: failed to read checkpoint ({e}). Skipping.")
            continue

        MODEL_NAME = None
        if "resnet18" in tag:
            MODEL_NAME = "resnet18"
        elif "resnet152" in tag:
            MODEL_NAME = "resnet152"
        elif "resnet50" in tag:
            MODEL_NAME = "resnet50"
        elif "resnext101" in tag or "resnext-101" in tag or "resnext_101" in tag:
            MODEL_NAME = "resnext101"
        elif "densenet201" in tag:
            MODEL_NAME = "densenet201"
        elif "efficientnet-b7" in tag or "efficientnetb7" in tag or "efn_b7" in tag:
            MODEL_NAME = "efficientnet-b7"
        else:
            MODEL_NAME = _infer_model_name_from_state_dict(sd)

        if MODEL_NAME is None:
            continue

        if MODEL_NAME == "resnet18":
            net = models.resnet18(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif MODEL_NAME == "resnet50":
            net = models.resnet50(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 32
        elif MODEL_NAME == "resnet152":
            net = models.resnet152(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif MODEL_NAME == "resnext101":
            net = models.resnext101_32x8d(weights=None)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif MODEL_NAME == "densenet201":
            net = models.densenet201(weights=None)
            net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif MODEL_NAME == "efficientnet-b7":
            if not _HAS_EFFICIENTNET:
                print(
                    f"{basename}: EfficientNet checkpoint present but efficientnet_pytorch not installed. Skipping."
                )
                continue
            net = EfficientNet.from_name(MODEL_NAME)
            net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            continue

        print(
            f"{os.path.splitext(os.path.basename(pretrained_model))[0]}: {MODEL_NAME}"
        )

        if not _looks_like_backbone_checkpoint(sd, MODEL_NAME):
            print(
                f"{basename}: checkpoint does not look like {MODEL_NAME} backbone. Skipping."
            )
            continue

        try:
            missing, unexpected = _load_checkpoint_into_wrapper(net, sd, MODEL_NAME)
            if len(missing) > 0 or len(unexpected) > 0:
                print(
                    f"{basename}: load_state_dict missing={len(missing)} unexpected={len(unexpected)}"
                )
            loaded_any = True
        except Exception as e:
            print(f"{basename}: failed to load checkpoint ({e}). Skipping.")
            continue

        for param in net.parameters():
            param.requires_grad = False

        for tid, transform_ in enumerate(transform["test"]):
            print(f"transform loop={tid}")
            dataset = {"test": TestDataset(df_test, transform=transform_)}
            dataloader = {
                "test": torch.utils.data.DataLoader(
                    dataset["test"],
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                )
            }

            proba = predict_model(basename, net, dataloader)
            probability.append(proba)

        del net
        torch.cuda.empty_cache()

    if (not loaded_any) or (len(probability) == 0):
        print(
            "All models were skipped/failed to load; falling back to sample_submission.csv."
        )
    else:
        mean_proba = np.stack(probability, axis=0).mean(axis=0)
        df_test["label"] = mean_proba.argmax(axis=1).astype(int)
        print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 17
df_test.head()



## === cell 18
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_test[["image_id", "label"]].shape)
print(df_test[["image_id", "label"]].head())
