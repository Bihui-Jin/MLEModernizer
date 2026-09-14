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

0.8952855847688124

# 6. Current score

0.76831

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0852) has done: 'I fix the run-blocking checkpoint path issue by making the script automatically locate an available `.pt/.pth` checkpoint under `../input/` (or fall back to `weights=IMAGENET1K_V1` if none exists) so inference always runs end-to-end. I also ensure `test["label"]` is always created (even if inference fails) so the submission merge cannot KeyError. Finally, I correct the seeding function to avoid the contradictory `deterministic=True` with `benchmark=True` (stability-only) and keep all core inference logic (TTA loop, model, transforms, argmax) unchanged.'
- What this solution (achieved 0.08707) has done: 'Your current score (0.0852) is far below the target (0.8953), and the most likely reason is that inference is effectively random because the code is not loading the intended cassava-trained checkpoint (it falls back to ImageNet weights or a mismatched checkpoint with many missing/unexpected keys). I make the smallest change that materially improves accuracy: prefer loading a checkpoint whose classifier head matches 5 classes and fail fast (instead of silently running with a bad/mismatched state_dict). I also fix the inference augmentation pipeline to use a deterministic center-crop/resize-only transform for test-time inference (your current pipeline uses strong random augmentations and coarse dropout during inference, which can severely hurt accuracy even when averaging). These changes keep the same model architecture, same inference averaging loop, same argmax post-processing, and still produce `submission.csv`.'
- What this solution (achieved 0.08707) has done: 'Your score is far below the target, so the most likely issue is still that you’re not actually loading cassava-trained weights (and/or you’re not finding the real test images directory), which makes predictions near-random. I make the smallest changes that (1) robustly locate the correct `test_images/` folder from the known Kaggle paths, and (2) robustly locate a compatible 5-class EfficientNet-B4 checkpoint by scanning only plausible cassava locations (instead of all `../input`), preferring filenames that look like cassava models and verifying the state_dict keys. I also make the checkpoint loader accept both `net.*` and plain torchvision `classifier.*` key formats (common mismatch), without changing the architecture or inference averaging logic. These changes are directly aimed at turning “random inference” into “using the intended trained weights”, which is the main lever to move accuracy toward ~0.89.'
- What this solution (achieved 0.08707) has done: 'Your score is far below target, so the primary issue is still that inference is running with wrong/untrained weights (effectively near-random). I make the smallest change that materially improves accuracy: load the official cassava-trained EfficientNet-B4 checkpoint from the competition dataset (`../input/cassava-leaf-disease-classification/`), which matches your model and 5-class head. To keep the same architecture and inference loop, I only adjust the checkpoint search order and make the loader remap common key prefixes (e.g., `model.` / `net.`) so the correct weights actually load. This should move accuracy toward the target without altering your TTA/averaging or post-processing.'
- What this solution (achieved 0.08707) has done: 'Your current score is far below target, so the highest-impact minimal fix is to ensure we are actually using cassava-trained weights rather than an incompatible/empty checkpoint fallback. I keep your model, transforms, and inference averaging loop unchanged, but make checkpoint discovery prioritize only files that can be verified as EfficientNet-B4 5-class weights (by checking multiple possible classifier key patterns) and correctly remap common key prefixes so good checkpoints don’t get rejected. I also reduce false “incompatible” rejections by validating compatibility more directly (matching tensor shapes) instead of counting missing/unexpected keys, which can be high for harmless metadata but still load correctly. This should move predictions from near-random toward the target accuracy while still producing the same `submission.csv` format end-to-end.'
- What this solution (achieved 0.08707) has done: 'Your score is far below the target, which strongly suggests the model is still not loading meaningful cassava-trained weights and/or predictions are being misaligned with the submission order. I make two minimal, high-impact fixes: (1) remove the “fail fast” incompatibility rejection so we always load the best available checkpoint with `strict=False` (instead of aborting or silently falling back to ImageNet), and (2) enforce prediction-to-row alignment by building the test dataloader from `sample_submission.csv` order and writing the submission directly in that same order (no merge-induced NaNs/misorder). I also average **probabilities** (softmax) across the existing inference loop epochs (same loop, same model) which is a small semantic improvement for multi-class accuracy compared to averaging logits.'
- What this solution (achieved 0.0867) has done: 'Your score is far below the target, so the smallest high-impact fix is to stop averaging identical deterministic predictions across “epochs” (which currently does nothing) and instead do real test-time augmentation by adding a small set of deterministic geometric flips/transpose while keeping your same model/inference loop/argmax semantics. I also add a strict sanity check to warn you when no meaningful cassava-trained checkpoint was loaded (shape overlap too low), because that typically produces near-random ~0.08–0.10 accuracy regardless of TTA. Finally, I keep submission alignment exactly in `sample_submission.csv` order and ensure the image path resolution is robust, without changing I/O paths or the model architecture.'
- What this solution (achieved 0.0867) has done: 'Your score is far below the target, so the most likely remaining issue is that you’re still not actually loading a cassava-trained checkpoint (so predictions are near-random). I make the smallest high-impact change: first try to load an EfficientNet-B4 cassava checkpoint from a local `../working` path (where you can place/upload your trained weights), and only if that fails fall back to scanning `../input`; this preserves your architecture and inference loop. I also ensure the checkpoint compatibility check prefers checkpoints with strong shape overlap (not just filename keywords) and clearly report which path was loaded, so you can verify you’re not accidentally using ImageNet weights. Everything else (transforms, TTA loop, softmax averaging, submission order/format) stays the same.'
- What this solution (achieved 0.0867) has done: 'Your score is far below the target, so the most likely remaining cause is still “near-random inference” from not actually loading cassava-trained weights. I make the smallest high-impact change: explicitly load a known-good EfficientNet-B4 cassava checkpoint if it exists in common Kaggle input locations (including the widely used public `cassava-b4`-style datasets), and only then fall back to the current broad scan / ImageNet weights. I also tighten the checkpoint selection to strongly prefer files that (a) contain EfficientNet-B4 feature keys and (b) have a 5-class classifier tensor, while keeping your exact model architecture, softmax-prob averaging, TTA loop, and submission ordering unchanged. This should move accuracy significantly upward toward the target without changing the training/inference semantics beyond using the intended weights.'
- What this solution (achieved 0.0867) has done: 'Your current accuracy (0.0867) is far below the target (0.8953), which strongly indicates you’re still not loading a real cassava-trained checkpoint and are effectively running near-random inference. The most direct minimal fix is to stop scanning arbitrary inputs and instead load a known-good baseline checkpoint that is commonly shipped inside this competition’s dataset (`../input/cassava-leaf-disease-classification/`), specifically `baselineepoch20.pt` if present; if it’s not present, we keep your existing fallback behavior. I keep your model architecture, inference loop, TTA passes, and argmax semantics the same, but add a deterministic, compatibility-aware checkpoint resolver that prefers that exact file and verifies it actually loads meaningful weights (high shape-match ratio). This should move the score substantially upward toward the target without changing your core approach.'
- What this solution (achieved 0.0867) has done: 'Your score is far below the target, which strongly suggests the checkpoint being loaded is not actually a cassava-trained EfficientNet-B4 (so predictions are near-random). I make a minimal, high-impact change to checkpoint selection: only consider checkpoints that (a) look like EfficientNet-B4 feature weights and (b) have a 5-class classifier tensor, and stop “falling back to ImageNet” when compatibility is low (because that guarantees poor competition accuracy). I also make the loader correctly handle the common mismatch where a checkpoint stores EfficientNet head weights under `net.classifier.1.*` but your wrapper uses `net.net.classifier.1.*`, by remapping keys before loading. Everything else (model architecture, transforms, TTA loop, softmax-avg, argmax, submission order/format) stays the same.'
- What this solution (achieved 0.74514) has done: 'Your current score is far below the target, so the most likely issue is still “near-random inference” caused by not actually loading a cassava-trained EfficientNet-B4 checkpoint (the competition dataset itself does not include `baselineepoch20.pt`). I make the smallest high-impact change: if no suitable checkpoint is found, run a quick, deterministic fine-tune of the existing EfficientNet-B4 head (and optionally last block) on `train.csv` within the same notebook, then use those weights for test inference—this preserves the core model architecture and loss (CrossEntropy) and still produces `submission.csv`. I keep your existing inference transforms/TTA averaging loop unchanged, only adding a lightweight training stage and the minimal dataset plumbing to read `train_images/`. This should move accuracy substantially upward toward the target band while staying within Kaggle runtime constraints.'
- What this solution (achieved 0.78027) has done: 'Your score (0.745) is below the target (0.895), so we should improve generalization with the smallest changes that keep your core model/inference logic intact. The highest-impact minimal fix is to change the finetune stage from “train-on-all-data for 1 epoch” (which can overfit/miscalibrate) to a small stratified train/validation split while still training the same head/last block with the same loss/optimizer. We then select the best epoch by validation accuracy (no early stopping—just picking among the fixed number of epochs) and save/load those weights for inference. This typically increases test accuracy materially while preserving your EfficientNet-B4 model, CrossEntropy training, and the same TTA inference/softmax-averaging/argmax submission pipeline.'
- What this solution (achieved 0.76831) has done: 'Your current score (0.78027) is below the target (0.8953), so we should make a small, low-risk generalization improvement without changing the model architecture or loss. The biggest issue is that your fine-tune stage never runs when a checkpoint is found—even if that checkpoint is weak or partially incompatible—so you may be leaving accuracy on the table. I keep the exact same EfficientNet-B4 model, CrossEntropy training, and TTA inference loop, but (1) enable fine-tuning also when the loaded checkpoint looks low-quality (based on the already-computed shape-match ratio), and (2) add a lightweight learning-rate schedule during the fixed number of finetune epochs to improve convergence stability. These changes are directly aimed at moving accuracy upward toward the target while staying within Kaggle constraints and preserving core logic.'

# 9. Code solution

## === cell 0
import os
import gc
import time
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from tqdm import tqdm
from torchvision import models

warnings.filterwarnings("ignore")



## === cell 1
image_size = 380



## === cell 2
config = dict(
    seed=22,
    experiment_name="modified",
    test_location="../input/cassava-leaf-disease-classification/test_images",
    checkpoint_path="../input/saved-models",
    checkpoint="baselineepoch20.pt",
    model="efficientnet-b4",  # kept for compatibility with existing config
    epochs=10,
    batch_size=16,
    workers=8,
    inference_augmentations=[
        dict(name="LongestMaxSize", params=dict(max_size=image_size, p=1.0)),
        dict(
            name="PadIfNeeded",
            params=dict(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0,
            ),
        ),
        dict(
            name="CenterCrop", params=dict(height=image_size, width=image_size, p=1.0)
        ),
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ),
    ],
)

config.update(
    dict(
        train_csv="../input/cassava-leaf-disease-classification/train.csv",
        train_images="../input/cassava-leaf-disease-classification/train_images",
        finetune=True,  # turn on quick finetune if no usable ckpt is found
        finetune_epochs=3,  # fixed small number of epochs (no early stopping)
        finetune_lr=3e-4,
        finetune_batch_size=24,
        finetune_workers=8,
        finetune_train_frac=1.0,
        finetune_val_frac=0.10,
        finetune_unfreeze_last_blocks=1,
        finetune_even_if_ckpt_loaded=True,
        finetune_ckpt_shape_ratio_threshold=0.80,  # if loaded ckpt has lower ratio, finetune anyway
    )
)



## === cell 3
sample_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
sample = pd.read_csv(sample_path)

test = sample[["image_id"]].copy()
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
print("Using device:", device)


def resolve_test_location():
    candidates = [
        Path(config["test_location"]),
        Path("../input/test_images"),
        Path("../input/cassava-leaf-disease-classification/test_images"),
        Path(
            "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images"
        ),
        Path("/kaggle/input/cassava-leaf-disease-classification/test_images"),
        Path(
            "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images"
        ),
        Path("/kaggle/data/cassava-leaf-disease-classification/test_images"),
        Path("/kaggle/data/input/cassava-leaf-disease-classification/test_images"),
        Path("/kaggle/data/input/test_images"),
    ]
    for p in candidates:
        try:
            if p.exists():
                jpgs = list(p.glob("*.jpg"))
                if len(jpgs) > 0:
                    return str(p)
        except Exception:
            continue
    return config["test_location"]


def resolve_train_location():
    candidates = [
        Path(config["train_images"]),
        Path("../input/cassava-leaf-disease-classification/train_images"),
        Path(
            "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images"
        ),
        Path("/kaggle/input/cassava-leaf-disease-classification/train_images"),
        Path(
            "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images"
        ),
        Path("/kaggle/data/cassava-leaf-disease-classification/train_images"),
        Path("/kaggle/data/input/cassava-leaf-disease-classification/train_images"),
    ]
    for p in candidates:
        try:
            if p.exists():
                jpgs = list(p.glob("*.jpg"))
                if len(jpgs) > 0:
                    return str(p)
        except Exception:
            continue
    return config["train_images"]


config["test_location"] = resolve_test_location()
config["train_images"] = resolve_train_location()
print("Resolved test_location:", config["test_location"])
print("Resolved train_images:", config["train_images"])




## === cell 6
class EfficientNetB4Like(nn.Module):
    """
    Replace efficientnet_pytorch dependency with torchvision EfficientNet-B4.
    Keep the same 'num_classes=5' head.
    """

    def __init__(self, num_classes=5, weights=None):
        super().__init__()
        self.net = models.efficientnet_b4(weights=weights)
        in_features = self.net.classifier[1].in_features
        self.net.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(checkpoint):
    state_dict = None
    meta = {}
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            tensor_values = [v for v in checkpoint.values() if torch.is_tensor(v)]
            if len(tensor_values) == len(checkpoint):
                state_dict = checkpoint
        meta = checkpoint
    else:
        state_dict = checkpoint
    return state_dict, meta


def _remap_state_dict_for_wrapper(sd):
    if not isinstance(sd, dict) or len(sd) == 0:
        return sd

    if any(k.startswith("net.") for k in sd.keys()):
        return sd

    if any(k.startswith("model.") for k in sd.keys()):
        return {
            ("net." + k[len("model.") :]) if k.startswith("model.") else k: v
            for k, v in sd.items()
        }

    if any(k.startswith("features.") or k.startswith("classifier.") for k in sd.keys()):
        return {f"net.{k}": v for k, v in sd.items()}

    return sd


def _remap_wrapper_depth(sd):
    if not isinstance(sd, dict) or len(sd) == 0:
        return sd

    msd_keys = set(EfficientNetB4Like(num_classes=5, weights=None).state_dict().keys())

    overlap_direct = sum(1 for k in sd.keys() if k in msd_keys)
    if overlap_direct > 50:
        return sd

    remapped = {}
    for k, v in sd.items():
        nk = k

        if nk.startswith("net.features."):
            nk = "net.net." + nk[len("net.") :]

        if nk.startswith("net.classifier."):
            nk = "net.net." + nk[len("net.") :]

        if nk.startswith("classifier."):
            nk = "net.net." + nk

        if nk.startswith("features."):
            nk = "net.net." + nk

        remapped[nk] = v

    overlap_after = sum(1 for k in remapped.keys() if k in msd_keys)
    if overlap_after >= overlap_direct:
        return remapped
    return sd


def _classifier_out_features_from_state_dict(sd):
    prefixes = [
        "net.net.classifier.1",
        "net.classifier.1",
        "classifier.1",
        "model.classifier.1",
        "net.fc",
        "fc",
        "model.fc",
        "net.head",
        "head",
        "model.head",
    ]
    for prefix in prefixes:
        w = sd.get(f"{prefix}.weight", None)
        b = sd.get(f"{prefix}.bias", None)
        if torch.is_tensor(w) and w.ndim == 2:
            return int(w.shape[0])
        if torch.is_tensor(b) and b.ndim == 1:
            return int(b.shape[0])
    return None


def _shape_compatibility_ratio(model, sd):
    msd = model.state_dict()
    overlap = 0
    good = 0
    for k, v in sd.items():
        if k in msd and torch.is_tensor(v) and torch.is_tensor(msd[k]):
            overlap += 1
            if tuple(v.shape) == tuple(msd[k].shape):
                good += 1
    if overlap == 0:
        return 0.0, 0
    return good / overlap, overlap


def _has_effnetb4_feature_keys(sd):
    if not isinstance(sd, dict) or len(sd) == 0:
        return False
    keys = list(sd.keys())
    return any(
        k.startswith("net.net.features.")
        or k.startswith("net.features.")
        or k.startswith("features.")
        or "efficientnet" in k
        for k in keys
    )


def _explicit_preferred_ckpts():
    return [
        Path("../input/cassava-leaf-disease-classification/baselineepoch20.pt"),
        Path(
            "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/baselineepoch20.pt"
        ),
        Path(config["checkpoint_path"]) / config["checkpoint"],
    ]


def _likely_ckpt_dirs():
    return [
        Path("../input/cassava-leaf-disease-classification"),
        Path(
            "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification"
        ),
        Path("../working"),
        Path("../working/cassava-leaf-disease-classification"),
        Path("../working/checkpoints"),
        Path("../working/weights"),
        Path("../input/saved-models"),
        Path("../input"),
        Path("/kaggle/input/cassava-leaf-disease-classification"),
        Path("/kaggle/input"),
        Path("/kaggle/data/input/cassava-leaf-disease-classification"),
        Path("/kaggle/data/input"),
    ]


def _find_best_checkpoint():
    for p in _explicit_preferred_ckpts():
        try:
            if p.exists() and p.is_file():
                return str(p)
        except Exception:
            pass

    candidates = []
    patterns = ["*.pt", "*.pth", "*.bin"]
    for root in _likely_ckpt_dirs():
        try:
            if root.exists():
                for pat in patterns:
                    candidates.extend(root.rglob(pat))
        except Exception:
            continue

    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None

    probe_model = EfficientNetB4Like(num_classes=5, weights=None)

    scored = []
    for p in candidates:
        readable = 0
        ratio = 0.0
        overlap = 0
        out5 = 0
        has_eff_keys = 0

        try:
            ckpt = torch.load(str(p), map_location="cpu")
            sd, _ = _extract_state_dict(ckpt)
            if isinstance(sd, dict):
                readable = 1
                sd = _strip_module_prefix(sd)
                sd = _remap_state_dict_for_wrapper(sd)
                sd = _remap_wrapper_depth(sd)
                has_eff_keys = 1 if _has_effnetb4_feature_keys(sd) else 0
                of = _classifier_out_features_from_state_dict(sd)
                if of == 5:
                    out5 = 1
                ratio, overlap = _shape_compatibility_ratio(probe_model, sd)
        except Exception:
            readable = 0

        if not (readable and out5 and has_eff_keys and overlap >= 50 and ratio >= 0.20):
            continue

        name = p.name.lower()
        kw = 0
        for k in [
            "cassava",
            "leaf",
            "disease",
            "efficientnet",
            "eff",
            "b4",
            "baseline",
            "epoch",
            "best",
            "fold",
        ]:
            if k in name:
                kw += 1

        try:
            sz = p.stat().st_size
        except Exception:
            sz = 0

        in_comp_folder = (
            1 if "cassava-leaf-disease-classification" in str(p).lower() else 0
        )

        scored.append((in_comp_folder, kw, round(ratio, 6), overlap, sz, p))

    if not scored:
        return None

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]), reverse=True)
    return str(scored[0][-1])


def load_model():
    ckpt_path = _find_best_checkpoint()
    if ckpt_path is None:
        print(
            "WARNING: No compatible cassava EfficientNet-B4 5-class checkpoint found. "
            "Will initialize from ImageNet weights; finetune stage (if enabled) will make it cassava-specific."
        )
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        model = EfficientNetB4Like(num_classes=5, weights=weights).to(device)
        return model, None, 0.0

    print(f"Loading checkpoint: {ckpt_path}")
    model = EfficientNetB4Like(num_classes=5, weights=None)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict, meta = _extract_state_dict(checkpoint)

    if state_dict is None or not isinstance(state_dict, dict):
        print(
            "WARNING: Could not find model weights in checkpoint (expected dict or keys: 'model'/'state_dict'). "
            "Falling back to ImageNet weights; finetune stage (if enabled) will make it cassava-specific."
        )
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        model = EfficientNetB4Like(num_classes=5, weights=weights).to(device)
        return model, None, 0.0

    state_dict = _strip_module_prefix(state_dict)
    state_dict = _remap_state_dict_for_wrapper(state_dict)
    state_dict = _remap_wrapper_depth(state_dict)

    out_features = _classifier_out_features_from_state_dict(state_dict)
    if out_features is not None and out_features != 5:
        print(
            f"WARNING: Checkpoint head out_features={out_features} but expected 5 classes. "
            f"Will load with strict=False and rely on the model's 5-class head init for missing params. Path: {ckpt_path}"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    ratio, overlap = _shape_compatibility_ratio(model, state_dict)

    if isinstance(meta, dict):
        for k in ["epoch", "train_loss", "val_loss", "metrics", "lr"]:
            if k in meta:
                print(f"{k}: {meta[k]}")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
    print(f"Shape match ratio over overlap keys: {ratio:.3f} (overlap={overlap})")

    if ratio < 0.20 or overlap < 50:
        print(
            "WARNING: Low checkpoint compatibility detected, but keeping loaded weights (no ImageNet fallback) "
            "because selection already filtered for best available cassava-like checkpoint."
        )

    model.to(device)
    return model, ckpt_path, float(ratio)




## === cell 7
def get_transforms():
    transforms = []
    for item in config["inference_augmentations"]:
        name = item["name"]
        params = item["params"]
        if not hasattr(A, name):
            continue
        transforms.append(getattr(A, name)(**params))
    return A.Compose(transforms)


def get_train_transforms():
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, p=1.0),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0,
            ),
            A.CenterCrop(height=image_size, width=image_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ]
    )




## === cell 8
class CassavaDataset(Dataset):
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __getitem__(self, n):
        img_name = self.images[n]
        img_path = os.path.join(config["test_location"], img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        image = np.moveaxis(image, -1, 0)
        image = torch.tensor(image, dtype=torch.float32)
        return image

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(Dataset):
    def __init__(self, df, transforms, root_dir):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.root_dir = root_dir

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image_id"]
        y = int(row["label"])
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        image = np.moveaxis(image, -1, 0)
        x = torch.tensor(image, dtype=torch.float32)
        return x, torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.df)




## === cell 9
def get_dataloader():
    transforms = get_transforms()
    test_data = np.array(sample["image_id"].values)

    data = CassavaDataset(test_data, transforms)
    dataloader = DataLoader(
        data,
        shuffle=False,
        batch_size=config["batch_size"],
        pin_memory=torch.cuda.is_available(),
        num_workers=min(config["workers"], os.cpu_count() or 1),
        drop_last=False,
    )
    return dataloader


def get_train_val_dataloaders():
    train_df = pd.read_csv(config["train_csv"])

    if config["finetune_train_frac"] < 1.0:
        n = int(len(train_df) * config["finetune_train_frac"])
        train_df = train_df.sample(n=n, random_state=config["seed"]).reset_index(
            drop=True
        )

    val_frac = float(config.get("finetune_val_frac", 0.0))
    if not (0.0 < val_frac < 0.5):
        ds = CassavaTrainDataset(
            train_df, get_train_transforms(), config["train_images"]
        )
        dl = DataLoader(
            ds,
            shuffle=True,
            batch_size=config["finetune_batch_size"],
            pin_memory=torch.cuda.is_available(),
            num_workers=min(config["finetune_workers"], os.cpu_count() or 1),
            drop_last=False,
        )
        return dl, None

    rng = np.random.RandomState(config["seed"])
    val_indices = []
    train_indices = []
    for lbl, grp in train_df.groupby("label"):
        idxs = grp.index.values.copy()
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_frac)))
        val_indices.extend(idxs[:n_val].tolist())
        train_indices.extend(idxs[n_val:].tolist())

    train_split = train_df.loc[train_indices].reset_index(drop=True)
    val_split = train_df.loc[val_indices].reset_index(drop=True)

    train_ds = CassavaTrainDataset(
        train_split, get_train_transforms(), config["train_images"]
    )
    val_ds = CassavaTrainDataset(val_split, get_transforms(), config["train_images"])

    train_dl = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=config["finetune_batch_size"],
        pin_memory=torch.cuda.is_available(),
        num_workers=min(config["finetune_workers"], os.cpu_count() or 1),
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=config["finetune_batch_size"],
        pin_memory=torch.cuda.is_available(),
        num_workers=min(config["finetune_workers"], os.cpu_count() or 1),
        drop_last=False,
    )
    print(
        f"Finetune split: train={len(train_split)} val={len(val_split)} (val_frac={val_frac})"
    )
    return train_dl, val_dl




## === cell 10
def infer(model, dataloader):
    print("Running inference...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device, non_blocking=torch.cuda.is_available())
            batch_hat = model(batch)
            batch_prob = torch.softmax(batch_hat, dim=1)
            predictions.append(batch_prob.detach().cpu())

    return torch.cat(predictions, dim=0)


def _evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    seen = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=torch.cuda.is_available())
            y = y.to(device, non_blocking=torch.cuda.is_available())
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().detach().cpu())
            seen += int(y.numel())
    return correct / max(seen, 1)


def finetune_if_needed(model, loaded_ckpt_path, loaded_ckpt_shape_ratio: float):
    if loaded_ckpt_path is not None and not config.get(
        "finetune_even_if_ckpt_loaded", True
    ):
        print("Checkpoint provided; skipping finetune.")
        return model

    if loaded_ckpt_path is not None and config.get(
        "finetune_even_if_ckpt_loaded", True
    ):
        thr = float(config.get("finetune_ckpt_shape_ratio_threshold", 0.0))
        if loaded_ckpt_shape_ratio >= thr:
            print(
                f"Checkpoint provided and looks compatible enough (shape_ratio={loaded_ckpt_shape_ratio:.3f} >= {thr:.3f}); skipping finetune."
            )
            return model
        else:
            print(
                f"Checkpoint provided but looks weak/partial (shape_ratio={loaded_ckpt_shape_ratio:.3f} < {thr:.3f}); running short finetune to improve generalization."
            )

    if not config.get("finetune", True):
        print(
            "Finetune disabled; proceeding without cassava-specific training (likely low score)."
        )
        return model

    train_dl, val_dl = get_train_val_dataloaders()
    model.train()

    for p in model.parameters():
        p.requires_grad = False
    for p in model.net.classifier.parameters():
        p.requires_grad = True

    if config.get("finetune_unfreeze_last_blocks", 0) >= 1:
        if hasattr(model.net, "features") and len(model.net.features) > 0:
            for p in model.net.features[-1].parameters():
                p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Finetune trainable params:", sum(p.numel() for p in trainable_params))

    optimizer = torch.optim.AdamW(trainable_params, lr=config["finetune_lr"])
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(config["finetune_epochs"]))
    )

    best_acc = -1.0
    best_state = None

    for epoch in range(config["finetune_epochs"]):
        running_loss = 0.0
        correct = 0
        seen = 0
        pbar = tqdm(
            train_dl, desc=f"Finetune epoch {epoch+1}/{config['finetune_epochs']}"
        )
        model.train()
        for x, y in pbar:
            x = x.to(device, non_blocking=torch.cuda.is_available())
            y = y.to(device, non_blocking=torch.cuda.is_available())

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().detach().cpu())
            seen += int(x.size(0))
            pbar.set_postfix(
                loss=running_loss / max(seen, 1),
                acc=correct / max(seen, 1),
                lr=float(optimizer.param_groups[0]["lr"]),
            )

        scheduler.step()

        if val_dl is not None:
            val_acc = _evaluate_accuracy(model, val_dl)
            print(f"Validation accuracy after epoch {epoch+1}: {val_acc:.5f}")
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        print(f"Loaded best finetune epoch by val acc: {best_acc:.5f}")

    model.eval()
    return model




## === cell 11
def set_tta_for_epoch(epoch_idx: int):
    t = config["inference_augmentations"]

    def _drop(names):
        return [x for x in t if x.get("name") not in set(names)]

    def _insert_before_normalize(extra_ops):
        base = _drop(["HorizontalFlip", "VerticalFlip", "Transpose", "RandomRotate90"])
        out = []
        for op in base:
            if op["name"] == "Normalize":
                out.extend(extra_ops)
            out.append(op)
        return out

    mode = epoch_idx % 4
    extra = []
    if mode == 1:
        extra = [dict(name="HorizontalFlip", params=dict(p=1.0))]
    elif mode == 2:
        extra = [dict(name="VerticalFlip", params=dict(p=1.0))]
    elif mode == 3:
        extra = [dict(name="Transpose", params=dict(p=1.0))]

    config["inference_augmentations"] = _insert_before_normalize(extra)




## === cell 12
if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model, ckpt_path, ckpt_shape_ratio = load_model()
    model = finetune_if_needed(model, ckpt_path, ckpt_shape_ratio)

    predictions = None
    print("Inferring experiment", config["experiment_name"])

    for epoch in range(config["epochs"]):
        set_tta_for_epoch(epoch)
        dataloader = get_dataloader()

        print("Epoch (TTA pass):", epoch)
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

    pred_labels = np.argmax(results, axis=-1).astype(int)

    sub = sample.copy()
    sub["label"] = pred_labels
    sub.to_csv("submission.csv", index=False)
    print(sub.head())
    print("Wrote submission.csv with rows:", len(sub))
