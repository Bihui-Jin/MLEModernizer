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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.8230583257781807

# 6. Current score

0.48019

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I fix the immediate runtime blockers by (1) making the code robust to the absence of the external `.pth` checkpoint dataset and (2) ensuring `model` is always defined so inference can run and a valid `submission.csv` is written. Because your current score is “Not yielded”, the priority is to produce a valid end-to-end pipeline; if the checkpoint is missing, I fall back to a standard torchvision ResNet-50 backbone with a 5-class head so predictions can still be generated. I also fix the checkpoint loading logic to correctly handle either a full saved `nn.Module` or a `state_dict` (with safe, minimal assumptions). These changes keep the inference/preprocessing loop semantics the same and only add a necessary fallback to make the notebook executable in this environment.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, and the main reason is that the current script likely runs with an untrained random ResNet50 (or partially mismatched checkpoint), plus a preprocessing mismatch (mean/std and resize) that can destroy accuracy for a pretrained cassava model. I make the smallest changes that keep the same core inference loop and model family: (1) use ImageNet normalization (the most common for ResNet checkpoints) and (2) load the checkpoint more robustly by stripping common prefixes (e.g., `module.`) and inferring the correct `num_classes` from the checkpoint’s `fc.weight` when possible. These changes should substantially increase accuracy toward your target without changing the architecture/training approach (still ResNet50 + linear head, single-image loop, argmax). The script still always produce a valid `submission.csv`.'
- What this solution (achieved 0.3722) has done: 'Your current score is far below the target, which strongly suggests the checkpoint isn’t being loaded into the same architecture it was trained with (so you’re effectively running a near-random head). I keep your ResNet50 + argmax inference core unchanged, but make checkpoint loading stricter and more compatible: (1) build the model with the correct `num_classes` inferred from the checkpoint, (2) correctly handle common head key variants (`fc.*`, `classifier.*`, `head.*`) by remapping them to `fc.*`, and (3) if the checkpoint indicates ImageNet pretraining (missing backbone weights), initialize the backbone with ImageNet weights instead of `weights=None`. These are minimal, directly score-relevant changes that should move accuracy substantially upward toward your target while still producing the same `submission.csv` format.'
- What this solution (achieved 0.15695) has done: 'Your current score is far below target, so the most likely issue is a mismatch between the checkpoint’s expected input preprocessing and the model you’re running at inference. I keep your ResNet50+argmax inference exactly the same, but add a minimal, safe test-time augmentation (TTA) path (original + horizontal flip) to improve accuracy without changing training or architecture. I also make the checkpoint head remapping slightly more compatible by handling `*.fc.*` and `*.classifier.*` keys (common in wrappers) while still preferring strict loading when possible. These changes are directly score-relevant and should move you closer to the target band while still producing a valid `submission.csv`.'
- What this solution (achieved 0.13976) has done: 'Your score is far below the target, so we should increase accuracy with minimal, score-relevant fixes while keeping your ResNet50 + argmax inference core unchanged. The biggest likely issue is input preprocessing mismatch: cassava ResNet50 checkpoints commonly use a centered-crop style eval pipeline (resize shorter side then center crop), not a hard `Resize((512,512))` which distorts aspect ratio and can tank accuracy. I change preprocessing to `Resize(512) + CenterCrop(512)` (and the same for the deterministic hflip path) while keeping the same normalization, image size, model, and TTA averaging. I also add an optional, safe softmax-averaging switch (default on) because averaging probabilities is typically better calibrated than averaging logits, without changing the final argmax semantics.'
- What this solution (achieved 0.55493) has done: 'Your current score is far below the target, so we should improve accuracy with minimal, score-relevant changes while keeping your ResNet50 + argmax inference core unchanged. The biggest likely issue is still a checkpoint/model mismatch: many cassava ResNet50 checkpoints are saved with a different head name (e.g., `classifier`/`head`), and sometimes include wrapper prefixes; if the head weights don’t load into `fc`, predictions collapse. I make the head-key remapping slightly more comprehensive (handle `*.classifier.*`, `*.head.*`, and common `backbone.*` prefixes) and also ensure we apply the ResNet50 recommended inference resize/crop exactly (Resize to 512 shorter side + CenterCrop 512 is already correct, we just make interpolation explicitly match common training setups). These are small, low-risk compatibility changes intended to load the intended trained weights and move accuracy upward toward your target, without changing the model family, prediction method, or submission format.'
- What this solution (achieved 0.10575) has done: 'Your current score (0.55493) is far below the target (0.82306), so we should improve accuracy with the smallest changes that keep your ResNet50 + argmax + (hflip) TTA inference logic intact. The most likely remaining gap is still checkpoint incompatibility: many Cassava ResNet50 checkpoints are saved as a nested dict and/or under keys like `model_state_dict`, `state_dict_ema`, and the classifier weights may be named `classifier.*`/`head.*` and wrapped with extra prefixes; if we don’t load the real head/backbone weights, accuracy collapses. I minimally extend the checkpoint parsing to detect these common containers, strip prefixes repeatedly (not just when *all* keys share a prefix), and remap head keys more robustly while keeping strict loading when possible. This should move you materially upward toward the target without changing model family, preprocessing intent, or prediction semantics, and it still always write a valid `submission.csv`.'
- What this solution (achieved 0.46375) has done: 'Your score is far below the target, so we should increase accuracy with the smallest changes that keep your ResNet50 + (hflip) TTA + argmax inference core intact. The biggest likely issue now is that the loaded checkpoint’s classification head is still not being mapped into `model.fc` (common when checkpoints are saved with keys like `fc.1.weight`, `classifier.1.weight`, etc.), causing near-random predictions. I extend the head remapping to also handle these “indexed head” patterns and improve `num_classes` inference accordingly, while keeping the same preprocessing and inference loop. I also print a brief load summary (missing/unexpected counts) to verify that backbone + head weights actually load, without changing submission format.'
- What this solution (achieved 0.15845) has done: 'The score gap to your target is large, so we should focus on one likely root cause: the checkpoint weights still aren’t being loaded into the exact torchvision ResNet50 parameter names (especially if the checkpoint comes from a `timm`/custom ResNet50 where blocks are named `stem.*`, `stages.*`, etc.), leaving most of the backbone random and accuracy low. I keep your exact inference core (ResNet50 + same preprocessing + hflip TTA + argmax) but add a minimal, score-relevant checkpoint adapter that detects common `timm`-style ResNet keys and remaps them to torchvision’s naming so the backbone and head actually load. I also make the “fc already present” early-return in head remapping more precise (so it doesn’t skip remapping when the checkpoint only has `fc.1.*` but not `fc.weight`). These are small compatibility changes intended to move accuracy upward toward your target while preserving your pipeline semantics and still writing a valid `submission.csv`.'
- What this solution (achieved 0.16143) has done: 'Your score is far below the target, so the most likely issue is still that the checkpoint is not being loaded into the same parameter naming scheme as the model you build (so you’re effectively running an ImageNet/random backbone + random head). I keep your exact ResNet50 + preprocessing + (hflip) TTA + argmax inference core unchanged, and only tighten checkpoint compatibility: (1) better detect timm-style “layerX.Y.*” vs “layerX.Y.*” (including `blocks.*` inside `layer*`) and (2) add a small, safe ResNet “stem” remap variant (`stem.conv1.*`, `stem.bn1.*`) seen in some checkpoints. I also add a small diagnostic that prints how many backbone keys actually loaded (without changing outputs), so you can confirm whether you’re still running mostly-uninitialized weights.'
- What this solution (achieved 0.12855) has done: 'Your current score (0.16143) is far below the target (0.82306), so we should improve accuracy with the smallest changes that keep your ResNet50 + preprocessing + (hflip) TTA + argmax inference core intact. The most likely remaining issue is still checkpoint incompatibility: many cassava ResNet50 weights come from timm-style ResNet implementations where key names differ substantially from torchvision (e.g., `conv1` vs `conv1.0`, `act1`, `bn1.num_batches_tracked`, `fc` vs `head.fc`, etc.), so most of the backbone may not actually load. I add a minimal, conservative key-adapter for common timm ResNet patterns (including `conv1.0/conv1.1/conv1.2` and `act1` handling) and also set the model’s `fc` to `Identity()` when the checkpoint already contains a separate `head.*` classifier (so we don’t double-classify). These are compatibility-only changes intended to actually load the trained weights and move the score upward toward your target without changing inference semantics or submission format.'
- What this solution (achieved 0.31016) has done: 'Your current score (0.12855) is far below the target (0.82306), so we need a real accuracy lift with minimal, core-logic-preserving changes. The most likely cause is still that the checkpoint is not being loaded into the torchvision ResNet50 naming (so you’re effectively running mostly ImageNet/random weights), so I’m adding a conservative, ResNet-specific key adapter for the most common remaining timm patterns (notably `act1`→`relu`, `bn1.num_batches_tracked` handling, and `fc` nested under `head.classifier`/`classifier.fc`). I also make `num_classes` inference and head remapping catch these additional suffixes so the trained classifier actually lands in `model.fc`. These are strictly compatibility changes: same model family (resnet50), same preprocessing, same hflip TTA + argmax semantics, same submission writing.'
- What this solution (achieved 0.48019) has done: 'Your current score (0.31016) is far below the target (0.82306), so we should improve accuracy with the smallest changes that don’t alter your core model/inference logic (ResNet50 + preprocessing + hflip TTA + argmax). The most likely remaining issue is that your checkpoint is actually an EfficientNet (or other timm model) despite the filename, so your ResNet50 loader silently fails to load most weights and you end up with near-random predictions; I add a minimal “architecture sniff” to build the right timm model only when the checkpoint clearly indicates it, otherwise keeping your exact ResNet50 path. I also ensure we apply the correct timm default normalization/input-size only in that timm branch (so ResNet behavior stays unchanged), and keep the same submission writing and ordering. These changes are directly score-relevant (proper weight loading + matching preprocessing) and should move you materially toward the target band without changing the overall pipeline semantics.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms, models



## === cell 1
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

MODEL_PATH = (
    "/kaggle/input/resnet50_70_512x512/pytorch/default/1/Resnet50_70_512x512.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_model_preprocess = transforms.Compose(
    [
        transforms.Resize(
            512,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

main_model_preprocess_hflip = transforms.Compose(
    [
        transforms.Resize(
            512,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.CenterCrop(512),
        transforms.Lambda(lambda im: transforms.functional.hflip(im)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

USE_SOFTMAX_TTA_AVG = True




## === cell 2
def _find_checkpoint_fallback():
    candidates = glob.glob("/kaggle/input/**/*.pth", recursive=True)
    if not candidates:
        return None

    def score(p):
        s = p.lower()
        sc = 0
        for token, w in [
            ("cassava", 5),
            ("resnet50", 5),
            ("resnet", 2),
            ("512", 2),
            ("leaf", 1),
            ("disease", 1),
        ]:
            if token in s:
                sc += w
        sc -= 0.0001 * len(p)
        return sc

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]


def _strip_state_dict_prefixes(state_dict: dict) -> dict:
    """
    Score-relevant compatibility fix:
    Strip common wrappers/prefixes key-by-key (iteratively) so weights land on expected names.
    """
    if not state_dict:
        return state_dict

    prefixes = (
        "module.",
        "model.",
        "net.",
        "backbone.",
        "encoder.",
    )

    new_sd = {}
    for k, v in state_dict.items():
        if not isinstance(k, str):
            new_sd[k] = v
            continue
        kk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if kk.startswith(p):
                    kk = kk[len(p) :]
                    changed = True
        new_sd[kk] = v
    return new_sd


def _remap_timm_resnet_to_torchvision(state_dict: dict) -> dict:
    """
    Score-relevant compatibility fix:
    Remap common timm/custom ResNet key patterns to torchvision's naming.
    """
    if not state_dict:
        return state_dict

    keys = [k for k in state_dict.keys() if isinstance(k, str)]
    has_timm_stage = any(k.startswith("stages.") for k in keys)
    has_timm_stem = any(k.startswith("stem.") for k in keys)
    has_layer_blocks = any(".blocks." in k and k.startswith("layer") for k in keys)

    has_timm_conv_stem = any(k.startswith("conv1.") for k in keys) and any(
        k.startswith("bn1.") for k in keys
    )
    has_timm_act = any(k.startswith("act1.") or k == "act1" for k in keys)

    has_head_classifier = any(k.startswith("head.classifier.") for k in keys) or any(
        k.endswith(".head.classifier.weight") for k in keys
    )

    if not (
        has_timm_stage
        or has_timm_stem
        or has_layer_blocks
        or has_timm_conv_stem
        or has_timm_act
        or has_head_classifier
    ):
        return state_dict

    def map_key(k: str) -> str:
        kk = k

        if kk.startswith("stem.conv."):
            kk = "conv1." + kk[len("stem.conv.") :]
        if kk.startswith("stem.bn."):
            kk = "bn1." + kk[len("stem.bn.") :]

        if kk.startswith("stem.conv1."):
            kk = "conv1." + kk[len("stem.conv1.") :]
        if kk.startswith("stem.bn1."):
            kk = "bn1." + kk[len("stem.bn1.") :]

        if kk.startswith("conv1.0."):
            kk = "conv1." + kk[len("conv1.0.") :]
        if kk.startswith("conv1.1."):
            kk = "bn1." + kk[len("conv1.1.") :]

        if kk.startswith("act1."):
            kk = "relu." + kk[len("act1.") :]
        if kk == "act1":
            kk = "relu"

        if kk.startswith("stages."):
            parts = kk.split(".")
            if len(parts) >= 5 and parts[2] == "blocks":
                try:
                    stage_i = int(parts[1])
                    block_j = int(parts[3])
                    rest = ".".join(parts[4:])
                    kk = f"layer{stage_i+1}.{block_j}." + rest
                except Exception:
                    pass

        if kk.startswith("layer") and ".blocks." in kk:
            parts = kk.split(".")
            if len(parts) >= 5 and parts[2] == "blocks":
                try:
                    layer_name = parts[0]  # layer1..layer4
                    block_j = int(parts[1])
                    rest = ".".join(parts[4:])
                    kk = f"{layer_name}.{block_j}." + rest
                except Exception:
                    pass

        kk = kk.replace(".downsample.conv.", ".downsample.0.")
        kk = kk.replace(".downsample.bn.", ".downsample.1.")
        kk = kk.replace(".shortcut.conv.", ".downsample.0.")
        kk = kk.replace(".shortcut.bn.", ".downsample.1.")

        if kk.startswith("head.fc."):
            kk = "fc." + kk[len("head.fc.") :]
        if kk.startswith("head.classifier."):
            kk = "fc." + kk[len("head.classifier.") :]
        if kk.startswith("head."):
            kk = kk[len("head.") :]

        return kk

    new_sd = {}
    for k, v in state_dict.items():
        if isinstance(k, str):
            new_sd[map_key(k)] = v
        else:
            new_sd[k] = v
    return new_sd


def _remap_common_head_keys_to_fc(state_dict: dict) -> dict:
    """
    Score-relevant compatibility fix:
    Remap various classifier head key names into torchvision's "fc.*".
    """
    if not state_dict:
        return state_dict

    if ("fc.weight" in state_dict) or ("fc.bias" in state_dict):
        return state_dict

    keys = [k for k in state_dict.keys() if isinstance(k, str)]

    suffix_candidates = [
        ("fc.weight", "fc.bias"),
        ("fc.1.weight", "fc.1.bias"),
        ("classifier.weight", "classifier.bias"),
        ("classifier.1.weight", "classifier.1.bias"),
        ("head.weight", "head.bias"),
        ("head.1.weight", "head.1.bias"),
        ("head.fc.weight", "head.fc.bias"),
        ("head.classifier.weight", "head.classifier.bias"),
        ("last_linear.weight", "last_linear.bias"),
        ("output.weight", "output.bias"),
        ("logits.weight", "logits.bias"),
    ]

    for w_suf, b_suf in suffix_candidates:
        w_keys = [k for k in keys if k.endswith(w_suf)]
        if w_keys:
            w_key = sorted(w_keys, key=len)[0]
            b_key = w_key[: -len(w_suf)] + b_suf

            new_sd = dict(state_dict)
            new_sd["fc.weight"] = new_sd.pop(w_key)
            if b_key in new_sd:
                new_sd["fc.bias"] = new_sd.pop(b_key)
            return new_sd

    for prefix in (
        "classifier.",
        "head.",
        "head.fc.",
        "head.classifier.",
        "last_linear.",
        "output.",
        "logits.",
        "fc.",
    ):
        for idx in ("", "1."):
            w_key = prefix + idx + "weight"
            b_key = prefix + idx + "bias"
            if w_key in state_dict:
                new_sd = dict(state_dict)
                new_sd["fc.weight"] = new_sd.pop(w_key)
                if b_key in new_sd:
                    new_sd["fc.bias"] = new_sd.pop(b_key)
                return new_sd

    return state_dict


def _infer_num_classes_from_state_dict(state_dict: dict, default: int = 5) -> int:
    if not state_dict:
        return default

    w = state_dict.get("fc.weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[0] >= 2:
        return int(w.shape[0])

    candidates = (
        ".fc.weight",
        ".fc.1.weight",
        ".classifier.weight",
        ".classifier.1.weight",
        ".head.weight",
        ".head.1.weight",
        ".head.fc.weight",
        ".head.classifier.weight",
        ".last_linear.weight",
    )
    for kk, vv in state_dict.items():
        if not isinstance(kk, str) or not isinstance(vv, torch.Tensor):
            continue
        if kk.endswith(candidates) and vv.ndim == 2 and vv.shape[0] >= 2:
            return int(vv.shape[0])

    return default


def _should_use_imagenet_backbone(state_dict: dict) -> bool:
    if not state_dict:
        return True
    return ("conv1.weight" not in state_dict) and (
        "layer1.0.conv1.weight" not in state_dict
    )


def _build_fallback_model(num_classes=5, use_imagenet_backbone=False):
    weights = models.ResNet50_Weights.DEFAULT if use_imagenet_backbone else None
    m = models.resnet50(weights=weights)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m


def _extract_state_dict_from_checkpoint(ckpt: object):
    """
    Score-relevant compatibility fix:
    Prefer EMA if present; otherwise pick the most plausible tensor dict.
    """
    if isinstance(ckpt, dict):
        preferred_keys = [
            "state_dict_ema",
            "ema_state_dict",
            "model_ema",
            "model_state_dict",
            "state_dict",
            "model",
            "net",
            "weights",
        ]
        for k in preferred_keys:
            v = ckpt.get(k, None)
            if (
                isinstance(v, dict)
                and v
                and all(isinstance(kk, str) for kk in v.keys())
                and any(isinstance(vv, torch.Tensor) for vv in v.values())
            ):
                return v
        if ckpt and all(isinstance(kk, str) for kk in ckpt.keys()):
            if any(isinstance(vv, torch.Tensor) for vv in ckpt.values()):
                return ckpt
    return None


def _print_load_diagnostics(model, state_dict, incompatible=None):
    """
    Score-relevant debugging aid (no effect on predictions).
    """
    if state_dict is None:
        print("Load diagnostics: state_dict is None.")
        return
    sd_keys = set(k for k in state_dict.keys() if isinstance(k, str))
    model_keys = set(model.state_dict().keys())
    overlap = len(sd_keys & model_keys)
    print(
        f"Load diagnostics: state_dict keys={len(sd_keys)}, model keys={len(model_keys)}, overlap={overlap}."
    )
    if incompatible is not None:
        missing = getattr(incompatible, "missing_keys", None) or []
        unexpected = getattr(incompatible, "unexpected_keys", None) or []
        print(
            f"Load diagnostics: missing={len(missing)}, unexpected={len(unexpected)}."
        )
        anchors = [
            "conv1.weight",
            "bn1.weight",
            "layer1.0.conv1.weight",
            "layer4.2.conv3.weight",
            "fc.weight",
        ]
        present = [a for a in anchors if a in sd_keys]
        print(f"Load diagnostics: anchor keys present in ckpt: {present}")


def _looks_like_efficientnet_or_timm_non_resnet(state_dict: dict) -> bool:
    """
    Score-relevant fix:
    Detect when a .pth is NOT a torchvision-ResNet-style checkpoint (often EfficientNet/timm),
    because forcing it into a ResNet50 leaves most weights unloaded and accuracy collapses.
    We only switch behavior if strong indicators are present.
    """
    if not state_dict:
        return False
    keys = [k for k in state_dict.keys() if isinstance(k, str)]
    eff_indicators = (
        any(k.startswith("conv_stem.") for k in keys)
        or any(
            k.startswith("bn1.") is False and k.startswith("bn1") is False for k in []
        )  # no-op, keep minimal
        or any("blocks." in k and "bn" in k for k in keys)  # common in efficientnet
        or any(
            k.startswith("classifier.") for k in keys
        )  # timm heads often classifier.*
        or any(k.startswith("features.") for k in keys)  # torchvision efficientnet
        or any(
            k.startswith("encoder.") and "blocks" in k for k in keys
        )  # some wrappers
    )
    strong = any(k.startswith("conv_stem.") for k in keys) and any(
        k.startswith("blocks.") for k in keys
    )
    looks_resnet = any(k.startswith("layer1.") for k in keys) and any(
        k.startswith("layer4.") for k in keys
    )
    return (strong or eff_indicators) and (not looks_resnet)


def _try_build_timm_efficientnet_from_state_dict(state_dict: dict, num_classes: int):
    """
    Score-relevant fix:
    If timm is available and checkpoint is EfficientNet-like, build the matching timm model.
    This keeps the rest of the pipeline (TTA+argmax) identical but allows real weight loading.
    """
    try:
        import timm  # noqa: F401
    except Exception:
        return None, None, None

    import timm

    candidates = [
        "tf_efficientnet_b4_ns",
        "tf_efficientnet_b4",
        "efficientnet_b4",
        "tf_efficientnet_b5_ns",
        "tf_efficientnet_b5",
        "efficientnet_b5",
        "tf_efficientnet_b3_ns",
        "tf_efficientnet_b3",
        "efficientnet_b3",
    ]

    best = None
    best_name = None
    best_incompat = None

    for name in candidates:
        try:
            m = timm.create_model(name, pretrained=False, num_classes=num_classes)
        except Exception:
            continue
        try:
            incompat = m.load_state_dict(state_dict, strict=False)
        except Exception:
            continue
        missing = getattr(incompat, "missing_keys", None) or []
        unexpected = getattr(incompat, "unexpected_keys", None) or []
        score = len(missing) + len(unexpected)
        if best is None or score < best:
            best = score
            best_name = name
            best_incompat = incompat
            best_model = m

    if best_name is None:
        return None, None, None

    cfg = best_model.default_cfg if hasattr(best_model, "default_cfg") else {}
    return best_model, best_name, cfg


def _build_timm_preprocess_from_cfg(cfg: dict, img_size_fallback: int = 512):
    """
    Score-relevant fix:
    Use timm model's expected normalization & input-size when we detect timm model usage.
    This avoids preprocessing mismatch that can heavily reduce accuracy.
    """
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))

    input_size = cfg.get("input_size", None)  # e.g., (3, 380, 380)
    if isinstance(input_size, (list, tuple)) and len(input_size) == 3:
        img_size = int(input_size[1])
    else:
        img_size = int(img_size_fallback)

    p1 = transforms.Compose(
        [
            transforms.Resize(
                img_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )
    p2 = transforms.Compose(
        [
            transforms.Resize(
                img_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(img_size),
            transforms.Lambda(lambda im: transforms.functional.hflip(im)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )
    return p1, p2


if not os.path.exists(MODEL_PATH):
    alt = _find_checkpoint_fallback()
    if alt is not None:
        print(f"MODEL_PATH not found. Using fallback checkpoint: {alt}")
        MODEL_PATH = alt
    else:
        print(
            f"Warning: MODEL_PATH not found: {MODEL_PATH}\n"
            f"Also did not find any .pth under /kaggle/input.\n"
            f"Proceeding with an untrained fallback ResNet50 model so a valid submission.csv is produced."
        )
        MODEL_PATH = None

model = None
_using_timm = False

if MODEL_PATH is not None:
    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
        print("Loaded checkpoint as a full nn.Module.")
    else:
        state_dict = _extract_state_dict_from_checkpoint(ckpt)

        if state_dict is not None:
            state_dict = _strip_state_dict_prefixes(state_dict)

            num_classes_guess = _infer_num_classes_from_state_dict(
                state_dict, default=5
            )
            if _looks_like_efficientnet_or_timm_non_resnet(state_dict):
                timm_model, timm_name, timm_cfg = (
                    _try_build_timm_efficientnet_from_state_dict(
                        state_dict, num_classes=num_classes_guess
                    )
                )
                if timm_model is not None:
                    model = timm_model
                    _using_timm = True
                    main_model_preprocess, main_model_preprocess_hflip = (
                        _build_timm_preprocess_from_cfg(timm_cfg, img_size_fallback=512)
                    )
                    print(
                        f"Loaded checkpoint via timm model: {timm_name} (num_classes={num_classes_guess})."
                    )
                    missing = getattr(timm_cfg, "missing_keys", None)
                else:
                    print(
                        "Checkpoint looks non-ResNet, but timm load was unavailable/failed; falling back to ResNet path."
                    )

            if model is None:
                state_dict = _remap_timm_resnet_to_torchvision(state_dict)
                state_dict = _remap_common_head_keys_to_fc(state_dict)

                num_classes = _infer_num_classes_from_state_dict(state_dict, default=5)
                use_imagenet = _should_use_imagenet_backbone(state_dict)

                model = _build_fallback_model(
                    num_classes=num_classes, use_imagenet_backbone=use_imagenet
                )

                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if not (isinstance(k, str) and k.endswith("num_batches_tracked"))
                }

                try:
                    model.load_state_dict(state_dict, strict=True)
                    print(
                        f"Loaded state_dict with strict=True (num_classes={num_classes}, imagenet_backbone={use_imagenet})."
                    )
                    _print_load_diagnostics(model, state_dict, incompatible=None)
                except Exception as e:
                    print(
                        f"Warning: strict=True load failed ({type(e).__name__}: {e}). Falling back to strict=False."
                    )
                    incompatible = model.load_state_dict(state_dict, strict=False)
                    missing = getattr(incompatible, "missing_keys", None)
                    unexpected = getattr(incompatible, "unexpected_keys", None)
                    if missing:
                        print(
                            f"Missing keys: {len(missing)} (showing up to 10): {missing[:10]}"
                        )
                    if unexpected:
                        print(
                            f"Unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}"
                        )
                    print(
                        f"Loaded state_dict with strict=False (num_classes={num_classes}, imagenet_backbone={use_imagenet})."
                    )
                    _print_load_diagnostics(
                        model, state_dict, incompatible=incompatible
                    )
        else:
            print(
                "Warning: Unrecognized checkpoint format. Proceeding with an untrained fallback model."
            )
            model = _build_fallback_model(num_classes=5, use_imagenet_backbone=True)

if model is None:
    model = _build_fallback_model(num_classes=5, use_imagenet_backbone=True)

model.to(device)
model.eval()



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
image_ids = sample_sub["image_id"].astype(str).tolist()

prediction = []
missing = 0
failed = 0

softmax = torch.nn.Softmax(dim=1)

with torch.no_grad():
    for image_id in image_ids:
        img_path = os.path.join(TEST_IMG_DIR, image_id)

        if not os.path.exists(img_path):
            missing += 1
            prediction.append(0)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            failed += 1
            prediction.append(0)
            continue

        x1 = main_model_preprocess(img).unsqueeze(0).to(device)
        x2 = main_model_preprocess_hflip(img).unsqueeze(0).to(device)

        logits1 = model(x1)
        logits2 = model(x2)

        if USE_SOFTMAX_TTA_AVG:
            probs = (softmax(logits1) + softmax(logits2)) / 2.0
            pred = int(torch.argmax(probs, dim=1).item())
        else:
            logits = (logits1 + logits2) / 2.0
            pred = int(torch.argmax(logits, dim=1).item())

        prediction.append(pred)

if missing:
    print(f"Warning: {missing} test images were missing. Filled with class 0.")
if failed:
    print(f"Warning: {failed} test images failed to load. Filled with class 0.")

assert len(prediction) == len(
    image_ids
), "Internal error: prediction and image_ids length mismatch."

submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print(f"Wrote submission.csv with {len(submission)} rows.")
print(f"Used timm model branch: {_using_timm}")
