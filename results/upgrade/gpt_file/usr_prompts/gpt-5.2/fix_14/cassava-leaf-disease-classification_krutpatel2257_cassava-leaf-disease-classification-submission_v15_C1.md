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

0.8856

# 6. Current score

0.19768

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24178) has done: 'I first fix the missing pretrained weights issue by loading a locally available ResNeXt50 ImageNet backbone (instead of a non-existent `.pth` path) so the notebook runs end-to-end and produces a valid `submission.csv`. Next, I update the Albumentations augmentation definitions to the v2 API (the current `RandomResizedCrop(256, 256)` signature is invalid) and replace removed transforms (Cutout) with their supported equivalents. Finally, I fix inference correctness/stability issues (proper RGB conversion, `torch.no_grad()`, device handling, and ensuring transforms output is used correctly) so predictions are generated for all test images and written in the required format.'
- What this solution (achieved 0.23468) has done: 'Your score is far below the target, and the main reason is that you’re running a fresh ImageNet ResNeXt with a randomly initialized 5-class head (and you never load the provided fine-tuned checkpoint), so predictions are essentially garbage. The smallest change that should move accuracy sharply upward toward your target is to correctly load the local `model(11).pth` weights into the model (handling common checkpoint key formats) and then run the same TTA inference as you already do. I also ensure the Albumentations output is converted to a float tensor correctly (right now `ToTensor()` can mis-handle already-normalized float arrays), without changing the model or overall approach. The rest of your pipeline (TTA loop, argmax, submission formatting) stays the same.'
- What this solution (achieved 0.24215) has done: 'Your score is far below the target because the fine-tuned checkpoint still isn’t being applied to the model’s 5-class head (most likely the checkpoint stores a different head name/shape), so you’re effectively running an ImageNet backbone with a randomly initialized classifier. I make the smallest change that directly fixes this: load the checkpoint in a “compatible” way by (1) mapping common key prefixes and (2) skipping only the final-layer weights when shapes don’t match, while keeping everything else identical. This preserves your architecture and TTA inference loop, but ensures you actually use the learned cassava features, which should move accuracy sharply upward toward your target band. I also keep preprocessing identical, only ensuring tensor dtype is float for safety.'
- What this solution (achieved 0.19768) has done: 'Your score is far below the target because you’re still doing “train-time style” augmentations at inference (RandomResizedCrop/color jitter/dropout), which destroys accuracy for classification; the smallest change that should move accuracy sharply upward is to switch to a deterministic test-time transform (resize/center-crop + normalize) while keeping your exact model, checkpoint-loading logic, and TTA averaging loop intact. I also set `model.eval()` right before inference (to be safe) and make the checkpoint load slightly stricter for the classifier head by explicitly remapping common `fc`/`classifier` key variants so the fine-tuned 5-class head is more likely to load instead of being skipped. These are minimal changes that preserve core semantics (argmax over averaged logits) but should improve correctness and move accuracy toward your 0.8856 target. The script still runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.19768) has done: 'Your score is far below target, and the most likely cause is that the fine-tuned checkpoint still isn’t actually being applied to the model (so you’re effectively predicting with an ImageNet backbone + mostly random 5-class head). I make the smallest change that directly increases accuracy: load the checkpoint with a more robust key-remapping that handles common ResNeXt `fc`/`head`/`classifier` naming and also avoids accidentally skipping the classifier when it *does* match. I also print a quick sanity check that the `fc.weight` tensor actually changes after loading (to confirm the checkpoint took effect) without changing your model, transforms, TTA loop, or submission formatting. Everything else (deterministic test transform, averaging logits across TTA, argmax, `submission.csv`) stays identical.'
- What this solution (achieved 0.19768) has done: 'Your current score is far below the target, so we should increase accuracy with the smallest changes that keep your model and inference logic intact. The main likely issue is that your checkpoint contains the *fine-tuned head*, but your “compat” loader can accidentally skip it if the checkpoint uses different key names (or stores the head under a different attribute), leaving you with an essentially random 5-class classifier. I minimally improve the checkpoint loading to (1) try a strict load after key remapping, and if that fails (2) fall back to shape-compatible loading, while explicitly prioritizing loading the `fc.*` head when present. I also enforce deterministic-ish behavior (without changing training/inference semantics) and keep your deterministic test-time transform + TTA loop unchanged, still writing a valid `submission.csv`.'
- What this solution (achieved 0.19768) has done: 'Your current score is far below the target, so we should increase accuracy with minimal, inference-only fixes that keep your model and TTA loop intact. The biggest likely issue now is that you still may not actually be loading the fine-tuned checkpoint weights due to key mismatches (so you’re effectively using an ImageNet backbone + near-random 5-class head), so I make the loader try several common key mappings (including `model.fc.*` → `fc.*`, `backbone.*` stripping, and `module.model.*` patterns) and ensure the `fc.*` weights load when shapes match. I also remove “fake TTA” by making TTA_n=1 because your test transform is deterministic (so repeating it just adds compute without benefit), which keeps semantics identical but reduces runtime risk. Finally, I ensure the submission rows exactly follow `sample_submission.csv` order and always write a valid `submission.csv`.'
- What this solution (achieved 0.19768) has done: 'Your current score is far below the target, so we should increase accuracy with the smallest changes that keep your model and inference approach intact. The most likely root cause is that the checkpoint is not actually being loaded correctly into this torchvision ResNeXt (key mismatches), leaving you with an ImageNet backbone + effectively random 5‑class head. I minimally improve the checkpoint loader to additionally handle the very common pattern where the saved classifier is named `classifier.*`/`head.*` in the *target model* but needs to be mapped to `fc.*` for torchvision, while also trying both directions safely and confirming whether `fc.*` was truly loaded. Everything else (same model, deterministic test transform, argmax, submission format and ordering) stays the same.'
- What this solution (achieved 0.15247) has done: 'Your score is far below the target, so we should increase accuracy with the smallest inference-only fixes while keeping your model, checkpoint, and argmax-on-logits semantics unchanged. The most likely remaining issue is that the loaded checkpoint is not actually the intended fine-tuned ResNeXt weights (or it’s being partially loaded), so we (1) make checkpoint loading more robust and verifiable by selecting the variant that loads the most keys and explicitly preferring a successfully loaded 5-class `fc.*` head. Next, we align preprocessing with torchvision’s ResNeXt50 ImageNet weights by using the weights’ built-in transforms (same mean/std/resize/crop) instead of a hand-rolled pipeline that can be subtly mismatched. Finally, we ensure test-time ordering exactly matches `sample_submission.csv` (already) and keep TTA at 1 (deterministic), producing a valid `submission.csv`.'
- What this solution (achieved 0.15247) has done: 'Your score is far below the target, so we should increase accuracy with minimal inference-only fixes while keeping your model and argmax-on-logits semantics unchanged. The biggest likely issue is still that the checkpoint isn’t being loaded into the torchvision ResNeXt reliably (so you’re effectively using an ImageNet backbone with an untrained 5-class head), because your current “best variant” selection compares `fc_before` from a different model instance and can fail to select the real fine-tuned weights. I change the checkpoint-loading selection to choose the state_dict variant that loads the most tensors (and specifically prefers variants that successfully load `fc.weight`/`fc.bias`), and I compute the `fc` delta against the reinitialized model within each attempt (so the sanity check and selection are meaningful). This is a small, targeted change that should move accuracy sharply upward toward your 0.8856 target without changing architecture, transforms, or inference logic, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.19768) has done: 'Your current score is far below the target, so we should increase accuracy with the smallest inference-only fixes that keep your model and argmax-on-logits semantics intact. The most likely remaining issue is that you’re applying the ImageNet weights’ built-in preprocessing (which resizes to 232 and center-crops to 224 for ResNeXt) while your fine-tuned checkpoint was almost certainly trained at 256×256 (your Albumentations pipeline), causing a harmful train/test preprocessing mismatch. I always use your existing 256-center-crop normalization pipeline for inference (so preprocessing matches the likely fine-tuning setup), while keeping the same model, checkpoint-loading logic, and submission ordering. I also add a tiny safety fix to ensure the tensor is `float32` on the correct device in all cases.'
- What this solution (achieved 0.19768) has done: 'Your score is far below the target, so we should increase accuracy with the smallest inference-only change that likely fixes a correctness issue. The most probable remaining problem is that the fine-tuned checkpoint contains a different classifier head name (or a full model wrapper) and our current “variant” search is too narrow, so we still end up using an ImageNet backbone with a near-random 5‑class head. I minimally expand the checkpoint key remapping/variant attempts to cover the most common patterns for cassava solutions (`model.*`, `module.*`, `backbone.*`, plus `fc`↔`classifier`/`head` mapping both directions), and select the variant that loads the most tensors while explicitly requiring the 5-class head to be loaded (by checking the actual loaded `fc.weight`/`fc.bias` match the checkpoint tensors). Everything else (model, transform, argmax, submission formatting/order) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.19768) has done: 'Your score is far below the target, so we need a real accuracy boost with minimal semantic change to your existing inference pipeline. The most likely issue is still that `model(11).pth` is not being loaded into the torchvision ResNeXt correctly because the checkpoint may store a *full model object* or a state_dict under uncommon keys; your current loader only explores a few dict-key variants. I minimally extend checkpoint extraction and key normalization to handle (a) checkpoints saved as `nn.Module`, (b) nested keys like `ema`, `student`, `teacher`, and (c) deeper wrapper prefixes, while keeping the same model, same deterministic test transform, same argmax-on-averaged-logits logic, and same submission formatting. I also make the “best variant” selection actually choose the variant that loads the most parameters and prefers loading the classifier head when possible, which should move accuracy substantially upward toward your target band.'

# 9. Code solution

## === cell 0
import os
import random

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models



## === cell 1
model_path = "../input/rn-tta-calr-ft-ofasf/model(11).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 2
try:
    weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
except Exception:
    weights = None

model = models.resnext50_32x4d(weights=weights)
model.fc = nn.Linear(2048, 5)
model = model.to(device)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, torch.nn.Module):
        return ckpt_obj.state_dict()

    if isinstance(ckpt_obj, dict):
        candidate_keys = [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "module",
            "ema",
            "ema_state_dict",
            "student",
            "teacher",
            "swa_model",
            "best_state_dict",
            "weights",
        ]
        for k in candidate_keys:
            if k in ckpt_obj:
                v = ckpt_obj[k]
                if isinstance(v, torch.nn.Module):
                    return v.state_dict()
                if isinstance(v, dict):
                    if any(torch.is_tensor(t) for t in v.values()):
                        return v
                    for k2 in candidate_keys:
                        if (
                            k2 in v
                            and isinstance(v[k2], dict)
                            and any(torch.is_tensor(t) for t in v[k2].values())
                        ):
                            return v[k2]

        if any(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj

    return ckpt_obj


def _strip_prefixes_multi(sd, prefixes):
    for prefix in prefixes:
        sd = {
            (k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in sd.items()
        }
    return sd


def _remap_keys(sd, mapping_rules):
    out = {}
    for k, v in sd.items():
        nk = k
        for src_prefix, dst_prefix in mapping_rules:
            if nk.startswith(src_prefix):
                nk = dst_prefix + nk[len(src_prefix) :]
        out[nk] = v
    return out


def _normalize_common_wrappers(sd):
    return _strip_prefixes_multi(
        sd,
        prefixes=(
            "module.",
            "module.model.",
            "module.backbone.",
            "model.",
            "model.model.",
            "net.",
            "network.",
            "backbone.",
            "encoder.",
            "base_model.",
        ),
    )


def _load_ckpt_try_strict_then_fallback(model_, state_dict_):
    try:
        missing, unexpected = model_.load_state_dict(state_dict_, strict=True)
        return {
            "mode": "strict",
            "missing": list(missing) if missing is not None else [],
            "unexpected": list(unexpected) if unexpected is not None else [],
            "skipped": [],
            "loaded_n": len(state_dict_),
        }
    except Exception as e_strict:
        model_sd = model_.state_dict()
        filtered = {}
        skipped = []
        for k, v in state_dict_.items():
            if k in model_sd and hasattr(v, "shape") and model_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                if k in model_sd and hasattr(v, "shape"):
                    skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        missing, unexpected = model_.load_state_dict(filtered, strict=False)
        return {
            "mode": f"fallback_compat (strict failed: {type(e_strict).__name__})",
            "missing": list(missing),
            "unexpected": list(unexpected),
            "skipped": skipped,
            "loaded_n": len(filtered),
        }


def _sd_has_key(sd, k):
    return isinstance(sd, dict) and (k in sd)


def _head_loaded_exact(m, sd_try):
    if _sd_has_key(sd_try, "fc.weight") and _sd_has_key(sd_try, "fc.bias"):
        with torch.no_grad():
            w_ok = torch.allclose(
                m.fc.weight.detach().cpu(),
                sd_try["fc.weight"].detach().cpu(),
                atol=0,
                rtol=0,
            )
            b_ok = torch.allclose(
                m.fc.bias.detach().cpu(),
                sd_try["fc.bias"].detach().cpu(),
                atol=0,
                rtol=0,
            )
        return bool(w_ok and b_ok)
    return False


def _pick_best_loaded_variant(variants, weights_):
    best = None  # (priority_tuple, name, info, m)
    for name, sd_try in variants:
        m = models.resnext50_32x4d(weights=weights_)
        m.fc = nn.Linear(2048, 5)
        m = m.to(device)

        info = _load_ckpt_try_strict_then_fallback(m, sd_try)

        head_in_sd = _sd_has_key(sd_try, "fc.weight") and _sd_has_key(sd_try, "fc.bias")
        head_loaded = _head_loaded_exact(m, sd_try)

        priority = (
            2 if head_loaded else (1 if head_in_sd else 0),
            int(info["loaded_n"]),
        )
        cand = (priority, name, info, m)
        if best is None or cand[0] > best[0]:
            best = cand

    return best


if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    sd0 = _extract_state_dict(ckpt)

    if isinstance(sd0, dict):
        sd_base = _normalize_common_wrappers(sd0)

        variants = []
        variants.append(("as_is", sd_base))

        to_fc_rules = [
            ("classifier.", "fc."),
            ("head.", "fc."),
            ("last_linear.", "fc."),
            ("logits.", "fc."),
            ("linear.", "fc."),
            ("_fc.", "fc."),
            ("final.", "fc."),
        ]
        variants.append(("map_to_fc", _remap_keys(sd_base, to_fc_rules)))

        variants.append(
            ("map_modelfc_to_fc", _remap_keys(sd_base, [("model.fc.", "fc.")]))
        )
        variants.append(
            ("map_backbonefc_to_fc", _remap_keys(sd_base, [("backbone.fc.", "fc.")]))
        )

        variants.append(
            (
                "strip_model_backbone",
                _strip_prefixes_multi(
                    sd_base,
                    prefixes=(
                        "model.backbone.",
                        "backbone.model.",
                    ),
                ),
            )
        )

        best = _pick_best_loaded_variant(variants, weights)
        _, best_name, info, model = best

        print("Loaded checkpoint:", model_path)
        print("Key remap variant used:", best_name)
        print("Load mode:", info["mode"])
        print(
            "Loaded keys:",
            info["loaded_n"],
            "Missing keys:",
            len(info["missing"]),
            "Unexpected keys:",
            len(info["unexpected"]),
            "Skipped (shape-mismatch):",
            len(info["skipped"]),
        )
        if len(info["skipped"]) > 0:
            print("Example skipped keys:", info["skipped"][:5])

        sd_used = dict(variants)[best_name]
        print(
            "Sanity check: fc.* loaded exactly from checkpoint:",
            _head_loaded_exact(model, sd_used),
        )
    else:
        print(
            "WARNING: checkpoint format not understood; running without fine-tuned weights."
        )
else:
    print(
        "WARNING: model_path does not exist, running without fine-tuned weights:",
        model_path,
    )

model.eval()



## === cell 3
from albumentations.pytorch import ToTensorV2

sub_aug = A.Compose(
    [
        A.SmallestMaxSize(max_size=256, p=1.0),
        A.CenterCrop(height=256, width=256, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)



## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

predictions = []

tta_n = 1  # deterministic preprocess; repeating adds no benefit

model.eval()
with torch.no_grad():
    for _, sample_row in sample_sub.iterrows():
        image_path = os.path.join(test_images_path, sample_row.image_id)
        image = Image.open(image_path).convert("RGB")

        image_pred = None
        for _ in range(tta_n):
            image_np = np.array(image)
            aug_out = sub_aug(image=image_np)

            x = aug_out["image"].to(device).float()

            outputs = model(x.unsqueeze(0))
            if image_pred is None:
                image_pred = outputs
            else:
                image_pred += outputs

        image_pred /= tta_n
        pred_label = int(torch.argmax(image_pred, dim=1).item())
        predictions.append(pred_label)

sub_df = sample_sub.copy()
sub_df["label"] = predictions
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with", len(sub_df), "rows")
