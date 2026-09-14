# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8278936234511937

# 6. Current score

0.10987

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24664) has done: 'I fix the missing model file issue by falling back to a torchvision pretrained SqueezeNet (same architecture/head) when the provided `.pth` path doesn’t exist, so the notebook runs end-to-end and still yields reasonable accuracy. I also update the Albumentations augmentation definitions to match albumentations==2.x API (notably `RandomResizedCrop` and deprecated transforms like `Cutout`), which is why `sub_aug` never got defined. Finally, I make loading weights device-safe with `map_location`, ensure images are converted to RGB, and keep the same TTA averaging logic while producing a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.27205) has done: 'Your current score is low mainly because the inference-time augmentation is extremely strong and stochastic (random crops, heavy color jitter, dropout), which can destroy class evidence and make logits noisy even with TTA. To move accuracy toward your target with minimal semantic change, I keep the same model and TTA averaging loop but switch the submission-time transform to a deterministic, evaluation-appropriate pipeline (resize/center-crop + normalize) so that each of the 10 TTA passes becomes stable rather than randomly destructive. I also fix a subtle but important normalization bug: you’re applying Albumentations `Normalize` and then `transforms.ToTensor()` (which divides by 255 again), so inputs are mis-scaled; I instead convert to tensor directly from the already-normalized numpy image. Finally, I run the model under `torch.inference_mode()` to avoid any overhead while keeping outputs identical.'
- What this solution (achieved 0.33819) has done: 'Your score is far below the target, so we should improve accuracy with the smallest changes that keep the same model and the same 10-pass TTA averaging logic. The biggest remaining issue is that although the transform is now deterministic, the model is still receiving inputs in the wrong numeric range/dtype: Albumentations `Normalize` produces float32, but the current conversion can yield float64 and (depending on upstream) can be inconsistent; we force `float32` and ensure the tensor is contiguous. We also make sure the loaded checkpoint format is handled robustly (`state_dict` vs full checkpoint dict) to actually load the trained weights when present; failing to load correct weights is a common reason for ~0.27 accuracy. These changes preserve the architecture, inference approach, and evaluation semantics, and should move accuracy materially toward your target.'
- What this solution (achieved 0.15022) has done: 'Your current score (0.33819) is far below the target (0.82789), and the most likely cause is that the model weights you think you’re loading aren’t actually being applied because the checkpoint keys don’t exactly match (common with `model.` prefixes, `module.` prefixes, or different classifier naming). I keep the exact same architecture and 10-pass TTA averaging loop, but make checkpoint loading robust by (1) trying multiple common key-prefix cleanups and (2) allowing `strict=False` only as a fallback while reporting how many keys matched, so you actually get the best-available weights instead of effectively random-ish ones. I also ensure the input tensor has the expected shape/dtype and avoid any accidental dtype/device mismatch, without changing the augmentation semantics (still deterministic center-crop). These are minimal, directly score-relevant changes that should materially increase accuracy toward your target.'
- What this solution (achieved 0.0852) has done: 'Your score is far below the target, so the most likely “minimal but big impact” issue is that the checkpoint still isn’t being applied correctly (so you’re effectively using weak/random weights). I keep the exact same SqueezeNet architecture and the same 10-pass TTA averaging loop, but (1) make checkpoint loading robust to common wrapper formats (`{'model': ...}`, `{'state_dict': ...}`, nested dicts) and (2) explicitly handle the frequent `DataParallel`/Lightning `model.`/`module.` prefixes while preferring the variant that matches the most keys. I also enforce the correct inference input shape by unsqueezing the batch and ensure contiguous float32 tensors, without changing the deterministic center-crop evaluation pipeline. These changes are directly aimed at moving accuracy upward toward your target without changing the modeling approach.'
- What this solution (achieved 0.1491) has done: 'Your score is far below the target, so we should make the smallest changes that plausibly recover the intended trained-weight performance without changing the model or inference semantics. The biggest likely cause is the checkpoint not being applied correctly due to `classifier.*` key mismatches (common with SqueezeNet heads), so we load the checkpoint in a more shape-aware way that preferentially restores all feature weights even if the head keys differ, while keeping the same architecture and 10-pass averaging loop. We also ensure `Softmax` is applied before averaging logits across TTA passes (averaging probabilities is typically more stable for accuracy with augmentations) while keeping the same TTA count and final `argmax`. Finally, we make the test loop slightly more robust by reusing image load outside the TTA loop (no semantic change) and ensuring the submission ordering matches `sample_submission.csv`.'
- What this solution (achieved 0.10426) has done: 'Your current score is far below the target, so the most likely minimal high-impact issue is a train/test preprocessing mismatch: SqueezeNet typically expects 224×224 center-crop inputs (ImageNet-style), while your inference uses 256×256, which can noticeably hurt accuracy. I keep the exact same model, checkpoint-loading logic, and 10-pass TTA averaging loop, but change only the submission-time transform to produce a deterministic 224×224 center-crop with ImageNet normalization. This is a small, metric-aligned change that should move accuracy upward toward the target without altering core modeling/training logic. I also ensure the output tensor is explicitly float32 on-device (no semantic change, just avoids dtype surprises).'
- What this solution (achieved 0.09567) has done: 'Your current score (0.10426) is far below the target (0.82789), so we should make a small, high-impact fix that preserves your model and inference/TTA loop while correcting a likely train/test mismatch. The biggest issue is that the SqueezeNet head is missing a required `Dropout` layer; changing `model.classifier[1]` keeps the old dropout but your current `model.classifier[1] = Conv2d(...)` removes it and alters the classifier structure, which makes most checkpoints load poorly and harms accuracy. I restore the canonical SqueezeNet classifier block `[Dropout, Conv2d, ReLU, AvgPool]` while keeping the same Conv2d parameters and the same checkpoint-loading logic (still shape-compatible filtering). Everything else (224 center-crop, normalization, 10-pass probability averaging TTA, submission formatting) stays the same.'
- What this solution (achieved 0.10015) has done: 'Your score is far below the target, so we should make a minimal, high-impact fix that preserves your model and 10-pass averaging loop while removing the main source of systematic accuracy collapse: the classifier head currently uses `AdaptiveAvgPool2d`, which often prevents proper checkpoint loading if the trained model used the canonical SqueezeNet `AvgPool2d(kernel_size=13)` head. I restore the canonical classifier block (`Dropout -> Conv2d -> ReLU -> AvgPool2d(13)`) so pretrained/competition checkpoints match key shapes and load fully, while keeping the same backbone and inference semantics. I also print how many tensors were actually loaded (exact key+shape matches) so you can verify the checkpoint is being applied, and keep the same deterministic 224 center-crop + ImageNet normalization and the same submission formatting.'
- What this solution (achieved 0.09978) has done: 'Your score is far below the target, so the most likely minimal high-impact issue is that inference is being run with dropout still active (SqueezeNet’s classifier starts with `Dropout`, and `model.eval()` should disable it, but we also explicitly force dropout layers to eval to avoid any accidental train-mode toggles). Next, we make the submission ordering and label dtype airtight by deriving `image_id` order strictly from `sample_submission.csv` and casting predictions to `int64`, preventing any subtle merge/NaN issues. Finally, we add deterministic settings and a quick sanity check that the model output has 5 classes so we fail fast if a checkpoint mismatch silently changes shapes; these changes don’t alter the core model, TTA averaging, or transforms, but should stabilize and improve accuracy toward your target.'
- What this solution (achieved 0.09978) has done: 'Your score is far below the target, so the most likely minimal high-impact fix is that the intended checkpoint still isn’t being applied (or is being applied to the wrong tensors), leaving you close to random guessing. I keep the exact same SqueezeNet architecture, deterministic eval transform, and 10-pass probability-averaging TTA loop, but make checkpoint loading shape-aware with an explicit preference order: load a fully matching state_dict strictly when possible, otherwise load all shape-matching weights but *also* re-map common SqueezeNet head key patterns (e.g., `classifier.1.*` vs `classifier.1.0.*`) so the classifier conv actually loads. I also add a simple, score-relevant sanity check: run one batch through the model and confirm logits look non-degenerate; this doesn’t change inference, it just prevents silently-bad submissions. These changes are directly aimed at moving accuracy up toward your target without changing core evaluation semantics.'
- What this solution (achieved 0.10987) has done: 'Your current score (0.09978) is far below the target (0.82789), so we need a minimal but high-impact fix that keeps your model/TTA loop intact while addressing the most likely root cause: the intended checkpoint is not being loaded at all because `model_path` points to a non-existent location, so you fall back to weak ImageNet features and get near-random performance. I keep the same SqueezeNet architecture, the same deterministic eval transform, and the same 10-pass “TTA” averaging logic, but add a tiny path-resolution step that searches a few likely input locations for the `.pth` file and only falls back if it truly cannot be found. I also make the fallback path still load ImageNet weights correctly for *all* layers (including the classifier conv) if no competition checkpoint is present, which is a legitimate accuracy boost without changing core logic. These changes are directly score-relevant and should move accuracy substantially upward toward your target.'

# 9. Code solution

## === cell 0
import os
import random

import albumentations
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models



## === cell 1
model_path = "../input/sn-wc-aug/model(4).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _resolve_checkpoint_path(path: str) -> str:
    if os.path.exists(path):
        return path

    fname = os.path.basename(path)

    search_roots = [
        "../input",
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/data",
    ]

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if fname in filenames:
                cand = os.path.join(dirpath, fname)
                return cand

    return path


model_path = _resolve_checkpoint_path(model_path)
print("Resolved model_path:", model_path, "| exists:", os.path.exists(model_path))



## === cell 2
model = models.squeezenet1_0(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Conv2d(in_channels=512, out_channels=5, kernel_size=(1, 1), stride=(1, 1)),
    nn.ReLU(inplace=True),
    nn.AvgPool2d(kernel_size=13, stride=1),
)
model.to(DEVICE)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        candidate_keys = [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
        ]
        for k in candidate_keys:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]

        tensorish = 0
        dotted = 0
        for kk, vv in ckpt_obj.items():
            if isinstance(kk, str) and "." in kk:
                dotted += 1
            if torch.is_tensor(vv):
                tensorish += 1
        if tensorish > 0 and dotted > 0:
            return ckpt_obj

    return ckpt_obj


def _clean_keys(sd, prefixes_to_strip):
    out = {}
    for k, v in sd.items():
        nk = k
        for prefix in prefixes_to_strip:
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        out[nk] = v
    return out


def _remap_squeezenet_head_keys(sd):
    out = dict(sd)

    for suffix in ["weight", "bias"]:
        k_from = f"classifier.1.0.{suffix}"
        k_to = f"classifier.1.{suffix}"
        if (k_from in out) and (k_to not in out):
            out[k_to] = out[k_from]

    for suffix in ["weight", "bias"]:
        k_from = f"classifier.1.conv.{suffix}"
        k_to = f"classifier.1.{suffix}"
        if (k_from in out) and (k_to not in out):
            out[k_to] = out[k_from]

    return out


def _score_variant_by_shape_compat(model, cleaned_sd):
    model_sd = model.state_dict()
    score_shape_ok = 0
    score_matched = 0
    for k, v in cleaned_sd.items():
        if k in model_sd:
            score_matched += 1
            try:
                if tuple(model_sd[k].shape) == tuple(v.shape):
                    score_shape_ok += 1
            except Exception:
                pass
    return (score_shape_ok, score_matched)


def _filter_shape_compatible(model, cleaned_sd):
    model_sd = model.state_dict()
    out = {}
    for k, v in cleaned_sd.items():
        if k in model_sd:
            try:
                if tuple(model_sd[k].shape) == tuple(v.shape):
                    out[k] = v
            except Exception:
                pass
    return out


def _try_load_with_variants(model, state_dict):
    variants = [
        [],  # as-is
        ["module."],
        ["model."],
        ["net."],
        ["module.", "model."],
        ["model.", "module."],
        ["module.", "net."],
        ["net.", "module."],
        ["module.", "model.", "net."],
    ]

    best = None  # (score_tuple, rules, cleaned_dict)

    for rules in variants:
        cleaned = _clean_keys(state_dict, rules)
        cleaned = _remap_squeezenet_head_keys(cleaned)
        score = _score_variant_by_shape_compat(model, cleaned)
        if (best is None) or (score > best[0]):
            best = (score, rules, cleaned)

    if best is None:
        return False, "Failed to find any usable checkpoint key variant.", 0, 0, 0

    (shape_ok, matched), rules, cleaned = best

    model_sd = model.state_dict()
    strict_possible = True
    for k, v in model_sd.items():
        if k not in cleaned:
            strict_possible = False
            break
        try:
            if tuple(cleaned[k].shape) != tuple(v.shape):
                strict_possible = False
                break
        except Exception:
            strict_possible = False
            break

    if strict_possible:
        model.load_state_dict(cleaned, strict=True)
        msg = (
            f"Selected checkpoint variant rules={rules} by (shape_ok={shape_ok}, matched_keys={matched}). "
            f"Loaded with strict=True (full match)."
        )
        return True, msg, len(cleaned), 0, 0

    filtered = _filter_shape_compatible(model, cleaned)
    incompatible = model.load_state_dict(filtered, strict=False)

    missing = len(incompatible.missing_keys)
    unexpected = len(incompatible.unexpected_keys)
    loaded = len(filtered)

    msg = (
        f"Selected checkpoint variant rules={rules} by (shape_ok={shape_ok}, matched_keys={matched}). "
        f"Loaded tensors={loaded} with strict=False (missing={missing}, unexpected={unexpected})."
    )
    return True, msg, loaded, missing, unexpected


if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location=DEVICE)
    state_dict = _extract_state_dict(ckpt)
    ok, msg, loaded, missing, unexpected = _try_load_with_variants(model, state_dict)
    print(msg)
    if not ok:
        raise RuntimeError(
            "Checkpoint exists but could not be loaded in any supported format."
        )
else:
    backbone = models.squeezenet1_0(pretrained=True)
    model.load_state_dict(
        backbone.state_dict(), strict=False
    )  # keep as much as possible
    nn.init.normal_(model.classifier[1].weight, mean=0.0, std=0.01)
    if model.classifier[1].bias is not None:
        nn.init.constant_(model.classifier[1].bias, 0.0)
    print(
        "Model checkpoint not found; using full ImageNet-pretrained SqueezeNet fallback (reinit final conv only)."
    )

model.eval()
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.eval()

with torch.inference_mode():
    x = torch.zeros(1, 3, 224, 224, device=DEVICE, dtype=torch.float32)
    y = model(x)
    if y.ndim != 2 or y.shape[1] != 5:
        raise RuntimeError(
            f"Model sanity check failed: got logits shape {tuple(y.shape)}"
        )
    if not torch.isfinite(y).all():
        raise RuntimeError("Model sanity check failed: non-finite logits detected.")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2242315273.py in <cell line: 0>()
    169     # (features + classifier) and then only replace the final conv to 5 classes.
    170     backbone = models.squeezenet1_0(pretrained=True)
--> 171     model.load_state_dict(
    172         backbone.state_dict(), strict=False
    173     )  # keep as much as possible

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in load_state_dict(self, state_dict, strict, assign)
   2579 
   2580         if len(error_msgs) > 0:
-> 2581             raise RuntimeError(
   2582                 "Error(s) in loading state_dict for {}:\n\t{}".format(
   2583                     self.__class__.__name__, "\n\t".join(error_msgs)

RuntimeError: Error(s) in loading state_dict for SqueezeNet:
	size mismatch for classifier.1.weight: copying a param with shape torch.Size([1000, 512, 1, 1]) from checkpoint, the shape in current model is torch.Size([5, 512, 1, 1]).
	size mismatch for classifier.1.bias: copying a param with shape torch.Size([1000]) from checkpoint, the shape in current model is torch.Size([5]).

## === cell 3
sub_aug = albumentations.Compose(
    [
        albumentations.LongestMaxSize(max_size=256, p=1.0),
        albumentations.PadIfNeeded(
            min_height=256, min_width=256, border_mode=0, value=0, p=1.0
        ),
        albumentations.CenterCrop(height=224, width=224, p=1.0),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)




## === cell 4
def normalized_numpy_to_tensor(x: np.ndarray) -> torch.Tensor:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(
            f"Expected HxWx3 image after augmentation, got shape {x.shape}"
        )
    t = torch.from_numpy(x).permute(2, 0, 1).contiguous()
    return t




## === cell 5
sample_sub = pd.read_csv(sample_sub_path)
image_ids = sample_sub["image_id"].tolist()

pred_labels = np.empty(len(image_ids), dtype=np.int64)
softmax = torch.nn.Softmax(dim=1)

with torch.inference_mode():
    for i, image_id in enumerate(image_ids):
        image_path = os.path.join(test_images_path, image_id)

        image0 = Image.open(image_path).convert("RGB")
        image0 = np.array(image0)

        prob_sum = None
        for _ in range(10):  # keep same TTA count / averaging logic
            image = sub_aug(image=image0)["image"]
            image_t = normalized_numpy_to_tensor(image).to(
                DEVICE, non_blocking=True, dtype=torch.float32
            )

            outputs = model(image_t.unsqueeze(0))

            if outputs.ndim != 2 or outputs.shape[1] != 5:
                raise RuntimeError(
                    f"Unexpected model output shape {tuple(outputs.shape)} for image {image_id}"
                )

            probs = softmax(outputs)  # average probabilities for stability

            if prob_sum is None:
                prob_sum = probs
            else:
                prob_sum += probs

        prob_avg = prob_sum / 10.0
        pred_labels[i] = int(torch.argmax(prob_avg, dim=1).item())

sub_df = pd.DataFrame({"image_id": image_ids, "label": pred_labels.astype(np.int64)})
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with", len(sub_df), "rows")
