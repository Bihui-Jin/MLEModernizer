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

0.8203384708371109

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the two blockers preventing an end-to-end run: (1) make the ViT weights loading robust to missing external weight files by falling back to timm’s built-in pretrained weights (same architecture, same inference semantics), and (2) remove unconditional `.cuda()` calls by selecting CPU vs GPU at runtime so it works in Kaggle notebook environments without a GPU driver. I also fix the test dataloader ordering/alignment by using `sample_submission.csv` image_id order and disabling shuffle, ensuring the submission exactly matches the required format and ordering. Finally, I replace the deprecated/slow `DataFrame.append` in the inference loop with list accumulation to avoid runtime/performance issues while keeping predictions identical.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target (0.82034), which strongly suggests the finetuned checkpoint is not actually being loaded (or is mismatched), so you’re effectively submitting near-random predictions from a generic ImageNet-pretrained ViT head. I make the checkpoint loading robust to common Kaggle checkpoint formats (e.g., `state_dict`, `model`, `module.` prefixes) while keeping the exact same model architecture and inference logic. I also add a quick sanity print of missing/unexpected keys to confirm whether weights were applied, and keep the submission ordering exactly aligned to `sample_submission.csv`. These minimal fixes should move accuracy sharply upward toward the target without changing the model or evaluation semantics.'
- What this solution (achieved 0.11584) has done: 'Your current score is so far below the target that the most likely cause is the finetuned checkpoint isn’t actually being applied to the model (so predictions are essentially random). I keep your model and inference logic intact, but make checkpoint loading more robust specifically for common “timm saved model” cases (e.g., keys prefixed with `model.model.` and classifier saved as `fc.*` instead of `head.*`), and I verify the loaded head weight shapes to ensure the 5-class classifier is restored. I also keep the submission ordering driven by `sample_submission.csv` exactly as you already do. These are minimal changes aimed directly at moving accuracy sharply upward toward the target without altering architecture or evaluation semantics.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so the smallest likely fix is ensuring the finetuned checkpoint truly matches your ViT head and is actually being loaded (not silently skipped by `strict=False`). I keep your model and inference identical, but make the checkpoint loader “strict when possible”: it (1) adapt common key patterns, (2) explicitly validate that `head.weight/head.bias` exist and match shape `(5, in_features)`, and (3) fail over to a second mapping attempt for other frequent wrappers. I also switch the test DataLoader to a larger batch size (no semantic change) to keep runtime comfortably under limits while producing the same predictions. The submission order/format remains driven by `sample_submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so the smallest likely improvement is to ensure the finetuned checkpoint is actually being applied to the exact parameter names/shapes your `timm` ViT expects. I keep your architecture and inference identical, but make the checkpoint loader handle more real-world key patterns (especially `model.` vs `model.model.` nesting) and explicitly map head keys into the nested `self.model.head.*` that your wrapper uses. I also add a hard validation that the loaded classifier weights are non-default and shape-correct (without changing predictions), so we don’t silently submit near-random logits again. Everything else (transforms, argmax, submission ordering/format) stays the same.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is so far below the target (0.82034) that the most likely remaining issue is still “finetuned weights not actually being applied,” especially the classifier head, despite the existing robust loader. I make one minimal but high-impact fix: validate and load the finetuned checkpoint directly into the underlying `timm` model (`cassava_model.model`) in addition to the wrapper, using the same key-remapping logic, so we don’t get trapped by wrapper-prefix mismatches. I also add a hard check that the classifier weights changed from the freshly-initialized head (detecting silent no-load), and only then proceed to inference. This preserves the exact architecture/inference semantics and should move accuracy sharply upward toward the target band.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, so the most likely issue is still that the finetuned checkpoint is not being correctly loaded into the 5-class ViT head (or it’s being overwritten/ignored). I keep your exact model architecture and inference logic, but make the checkpoint loader (1) prefer loading into the underlying `timm` model with correctly-remapped `head.*` keys, (2) only then load into the wrapper if needed, and (3) enforce a hard validation that the head weights match the expected 5-class shape and actually changed—otherwise we fail fast instead of silently submitting near-random predictions. I also ensure the finetuned head isn’t accidentally clobbered by any subsequent reinitialization and keep submission ordering aligned to `sample_submission.csv`. These are minimal, directly score-relevant changes intended to move accuracy sharply upward toward your target band.'
- What this solution (achieved 0.11584) has done: 'Your current score is far below the target, so the most likely remaining issue is that the finetuned checkpoint still isn’t being applied correctly (despite loading “something”), especially the classifier head vs wrapper key names. I keep your exact model/inference logic, but make one minimal, high-impact change: load the finetuned checkpoint into the underlying `timm` model only after explicitly remapping common prefixes and forcing the head keys to exactly match `head.weight/head.bias` with the correct `(5, …)` shape. To avoid “half-loaded” models, I also enforce that the head keys must be present in the checkpoint and raise an error if they aren’t, because submitting without a trained head is what produces near-random accuracy like 0.115. Everything else (transforms, argmax, submission ordering via `sample_submission.csv`) remains the same.'
- What this solution (achieved 0.10874) has done: 'The current score is so far below the target that the most likely remaining issue is still that the finetuned checkpoint isn’t actually being loaded into the model parameters your `timm` ViT expects, even though the head “changed” check passes. I keep your exact model, transforms, and argmax inference, but make the checkpoint loader try a small set of additional, common key-remappings (including `model.` / `model.module.` / wrapper nesting) and explicitly verify that a large fraction of non-head weights also changed (to avoid loading only the head or only a tiny subset). I also ensure the model is created with `pretrained=True` directly (same architecture) so that if the finetuned checkpoint is partial, the backbone is still sensible rather than random. These are minimal, score-relevant changes aimed at moving accuracy sharply upward toward your target while preserving evaluation semantics and producing the same submission format/order.'
- What this solution (achieved 0.10874) has done: 'Your score is far below the target, so the most likely remaining issue is that you’re still not actually using a properly finetuned 5-class cassava checkpoint at inference (even if “something loads”), or you’re loading weights from a different architecture/version than your `timm` model expects. I make the checkpoint loading both more compatible and safer by (1) allowing `timm`’s own `checkpoint_filter_fn` to adapt keys when possible, (2) trying a small additional set of common key patterns (including `head.*` vs `model.head.*` nesting), and (3) enforcing that the checkpoint contains a valid 5-class head and that a meaningful fraction of backbone tensors changed. These are minimal, score-critical changes that preserve your model architecture and inference logic and should move accuracy sharply upward toward the target band while still producing the same submission format and order.'
- What this solution (achieved 0.10874) has done: 'Your score is far below the target, so the smallest likely score-moving fix is to stop silently running with a wrong/partial checkpoint and instead load the finetuned weights using timm’s native `load_checkpoint` (which applies the correct internal key filtering for ViT) as the first attempt. If that fails, we keep your existing robust remapping loader as a fallback, but we tighten validation to ensure the backbone and head are truly loaded (otherwise we fail fast rather than producing near-random submissions). I also remove the unused loss/optimizer creation (no training happens here) to avoid any accidental side effects and keep runtime stable. Submission ordering/format stays driven by `sample_submission.csv`, identical to your current semantics.'
- What this solution (achieved 0.10874) has done: 'Your score is far below the target, so the smallest likely score-moving change is to ensure we are actually loading the intended finetuned checkpoint (and not silently falling back to ImageNet weights or partial loads). I keep your exact ViT model and argmax inference, but (1) make checkpoint path discovery robust across `/kaggle/input/*` mount variants, and (2) add one more safe, common checkpoint-key normalization (`"model.model."` double-nesting) before loading. I also keep your existing “fail fast if too few non-head weights changed” guard so we don’t accidentally submit a near-random model again. Everything else (transforms, DataLoader ordering via `sample_submission.csv`, submission schema) remains unchanged.'
- What this solution (achieved 0.10874) has done: 'The huge gap to the target accuracy strongly suggests your finetuned checkpoint still isn’t being correctly applied (or you’re accidentally using a mismatched model config), so the smallest score-relevant fix is to create the exact same ViT variant that the common cassava finetuned checkpoints use and then load the finetuned weights into that exact key space. I keep your core logic (ViT + argmax inference + sample_submission ordering) but adjust the model creation to use `vit_base_patch16_224_in21k` when available (falling back to your current model name if not), and I tighten/expand key normalization once more to handle `head.*` saved as `model.head.*` or deeper nestings. Finally, I keep submission format/order identical and only touch loader/model-instantiation pieces that directly affect accuracy.'

# 9. Code solution

## === cell 0
import os
import time
import random
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt



## === cell 1
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



## === cell 2
BASE1 = "../input/cassava-leaf-disease-classification"
BASE2 = "/kaggle/input/cassava-leaf-disease-classification"

data_path = BASE1 if os.path.exists(BASE1) else BASE2
train_path = os.path.join(data_path, "train_images") + os.sep
test_path = os.path.join(data_path, "test_images") + os.sep


def _resolve_existing_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _find_in_kaggle_input_by_filename(filename):
    roots = [
        "../input",
        "/kaggle/input",
        "/kaggle/data/input",
    ]
    for root in roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, _, files in os.walk(root):
            if filename in files:
                return os.path.join(dirpath, filename)
    return None


model_path = _resolve_existing_path(
    [
        "../input/vitbase16224/jx_vit_base_p16_224-80ecf9dd.pth",
        "/kaggle/input/vitbase16224/jx_vit_base_p16_224-80ecf9dd.pth",
    ]
)
if model_path is None:
    model_path = _find_in_kaggle_input_by_filename("jx_vit_base_p16_224-80ecf9dd.pth")

Cassava_model = _resolve_existing_path(
    [
        "../input/cassavaaugmtp98epochs4lr175/CassavaViT_Augm_TP98_Epochs4_LR1-75e05.pt",
        "/kaggle/input/cassavaaugmtp98epochs4lr175/CassavaViT_Augm_TP98_Epochs4_LR1-75e05.pt",
    ]
)
if Cassava_model is None:
    Cassava_model = _find_in_kaggle_input_by_filename(
        "CassavaViT_Augm_TP98_Epochs4_LR1-75e05.pt"
    )

print("data_path:", data_path)
print("test_path exists:", os.path.exists(test_path))
print("Cassava finetuned checkpoint path:", Cassava_model)
print("Base ViT weights path:", model_path)



## === cell 3
import timm

print("timm version:", getattr(timm, "__version__", "unknown"))



## === cell 4
print("Available ViT Models (subset):")
print([m for m in timm.list_models("vit*")][:10], "...")




## === cell 5
def _choose_vit_model_name():
    preferred = "vit_base_patch16_224_in21k"
    fallback = "vit_base_patch16_224"
    try:
        avail = set(timm.list_models(pretrained=True))
        if preferred in avail:
            return preferred
    except Exception:
        pass
    return fallback


VIT_MODEL_NAME = _choose_vit_model_name()
print("Using timm model name:", VIT_MODEL_NAME)


class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ViTBase16, self).__init__()

        if pretrained:
            if (
                model_path is not None
                and os.path.exists(model_path)
                and VIT_MODEL_NAME == "vit_base_patch16_224"
            ):
                self.model = timm.create_model(VIT_MODEL_NAME, pretrained=False)
                state = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(state, strict=True)
                print(f"Loaded base ViT weights from: {model_path}")
            else:
                self.model = timm.create_model(VIT_MODEL_NAME, pretrained=True)
                if (
                    model_path is not None
                    and os.path.exists(model_path)
                    and VIT_MODEL_NAME != "vit_base_patch16_224"
                ):
                    print(
                        "External base ViT weights exist but were trained for vit_base_patch16_224; "
                        f"current model is {VIT_MODEL_NAME}. Using timm pretrained weights for compatibility."
                    )
                else:
                    print("Using timm pretrained=True weights.")
        else:
            self.model = timm.create_model(VIT_MODEL_NAME, pretrained=False)

        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        return self.model(x)




## === cell 6
def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    return ckpt_obj


def _strip_prefix_from_state_dict(state_dict, prefixes):
    if not isinstance(state_dict, dict):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        new_sd[nk] = v
    return new_sd


def _remap_classifier_keys_for_vit(sd):
    if not isinstance(sd, dict):
        return sd
    out = dict(sd)

    if "fc.weight" in out and "head.weight" not in out:
        out["head.weight"] = out.pop("fc.weight")
    if "fc.bias" in out and "head.bias" not in out:
        out["head.bias"] = out.pop("fc.bias")

    if "classifier.weight" in out and "head.weight" not in out:
        out["head.weight"] = out.pop("classifier.weight")
    if "classifier.bias" in out and "head.bias" not in out:
        out["head.bias"] = out.pop("classifier.bias")

    if "model.head.weight" in out and "head.weight" not in out:
        out["head.weight"] = out.pop("model.head.weight")
    if "model.head.bias" in out and "head.bias" not in out:
        out["head.bias"] = out.pop("model.head.bias")
    if "model.model.head.weight" in out and "head.weight" not in out:
        out["head.weight"] = out.pop("model.model.head.weight")
    if "model.model.head.bias" in out and "head.bias" not in out:
        out["head.bias"] = out.pop("model.model.head.bias")

    return out


def _tensor_fingerprint(t: torch.Tensor, n=32):
    x = t.detach().float().reshape(-1)
    if x.numel() == 0:
        return (0.0, 0.0, 0.0)
    idx = torch.linspace(0, max(0, x.numel() - 1), steps=min(n, x.numel())).long()
    samp = x[idx].cpu().numpy()
    return (float(samp.mean()), float(samp.std()), float(samp[0]))


def _count_changed_params(before_sd, after_sd, ignore_prefixes=("head.",)):
    changed = 0
    total = 0
    for k, v_before in before_sd.items():
        if any(k.startswith(p) for p in ignore_prefixes):
            continue
        v_after = after_sd.get(k, None)
        if v_after is None:
            continue
        if not (
            isinstance(v_before, torch.Tensor) and isinstance(v_after, torch.Tensor)
        ):
            continue
        total += 1
        if not torch.equal(v_before.cpu(), v_after.cpu()):
            changed += 1
    return changed, total


def _maybe_apply_timm_checkpoint_filter(model_timm, sd: dict):
    try:
        fn = getattr(model_timm, "checkpoint_filter_fn", None)
        if callable(fn):
            return fn(sd, model_timm)
    except Exception as e:
        print("timm checkpoint_filter_fn failed (continuing without it):", repr(e))
    return sd


def load_finetuned_weights_strictish(
    model: ViTBase16, ckpt_path: str, n_classes: int = 5
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd0 = _extract_state_dict(ckpt)
    if not isinstance(sd0, dict):
        raise ValueError("Checkpoint did not contain a recognizable state_dict dict.")

    before_sd = {
        k: v.detach().clone().cpu() for k, v in model.model.state_dict().items()
    }
    base_sd = dict(sd0)

    prefix_variants = [
        tuple(),
        ("module.",),
        ("model.",),
        ("model.model.",),
        ("model.module.",),
        ("model.model.model.",),
        ("net.",),
        ("backbone.",),
        ("module.", "model.", "net.", "backbone.", "model.model.", "model.module."),
        ("model.", "module.", "net.", "backbone.", "model.model.", "model.module."),
        ("model.module.", "module.", "net.", "backbone.", "model.model.", "model."),
        ("model.model.", "model.", "module.", "net.", "backbone.", "model.module."),
        (
            "model.model.model.",
            "model.model.",
            "model.",
            "module.",
            "net.",
            "backbone.",
        ),
        (
            "model.model.module.",
            "model.model.",
            "model.",
            "module.",
            "net.",
            "backbone.",
        ),
    ]

    last_missing = last_unexpected = None
    loaded = False

    for prefixes in prefix_variants:
        sd = _strip_prefix_from_state_dict(base_sd, prefixes=prefixes)
        sd = _remap_classifier_keys_for_vit(sd)
        sd = _maybe_apply_timm_checkpoint_filter(model.model, sd)

        if "head.weight" not in sd or "head.bias" not in sd:
            continue

        hw = sd["head.weight"]
        hb = sd["head.bias"]
        if not (isinstance(hw, torch.Tensor) and isinstance(hb, torch.Tensor)):
            continue
        if tuple(hw.shape)[0] != n_classes or tuple(hb.shape)[0] != n_classes:
            continue

        before_hw = _tensor_fingerprint(model.model.head.weight)
        before_hb = _tensor_fingerprint(model.model.head.bias)

        missing, unexpected = model.model.load_state_dict(sd, strict=False)
        last_missing, last_unexpected = missing, unexpected

        after_hw = _tensor_fingerprint(model.model.head.weight)
        after_hb = _tensor_fingerprint(model.model.head.bias)

        head_changed = (before_hw != after_hw) or (before_hb != after_hb)
        if not head_changed:
            continue

        loaded = True
        print(f"Loaded finetuned checkpoint from: {ckpt_path}")
        print(f"Used prefix stripping: {prefixes if prefixes else '(none)'}")
        print(
            f"[timm_load(strict=False)] missing: {len(missing)} | unexpected: {len(unexpected)}"
        )
        if len(missing) > 0:
            print("First 10 missing keys:", missing[:10])
        if len(unexpected) > 0:
            print("First 10 unexpected keys:", unexpected[:10])
        print("Head weight fingerprint before:", before_hw, "after:", after_hw)
        print("Head bias  fingerprint before:", before_hb, "after:", after_hb)
        print("Model head weight shape:", tuple(model.model.head.weight.shape))
        print("Model head bias shape:", tuple(model.model.head.bias.shape))
        break

    if not loaded:
        head_like = [
            k for k in base_sd.keys() if ("head" in k or "fc" in k or "classifier" in k)
        ]
        raise RuntimeError(
            "Failed to load finetuned checkpoint with expected 5-class head after trying common key remaps. "
            f"Head-like keys (first 30): {head_like[:30]}"
        )

    after_sd = {
        k: v.detach().clone().cpu() for k, v in model.model.state_dict().items()
    }
    changed, total = _count_changed_params(
        before_sd, after_sd, ignore_prefixes=("head.",)
    )
    frac = (changed / total) if total > 0 else 0.0
    print(f"Non-head tensors changed: {changed}/{total} ({frac:.3f})")

    if frac < 0.10:
        raise RuntimeError(
            "Checkpoint load appears to have changed too few non-head weights (<10%). "
            "This strongly suggests key mismatch and would produce low accuracy. "
            f"Last missing={len(last_missing) if last_missing is not None else 'NA'}, "
            f"unexpected={len(last_unexpected) if last_unexpected is not None else 'NA'}."
        )


def load_finetuned_weights_prefer_timm(
    model: ViTBase16, ckpt_path: str, n_classes: int = 5
):
    before_sd = {
        k: v.detach().clone().cpu() for k, v in model.model.state_dict().items()
    }
    before_hw = _tensor_fingerprint(model.model.head.weight)
    before_hb = _tensor_fingerprint(model.model.head.bias)

    try:
        from timm.models import load_checkpoint as timm_load_checkpoint

        timm_load_checkpoint(model.model, ckpt_path, strict=False)
        after_hw = _tensor_fingerprint(model.model.head.weight)
        after_hb = _tensor_fingerprint(model.model.head.bias)

        if (tuple(model.model.head.weight.shape)[0] != n_classes) or (
            tuple(model.model.head.bias.shape)[0] != n_classes
        ):
            raise RuntimeError(
                "Loaded checkpoint but head shape is not 5-class; likely wrong checkpoint/model."
            )
        if (before_hw == after_hw) and (before_hb == after_hb):
            raise RuntimeError(
                "timm load_checkpoint did not change head weights; likely key mismatch."
            )

        after_sd = {
            k: v.detach().clone().cpu() for k, v in model.model.state_dict().items()
        }
        changed, total = _count_changed_params(
            before_sd, after_sd, ignore_prefixes=("head.",)
        )
        frac = (changed / total) if total > 0 else 0.0
        print(f"Loaded finetuned checkpoint via timm.load_checkpoint: {ckpt_path}")
        print(f"Non-head tensors changed: {changed}/{total} ({frac:.3f})")
        if frac < 0.10:
            raise RuntimeError(
                "timm load_checkpoint changed too few non-head weights; refusing to submit near-random model."
            )
        return
    except Exception as e:
        print(
            "timm.load_checkpoint path failed; falling back to manual remap loader. Reason:",
            repr(e),
        )

    load_finetuned_weights_strictish(model, ckpt_path, n_classes=n_classes)




## === cell 7
cassava_model = ViTBase16(n_classes=5, pretrained=True)

if Cassava_model is not None and os.path.exists(Cassava_model):
    load_finetuned_weights_prefer_timm(cassava_model, Cassava_model, n_classes=5)
else:
    raise FileNotFoundError(
        "Finetuned Cassava checkpoint not found. This would produce near-random accuracy. "
        "Searched fixed paths and /kaggle/input recursively. "
        f"Got Cassava_model={Cassava_model!r}"
    )

cassava_model = cassava_model.to(device)
cassava_model.eval()

cassava_model



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/1288268647.py in <cell line: 0>()
      4     load_finetuned_weights_prefer_timm(cassava_model, Cassava_model, n_classes=5)
      5 else:
----> 6     raise FileNotFoundError(
      7         "Finetuned Cassava checkpoint not found. This would produce near-random accuracy. "
      8         "Searched fixed paths and /kaggle/input recursively. "

FileNotFoundError: Finetuned Cassava checkpoint not found. This would produce near-random accuracy. Searched fixed paths and /kaggle/input recursively. Got Cassava_model=None

## === cell 8
sample_sub_path = os.path.join(data_path, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()
print("Num test ids from sample_submission:", len(test_image_ids))
print("First 3 ids:", test_image_ids[:3])




## === cell 9
class TestSet2(Dataset):
    """Cassava Test Dataset driven by image_id list (ensures stable ordering)."""

    def __init__(self, test_dir, image_ids, transform=None):
        super().__init__()
        self.test_dir = test_dir
        self.image_ids = list(image_ids)
        self.transform = transform
        print("Cassava Disease Test Dataset Length = ", len(self.image_ids))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_name = self.image_ids[idx]
        img_path = os.path.join(self.test_dir, image_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(img)
        else:
            image = transforms.ToTensor()(img)
        return image, image_name




## === cell 10
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

testset = TestSet2(
    test_dir=test_path, image_ids=test_image_ids, transform=test_transform
)
print(testset)



## === cell 11
test_batch_size = 32
test_loader = DataLoader(
    dataset=testset,
    batch_size=test_batch_size,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=2,
)
print(test_loader)



## === cell 12
images, names = next(iter(test_loader))
im = make_grid(images[: min(len(images), 16)], nrow=4)
print("Batch names:", list(names[:4]))

plt.figure(figsize=(12, 4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
plt.axis("off")
plt.show()



## === cell 13
pred_rows = []
tic = time.time()

with torch.no_grad():
    for X_test, name in test_loader:
        X_test = X_test.to(device, non_blocking=True)
        logits = cassava_model(X_test)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
        for n, p in zip(list(name), preds):
            pred_rows.append((n, int(p)))

toc = time.time() - tic
print("Time for test inference is", toc, "seconds")

submission_df = pd.DataFrame(pred_rows, columns=["image_id", "label"])
submission_df = submission_df.set_index("image_id").loc[test_image_ids].reset_index()

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_df.head())



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_56/2378434827.py in <cell line: 0>()
      5     for X_test, name in test_loader:
      6         X_test = X_test.to(device, non_blocking=True)
----> 7         logits = cassava_model(X_test)
      8         preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
      9         for n, p in zip(list(name), preds):

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_56/2050697629.py in forward(self, x)
     48 
     49     def forward(self, x):
---> 50         return self.model(x)
     51 
     52 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/timm/models/vision_transformer.py in forward(self, x, attn_mask)
    991 
    992     def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
--> 993         x = self.forward_features(x, attn_mask=attn_mask)
    994         x = self.forward_head(x)
    995         return x

/usr/local/lib/python3.11/dist-packages/timm/models/vision_transformer.py in forward_features(self, x, attn_mask)
    934     def forward_features(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    935         """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)."""
--> 936         x = self.patch_embed(x)
    937         x = self._pos_embed(x)
    938         x = self.patch_drop(x)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/timm/layers/patch_embed.py in forward(self, x)
    129             pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
    130             x = F.pad(x, (0, pad_w, 0, pad_h))
--> 131         x = self.proj(x)
    132         if self.flatten:
    133             x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    552 
    553     def forward(self, input: Tensor) -> Tensor:
--> 554         return self._conv_forward(input, self.weight, self.bias)
    555 
    556 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    547                 self.groups,
    548             )
--> 549         return F.conv2d(
    550             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    551         )

RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same

## === cell 14
df_sub_test = pd.read_csv("submission.csv")
print("submission shape:", df_sub_test.shape)
print("submission columns:", df_sub_test.columns.tolist())
print(df_sub_test.head())

print("Files in cwd:", os.listdir("."))

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/148232022.py in <cell line: 0>()
----> 1 df_sub_test = pd.read_csv("submission.csv")
      2 print("submission shape:", df_sub_test.shape)
      3 print("submission columns:", df_sub_test.columns.tolist())
      4 print(df_sub_test.head())
      5 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'
