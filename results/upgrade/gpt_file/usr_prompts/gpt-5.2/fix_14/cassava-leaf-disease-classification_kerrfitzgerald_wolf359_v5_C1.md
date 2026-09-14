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

0.7840737382895134

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.14163) has done: 'I fix the execution blocker by removing the hard dependency on the missing fine-tuned checkpoint and instead loading only what is available (the ViT base weights if present, otherwise run with random init). I also make the model weight loading robust to common checkpoint key mismatches (e.g., `module.` prefixes) so it doesn’t crash on Kaggle. To improve accuracy toward your target (and keep core logic the same: same model + same inference loop), I enable standard test-time augmentation (simple flips) and average logits, which is a minimal inference-only change. Finally, I ensure the submission file is always created as `submission.csv` with the correct `image_id,label` columns and sample_submission alignment.'
- What this solution (achieved 0.14163) has done: 'Your score is far below the target because the model is likely running with random/base weights (fine-tuned checkpoint missing/unloaded), so the smallest reliable way to move accuracy toward the target is to (1) correctly load the fine-tuned checkpoint even when keys are nested/prefixed, and (2) ensure the model head shape matches by rebuilding the classifier after loading when needed. I keep the exact same ViT model and inference loop, but make the checkpoint loading more robust (handle common formats like `model`, `net`, `state_dict`, and `module.`) and report how many keys were actually loaded so we can confirm it’s not silently failing. I also keep your simple flip-TTA, and ensure the submission is aligned to `sample_submission.csv` exactly as you already do. These are minimal changes that should substantially increase accuracy if the fine-tuned weights exist in the provided input path.'
- What this solution (achieved 0.14163) has done: 'Your current score strongly suggests the fine-tuned checkpoint isn’t actually being applied to the timm ViT module (most likely due to key mismatches like `model.model.*` vs `model.*` and/or a saved dict with nested prefixes), so the smallest meaningful change is to make the checkpoint loader also strip the extra `model.model.` prefix and retry loading into the correct module. This keeps the exact same model, transforms, and inference/TTA logic, but increases the chance we truly load the trained weights and move accuracy up toward your target band. I also add a simple sanity print that checks whether the classifier head weights changed after loading (to detect silent no-op loads). Submission writing and alignment remain unchanged.'
- What this solution (achieved 0.14163) has done: 'Your score suggests the fine-tuned checkpoint still isn’t being applied to the actual timm ViT weights (so the model behaves close to random). I keep the exact same model, transforms, and flip-TTA inference, but make the checkpoint loader try additional common key-prefix patterns (notably `model.module.` and `module.model.`) and, crucially, attempt loading directly into `cassava_model.model` first (the timm backbone), which is where most saved state_dicts for timm models actually match. I also ensure the classifier head is set to 5 classes *before* loading the fine-tuned checkpoint so head weights can load when present (instead of becoming “unexpected” and staying random). These are minimal, score-relevant changes that should move accuracy substantially toward your target if the checkpoint contains the trained weights.'
- What this solution (achieved 0.14163) has done: 'Your current score (0.14163) is far below the target (0.78407), which is consistent with the fine-tuned checkpoint not actually being loaded into the model (so predictions are close to random). I keep your exact ViT model and inference/TTA logic, but make the checkpoint loader try a few additional, very common timm/Lightning key patterns (including handling `head.*` vs `classifier.*` and nested `state_dict` entries) and verify loading by checking multiple layer deltas (not just the head). I also ensure the checkpoint is loaded onto the correct module (`cassava_model.model`) with a last-resort targeted key-rename for the classification head so the 5-class head weights can load when present. These are minimal, score-relevant changes intended to move accuracy up toward your target band without changing architecture, transforms, or evaluation semantics.'
- What this solution (achieved 0.14163) has done: 'Your score is far below the target, which is most consistent with the fine-tuned checkpoint still not being correctly loaded (so predictions are near-random). I keep the exact same ViT model and the same inference loop (including your flip-TTA), but make one minimal, score-critical adjustment: try additional very common head key renames used in timm ViT checkpoints (e.g., `model.head.*` / `model.fc.*` / `classifier.*`) and apply the rename only when tensor shapes match. I also ensure we always load the fine-tuned checkpoint into the timm backbone first (where keys usually match), and add a small verification print of how many head/backbone tensors actually changed to detect silent no-op loads. Submission writing, alignment to `sample_submission.csv`, and output filename `submission.csv` remain unchanged.'
- What this solution (achieved 0.14163) has done: 'Your score (0.14163) is far below the target (0.78407), which strongly indicates the fine-tuned checkpoint still isn’t actually loading into the timm ViT (so predictions are near-random). I keep your exact ViT model, transforms, and flip-TTA inference, but make one minimal, score-critical adjustment: load the fine-tuned checkpoint with a smarter key-mapping that also fixes common timm ViT naming differences like `head` vs `fc` and nested `model.*` prefixes, applied only when tensor shapes match. I also ensure we try loading into both `cassava_model.model` and the full wrapper with the remapped state dict, and add a small verification that a few backbone tensors changed (to catch silent no-op loads). Submission writing, alignment to `sample_submission.csv`, and `submission.csv` output remain unchanged.'
- What this solution (achieved 0.14163) has done: 'Your current score is far below the target, which is most consistent with the fine-tuned checkpoint not actually being loaded (so the model behaves close to random). I keep your exact ViT model, transforms, and flip-TTA inference, but make the checkpoint loader more aggressive in two minimal ways: (1) handle “timm checkpoint as list/EMA” patterns and (2) automatically infer and strip the *best* common prefix by matching keys against the model’s state_dict, then apply shape-safe head renames. I also add a strict, score-relevant sanity check that aborts (instead of silently submitting near-random predictions) if essentially no backbone weights changed after the “fine-tuned” load attempt—because submitting random weights is what produced ~0.14. These are minimal, execution-safe changes aimed specifically at getting the real fine-tuned weights applied so accuracy moves toward your target.'
- What this solution (achieved 0.14163) has done: 'Your score (0.14163) is far below the target (0.78407), which most strongly indicates the fine-tuned checkpoint is still not actually being applied (so predictions are near-random). I keep your exact ViT model and the same inference/TTA loop, but make one minimal, score-critical change: load the fine-tuned checkpoint into the full wrapper first (so both `model.*` and any wrapper-level keys can match), and only then fall back to backbone-only loading. I also strengthen the “no-op load” detection to check a few backbone tensors deterministically and, if the load is a no-op, fall back to using the most frequent class from `train.csv` instead of submitting random predictions (this legitimately improves accuracy on Cassava because class imbalance makes the majority-class baseline much higher than ~0.14). Submission creation, alignment to `sample_submission.csv`, and output filename `submission.csv` remain unchanged.'
- What this solution (achieved 0.14163) has done: 'Your score (0.14163) is far below the target (0.78407), which is most consistent with the fine-tuned checkpoint still not being loaded into the actual timm ViT backbone (so predictions are near-random). I keep your model, transforms, and inference/TTA logic identical, but make one minimal, score-critical improvement: infer the best matching key-prefix **by longest-common-suffix matching against the model keys**, which fixes cases where the checkpoint keys are like `something.backbone.model.*` and simple prefix-stripping can’t recover them. I also ensure we always try loading into `cassava_model.model` (timm backbone) first with this improved mapping, because that’s where most timm checkpoints match, and keep the majority-class fallback only if loading is detected as a near no-op. These changes are narrowly targeted to increase the chance your fine-tuned weights actually apply, which should move accuracy sharply toward your target without changing core logic.'
- What this solution (achieved 0.14163) has done: 'Your score is far below the target, which is most consistent with the fine-tuned checkpoint still not being effectively applied and/or the submission falling back to near-random predictions. I keep your exact ViT model, transforms, and inference loop (including flip-TTA), but make one minimal, score-critical improvement: reliably detect “no-op” loading by measuring backbone deltas on stable parameter names and, if loading is a no-op, use the legitimate majority-class baseline (which is much stronger than ~0.14 on this dataset). I also make the checkpoint loading try one additional very common container key (`"model_state"`) and ensure we load into the correct module in the right order without changing architecture. These changes are narrowly targeted to move accuracy upward toward your target band while preserving core logic and producing the same submission format.'
- What this solution (achieved 0.61099) has done: 'Your score (~0.14) is consistent with the “fine-tuned” checkpoint not being applied (so predictions are near-random); the smallest score-relevant change is to make checkpoint discovery and loading actually find a valid .pt/.pth in the given input folders and then load it into the existing timm ViT backbone. I keep your exact model, transforms, and flip-TTA inference, but add a minimal glob-based search for plausible checkpoints and a stricter “loaded vs no-op” detection so we only fall back to majority-class when we truly failed to load meaningful weights. This should move accuracy upward toward your target because a real fine-tuned ViT on Cassava typically scores far above 0.14, while the rest of your pipeline (submission alignment/format) is already correct. No training, architecture, or loss changes are introduced, and it still always writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import time
import random
import warnings
import glob

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
try:
    import timm  # type: ignore
except Exception as e:
    wheel_path = "../input/timm034/timm-0.3.4-py3-none-any.whl"
    if os.path.exists(wheel_path):
        import sys
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-deps", "-q", wheel_path]
        )
        import timm  # type: ignore
    else:
        raise RuntimeError(
            "timm is not available and local wheel was not found at "
            f"{wheel_path}. Original import error: {e}"
        )



## === cell 2
data_path = "../input/cassava-leaf-disease-classification/"
train_path = "../input/cassava-leaf-disease-classification/train_images/"
test_path = "../input/cassava-leaf-disease-classification/test_images/"

model_path = "../input/vitbase16224/jx_vit_base_p16_224-80ecf9dd.pth"
Cassava_model = (
    "../input/cassavaaugmtp99epochs2/CassavaViT_Augm_TP99_Epochs2_LR1-75e05.pt"
)

if not os.path.isdir(test_path):
    nested = os.path.join(
        data_path, "cassava-leaf-disease-classification", "test_images"
    )
    if os.path.isdir(nested):
        test_path = nested if nested.endswith("/") else (nested + "/")
if not os.path.isdir(train_path):
    nested = os.path.join(
        data_path, "cassava-leaf-disease-classification", "train_images"
    )
    if os.path.isdir(nested):
        train_path = nested if nested.endswith("/") else (nested + "/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 3
available = timm.list_models("vit*")
print("Available ViT Models (count):", len(available))
print("Example:", available[:5])




## === cell 4
def _extract_state_dict(ckpt_obj):
    """
    Score-relevant robustness: many training frameworks wrap the real state dict under
    different keys; also sometimes provide EMA weights. Extract the most likely weights dict.
    """
    if isinstance(ckpt_obj, dict):
        for key in ["ema", "model_ema", "teacher", "student_ema"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                inner = ckpt_obj[key]
                for k2 in [
                    "state_dict",
                    "model",
                    "net",
                    "network",
                    "model_state_dict",
                    "model_state",
                ]:
                    if k2 in inner and isinstance(inner[k2], dict):
                        return inner[k2]
                return inner

        for key in [
            "state_dict",
            "model",
            "net",
            "network",
            "model_state_dict",
            "model_state",  # minimal robustness: some scripts save under this key
            "student",
        ]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        return ckpt_obj

    if isinstance(ckpt_obj, (list, tuple)):
        for item in ckpt_obj:
            if isinstance(item, dict):
                tensor_like = 0
                for v in item.values():
                    if torch.is_tensor(v):
                        tensor_like += 1
                    if tensor_like >= 5:
                        return item
        for item in ckpt_obj:
            if isinstance(item, dict):
                return item

    return ckpt_obj


def _strip_prefix(k, pref):
    return k[len(pref) :] if k.startswith(pref) else k


def _auto_prefix_strip(state, model, extra_prefixes=None):
    """
    Minimal robustness: select prefix-removal that maximizes key overlap with model.state_dict().
    """
    if not isinstance(state, dict):
        return state

    model_keys = set(model.state_dict().keys())
    keys = list(state.keys())
    prefixes = [
        "",
        "module.",
        "model.",
        "model.module.",
        "module.model.",
        "model.model.",
        "net.",
        "network.",
    ]
    if extra_prefixes:
        prefixes.extend(extra_prefixes)

    best_pref = ""
    best_hits = -1
    for pref in prefixes:
        hits = 0
        for k in keys:
            kk = _strip_prefix(k, pref)
            if kk in model_keys:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_pref = pref

    if best_pref != "":
        stripped = {(_strip_prefix(k, best_pref)): v for k, v in state.items()}
        print(
            f"auto_prefix_strip: chose prefix='{best_pref}' with key_hits={best_hits}/{len(keys)}"
        )
        return stripped

    print(f"auto_prefix_strip: no prefix stripped; key_hits={best_hits}/{len(keys)}")
    return state


def _maybe_rename_head_keys_for_timm(state, model):
    """
    Score-relevant: alias common classifier naming to timm ViT 'head.*' when shapes match.
    """
    if not isinstance(state, dict):
        return state

    model_sd = model.state_dict()
    target = {}
    for k in ["head.weight", "head.bias"]:
        if k in model_sd:
            target[k] = tuple(model_sd[k].shape)

    if not target:
        return state

    candidates = [
        ("classifier.weight", "head.weight"),
        ("classifier.bias", "head.bias"),
        ("fc.weight", "head.weight"),
        ("fc.bias", "head.bias"),
        ("model.head.weight", "head.weight"),
        ("model.head.bias", "head.bias"),
        ("model.fc.weight", "head.weight"),
        ("model.fc.bias", "head.bias"),
        ("classifier.1.weight", "head.weight"),
        ("classifier.1.bias", "head.bias"),
    ]

    out = dict(state)
    for src, dst in candidates:
        if src in out and dst in target and dst not in out:
            v = out[src]
            if torch.is_tensor(v) and tuple(v.shape) == target[dst]:
                out[dst] = v
    return out


def _shape_safe_alias_keys(state, aliases, model):
    """
    Alias only if destination exists and shapes match (keeps core logic identical).
    """
    if not isinstance(state, dict):
        return state
    model_sd = model.state_dict()
    out = dict(state)
    for src, dst in aliases:
        if src in out and dst in model_sd and dst not in out:
            v = out[src]
            if torch.is_tensor(v) and tuple(v.shape) == tuple(model_sd[dst].shape):
                out[dst] = v
    return out


def _best_suffix_key_remap(state, model, min_match=50):
    """
    Score-critical minimal improvement: if checkpoint keys have unknown leading prefixes
    (e.g., 'something.backbone.model.blocks.0...'), prefix-stripping can't fix it.

    This remaps keys by matching the LONGEST suffix that exactly equals a model key.
    Applied only when it yields enough matches to matter, otherwise returns state unchanged.
    """
    if not isinstance(state, dict):
        return state

    model_keys = list(model.state_dict().keys())
    model_key_set = set(model_keys)

    last_token_index = {}
    for mk in model_keys:
        lt = mk.split(".")[-1]
        last_token_index.setdefault(lt, []).append(mk)

    remapped = {}
    matched = 0
    collisions = 0

    for k, v in state.items():
        if k in model_key_set:
            remapped[k] = v
            matched += 1
            continue

        lt = k.split(".")[-1]
        candidates = last_token_index.get(lt, [])
        best_mk = None
        best_len = -1
        for mk in candidates:
            if k.endswith(mk):
                if len(mk) > best_len:
                    best_len = len(mk)
                    best_mk = mk
        if best_mk is not None:
            if best_mk in remapped:
                collisions += 1
            remapped[best_mk] = v
            matched += 1
        else:
            remapped[k] = v

    if matched >= min_match:
        print(
            f"suffix_remap: enabled; matched={matched}/{len(state)} collisions={collisions}"
        )
        return remapped
    else:
        print(f"suffix_remap: not used; matched={matched}/{len(state)} (<{min_match})")
        return state


def _load_state(model, ckpt_obj, strict=False, tag="checkpoint"):
    state = _extract_state_dict(ckpt_obj)
    if not isinstance(state, dict):
        raise ValueError(
            f"{tag}: extracted state is not a dict; got type={type(state)}"
        )

    state = _best_suffix_key_remap(state, model, min_match=50)
    state = _auto_prefix_strip(state, model)
    state = _shape_safe_alias_keys(
        state,
        aliases=[
            ("fc.weight", "head.weight"),
            ("fc.bias", "head.bias"),
            ("head.weight", "fc.weight"),
            ("head.bias", "fc.bias"),
        ],
        model=model,
    )
    state = _maybe_rename_head_keys_for_timm(state, model)

    missing, unexpected = model.load_state_dict(state, strict=strict)
    loaded_keys = len(state) - len(unexpected)
    print(
        f"{tag}: attempted keys={len(state)} loaded_keys~={loaded_keys} missing={len(missing)} unexpected={len(unexpected)}"
    )
    if len(missing) > 0:
        print(f"{tag}: Missing keys (truncated):", missing[:10])
    if len(unexpected) > 0:
        print(f"{tag}: Unexpected keys (truncated):", unexpected[:10])
    return missing, unexpected


def _find_candidate_checkpoints():
    patterns = [
        "../input/**/*.pt",
        "../input/**/*.pth",
        "../input/**/*.bin",
    ]
    hits = []
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            lp = p.lower()
            if any(
                s in lp
                for s in ["/__pycache__/", ".ipynb_checkpoints", "sample_submission"]
            ):
                continue
            if os.path.isdir(p):
                continue
            score = 0
            base = os.path.basename(lp)
            if "cassava" in lp:
                score += 5
            if "vit" in lp:
                score += 3
            if "epoch" in lp or "best" in lp:
                score += 2
            if base.endswith(".pt"):
                score += 1
            size_mb = 0.0
            try:
                size_mb = os.path.getsize(p) / (1024 * 1024)
            except Exception:
                pass
            if size_mb < 1.0:
                score -= 10
            hits.append((score, size_mb, p))
    hits.sort(reverse=True)
    return hits


class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ViTBase16, self).__init__()

        self.model = timm.create_model("vit_base_patch16_224", pretrained=False)

        if pretrained and os.path.exists(model_path):
            state = torch.load(model_path, map_location="cpu")
            _load_state(self.model, state, strict=False, tag="base_weights")
            print(f"Loaded base weights from: {model_path}")
        else:
            if pretrained:
                print(
                    f"Base weights not found at {model_path}; proceeding without them."
                )

        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        return self.model(x)




## === cell 5
cassava_model = ViTBase16(n_classes=5, pretrained=True)

if (
    not isinstance(cassava_model.model.head, nn.Linear)
    or cassava_model.model.head.out_features != 5
):
    cassava_model.model.head = nn.Linear(cassava_model.model.head.in_features, 5)

train_csv_path = os.path.join(data_path, "train.csv")
if not os.path.exists(train_csv_path):
    nested = os.path.join(data_path, "cassava-leaf-disease-classification", "train.csv")
    train_csv_path = nested
train_df = pd.read_csv(train_csv_path)
majority_label = int(train_df["label"].value_counts().idxmax())
print("Majority label from train.csv:", majority_label)

head_w_before = cassava_model.model.head.weight.detach().cpu().clone()
head_b_before = cassava_model.model.head.bias.detach().cpu().clone()

probe_names = []
for n, p in cassava_model.model.named_parameters():
    if p.requires_grad and p.ndim >= 2 and ("head" not in n):
        probe_names.append(n)
    if len(probe_names) >= 6:
        break

probe_params_before = {
    n: dict(cassava_model.model.named_parameters())[n].detach().cpu().clone()
    for n in probe_names
}

loaded_ok = False
used_majority_fallback = False

ckpt_path_to_use = Cassava_model
if not os.path.exists(ckpt_path_to_use):
    candidates = _find_candidate_checkpoints()
    print(f"Hard-coded checkpoint not found: {ckpt_path_to_use}")
    print(f"Discovered checkpoint candidates: {len(candidates)}")
    if len(candidates) > 0:
        print("Top 5 candidates (score, size_mb, path):")
        for item in candidates[:5]:
            print(item)
        ckpt_path_to_use = candidates[0][2]
        print("Using discovered checkpoint:", ckpt_path_to_use)

if os.path.exists(ckpt_path_to_use):
    ckpt = torch.load(ckpt_path_to_use, map_location="cpu")

    try:
        _load_state(
            cassava_model.model,
            ckpt,
            strict=False,
            tag="finetuned_weights(timm_backbone_first)",
        )
        loaded_ok = True
        print(
            f"Loaded fine-tuned weights from: {ckpt_path_to_use} (timm_backbone_first)"
        )
    except Exception as e:
        print("Backbone-first fine-tuned load failed with:", repr(e))

    if not loaded_ok:
        try:
            _load_state(
                cassava_model,
                ckpt,
                strict=False,
                tag="finetuned_weights(wrapper_fallback)",
            )
            loaded_ok = True
            print(
                f"Loaded fine-tuned weights from: {ckpt_path_to_use} (wrapper_fallback)"
            )
        except Exception as e2:
            print("Wrapper fallback fine-tuned load also failed with:", repr(e2))
            print(
                "WARNING: Proceeding without fine-tuned weights; accuracy will likely be low."
            )
else:
    print(
        f"WARNING: Fine-tuned checkpoint not found at: {ckpt_path_to_use}. "
        "Proceeding with base/random weights so the notebook can run and produce a submission."
    )

head_w_after = cassava_model.model.head.weight.detach().cpu().clone()
head_b_after = cassava_model.model.head.bias.detach().cpu().clone()
delta_head_w = float((head_w_after - head_w_before).abs().mean().item())
delta_head_b = float((head_b_after - head_b_before).abs().mean().item())
print(f"Head weight mean(|delta|) after finetune load attempt: {delta_head_w:.6f}")
print(f"Head bias   mean(|delta|) after finetune load attempt: {delta_head_b:.6f}")

backbone_deltas = []
for n in probe_names:
    before = probe_params_before[n]
    after = dict(cassava_model.model.named_parameters())[n].detach().cpu().clone()
    delta = float((after - before).abs().mean().item())
    backbone_deltas.append(delta)
    print(f"Backbone probe '{n}' mean(|delta|) after load attempt: {delta:.6f}")

if os.path.exists(ckpt_path_to_use) and loaded_ok:
    mean_backbone_delta = float(np.mean(backbone_deltas)) if backbone_deltas else 0.0
    if (delta_head_w < 1e-6) and (mean_backbone_delta < 1e-7):
        print(
            "WARNING: Fine-tuned checkpoint load appears to be a no-op (head/backbone unchanged). "
            f"Will use majority-label fallback={majority_label} at submission time to avoid near-random score."
        )
        used_majority_fallback = True
else:
    used_majority_fallback = True

cassava_gpu_model = cassava_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cassava_gpu_model.parameters(), lr=1.5e-05)

cassava_gpu_model.eval()
type(cassava_gpu_model)



## === cell 6
test_img_names = []
if not os.path.isdir(test_path):
    raise FileNotFoundError(f"test_path does not exist: {test_path}")

for img in os.listdir(test_path):
    if img.lower().endswith(".jpg"):
        test_img_names.append(img)

print("Test folder:", test_path)
print("Testing Images:", len(test_img_names))
print("First 5:", test_img_names[:5])




## === cell 7
class TestSet2(Dataset):
    """Cassava Disease Dataset (test)"""

    def __init__(self, root_dir, test_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.test_dir = test_dir
        self.transform = transform

        self.files = sorted(
            [f for f in os.listdir(self.test_dir) if f.lower().endswith(".jpg")]
        )
        print(self.root_dir)
        print(self.test_dir)
        print("Cassava Disease Test Dataset Length = ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.test_dir, fname)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(img)
        else:
            image = transforms.ToTensor()(img)

        return (image, fname)




## === cell 8
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 9
testset = TestSet2(root_dir="", test_dir=test_path, transform=test_transform)
print(testset)



## === cell 10
test_batch_size = 16
test_loader = DataLoader(
    dataset=testset,
    batch_size=test_batch_size,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=2,
)
print(test_loader)



## === cell 11
images, names = next(iter(test_loader))
im = make_grid(images[:8], nrow=4)
print("Batch names example:", list(names)[:8])



## === cell 12
cassava_gpu_model.eval()

rows = []
tic = time.time()
with torch.no_grad():
    for X_test, names in test_loader:
        if used_majority_fallback:
            predicted = np.full((len(names),), majority_label, dtype=int)
        else:
            X_test = X_test.to(device, non_blocking=True)

            logits1 = cassava_gpu_model(X_test)
            logits2 = cassava_gpu_model(torch.flip(X_test, dims=[3]))  # horizontal flip
            y_test_pred = (logits1 + logits2) / 2.0

            predicted = (
                torch.argmax(y_test_pred, dim=1).detach().cpu().numpy().astype(int)
            )

        for fname, label in zip(list(names), predicted.tolist()):
            rows.append((fname, label))

toc = time.time() - tic
print("Time for model inference is ", toc)

submission_df = pd.DataFrame(rows, columns=["image_id", "label"])

sample_sub_path = os.path.join(data_path, "sample_submission.csv")
if not os.path.exists(sample_sub_path):
    nested = os.path.join(
        data_path, "cassava-leaf-disease-classification", "sample_submission.csv"
    )
    sample_sub_path = nested

sample_sub = pd.read_csv(sample_sub_path)
submission_df = sample_sub[["image_id"]].merge(submission_df, on="image_id", how="left")
submission_df["label"] = submission_df["label"].fillna(majority_label).astype(int)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", list(submission_df.columns))
print("Null labels after merge:", int(submission_df["label"].isna().sum()))
print("Label value counts (top):")
print(submission_df["label"].value_counts().head())
