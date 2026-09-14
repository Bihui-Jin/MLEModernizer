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

0.8757932910244787

# 6. Current score

0.08595

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.38117) has done: 'I make the model checkpoint loading robust so the notebook runs even when the external `/kaggle/input/ensemblev5/*.pth` files are unavailable, by falling back to `pretrained=True` weights (same architectures, same inference logic). I also fix the downstream `NameError` by ensuring the model variables are always defined and moved to the correct device. To avoid internet dependency/timeouts, I disable MiDaS depth-cropping by default (it currently uses `torch.hub` which can fail offline), while keeping the rest of preprocessing and ensemble averaging unchanged. Finally, I keep the submission creation logic but ensure it always writes a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved 0.1136) has done: 'Your current low score is mainly because the “fallback” models are ImageNet-pretrained but not cassava-fine-tuned, so accuracy collapses. To move toward the target with minimal logic changes, I (1) load the official cassava fine-tuned weights from Kaggle’s public `cassava-model` dataset if present, keeping the same two-model ensemble and identical inference flow, and only fall back to ImageNet if those weights truly aren’t available. I also (2) fix a bug in `load_state_dict_forgiving` where non-`module.` keys get mishandled, which can silently break checkpoint loading and hurt accuracy. Everything else (preprocessing, MiDaS disabled by default, averaging logits, submission formatting) stays the same.'
- What this solution (achieved 0.10575) has done: 'Your score is far below the target, so the most likely issue is that the cassava-finetuned checkpoints are still not being found/loaded and you’re effectively submitting ImageNet-pretrained predictions. I make checkpoint discovery robust by searching all plausible Kaggle input locations (including nested competition folder copies) and also accept common filename variants, while keeping the exact same two-model ensemble and inference logic. I also harden checkpoint loading to correctly handle common checkpoint formats (`state_dict`, `model`, `model_state_dict`) and strip prefixes (`module.`, `model.`) without altering any other behavior. These minimal changes should substantially increase accuracy toward your target if the finetuned weights are present anywhere in `/kaggle/input`.'
- What this solution (achieved 0.2272) has done: 'Your current score strongly suggests the finetuned cassava checkpoints still aren’t being loaded (so you’re effectively using ImageNet-pretrained classifiers). I keep the exact same two-model timm ensemble and inference flow, but (1) broaden checkpoint discovery to include common file extensions and names, and (2) make the state_dict loading more robust to additional real-world key prefixes (e.g., `backbone.` / `encoder.`) that can otherwise cause most weights to remain randomly initialized. Finally, I add a small sanity print that reports whether each model ended up using finetuned weights vs fallback, so you can confirm the fix correlates with the score jump toward the target.'
- What this solution (achieved 0.1648) has done: 'Your score (0.2272) is far below the target (0.8758), so we should only make minimal, high-impact fixes that increase accuracy without changing the core ensemble/inference logic. The most likely root cause is that the finetuned checkpoints are still not being loaded correctly: `create_model(..., pretrained=False, num_classes=5)` initialize a new 5-class head, but many finetuned checkpoints were saved with different classifier key names (e.g., `classifier.*`, `fc.*`, `head.fc.*`), so the head stays random and accuracy collapses. I keep the exact same two-model timm ensemble and averaging, but (1) add a “head remap” step that maps common checkpoint classifier keys into the current model’s classifier keys when shapes match, and (2) report how many classifier tensors were actually loaded so you can verify you’re not running with a random head. This is a surgical change confined to checkpoint loading and should move accuracy sharply toward the target if finetuned weights exist in the environment.'
- What this solution (achieved 0.11996) has done: 'Your current score is far below the target, so we should only make minimal, high-impact fixes that increase accuracy without changing the ensemble/inference logic. The biggest likely issue is that even when a finetuned checkpoint is found, it may have been saved from a different timm version/variant (or wrapped in a different module structure), so most weights silently fail to load and you effectively run with a near-random 5-class head. I keep the same two-model timm ensemble and the same preprocessing/inference loop, but improve checkpoint compatibility by (1) creating the model using `checkpoint_path` when available (timm’s native loader handles many key-name differences), then (2) falling back to your forgiving loader, and finally (3) if still not finetuned, refuse the ImageNet 5-class head by loading pretrained backbone first and then replacing the classifier (this avoids random-head behavior). These changes are confined to model/weight loading and should move accuracy substantially toward the target while preserving the core approach.'
- What this solution (achieved 0.08595) has done: 'Your score is extremely low for this competition, which is consistent with models running with the wrong head / partially-loaded checkpoints (effectively near-random predictions). I keep your ensemble and preprocessing exactly the same, but make checkpoint loading stricter in a targeted way: if the finetuned checkpoint isn’t actually compatible with a 5-class head, we rebuild the model with the checkpoint’s own `num_classes` (when detectable) so the head loads correctly and then use that for inference. I also switch prediction from `argmax(softmax(logits))` to `argmax(logits)` (mathematically identical) to remove any tiny numeric instability, and I print a concise “what loaded” summary so you can confirm you’re not silently falling back to ImageNet. Submission writing/ordering stays unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import json
import warnings

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import timm

warnings.filterwarnings("ignore")

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(DATA_DIR), f"Missing DATA_DIR: {DATA_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing TEST_IMG_DIR: {TEST_IMG_DIR}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv: {SAMPLE_SUB_PATH}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)




## === cell 1
def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    return ckpt_obj


def _strip_state_dict_prefixes(sd: dict) -> dict:
    """
    Score fix: normalize common training-time prefixes so weights actually load into the
    current timm model (otherwise many tensors remain random).
    """
    if not isinstance(sd, dict):
        return sd

    strip_prefixes = ["module.", "model.", "net.", "backbone.", "encoder."]

    def _strip(k: str) -> str:
        changed = True
        while changed:
            changed = False
            for p in strip_prefixes:
                if k.startswith(p):
                    k = k[len(p) :]
                    changed = True
        return k

    new_sd = {}
    for k, v in sd.items():
        if isinstance(k, str):
            new_sd[_strip(k)] = v
    return new_sd


def _infer_num_classes_from_sd(sd: dict) -> int | None:
    """
    Score fix: if the checkpoint was saved with a different classifier naming, creating
    the model with the wrong num_classes causes the head not to load (catastrophic).
    We try to infer num_classes from the head weight shape.
    """
    if not isinstance(sd, dict):
        return None
    cand_keys = [
        "classifier.weight",
        "fc.weight",
        "head.fc.weight",
        "head.weight",
        "last_linear.weight",
    ]
    for k in cand_keys:
        w = sd.get(k, None)
        if hasattr(w, "shape") and len(w.shape) == 2:
            nc = int(w.shape[0])
            if 2 <= nc <= 1000:
                return nc
    return None


def _remap_classifier_keys_to_model(sd: dict, model: torch.nn.Module) -> dict:
    """
    Score fix: remap common classifier key names so the 5-class head loads correctly
    (otherwise head stays random and accuracy collapses).
    """
    if not isinstance(sd, dict):
        return sd

    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())

    dst_weight = None
    dst_bias = None
    for cand_w, cand_b in [
        ("classifier.weight", "classifier.bias"),
        ("fc.weight", "fc.bias"),
        ("head.fc.weight", "head.fc.bias"),
        ("head.weight", "head.bias"),
        ("last_linear.weight", "last_linear.bias"),
    ]:
        if cand_w in model_keys and cand_b in model_keys:
            dst_weight, dst_bias = cand_w, cand_b
            break

    if dst_weight is None:
        return sd

    src_pairs = [
        ("classifier.weight", "classifier.bias"),
        ("fc.weight", "fc.bias"),
        ("head.fc.weight", "head.fc.bias"),
        ("head.weight", "head.bias"),
        ("last_linear.weight", "last_linear.bias"),
    ]

    src_weight = None
    src_bias = None
    for sw, sb in src_pairs:
        if sw in sd and sb in sd:
            src_weight, src_bias = sw, sb
            break

    if src_weight is None:
        return sd

    try:
        if tuple(sd[src_weight].shape) == tuple(model_sd[dst_weight].shape) and tuple(
            sd[src_bias].shape
        ) == tuple(model_sd[dst_bias].shape):
            if src_weight != dst_weight:
                sd[dst_weight] = sd[src_weight]
            if src_bias != dst_bias:
                sd[dst_bias] = sd[src_bias]
    except Exception:
        return sd

    return sd


def load_state_dict_forgiving(model: torch.nn.Module, checkpoint_path: str):
    """
    Score fix: robustly load finetuned weights, handling common checkpoint wrappers and
    prefixes. This directly affects accuracy because failing to load the head/backbone
    yields near-random predictions.
    """
    if checkpoint_path is None or (not os.path.exists(checkpoint_path)):
        raise FileNotFoundError(checkpoint_path)

    sd = torch.load(checkpoint_path, map_location="cpu")
    sd = _extract_state_dict(sd)

    if not isinstance(sd, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} is not a state_dict dict.")

    sd = _strip_state_dict_prefixes(sd)
    sd = _remap_classifier_keys_to_model(sd, model)

    missing, unexpected = model.load_state_dict(sd, strict=False)

    model_keys = set(model.state_dict().keys())
    head_keys = [
        k
        for k in [
            "classifier.weight",
            "fc.weight",
            "head.fc.weight",
            "last_linear.weight",
            "head.weight",
        ]
        if k in model_keys
    ]
    head_loaded = [k for k in head_keys if (k in sd)]
    print(
        f"Loaded {os.path.basename(checkpoint_path)} via forgiving loader: missing={len(missing)}, unexpected={len(unexpected)}, head_loaded={head_loaded}"
    )
    return model


def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def expand_ckpt_candidates(candidate_files):
    """
    Score fix: broaden checkpoint discovery patterns under /kaggle/input to actually find
    finetuned weights if they exist.
    """
    roots = [
        "/kaggle/input",
        "/kaggle/input/cassava-model",
        "/kaggle/input/ensemblev5",
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
    out = []
    out.extend(candidate_files)

    exts = [".pth", ".pt", ".bin"]

    basenames = list({os.path.basename(p) for p in candidate_files if p})
    for r in roots:
        for bn in basenames:
            out.append(os.path.join(r, bn))
            out.append(os.path.join(r, "**", bn))

            base, ext = os.path.splitext(bn)
            if ext.lower() in exts:
                for e in exts:
                    out.append(os.path.join(r, base + e))
                    out.append(os.path.join(r, "**", base + e))

    alt_names = [
        "eff_best.pth",
        "eff_best.pt",
        "eff_best.bin",
        "efficientnet_b5.pth",
        "tf_efficientnet_b5.pth",
        "tf_efficientnet_b5_ns.pth",
        "efficientnetb5.pth",
        "eff.pth",
        "seresnext_best.pth",
        "seresnext_best.pt",
        "seresnext_best.bin",
        "seresnext101_32x4d.pth",
        "seresnext101.pth",
        "se_resnext101_32x4d.pth",
        "seresnext.pth",
    ]
    for r in roots:
        for bn in alt_names:
            out.append(os.path.join(r, bn))
            out.append(os.path.join(r, "**", bn))

    expanded = []
    for p in out:
        if "*" in p:
            expanded.extend(sorted(glob.glob(p, recursive=True)))
        else:
            expanded.append(p)

    seen = set()
    uniq = []
    for p in expanded:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def create_model_with_best_effort_weights(
    model_name: str, ckpt_path: str | None, num_classes: int = 5
):
    """
    Score fix (minimal, preserves ensemble/inference): If a finetuned checkpoint exists but
    was saved with a different classifier naming/shape, creating the model with num_classes=5
    can cause the head to not load. We infer num_classes from the checkpoint head weight and
    instantiate accordingly, then load weights. Inference still outputs argmax over 5 classes
    when the checkpoint is actually cassava-finetuned.
    """
    loaded_finetuned = False
    model = None

    if ckpt_path is not None and os.path.exists(ckpt_path):
        try:
            raw = torch.load(ckpt_path, map_location="cpu")
            sd0 = _extract_state_dict(raw)
            if isinstance(sd0, dict):
                sd0 = _strip_state_dict_prefixes(sd0)
                inferred_nc = _infer_num_classes_from_sd(sd0)
            else:
                inferred_nc = None
        except Exception as e:
            print(
                f"{model_name}: could not inspect checkpoint for num_classes: {repr(e)}"
            )
            inferred_nc = None

        effective_nc = inferred_nc if inferred_nc is not None else num_classes
        if inferred_nc is not None and inferred_nc != num_classes:
            print(
                f"{model_name}: inferred checkpoint num_classes={inferred_nc} (requested {num_classes}); will instantiate with inferred value to load head correctly."
            )

        try:
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=effective_nc,
                checkpoint_path=ckpt_path,
            )
            loaded_finetuned = True
            print(
                f"{model_name}: loaded finetuned via timm checkpoint_path from {ckpt_path}"
            )
            return model, loaded_finetuned
        except Exception as e:
            print(
                f"{model_name}: timm checkpoint_path load failed, will try forgiving loader. Error: {repr(e)}"
            )

        try:
            model = timm.create_model(
                model_name, pretrained=False, num_classes=effective_nc
            )
            model = load_state_dict_forgiving(model, ckpt_path)
            loaded_finetuned = True
            print(
                f"{model_name}: loaded finetuned via forgiving loader from {ckpt_path}"
            )
            return model, loaded_finetuned
        except Exception as e:
            print(
                f"{model_name}: forgiving load failed; will fall back to pretrained=True. Error: {repr(e)}"
            )

    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    loaded_finetuned = False
    print(f"{model_name}: no finetuned ckpt loaded; using pretrained=True fallback.")
    return model, loaded_finetuned


eff_ckpt_candidates = expand_ckpt_candidates(
    [
        "/kaggle/input/ensemblev5/eff_best.pth",
        "/kaggle/input/cassava-model/eff_best.pth",
        "/kaggle/input/cassava-model/tf_efficientnet_b5.pth",
        "/kaggle/input/cassava-model/efficientnet_b5.pth",
    ]
)
se_ckpt_candidates = expand_ckpt_candidates(
    [
        "/kaggle/input/ensemblev5/seresnext_best.pth",
        "/kaggle/input/cassava-model/seresnext_best.pth",
        "/kaggle/input/cassava-model/seresnext101_32x4d.pth",
        "/kaggle/input/cassava-model/seresnext101.pth",
    ]
)

eff_ckpt = find_first_existing(eff_ckpt_candidates)
se_ckpt = find_first_existing(se_ckpt_candidates)

print("EfficientNet ckpt:", eff_ckpt)
print("SE-ResNeXt ckpt:", se_ckpt)

efficient, efficient_loaded_finetuned = create_model_with_best_effort_weights(
    "tf_efficientnet_b5", eff_ckpt, num_classes=5
)
seresnext, seresnext_loaded_finetuned = create_model_with_best_effort_weights(
    "seresnext101_32x4d", se_ckpt, num_classes=5
)

efficient.eval().to(device)
seresnext.eval().to(device)

print("Models are ready.")
print(
    f"Finetuned loaded? efficient={efficient_loaded_finetuned}, seresnext={seresnext_loaded_finetuned}\n"
)



## === cell 2
USE_MIDAS = False
midas = None
transform = None

if USE_MIDAS:
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=True)
        midas.to(device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.default_transform
        print("MiDaS loaded.")
    except Exception as e:
        USE_MIDAS = False
        print(
            "MiDaS could not be loaded; proceeding without depth-based cropping.\nError:",
            repr(e),
        )


def processor(image_bgr: np.ndarray) -> torch.Tensor:
    img = (
        cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        / 255.0
    )
    img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    image = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    return image


def get_depth(img_bgr: np.ndarray) -> np.ndarray:
    if (not USE_MIDAS) or (midas is None) or (transform is None):
        h, w = img_bgr.shape[:2]
        return np.ones((h, w), dtype=bool)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = (
            F.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
    output = prediction.detach().float().cpu().numpy()
    img_min = float(np.min(output))
    img_max = float(np.max(output))
    thr = (img_min + img_max) / 3.0
    return output > thr


def crop_image(image_bgr: np.ndarray, depth_mask: np.ndarray) -> np.ndarray:
    depth = depth_mask.astype(np.uint8)
    mask_3d = np.stack((depth, depth, depth), axis=2)
    masked_arr = np.where(mask_3d == 1, image_bgr, 0).astype(np.uint8)

    c = np.where(masked_arr != 0)
    if c[0].size == 0 or c[1].size == 0:
        return image_bgr

    x_max = int(np.max(c[1]))
    x_min = int(np.min(c[1]))
    y_max = int(np.max(c[0]))
    y_min = int(np.min(c[0]))

    h, w = image_bgr.shape[:2]
    x_min = max(0, min(x_min, w - 1))
    x_max = max(0, min(x_max, w - 1))
    y_min = max(0, min(y_min, h - 1))
    y_max = max(0, min(y_max, h - 1))
    if x_max <= x_min or y_max <= y_min:
        return image_bgr

    return masked_arr[y_min:y_max, x_min:x_max]




## === cell 3
sample_df = pd.read_csv(SAMPLE_SUB_PATH)
test_image_ids = sample_df["image_id"].tolist()

test_paths = [os.path.join(TEST_IMG_DIR, iid) for iid in test_image_ids]
missing_files = [p for p in test_paths if not os.path.exists(p)]
assert len(missing_files) == 0, f"Missing test images (first 5): {missing_files[:5]}"

names, labels = [], []

with torch.no_grad():
    for file_path, image_id in zip(test_paths, test_image_ids):
        img = cv2.imread(file_path)
        if img is None:
            names.append(image_id)
            labels.append(0)
            continue

        depth = get_depth(img)
        img_c = crop_image(img, depth)
        x = processor(img_c)

        se_out = seresnext(x)
        eff_out = efficient(x)
        total = (se_out + eff_out) / 2.0

        pred = int(torch.argmax(total, dim=1).detach().cpu().item())

        names.append(image_id)
        labels.append(pred)

sub_df = pd.DataFrame({"image_id": names, "label": labels})
sub_df = sub_df.set_index("image_id").loc[test_image_ids].reset_index()

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub_df.head())
print("rows:", len(sub_df))
assert out_path.endswith(".csv") and os.path.exists(out_path)
assert list(sub_df.columns) == ["image_id", "label"]
assert len(sub_df) == len(sample_df)
