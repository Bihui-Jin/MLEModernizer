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

0.8656693865216077

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I remove the notebook-style shell/cd/pip cells that don’t run in a plain Kaggle Python script environment and instead rely on the already-available Kaggle input dataset paths. Then I fix the `ModuleNotFoundError` by providing a small fallback EfficientNet-B5 implementation via `torchvision` (while still loading your `b5.pth` if it’s present), keeping the same single-model inference flow and argmax labeling. I also make the inference robust to missing CUDA and ensure test image file discovery matches the provided dataset structure. Finally, I always write a valid `submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.11584) has done: 'Your current score is near random because the checkpoint is likely not loading (or not matching) and the input normalization does not match EfficientNet’s expected preprocessing. To move the score toward the 0.865 target with minimal changes, I (1) make checkpoint loading robust to common key prefixes (`module.`, `model.`) and enforce strict loading when possible, and (2) switch preprocessing to standard ImageNet mean/std normalization while keeping your same CLAHE + resize + single-image inference + argmax flow. I also ensure we always output predictions for every image in `sample_submission.csv` order (no accidental drops from unreadable images). These are small, directly score-relevant fixes and should substantially improve accuracy if `b5.pth` is a real trained cassava checkpoint.'
- What this solution (achieved 0.11584) has done: 'Your current score suggests the model is effectively random, so the most likely issue is a checkpoint architecture mismatch (your code builds `torchvision.efficientnet_b5`, but many Cassava “B5” checkpoints were trained with `efficientnet_pytorch` and won’t load meaningfully). To move accuracy toward the 0.8657 target with minimal disruption, I keep your same single-model argmax inference, but add a robust fallback model builder: try `efficientnet_pytorch` (if available in the Kaggle environment) and otherwise use `torchvision`, then load the checkpoint in the first compatible way. I also make preprocessing match EfficientNet defaults more closely by resizing to 456 (EffNet-B5 default) while keeping your CLAHE + ImageNet normalization semantics. Finally, I keep the submission aligned exactly to `sample_submission.csv` order and still guarantee a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random, so the most likely cause is that the checkpoint isn’t actually being applied to the model you’re running (either no checkpoint is found, or the keys don’t match and most weights stay random). To move accuracy upward toward the 0.8657 target with minimal disruption, I (1) expand checkpoint discovery to include common Kaggle dataset locations under `/kaggle/input/**/b5*.pth`, and (2) make checkpoint loading handle the very common “wrapped” formats (e.g., `{"model": {"state_dict": ...}}`, `{"model_state_dict": ...}`) and also handle the case where the state dict corresponds to the EfficientNet backbone without the classifier head (by loading all matching keys and leaving only the head randomly initialized). I also add a single sanity check print showing what fraction of parameters were actually loaded, so you can immediately confirm you’re no longer running random weights; this is directly score-relevant while keeping your same single-model argmax inference and preprocessing flow.'
- What this solution (achieved 0.11584) has done: 'Your current score is near-random, which strongly suggests the checkpoint isn’t actually being used correctly (either not found, or loaded into a non-matching model so most weights remain random). I keep your single-model EfficientNet-B5 argmax inference and preprocessing flow, but make checkpoint discovery and loading more robust to common Kaggle Cassava checkpoint formats (nested dicts and key prefixes like `encoder.`/`backbone.`/`model.module.`), and ensure we only accept a loaded model if a meaningful fraction of parameters match by shape. I also align the final classifier replacement to handle both torchvision-style (`classifier[1]`) and efficientnet_pytorch-style (`_fc`) consistently, without changing the overall architecture choice you already have. These minimal changes should move accuracy up substantially toward your 0.8657 target if a real trained B5 checkpoint exists in `/kaggle/input`.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random, so the smallest likely fix is to ensure we’re actually loading the correct weights into the correct model keys. I keep the same single-model EfficientNet-B5 argmax inference, but (1) make checkpoint key “cleaning” less destructive (don’t strip essential prefixes like `features.`), and (2) add a tiny head-mapping shim for torchvision EfficientNet checkpoints where the classifier is saved as `classifier.weight/bias` (common in many training scripts) so it loads into `classifier.1.*`. These changes are narrowly targeted at getting a real trained checkpoint to load meaningfully, which should move accuracy up toward your target. Everything else (CLAHE + resize + ImageNet normalization + per-image loop + submission format/order) stays the same.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target (random-like), so the smallest meaningful path toward the target is to ensure the checkpoint is truly being applied to the model backbone and that the classifier head matches (5 classes) for the chosen EfficientNet implementation. I keep your single-model EfficientNet-B5 argmax inference and the same preprocessing flow (CLAHE + resize + ImageNet normalization), but make checkpoint extraction/key-cleaning less destructive and add common Cassava training-key remaps (`backbone.`, `encoder.`, `model.module.` etc.) so more weights actually match by name/shape. I also add a deterministic “acceptance” rule: if too few parameters match, we automatically try the other implementation (torchvision vs efficientnet_pytorch) and only select a variant that meaningfully loads (otherwise your score stays near random). This should move accuracy sharply upward toward the 0.8657 target when a real B5 Cassava checkpoint exists in `/kaggle/input`, without changing your overall modeling/inference semantics.'
- What this solution (achieved 0.11584) has done: 'Your current score is near-random, so the most direct way to move toward the 0.8657 target (without changing core inference semantics) is to ensure the checkpoint is actually compatible and being loaded in a way that matches the model’s parameter names. I make key-cleaning less destructive (stop stripping important prefixes like `backbone.`/`encoder.` that often must stay for matching), and instead add a small “key-alias” loader that tries several common prefix mappings and selects the one that matches the most tensors by name+shape. I also avoid CLAHE (which can shift the input distribution away from what many pretrained/finetuned EfficientNet checkpoints expect) by gating it off by default while keeping the rest of your preprocessing (resize + ImageNet normalization) and the same single-model argmax inference loop. These are minimal, score-relevant changes aimed at turning the run from “random weights” into “real checkpoint weights applied”, which should substantially increase accuracy if a real Cassava-trained B5 checkpoint exists.'
- What this solution (achieved 0.11584) has done: 'Your score (0.11584) is random-like and far below the target (0.8657), so the smallest likely improvement is to ensure the model is actually receiving inputs in the same scale it was trained with and that inference is done in a numerically-correct way for EfficientNet. I keep your single-model EfficientNet-B5 argmax inference exactly the same, but (1) switch `process()` to use the model’s native `weights.transforms()` preprocessing when using the torchvision EfficientNet implementation (this fixes resize/crop/interpolation/normalization mismatches), and (2) run inference under `torch.cuda.amp.autocast` on GPU for correctness/perf parity (no semantic change). Everything else (checkpoint discovery/loading logic, per-image loop, and submission alignment to `sample_submission.csv`) stays the same to minimize risk.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import tqdm

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
BASE = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(BASE, "test_images")

if not os.path.isdir(TEST_DIR):
    alt = "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images"
    if os.path.isdir(alt):
        TEST_DIR = alt

_ckpt_globs = [
    "/kaggle/input/**/b5.pth",
    "/kaggle/input/**/B5.pth",
    "/kaggle/input/**/b5*.pth",
    "/kaggle/input/**/efficientnet*b5*.pth",
    "/kaggle/input/**/*efficientnet*B5*.pth",
    "/kaggle/input/**/*effnet*b5*.pth",
    "/kaggle/input/**/*cassava*b5*.pth",
    "/kaggle/input/**/*.pth",
]
_ckpt_found = []
for pat in _ckpt_globs:
    _ckpt_found.extend(glob.glob(pat, recursive=True))
_ckpt_found = sorted({p for p in _ckpt_found if os.path.isfile(p)})

CKPT_CANDIDATES = [
    "b5.pth",
    "/kaggle/working/b5.pth",
    "/kaggle/input/b5v3checkpoint/b5.pth",
] + _ckpt_found

CKPT_PATH = next((p for p in CKPT_CANDIDATES if os.path.isfile(p)), None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"TEST_DIR={TEST_DIR}")
print(f"CKPT_PATH={CKPT_PATH}")




## === cell 2
def _extract_state_dict(ckpt_obj):
    """
    Score-relevant: robustly extract the actual tensor state_dict from common nested checkpoint formats,
    so we don't silently fall back to random weights.
    """
    if ckpt_obj is None:
        return None

    if isinstance(ckpt_obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
            "ema_state_dict",
            "student",
            "teacher",
            "checkpoint",
        ]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                inner = ckpt_obj[k]
                for kk in [
                    "state_dict",
                    "model_state_dict",
                    "model",
                    "net",
                    "weights",
                    "params",
                ]:
                    if kk in inner and isinstance(inner[kk], dict):
                        return inner[kk]
                return inner

        if all(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj

    return None


def _maybe_remap_torchvision_efficientnet_head_keys(state: dict) -> dict:
    """
    Score-relevant: load torchvision EfficientNet head if saved as classifier.weight/bias.
    """
    if not isinstance(state, dict):
        return state
    st = state
    if "classifier.weight" in st and "classifier.1.weight" not in st:
        st = dict(st)
        st["classifier.1.weight"] = st.pop("classifier.weight")
    if "classifier.bias" in st and "classifier.1.bias" not in st:
        st = dict(st)
        st["classifier.1.bias"] = st.pop("classifier.bias")
    return st


def _maybe_remap_effnet_pytorch_head_keys(state: dict) -> dict:
    """
    Score-relevant: load efficientnet_pytorch head under common alternative names.
    """
    if not isinstance(state, dict):
        return state
    st = state
    if "fc.weight" in st and "_fc.weight" not in st:
        st = dict(st)
        st["_fc.weight"] = st.pop("fc.weight")
    if "fc.bias" in st and "_fc.bias" not in st:
        st = dict(st)
        st["_fc.bias"] = st.pop("fc.bias")
    if "classifier.weight" in st and "_fc.weight" not in st:
        st = dict(st)
        st["_fc.weight"] = st.pop("classifier.weight")
    if "classifier.bias" in st and "_fc.bias" not in st:
        st = dict(st)
        st["_fc.bias"] = st.pop("classifier.bias")
    return st


def build_model_torchvision(num_classes: int = 5) -> nn.Module:
    m = models.efficientnet_b5(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


def build_model_efficientnet_pytorch(num_classes: int = 5) -> nn.Module:
    from efficientnet_pytorch import EfficientNet  # may or may not be available

    m = EfficientNet.from_name("efficientnet-b5")
    in_features = m._fc.in_features
    m._fc = nn.Linear(in_features, num_classes)
    return m


def _count_name_shape_matches(model: nn.Module, sd: dict):
    msd = model.state_dict()
    matched_tensors = 0
    total_tensors = len(msd)
    matched_elems = 0
    total_elems = 0
    for k, v in msd.items():
        total_elems += int(v.numel())
        if k in sd and hasattr(sd[k], "shape") and tuple(sd[k].shape) == tuple(v.shape):
            matched_tensors += 1
            matched_elems += int(v.numel())
    return matched_tensors, total_tensors, matched_elems, total_elems


def _apply_prefix_mapping(sd: dict, mapping_pairs):
    """
    Score-relevant: instead of stripping many prefixes (which can destroy correct names),
    try a small set of alias mappings and pick the one that maximizes name+shape matches.
    mapping_pairs: list of (old_prefix, new_prefix) applied only when key startswith old_prefix.
    """
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        nk = k
        for old, new in mapping_pairs:
            if nk.startswith(old):
                nk = new + nk[len(old) :]
        out[nk] = v
    return out


def _candidate_mappings():
    """
    Score-relevant: common training scripts wrap the model under different prefixes.
    We try a few safe, minimal mappings rather than aggressive stripping.
    """
    return [
        [],  # identity
        [("module.", "")],
        [("model.", "")],
        [("model.module.", "")],
        [("net.", "")],
        [("state_dict.", "")],
        [("encoder.", "")],
        [("backbone.", "")],
        [("feature_extractor.", "")],
        [("extractor.", "")],
        [("student.", "")],
        [("teacher.", "")],
        [("ema.", "")],
        [("model.", ""), ("module.", "")],
        [("model.module.", ""), ("module.", ""), ("model.", "")],
    ]


def try_build_and_load(num_classes: int = 5):
    model_variants = []
    try:
        model_variants.append(
            ("efficientnet_pytorch", build_model_efficientnet_pytorch(num_classes))
        )
    except Exception:
        pass
    model_variants.append(("torchvision", build_model_torchvision(num_classes)))

    load_report = []
    raw_state = None
    if CKPT_PATH is not None:
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        raw_state = _extract_state_dict(ckpt)

    best = None  # (score_tuple, model, tag, strict_loaded, loaded_sd)
    for tag, m in model_variants:
        if raw_state is None:
            load_report.append(
                f"No checkpoint found; using random weights with {tag} (will score poorly)."
            )
            best = ((-1, -1, -1.0), m, tag, False, None)
            break

        base_sd = raw_state
        if tag == "torchvision":
            base_sd = _maybe_remap_torchvision_efficientnet_head_keys(base_sd)
        else:
            base_sd = _maybe_remap_effnet_pytorch_head_keys(base_sd)

        best_for_variant = None
        for mp in _candidate_mappings():
            sd_try = _apply_prefix_mapping(base_sd, mp)
            mt, tt, me, te = _count_name_shape_matches(m, sd_try)
            score_tuple = (mt, me, me / max(1, te))
            if best_for_variant is None or score_tuple > best_for_variant[0]:
                best_for_variant = (score_tuple, sd_try, mp)

        (mt, me, frac) = best_for_variant[0]
        sd_best = best_for_variant[1]
        mp_best = best_for_variant[2]
        load_report.append(
            f"{tag}: best key-mapping matched_tensors={mt}/{len(m.state_dict())}, matched_frac={frac:.4f}, mapping={mp_best}"
        )

        try:
            m.load_state_dict(sd_best, strict=True)
            strict_loaded = True
            load_report.append(f"{tag}: strict load succeeded.")
        except Exception as e_strict:
            incompat = m.load_state_dict(sd_best, strict=False)
            strict_loaded = False
            missing = list(getattr(incompat, "missing_keys", []))
            unexpected = list(getattr(incompat, "unexpected_keys", []))
            load_report.append(
                f"{tag}: strict load failed ({repr(e_strict)}); non-strict used (missing={len(missing)}, unexpected={len(unexpected)})."
            )

        score_tuple = (mt, me, frac)
        if best is None or score_tuple > best[0]:
            best = (score_tuple, m, tag, strict_loaded, sd_best)

    (score_tuple, model, model_tag, strict_loaded, used_sd) = best
    mt, me, frac = score_tuple

    if CKPT_PATH is not None and mt < 50:
        load_report.append(
            f"WARNING: matched_tensors={mt} is very low; model may still be effectively random (score ~0.1)."
        )

    return model, model_tag, strict_loaded, mt, frac, load_report


model, model_tag, strict_loaded, matched_tensors, matched_frac, report = (
    try_build_and_load(num_classes=5)
)

for line in report:
    print(line)

print(
    f"Selected model impl: {model_tag} (strict_loaded={strict_loaded}, matched_tensors={matched_tensors}, matched_frac≈{matched_frac:.4f})"
)

model = model.to(device).eval()




## === cell 3
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

_TV_HAS_WEIGHTS = hasattr(models, "EfficientNet_B5_Weights")
_TV_B5_WEIGHTS = (
    models.EfficientNet_B5_Weights.IMAGENET1K_V1 if _TV_HAS_WEIGHTS else None
)
_tv_preprocess = _TV_B5_WEIGHTS.transforms() if _TV_B5_WEIGHTS is not None else None

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
    1, 3, 1, 1
)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
    1, 3, 1, 1
)

IMG_SIZE = 456
USE_CLAHE = False


@torch.no_grad()
def process(image_bgr: np.ndarray) -> torch.Tensor:
    if USE_CLAHE:
        img0 = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img0 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        img0 = image_bgr

    img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    if (model_tag == "torchvision") and (_tv_preprocess is not None):
        x = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()
        x = _tv_preprocess(x).unsqueeze(0)
        return x.to(device)

    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return x.to(device)




## === cell 4
files = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
if len(files) == 0:
    raise FileNotFoundError(f"No test images found in {TEST_DIR}")

sample_path = os.path.join(BASE, "sample_submission.csv")
if not os.path.isfile(sample_path):
    raise FileNotFoundError(f"sample_submission.csv not found at {sample_path}")

sample = pd.read_csv(sample_path)
id_to_path = {os.path.basename(p): p for p in files}

names, labels = [], []
fallback_label = 0  # updated after first successful prediction

use_amp = device.type == "cuda"

for image_id in tqdm.tqdm(sample["image_id"].tolist(), total=len(sample)):
    file = id_to_path.get(image_id, None)
    if file is None:
        names.append(image_id)
        labels.append(fallback_label)
        continue

    img = cv2.imread(file)
    if img is None:
        names.append(image_id)
        labels.append(fallback_label)
        continue

    x = process(img)

    with torch.cuda.amp.autocast(enabled=use_amp):
        out = model(x)
    pred = int(torch.argmax(out, dim=1).detach().cpu().item())

    names.append(image_id)
    labels.append(pred)
    fallback_label = pred




## === cell 5
sub = pd.DataFrame({"image_id": names, "label": labels})

sub = sample[["image_id"]].merge(sub, on="image_id", how="left")
if sub["label"].isna().any():
    fill = int(pd.Series(labels).mode().iloc[0]) if len(labels) else 0
    sub["label"] = sub["label"].fillna(fill).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(
    f"Wrote {out_path} with shape {sub.shape} (model_impl={model_tag}, ckpt={CKPT_PATH}, matched_tensors={matched_tensors}, matched_frac≈{matched_frac:.4f}, use_clahe={USE_CLAHE})"
)
print(sub.head())
