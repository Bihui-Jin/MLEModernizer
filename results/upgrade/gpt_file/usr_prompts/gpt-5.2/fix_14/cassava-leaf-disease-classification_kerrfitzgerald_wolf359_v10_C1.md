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

0.852825627077667

# 6. Current score

0.32623

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24477) has done: 'I fix the two issues that prevent an end-to-end run and submission creation: (1) the fine-tuned checkpoint path is missing, so I make the loader robust by falling back to a standard pretrained ViT (timm weights) when the Kaggle input checkpoint isn’t available, and (2) the model weights are currently on CPU while inputs are on GPU, so I ensure the model is moved to the same device after all weight-loading. These changes preserve the core ViT architecture and inference flow, but guarantee the notebook produces a valid `submission.csv`. This should also yield a reasonable score (likely below your target without the provided fine-tuned weights, but it run and submit correctly).'
- What this solution (achieved 0.50747) has done: 'Your score is far below the target, and the main cause is that your notebook is likely not loading the intended fine-tuned checkpoint (so it falls back to a generic ImageNet ViT head for 5 classes, which performs near-random on cassava). I keep your ViT architecture and inference flow identical, but make the checkpoint loading robust to common training wrappers (e.g., `module.`, `model.`, `net.`, `state_dict` nesting) so the fine-tuned weights actually land in the model. I also ensure the fallback still uses your `ViTBase16` class (same head wiring) and prefer `pretrained=True` there, so even without the finetuned file you don’t end up with an untrained head by accident. These are minimal, execution-safe changes aimed specifically at moving accuracy up toward your target without changing the modeling approach.'
- What this solution (achieved 0.537) has done: 'Your current score (0.50747) is far below the target (0.85283), so we should safely increase performance without changing the model architecture or training approach. The biggest likely remaining issue is that the fine-tuned checkpoint may still not be loading correctly due to key mismatches (e.g., `head.*` vs `model.head.*`, or nested prefixes), so I make the checkpoint loader map keys onto your `ViTBase16` wrapper deterministically and verify that head weights load. I also ensure test-time preprocessing matches typical ViT fine-tuning (center-crop style) while keeping image size and normalization the same, as a minimal transform tweak that can improve accuracy without altering semantics. Finally, I add lightweight sanity prints (counts of loaded/missing keys and head weight stats) to confirm the intended weights are actually used before producing `submission.csv`.'
- What this solution (achieved 0.32623) has done: 'Your score is still far below the target, so we should safely increase accuracy without changing the model or inference logic. The highest-probability remaining issue is input preprocessing mismatch: timm ViT models expect the timm default interpolation/crop behavior and normalization tied to the model’s `default_cfg`; using generic torchvision transforms can materially hurt accuracy even with correct weights. I keep your same Resize→CenterCrop→Normalize pipeline, but switch it to timm’s `resolve_data_config` + `create_transform` for the exact model variant you’re using, which is a minimal, metric-aligned change. I also add a small safeguard to ensure we’re reading the correct `sample_submission.csv` path if `data_path` is off, without changing outputs otherwise.'
- What this solution (achieved 0.32623) has done: 'Your current score is far below the target, so we should improve accuracy with the smallest changes that most likely fix a “wrong weights loaded” situation without changing your ViT architecture or inference loop. The most probable remaining issue is that your remapping always forces keys to `model.*`, but many fine-tuned checkpoints already contain `model.*` (or a mix), causing `model.model.*` mismatches and silently dropping important weights under `strict=False`. I make the checkpoint key normalization smarter (only add `model.` when needed, and also handle `backbone.*` / `encoder.*`), and I print a concise load report (missing/unexpected + whether head weights loaded) to verify the fine-tuned weights truly land. This keeps the same model, transforms, and argmax prediction, but should move the score up toward your target by actually using the intended fine-tuned parameters.'
- What this solution (achieved 0.32623) has done: 'Your current score (0.32623) is far below the target (0.85283), so we should increase accuracy with the smallest changes most likely to fix “fine-tuned weights not actually loaded.” I keep your ViT model, inference loop, and argmax unchanged, but make the checkpoint key mapping explicitly compatible with common ViT fine-tune saves where the classifier is stored as `head.*` while your wrapper expects `model.head.*`. I also add a tiny key-rewrite that maps `fc.*` or `classifier.*` to `head.*` (another common convention) and print a concise confirmation that head weights were loaded (to avoid silently running with an effectively random head). These changes are directly score-relevant and should move accuracy upward toward your target without changing the architecture or training semantics, and the script still write a valid `submission.csv`.'
- What this solution (achieved 0.32623) has done: 'Your current score is far below the target, so we should increase accuracy with the smallest change most likely to fix a “wrong head weights loaded” problem while keeping your ViT architecture and argmax inference unchanged. Right now, your loader strips the `model.` prefix and then re-adds it unconditionally, which can easily cause key mismatches if the checkpoint already contains wrapper-style keys and lead to silently using an effectively random classifier head. I replace that logic with a single “normalize keys to exactly match this model’s state_dict” routine (auto-handling `module.`, `model.`, `backbone.` etc. and mapping `fc/classifier`→`head`) and then load strictly when possible. This should make the fine-tuned checkpoint actually land in `cassava_model.model.*`, improving score toward your target without changing the core modeling approach.'
- What this solution (achieved 0.32623) has done: 'Your score is far below the target, so the most likely “minimal-change” path is to ensure the fine-tuned checkpoint actually loads fully (especially the classifier head) rather than partially/silently. I keep your ViT model, argmax inference, and timm transform logic the same, but improve the checkpoint key normalization to explicitly map common nesting like `model.model.*` and to handle the frequent case where a checkpoint stores `head.*` while your module expects `model.head.*`. I also tighten loading to prefer a strict load when the remapped keys perfectly match, and print a concise confirmation that head weights were loaded (without changing outputs). These changes are directly score-relevant and should move accuracy upward toward your target while preserving the core approach.'
- What this solution (achieved 0.32623) has done: 'Your current score (0.32623) is far below the target (0.85283), so the most likely minimal-change improvement is to ensure you’re actually using the intended fine-tuned weights and not silently falling back to an almost-random head. I keep your ViT model, argmax inference, and timm test transform unchanged, but make the checkpoint loader *guarantee* the classifier head gets loaded by (1) detecting/deriving the correct prefix mapping from the checkpoint to your model keys, and (2) explicitly remapping `head.*` ↔ `model.head.*` and common aliases while verifying the head tensors match and were loaded. If head weights still can’t be loaded, the code clearly report it (instead of silently proceeding), which is directly score-relevant. The rest of the pipeline (dataset, dataloader, submission merge) stays the same to preserve evaluation semantics and produce a valid `submission.csv`.'
- What this solution (achieved 0.32623) has done: 'Your score gap to the target is large, so the most likely minimal fix is ensuring you’re actually using the fine-tuned checkpoint (including the classifier head) instead of effectively running with an ImageNet/random head. I keep your ViT model and argmax inference unchanged, but (1) make the checkpoint search robust (the provided `../input/...` paths often differ), and (2) change the key-normalization to match checkpoint keys to your model keys by suffix (a common, safe way to resolve `module/model/backbone` prefix differences without changing architecture). I also enforce that if a checkpoint is found but the head can’t be loaded with the correct shape, we reinitialize the head deterministically and clearly report it (so we avoid silent partial loads that crush accuracy). The rest of your pipeline (timm transforms, dataloader, submission merge) stays the same.'
- What this solution (achieved 0.32623) has done: 'Your score is far below the target, so we should increase accuracy with the smallest change most likely to fix “fine-tuned weights aren’t actually being used.” The most common silent failure here is that the checkpoint is saved from a wrapper with different key names, and suffix-matching can accidentally map the wrong tensors (or drop most of them), leaving you effectively with a random/imagenet head. I keep your ViT model and argmax inference identical, but change the loader to: (1) choose the best-matching prefix mapping from the checkpoint to your model keys (instead of loose suffix mapping), (2) explicitly support `head.*`↔`model.head.*` and only accept head weights when shapes match, and (3) if a checkpoint exists but loads too few keys, fall back to timm pretrained backbone without reinitializing the head randomly. This is directly score-relevant and should move you materially toward the target if the fine-tuned file is present and compatible.'
- What this solution (achieved 0.32623) has done: 'Your score is far below the target, so we should increase accuracy with the smallest possible change that’s most likely to fix the remaining “wrong checkpoint / wrong weights loaded” issue without changing the ViT architecture or inference semantics. The key issue is that your loader currently skips loading the fine-tuned checkpoint if <60% of keys match; that’s a reasonable safety check generally, but it can wrongly reject valid fine-tuned checkpoints that only store a subset (e.g., model EMA, only backbone+head, or different naming), causing you to fall back to a weak pretrained-only model and tank accuracy. I keep your model, argmax inference, and timm transforms identical, but (1) lower that skip-threshold and (2) explicitly ensure head weights are loaded when present and shape-compatible (otherwise we still fall back), because head loading is the single most score-critical part. I also make the checkpoint search slightly more robust (still only under `../input`) to reduce the chance you’re simply not finding the intended `.pt` file.'

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
from torchvision.utils import make_grid

from PIL import Image
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
data_path = "../input/cassava-leaf-disease-classification/"
train_path = "../input/cassava-leaf-disease-classification/train_images/"
test_path = "../input/cassava-leaf-disease-classification/test_images/"

model_path = "../input/vitbase16224/jx_vit_base_p16_224-80ecf9dd.pth"
Cassava_model = (
    "../input/cassavanewaugtp95epochs3/CassavaViT_newaug_TP95_Epochs3_LR1-75e05.pt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



## === cell 2
wheel_path = "../input/timm034/timm-0.3.4-py3-none-any.whl"
if not os.path.exists(wheel_path):
    print(
        "timm wheel not found at:",
        wheel_path,
        " -> assuming timm is already available.",
    )
else:
    import sys, subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_path, "-q"])



## === cell 3
import timm



## === cell 4
print("Available ViT Models (first 20):")
print(timm.list_models("vit*")[:20])




## === cell 5
class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False):
        super(ViTBase16, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False)

        if pretrained:
            if os.path.exists(model_path):
                state = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(state)
                print("Loaded backbone weights from:", model_path)
            else:
                print(
                    "Backbone weights not found at:",
                    model_path,
                    " -> using timm pretrained=True weights for backbone.",
                )
                self.model = timm.create_model("vit_base_patch16_224", pretrained=True)

        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        return self.model(x)




## === cell 6
def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        if len(ckpt_obj) and all(isinstance(kk, str) for kk in ckpt_obj.keys()):
            return ckpt_obj
    return ckpt_obj


def _rewrite_classifier_aliases(k: str) -> str:
    if k.startswith("fc."):
        return "head." + k[len("fc.") :]
    if k.startswith("classifier."):
        return "head." + k[len("classifier.") :]
    return k


def _find_existing_checkpoint_path(path_like: str) -> str:
    if path_like and os.path.exists(path_like):
        return path_like

    base = os.path.basename(path_like) if path_like else ""
    if not base:
        return path_like

    root_dir = os.path.dirname(path_like)
    name_no_ext, ext = os.path.splitext(base)
    if root_dir and os.path.exists(root_dir):
        for try_ext in [ext, ".pt", ".pth", ".bin"]:
            if try_ext and (name_no_ext + try_ext) != base:
                cand = os.path.join(root_dir, name_no_ext + try_ext)
                if os.path.exists(cand):
                    return cand

    search_root = "../input"
    if not os.path.exists(search_root):
        return path_like

    targets = {base}
    if name_no_ext:
        for try_ext in [".pt", ".pth", ".bin"]:
            targets.add(name_no_ext + try_ext)

    for root, _, files in os.walk(search_root):
        fs = set(files)
        hit = targets.intersection(fs)
        if hit:
            chosen = sorted(hit)[0]
            return os.path.join(root, chosen)

    return path_like


def _prepare_ckpt_key(k: str) -> str:
    k = _rewrite_classifier_aliases(k)
    for p in ("module.",):
        while k.startswith(p):
            k = k[len(p) :]
    return k


def _choose_best_prefix_mapping(model: nn.Module, raw_sd: dict) -> tuple[dict, dict]:
    """
    Choose best deterministic mapping among common prefix conventions to maximize exact key matches.
    Returns (mapped_sd, stats).
    """
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())

    def t_identity(k: str) -> str:
        return k

    def t_add_model(k: str) -> str:
        return "model." + k

    def t_strip_model(k: str) -> str:
        return k[len("model.") :] if k.startswith("model.") else k

    def t_strip_backbone(k: str) -> str:
        return k[len("backbone.") :] if k.startswith("backbone.") else k

    def t_strip_encoder(k: str) -> str:
        return k[len("encoder.") :] if k.startswith("encoder.") else k

    def t_strip_net(k: str) -> str:
        return k[len("net.") :] if k.startswith("net.") else k

    def t_strip_model_model(k: str) -> str:
        return k[len("model.model.") :] if k.startswith("model.model.") else k

    candidates = [
        ("identity", t_identity),
        ("strip_model_model", t_strip_model_model),
        ("strip_model", t_strip_model),
        ("strip_backbone", t_strip_backbone),
        ("strip_encoder", t_strip_encoder),
        ("strip_net", t_strip_net),
        ("add_model", t_add_model),
    ]

    def _head_swap(k: str) -> list[str]:
        ks = [k]
        if k.startswith("head."):
            ks.append("model." + k)
        if k.startswith("model.head."):
            ks.append(k[len("model.") :])
        return ks

    best = None
    best_mapped = None

    for name, tfm in candidates:
        mapped = {}
        matched = 0
        shape_ok = 0

        for rk, rv in raw_sd.items():
            kk = _prepare_ckpt_key(rk)
            kk = tfm(kk)

            for ktry in _head_swap(kk):
                if ktry in model_keys:
                    mapped[ktry] = rv
                    matched += 1
                    if tuple(model_sd[ktry].shape) == tuple(rv.shape):
                        shape_ok += 1
                    break

        stats = {
            "name": name,
            "matched": matched,
            "shape_ok": shape_ok,
            "raw_keys": len(raw_sd),
            "model_keys": len(model_sd),
        }
        if best is None or (matched, shape_ok) > (best["matched"], best["shape_ok"]):
            best = stats
            best_mapped = mapped

    return best_mapped if best_mapped is not None else {}, (
        best if best is not None else {}
    )


def _filter_to_shape_compatible(model: nn.Module, sd: dict) -> dict:
    model_sd = model.state_dict()
    out = {}
    for k, v in sd.items():
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape):
            out[k] = v
    return out


def _load_finetuned_into_model(model: nn.Module, ckpt_path: str) -> bool:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_sd = _extract_state_dict(ckpt)
    if not isinstance(raw_sd, dict) or not raw_sd:
        print("Checkpoint did not contain a valid state_dict dict; skipping.")
        return False

    mapped_sd, stats = _choose_best_prefix_mapping(model, raw_sd)
    mapped_sd = _filter_to_shape_compatible(model, mapped_sd)

    model_sd = model.state_dict()
    expected_hw_key = (
        "model.head.weight" if "model.head.weight" in model_sd else "head.weight"
    )
    expected_hb_key = (
        "model.head.bias" if "model.head.bias" in model_sd else "head.bias"
    )
    head_loaded = (expected_hw_key in mapped_sd) and (expected_hb_key in mapped_sd)

    print("Checkpoint mapping choice:", stats)
    print("Mapped keys (shape-compatible):", len(mapped_sd), "/", len(model_sd))
    print("Head weights present & shape-compatible:", head_loaded)

    min_match_ratio = 0.20
    if len(mapped_sd) < int(min_match_ratio * len(model_sd)) and not head_loaded:
        print(
            f"WARNING: too few keys matched ({len(mapped_sd)}/{len(model_sd)}) and head not loaded. "
            "Skipping finetuned load to avoid wrong-weight partial load."
        )
        return False

    try:
        incompatible = model.load_state_dict(mapped_sd, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        print("Load report | strict=False")
        print(
            "Missing keys count:",
            len(missing),
            "| Unexpected keys count:",
            len(unexpected),
        )
        if any("head" in k for k in missing):
            print(
                "WARNING: head missing keys detected; finetuned accuracy will be poor."
            )
            return False
        return True
    except RuntimeError as e:
        print("ERROR during load_state_dict:", str(e))
        return False


model_path = _find_existing_checkpoint_path(model_path)
Cassava_model = _find_existing_checkpoint_path(Cassava_model)
print("Resolved model_path:", model_path)
print("Resolved Cassava_model:", Cassava_model)

cassava_model = ViTBase16(n_classes=5, pretrained=True)

loaded_finetuned = False
if os.path.exists(Cassava_model):
    loaded_finetuned = _load_finetuned_into_model(cassava_model, Cassava_model)
    print(
        "Loaded fine-tuned checkpoint from:",
        Cassava_model,
        "| success:",
        loaded_finetuned,
    )

    if not loaded_finetuned:
        print(
            "Falling back to timm-pretrained backbone (no finetuned weights applied)."
        )
        cassava_model = ViTBase16(n_classes=5, pretrained=True)
else:
    print(
        "Fine-tuned model checkpoint not found at:",
        Cassava_model,
        " -> using timm pretrained backbone weights instead.",
    )
    cassava_model = ViTBase16(n_classes=5, pretrained=True)

cassava_model = cassava_model.to(device)
cassava_model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cassava_model.parameters(), lr=1.5e-05)

with torch.no_grad():
    hw = cassava_model.model.head.weight.detach().float().cpu()
    hb = cassava_model.model.head.bias.detach().float().cpu()
    print("Head weight mean/std:", float(hw.mean()), float(hw.std()))
    print("Head bias mean/std:", float(hb.mean()), float(hb.std()))

cassava_model



## === cell 7
test_img_names = [f for f in os.listdir(test_path) if f.lower().endswith(".jpg")]
print("Testing Images:", len(test_img_names))
print("First 5:", test_img_names[:5])




## === cell 8
class TestSet2(Dataset):
    """Cassava Disease Test Dataset"""

    def __init__(self, root_dir, test_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.test_dir = test_dir
        self.transform = transform

        self.files = sorted(
            [f for f in os.listdir(self.test_dir) if f.lower().endswith(".jpg")]
        )
        print(root_dir)
        print(test_dir)
        print("Cassava Disease Test Dataset Length = ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_name = self.files[idx]
        img_path = os.path.join(self.test_dir, image_name)
        img = Image.open(img_path).convert("RGB")

        image = self.transform(img) if self.transform else img
        return (image, image_name)




## === cell 9
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

data_cfg = resolve_data_config({}, model=cassava_model.model)
test_transform = create_transform(**data_cfg, is_training=False)

print("timm data config:", data_cfg)
print("test_transform:", test_transform)



## === cell 10
testset = TestSet2(root_dir="", test_dir=test_path, transform=test_transform)
print(testset)



## === cell 11
test_batch_size = 32
test_loader = DataLoader(
    dataset=testset,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
print(test_loader)



## === cell 12
images, names = next(iter(test_loader))
im = make_grid(images[:8], nrow=4)
print("Example names:", list(names[:8]))
plt.figure(figsize=(12, 4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
plt.axis("off")
plt.show()



## === cell 13
rows = []
tic = time.time()

with torch.no_grad():
    for X_test, name in test_loader:
        X_test = X_test.to(device, non_blocking=True)
        logits = cassava_model(X_test)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)

        for img_name, label in zip(name, preds):
            rows.append((img_name, int(label)))

toc = time.time() - tic
print("Time for test inference is", toc, "seconds")

pred_df = pd.DataFrame(rows, columns=["image_id", "label"])

sample_sub_path = os.path.join(data_path, "sample_submission.csv")
if not os.path.exists(sample_sub_path):
    alt_path = "../input/sample_submission.csv"
    if os.path.exists(alt_path):
        sample_sub_path = alt_path

sample_sub = pd.read_csv(sample_sub_path)
sub = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if sub["label"].isna().any():
    fill_label = int(pred_df["label"].mode().iloc[0]) if len(pred_df) else 0
    sub["label"] = sub["label"].fillna(fill_label).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Saved submission.csv with shape:", sub.shape)

print("Files in current directory:", os.listdir("."))



## === cell 14
df_sub_test = pd.read_csv("submission.csv")
print(df_sub_test.shape)
print(df_sub_test.columns.tolist())
print(df_sub_test.head())
assert df_sub_test.columns.tolist() == ["image_id", "label"]
assert df_sub_test["image_id"].nunique() == len(df_sub_test)
assert df_sub_test["label"].between(0, 4).all()
print("Submission format looks valid.")
