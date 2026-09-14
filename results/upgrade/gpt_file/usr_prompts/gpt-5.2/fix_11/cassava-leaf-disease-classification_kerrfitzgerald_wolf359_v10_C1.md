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


def _strip_known_prefixes(s: str) -> str:
    prefixes = ("module.", "model.", "net.", "backbone.", "encoder.")
    prev = None
    cur = s
    while prev != cur:
        prev = cur
        for p in prefixes:
            if cur.startswith(p):
                cur = cur[len(p) :]
                break
    return cur


def _choose_best_global_mapping(raw_sd: dict, model_keys: set):
    """
    Score-relevant fix: infer the best *global* prefix mapping from checkpoint keys
    to the current model's keys (prevents the common 'model.model.*' mismatch).
    """
    if not isinstance(raw_sd, dict) or not raw_sd:
        return (lambda k: k, "identity")

    def t_identity(k):  # keep as-is
        return _rewrite_classifier_aliases(k)

    def t_strip(k):  # strip wrappers
        return _rewrite_classifier_aliases(_strip_known_prefixes(k))

    def t_add_model(k):  # ensure 'model.' prefix after stripping
        kk = _rewrite_classifier_aliases(_strip_known_prefixes(k))
        return kk if kk.startswith("model.") else ("model." + kk)

    def t_collapse_model_model(k):  # collapse model.model. -> model.
        kk = _rewrite_classifier_aliases(k)
        if kk.startswith("model.model."):
            kk = "model." + kk[len("model.model.") :]
        kk = _strip_known_prefixes(kk)  # then strip wrappers just in case
        return kk if kk.startswith("model.") else ("model." + kk)

    def t_head_to_model_head(k):  # head.* -> model.head.*
        kk = _rewrite_classifier_aliases(_strip_known_prefixes(k))
        if kk.startswith("head."):
            return "model." + kk
        return kk if kk.startswith("model.") else ("model." + kk)

    def t_model_head_to_head(k):  # model.head.* -> head.* (for models without wrapper)
        kk = _rewrite_classifier_aliases(_strip_known_prefixes(k))
        if kk.startswith("model.head."):
            return kk[len("model.") :]
        return kk

    candidates = [
        ("identity", t_identity),
        ("strip", t_strip),
        ("add_model", t_add_model),
        ("collapse_model_model", t_collapse_model_model),
        ("head_to_model_head", t_head_to_model_head),
        ("model_head_to_head", t_model_head_to_head),
    ]

    raw_keys = list(raw_sd.keys())
    best_name, best_t, best_hits = None, None, -1
    for name, tf in candidates:
        hits = 0
        for k in raw_keys[:: max(1, len(raw_keys) // 2000)]:
            if tf(k) in model_keys:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_name, best_t = name, tf

    return best_t, best_name


def _normalize_ckpt_keys_to_model(model: nn.Module, raw_sd: dict) -> dict:
    """
    Score-relevant fix: normalize keys using a best global mapping + a small set of
    fallback per-key alternatives to maximize correct weight loading (esp. head).
    """
    if not isinstance(raw_sd, dict) or not len(raw_sd):
        return raw_sd

    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())

    best_tf, best_name = _choose_best_global_mapping(raw_sd, model_keys)

    normalized = {}
    matched = 0

    for k, v in raw_sd.items():
        kk = best_tf(k)

        fallbacks = [kk]
        if kk.startswith("model.model."):
            fallbacks.append("model." + kk[len("model.model.") :])
        if kk.startswith("head."):
            fallbacks.append("model." + kk)
        if kk.startswith("model.head."):
            fallbacks.append(kk[len("model.") :])  # head.*
        if not kk.startswith("model."):
            fallbacks.append("model." + kk)

        placed = False
        for cand in fallbacks:
            if cand in model_keys:
                normalized[cand] = v
                matched += 1
                placed = True
                break
        if not placed:
            pass

    print(
        "Checkpoint key normalization:",
        f"strategy={best_name}",
        f"raw_keys={len(raw_sd)}",
        f"kept_keys={len(normalized)}",
        f"matched_keys={matched}",
        f"model_keys={len(model_keys)}",
    )
    return normalized


def _load_finetuned_into_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_sd = _extract_state_dict(ckpt)

    sd = _normalize_ckpt_keys_to_model(model, raw_sd)

    expected_hw_key = (
        "model.head.weight"
        if "model.head.weight" in model.state_dict()
        else "head.weight"
    )
    expected_hb_key = (
        "model.head.bias" if "model.head.bias" in model.state_dict() else "head.bias"
    )
    head_in_sd = (expected_hw_key in sd) and (expected_hb_key in sd)

    if head_in_sd:
        w_ok = tuple(sd[expected_hw_key].shape) == tuple(
            model.state_dict()[expected_hw_key].shape
        )
        b_ok = tuple(sd[expected_hb_key].shape) == tuple(
            model.state_dict()[expected_hb_key].shape
        )
        print(
            "Head present in remapped state_dict:", True, "| shape_ok:", (w_ok and b_ok)
        )
    else:
        print(
            "Head present in remapped state_dict:",
            False,
            "| This will hurt score if true.",
        )

    strict_used = False
    if len(sd) == len(model.state_dict()):
        model.load_state_dict(sd, strict=True)
        strict_used = True
        print("Load report | strict=True (all keys matched)")
    else:
        incompatible = model.load_state_dict(sd, strict=False)
        strict_used = False
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        print("Load report | strict=False")
        print(
            "Missing keys count:",
            len(missing),
            "| Unexpected keys count:",
            len(unexpected),
        )
        head_missing = [k for k in missing if "head" in k]
        if head_missing:
            print(
                "WARNING: head-related missing keys (will hurt score):",
                head_missing[:20],
            )

    with torch.no_grad():
        hw = model.model.head.weight.detach().float().cpu()
        hb = model.model.head.bias.detach().float().cpu()
        print("Post-load head weight abs-mean:", float(hw.abs().mean()))
        print("Post-load head bias abs-mean:", float(hb.abs().mean()))
    return strict_used


cassava_model = ViTBase16(n_classes=5, pretrained=True)

loaded_finetuned = False
if os.path.exists(Cassava_model):
    strict_used = _load_finetuned_into_model(cassava_model, Cassava_model)
    print(
        "Loaded fine-tuned checkpoint from:",
        Cassava_model,
        "| strict_used:",
        strict_used,
    )
    loaded_finetuned = True
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
