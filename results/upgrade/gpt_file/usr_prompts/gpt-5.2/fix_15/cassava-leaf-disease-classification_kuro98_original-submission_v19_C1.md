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

3.12

# 3. Installed packages

geopandas==0.14.4
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

0.8845572680568148

# 6. Current score

0.4787

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.55456) has done: 'I fix the immediate runtime blocker by loading a model checkpoint from an available local path (and using a safe fallback if the external `/kaggle/input/efficient-net/...` file doesn’t exist), so `model` is always defined. I also fix submission correctness by iterating over the test set in a deterministic order (no shuffle), and by aligning predictions to `sample_submission.csv` so the output length and ordering exactly match Kaggle’s expected test set. Finally, I make the dataset robust to image mode issues (e.g., convert to RGB) and ensure `torch.load` uses `map_location` for CPU/GPU portability; these are score-neutral stability fixes.'
- What this solution (achieved 0.4787) has done: 'Your current score suggests you’re mostly using an ImageNet-pretrained EfficientNet/ViT head without a reliably loaded cassava-finetuned checkpoint, which severely cap accuracy. To move the score toward the 0.8846 target with minimal core-logic change, I (1) make checkpoint loading robust for both full-model and state_dict checkpoints for EfficientNet-B0 as well (not just ViT), and (2) ensure we use the correct evaluation preprocessing by switching from a generic ImageNet normalize/resize to the model’s official torchvision weights transforms when available (still the same “resize->tensor->normalize” semantics, just correct parameters). These two small changes should materially improve accuracy if the provided `.pt` checkpoint is an EfficientNet fine-tune or expects specific preprocessing, without altering your inference loop or submission alignment logic. The script still writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.4787) has done: 'Your score gap is large (0.4787 → target 0.8846), and the most likely cause is that you are not actually loading a fine-tuned cassava checkpoint (or you’re loading it but into the wrong architecture), so you’re effectively submitting near-ImageNet predictions. With minimal core-logic change, I (1) broaden checkpoint discovery to also look in the provided cassava dataset directory and accept common `.pth/.ckpt` names, (2) add a safe architecture “auto-probe” that tries EfficientNet-B0, EfficientNet-B3, and ViT-B/16 using the same strict=False load you already use and picks the one with the best key-match (so we stop mis-loading weights), and (3) ensure transforms match the selected torchvision weights (and force deterministic center-crop-like resizing) so preprocessing aligns with the loaded backbone. This keeps your inference loop and submission alignment logic the same, but substantially increases the chance you’re using the intended fine-tuned weights, which should move accuracy toward the target.'
- What this solution (achieved 0.4787) has done: 'Your score is far below the target, so the most likely issue is that you’re still effectively running an ImageNet model (or mis-loading a fine-tuned checkpoint) rather than a cassava-finetuned classifier. To move accuracy upward with minimal disruption, I (1) expand checkpoint discovery to also recursively scan the cassava dataset folder for any `.pt/.pth/.ckpt` (the current scan only looks one directory level deep), and (2) make state_dict loading robust to common prefix patterns (e.g., `backbone.`, `encoder.`, `model.module.`) and to Lightning-style keys, so the right weights actually land in the model. These changes keep your inference loop and model architectures the same, but substantially increase the chance the intended fine-tuned weights are used, which should move score toward your 0.8846 target. Submission alignment and CSV writing remain unchanged.'
- What this solution (achieved 0.4787) has done: 'Your score gap to the target is large, and the most likely reason is that you’re still running an ImageNet-pretrained classifier head (randomly initialized for 5 classes) because no true cassava-finetuned checkpoint is being found/loaded. With minimal changes and no change to the inference/training paradigm (still pure inference, same architectures tried), I (1) expand checkpoint discovery to include common Kaggle “input dataset” patterns (single-file datasets, nested `*/input/*`), and (2) make state_dict loading robust to `DataParallel`/Lightning key prefixes while also fixing a subtle bug where stripping `"classifier."` can break correct head loading. Finally, I ensure the chosen model’s preprocessing is correct by using the official torchvision weights transforms for the selected backbone (as you already intended), and keep deterministic ordering aligned to `sample_submission.csv` to avoid accidental score loss.'
- What this solution (achieved 0.4787) has done: 'Your current score is far below target, so the smallest meaningful improvement is to ensure inference uses a genuinely cassava-finetuned checkpoint if one exists, and that it loads correctly into the chosen backbone (right now you often end up with an ImageNet model with a mostly-random 5-class head). I keep your overall approach (single-model inference over test_images, same architectures tried, same softmax/argmax, same submission alignment), but (1) make checkpoint selection prefer “best/finetune/fold” files and avoid obviously irrelevant optimizer-only checkpoints, (2) strengthen state_dict key cleaning to handle Lightning (`state_dict` with `model.`), `fc.`→`classifier.1.` mappings, and `heads.head.` vs `classifier` variants so the classifier head weights actually load, and (3) set `num_workers=0` for Kaggle stability and add a safe `pin_memory` toggle based on CUDA (score-neutral, prevents occasional loader issues). These are minimal, execution-safe changes that should materially increase accuracy when a real fine-tuned checkpoint is present, moving the score toward your 0.8846 target while preserving core logic and output format.'
- What this solution (achieved 0.4787) has done: 'Your score is far below the target, so we should focus on the most likely small-but-high-impact issue: the model’s preprocessing does not match what the backbone expects at inference. I keep your exact inference loop and model choices, but I (1) fix backbone detection so EfficientNet-B3 is correctly recognized (your current `type(model).__name__` check never matches), and (2) make the fallback transforms match standard EfficientNet/ViT eval semantics by using `Resize(shorter)->CenterCrop` instead of a direct square resize, which can noticeably hurt accuracy. These are minimal changes that preserve evaluation semantics (deterministic eval preprocessing, same argmax over softmax outputs) while making predictions much more consistent with pretrained/fine-tuned checkpoints.'
- What this solution (achieved 0.4787) has done: 'Your gap to the target is large (0.4787 → 0.8846), so the smallest meaningful fix is to stop silently using partially-mismatched weights (which effectively behaves like an ImageNet model with a random 5-class head). I (1) fix a bug in the ViT weights transform selector (`IMAGEN1K` typo) so preprocessing matches the backbone when ViT is chosen, and (2) make checkpoint loading prefer candidates where the 5-class classifier/head weights actually load (instead of just “few missing keys” overall), which is a minimal change that directly improves inference correctness without changing the model families or inference loop. I also keep the submission alignment logic intact and still write `submission.csv` in the exact expected format. These changes are designed to increase accuracy toward your target without altering the core approach.'
- What this solution (achieved 0.4787) has done: 'Your score is far below the target, so the smallest meaningful improvement is to make sure inference is actually using a valid cassava-finetuned checkpoint when one exists, rather than silently falling back to an ImageNet-only model. I keep your exact inference approach, but (1) broaden checkpoint discovery to include more likely locations/names (including nested `/kaggle/input/*` datasets) and (2) make state_dict cleaning/loading more robust to common Lightning and DataParallel key patterns so the classifier head weights load correctly. I also fix the current TTA implementation so it uses deterministic test-time augmentations (instead of random transforms at inference) without changing your “average predictions then argmax” semantics. These changes are directly aimed at improving accuracy toward your 0.8846 target while preserving your core logic and producing the same valid `submission.csv` format.'
- What this solution (achieved 0.4787) has done: 'Your current accuracy is far below the target, so the smallest meaningful improvement is to ensure the loaded checkpoint (if any) actually lands on the correct architecture and that the inference preprocessing matches what the checkpoint/backbone expects. I keep your single-model inference pipeline intact, but (1) make checkpoint selection prefer those whose 5-class head weights match (so we stop picking “best” files that don’t actually fit), and (2) make state_dict key cleaning handle a couple of common head/key patterns seen in EfficientNet/ViT fine-tunes so the classifier weights load instead of being left random. These changes are narrowly targeted at fixing mis-loaded weights/preprocessing (the main cause of ~0.48 accuracy) while keeping your loops, argmax logic, and submission alignment unchanged. The script still writes a valid `submission.csv`.'
- What this solution (achieved 0.4787) has done: 'Your score gap to the target is large, and the most likely remaining blocker is still “not actually using a cassava-finetuned head”: the current `_remap_head_keys()` ends by rewriting any `classifier.*` keys into `heads.head.*`, which can silently destroy correct EfficientNet head weights and lead to a mostly-random 5-class classifier (≈0.48 accuracy). I make the head key remapping architecture-aware (only apply EfficientNet-style or ViT-style mappings depending on which candidate model we’re testing) and remove the unconditional `classifier -> heads.head` rewrite. I also make the EfficientNet/VIT candidate selection prefer checkpoints where the 5-class head tensors match in both key coverage and shape (a minimal change that directly increases the chance the right weights load). Everything else (model families tried, inference loop, softmax/argmax, submission alignment and CSV writing) stays the same.'
- What this solution (achieved 0.4787) has done: 'Your current gap to the target is large, and the most likely remaining cause is still “weights don’t actually load into the right places”, especially for EfficientNet checkpoints saved with common alternative head key names (e.g., `classifier.weight`, `classifier.bias`, or `fc.*`) that your current remapping doesn’t fully cover. I make a minimal, architecture-aware improvement to the state_dict key cleaning so EfficientNet head weights are mapped correctly into `classifier.1.*` (and avoid breaking ViT keys), while keeping the same candidate-model probing/inference loop and submission alignment. I also make the candidate selection slightly more head-focused (still the same logic, just a safer tie-break) so we pick the checkpoint/architecture where the 5-class head actually matches in shape, which should move accuracy upward toward your 0.8846 target if a fine-tuned checkpoint exists in the scanned folders. The script still run end-to-end and write a valid `submission.csv` with the required columns/order.'
- What this solution (achieved 0.4787) has done: 'Your score is far below the target, so the smallest high-impact changes should focus on getting correct cassava-finetuned weights to actually load and using the exact eval preprocessing the chosen backbone expects. I keep your single-model inference pipeline and candidate-architecture probing, but improve the checkpoint matching signal by (1) filtering out non-model checkpoints more strongly and (2) selecting the candidate based primarily on “5-class head keys with correct shapes” (the main difference between a real fine-tuned model and an ImageNet fallback). I also make the torchvision “weights transforms” usage deterministic w.r.t. `img_size` (override resize/crop size to your configured `img_size`) so you don’t accidentally evaluate at a mismatched resolution that can hurt accuracy. The submission alignment logic and CSV output stay unchanged.'
- What this solution (achieved 0.4787) has done: 'Your score gap to the target is large, so the most likely minimal, high-impact fix is that inference preprocessing still doesn’t match what most cassava EfficientNet/ViT fine-tuned checkpoints expect (224/300/384 + center-crop), and forcing `img_size=528` can badly hurt accuracy even when weights load correctly. I keep your model families, checkpoint probing logic, and inference loop identical, but change only the transform sizing behavior: use the official torchvision weights’ default eval transforms (including their native resize/crop sizes) and only override the final crop size when it is smaller than your configured `img_size` (so we don’t upsample beyond what weights were trained on). I also make TTA deterministic (using v2 functional flips instead of Random* modules) without changing your “average predictions then argmax” semantics, so turning `tta=True` won’t introduce randomness. These changes are directly score-relevant (better-calibrated preprocessing + deterministic TTA) and should move accuracy upward toward your target without altering core logic.'

# 9. Code solution

## === cell 0
import os

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"

img_size = 528
batch_size = 16

num_workers = 0

num_classes = 5
tta = False

ckpt_candidates = [
    "/kaggle/input/efficient-net/vit_cont_3.pt",
    "/kaggle/input/vit_cont_3.pt",
    "/kaggle/working/vit_cont_3.pt",
    "/kaggle/input/cassava-leaf-disease-classification/vit_cont_3.pt",
    "/kaggle/input/cassava-leaf-disease-classification/model.pt",
    "/kaggle/input/cassava-leaf-disease-classification/model.pth",
    "/kaggle/input/cassava-leaf-disease-classification/checkpoint.pth",
    "/kaggle/input/cassava-leaf-disease-classification/checkpoint.pt",
]

scan_dirs = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input",
]


def _iter_ckpt_files(root_dir):
    try:
        for r, _, files in os.walk(root_dir):
            for fn in files:
                lfn = fn.lower()
                if lfn.endswith((".pt", ".pth", ".ckpt")):
                    yield os.path.join(r, fn)
    except Exception:
        return


priority_keywords = [
    "best",
    "finetune",
    "fine-tune",
    "finetuned",
    "fold",
    "cassava",
    "eff",
    "enet",
    "efficientnet",
    "vit",
    "model",
    "checkpoint",
]
avoid_keywords = [
    "optimizer",
    "optim",
    "sched",
    "scheduler",
    "scaler",
    "amp",
    "ema",
    "metrics",
    "history",
    "log",
    "events",
    "train_state",
    "trainer",
    "callback",
    "wandb",
    "tensorboard",
    "args",
    "config",
    "hparams",
]

for d in scan_dirs:
    if not os.path.isdir(d):
        continue
    for p in _iter_ckpt_files(d):
        base = os.path.basename(p).lower()
        if any(k in base for k in avoid_keywords):
            continue
        if any(k in base for k in priority_keywords):
            ckpt_candidates.append(p)

_seen = set()
ckpt_candidates = [p for p in ckpt_candidates if not (p in _seen or _seen.add(p))]


def _ckpt_priority(p: str) -> int:
    b = os.path.basename(p).lower()
    score = 0
    if "best" in b:
        score -= 50
    if "finetune" in b or "fine-tune" in b or "finetuned" in b:
        score -= 30
    if "fold" in b:
        score -= 10
    if "cassava" in b:
        score -= 10
    if "vit" in b:
        score -= 5
    if "efficientnet" in b or "enet" in b or "eff" in b:
        score -= 5
    if "checkpoint" in b:
        score -= 2
    return score


existing = [p for p in ckpt_candidates if os.path.exists(p)]
existing.sort(key=_ckpt_priority)
ckpt_path = existing[0] if existing else None


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in (
            "state_dict",
            "model",
            "net",
            "module",
            "model_state_dict",
            "weights",
        ):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if "checkpoint" in obj and isinstance(obj["checkpoint"], dict):
            inner = obj["checkpoint"]
            for k in ("state_dict", "model", "model_state_dict"):
                if k in inner and isinstance(inner[k], dict):
                    return inner[k]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    return None


def _strip_prefix(sd, prefix):
    if sd is None:
        return None
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in sd.items()}


def _try_common_prefix_strips(sd):
    prefixes = [
        "module.",
        "model.module.",
        "model.",
        "net.",
        "backbone.",
        "encoder.",
        "student.",
        "teacher.",
        "wrapped.",
        "ema.",
    ]
    out = sd
    for p in prefixes:
        out = _strip_prefix(out, p)
    return out


def _remap_head_keys(sd, arch_name: str):
    """
    Minimal, architecture-aware head remapping to improve likelihood that a true
    cassava-finetuned 5-class head loads (and we don't silently keep a random head).

    This is directly score-relevant (correct weights) while preserving the same
    model families and inference semantics.
    """
    if sd is None:
        return None
    out = dict(sd)

    def _rename_prefix(d, old_prefix, new_prefix):
        if not any(k.startswith(old_prefix) for k in d.keys()):
            return d
        nd = {}
        for k, v in d.items():
            if k.startswith(old_prefix):
                nd[new_prefix + k[len(old_prefix) :]] = v
            else:
                nd[k] = v
        return nd

    def _rename_exact(d, old_key, new_key):
        if old_key not in d:
            return d
        nd = dict(d)
        nd[new_key] = nd.pop(old_key)
        return nd

    if arch_name.startswith("efficientnet"):
        out = _rename_prefix(out, "fc.", "classifier.1.")
        out = _rename_prefix(out, "head.", "classifier.1.")
        out = _rename_exact(out, "classifier.weight", "classifier.1.weight")
        out = _rename_exact(out, "classifier.bias", "classifier.1.bias")
        out = _rename_exact(out, "classifier.0.weight", "classifier.1.weight")
        out = _rename_exact(out, "classifier.0.bias", "classifier.1.bias")
        return out

    if arch_name.startswith("vit"):
        out = _rename_prefix(out, "head.", "heads.head.")
        out = _rename_prefix(out, "classifier.", "heads.head.")
        return out

    return out


def _head_key_coverage(sd: dict, arch_name: str) -> int:
    if sd is None:
        return 0
    keys = sd.keys()
    if arch_name.startswith("efficientnet"):
        head_prefix = "classifier.1."
    elif arch_name.startswith("vit"):
        head_prefix = "heads.head."
    else:
        return 0
    return sum(1 for k in keys if k.startswith(head_prefix))


def _head_shape_match(sd: dict, arch_name: str) -> int:
    if sd is None:
        return 0
    try:
        if arch_name.startswith("efficientnet"):
            w = sd.get("classifier.1.weight", None)
            b = sd.get("classifier.1.bias", None)
            if (
                isinstance(w, torch.Tensor)
                and w.ndim == 2
                and w.shape[0] == num_classes
            ):
                return 1
            if (
                isinstance(b, torch.Tensor)
                and b.ndim == 1
                and b.shape[0] == num_classes
            ):
                return 1
        if arch_name.startswith("vit"):
            w = sd.get("heads.head.weight", None)
            b = sd.get("heads.head.bias", None)
            if (
                isinstance(w, torch.Tensor)
                and w.ndim == 2
                and w.shape[0] == num_classes
            ):
                return 1
            if (
                isinstance(b, torch.Tensor)
                and b.ndim == 1
                and b.shape[0] == num_classes
            ):
                return 1
    except Exception:
        return 0
    return 0


def _score_state_dict_match(
    missing_keys, unexpected_keys, head_cov: int, head_shape_ok: int
):
    return (
        (len(missing_keys) * 1.0)
        + (len(unexpected_keys) * 2.0)
        - (head_cov * 300.0)
        - (head_shape_ok * 500.0)
    )


model = None
loaded_weights_name = None  # used to pick correct transforms later

obj = None
if ckpt_path is not None:
    try:
        obj = torch.load(ckpt_path, map_location=device)
    except Exception:
        obj = None

if isinstance(obj, torch.nn.Module):
    model = obj
    loaded_weights_name = "custom_fullmodel"


def _build_candidates_from_state_dict(sd_raw):
    candidates = []

    try:
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        sd_vit = _remap_head_keys(sd_raw, "vit_b_16")
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        vit.heads.head = torch.nn.Linear(vit.heads.head.in_features, num_classes)
        missing, unexpected = vit.load_state_dict(sd_vit, strict=False)
        head_cov = _head_key_coverage(sd_vit, "vit_b_16")
        head_shape_ok = _head_shape_match(sd_vit, "vit_b_16")
        candidates.append(
            (
                _score_state_dict_match(missing, unexpected, head_cov, head_shape_ok),
                vit,
                "vit_b_16",
            )
        )
    except Exception:
        pass

    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        sd_eff0 = _remap_head_keys(sd_raw, "efficientnet_b0")
        eff0 = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        eff0.classifier[1] = torch.nn.Linear(
            eff0.classifier[1].in_features, num_classes
        )
        missing, unexpected = eff0.load_state_dict(sd_eff0, strict=False)
        head_cov = _head_key_coverage(sd_eff0, "efficientnet_b0")
        head_shape_ok = _head_shape_match(sd_eff0, "efficientnet_b0")
        candidates.append(
            (
                _score_state_dict_match(missing, unexpected, head_cov, head_shape_ok),
                eff0,
                "efficientnet_b0",
            )
        )
    except Exception:
        pass

    try:
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

        sd_eff3 = _remap_head_keys(sd_raw, "efficientnet_b3")
        eff3 = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        eff3.classifier[1] = torch.nn.Linear(
            eff3.classifier[1].in_features, num_classes
        )
        missing, unexpected = eff3.load_state_dict(sd_eff3, strict=False)
        head_cov = _head_key_coverage(sd_eff3, "efficientnet_b3")
        head_shape_ok = _head_shape_match(sd_eff3, "efficientnet_b3")
        candidates.append(
            (
                _score_state_dict_match(missing, unexpected, head_cov, head_shape_ok),
                eff3,
                "efficientnet_b3",
            )
        )
    except Exception:
        pass

    return candidates


best_global = None  # (score, model, name, ckpt_path_used)

if model is None:
    for p in existing[:25]:  # bounded for runtime stability
        try:
            obj_i = torch.load(p, map_location=device)
        except Exception:
            continue

        if isinstance(obj_i, dict) and (
            "optimizer_states" in obj_i or "lr_schedulers" in obj_i
        ):
            continue

        sd_i = _extract_state_dict(obj_i)
        sd_i = _try_common_prefix_strips(sd_i)
        if sd_i is None:
            continue

        cands = _build_candidates_from_state_dict(sd_i)
        if not cands:
            continue

        cands.sort(key=lambda x: x[0])
        score_i, model_i, name_i = cands[0]

        if best_global is None or score_i < best_global[0]:
            best_global = (score_i, model_i, name_i, p)

    if best_global is not None and best_global[0] < 2000:
        model = best_global[1]
        loaded_weights_name = best_global[2]
        ckpt_path = best_global[3]

if model is None:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    loaded_weights_name = "efficientnet_b0_imagenet"

model.to(device)

print("Checkpoint used:", ckpt_path)
print("Model type:", type(model).__name__, "loaded_weights_name:", loaded_weights_name)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data (test-only: returns (image_tensor, filename))."""

    def __init__(self, data_dir, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.images = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        self.ttas = ttas

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        if self.ttas is not None and self.transform is not None:
            img = [self.transform(t(img)) for t in self.ttas]
        elif self.transform:
            img = self.transform(img)

        return img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
def _weights_transform_for_model(model, img_size):
    """
    Score-relevant minimal change:
    - Use the official torchvision weights' eval transforms *as-is* (they encode the
      expected resize/crop sizes and normalization).
    - Only override crop/resize sizes when the official crop is larger than img_size
      (downscale) to avoid harmful upscaling (e.g., forcing 528) that can reduce accuracy.
    This keeps the same deterministic "resize/crop -> tensor -> normalize" semantics.
    """
    try:
        from torchvision.models import (
            EfficientNet_B0_Weights,
            EfficientNet_B3_Weights,
            ViT_B_16_Weights,
        )
        from torchvision.models.efficientnet import EfficientNet
        from torchvision.models.vision_transformer import VisionTransformer

        def _override_size_conservatively(tfm):
            if not hasattr(tfm, "transforms"):
                return tfm
            new_list = []
            for tr in tfm.transforms:
                if isinstance(tr, v2.Resize):
                    size = tr.size
                    try:
                        if isinstance(size, int):
                            new_size = min(size, img_size)
                        elif isinstance(size, (tuple, list)) and len(size) == 2:
                            new_size = (min(size[0], img_size), min(size[1], img_size))
                        else:
                            new_size = size
                    except Exception:
                        new_size = size
                    new_list.append(
                        v2.Resize(new_size, interpolation=InterpolationMode.BICUBIC)
                    )
                elif isinstance(tr, v2.CenterCrop):
                    size = tr.size
                    try:
                        if isinstance(size, int):
                            new_size = min(size, img_size)
                        elif isinstance(size, (tuple, list)) and len(size) == 2:
                            new_size = (min(size[0], img_size), min(size[1], img_size))
                        else:
                            new_size = size
                    except Exception:
                        new_size = size
                    new_list.append(v2.CenterCrop(new_size))
                else:
                    new_list.append(tr)
            return v2.Compose(new_list)

        if isinstance(model, EfficientNet):
            try:
                last_conv = None
                for m in reversed(list(model.features.modules())):
                    if isinstance(m, torch.nn.Conv2d):
                        last_conv = m
                        break
                out_ch = last_conv.out_channels if last_conv is not None else None
            except Exception:
                out_ch = None

            if out_ch is not None and out_ch >= 1536:
                return _override_size_conservatively(
                    EfficientNet_B3_Weights.IMAGENET1K_V1.transforms()
                )
            return _override_size_conservatively(
                EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
            )

        if isinstance(model, VisionTransformer):
            return _override_size_conservatively(
                ViT_B_16_Weights.IMAGENET1K_V1.transforms()
            )
    except Exception:
        pass

    return v2.Compose(
        [
            v2.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            v2.CenterCrop(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


test_transforms = _weights_transform_for_model(model, img_size)

if tta:

    def _hflip(img):  # PIL in, PIL out
        return v2.functional.horizontal_flip(img)

    def _vflip(img):
        return v2.functional.vertical_flip(img)

    ttas = [
        lambda x: x,
        _hflip,
        _vflip,
    ]
else:
    ttas = None

test_dataset = CassavaDataset(test_dir, transform=test_transforms, ttas=ttas)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
all_names = []
all_preds = []

model.eval()
with torch.no_grad():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        if tta:
            inputs, filenames = torch.cat(inputs, dim=0).to(device), list(filenames)
            preds = normalizer(model(inputs))
            batch_preds = torch.stack(torch.split(preds, len(filenames)), dim=0)
            mean_preds = torch.mean(batch_preds, dim=0)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            inputs, filenames = inputs.to(device), list(filenames)
            preds = normalizer(model(inputs))
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)



## === cell 4
sample = pd.read_csv(sample_path)
pred_map = dict(zip(all_names, all_preds))

missing = [img_id for img_id in sample["image_id"].tolist() if img_id not in pred_map]
if missing:
    for m in missing:
        pred_map[m] = 0

my_submission = sample.copy()
my_submission["label"] = my_submission["image_id"].map(pred_map).astype(int)

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)



## === cell 5
my_submission.head()
