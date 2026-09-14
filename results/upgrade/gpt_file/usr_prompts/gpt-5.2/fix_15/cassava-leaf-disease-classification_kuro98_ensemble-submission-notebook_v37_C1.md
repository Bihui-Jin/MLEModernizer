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

0.8974010275007556

# 6. Current score

0.42152

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.38266) has done: 'I fix the dataset root detection so it doesn’t incorrectly append `test_images` twice when you already pass the `.../test_images/` directory, which currently causes the “No image files found” crash and prevents `test_loader` from being created. I keep your inference/ensemble logic unchanged and only adjust the path resolution to robustly handle both “competition root” and “direct images folder” inputs. I also add a small safety check to print the resolved image root and ensure the submission merges correctly against `sample_submission.csv`. This should run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.38266) has done: 'Your current low score is consistent with your models not actually using the competition’s fine-tuned weights: the code tries to load from `/kaggle/input/vit-v1/...` etc., but those datasets typically don’t exist, so it silently falls back to ImageNet weights and a random new 5-class head, producing near-random predictions. I keep your exact ensemble/inference logic, but add a robust search that automatically finds `.pt/.pth` model files inside `/kaggle/input/` and loads them (handling common checkpoint formats) before falling back to torchvision defaults. I also ensure the classifier is replaced after loading (and reinitialized if shapes mismatch) so the forward pass stays valid while actually leveraging your provided weights when present. This should move accuracy substantially upward toward your target without changing architecture or inference semantics.'
- What this solution (achieved 0.38266) has done: 'Your current score suggests the loaded checkpoints (if any) aren’t being applied correctly to the exact model parameter names/shapes, so inference effectively uses mostly random/new heads. I keep your ensemble and inference exactly the same, but make checkpoint loading more robust by (1) filtering out classifier/head weights when their shapes don’t match your 5-class head, and (2) automatically remapping common `timm`-style keys (e.g., `head.weight`/`head.bias`) onto torchvision ViT (`heads.head.*`) so fine-tuned weights actually load. This should substantially increase accuracy toward your target without changing architecture, transforms, averaging, or prediction logic. I also ensure we print how many keys were loaded so you can confirm weights are truly applied.'
- What this solution (achieved 0.38266) has done: 'Your score is far below the target, and the most likely cause (given your current logic) is that at least one ensemble member is effectively untrained on Cassava due to a classifier/head mismatch: `_replace_classifier` doesn’t correctly replace the torchvision ViT head, so your ViT may still output 1000 ImageNet logits, making the ensemble nearly random. I make a minimal, score-relevant fix to `_replace_classifier` to correctly handle torchvision ViT (`model.heads.head`) and also make `_filter_mismatched_shapes` stricter so it drops keys that don’t exist in the current model (avoiding “unexpected” keys from interfering with loading). These changes keep your architecture/ensemble/inference semantics the same, but ensure checkpoints (or at least correct 5-class heads) are actually applied so accuracy moves toward the target. The rest of your pipeline (paths, transforms, averaging, submission merge) remains unchanged.'
- What this solution (achieved 0.33707) has done: 'Your current score is far below the target, so we should improve accuracy while keeping your ensemble/inference logic intact. The biggest score-relevant issue is that you apply `Softmax` after averaging raw logits from different models; this is poorly calibrated because each model’s logits have different scales, which can collapse the ensemble toward near-random choices. I make the minimal change to ensemble probabilities instead: apply `Softmax` to each model output first, then average probabilities (and similarly for TTA), preserving your three-model averaging logic but fixing the aggregation semantics. I also ensure the dataset crop size doesn’t exceed image bounds (to avoid unintended behavior on smaller images) without changing the overall preprocessing intent.'
- What this solution (achieved 0.34043) has done: 'Your current score is far below the target, so the smallest likely high-impact fix is to ensure inference uses the same input normalization and resize/crop policy the fine-tuned checkpoints expect. Right now you use ImageNet mean/std for all models, which can severely degrade accuracy for Cassava-specific fine-tuned weights; we switch to the Cassava competition’s commonly used dataset normalization (kept as a simple constant swap) while leaving your ensemble, model loading, and inference loop unchanged. I also make the center-crop size follow the *model’s* expected image size (instead of a fixed 600) to avoid unintended scale/cropping mismatch, again without changing the overall preprocessing intent (still center-crop then resize). Submission writing/merging stays identical.'
- What this solution (achieved 0.42152) has done: 'Your current accuracy is far below target, so we should increase performance with the smallest safe changes that don’t alter your ensemble/model logic. The most likely score-killer here is a preprocessing mismatch: you normalize with ImageNet stats but your checkpoints (if they exist) were likely trained with Cassava-specific mean/std; using the wrong normalization can collapse accuracy toward random. I switch `CASSAVA_MEAN/STD` to the commonly used Cassava competition normalization constants (minimal, localized change), while leaving your model loading, resizing/cropping, softmax-per-model averaging, and submission merge untouched. I also add a one-line print of the chosen normalization so you can verify the change is actually applied.'
- What this solution (achieved 0.42152) has done: 'Your current gap to the target is large (0.42152 → 0.8974), so the most likely score-killer is that at least one ensemble member is still effectively using an untrained/random 5-class head because the checkpoint head keys are not being remapped for non-ViT models (e.g., EfficientNet uses `classifier.1.*`, ResNet uses `fc.*`). I keep your ensemble, softmax-per-model averaging, and inference loop identical, but make checkpoint loading more robust by remapping common head keys into the *actual* classifier module that your `_replace_classifier` creates, then filtering mismatched shapes as you already do. This should materially increase accuracy toward the target when fine-tuned checkpoints exist, while remaining a minimal, localized change. I also print which head mapping was applied so you can confirm weights are truly being used.'
- What this solution (achieved 0.42152) has done: 'Your current score is far below the target, so we should increase accuracy with the smallest changes that keep your ensemble/inference logic intact. The biggest likely issue now is still “bad/absent fine-tuned weights”: your `_discover_checkpoint` only considers files if *any* keyword matches, so if the actual checkpoint filenames don’t contain your chosen keywords you silently fall back to ImageNet with a random 5-class head. I make checkpoint discovery minimally more robust by (1) preferring keyword matches but (2) falling back to the “best available” `.pt/.pth/.bin` in `/kaggle/input` (excluding obvious non-model folders) when no keyword hit is found, and (3) improving head-key remapping to cover common torchvision patterns (e.g., EfficientNet `classifier.1.*` and also `classifier.0.*`, plus ViT `heads.head.*`). This preserves your architecture, transforms, and softmax-per-model averaging, but greatly increases the chance your intended fine-tuned checkpoints are actually loaded, moving the score toward your target.'
- What this solution (achieved 0.42152) has done: 'Your score is far below the target, so we should improve accuracy while keeping your ensemble and inference loop unchanged. The most likely remaining score-killer is that even when a checkpoint is found, the final classifier/head weights often don’t get loaded because the key names don’t match your torchvision model’s actual head structure (especially for EfficientNet where the head is typically `classifier.1.*`). I make a minimal, localized change to remap additional common head key patterns (including mapping between `classifier.1.*` and `classifier.0.*`, and handling `heads.*`/`heads.head.*`) so the 5-class fine-tuned head is much more likely to load correctly. I also ensure the head replacement for EfficientNet always targets the last Linear layer in the classifier sequential (as you intended), without changing architecture or averaging semantics, and keep submission writing identical.'

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

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

model_a_img_size = 384
model_b_img_size = 528
model_c_img_size = 384

batch_size = 16
num_workers = 4
num_classes = 5
tta = False


def _replace_classifier(model: torch.nn.Module, num_classes: int) -> torch.nn.Module:
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Module):
        in_f = model.fc.in_features
        model.fc = torch.nn.Linear(in_f, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Module):
        if isinstance(model.classifier, torch.nn.Sequential):
            seq = list(model.classifier)
            last_linear_idx = None
            for i in range(len(seq) - 1, -1, -1):
                if isinstance(seq[i], torch.nn.Linear):
                    last_linear_idx = i
                    break
            if last_linear_idx is not None:
                in_f = seq[last_linear_idx].in_features
                seq[last_linear_idx] = torch.nn.Linear(in_f, num_classes)
                model.classifier = torch.nn.Sequential(*seq)
            else:
                model.classifier = torch.nn.Sequential(*seq)
        elif isinstance(model.classifier, torch.nn.Linear):
            in_f = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_f, num_classes)
    elif hasattr(model, "heads"):
        heads = getattr(model, "heads")
        if hasattr(heads, "head") and isinstance(
            getattr(heads, "head"), torch.nn.Linear
        ):
            in_f = heads.head.in_features
            heads.head = torch.nn.Linear(in_f, num_classes)
            model.heads = heads
        elif isinstance(heads, torch.nn.Sequential):
            seq = list(heads)
            for i in range(len(seq) - 1, -1, -1):
                if isinstance(seq[i], torch.nn.Linear):
                    in_f = seq[i].in_features
                    seq[i] = torch.nn.Linear(in_f, num_classes)
                    break
            model.heads = torch.nn.Sequential(*seq)
        elif isinstance(heads, torch.nn.Linear):
            in_f = heads.in_features
            model.heads = torch.nn.Linear(in_f, num_classes)
    elif hasattr(model, "head") and isinstance(model.head, torch.nn.Linear):
        in_f = model.head.in_features
        model.head = torch.nn.Linear(in_f, num_classes)
    return model


def _safe_construct(fallback_ctor, num_classes: int):
    """
    Keep existing behavior: try DEFAULT weights (if available offline), else weights=None.
    """
    try:
        model = fallback_ctor(weights="DEFAULT")
    except Exception as e:
        print(
            f"[WARN] Could not load DEFAULT weights ({type(e).__name__}: {e}). Using weights=None."
        )
        model = fallback_ctor(weights=None)
    model = _replace_classifier(model, num_classes)
    return model


def _infer_required_image_size(model: torch.nn.Module, default: int) -> int:
    """
    Keep existing behavior: ViT models may require fixed image_size.
    """
    sz = getattr(model, "image_size", None)
    if isinstance(sz, int) and sz > 0:
        return sz
    return default


def _discover_checkpoint(preferred_path: str, keywords: list[str]) -> str | None:
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    search_roots = ["/kaggle/input"]
    exts = (".pt", ".pth", ".bin")
    keyword_hits = []
    all_ckpt_files = []

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            base = os.path.basename(dirpath)
            if base in {
                "train_images",
                "test_images",
                "train_tfrecords",
                "test_tfrecords",
            }:
                dirnames[:] = []
                continue

            for fn in filenames:
                lfn = fn.lower()
                if not lfn.endswith(exts):
                    continue
                fullp = os.path.join(dirpath, fn)
                all_ckpt_files.append(fullp)

                score = 0
                for kw in keywords:
                    if kw.lower() in lfn:
                        score += 1
                if score > 0:
                    keyword_hits.append((score, fullp))

    if keyword_hits:
        keyword_hits.sort(key=lambda x: (-x[0], x[1]))
        return keyword_hits[0][1]

    if not all_ckpt_files:
        return None

    def _bad_path(p: str) -> bool:
        lp = p.lower()
        bad_tokens = [
            "/train_images/",
            "/test_images/",
            "/train_tfrecords/",
            "/test_tfrecords/",
            "optimizer",
            "sched",
            "scheduler",
        ]
        return any(t in lp for t in bad_tokens)

    candidates = [p for p in all_ckpt_files if not _bad_path(p)]
    if not candidates:
        candidates = all_ckpt_files

    candidates.sort(key=lambda p: (len(p), p))
    chosen = candidates[0]
    print(
        f"[WARN] No keyword-matching checkpoint found for keywords={keywords}. "
        f"Falling back to candidate checkpoint: {chosen}"
    )
    return chosen


def _load_checkpoint_state(path: str):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    return obj  # could be a full nn.Module


def _strip_state_dict_prefixes(sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for pref in ("module.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        new_sd[nk] = v
    return new_sd


def _remap_common_head_keys_for_torchvision(model: torch.nn.Module, sd: dict) -> dict:
    """
    Score-relevant robustness: remap common fine-tuned head keys to the actual
    classifier module name used by this torchvision model, so we don't end up
    with a randomly-initialized 5-class head at inference.

    Minimal change: expand remaps for EfficientNet/ViT/ResNet head naming variants.
    """
    if not isinstance(sd, dict):
        return sd

    model_keys = set(model.state_dict().keys())
    out = dict(sd)
    mapped = []

    def _maybe_copy(src_k: str, dst_k: str):
        if src_k in out and dst_k in model_keys and dst_k not in out:
            out[dst_k] = out[src_k]
            mapped.append((src_k, dst_k))

    if "heads.head.weight" in model_keys:
        _maybe_copy("head.weight", "heads.head.weight")
        _maybe_copy("head.bias", "heads.head.bias")
        _maybe_copy("classifier.weight", "heads.head.weight")
        _maybe_copy("classifier.bias", "heads.head.bias")
        _maybe_copy("heads.weight", "heads.head.weight")
        _maybe_copy("heads.bias", "heads.head.bias")
        _maybe_copy("heads.head.weight", "heads.head.weight")
        _maybe_copy("heads.head.bias", "heads.head.bias")

    if "fc.weight" in model_keys:
        _maybe_copy("head.weight", "fc.weight")
        _maybe_copy("head.bias", "fc.bias")
        _maybe_copy("classifier.weight", "fc.weight")
        _maybe_copy("classifier.bias", "fc.bias")

    if "classifier.1.weight" in model_keys:
        _maybe_copy("head.weight", "classifier.1.weight")
        _maybe_copy("head.bias", "classifier.1.bias")
        _maybe_copy("classifier.weight", "classifier.1.weight")
        _maybe_copy("classifier.bias", "classifier.1.bias")
        _maybe_copy("classifier.0.weight", "classifier.1.weight")
        _maybe_copy("classifier.0.bias", "classifier.1.bias")

    if "classifier.0.weight" in model_keys:
        _maybe_copy("head.weight", "classifier.0.weight")
        _maybe_copy("head.bias", "classifier.0.bias")
        _maybe_copy("classifier.weight", "classifier.0.weight")
        _maybe_copy("classifier.bias", "classifier.0.bias")
        _maybe_copy("classifier.1.weight", "classifier.0.weight")
        _maybe_copy("classifier.1.bias", "classifier.0.bias")

    if mapped:
        print("[INFO] Remapped head keys (first few):", mapped[:10])
    return out


def _filter_mismatched_shapes(
    model: torch.nn.Module, sd: dict
) -> tuple[dict, int, int]:
    if not isinstance(sd, dict):
        return sd, 0, 0

    model_sd = model.state_dict()
    kept = {}
    dropped = 0
    for k, v in sd.items():
        if k not in model_sd:
            dropped += 1
            continue
        if hasattr(v, "shape") and hasattr(model_sd[k], "shape"):
            if tuple(v.shape) == tuple(model_sd[k].shape):
                kept[k] = v
            else:
                dropped += 1
        else:
            kept[k] = v
    return kept, len(kept), dropped


def _try_load_or_fallback(
    pt_path: str, fallback_ctor, num_classes: int, discover_keywords: list[str]
):
    chosen = _discover_checkpoint(pt_path, discover_keywords)

    if chosen is not None:
        print(f"[INFO] Using checkpoint: {chosen}")
        obj = _load_checkpoint_state(chosen)

        if isinstance(obj, torch.nn.Module):
            model = obj
            model = _replace_classifier(model, num_classes)
            return model

        if isinstance(obj, dict):
            model = _safe_construct(fallback_ctor, num_classes)

            sd = _strip_state_dict_prefixes(obj)
            sd = _remap_common_head_keys_for_torchvision(model, sd)
            sd, kept_n, dropped_n = _filter_mismatched_shapes(model, sd)

            try:
                missing, unexpected = model.load_state_dict(sd, strict=False)
                print(
                    f"[INFO] load_state_dict: kept={kept_n} dropped={dropped_n} missing={len(missing)} unexpected={len(unexpected)}"
                )
                if missing:
                    print(f"[WARN] Missing keys (truncated): {missing[:8]} ...")
                if unexpected:
                    print(f"[WARN] Unexpected keys (truncated): {unexpected[:8]} ...")
            except RuntimeError as e:
                print(
                    f"[WARN] load_state_dict RuntimeError: {e}. Falling back to torchvision model."
                )
                return _safe_construct(fallback_ctor, num_classes)

            return model

        print(
            "[WARN] Unrecognized checkpoint format; falling back to torchvision model."
        )
        return _safe_construct(fallback_ctor, num_classes)

    print(
        f"[WARN] Checkpoint not found for keywords={discover_keywords}; using torchvision fallback."
    )
    return _safe_construct(fallback_ctor, num_classes)


from torchvision import models as tvm

path_a = "/kaggle/input/vit-v1/vit_v1.pt"
path_b = "/kaggle/input/efficient-net/efficient_net.pt"
path_c = "/kaggle/input/vit-v6/vit_v6.pt"

model_a = _try_load_or_fallback(
    path_a,
    lambda weights="DEFAULT": tvm.vit_b_16(weights=weights),
    num_classes,
    discover_keywords=["vit", "b16", "vit_v1", "vit-v1"],
)
model_b = _try_load_or_fallback(
    path_b,
    lambda weights="DEFAULT": tvm.efficientnet_b0(weights=weights),
    num_classes,
    discover_keywords=["efficient", "efficientnet", "b0", "efficient_net"],
)
model_c = _try_load_or_fallback(
    path_c,
    lambda weights="DEFAULT": tvm.resnet18(weights=weights),
    num_classes,
    discover_keywords=["resnet18", "resnet_18", "resnet"],
)

model_a_img_size = _infer_required_image_size(model_a, model_a_img_size)
model_b_img_size = _infer_required_image_size(model_b, model_b_img_size)
model_c_img_size = _infer_required_image_size(model_c, model_c_img_size)
print(
    "Using image sizes:",
    {"a": model_a_img_size, "b": model_b_img_size, "c": model_c_img_size},
)

model_a.to(device)
model_b.to(device)
model_c.to(device)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data.

    Args:
        data_dir: base directory or direct directory to images.
        transforms: set of transforms to be used.
    """

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        model_c_size,
        transform=None,
        ttas=None,
        img_size=384,
    ):
        data_dir = os.path.abspath(data_dir)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        def _has_images(d):
            if not os.path.isdir(d):
                return False
            try:
                for n in os.listdir(d):
                    p = os.path.join(d, n)
                    if os.path.isfile(p) and os.path.splitext(n.lower())[1] in exts:
                        return True
            except Exception:
                return False
            return False

        if _has_images(data_dir):
            img_root = data_dir
        else:
            cand_test = os.path.join(data_dir, "test_images")
            cand_train = os.path.join(data_dir, "train_images")
            if _has_images(cand_test):
                img_root = cand_test
            elif _has_images(cand_train):
                img_root = cand_train
            else:
                img_root = data_dir  # will error with a clearer message below

        super().__init__(root=img_root)

        self.transform = transform
        self.ttas = ttas

        self.crop_a = int(model_a_size)
        self.crop_b = int(model_b_size)
        self.crop_c = int(model_c_size)

        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_c = v2.Resize(
            (model_c_size, model_c_size), interpolation=InterpolationMode.BICUBIC
        )

        files = []
        if not os.path.isdir(self.root):
            raise RuntimeError(f"Image root directory does not exist: {self.root}")

        for name in os.listdir(self.root):
            p = os.path.join(self.root, name)
            if os.path.isfile(p) and os.path.splitext(name.lower())[1] in exts:
                files.append(name)
        self.images = sorted(files)

        if len(self.images) == 0:
            raise RuntimeError(
                f"No image files found under: {self.root}. "
                f"Please verify test_dir/train_dir points to a folder containing images."
            )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        w, h = img.size

        ca = min(self.crop_a, w, h)
        cb = min(self.crop_b, w, h)
        cc = min(self.crop_c, w, h)

        model_a_img = v2.CenterCrop((ca, ca))(img)
        model_b_img = v2.CenterCrop((cb, cb))(img)
        model_c_img = v2.CenterCrop((cc, cc))(img)

        model_a_img = self.resize_model_a(model_a_img)
        model_b_img = self.resize_model_b(model_b_img)
        model_c_img = self.resize_model_c(model_c_img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
            model_c_img = [self.transform(t(model_c_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)
            model_c_img = self.transform(model_c_img)

        return model_a_img, model_b_img, model_c_img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
CASSAVA_MEAN = [0.4304, 0.4968, 0.3135]
CASSAVA_STD = [0.2358, 0.2457, 0.2066]
print("Using normalization:", {"mean": CASSAVA_MEAN, "std": CASSAVA_STD})

test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=CASSAVA_MEAN, std=CASSAVA_STD),
    ]
)

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomAffine(180),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    model_c_img_size,
    transform=test_transforms,
    ttas=ttas,
)
print("Resolved test image root:", test_dataset.root)
print("Num test images:", len(test_dataset))

use_workers = num_workers if os.name != "nt" else 0
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=use_workers,
    pin_memory=torch.cuda.is_available(),
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
all_names = []
all_preds = []

model_a.eval()
model_b.eval()
model_c.eval()

with torch.no_grad():
    for batch_idx, (
        model_a_inputs,
        model_b_inputs,
        model_c_inputs,
        filenames,
    ) in enumerate(test_loader):
        bsz = len(filenames)

        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(device)
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(device)
            model_c_inputs = torch.cat(model_c_inputs, dim=0).to(device)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)
            model_c_outputs = model_c(model_c_inputs)

            model_a_batch_probs = torch.stack(
                torch.split(normalizer(model_a_outputs), bsz), dim=0
            )
            model_a_mean_probs = torch.mean(model_a_batch_probs, dim=0)

            model_b_batch_probs = torch.stack(
                torch.split(normalizer(model_b_outputs), bsz), dim=0
            )
            model_b_mean_probs = torch.mean(model_b_batch_probs, dim=0)

            model_c_batch_probs = torch.stack(
                torch.split(normalizer(model_c_outputs), bsz), dim=0
            )
            model_c_mean_probs = torch.mean(model_c_batch_probs, dim=0)

            mean_probs = (
                model_a_mean_probs + model_b_mean_probs + model_c_mean_probs
            ) / 3.0
            pred_labels = torch.argmax(mean_probs, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device)
            model_b_inputs = model_b_inputs.to(device)
            model_c_inputs = model_c_inputs.to(device)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)
            model_c_outputs = model_c(model_c_inputs)

            probs = (
                normalizer(model_a_outputs)
                + normalizer(model_b_outputs)
                + normalizer(model_c_outputs)
            ) / 3.0
            pred_labels = torch.argmax(probs, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

sample_sub = pd.read_csv(sample_sub_path)
pred_map = pd.DataFrame({"image_id": all_names, "label": all_preds}).drop_duplicates(
    "image_id"
)

my_submission = sample_sub[["image_id"]].merge(pred_map, on="image_id", how="left")
my_submission["label"] = my_submission["label"].fillna(0).astype(int)

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())
print("Missing predictions filled with 0:", int(my_submission["label"].isna().sum()))
print("Unique predicted labels:", sorted(my_submission["label"].unique().tolist()))
