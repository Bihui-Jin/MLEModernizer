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

0.8689936536718041

# 6. Current score

0.08184

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11061) has done: 'I remove the failing `pip install` dependency and instead build the exact EfficientNet-B4 architecture using `torchvision` so the provided `.pth` can load without `efficientnet_pytorch`. I fix the Albumentations v2 API break by using the correct `size=(H,W)` signature and replacing removed transforms (`Cutout`) with their current equivalents so TTA runs. I also fix inference-time bugs that would silently hurt accuracy (wrong averaging divisor, missing `torch.no_grad()`, and missing softmax/logit handling), while keeping the same core approach: EfficientNet-B4 + 5x TTA over the sample submission list and argmax for labels. Finally, the script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.11024) has done: 'I fix the immediate runtime failure by making the checkpoint path robust: the current `model_path` points to a dataset that isn’t present, so I search common Kaggle input locations for `model_50.pth` (and fail fast with a clear message if it truly doesn’t exist). I also make the model weight loading tolerant to minor key mismatches (`strict=False`) while keeping the same EfficientNet-B4 architecture, which prevents silent “all-random-weights” behavior and should raise accuracy substantially toward your target. Finally, I fix a subtle inference bug: Albumentations `Normalize` already outputs float images, so applying `torchvision.transforms.ToTensor()` rescales by 1/255 again; I replace it with a correct HWC->CHW tensor conversion to avoid destroying the input scale and hurting score. The rest of the core logic (EfficientNet-B4 + 5x TTA over `sample_submission.csv` and argmax) stays the same.'
- What this solution (achieved 0.11024) has done: 'Your run is failing because the referenced pretrained checkpoint (`model_50.pth`) is not present anywhere in this Kaggle environment, so the code never gets past checkpoint resolution/loading. Since we must keep the same core logic (EfficientNet-B4 + TTA + argmax) but still produce a valid submission, I (1) make the checkpoint resolution robust to this environment by allowing a clean fallback to an untrained model (so the notebook runs end-to-end), and (2) add a clear warning so it’s obvious why the score be low without weights. I also make the file paths robust to both `../input/...` and `/kaggle/data/input/...` layouts so inference can always find `sample_submission.csv` and `test_images`. This fix the runtime errors and always write a valid `submission.csv`; meaningful score improvement require providing the missing checkpoint file in `/kaggle/input`.'
- What this solution (achieved 0.11024) has done: 'Your current score is extremely low because the model is effectively untrained in this environment (the checkpoint is missing), so the smallest change that can legitimately move accuracy toward your target is to load real pretrained weights without changing the architecture or inference semantics. I keep the same EfficientNet-B4 + 5x TTA + argmax core logic, but (1) point `model_path` to a checkpoint that actually exists in this dataset (`final_50.pth` is present in `/kaggle/data/...`), and (2) make weight loading stricter by auto-matching common key prefixes and requiring a high key-match ratio so we don’t silently run random weights. I also switch the TTA crop from `RandomResizedCrop` to deterministic `Resize+CenterCrop` (still 5 passes with the same flips/affine jitter) to reduce harmful randomness at inference; this typically improves accuracy materially when weights are correct and should move you much closer to the target. The script still runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 0.11024) has done: 'Your score is far below the target because the inference-time augmentation pipeline is extremely destructive for test-time prediction: it applies heavy geometric/color jitter plus CoarseDropout during TTA, which typically collapses accuracy even with good weights. To move accuracy upward toward the target while preserving the same core logic (EfficientNet-B4, 5x TTA loop, argmax over averaged logits), I keep TTA=5 but make the TTA transforms inference-safe by removing CoarseDropout and replacing random heavy transforms with mild, realistic TTA (small rotate/scale + flips only) while keeping the same normalization. I also switch to a 380px resize (EfficientNet-B4’s native resolution) with center crop to better match the model’s expected input distribution, without changing the model or loop structure. These are minimal, directly score-relevant changes that should materially increase accuracy toward your target.'
- What this solution (achieved 0.11024) has done: 'Your score is far below the target because the checkpoint you load is almost certainly not aligned with `torchvision.models.efficientnet_b4`’s internal layer naming/structure, so even with a high “key overlap” you can still end up with a badly-initialized network (effectively near-random). The smallest change that preserves your core logic (EfficientNet-B4 + 5x TTA + argmax) but should materially increase accuracy is to switch to the exact same EfficientNet-B4 implementation used by the checkpoint via `efficientnet_pytorch`, and then load the checkpoint strictly so we don’t silently run mismatched weights. I also keep TTA=5 but make it inference-safe/deterministic (no random rotate/scale) so predictions don’t degrade due to unnecessary randomness at test-time. Paths and the submission writing remain identical, and the script still runs end-to-end and always produces `submission.csv`.'
- What this solution (achieved 0.09753) has done: 'Your score is far below the target because the current inference pipeline is effectively producing near-random predictions: (1) it depends on `efficientnet_pytorch`, which is not installed here, so you’re forced onto a different EfficientNet-B4 implementation; and (2) `strict=True` often fail (or if you relax it, it can silently load mismatched weights), leaving the network poorly initialized. To move accuracy up toward your target while preserving the same core logic (EfficientNet-B4 + 5x TTA + argmax), I switch to `torchvision`’s EfficientNet-B4 with ImageNet pretrained weights as a legitimate, available fallback when the custom checkpoint can’t be loaded correctly. I also make checkpoint loading robust by auto-adapting common classifier-head key names between implementations and only using the checkpoint if the load is clean; otherwise we fall back to ImageNet weights (much better than random). The TTA loop count, inference semantics, and submission format remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.0994) has done: 'Your score is far below the target, so we should make a small, legitimate change that increases accuracy without changing the core approach (EfficientNet-B4 + 5x TTA + argmax). The biggest likely issue is that the model is often running with ImageNet weights (or partially-loaded weights) because `strict=True` checkpoint loading fails, which yields near-random cassava predictions. I keep the same architecture and inference loop, but make checkpoint loading robust by (1) verifying key overlap, (2) resizing only the classifier weights if needed, and (3) loading with `strict=False` only when we can confirm a high match ratio—otherwise keep the ImageNet fallback. This preserves evaluation semantics while making it much more likely you actually use the provided cassava-trained checkpoint, which should move the score strongly toward your target.'
- What this solution (achieved 0.1009) has done: 'Your current score (0.0994) is far below the target (0.86899), so the most likely issue is that you’re still not actually loading the cassava-trained checkpoint correctly and are effectively predicting with ImageNet/random-ish weights. I keep the same core logic (EfficientNet-B4 + 5x TTA + argmax) but make checkpoint loading stricter and more compatible by (1) detecting whether the checkpoint is from `efficientnet_pytorch` (common for B4 cassava checkpoints), and (2) instantiating the matching model implementation so weights load cleanly. If `efficientnet_pytorch` isn’t available, the script fall back to torchvision ImageNet weights exactly as now (so it still runs end-to-end), but when the package is available this should move accuracy sharply upward toward your target. I also make the TTA deterministic across runs by reseeding per image/tta so random flips don’t add uncontrolled variance, without changing the TTA count or semantics.'
- What this solution (achieved 0.09567) has done: 'Your current score is far below the target, which strongly suggests the cassava-specific checkpoint still isn’t being used correctly and you’re effectively submitting near-ImageNet/random predictions. I make the smallest change that increases the probability of actually loading the provided `final_50.pth` by (1) prioritizing that exact file path (and its dataset folder) during checkpoint resolution, and (2) adding a robust “classifier head remap” for the common `efficientnet_pytorch -> torchvision` naming mismatch (especially `._fc` vs `.classifier.1`) while still keeping the exact EfficientNet-B4 + 5x TTA + argmax inference semantics. I also remove per-image reseeding (which currently makes HorizontalFlip deterministic per-image across all TTAs, reducing TTA diversity) while keeping `tta_n=5` unchanged, which should nudge accuracy upward without altering the approach. The rest of the pipeline (paths, preprocessing, loop structure, submission writing) remains the same and it still always write a valid `submission.csv`.'
- What this solution (achieved 0.09903) has done: 'I fix the runtime error by replacing the removed `albumentations.set_seed` call with a small helper that deterministically seeds Python, NumPy, and Albumentations’ internal RNG (`A.seed_everything`) for each (image, TTA) pass. This keeps your core inference logic identical (EfficientNet-B4, 5x TTA, averaged logits, argmax) while restoring the intended deterministic TTA diversity. I also add a tiny safety check for missing test images so the run fails fast with a clear message rather than crashing mid-loop. The script then run end-to-end and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.08184) has done: 'Your current score is far below the target, so the smallest likely score-improving change is to stop introducing randomness during test-time augmentation: your “TTA” currently uses probabilistic flips, which adds noise and can easily tank accuracy when averaged over only 5 passes. I keep the exact same core logic (EfficientNet-B4, 5-pass loop, averaged logits, argmax) but make TTA deterministic by replacing random flips with a fixed set of 5 deterministic transforms (identity, hflip, vflip, hvflip, rotate90). I also ensure the resize/crop stays consistent with B4 (380) and keep normalization identical, so evaluation semantics remain unchanged while reducing harmful inference variance toward the target score. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.08184) has done: 'Your current score (0.08184) is far below the target, so the most likely remaining issue is that inference-time preprocessing still doesn’t match what the checkpoint expects, causing near-random predictions even if weights load. I keep the same core logic (EfficientNet-B4, 5 deterministic TTA passes, averaged logits, argmax) but make two minimal score-relevant fixes: (1) use a safe resize-to-380 without a redundant CenterCrop (currently it’s a no-op but can introduce subtle library-dependent behavior), and (2) set the PIL interpolation explicitly and add a `torch.inference_mode()` context (same semantics as no_grad, less overhead/safer). Finally, I ensure the submission rows exactly follow `sample_submission.csv` order and add a small sanity check that the predicted labels are within `[0..4]` to catch any silent corruption.'

# 9. Code solution

## === cell 0
import os
import random
import glob

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

import albumentations as A



## === cell 1
model_path = "../input/en-b4-tta-calr-50/final_50.pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def resolve_existing_path(p: str, must_be_dir: bool = False) -> str:
    if os.path.isdir(p) if must_be_dir else os.path.exists(p):
        return p

    base = p.replace("\\", "/")
    candidates = []

    for root in ["../input", "/kaggle/input", "/kaggle/data/input", "/kaggle/data"]:
        alt = os.path.join(
            root, os.path.basename(os.path.dirname(base)), os.path.basename(base)
        )
        candidates.append(alt)

    for root in ["/kaggle/input", "/kaggle/data/input", "/kaggle/data"]:
        if base.startswith("../input/"):
            candidates.append(os.path.join(root, base[len("../input/") :]))

    for c in candidates:
        if os.path.isdir(c) if must_be_dir else os.path.exists(c):
            return c

    search_name = os.path.basename(base)
    for root in ["../input", "/kaggle/input", "/kaggle/data/input", "/kaggle/data"]:
        if os.path.isdir(root):
            pattern = os.path.join(root, "**", search_name)
            for hit in glob.glob(pattern, recursive=True):
                if os.path.isdir(hit) if must_be_dir else os.path.exists(hit):
                    return hit

    return p


def resolve_checkpoint_path_or_none(p: str):
    """
    Change (score-relevant): prioritize the exact intended dataset folder/name first.
    This reduces the chance we accidentally pick an unrelated .pth with same basename,
    which can lead to mismatched weights and near-random predictions (low accuracy).
    """
    if os.path.exists(p):
        return p

    base = p.replace("\\", "/")
    fname = os.path.basename(base)

    preferred_rel = base
    if preferred_rel.startswith("../input/"):
        preferred_rel = preferred_rel[len("../input/") :]

    preferred_candidates = []
    for root in ["../input", "/kaggle/input", "/kaggle/data/input", "/kaggle/data"]:
        if os.path.isdir(root):
            cand = os.path.join(root, preferred_rel)
            preferred_candidates.append(cand)
    for c in preferred_candidates:
        if os.path.exists(c):
            return c

    search_roots = ["../input", "/kaggle/input", "/kaggle/data/input", "/kaggle/data"]
    candidates = []
    for root in search_roots:
        if os.path.isdir(root):
            candidates.extend(
                glob.glob(os.path.join(root, "**", fname), recursive=True)
            )

    slug = os.path.basename(os.path.dirname(base))
    exact_slug = [c for c in candidates if f"/{slug}/" in c.replace("\\", "/")]
    if exact_slug:
        return exact_slug[0]

    preferred = [c for c in candidates if "en-b4-tta-calr-50" in c.replace("\\", "/")]
    if preferred:
        return preferred[0]

    if candidates:
        return candidates[0]
    return None


sample_sub_path = resolve_existing_path(sample_sub_path, must_be_dir=False)
test_images_path = resolve_existing_path(test_images_path, must_be_dir=True)

ckpt_path = resolve_checkpoint_path_or_none(model_path)
if ckpt_path is None:
    print(
        "WARNING: Checkpoint not found. Will fall back to torchvision ImageNet weights.\n"
        "This preserves the same EfficientNet-B4 + TTA + argmax core logic, but accuracy will likely be far below target."
    )
else:
    print("Found model checkpoint:", ckpt_path)

print("Using sample_submission:", sample_sub_path)
print("Using test_images dir:", test_images_path)



## === cell 2
from torchvision import models as tv_models


def _extract_state_dict(ckpt_obj):
    state_dict = ckpt_obj
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                state_dict = ckpt_obj[key]
                break
    return state_dict


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    out = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _adapt_classifier_keys_for_torchvision(sd: dict) -> dict:
    """
    Change (score-relevant): add robust head-key remapping so cassava-trained checkpoints
    using efficientnet_pytorch naming can load into torchvision EfficientNet without silently
    keeping an ImageNet/random head (which destroys accuracy).
    """
    out = dict(sd)

    if "_fc.weight" in out and "classifier.1.weight" not in out:
        out["classifier.1.weight"] = out.pop("_fc.weight")
    if "_fc.bias" in out and "classifier.1.bias" not in out:
        out["classifier.1.bias"] = out.pop("_fc.bias")

    if "fc.weight" in out and "classifier.1.weight" not in out:
        out["classifier.1.weight"] = out.pop("fc.weight")
    if "fc.bias" in out and "classifier.1.bias" not in out:
        out["classifier.1.bias"] = out.pop("fc.bias")

    if "classifier.weight" in out and "classifier.1.weight" not in out:
        out["classifier.1.weight"] = out.pop("classifier.weight")
    if "classifier.bias" in out and "classifier.1.bias" not in out:
        out["classifier.1.bias"] = out.pop("classifier.bias")

    return out


def _detect_efficientnet_pytorch(sd: dict) -> bool:
    keys = [k for k in sd.keys() if isinstance(k, str)]
    ep_hits = 0
    for pat in ["_conv_stem.", "_bn0.", "_blocks.", "_conv_head.", "_bn1.", "_fc."]:
        if any(pat in k for k in keys):
            ep_hits += 1
    return ep_hits >= 2


def _build_model_torchvision() -> nn.Module:
    try:
        eff_weights = tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1
    except Exception:
        eff_weights = None

    m = tv_models.efficientnet_b4(weights=eff_weights)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, 5)
    return m


def _build_model_efficientnet_pytorch() -> nn.Module:
    from efficientnet_pytorch import EfficientNet  # imported only if available

    m = EfficientNet.from_name("efficientnet-b4")
    in_features = m._fc.in_features
    m._fc = nn.Linear(in_features, 5)
    return m


def _load_checkpoint_strict(model: nn.Module, sd: dict) -> None:
    model.load_state_dict(sd, strict=True)
    print("Loaded checkpoint with strict=True into matching architecture.")


def _prepare_state_dict_for_model(sd: dict, model: nn.Module) -> dict:
    out = dict(sd)
    if any(isinstance(k, str) and k.startswith("module.") for k in out.keys()):
        out = _strip_prefix(out, "module.")
    for pref in ["model.", "net."]:
        if any(isinstance(k, str) and k.startswith(pref) for k in out.keys()):
            out = _strip_prefix(out, pref)

    if isinstance(model, tv_models.EfficientNet):
        out = _adapt_classifier_keys_for_torchvision(out)

    return out


model = _build_model_torchvision().to(device)

if ckpt_path is not None:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_sd = _extract_state_dict(ckpt)
        if not isinstance(raw_sd, dict):
            raise RuntimeError("Checkpoint did not contain a state_dict-like mapping.")

        use_ep = _detect_efficientnet_pytorch(raw_sd)
        if use_ep:
            try:
                ep_model = _build_model_efficientnet_pytorch().to(device)
                sd_ep = _prepare_state_dict_for_model(raw_sd, ep_model)
                _load_checkpoint_strict(ep_model, sd_ep)
                model = ep_model
            except Exception as e:
                print(
                    "WARNING: Detected efficientnet_pytorch-style checkpoint, but couldn't load with efficientnet_pytorch.\n"
                    "Falling back to torchvision model; will still attempt strict load after key adaptation.\n"
                    f"Load error: {repr(e)}"
                )
                sd_tv = _prepare_state_dict_for_model(raw_sd, model)
                try:
                    _load_checkpoint_strict(model, sd_tv)
                except Exception as e2:
                    print(
                        "WARNING: Could not strict-load checkpoint into torchvision EfficientNet-B4; using ImageNet weights.\n"
                        f"Load error: {repr(e2)}"
                    )
        else:
            sd_tv = _prepare_state_dict_for_model(raw_sd, model)
            try:
                _load_checkpoint_strict(model, sd_tv)
            except Exception as e:
                print(
                    "WARNING: Could not strict-load checkpoint into torchvision EfficientNet-B4; using ImageNet weights.\n"
                    f"Load error: {repr(e)}"
                )
    except Exception as e:
        print(
            "WARNING: Failed reading checkpoint; using torchvision ImageNet weights instead.\n"
            f"Load error: {repr(e)}"
        )

model.eval()



## === cell 3
tta_transforms = [
    A.Compose(
        [
            A.Resize(380, 380, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    ),
    A.Compose(
        [
            A.Resize(380, 380, p=1.0),
            A.HorizontalFlip(p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    ),
    A.Compose(
        [
            A.Resize(380, 380, p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    ),
    A.Compose(
        [
            A.Resize(380, 380, p=1.0),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    ),
    A.Compose(
        [
            A.Resize(380, 380, p=1.0),
            A.Rotate(limit=(90, 90), border_mode=0, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    ),
]


def alb_to_tensor(img_hwc: np.ndarray) -> torch.Tensor:
    if img_hwc.dtype != np.float32:
        img_hwc = img_hwc.astype(np.float32)
    return torch.from_numpy(img_hwc).permute(2, 0, 1).contiguous()




## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

pred_labels = []
tta_n = 5  # keep identical TTA loop count

with torch.inference_mode():
    for row_idx, sample_row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, sample_row.image_id)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing test image: {img_path}")

        image_np = np.array(Image.open(img_path).convert("RGB"))

        image_logits = torch.zeros((1, 5), device=device, dtype=torch.float32)

        for tta_i in range(tta_n):
            aug = tta_transforms[tta_i](image=image_np)[
                "image"
            ]  # HWC float32 normalized
            img_t = alb_to_tensor(aug).to(device)  # CHW float32 (already normalized)
            outputs = model(img_t.unsqueeze(0))  # logits
            image_logits = image_logits + outputs

        image_logits = image_logits / float(tta_n)
        pred_label = int(torch.argmax(image_logits, dim=1).item())
        pred_labels.append(pred_label)

sub_df = sample_sub.copy()
sub_df["label"] = pred_labels

if not sub_df["label"].between(0, 4).all():
    bad = sub_df.loc[~sub_df["label"].between(0, 4), "label"].unique().tolist()
    raise ValueError(f"Found out-of-range labels in predictions: {bad}")

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("Label distribution:\n", sub_df["label"].value_counts(dropna=False).sort_index())
