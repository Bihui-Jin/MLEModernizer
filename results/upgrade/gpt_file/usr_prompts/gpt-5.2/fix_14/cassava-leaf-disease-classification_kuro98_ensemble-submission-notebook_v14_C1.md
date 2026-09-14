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

0.9008763977032336

# 6. Current score

0.09753

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the ViT input-size assertion by automatically aligning `vit_img_size`/`eff_img_size` to the loaded models’ expected image sizes (224 for the torchvision fallback ViT), without changing the ensemble/training-free inference logic. I also make the TTA deterministic and collate-safe: `Random*` transforms be applied with fixed seeds per item/tta so the DataLoader can stack tensors reliably and results are reproducible. Finally, I ensure `pred_map` is always created (even if inference fails early) and that the submission strictly matches `sample_submission.csv` ordering/length and is written as `submission.csv`. These changes are correctness/stability fixes and should also improve score versus the broken run by enabling the intended inference/ensemble.'
- What this solution (achieved 0.22235) has done: 'I fix the DataLoader crash by ensuring the dataset only indexes actual image files (filtering out nested `test_images/` directories that exist inside the provided path). This is a correctness fix that allows inference to run end-to-end and produce a valid `submission.csv`. I also add a small safety fallback to build the image list from `sample_submission.csv` if needed, ensuring the prediction map aligns with the required submission rows and avoids missing-label issues. No model/ensemble logic is changed, so score should improve substantially versus the current broken/mostly-empty prediction run.'
- What this solution (achieved 0.22235) has done: 'Your current score (0.22235) is far below the target (0.90088), so we should improve accuracy while keeping the same inference-only ensemble core. The biggest issue is that your torchvision fallback models are ImageNet-pretrained but *not* Cassava-trained, and `linear_head` is never actually applied—so predictions are essentially random for the competition labels. The minimal, core-logic-preserving fix is to (1) correctly load the provided `.pt` checkpoints as `state_dict` when needed, (2) ensure the loaded objects are put into eval mode on the right device, and (3) actually apply `linear_head` (when present) to the combined logits before softmax. These changes keep your architecture/loop/TTA/ensemble intact but make the pipeline use the intended trained weights and head, which should move the score sharply toward the target.'
- What this solution (achieved 0.21226) has done: 'Your current score (0.22235) is far below the target (0.90088), so we should make minimal changes that improve correctness of inference rather than tuning for marginal gains. The biggest likely issue is a mismatch between your pretrained normalization/resize/crop pipeline and what your Cassava-trained checkpoints expect, plus the `linear_head` being applied to already-class logits (which can severely distort predictions if the head was trained on features). I (1) infer whether the loaded checkpoints already output 5-class logits and only apply `linear_head` when its input dimension matches the logits/features, and (2) make the input preprocessing closer to standard Cassava pipelines by switching from a fixed CenterCrop(600) to a simpler Resize to each model’s native size (keeping the same transforms/TTA logic), which often fixes large accuracy drops caused by aggressive cropping. These are small, inference-only adjustments that preserve your ensemble/TTA approach and should move the score sharply upward toward the target.'
- What this solution (achieved 0.17227) has done: 'Your score is far below the target, so we should focus on a small, correctness-focused fix that can materially improve accuracy without changing the ensemble/inference core. The biggest likely remaining issue is a preprocessing mismatch: using ImageNet mean/std for Cassava checkpoints often collapses accuracy, while Cassava solutions commonly use simple `[0.5,0.5,0.5]` normalization (or none) depending on training. I keep your resize/TTA/ensemble and model-loading logic intact, but switch to a safer Cassava-style normalization and also add a minimal safeguard to ensure both models really output 5 logits (otherwise fallback to argmax over whatever they output is effectively random). This should move the score substantially upward toward the target while remaining within your constraints and still producing `submission.csv`.'
- What this solution (achieved 0.16592) has done: 'Your current score (0.17227) is far below the target (0.90088), so we should make a minimal, correctness-focused fix that addresses the most likely source of near-random predictions: `ensure_5_logits()` currently creates a brand-new random Linear projection at inference time whenever a model’s output isn’t already 5 logits, which destroys accuracy. I change it to instead extract the correct logits from common Cassava checkpoint wrappers (e.g., `{"logits": ...}` / `{"pred": ...}`) and, if dimensions still don’t match 5, fall back to a safe “use the first 5 channels” behavior (no random weights). I also change ensembling to average probabilities (post-softmax) rather than logits to reduce calibration mismatch between the two models with minimal semantic change, while keeping your models, TTA, and overall inference pipeline intact. This should move the score materially upward toward the target while still producing a valid `submission.csv` in the required order.'
- What this solution (achieved 0.1988) has done: 'Your current score is far below target, so the most likely way to move toward 0.90 while keeping your inference-only ensemble logic is to fix a key preprocessing mismatch that can make Cassava-trained checkpoints behave almost randomly. I keep your models/TTA/ensemble structure intact, but change normalization to standard ImageNet mean/std (what torchvision ViT/EfficientNet weights and many Cassava pipelines expect) and ensure the EfficientNet resize uses a more typical 380px default (B4’s native), instead of 528 which can hurt accuracy if the checkpoint wasn’t trained that way. I also switch the voting weights to a neutral 0.5/0.5 (still the same “average probs” ensemble) to reduce the risk of overweighting a miscalibrated member. These are minimal, inference-only changes that should materially increase accuracy versus the current near-random output without altering your core loop or architecture.'
- What this solution (achieved 0.1932) has done: 'The current score is far below target, so the most likely “minimal but meaningful” gain is to stop feeding the models inconsistent TTA views: right now each TTA transform is applied independently to the ViT-resized and EfficientNet-resized images, so the two models ensemble different augmentations per view, which makes the averaged probabilities noisy and can look near-random. I change TTA so that each view uses a *deterministic paired transform* (same flip/rotation/affine/perspective parameters) applied to both resized images, while keeping your ensemble, weights (0.5/0.5), and inference-only loop intact. This preserves your core logic (same models, same TTA idea, same averaging) but fixes a correctness issue in how TTA is coupled across ensemble members. Everything else (paths, submission writing, ordering via sample_submission) stays the same.'
- What this solution (achieved 0.1932) has done: 'Your current score (0.1932) is far below the target (0.90088), so we should make a small, high-impact *correctness* fix rather than tuning. The biggest likely issue is that your torchvision fallback architectures may not match the Cassava checkpoints you load with `strict=False`, leaving large parts randomly initialized (and therefore predictions near-random). I (1) load checkpoints more robustly by stripping common prefixes and, crucially, by resolving key mismatches between torchvision ViT/EfficientNet naming and many training scripts, and (2) add a minimal sanity check that warns (and falls back) if too few parameters were actually loaded. This keeps your ensemble/TTA/inference loop intact, but increases the chance you’re truly using trained weights—moving accuracy sharply toward the target.'
- What this solution (achieved 0.1932) has done: 'The current score is far below the target, so we should make a small correctness-focused change that increases accuracy without changing your ensemble/TTA/inference structure. The most likely remaining issue is that your ViT/EfficientNet checkpoints are not actually being loaded into the right architecture keys (coverage is low), so the models are effectively random; we add a minimal “key remapping” step for the most common torchvision ViT/EfficientNet naming differences and then reload with higher real coverage. We also make sure we only replace classifier/head weights when the checkpoint actually contains matching keys (so we don’t accidentally keep random 5-class heads when the checkpoint is 5-class but named differently). Everything else (paired TTA, probability averaging, submission alignment to sample_submission) stays the same, and we still write `submission.csv`.'
- What this solution (achieved 0.1932) has done: 'We make the smallest change that is most likely to move accuracy toward your 0.90 target: ensure the Cassava checkpoints actually load into the torchvision fallback architectures with high key-coverage, because low coverage leaves models effectively random (matching your ~0.19 score). Concretely, we (1) extend the state_dict key remapping for common ViT/EfficientNet naming patterns seen in timm/Lightning-style checkpoints and (2) automatically “route” checkpoint classifier/head weights into the correct torchvision head/classifier keys when shapes match. This preserves your ensemble/TTA/inference loop and does not change the architecture or training approach—only makes weight-loading correct. Submission writing/ordering stays identical and still produces `submission.csv`.'
- What this solution (achieved 0.09753) has done: 'Your score (0.1932) is far below the target (0.9009), so we need a high-impact correctness fix while keeping your ensemble/TTA inference core unchanged. The most likely cause of near-random accuracy is that your checkpoints aren’t actually loading into the torchvision fallback models (low key coverage), leaving most weights random; I add a minimal “auto-architecture selection” step that builds the correct torchvision variant (e.g., ViT-B/16 384px, EfficientNet-B3/B4/B5) by inspecting checkpoint tensor shapes, then reloads with much higher coverage. I also ensure the classifier/head weights are routed correctly when shapes match, but won’t change your TTA, averaging, softmax, or submission alignment logic. This should move accuracy substantially toward the target without changing the fundamental approach.'

# 9. Code solution

## === cell 0
import os
import random
from pathlib import Path

import numpy as np
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

random.seed(3407)
np.random.seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

eff_img_size = 380
vit_img_size = 384

batch_size = 16
num_workers = 4
num_classes = 5
tta = True

VIT_PATH = "/kaggle/input/vit-v1-update/vit_v1_1.pt"
EFF_PATH = "/kaggle/input/efficient-net/vit_cont_3.pt"
HEAD_PATH = "/kaggle/input/linear-head/linear_cls.pt"


def _try_load_torch_obj(path: str):
    try:
        if os.path.exists(path):
            return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
    return None


def _is_state_dict(x):
    return isinstance(x, dict) and any(
        isinstance(k, str) and (k.endswith("weight") or k.endswith("bias"))
        for k in x.keys()
    )


def _clean_state_dict_keys(sd: dict) -> dict:
    cleaned = {}
    for k, v in sd.items():
        nk = k
        for pref in ("module.", "model.", "net.", "backbone.", "encoder."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        cleaned[nk] = v
    return cleaned


def _extract_state_dict(obj):
    if obj is None:
        return None
    if _is_state_dict(obj):
        return obj
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in obj and _is_state_dict(obj[key]):
                return obj[key]
    return None


vit_obj = _try_load_torch_obj(VIT_PATH)
eff_obj = _try_load_torch_obj(EFF_PATH)
head_obj = _try_load_torch_obj(HEAD_PATH)

vit_sd = _extract_state_dict(vit_obj)
eff_sd = _extract_state_dict(eff_obj)

if vit_sd is not None:
    vit_sd = _clean_state_dict_keys(vit_sd)
if eff_sd is not None:
    eff_sd = _clean_state_dict_keys(eff_sd)

vit_model = vit_obj if isinstance(vit_obj, torch.nn.Module) else None
eff_model = eff_obj if isinstance(eff_obj, torch.nn.Module) else None
linear_head = head_obj if isinstance(head_obj, torch.nn.Module) else None

need_fallback_arch = (vit_model is None) or (eff_model is None)

import torchvision


def _infer_vit_variant_and_image_size_from_sd(sd: dict):
    """
    Score-related fix (minimal, preserves inference/ensemble logic):
    If we must instantiate a torchvision ViT to host a checkpoint, choose the
    correct variant by inspecting common ViT tensor shapes in the checkpoint.
    Wrong variant -> very low weight coverage -> near-random predictions.
    """
    if not isinstance(sd, dict) or len(sd) == 0:
        return ("vit_b_16", 224)

    embed_dim = None
    patch = None

    for k in ("conv_proj.weight", "patch_embed.proj.weight"):
        if k in sd and isinstance(sd[k], torch.Tensor) and sd[k].ndim == 4:
            embed_dim = int(sd[k].shape[0])
            patch = int(sd[k].shape[-1])
            break

    img_size = 224
    for k in ("encoder.pos_embedding", "pos_embed", "pos_embedding"):
        if k in sd and isinstance(sd[k], torch.Tensor) and sd[k].ndim == 3:
            n_tokens = int(sd[k].shape[1])  # includes class token
            n_patches = n_tokens - 1
            if n_patches > 0:
                side = int(round(n_patches**0.5))
                if side * side == n_patches and patch in (16, 32):
                    img_size = side * int(patch)
            break

    if embed_dim == 768 and patch == 16:
        return ("vit_b_16", img_size)
    if embed_dim == 768 and patch == 32:
        return ("vit_b_32", img_size)
    if embed_dim == 1024 and patch == 16:
        return ("vit_l_16", img_size)
    if embed_dim == 1024 and patch == 32:
        return ("vit_l_32", img_size)

    return ("vit_b_16", img_size)


def _infer_eff_variant_from_sd(sd: dict):
    """
    Score-related fix (minimal, preserves inference/ensemble logic):
    EfficientNet variants differ in stage widths. Pick B3/B4/B5 based on the
    stem conv out_channels when possible.
    """
    if not isinstance(sd, dict) or len(sd) == 0:
        return "efficientnet_b4"

    stem_key_candidates = [
        "features.0.0.weight",  # torchvision
        "conv_stem.weight",  # timm-ish
    ]
    stem_w = None
    for k in stem_key_candidates:
        if k in sd and isinstance(sd[k], torch.Tensor) and sd[k].ndim == 4:
            stem_w = sd[k]
            break

    if stem_w is not None:
        out_ch = int(stem_w.shape[0])
        if out_ch == 40:
            return "efficientnet_b3"
        if out_ch == 48:
            return "efficientnet_b4"
        if out_ch == 56:
            return "efficientnet_b6"
        if out_ch == 64:
            return "efficientnet_b7"
        if out_ch == 32:
            return "efficientnet_b2"

    return "efficientnet_b4"


if need_fallback_arch:
    print(
        "Loading torchvision backbones to host state_dict checkpoints (or as a final fallback)."
    )

    if vit_model is None:
        vit_variant, vit_img_size_inferred = _infer_vit_variant_and_image_size_from_sd(
            vit_sd or {}
        )
        vit_img_size = (
            int(vit_img_size_inferred) if vit_img_size_inferred else vit_img_size
        )
        vit_ctor = getattr(torchvision.models, vit_variant)
        vit_model = vit_ctor(weights=None)
        vit_model.heads = torch.nn.Linear(vit_model.heads.head.in_features, num_classes)
        print(
            f"Instantiated torchvision {vit_variant} for checkpoint hosting; vit_img_size={vit_img_size}"
        )

    if eff_model is None:
        eff_variant = _infer_eff_variant_from_sd(eff_sd or {})
        eff_ctor = getattr(torchvision.models, eff_variant)
        eff_model = eff_ctor(weights=None)
        eff_model.classifier[1] = torch.nn.Linear(
            eff_model.classifier[1].in_features, num_classes
        )
        print(f"Instantiated torchvision {eff_variant} for checkpoint hosting.")


def _remap_state_dict_for_torchvision(model: torch.nn.Module, sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd

    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    out = dict(sd)

    def _maybe_rename(src: str, dst: str):
        if src in out and dst in model_keys and dst not in out:
            out[dst] = out.pop(src)

    def _maybe_copy(src: str, dst: str):
        if src in out and dst in model_keys and dst not in out:
            out[dst] = out[src]

    def _maybe_route_by_shape(src: str, dst: str):
        if src in out and dst in model_keys and dst not in out:
            sv = out[src]
            dv = model_sd[dst]
            if (
                isinstance(sv, torch.Tensor)
                and isinstance(dv, torch.Tensor)
                and tuple(sv.shape) == tuple(dv.shape)
            ):
                out[dst] = sv

    for src_w in (
        "head.weight",
        "heads.head.weight",
        "classifier.weight",
        "fc.weight",
        "cls_head.weight",
        "head.fc.weight",
        "mlp_head.weight",
    ):
        _maybe_route_by_shape(src_w, "heads.head.weight")
        _maybe_copy(src_w, "heads.head.weight")
    for src_b in (
        "head.bias",
        "heads.head.bias",
        "classifier.bias",
        "fc.bias",
        "cls_head.bias",
        "head.fc.bias",
        "mlp_head.bias",
    ):
        _maybe_route_by_shape(src_b, "heads.head.bias")
        _maybe_copy(src_b, "heads.head.bias")

    _maybe_route_by_shape("head.0.weight", "heads.head.weight")
    _maybe_route_by_shape("head.0.bias", "heads.head.bias")
    _maybe_copy("head.0.weight", "heads.head.weight")
    _maybe_copy("head.0.bias", "heads.head.bias")

    for src_w in (
        "classifier.weight",
        "head.weight",
        "fc.weight",
        "classifier.1.weight",
        "classifier.fc.weight",
        "classifier.linear.weight",
        "classif.weight",
    ):
        _maybe_route_by_shape(src_w, "classifier.1.weight")
        _maybe_copy(src_w, "classifier.1.weight")
    for src_b in (
        "classifier.bias",
        "head.bias",
        "fc.bias",
        "classifier.1.bias",
        "classifier.fc.bias",
        "classifier.linear.bias",
        "classif.bias",
    ):
        _maybe_route_by_shape(src_b, "classifier.1.bias")
        _maybe_copy(src_b, "classifier.1.bias")

    _maybe_rename("conv_stem.weight", "features.0.0.weight")
    _maybe_rename("bn1.weight", "features.0.1.weight")
    _maybe_rename("bn1.bias", "features.0.1.bias")
    _maybe_rename("bn1.running_mean", "features.0.1.running_mean")
    _maybe_rename("bn1.running_var", "features.0.1.running_var")
    _maybe_rename("bn1.num_batches_tracked", "features.0.1.num_batches_tracked")

    return out


def _load_state_dict_forgiving(model, sd_or_obj, name: str):
    sd = _extract_state_dict(sd_or_obj)
    if sd is None and isinstance(sd_or_obj, dict):
        sd = sd_or_obj
    if sd is None:
        return False

    sd = _clean_state_dict_keys(sd)
    sd = _remap_state_dict_for_torchvision(model, sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)

    model_keys = set(model.state_dict().keys())
    sd_keys = set(sd.keys())
    matched = len(model_keys & sd_keys)
    coverage = matched / max(1, len(model_keys))

    print(
        f"{name}: loaded state_dict strict=False; "
        f"matched={matched}/{len(model_keys)} ({coverage:.1%}), missing={len(missing)}, unexpected={len(unexpected)}"
    )

    if coverage < 0.10:
        print(
            f"WARNING: {name} checkpoint coverage is very low ({coverage:.1%}). "
            "Predictions may be near-random; verify checkpoint matches architecture."
        )
    return True


if vit_sd is not None and not isinstance(vit_obj, torch.nn.Module):
    _ = _load_state_dict_forgiving(vit_model, vit_sd, "vit_model")
if eff_sd is not None and not isinstance(eff_obj, torch.nn.Module):
    _ = _load_state_dict_forgiving(eff_model, eff_sd, "eff_model")

if linear_head is None:
    sd = _extract_state_dict(head_obj)
    if sd is not None:
        sd = _clean_state_dict_keys(sd)
        w = None
        for k, v in sd.items():
            if (
                isinstance(k, str)
                and k.endswith("weight")
                and isinstance(v, torch.Tensor)
            ):
                w = v
                break
        if w is not None and w.ndim == 2 and w.shape[0] == num_classes:
            linear_head = torch.nn.Linear(int(w.shape[1]), num_classes)
            _ = _load_state_dict_forgiving(linear_head, sd, "linear_head")
        else:
            linear_head = torch.nn.Identity()
    else:
        linear_head = torch.nn.Identity()


def _infer_vit_image_size(model):
    if hasattr(model, "image_size"):
        try:
            return int(model.image_size)
        except Exception:
            pass
    return int(vit_img_size) if vit_img_size else 224


def _infer_eff_image_size(model, default):
    return int(default)


vit_img_size = _infer_vit_image_size(vit_model)
eff_img_size = _infer_eff_image_size(eff_model, eff_img_size)
print(f"Using vit_img_size={vit_img_size}, eff_img_size={eff_img_size}")

vit_model = vit_model.to(device)
eff_model = eff_model.to(device)
linear_head = linear_head.to(device)


def _get_linear_in_features(m):
    if isinstance(m, torch.nn.Linear):
        return int(m.in_features)
    return None


linear_in = _get_linear_in_features(linear_head)
print(f"linear_head type={type(linear_head).__name__}, in_features={linear_in}")


def apply_head_if_compatible(x: torch.Tensor) -> torch.Tensor:
    if isinstance(linear_head, torch.nn.Identity):
        return x
    if linear_in is not None and x.shape[-1] == linear_in:
        return linear_head(x)
    return x


def _unwrap_model_output(out):
    if isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, dict):
        for k in ("logits", "pred", "out", "output", "y", "cls", "classification"):
            if k in out:
                out = out[k]
                break
    return out


def ensure_5_logits(x: torch.Tensor, name: str) -> torch.Tensor:
    if x is None:
        raise RuntimeError(f"{name}: got None output")
    if not isinstance(x, torch.Tensor):
        raise RuntimeError(f"{name}: expected Tensor output, got {type(x)}")

    if x.ndim != 2:
        x = x.view(x.shape[0], -1)

    if x.shape[1] == num_classes:
        return x

    if x.shape[1] > num_classes:
        return x[:, :num_classes]
    else:
        pad = x.new_zeros((x.shape[0], num_classes - x.shape[1]))
        return torch.cat([x, pad], dim=1)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava test images.

    Score-related fix (kept as-is): paired TTA across the two model inputs.
    """

    def __init__(
        self,
        data_dir,
        vit_size,
        efficient_size,
        transform=None,
        tta_transforms=None,
        tta_seed=3407,
        image_ids=None,
    ):
        super().__init__(root=data_dir)

        self.transform = transform

        if image_ids is not None:
            self.images = list(image_ids)
        else:
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            items = []
            for name in os.listdir(data_dir):
                p = os.path.join(data_dir, name)
                if os.path.isfile(p) and Path(name).suffix.lower() in exts:
                    items.append(name)
            self.images = sorted(items)

        self.tta_transforms = tta_transforms or []
        self.tta_seed = int(tta_seed)

        self.resize_vit = v2.Resize(
            (vit_size, vit_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")

        vit_base = self.resize_vit(img)
        eff_base = self.resize_efficient(img)

        if self.transform is None:
            raise RuntimeError("transform must be provided to produce tensors.")

        if len(self.tta_transforms) > 0:
            vit_views = []
            eff_views = []
            for j, t in enumerate(self.tta_transforms):
                rng_seed = self.tta_seed + idx * 1000 + j
                torch.manual_seed(rng_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(rng_seed)

                if hasattr(t, "make_params") and hasattr(t, "transform"):
                    params = t.make_params([vit_base])
                    vit_aug = t.transform(vit_base, params)
                    eff_aug = t.transform(eff_base, params)
                else:
                    vit_aug = t(vit_base)
                    torch.manual_seed(rng_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(rng_seed)
                    eff_aug = t(eff_base)

                vit_views.append(self.transform(vit_aug))
                eff_views.append(self.transform(eff_aug))

            return (
                torch.stack(vit_views, dim=0),
                torch.stack(eff_views, dim=0),
                filename,
            )

        vit_t = self.transform(vit_base)
        eff_t = self.transform(eff_base)
        return vit_t, eff_t, filename

    def __len__(self):
        return len(self.images)




## === cell 2
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    tta_transforms = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomAffine(180),
        v2.RandomPerspective(p=1),
    ]
else:
    tta_transforms = []

sample_sub_for_ids = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub_for_ids["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir,
    vit_img_size,
    eff_img_size,
    transform=test_transforms,
    tta_transforms=tta_transforms,
    tta_seed=3407,
    image_ids=test_image_ids,
)


def seed_worker(worker_id):
    worker_seed = (3407 + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(3407)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
all_names = []
all_preds = []

vit_model.eval()
eff_model.eval()
linear_head.eval()

pred_map = {}

with torch.no_grad():
    for vit_inputs, eff_inputs, filenames in test_loader:
        filenames = list(filenames)
        base_bs = len(filenames)

        if tta:
            if vit_inputs.ndim != 5 or eff_inputs.ndim != 5:
                raise RuntimeError(
                    f"Expected TTA tensors with shape [B,T,C,H,W], got vit={tuple(vit_inputs.shape)}, eff={tuple(eff_inputs.shape)}"
                )
            B, T = vit_inputs.shape[0], vit_inputs.shape[1]
            vit_inputs = vit_inputs.permute(1, 0, 2, 3, 4).reshape(
                T * B, *vit_inputs.shape[2:]
            )
            eff_inputs = eff_inputs.permute(1, 0, 2, 3, 4).reshape(
                T * B, *eff_inputs.shape[2:]
            )

            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)

            vit_outputs = _unwrap_model_output(vit_model(vit_inputs))
            eff_outputs = _unwrap_model_output(eff_model(eff_inputs))

            vit_batch_logits = vit_outputs.view(T, base_bs, -1)
            vit_mean_logits = vit_batch_logits.mean(dim=0)

            eff_batch_logits = eff_outputs.view(T, base_bs, -1)
            eff_mean_logits = eff_batch_logits.mean(dim=0)

            vit_mean_logits = ensure_5_logits(vit_mean_logits, "vit")
            eff_mean_logits = ensure_5_logits(eff_mean_logits, "eff")

            vit_probs = normalizer(apply_head_if_compatible(vit_mean_logits))
            eff_probs = normalizer(apply_head_if_compatible(eff_mean_logits))

            mean_probs = 0.5 * vit_probs + 0.5 * eff_probs
            pred_labels = torch.argmax(mean_probs, 1).tolist()
        else:
            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)

            vit_outputs = _unwrap_model_output(vit_model(vit_inputs))
            eff_outputs = _unwrap_model_output(eff_model(eff_inputs))

            vit_outputs = ensure_5_logits(vit_outputs, "vit")
            eff_outputs = ensure_5_logits(eff_outputs, "eff")

            vit_probs = normalizer(apply_head_if_compatible(vit_outputs))
            eff_probs = normalizer(apply_head_if_compatible(eff_outputs))
            mean_probs = (vit_probs + eff_probs) / 2.0
            pred_labels = torch.argmax(mean_probs, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

pred_map = dict(zip(all_names, all_preds))
print(
    f"Predicted {len(pred_map)} unique test images out of {len(all_names)} total entries."
)



## === cell 4
sample_sub = pd.read_csv(sample_sub_path)
sample_sub["label"] = sample_sub["image_id"].map(pred_map)

if sample_sub["label"].isna().any():
    mode_label = (
        int(pd.Series(list(pred_map.values())).mode().iloc[0]) if len(pred_map) else 0
    )
    sample_sub["label"] = sample_sub["label"].fillna(mode_label).astype(int)
else:
    sample_sub["label"] = sample_sub["label"].astype(int)

submission_path = "submission.csv"
sample_sub.to_csv(submission_path, index=False)
print(f"Wrote {submission_path} with shape={sample_sub.shape}")
print(sample_sub.head())
