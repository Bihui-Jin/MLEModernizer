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

0.8715624055605923

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11211) has done: 'I fix the missing `efficientnet_pytorch` import by removing the unavailable wheel install and instead using `torchvision`’s built-in EfficientNet-B4 with a 5-class classifier head, which preserves the same core model family and allows the provided `.pth` weights to load (with key/shape handling). I update the Albumentations `RandomResizedCrop` call to the new v2 API (expects `size=(h,w)`) and replace deprecated transforms (`Cutout`) with `CoarseDropout` only so augmentation runs. I also fix the test-time augmentation loop bug (it loops 5 times but divides by 10) and ensure tensors are correctly normalized and moved to the right device. Finally, I guarantee a valid `submission.csv` with the exact required columns is written.'
- What this solution (achieved 0.11024) has done: 'I fix the path/IO bug that prevents the model weights from being found by searching for the `.pth` file inside the provided `/kaggle/input` directory structure and falling back safely if it’s not present. Then I correct the preprocessing bug: you currently `A.Normalize()` in Albumentations and then apply `transforms.ToTensor()` which re-scales values again, badly breaking inference and causing the very low score; I replace this with a single consistent “to tensor + normalize” step that matches the EfficientNet expectations. I also switch TTA to deterministic test-time transforms (flips/transpose) only—keeping the same TTA averaging approach but removing training-only random crops/rotations that can harm accuracy at inference. Finally, I ensure `submission.csv` is written with the exact required columns and row order from `sample_submission.csv`.'
- What this solution (achieved 0.11024) has done: 'Your current score is far below the target, so the smallest likely win is to make inference preprocessing match what the EfficientNet-B4 checkpoint expects. I (1) add the missing EfficientNet resize/crop step (B4 typically expects a centered 380×380 crop) before normalization, (2) ensure the transpose-based TTA variants produce a valid square input by resizing after the geometric transform, and (3) speed up/clean inference slightly (batch the TTA stack per image) without changing the model or the TTA averaging logic. These changes keep the same architecture, checkpoint loading, loss/eval semantics, and produce the same submission format, but should materially improve accuracy if the checkpoint was trained with standard EfficientNet sizing.'
- What this solution (achieved 0.11024) has done: 'Your score indicates the model is effectively guessing, which is most consistent with loading the wrong checkpoint (or none) even though inference preprocessing is now sane. I make the checkpoint selection stricter by preferring files that (a) contain “b4/efficientnet/cassava”, (b) are reasonably sized (to avoid tiny optimizer-only stubs), and (c) actually match the model’s tensor shapes, by trying the top candidates and selecting the first that loads cleanly. This keeps the same EfficientNet-B4 architecture and inference/TTA logic, but increases the chance you’re actually using the intended trained weights, which should move accuracy toward your target. Everything still runs end-to-end and writes a valid `submission.csv` in the required format.'
- What this solution (achieved 0.11024) has done: 'Your current score is far below the target, so the most likely cause is that you’re still not loading the intended Cassava-trained checkpoint (so the model is effectively random). I make checkpoint selection stricter by (1) requiring EfficientNet-B4-compatible key patterns and (2) preferring checkpoints whose classifier weights are exactly shape `(5, in_features)` and whose missing-key ratio is tiny, and then I verify loading by printing a small sanity check (logit stats) on a few images. I also fix one likely pitfall: `A.CenterCrop(B4_SIZE, B4_SIZE)` right after `A.Resize(B4_SIZE, B4_SIZE)` is redundant; keeping it can’t help and occasionally interacts badly if a variant ever becomes non-square—so I remove it but keep the same resize/normalize/tensor pipeline. These changes keep the same model architecture (torchvision EfficientNet-B4), same inference/TTA averaging core logic, and should move accuracy sharply upward if a proper checkpoint exists in `/kaggle/input`.'
- What this solution (achieved 0.11024) has done: 'Your score is far below the target, so the most likely remaining issue is that you’re still not actually loading the intended Cassava-trained checkpoint (the current loader is overly strict by requiring `classifier.1.*` keys, which many EfficientNet checkpoints don’t have, and it also “tests” candidates on the GPU which is unnecessary). I keep the same EfficientNet-B4 architecture and inference/TTA logic, but change checkpoint selection to (1) evaluate candidates on CPU, (2) accept multiple common classifier head key patterns (including `classifier.weight`, `head.fc.*`, etc.), and (3) pick the checkpoint that best matches the backbone + has a plausible 5-class head, then remap classifier keys into `classifier.1.*` when needed. This is a minimal, score-relevant change: it increases the chance the model uses real trained weights instead of random initialization, which should move accuracy sharply toward your target. The rest of the pipeline (380 resize, ImageNet normalization, 5-way argmax, submission format) stays the same.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, so the most likely issue is still that you aren’t actually loading a real Cassava-trained checkpoint and are effectively predicting with random weights. I keep your EfficientNet-B4 + 5-class head and the same inference/TTA averaging, but make checkpoint loading more compatible by (1) preferring torchvision EfficientNet-B4 pretrained weights as a safe backbone fallback, and (2) expanding checkpoint key remapping to handle the very common `model.classifier.*` / `classifier.*` patterns (including `classifier.0.*`) so more real checkpoints can load cleanly. These are minimal, score-relevant changes that should move accuracy sharply upward if a valid checkpoint exists, and otherwise still improve versus pure random init while producing the same `submission.csv` format.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is far below the target (0.87156), which strongly suggests the model is still effectively not using a real Cassava-trained checkpoint (or the wrong weights are being partially loaded), so the smallest score-relevant fix is to (1) broaden and harden checkpoint extraction/remapping so common EfficientNet implementations load correctly, and (2) ensure we select the checkpoint that best matches the full backbone (not just “has some EfficientNet keys”). I keep the same EfficientNet-B4 architecture, the same preprocessing/normalization, and the same 5-variant TTA averaging, but change checkpoint scoring to prefer candidates with a low missing-keys ratio specifically on `features.*` and a valid 5-class head after remapping. I also handle the very common EfficientNet-PyTorch naming (`_conv_stem`, `_blocks`, `_fc`) by translating those keys into torchvision’s `features.*` layout when possible (only as a loading compatibility shim; inference logic stays identical). This should materially increase the chance you’re actually running a trained Cassava model, moving accuracy toward your target while still producing the same `submission.csv` format.'
- What this solution (achieved 0.05531) has done: 'Your current score is far below the target, so we should focus on the smallest inference-time fixes that can materially improve accuracy without changing the model family or evaluation semantics. The biggest likely issue is that torchvision EfficientNet expects specific input normalization *and* often benefits from a standard “resize then center-crop” evaluation pipeline; right now you always warp images to 380×380 which can distort leaves and hurt accuracy. I keep the same EfficientNet-B4 + 5-class head + 5-variant TTA averaging, but adjust preprocessing to `Resize(IMG_SCALE) -> CenterCrop(380)` (no warping) and add a safe PIL-exif transpose to avoid rotated images. I also make submission row order exactly match `sample_submission.csv` (by merging on `image_id`) to avoid any rare alignment issues.'
- What this solution (achieved 0.05531) has done: 'Your current score is far below the target, so the smallest likely fix is to ensure the model is actually using a Cassava-trained checkpoint rather than (mostly) ImageNet/random weights. I keep the exact EfficientNet-B4 architecture and the same 5-variant TTA averaging, but change checkpoint selection to explicitly prefer checkpoints that contain a real 5-class head (keys with shape `[5, ...]`) and that match EfficientNet-B4 `features.*` weights well, then load the best candidate. I also add support for another common checkpoint layout where the final classifier is stored as `classifier.1.*` or `classifier.*` but the rest is already torchvision-compatible, without changing inference preprocessing. These are minimal, score-relevant changes that should move accuracy sharply upward toward your target if a valid Cassava checkpoint exists in `/kaggle/input`.'

# 9. Code solution

## === cell 0
import os
import warnings
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torchvision import models

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 1
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

assert os.path.exists(
    sample_sub_path
), f"Missing sample submission at: {sample_sub_path}"
assert os.path.isdir(
    test_images_path
), f"Missing test images dir at: {test_images_path}"

preferred_model_path = "../input/en-b4-tta-calr-15-v2/model(13).pth"


def find_checkpoint_candidates():
    roots = ["../input", "/kaggle/input"]
    candidates = []

    if os.path.exists(preferred_model_path):
        candidates.append(preferred_model_path)

    for r in roots:
        rp = Path(r)
        if rp.exists():
            candidates.extend([str(p) for p in rp.rglob("*.pth")])
            candidates.extend([str(p) for p in rp.rglob("*.pt")])

    seen = set()
    uniq = []
    for p in candidates:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def checkpoint_rank_score(p: str) -> float:
    name = Path(p).name.lower()
    parent = str(Path(p).parent).lower()
    s = 0.0

    for key, val in [
        ("efficientnet", 8),
        ("tf_efficientnet", 8),
        ("efficientnet_b4", 8),
        ("b4", 7),
        ("enb4", 7),
        ("cassava", 6),
        ("leaf", 2),
        ("disease", 2),
        ("tta", 2),
        ("fold", 1),
        ("best", 1),
        ("model", 1),
    ]:
        if key in name or key in parent:
            s += val

    try:
        sz = os.path.getsize(p)
        if sz < 1_000_000:
            s -= 50
        elif sz < 10_000_000:
            s -= 10
        elif sz > 40_000_000:
            s += 5
    except OSError:
        s -= 5

    if name.endswith(".pth"):
        s += 1

    return s


candidates = find_checkpoint_candidates()
candidates = sorted(candidates, key=checkpoint_rank_score, reverse=True)
print(f"Found {len(candidates)} checkpoint candidates (top 10 shown):")
for p in candidates[:10]:
    try:
        print(
            "  ",
            p,
            "| size(MB)=",
            round(os.path.getsize(p) / 1024 / 1024, 2),
            "| score=",
            checkpoint_rank_score(p),
        )
    except OSError:
        print("  ", p, "| size(MB)=?", "| score=", checkpoint_rank_score(p))



## === cell 2
try:
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    print("Backbone init: torchvision EfficientNet_B4 ImageNet weights")
except Exception as e:
    print(
        "WARNING: could not load torchvision weights, falling back to random init:",
        repr(e),
    )
    model = models.efficientnet_b4(weights=None)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 5)
model = model.to(device)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        sd = ckpt_obj.get(
            "state_dict",
            ckpt_obj.get("model_state_dict", ckpt_obj.get("model", ckpt_obj)),
        )
    else:
        sd = ckpt_obj
    if not isinstance(sd, dict):
        return None
    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


def _state_dict_looks_like_efficientnet(sd: dict) -> bool:
    keys = list(sd.keys())
    has_tv_features = any(k.startswith("features.") for k in keys)
    has_enp_features = any(
        k.startswith("_conv_stem") or k.startswith("_blocks") for k in keys
    )
    has_any_head = any(
        k.startswith("classifier.")
        or k.startswith("head.")
        or k.startswith("fc.")
        or k.startswith("_fc.")
        for k in keys
    )
    return (has_tv_features or has_enp_features) and has_any_head


def _remap_head_keys_to_torchvision(sd: dict, expected_w_shape, expected_b_shape):
    if "classifier.1.weight" in sd and "classifier.1.bias" in sd:
        if (
            tuple(sd["classifier.1.weight"].shape) == expected_w_shape
            and tuple(sd["classifier.1.bias"].shape) == expected_b_shape
        ):
            return sd, True, "classifier.1.*"

    head_candidates = [
        ("classifier.weight", "classifier.bias"),
        ("classifier.0.weight", "classifier.0.bias"),
        ("head.weight", "head.bias"),
        ("head.fc.weight", "head.fc.bias"),
        ("fc.weight", "fc.bias"),
        ("_fc.weight", "_fc.bias"),
    ]

    for wk, bk in head_candidates:
        if wk in sd and bk in sd:
            if (
                tuple(sd[wk].shape) == expected_w_shape
                and tuple(sd[bk].shape) == expected_b_shape
            ):
                sd2 = dict(sd)
                sd2["classifier.1.weight"] = sd2.pop(wk)
                sd2["classifier.1.bias"] = sd2.pop(bk)
                return sd2, True, f"remap {wk}->classifier.1.*"

    return sd, False, None


def _maybe_remap_efficientnet_pytorch_to_torchvision(
    sd: dict,
) -> tuple[dict, bool, str | None]:
    keys = list(sd.keys())
    has_enp = any(k.startswith("_conv_stem") or k.startswith("_blocks") for k in keys)
    has_tv = any(k.startswith("features.") for k in keys)
    if (not has_enp) or has_tv:
        return sd, False, None

    sd2 = dict(sd)

    stem_map = {
        "_conv_stem.weight": "features.0.0.weight",
        "_bn0.weight": "features.0.1.weight",
        "_bn0.bias": "features.0.1.bias",
        "_bn0.running_mean": "features.0.1.running_mean",
        "_bn0.running_var": "features.0.1.running_var",
        "_bn0.num_batches_tracked": "features.0.1.num_batches_tracked",
    }

    moved = 0
    for src, dst in stem_map.items():
        if src in sd2 and dst not in sd2:
            sd2[dst] = sd2.pop(src)
            moved += 1

    head_map = {
        "_conv_head.weight": "features.8.0.weight",
        "_bn1.weight": "features.8.1.weight",
        "_bn1.bias": "features.8.1.bias",
        "_bn1.running_mean": "features.8.1.running_mean",
        "_bn1.running_var": "features.8.1.running_var",
        "_bn1.num_batches_tracked": "features.8.1.num_batches_tracked",
    }
    for src, dst in head_map.items():
        if src in sd2 and dst not in sd2:
            sd2[dst] = sd2.pop(src)
            moved += 1

    if moved > 0:
        return (
            sd2,
            True,
            f"remapped {moved} stem/head keys (efficientnet_pytorch -> torchvision)",
        )
    return sd, False, None


def _count_5class_head_tensors(sd: dict) -> int:
    n = 0
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if v.ndim == 2 and v.shape[0] == 5:
            n += 1
        if v.ndim == 1 and v.shape[0] == 5:
            n += 1
    return n


loaded = False
selected_model_path = None
last_err = None

cls_w_key = "classifier.1.weight"
cls_b_key = "classifier.1.bias"
expected_w_shape = tuple(model.state_dict()[cls_w_key].shape)
expected_b_shape = tuple(model.state_dict()[cls_b_key].shape)

best_path = None
best_details = None  # (score_tuple, path, head_ok, miss_ratio_all, miss_ratio_features, unexpected_len, head_src, compat_note, head5_count)

probe_model_cpu = models.efficientnet_b4(weights=None)
probe_in_features = probe_model_cpu.classifier[1].in_features
probe_model_cpu.classifier[1] = nn.Linear(probe_in_features, 5)
probe_model_cpu.eval()

probe_keys = list(probe_model_cpu.state_dict().keys())
probe_feature_keys = [k for k in probe_keys if k.startswith("features.")]

for p in candidates[:400]:
    try:
        if not os.path.exists(p):
            continue

        ckpt = torch.load(p, map_location="cpu")
        sd = _extract_state_dict(ckpt)
        if sd is None:
            continue
        if not _state_dict_looks_like_efficientnet(sd):
            continue

        head5_count = _count_5class_head_tensors(sd)

        sd, compat_used, compat_note = _maybe_remap_efficientnet_pytorch_to_torchvision(
            sd
        )
        sd, head_ok, head_src = _remap_head_keys_to_torchvision(
            sd, expected_w_shape, expected_b_shape
        )

        missing, unexpected = probe_model_cpu.load_state_dict(sd, strict=False)

        total_params = len(probe_model_cpu.state_dict())
        miss_ratio_all = len(missing) / max(1, total_params)

        missing_set = set(missing)
        miss_feat = sum(1 for k in probe_feature_keys if k in missing_set)
        miss_ratio_features = miss_feat / max(1, len(probe_feature_keys))

        score_tuple = (
            0 if head_ok else 1,
            0 if head5_count > 0 else 1,
            miss_ratio_features,
            miss_ratio_all,
            len(unexpected),
            -checkpoint_rank_score(p),
        )

        if best_details is None or score_tuple < best_details[0]:
            best_path = p
            best_details = (
                score_tuple,
                p,
                head_ok,
                miss_ratio_all,
                miss_ratio_features,
                len(unexpected),
                head_src,
                compat_note if compat_used else None,
                head5_count,
            )

            if (
                head_ok
                and head5_count > 0
                and miss_ratio_features <= 0.02
                and miss_ratio_all <= 0.02
            ):
                break

        probe_model_cpu = models.efficientnet_b4(weights=None)
        probe_in_features = probe_model_cpu.classifier[1].in_features
        probe_model_cpu.classifier[1] = nn.Linear(probe_in_features, 5)
        probe_model_cpu.eval()

    except Exception as e:
        last_err = e
        continue

if best_path is not None:
    try:
        ckpt = torch.load(best_path, map_location="cpu")
        sd = _extract_state_dict(ckpt)
        sd, compat_used, compat_note = _maybe_remap_efficientnet_pytorch_to_torchvision(
            sd
        )
        sd, _, head_src = _remap_head_keys_to_torchvision(
            sd, expected_w_shape, expected_b_shape
        )

        model.load_state_dict(sd, strict=False)
        loaded = True
        selected_model_path = best_path
        print("Selected checkpoint:", selected_model_path)
        print(
            "  selection_details(score_tuple, head_ok, miss_ratio_all, miss_ratio_features, unexpected_len, head_src, compat_note, head5_count) =",
            best_details[0],
            best_details[2],
            round(best_details[3], 6),
            round(best_details[4], 6),
            best_details[5],
            best_details[6],
            best_details[7],
            best_details[8],
        )
    except Exception as e:
        last_err = e
        loaded = False

if not loaded:
    print(
        "WARNING: No compatible EfficientNet-B4 checkpoint found; using ImageNet-initialized backbone + random 5-class head (submission valid, but likely below target)."
    )
    if last_err is not None:
        print("Last checkpoint load error:", repr(last_err))
else:
    print("Checkpoint loaded:", selected_model_path)

model.eval()



## === cell 3
B4_SIZE = 380
IMG_SCALE = (
    416  # resize then center-crop (no warping), consistent with prior code intent
)

tta_aug = A.Compose(
    [
        A.ToFloat(max_value=255.0),
        A.SmallestMaxSize(max_size=IMG_SCALE, interpolation=1),
        A.CenterCrop(height=B4_SIZE, width=B4_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.transforms.ToTensorV2(),
    ]
)


def apply_tta_variant(img_np, variant_id: int):
    if variant_id == 0:
        x = img_np
    elif variant_id == 1:
        x = np.ascontiguousarray(np.flip(img_np, axis=1))  # H-flip
    elif variant_id == 2:
        x = np.ascontiguousarray(np.flip(img_np, axis=0))  # V-flip
    elif variant_id == 3:
        x = np.ascontiguousarray(np.transpose(img_np, (1, 0, 2)))  # transpose
    elif variant_id == 4:
        x = np.ascontiguousarray(
            np.flip(np.transpose(img_np, (1, 0, 2)), axis=1)
        )  # transpose + H-flip
    else:
        x = img_np
    return x




## === cell 4
sample_sub = pd.read_csv(sample_sub_path)


def _read_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.array(img)


def _sanity_check(n=3):
    n = min(n, len(sample_sub))
    rows = sample_sub.head(n)
    with torch.no_grad():
        for i, r in rows.iterrows():
            img_path = os.path.join(test_images_path, r.image_id)
            image_np = _read_image_rgb(img_path)
            x = tta_aug(image=image_np)["image"].unsqueeze(0).to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1).detach().cpu().numpy()[0]
            print(
                f"Sanity[{i}] {r.image_id} probs(min/mean/max)={probs.min():.4f}/{probs.mean():.4f}/{probs.max():.4f} pred={probs.argmax()}"
            )


_sanity_check(n=3)

predictions = []
tta_n = 5  # keep consistent with variants and averaging

with torch.no_grad():
    for _, sample_row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, sample_row.image_id)
        image_np = _read_image_rgb(img_path)

        xs = []
        for v in range(tta_n):
            img_v = apply_tta_variant(image_np, v)
            xs.append(tta_aug(image=img_v)["image"])
        x = torch.stack(xs, dim=0).to(device)  # tta_n x C x H x W

        out = model(x)  # tta_n x 5 logits
        logits_avg = out.mean(dim=0, keepdim=True)  # 1 x 5
        pred_label = int(torch.argmax(logits_avg, dim=1).item())
        predictions.append([sample_row.image_id, pred_label])

pred_df = pd.DataFrame(predictions, columns=["image_id", "label"])

sub_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
assert sub_df["label"].isna().sum() == 0, "Some test images were not predicted."

sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print(
    "Wrote submission.csv with shape:",
    sub_df.shape,
    "| weights_loaded:",
    loaded,
    "| ckpt:",
    selected_model_path,
)
assert sub_df.shape[0] == sample_sub.shape[0]
assert list(sub_df.columns) == ["image_id", "label"]
assert os.path.exists("submission.csv")
